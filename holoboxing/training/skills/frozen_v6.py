"""Frozen v6 teacher inference with an explicit action/transition boundary.

The archived step is executed until it submits targets. Only its unused reward
calculation is omitted; phase, contact clock and resets retain v6 semantics.
"""

from dataclasses import fields
import hashlib
import importlib.util
from pathlib import Path
import sys

import torch

from holoboxing.training.skills.actor import SkillActor


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_v6(path, source_root, device):
    saved = torch.load(path, map_location=device, weights_only=False)
    if (
        saved.get("format"),
        saved.get("update"),
        saved.get("observation_dim"),
        saved.get("action_dim"),
    ) != ("kangning037_range_teacher_v6", 9536, 1356, 30):
        raise ValueError(
            "Require the frozen v6 update9536, 1356 -> 30 teacher"
        )
    package = (Path(source_root) / "holoboxing").resolve()
    closure = saved["implementation_sources"]
    for relative, expected in closure.items():
        file = (package / relative).resolve()
        if not file.is_relative_to(package) or digest(file) != expected:
            raise ValueError("Archived teacher source differs: " + relative)
    current = Path(__file__).resolve().parents[2]
    dependencies = [
        p for p in closure if p.startswith(("data/", "evaluation/"))
    ]
    dependencies += [
        "training/skills/" + p
        for p in (
            "actor.py",
            "env.py",
            "reference.py",
            "physics.py",
            "punch_reach.py",
        )
    ]
    dependencies += [
        "training/envs/" + p
        for p in (
            "mujoco_env.py",
            "mujoco_physics.py",
            "control.py",
            "onnx_lower.py",
        )
    ]
    dependencies += [
        "training/tasks/combat/config/sandbag.py",
        "training/tasks/combat/mdp/geometry.py",
    ]
    for relative in dependencies:
        if digest(current / relative) != closure[relative]:
            raise ValueError(
                "Frozen controller dependency differs: " + relative
            )
    # Only two extra diagnostic counters are permitted in the shared referee.
    name = "training/envs/official_referee.py"
    text = (current / name).read_text()
    for line in (
        '                "official_left_hits",\n',
        '                "official_right_hits",\n',
        '                    metrics[f"official_{hand}_hits"][world, actor] += 1\n',
    ):
        text = text.replace(line, "")
    text = text.replace(
        "for (actor, hand, _), rows in samples.items():",
        "for (actor, _, _), rows in samples.items():",
    )
    if text != (package / name).read_text():
        raise ValueError("Referee changed beyond additive per-hand counters")
    source = package / "training/skills/fixed_punch.py"
    name = "_frozen_v6_" + digest(source)[:12]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    model = SkillActor(1356, 30, saved["hidden_dims"]).to(device)
    model.load_state_dict(saved["model"], strict=True)
    model.eval().requires_grad_(False)
    if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError("Nonfinite frozen teacher")
    return (
        model,
        saved,
        module,
        {
            "checkpoint": str(path),
            "sha256": digest(path),
            "source_root": str(source_root),
            "update": 9536,
            "source_sha256": digest(source),
        },
    )


class TargetsReady(Exception):
    def __init__(self, targets):
        self.targets = targets


def swapped(obs):
    return {"robot_A": obs["robot_B"], "robot_B": obs["robot_A"]}


class PhysicsView:
    def __init__(self, physics):
        self.base = physics

    def __getattr__(self, name):
        return getattr(self.base, name)

    def state(self):
        return {k: v.flip(1) for k, v in self.base.state().items()}


class CombatView:
    def __init__(self, combat, observations):
        self.physics = PhysicsView(combat.physics)
        self.device, self.num_envs = combat.device, combat.num_envs
        self.official_rules = self.capture_transition_state = True
        self.standing_reference = None
        self.observations = observations

    def reset(self):
        return swapped(self.observations)

    def step(self, actions):
        # Do not advance the simulator: red and blue must be submitted together.
        raise TargetsReady(actions["robot_A"])


class FrozenV6Opponent:
    def __init__(self, env, model, saved, module, identity):
        self.model, self.identity = model, identity
        recipe = saved["environment_recipe"]
        if (
            env.library.manifest != saved["reference_manifest"]
            or env.movement_library.manifest != recipe["motion"]
        ):
            raise ValueError("Frozen teacher motion manifests differ")
        view = CombatView(env.combat, env.obs)
        config = module.FixedPunchConfig(
            **{f.name: recipe[f.name] for f in fields(module.FixedPunchConfig)}
        )
        self.teacher = module.FixedPunchCombatEnv(
            view,
            env.library,
            env.movement_library,
            config=config,
            seed=saved["seed"],
            opponent=lambda obs: torch.zeros(
                obs.shape[0], 225, device=obs.device
            ),
        )
        self.versions = tuple(p._version for p in model.parameters())
        self.pending = False

    @torch.no_grad()
    def __call__(self, observation):
        if self.pending:
            raise RuntimeError(
                "Frozen blue requires one transition per action"
            )
        if any(
            p.requires_grad for p in self.model.parameters()
        ) or self.versions != tuple(
            p._version for p in self.model.parameters()
        ):
            raise RuntimeError("Frozen teacher weights changed")
        t = self.teacher
        raw = self.model.act(t.get_observations(), deterministic=True)
        self.bounded = raw.tanh().reshape(t.num_envs, 5, 6).clone()
        self.bounded[~t.punching, 1:3] = 0
        self.close_align = t.attack & ~t.punching
        self.bounded[self.close_align, :, :3] = 0
        try:
            t.step(raw)
        except TargetsReady as result:
            self.pending = True
            return result.targets
        raise RuntimeError("Archived teacher did not submit targets")

    def after_step(self, observations, stats, done):
        if not self.pending:
            raise RuntimeError("Missing frozen blue action")
        t = self.teacher
        t.obs = swapped(observations)
        t.phase += (~self.close_align).long()
        t.age += 1
        t.no_torso_contact_steps = torch.where(
            stats["official_torso_contact_samples"] > 0,
            torch.zeros_like(t.no_torso_contact_steps),
            t.no_torso_contact_steps + 1,
        )
        t.last_action = self.bounded.flatten(1).detach().clone()
        t._prepare(done)
        self.pending = False

    def reset(self, observations, mask):
        self.teacher.obs = swapped(observations)
        self.teacher._prepare(mask)
        self.pending = False
