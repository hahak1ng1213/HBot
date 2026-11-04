"""Train/evaluate kangning037 at 1.5x with a strict 0.6 m attack gate.

This entry uses the competition CPU physics and official referee, with a
passive world-anchored opponent. Checkpoints are task teachers, not submissions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.skills.actor import SkillActor
from holoboxing.training.skills.fixed_punch import (
    FixedPunchCombatEnv,
    FixedPunchConfig,
)
from holoboxing.training.skills.train import (
    record_failure,
    train_teacher,
    validate_training_arguments,
    write_json,
)


def source_identity():
    # Bind the task, CPU physics, reference loader and referee implementations.
    root = Path(__file__).resolve().parents[2]
    files = sorted(
        {
            path
            for directory in ("training", "data", "evaluation")
            for path in (root / directory).rglob("*.py")
        }
    )
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in files
    }


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--movement", choices=("demo", "explore"), default="demo"
    )
    parser.add_argument(
        "--mode", choices=("train", "baseline", "evaluate"), default="train"
    )
    parser.add_argument("--checkpoint", type=existing_path)
    parser.add_argument("--opponent-onnx", type=existing_path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--updates", type=int, default=256)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--eval-steps", type=int, default=3000)
    parser.add_argument("--guard-offset-deg", type=float, default=0.0)
    parser.add_argument("--physics-threads", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        validate_training_arguments(args)
        FixedPunchConfig(
            movement=args.movement, guard_offset_deg=args.guard_offset_deg
        )
        if args.eval_steps < 1 or args.physics_threads < 1 or args.seed < 0:
            raise ValueError(
                "Require positive steps/threads and nonnegative seed"
            )
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be new")
    runtime = args.output.parent / (args.output.name + ".runtime")
    if runtime.exists() or runtime.is_symlink():
        parser.error("Associated .runtime directory must also be new")
    if (args.mode == "evaluate") != (args.checkpoint is not None):
        parser.error("Only evaluate requires --checkpoint")
    args.skill = "kangning037_combat"
    return args


class FrozenOpponent:
    """Stateless competition ONNX, including its exported preprocessing."""

    def __init__(self, path):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), options, providers=["CPUExecutionProvider"]
        )
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (
            len(inputs) != 1
            or len(outputs) != 1
            or inputs[0].name != "obs"
            or outputs[0].name != "target_pose"
            or inputs[0].shape != [1, 1097]
            or outputs[0].shape != [1, 225]
        ):
            raise ValueError(
                "Opponent must use stateless competition [1,1097] -> [1,225] contract"
            )
        self.identity = {
            "path": str(path),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "provider": "CPUExecutionProvider",
        }

    def __call__(self, observation):
        import numpy as np

        rows = observation.detach().cpu().numpy().astype(np.float32)
        output = np.concatenate(
            [
                self.session.run(["target_pose"], {"obs": row[None]})[0]
                for row in rows
            ]
        )
        if output.shape != (len(rows), 225) or not np.isfinite(output).all():
            raise ValueError("Invalid frozen opponent output")
        return torch.as_tensor(output, device=observation.device)


def build_environment(args):
    # These imports require the standalone boxing runtime, not Isaac Lab.
    from holoboxing.data.contract import CompetitionAssets
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.resources import (
        load_neutral_reference,
        resolve_resources,
    )
    from holoboxing.training.skills.physics import load_solo_model
    from holoboxing.training.skills.reference import MotionLibrary
    from holoboxing.training.standing_reference import WorldStandingReference
    from holoboxing.training.tasks.combat.config.sandbag import (
        BODY_NAMES,
        JOINT_NAMES,
    )

    assets = CompetitionAssets.load(args.assets)
    _, paths, resource_identity = resolve_resources(args.assets, args.lower)
    neutral = load_neutral_reference(paths["lower_onnx"], paths["arena_scene"])
    # Kinematics only; this does not create another lower inference session.
    solo = load_solo_model(assets)
    punch = MotionLibrary(
        args.source,
        solo,
        JOINT_NAMES,
        BODY_NAMES,
        skill="punch",
        punch_groups=("kangning-037",),
        # Original 50 Hz frames: both ends are boxing-ready, excluding
        # the wide-arm calibration/entry and exit sections of the source.
        punch_bounds=(90, 1321),
        speed_scale=1.5,
        device=args.device,
    )
    movement = None
    if args.movement == "demo":
        movement = MotionLibrary(
            args.source,
            solo,
            JOINT_NAMES,
            BODY_NAMES,
            skill="locomotion",
            locomotion_mode="forward",
            device=args.device,
        )
    opponent = (
        FrozenOpponent(args.opponent_onnx) if args.opponent_onnx else None
    )
    standing = (
        WorldStandingReference(neutral, args.num_envs, args.device)
        if opponent is None
        else None
    )
    combat = MujocoSandbagEnv(
        assets,
        paths,
        args.num_envs,
        args.device,
        args.output.parent / (args.output.name + ".runtime"),
        args.seed,
        standing_reference=standing,
        official_rules=True,
        capture_transition_state=True,
        physics_threads=args.physics_threads,
    )
    try:
        env = FixedPunchCombatEnv(
            combat,
            punch,
            movement,
            config=FixedPunchConfig(
                movement=args.movement, guard_offset_deg=args.guard_offset_deg
            ),
            seed=args.seed,
            opponent=opponent,
        )
        env.implementation_sources = source_identity()
        env.recipe["resources"] = resource_identity
        env.recipe["opponent"] = (
            opponent.identity if opponent is not None else "world_standing"
        )
        return env
    except BaseException:
        combat.close()
        raise


def evaluate(env, args):
    model = None
    if args.mode == "evaluate":
        saved = torch.load(
            args.checkpoint, map_location=env.device, weights_only=False
        )
        expected = {
            "format": env.checkpoint_format,
            "environment_recipe": env.recipe,
            "reference_manifest": env.library.manifest,
            "observation_dim": env.observation_dim,
            "implementation_sources": source_identity(),
        }
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Checkpoint task, references or implementation differs"
            )
        model = SkillActor(env.observation_dim).to(env.device)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
    args.output.mkdir(parents=True, exist_ok=False)
    env.obs = env.combat.reset()
    env._prepare(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    write_json(
        args.output / "recipe.json",
        {"environment": env.recipe, "reference": env.library.manifest},
    )
    write_json(
        args.output / "status.json", {"state": "running", "kind": args.mode}
    )
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    hit_counts = torch.zeros(env.num_envs, device=env.device)
    gated_hits = torch.zeros_like(hit_counts)
    falls = torch.zeros_like(alive)
    invalids = torch.zeros_like(alive)
    completed = torch.zeros_like(alive)
    with (args.output / "metrics.jsonl").open("x", encoding="utf-8") as stream:
        for step in range(args.eval_steps):
            with torch.no_grad():
                action = (
                    torch.zeros(env.num_envs, 30, device=env.device)
                    if model is None
                    else model.act(env.get_observations(), deterministic=True)
                )
                _, _, done, info = env.step(action)
            m = info["metrics"]
            hit_counts += m["official_hits"] * alive
            gated_hits += m["official_hit"] / 5 * alive
            falls |= info["failed"] & alive
            invalids |= m["invalid"].bool() & alive
            completed |= info["completed"] & alive
            stream.write(
                json.dumps(
                    {
                        "step": step,
                        **{k: v.detach().cpu().tolist() for k, v in m.items()},
                    },
                    allow_nan=False,
                )
                + "\n"
            )
            alive &= ~done
            if not alive.any():
                break
    report = {
        "execution": "completed",
        "kind": args.mode,
        "steps": step + 1,
        "official_hits": hit_counts.tolist(),
        "fallen_or_invalid": falls.tolist(),
        "in_range_facing_hits": gated_hits.tolist(),
        "invalid": invalids.tolist(),
        "complete_bouts": completed.tolist(),
        "scored_attack_gate_passed": bool(
            (completed & ~falls & ~invalids & (gated_hits > 0)).all()
        ),
        "submission_qualified": False,
    }
    write_json(args.output / "evaluation.json", report)
    write_json(args.output / "status.json", {"state": "completed", **report})
    return report


def main(argv=None):
    args = parse_arguments(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = None
    try:
        env = build_environment(args)
        result = (
            train_teacher(env, args.output, args)
            if args.mode == "train"
            else evaluate(env, args)
        )
        print(json.dumps(result), flush=True)
    except BaseException as error:
        # Existing output conflicts never authorize changing somebody's run.
        if not isinstance(error, FileExistsError):
            try:
                if not args.output.exists():
                    args.output.mkdir(parents=True, exist_ok=False)
                if (
                    env is not None
                    or not (args.output / "status.json").exists()
                ):
                    record_failure(
                        args.output, error, getattr(env, "physics", None)
                    )
            except Exception as diagnostic_error:
                print(
                    f"Could not record task failure: {diagnostic_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        if env is not None:
            env.combat.close()


if __name__ == "__main__":
    main()
