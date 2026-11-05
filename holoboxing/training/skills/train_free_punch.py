"""Free punch/recovery curriculum; frozen red prior and blue are v6/update9536.

Example: python -m holoboxing.training.skills.train_free_punch --help
Provide --teacher-checkpoint and --teacher-source-root from the preserved v6
run plus the ordinary fixed-punch resource, output and training arguments.
"""

import argparse
from dataclasses import fields
import json

import torch

from holoboxing.training.cli import existing_path
from holoboxing.training.skills.actor import SkillActor
from holoboxing.training.skills.fixed_punch import FixedPunchConfig
from holoboxing.training.skills.free_punch import FreePunchCombatEnv
from holoboxing.training.skills.frozen_v6 import FrozenV6Opponent, load_v6
from holoboxing.training.skills.train import (
    record_failure,
    train_teacher,
    write_json,
)
from holoboxing.training.skills.train_fixed_punch import (
    build_environment,
    parse_arguments,
    source_identity,
)


class PendingOpponent:
    def __init__(self, identity):
        self.identity = identity

    def __call__(self, observation):
        raise RuntimeError("Frozen blue has not been attached")


def build_free_environment(args):
    model, saved, module, identity = load_v6(
        args.teacher_checkpoint, args.teacher_source_root, args.device
    )
    env = build_environment(
        args,
        env_class=FreePunchCombatEnv,
        opponent_override=PendingOpponent(identity),
    )
    try:
        if env.recipe["resources"] != saved["environment_recipe"]["resources"]:
            raise ValueError("Teacher resources differ")
        for field in fields(FixedPunchConfig):
            if (
                getattr(env.config, field.name)
                != saved["environment_recipe"][field.name]
            ):
                raise ValueError(
                    "Red prior execution setting differs: " + field.name
                )
        blue = FrozenV6Opponent(env, model, saved, module, identity)
        env.configure(model, blue, identity, args.rollout_steps)
        return env
    except BaseException:
        env.combat.close()
        raise


def evaluate_free(env, args):
    model = None
    if args.mode == "evaluate":
        saved = torch.load(
            args.checkpoint, map_location=env.device, weights_only=False
        )
        for key, expected in {
            "format": env.checkpoint_format,
            "environment_recipe": env.recipe,
            "reference_manifest": env.library.manifest,
            "observation_dim": env.observation_dim,
            "implementation_sources": source_identity(),
        }.items():
            if saved.get(key) != expected:
                raise ValueError(
                    "Free-punch evaluation contract differs: " + key
                )
        model = SkillActor(env.observation_dim).to(env.device)
        model.load_state_dict(saved["model"], strict=True)
        model.eval()
        env.restore_curriculum(saved["curriculum_state"])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "recipe.json", env.recipe)
    env.obs = env.combat.reset()
    env._prepare(torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    hits = torch.zeros(env.num_envs, device=env.device)
    cycles = torch.zeros_like(hits)
    failed = ~alive
    completed = ~alive
    with (args.output / "metrics.jsonl").open("x") as stream:
        for step in range(args.eval_steps):
            with torch.no_grad():
                action = (
                    torch.zeros(env.num_envs, 30, device=env.device)
                    if model is None
                    else model.act(env.get_observations(), deterministic=True)
                )
                _, _, done, info = env.step(action)
            m = info["metrics"]
            hits += m["official_hits"] * alive
            cycles += m["hit_complete_cycles"] * alive
            failed |= (m["low_pose"].bool() | m["invalid"].bool()) & alive
            completed |= done & alive & ~failed
            stream.write(
                json.dumps(
                    {"step": step, **{k: v.tolist() for k, v in m.items()}},
                    allow_nan=False,
                )
                + "\n"
            )
            alive &= ~done
            if not alive.any():
                break
    report = {
        "state": "completed",
        "steps": step + 1,
        "official_hits": hits.tolist(),
        "hit_recovery_cycles": cycles.tolist(),
        "fallen_or_invalid": failed.tolist(),
        "complete_bouts": completed.tolist(),
        "curriculum": env.curriculum_state(),
        "submission_qualified": False,
    }
    write_json(args.output / "evaluation.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--teacher-checkpoint", type=existing_path)
    parser.add_argument("--teacher-source-root", type=existing_path)
    extra, remaining = parser.parse_known_args(argv)
    if "--help" in remaining or "-h" in remaining:
        print(__doc__)
        print(parser.format_help())
    elif extra.teacher_checkpoint is None or extra.teacher_source_root is None:
        parser.error("Require --teacher-checkpoint and --teacher-source-root")
    args = parse_arguments(remaining)
    if "--learning-rate" not in remaining and not any(
        x.startswith("--learning-rate=") for x in remaining
    ):
        args.learning_rate = 1e-5
    if (
        args.init_checkpoint
        or args.opponent_onnx
        or args.standing_guard
        or args.movement != "demo"
    ):
        parser.error(
            "Use the frozen9536 teacher and demo gait; no other initialization/opponent overrides"
        )
    args.teacher_checkpoint, args.teacher_source_root = (
        extra.teacher_checkpoint,
        extra.teacher_source_root,
    )
    args.skill = "free_punch_v7"
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    env = None
    try:
        env = build_free_environment(args)
        result = (
            train_teacher(env, args.output, args)
            if args.mode == "train"
            else evaluate_free(env, args)
        )
        print(json.dumps(result), flush=True)
    except BaseException as error:
        # A pre-existing output never authorizes replacing another run's data.
        if (
            env is not None
            and args.output.exists()
            and not isinstance(error, FileExistsError)
        ):
            record_failure(args.output, error, env.physics)
        raise
    finally:
        if env is not None:
            env.combat.close()


if __name__ == "__main__":
    main()
