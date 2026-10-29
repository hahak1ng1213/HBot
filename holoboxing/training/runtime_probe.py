"""Bounded fixed-controller smoke test; never reports a learned boxing baseline."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import traceback


def probe(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        status="RUNNING",
        training_run=False,
        boxing_quality_verified=False,
        seed=args.seed,
        worlds=args.num_envs,
        requested_steps=args.steps,
    )
    started = time.monotonic()
    env = None
    try:
        import torch
        from holoboxing.data.contract import CompetitionAssets
        from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
        from holoboxing.training.resources import (
            resolve_resources,
            load_neutral_reference,
        )
        from holoboxing.training.rsl_rl.runners.on_policy_runner import (
            seed_everything,
        )
        from holoboxing.training.standing_reference import (
            WorldStandingReference,
        )
        from holoboxing.training.tasks.combat.config.sandbag import AGENTS

        seed_everything(args.seed)
        report["versions"] = {
            name: version(name)
            for name in (
                "torch",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
                "onnxruntime-gpu",
            )
        }
        report["gpu"] = torch.cuda.get_device_name(args.device)
        report["capability"] = torch.cuda.get_device_capability(args.device)
        report["torch_architectures"] = torch.cuda.get_arch_list()
        # Exercise an actual CUDA kernel, rather than accepting is_available().
        assert torch.ones(8, device=args.device).sum().item() == 8
        assets = CompetitionAssets.load(args.assets)
        _, paths, identity = resolve_resources(args.assets, args.lower)
        report["resources"] = identity
        reference = load_neutral_reference(
            paths["lower_onnx"], paths["arena_scene"]
        )
        env = MujocoSandbagEnv(
            assets,
            paths,
            args.num_envs,
            args.device,
            output / ".runtime",
            args.seed,
            official_rules=True,
            standing_reference=WorldStandingReference(
                reference, args.num_envs, args.device
            ),
        )
        observations = env.reset()
        actions = {
            a: reference.to(args.device).expand(args.num_envs, -1).clone()
            for a in AGENTS
        }
        trajectory = []
        for step in range(args.steps):
            observations, rewards, done, info = env.step(actions)
            for a in AGENTS:
                assert torch.isfinite(observations[a]["task_policy"]).all()
                assert torch.isfinite(rewards[a]).all()
                assert "official_score_delta" in env.reward_terms[a]
            trajectory.append(
                dict(
                    step=step + 1,
                    done=done.cpu().tolist(),
                    context={
                        a: observations[a]["task_policy"][:, -2:]
                        .cpu()
                        .tolist()
                        for a in AGENTS
                    },
                    reward={a: rewards[a].cpu().tolist() for a in AGENTS},
                    score=[m.score.tolist() for m in env.referees.matches],
                )
            )
        report.update(
            status="PASS",
            steps_completed=len(trajectory),
            trajectory=trajectory,
        )
    except Exception as error:
        report.update(
            status="FAIL",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    finally:
        if env is not None:
            env.close()
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--lower", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= args.steps <= 3000 or not 1 <= args.num_envs <= 8:
        parser.error("Bound this probe to 1..3000 steps and 1..8 environments")
    result = probe(args)
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
