"""Standalone frozen-lower skill PPO, independent of combat objectives."""

import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.skills.actor import (
    SkillActor,
    implementation_identity,
    load_teacher,
)


def json_value(value, *, diagnostic=False):
    """Preserve numerical failure evidence in valid, inspectable JSON."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "values": json_value(value.tolist(), diagnostic=diagnostic),
        }
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {
            str(key): json_value(item, diagnostic=diagnostic)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_value(item, diagnostic=diagnostic) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        if not diagnostic:
            raise ValueError("Nonfinite value in ordinary training metadata")
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if diagnostic:
        return repr(value)
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def write_json(path, value, *, diagnostic=False):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            json_value(value, diagnostic=diagnostic), indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_failure(output, error, physics=None):
    write_json(
        output / "failure.json",
        {
            "error": str(error),
            "exception_type": type(error).__name__,
            "physics_failure": getattr(physics, "failure_state", None),
            "nonfinite_representation": "string values with array dtype/shape",
        },
        diagnostic=True,
    )
    write_json(
        output / "status.json", {"state": "failed", "error": str(error)}
    )


def advantages(reward, done, values, final_value, gamma=0.995, lam=0.95):
    if (
        reward.ndim != 2
        or reward.shape != values.shape
        or done.shape != reward.shape
        or done.dtype != torch.bool
        or final_value.shape != reward.shape[1:]
    ):
        raise ValueError(
            "GAE requires [T,N] rewards/values and boolean terminals"
        )
    if not all(torch.isfinite(x).all() for x in (reward, values, final_value)):
        raise ValueError("Nonfinite PPO rollout")
    result = torch.zeros_like(reward)
    carry = torch.zeros_like(final_value)
    for step in reversed(range(len(reward))):
        next_value = (
            final_value if step == len(reward) - 1 else values[step + 1]
        )
        live = (~done[step]).float()
        delta = reward[step] + gamma * live * next_value - values[step]
        carry = delta + gamma * lam * live * carry
        result[step] = carry
    return result, result + values


def reset_training_rows(env, ids):
    if not len(ids):
        return
    if hasattr(env, "reset_training_rows"):
        env.reset_training_rows(ids)
        return
    clips = torch.randint(
        len(env.library.clips),
        (len(ids),),
        generator=env.rng,
        device=env.device,
    )
    neutral = torch.rand(len(ids), generator=env.rng, device=env.device) < 0.35
    yaw = (
        torch.rand(len(ids), generator=env.rng, device=env.device) * 2 - 1
    ) * np.pi
    for normal in (True, False):
        use = neutral == normal
        if use.any():
            env.start(
                clips[use],
                ids=ids[use],
                reset_physics=True,
                neutral=normal,
                random_phase=not normal,
                yaw=yaw[use],
            )


def checkpoint(env, model, optimizer, update, args):
    return {
        "format": getattr(env, "checkpoint_format", "solo_skill_teacher_v1"),
        "skill": args.skill,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "observation_dim": model.observation_dim,
        "action_dim": model.action_dim,
        "hidden_dims": model.hidden_dims,
        "update": update,
        "seed": args.seed,
        "reference_manifest": env.library.manifest,
        "environment_recipe": env.recipe,
        "physics_metadata": env.physics.metadata,
        "arguments": vars(args),
        "torch_rng_state": torch.get_rng_state(),
        "environment_rng_state": env.rng.get_state(),
        "is_competition_submission": False,
        "simulator_continuity_saved": False,
        "initialization": getattr(env, "initialization", None),
        "curriculum_state": getattr(env, "curriculum_state", lambda: None)(),
        "implementation_sources": getattr(env, "implementation_sources", None)
        or implementation_identity(),
    }


def save_model(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def bounded_optimizer_step(
    model, optimizer, observations, old_mean, old_std, target_kl
):
    """Reject a minibatch update that exceeds the full-rollout Gaussian KL.

    Restore Adam moments as well as weights, then reduce the next step size.
    This is opt-in for task environments declaring ppo_target_kl.
    """
    model_before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    optimizer.step()
    with torch.no_grad():
        divergence = float("inf")
        if all(torch.isfinite(p).all() for p in model.parameters()):
            current = model.distribution(observations)
            old = torch.distributions.Normal(old_mean, old_std)
            divergence = float(
                torch.distributions.kl_divergence(old, current).sum(-1).mean()
            )
    accepted = math.isfinite(divergence) and divergence <= target_kl
    if not accepted:
        model.load_state_dict(model_before)
        optimizer.load_state_dict(optimizer_before)
        for group in optimizer.param_groups:
            group["lr"] = max(1e-6, group["lr"] * 0.5)
    return accepted, divergence


def validate_training_arguments(args):
    for name in ("num_envs", "updates", "rollout_steps", "save_every"):
        if type(getattr(args, name)) is not int or getattr(args, name) < 1:
            raise ValueError(f"Expected positive integer {name}")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("Learning rate must be finite and positive")


def train_teacher(env, output, args):
    validate_training_arguments(args)
    if env.num_envs != args.num_envs:
        raise ValueError("Training environment count differs from arguments")
    output = Path(output)
    output.mkdir(exist_ok=False, parents=True)
    torch.manual_seed(args.seed)
    try:
        return _train_teacher(env, output, args)
    except BaseException as error:
        try:
            record_failure(output, error, env.physics)
        except Exception as diagnostic_error:
            print(
                f"Could not save failure evidence: {diagnostic_error}",
                file=sys.stderr,
            )
        raise


def _train_teacher(env, output, args):
    model = SkillActor(env.observation_dim).to(env.device)
    initial = getattr(env, "initial_model_state", None)
    if initial is not None:
        model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    write_json(
        output / "recipe.json",
        {
            "environment": env.recipe,
            "reference": env.library.manifest,
            "physics": env.physics.metadata,
            "updates": args.updates,
            "num_envs": env.num_envs,
            "rollout_steps": args.rollout_steps,
            "fresh_residual_policy": initial is None,
            "initialization": getattr(env, "initialization", None),
            "fresh_optimizer": True,
        },
    )
    start = time.monotonic()
    reset_training_rows(env, torch.arange(env.num_envs, device=env.device))
    observation = env.get_observations()
    last_metrics = {}
    with (output / "metrics.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(1, args.updates + 1):
            update_start = time.monotonic()
            batch = {
                key: []
                for key in (
                    "obs",
                    "actions",
                    "logp",
                    "value",
                    "reward",
                    "done",
                )
            }
            metric_sum = {}
            completions = failures = 0
            for _ in range(args.rollout_steps):
                if not torch.isfinite(observation).all():
                    raise RuntimeError("Nonfinite teacher observation")
                # Physics or partial resets may reuse the observation buffer.
                observation = observation.detach().clone()
                with torch.no_grad():
                    action = model.act(observation)
                    logp, _, value = model.evaluate(observation, action)
                for key, value_ in (
                    ("obs", observation),
                    ("actions", action),
                    ("logp", logp),
                    ("value", value),
                ):
                    batch[key].append(value_.detach().clone())
                next_obs, reward, done, info = env.step(action)
                for key, value_ in (
                    ("reward", reward),
                    ("done", done),
                ):
                    batch[key].append(value_.detach().clone())
                if not torch.isfinite(reward).all():
                    raise RuntimeError("Nonfinite teacher reward")
                completions += int(info["completed"].sum())
                failures += int(info["failed"].sum())
                for key, metric in info["metrics"].items():
                    if not torch.isfinite(metric).all():
                        raise RuntimeError(f"Nonfinite teacher metric: {key}")
                    metric_sum[key] = metric_sum.get(key, 0.0) + float(
                        metric.mean()
                    )
                reset_training_rows(
                    env, done.nonzero(as_tuple=False).flatten()
                )
                observation = (
                    env.get_observations() if done.any() else next_obs
                )
            batch = {key: torch.stack(value) for key, value in batch.items()}
            with torch.no_grad():
                adv, returns = advantages(
                    batch["reward"],
                    batch["done"],
                    batch["value"],
                    model.value(observation),
                )
            adv = (
                (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            ).flatten()
            returns = returns.flatten()
            observations = batch["obs"].flatten(0, 1)
            actions = batch["actions"].flatten(0, 1)
            old_logp = batch["logp"].flatten()
            old_value = batch["value"].flatten()
            kl = 0.0
            target_kl = getattr(env, "ppo_target_kl", None)
            if target_kl is not None:
                with torch.no_grad():
                    old_distribution = model.distribution(observations)
                    old_mean = old_distribution.mean.detach().clone()
                    old_std = old_distribution.stddev.detach().clone()
            rejected = False
            rejected_kl = 0.0
            epochs_done = 0
            for epoch in range(4):
                order = torch.randperm(len(adv), device=env.device)
                for ids in order.chunk(4):
                    logp, entropy, value = model.evaluate(
                        observations[ids], actions[ids]
                    )
                    ratio = (logp - old_logp[ids]).exp()
                    policy_loss = -torch.minimum(
                        ratio * adv[ids], ratio.clamp(0.8, 1.2) * adv[ids]
                    ).mean()
                    clipped = old_value[ids] + (value - old_value[ids]).clamp(
                        -0.2, 0.2
                    )
                    value_loss = torch.maximum(
                        (value - returns[ids]).square(),
                        (clipped - returns[ids]).square(),
                    ).mean()
                    loss = (
                        policy_loss
                        + 0.5 * value_loss
                        - 0.0005 * entropy.mean()
                    )
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, error_if_nonfinite=True
                    )
                    if target_kl is None:
                        optimizer.step()
                    else:
                        accepted, proposed_kl = bounded_optimizer_step(
                            model,
                            optimizer,
                            observations,
                            old_mean,
                            old_std,
                            target_kl,
                        )
                        if not accepted:
                            rejected = True
                            rejected_kl = (
                                proposed_kl
                                if math.isfinite(proposed_kl)
                                else -1.0
                            )
                            break
                        kl = proposed_kl
                with torch.no_grad():
                    if target_kl is None:
                        current_logp, _, _ = model.evaluate(
                            observations, actions
                        )
                        difference = current_logp - old_logp
                        kl = float((difference.exp() - 1 - difference).mean())
                    if not math.isfinite(kl):
                        raise RuntimeError("Nonfinite PPO divergence")
                epochs_done = epoch + 1
                if rejected or kl > 0.03:
                    break
            last_metrics = {
                "update": update,
                "reward_mean": float(batch["reward"].mean()),
                "completed_episodes": completions,
                "failed_episodes": failures,
                "tracking": {
                    k: v / args.rollout_steps for k, v in metric_sum.items()
                },
                "ppo_kl": kl,
                "ppo_rejected_update": int(rejected),
                "ppo_rejected_kl": rejected_kl,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimization_epochs": epochs_done,
                "update_wall_seconds": time.monotonic() - update_start,
                "wall_seconds": time.monotonic() - start,
            }
            if hasattr(env, "advance_curriculum"):
                last_metrics["curriculum"] = env.advance_curriculum(
                    update, last_metrics["tracking"]
                )
                observation = env.get_observations()
            stream.write(json.dumps(last_metrics, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(last_metrics), flush=True)
            if update % args.save_every == 0 or update == args.updates:
                save_model(
                    output / f"model_{update:06d}.pt",
                    checkpoint(env, model, optimizer, update, args),
                )
            write_json(
                output / "status.json",
                {
                    "state": "running",
                    "update": update,
                    "target_updates": args.updates,
                },
            )
    result = {
        "state": "completed",
        "updates": args.updates,
        "checkpoint": str(output / f"model_{args.updates:06d}.pt"),
        "wall_seconds": time.monotonic() - start,
        "last_metrics": last_metrics,
        "skill_quality": "requires_continuous_physical_evaluation",
    }
    write_json(output / "training-result.json", result)
    write_json(output / "status.json", result)
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument(
        "--punch-groups", nargs="+", default=["kangning-001", "kangning-003"]
    )
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument(
        "--punch-bounds", type=int, nargs=2, metavar=("START", "END")
    )
    parser.add_argument(
        "--skill", choices=("punch", "locomotion"), required=True
    )
    parser.add_argument(
        "--locomotion-mode",
        choices=("directional", "full", "forward"),
        default="directional",
        help=(
            "Step sources: directional crops, complete 010/011 motions, "
            "or forward-only crops; ignored for punch"
        ),
    )
    parser.add_argument(
        "--mode", choices=("train", "baseline", "evaluate"), default="train"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=existing_path)
    parser.add_argument(
        "--initialization", choices=("neutral", "reference"), default="neutral"
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate_training_arguments(args)
    except ValueError as error:
        parser.error(str(error))
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be new")
    if not math.isfinite(args.speed_scale) or args.speed_scale <= 0:
        parser.error("--speed-scale must be finite and positive")
    if args.mode == "evaluate" and not args.checkpoint:
        parser.error("Evaluation requires --checkpoint")
    if args.mode != "evaluate" and args.checkpoint:
        parser.error("Only evaluate accepts an existing teacher checkpoint")
    return args


def main(argv=None):
    args = parse_arguments(argv)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    from holoboxing.training.resources import resolve_resources
    from holoboxing.training.skills.env import SkillTrackingEnv
    from holoboxing.training.skills.physics import SoloPhysics
    from holoboxing.training.skills.reference import MotionLibrary

    physics = None
    try:
        resources, _, _ = resolve_resources(args.assets, args.lower)
        assets = SimpleNamespace(resources=resources)
        physics = SoloPhysics(
            assets,
            args.lower,
            args.num_envs if args.mode == "train" else 1,
            args.device,
        )
        library = MotionLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            skill=args.skill,
            device=args.device,
            locomotion_mode=args.locomotion_mode,
            punch_groups=args.punch_groups,
            speed_scale=args.speed_scale,
            punch_bounds=args.punch_bounds,
        )
        env = SkillTrackingEnv(physics, library, seed=args.seed)
        if args.mode == "train":
            print(
                json.dumps(train_teacher(env, args.output, args)), flush=True
            )
        else:
            from holoboxing.training.skills.evaluate import evaluate_skills

            teacher = None
            if args.mode == "evaluate":
                teacher, saved = load_teacher(args.checkpoint, args.device)
                if (
                    saved["reference_manifest"] != library.manifest
                    or saved["environment_recipe"] != env.recipe
                    or saved["physics_metadata"] != physics.metadata
                    or saved["observation_dim"] != env.observation_dim
                    or saved["skill"] != args.skill
                ):
                    raise ValueError(
                        "Teacher data or environment contract differs"
                    )
            result = evaluate_skills(
                env,
                teacher,
                args.output,
                record=not args.no_video,
                initialization=args.initialization,
            )
            print(json.dumps(result), flush=True)
    except BaseException as error:
        # Output conflicts do not authorize modifying another existing run.
        if not isinstance(error, FileExistsError):
            try:
                args.output.mkdir(parents=True, exist_ok=True)
                record_failure(args.output, error, physics)
            except Exception as diagnostic_error:
                print(
                    f"Could not save failure evidence: {diagnostic_error}",
                    file=sys.stderr,
                )
        raise
    finally:
        if physics is not None:
            active_exception = sys.exc_info()[0] is not None
            try:
                physics.close()
            except Exception as close_error:
                if not active_exception:
                    raise
                print(
                    f"Physics cleanup also failed: {close_error}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
