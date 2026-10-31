"""Train or resume the boxing planner on competition physics."""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from holoboxing.training import cli as _common
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig
from holoboxing.training.time_budget import (
    add_budget_arguments,
    budget_from_arguments,
)


def resolve_training_config(args, checkpoint):
    """Keep restart settings, but use stage-specific exploration for new actors."""
    from holoboxing.training.motion_prior import FORMAT as PRIOR_FORMAT
    from holoboxing.training.spherical_actor import FORMAT as IMITATION_FORMAT
    from holoboxing.training.spherical_rl import FORMAT as LATENT_RL_FORMAT

    if args.config:
        config = TrainConfig(**json.loads(args.config.read_text()))
    elif args.resume:
        config = TrainConfig(**checkpoint["config"])
    elif args.policy_checkpoint or args.supervised_checkpoint:
        # Architecture and physical action mapping must match the saved actor.
        config = TrainConfig(**checkpoint["config"])
    else:
        config = TrainConfig()

    if not args.config and not args.resume:
        changes = {"num_envs": 8}
        if checkpoint.get("format") in (
            PRIOR_FORMAT,
            IMITATION_FORMAT,
            LATENT_RL_FORMAT,
        ):
            changes.update(
                horizon_rho=0,
                init_noise_std=0.2
                if args.training_mode == "selfplay"
                else 0.1,
            )
        config = replace(config, **changes)
    return replace(
        config,
        **{
            key: value
            for key, value in {
                "num_envs": args.num_envs,
                "seed": args.seed,
            }.items()
            if value is not None
        },
    )


def main(argv=None):
    started_at = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    _common.add_controller_arguments(parser)
    add_budget_arguments(parser)
    parser.add_argument(
        "--amp-motions",
        type=_common.existing_path,
        required=True,
        help="prepared directory from holoboxing.data prepare containing 50 Hz AMP references",
    )
    start = parser.add_mutually_exclusive_group(required=True)
    start.add_argument(
        "--supervised-checkpoint",
        type=_common.existing_path,
        help="epoch_*.pt produced by holoboxing.training.train_sl",
    )
    start.add_argument("--resume", type=_common.existing_path)
    start.add_argument(
        "--policy-checkpoint",
        type=_common.existing_path,
        help="RL actor initialization for a new stage; resets learner state",
    )
    start.add_argument(
        "--prior-checkpoint",
        type=_common.existing_path,
        help="start latent RL without imitation",
    )
    parser.add_argument(
        "--config",
        type=_common.existing_path,
        help="experiment-specific TrainConfig JSON",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--training-mode",
        choices=("paired", "standing", "selfplay"),
        default="paired",
        help="paired learning, standing targets, or historical opponents",
    )
    parser.add_argument("--opponent-pool-capacity", type=int, default=8)
    parser.add_argument("--opponent-refresh-updates", type=int, default=256)
    parser.add_argument(
        "--official-rules",
        action="store_true",
        help="Use the shared official referee; correctness-first CPU contact bridge",
    )
    parser.add_argument(
        "--observation-mode", choices=("current_state", "history10")
    )
    parser.add_argument(
        "--match-context",
        action="store_true",
        help="Enable time/score input for a new spherical RL stage",
    )
    parser.add_argument(
        "--opponent-checkpoint",
        action="append",
        default=[],
        type=_common.existing_path,
    )
    parser.add_argument(
        "--prior-normalization-dataset",
        choices=("interhuman", "robostriker"),
        default="interhuman",
        help="Measured training records for prior-only initialization; SL still requires InterHuman",
    )
    parser.add_argument("--allow-resume-code-change", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="new empty training output directory",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=40000,
        help="Additional PPO iterations to run (default: 40000)",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Neural-network device (cpu or cuda:N); physics always uses native MuJoCo CPU",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument(
        "--distance-curriculum",
        action="store_true",
        help="Standing training: scored bouts unlock 1.0/1.4/1.8 m then normal resets",
    )
    parser.add_argument(
        "--rapid-approach",
        action="store_true",
        help="Distance-gated approach/attack rewards; also enabled by distance curriculum",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=1024,
        help="Save candidate checkpoints every N PPO updates",
    )
    args = parser.parse_args(argv)
    if args.distance_curriculum and (
        not args.official_rules or args.training_mode != "standing"
    ):
        parser.error(
            "--distance-curriculum requires --official-rules --training-mode standing"
        )
    if args.rapid_approach and not args.official_rules:
        parser.error("--rapid-approach requires --official-rules")
    args.rapid_approach = args.rapid_approach or args.distance_curriculum
    if not (
        args.supervised_checkpoint
        or args.resume
        or args.prior_checkpoint
        or args.policy_checkpoint
    ).is_file():
        parser.error("Initialization must be a checkpoint file")
    if args.allow_resume_code_change and not args.resume:
        parser.error("--allow-resume-code-change requires --resume")
    if args.resume and (
        args.observation_mode or args.match_context or args.opponent_checkpoint
    ):
        parser.error(
            "Change observation or pool only in a new stage, not resume"
        )
    if args.match_context and not args.official_rules:
        parser.error("--match-context requires --official-rules")
    if args.opponent_checkpoint and args.training_mode != "selfplay":
        parser.error("External opponents require selfplay")
    if (
        args.prior_normalization_dataset != "interhuman"
        and not args.prior_checkpoint
    ):
        parser.error(
            "--prior-normalization-dataset only applies to --prior-checkpoint"
        )
    if args.updates <= 0:
        parser.error("--updates must be positive")
    if args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    if args.opponent_pool_capacity < 2 or args.opponent_refresh_updates < 1:
        parser.error(
            "Opponent pool needs >=2 slots and a positive refresh interval"
        )
    try:
        budget = budget_from_arguments(args, started_at=started_at)
    except ValueError as error:
        parser.error(str(error))
    import torch

    initialization = (
        args.resume
        or args.supervised_checkpoint
        or args.prior_checkpoint
        or args.policy_checkpoint
    )
    checkpoint = torch.load(
        initialization, map_location="cpu", weights_only=False
    )
    config = resolve_training_config(args, checkpoint)
    args.num_envs, args.seed, args.action_dim = (
        config.num_envs,
        config.seed,
        config.action_dim,
    )
    from holoboxing.training.scripts.train_mujoco import train

    with budget.handle_signals():
        train(args, config, time_budget=budget, checkpoint=checkpoint)


if __name__ == "__main__":
    main()
