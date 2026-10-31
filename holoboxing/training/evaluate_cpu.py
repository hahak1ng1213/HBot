"""Evaluate frozen checkpoints in CPU training physics with the shared referee.

This diagnostic retains training randomization. It is not the official Docker
evaluator and does not establish competition qualification.
"""

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import torch

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.tasks.combat.config.sandbag import AGENTS


CONTACT_COUNTS = (
    "official_legal_contact_samples",
    "official_subthreshold_contact_samples",
    "official_cooldown_contact_samples",
)
IMPULSE_PEAK = "official_contact_impulse_max_ns"


@torch.no_grad()
def run_match(env, policies, *, candidate_side, max_steps):
    """Read pre-reset metric snapshots; the environment auto-resets at done."""
    observations = env.reset()
    hits, knockouts, fallen_frames = [0, 0], [0, 0], [0, 0]
    contact_counts = {name: [0, 0] for name in CONTACT_COUNTS}
    peak_impulse, scores = [0.0, 0.0], [0, 0]
    complete, invalid, winner = False, False, None
    for frame in range(max_steps):
        actions = {
            agent: policies[role](observations[agent]["task_policy"])
            for role, agent in enumerate(AGENTS)
        }
        observations, _, done, infos = env.step(actions)
        invalid = bool(infos["invalid_envs"][0])
        if invalid:
            # Invalid rows have been cleared by env.step; never count them as
            # a successful contact, terminal, or an ordinary zero-score frame.
            break
        for role, agent in enumerate(AGENTS):
            metrics = env.metrics[agent]
            scores[role] = int(metrics["official_score"][0])
            hits[role] += int(metrics["official_hits"][0])
            knockouts[role] += int(metrics["official_ko"][0])
            fallen_frames[role] += int(metrics["fallen"][0])
            for name in CONTACT_COUNTS:
                contact_counts[name][role] += int(metrics[name][0])
            peak_impulse[role] = max(
                peak_impulse[role], float(metrics[IMPULSE_PEAK][0])
            )
        if bool(done[0]):
            complete = bool(env.metrics[AGENTS[0]]["official_match_over"][0])
            if complete:
                # The referee's outcome also handles simultaneous KO. Scores
                # alone are insufficient to infer the official winner.
                outcome = float(
                    env.reward_terms[AGENTS[0]]["official_outcome"][0]
                )
                winner = 0 if outcome > 0 else 1 if outcome < 0 else None
            break
    return {
        "candidate_side": candidate_side,
        "steps": frame + 1,
        "simulation_seconds": (frame + 1) * 0.02,
        "complete_match": complete,
        "invalid_simulation": invalid,
        "end_reason": "INVALID"
        if invalid
        else (
            "KO"
            if complete and any(knockouts)
            else "TIME"
            if complete
            else "STEP_LIMIT"
        ),
        "score": scores,
        "hits": hits,
        "knockouts_awarded": knockouts,
        "fallen_frames": fallen_frames,
        "contact_samples": contact_counts,
        "peak_hand_impulse_ns": peak_impulse,
        "winner": winner,
        "candidate_win": (winner == candidate_side) if complete else None,
    }


def summarize(matches):
    complete = bool(matches) and all(
        match["complete_match"] and not match["invalid_simulation"]
        for match in matches
    )
    scored_attack = complete and all(
        match["hits"][match["candidate_side"]] > 0 for match in matches
    )
    return {
        "all_matches_complete": complete,
        "candidate_hits": sum(
            match["hits"][match["candidate_side"]] for match in matches
        ),
        # A KO from an opponent's unforced fall does not pass this gate.
        "scored_attack_gate_passed": scored_attack,
        "scored_attack_gate": "at least one candidate HIT in every complete, valid match",
        "candidate_win_rate": (
            sum(match["candidate_win"] for match in matches) / len(matches)
            if complete
            else None
        ),
        "boxing_quality_verified": False,
    }


def evaluate(args):
    from holoboxing.data.contract import CompetitionAssets
    from holoboxing.evaluation.runtime.model_contract import sha256
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.export import load_policy
    from holoboxing.training.resources import (
        load_neutral_reference,
        resolve_resources,
    )
    from holoboxing.training.rsl_rl.runners.on_policy_runner import (
        seed_everything,
    )
    from holoboxing.training.standing_reference import WorldStandingReference

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {
        "status": "RUNNING",
        "official_evaluation": False,
        "physics": "mujoco_cpu",
        "policy_device": args.device,
        "training_randomization": "COM, reset, observation noise and velocity pushes retained",
        "requested_steps_per_match": args.steps,
        "seeds": args.seeds,
        "matches": [],
        "boxing_quality_verified": False,
        "scored_attack_gate_passed": False,
    }
    try:
        torch.set_num_threads(1)
        assets = CompetitionAssets.load(args.assets)
        _, paths, resources = resolve_resources(args.assets, args.lower)
        report["resources"] = resources
        report["candidate"] = {
            "checkpoint": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
        }
        candidate = load_policy(args.checkpoint).to(args.device).eval()
        candidate.requires_grad_(False)
        if args.standing_opponent:
            reference = load_neutral_reference(
                paths["lower_onnx"], paths["arena_scene"]
            ).to(args.device)
            opponent = lambda obs: reference.expand(obs.shape[0], -1)
            report["opponent"] = {
                "kind": "neutral_world_targets",
                "candidate_sides": [0],
            }
        else:
            reference = None
            opponent = (
                load_policy(args.opponent_checkpoint).to(args.device).eval()
            )
            opponent.requires_grad_(False)
            report["opponent"] = {
                "checkpoint": str(args.opponent_checkpoint),
                "sha256": sha256(args.opponent_checkpoint),
                "candidate_sides": [0, 1],
            }
        for seed in args.seeds:
            for side in report["opponent"]["candidate_sides"]:
                seed_everything(seed)
                standing = (
                    WorldStandingReference(reference, 1, args.device)
                    if args.standing_opponent
                    else None
                )
                # Reuse one runtime directory: do not duplicate the 1.65 GB
                # specialized lower-controller file for every seed/side.
                env = MujocoSandbagEnv(
                    assets,
                    paths,
                    1,
                    args.device,
                    output / ".runtime",
                    seed,
                    standing_reference=standing,
                    official_rules=True,
                )
                try:
                    policies = (
                        (candidate, opponent)
                        if side == 0
                        else (opponent, candidate)
                    )
                    match = run_match(
                        env,
                        policies,
                        candidate_side=side,
                        max_steps=args.steps,
                    )
                    match["seed"] = seed
                    report["matches"].append(match)
                    print(json.dumps(match, allow_nan=False), flush=True)
                finally:
                    env.close()
                    del env
                    # CommonLower holds an environment reference; release the
                    # cycle before loading another large ONNX session.
                    gc.collect()
                report.update(summarize(report["matches"]))
                (output / "report.json").write_text(
                    json.dumps(report, indent=2, allow_nan=False)
                )
        report["status"] = (
            "FAIL"
            if any(match["invalid_simulation"] for match in report["matches"])
            else "PASS"
        )
    except Exception as error:
        report.update(
            status="FAIL",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
            all_matches_complete=False,
            scored_attack_gate_passed=False,
        )
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False)
        )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--checkpoint", type=existing_path, required=True)
    opponent = parser.add_mutually_exclusive_group(required=True)
    opponent.add_argument("--opponent-checkpoint", type=existing_path)
    opponent.add_argument(
        "--standing-opponent",
        action="store_true",
        help="Candidate red only; blue holds neutral world targets",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=[43])
    parser.add_argument(
        "--steps",
        type=int,
        default=3000,
        help="Per-match cap; partial matches do not pass the attack gate",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 3000:
        parser.error("--steps must be 1..3000")
    if len(set(args.seeds)) != len(args.seeds) or any(
        not 0 <= s < 2**32 for s in args.seeds
    ):
        parser.error("Use distinct nonnegative 32-bit seeds")
    report = evaluate(args)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
