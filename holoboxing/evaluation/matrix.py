"""Independent held-out opponents, both sides, and validation-only selection.

Each candidate is an exported checkpoint with its training seed and ablation
label. Repeated deterministic games are deliberately not generated.
"""

import argparse
import json
import re
from pathlib import Path

from .runtime.model_contract import sha256


def validate_plan(plan):
    if plan["split"] not in ("validation", "test"):
        raise ValueError("Expected validation or final test split")
    for group in ("candidates", "opponents"):
        rows = plan[group]
        ids = [r["id"] for r in rows]
        if (
            not rows
            or len(set(ids)) != len(ids)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in ids)
        ):
            raise ValueError(
                "Nonempty populations with unique safe IDs required"
            )
    if plan.get("baseline_variant") is not None:
        keys = [(c["variant"], c["seed"]) for c in plan["candidates"]]
        if len(set(keys)) != len(keys):
            raise ValueError(
                "Ablations require one preselected checkpoint per variant/seed"
            )
        if plan["baseline_variant"] not in {
            c["variant"] for c in plan["candidates"]
        }:
            raise ValueError("Unknown baseline variant")
    hashes = {r["id"]: sha256(Path(r["model"])) for r in plan["opponents"]}
    if len(set(hashes.values())) != len(hashes):
        raise ValueError(
            "Duplicate opponent weights are not independent opponents"
        )
    candidate_hashes = set()
    for candidate in plan["candidates"]:
        if type(candidate["seed"]) is not int or not candidate["variant"]:
            raise ValueError("Record training seed and observation variant")
        trained = candidate["training_opponent_sha256"]
        if set(trained) & set(hashes.values()):
            raise ValueError(
                "Training opponent leaked into held-out population"
            )
        digest = sha256(Path(candidate["model"]))
        if digest in candidate_hashes:
            raise ValueError(
                "Duplicate candidate weights are not independent seeds"
            )
        candidate_hashes.add(digest)
        if digest in hashes.values():
            raise ValueError("A candidate cannot be its own held-out opponent")
    return hashes


def aggregate(rows):
    if not rows or any(r["status"] != "PASS" for r in rows):
        return dict(
            eligible=False, failures=sum(r["status"] != "PASS" for r in rows)
        )
    wins = sum(r["outcome"] == "win" for r in rows)
    draws = sum(r["outcome"] == "draw" for r in rows)
    return dict(
        eligible=True,
        games=len(rows),
        wins=wins,
        draws=draws,
        losses=len(rows) - wins - draws,
        win_rate=wins / len(rows),
        match_points=(wins + 0.5 * draws) / len(rows),
        mean_score=sum(r["score"] for r in rows) / len(rows),
        mean_score_difference=sum(r["difference"] for r in rows) / len(rows),
        hits=sum(
            r["summary"].get("hit_count", [0, 0])[r["side"]] for r in rows
        ),
        hits_received=sum(
            r["summary"].get("hit_count", [0, 0])[1 - r["side"]] for r in rows
        ),
        kos=sum(r["summary"].get("ko_count", [0, 0])[r["side"]] for r in rows),
        kos_received=sum(
            r["summary"].get("ko_count", [0, 0])[1 - r["side"]] for r in rows
        ),
    )


def compare_variants(candidates, summaries, baseline):
    """Pair training seeds; games from one deterministic policy are not seeds."""
    if baseline is None:
        return {}
    groups = {}
    for candidate in candidates:
        group = groups.setdefault(candidate["variant"], {})
        seed = candidate["seed"]
        if seed in group:
            raise ValueError(
                "Ablations require one preselected checkpoint per variant/seed"
            )
        group[seed] = summaries[candidate["id"]]
    if baseline not in groups:
        raise ValueError("Unknown baseline variant")
    reports = {}
    for variant, results in groups.items():
        if variant == baseline:
            continue
        seeds = sorted(set(results) & set(groups[baseline]))
        valid = [
            s
            for s in seeds
            if results[s]["eligible"] and groups[baseline][s]["eligible"]
        ]
        deltas = [
            results[s]["match_points"] - groups[baseline][s]["match_points"]
            for s in valid
        ]
        reports[variant] = dict(
            baseline=baseline,
            paired_seeds=valid,
            unpaired_or_failed_seeds=sorted(
                (set(results) | set(groups[baseline])) - set(valid)
            ),
            match_point_deltas=deltas,
            mean_delta=sum(deltas) / len(deltas) if deltas else None,
            sufficient_seed_count=len(valid) >= 3,
            note="Descriptive paired seed comparison; no significance or causal benefit claim.",
        )
    return reports


def run_matrix(plan, output, match_fn=None):
    hashes = validate_plan(plan)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    if match_fn is None:
        from .match import evaluate_match

        match_fn = evaluate_match
    frozen_plan = json.loads(json.dumps(plan))
    frozen_plan["opponent_hashes"] = hashes
    frozen_plan["candidate_hashes"] = {
        c["id"]: sha256(c["model"]) for c in plan["candidates"]
    }
    (root / "plan.json").write_text(json.dumps(frozen_plan, indent=2))
    rows, summaries = [], {}
    for c in plan["candidates"]:
        for o in plan["opponents"]:
            for side in (0, 1):
                row = dict(
                    candidate=c["id"],
                    opponent=o["id"],
                    side=side,
                    seed=c["seed"],
                    variant=c["variant"],
                    output=str(root / f"{c['id']}--{o['id']}--{side}"),
                )
                try:
                    models = (
                        [c["model"], o["model"]]
                        if side == 0
                        else [o["model"], c["model"]]
                    )
                    result = match_fn(
                        *models,
                        plan["assets"],
                        plan["lower"],
                        root / f"{c['id']}--{o['id']}--{side}",
                    )
                    summary = result["summary"]
                    if result["status"] != "PASS" or summary[
                        "end_reason"
                    ] not in ("KO", "TIME"):
                        raise ValueError("Incomplete or invalid match")
                    winner = summary["winner"]
                    if winner is not None and (
                        type(winner) is not int or winner not in (0, 1)
                    ):
                        raise ValueError("Invalid match winner")
                    scores = summary["final_score"]
                    if len(scores) != 2 or any(
                        type(s) is not int or s < 0 for s in scores
                    ):
                        raise ValueError("Invalid final score")
                    json.dumps(summary, allow_nan=False)
                    row.update(
                        status="PASS",
                        outcome="draw"
                        if winner is None
                        else "win"
                        if winner == side
                        else "loss",
                        score=summary["final_score"][side],
                        difference=summary["final_score"][side]
                        - summary["final_score"][1 - side],
                        summary=summary,
                        match=result.get("match"),
                        replay=result.get("replay"),
                    )
                except Exception as error:
                    row.update(
                        status="FAIL", error=f"{type(error).__name__}: {error}"
                    )
                rows.append(row)
                with (root / "matches.jsonl").open("a") as f:
                    f.write(json.dumps(row, allow_nan=False) + "\n")
        summaries[c["id"]] = aggregate(
            [r for r in rows if r["candidate"] == c["id"]]
        )
    eligible = [
        c for c in plan["candidates"] if summaries[c["id"]]["eligible"]
    ]
    # Ties stay deterministic; points determine selection, score only diagnoses it.
    selected = (
        max(eligible, key=lambda c: summaries[c["id"]]["match_points"])["id"]
        if eligible and plan["split"] == "validation"
        else None
    )
    report = dict(
        split=plan["split"],
        selected=selected,
        candidates=summaries,
        ablations=compare_variants(
            plan["candidates"], summaries, plan.get("baseline_variant")
        ),
        all_matches_valid=all(r["status"] == "PASS" for r in rows),
        note="Match points are not an estimate of official Elo. Test results never select a checkpoint.",
    )
    (root / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text())
    for row in [*plan["candidates"], *plan["opponents"]]:
        row["model"] = str((args.plan.parent / row["model"]).resolve())
    for key in ("assets", "lower"):
        plan[key] = str((args.plan.parent / plan[key]).resolve())
    report = run_matrix(plan, args.output)
    print(json.dumps(report, indent=2))
    return 0 if report["all_matches_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
