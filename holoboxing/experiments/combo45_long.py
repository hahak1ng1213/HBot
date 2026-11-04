"""Longer, previously untested 045 combos: solo tracking and paired impacts."""

# ruff: noqa: E402

import argparse
import json
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from holoboxing.experiments import (
    combo45_impact as impact,
    combo45_reference as reference_module,
)
from holoboxing.experiments.provenance import code_identity
from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.resources import resolve_resources
from holoboxing.training.skills.env import SkillTrackingEnv
from holoboxing.training.skills.evaluate import evaluate_skills
from holoboxing.training.skills.physics import SoloPhysics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    segments = selection["segments"]
    if not segments or any(
        not (0 <= s["source_bounds"][0] < s["source_bounds"][1] <= 1475)
        for s in segments
    ):
        raise ValueError("Invalid source selection")
    reference_module.SEGMENTS = tuple(
        (s["name"], *s["source_bounds"]) for s in segments
    )
    reference_module.EVENTS.clear()
    reference_module.EVENTS.update(
        {s["name"]: tuple(tuple(e) for e in s["events"]) for s in segments}
    )
    args.output.mkdir(parents=True, exist_ok=False)
    write = reference_module.write_json
    digest = reference_module.sha256
    write(args.output / "selection.json", selection)
    torch.set_num_threads(1)
    torch.manual_seed(92)
    resources, _, _ = resolve_resources(
        args.assets,
        args.lower,
    )
    assets = SimpleNamespace(resources=resources)
    lower = args.lower
    paths = {
        n: reference_module.BASE / "holoboxing/training/skills" / (n + ".py")
        for n in ("actor", "train", "env", "physics", "reference")
    }
    hashes = {n: digest(p) for n, p in paths.items()}
    summary = {
        "state": "running",
        "code": code_identity(),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": sys.executable,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": digest(__file__),
        "selection_sha256": digest(args.selection),
        "helper_sha256": {
            n: digest(Path(__file__).parent / n)
            for n in (
                "combo45_reference.py",
                "combo45_impact.py",
                "impulse_metrics.py",
            )
        },
        "frozen_source_sha256": hashes,
        "scope": "045 only; original speed; frozen lower; no training",
        "entry_s": 0.4,
        "artificial_endpoint_hold_s": 0,
        "midclip_resets": False,
        "cases": [],
        "limitation": (
            "Stationary-opponent test without learned aiming or "
            "distance correction; source movement is preserved."
        ),
    }
    write(args.output / "summary.json", summary)
    physics = None
    try:
        print(
            json.dumps({"stage": "solo_tracking", "clips": len(segments)}),
            flush=True,
        )
        physics = SoloPhysics(assets, lower, 1, "cuda:0")
        library = reference_module.ComboReferenceLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            device="cuda:0",
        )
        write(args.output / "reference-manifest.json", library.manifest)
        env = SkillTrackingEnv(
            physics, library, seed=92, entry_steps=20, settle_steps=0
        )
        solo = evaluate_skills(
            env,
            None,
            args.output / "solo",
            record=not args.no_video,
            initialization="neutral",
        )
        summary["solo_all_completed"] = solo[
            "all_clips_completed_without_environment_failure"
        ]
        summary["solo_clips"] = [
            {
                "clip": c["clip"],
                "completed": c["completed_clip_and_settle"],
                "seconds": c["simulation_seconds"],
            }
            for c in solo["clips"]
        ]
        physics.close()
        physics = None
        write(args.output / "summary.json", summary)
        print(
            json.dumps(
                {"stage": "solo_complete", "clips": summary["solo_clips"]}
            ),
            flush=True,
        )
        physics = impact.ImpactPhysics(assets, lower)
        library = reference_module.ComboReferenceLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            device="cuda:0",
        )
        summary["lower_and_fk_metadata"] = physics.solo.metadata
        summary["opponent_total_mass_kg"] = float(
            sum(
                physics.pair.host.body_mass[b]
                for b in range(physics.pair.host.nbody)
                if physics.pair.host.body(b).name.startswith("p2_")
            )
        )
        impact_root = args.output / "impact"
        impact_root.mkdir()
        for clip_id, selected in enumerate(segments):
            for distance in selected["initial_distances_m"]:
                name = f"{clip_id:02d}-{selected['name']}-d{distance:.3f}"
                print(json.dumps({"starting": name}), flush=True)
                case = impact.run_case(
                    physics,
                    library,
                    clip_id,
                    distance,
                    impact_root / name,
                    record=not args.no_video,
                    settle_steps=0,
                    aim_events=selected["aim_events"],
                )
                summary["cases"].append(case)
                write(args.output / "summary.json", summary)
                print(
                    json.dumps(
                        {
                            "finished": name,
                            "state": case["state"],
                            "seconds": case.get("actual_seconds"),
                            "end_reason": case.get("end_reason"),
                        }
                    ),
                    flush=True,
                )
        assert hashes == {n: digest(p) for n, p in paths.items()}
        summary.update(
            state="completed", original_five_source_files_unchanged=True
        )
        write(args.output / "summary.json", summary)
        print(
            json.dumps({"state": "completed", "cases": len(summary["cases"])}),
            flush=True,
        )
    except BaseException as error:
        summary.update(
            state="failed", error_type=type(error).__name__, error=str(error)
        )
        write(args.output / "summary.json", summary)
        raise
    finally:
        if physics is not None:
            physics.close()


if __name__ == "__main__":
    main()
