"""Run three distances for one selected longer 045 combo, without solo work."""

# ruff: noqa: E402

import argparse
import json
import math
import os
import platform
import sys
import time
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--clip-id", type=int, required=True)
    parser.add_argument(
        "--distances", type=float, nargs="+", default=[0.50, 0.575, 0.65]
    )
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value <= 0 for value in args.distances
    ) or len({f"{value:.3f}" for value in args.distances}) != len(
        args.distances
    ):
        raise ValueError("Distances must be positive, finite and distinct")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    segments = selection["segments"]
    if not segments or not 0 <= args.clip_id < len(segments):
        raise ValueError("Invalid clip ID")
    if any(
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
    write, digest = reference_module.write_json, reference_module.sha256
    write(args.output / "selection.json", selection)
    paths = {
        name: reference_module.BASE
        / "holoboxing/training/skills"
        / (name + ".py")
        for name in ("actor", "train", "env", "physics", "reference")
    }
    hashes = {name: digest(path) for name, path in paths.items()}
    selected = segments[args.clip_id]
    started = time.monotonic()
    summary = {
        "format": "other045_impact_shard_v1",
        "state": "starting",
        "code": code_identity(),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": sys.executable,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "script_sha256": digest(__file__),
        "selection_sha256": digest(args.selection),
        "helper_sha256": {
            name: digest(Path(__file__).parent / name)
            for name in (
                "combo45_reference.py",
                "combo45_impact.py",
                "impulse_metrics.py",
            )
        },
        "frozen_source_sha256": hashes,
        "clip_id": args.clip_id,
        "clip": selected["name"],
        "scope": "045 impact shard only; frozen lower; no training",
        "solo_evaluation_run": False,
        "entry_s": 0.4,
        "artificial_endpoint_hold_s": 0,
        "midclip_resets": False,
        "distances_m": args.distances,
        "cases": [],
    }
    write(args.output / "summary.json", summary)
    physics = None
    try:
        torch.set_num_threads(1)
        torch.manual_seed(92)
        resources, _, _ = resolve_resources(
            args.assets,
            args.lower,
        )
        physics = impact.ImpactPhysics(
            SimpleNamespace(resources=resources),
            args.lower,
        )
        library = reference_module.ComboReferenceLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            device="cuda:0",
        )
        write(args.output / "reference-manifest.json", library.manifest)
        summary.update(
            state="running",
            lower_and_fk_metadata=physics.solo.metadata,
            physics_backend="native MuJoCo 200 Hz; frozen ONNX on GPU",
            opponent_total_mass_kg=float(
                sum(
                    physics.pair.host.body_mass[body]
                    for body in range(physics.pair.host.nbody)
                    if physics.pair.host.body(body).name.startswith("p2_")
                )
            ),
        )
        write(args.output / "summary.json", summary)
        for distance in summary["distances_m"]:
            name = f"{args.clip_id:02d}-{selected['name']}-d{distance:.3f}"
            print(json.dumps({"starting": name}), flush=True)
            case = impact.run_case(
                physics,
                library,
                args.clip_id,
                distance,
                args.output / name,
                record=not args.no_video,
                settle_steps=0,
                aim_events=selected["aim_events"],
            )
            summary["cases"].append(case)
            summary["wall_seconds"] = time.monotonic() - started
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
        assert hashes == {name: digest(path) for name, path in paths.items()}
        summary.update(
            state="completed",
            original_five_source_files_unchanged=True,
            wall_seconds=time.monotonic() - started,
        )
        write(args.output / "summary.json", summary)
        print(
            json.dumps({"state": "completed", "cases": len(summary["cases"])}),
            flush=True,
        )
    except BaseException as error:
        summary.update(
            state="failed",
            error_type=type(error).__name__,
            error=str(error),
            wall_seconds=time.monotonic() - started,
        )
        write(args.output / "summary.json", summary)
        raise
    finally:
        if physics is not None:
            physics.close()


if __name__ == "__main__":
    main()
