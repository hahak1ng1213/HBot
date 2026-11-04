"""045 short-combo physical baseline, without a learned high-level policy."""

# ruff: noqa: E402

import argparse
import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from holoboxing.data.augment_punch import CATALOG
from holoboxing.data.common import sha256
from holoboxing.data.records import load_trajectory
from holoboxing.experiments.provenance import code_identity
from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.resources import resolve_resources
from holoboxing.training.skills.env import SkillTrackingEnv
from holoboxing.training.skills.evaluate import evaluate_skills
from holoboxing.training.skills.physics import SoloPhysics
from holoboxing.training.skills.reference import (
    TENSOR_FIELDS,
    MotionLibrary,
    _verified_source,
    reference_kinematics,
)

BASE = Path(__file__).resolve().parents[2]

SEGMENTS = (
    ("short_arc_pair", 135, 225),
    ("longer_extension_pair", 440, 506),
    ("dense_alternation", 225, 340),
)
EVENTS = {
    "short_arc_pair": (("right", 166), ("left", 186)),
    "longer_extension_pair": (("right", 463), ("left", 477)),
    "dense_alternation": (
        ("right", 252),
        ("left", 266),
        ("right", 285),
        ("left", 308),
    ),
}


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def root_yaw(q):
    w, x, y, z = np.asarray(q).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotate_world(arrays, angle):
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c2, s2 = math.cos(angle / 2), math.sin(angle / 2)
    result = {}
    for key, value in arrays.items():
        if key.startswith("joint"):
            result[key] = value.copy()
        elif "quaternion" in key:
            w, x, y, z = np.moveaxis(value, -1, 0)
            result[key] = np.stack(
                (
                    c2 * w - s2 * z,
                    c2 * x - s2 * y,
                    c2 * y + s2 * x,
                    c2 * z + s2 * w,
                ),
                axis=-1,
            )
        else:
            result[key] = np.einsum("ij,...j->...i", rotation, value)
    return result


class ComboReferenceLibrary(MotionLibrary):
    """Use canonical FK/gather, preserving all original joint/root motion."""

    def __init__(self, source_root, model, joint_names, body_names, *, device):
        self.device = torch.device(device)
        self.joint_names, self.body_names = (
            tuple(joint_names),
            tuple(body_names),
        )
        source_root = Path(source_root).resolve(strict=True)
        path, row = _verified_source(
            source_root, "kangning-045", json.loads(CATALOG.read_text())
        )
        trajectory = load_trajectory(path, self.joint_names)
        self.clips, summaries = [], []
        for name, start, end in SEGMENTS:
            original = reference_kinematics(
                model,
                trajectory,
                self.joint_names,
                self.body_names,
                (start, end),
            )
            heading = float(root_yaw(original["root_quaternion"])[0])
            arrays = rotate_world(original, -heading)
            restored = rotate_world(arrays, heading)
            for key in TENSOR_FIELDS:
                np.testing.assert_allclose(
                    restored[key], original[key], atol=2e-7, rtol=2e-7
                )
            np.testing.assert_array_equal(
                arrays["joint_position"],
                trajectory["joint_position"][start:end, 0],
            )
            np.testing.assert_array_equal(
                arrays["joint_velocity"],
                trajectory["joint_velocity"][start:end, 0],
            )
            assert abs(root_yaw(arrays["root_quaternion"])[0]) < 1e-6
            assert np.linalg.norm(arrays["root_position"][0, :2]) < 1e-7
            summary = {
                "name": name,
                "group": "kangning-045",
                "source_path": row["path"],
                "source_sha256": row["sha256"],
                "source_bounds": [start, end],
                "frames": end - start,
                "direction": [0.0, 0.0],
                "guard_heading_world": 0.0,
                "original_initial_pelvis_yaw_rad": heading,
                "world_yaw_rotation_rad": -heading,
                "candidate_events_are_not_verified_hits": True,
                "candidate_events": list(EVENTS[name]),
                "endpoint_stationarity_verified": False,
            }
            self.clips.append(
                {
                    **summary,
                    **{
                        key: torch.as_tensor(
                            value, device=self.device, dtype=torch.float32
                        )
                        for key, value in arrays.items()
                    },
                    "direction": torch.zeros(2, device=self.device),
                    "motion_axis_world": torch.tensor(
                        [1.0, 0.0], device=self.device
                    ),
                }
            )
            summaries.append(summary)
        self.names = [clip["name"] for clip in self.clips]
        self.lengths = torch.tensor(
            [clip["frames"] for clip in self.clips],
            device=self.device,
            dtype=torch.long,
        )
        self.directions = torch.stack(
            [clip["direction"] for clip in self.clips]
        )
        self.guard_headings = torch.zeros(len(self.clips), device=self.device)
        frames = torch.arange(int(self.lengths.max()), device=self.device)
        self._packed = {
            key: torch.stack(
                [
                    clip[key][frames.clamp_max(clip["frames"] - 1)]
                    for clip in self.clips
                ]
            )
            for key in TENSOR_FIELDS
        }
        self.manifest = {
            "format": "combo45_original_reference_baseline_v1",
            "skill": "punch",
            "source_root": str(source_root),
            "catalog_sha256": sha256(CATALOG),
            "clips": summaries,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "fps": 50,
            "speed_scale": 1.0,
            "coordinate_transform": (
                "subtract crop initial root XY; rotate all world "
                "poses/velocities by negative initial pelvis yaw"
            ),
            "original_joint_values_preserved": True,
            "rigid_transform_roundtrip_checked": True,
            "phase_condition_offsets": [0, 5, 10, 25],
            "lower_target_offsets": [1, 2, 3, 4, 5],
            "clip_end": "clamp; no cyclic splice",
            "physics_rollouts_collected": False,
        }
        self.manifest["sha256"] = hashlib.sha256(
            json.dumps(
                self.manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()


def analyze(output, library):
    measurements = []
    for initialization in ("reference", "neutral"):
        for clip_id, clip in enumerate(library.clips):
            directory = (
                output / initialization / f"{clip_id:02d}-{clip['name']}"
            )
            report = json.loads((directory / "report.json").read_text())
            with np.load(
                directory / "states.npz", allow_pickle=False
            ) as states:
                phase = states["source_frame"]
                active = (phase >= 0) & (phase < clip["frames"] - 1)
                source = {
                    key.removeprefix("metric_"): {
                        "mean": np.mean(states[key][active], axis=0).tolist()
                        if active.any()
                        else None
                    }
                    for key in states.files
                    if key.startswith("metric_")
                }
                speed = states["metric_wrist_arm_speed_mps"]
                ref_speed = states["metric_reference_wrist_arm_speed_mps"]
                event_metrics = []
                for hand, absolute_frame in EVENTS[clip["name"]]:
                    hand_index = 0 if hand == "left" else 1
                    frame = absolute_frame - clip["source_bounds"][0]
                    window = active & (np.abs(phase - frame) <= 7)
                    indices = np.flatnonzero(window)
                    if len(indices):
                        ai = indices[np.argmax(speed[indices, hand_index])]
                        ri = indices[np.argmax(ref_speed[indices, hand_index])]
                        event_metrics.append(
                            {
                                "hand": hand,
                                "source_candidate_frame": absolute_frame,
                                "window_s": 0.14,
                                "reference_peak_mps": float(
                                    ref_speed[ri, hand_index]
                                ),
                                "actual_peak_mps": float(
                                    speed[ai, hand_index]
                                ),
                                "actual_minus_reference_peak_s": float(
                                    (phase[ai] - phase[ri]) * 0.02
                                ),
                                "peak_ratio": float(
                                    speed[ai, hand_index]
                                    / max(ref_speed[ri, hand_index], 1e-8)
                                ),
                            }
                        )
                q = states["actual_root_quaternion"]
                tilt = np.arccos(
                    np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)
                )
                stable = (
                    (
                        np.linalg.norm(
                            states["actual_root_linear_velocity"][:, :2],
                            axis=-1,
                        )
                        < 0.2
                    )
                    & (
                        np.linalg.norm(
                            states["actual_root_angular_velocity"], axis=-1
                        )
                        < 0.4
                    )
                    & (tilt < 0.5)
                    & ~states["actual_fallen"].astype(bool)
                )
                last50 = np.arange(max(0, len(phase) - 50), len(phase))
                measurements.append(
                    {
                        "initialization": initialization,
                        "clip": clip["name"],
                        "source_bounds": clip["source_bounds"],
                        "completed_clip_and_settle": report[
                            "completed_clip_and_settle"
                        ],
                        "fallen_samples": report["metrics"]["fallen_samples"],
                        "active_frames_observed": int(active.sum()),
                        "source_frames": clip["frames"],
                        "active_wrist_relative_error_cm": source[
                            "wrist_relative_error_m"
                        ]["mean"]
                        * 100
                        if active.any()
                        else None,
                        "active_world_wrist_error_cm": source["wrist_error_m"][
                            "mean"
                        ]
                        * 100
                        if active.any()
                        else None,
                        "active_shoulder_error_deg": math.degrees(
                            source["shoulder_error_rad"]["mean"]
                        )
                        if active.any()
                        else None,
                        "active_elbow_error_deg": math.degrees(
                            source["elbow_error_rad"]["mean"]
                        )
                        if active.any()
                        else None,
                        "active_root_error_cm": source["root_error_m"]["mean"]
                        * 100
                        if active.any()
                        else None,
                        "active_wrist_relative_velocity_error_mps": source[
                            "wrist_relative_velocity_mps"
                        ]["mean"],
                        "contact_conditioned_foot_speed_mps": report[
                            "metrics"
                        ]["contact_conditioned_foot_speed_mps"],
                        "last_1s_stable": bool(
                            len(last50) == 50
                            and (phase[last50] >= clip["frames"]).all()
                            and stable[last50].all()
                        ),
                        "candidate_events": event_metrics,
                        "directory": str(directory),
                    }
                )
    return measurements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--entry-steps", default=20, type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if args.analyze_existing:
        receipt = json.loads((args.output / "receipt.json").read_text())
        manifest = json.loads(
            (args.output / "reference-manifest.json").read_text()
        )
        for initialization in ("reference", "neutral"):
            summary = json.loads(
                (args.output / initialization / "summary.json").read_text()
            )
            if summary["state"] != "completed":
                raise RuntimeError("Physical evaluation is not complete")
        for path, digest in receipt["frozen_source_sha256"].items():
            if sha256(path) != digest:
                raise RuntimeError("Frozen source changed")
        measurements = analyze(
            args.output, SimpleNamespace(clips=manifest["clips"])
        )
        write_json(
            args.output / "measurements.json",
            {
                "kind": "045_original_reference_physical_baseline",
                "measurements": measurements,
            },
        )
        receipt.update(
            state="completed",
            postprocess_recovered=True,
            analyzer_sha256=sha256(__file__),
            physical_rollouts_repeated=False,
            original_five_source_files_unchanged=True,
        )
        write_json(args.output / "receipt.json", receipt)
        print(json.dumps({"state": "completed", "cases": len(measurements)}))
        return
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(90)
    frozen_paths = [
        BASE / "holoboxing/training/skills" / (name + ".py")
        for name in ("actor", "train", "env", "physics", "reference")
    ]
    hashes = {str(path): sha256(path) for path in frozen_paths}
    receipt = {
        "state": "starting",
        "code": code_identity(),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": sys.executable,
        "script_sha256": sha256(__file__),
        "frozen_source_sha256": hashes,
        "gpu_visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "seed": 90,
        "entry_seconds_neutral_only": args.entry_steps * 0.02,
        "settle_seconds": 2.0,
        "lower_controller_frozen": True,
        "high_level_training": False,
        "high_level_residual_zero": True,
        "midclip_resets": False,
        "reference_initialization_cold_controller_cache": True,
        "limitations": [
            "short crops only, no opponent or hit test",
            "reference state reset is diagnostic, not deployment",
            "cold controller cache; not an upper bound on achievable tracking",
            "endpoint hold tests a cropped pose, not a learned recovery",
        ],
    }
    write_json(args.output / "receipt.json", receipt)
    physics = None
    try:
        resources, _, _ = resolve_resources(
            args.assets,
            args.lower,
        )
        physics = SoloPhysics(
            SimpleNamespace(resources=resources),
            args.lower,
            1,
            "cuda:0",
        )
        library = ComboReferenceLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            device="cuda:0",
        )
        write_json(args.output / "reference-manifest.json", library.manifest)
        env = SkillTrackingEnv(
            physics,
            library,
            seed=90,
            entry_steps=args.entry_steps,
            settle_steps=100,
        )
        for initialization in ("reference", "neutral"):
            print(
                json.dumps(
                    {"phase": "evaluate", "initialization": initialization}
                ),
                flush=True,
            )
            evaluate_skills(
                env,
                None,
                args.output / initialization,
                record=not args.no_video,
                initialization=initialization,
            )
        measurements = analyze(args.output, library)
        write_json(
            args.output / "measurements.json",
            {
                "kind": "045_original_reference_physical_baseline",
                "measurements": measurements,
            },
        )
        assert hashes == {str(path): sha256(path) for path in frozen_paths}
        receipt.update(
            state="completed", original_five_source_files_unchanged=True
        )
        write_json(args.output / "receipt.json", receipt)
        print(
            json.dumps({"state": "completed", "measurements": measurements}),
            flush=True,
        )
    except BaseException as error:
        receipt.update(
            state="failed", error_type=type(error).__name__, error=str(error)
        )
        write_json(args.output / "receipt.json", receipt)
        raise
    finally:
        if physics is not None:
            physics.close()


if __name__ == "__main__":
    main()
