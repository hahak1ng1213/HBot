"""Pinned, unwarped turn references for physical preflight and skill learning.

These are demonstration targets, never claims of learned or stable turning.
The existing forward teacher is not compatible with this separate library.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from holoboxing.data.augment_punch import CATALOG
from holoboxing.data.common import sha256
from holoboxing.data.records import load_trajectory
from holoboxing.training.skills.reference import (
    TENSOR_FIELDS,
    MotionLibrary,
    _verified_source,
    reference_kinematics,
)


def root_yaw(quaternions):
    w, x, y, z = np.asarray(quaternions).T
    return np.unwrap(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rotate_world_arrays(arrays, angle):
    """Rigidly change the world heading of the entire named FK trajectory."""
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    c2, s2 = math.cos(angle / 2), math.sin(angle / 2)
    for key, value in arrays.items():
        if key.startswith("joint"):
            continue
        if "quaternion" in key:
            w, x, y, z = np.moveaxis(value, -1, 0)
            arrays[key] = np.stack(
                (
                    c2 * w - s2 * z,
                    c2 * x - s2 * y,
                    c2 * y + s2 * x,
                    c2 * z + s2 * w,
                ),
                axis=-1,
            )
        elif "position" in key or "velocity" in key:
            arrays[key] = np.einsum("ij,...j->...i", rotation, value)


class TurnReferenceLibrary(MotionLibrary):
    """Reuse the canonical FK and gather without modifying teacher sources."""

    def __init__(
        self,
        source_root,
        model,
        joint_names,
        body_names,
        *,
        device="cpu",
        segments=None,
        normalize_initial_yaw=True,
    ):
        self.device = torch.device(device)
        self.joint_names, self.body_names = (
            tuple(joint_names),
            tuple(body_names),
        )
        source_root = Path(source_root).resolve(strict=True)
        if type(normalize_initial_yaw) is not bool:
            raise ValueError("normalize_initial_yaw must be boolean")
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        path, row = _verified_source(source_root, "kangning-028", catalog)
        trajectory = load_trajectory(path, self.joint_names)
        count = len(trajectory["joint_position"])
        segments = (
            (("turn_full", 0, count),) if segments is None else tuple(segments)
        )
        if not segments or any(
            not isinstance(segment, (tuple, list))
            or len(segment) != 3
            or not isinstance(segment[0], str)
            or not segment[0].strip()
            or type(segment[1]) is not int
            or type(segment[2]) is not int
            or not 0 <= segment[1] < segment[2] <= count
            or segment[2] - segment[1] < 6
            for segment in segments
        ):
            raise ValueError(
                "Expected named turn crops with integer bounds and >=6 frames"
            )
        if len({s[0] for s in segments}) != len(segments):
            raise ValueError("Turn segment names must be unique")
        self.clips, summaries = [], []
        for name, start, end in segments:
            arrays = reference_kinematics(
                model,
                trajectory,
                self.joint_names,
                self.body_names,
                (start, end),
            )
            yaw = root_yaw(arrays["root_quaternion"])
            original_heading = float(yaw[0])
            world_rotation = (
                -original_heading if normalize_initial_yaw else 0.0
            )
            if normalize_initial_yaw:
                rotate_world_arrays(arrays, world_rotation)
            heading = float(root_yaw(arrays["root_quaternion"])[0])
            summary = {
                "name": name,
                "group": "kangning-028",
                "source_path": row["path"],
                "source_sha256": row["sha256"],
                "source_bounds": [start, end],
                "frames": end - start,
                "direction": [0.0, 0.0],
                "guard_heading_world": heading,
                "original_initial_pelvis_yaw_rad": original_heading,
                "world_yaw_rotation_rad": world_rotation,
                "guard_axis_calibrated": False,
                "direction_frame": (
                    "relative pelvis yaw; guard axis not calibrated"
                ),
                "source_net_yaw_deg": math.degrees(float(yaw[-1] - yaw[0])),
                "source_yaw_range_deg": math.degrees(float(np.ptp(yaw))),
                "net_displacement_m": arrays["root_position"][-1, :2].tolist(),
                "neutral_start_and_stationary_end_verified": False,
                "navigation_direction_primitive": False,
            }
            self.clips.append(
                {
                    **summary,
                    **{
                        k: torch.as_tensor(
                            v, dtype=torch.float32, device=self.device
                        )
                        for k, v in arrays.items()
                    },
                    "direction": torch.zeros(2, device=self.device),
                    "motion_axis_world": torch.tensor(
                        [math.cos(heading), math.sin(heading)],
                        device=self.device,
                    ),
                }
            )
            summaries.append(summary)
        self.names = [c["name"] for c in self.clips]
        self.lengths = torch.tensor(
            [c["frames"] for c in self.clips],
            dtype=torch.long,
            device=self.device,
        )
        self.directions = torch.stack([c["direction"] for c in self.clips])
        self.guard_headings = torch.tensor(
            [c["guard_heading_world"] for c in self.clips],
            dtype=torch.float32,
            device=self.device,
        )
        frames = torch.arange(int(self.lengths.max()), device=self.device)
        self._packed = {
            k: torch.stack(
                [c[k][frames.clamp_max(c["frames"] - 1)] for c in self.clips]
            )
            for k in TENSOR_FIELDS
        }
        self.manifest = {
            "format": "solo_turn_reference_v1",
            "skill": "locomotion",
            "primitive": "turn",
            "source_root": str(source_root),
            "catalog_sha256": sha256(CATALOG),
            "clips": summaries,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "fps": 50,
            "speed_scale": 1.0,
            "coordinate_transform": (
                "subtract crop initial root XY; rigid world yaw rotation "
                "by negative initial pelvis yaw for all FK poses/velocities"
                if normalize_initial_yaw
                else "subtract each crop initial root XY only"
            ),
            "normalize_initial_yaw": normalize_initial_yaw,
            "relative_turn_angle_modified": False,
            "angle_augmentation": False,
            "mirrored": False,
            "physics_rollouts_collected": False,
            "phase_condition_offsets": [0, 5, 10, 25],
            "lower_target_offsets": [1, 2, 3, 4, 5],
            "navigation_direction_primitives": False,
            "clip_end": "clamp; no cyclic splice",
        }
        self.manifest["sha256"] = hashlib.sha256(
            json.dumps(
                self.manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
