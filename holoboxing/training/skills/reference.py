"""Pinned, single-robot full-body references for physical skill learning.

This library contains targets, not measured controller rollouts. It never
creates an opponent, locks joints, mirrors a motion or joins the clip ends.
"""

from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

from holoboxing.data.augment_punch import CATALOG
from holoboxing.data.common import sha256
from holoboxing.data.records import load_trajectory
from holoboxing.evaluation.runtime.model_contract import INPUT_LINK_NAMES

# Bounds are in the pinned original 50 Hz files, end-exclusive. Context at
# reversals is retained: a direction label describes the active motion, not
# an assertion that every entry/exit frame has positive commanded velocity.
LOCOMOTION_CLIPS = (
    ("forward_1", "kangning-010", (85, 275), (1.0, 0.0)),
    ("backward_1", "kangning-010", (271, 429), (-1.0, 0.0)),
    ("forward_2", "kangning-010", (425, 574), (1.0, 0.0)),
    ("backward_2", "kangning-010", (570, 765), (-1.0, 0.0)),
    ("left_1", "kangning-011", (80, 203), (0.0, 1.0)),
    ("right_1", "kangning-011", (198, 345), (0.0, -1.0)),
    ("left_2", "kangning-011", (336, 475), (0.0, 1.0)),
    ("right_2", "kangning-011", (470, 650), (0.0, -1.0)),
)
# Virtual task-forward is the audited travel axis for 010 and the clockwise
# perpendicular to the audited leftward travel axis for 011, not pelvis X.
TASK_FORWARD_RADIANS = {
    "kangning-010": math.radians(41.61694687749417),
    "kangning-011": math.atan2(0.920631784334657, -0.390431962924125)
    - math.pi / 2,
}
TENSOR_FIELDS = (
    "root_position",
    "root_quaternion",
    "root_linear_velocity",
    "root_angular_velocity",
    "joint_position",
    "joint_velocity",
    "body_position",
    "body_quaternion",
    "body_linear_velocity",
    "body_angular_velocity",
    "point_position",
    "point_quaternion",
    "point_velocity",
)


def retime_trajectory(trajectory, speed_scale):
    """Resample source time at 50 Hz; scale ALL supplied velocity labels.

    Original arrays/files are never changed. The final sample is the last
    complete control interval (duration rounding is less than 20 ms).
    """
    if not math.isfinite(speed_scale) or speed_scale <= 0:
        raise ValueError("speed_scale must be finite and positive")
    if speed_scale == 1:
        return trajectory
    frames = len(trajectory["joint_position"])
    query = (
        np.arange(int(np.floor((frames - 1) / speed_scale)) + 1) * speed_scale
    )
    if len(query) < 6:
        raise ValueError("Retimed motion needs at least six control frames")
    result = dict(trajectory)
    for key in (
        "root_position",
        "joint_position",
        "root_origin_velocity",
        "root_angular_velocity_world",
        "joint_velocity",
    ):
        value = trajectory[key]
        flat = value.reshape(frames, -1)
        sampled = np.stack(
            [np.interp(query, np.arange(frames), column) for column in flat.T],
            -1,
        ).reshape((len(query), *value.shape[1:]))
        result[key] = sampled * speed_scale if "velocity" in key else sampled
    result["root_quaternion"] = np.stack(
        [
            Slerp(
                np.arange(frames),
                Rotation.from_quat(
                    trajectory["root_quaternion"][:, role, [1, 2, 3, 0]]
                ),
            )(query).as_quat()[:, [3, 0, 1, 2]]
            for role in range(trajectory["roles"])
        ],
        1,
    )
    result["amp_root_quaternion"] = result["root_quaternion"].copy()
    return result


def _model_indices(model, joint_names, body_names):
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("References require one floating 29-joint robot")
    if len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("Require 29 unique scene-ordered joint names")
    if not body_names or len(set(body_names)) != len(body_names):
        raise ValueError("Require unique body names")
    pelvis = model.body("pelvis")
    root_joint = int(pelvis.jntadr[0])
    if model.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Pelvis must have a free joint")
    joints = np.array([model.joint(name).id for name in joint_names])
    if np.any(model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("Named controlled joints cannot be free joints")
    return {
        "root_qi": int(model.jnt_qposadr[root_joint]),
        "root_vi": int(model.jnt_dofadr[root_joint]),
        "qi": model.jnt_qposadr[joints],
        "vi": model.jnt_dofadr[joints],
        "bodies": np.array([model.body(name).id for name in body_names]),
        "points": np.array([model.body(name).id for name in INPUT_LINK_NAMES]),
    }


def reference_kinematics(model, trajectory, joint_names, body_names, bounds):
    """Reconstruct bodies with current-state FK and origin velocities."""
    index = _model_indices(model, joint_names, body_names)
    if trajectory["roles"] != 1:
        raise ValueError("Skill references must contain exactly one robot")
    start, end = bounds
    count = len(trajectory["joint_position"])
    if not (0 <= start < end <= count) or end - start < 6:
        raise ValueError("Invalid 50 Hz source bounds")
    result = {
        "root_position": trajectory["root_position"][start:end, 0].copy(),
        "root_quaternion": trajectory["root_quaternion"][start:end, 0].copy(),
        "root_linear_velocity": trajectory["root_origin_velocity"][
            start:end, 0
        ].copy(),
        "root_angular_velocity": trajectory["root_angular_velocity_world"][
            start:end, 0
        ].copy(),
        "joint_position": trajectory["joint_position"][start:end, 0].copy(),
        "joint_velocity": trajectory["joint_velocity"][start:end, 0].copy(),
    }
    # A common XY translation preserves source height and body orientation.
    result["root_position"][:, :2] -= result["root_position"][0, :2].copy()
    frame_count = end - start
    for key, width in (
        ("position", 3),
        ("quaternion", 4),
        ("linear_velocity", 3),
        ("angular_velocity", 3),
    ):
        result[f"body_{key}"] = np.empty((frame_count, len(body_names), width))
    for key, width in (("position", 3), ("quaternion", 4), ("velocity", 3)):
        result[f"point_{key}"] = np.empty((frame_count, 5, width))
    scratch = mujoco.MjData(model)
    qi, vi = index["root_qi"], index["root_vi"]
    for frame in range(frame_count):
        scratch.qpos[qi : qi + 3] = result["root_position"][frame]
        scratch.qpos[qi + 3 : qi + 7] = result["root_quaternion"][frame]
        scratch.qpos[index["qi"]] = result["joint_position"][frame]
        rotation = Rotation.from_quat(
            result["root_quaternion"][frame, [1, 2, 3, 0]]
        ).as_matrix()
        scratch.qvel[vi : vi + 3] = result["root_linear_velocity"][frame]
        scratch.qvel[vi + 3 : vi + 6] = (
            rotation.T @ result["root_angular_velocity"][frame]
        )
        scratch.qvel[index["vi"]] = result["joint_velocity"][frame]
        mujoco.mj_forward(model, scratch)
        for name, ids in (
            ("body", index["bodies"]),
            ("point", index["points"]),
        ):
            position = scratch.xpos[ids]
            omega = scratch.cvel[ids, :3]
            center = scratch.subtree_com[model.body_rootid[ids]]
            velocity = scratch.cvel[ids, 3:] + np.cross(
                omega, position - center
            )
            result[f"{name}_position"][frame] = position
            result[f"{name}_quaternion"][frame] = scratch.xquat[ids]
            result[
                f"{name}_{'linear_velocity' if name == 'body' else 'velocity'}"
            ][frame] = velocity
            if name == "body":
                result["body_angular_velocity"][frame] = omega
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("Nonfinite source kinematics")
    return result


def _verified_source(source_root, group, catalog):
    rows = [row for row in catalog["files"] if Path(row["path"]).stem == group]
    if len(rows) != 1:
        raise ValueError(f"Expected one pinned original source: {group}")
    row = rows[0]
    path = (source_root / row["path"]).resolve(strict=True)
    if not path.is_relative_to(source_root) or sha256(path) != row["sha256"]:
        raise ValueError(
            f"Original source differs from pinned catalog: {group}"
        )
    with np.load(path, allow_pickle=False) as packet:
        fps = np.asarray(packet["fps"])
        if (
            fps.size != 1
            or not np.isfinite(fps).all()
            or float(fps.reshape(-1)[0]) != 50.0
        ):
            raise ValueError(
                "Pinned skill sources must be original 50 Hz motion"
            )
    return path, row


class MotionLibrary:
    """Canonical references and clamped batched lookahead, with no looping."""

    def __init__(
        self,
        source_root,
        model,
        joint_names,
        body_names,
        *,
        skill,
        device="cpu",
        punch_groups=("kangning-001", "kangning-003"),
        locomotion_mode="directional",
        speed_scale=1.0,
        punch_bounds=None,
    ):
        if skill not in {"punch", "locomotion"}:
            raise ValueError("Choose punch or locomotion")
        if punch_bounds is not None and (
            skill != "punch"
            or len(punch_bounds) != 2
            or any(type(v) is not int for v in punch_bounds)
            or not 0 <= punch_bounds[0] < punch_bounds[1]
        ):
            raise ValueError(
                "punch_bounds must be a punch-only [start,end) source interval"
            )
        if skill == "locomotion" and locomotion_mode not in {
            "directional",
            "full",
            "forward",
        }:
            raise ValueError("Choose directional, full or forward locomotion")
        self.device = torch.device(device)
        if not math.isfinite(speed_scale) or speed_scale <= 0:
            raise ValueError("speed_scale must be finite and positive")
        self.joint_names, self.body_names = (
            tuple(joint_names),
            tuple(body_names),
        )
        _model_indices(model, self.joint_names, self.body_names)
        source_root = Path(source_root).resolve(strict=True)
        catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
        if not punch_groups or len(set(punch_groups)) != len(punch_groups):
            raise ValueError("Punch groups must be a nonempty unique sequence")
        if skill == "punch":
            descriptions = tuple(
                (
                    "punch_" + group.removeprefix("kangning-"),
                    group,
                    punch_bounds,
                    (0.0, 0.0),
                )
                for group in punch_groups
            )
        elif locomotion_mode == "full":
            descriptions = tuple(
                ("footwork_" + number, "kangning-" + number, None, (0.0, 0.0))
                for number in ("010", "011")
            )
        else:
            descriptions = tuple(
                row
                for row in LOCOMOTION_CLIPS
                if locomotion_mode == "directional"
                or row[0].startswith("forward_")
            )
        groups = {row[1] for row in descriptions}
        sources = {}
        for group in sorted(groups):
            path, row = _verified_source(source_root, group, catalog)
            sources[group] = (load_trajectory(path, self.joint_names), row)
        self.clips = []
        summaries = []
        for name, group, bounds, direction in descriptions:
            trajectory, row = sources[group]
            bounds = bounds or (0, len(trajectory["joint_position"]))
            source_bounds = bounds
            if (
                not 0
                <= bounds[0]
                < bounds[1]
                <= len(trajectory["joint_position"])
            ):
                raise ValueError("Source bounds exceed original motion")
            if speed_scale != 1:
                trajectory = dict(trajectory)
                for key, value in tuple(trajectory.items()):
                    if isinstance(value, np.ndarray):
                        trajectory[key] = value[bounds[0] : bounds[1]].copy()
                trajectory = retime_trajectory(trajectory, speed_scale)
                bounds = (0, len(trajectory["joint_position"]))
            arrays = reference_kinematics(
                model, trajectory, self.joint_names, self.body_names, bounds
            )
            initial_rotation = Rotation.from_quat(
                arrays["root_quaternion"][0, [1, 2, 3, 0]]
            ).as_matrix()
            heading = TASK_FORWARD_RADIANS.get(
                group,
                math.atan2(initial_rotation[1, 0], initial_rotation[0, 0]),
            )
            net = (
                arrays["root_position"][-1, :2]
                - arrays["root_position"][0, :2]
            )
            direction_world = (
                np.array(
                    [
                        [math.cos(heading), -math.sin(heading)],
                        [math.sin(heading), math.cos(heading)],
                    ]
                )
                @ direction
            )
            directional = skill == "locomotion" and locomotion_mode != "full"
            if directional and float(net @ direction_world) <= 0.2:
                raise ValueError(
                    f"Source motion does not match audited direction: {name}"
                )
            summary = {
                "name": name,
                "group": group,
                "source_path": row["path"],
                "source_sha256": row["sha256"],
                "source_bounds": list(source_bounds),
                "frames": bounds[1] - bounds[0],
                "direction": list(direction),
                "guard_heading_world": heading,
                "direction_frame": "audited task-forward, not pelvis-X",
                "net_displacement_m": net.tolist(),
                "entry_source_speed_mps": float(
                    np.linalg.norm(arrays["root_linear_velocity"][0, :2])
                ),
                "exit_source_speed_mps": float(
                    np.linalg.norm(arrays["root_linear_velocity"][-1, :2])
                ),
                "neutral_start_and_stationary_end_verified": False,
            }
            if skill == "locomotion":
                summary["navigation_direction_primitive"] = directional
            clip = {
                **summary,
                **{
                    key: torch.tensor(
                        value, dtype=torch.float32, device=self.device
                    )
                    for key, value in arrays.items()
                },
            }
            clip["direction"] = torch.tensor(
                direction, dtype=torch.float32, device=self.device
            )
            clip["motion_axis_world"] = torch.tensor(
                direction_world, dtype=torch.float32, device=self.device
            )
            self.clips.append(clip)
            summaries.append(summary)
        self.names = [clip["name"] for clip in self.clips]
        self.lengths = torch.tensor(
            [clip["frames"] for clip in self.clips],
            dtype=torch.long,
            device=self.device,
        )
        self.directions = torch.stack(
            [clip["direction"] for clip in self.clips]
        )
        self.guard_headings = torch.tensor(
            [clip["guard_heading_world"] for clip in self.clips],
            dtype=torch.float32,
            device=self.device,
        )
        max_frames = int(self.lengths.max())
        self._packed = {
            key: torch.stack(
                [
                    clip[key][
                        torch.arange(max_frames, device=self.device).clamp_max(
                            clip["frames"] - 1
                        )
                    ]
                    for clip in self.clips
                ]
            )
            for key in TENSOR_FIELDS
        }
        self.manifest = {
            "format": "solo_skill_reference_v1",
            "skill": skill,
            "catalog_sha256": sha256(CATALOG),
            "source_root": str(source_root),
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "clips": summaries,
            "fps": 50,
            "speed_scale": speed_scale,
            "physics_rollouts_collected": False,
            "coordinate_transform": (
                "subtract clip initial root XY from all positions; "
                "preserve all orientations and heights"
            ),
            "clip_end": "clamp lookahead with validity mask; no cyclic splice",
            "phase_condition_offsets": [0, 5, 10, 25],
            "lower_target_offsets": [1, 2, 3, 4, 5],
        }
        if skill == "locomotion":
            self.manifest["locomotion_mode"] = locomotion_mode
            self.manifest["navigation_direction_primitives"] = (
                locomotion_mode != "full"
            )
        self.manifest["sha256"] = hashlib.sha256(
            json.dumps(
                self.manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()

    def gather(self, clip_ids, frame_ids, offsets=(0,)):
        """Return [B,T,...] targets and validity for integer frame phases."""
        clips = torch.as_tensor(clip_ids, device=self.device)
        frames = torch.as_tensor(frame_ids, device=self.device)
        shifts = torch.as_tensor(offsets, device=self.device)
        if (
            clips.ndim != 1
            or frames.shape != clips.shape
            or shifts.ndim != 1
            or shifts.numel() == 0
        ):
            raise ValueError("Expected clip_ids[B], frame_ids[B], offsets[T]")
        for value in (clips, frames, shifts):
            if (
                value.dtype not in (torch.int32, torch.int64)
                or (value < 0).any()
            ):
                raise ValueError(
                    "Reference indices must be nonnegative integers"
                )
        if (clips >= len(self.clips)).any():
            raise ValueError("Unknown skill clip index")
        query = frames[:, None] + shifts[None]
        lengths = self.lengths[clips, None]
        valid = query < lengths
        query = torch.minimum(query, lengths - 1)
        return {
            **{
                key: values[clips[:, None], query]
                for key, values in self._packed.items()
            },
            "valid": valid,
        }
