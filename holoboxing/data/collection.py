"""Measured paired supervision using the frozen lower controller, without an upper model."""

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .common import sha256
from .contract import CompetitionHistory, target_from_world

MEASURED_SOURCE = "measured_frozen_lower_rollout_current_state"
RECIPE = {
    "version": "measured_pair_short_window_cpu_v2",
    "physics_backend": "mujoco_cpu",
    "observation_source": MEASURED_SOURCE,
    "core_frames": 100,
    "warmup_frames": 10,
    "segment_frames": 3000,
    "future_offsets": [1, 2, 3, 4, 5],
    "initial_velocity": "reference_world_linear_local_angular_named_joints",
    "target_translation": "pair_mean_displacement_error_xy_from_reset",
    "velocity": "same_instant_link_origin",
    "target_frame": "actual_current_full_pelvis",
    "cutoff": "first_role_pelvis_or_torso_below_floor_plus_0.3m",
    "early_terminal_tail": "exclude_labels_reaching_failure",
    "remaining_time": "source_global_segment",
}


def windows(frames):
    for segment in range(0, frames, 3000):
        end = min(segment + 3000, frames) - 5
        for core in range(segment, end, 100):
            yield max(segment, core - 10), core, min(core + 100, end)


def sample_indices(frames, roles=2):
    current = np.arange(max(0, frames - 5), dtype=np.int64)
    current = current[current // 3000 == (current + 5) // 3000]
    return np.concatenate(
        [
            np.column_stack((current, np.full_like(current, role)))
            for role in range(roles)
        ]
    )


def reference_velocity(trajectory, frame):
    q = trajectory["root_quaternion"][frame]
    rotation = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()
    angular = np.einsum(
        "rji,rj->ri",
        rotation,
        trajectory["root_angular_velocity_world"][frame],
    )
    return np.concatenate(
        (
            trajectory["root_origin_velocity"][frame],
            angular,
            trajectory["joint_velocity"][frame],
        ),
        axis=-1,
    )


def shared_translation(actual, reference, initial_actual, initial_reference):
    delta = ((actual - initial_actual) - (reference - initial_reference)).mean(
        0
    )
    delta[2] = 0
    return delta


def retained_frames(seen, core, stop):
    if not seen:
        return []
    terminal = seen[-1] + 1
    return [
        f for f in seen if f >= core and (terminal == stop or f + 5 < terminal)
    ]


def expected_retained_indices(frames, failures, roles=2):
    plan = {core: (start, stop) for start, core, stop in windows(frames)}
    cutoffs = {}
    for event in failures:
        core, frame = event["core"], event["frame"]
        if core not in plan or core in cutoffs:
            raise ValueError("Unknown or duplicate failed window")
        start, stop = plan[core]
        if (
            type(frame) is not int
            or not start <= frame < stop
            or not event["roles"]
            or any(r not in (0, 1) for r in event["roles"])
        ):
            raise ValueError("Invalid early-failure record")
        cutoffs[core] = frame
    accepted = []
    for core, (_, stop) in plan.items():
        accepted.extend(
            range(
                core, min(stop, cutoffs[core] - 5) if core in cutoffs else stop
            )
        )
    indices = sample_indices(frames, roles)
    return indices[np.isin(indices[:, 0], accepted)]


def measured_state(model, data, point_ids):
    position = data.xpos[point_ids].copy()
    angular = data.cvel[point_ids, :3]
    center = data.subtree_com[model.body_rootid[point_ids]]
    velocity = data.cvel[point_ids, 3:] + np.cross(angular, position - center)
    omega = np.stack(
        [
            data.xmat[point_ids[r, 0]].reshape(3, 3).T @ angular[r, 0]
            for r in range(2)
        ]
    )
    return position, data.xquat[point_ids].copy(), velocity, omega


def pack_reference(schema, target, height, *, out=None):
    """Fill the existing lower schema, retaining masks and validity bits."""
    points = target.reshape(5, 5, 9)
    template = schema.template.to(target.device)
    value = template.clone() if out is None else out.copy_(template)
    task = value[: schema.task_dim].reshape(5, schema.width)
    task[:, : schema.num_links * 3].reshape(5, schema.num_links, 3)[
        :, schema.active
    ] = points[..., :3]
    task[:, schema.num_links * 3 : -1].reshape(5, schema.num_links, 6)[
        :, schema.active
    ] = points[..., 3:]
    value[schema.task_dim] = height
    return value


def lower_identity(directory):
    root = Path(directory).expanduser().resolve(strict=True)
    return {
        name: sha256(root / name)
        for name in ("exported/model_10500.onnx", "config.yaml")
    }


class MeasuredCollector:
    """Native CPU arena with frozen lower inference on the selected Torch device."""

    def __init__(self, assets, lower, device="cuda:0"):
        from tempfile import TemporaryDirectory

        from holoboxing.training.envs.control import (
            FrozenHoloMotionV2Controller,
        )
        from holoboxing.training.envs.mujoco_physics import CompetitionPhysics
        from holoboxing.training.envs.onnx_lower import OnnxLower

        from .records import scene_layout
        from .robot import robot_file

        self.device = torch.device(device)
        self.physics = CompetitionPhysics(assets, 1, self.device)
        self.scene = SimpleNamespace(
            model=self.physics.models[0], data=self.physics.native[0]
        )
        self.names, self.roots, self.joints, self.point_ids = scene_layout(
            self.scene.model
        )
        self.scratch = TemporaryDirectory(prefix="boxing-cpu-collector-")
        session = OnnxLower(
            Path(lower) / "exported/model_10500.onnx",
            1,
            self.device,
            self.scratch.name,
        )
        names = session.metadata["joint_names"].split(",")
        order = [names.index(n) for n in self.names]
        gains = [
            torch.tensor([float(v) for v in session.metadata[k].split(",")])[
                order
            ]
            for k in ("joint_stiffness", "joint_damping")
        ]
        # Physics uses canonical training joint order; collector uses scene order.
        from holoboxing.training.tasks.combat.config.sandbag import JOINT_NAMES

        self.physics_order = [self.names.index(n) for n in JOINT_NAMES]
        self.physics.set_gains(*(g[self.physics_order] for g in gains))
        self.policies = []
        for side in ("red", "blue"):
            with robot_file(assets.resources.scene, side) as xml:
                self.policies.append(
                    FrozenHoloMotionV2Controller(
                        config_path=Path(lower) / "config.yaml",
                        lower_session=session,
                        robot_file_path=xml,
                        batch_size=1,
                        scene_joint_names=JOINT_NAMES,
                        device=self.device,
                    )
                )
        self.cpu = mujoco.MjData(self.scene.model)
        self.torso = [
            self.scene.model.body(p + "torso_link").id for p in ("p1_", "p2_")
        ]
        self.floor = self.physics.floor_height
        self.recipe = {**RECIPE, "lower_identity": lower_identity(lower)}

    def close(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.scratch.cleanup()

    def reset(self, trajectory, frame):
        qpos = np.concatenate(
            [
                trajectory[key][frame]
                for key in (
                    "root_position",
                    "root_quaternion",
                    "joint_position",
                )
            ],
            -1,
        )
        qvel = reference_velocity(trajectory, frame)
        if trajectory["roles"] == 1:
            qpos = np.concatenate((qpos, trajectory["standing_qpos"][None]))
            qvel = np.concatenate((qvel, np.zeros_like(qvel)))
        self.physics.reset(torch.tensor([0], device=self.device))
        for role, ((qi, vi), (jqi, jvi)) in enumerate(
            zip(self.roots, self.joints, strict=True)
        ):
            self.scene.data.qpos[qi : qi + 7] = qpos[role, :7]
            self.scene.data.qpos[jqi] = qpos[role, 7:]
            self.scene.data.qvel[vi : vi + 6] = qvel[role, :6]
            self.scene.data.qvel[jvi] = qvel[role, 6:]
        mujoco.mj_forward(self.scene.model, self.scene.data)
        self.physics._pull()
        for policy in self.policies:
            policy.reset()

    def collect(self, arrays, trajectory):
        if tuple(trajectory["joint_names"]) != tuple(self.names):
            raise ValueError("Reference and collector joint order differ")
        count = len(arrays["position"])
        obs, targets, row_window, accepted, events = {}, {}, {}, [], []
        max_velocity_error = max_target_error = 0.0
        window_count = 0
        for start, core, stop in windows(count):
            self.reset(trajectory, start)
            history = CompetitionHistory(1)
            seen = []
            initial_actual = initial_reference = None
            for frame in range(start, stop):
                self.cpu.qpos[:] = self.scene.data.qpos
                self.cpu.qvel[:] = self.scene.data.qvel
                if not (
                    np.isfinite(self.cpu.qpos).all()
                    and np.isfinite(self.cpu.qvel).all()
                ):
                    raise ValueError(
                        f"Nonfinite measured state at source frame {frame}"
                    )
                mujoco.mj_forward(self.scene.model, self.cpu)
                state = measured_state(
                    self.scene.model, self.cpu, self.point_ids
                )
                position, quaternion, velocity, _ = state
                low = (
                    np.minimum(position[:, 0, 2], self.cpu.xpos[self.torso, 2])
                    < self.floor + 0.3
                )
                if low.any():
                    events.append(
                        {
                            "frame": frame,
                            "core": core,
                            "roles": np.flatnonzero(low).tolist(),
                        }
                    )
                    break
                if frame - start < 2:
                    for role, slot in np.ndindex(2, 5):
                        jac = np.empty((3, self.scene.model.nv))
                        mujoco.mj_jacBody(
                            self.scene.model,
                            self.cpu,
                            jac,
                            None,
                            int(self.point_ids[role, slot]),
                        )
                        max_velocity_error = max(
                            max_velocity_error,
                            float(
                                np.abs(
                                    jac @ self.cpu.qvel - velocity[role, slot]
                                ).max()
                            ),
                        )
                tensors = [
                    torch.tensor(v[None], dtype=torch.float32) for v in state
                ]
                observation = history.observe(
                    *tensors,
                    torch.tensor([60.0 - (frame % 3000) / 50]),
                    torch.zeros(1, 2),
                    reset=torch.tensor([frame == start]),
                    step=frame - start,
                )
                reference_pelvis = arrays["position"][frame, :, 0].numpy()
                if initial_actual is None:
                    initial_actual = position[:, 0].copy()
                    initial_reference = reference_pelvis.copy()
                shift = shared_translation(
                    position[:, 0],
                    reference_pelvis,
                    initial_actual,
                    initial_reference,
                )
                future = torch.arange(frame + 1, frame + 6)
                world = arrays["position"][future] + torch.tensor(
                    shift, dtype=torch.float32
                )
                target = target_from_world(
                    tensors[0][0],
                    tensors[1][0],
                    world.transpose(0, 1),
                    arrays["quaternion"][future].transpose(0, 1),
                )
                if frame >= core:
                    obs[frame], targets[frame] = (
                        observation[0].numpy(),
                        target.numpy(),
                    )
                    row_window[frame] = (start, core)
                seen.append(frame)
                # Feed identical world targets to the same frozen controller used in RL.
                from holoboxing.training.tasks.combat.mdp.geometry import (
                    decode_output,
                )
                from holomotion.src.utils.torch_utils import quat_from_matrix

                p = tensors[0][0].to(self.device)
                q = tensors[1][0].to(self.device)
                decoded_p, decoded_r = decode_output(
                    p, q, target.to(self.device), action_dim=225
                )
                max_target_error = max(
                    max_target_error,
                    float(
                        (decoded_p.cpu() - world.transpose(0, 1)).abs().max()
                    ),
                )
                decoded_q = torch.nn.functional.normalize(
                    quat_from_matrix(decoded_r), dim=-1
                )
                for role, policy in enumerate(self.policies):
                    jqi, jvi = self.joints[role]

                    def tensor(value):
                        return torch.as_tensor(
                            np.asarray(value).copy(),
                            device=self.device,
                            dtype=torch.float32,
                        )[None]

                    jt = policy(
                        reference_position_w=decoded_p[role : role + 1],
                        reference_quaternion_wxyz=decoded_q[role : role + 1],
                        current_reference_root_position_w=p[
                            role : role + 1, 0
                        ],
                        reference_root_height=p[role : role + 1, 0, 2:3],
                        robot_root_quaternion_wxyz=q[role : role + 1, 0],
                        robot_root_angular_velocity_b=tensor(state[3][role]),
                        robot_joint_position=tensor(
                            self.scene.data.qpos[jqi][self.physics_order]
                        ),
                        robot_joint_velocity=tensor(
                            self.scene.data.qvel[jvi][self.physics_order]
                        ),
                    )
                    # Controller and physics both use canonical ONNX order.
                    self.physics.targets[0, role] = jt[0]
                for _ in range(4):
                    self.physics.substep()
            retained = retained_frames(seen, core, stop)
            accepted.extend(retained)
            window_count += 1
        if max_velocity_error >= 1e-7 or max_target_error >= 3e-5:
            raise ValueError(
                "Measured velocity/target geometry verification failed"
            )
        indices = sample_indices(count, trajectory["roles"])
        planned = len(indices)
        indices = indices[np.isin(indices[:, 0], accepted)]
        if not len(indices):
            raise ValueError(
                "Source produced zero measured SL rows; refusing to drop its AMP motion"
            )
        packet = {
            "obs": np.stack([obs[int(f)][int(r)] for f, r in indices]).astype(
                np.float32
            ),
            "target_pose": np.stack(
                [targets[int(f)][int(r)] for f, r in indices]
            ).astype(np.float32),
            "sample_index": indices,
            "window_index": np.asarray(
                [row_window[int(f)] for f, _ in indices], dtype=np.int64
            ),
        }
        return packet, {
            "observation_source": MEASURED_SOURCE,
            "samples": len(indices),
            "planned_samples": planned,
            "rejected_samples": planned - len(indices),
            "windows": window_count,
            "early_failures": events,
            "max_velocity_jacobian_error": max_velocity_error,
            "max_target_roundtrip_error": max_target_error,
            "collection_recipe": self.recipe,
        }
