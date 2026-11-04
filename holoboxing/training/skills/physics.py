"""Single G1 on a plane with the pinned, five-point HoloMotion controller.

CPU MuJoCo integrates independent worlds; the frozen lower network is batched
on the requested device. This is a skill-training scene, not match evaluation.
"""

import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holoboxing.data.robot import load_scene_model, robot_file, robot_spec
from holoboxing.evaluation.models import frozen_manifest, verify_bundle
from holoboxing.evaluation.runtime.model_contract import INPUT_LINK_NAMES
from holoboxing.training.envs.control import FrozenHoloMotionV2Controller
from holoboxing.training.envs.onnx_lower import OnnxLower
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


def _arena_floor(assets):
    arena = load_scene_model(assets)
    data = mujoco.MjData(arena)
    mujoco.mj_forward(arena, data)
    floor = arena.geom("smpl_ring_baseLink").id
    height = float(data.geom_xpos[floor, 2] + arena.geom_size[floor, 2])
    return arena, floor, height


def load_solo_model(assets):
    """Create one robot with evaluator joint dynamics and a ground plane."""
    arena, floor, _ = _arena_floor(assets)
    spec = robot_spec(assets.resources.scene, "red")
    for name in (
        "timestep",
        "gravity",
        "integrator",
        "solver",
        "iterations",
        "tolerance",
        "cone",
        "impratio",
        "noslip_iterations",
        "noslip_tolerance",
        "ls_iterations",
        "ls_tolerance",
        "enableflags",
        "disableflags",
    ):
        setattr(spec.option, name, getattr(arena.opt, name))
    spec.option.timestep = 0.005
    spec.worldbody.add_geom(
        name="skill_ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[0, 0, 0.1],
        rgba=[0.35, 0.38, 0.4, 1],
        friction=arena.geom_friction[floor],
        solref=arena.geom_solref[floor],
        solimp=arena.geom_solimp[floor],
        condim=int(arena.geom_condim[floor]),
        contype=int(arena.geom_contype[floor]),
        conaffinity=int(arena.geom_conaffinity[floor]),
        margin=float(arena.geom_margin[floor]),
        gap=float(arena.geom_gap[floor]),
    )
    spec.worldbody.add_light(pos=[0, 0, 5], dir=[0, 0, -1])
    for joint in spec.joints:
        if joint.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        joint.damping = 0.05
        joint.armature = 0.01
        wrist = "_wrist_pitch_" in joint.name or "_wrist_yaw_" in joint.name
        joint.frictionloss = 0.1 if wrist else 0.2
        actuator = spec.add_actuator(name=joint.name.removesuffix("_joint"))
        actuator.set_to_motor()
        actuator.target = joint.name
        actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
        actuator.ctrllimited = True
        actuator.ctrlrange = joint.actfrcrange
    model = spec.compile()
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Solo scene must contain exactly one 29-joint G1")
    return model


def initial_solo_qpos(assets, model):
    """Translate the official red reset onto z=0 ground, centered in xy."""
    _, _, floor_height = _arena_floor(assets)
    reset = json.loads(Path(assets.resources.initial_state).read_text())
    names = reset["dof_names"]
    if len(names) != 29 or set(names) != set(JOINT_NAMES):
        raise ValueError("Initial reset does not describe the G1 joint set")
    saved = np.asarray(reset["qpos"], dtype=np.float64)[0]
    if saved.shape != (36,) or not np.isfinite(saved).all():
        raise ValueError("Invalid official initial pose")
    result = model.qpos0.copy()
    root = model.body("pelvis").jntadr[0]
    start = int(model.jnt_qposadr[root])
    result[start : start + 7] = saved[:7]
    result[start : start + 2] = 0
    result[start + 2] -= floor_height
    for name in JOINT_NAMES:
        result[model.jnt_qposadr[model.joint(name).id]] = saved[
            7 + names.index(name)
        ]
    return result


class SoloPhysics:
    """One real robot per world; no opponent, arena walls, or referee."""

    dt = 0.02
    joint_names = JOINT_NAMES
    body_names = BODY_NAMES
    point_names = INPUT_LINK_NAMES

    def __init__(self, assets, lower, worlds=1, device="cuda:0"):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("worlds must be a positive integer")
        self.device = torch.device(device)
        self.num_envs = worlds
        self.host = load_solo_model(assets)
        self.models = [copy.copy(self.host) for _ in range(worlds)]
        self.native = [mujoco.MjData(model) for model in self.models]
        self.fk = [mujoco.MjData(model) for model in self.models]
        self.initial_qpos = initial_solo_qpos(assets, self.host)
        self.floor_height = 0.0
        self.body_ids = np.asarray(
            [self.host.body(name).id for name in self.body_names]
        )
        self.point_ids = np.asarray(
            [self.host.body(name).id for name in self.point_names]
        )
        joint_ids = [self.host.joint(name).id for name in self.joint_names]
        self.qi = self.host.jnt_qposadr[joint_ids].copy()
        self.vi = self.host.jnt_dofadr[joint_ids].copy()
        actuator_by_joint = {
            int(value[0]): i
            for i, value in enumerate(self.host.actuator_trnid)
        }
        self.ai = np.asarray([actuator_by_joint[j] for j in joint_ids])
        root_joint = int(self.host.body("pelvis").jntadr[0])
        self.root_qi = int(self.host.jnt_qposadr[root_joint])
        self.root_vi = int(self.host.jnt_dofadr[root_joint])
        if self.host.jnt_type[root_joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("The pelvis must have a free joint")
        self.torso_id = self.host.body("torso_link").id
        self.floor_id = self.host.geom("skill_ground").id
        limits = self.host.actuator_ctrlrange[self.ai]
        if not self.host.actuator_ctrllimited[
            self.ai
        ].all() or not np.allclose(limits[:, 0], -limits[:, 1]):
            raise ValueError("Solo motors require symmetric torque limits")
        self.torque_limit = limits[:, 1].copy()
        self._foot_body_sets = []
        for name in ("left_ankle_roll_link", "right_ankle_roll_link"):
            ancestor = self.host.body(name).id
            descendants = set()
            for body in range(self.host.nbody):
                node = body
                while node:
                    if node == ancestor:
                        descendants.add(body)
                        break
                    node = int(self.host.body_parentid[node])
            self._foot_body_sets.append(descendants)
        bundle = verify_bundle(lower)
        identity = frozen_manifest()["files"]
        self.scratch = TemporaryDirectory(prefix="solo-g1-lower-")
        try:
            session = OnnxLower(
                bundle / "exported/model_10500.onnx",
                worlds,
                self.device,
                self.scratch.name,
            )
            names = tuple(session.metadata["joint_names"].split(","))
            if names != self.joint_names:
                raise ValueError(
                    "Solo joint order must match pinned ONNX order"
                )
            self.kp, self.kd = [
                np.asarray(
                    [float(x) for x in session.metadata[key].split(",")]
                )
                for key in ("joint_stiffness", "joint_damping")
            ]
            if any(
                value.shape != (29,)
                or not np.isfinite(value).all()
                or (value < 0).any()
                for value in (self.kp, self.kd)
            ):
                raise ValueError("Invalid frozen lower PD gains")
            with robot_file(assets.resources.scene, "red") as xml:
                self.lower = FrozenHoloMotionV2Controller(
                    config_path=bundle / "config.yaml",
                    lower_session=session,
                    robot_file_path=xml,
                    batch_size=worlds,
                    scene_joint_names=self.joint_names,
                    device=self.device,
                )
        except BaseException:
            self.scratch.cleanup()
            raise
        self.metadata = {
            "scene": "single_g1_plane",
            "robots_per_world": 1,
            "physics_backend": "native_mujoco_cpu",
            "mujoco_version": mujoco.__version__,
            "physics_dt_s": float(self.host.opt.timestep),
            "control_dt_s": self.dt,
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
            "target_points": list(self.point_names),
            "target_offsets": [1, 2, 3, 4, 5],
            "lower_sha256": {
                name: identity[name]["sha256"]
                for name in ("config.yaml", "exported/model_10500.onnx")
            },
            "joint_damping": self.host.dof_damping[self.vi].tolist(),
            "joint_armature": self.host.dof_armature[self.vi].tolist(),
            "joint_frictionloss": self.host.dof_frictionloss[self.vi].tolist(),
            "torque_limits": self.torque_limit.tolist(),
            "ground_friction": self.host.geom_friction[self.floor_id].tolist(),
            "floor_height": 0.0,
            "reset": "official red pose centered in xy, arena floor removed",
            "match_rules_used": False,
        }
        self.steps = 0
        self.failure_state = None
        self._closed = False
        self.targets = np.zeros((worlds, 29))
        for model, data in zip(self.models, self.native, strict=True):
            data.qpos[:] = self.initial_qpos
            mujoco.mj_forward(model, data)
        self.targets[:] = self.initial_qpos[self.qi]

    def _tensor(self, value):
        return torch.as_tensor(
            np.asarray(value), dtype=torch.float32, device=self.device
        ).clone()

    def state(self):
        """Return same-instant FK without touching solver/contact buffers."""
        body_p, body_q, body_v, body_w = [], [], [], []
        for model, source, data in zip(
            self.models, self.native, self.fk, strict=True
        ):
            data.qpos[:] = source.qpos
            data.qvel[:] = source.qvel
            mujoco.mj_forward(model, data)
            ids = self.body_ids
            position = data.xpos[ids].copy()
            angular = data.cvel[ids, :3].copy()
            center = data.subtree_com[model.body_rootid[ids]]
            velocity = data.cvel[ids, 3:] + np.cross(
                angular, position - center
            )
            body_p.append(position)
            body_q.append(data.xquat[ids].copy())
            body_v.append(velocity)
            body_w.append(angular)
        positions, quaternions, velocities, angulars = [
            self._tensor(value) for value in (body_p, body_q, body_v, body_w)
        ]
        root_index = self.body_names.index("pelvis")
        point_indices = [
            self.body_names.index(name) for name in self.point_names
        ]
        root_quaternion = quaternions[:, root_index]
        root_angular_local = self._tensor(
            [
                d.qvel[self.root_vi + 3 : self.root_vi + 6].copy()
                for d in self.native
            ]
        )
        result = {
            "root_position": positions[:, root_index],
            "root_quaternion": root_quaternion,
            "root_linear_velocity": velocities[:, root_index],
            "root_angular_velocity": angulars[:, root_index],
            "root_angular_velocity_local": root_angular_local,
            "joint_position": self._tensor(
                [d.qpos[self.qi] for d in self.native]
            ),
            "joint_velocity": self._tensor(
                [d.qvel[self.vi] for d in self.native]
            ),
            "body_position": positions,
            "body_quaternion": quaternions,
            "body_linear_velocity": velocities,
            "body_angular_velocity": angulars,
            "point_position": positions[:, point_indices],
            "point_quaternion": quaternions[:, point_indices],
            "point_velocity": velocities[:, point_indices],
            "foot_contact": self._foot_contacts(),
            "time": self._tensor([d.time for d in self.native]),
        }
        torso_index = self.body_names.index("torso_link")
        result["fallen"] = (
            torch.minimum(
                positions[:, root_index, 2], positions[:, torso_index, 2]
            )
            < 0.3
        )
        if any(not torch.isfinite(value).all() for value in result.values()):
            self._fail("Nonfinite solo robot state")
        return result

    def _foot_contacts(self):
        contacts = np.zeros((self.num_envs, 2), dtype=bool)
        for world, (model, data) in enumerate(
            zip(self.models, self.native, strict=True)
        ):
            for contact in range(data.ncon):
                pair = list(data.contact.geom[contact])
                if self.floor_id not in pair:
                    continue
                other = pair[1] if pair[0] == self.floor_id else pair[0]
                body = int(model.geom_bodyid[other])
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact, force)
                if force[0] <= 1.0:
                    continue
                for foot, descendants in enumerate(self._foot_body_sets):
                    contacts[world, foot] |= body in descendants
        return torch.as_tensor(contacts, device=self.device)

    def reset(
        self,
        ids,
        root_position,
        root_quaternion,
        joint_position,
        root_linear_velocity=None,
        root_angular_velocity=None,
        joint_velocity=None,
    ):
        """Reset selected worlds; input rows correspond exactly to ids.

        Both supplied root velocities are world-frame; MuJoCo's free-joint
        angular component is converted to local coordinates on assignment.
        """
        ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        if (
            ids.ndim != 1
            or len(ids.unique()) != len(ids)
            or bool(((ids < 0) | (ids >= self.num_envs)).any())
        ):
            raise ValueError("Reset ids must be distinct valid world indices")
        count = len(ids)

        def array(value, width, default=False):
            if value is None and default:
                return np.zeros((count, width))
            result = (
                value.detach().cpu().numpy().astype(float)
                if isinstance(value, torch.Tensor)
                else np.asarray(value, dtype=float)
            )
            if result.shape != (count, width) or not np.isfinite(result).all():
                raise ValueError(
                    f"Expected finite reset array[{count},{width}]"
                )
            return result

        position = array(root_position, 3)
        quaternion = array(root_quaternion, 4)
        norms = np.linalg.norm(quaternion, axis=-1, keepdims=True)
        if (norms < 1e-6).any():
            raise ValueError("Invalid reset root quaternion")
        quaternion /= norms
        joints = array(joint_position, 29)
        linear = array(root_linear_velocity, 3, True)
        angular = array(root_angular_velocity, 3, True)
        joint_v = array(joint_velocity, 29, True)
        rotations = Rotation.from_quat(quaternion[:, [1, 2, 3, 0]])
        local_angular = rotations.inv().apply(angular)
        for row, index in enumerate(ids.cpu().tolist()):
            model, data = self.models[index], self.native[index]
            mujoco.mj_resetData(model, data)
            data.qpos[:] = self.initial_qpos
            qi, vi = self.root_qi, self.root_vi
            data.qpos[qi : qi + 3] = position[row]
            data.qpos[qi + 3 : qi + 7] = quaternion[row]
            data.qpos[self.qi] = joints[row]
            data.qvel[vi : vi + 3] = linear[row]
            data.qvel[vi + 3 : vi + 6] = local_angular[row]
            data.qvel[self.vi] = joint_v[row]
            self.targets[index] = joints[row]
            mujoco.mj_forward(model, data)
        self.lower.reset(ids)
        return self.state()

    @torch.inference_mode()
    def step(self, target_positions, target_quaternions):
        """Execute five world-frame future poses, then four 200-Hz substeps."""
        if self._closed:
            raise RuntimeError("SoloPhysics is closed")
        state = self.state()
        self.targets[:] = (
            self.lower(
                reference_position_w=torch.as_tensor(
                    target_positions, dtype=torch.float32, device=self.device
                ),
                reference_quaternion_wxyz=torch.as_tensor(
                    target_quaternions, dtype=torch.float32, device=self.device
                ),
                current_reference_root_position_w=state["root_position"],
                reference_root_height=state["root_position"][:, 2:3],
                robot_root_quaternion_wxyz=state["root_quaternion"],
                robot_root_angular_velocity_b=state[
                    "root_angular_velocity_local"
                ],
                robot_joint_position=state["joint_position"],
                robot_joint_velocity=state["joint_velocity"],
            )
            .detach()
            .cpu()
            .numpy()
        )
        if not np.isfinite(self.targets).all():
            self._fail("Nonfinite frozen lower joint targets")
        for _ in range(4):
            before = self.diagnostic_state()
            for world, (model, data) in enumerate(
                zip(self.models, self.native, strict=True)
            ):
                effort = self.kp * (self.targets[world] - data.qpos[self.qi])
                effort -= self.kd * data.qvel[self.vi]
                data.ctrl[self.ai] = np.clip(
                    effort, -self.torque_limit, self.torque_limit
                )
                warnings = data.warning.number.copy()
                mujoco.mj_step(model, data)
                if np.any(data.warning.number > warnings):
                    self._fail(f"MuJoCo warning in solo world {world}", before)
        self.steps += 1
        return self.state()

    def diagnostic_state(self):
        """Snapshot integration state and solver warning counts."""
        return {
            name: np.stack(
                [np.asarray(getattr(d, name)).copy() for d in self.native]
            )
            for name in (
                "qpos",
                "qvel",
                "qacc",
                "qacc_warmstart",
                "ctrl",
                "time",
            )
        } | {
            "warning_counts": np.stack(
                [d.warning.number.copy() for d in self.native]
            ),
            "targets": self.targets.copy(),
        }

    def _fail(self, reason, before=None):
        self.failure_state = {
            "reason": reason,
            "before": before,
            "after": self.diagnostic_state(),
        }
        raise RuntimeError(reason)

    def close(self):
        if self._closed:
            return
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._closed = True
        del self.lower
        self.scratch.cleanup()
