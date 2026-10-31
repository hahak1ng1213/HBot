"""Native MuJoCo CPU physics; Torch policy tensors may live on CPU or CUDA.

The previous Warp implementation is archived in .cpu-backend-original/.
Each world owns an independent MjModel/MjData, including randomized inertia.
Torch qpos/qvel are writable staging buffers; native solver/contact data are
kept intact while separate FK data supplies same-instant observations.
"""

import copy
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from holoboxing.data.records import initial_qpos
from holoboxing.data.robot import load_scene_model
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)
from holoboxing.training.tasks.combat.mdp.geometry import matrix, rotate


class CompetitionPhysics:
    backend = "mujoco_cpu"
    fields = (
        "qpos",
        "qvel",
        "qacc",
        "qacc_warmstart",
        "ctrl",
        "time",
        "xpos",
        "xquat",
        "xipos",
        "cvel",
        "subtree_com",
    )

    def __init__(
        self,
        assets,
        worlds,
        device,
        *,
        contact_capacity=None,
        constraint_capacity=None,
        ordered=None,
    ):
        if type(worlds) is not int or worlds < 1:
            raise ValueError("Physics requires positive worlds")
        self.device = torch.device(device)
        self.num_envs = worlds
        self.host = load_scene_model(assets)
        if not np.isclose(self.host.opt.timestep, 0.005):
            raise ValueError("Competition physics requires a 0.005 s timestep")
        self.initial = mujoco.MjData(self.host)
        self.initial.qpos[:] = initial_qpos(assets, self.host)
        mujoco.mj_forward(self.host, self.initial)
        floor = self.host.geom("smpl_ring_baseLink").id
        self.floor_height = float(
            self.initial.geom_xpos[floor, 2] + self.host.geom_size[floor, 2]
        )
        self.models = [copy.copy(self.host) for _ in range(worlds)]
        self.native = [mujoco.MjData(m) for m in self.models]
        self.fk = [mujoco.MjData(m) for m in self.models]
        for m, d in zip(self.models, self.native, strict=True):
            d.qpos[:] = self.initial.qpos
            mujoco.mj_forward(m, d)
        self.data = SimpleNamespace()
        self.model = SimpleNamespace(
            body_ipos=torch.tensor(
                np.stack([m.body_ipos for m in self.models]),
                dtype=torch.float32,
                device=device,
            )
        )
        self.ordered = (
            False  # Native solver, not the official Warp ordering patch.
        )
        qi, vi, ai, bids, roots = [], [], [], [], []
        actuator_by_joint = {
            int(j[0]): i for i, j in enumerate(self.host.actuator_trnid)
        }
        for prefix in ("p1_", "p2_"):
            joints = [
                self.host.joint(prefix + name).id for name in JOINT_NAMES
            ]
            qi.append(self.host.jnt_qposadr[joints])
            vi.append(self.host.jnt_dofadr[joints])
            ai.append([actuator_by_joint[j] for j in joints])
            bids.append(
                [self.host.body(prefix + name).id for name in BODY_NAMES]
            )
            roots.append(
                int(
                    self.host.jnt_qposadr[
                        self.host.body(prefix + "pelvis").jntadr[0]
                    ]
                )
            )
        self.qi, self.vi, self.ai = [
            torch.tensor(np.asarray(x), device=device) for x in (qi, vi, ai)
        ]
        self.body_ids = torch.tensor(bids, device=device)
        self.native_body_ids = np.asarray(bids)
        self.native_root_ids = self.host.body_rootid[self.native_body_ids]
        self.root_qi = roots
        self.root_ids = torch.tensor(
            self.host.body_rootid[np.asarray(bids)], device=device
        )
        limits = self.host.actuator_ctrlrange[np.asarray(ai)]
        if not self.host.actuator_ctrllimited[
            np.asarray(ai)
        ].all() or not np.allclose(limits[..., 0], -limits[..., 1]):
            raise ValueError("Expected symmetric scene torque limits")
        self.limits = limits[..., 1].copy()
        self.indices = [np.asarray(x, dtype=np.int64) for x in (qi, vi, ai)]
        self.targets = (
            torch.tensor(
                self.initial.qpos[np.asarray(qi)],
                device=device,
                dtype=torch.float32,
            )
            .expand(worlds, -1, -1)
            .clone()
        )
        mapping = np.full(self.host.nbody, -1, dtype=np.int32)
        mapping[np.asarray(bids).flatten()] = np.arange(60)
        # Fixed glove/head links belong to the same rigid body as their
        # actuated parent. Retain their forces in the existing 30-body schema.
        rigid_body = self.host.body_weldid[self.host.geom_bodyid]
        self.geom_bodies = mapping[rigid_body]
        slots = np.full((self.host.ngeom, self.host.ngeom), -1, dtype=np.int32)
        for role, prefix in enumerate(("p1_", "p2_")):
            other = "p2_" if role == 0 else "p1_"
            torso = self.host.body(prefix + "torso_link").id
            for hand, side in enumerate(("left", "right")):
                wrist = self.host.body(other + side + "_wrist_yaw_link").id
                slots[
                    np.ix_(
                        rigid_body == torso,
                        rigid_body == wrist,
                    )
                ] = role * 2 + hand
        self.slots = slots
        self.net_history = torch.zeros((worlds, 6, 2, 30, 3), device=device)
        self.paired_history = torch.zeros((worlds, 6, 2, 2, 3), device=device)
        self.joint_acceleration = torch.zeros((worlds, 2, 29), device=device)
        self.steps = 0
        self.kp = self.kd = None
        self._contacts = [[] for _ in range(worlds)]
        self._pull()

    def _pull(self, ids=None):
        ids = list(range(self.num_envs)) if ids is None else ids
        for name in self.fields:
            values = np.stack(
                [np.asarray(getattr(self.native[i], name)) for i in ids]
            )
            tensor = torch.as_tensor(values, device=self.device)
            if not hasattr(self.data, name):
                setattr(self.data, name, tensor.clone())
            else:
                getattr(self.data, name)[ids] = tensor

    def _push(self):
        # Explicit synchronization of only writable integration inputs.
        qpos = self.data.qpos.detach().cpu().numpy()
        qvel = self.data.qvel.detach().cpu().numpy()
        for i, d in enumerate(self.native):
            d.qpos[:] = qpos[i]
            d.qvel[:] = qvel[i]

    def forward(self):
        self._push()
        for m, d in zip(self.models, self.native, strict=True):
            mujoco.mj_forward(m, d)
        self._pull()

    def set_gains(self, kp, kd):
        self.kp, self.kd = [x.detach().cpu().numpy().copy() for x in (kp, kd)]

    def push(self, ids, role, delta_velocity_w):
        """Add a world-frame velocity impulse at the root center of mass."""
        root = self.host.body(("p1_", "p2_")[role] + "pelvis")
        qi = self.root_qi[role]
        vi = int(self.host.jnt_dofadr[root.jntadr[0]])
        qpos, qvel = self.data.qpos, self.data.qvel
        rotation = matrix(qpos[ids, qi + 3 : qi + 7].float())
        ipos = self.model.body_ipos
        local_com = (
            ipos[0, root.id] if ipos.shape[0] == 1 else ipos[ids, root.id]
        )
        offset_w = rotate(rotation, local_com)
        # Free-joint translation is at the body origin; angular velocity is local.
        qvel[ids, vi : vi + 3] += delta_velocity_w[:, :3] - torch.cross(
            delta_velocity_w[:, 3:], offset_w, dim=-1
        )
        qvel[ids, vi + 3 : vi + 6] += rotate(
            rotation.transpose(-1, -2), delta_velocity_w[:, 3:]
        )

    def randomize_torso_com(self, offsets):
        offsets = offsets.detach().cpu().numpy()
        if (
            offsets.shape != (self.num_envs, 2, 3)
            or not np.isfinite(offsets).all()
        ):
            raise ValueError("Invalid per-world torso COM offsets")
        for i, m in enumerate(self.models):
            for role, prefix in enumerate(("p1_", "p2_")):
                m.body_ipos[m.body(prefix + "torso_link").id] += offsets[
                    i, role
                ]
            # Use scratch data so recomputing model constants cannot reset a world.
            mujoco.mj_setConst(m, mujoco.MjData(m))
        self.model.body_ipos.copy_(
            torch.as_tensor(
                np.stack([m.body_ipos for m in self.models]),
                device=self.device,
            )
        )
        self.forward()

    def substep(self):
        if self.kp is None:
            raise RuntimeError("Set controller gains before stepping")
        self._push()
        targets = self.targets.detach().cpu().numpy()
        qi, vi, ai = self.indices
        previous = np.stack([d.qvel[vi].copy() for d in self.native])
        for i, (m, d) in enumerate(zip(self.models, self.native, strict=True)):
            effort = self.kp * (targets[i] - d.qpos[qi]) - self.kd * d.qvel[vi]
            d.ctrl[ai] = np.clip(effort, -self.limits, self.limits)
            warnings = d.warning.number.copy()
            mujoco.mj_step(m, d)
            # MuJoCo can auto-reset a divergent world. Never silently accept it.
            if np.any(d.warning.number > warnings):
                self._pull()
                raise RuntimeError(
                    f"MuJoCo solver warning in world {i}; inspect diagnostic_state"
                )
        self._pull()
        velocity = np.stack([d.qvel[vi] for d in self.native])
        self.joint_acceleration.copy_(
            torch.as_tensor(
                (velocity - previous) / self.host.opt.timestep,
                device=self.device,
            )
        )
        self.read_contact_forces()
        self.steps += 1

    def read_contact_forces(self):
        from holoboxing.training.envs.official_referee import contact_records

        net = np.zeros((self.num_envs, 2, 30, 3))
        paired = np.zeros((self.num_envs, 2, 2, 3))
        self._contacts = []
        for w, (m, d) in enumerate(zip(self.models, self.native, strict=True)):
            forces = np.zeros((d.ncon, 6))
            for c in range(d.ncon):
                local = np.zeros(6)
                mujoco.mj_contactForce(m, d, c, local)
                frame = d.contact.frame[c].reshape(3, 3)
                forces[c, :3] = local[:3] @ frame
                forces[c, 3:] = local[3:] @ frame
                pair = d.contact.geom[c]
                for side in (0, 1):
                    body = self.geom_bodies[pair[side]]
                    force = (2 * side - 1) * forces[c, :3]
                    if body >= 0:
                        net[w, body // 30, body % 30] += force
                    slot = self.slots[pair[side], pair[1 - side]]
                    if slot >= 0:
                        paired[w, slot // 2, slot % 2] += force
            self._contacts.append(
                contact_records(
                    m,
                    np.zeros(d.ncon, dtype=int),
                    d.contact.geom[: d.ncon],
                    d.contact.pos[: d.ncon],
                    forces,
                    d.contact.frame[: d.ncon].reshape(-1, 3, 3)[:, 0],
                    d.contact.dist[: d.ncon],
                    d.ncon,
                    1,
                )[0]
            )
        self.net_history[:, 1:] = self.net_history[:, :-1].clone()
        self.paired_history[:, 1:] = self.paired_history[:, :-1].clone()
        self.net_history[:, 0] = torch.as_tensor(net, device=self.device)
        self.paired_history[:, 0] = torch.as_tensor(paired, device=self.device)

    def official_contacts(self):
        return self._contacts

    def reset(self, ids):
        rows = ids.detach().cpu().tolist()
        if not rows:
            return
        for i in rows:
            m, d = self.models[i], self.native[i]
            mujoco.mj_resetData(m, d)
            d.qpos[:] = self.initial.qpos
            mujoco.mj_forward(m, d)
            self._contacts[i] = []
        self._pull(rows)
        self.targets[ids] = torch.as_tensor(
            self.initial.qpos[self.indices[0]],
            device=self.device,
            dtype=torch.float32,
        )
        self.net_history[ids] = 0
        self.paired_history[ids] = 0
        self.joint_acceleration[ids] = 0

    def state(self):
        self._push()
        for m, source, d in zip(
            self.models, self.native, self.fk, strict=True
        ):
            d.qpos[:] = source.qpos
            d.qvel[:] = source.qvel
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)

        def tensor(name):
            return torch.as_tensor(
                np.stack([getattr(d, name) for d in self.fk]),
                device=self.device,
                dtype=torch.float32,
            )

        pos = tensor("xpos")[:, self.body_ids]
        quat = tensor("xquat")[:, self.body_ids]
        cv = tensor("cvel")[:, self.body_ids]
        # Compute the origin shift in native precision before casting policy tensors.
        bids = self.native_body_ids
        roots = self.native_root_ids
        velocities = []
        for d in self.fk:
            cvel = d.cvel[bids]
            velocities.append(
                cvel[..., 3:]
                + np.cross(cvel[..., :3], d.xpos[bids] - d.subtree_com[roots])
            )
        vel = torch.as_tensor(
            np.stack(velocities), device=self.device, dtype=torch.float32
        )
        return dict(
            position=pos,
            quaternion=quat,
            velocity=vel,
            angular=cv[..., :3],
            joint_position=self.data.qpos[:, self.qi].float(),
            joint_velocity=self.data.qvel[:, self.vi].float(),
            joint_acceleration=self.joint_acceleration,
        )

    def diagnostic_state(self):
        return {
            "physics_backend": self.backend,
            "physics_steps": self.steps,
            "state": {
                name: getattr(self.data, name).detach().cpu().clone()
                for name in self.fields
            },
            "contacts": [
                {
                    name: np.array(
                        getattr(d.contact, name)[: d.ncon], copy=True
                    )
                    for name in (
                        "geom",
                        "dim",
                        "dist",
                        "frame",
                        "pos",
                        "efc_address",
                        "friction",
                    )
                }
                for d in self.native
            ],
            "constraint_force": [d.efc_force.copy() for d in self.native],
            "body_ipos": self.model.body_ipos.cpu().clone(),
            "targets": self.targets.cpu().clone(),
            "joint_acceleration": self.joint_acceleration.cpu().clone(),
            "net_history": self.net_history.cpu().clone(),
            "paired_history": self.paired_history.cpu().clone(),
        }
