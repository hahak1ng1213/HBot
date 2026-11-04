"""Continuous, reference-conditioned skill learning without opponents."""

import math
from copy import deepcopy

import torch
from torch.nn import functional as F

SHOULDER_JOINT_NAMES = tuple(
    f"{side}_shoulder_{axis}_joint"
    for side in ("left", "right")
    for axis in ("pitch", "roll", "yaw")
)
ELBOW_JOINT_NAMES = ("left_elbow_joint", "right_elbow_joint")
PUNCH_TRACKING_RECIPE = {
    "version": "punch_anatomy_tracking_v3",
    "weights": {
        "shoulder_elbow_pose": 0.35,
        "wrist_pose": 0.30,
        "wrist_relative_velocity": 0.15,
        "full_body_root_tracking": 0.20,
    },
    "scales": {
        "shoulder_error_rad": 0.35,
        "elbow_error_rad": 0.30,
        "wrist_relative_error_m": 0.10,
        "wrist_relative_rotation_rad": 0.40,
        "wrist_relative_velocity_mps": 1.0,
    },
    "kernel": "exp(-(RMS error / scale)^2)",
    "shoulder_elbow_mix": [0.5, 0.5],
    "wrist_position_rotation_mix": [2 / 3, 1 / 3],
    "shoulder_joints": list(SHOULDER_JOINT_NAMES),
    "elbow_joints": list(ELBOW_JOINT_NAMES),
    "wrist_frame": "own root frame; subtract root v + omega x r from velocity",
    "full_body_root_tracking": {
        "version": "full_body_tracking_v1",
        "weights": {
            "root_error_m": 0.125,
            "body_position_m": 0.25,
            "body_rotation_rad": 0.25,
            "joint_error_rad": 0.125,
            "body_velocity_mps": 0.25,
        },
        "scales": {
            "root_error_m": 0.3,
            "body_position_m": 0.12,
            "body_rotation_rad": 0.4,
            "joint_error_rad": 0.35,
            "body_velocity_mps": 1.0,
        },
    },
    "entry_override": "five-point tracking only; no interpolated joint reward",
    "limitation": "five-point residual has no direct elbow redundancy control",
}


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quat_matrix(q):
    w, x, y, z = F.normalize(q, dim=-1).unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(*q.shape[:-1], 3, 3)


def quat_blend(a, b, weight):
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    return F.normalize(a + weight * (b - a), dim=-1)


def blend_angular_velocity(a, b, weight, rate):
    """World angular velocity of the shortest-arc normalized linear blend."""
    b = torch.where((a * b).sum(-1, keepdim=True) < 0, -b, b)
    delta = b - a
    raw = a + weight * delta
    length = raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    quaternion = raw / length
    derivative = (
        (delta - quaternion * (quaternion * delta).sum(-1, keepdim=True))
        * rate
        / length
    )
    conjugate = quaternion * quaternion.new_tensor((1, -1, -1, -1))
    return 2 * quat_mul(derivative, conjugate)[..., 1:]


def rotation_residual(vector):
    angle = vector.norm(dim=-1, keepdim=True)
    return torch.cat(
        (
            torch.cos(angle / 2),
            vector * (0.5 * torch.sinc(angle / (2 * math.pi))),
        ),
        -1,
    )


def tracking_metrics(actual, reference):
    a = actual["body_position"] - actual["root_position"][:, None]
    b = reference["body_position"] - reference["root_position"][:, None]
    dot = (
        (actual["body_quaternion"] * reference["body_quaternion"])
        .sum(-1)
        .abs()
        .clamp(0, 1)
    )
    point_dot = (
        (actual["point_quaternion"] * reference["point_quaternion"])
        .sum(-1)
        .abs()
        .clamp(0, 1)
    )
    return {
        "point_error_m": (
            actual["point_position"] - reference["point_position"]
        )
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt(),
        "point_rotation_rad": (2 * torch.acos(point_dot))
        .square()
        .mean(-1)
        .sqrt(),
        "body_position_m": (a - b).square().sum(-1).mean(-1).sqrt(),
        "body_rotation_rad": (2 * torch.acos(dot)).square().mean(-1).sqrt(),
        "joint_error_rad": (
            actual["joint_position"] - reference["joint_position"]
        )
        .square()
        .mean(-1)
        .sqrt(),
        "body_velocity_mps": (
            actual["body_linear_velocity"] - reference["body_linear_velocity"]
        )
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt(),
        "root_error_m": (
            actual["root_position"] - reference["root_position"]
        ).norm(dim=-1),
        "wrist_error_m": (
            actual["point_position"][:, 1:3]
            - reference["point_position"][:, 1:3]
        )
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt(),
        "root_speed_mps": actual["root_linear_velocity"][:, :2].norm(dim=-1),
        "foot_slip_mps": (
            actual["point_velocity"][:, 3:, :2].norm(dim=-1)
            * actual["foot_contact"].float()
        ).mean(-1),
    }


def root_relative_wrists(state):
    """Wrist pose and its velocity relative to the moving root frame."""
    offset = state["point_position"][:, 1:3] - state["root_position"][:, None]
    rotation = quat_matrix(state["root_quaternion"]).transpose(-1, -2)
    relative_velocity = (
        state["point_velocity"][:, 1:3]
        - state["root_linear_velocity"][:, None]
        - torch.cross(
            state["root_angular_velocity"][:, None].expand_as(offset),
            offset,
            dim=-1,
        )
    )
    conjugate = state["root_quaternion"] * offset.new_tensor((1, -1, -1, -1))
    return (
        torch.einsum("nij,nbj->nbi", rotation, offset),
        quat_mul(conjugate[:, None], state["point_quaternion"][:, 1:3]),
        torch.einsum("nij,nbj->nbi", rotation, relative_velocity),
    )


def punch_tracking_metrics(actual, reference, shoulder_indices, elbow_indices):
    """Keep elbow and shoulder errors separate from the other 21 joints."""
    joint_error = actual["joint_position"] - reference["joint_position"]
    ap, aq, av = root_relative_wrists(actual)
    rp, rq, rv = root_relative_wrists(reference)
    dot = (aq * rq).sum(-1).abs().clamp(0, 1)
    return {
        "shoulder_error_rad": joint_error[:, shoulder_indices]
        .square()
        .mean(-1)
        .sqrt(),
        "elbow_error_rad": joint_error[:, elbow_indices]
        .square()
        .mean(-1)
        .sqrt(),
        "wrist_relative_error_m": (ap - rp).square().sum(-1).mean(-1).sqrt(),
        "wrist_relative_rotation_rad": (2 * torch.acos(dot))
        .square()
        .mean(-1)
        .sqrt(),
        "wrist_relative_velocity_mps": (av - rv)
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt(),
    }


class SkillTrackingEnv:
    """A privileged skill teacher; this is NOT the competition ONNX input."""

    action_dim = 30
    dt = 0.02

    def __init__(
        self, physics, library, *, seed=54, entry_steps=50, settle_steps=50
    ):
        self.physics, self.library = physics, library
        self.device, self.num_envs = physics.device, physics.num_envs
        self.skill = library.manifest["skill"]
        if self.skill not in {"punch", "locomotion"}:
            raise ValueError("Expected punch or locomotion reference library")
        self.shoulder_indices = [
            physics.joint_names.index(name) for name in SHOULDER_JOINT_NAMES
        ]
        self.elbow_indices = [
            physics.joint_names.index(name) for name in ELBOW_JOINT_NAMES
        ]
        if any(
            type(value) is not int or value < 0
            for value in (entry_steps, settle_steps)
        ):
            raise ValueError(
                "Entry and settling lengths must be nonnegative integers"
            )
        self.entry_steps, self.settle_steps = (
            int(entry_steps),
            int(settle_steps),
        )
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        n = self.num_envs
        self.clip_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        self.frame = torch.zeros_like(self.clip_ids)
        self.entry_length = torch.zeros_like(self.clip_ids)
        self.yaw = torch.zeros(n, device=self.device)
        self.shift = torch.zeros((n, 3), device=self.device)
        self.last_action = torch.zeros((n, 30), device=self.device)
        self.failed = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.elapsed = torch.zeros_like(self.clip_ids)
        self.entry_state = {k: v.clone() for k, v in physics.state().items()}
        self._state = physics.state()
        self.start(self.clip_ids, reset_physics=True, neutral=True)
        self.observation_dim = self.get_observations().shape[-1]
        self.recipe = {
            "format": "solo_reference_residual_teacher_v3",
            "skill": self.skill,
            "action_dim": 30,
            "control_dt": self.dt,
            "reference_condition_offsets": [0, 5, 10, 25],
            "output_offsets": [1, 2, 3, 4, 5],
            "reference_world_anchor": "fixed; never follows physical drift",
            "position_residual_scales_m": [0.025, 0.06, 0.06, 0.025, 0.025],
            "rotation_residual_scale_rad": 0.15,
            "entry": "candidate smooth transition through real frozen lower",
            "entry_reward": "five-point tracking only; no joint/body FK claim",
            "entry_steps": self.entry_steps,
            "settle_steps": self.settle_steps,
            "no_opponent_no_hit_reward": True,
            "deployable_competition_actor": False,
            "tracking_reward": (
                deepcopy(PUNCH_TRACKING_RECIPE)
                if self.skill == "punch"
                else {"version": "full_body_tracking_v1", "unchanged": True}
            ),
        }

    def _rotate(self, value):
        c, s = self.yaw.cos(), self.yaw.sin()
        shape = (self.num_envs,) + (1,) * (value.ndim - 2)
        c, s = c.reshape(shape), s.reshape(shape)
        x, y, z = value.unbind(-1)
        return torch.stack((c * x - s * y, s * x + c * y, z), -1)

    def reference(self, offsets=(0,)):
        query = (
            self.frame[:, None]
            + torch.tensor(offsets, device=self.device)[None]
        )
        n, t = query.shape
        gathered = self.library.gather(
            self.clip_ids[:, None].expand(n, t).flatten(),
            query.clamp_min(0).flatten(),
        )
        out = {
            k: v.reshape(n, t, *v.shape[2:])
            for k, v in gathered.items()
            if k != "valid"
        }
        yawq = torch.stack(
            (
                torch.cos(self.yaw / 2),
                self.yaw * 0,
                self.yaw * 0,
                torch.sin(self.yaw / 2),
            ),
            -1,
        )
        entry_alpha = (
            (query + self.entry_length[:, None]).float()
            / self.entry_length[:, None].clamp_min(1)
        ).clamp(0, 1)
        entry_alpha = torch.where(
            self.entry_length[:, None] == 0,
            torch.ones_like(entry_alpha),
            entry_alpha,
        )
        alpha = entry_alpha.square() * (3 - 2 * entry_alpha)
        derivative = (
            6
            * entry_alpha
            * (1 - entry_alpha)
            / (self.entry_length[:, None].clamp_min(1) * self.dt)
        )
        past_end = query >= self.library.lengths[self.clip_ids, None] - 1
        for key, value in out.items():
            if key.endswith("position") and not key.startswith("joint"):
                value = self._rotate(value) + self.shift.reshape(
                    n, *([1] * (value.ndim - 2)), 3
                )
            elif "quaternion" in key:
                value = quat_mul(
                    yawq.reshape(n, *([1] * (value.ndim - 2)), 4), value
                )
            elif "velocity" in key and not key.startswith("joint"):
                value = self._rotate(value)
            if "velocity" in key:
                value = torch.where(
                    past_end.reshape(n, t, *([1] * (value.ndim - 2))),
                    torch.zeros_like(value),
                    value,
                )
            if key in self.entry_state:
                a = alpha.reshape(n, t, *([1] * (value.ndim - 2)))
                old = self.entry_state[key][:, None]
                if "quaternion" in key:
                    value = quat_blend(old, value, a)
                elif "velocity" not in key:
                    value = old + a * (value - old)
                else:
                    # Entry is a generated candidate, not measured motion.
                    value = a * value
            out[key] = value
        # Exact translational/joint derivative of the smooth entry reference.
        for vkey, pkey in (
            ("root_linear_velocity", "root_position"),
            ("joint_velocity", "joint_position"),
            ("body_linear_velocity", "body_position"),
            ("point_velocity", "point_position"),
        ):
            source = gathered[pkey].reshape(n, t, *gathered[pkey].shape[2:])
            if not pkey.startswith("joint"):
                source = self._rotate(source) + self.shift.reshape(
                    n, *([1] * (source.ndim - 2)), 3
                )
            delta = source - self.entry_state[pkey][:, None]
            mask = (query < 0).reshape(n, t, *([1] * (delta.ndim - 2)))
            rate = derivative.reshape(n, t, *([1] * (delta.ndim - 2)))
            out[vkey] = torch.where(mask, delta * rate, out[vkey])
        for vkey, qkey in (
            ("root_angular_velocity", "root_quaternion"),
            ("body_angular_velocity", "body_quaternion"),
        ):
            source = gathered[qkey].reshape(n, t, *gathered[qkey].shape[2:])
            source = quat_mul(
                yawq.reshape(n, *([1] * (source.ndim - 2)), 4), source
            )
            shape = (n, t, *([1] * (source.ndim - 2)))
            velocity = blend_angular_velocity(
                self.entry_state[qkey][:, None],
                source,
                alpha.reshape(shape),
                derivative.reshape(shape),
            )
            out[vkey] = torch.where(
                (query < 0).reshape(shape), velocity, out[vkey]
            )
        return out

    def start(
        self,
        clip_ids,
        *,
        reset_physics=False,
        neutral=False,
        random_phase=False,
        yaw=None,
        ids=None,
    ):
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if ids is None
            else torch.as_tensor(ids, device=self.device, dtype=torch.long)
        )
        clips = torch.as_tensor(
            clip_ids, device=self.device, dtype=torch.long
        ).clone()
        if clips.shape != ids.shape:
            raise ValueError("clip_ids must match selected environment IDs")
        if (
            ids.ndim != 1
            or len(ids.unique()) != len(ids)
            or bool(((ids < 0) | (ids >= self.num_envs)).any())
        ):
            raise ValueError("Expected distinct valid environment IDs")
        if bool(((clips < 0) | (clips >= len(self.library.clips))).any()):
            raise ValueError("Invalid clip selection")
        if random_phase and not reset_physics:
            raise ValueError(
                "Reference-phase initialization requires a physics reset"
            )
        angles = (
            torch.zeros(len(ids), device=self.device, dtype=self.yaw.dtype)
            if yaw is None
            else torch.as_tensor(yaw, device=self.device, dtype=self.yaw.dtype)
        )
        if angles.ndim == 0:
            angles = angles.expand(len(ids))
        if angles.shape != ids.shape or not torch.isfinite(angles).all():
            raise ValueError(
                "Expected finite yaw for each selected environment"
            )
        self.clip_ids[ids] = clips
        self.yaw[ids] = angles.clone()
        phase = torch.zeros_like(clips)
        if random_phase and not neutral:
            phase = (
                torch.rand(len(ids), generator=self.rng, device=self.device)
                * (self.library.lengths[clips] - 6).clamp_min(1)
            ).long()
        self.frame[ids] = phase
        self.entry_length[ids] = (
            self.entry_steps if neutral or not reset_physics else 0
        )
        source = self.library.gather(clips, phase)
        # Fix world translation at the start; retain that anchor thereafter.
        self.shift[ids] = 0
        ref = self.reference()
        if reset_physics:
            rp, rq = (
                ref["root_position"][ids, 0].clone(),
                ref["root_quaternion"][ids, 0].clone(),
            )
            jp = source["joint_position"][:, 0].clone()
            rv, rw, jv = (
                ref[k][ids, 0].clone()
                for k in (
                    "root_linear_velocity",
                    "root_angular_velocity",
                    "joint_velocity",
                )
            )
            if neutral:
                q = torch.as_tensor(
                    self.physics.initial_qpos,
                    device=self.device,
                    dtype=torch.float32,
                )
                rp = (
                    q[self.physics.root_qi : self.physics.root_qi + 3]
                    .expand(len(ids), -1)
                    .clone()
                )
                rq = (
                    q[self.physics.root_qi + 3 : self.physics.root_qi + 7]
                    .expand(len(ids), -1)
                    .clone()
                )
                angle = self.yaw[ids]
                rq = quat_mul(
                    torch.stack(
                        (
                            torch.cos(angle / 2),
                            angle * 0,
                            angle * 0,
                            torch.sin(angle / 2),
                        ),
                        -1,
                    ),
                    rq,
                )
                jp = q[self.physics.qi].expand(len(ids), -1).clone()
                rv.zero_()
                rw.zero_()
                jv.zero_()
            rp[:, :2] = 0
            self.physics.reset(
                ids,
                rp,
                rq,
                jp,
                root_linear_velocity=rv,
                root_angular_velocity=rw,
                joint_velocity=jv,
            )
        self._state = self.physics.state()
        # Anchor world XY using the unblended source phase.
        base = self.library.gather(self.clip_ids, self.frame.clamp_min(0))[
            "root_position"
        ][:, 0]
        base = self._rotate(base)
        self.shift[ids, :2] = (
            self._state["root_position"][ids, :2] - base[ids, :2]
        )
        for key in self.entry_state:
            self.entry_state[key][ids] = self._state[key][ids]
        self.frame[ids] -= self.entry_length[ids]
        self.last_action[ids] = 0
        self.failed[ids] = False
        self.elapsed[ids] = 0
        return self.get_observations()

    def get_observations(self):
        state = self._state
        ref = self.reference((0, 5, 10, 25))
        rotation = quat_matrix(state["root_quaternion"]).transpose(-1, -2)

        def local(value):
            return torch.einsum("nij,n...j->n...i", rotation, value)

        p = local(
            ref["point_position"] - state["root_position"][:, None, None]
        )
        r = torch.einsum(
            "nij,ntbjk->ntbik", rotation, quat_matrix(ref["point_quaternion"])
        )
        body_p = local(
            state["point_position"] - state["root_position"][:, None]
        )
        phase = self.frame.float() / self.library.lengths[self.clip_ids]
        gravity = local(
            state["root_position"]
            .new_tensor((0, 0, -1))
            .expand(self.num_envs, -1)
        )
        values = [
            state["joint_position"],
            state["joint_velocity"] * 0.1,
            gravity,
            local(state["root_linear_velocity"]),
            local(state["root_angular_velocity"]) * 0.25,
            body_p.flatten(1),
            local(state["point_velocity"]).flatten(1) * 0.25,
            p.flatten(1),
            r[..., (0, 2)].flatten(1),
            ref["joint_position"].flatten(1),
            torch.stack(
                (
                    torch.sin(phase * 2 * math.pi),
                    torch.cos(phase * 2 * math.pi),
                ),
                -1,
            ),
            F.one_hot(self.clip_ids, len(self.library.clips)).float(),
            self.last_action,
        ]
        observation = torch.cat(values, -1)
        if not torch.isfinite(observation).all():
            raise RuntimeError("Nonfinite skill observation")
        return observation.clamp(-10, 10)

    def _tracking_reward(self, actual, reference):
        metrics = tracking_metrics(actual, reference)
        full_body = (
            0.5 * torch.exp(-(metrics["root_error_m"] / 0.3).square())
            + torch.exp(-(metrics["body_position_m"] / 0.12).square())
            + torch.exp(-(metrics["body_rotation_rad"] / 0.4).square())
            + 0.5 * torch.exp(-(metrics["joint_error_rad"] / 0.35).square())
            + torch.exp(-(metrics["body_velocity_mps"] / 1.0).square())
        ) / 4
        if self.skill == "locomotion":
            return full_body, metrics
        metrics.update(
            punch_tracking_metrics(
                actual, reference, self.shoulder_indices, self.elbow_indices
            )
        )
        recipe = self.recipe["tracking_reward"]
        scores = {
            key: torch.exp(-(metrics[key] / scale).square())
            for key, scale in recipe["scales"].items()
        }
        shoulder, elbow = recipe["shoulder_elbow_mix"]
        position, rotation = recipe["wrist_position_rotation_mix"]
        weights = recipe["weights"]
        reward = (
            weights["shoulder_elbow_pose"]
            * (
                shoulder * scores["shoulder_error_rad"]
                + elbow * scores["elbow_error_rad"]
            )
            + weights["wrist_pose"]
            * (
                position * scores["wrist_relative_error_m"]
                + rotation * scores["wrist_relative_rotation_rad"]
            )
            + weights["wrist_relative_velocity"]
            * scores["wrist_relative_velocity_mps"]
            + weights["full_body_root_tracking"] * full_body
        )
        return reward, metrics

    def step(self, action):
        if (
            action.shape != (self.num_envs, 30)
            or not torch.isfinite(action).all()
        ):
            raise ValueError("Expected finite raw Gaussian residual [N,30]")
        bounded = action.tanh().reshape(self.num_envs, 5, 6)
        ref = self.reference((1, 2, 3, 4, 5))
        scales = action.new_tensor((0.025, 0.06, 0.06, 0.025, 0.025))[
            None, :, None
        ]
        delta = bounded[..., :3] * scales
        rotation = quat_matrix(self._state["root_quaternion"])
        delta_world = torch.einsum("nij,nbj->nbi", rotation, delta)
        target_p = ref["point_position"] + delta_world[:, None]
        # Apply the same local-axis rotation correction across future frames.
        target_q = quat_mul(
            ref["point_quaternion"],
            rotation_residual(bounded[..., 3:] * 0.15)[:, None],
        )
        self._state = self.physics.step(target_p, target_q)
        self.frame += 1
        self.elapsed += 1
        current = {k: v[:, 0] for k, v in self.reference().items()}
        reward, metrics = self._tracking_reward(self._state, current)
        # Entry poses are independently interpolated target candidates. Their
        # joint and body interpolants need not be FK-consistent; do not ask the
        # physical robot to satisfy both representations simultaneously.
        is_entry = self.frame < 0
        entry_reward = (
            0.25 * torch.exp(-(metrics["root_error_m"] / 0.3).square())
            + 0.5 * torch.exp(-(metrics["point_error_m"] / 0.15).square())
            + 0.25 * torch.exp(-(metrics["point_rotation_rad"] / 0.4).square())
        )
        reward = torch.where(is_entry, entry_reward, reward)
        reward -= 0.01 * metrics["foot_slip_mps"].clamp_max(3)
        reward -= 0.003 * bounded.square().mean((1, 2))
        reward -= 0.005 * (
            bounded.flatten(1) - self.last_action
        ).square().mean(-1)
        self.last_action = bounded.flatten(1).detach().clone()
        self.failed |= self._state["fallen"] | (metrics["root_error_m"] > 1.5)
        completed = (
            self.frame
            >= self.library.lengths[self.clip_ids] - 1 + self.settle_steps
        )
        done = completed | self.failed
        reward -= self.failed.float()
        return (
            self.get_observations(),
            reward,
            done,
            {
                "completed": completed & ~self.failed,
                "failed": self.failed.clone(),
                "entry": is_entry,
                "motion": (self.frame >= 0)
                & (self.frame < self.library.lengths[self.clip_ids] - 1),
                "settle": self.frame
                >= self.library.lengths[self.clip_ids] - 1,
                "metrics": metrics,
            },
        )
