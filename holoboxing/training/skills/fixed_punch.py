"""Kangning037 task adapter: explicit range gate above a frozen controller.

The gate constrains REQUESTED targets, not instantaneous physical velocity.
After an exit the arms retract continuously; no controller can stop inertia
instantaneously. This reference/phase-conditioned teacher is not an ONNX entry.
"""

from dataclasses import asdict, dataclass
import math

import torch

from holoboxing.training.skills.env import (
    quat_blend,
    quat_matrix,
    quat_mul,
    rotation_residual,
)
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
    POINTS,
)
from holoboxing.training.tasks.combat.mdp.geometry import encode_output


@dataclass(frozen=True)
class FixedPunchConfig:
    attack_exit_m: float = 0.6
    attack_enter_m: float = 0.6
    stopping_distance_m: float = 0.56
    facing_enter_deg: float = 90.0
    facing_exit_deg: float = 90.0
    guard_offset_deg: float = 0.0
    transition_steps: int = 10
    cycle_bridge_steps: int = 20
    turn_rate_rad_s: float = 0.6
    movement: str = "demo"

    def __post_init__(self):
        if any(
            not math.isfinite(v)
            for v in asdict(self).values()
            if isinstance(v, (int, float))
        ):
            raise ValueError("Nonfinite task setting")
        if (
            not 0
            < self.stopping_distance_m
            < self.attack_enter_m
            <= self.attack_exit_m
            <= 0.6
        ):
            raise ValueError("Require stop < enter <= exit <= 0.6 m")
        if not 0 < self.facing_enter_deg <= self.facing_exit_deg <= 90:
            raise ValueError("Invalid facing gates")
        if any(
            type(v) is not int or v < 2
            for v in (self.transition_steps, self.cycle_bridge_steps)
        ):
            raise ValueError("Transitions require at least two control steps")
        if self.turn_rate_rad_s <= 0 or self.movement not in (
            "demo",
            "explore",
        ):
            raise ValueError("Invalid movement recipe")


def yaw(q):
    w, x, y, z = q.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y.square() + z.square()))


def yaw_quaternion(angle):
    return torch.stack(
        ((angle / 2).cos(), angle * 0, angle * 0, (angle / 2).sin()), -1
    )


def geometry(state, config):
    delta = state["opponent"][:, :2] - state["position"][:, 0, :2]
    distance = delta.norm(dim=-1)
    error = torch.atan2(delta[:, 1], delta[:, 0]) - yaw(
        state["quaternion"][:, 0]
    )
    error -= math.radians(config.guard_offset_deg)
    return distance, torch.atan2(error.sin(), error.cos())


def in_front(angle, limit_deg=90.0):
    """Front half-plane; exactly sideways and behind never qualify."""
    return (angle.abs() < math.pi / 2) & (
        angle.abs() <= math.radians(limit_deg)
    )


def attack_gate(previous, distance, angle, config):
    """Range gate with configurable front-facing limits."""
    finite = torch.isfinite(distance) & torch.isfinite(angle)
    enter = (distance <= config.attack_enter_m) & in_front(
        angle, config.facing_enter_deg
    )
    remain = (distance <= config.attack_exit_m) & in_front(
        angle, config.facing_exit_deg
    )
    return finite & torch.where(previous, remain, enter)


def guard_pose(clip):
    """Choose a raised, compact two-hand pose instead of the clip's T-pose entry."""
    inverse = quat_matrix(clip["point_quaternion"][:, 0]).transpose(-1, -2)
    local = torch.einsum(
        "nij,nbj->nbi",
        inverse,
        clip["point_position"][:, 1:3] - clip["point_position"][:, :1],
    )
    desired = local.new_tensor([[0.18, 0.12, 0.34], [0.18, -0.12, 0.34]])
    score = (local - desired).square().mean((1, 2))
    if "body_position" in clip:
        elbows = [
            BODY_NAMES.index(f"{side}_elbow_link")
            for side in ("left", "right")
        ]
        torso = BODY_NAMES.index("torso_link")
        elbow_local = torch.einsum(
            "nij,nbj->nbi",
            inverse,
            clip["body_position"][:, elbows]
            - clip["body_position"][:, torso : torso + 1],
        )
        # Prefer reference wrist orientations associated with tucked elbows.
        score += (
            (elbow_local[..., 1].abs() - 0.18).clamp_min(0).square().mean(-1)
        )
    frame = int(score.argmin())
    invq = clip["point_quaternion"][frame, 0].clone()
    invq[1:] *= -1
    return desired, quat_mul(invq, clip["point_quaternion"][frame, 1:3]), frame


def training_start_poses(qpos, root_indices, ids, rng, config):
    """Training only: mix reachable punches and approaches from all bearings.

    Joint poses, height and root tilt remain the combat reset's values. The
    official evaluation path never calls this reset distribution.
    """
    count = len(ids)
    samples = torch.rand(count, 4, generator=rng, device=qpos.device)
    near = samples[:, 0] < 0.5
    distance = torch.where(
        near, 0.52 + samples[:, 1] * 0.06, 0.65 + samples[:, 1] * 1.35
    )
    bearing = (samples[:, 2] * 2 - 1) * math.pi
    error = (samples[:, 3] * 2 - 1) * torch.where(
        near, math.radians(10), math.pi
    )
    a, b = root_indices
    center = (qpos[ids, a : a + 2] + qpos[ids, b : b + 2]) / 2
    delta = (
        torch.stack((bearing.cos(), bearing.sin()), -1) * distance[:, None] / 2
    )
    qpos[ids, a : a + 2], qpos[ids, b : b + 2] = center - delta, center + delta
    for root, heading in (
        (a, bearing - error - math.radians(config.guard_offset_deg)),
        (b, bearing + math.pi),
    ):
        original = qpos[ids, root + 3 : root + 7]
        qpos[ids, root + 3 : root + 7] = quat_mul(
            yaw_quaternion(heading - yaw(original)), original
        ).to(qpos.dtype)


def cyclic_points(library, phases, offsets, bridge_steps, *, locomotion=False):
    """Smooth end-to-start target bridge, explicitly outside source motion.

    Locomotion carries forward net translation across cycles. A punch returns
    to its original anchor. This is a candidate transition, not source data.
    """
    clip = library.clips[0]
    count = clip["frames"]
    total = count - 1 + bridge_steps
    query = phases[:, None] + phases.new_tensor(offsets)[None]
    cycle = torch.div(query, total, rounding_mode="floor")
    local = query.remainder(total)
    index = local.clamp_max(count - 1).long()
    alpha = ((local - (count - 1)).float() / bridge_steps).clamp(0, 1)
    alpha = alpha.square() * (3 - 2 * alpha)
    net = clip["root_position"][-1] - clip["root_position"][0]
    net = net.clone() if locomotion else torch.zeros_like(net)
    net[2] = 0
    result = {}
    for key in ("point_position", "point_quaternion", "joint_position"):
        values = clip[key]
        if key == "joint_position":
            bridge = values[-1] + alpha[..., None] * (values[0] - values[-1])
            result[key] = torch.where(
                (local >= count - 1)[..., None], bridge, values[index]
            )
            continue
        start = values[0] + net if key == "point_position" else values[0]
        a = alpha[..., None, None]
        if key.endswith("position"):
            bridge = values[-1] + a * (start - values[-1])
            sampled = torch.where(
                (local >= count - 1)[..., None, None], bridge, values[index]
            )
            result[key] = sampled + cycle[..., None, None] * net
        else:
            bridge = quat_blend(values[-1], start, a)
            result[key] = torch.where(
                (local >= count - 1)[..., None, None], bridge, values[index]
            )
    result["bridge"] = local >= count - 1
    return result


def task_reward(
    before,
    after,
    reference,
    attack,
    settled,
    hits,
    fallen,
    invalid,
    action,
    previous_action,
    config,
    dt=0.02,
):
    """Shaping is dt-scaled; HIT, per-step fallen and invalid costs are not."""
    distance0, angle0 = geometry(before, config)
    distance, angle = geometry(after, config)
    own = after["position"]
    target = reference["point_position"][:, 0]
    wrist_error = (
        ((own[:, 1:3] - own[:, :1]) - (target[:, 1:3] - target[:, :1]))
        .square()
        .mean((1, 2))
        .sqrt()
    )
    feet_error = (own[:, 3:5] - target[:, 3:5]).square().mean((1, 2)).sqrt()
    pelvis_error = (own[:, 0] - target[:, 0]).norm(dim=-1)
    actual_rotation = quat_matrix(after["quaternion"][:, 1:3])
    target_rotation = quat_matrix(reference["point_quaternion"][:, 0, 1:3])
    rotation_error = (
        (actual_rotation - target_rotation).square().mean((1, 2, 3))
    )
    desired_velocity = (reference["point_position"][:, 1] - target) / dt
    actual_relative = after["velocity"][:, 1:3] - after["velocity"][:, :1]
    desired_relative = desired_velocity[:, 1:3] - desired_velocity[:, :1]
    velocity_error = (actual_relative - desired_relative).square().mean((1, 2))
    a = attack.float()
    # Linear five-point transitions and linear joint transitions need not
    # have the same FK. Only supervise anatomy on original motion frames.
    anatomy = a * settled.float() * (~reference["bridge"][:, 0]).float()
    moving = 1 - a
    # Track the stop distance; also discourage crowding the opponent.
    old_gap = (distance0 - config.stopping_distance_m).abs()
    new_gap = (distance - config.stopping_distance_m).abs()
    progress = ((old_gap - new_gap) / dt).clamp(-1, 1)
    facing_progress = ((angle0.abs() - angle.abs()) / dt).clamp(-2, 2)
    upright = quat_matrix(after["quaternion"][:, 0])[:, 2, 2].clamp(0, 1)
    joint_error = after["joint_position"] - reference["joint_position"][:, 0]
    arm_ids = [
        i
        for i, name in enumerate(JOINT_NAMES)
        if "shoulder" in name or "elbow" in name
    ]
    leg_ids = [
        i
        for i, name in enumerate(JOINT_NAMES)
        if any(part in name for part in ("hip", "knee", "ankle"))
    ]
    arm_error = joint_error[:, arm_ids].square().mean(-1).sqrt()
    leg_error = joint_error[:, leg_ids].square().mean(-1).sqrt()
    terms = {
        "approach_progress": dt
        * moving
        * (distance0 > config.attack_exit_m).float()
        * 2.0
        * progress,
        "move_guard_tracking": dt
        * moving
        * 2.0
        * torch.exp(-(wrist_error / 0.08).square()),
        "facing_progress": dt * 0.5 * facing_progress,
        "facing_alignment": dt * 1.0 * angle.cos(),
        "punch_wrist_position": dt
        * a
        * 2.0
        * torch.exp(-(wrist_error / 0.10).square()),
        "punch_wrist_orientation": dt
        * a
        * 0.3
        * torch.exp(-rotation_error / 0.16),
        "punch_relative_velocity": dt
        * a
        * settled.float()
        * 0.5
        * torch.exp(-velocity_error),
        "punch_feet_tracking": dt
        * a
        * 0.8
        * torch.exp(-(feet_error / 0.06).square()),
        "punch_pelvis_tracking": dt
        * a
        * 0.5
        * torch.exp(-(pelvis_error / 0.08).square()),
        "punch_arm_joints": dt
        * anatomy
        * 1.0
        * torch.exp(-(arm_error / 0.35).square()),
        "punch_leg_joints": dt
        * anatomy
        * 0.5
        * torch.exp(-(leg_error / 0.25).square()),
        "balance": dt * 0.2 * upright,
        "time": torch.full_like(distance, -dt * 0.1),
        "residual": -dt * 0.02 * action.square().mean(-1),
        "action_rate": -dt
        * 0.02
        * (action - previous_action).square().mean(-1),
        "official_hit": 5.0
        * hits
        * a
        * (distance <= config.attack_exit_m)
        * in_front(angle, config.facing_exit_deg),
        "fallen": -2.0 * fallen.float(),
    }
    # Invalid physics remains a failed trial with a -10 cost. Mask diagnostic
    # NaNs only on those explicitly invalid rows; never hide healthy-row NaNs.
    terms = {
        key: torch.where(invalid, torch.zeros_like(value), value)
        for key, value in terms.items()
    }
    reward = sum(terms.values())
    reward = torch.where(invalid, torch.full_like(reward, -10), reward)
    metrics = {
        **terms,
        "distance_m": distance,
        "facing_error_rad": angle.abs(),
        "attack_mode": a,
        "wrist_error_m": wrist_error,
        "feet_error_m": feet_error,
        "arm_error_rad": arm_error,
        "leg_error_rad": leg_error,
        "official_hits": hits,
    }
    elbow_local = torch.einsum(
        "nij,nbj->nbi",
        quat_matrix(after["quaternion"][:, 0]).transpose(-1, -2),
        after["elbow_position"] - after["torso_position"][:, None],
    )
    metrics["elbow_lateral_extent_m"] = elbow_local[..., 1].abs().amax(-1)
    metrics = {
        key: torch.where(invalid, torch.zeros_like(value), value).float()
        for key, value in metrics.items()
    }
    metrics["invalid"] = invalid.float()
    return reward, metrics


class FixedPunchCombatEnv:
    checkpoint_format = "kangning037_range_teacher_v3"
    dt = 0.02

    def __init__(
        self, combat, punch, movement, *, config=None, seed=37, opponent=None
    ):
        self.opponent = opponent
        self.combat, self.library, self.movement_library = (
            combat,
            punch,
            movement,
        )
        self.config = config or FixedPunchConfig()
        self.device, self.num_envs = combat.device, combat.num_envs
        if (
            len(punch.clips) != 1
            or punch.clips[0]["group"] != "kangning-037"
            or punch.manifest["speed_scale"] != 1.5
        ):
            raise ValueError(
                "This task requires original kangning-037 at 1.5x"
            )
        if not combat.official_rules or not combat.capture_transition_state:
            raise ValueError(
                "Require official referee and pre-reset transition capture"
            )
        if (
            opponent is None
            and getattr(combat.standing_reference, "role", None) != 1
        ):
            raise ValueError(
                "Require a world-anchored standing opponent in role 1"
            )
        if opponent is not None and combat.standing_reference is not None:
            raise ValueError("Learned opponent cannot have standing override")
        if self.config.movement == "demo" and movement is None:
            raise ValueError("Demo movement requires a forward reference")
        self.physics = combat.physics
        self.physics.metadata = {
            "backend": "competition_cpu",
            "official_rules": True,
            "frozen_lower": True,
            "control_dt": self.dt,
        }
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        n = self.num_envs
        self.attack = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.age = self.phase.clone()
        self.last_action = torch.zeros(n, 30, device=self.device)
        self.anchor = torch.zeros(n, 3, device=self.device)
        self.heading = torch.zeros(n, device=self.device)
        self.entry_p = torch.zeros(n, 5, 3, device=self.device)
        self.entry_q = torch.zeros(n, 5, 4, device=self.device)
        self.entry_joints = torch.zeros(
            n, len(JOINT_NAMES), device=self.device
        )
        self.guard_position, self.guard_rotation, guard_frame = guard_pose(
            punch.clips[0]
        )
        self.recipe = {
            "version": self.checkpoint_format,
            **asdict(self.config),
            "punch_source": "kangning-037",
            "punch_speed": 1.5,
            "movement_guard": {
                "pelvis_local_wrists": self.guard_position.tolist(),
                "orientation_source_frame": guard_frame,
                "orientation_selection": "wrist error plus elbow lateral excess beyond 0.18 m",
            },
            "close_alignment": "distance <= 0.6: stop locomotion phase and translation, retain yaw alignment before punch",
            "motion": None if movement is None else movement.manifest,
            "training_resets": "50% distance [0.52,0.58], facing +/-10 deg; 50% [0.65,2.0], facing +/-180 deg; random world bearing; evaluation uses combat reset",
            "guard_axis": "pelvis yaw plus explicit guard_offset_deg; calibration pending",
            "attack_alignment": "rotate source anchor heading toward opponent at <=0.6 rad/s; preserve source joints and relative motion",
            "facing_reward": {
                "alignment": "dt*cos(error)",
                "progress": "dt*0.5*clip((abs(old)-abs(new))/dt,-2,2)",
                "modes": "move and punch",
            },
            "deployable_competition_actor": False,
            "reward": "fixed_punch.task_reward; shaping dt=0.02, HIT=5, fall=-2, invalid=-10",
        }
        self.obs = combat.reset()
        self._prepare(torch.ones(n, dtype=torch.bool, device=self.device))
        self.observation_dim = self.get_observations().shape[-1]

    def state(self, raw=None):
        raw = self.physics.state() if raw is None else raw
        ids = [BODY_NAMES.index(name) for name in POINTS]
        return {
            "position": raw["position"][:, 0, ids].clone(),
            "quaternion": raw["quaternion"][:, 0, ids].clone(),
            "velocity": raw["velocity"][:, 0, ids].clone(),
            "joint_position": raw["joint_position"][:, 0].clone(),
            "elbow_position": raw["position"][
                :,
                0,
                [
                    BODY_NAMES.index(f"{side}_elbow_link")
                    for side in ("left", "right")
                ],
            ].clone(),
            "torso_position": raw["position"][
                :, 0, BODY_NAMES.index("torso_link")
            ].clone(),
            "opponent": raw["position"][:, 1, 0].clone(),
        }

    def _prepare(self, reset=None):
        current = self.state()
        d, a = geometry(current, self.config)
        reset = torch.zeros_like(self.attack) if reset is None else reset
        mode = attack_gate(self.attack & ~reset, d, a, self.config)
        change = (mode != self.attack) | reset
        self.attack = mode
        self.phase[change] = 0
        self.age[change] = 0
        self.anchor[change] = current["position"][change, 0]
        self.heading[change] = yaw(current["quaternion"][change, 0])
        desired_heading = yaw(current["quaternion"][:, 0]) + a
        difference = desired_heading - self.heading
        difference = torch.atan2(difference.sin(), difference.cos())
        turning = difference.clamp(
            -self.config.turn_rate_rad_s * self.dt,
            self.config.turn_rate_rad_s * self.dt,
        )
        self.heading += torch.where(
            mode & ~change, turning, torch.zeros_like(turning)
        )
        self.entry_p[change] = current["position"][change]
        self.entry_q[change] = current["quaternion"][change]
        self.entry_joints[change] = current["joint_position"][change]
        self.last_action[change] = 0

    def reset_training_rows(self, ids):
        # Combat resets terminals itself; explicit initial reset also resets
        # referee, lower caches and history. No reference-pose teleporting.
        self.combat._reset(ids)
        training_start_poses(
            self.physics.data.qpos,
            self.physics.root_qi,
            ids,
            self.rng,
            self.config,
        )
        mask = torch.zeros_like(self.attack)
        mask[ids] = True
        self.obs = self.combat._read(mask, previous=self.obs)
        self._prepare(mask)

    def references(self, offsets=(0, 1, 2, 3, 4, 5)):
        current = self.state()
        config = self.config
        d, bearing = geometry(current, config)
        close_align = ~self.attack & (d <= config.attack_exit_m)
        ref = cyclic_points(
            self.library, self.phase, offsets, config.cycle_bridge_steps
        )
        first = self.library.clips[0]
        angle = self.heading - yaw(first["point_quaternion"][0, 0])
        rotation = quat_matrix(yaw_quaternion(angle))
        attack_p = torch.einsum(
            "nij,ntbj->ntbi",
            rotation,
            ref["point_position"] - first["point_position"][0, 0],
        )
        attack_p += self.anchor[:, None, None]
        attack_q = quat_mul(
            yaw_quaternion(angle)[:, None, None], ref["point_quaternion"]
        )

        library = (
            self.movement_library
            if config.movement == "demo"
            else self.library
        )
        phases = self.phase if config.movement == "demo" else self.phase * 0
        move = cyclic_points(
            library,
            phases,
            offsets,
            config.cycle_bridge_steps,
            locomotion=True,
        )
        now = cyclic_points(
            library, phases, (0,), config.cycle_bridge_steps, locomotion=True
        )
        if config.movement == "explore":
            for key in ("point_position", "point_quaternion"):
                move[key] = now[key].expand(-1, len(offsets), -1, -1)
        for key in ("point_position", "point_quaternion"):
            move[key] = torch.where(
                close_align[:, None, None, None],
                now[key].expand(-1, len(offsets), -1, -1),
                move[key],
            )
        d, bearing = geometry(current, config)
        times = self.heading.new_tensor(offsets)[None] * self.dt
        turning = (bearing * 2).clamp(
            -config.turn_rate_rad_s, config.turn_rate_rad_s
        )
        angle = (
            yaw(current["quaternion"][:, 0])[:, None]
            - yaw(now["point_quaternion"][:, 0, 0])[:, None]
            + turning[:, None] * times
        )
        mq = yaw_quaternion(angle)
        mp = torch.einsum(
            "ntij,ntbj->ntbi",
            quat_matrix(mq),
            move["point_position"] - move["point_position"][:, :, :1],
        )
        travel_heading = library.clips[0]["guard_heading_world"]
        travel_angle = (
            yaw(current["quaternion"][:, 0])[:, None]
            + math.radians(config.guard_offset_deg)
            - travel_heading
            + turning[:, None] * times
        )
        travel = torch.einsum(
            "ntij,ntbj->ntbi",
            quat_matrix(yaw_quaternion(travel_angle)),
            move["point_position"][:, :, :1] - now["point_position"][:, :, :1],
        )
        if config.movement == "demo":
            # Slow root translation near stopping range or while turning;
            # preserve the demonstrated stepping shape relative to its pelvis.
            gain = ((d - config.stopping_distance_m) * 4).clamp(
                0, 1
            ) * bearing.cos().clamp(0, 1)
            travel *= gain[:, None, None, None]
        else:
            # Explore starts from a static stance; no punching-clip leg or
            # pelvis time evolution is used as a hidden movement reference.
            travel.zero_()
        mp += travel
        mp += current["position"][:, None, :1]
        mquat = quat_mul(mq[:, :, None], move["point_quaternion"])
        # In MOVE neither demonstrations nor policy residuals can request a
        # punch. Both wrists are rigid guard offsets of the requested pelvis.
        guard = self.guard_position
        mp[:, :, 1:3] = mp[:, :, :1] + torch.einsum(
            "ntij,bj->ntbi", quat_matrix(mquat[:, :, 0]), guard
        )
        guard_rot = self.guard_rotation
        mquat[:, :, 1:3] = quat_mul(mquat[:, :, :1], guard_rot)
        mask = self.attack[:, None, None, None]
        p = torch.where(mask, attack_p, mp)
        q = torch.where(mask, attack_q, mquat)
        alpha = (
            (self.age[:, None] + self.phase.new_tensor(offsets)[None]).float()
            / config.transition_steps
        ).clamp(0, 1)
        alpha = (alpha.square() * (3 - 2 * alpha))[..., None, None]
        p = self.entry_p[:, None] + alpha * (p - self.entry_p[:, None])
        q = quat_blend(self.entry_q[:, None], q, alpha)
        joints = self.entry_joints[:, None] + alpha[..., 0] * (
            ref["joint_position"] - self.entry_joints[:, None]
        )
        return {
            "point_position": p,
            "point_quaternion": q,
            "joint_position": joints,
            "bridge": ref["bridge"],
        }

    def get_observations(self):
        current = self.state()
        ref = self.references((1, 2, 3, 4, 5))
        targets = encode_output(
            current["position"],
            current["quaternion"],
            ref["point_position"],
            ref["point_quaternion"],
        )
        punch_period = (
            self.library.clips[0]["frames"]
            - 1
            + self.config.cycle_bridge_steps
        )
        move_period = (
            (
                self.movement_library.clips[0]["frames"]
                - 1
                + self.config.cycle_bridge_steps
            )
            if self.movement_library is not None
            else 1
        )
        period = torch.where(self.attack, punch_period, move_period)
        phase = 2 * math.pi * self.phase.remainder(period).float() / period
        return torch.cat(
            (
                self.obs["robot_A"]["task_policy"],
                targets,
                self.attack[:, None].float(),
                phase[:, None].sin(),
                phase[:, None].cos(),
                self.last_action,
            ),
            -1,
        ).clamp(-20, 20)

    def step(self, raw_action):
        if (
            raw_action.shape != (self.num_envs, 30)
            or not torch.isfinite(raw_action).all()
        ):
            raise ValueError("Expected finite residual [N,30]")
        before = self.state()
        attack = self.attack.clone()
        bounded = raw_action.tanh().reshape(self.num_envs, 5, 6)
        bounded = bounded.clone()
        bounded[~attack, 1:3] = 0
        distance, _ = geometry(before, self.config)
        close_align = ~attack & (distance <= self.config.attack_exit_m)
        bounded[close_align, :, :3] = 0
        ref = self.references((1, 2, 3, 4, 5, 6))
        scales = torch.where(
            attack[:, None],
            raw_action.new_tensor([0.015, 0.06, 0.06, 0.015, 0.015]),
            raw_action.new_tensor([0.12, 0, 0, 0.16, 0.16]),
        )
        delta = bounded[..., :3] * scales[..., None]
        dp = torch.einsum(
            "nij,nbj->nbi", quat_matrix(before["quaternion"][:, 0]), delta
        )
        p = ref["point_position"][:, :5] + dp[:, None]
        q = quat_mul(
            ref["point_quaternion"][:, :5],
            rotation_residual(bounded[..., 3:] * 0.15)[:, None],
        )
        # Both translation AND rotation of a pelvic residual carry the guard.
        old_p = ref["point_position"][:, :5]
        inverse = ref["point_quaternion"][:, :5, :1].clone()
        inverse[..., 1:] *= -1
        correction = quat_mul(q[:, :, :1], inverse)
        guarded_p = p[:, :, :1] + torch.einsum(
            "ntbij,ntbj->ntbi",
            quat_matrix(correction),
            old_p[:, :, 1:3] - old_p[:, :, :1],
        )
        guarded_q = quat_mul(correction, ref["point_quaternion"][:, :5, 1:3])
        p[~attack, :, 1:3] = guarded_p[~attack]
        q[~attack, :, 1:3] = guarded_q[~attack]
        encoded = encode_output(before["position"], before["quaternion"], p, q)
        blue = (
            encoded * 0
            if self.opponent is None
            else self.opponent(self.obs["robot_B"]["task_policy"])
        )
        self.obs, _, done, info = self.combat.step(
            {"robot_A": encoded, "robot_B": blue}
        )
        after = self.state(self.combat.last_transition_state)
        self.phase += (~close_align).long()
        self.age += 1
        stats = self.combat.metrics["robot_A"]
        invalid = info["invalid_envs"]
        reward, metrics = task_reward(
            before,
            after,
            ref,
            attack,
            self.age >= self.config.transition_steps,
            stats["official_hits"],
            stats["fallen"],
            invalid,
            bounded.flatten(1),
            self.last_action,
            self.config,
        )
        self.last_action = bounded.flatten(1).detach().clone()
        self._prepare(done)
        return (
            self.get_observations(),
            reward,
            done,
            {
                "completed": done & ~invalid & ~stats["fallen"],
                "failed": invalid | stats["fallen"],
                "metrics": metrics,
            },
        )
