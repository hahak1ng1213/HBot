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
    attack_exit_m: float = 1.5
    attack_enter_m: float = 0.65
    stopping_distance_m: float = 0.56
    facing_enter_deg: float = 90.0
    facing_exit_deg: float = 90.0
    guard_offset_deg: float = 0.0
    transition_steps: int = 10
    cycle_bridge_steps: int = 20
    turn_rate_rad_s: float = 0.6
    movement: str = "demo"
    facing_alignment_weight: float = 3.0
    facing_progress_weight: float = 1.5
    punch_tracking_scale: float = 2.0
    no_contact_grace_s: float = 5.0
    no_contact_cost_per_s: float = 0.5
    reach_weight: float = 4.0
    reach_gap_scale_m: float = 0.18
    reach_forward_allowance_m: float = 0.12

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
            < self.attack_exit_m
        ):
            raise ValueError("Require 0 < stop < enter < exit")
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
        if (
            min(
                self.facing_alignment_weight,
                self.facing_progress_weight,
                self.punch_tracking_scale,
                self.no_contact_grace_s,
                self.no_contact_cost_per_s,
                self.reach_weight,
                self.reach_gap_scale_m,
                self.reach_forward_allowance_m,
            )
            <= 0
        ):
            raise ValueError(
                "Reward weights and contact grace must be positive"
            )


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
    """Latch attack below enter; release only strictly above exit.

    Heading controls punch execution separately, never the distance latch.
    """
    finite = torch.isfinite(distance) & torch.isfinite(angle)
    enter = distance < config.attack_enter_m
    remain = distance <= config.attack_exit_m
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
    engaged=None,
    no_torso_contact_steps=None,
    reach=None,
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
    engaged = attack if engaged is None else engaged
    approaching = (~engaged).float()
    contact_age = (
        torch.zeros_like(distance)
        if no_torso_contact_steps is None
        else no_torso_contact_steps.float() * dt
    )
    # Linear five-point transitions and linear joint transitions need not
    # have the same FK. Only supervise anatomy on original motion frames.
    anatomy = a * settled.float() * (~reference["bridge"][:, 0]).float()
    # Reach shaping is enabled only for the outward part of the demonstrated
    # punch, never the recovery, guard or transition. Body motion is removed
    # by the measured geometry before calculating arm speed.
    forward = quat_matrix(after["quaternion"][:, 0])[:, :, 0]
    reference_forward_speed = (desired_relative * forward[:, None]).sum(-1)
    outward_phase = reference_forward_speed > 0.02
    strike_window = (
        outward_phase
        & anatomy[:, None].bool()
        & in_front(angle)[:, None]
        & (distance <= 0.85)[:, None]
    )
    reach_value = torch.zeros_like(distance)
    min_gap = torch.zeros_like(distance)
    if reach is not None:
        valid = reach["target_valid"]
        if bool((~valid & ~invalid[:, None]).any()):
            raise FloatingPointError(
                "Nonfinite reach geometry in a healthy world"
            )
        gap = reach["gap_m"].clamp_min(0)
        min_gap = gap.amin(-1)
        aimed_speed = ((reach["arm_aim_speed_mps"] - 0.05) / 0.75).clamp(0, 1)
        outward_speed = (reach["arm_outward_speed_mps"] / 0.3).clamp(0, 1)
        closing = (reach["closing_speed_mps"] / 0.3).clamp(0, 1)
        reach_value = (
            strike_window.float()
            * torch.exp(-gap / config.reach_gap_scale_m)
            * aimed_speed
            * outward_speed
            * closing
        ).amax(-1)
    # Allow a modest extra extension along the opponent direction without
    # weakening lateral/height tracking or any recovery-frame tracking.
    direction = after["opponent"] - own[:, 0]
    direction[:, 2] = 0
    direction = torch.nn.functional.normalize(direction, dim=-1)
    wrist_difference = (own[:, 1:3] - own[:, :1]) - (
        target[:, 1:3] - target[:, :1]
    )
    extension = (wrist_difference * direction[:, None]).sum(-1)
    # Taper the allowance around the punch apex instead of requesting an
    # abrupt 12 cm snap-back when the reference velocity changes sign.
    tracking_window = (
        ((reference_forward_speed + 0.10) / 0.20).clamp(0, 1)
        * anatomy[:, None]
        * in_front(angle)[:, None]
        * (distance <= 0.85)[:, None]
    )
    allowed = torch.minimum(
        extension.clamp_min(0),
        config.reach_forward_allowance_m * tracking_window,
    )
    adjusted = wrist_difference - allowed[..., None] * direction[:, None]
    tracking_wrist_error = adjusted.square().mean((1, 2)).sqrt()
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
    arm_tracking = []
    for hand, side in enumerate(("left", "right")):
        ids = [i for i in arm_ids if JOINT_NAMES[i].startswith(side + "_")]
        error = joint_error[:, ids].square().mean(-1).sqrt()
        arm_tracking.append(
            (1 - 0.5 * tracking_window[:, hand])
            * torch.exp(-(error / 0.35).square())
        )
    leg_error = joint_error[:, leg_ids].square().mean(-1).sqrt()
    terms = {
        "approach_progress": dt
        * approaching
        * (distance0 >= config.attack_enter_m).float()
        * 2.0
        * progress,
        "move_guard_tracking": dt
        * moving
        * 2.0
        * torch.exp(-(wrist_error / 0.08).square()),
        "facing_progress": dt
        * config.facing_progress_weight
        * facing_progress,
        "facing_alignment": dt * config.facing_alignment_weight * angle.cos(),
        "no_torso_contact": -dt
        * config.no_contact_cost_per_s
        * (contact_age > config.no_contact_grace_s).float(),
        "punch_reach": dt * config.reach_weight * reach_value * upright,
        "punch_wrist_position": dt
        * a
        * 2.0
        * config.punch_tracking_scale
        * torch.exp(-(tracking_wrist_error / 0.10).square()),
        "punch_wrist_orientation": dt
        * a
        * 0.3
        * config.punch_tracking_scale
        * torch.exp(-rotation_error / 0.16),
        "punch_relative_velocity": dt
        * a
        * settled.float()
        * 0.5
        * config.punch_tracking_scale
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
        * config.punch_tracking_scale
        * torch.stack(arm_tracking, -1).mean(-1),
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
        "attack_mode": engaged.float(),
        "punch_active": a,
        "alignment_only": (engaged & ~attack).float(),
        "no_torso_contact_s": contact_age,
        "wrist_error_m": wrist_error,
        "reach_tracking_wrist_error_m": tracking_wrist_error,
        "glove_torso_gap_m": min_gap,
        "reach_window": strike_window.any(-1).float(),
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
    checkpoint_format = "kangning037_range_teacher_v6"
    ppo_target_kl = 0.03
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
        from holoboxing.training.skills.punch_reach import PunchReachGeometry

        self.reach_geometry = PunchReachGeometry(self.physics.host)
        self.combat.transition_measurement = lambda env: {
            "reach_" + key: value
            for key, value in self.reach_geometry.measure(env).items()
        }
        self.physics.metadata = {
            "backend": "competition_cpu",
            "official_rules": True,
            "frozen_lower": True,
            "control_dt": self.dt,
        }
        self.rng = torch.Generator(device=self.device).manual_seed(seed)
        n = self.num_envs
        self.attack = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.punching = self.attack.clone()
        self.phase = torch.zeros(n, dtype=torch.long, device=self.device)
        self.age = self.phase.clone()
        self.no_torso_contact_steps = self.phase.clone()
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
            "distance_latch": "enter strictly below 0.65 m; retain through 1.5 m; exit strictly above 1.5 m; reset clears latch",
            "close_alignment": "while latched and opponent not in front: hold guard and turn, pause punch phase; heading never clears distance latch",
            "motion": None if movement is None else movement.manifest,
            "training_resets": "50% distance [0.52,0.58], facing +/-10 deg; 50% [0.65,2.0], facing +/-180 deg; random world bearing; evaluation uses combat reset",
            "guard_axis": "pelvis yaw plus explicit guard_offset_deg; calibration pending",
            "attack_alignment": "rotate source anchor heading toward opponent at <=0.6 rad/s; preserve source joints and relative motion",
            "facing_reward": {
                "alignment": "dt*facing_alignment_weight*cos(error)",
                "progress": "dt*facing_progress_weight*clip((abs(old)-abs(new))/dt,-2,2)",
                "modes": "move and punch",
            },
            "contact_timeout": "from episode start or last physical hand-torso contact; all modes; no positive reward for resting contact; official scoring unchanged",
            "reach_reward": {
                "geometry": "signed surface gap of collidable official glove to opponent torso/chest; excludes head/pelvis",
                "gate": "settled original outward punch phase, front half-plane, pelvis distance <= 0.85 m",
                "value": "dt*4*max_h(exp(-max(gap,0)/0.18)*clip((arm_aim_speed-0.05)/0.75,0,1)*clip(outward_arm_speed/0.3,0,1)*clip(closing_speed/0.3,0,1))*upright",
                "tracking": "allow at most 0.12 m extra toward-opponent extension, taper over reference forward speed [-0.1,0.1] m/s; halve only active hand arm-joint shaping; full recovery unchanged",
                "residual": "positive wrist local-x bound 0.18 m; negative/lateral/vertical 0.06 m; pelvis/feet unchanged",
            },
            "ppo_update_guard": {
                "target_kl": self.ppo_target_kl,
                "measure": "exact Gaussian old-to-new KL over full rollout",
                "on_excess": "rollback model and optimizer; halve learning rate; stop this update",
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
        self.no_torso_contact_steps[reset] = 0
        mode = attack_gate(self.attack & ~reset, d, a, self.config)
        mode_change = (mode != self.attack) | reset
        heading_limit = torch.where(
            self.punching & ~reset,
            self.config.facing_exit_deg,
            self.config.facing_enter_deg,
        )
        punching = (
            mode & in_front(a) & (a.abs() <= torch.deg2rad(heading_limit))
        )
        change = mode_change | (punching != self.punching)
        self.attack = mode
        self.punching = punching
        self.phase[mode_change] = 0
        self.age[change] = 0
        self.anchor[mode_change] = current["position"][mode_change, 0]
        self.heading[mode_change] = yaw(current["quaternion"][mode_change, 0])
        desired_heading = yaw(current["quaternion"][:, 0]) + a
        difference = desired_heading - self.heading
        difference = torch.atan2(difference.sin(), difference.cos())
        turning = difference.clamp(
            -self.config.turn_rate_rad_s * self.dt,
            self.config.turn_rate_rad_s * self.dt,
        )
        self.heading += torch.where(
            mode & ~mode_change, turning, torch.zeros_like(turning)
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
        close_align = self.attack & ~self.punching
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
        mask = self.punching[:, None, None, None]
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
                (
                    self.no_torso_contact_steps.float()
                    * self.dt
                    / self.config.no_contact_grace_s
                ).clamp(0, 2)[:, None],
            ),
            -1,
        ).clamp(-20, 20)

    def action_targets(self, raw_action):
        if (
            raw_action.shape != (self.num_envs, 30)
            or not torch.isfinite(raw_action).all()
        ):
            raise ValueError("Expected finite residual [N,30]")
        before = self.state()
        attack = self.attack.clone()
        punching = self.punching.clone()
        bounded = raw_action.tanh().reshape(self.num_envs, 5, 6)
        bounded = bounded.clone()
        bounded[~punching, 1:3] = 0
        close_align = attack & ~punching
        bounded[close_align, :, :3] = 0
        ref = self.references((1, 2, 3, 4, 5, 6))
        scales = torch.where(
            punching[:, None],
            raw_action.new_tensor([0.015, 0.06, 0.06, 0.015, 0.015]),
            raw_action.new_tensor([0.12, 0, 0, 0.16, 0.16]),
        )
        delta = bounded[..., :3] * scales[..., None]
        delta[:, 1:3, 0] += (
            punching[:, None]
            * bounded[:, 1:3, 0].clamp_min(0)
            * self.config.reach_forward_allowance_m
        )
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
        p[~punching, :, 1:3] = guarded_p[~punching]
        q[~punching, :, 1:3] = guarded_q[~punching]
        return bounded, ref, p, q

    def step(self, raw_action):
        before = self.state()
        attack = self.attack.clone()
        punching = self.punching.clone()
        phase_before = self.phase.clone()
        close_align = attack & ~punching
        bounded, ref, p, q = self.action_targets(raw_action)
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
        torso_contact = stats["official_torso_contact_samples"] > 0
        self.no_torso_contact_steps = torch.where(
            torso_contact,
            torch.zeros_like(self.no_torso_contact_steps),
            self.no_torso_contact_steps + 1,
        )
        reward, metrics = task_reward(
            before,
            after,
            ref,
            punching,
            self.age >= self.config.transition_steps,
            stats["official_hits"],
            stats["fallen"],
            invalid,
            bounded.flatten(1),
            self.last_action,
            self.config,
            engaged=attack,
            no_torso_contact_steps=self.no_torso_contact_steps,
            reach={
                key.removeprefix("reach_"): value
                for key, value in self.combat.last_transition_state.items()
                if key.startswith("reach_")
            },
        )
        self.last_action = bounded.flatten(1).detach().clone()
        self._prepare(done)
        metrics["reference_phase"] = phase_before.float()
        metrics["torso_contact"] = (torso_contact & ~invalid).float()
        metrics["mode_changed"] = (self.attack != attack).float()
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
