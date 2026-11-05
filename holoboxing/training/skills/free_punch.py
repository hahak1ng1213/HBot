"""Explore complete hit-and-recover punches on top of a frozen v6 skill.

No upper-body pose/velocity/joint imitation reward. The v6 motion is a prior,
not the objective: wrist corrections span 0.40 m and 0.8 rad in either direction.
"""

import torch

from holoboxing.training.skills.actor import SkillActor
from holoboxing.training.skills.env import (
    quat_matrix,
    quat_mul,
    rotation_residual,
)
from holoboxing.training.skills.fixed_punch import (
    FixedPunchCombatEnv,
    geometry,
)


class PunchCycles:
    """Per-hand guard -> outward -> extension/hit -> guard state machine."""

    def __init__(self, n, device):
        self.phase = torch.zeros(n, 2, device=device, dtype=torch.long)
        self.age = torch.zeros(n, 2, device=device)
        self.hit = torch.zeros(n, 2, device=device, dtype=torch.bool)
        self.interval = torch.zeros(n, 2, device=device)

    def reset(self, mask):
        for value in (self.phase, self.age, self.hit, self.interval):
            value[mask] = 0

    def observations(self):
        return torch.cat(
            (
                self.phase.float() / 3,
                (self.age / 3).clamp(0, 2),
                self.hit.float(),
                (self.interval / 5).clamp(0, 2),
            ),
            -1,
        )

    def advance(self, guard_distance, extension, hits, active, stable, dt):
        self.interval += active[:, None] * dt
        self.interval.clamp_(max=10)
        self.age += (self.phase >= 2) * dt
        guard = guard_distance < 0.12
        self.phase = torch.where((self.phase == 0) & guard, 1, self.phase)
        launch = (self.phase == 1) & (guard_distance > 0.15) & active[:, None]
        self.phase[launch] = 2
        self.age[launch] = 0
        self.hit[launch] = False
        self.hit |= (hits > 0) & (self.phase >= 2)
        extended = (self.phase == 2) & ((extension > 0.20) | (hits > 0))
        self.phase[extended] = 3
        completed = (
            (self.phase == 3) & guard & (self.age >= 0.25) & (self.age <= 3)
        )
        completed &= active[:, None] & stable[:, None]
        hit_completed = completed & self.hit
        duration, interval = self.age.clone(), self.interval.clone()
        clear = (
            completed | (self.age > 3) | ~active[:, None] | ~stable[:, None]
        )
        self.phase[clear] = 0
        self.age[clear] = 0
        self.hit[clear] = False
        self.interval[hit_completed] = 0
        return completed, hit_completed, duration, interval


class PaceCurriculum:
    durations = (1.4, 1.2, 1.0, 0.85, 0.70)
    window_updates = 64

    def __init__(self):
        self.stage = 0
        self.window = []

    @property
    def duration(self):
        return self.durations[self.stage]

    def observe(self, update, metrics, transitions):
        self.window.append(
            {
                k: metrics[k]
                for k in (
                    "hit_complete_cycles",
                    "complete_cycles",
                    "low_pose",
                    "invalid",
                )
            }
        )
        change = "hold"
        if len(self.window) >= self.window_updates:
            hits = (
                sum(x["hit_complete_cycles"] for x in self.window)
                * transitions
            )
            cycles = (
                sum(x["complete_cycles"] for x in self.window) * transitions
            )
            fall = sum(x["low_pose"] for x in self.window) / len(self.window)
            invalid = sum(x["invalid"] for x in self.window)
            if (fall > 0.03 or invalid > 0) and self.stage > 0:
                self.stage -= 1
                change = "slow_down_for_stability"
            elif (
                hits >= 8
                and cycles >= 12
                and fall <= 0.01
                and invalid == 0
                and self.stage < len(self.durations) - 1
            ):
                self.stage += 1
                change = "speed_up_after_effective_recovery"
            self.window = []
        return {
            "update": update,
            "stage": self.stage,
            "target_cycle_s": self.duration,
            "change": change,
            "window_updates": len(self.window),
        }


class FreePunchCombatEnv(FixedPunchCombatEnv):
    checkpoint_format = "kangning037_free_punch_v7"

    def configure(self, mentor, frozen_blue, identity, rollout_steps):
        self.mentor, self.opponent = mentor, frozen_blue
        self.cycles = PunchCycles(self.num_envs, self.device)
        self.pace = PaceCurriculum()
        self.learner_last_action = torch.zeros(
            self.num_envs, 30, device=self.device
        )
        self.learner_action = self.learner_last_action.clone()
        self.no_hit_steps = torch.zeros(self.num_envs, device=self.device)
        self.transitions_per_update = rollout_steps * self.num_envs
        self._inside_step = False
        self.observation_dim = self.get_observations().shape[-1]
        actor = SkillActor(self.observation_dim).to(self.device)
        with torch.no_grad():
            actor.log_std.fill_(-3.0)
        self.initial_model_state = actor.state_dict()
        self.initialization = {
            "kind": "new_zero_residual_actor_over_frozen_v6_9536",
            "mentor": identity,
            "fresh_optimizer": True,
        }
        prior_recipe = self.recipe
        self.recipe = {
            "prior_execution_recipe": prior_recipe,
            "resources": prior_recipe.get("resources"),
        }
        self.recipe.update(
            version=self.checkpoint_format,
            opponent=identity,
            mentor=identity,
            reward="free_punch.step: HIT +10 unconditionally, hit/recovery cycle <=1.5, no upper imitation",
            reward_weights={
                "official_hit": 10,
                "received_hit": -1,
                "hit_recovery": "0.5 + speed_quality * effective_cycle_frequency (max 1.5)",
                "dt_scaled": {
                    "approach": 2,
                    "facing": 1,
                    "turn_progress": 0.5,
                    "reach": 1,
                    "retract": 0.4,
                    "no_hit": -0.5,
                    "upright": 0.3,
                    "tilt": -2,
                    "feet": 0.3,
                    "too_low": -4,
                    "action_rate": -0.1,
                    "residual": -0.01,
                },
                "fallen_each_step": -2,
                "invalid_override": -10,
            },
            free_action={
                "wrist_translation_m": 0.40,
                "wrist_rotation_rad": 0.8,
                "pelvis_crouch_m": 0.08,
                "feet_translation_m": 0.015,
                "move_wrists": "original compact guard; no learner punch",
            },
            curriculum={
                "cycle_duration_s": self.pace.durations,
                "window_updates": 64,
                "promote": "8 hit-recovery cycles, 12 total full cycles, low-pose <=1%, invalid=0",
                "demote": "low-pose >3% or invalid state",
                "not_wall_clock_driven": True,
            },
            observation="1356 mentor inputs + 8 cycle states + 30 previous learner actions + pace stage/duration + no-hit clock",
        )

    def get_observations(self):
        base = super().get_observations()
        if not hasattr(self, "cycles"):
            return base
        hints = base.new_tensor(
            [self.pace.stage / 4, self.pace.duration]
        ).expand(self.num_envs, -1)
        return torch.cat(
            (
                base,
                self.cycles.observations(),
                self.learner_last_action,
                hints,
                (self.no_hit_steps * self.dt / 5).clamp(0, 2)[:, None],
            ),
            -1,
        )

    def _prepare(self, reset=None):
        super()._prepare(reset)
        if (
            hasattr(self, "cycles")
            and reset is not None
            and not self._inside_step
        ):
            self.cycles.reset(reset)
            self.no_hit_steps[reset] = 0
            self.learner_last_action[reset] = 0
            self.opponent.reset(self.obs, reset)

    def action_targets(self, raw_action):
        bounded, ref, p, q = super().action_targets(raw_action)
        extra = self.learner_action.reshape(self.num_envs, 5, 6).clone()
        extra[~self.punching] = (
            0  # preserve learned movement and compact guard
        )
        rotation = quat_matrix(self.state()["quaternion"][:, 0])
        scales = p.new_tensor([0.02, 0.40, 0.40, 0.015, 0.015])
        translation = extra[..., :3] * scales[None, :, None]
        translation[:, 0, 2] = -extra[:, 0, 2].clamp_min(0) * 0.08
        world = torch.einsum("nij,nbj->nbi", rotation, translation)
        p = p + world[:, None]
        # A crouch carries both hand targets down while leaving feet planted.
        p[:, :, 1:3, 2] += world[:, None, :1, 2]
        rotation_scale = p.new_tensor([0.05, 0.8, 0.8, 0.05, 0.05])
        q = quat_mul(
            q,
            rotation_residual(extra[..., 3:] * rotation_scale[None, :, None])[
                :, None
            ],
        )
        return bounded, ref, p, q

    @torch.no_grad()
    def step(self, raw_action):
        if (
            raw_action.shape != (self.num_envs, 30)
            or not torch.isfinite(raw_action).all()
        ):
            raise ValueError("Expected finite free punch actions [N,30]")
        before = self.state()
        ref = self.references((1,))["point_position"][:, 0]
        active = self.punching.clone()
        self.learner_action = raw_action.tanh()
        base_action = self.mentor.act(
            super().get_observations(), deterministic=True
        )
        self._inside_step = True
        try:
            _, _, done, old_info = super().step(base_action)
        finally:
            self._inside_step = False
        stats = self.combat.metrics["robot_A"]
        other = self.combat.metrics["robot_B"]
        invalid = old_info["metrics"]["invalid"].bool()
        self.opponent.after_step(self.obs, other, done)
        after = self.state(self.combat.last_transition_state)
        distance0, angle0 = geometry(before, self.config)
        distance, angle = geometry(after, self.config)
        rot = quat_matrix(after["quaternion"][:, 0])
        local = torch.einsum(
            "nij,nbj->nbi",
            rot.transpose(-1, -2),
            after["position"][:, 1:3] - after["position"][:, :1],
        )
        old_local = torch.einsum(
            "nij,nbj->nbi",
            quat_matrix(before["quaternion"][:, 0]).transpose(-1, -2),
            before["position"][:, 1:3] - before["position"][:, :1],
        )
        guard_distance = (local - self.guard_position).norm(dim=-1)
        old_guard_distance = (old_local - self.guard_position).norm(dim=-1)
        upright = rot[:, 2, 2].clamp(0, 1)
        stable = (upright > 0.8) & ~stats["fallen"] & ~invalid
        per_hand = torch.stack(
            (stats["official_left_hits"], stats["official_right_hits"]), -1
        )
        retracting = self.cycles.phase == 3
        complete, hit_complete, duration, interval = self.cycles.advance(
            guard_distance, guard_distance, per_hand, active, stable, self.dt
        )
        self.no_hit_steps = torch.where(
            stats["official_hits"] > 0, 0, self.no_hit_steps + 1
        )
        reach = {
            k.removeprefix("reach_"): v
            for k, v in self.combat.last_transition_state.items()
            if k.startswith("reach_")
        }
        aimed = ((reach["arm_aim_speed_mps"] - 0.05) / 0.75).clamp(0, 1)
        outward = (reach["arm_outward_speed_mps"] / 0.3).clamp(0, 1)
        closing = (reach["closing_speed_mps"] / 0.3).clamp(0, 1)
        reach_value = (
            torch.exp(-reach["gap_m"].clamp_min(0) / 0.18)
            * aimed
            * outward
            * closing
        ).amax(-1)
        quality = torch.exp(
            -(duration - self.pace.duration).clamp_min(0) / self.pace.duration
        )
        frequency = (
            self.pace.duration / interval.clamp_min(self.pace.duration)
        ).clamp(0, 1)
        terms = {
            "official_hit": 10 * stats["official_hits"],
            "received_hit": -other["official_hits"],
            "hit_recovery": (hit_complete * (0.5 + quality * frequency)).sum(
                -1
            ),
            "approach": self.dt
            * 2
            * (
                ((distance0 - 0.55).abs() - (distance - 0.55).abs()) / self.dt
            ).clamp(-1, 1),
            "facing": self.dt * angle.cos(),
            "turn_progress": self.dt
            * 0.5
            * ((angle0.abs() - angle.abs()) / self.dt).clamp(-2, 2),
            "reach": self.dt * reach_value * active * stable,
            "retract": self.dt
            * 0.4
            * (
                retracting
                * ((old_guard_distance - guard_distance) / self.dt).clamp(
                    -1, 1
                )
            ).mean(-1)
            * active,
            "no_hit": -self.dt
            * 0.5
            * ((self.no_hit_steps * self.dt - 5) / 5).clamp(0, 1),
            "balance": self.dt * (0.3 * upright - 2 * (1 - upright)),
            "feet": self.dt
            * 0.3
            * torch.exp(
                -(
                    (after["position"][:, 3:5] - ref[:, 3:5])
                    .square()
                    .mean((1, 2))
                    / 0.01
                )
            ),
            "too_low": -self.dt
            * 4
            * ((ref[:, 0, 2] - 0.10 - after["position"][:, 0, 2]) / 0.1).clamp(
                0, 1
            ),
            "action_rate": -self.dt
            * 0.1
            * (self.learner_action - self.learner_last_action)
            .square()
            .mean(-1),
            "residual": -self.dt
            * 0.01
            * self.learner_action.square().mean(-1),
            "fallen": -2 * stats["fallen"].float(),
        }
        terms = {
            k: torch.where(invalid, torch.zeros_like(v), v)
            for k, v in terms.items()
        }
        reward = torch.where(invalid, -10.0, sum(terms.values()))
        diagnostics = {
            "official_hits": stats["official_hits"],
            "complete_cycles": complete.sum(-1).float(),
            "hit_complete_cycles": hit_complete.sum(-1).float(),
            "low_pose": stats["fallen"].float(),
            "distance_m": distance,
            "facing_error_rad": angle.abs(),
            "no_hit_s": self.no_hit_steps * self.dt,
            "target_cycle_s": distance.new_full(
                distance.shape, self.pace.duration
            ),
        }
        metrics = {
            **terms,
            **{k: torch.where(invalid, 0, v) for k, v in diagnostics.items()},
            "invalid": invalid.float(),
        }
        metrics = {key: value.float() for key, value in metrics.items()}
        self.cycles.reset(done)
        self.no_hit_steps[done] = 0
        self.learner_last_action = self.learner_action.clone()
        self.learner_last_action[done] = 0
        return (
            self.get_observations(),
            reward,
            done,
            {
                "metrics": metrics,
                "failed": done & (invalid | stats["fallen"]),
                "completed": done & ~invalid & ~stats["fallen"],
            },
        )

    def advance_curriculum(self, update, metrics):
        return self.pace.observe(update, metrics, self.transitions_per_update)

    def curriculum_state(self):
        return {"stage": self.pace.stage, "window": self.pace.window}

    def restore_curriculum(self, state):
        if (
            not state
            or type(state.get("stage")) is not int
            or not 0 <= state["stage"] < len(self.pace.durations)
        ):
            raise ValueError("Invalid free-punch curriculum checkpoint")
        self.pace.stage, self.pace.window = state["stage"], state["window"]
