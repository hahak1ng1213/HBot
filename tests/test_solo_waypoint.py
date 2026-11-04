"""Controller accounting tests; fake physics is not locomotion evidence."""

import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holoboxing.training.skills.waypoint import (
    WaypointConfig,
    capture_hold,
    hold_targets,
    measured_goal,
    run_waypoint,
    turn_targets,
    verify_teacher_contract,
    yaw,
)


def state(x=0.0, y=0.0, angle=0.0, speed=0.0, fallen=False):
    root = torch.tensor([[x, y, 0.8]], dtype=torch.float32)
    quaternion = torch.tensor(
        [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]]
    )
    points = root[:, None] + torch.tensor(
        [
            [
                [0, 0, 0],
                [0.2, 0.2, 0.1],
                [0.2, -0.2, 0.1],
                [0, 0.1, -0.75],
                [0, -0.1, -0.75],
            ]
        ]
    )
    return {
        "root_position": root,
        "root_quaternion": quaternion,
        "root_linear_velocity": torch.tensor([[speed, 0.0, 0.0]]),
        "root_angular_velocity": torch.zeros((1, 3)),
        "point_position": points,
        "point_quaternion": quaternion[:, None].expand(-1, 5, -1).clone(),
        "fallen": torch.tensor([fallen]),
    }


class Teacher:
    def eval(self):
        pass

    def act(self, observation, deterministic=False):
        assert deterministic
        return torch.zeros((1, 30))


class FakeEnv:
    """Records resets and models enough motion to test the controller."""

    num_envs = 1
    device = torch.device("cpu")
    dt = 0.02
    root_qi = 0
    initial_qpos = np.array([0, 0, 0.8, 1, 0, 0, 0])
    metadata = {"fake": True}
    recipe = {"fake": True}
    observation_dim = 2

    def __init__(self, fall_on_hold=False):
        clip = {
            "name": "forward_1",
            "frames": 3,
            "guard_heading_world": 0.0,
            "root_quaternion": torch.tensor([[1.0, 0, 0, 0]]),
            "direction": torch.tensor([1.0, 0.0]),
        }
        self.library = SimpleNamespace(
            clips=[clip], manifest={"test_only": True}
        )
        self.physics = self
        self._state = state()
        self.frame = torch.zeros(1, dtype=torch.long)
        self.last_action = torch.zeros((1, 30))
        self.resets = []
        self.holds = []
        self.turn_requests = []
        self.fall_on_hold = fall_on_hold

    def start(self, clips, *, reset_physics, neutral, random_phase, yaw):
        assert not random_phase
        self.resets.append(reset_physics)
        if reset_physics:
            assert neutral
            self._state = state(angle=float(yaw))
        self.frame[:] = 0
        # Match SkillTrackingEnv.start's mutation of saved previous actions.
        self.last_action[:] = 0
        return self.get_observations()

    def state(self):
        return self._state

    def get_observations(self):
        return torch.zeros((1, 2))

    def step(self, *args):
        if len(args) == 2:
            p, q = args
            self.holds.append((p.clone(), q.clone()))
            self._state = state(
                float(p[0, 0, 0, 0]),
                float(p[0, 0, 0, 1]),
                yaw(q[0, 0, 0]),
                fallen=self.fall_on_hold,
            )
            return self._state
        self.frame += 1
        # A tensor produced under inference_mode cannot be mutated by start
        # outside that context; this catches repeated-primitive integration.
        self.last_action = args[0].tanh().detach().clone()
        angle = yaw(self._state["root_quaternion"])
        position = self._state["root_position"][0]
        self._state = state(
            float(position[0]) + 0.1 * math.cos(angle),
            float(position[1]) + 0.1 * math.sin(angle),
            angle,
            speed=1.0,
        )
        info = {"failed": torch.tensor([False]), "completed": self.frame >= 3}
        return self.get_observations(), None, None, info


def test_hold_remains_world_fixed_and_does_not_modify_measured_state():
    actual = state(1, 2, speed=1)
    snapshot = capture_hold(actual)
    actual["point_position"] += 5
    actual["root_linear_velocity"][:] = 4
    p, q = hold_targets(snapshot)
    assert p.shape == (1, 5, 5, 3)
    torch.testing.assert_close(p[:, 0], p[:, -1])
    assert float(p[0, 0, 0, 0]) == 1
    assert float(actual["root_linear_velocity"][0, 0]) == 4
    assert q.shape == (1, 5, 5, 4)


def test_turn_targets_limit_rate_and_preserve_rigid_geometry():
    actual = state(2, 3)
    snapshot = capture_hold(actual)
    cfg = WaypointConfig(turn_rate_radps=0.2)
    p, q = turn_targets(snapshot, actual["root_position"], 0, 0.1, cfg)
    assert yaw(q[0, 0, 0]) == pytest.approx(0.004, abs=1e-6)
    assert yaw(q[0, -1, 0]) == pytest.approx(0.020, abs=1e-6)
    torch.testing.assert_close(
        (p[:, 0, 1] - p[:, 0, 0]).norm(dim=-1),
        (snapshot[0][:, 1] - snapshot[0][:, 0]).norm(dim=-1),
    )
    torch.testing.assert_close(q.norm(dim=-1), torch.ones((1, 5, 5)))
    p2, q2 = turn_targets(snapshot, actual["root_position"], 10, -0.1, cfg)
    assert yaw(q2[0, -1, 0]) == pytest.approx(-0.1, abs=1e-6)
    torch.testing.assert_close(p2[:, 0], p2[:, -1])
    torch.testing.assert_close(actual["point_position"], snapshot[0])


def test_forward_chaining_and_early_goal_hold_never_reset_physics():
    env = FakeEnv()
    report, rows = run_waypoint(
        env, Teacher(), (1.0, 0), WaypointConfig(max_seconds=3)
    )
    assert report["arrival"]
    assert report["stable_arrival_seconds"] >= 1
    assert env.resets[0] is True
    assert len(env.resets) > 2
    assert not any(env.resets[1:])
    assert report["learned_navigation"] is False
    assert report["final_speed_mps"] == 0
    assert report["final_distance_m"] <= 0.3
    hold = next(e for e in report["events"] if e["event"] == "early_goal_hold")
    assert hold["distance_m"] <= 0.3
    # Stops before the next complete three-step primitive endpoint.
    first_hold = np.flatnonzero(rows["mode"] == "hold")[0]
    assert rows["source_frame"][first_hold] < 3
    np.testing.assert_allclose(np.diff(rows["simulation_time_s"]), 0.02)


def test_crossing_goal_and_falling_cannot_count_as_arrival():
    env = FakeEnv(fall_on_hold=True)
    report, _ = run_waypoint(
        env, Teacher(), (0.5, 0), WaypointConfig(max_seconds=3)
    )
    assert report["minimum_distance_m"] <= 0.3
    assert not report["arrival"]
    assert report["failed"]
    assert report["fallen_samples"] == 1


def test_far_goal_timeout_is_not_success_and_budget_is_exact():
    env = FakeEnv()
    report, rows = run_waypoint(
        env, Teacher(), (10, 0), WaypointConfig(max_seconds=0.2)
    )
    assert not report["arrival"]
    assert report["end_reason"] == "time_limit"
    assert report["physical_steps"] == 10
    assert len(rows["mode"]) == 11


def test_oblique_goal_turns_physically_without_goal_facing_reset():
    env = FakeEnv()
    report, _ = run_waypoint(
        env, Teacher(), (3, 3), WaypointConfig(max_seconds=2)
    )
    assert env.resets == [True]
    assert env.holds
    turns = [e for e in report["events"] if e["event"] == "scripted_turn"]
    assert turns
    assert all(
        abs(e["requested_yaw_delta_rad"]) <= math.radians(10) for e in turns
    )
    assert not report["arrival"]


def test_arrival_needs_low_speed_not_only_distance():
    info = measured_goal(state(0.1, 0, speed=3), (0, 0), 0)
    assert info["distance_m"] < 0.3
    assert info["speed_mps"] > 0.15


def test_stationary_tilted_robot_is_not_accepted_as_stable_arrival():
    class TiltedEnv(FakeEnv):
        def step(self, *args):
            result = super().step(*args)
            angle = 0.7
            self._state["root_quaternion"] = torch.tensor(
                [[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]]
            )
            return self._state if len(args) == 2 else result

    env = TiltedEnv()
    report, _ = run_waypoint(
        env, Teacher(), (0.0, 0.0), WaypointConfig(max_seconds=4)
    )
    assert report["final_distance_m"] == 0
    assert report["final_speed_mps"] == 0
    assert report["final_root_tilt_rad"] == pytest.approx(0.7, abs=1e-6)
    assert not report["arrival"]
    assert report["end_reason"] == "hold_failed_to_stop_at_goal"


def test_contract_rejects_other_library_and_non_forward_teacher():
    env = FakeEnv()
    saved = {
        "reference_manifest": env.library.manifest,
        "environment_recipe": env.recipe,
        "physics_metadata": env.metadata,
        "observation_dim": env.observation_dim,
        "skill": "locomotion",
    }
    verify_teacher_contract(env, saved)
    with pytest.raises(ValueError, match="contract"):
        verify_teacher_contract(env, {**saved, "observation_dim": 999})
    env.library.clips[0]["direction"] = torch.tensor([0.0, 1.0])
    with pytest.raises(ValueError, match="forward-only"):
        verify_teacher_contract(env, saved)


def test_standoff_accepts_measured_ring_without_walking_to_center():
    cfg = WaypointConfig(
        max_seconds=4, arrival_min_distance_m=0.5, goal_tolerance_m=0.65
    )
    report, rows = run_waypoint(FakeEnv(), Teacher(), (1.0, 0.0), cfg)
    assert report["arrival"]
    assert 0.5 <= report["final_distance_m"] <= 0.65
    assert report["stable_arrival_seconds"] >= 1
    assert report["arrival_distance_band_m"] == [0.5, 0.65]
    assert "opponent-center" in report["goal_semantics"]
    first_hold = np.flatnonzero(rows["mode"] == "hold")[0]
    assert 0.5 <= rows["distance_m"][first_hold] <= 0.65


def test_standoff_start_too_close_fails_without_forward_or_reset_repair():
    env = FakeEnv()
    report, _ = run_waypoint(
        env,
        Teacher(),
        (0.3, 0),
        WaypointConfig(arrival_min_distance_m=0.5, goal_tolerance_m=0.65),
    )
    assert not report["arrival"]
    assert report["end_reason"] == "too_close_no_reverse_skill"
    assert report["physical_steps"] == 0
    assert env.resets == [True]


def test_standoff_hold_drift_across_inner_boundary_is_failure():
    class DriftingEnv(FakeEnv):
        def step(self, *args):
            result = super().step(*args)
            if len(args) == 2:
                self._state = state(x=0.7)
                return self._state
            return result

    report, _ = run_waypoint(
        DriftingEnv(),
        Teacher(),
        (1.0, 0),
        WaypointConfig(arrival_min_distance_m=0.5, goal_tolerance_m=0.65),
    )
    assert not report["arrival"]
    assert report["end_reason"] == "too_close_no_reverse_skill"
    assert report["minimum_distance_m"] < 0.5


def test_standoff_stationary_but_facing_away_does_not_pass():
    report, _ = run_waypoint(
        FakeEnv(),
        Teacher(),
        (-0.6, 0),
        WaypointConfig(
            max_seconds=4, arrival_min_distance_m=0.5, goal_tolerance_m=0.65
        ),
    )
    assert 0.5 <= report["final_distance_m"] <= 0.65
    assert report["final_speed_mps"] == 0
    assert not report["arrival"]
    assert report["end_reason"] == "hold_failed_to_stop_at_goal"


@pytest.mark.parametrize(
    "config",
    [
        {"max_seconds": float("nan")},
        {"turn_rate_radps": 0},
        {"max_heading_step_rad": 1},
        {"stable_seconds": 4},
        {"arrival_min_distance_m": -0.1},
        {"arrival_min_distance_m": 0.30},
    ],
)
def test_invalid_controller_settings_rejected(config):
    with pytest.raises(ValueError):
        WaypointConfig(**config)
