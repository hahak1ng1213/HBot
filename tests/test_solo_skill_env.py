"""Real native physics with synthetic references and a mock frozen lower."""

import math

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_solo_skill_physics import (
    assets as assets,
    physics as physics,
)

from holoboxing.training.skills.env import (
    SkillTrackingEnv,
    blend_angular_velocity,
    quat_matrix,
    quat_mul,
    root_relative_wrists,
    tracking_metrics,
)
from holoboxing.training.skills.reference import reference_kinematics


class SyntheticLibrary:
    def __init__(self, physics, skill="locomotion"):
        self.device = physics.device
        self.manifest = {"skill": skill}
        self.clips = []
        frames = 60
        for yaw in (0.3, -0.4):
            position = np.zeros((frames, 1, 3))
            position[:, 0, 2] = 0.9
            position[:, 0, 0] = np.arange(frames) * 0.002
            quaternion = Rotation.from_euler("z", yaw).as_quat()[[3, 0, 1, 2]]
            trajectory = {
                "roles": 1,
                "root_position": position,
                "root_quaternion": np.tile(quaternion, (frames, 1, 1)),
                "root_origin_velocity": np.tile([0.1, 0, 0], (frames, 1, 1)),
                "root_angular_velocity_world": np.zeros((frames, 1, 3)),
                "joint_position": np.full((frames, 1, 29), 0.2),
                "joint_velocity": np.zeros((frames, 1, 29)),
            }
            data = reference_kinematics(
                physics.host,
                trajectory,
                physics.joint_names,
                physics.body_names,
                (0, frames),
            )
            self.clips.append(
                {
                    key: torch.tensor(value, dtype=torch.float32)
                    for key, value in data.items()
                }
            )
        self.lengths = torch.tensor([frames, frames])

    def gather(self, clip_ids, frame_ids):
        query = frame_ids.clamp(0, int(self.lengths[0]) - 1)
        result = {
            key: torch.stack(
                [
                    self.clips[int(c)][key][int(f)]
                    for c, f in zip(clip_ids, query, strict=True)
                ]
            )[:, None]
            for key in self.clips[0]
        }
        result["valid"] = (frame_ids < self.lengths[clip_ids])[:, None]
        return result


@pytest.fixture
def env(physics):
    return SkillTrackingEnv(
        physics, SyntheticLibrary(physics), entry_steps=10, settle_steps=5
    )


def test_no_entry_frame_zero_never_uses_stale_entry_state(env):
    env.start([0, 1], reset_physics=True, neutral=False)
    env.entry_state["joint_position"][:] = -2
    ref = env.reference()
    assert torch.equal(env.entry_length, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(
        ref["joint_position"][:, 0], torch.full((2, 29), 0.2)
    )
    torch.testing.assert_close(
        ref["joint_velocity"][:, 0], torch.zeros((2, 29))
    )
    torch.testing.assert_close(
        ref["root_linear_velocity"][:, 0], torch.tensor([[0.1, 0, 0]] * 2)
    )


def test_world_reference_anchor_does_not_follow_robot_drift(env):
    env.start([0, 1], reset_physics=True, neutral=False)
    original = env.reference((0, 5, 10, 25))
    env.physics.native[0].qpos[env.physics.root_qi] += 2
    env._state = env.physics.state()
    changed = env.reference((0, 5, 10, 25))
    for key in original:
        torch.testing.assert_close(changed[key], original[key])
    assert env.get_observations().isfinite().all()
    assert (
        tracking_metrics(
            env._state, {key: value[:, 0] for key, value in changed.items()}
        )["root_error_m"][0]
        > 1.9
    )


def test_yaw_rotates_positions_quaternions_and_world_velocity_together(env):
    env.start([0, 1], reset_physics=True, neutral=False, yaw=math.pi / 2)
    env.frame[:] = 10
    ref = env.reference()
    source = env.library.gather(env.clip_ids, env.frame)
    expected_p = source["point_position"].clone()
    expected_p[..., 0] = -source["point_position"][..., 1]
    expected_p[..., 1] = source["point_position"][..., 0]
    torch.testing.assert_close(
        ref["point_position"], expected_p, atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        ref["root_linear_velocity"][:, 0],
        torch.tensor([[0, 0.1, 0]] * 2),
        atol=1e-6,
        rtol=1e-6,
    )
    expected_yaw = torch.tensor([0.3, -0.4]) + math.pi / 2
    measured_rotation = quat_matrix(ref["root_quaternion"][:, 0])
    torch.testing.assert_close(
        measured_rotation[:, 1, 0], expected_yaw.sin(), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    "yaw",
    [
        0,
        [0],
        0.0,
        [0.0],
        torch.tensor(0),
        torch.tensor([0.0], dtype=torch.float64),
    ],
)
def test_zero_yaw_numeric_forms_match_float_state_during_partial_reset(
    env, yaw
):
    env.yaw[0] = 0.2
    env.start([1], ids=[1], reset_physics=True, neutral=False, yaw=yaw)
    torch.testing.assert_close(env.yaw, torch.tensor([0.2, 0.0]))
    expected = env.library.gather(torch.tensor([1]), torch.tensor([0]))
    torch.testing.assert_close(
        env.physics.state()["root_quaternion"][1],
        expected["root_quaternion"][0, 0],
    )
    assert env.get_observations().isfinite().all()


def test_live_skill_switch_keeps_physics_and_starts_from_actual_pose(env):
    env.physics.native[0].qpos[env.physics.root_qi] += 1.3
    env.physics.native[0].qpos[env.physics.qi] -= 0.2
    before = [d.qpos.copy() for d in env.physics.native]
    current = env.physics.state()
    env.start([1, 0], reset_physics=False)
    assert torch.equal(env.frame, torch.full((2,), -10))
    for saved, data in zip(before, env.physics.native, strict=True):
        np.testing.assert_array_equal(saved, data.qpos)
    ref = env.reference()
    for key in ("point_position", "point_quaternion", "joint_position"):
        torch.testing.assert_close(ref[key][:, 0], current[key])
    env.frame[:] = -5
    middle = env.reference()
    target = env.library.gather(env.clip_ids, torch.zeros(2, dtype=torch.long))
    expected = (
        target["joint_position"][:, 0] - current["joint_position"]
    ) * 7.5
    torch.testing.assert_close(middle["joint_velocity"][:, 0], expected)


def test_partial_start_preserves_other_world_reference_and_progress(env):
    env.start([0, 1], reset_physics=True, neutral=False, yaw=[0.2, -0.5])
    env.frame[:] = torch.tensor([5, 9])
    env.elapsed[:] = torch.tensor([5, 9])
    env.last_action[1] = 0.3
    before = {
        key: value[1].clone() for key, value in env.reference((0, 5)).items()
    }
    physical = env.physics.native[1].qpos.copy()
    env.start([1], ids=[0], reset_physics=True, neutral=True)
    assert env.frame[1] == 9 and env.elapsed[1] == 9
    assert torch.all(env.last_action[1] == 0.3)
    np.testing.assert_array_equal(env.physics.native[1].qpos, physical)
    for key, value in env.reference((0, 5)).items():
        torch.testing.assert_close(value[1], before[key])


def test_raw_action_preserved_and_residuals_share_one_so3_transform(env):
    env.start([0, 1], reset_physics=True, neutral=False)
    raw = torch.linspace(-5, 5, 60).reshape(2, 30)
    saved = raw.clone()
    reference = env.reference((1, 2, 3, 4, 5))
    _, reward, _, info = env.step(raw)
    torch.testing.assert_close(raw, saved)
    assert reward.isfinite().all()
    assert info["motion"].all() and not info["entry"].any()
    sent = env.physics.lower.last_inputs
    delta = sent["reference_position_w"] - reference["point_position"]
    for frame in range(1, 5):
        torch.testing.assert_close(delta[:, frame], delta[:, 0])
    ref_conjugate = reference["point_quaternion"] * torch.tensor(
        [1, -1, -1, -1]
    )
    correction = quat_mul(ref_conjugate, sent["reference_quaternion_wxyz"])
    for frame in range(1, 5):
        torch.testing.assert_close(correction[:, frame], correction[:, 0])
    rotation = quat_matrix(sent["reference_quaternion_wxyz"])
    torch.testing.assert_close(
        rotation.transpose(-1, -2) @ rotation,
        torch.eye(3).expand(2, 5, 5, 3, 3),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(env.last_action.abs() < 1)


def test_full_body_metrics_notice_error_outside_five_target_points(env):
    actual = env.physics.state()
    reference = {key: value.clone() for key, value in actual.items()}
    elbow = env.physics.body_names.index("left_elbow_link")
    actual["body_position"][:, elbow, 0] += 0.2
    actual["joint_position"][:, 0] += 0.3
    metrics = tracking_metrics(actual, reference)
    assert torch.all(metrics["body_position_m"] > 0)
    assert torch.all(metrics["joint_error_rad"] > 0)
    assert torch.all(metrics["point_error_m"] == 0)


def test_clip_end_zeroes_velocity_and_finishes_after_settle(
    env,
):
    env.start([0, 1], reset_physics=True, neutral=False)
    env.frame[:] = env.library.lengths[env.clip_ids] - 1
    ref = env.reference((0, 5))
    for key in ref:
        if "velocity" in key:
            assert torch.all(ref[key] == 0)
    env.frame[:] = env.library.lengths[env.clip_ids] - 2 + env.settle_steps
    _, _, done, info = env.step(torch.zeros((2, 30)))
    assert done.all() and info["completed"].all()
    assert info["settle"].all()
    assert not info["failed"].any()


def test_entry_angular_velocity_matches_quaternion_finite_difference():
    from holoboxing.training.skills.env import quat_blend

    a = torch.tensor([1.0, 0, 0, 0], dtype=torch.float64)
    b = torch.tensor([math.cos(0.8), 0, 0, math.sin(0.8)], dtype=torch.float64)
    weight, rate, eps = 0.4, 1.3, 1e-6
    center = quat_blend(a, b, weight)
    derivative = (
        quat_blend(a, b, weight + eps * rate)
        - quat_blend(a, b, weight - eps * rate)
    ) / (2 * eps)
    expected = (
        2 * quat_mul(derivative, center * torch.tensor([1, -1, -1, -1]))[1:]
    )
    torch.testing.assert_close(
        blend_angular_velocity(a, b, weight, rate), expected
    )


@pytest.mark.parametrize("skill", ["locomotion", "punch"])
def test_entry_reward_does_not_require_inconsistent_joint_body_interpolants(
    physics, monkeypatch, skill
):
    env = SkillTrackingEnv(
        physics,
        SyntheticLibrary(physics, skill),
        entry_steps=10,
        settle_steps=5,
    )
    current = {
        key: value[:, 0].clone() for key, value in env.reference((1,)).items()
    }
    current["foot_contact"] = torch.zeros((2, 2), dtype=torch.bool)
    current["fallen"] = torch.zeros(2, dtype=torch.bool)
    monkeypatch.setattr(env.physics, "step", lambda *_: current)
    _, expected, _, info = env.step(torch.zeros((2, 30)))
    assert info["entry"].all()
    current["joint_position"] += 1.0
    current["body_position"][:, 3, 0] += 0.8
    env.frame -= 1
    _, changed, _, altered_info = env.step(torch.zeros((2, 30)))
    torch.testing.assert_close(changed, expected)
    assert (altered_info["metrics"]["joint_error_rad"] > 0.9).all()


@pytest.mark.parametrize(
    "joint,degrees,metric",
    [
        ("left_shoulder_pitch_joint", 32, "shoulder_error_rad"),
        ("left_shoulder_yaw_joint", 27, "shoulder_error_rad"),
        ("left_elbow_joint", 20, "elbow_error_rad"),
    ],
)
def test_punch_reward_notices_local_arm_errors_hidden_by_full_body_average(
    physics, joint, degrees, metric
):
    punch = SkillTrackingEnv(physics, SyntheticLibrary(physics, "punch"))
    locomotion = SkillTrackingEnv(physics, SyntheticLibrary(physics))
    reference = physics.state()
    index = physics.joint_names.index(joint)
    for data in physics.native:
        data.qpos[physics.qi[index]] += math.radians(degrees)
    # Recompute actual FK after changing physical joint positions.
    actual = physics.state()
    baseline, _ = punch._tracking_reward(reference, reference)
    changed, metrics = punch._tracking_reward(actual, reference)
    old_baseline, _ = locomotion._tracking_reward(reference, reference)
    old_changed, _ = locomotion._tracking_reward(actual, reference)
    assert torch.all(metrics[metric] > 0)
    assert torch.all(baseline - changed > 2 * (old_baseline - old_changed))
    assert (
        punch.recipe["tracking_reward"]["version"]
        == "punch_anatomy_tracking_v3"
    )


def test_wrist_reward_tracks_pose_and_velocity_direction(physics):
    env = SkillTrackingEnv(physics, SyntheticLibrary(physics, "punch"))
    reference = physics.state()
    reference["point_velocity"][:, 1:3] = torch.tensor([1.0, 0, 0])
    expected, _ = env._tracking_reward(reference, reference)

    actual = {key: value.clone() for key, value in reference.items()}
    actual["point_position"][:, 1, 0] += 0.1
    moved, metrics = env._tracking_reward(actual, reference)
    assert torch.all(metrics["wrist_relative_error_m"] > 0.07)
    assert torch.all(expected - moved > 0.07)

    actual = {key: value.clone() for key, value in reference.items()}
    actual["point_quaternion"][:, 1] = quat_mul(
        actual["point_quaternion"][:, 1],
        torch.tensor([math.cos(0.2), 0, 0, math.sin(0.2)]),
    )
    rotated, metrics = env._tracking_reward(actual, reference)
    assert torch.all(metrics["wrist_relative_rotation_rad"] > 0.28)
    assert torch.all(expected - rotated > 0.03)

    actual = {key: value.clone() for key, value in reference.items()}
    actual["point_velocity"][:, 1:3] = torch.tensor([0, 1.0, 0])
    wrong_direction, metrics = env._tracking_reward(actual, reference)
    # Equal speed is insufficient: the velocity vector must match the punch.
    torch.testing.assert_close(
        actual["point_velocity"][:, 1:3].norm(dim=-1),
        reference["point_velocity"][:, 1:3].norm(dim=-1),
    )
    assert torch.all(metrics["wrist_relative_velocity_mps"] > 1.4)
    assert torch.all(expected - wrong_direction > 0.12)


def test_wrist_velocity_removes_root_translation_and_rotation(physics):
    state = physics.state()
    state["root_linear_velocity"][:] = torch.tensor([0.5, -0.2, 0.1])
    state["root_angular_velocity"][:] = torch.tensor([0.0, 0, 2.0])
    offsets = state["point_position"][:, 1:3] - state["root_position"][:, None]
    state["point_velocity"][:, 1:3] = state["root_linear_velocity"][
        :, None
    ] + torch.cross(
        state["root_angular_velocity"][:, None].expand_as(offsets),
        offsets,
        dim=-1,
    )
    _, _, relative = root_relative_wrists(state)
    torch.testing.assert_close(
        relative, torch.zeros_like(relative), atol=1e-7, rtol=0
    )


def test_locomotion_tracking_reward_remains_exactly_the_original_formula(env):
    reference = env.physics.state()
    actual = {key: value.clone() for key, value in reference.items()}
    actual["joint_position"][:, 0] += 0.4
    actual["root_position"][:, 0] += 0.15
    actual["body_position"][:, 3, 0] += 0.2
    actual["body_linear_velocity"][:, 4, 1] += 0.8
    actual["body_quaternion"][:, 5] = torch.tensor(
        [math.cos(0.2), 0, 0, math.sin(0.2)]
    )
    reward, metrics = env._tracking_reward(actual, reference)
    expected = (
        0.5 * torch.exp(-(metrics["root_error_m"] / 0.3).square())
        + torch.exp(-(metrics["body_position_m"] / 0.12).square())
        + torch.exp(-(metrics["body_rotation_rad"] / 0.4).square())
        + 0.5 * torch.exp(-(metrics["joint_error_rad"] / 0.35).square())
        + torch.exp(-(metrics["body_velocity_mps"] / 1.0).square())
    ) / 4
    assert torch.equal(reward, expected)
    assert "shoulder_error_rad" not in metrics


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ids": [0, 0], "clip_ids": [0, 1]},
        {"ids": [0], "clip_ids": [-1]},
        {"ids": [0], "clip_ids": [0], "yaw": float("nan")},
        {"ids": [0], "clip_ids": [0], "random_phase": True},
    ],
)
def test_invalid_start_rejected_without_changing_reference(env, kwargs):
    before = env.clip_ids.clone(), env.frame.clone(), env.yaw.clone()
    with pytest.raises(ValueError):
        env.start(**kwargs)
    for saved, current in zip(
        before, (env.clip_ids, env.frame, env.yaw), strict=True
    ):
        torch.testing.assert_close(saved, current)
