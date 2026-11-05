"""Range-gated task contracts; synthetic physics/lower are not skill evidence."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_boxing_cpu_physics import fake_lower_setup
from test_boxing_cpu_physics import physics as native_physics  # noqa: F401

from holoboxing.training.skills import fixed_punch as task
from holoboxing.training.skills.env import quat_matrix
from holoboxing.training.skills.reference import retime_trajectory
from holoboxing.training.skills.train import train_teacher
from holoboxing.training.skills.train_fixed_punch import (
    evaluate,
    source_identity,
)
from holoboxing.training.tasks.combat.mdp.geometry import decode_output


def library(*, movement=False):
    count = 13
    p = torch.tensor(
        [
            [0, 0, 1.0],
            [0.3, 0.2, 1.2],
            [0.3, -0.2, 1.2],
            [0, 0.1, 0],
            [0, -0.1, 0],
        ]
    )
    p = p.repeat(count, 1, 1)
    p[:, 1:3, 0] += torch.linspace(0, 0.25, count)[:, None]
    if movement:
        p[..., 0] += torch.linspace(0, 0.3, count)[:, None]
    q = torch.zeros(count, 5, 4)
    q[..., 0] = 1
    clip = dict(
        frames=count,
        group="kangning-010" if movement else "kangning-037",
        point_position=p,
        point_quaternion=q,
        root_position=p[:, 0],
        guard_heading_world=0.0,
        joint_position=torch.zeros(count, 29),
    )
    return SimpleNamespace(
        clips=[clip], manifest={"speed_scale": 1.5, "fixture": True}
    )


@pytest.fixture
def env(monkeypatch, tmp_path, request):
    request.getfixturevalue("native_physics")
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.standing_reference import WorldStandingReference

    assets = fake_lower_setup(monkeypatch)
    combat = MujocoSandbagEnv(
        assets,
        {"lower_onnx": "unused", "lower_config": "unused"},
        2,
        "cpu",
        tmp_path / "runtime",
        official_rules=True,
        capture_transition_state=True,
        standing_reference=WorldStandingReference(torch.zeros(225), 2, "cpu"),
    )
    # The base CPU fixture enables only the left glove. Reach supervision
    # intentionally requires both real collision pairs instead of a fallback.
    for model in (combat.physics.host, *combat.physics.models):
        wrist = model.body("p1_right_wrist_yaw_link").id
        geoms = model.geom_bodyid == wrist
        model.geom_contype[geoms] = 1
        model.geom_conaffinity[geoms] = 1
    result = task.FixedPunchCombatEnv(
        combat, library(), library(movement=True)
    )
    result.implementation_sources = source_identity()
    yield result
    combat.close()


def place(env, distance, angle=0.0):
    qpos = env.physics.data.qpos
    a, b = env.physics.root_qi
    qpos[:, a : a + 3] = qpos.new_tensor([0, 0, 1])
    qpos[:, b : b + 3] = qpos.new_tensor(
        [distance * math.cos(angle), distance * math.sin(angle), 1]
    )
    qpos[:, a + 3 : a + 7] = qpos.new_tensor([1, 0, 0, 0])
    mask = torch.ones(env.num_envs, dtype=torch.bool)
    env.obs = env.combat._read(mask, previous=env.obs)
    env._prepare(mask)
    env.age[:] = env.config.transition_steps


def test_distance_latch_strict_entry_and_strict_exit():
    cfg = task.FixedPunchConfig()
    d = torch.tensor(
        [0.6499, 0.65, 0.6501, 1.4999, 1.5, 1.5001, float("nan")],
        dtype=torch.float64,
    )
    a = torch.zeros_like(d)
    assert task.attack_gate(
        torch.zeros(7, dtype=torch.bool), d, a, cfg
    ).tolist() == [True, False, False, False, False, False, False]
    assert task.attack_gate(
        torch.ones(7, dtype=torch.bool), d, a, cfg
    ).tolist() == [True, True, True, True, True, False, False]


def test_heading_never_clears_latch_but_punch_requires_front():
    angles = torch.deg2rad(
        torch.tensor(
            [0.0, 45.0, 89.9, 90.0, 90.1, 180.0, -89.9, -90.0, float("nan")]
        )
    )
    for previous in (True, False):
        result = task.attack_gate(
            torch.full((9,), previous),
            torch.full((9,), 0.599),
            angles,
            task.FixedPunchConfig(),
        )
        assert result.tolist() == [True] * 8 + [False]
    assert task.in_front(angles).tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        False,
        False,
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attack_exit_m": 0.61},
        {"transition_steps": 0},
        {"guard_offset_deg": float("nan")},
        {"movement": "mixed"},
    ],
)
def test_invalid_task_config(kwargs):
    with pytest.raises(ValueError):
        task.FixedPunchConfig(**kwargs)


def test_retime_resamples_not_relabels_fps_and_scales_velocity():
    t = np.arange(13, dtype=float)
    q = np.zeros((13, 1, 4))
    q[:, 0, 0], q[:, 0, 3] = np.cos(t * 0.05), np.sin(t * 0.05)
    trajectory = {
        "roles": 1,
        "root_quaternion": q,
        "root_position": np.repeat(t[:, None, None], 3, -1),
        "joint_position": np.repeat(t[:, None, None], 29, -1),
        "root_origin_velocity": np.ones((13, 1, 3)),
        "root_angular_velocity_world": np.ones((13, 1, 3)),
        "joint_velocity": np.ones((13, 1, 29)),
    }
    result = retime_trajectory(trajectory, 1.5)
    assert len(result["joint_position"]) == 9
    np.testing.assert_allclose(
        result["joint_position"][:, 0, 0], np.arange(9) * 1.5
    )
    for key in (
        "root_origin_velocity",
        "root_angular_velocity_world",
        "joint_velocity",
    ):
        np.testing.assert_allclose(result[key], 1.5)
        np.testing.assert_allclose(trajectory[key], 1)
    np.testing.assert_allclose(
        result["root_quaternion"][:, 0, 3], np.sin(np.arange(9) * 1.5 * 0.05)
    )
    assert retime_trajectory(trajectory, 1) is trajectory


def test_cycle_bridge_is_continuous_and_forward_gait_does_not_teleport_back():
    lib = library(movement=True)
    period = lib.clips[0]["frames"] - 1 + 20
    ref = task.cyclic_points(
        lib, torch.tensor([period - 1]), (0, 1, 2), 20, locomotion=True
    )
    root = ref["point_position"][0, :, 0, 0]
    assert bool((root[1:] >= root[:-1]).all())
    assert float(root[1]) == pytest.approx(0.3)
    assert (
        float(
            (ref["point_position"][:, 1] - ref["point_position"][:, 0])
            .abs()
            .max()
        )
        < 0.01
    )
    torch.testing.assert_close(
        ref["point_quaternion"].norm(dim=-1), torch.ones(1, 3, 5)
    )


def test_horizontal_distance_and_bad_facing_stop_punch_phase(env):
    place(env, 0.55)
    assert env.attack.all()
    env.phase[:] = 7
    place(env, 0.65001)
    assert not env.attack.any()
    assert not env.phase.any()
    place(env, 0.55, math.pi / 2)
    assert env.attack.all() and not env.punching.any()
    place(env, 0.55)
    state = env.state()
    state["opponent"][:, 2] += 10
    torch.testing.assert_close(
        task.geometry(state, env.config)[0], torch.full((2,), 0.55)
    )


def test_distance_band_preserves_mode_phase_and_anchor_until_exit(env):
    place(env, 0.8)
    a, b = env.physics.root_qi

    def move(distance):
        env.physics.data.qpos[:, b] = env.physics.data.qpos[:, a] + distance
        env._prepare()

    move(0.66)
    assert not env.attack.any()
    move(0.649)
    assert env.attack.all() and env.punching.all()
    env.phase[:] = 7
    env.age[:] = 20
    anchor = env.anchor.clone()
    for distance in (0.651, 0.7, 1.0, 1.5, 0.64, 1.49):
        move(distance)
        assert env.attack.all() and env.punching.all()
        assert (env.phase == 7).all() and (env.age == 20).all()
        torch.testing.assert_close(env.anchor, anchor)
    move(1.501)
    assert not env.attack.any() and not env.punching.any()
    assert not env.phase.any()
    move(0.7)
    assert not env.attack.any()
    move(0.64)
    assert env.attack.all()


def test_heading_pause_resume_preserves_latched_punch_phase(env):
    place(env, 0.55)
    env.phase[:] = 7
    anchor = env.anchor.clone()
    a, b = env.physics.root_qi
    env.physics.data.qpos[:, b : b + 2] = torch.tensor([-0.8, 0.0])
    env._prepare()
    assert env.attack.all() and not env.punching.any()
    assert (env.phase == 7).all()
    _, _, done, info = env.step(torch.zeros(2, 30))
    assert not done.any()
    assert env.attack.all() and not env.punching.any()
    assert (env.phase == 7).all()
    assert info["metrics"]["attack_mode"].all()
    assert info["metrics"]["alignment_only"].all()
    assert not info["metrics"]["punch_active"].any()
    assert not info["metrics"]["approach_progress"].any()
    env.physics.data.qpos[:, b : b + 2] = env.physics.data.qpos[
        :, a : a + 2
    ] + torch.tensor([0.8, 0.0])
    env._prepare()
    assert env.attack.all() and env.punching.all()
    assert (env.phase == 7).all()
    torch.testing.assert_close(env.anchor, anchor)
    env.step(torch.zeros(2, 30))
    assert (env.phase == 8).all()


def test_episode_reset_clears_only_reset_rows_distance_memory(env):
    place(env, 0.55)
    env.phase[:] = 7
    a, b = env.physics.root_qi
    env.physics.data.qpos[:, b] = env.physics.data.qpos[:, a] + 0.8
    env._prepare(torch.tensor([True, False]))
    assert env.attack.tolist() == [False, True]
    assert env.punching.tolist() == [False, True]
    assert env.phase.tolist() == [0, 7]


def test_unentered_distance_band_still_approaches_and_receives_progress(env):
    place(env, 0.8)
    before = env.state()
    ref = env.references((1, 2))
    assert not env.attack.any()
    assert (
        ref["point_position"][:, -1, 0, 0] > before["position"][:, 0, 0]
    ).all()
    after = {key: value.clone() for key, value in before.items()}
    after["opponent"][:, 0] -= 0.01
    flags = torch.zeros(2, dtype=torch.bool)
    action = torch.zeros(2, 30)
    _, metrics = task.task_reward(
        before,
        after,
        ref,
        flags,
        ~flags,
        flags.float(),
        flags,
        flags,
        action,
        action,
        env.config,
        engaged=flags,
    )
    assert (metrics["approach_progress"] > 0).all()
    assert not metrics["punch_wrist_position"].any()


def test_move_masks_wrist_actions_and_preserves_guard_under_pelvis_rotation(
    env, monkeypatch
):
    place(env, 1.0)
    before = env.state()
    reference = env.references((1, 2, 3, 4, 5))
    actions = torch.ones(2, 30) * 8
    sent = {}
    original_step = env.combat.step

    def capture(values):
        sent.update(values)
        return original_step(values)

    monkeypatch.setattr(env.combat, "step", capture)
    _, _, _, info = env.step(actions)
    p, r = decode_output(
        before["position"], before["quaternion"], sent["robot_A"]
    )
    actual_guard = torch.einsum(
        "ntij,ntbj->ntbi",
        r[:, :, 0].transpose(-1, -2),
        p[:, :, 1:3] - p[:, :, :1],
    )
    rq = quat_matrix(reference["point_quaternion"][:, :, 0])
    expected_guard = torch.einsum(
        "ntij,ntbj->ntbi",
        rq.transpose(-1, -2),
        reference["point_position"][:, :, 1:3]
        - reference["point_position"][:, :, :1],
    )
    torch.testing.assert_close(
        actual_guard, expected_guard, atol=1e-6, rtol=1e-5
    )
    assert not env.last_action.reshape(2, 5, 6)[:, 1:3].any()
    assert not info["metrics"]["attack_mode"].any()
    assert not info["metrics"]["punch_wrist_position"].any()


def test_explore_has_static_base_gait_and_can_turn(env):
    env.config = task.FixedPunchConfig(movement="explore")
    place(env, 1, math.pi / 2)
    before = env.state()
    ref = env.references((1, 2, 3, 4, 5))
    torch.testing.assert_close(
        ref["point_position"][:, :, 0],
        before["position"][:, None, 0].expand(-1, 5, -1),
    )
    assert bool((task.yaw(ref["point_quaternion"][:, -1, 0]) > 0).all())


def test_near_distance_attacks_when_facing_and_aligns_without_advance(env):
    place(env, 0.599)
    assert env.attack.all()
    place(env, 0.4, math.pi / 2)
    assert env.attack.all() and not env.punching.any()
    ref = env.references((1, 2, 3, 4, 5))
    before = env.state()
    torch.testing.assert_close(
        ref["point_position"][:, :, 0],
        before["position"][:, None, 0].expand(-1, 5, -1),
    )
    assert (task.yaw(ref["point_quaternion"][:, -1, 0]) > 0).all()
    phase = env.phase.clone()
    env.step(torch.full((2, 30), 8.0))
    torch.testing.assert_close(env.phase, phase)
    assert not env.last_action.reshape(2, 5, 6)[..., :3].any()


def test_move_wrists_are_raised_compact_guard(env):
    place(env, 1.0)
    ref = env.references((1, 2, 3, 4, 5))
    inverse = quat_matrix(ref["point_quaternion"][:, :, 0]).transpose(-1, -2)
    local = torch.einsum(
        "ntij,ntbj->ntbi",
        inverse,
        ref["point_position"][:, :, 1:3] - ref["point_position"][:, :, :1],
    )
    expected = torch.tensor([[0.18, 0.12, 0.34], [0.18, -0.12, 0.34]])
    torch.testing.assert_close(
        local, expected.expand(2, 5, -1, -1), atol=1e-6, rtol=1e-5
    )


def test_no_torso_contact_timeout_strict_boundary_and_contact_reset(
    env, monkeypatch
):
    place(env, 0.8)
    env.no_torso_contact_steps[:] = torch.tensor([249, 250])
    _, _, done, info = env.step(torch.zeros(2, 30))
    assert not done.any()
    torch.testing.assert_close(
        info["metrics"]["no_torso_contact"], torch.tensor([0.0, -0.01])
    )
    assert env.no_torso_contact_steps.tolist() == [250, 251]
    original_step = env.combat.step

    def touch_torso(values):
        result = original_step(values)
        env.combat.metrics["robot_A"]["official_torso_contact_samples"] = (
            torch.tensor([1.0, 0.0])
        )
        return result

    monkeypatch.setattr(env.combat, "step", touch_torso)
    _, _, _, info = env.step(torch.zeros(2, 30))
    assert env.no_torso_contact_steps.tolist() == [0, 252]
    assert info["metrics"]["torso_contact"].tolist() == [1, 0]
    torch.testing.assert_close(
        info["metrics"]["no_torso_contact"], torch.tensor([0.0, -0.01])
    )
    assert env.get_observations()[1, -1] > 1


def test_contact_clock_survives_mode_change_but_not_episode_reset(env):
    place(env, 0.8)
    env.no_torso_contact_steps[:] = 300
    a, b = env.physics.root_qi
    for distance in (0.55, 1.6):
        env.physics.data.qpos[:, b] = env.physics.data.qpos[:, a] + distance
        env._prepare()
        assert (env.no_torso_contact_steps == 300).all()
    env._prepare(torch.tensor([True, False]))
    assert env.no_torso_contact_steps.tolist() == [0, 300]


def test_stronger_facing_and_upper_tracking_weights(env):
    from dataclasses import replace

    place(env, 0.55)
    state = env.state()
    ref = env.references((1, 2))
    flags = torch.ones(2, dtype=torch.bool)
    action = torch.zeros(2, 30)

    def metrics(config):
        return task.task_reward(
            state,
            state,
            ref,
            flags,
            flags,
            flags.float() * 0,
            ~flags,
            ~flags,
            action,
            action,
            config,
        )[1]

    previous = metrics(
        replace(
            env.config,
            facing_alignment_weight=1.0,
            facing_progress_weight=0.5,
            punch_tracking_scale=1.0,
        )
    )
    current = metrics(env.config)
    torch.testing.assert_close(
        current["facing_alignment"], 3 * previous["facing_alignment"]
    )
    for key in (
        "punch_wrist_position",
        "punch_wrist_orientation",
        "punch_relative_velocity",
        "punch_arm_joints",
    ):
        torch.testing.assert_close(current[key], 2 * previous[key])


def test_reward_prefers_tracking_and_does_not_pay_out_of_range_hit(env):
    place(env, 0.55)
    ref = env.references((1, 2))
    state = env.state()
    state["position"] = ref["point_position"][:, 0].clone()
    state["quaternion"] = ref["point_quaternion"][:, 0].clone()
    state["joint_position"] = ref["joint_position"][:, 0].clone()
    flags = torch.ones(2, dtype=torch.bool)
    action = torch.zeros(2, 30)

    def score(after, *, invalid=~flags):
        return task.task_reward(
            state,
            after,
            ref,
            flags,
            flags,
            flags.float(),
            ~flags,
            invalid,
            action,
            action,
            env.config,
        )

    good, gm = score(state)
    _, entering = task.task_reward(
        state,
        state,
        ref,
        flags,
        ~flags,
        flags.float(),
        ~flags,
        ~flags,
        action,
        action,
        env.config,
    )
    assert not entering["punch_arm_joints"].any()
    assert not entering["punch_leg_joints"].any()
    assert (entering["punch_wrist_position"] > 0).all()
    bad = {k: v.clone() for k, v in state.items()}
    bad["position"][:, 1:3, 0] += 0.4
    bad["joint_position"] += 0.5
    bad_reward, bm = score(bad)
    assert bool((good > bad_reward).all())
    assert bool((gm["punch_arm_joints"] > bm["punch_arm_joints"]).all())
    bad["opponent"][:, 0] = 1.6
    assert not score(bad)[1]["official_hit"].any()
    bad["position"][0] = float("nan")
    reward, metrics = score(bad, invalid=torch.tensor([True, False]))
    assert float(reward[0]) == -10
    assert all(torch.isfinite(v).all() for v in metrics.values())
    assert metrics["invalid"].tolist() == [1, 0]


def test_training_resets_cover_near_far_and_heading_without_joint_change():
    qpos = torch.zeros(256, 72)
    qpos[:, 3] = qpos[:, 39] = 1
    qpos[:, 2] = qpos[:, 38] = 1
    original = qpos.clone()
    task.training_start_poses(
        qpos,
        (0, 36),
        torch.arange(256),
        torch.Generator().manual_seed(37),
        task.FixedPunchConfig(),
    )
    delta = qpos[:, 36:38] - qpos[:, :2]
    d = delta.norm(dim=-1)
    assert ((d >= 0.52) & (d <= 2)).all()
    assert (d < 0.6).any() and (d > 0.6).any()
    assert delta[:, 0].min() < 0 and delta[:, 0].max() > 0
    torch.testing.assert_close(qpos[:, 7:36], original[:, 7:36])
    torch.testing.assert_close(qpos[:, 43:], original[:, 43:])


def test_guard_orientation_selection_penalizes_flared_elbows():
    from holoboxing.training.tasks.combat.config.sandbag import BODY_NAMES

    clip = library().clips[0]
    clip["point_position"][:, 1:3] = torch.tensor(
        [[0.18, 0.12, 1.34], [0.18, -0.12, 1.34]]
    )
    clip["body_position"] = torch.zeros(clip["frames"], len(BODY_NAMES), 3)
    for side, sign in (("left", 1), ("right", -1)):
        index = BODY_NAMES.index(f"{side}_elbow_link")
        clip["body_position"][:, index, 1] = sign * 0.12
        clip["body_position"][0, index, 1] = sign * 0.45
    assert task.guard_pose(clip)[2] == 1


def test_alignment_rewards_both_modes_and_hit_accepts_front(env):
    place(env, 0.55)
    before = env.state()
    after = {key: value.clone() for key, value in before.items()}
    before["opponent"][:, :2] = torch.tensor(
        [0.55 * math.cos(math.pi / 3), 0.55 * math.sin(math.pi / 3)]
    )
    after["opponent"][:, :2] = torch.tensor(
        [0.55 * math.cos(math.pi / 6), 0.55 * math.sin(math.pi / 6)]
    )
    action = torch.zeros(2, 30)
    attack = torch.tensor([True, False])
    flags = torch.zeros(2, dtype=torch.bool)
    ref = env.references((1, 2))

    def score(state):
        return task.task_reward(
            before,
            state,
            ref,
            attack,
            ~flags,
            torch.ones(2),
            flags,
            flags,
            action,
            action,
            env.config,
        )[1]

    metrics = score(after)
    assert (metrics["facing_progress"] > 0).all()
    assert (
        metrics["facing_alignment"] > score(before)["facing_alignment"]
    ).all()
    assert metrics["official_hit"].tolist() == [5, 0]
    after["opponent"][:, :2] = torch.tensor([-0.3, 0.4])
    assert not score(after)["official_hit"].any()


def test_punch_heading_tracks_opponent_with_bounded_turn(env):
    place(env, 0.55, math.pi / 3)
    assert env.attack.all()
    previous = env.heading.clone()
    anchor = env.anchor.clone()
    env.phase[:] = 5
    env._prepare()
    delta = env.heading - previous
    assert (delta > 0).all()
    assert (delta <= env.config.turn_rate_rad_s * env.dt + 1e-6).all()
    assert (env.phase == 5).all()
    torch.testing.assert_close(env.anchor, anchor)


def test_mode_changes_blend_from_measured_pose_without_physics_reset(
    env, monkeypatch
):
    def reject_reset(*args, **kwargs):
        raise AssertionError("Mode switch must never reset physical pose")

    place(env, 0.7)
    monkeypatch.setattr(env.combat, "_reset", reject_reset)
    for distance, expected in ((0.55, True), (1.6, False), (0.55, True)):
        a, b = env.physics.root_qi
        env.physics.data.qpos[:, b] = env.physics.data.qpos[:, a] + distance
        env._prepare()
        assert bool(env.attack.all()) == expected
        state = env.state()
        ref = env.references((0,))
        torch.testing.assert_close(
            ref["point_position"][:, 0], state["position"]
        )
        torch.testing.assert_close(
            ref["point_quaternion"][:, 0], state["quaternion"]
        )


def test_move_reward_rewards_progress_and_turning(env):
    place(env, 1, math.pi / 4)
    before = env.state()
    after = {key: value.clone() for key, value in before.items()}
    after["position"][:, :, :2] += 0.01
    after["quaternion"][:, 0] = task.yaw_quaternion(torch.full((2,), 0.01))
    flags = torch.zeros(2, dtype=torch.bool)
    action = torch.zeros(2, 30)
    _, metrics = task.task_reward(
        before,
        after,
        env.references((1, 2)),
        flags,
        flags,
        flags.float(),
        flags,
        flags,
        action,
        action,
        env.config,
    )
    assert (metrics["approach_progress"] > 0).all()
    assert (metrics["facing_progress"] > 0).all()
    assert not metrics["punch_arm_joints"].any()


def test_cli_cannot_silently_load_old_checkpoint_or_overwrite_run(tmp_path):
    from holoboxing.training.skills.train_fixed_punch import parse_arguments

    common = [
        "--assets",
        str(tmp_path),
        "--lower",
        str(tmp_path),
        "--source",
        str(tmp_path),
        "--output",
        str(tmp_path / "run"),
    ]
    assert (
        parse_arguments(common + ["--movement", "explore"]).movement
        == "explore"
    )
    with pytest.raises(SystemExit):
        parse_arguments(common + ["--mode", "evaluate"])
    with pytest.raises(SystemExit):
        parse_arguments(common + ["--guard-offset-deg", "nan"])
    (tmp_path / "run.runtime").mkdir()
    with pytest.raises(SystemExit):
        parse_arguments(common)


def test_transition_reward_captures_state_before_automatic_reset(
    env, monkeypatch
):
    place(env, 1.0)
    original_read = env.combat._read
    calls = 0

    def invalid_first_read(reset, previous=None):
        nonlocal calls
        result = original_read(reset, previous)
        calls += 1
        if calls == 1:
            result["robot_A"]["task_policy"][0, 0] = float("nan")
        return result

    monkeypatch.setattr(env.combat, "_read", invalid_first_read)
    _, reward, done, info = env.step(torch.zeros(2, 30))
    assert done.tolist() == [True, False]
    assert reward[0] == -10
    assert info["failed"][0] and not info["completed"][0]
    snapshot = env.combat.last_transition_state
    assert float(snapshot["position"][0, 1, 0, 0]) == pytest.approx(
        1, abs=0.01
    )
    assert float(env.physics.state()["position"][0, 1, 0, 0]) > 1.9
    assert all(torch.isfinite(v).all() for v in info["metrics"].values())


def test_native_synthetic_ppo_save_and_evaluation_contract(env, tmp_path):
    torch.set_num_threads(1)
    args = SimpleNamespace(
        num_envs=2,
        updates=1,
        rollout_steps=4,
        save_every=1,
        learning_rate=1e-4,
        seed=37,
        skill="kangning037_combat",
    )
    result = train_teacher(env, tmp_path / "train", args)
    saved = torch.load(result["checkpoint"], weights_only=False)
    assert saved["format"] == "kangning037_range_teacher_v6"
    assert saved["observation_dim"] == 1356
    assert env.physics.steps == 16
    assert len(saved["optimizer"]["state"]) > 0
    assert saved["model"]["actor.4.weight"].abs().sum() > 0
    assert all(torch.isfinite(x).all() for x in saved["model"].values())
    args.mode, args.checkpoint = "evaluate", result["checkpoint"]
    args.output, args.eval_steps = tmp_path / "eval", 2
    report = evaluate(env, args)
    assert not report["submission_qualified"]
    assert not report["scored_attack_gate_passed"]
    assert len((args.output / "metrics.jsonl").read_text().splitlines()) == 2
    assert (
        json.loads((args.output / "evaluation.json").read_text())["steps"] == 2
    )


def test_reach_rewards_active_close_extension_not_holding_or_recovery(env):
    place(env, 0.55)
    state = env.state()
    ref = {
        "point_position": state["position"][:, None].repeat(1, 2, 1, 1),
        "point_quaternion": state["quaternion"][:, None].repeat(1, 2, 1, 1),
        "joint_position": state["joint_position"][:, None].repeat(1, 2, 1),
        "bridge": torch.zeros(2, 2, dtype=torch.bool),
    }
    ref["point_position"][:, 1, 1:3, 0] += 0.02
    reach = {
        "gap_m": torch.tensor([[0.03, 0.03], [0.3, 0.3]]),
        "arm_aim_speed_mps": torch.ones(2, 2),
        "arm_outward_speed_mps": torch.ones(2, 2),
        "closing_speed_mps": torch.ones(2, 2),
        "target_valid": torch.ones(2, 2, dtype=torch.bool),
    }
    yes = torch.ones(2, dtype=torch.bool)
    action = torch.zeros(2, 30)

    def score():
        return task.task_reward(
            state,
            state,
            ref,
            yes,
            yes,
            action[:, 0],
            ~yes,
            ~yes,
            action,
            action,
            env.config,
            reach=reach,
        )[1]

    metrics = score()
    assert metrics["punch_reach"][0] > metrics["punch_reach"][1] > 0
    assert metrics["punch_reach"].max() <= 0.02 * env.config.reach_weight
    state["position"][:, 1:3, 0] += 0.10
    assert score()["reach_tracking_wrist_error_m"].max() < 1e-6
    state["position"][:, 1:3, 1] += 0.10
    assert score()["reach_tracking_wrist_error_m"].min() > 0.05
    state["position"][:, 1:3, 1] -= 0.10
    ref["point_position"][:, 1, 1:3, 0] -= 0.02
    apex_error = score()["reach_tracking_wrist_error_m"]
    ref["point_position"][:, 1, 1:3, 0] -= 0.02
    assert (score()["reach_tracking_wrist_error_m"] > apex_error).all()
    ref["point_position"][:, 1, 1:3, 0] += 0.04
    state["position"][:, 1:3, 0] -= 0.10
    reach["arm_aim_speed_mps"].zero_()
    assert not score()["punch_reach"].any()  # body movement alone cannot score
    reach["arm_aim_speed_mps"].fill_(1)
    ref["point_position"][:, 1, 1:3, 0] -= 0.04
    assert not score()["punch_reach"].any()  # reference recovery cannot score
    ref["point_position"][:, 1, 1:3, 0] += 0.04
    ref["bridge"].fill_(True)
    assert not score()["punch_reach"].any()
    ref["bridge"].zero_()
    state["opponent"][:, 0] = 0.9
    assert not score()["punch_reach"].any()  # no stretching across large gaps
    reach["target_valid"][0, 0] = False
    with pytest.raises(FloatingPointError):
        score()


def test_explicit_v5_weights_initialization_verifies_sources_and_contract(
    env, tmp_path
):
    import hashlib
    from holoboxing.training.skills.actor import SkillActor
    from holoboxing.training.skills.train_fixed_punch import initialize_teacher

    original = SkillActor(env.observation_dim)
    with torch.no_grad():
        original.actor[-1].bias.fill_(0.12)
    source = tmp_path / "original" / "holoboxing" / "training"
    source.mkdir(parents=True)
    file = source / "test_source.py"
    file.write_text("verified source", encoding="utf-8")
    recipe = dict(env.recipe, version="kangning037_range_teacher_v5")
    for key in (
        "reach_reward",
        "reach_weight",
        "reach_gap_scale_m",
        "reach_forward_allowance_m",
    ):
        recipe.pop(key)
    saved = {
        "format": "kangning037_range_teacher_v5",
        "observation_dim": env.observation_dim,
        "action_dim": 30,
        "hidden_dims": (256, 128),
        "model": original.state_dict(),
        "reference_manifest": env.library.manifest,
        "environment_recipe": recipe,
        "implementation_sources": {
            "training/test_source.py": hashlib.sha256(
                file.read_bytes()
            ).hexdigest()
        },
        "update": 512,
    }
    checkpoint = tmp_path / "v5.pt"
    torch.save(saved, checkpoint)
    initialize_teacher(env, checkpoint, tmp_path / "original")
    for key, value in original.state_dict().items():
        torch.testing.assert_close(env.initial_model_state[key], value)
    assert env.initialization["source_update"] == 512
    assert env.initialization["source_closure_verified"]
    file.write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="hash differs"):
        initialize_teacher(env, checkpoint, tmp_path / "original")
