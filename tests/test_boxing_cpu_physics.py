"""Native CPU physics regression tests; synthetic MJCF, not competition results."""

import copy
import sys

import mujoco
import numpy as np
import pytest
import torch

from holoboxing.training.envs import mujoco_physics as physics_module
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


def fixture_model():
    parts = [
        '<mujoco><option timestep="0.005"/><worldbody>',
        '<geom name="smpl_ring_baseLink" type="box" size="5 5 .05" pos="0 0 -.05"/>',
    ]
    motors = []
    for role, prefix in enumerate(("p1_", "p2_")):
        parts.append(
            f'<body name="{prefix}pelvis" pos="{role * 2} 0 1"><freejoint/>'
        )
        parts.append(
            '<geom type="sphere" size=".05" mass="3" contype="0" conaffinity="0"/>'
        )
        for j, (name, joint) in enumerate(
            zip(BODY_NAMES[1:], JOINT_NAMES, strict=True)
        ):
            contact = (role == 0 and name == "left_wrist_yaw_link") or (
                role == 1 and name == "torso_link"
            )
            pos = (
                ("1 0 0" if role == 0 else "-.85 0 0")
                if contact
                else f"0 {j * 0.02} 0"
            )
            parts.append(
                f'<body name="{prefix}{name}" pos="{pos}"><joint name="{prefix}{joint}" axis="0 1 0" limited="true" range="-90 90"/>'
            )
            parts.append(
                f'<geom type="sphere" size="{0.1 if contact else 0.01}" mass=".1" contype="{int(contact)}" conaffinity="{int(contact)}"/></body>'
            )
            motors.append(
                f'<motor joint="{prefix}{joint}" ctrllimited="true" ctrlrange="-10 10"/>'
            )
        parts.append("</body>")
    parts.append(
        "</worldbody><actuator>" + "".join(motors) + "</actuator></mujoco>"
    )
    return mujoco.MjModel.from_xml_string("".join(parts))


@pytest.fixture
def physics(monkeypatch):
    model = fixture_model()
    monkeypatch.setattr(
        physics_module, "load_scene_model", lambda _: copy.copy(model)
    )
    monkeypatch.setattr(
        physics_module, "initial_qpos", lambda _, m: m.qpos0.copy()
    )
    p = physics_module.CompetitionPhysics(None, 2, "cpu")
    p.set_gains(torch.ones(29) * 20, torch.ones(29) * 2)
    return p


def test_step_matches_independent_native_mujoco_and_torque_limit(physics):
    p = physics
    p.targets += 100
    reference = mujoco.MjData(p.models[0])
    reference.qpos[:] = p.initial.qpos
    reference.ctrl[p.indices[2]] = 10
    mujoco.mj_step(p.models[0], reference)
    p.substep()
    np.testing.assert_allclose(p.native[0].qpos, reference.qpos, atol=1e-12)
    np.testing.assert_allclose(p.native[0].qvel, reference.qvel, atol=1e-12)
    assert torch.all(p.data.ctrl == 10)
    assert p.steps == 1
    assert p.state()["position"].dtype == torch.float32


def test_reset_is_local_and_clears_history(physics):
    p = physics
    p.substep()
    before = p.native[1].qpos.copy()
    time = p.native[1].time
    p.net_history[:] = 1
    p.reset(torch.tensor([0]))
    np.testing.assert_array_equal(p.native[1].qpos, before)
    assert p.native[1].time == time
    assert not p.net_history[0].any()
    assert p.net_history[1].all()
    assert p.data.time[0] == 0


def test_world_com_randomization_is_independent(physics):
    p = physics
    offsets = torch.zeros(2, 2, 3)
    offsets[0, 0, 0] = 0.02
    original = p.models[1].body_ipos.copy()
    p.randomize_torso_com(offsets)
    bid = p.host.body("p1_torso_link").id
    assert p.models[0].body_ipos[bid, 0] == pytest.approx(0.02)
    np.testing.assert_array_equal(p.models[1].body_ipos, original)
    assert p.host.body_ipos[bid, 0] == 0


def test_fk_matches_jacobian_without_overwriting_solver_contacts(physics):
    p = physics
    p.data.qvel[:] = 0.1
    p.substep()
    before = p.native[0].efc_force.copy()
    state = p.state()
    for role in range(2):
        bid = int(p.body_ids[role, 0])
        jac = np.zeros((3, p.host.nv))
        mujoco.mj_jacBody(p.models[0], p.fk[0], jac, None, bid)
        np.testing.assert_allclose(
            state["velocity"][0, role, 0], jac @ p.native[0].qvel, atol=1e-6
        )
    np.testing.assert_array_equal(before, p.native[0].efc_force)


def test_push_and_staged_reset_position_reach_native_state(physics):
    p = physics
    p.data.qpos[0, 0] = 0.4
    p.push(torch.tensor([0]), 0, torch.tensor([[0.3, 0, 0, 0, 0, 0]]))
    p.state()
    assert p.native[0].qpos[0] == pytest.approx(0.4)
    assert p.native[0].qvel[0] == pytest.approx(0.3)
    assert p.native[1].qvel[0] == 0


def test_native_contact_force_to_referee_and_history(physics):
    p = physics
    p.forward()
    p.substep()
    rows = p.official_contacts()[0]
    assert rows and rows[0]["normal_force_n"] > 0
    assert rows[0]["sample_dt_s"] == 0.005
    assert p.net_history[:, 0].abs().sum() > 0
    assert p.paired_history[:, 0].abs().sum() > 0
    # Sum all body contact forces: equal-and-opposite internal pair.
    torch.testing.assert_close(
        p.net_history[:, 0].sum((1, 2)), torch.zeros(2, 3), atol=1e-5, rtol=0
    )
    record = p.diagnostic_state()
    assert record["physics_backend"] == "mujoco_cpu"
    assert len(record["contacts"]) == 2


def test_training_imports_without_warp():
    import importlib

    importlib.import_module("holoboxing.training.envs.mujoco_env")
    importlib.import_module("holoboxing.data.collection")

    assert "warp" not in sys.modules
    assert "mujoco_warp" not in sys.modules


@pytest.mark.parametrize("threads", [1, 2])
def test_batched_control_interval_preserves_trajectory_and_contacts(
    physics, monkeypatch, threads
):
    reference = physics
    fast = physics_module.CompetitionPhysics(None, 2, "cpu", threads=threads)
    fast.set_gains(torch.ones(29) * 20, torch.ones(29) * 2)
    for frame in range(12):
        if frame == 5:
            for p in (reference, fast):
                p.reset(torch.tensor([0]))
                p.push(
                    torch.tensor([1]), 0, torch.tensor([[0.3, 0, 0, 0, 0, 0]])
                )
        for p in (reference, fast):
            p.targets[:] = 0.01 * frame
        for substep in range(4):
            reference.substep()
            fast.substep(sync_input=substep == 0, sync_output=substep == 3)
            for left, right in zip(reference.native, fast.native, strict=True):
                np.testing.assert_array_equal(left.qpos, right.qpos)
                np.testing.assert_array_equal(left.qvel, right.qvel)
                np.testing.assert_array_equal(left.efc_force, right.efc_force)
            assert reference.official_contacts() == fast.official_contacts()
            torch.testing.assert_close(
                reference.net_history, fast.net_history, rtol=0, atol=0
            )
            torch.testing.assert_close(
                reference.paired_history, fast.paired_history, rtol=0, atol=0
            )
        for name in reference.fields:
            torch.testing.assert_close(
                getattr(reference.data, name),
                getattr(fast.data, name),
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(
            reference.joint_acceleration,
            fast.joint_acceleration,
            rtol=0,
            atol=0,
        )
    fast.close()


@pytest.mark.parametrize("speed,expected_score", [(0.0, 0), (5.0, 1)])
def test_native_impact_scores_only_above_official_impulse(
    physics, speed, expected_score
):
    from holoboxing.training.envs.official_referee import OfficialReferees

    # Synthetic legal wrist/torso impact, using real mj_contactForce values.
    # This verifies the CPU-to-referee path, not a learned policy's ability.
    physics.data.qvel[:, 0] = speed
    referee = OfficialReferees(2)
    for _ in range(4):
        physics.substep()
        referee.sample(physics.official_contacts())
    _, delta = referee.observe(
        np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=bool)
    )
    assert delta.tolist() == [[expected_score, 0], [expected_score, 0]]
    impulse = referee.contact_metrics()["official_contact_impulse_max_ns"][
        :, 0
    ]
    assert np.all(impulse >= 5) if expected_score else np.all(impulse < 5)


def test_cpu_onnx_binding(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from holoboxing.training.envs.onnx_lower import (
        TrainingBoundSession,
        training_session,
    )

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10
    )
    path = tmp_path / "identity.onnx"
    onnx.save(model, path)
    bound = TrainingBoundSession(
        training_session(path, torch.device("cpu")), "cpu"
    )
    bound.inputs["x"].copy_(torch.arange(6).reshape(2, 3))
    torch.testing.assert_close(bound.run()["y"], bound.inputs["x"])


def fake_lower_setup(monkeypatch):
    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace

    from holoboxing.data import robot
    from holoboxing.training.envs import control, mujoco_env, onnx_lower

    class Session:
        def __init__(self, *args):
            self.metadata = {
                "joint_names": ",".join(JOINT_NAMES),
                "joint_stiffness": ",".join(["20"] * 29),
                "joint_damping": ",".join(["2"] * 29),
            }

    class Controller:
        def __init__(self, **kwargs):
            assert tuple(kwargs["scene_joint_names"]) == tuple(JOINT_NAMES)

        def reset(self, *args):
            pass

        def __call__(self, **kwargs):
            return kwargs["robot_joint_position"].clone()

    monkeypatch.setattr(mujoco_env, "OnnxLower", Session)
    monkeypatch.setattr(onnx_lower, "OnnxLower", Session)
    monkeypatch.setattr(mujoco_env, "FrozenHoloMotionV2Controller", Controller)
    monkeypatch.setattr(control, "FrozenHoloMotionV2Controller", Controller)
    monkeypatch.setattr(
        robot, "robot_file", lambda *args: nullcontext(Path("synthetic.xml"))
    )
    monkeypatch.setattr(mujoco_env, "robot_file", robot.robot_file)
    return SimpleNamespace(
        resources=SimpleNamespace(scene=Path("synthetic.xml"))
    )


@pytest.mark.parametrize("curriculum", [False, True])
def test_cpu_environment_control_reward_reset_and_push(
    physics, monkeypatch, tmp_path, curriculum
):
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.tasks.combat.config.sandbag import AGENTS, POINTS
    from holoboxing.training.tasks.combat.mdp.geometry import encode_output

    assets = fake_lower_setup(monkeypatch)
    env = MujocoSandbagEnv(
        assets,
        {"lower_onnx": "unused", "lower_config": "unused"},
        2,
        "cpu",
        tmp_path,
        official_rules=True,
        distance_curriculum=curriculum,
        rapid_approach=curriculum,
    )
    obs = env.reset()
    for a in AGENTS:
        assert obs[a]["task_policy"].shape == (2, 1097)
    state = env.physics.state()
    if curriculum:
        torch.testing.assert_close(
            torch.linalg.vector_norm(
                state["position"][:, 0, 0, :2]
                - state["position"][:, 1, 0, :2],
                dim=-1,
            ),
            torch.ones(2),
            atol=1e-5,
            rtol=0,
        )
    ids = [BODY_NAMES.index(n) for n in POINTS]
    p, q = state["position"][:, :, ids], state["quaternion"][:, :, ids]
    actions = {
        a: encode_output(
            p[:, r],
            q[:, r],
            p[:, r, None].expand(-1, 5, -1, -1),
            q[:, r, None].expand(-1, 5, -1, -1),
        )
        for r, a in enumerate(AGENTS)
    }
    env.next_push.zero_()
    for _ in range(3):
        obs, reward, done, info = env.step(actions)
        assert all(torch.isfinite(obs[a]["task_policy"]).all() for a in AGENTS)
        assert all(
            "official_score_delta" in env.reward_terms[a] for a in AGENTS
        )
        # Exercise the actual step/aggregation, not only the term registry.
        attack = env.reward_terms["robot_A"]
        assert "active_hit_shaping_reward" in attack
        assert "offensive_hit_reward_punch" not in attack
        assert "terminate_reward" not in attack
        torch.testing.assert_close(reward["robot_A"], sum(attack.values()))
    assert env.physics.steps == 12
    env._reset(torch.tensor([0]))
    assert env.controllers
    env.close()


@pytest.mark.parametrize("permuted_scene", [False, True])
def test_cpu_measured_collection_uses_real_steps(
    physics, monkeypatch, permuted_scene
):
    from holoboxing.data import collection, records

    assets = fake_lower_setup(monkeypatch)
    if permuted_scene:
        original_layout = records.scene_layout

        def layout(model):
            names, roots, joints, points = original_layout(model)
            return (
                names[::-1],
                roots,
                [(qi[::-1], vi[::-1]) for qi, vi in joints],
                points,
            )

        monkeypatch.setattr(records, "scene_layout", layout)
    monkeypatch.setattr(
        collection, "lower_identity", lambda _: {"synthetic": True}
    )
    collector = collection.MeasuredCollector(assets, ".", "cpu")
    expected_targets = {}

    class CheckedController:
        def __init__(self, role):
            self.role = role

        def reset(self):
            pass

        def __call__(self, **kwargs):
            for key, values, addresses in (
                (
                    "robot_joint_position",
                    collector.scene.data.qpos,
                    collector.joints[self.role][0],
                ),
                (
                    "robot_joint_velocity",
                    collector.scene.data.qvel,
                    collector.joints[self.role][1],
                ),
            ):
                by_name = dict(zip(collector.names, values[addresses]))
                expected = torch.tensor(
                    [[by_name[n] for n in JOINT_NAMES]], dtype=torch.float32
                )
                torch.testing.assert_close(kwargs[key], expected)
            target = kwargs["robot_joint_position"] + torch.arange(29) * 0.001
            expected_targets[self.role] = target[0].clone()
            return target

    collector.policies = [CheckedController(role) for role in range(2)]
    model, data = collector.scene.model, collector.scene.data
    state = collection.measured_state(model, data, collector.point_ids)
    frames = 10
    arrays = {
        key: torch.tensor(
            np.repeat(value[None], frames, axis=0), dtype=torch.float32
        )
        for key, value in zip(
            ("position", "quaternion"), state[:2], strict=True
        )
    }
    trajectory = {"roles": 2, "joint_names": collector.names}
    trajectory["root_position"] = arrays["position"][:, :, 0].numpy()
    trajectory["root_quaternion"] = arrays["quaternion"][:, :, 0].numpy()
    for key, width in (
        ("root_origin_velocity", 3),
        ("root_angular_velocity_world", 3),
        ("joint_position", 29),
        ("joint_velocity", 29),
    ):
        trajectory[key] = np.zeros((frames, 2, width))
    packet, record = collector.collect(arrays, trajectory)
    assert packet["obs"].shape == (10, 1097)
    assert packet["target_pose"].shape == (10, 225)
    assert collector.physics.steps == 20
    for role in range(2):
        torch.testing.assert_close(
            collector.physics.targets[0, role], expected_targets[role]
        )
    assert record["collection_recipe"]["physics_backend"] == "mujoco_cpu"
    collector.close()


def test_robot_h5_writer_roundtrip_without_warp(tmp_path):
    import h5py
    from holomotion.src.training.data_production.robot_h5 import (
        RobotH5ShardWriter,
    )

    path = tmp_path / "motion.h5"
    arrays = {
        "ref_root_pos": np.zeros((6, 3), dtype=np.float32),
        "ref_root_rot": np.tile(
            np.array([0, 0, 0, 1], dtype=np.float32), (6, 1)
        ),
        "ref_dof_pos": np.zeros((6, 29), dtype=np.float32),
    }
    writer = RobotH5ShardWriter(path, chunks_t=6)
    assert writer.append_motion(
        motion_id=0, arrays=arrays, metadata_json="{}"
    ) == (0, 6)
    writer.finalize()
    with h5py.File(path, "r") as f:
        for name, value in arrays.items():
            np.testing.assert_array_equal(f[name][:], value)
    assert "warp" not in sys.modules


@pytest.mark.parametrize("bad_reset", [False, True])
def test_nonfinite_observation_recovery_still_checks_reset_rows(
    physics, monkeypatch, tmp_path, bad_reset
):
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.tasks.combat.config.sandbag import AGENTS

    assets = fake_lower_setup(monkeypatch)
    env = MujocoSandbagEnv(
        assets,
        {"lower_onnx": "unused", "lower_config": "unused"},
        2,
        "cpu",
        tmp_path,
        official_rules=True,
    )
    env.reset()
    original_read = env._read
    calls = 0

    def read(reset, previous=None):
        nonlocal calls
        result = original_read(reset, previous)
        calls += 1
        if calls == 1 or bad_reset:
            result[AGENTS[0]]["task_policy"][0, 0] = float("nan")
        return result

    monkeypatch.setattr(env, "_read", read)
    actions = {a: torch.zeros(2, 225) for a in AGENTS}
    if bad_reset:
        with pytest.raises(FloatingPointError, match="post-reset"):
            env.step(actions)
    else:
        obs, rewards, done, info = env.step(actions)
        assert info["invalid_envs"].tolist() == [True, False]
        assert done[0]
        assert all(torch.isfinite(obs[a]["task_policy"]).all() for a in AGENTS)
    env.close()


def test_nonfinite_motor_target_rejected_before_physics(
    physics, monkeypatch, tmp_path
):
    from holoboxing.training.envs.mujoco_env import MujocoSandbagEnv
    from holoboxing.training.tasks.combat.config.sandbag import AGENTS

    assets = fake_lower_setup(monkeypatch)
    env = MujocoSandbagEnv(
        assets,
        {"lower_onnx": "unused", "lower_config": "unused"},
        2,
        "cpu",
        tmp_path,
        official_rules=True,
    )
    env.reset()
    env.controllers[AGENTS[0]] = lambda **kwargs: torch.full(
        (2, 29), float("nan")
    )
    with pytest.raises(FloatingPointError, match="lower controller output"):
        env.step({a: torch.zeros(2, 225) for a in AGENTS})
    assert env.physics.steps == 0
    env.close()
