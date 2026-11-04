"""Single-robot native physics checks; mock lower is not a trained policy."""

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from holoboxing.training.skills import physics as module
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


def robot_fixture_spec():
    parts = [
        '<mujoco><compiler angle="radian"/><worldbody>',
        '<body name="pelvis" pos="0 0 .9"><freejoint/>',
        '<geom type="sphere" size=".04" mass="3"',
        ' contype="0" conaffinity="0"/>',
    ]
    for index, (body, joint) in enumerate(
        zip(BODY_NAMES[1:], JOINT_NAMES, strict=True)
    ):
        # Independent joint branches simplify the FK/Jacobian expectations.
        foot = body in ("left_ankle_roll_link", "right_ankle_roll_link")
        radius = 0.03 if foot else 0.01
        parts.extend(
            [
                f'<body name="{body}" pos=".1 {index * 0.03} 0">',
                f'<joint name="{joint}" axis="0 1 0" limited="true"',
                ' range="-3 3" actuatorfrclimited="true"',
                ' actuatorfrcrange="-10 10"/>',
                f'<geom type="sphere" size="{radius}" mass=".1"',
                f' contype="{int(foot)}" conaffinity="{int(foot)}"/></body>',
            ]
        )
    parts.append("</body></worldbody></mujoco>")
    return mujoco.MjSpec.from_string("".join(parts))


class FakeSession:
    def __init__(self, path, batch, device, scratch):
        self.batch_size = batch
        self.metadata = {
            "joint_names": ",".join(JOINT_NAMES),
            "joint_stiffness": ",".join(["20"] * 29),
            "joint_damping": ",".join(["2"] * 29),
        }


class FakeLower:
    def __init__(self, **kwargs):
        self.batch = kwargs["batch_size"]
        self.reset_ids = []
        self.delta = 0.0
        self.last_inputs = None
        assert tuple(kwargs["scene_joint_names"]) == JOINT_NAMES

    def reset(self, ids):
        self.reset_ids.append(ids.cpu().tolist())

    def __call__(self, **kwargs):
        self.last_inputs = {k: v.clone() for k, v in kwargs.items()}
        assert kwargs["reference_position_w"].shape == (self.batch, 5, 5, 3)
        assert kwargs["reference_quaternion_wxyz"].shape == (
            self.batch,
            5,
            5,
            4,
        )
        return kwargs["robot_joint_position"] + self.delta


@pytest.fixture
def assets(monkeypatch, tmp_path):
    arena = mujoco.MjModel.from_xml_string(
        '<mujoco><option timestep=".005"/>'
        '<worldbody><geom name="smpl_ring_baseLink" type="box" '
        'size="5 5 .1" pos="0 0 .2" friction=".8 .02 .01"/>'
        "</worldbody></mujoco>"
    )
    monkeypatch.setattr(module, "load_scene_model", lambda _: arena)
    monkeypatch.setattr(
        module, "robot_spec", lambda *args: robot_fixture_spec()
    )
    reset = tmp_path / "initial-state.json"
    names = list(reversed(JOINT_NAMES))
    values = {name: 0.001 * index for index, name in enumerate(JOINT_NAMES)}
    pose = [3, -2, 1.2, 1, 0, 0, 0] + [values[n] for n in names]
    reset.write_text(json.dumps({"qpos": [pose, pose], "dof_names": names}))
    monkeypatch.setattr(module, "verify_bundle", lambda value: Path(value))
    monkeypatch.setattr(
        module,
        "frozen_manifest",
        lambda: {
            "files": {
                "config.yaml": {"sha256": "config-hash"},
                "exported/model_10500.onnx": {"sha256": "model-hash"},
            }
        },
    )
    monkeypatch.setattr(module, "OnnxLower", FakeSession)
    monkeypatch.setattr(module, "FrozenHoloMotionV2Controller", FakeLower)
    monkeypatch.setattr(
        module, "robot_file", lambda *args: nullcontext(Path("unused.xml"))
    )
    return SimpleNamespace(
        resources=SimpleNamespace(
            initial_state=reset, scene=Path("unused.xml")
        )
    )


@pytest.fixture
def physics(assets):
    instance = module.SoloPhysics(assets, "unused-lower", 2, "cpu")
    yield instance
    instance.close()


def targets(physics):
    state = physics.state()
    return (
        state["point_position"][:, None].expand(-1, 5, -1, -1).clone(),
        state["point_quaternion"][:, None].expand(-1, 5, -1, -1).clone(),
    )


def test_scene_contains_one_robot_and_preserves_joint_and_ground_parameters(
    assets,
):
    model = module.load_solo_model(assets)
    assert (model.nq, model.nv, model.nu) == (36, 35, 29)
    assert sum(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE) == 1
    assert model.geom("skill_ground").type == mujoco.mjtGeom.mjGEOM_PLANE
    assert not any(
        (model.body(i).name or "").startswith(("p1_", "p2_"))
        for i in range(model.nbody)
    )
    np.testing.assert_allclose(
        model.geom("skill_ground").friction, [0.8, 0.02, 0.01]
    )
    for name in JOINT_NAMES:
        j = model.joint(name).id
        dof = model.jnt_dofadr[j]
        assert model.dof_damping[dof] == 0.05
        assert model.dof_armature[dof] == 0.01
        expected = (
            0.1 if "_wrist_pitch_" in name or "_wrist_yaw_" in name else 0.2
        )
        assert model.dof_frictionloss[dof] == expected


def test_official_reset_is_named_reordered_and_floor_normalized(physics):
    state = physics.state()
    torch.testing.assert_close(
        state["root_position"], torch.tensor([[0, 0, 0.9]] * 2)
    )
    torch.testing.assert_close(
        state["joint_position"],
        torch.arange(29).float()[None].repeat(2, 1) * 0.001,
    )
    assert physics.metadata["robots_per_world"] == 1
    assert physics.metadata["lower_sha256"]["config.yaml"] == "config-hash"


def test_reset_world_velocity_conversion_and_row_isolation(physics):
    before = physics.native[1].qpos.copy()
    physics.native[1].time = 12
    quat = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    state = physics.reset(
        [0],
        [[1, 2, 1.1]],
        [quat],
        torch.zeros((1, 29)),
        root_linear_velocity=[[0.2, 0.3, 0.4]],
        root_angular_velocity=[[1, 0, 0]],
    )
    np.testing.assert_array_equal(before, physics.native[1].qpos)
    assert physics.native[1].time == 12
    assert physics.lower.reset_ids == [[0]]
    torch.testing.assert_close(
        state["root_angular_velocity"][0], torch.tensor([1.0, 0, 0])
    )
    torch.testing.assert_close(
        state["root_angular_velocity_local"][0], torch.tensor([0.0, -1, 0])
    )
    torch.testing.assert_close(
        state["root_linear_velocity"][0], torch.tensor([0.2, 0.3, 0.4])
    )


def test_same_instant_body_velocities_match_independent_jacobians(physics):
    data = physics.native[0]
    data.qvel[:] = np.linspace(-0.2, 0.6, physics.host.nv)
    data.qpos[physics.qi] += 0.15
    before = physics.diagnostic_state()
    state = physics.state()
    check = mujoco.MjData(physics.models[0])
    check.qpos[:] = data.qpos
    check.qvel[:] = data.qvel
    mujoco.mj_forward(physics.models[0], check)
    for index, body in enumerate(physics.body_ids):
        jacobian = np.zeros((3, physics.host.nv))
        mujoco.mj_jacBody(physics.models[0], check, jacobian, None, int(body))
        np.testing.assert_allclose(
            state["body_linear_velocity"][0, index].numpy(),
            jacobian @ check.qvel,
            atol=1e-7,
        )
    for name, saved in before.items():
        np.testing.assert_array_equal(physics.diagnostic_state()[name], saved)


def test_control_step_matches_four_independent_torque_limited_substeps(
    physics,
):
    physics.lower.delta = 100
    reference = mujoco.MjData(physics.models[0])
    reference.qpos[:] = physics.native[0].qpos
    for _ in range(4):
        reference.ctrl[physics.ai] = 10
        mujoco.mj_step(physics.models[0], reference)
    state = physics.step(*targets(physics))
    np.testing.assert_allclose(
        physics.native[0].qpos, reference.qpos, atol=1e-12
    )
    np.testing.assert_allclose(
        physics.native[0].qvel, reference.qvel, atol=1e-12
    )
    assert physics.steps == 1
    torch.testing.assert_close(state["time"], torch.full((2,), 0.02))
    assert physics.lower.last_inputs[
        "current_reference_root_position_w"
    ].shape == (2, 3)
    assert torch.all(state["foot_contact"] == 0)


def test_solver_warning_preserves_before_and_after_state(physics, monkeypatch):
    native_step = mujoco.mj_step

    def warn(model, data):
        native_step(model, data)
        data.warning.number[0] += 1

    monkeypatch.setattr(mujoco, "mj_step", warn)
    with pytest.raises(RuntimeError, match="world 0"):
        physics.step(*targets(physics))
    record = physics.failure_state
    assert record["before"]["time"][0] == 0
    assert record["after"]["time"][0] == 0.005
    assert record["after"]["warning_counts"][0, 0] == 1
    assert physics.steps == 0


def test_foot_contact_flags_follow_real_ground_constraint_forces(physics):
    model, data = physics.models[0], physics.native[0]
    data.qpos[physics.root_qi + 2] = 0.02
    mujoco.mj_forward(model, data)
    assert data.ncon >= 2
    contacts = physics.state()["foot_contact"]
    assert contacts[0].all()
    assert not contacts[1].any()


@pytest.mark.parametrize("ids", [[0, 0], [-1], [2]])
def test_invalid_reset_indices_rejected_before_any_mutation(physics, ids):
    before = physics.native[0].qpos.copy()
    with pytest.raises(ValueError, match="Reset ids"):
        physics.reset(ids, [[0, 0, 1]], [[1, 0, 0, 0]], [[0] * 29])
    np.testing.assert_array_equal(before, physics.native[0].qpos)


def test_close_is_idempotent_and_rejects_new_steps(physics):
    reference = targets(physics)
    physics.close()
    physics.close()
    with pytest.raises(RuntimeError, match="closed"):
        physics.step(*reference)
