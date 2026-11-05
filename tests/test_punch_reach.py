"""Real MuJoCo geometry checks, not trained-skill performance evidence."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from holoboxing.training.skills.punch_reach import PunchReachGeometry


def scene(*, chest=False, missing_hand=False, collision_mask=1):
    hands = "".join(
        f"""<body name="p1_{side}_wrist_yaw_link" pos=".2 {y} .1">
          <joint name="{side}_extension" type="slide" axis="1 0 0"/>
          <geom name="{side}_glove" type="sphere" pos=".1 0 0" size=".05"
                contype="1" conaffinity="1"/>
        </body>"""
        for side, y in (("left", 0.2), ("right", -0.2))
        if not (missing_hand and side == "left")
    )
    target_name = "logo_link" if chest else "torso_link"
    target = '<body name="p2_torso_link" pos="0 0 .2"/>' if chest else ""
    return mujoco.MjModel.from_xml_string(
        f"""<mujoco><option gravity="0 0 0"/><worldbody>
          <body name="p1_pelvis" pos="0 0 1"><freejoint name="red_root"/>
            <geom type="sphere" size=".02"/>
            <body name="p1_torso_link" pos="0 0 .1">
              <inertial pos=".08 .03 .02" mass="2" diaginertia=".1 .1 .1"/>
              {hands}
            </body>
          </body>
          <body name="p2_pelvis" pos="1 0 1"><freejoint name="blue_root"/>
            <geom type="sphere" pos="-.6 -.2 .2" size=".09"/>
            {target}
            <body name="p2_{target_name}" pos="0 0 .2">
              <geom name="torso_target" type="sphere" size=".1"
                    contype="{collision_mask}" conaffinity="{collision_mask}"/>
            </body>
            <body name="p2_head_link" pos="-.6 .2 .2">
              <geom type="sphere" size=".09"/>
            </body>
          </body>
        </worldbody></mujoco>"""
    )


def context(model, worlds=1):
    data = [mujoco.MjData(model) for _ in range(worlds)]
    for item in data:
        mujoco.mj_forward(model, item)
    return SimpleNamespace(
        device="cpu", physics=SimpleNamespace(models=[model] * worlds, fk=data)
    )


@pytest.mark.parametrize("chest", [False, True])
def test_surface_gap_uses_geom_offsets_and_excludes_head_and_pelvis(chest):
    model = scene(chest=chest)
    combat = context(model)
    before = combat.physics.fk[0].qpos.copy()
    result = PunchReachGeometry(model).measure(combat)
    expected = np.hypot(0.7, 0.2) - 0.15
    assert result["gap_m"].shape == (1, 2)
    assert result["gap_m"].dtype == torch.float32
    assert result["target_valid"].tolist() == [[True, True]]
    np.testing.assert_allclose(result["gap_m"].numpy(), expected, rtol=1e-6)
    # A nearby head/pelvis would overlap both gloves if incorrectly included.
    assert torch.all(result["gap_m"] > 0.5)
    assert np.array_equal(before, combat.physics.fk[0].qpos)
    assert combat.physics.fk[0].time == 0


@pytest.mark.parametrize(
    "rigid_velocity", [(1.5, 0, 0, 0, 0, 0), (0, 0, 0, 0, 0, 2)]
)
def test_rigid_torso_motion_is_not_arm_extension(rigid_velocity):
    model = scene()
    combat = context(model)
    data = combat.physics.fk[0]
    data.qvel[:6] = rigid_velocity
    mujoco.mj_forward(model, data)
    result = PunchReachGeometry(model).measure(combat)
    assert torch.allclose(
        result["arm_aim_speed_mps"], torch.zeros(1, 2), atol=1e-6
    )
    assert torch.allclose(
        result["arm_outward_speed_mps"], torch.zeros(1, 2), atol=1e-6
    )
    assert result["closing_speed_mps"].abs().max() > 0.1


def test_arm_extension_and_opponent_motion_have_distinct_velocities():
    model = scene()
    combat = context(model)
    data = combat.physics.fk[0]
    data.qvel[model.joint("left_extension").dofadr[0]] = 2
    data.qvel[model.joint("blue_root").dofadr[0]] = -0.5
    mujoco.mj_forward(model, data)
    result = PunchReachGeometry(model).measure(combat)
    aim = 2 * 0.7 / np.hypot(0.7, 0.2)
    assert result["arm_aim_speed_mps"][0, 0].item() == pytest.approx(aim)
    assert result["arm_aim_speed_mps"][0, 1].item() == pytest.approx(0)
    assert result["arm_outward_speed_mps"][0, 0] > 0
    assert (
        result["closing_speed_mps"][0, 0] > result["arm_aim_speed_mps"][0, 0]
    )
    assert result["closing_speed_mps"][0, 1] > 0


def test_overlap_keeps_signed_gap_and_forward_normal():
    model = scene()
    combat = context(model)
    data = combat.physics.fk[0]
    blue = model.joint("blue_root").qposadr[0]
    data.qpos[blue : blue + 2] = [0.4, 0.2]
    data.qvel[model.joint("left_extension").dofadr[0]] = 1
    mujoco.mj_forward(model, data)
    result = PunchReachGeometry(model).measure(combat)
    assert result["gap_m"][0, 0].item() == pytest.approx(-0.05)
    assert result["arm_aim_speed_mps"][0, 0].item() == pytest.approx(1)


def test_invalid_runtime_world_retains_nan_and_does_not_poison_other_world():
    model = scene()
    combat = context(model, 2)
    combat.physics.fk[1].geom_xpos[0, 0] = np.nan
    result = PunchReachGeometry(model).measure(combat)
    assert result["target_valid"].tolist() == [[True, True], [False, False]]
    for name, value in result.items():
        if name != "target_valid":
            assert torch.isfinite(value[0]).all()
            assert torch.isnan(value[1]).all()


@pytest.mark.parametrize(
    "options", [{"missing_hand": True}, {"collision_mask": 2}]
)
def test_missing_or_incompatible_collision_pairs_are_configuration_errors(
    options,
):
    with pytest.raises(ValueError, match="no collidable official"):
        PunchReachGeometry(scene(**options))


def test_runtime_nonfinite_distance_is_not_silently_clamped(monkeypatch):
    model = scene()
    combat = context(model)
    reader = PunchReachGeometry(model)
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: np.nan)
    result = reader.measure(combat)
    assert not result["target_valid"].any()
    assert torch.isnan(result["gap_m"]).all()


def test_unsupported_distance_is_a_constructor_error(monkeypatch):
    monkeypatch.setattr(mujoco, "mj_geomDistance", lambda *args: np.inf)
    with pytest.raises(ValueError, match="unsupported/nonfinite"):
        PunchReachGeometry(scene())


def test_model_layout_change_is_not_treated_as_runtime_invalid():
    model = scene()
    reader = PunchReachGeometry(model)
    other = scene()
    other.geom_type[other.geom("torso_target").id] = (
        mujoco.mjtGeom.mjGEOM_CAPSULE
    )
    with pytest.raises(ValueError, match="layout differs"):
        reader.measure(context(other))
