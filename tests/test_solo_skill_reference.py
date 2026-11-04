"""Motion provenance and named single-robot FK; not control performance."""

import json

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from holoboxing.data.common import sha256
from holoboxing.training.skills import reference
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


def solo_model():
    parts = [
        '<mujoco><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size=".05" mass="3"/>'
    ]
    motors = []
    for i, (body, joint) in enumerate(
        zip(BODY_NAMES[1:], JOINT_NAMES, strict=True)
    ):
        parts.append(
            f'<body name="{body}" pos=".1 {i * 0.02} .03">'
            f'<joint name="{joint}" axis="0 1 0"/>'
            '<geom type="sphere" size=".01" mass=".1"/></body>'
        )
        motors.append(f'<motor joint="{joint}"/>')
    parts.append(
        "</body></worldbody><actuator>"
        + "".join(motors)
        + "</actuator></mujoco>"
    )
    return mujoco.MjModel.from_xml_string("".join(parts))


def packet(group, frames=900):
    time = np.arange(frames) / 50
    pos = np.zeros((frames, 1, 3))
    pos[..., 2] = 0.8
    if group == "kangning-010":
        distance = np.interp(
            np.arange(frames),
            [0, 85, 275, 429, 574, 765, frames - 1],
            [0, 0, 3, 0, 3, 0, 0],
        )
        heading = reference.TASK_FORWARD_RADIANS[group]
    elif group == "kangning-011":
        distance = np.interp(
            np.arange(frames),
            [0, 80, 203, 345, 475, 650, frames - 1],
            [0, 0, 2.5, 0, 2.5, 0, 0],
        )
        heading = reference.TASK_FORWARD_RADIANS[group] + np.pi / 2
    else:
        distance, heading = time * 0.05, 0.7
    pos[:, 0, :2] = distance[:, None] * np.array(
        [np.cos(heading), np.sin(heading)]
    ) + [4, 3]
    quaternion = np.zeros((frames, 1, 4))
    quaternion[..., 0] = 1
    return {
        "fps": np.array(50),
        "joint_names": np.array(JOINT_NAMES),
        "joint_pos": np.sin(time[:, None]) * 0.01
        + np.arange(29)[None] * 0.001,
        "joint_vel": np.cos(time[:, None]) * np.ones((1, 29)) * 0.01,
        "body_pos_w": pos,
        "body_quat_w": quaternion,
        "body_ang_vel_w": np.zeros_like(pos),
    }


@pytest.fixture
def originals(tmp_path, monkeypatch):
    root = tmp_path / "originals"
    (root / "boxing_20251226").mkdir(parents=True)
    rows = []
    for group in (
        "kangning-001",
        "kangning-003",
        "kangning-010",
        "kangning-011",
        "kangning-045",
    ):
        path = root / f"boxing_20251226/{group}.npz"
        np.savez_compressed(path, **packet(group))
        rows.append(
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path)}
        )
    catalog = tmp_path / "catalog.json"
    catalog.write_text(json.dumps({"files": rows}))
    monkeypatch.setattr(reference, "CATALOG", catalog)
    return root


@pytest.mark.parametrize("skill,count", [("punch", 2), ("locomotion", 8)])
def test_pinned_full_body_library_and_no_source_mutation(
    originals, skill, count
):
    before = {path: sha256(path) for path in originals.rglob("*.npz")}
    model = solo_model()
    library = reference.MotionLibrary(
        originals, model, JOINT_NAMES, BODY_NAMES, skill=skill
    )
    assert len(library.clips) == count
    assert library.manifest["physics_rollouts_collected"] is False
    for clip in library.clips:
        np.testing.assert_allclose(clip["root_position"][0, :2], 0)
        np.testing.assert_allclose(clip["root_position"][:, 2], 0.8)
        assert clip["joint_position"].shape == (clip["frames"], 29)
        assert clip["point_position"].shape == (clip["frames"], 5, 3)
        for slot, name in enumerate(reference.INPUT_LINK_NAMES):
            torch.testing.assert_close(
                clip["point_position"][:, slot],
                clip["body_position"][:, BODY_NAMES.index(name)],
            )
    assert all(sha256(path) == digest for path, digest in before.items())
    if skill == "locomotion":
        assert set(map(tuple, library.directions.tolist())) == {
            (1.0, 0.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (0.0, -1.0),
        }


def test_named_joints_reorder_without_altering_poses(originals):
    order = tuple(reversed(JOINT_NAMES))
    library = reference.MotionLibrary(
        originals, solo_model(), order, BODY_NAMES, skill="punch"
    )
    source = packet("kangning-001")
    np.testing.assert_allclose(
        library.clips[0]["joint_position"].numpy(),
        source["joint_pos"][:, ::-1],
        atol=2e-9,
    )
    np.testing.assert_allclose(
        library.clips[0]["root_quaternion"].numpy(),
        source["body_quat_w"][:, 0],
    )


@pytest.mark.parametrize(
    "mode,names,bounds,direction",
    [
        (
            "full",
            ["footwork_010", "footwork_011"],
            [[0, 900], [0, 900]],
            [[0.0, 0.0], [0.0, 0.0]],
        ),
        (
            "forward",
            ["forward_1", "forward_2"],
            [[85, 275], [425, 574]],
            [[1.0, 0.0], [1.0, 0.0]],
        ),
    ],
)
def test_locomotion_modes_keep_original_bounds_and_direction_semantics(
    originals, mode, names, bounds, direction
):
    library = reference.MotionLibrary(
        originals,
        solo_model(),
        JOINT_NAMES,
        BODY_NAMES,
        skill="locomotion",
        locomotion_mode=mode,
    )
    assert library.names == names
    assert [c["source_bounds"] for c in library.clips] == bounds
    assert library.directions.tolist() == direction
    assert library.manifest["locomotion_mode"] == mode
    assert library.manifest["navigation_direction_primitives"] == (
        mode != "full"
    )
    for clip in library.clips:
        assert clip["navigation_direction_primitive"] == (mode != "full")
        assert clip["guard_heading_world"] == pytest.approx(
            reference.TASK_FORWARD_RADIANS[clip["group"]]
        )
        source = packet(clip["group"])["joint_pos"]
        start, end = clip["source_bounds"]
        np.testing.assert_allclose(
            clip["joint_position"], source[start:end], atol=2e-9
        )


def test_locomotion_mode_does_not_change_punch_manifest(originals):
    model = solo_model()
    default = reference.MotionLibrary(
        originals, model, JOINT_NAMES, BODY_NAMES, skill="punch"
    )
    alternate = reference.MotionLibrary(
        originals,
        model,
        JOINT_NAMES,
        BODY_NAMES,
        skill="punch",
        locomotion_mode="full",
    )
    assert default.manifest == alternate.manifest
    with pytest.raises(ValueError, match="locomotion"):
        reference.MotionLibrary(
            originals,
            model,
            JOINT_NAMES,
            BODY_NAMES,
            skill="locomotion",
            locomotion_mode="unknown",
        )


def test_clamped_lookahead_exposes_validity_and_does_not_loop(originals):
    library = reference.MotionLibrary(
        originals, solo_model(), JOINT_NAMES, BODY_NAMES, skill="locomotion"
    )
    clips = torch.tensor([0, 3])
    frames = library.lengths[clips] - 2
    result = library.gather(clips, frames, offsets=(0, 1, 2, 25))
    torch.testing.assert_close(
        result["valid"],
        torch.tensor([[True, True, False, False]]).repeat(2, 1),
    )
    torch.testing.assert_close(
        result["joint_position"][:, 1], result["joint_position"][:, -1]
    )
    for ids, phases in (([8], [0]), ([0], [-1]), ([0], [0.5])):
        with pytest.raises(ValueError):
            library.gather(ids, phases)


def test_changed_original_rejected_before_loading(originals):
    path = originals / "boxing_20251226/kangning-001.npz"
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="pinned catalog"):
        reference.MotionLibrary(
            originals, solo_model(), JOINT_NAMES, BODY_NAMES, skill="punch"
        )


def test_fps_is_not_silently_reinterpreted(originals):
    path = originals / "boxing_20251226/kangning-001.npz"
    values = packet("kangning-001")
    values["fps"] = np.array(60)
    np.savez_compressed(path, **values)
    catalog = json.loads(reference.CATALOG.read_text())
    next(row for row in catalog["files"] if row["path"].endswith("001.npz"))[
        "sha256"
    ] = sha256(path)
    reference.CATALOG.write_text(json.dumps(catalog))
    with pytest.raises(ValueError, match="50 Hz"):
        reference.MotionLibrary(
            originals, solo_model(), JOINT_NAMES, BODY_NAMES, skill="punch"
        )


def test_world_angular_velocity_and_origin_velocity_match_jacobian(originals):
    model = solo_model()
    source = originals / "boxing_20251226/kangning-045.npz"
    trajectory = reference.load_trajectory(source, JOINT_NAMES)
    angle = 0.8
    q = Rotation.from_euler("z", angle).as_quat()[[3, 0, 1, 2]]
    trajectory["root_quaternion"][:] = q
    trajectory["root_angular_velocity_world"][:] = [0.3, 0.2, 0.5]
    arrays = reference.reference_kinematics(
        model, trajectory, JOINT_NAMES, BODY_NAMES, (10, 20)
    )
    d = mujoco.MjData(model)
    d.qpos[:3] = arrays["root_position"][0]
    d.qpos[3:7] = q
    d.qpos[7:] = arrays["joint_position"][0]
    d.qvel[:3] = arrays["root_linear_velocity"][0]
    d.qvel[3:6] = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix().T @ [
        0.3,
        0.2,
        0.5,
    ]
    d.qvel[6:] = arrays["joint_velocity"][0]
    mujoco.mj_forward(model, d)
    for slot, name in enumerate(BODY_NAMES):
        jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
        mujoco.mj_jacBody(model, d, jacp, jacr, model.body(name).id)
        np.testing.assert_allclose(
            arrays["body_linear_velocity"][0, slot], jacp @ d.qvel, atol=1e-12
        )
        np.testing.assert_allclose(
            arrays["body_angular_velocity"][0, slot], jacr @ d.qvel, atol=1e-12
        )
