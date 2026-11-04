"""Original turn crop/FK provenance, not physical turn performance."""

import hashlib
import json

import mujoco
import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from test_solo_skill_reference import packet, solo_model

from holoboxing.data.common import sha256
from holoboxing.evaluation.runtime.model_contract import INPUT_LINK_NAMES
from holoboxing.training.skills import turn_reference
from holoboxing.training.tasks.combat.config.sandbag import (
    BODY_NAMES,
    JOINT_NAMES,
)


@pytest.fixture
def turn_source(tmp_path, monkeypatch):
    root = tmp_path / "originals"
    path = root / "boxing_20251226/kangning-028.npz"
    path.parent.mkdir(parents=True)
    values = packet("kangning-028", frames=200)
    # A positive turn through the -pi/pi boundary; no root-yaw scaling.
    angles = np.linspace(np.deg2rad(170), np.deg2rad(250), 200)
    values["body_quat_w"][:, 0] = Rotation.from_euler("z", angles).as_quat()[
        :, [3, 0, 1, 2]
    ]
    values["body_ang_vel_w"][:, 0, 2] = (angles[1] - angles[0]) * 50
    np.savez_compressed(path, **values)
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": path.relative_to(root).as_posix(),
                        "sha256": sha256(path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(turn_reference, "CATALOG", catalog)
    return root, path, values


def library(turn_source, segments=None, joints=JOINT_NAMES, normalize=True):
    return turn_reference.TurnReferenceLibrary(
        turn_source[0],
        solo_model(),
        joints,
        BODY_NAMES,
        segments=segments,
        normalize_initial_yaw=normalize,
    )


def test_crops_preserve_source_order_original_yaw_and_full_named_pose(
    turn_source,
):
    _, path, values = turn_source
    digest = sha256(path)
    segments = [("later", 100, 170), ("earlier", 20, 80)]
    result = library(turn_source, segments, normalize=False)
    assert result.names == ["later", "earlier"]
    assert result.lengths.tolist() == [70, 60]
    for clip, (_, start, end) in zip(result.clips, segments, strict=True):
        np.testing.assert_allclose(
            clip["joint_position"], values["joint_pos"][start:end], atol=2e-9
        )
        np.testing.assert_allclose(
            clip["root_quaternion"],
            values["body_quat_w"][start:end, 0],
            atol=3e-8,
        )
        expected = values["body_pos_w"][start:end, 0].copy()
        expected[:, :2] -= expected[0, :2].copy()
        np.testing.assert_allclose(clip["root_position"], expected, atol=3e-8)
        np.testing.assert_allclose(
            clip["root_angular_velocity"],
            values["body_ang_vel_w"][start:end, 0],
            atol=3e-8,
        )
        assert clip["source_net_yaw_deg"] == pytest.approx(
            80 * (end - start - 1) / 199
        )
    assert sha256(path) == digest
    assert result.manifest["angle_augmentation"] is False
    assert result.manifest["mirrored"] is False
    assert result.manifest["physics_rollouts_collected"] is False


def test_named_joint_reorder_preserves_robot_body_kinematics(turn_source):
    segments = [("turn", 12, 74)]
    normal = library(turn_source, segments)
    reverse = library(turn_source, segments, tuple(reversed(JOINT_NAMES)))
    np.testing.assert_allclose(
        reverse.clips[0]["joint_position"],
        normal.clips[0]["joint_position"][:, torch.arange(28, -1, -1)],
    )
    for key in (
        "body_position",
        "body_quaternion",
        "point_position",
        "body_linear_velocity",
    ):
        torch.testing.assert_close(
            normal.clips[0][key], reverse.clips[0][key], atol=1e-7, rtol=0
        )
    assert normal.manifest["sha256"] != reverse.manifest["sha256"]


def test_reference_points_match_independent_named_mujoco_fk(turn_source):
    result = library(turn_source, [("turn", 20, 80)])
    clip = result.clips[0]
    model = solo_model()
    data = mujoco.MjData(model)
    root_joint = int(model.body("pelvis").jntadr[0])
    root_qi = int(model.jnt_qposadr[root_joint])
    for frame in (0, 27, 59):
        data.qpos[root_qi : root_qi + 3] = clip["root_position"][frame].numpy()
        data.qpos[root_qi + 3 : root_qi + 7] = clip["root_quaternion"][
            frame
        ].numpy()
        for joint, value in zip(
            JOINT_NAMES, clip["joint_position"][frame], strict=True
        ):
            data.qpos[int(model.joint(joint).qposadr[0])] = value
        mujoco.mj_forward(model, data)
        for point, name in enumerate(INPUT_LINK_NAMES):
            bid = model.body(name).id
            np.testing.assert_allclose(
                clip["point_position"][frame, point], data.xpos[bid], atol=5e-8
            )
            assert abs(
                np.dot(
                    clip["point_quaternion"][frame, point].numpy(),
                    data.xquat[bid],
                )
            ) == pytest.approx(1, abs=1e-7)


def test_clamped_lookahead_does_not_splice_or_loop_crop(turn_source):
    result = library(turn_source, [("a", 10, 30), ("b", 70, 105)])
    batch = result.gather(
        torch.tensor([0, 1]), torch.tensor([18, 33]), offsets=(0, 1, 2, 10)
    )
    assert batch["valid"].tolist() == [[True, True, False, False]] * 2
    torch.testing.assert_close(
        batch["joint_position"][:, 1], batch["joint_position"][:, -1]
    )
    torch.testing.assert_close(
        batch["root_quaternion"][:, 1], batch["root_quaternion"][:, -1]
    )


def test_provenance_digest_binds_crop_order_and_exact_source(turn_source):
    first = library(turn_source, [("a", 10, 30), ("b", 70, 105)])
    second = library(turn_source, [("b", 70, 105), ("a", 10, 30)])
    assert first.manifest["sha256"] != second.manifest["sha256"]
    manifest = dict(first.manifest)
    digest = manifest.pop("sha256")
    assert (
        digest
        == hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()
    )
    assert all(
        c["source_sha256"] == sha256(turn_source[1])
        for c in first.manifest["clips"]
    )
    with turn_source[1].open("ab") as stream:
        stream.write(b"changed source")
    with pytest.raises(ValueError, match="pinned catalog"):
        library(turn_source)


@pytest.mark.parametrize(
    "segments",
    [
        [],
        [("a", 1.5, 20)],
        [("a", -1, 20)],
        [("a", 10, 15)],
        [("a", 10, 201)],
        [("", 10, 30)],
        [("same", 10, 30), ("same", 30, 60)],
    ],
)
def test_invalid_turn_crop_contract_rejected(turn_source, segments):
    with pytest.raises(ValueError):
        library(turn_source, segments)


def test_full_sequence_keeps_unwrapped_signed_angle(turn_source):
    result = library(turn_source)
    assert result.names == ["turn_full"]
    assert result.clips[0]["source_net_yaw_deg"] == pytest.approx(80)
    assert result.clips[0]["source_yaw_range_deg"] == pytest.approx(80)


def test_initial_yaw_normalization_is_rigid_covariant_for_all_world_arrays(
    turn_source,
):
    segments = [("turn", 20, 80)]
    raw = library(turn_source, segments, normalize=False)
    normalized = library(turn_source, segments)
    before, after = raw.clips[0], normalized.clips[0]
    angle = before["guard_heading_world"]
    rotation = Rotation.from_euler("z", -angle)
    for key in turn_reference.TENSOR_FIELDS:
        source = before[key].numpy()
        actual = after[key].numpy()
        if key.startswith("joint"):
            np.testing.assert_array_equal(actual, source)
        elif "quaternion" in key:
            quaternions = Rotation.from_quat(
                source.reshape(-1, 4)[:, [1, 2, 3, 0]]
            )
            expected = (
                (rotation * quaternions)
                .as_quat()[:, [3, 0, 1, 2]]
                .reshape(source.shape)
            )
            np.testing.assert_allclose(
                np.abs((actual * expected).sum(-1)), 1, atol=1e-7
            )
        else:
            expected = rotation.apply(source.reshape(-1, 3)).reshape(
                source.shape
            )
            np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert after["guard_heading_world"] == pytest.approx(0, abs=1e-12)
    assert after["original_initial_pelvis_yaw_rad"] == pytest.approx(angle)
    assert after["world_yaw_rotation_rad"] == pytest.approx(-angle)
    assert after["source_net_yaw_deg"] == pytest.approx(
        before["source_net_yaw_deg"]
    )
    assert after["guard_axis_calibrated"] is False
    assert normalized.manifest["normalize_initial_yaw"] is True
    assert normalized.manifest["relative_turn_angle_modified"] is False
    assert normalized.manifest["sha256"] != raw.manifest["sha256"]
