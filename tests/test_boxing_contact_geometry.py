"""Native contact qualification matches evaluator geom semantics."""

from types import SimpleNamespace

import numpy as np
import pytest

from holoboxing.evaluation.runtime.contact_geometry import (
    opponent_target_semantic,
    strike_geometry,
)
from holoboxing.training.envs.official_referee import contact_records


def model_for(attacking, target="torso_link", mesh=None):
    return SimpleNamespace(
        geom_bodyid=np.arange(2),
        geom_type=np.array([2, 7 if mesh else 2]),
        geom_dataid=np.array([-1, 0]),
        body=lambda i: SimpleNamespace(
            name=("p1_" + attacking, "p2_" + target)[i]
        ),
        mesh=lambda i: SimpleNamespace(name="p2_" + mesh),
    )


def contacts(model):
    return contact_records(
        model,
        np.zeros(1, dtype=int),
        np.array([[0, 1]]),
        np.zeros((1, 3)),
        np.array([[300.0, 0, 0, 0, 0, 0]]),
        np.array([[1.0, 0, 0]]),
        np.zeros(1),
        1,
        1,
    )[0]


@pytest.mark.parametrize("name", ["left_wrist_yaw_link", "left_rubber_hand"])
def test_exact_official_fists_are_accepted(name):
    model = model_for(name)
    assert strike_geometry(model, 0) == (0, "left_hand")
    assert contacts(model)[0]["attacking_body"] == "left_hand"


@pytest.mark.parametrize(
    "name",
    [
        "left_wrist_roll_link",
        "left_wrist_pitch_link",
        "left_elbow_link",
        "left_ankle_roll_link",
        "pretend_left_glove",
    ],
)
def test_unqualified_wrist_or_foot_cannot_be_a_scoring_fist(name):
    assert contacts(model_for(name)) == []


@pytest.mark.parametrize(
    "body,mesh,semantic",
    [
        ("torso_link", None, "torso"),
        ("pelvis", None, "pelvis"),
        ("logo_link", None, "chest"),
        ("head_link", None, "head"),
        ("torso_link", "head_link", "head"),
        ("torso_link", "logo_link", "chest"),
        ("pelvis", "pelvis_contour_collision", "pelvis"),
    ],
)
def test_target_body_and_mesh_overrides_match_evaluator(body, mesh, semantic):
    model = model_for("left_rubber_hand", body, mesh)
    assert opponent_target_semantic(model, 0, 1) == (1, body, semantic)
    assert opponent_target_semantic(model, 1, 1) is None
    assert contacts(model)[0]["target_body"] == semantic


def test_similar_but_unscored_target_name_does_not_count():
    assert contacts(model_for("left_rubber_hand", "pretend_chest_link")) == []
