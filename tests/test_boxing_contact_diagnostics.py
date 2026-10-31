"""Measured contact evidence stays separate from official scores."""

import numpy as np
import pytest

from holoboxing.training.envs.official_referee import OfficialReferees


def contact(force=250.0, actor=0, hand="left", **extra):
    return (
        dict(
            attacker=actor,
            defender=1 - actor,
            attacking_body=hand + "_hand",
            target_body="head",
            peak_force=force,
            closing_speed=0.0,
            normal_force_n=force,
            sample_dt_s=0.005,
            position=[0.0, 0.0, 1.0],
            physical_contact=True,
        )
        | extra
    )


def frame(ref, contacts):
    for _ in range(4):
        ref.sample(contacts)
    return ref.observe(
        np.zeros((len(contacts), 2, 3)),
        np.zeros((len(contacts), 2), dtype=bool),
    )


def test_no_contacts_and_subthreshold_normal_impulse_are_distinguishable():
    ref = OfficialReferees(2)
    _, delta = frame(ref, [[], [contact(force=100, peak_force=10000)]])
    assert not delta.any()
    metrics = ref.contact_metrics()
    assert metrics["official_legal_contact_samples"].tolist() == [
        [0, 0],
        [4, 0],
    ]
    assert metrics["official_subthreshold_contact_samples"].tolist() == [
        [0, 0],
        [4, 0],
    ]
    assert metrics["official_contact_impulse_max_ns"][1, 0] == pytest.approx(
        2.0
    )
    assert not metrics["official_cooldown_contact_samples"].any()


def test_multiple_contact_points_count_one_hand_sample_and_hit():
    ref = OfficialReferees(1)
    rows = [contact(125, pair_id=0), contact(125, pair_id=1)]
    _, delta = frame(ref, [rows])
    assert delta.tolist() == [[1, 0]]
    metrics = ref.contact_metrics()
    assert metrics["official_legal_contact_samples"].tolist() == [[4, 0]]
    assert metrics["official_subthreshold_contact_samples"].tolist() == [
        [3, 0]
    ]
    assert metrics["official_contact_impulse_max_ns"].tolist() == [[5, 0]]
    # A second physical point at the scored substep must not look like a
    # separate cooldown rejection while the same hand scores a hit.
    assert not metrics["official_cooldown_contact_samples"].any()


def test_cooldown_diagnostics_follow_consumed_scoring_window():
    ref = OfficialReferees(1)
    frame(ref, [[contact()]])
    _, delta = frame(ref, [[contact()]])
    assert not delta.any()
    metrics = ref.contact_metrics()
    assert metrics["official_legal_contact_samples"].tolist() == [[4, 0]]
    assert metrics["official_subthreshold_contact_samples"].tolist() == [
        [3, 0]
    ]
    assert metrics["official_cooldown_contact_samples"].tolist() == [[1, 0]]
    assert metrics["official_contact_impulse_max_ns"].tolist() == [[5, 0]]


def test_impulse_peak_retains_earlier_substep_and_expires_between_frames():
    ref = OfficialReferees(1)
    ref.sample([[contact(1200)]])
    for _ in range(3):
        ref.sample([[]])
    ref.observe(np.zeros((1, 2, 3)), np.zeros((1, 2), dtype=bool))
    assert ref.matches[0].score.tolist() == [1, 0]
    # Scoring consumes the impulse, so inspecting only hand_gate.window at
    # the end of the frame would incorrectly report zero here.
    assert not ref.matches[0].hand_gate.window.any()
    metrics = ref.contact_metrics()
    assert metrics["official_contact_impulse_max_ns"].tolist() == [[6, 0]]
    frame(ref, [[]])
    assert all(not value.any() for value in ref.contact_metrics().values())


def test_both_fighters_hands_worlds_and_reset_are_independent():
    ref = OfficialReferees(2)
    _, delta = frame(
        ref, [[contact(), contact(hand="right")], [contact(actor=1)]]
    )
    assert delta.tolist() == [[2, 0], [0, 1]]
    metrics = ref.contact_metrics()
    assert metrics["official_legal_contact_samples"].tolist() == [
        [8, 0],
        [0, 4],
    ]
    ref.reset([0])
    assert ref.contact_metrics()[
        "official_legal_contact_samples"
    ].tolist() == [[0, 0], [0, 4]]
    # Report snapshots must survive a terminal-world reset by the environment.
    assert metrics["official_legal_contact_samples"].tolist() == [
        [8, 0],
        [0, 4],
    ]


def test_illegal_and_nonphysical_contacts_are_not_legal_contact_samples():
    ref = OfficialReferees(1)
    frame(
        ref,
        [
            [
                contact(attacking_body="left_ankle_roll_link"),
                contact(physical_contact=False),
                contact(target_body="left_knee_link"),
            ]
        ],
    )
    assert all(not value.any() for value in ref.contact_metrics().values())
    assert ref.matches[0].score.tolist() == [0, 0]
