"""Directed-punch feedback survives official scoring; motion alone cannot score."""

from types import SimpleNamespace

import torch

from holoboxing.training.envs.mujoco_rewards import (
    OFFICIAL_REPLACED_TERMS,
    reward_recipe,
    terms,
)


def test_official_recipe_preserves_dense_attack_and_only_replaces_score_proxies():
    legacy = {name for name, *_ in terms("robot_A")}
    official = {name for name, *_ in terms("robot_A", official_rules=True)}
    assert legacy - official == OFFICIAL_REPLACED_TERMS
    assert {
        "active_hit_shaping_reward",
        "complete_punch_reward",
        "vel_reward",
    } <= official
    assert (
        reward_recipe(official_rules=True)["shaping_weights"]["robot_A"][
            "active_hit_shaping_reward"
        ]
        == 1.5
    )


def test_selected_attack_reward_prefers_directed_punch_not_body_rush():
    names = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    # Rows: stationary, punch toward torso, retract, whole-body translation.
    position = torch.zeros(4, 3, 3)
    position[:, 1:, 0] = 0.5
    velocity = torch.zeros(4, 3, 6)
    velocity[1, 1:, 0] = 1.5
    velocity[2, 1:, 0] = -1.5
    velocity[3, :, 0] = 1.5
    own = SimpleNamespace(
        body_names=names,
        data=SimpleNamespace(body_pos_w=position, body_vel_w=velocity),
    )
    opponent_position = torch.zeros_like(position)
    opponent_position[:, :, 0] = 1.0
    opponent = SimpleNamespace(
        body_names=names, data=SimpleNamespace(body_pos_w=opponent_position)
    )
    env = SimpleNamespace(
        scene={"robot_A": own}, get_opponent=lambda _: opponent
    )
    _, weight, fn, params = next(
        term
        for term in terms("robot_A", official_rules=True)
        if term[0] == "active_hit_shaping_reward"
    )
    reward = weight * fn(env, **params) * 0.02
    assert reward[1] > 100 * reward[0]
    assert reward[1] > reward[2]
    torch.testing.assert_close(reward[0], reward[3])
    # No contact/collision evidence was needed for this *auxiliary* feedback.
    assert torch.isfinite(reward).all()
