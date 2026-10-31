from types import SimpleNamespace

import torch

from holoboxing.training.distance_curriculum import DistanceCurriculum
from holoboxing.training.envs.entities import SceneEntityCfg
from holoboxing.training.envs.mujoco_rewards import terms, reward_recipe
from holoboxing.training.tasks.combat.mdp.rewards import rapid_approach_reward


def test_far_reward_favors_fast_approach_not_crawl_retreat_or_opponent_motion():
    position = torch.zeros(7, 3)
    velocity = torch.zeros(7, 3)
    velocity[:, 0] = torch.tensor([0, 0.1, 0.5, 1, 2, -1, 0])
    other = SimpleNamespace(
        root_pos_w=torch.tensor([[2.0, 0, 0]]).repeat(7, 1),
        root_lin_vel_w=torch.zeros(7, 3),
    )
    other.root_lin_vel_w[-1, 0] = -3
    own = SimpleNamespace(root_pos_w=position, root_lin_vel_w=velocity)
    env = SimpleNamespace(
        scene={"robot_A": SimpleNamespace(data=own)},
        get_opponent=lambda _: SimpleNamespace(data=other),
    )
    reward = rapid_approach_reward(env, SceneEntityCfg("robot_A"))
    assert reward[0] < reward[1] < reward[2] < reward[3]
    assert reward[4] < reward[3] and reward[5] < reward[0]
    assert reward[-1] == reward[0]
    other.root_pos_w[:, 0] = 0.85
    assert not rapid_approach_reward(env, SceneEntityCfg("robot_A")).any()


def test_reset_moves_only_selected_xy_and_final_stage_leaves_reset_unchanged():
    c = DistanceCurriculum(2, "cpu")
    qpos = torch.randn(2, 14, dtype=torch.float64)
    qpos[:, :2] = torch.tensor([-1.0, 0.0])
    qpos[:, 7:9] = torch.tensor([1.0, 0.0])
    before = qpos.clone()
    c.reset(torch.tensor([0]), qpos, (0, 7))
    assert torch.linalg.vector_norm(qpos[0, :2] - qpos[0, 7:9]) == 1
    torch.testing.assert_close(
        qpos[0, :2] + qpos[0, 7:9], before[0, :2] + before[0, 7:9]
    )
    torch.testing.assert_close(qpos[1], before[1])
    torch.testing.assert_close(qpos[0, 2:7], before[0, 2:7])
    torch.testing.assert_close(qpos[0, 9:], before[0, 9:])
    c.stage = 3
    c.reset(torch.arange(2), qpos, (0, 7))
    torch.testing.assert_close(qpos[1], before[1])


def finish(c, hits=0, fallen=False, invalid=False):
    ids = torch.tensor([0])
    c.reset(ids, torch.tensor([[0.0, 0, 0, 0, 0, 0, 0, 2, 0.0]]), (0, 7))
    c.observe(
        torch.tensor([hits]),
        torch.tensor([fallen]),
        torch.tensor([True]),
        torch.tensor([invalid]),
    )


def test_no_promotion_for_no_hits_invalid_or_falling_bouts():
    for hits, fallen, invalid in [
        (0, False, False),
        (1, True, False),
        (1, False, True),
    ]:
        c = DistanceCurriculum(1, "cpu")
        for _ in range(32):
            finish(c, hits, fallen, invalid)
        assert c.stage == 0


def test_promotion_requires_completed_matches_and_survives_restart():
    c = DistanceCurriculum(1, "cpu")
    for _ in range(12):
        finish(c, 1)
    state = c.state_dict()
    restored = DistanceCurriculum(1, "cpu")
    restored.load_state_dict(state)
    for _ in range(4):
        finish(restored, 0)
    assert restored.stage == 1 and not restored.outcomes
    # Lots of hits inside an unfinished bout cannot promote the next stage.
    for _ in range(32):
        restored.observe(
            torch.ones(1),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
            torch.zeros(1, dtype=torch.bool),
        )
    assert restored.stage == 1
    # A bout that started at the old distance cannot pass the new stage.
    restored.episode_stage[0] = 0
    restored.observe(
        torch.ones(1),
        torch.zeros(1, dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        torch.zeros(1, dtype=torch.bool),
    )
    assert not restored.outcomes


def test_profile_keeps_official_rules_and_prior_profile_unchanged():
    original = {n: w for n, w, *_ in terms("robot_A", official_rules=True)}
    active = {
        n: w
        for n, w, *_ in terms(
            "robot_A", official_rules=True, rapid_approach=True
        )
    }
    assert "vel_reward" in original and "rapid_approach_reward" not in original
    assert "vel_reward" not in active and active["rapid_approach_reward"] == 2
    assert active["facing_reward"] < original["facing_reward"]
    assert "offensive_hit_reward_punch" not in active
    assert (
        reward_recipe(official_rules=True, rapid_approach=True)["version"]
        == "rapid_approach_v1"
    )
