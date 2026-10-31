"""Diagnostic gates must not confuse program completion with scored behavior."""

import pytest
import torch

from holoboxing.training.evaluate_cpu import (
    CONTACT_COUNTS,
    IMPULSE_PEAK,
    run_match,
    summarize,
)
from holoboxing.training.tasks.combat.config.sandbag import AGENTS


class MatchFixture:
    """Simulate env metric snapshots after an automatic terminal reset."""

    def __init__(
        self, *, hits=0, knockout=False, invalid=False, complete=True
    ):
        self.hits = hits
        self.knockout = knockout
        self.invalid = invalid
        self.complete = complete
        self.metrics, self.reward_terms = {}, {}

    def reset(self):
        return {
            agent: {"task_policy": torch.zeros(1, 1097)} for agent in AGENTS
        }

    def step(self, actions):
        for role, agent in enumerate(AGENTS):
            own = role == 0
            self.metrics[agent] = {
                "official_score": torch.tensor(
                    [self.hits + 1000 * self.knockout if own else 0]
                ),
                "official_hits": torch.tensor([self.hits if own else 0]),
                "official_ko": torch.tensor([int(self.knockout and own)]),
                "official_match_over": torch.tensor([self.complete]),
                "fallen": torch.tensor([self.knockout and not own]),
                IMPULSE_PEAK: torch.tensor(
                    [5.0 if own and self.hits else 0.0]
                ),
                **{name: torch.tensor([int(own)]) for name in CONTACT_COUNTS},
            }
            winner = self.hits or self.knockout
            self.reward_terms[agent] = {
                "official_outcome": torch.tensor(
                    [1 if own and winner else -1 if winner else 0]
                )
            }
        # No live referee is exposed: reading it after step would use reset data.
        return (
            self.reset(),
            {},
            torch.tensor([self.complete or self.invalid]),
            {"invalid_envs": torch.tensor([self.invalid])},
        )


def policy(obs):
    assert not torch.is_grad_enabled()
    return torch.zeros(obs.shape[0], 225)


@pytest.mark.parametrize(
    "hits,knockout,gate",
    [(0, False, False), (0, True, False), (1, False, True)],
)
def test_scored_attack_gate_requires_hits_not_completion_or_ko(
    hits, knockout, gate
):
    match = run_match(
        MatchFixture(hits=hits, knockout=knockout),
        (policy, policy),
        candidate_side=0,
        max_steps=3,
    )
    assert match["steps"] == 1
    assert match["score"][0] == hits + 1000 * knockout
    assert match["hits"][0] == hits
    summary = summarize([match])
    assert summary["scored_attack_gate_passed"] is gate
    assert summary["boxing_quality_verified"] is False


def test_partial_lead_is_not_a_win_or_pass():
    match = run_match(
        MatchFixture(hits=1, complete=False),
        (policy, policy),
        candidate_side=0,
        max_steps=2,
    )
    assert match["candidate_win"] is None
    assert match["end_reason"] == "STEP_LIMIT"
    assert match["hits"][0] == 2
    assert summarize([match])["scored_attack_gate_passed"] is False


def test_invalid_step_cannot_count_success_and_stops_immediately():
    match = run_match(
        MatchFixture(hits=1, invalid=True),
        (policy, policy),
        candidate_side=0,
        max_steps=3,
    )
    assert match["steps"] == 1
    assert match["hits"] == [0, 0]
    assert match["end_reason"] == "INVALID"
    assert not match["complete_match"]
    assert not summarize([match])["scored_attack_gate_passed"]


def test_gate_checks_candidate_side_and_every_match():
    red = run_match(
        MatchFixture(hits=1), (policy, policy), candidate_side=0, max_steps=3
    )
    blue = run_match(
        MatchFixture(hits=1), (policy, policy), candidate_side=1, max_steps=3
    )
    assert summarize([red])["scored_attack_gate_passed"]
    assert not summarize([red, blue])["scored_attack_gate_passed"]
