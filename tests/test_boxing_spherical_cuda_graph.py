"""Real CUDA capture/replay must preserve spherical actor outputs and gradients."""

import copy
from dataclasses import replace

import pytest
import torch

from holoboxing.training.motion_prior import MotionPrior, PriorConfig, state_sha256
from holoboxing.training.spherical_actor import KIND
from holoboxing.training.spherical_rl import LatentMeanActor
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("match_context", [False, True])
@pytest.mark.parametrize("mode", ["current_state", "history10"])
def test_spherical_actor_cuda_capture_replay_matches_eager(match_context, mode):
    torch.manual_seed(43)
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    actor = LatentMeanActor(
        prior.decoder,
        replace(TrainConfig(), hidden_dims=(16,)),
        dict(
            kind=KIND,
            decoder_sha256=state_sha256(prior.decoder),
            opponent_condition=True,
            self_observation_mode=mode,
            match_context=match_context,
        ),
    ).cuda()
    with torch.no_grad():
        actor.context_policy.weight.normal_(std=0.01)
    eager = copy.deepcopy(actor)
    assert "context_scale" not in actor.state_dict()
    actor.load_state_dict(eager.state_dict(), strict=True)
    inputs = torch.randn(4, 1097, device="cuda")
    graphed = torch.cuda.make_graphed_callables(actor, (inputs,))
    for _ in range(2):
        inputs = torch.randn_like(inputs)
        weights = torch.randn(4, 32, device="cuda")
        actor.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
        actual, expected = graphed(inputs), eager(inputs)
        torch.testing.assert_close(actual, expected)
        (actual * weights).sum().backward()
        (expected * weights).sum().backward()
        for p, q in zip(actor.parameters(), eager.parameters(), strict=True):
            if p.requires_grad:
                torch.testing.assert_close(p.grad, q.grad)
