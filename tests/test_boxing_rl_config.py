"""Saved architectures and explicit exploration settings survive RL launches."""

import json
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
import torch

from holoboxing.training import train_rl
from holoboxing.training.motion_prior import (
    MotionPrior,
    PriorConfig,
    state_sha256,
)
from holoboxing.training.spherical_actor import (
    KIND,
    SphericalActor,
    actor_from_state,
)
from holoboxing.training.spherical_rl import FORMAT, SphericalActorCritic
from holoboxing.training.tasks.combat.config.sandbag import TrainConfig


def saved_actor():
    config = replace(
        TrainConfig(),
        num_envs=4,
        hidden_dims=(16, 8),
        disc_hidden_dims=(8,),
        horizon_rho=0,
        init_noise_std=0.13,
        seed=73,
    )
    prior = MotionPrior(
        torch.zeros(225), torch.ones(225), PriorConfig(hidden_dims=(16,))
    )
    actor = SphericalActor(
        prior.decoder,
        config,
        dict(
            kind=KIND,
            prior_config=asdict(prior.config),
            decoder_sha256=state_sha256(prior.decoder),
            opponent_condition=True,
            self_observation_mode="current_state",
        ),
    )
    return config, actor


@pytest.mark.parametrize("mode,noise", [("standing", 0.1), ("selfplay", 0.2)])
def test_new_stage_reloads_compact_actor_and_resets_exploration(mode, noise):
    old_config, actor = saved_actor()
    args = SimpleNamespace(
        config=None,
        resume=None,
        policy_checkpoint="warmup.pt",
        supervised_checkpoint=None,
        training_mode=mode,
        num_envs=None,
        seed=None,
    )
    config = train_rl.resolve_training_config(
        args, {"format": FORMAT, "config": old_config.to_dict()}
    )
    restored = actor_from_state(actor.state_dict(), actor.spec, config)
    policy = SphericalActorCritic(restored, config)
    observations = torch.randn(2, 1097)
    torch.testing.assert_close(actor(observations), restored(observations))
    torch.testing.assert_close(policy.std_ego.exp(), torch.full((32,), noise))
    assert config.num_envs == 8
    assert config.hidden_dims == (16, 8)


def test_explicit_stage_config_controls_exploration(tmp_path):
    old_config, _ = saved_actor()
    requested = replace(old_config, init_noise_std=0.07, num_envs=2)
    path = tmp_path / "custom.json"
    path.write_text(json.dumps(requested.to_dict()))
    args = SimpleNamespace(
        config=path,
        resume=None,
        policy_checkpoint="warmup.pt",
        supervised_checkpoint=None,
        training_mode="selfplay",
        num_envs=None,
        seed=None,
    )
    assert (
        train_rl.resolve_training_config(
            args, {"format": FORMAT, "config": old_config.to_dict()}
        )
        == requested
    )


def test_resume_cli_inherits_saved_settings_before_building_environment(
    tmp_path, monkeypatch
):
    config, _ = saved_actor()
    checkpoint = tmp_path / "model.pt"
    torch.save({"format": FORMAT, "config": config.to_dict()}, checkpoint)
    captured = {}

    def fake_train(args, resolved, **kwargs):
        captured.update(
            config=resolved, args=args, checkpoint=kwargs["checkpoint"]
        )

    # Exercise CLI parsing and checkpoint/config loading without simulation.
    monkeypatch.setitem(
        sys.modules,
        "holoboxing.training.scripts.train_mujoco",
        SimpleNamespace(train=fake_train),
    )
    train_rl.main(
        [
            "--assets",
            str(tmp_path),
            "--lower",
            str(tmp_path),
            "--amp-motions",
            str(tmp_path),
            "--resume",
            str(checkpoint),
            "--training-mode",
            "selfplay",
            "--official-rules",
            "--output",
            str(tmp_path / "resumed"),
            "--updates",
            "1",
            "--device",
            "cpu",
        ]
    )
    assert captured["config"] == config
    assert captured["args"].num_envs == config.num_envs
    assert captured["args"].seed == config.seed
    assert captured["checkpoint"]["config"] == config.to_dict()


def test_fresh_prior_defaults_allow_explicit_world_count_and_seed():
    from holoboxing.training.motion_prior import FORMAT as PRIOR_FORMAT

    args = SimpleNamespace(
        config=None,
        resume=None,
        policy_checkpoint=None,
        supervised_checkpoint=None,
        training_mode="selfplay",
        num_envs=2,
        seed=17,
    )
    config = train_rl.resolve_training_config(args, {"format": PRIOR_FORMAT})
    assert config.num_envs == 2 and config.seed == 17
    assert config.horizon_rho == 0 and config.init_noise_std == 0.2
