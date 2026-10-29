"""Information ablations and frozen opponents, without claiming match quality."""

import copy

import pytest
import torch

from test_boxing_training_stages import make_runner, small_policy
from holoboxing.training.spherical_actor import actor_from_state


def test_prior_only_robot_statistics_do_not_leak_heldout_or_change_sl_default(
    tmp_path,
):
    from test_boxing_training_splits import population
    from holoboxing.training.supervised.data import Records
    from holoboxing.training.supervised.statistics import fit_statistics

    manifest = population(tmp_path)
    robot = Records(manifest, dataset="robostriker")
    _, normalizer = fit_statistics(robot)
    assert robot.sample_count() == 6
    assert normalizer.self_mean.eq(5).all()
    assert robot.source_selection["dataset"] == "robostriker"
    human = Records(manifest)
    assert human.source_selection["dataset"] == "interhuman"
    assert human.sample_count() == 12


def test_current_state_masks_history_and_disabled_context():
    _, policy = small_policy()
    actor = policy.taskpolicy_ego
    actor.spec.update(
        self_observation_mode="current_state", match_context=False
    )
    obs = torch.randn(2, 1097)
    changed = obs.clone()
    changed[:, :972] += 10
    changed[:, 1035:1080] += 10
    changed[:, -2:] += 10
    torch.testing.assert_close(actor(obs), actor(changed), rtol=0, atol=0)
    actor(obs).sum().backward()
    assert actor.context_policy.weight.grad is not None
    assert torch.count_nonzero(actor.context_policy.weight.grad) == 0
    policy.check_frozen()


def test_context_zero_initialization_then_receives_gradient():
    _, policy = small_policy()
    actor = policy.taskpolicy_ego
    obs = torch.randn(2, 1097)
    obs[:, -2:] = torch.tensor([60.0, 4.0])
    before = actor(obs).detach()
    actor.spec["match_context"] = True
    torch.testing.assert_close(before, actor(obs), rtol=0, atol=0)
    actor(obs).sum().backward()
    assert actor.context_policy.weight.grad.abs().sum() > 0
    with torch.no_grad():
        actor.context_policy.weight.copy_(torch.randn(32, 2))
    changed = obs.clone()
    changed[:, -2:] = 0
    assert not torch.allclose(actor(obs), actor(changed))
    policy.check_frozen()


def test_history_mode_reacts_to_existing_history():
    _, policy = small_policy()
    actor = policy.taskpolicy_ego
    actor.spec["self_observation_mode"] = "history10"
    obs = torch.randn(2, 1097)
    changed = obs.clone()
    changed[:, :972] += 5
    assert not torch.allclose(actor(obs), actor(changed))


def test_only_legacy_context_disabled_checkpoints_allow_missing_projection():
    config, policy = small_policy()
    actor = policy.taskpolicy_ego
    state = copy.deepcopy(actor.state_dict())
    del state["context_policy.weight"]
    loaded = actor_from_state(state, actor.spec, config)
    obs = torch.randn(2, 1097)
    torch.testing.assert_close(
        actor.latent(obs), loaded.latent(obs), rtol=0, atol=0
    )
    spec = {**actor.spec, "match_context": True}
    with pytest.raises(RuntimeError, match="context_policy"):
        actor_from_state(state, spec, config)


def test_external_opponent_retained_and_reproduces_after_resume(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    source = make_runner(tmp_path / "source", "standing", monkeypatch)
    source.learn(1)
    checkpoint = tmp_path / "source/model_1.pt"
    runner = make_runner(tmp_path / "target", "selfplay", monkeypatch)
    pool = runner.opponent_pool
    pool.add_checkpoint(checkpoint, runner.identity)
    digest = pool.snapshots[0]["sha256"]
    with pytest.raises(ValueError, match="Duplicate"):
        pool.add_checkpoint(checkpoint, runner.identity)
    runner.learn(5)
    assert (
        pool.snapshots[0]["external"] and pool.snapshots[0]["sha256"] == digest
    )
    assert len(pool.snapshots) == 3
    clone = make_runner(tmp_path / "resume", "selfplay", monkeypatch)
    clone.supervised_initialization = None
    clone.resume(tmp_path / "target/model_5.pt")
    pool.prepare(5)
    clone.opponent_pool.prepare(5)
    assert clone.opponent_pool.selected == pool.selected
    assert clone.opponent_pool.snapshots[0]["sha256"] == digest


def test_external_decoder_tampering_rejected(tmp_path, monkeypatch):
    source = make_runner(tmp_path / "source", "standing", monkeypatch)
    source.learn(1)
    checkpoint = tmp_path / "source/model_1.pt"
    state = torch.load(checkpoint, weights_only=False)
    key = next(
        k for k in state["policy"] if k.startswith("taskpolicy_ego.decoder.")
    )
    state["policy"][key] += 1
    torch.save(state, checkpoint)
    runner = make_runner(tmp_path / "target", "selfplay", monkeypatch)
    with pytest.raises(ValueError, match="decoder"):
        runner.opponent_pool.add_checkpoint(checkpoint, runner.identity)


@pytest.mark.parametrize(
    "mode,context",
    [("current_state", False), ("current_state", True), ("history10", True)],
)
def test_ablation_checkpoint_resume_and_onnx_parity(
    tmp_path, monkeypatch, mode, context
):
    from holoboxing.training.export import export_policy

    torch.set_num_threads(1)
    runner = make_runner(tmp_path / "run", "standing", monkeypatch)
    runner.policy.policy_spec.update(
        self_observation_mode=mode, match_context=context
    )
    for role in ("ego", "opponent"):
        getattr(runner.policy, "taskpolicy_" + role).spec.update(
            runner.policy.policy_spec
        )
    runner.identity["frozen_representation"] = copy.deepcopy(
        runner.policy.policy_spec
    )
    runner.learn(1)
    checkpoint = tmp_path / "run/model_1.pt"
    clone = make_runner(tmp_path / "resume", "standing", monkeypatch)
    clone.policy.policy_spec.update(runner.policy.policy_spec)
    for role in ("ego", "opponent"):
        getattr(clone.policy, "taskpolicy_" + role).spec.update(
            runner.policy.policy_spec
        )
    clone.identity["frozen_representation"] = copy.deepcopy(
        runner.policy.policy_spec
    )
    clone.supervised_initialization = None
    clone.resume(checkpoint)
    clone.learn(1)
    report = export_policy(checkpoint, tmp_path / "policy.onnx")
    assert report["max_abs_error"] < 1e-4
