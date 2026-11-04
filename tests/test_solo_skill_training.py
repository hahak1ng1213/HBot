"""Standalone PPO contracts with a mutable-buffer CPU environment."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holoboxing.training.skills import train
from holoboxing.training.skills.actor import SkillActor, load_teacher


class FakeEnv:
    num_envs = 2
    device = torch.device("cpu")
    observation_dim = 4

    def __init__(self, bad_reward=False):
        self.rng = torch.Generator().manual_seed(7)
        self.buffer = torch.zeros((2, 4))
        self.ages = torch.zeros(2, dtype=torch.long)
        self.generation = torch.zeros(2)
        self.library = SimpleNamespace(
            clips=[0, 1], manifest={"source": "fake"}
        )
        self.recipe = {"task": "fake-single-robot-tracking"}
        self.physics = SimpleNamespace(
            metadata={"lower": "frozen"}, failure_state=None
        )
        self.bad_reward = bad_reward
        self.transitions = {}
        self.resets = []

    def start(self, clips, *, ids, **kwargs):
        assert len(clips) == len(ids)
        self.resets.append(ids.tolist())
        self.ages[ids] = 0
        self.generation[ids] += 1
        self.buffer[ids, 0] = self.generation[ids]
        self.buffer[ids, 1] = 0
        self.buffer[ids, 2] = ids.float()
        self.buffer[ids, 3] = 1

    def get_observations(self):
        return self.buffer

    def step(self, action):
        for index in range(2):
            self.transitions[tuple(action[index].tolist())] = self.buffer[
                index
            ].clone()
        self.ages += 1
        self.buffer[:, 1] = self.ages.float()
        done = self.ages >= torch.tensor([2, 5])
        reward = 1 - action.tanh().square().mean(-1)
        if self.bad_reward:
            reward[0] = float("nan")
            self.physics.failure_state = {
                "qpos": np.array([np.nan, 1.0]),
                "qvel": torch.tensor([float("inf")]),
            }
        return (
            self.buffer,
            reward,
            done,
            {
                "completed": done,
                "failed": torch.zeros(2, dtype=torch.bool),
                "metrics": {"error": torch.full((2,), 0.1)},
            },
        )


def arguments(**overrides):
    return SimpleNamespace(
        **{
            "num_envs": 2,
            "updates": 2,
            "rollout_steps": 4,
            "save_every": 1,
            "learning_rate": 0.0003,
            "seed": 7,
            "skill": "punch",
            **overrides,
        }
    )


def test_raw_gaussian_likelihood_precedes_environment_tanh():
    model = SkillActor(4, hidden_dims=(8,))
    observation = torch.zeros((2, 4))
    raw = torch.full((2, 30), 2.0)
    log_probability, _, _ = model.evaluate(observation, raw)
    expected = (
        torch.distributions.Normal(torch.zeros_like(raw), model.log_std.exp())
        .log_prob(raw)
        .sum(-1)
    )
    torch.testing.assert_close(log_probability, expected)
    transformed, _, _ = model.evaluate(observation, raw.tanh())
    assert not torch.allclose(log_probability, transformed)
    assert torch.equal(
        model.act(observation, deterministic=True), torch.zeros_like(raw)
    )


def test_gae_cuts_terminal_resets_and_bootstraps_live_rows():
    reward = torch.tensor([[1.0, 1.0], [100.0, 2.0]])
    values = torch.zeros_like(reward)
    done = torch.tensor([[True, False], [False, False]])
    advantage, returns = train.advantages(
        reward, done, values, torch.tensor([10.0, 4.0]), gamma=0.9, lam=1.0
    )
    torch.testing.assert_close(
        advantage, torch.tensor([[1.0, 6.04], [109.0, 5.6]])
    )
    torch.testing.assert_close(returns, advantage)


def test_two_updates_preserve_raw_action_observation_pairs_and_save(
    tmp_path, monkeypatch
):
    torch.set_num_threads(1)
    env = FakeEnv()
    checked = []

    class CheckingActor(SkillActor):
        def __init__(self, observation_dim):
            super().__init__(observation_dim, hidden_dims=(8,))

        def evaluate(self, observations, raw_actions):
            if torch.is_grad_enabled():
                for observation, action in zip(
                    observations, raw_actions, strict=True
                ):
                    expected = env.transitions[tuple(action.tolist())]
                    torch.testing.assert_close(observation, expected)
                    checked.append(True)
            return super().evaluate(observations, raw_actions)

    monkeypatch.setattr(train, "SkillActor", CheckingActor)
    output = tmp_path / "fresh"
    result = train.train_teacher(env, output, arguments())
    assert checked
    assert result["state"] == "completed"
    assert result["skill_quality"] == "requires_continuous_physical_evaluation"
    assert [0] in env.resets  # first world terminates independently
    assert (output / "model_000001.pt").is_file()
    assert (output / "model_000002.pt").is_file()
    loaded, saved = load_teacher(result["checkpoint"])
    assert saved["update"] == 2
    assert saved["reference_manifest"] == env.library.manifest
    assert saved["physics_metadata"] == env.physics.metadata
    assert saved["implementation_sources"]
    assert loaded.action_dim == 30
    assert not saved["is_competition_submission"]
    before = (output / "recipe.json").read_bytes()
    with pytest.raises(FileExistsError):
        train.train_teacher(env, output, arguments())
    assert (output / "recipe.json").read_bytes() == before


def test_nonfinite_rollout_keeps_original_error_and_serializable_evidence(
    tmp_path,
):
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Nonfinite teacher reward"):
        train.train_teacher(FakeEnv(bad_reward=True), output, arguments())
    failure = json.loads((output / "failure.json").read_text())
    assert failure["physics_failure"]["qpos"]["values"] == ["nan", 1.0]
    assert failure["physics_failure"]["qvel"]["values"] == ["inf"]
    assert (
        json.loads((output / "status.json").read_text())["state"] == "failed"
    )
    assert not (output / "training-result.json").exists()


def cli(tmp_path):
    return [
        "--assets",
        str(tmp_path),
        "--lower",
        str(tmp_path),
        "--source",
        str(tmp_path),
        "--skill",
        "punch",
        "--output",
        str(tmp_path / "out"),
    ]


@pytest.mark.parametrize(
    "extra",
    [
        ["--mode", "evaluate"],
        ["--learning-rate", "nan"],
        ["--updates", "0"],
        ["--num-envs", "-1"],
        ["--locomotion-mode", "unknown"],
    ],
)
def test_bad_cli_rejected_before_physics(tmp_path, extra):
    with pytest.raises(SystemExit):
        train.parse_arguments(cli(tmp_path) + extra)


def test_baseline_forbids_checkpoint(tmp_path):
    with pytest.raises(SystemExit):
        train.parse_arguments(
            cli(tmp_path)
            + ["--mode", "baseline", "--checkpoint", str(tmp_path)]
        )


@pytest.mark.parametrize("locomotion_mode", ["directional", "full", "forward"])
def test_baseline_calls_evaluator_with_no_learned_actor(
    tmp_path, monkeypatch, locomotion_mode
):
    closed = []
    seen = []
    physics = SimpleNamespace(
        host=None,
        joint_names=[],
        body_names=[],
        metadata={},
        failure_state=None,
        close=lambda: closed.append(True),
    )
    library = SimpleNamespace(manifest={})
    env = SimpleNamespace(recipe={}, observation_dim=4)
    resources = SimpleNamespace(
        scene=tmp_path / "scene.xml", initial_state=tmp_path / "initial.npz"
    )

    def construct_physics(assets, *_):
        # resolve_resources returns ResourceTree, while physical consumers
        # require the resource tree wrapped under assets.resources.
        assert assets.resources is resources
        assert assets.resources.scene == tmp_path / "scene.xml"
        assert assets.resources.initial_state == tmp_path / "initial.npz"
        return physics

    def evaluate(actual_env, teacher, output, **kwargs):
        assert actual_env is env
        assert teacher is None
        seen.append(kwargs)
        return {"policy": "reference_baseline", "learned_actor": False}

    def construct_library(*args, **kwargs):
        assert kwargs["locomotion_mode"] == locomotion_mode
        return library

    replacements = {
        "resources": SimpleNamespace(
            resolve_resources=lambda *_: (resources, None, None)
        ),
        "skills.physics": SimpleNamespace(SoloPhysics=construct_physics),
        "skills.reference": SimpleNamespace(MotionLibrary=construct_library),
        "skills.env": SimpleNamespace(SkillTrackingEnv=lambda *a, **k: env),
        "skills.evaluate": SimpleNamespace(evaluate_skills=evaluate),
    }
    for name, module in replacements.items():
        monkeypatch.setitem(sys.modules, "holoboxing.training." + name, module)
    train.main(
        cli(tmp_path)
        + [
            "--mode",
            "baseline",
            "--no-video",
            "--locomotion-mode",
            locomotion_mode,
        ]
    )
    assert closed == [True]
    assert seen == [{"record": False, "initialization": "neutral"}]


def test_source_mutation_and_nan_checkpoint_refused(tmp_path):
    env = FakeEnv()
    model = SkillActor(4, hidden_dims=(8,))
    optimizer = torch.optim.Adam(model.parameters())
    saved = train.checkpoint(env, model, optimizer, 1, arguments())
    saved["implementation_sources"]["env.py"] = "bad"
    checkpoint = tmp_path / "wrong-source.pt"
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="implementation differs"):
        load_teacher(checkpoint)
    saved = train.checkpoint(env, model, optimizer, 1, arguments())
    saved["model"]["log_std"][0] = float("nan")
    torch.save(saved, checkpoint)
    with pytest.raises(ValueError, match="Nonfinite teacher"):
        load_teacher(checkpoint)
