"""Complete-clip, diagnostic and failure semantics without GPU rendering."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from holoboxing.training.skills import evaluate


class FakeEnv:
    num_envs, action_dim = 1, 30
    device, dt, entry_steps, settle_steps = "cpu", 0.02, 2, 2

    def __init__(self, *, failure=False, error=False, skill="locomotion"):
        self.skill = skill
        self.shoulder_indices, self.elbow_indices = [1, 2, 3, 4, 5, 6], [7, 8]
        self.library = SimpleNamespace(
            clips=[{"name": "test", "frames": 6}],
            manifest={"synthetic_test": True, "skill": skill},
        )
        self.recipe = {"synthetic_test": True}
        self.physics = SimpleNamespace(
            native=[SimpleNamespace(qpos=np.zeros(36), qvel=np.zeros(35))]
        )
        self.resets, self.actions = [], []
        self.failure, self.error = failure, error

    def start(self, clip_ids, **kwargs):
        self.resets.append((clip_ids, kwargs))
        self.frame = torch.tensor(
            [-self.entry_steps if kwargs["neutral"] else 0]
        )
        self.steps = 0
        self._update()
        return torch.zeros(1, 4)

    def _update(self):
        q = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
        self._state = {
            "root_position": torch.tensor([[self.steps * 0.01, 0.0, 0.8]]),
            "root_quaternion": q[:, 0],
            "root_linear_velocity": torch.tensor([[0.5, 0.0, 0.0]]),
            "root_angular_velocity": torch.zeros(1, 3),
            "joint_position": torch.zeros(1, 29),
            "joint_velocity": torch.zeros(1, 29),
            "body_position": torch.zeros(1, 30, 3),
            "body_quaternion": q.repeat(1, 30, 1),
            "body_linear_velocity": torch.zeros(1, 30, 3),
            "body_angular_velocity": torch.zeros(1, 30, 3),
            "point_position": torch.zeros(1, 5, 3),
            "point_quaternion": q.repeat(1, 5, 1),
            "point_velocity": torch.zeros(1, 5, 3),
            "foot_contact": torch.tensor([[True, False]]),
            "fallen": torch.tensor([self.failure and self.steps >= 3]),
        }
        self._state["point_velocity"][0, 3, 0] = 0.2

    def reference(self):
        return {
            key: value[:, None].clone()
            for key, value in self._state.items()
            if key not in {"fallen", "foot_contact"}
        }

    def step(self, action):
        self.actions.append(action.clone())
        if self.error and self.steps == 2:
            raise RuntimeError("synthetic solver failure")
        self.steps += 1
        self.frame += 1
        self._update()
        failed = self._state["fallen"]
        completed = (self.frame >= 7) & ~failed
        return (
            torch.ones(1, 4),
            torch.zeros(1),
            completed | failed,
            {"completed": completed, "failed": failed},
        )


class FakeTeacher:
    training = True

    def eval(self):
        self.training = False

    def train(self, state):
        self.training = state

    def act(self, observation, deterministic):
        assert deterministic is True and self.training is False
        return torch.ones(1, 30) * 0.25


@pytest.mark.parametrize(
    "initialization,steps", [("neutral", 9), ("reference", 7)]
)
def test_baseline_runs_full_motion_without_midclip_resets(
    tmp_path, initialization, steps
):
    env = FakeEnv()
    report = evaluate.evaluate_skills(
        env,
        None,
        tmp_path / "run",
        record=False,
        initialization=initialization,
    )
    assert len(env.resets) == 1
    assert env.resets[0][1]["random_phase"] is False
    assert env.steps == steps
    assert all(torch.count_nonzero(action) == 0 for action in env.actions)
    assert report["state"] == "completed"
    assert report["kind"] == "frozen_lower_reference_baseline"
    assert report["form_certified"] is False
    item = report["clips"][0]
    assert item["frames_executed"] == steps
    assert item["completed_clip_and_settle"] is True
    assert item["metrics"]["contact_conditioned_foot_speed_mps"][0][
        "mean"
    ] == pytest.approx(0.2)
    assert (
        item["metrics"]["contact_conditioned_foot_speed_mps"][1]["samples"]
        == 0
    )
    with np.load(tmp_path / "run/00-test/states.npz") as data:
        assert len(data["qpos"]) == steps + 1
        assert data["source_frame"][-1] == 7
        assert np.diff(data["source_frame"]).tolist() == [1] * steps
        assert (
            "actual_body_quaternion" in data
            and "reference_joint_velocity" in data
        )


def test_teacher_is_deterministic_and_training_mode_restored(tmp_path):
    env, teacher = FakeEnv(), FakeTeacher()
    report = evaluate.evaluate_skills(
        env, teacher, tmp_path / "run", record=False
    )
    assert teacher.training is True
    assert all(torch.all(action == 0.25) for action in env.actions)
    assert report["kind"] == "learned_reference_conditioned_teacher"
    assert report["deployable_competition_policy"] is False


def test_fall_is_completed_measurement_but_not_success(tmp_path):
    env = FakeEnv(failure=True)
    report = evaluate.evaluate_skills(
        env, None, tmp_path / "run", record=False
    )
    assert env.steps == 3
    assert report["state"] == "completed"
    assert not report["all_clips_completed_without_environment_failure"]
    item = report["clips"][0]
    assert item["failure_flags"] and item["metrics"]["fallen_samples"] == 1
    assert item["successful_motion_learning_claim"] is False


def test_exception_saves_partial_evidence_and_reraises(tmp_path):
    env, teacher = FakeEnv(error=True), FakeTeacher()
    with pytest.raises(RuntimeError, match="synthetic solver failure"):
        evaluate.evaluate_skills(env, teacher, tmp_path / "run", record=False)
    assert teacher.training is True
    report = json.loads((tmp_path / "run/summary.json").read_text())
    assert report["state"] == "failed"
    assert report["clips"][0]["error"]["type"] == "RuntimeError"
    with np.load(tmp_path / "run/00-test/states.npz") as data:
        assert len(data["qpos"]) == 3


def test_existing_output_and_invalid_ids_are_rejected(tmp_path):
    with pytest.raises(FileExistsError):
        evaluate.evaluate_skills(FakeEnv(), None, tmp_path, record=False)
    for ids in ([], [1], [0, 0], [False]):
        with pytest.raises(ValueError, match="clip_ids"):
            evaluate.evaluate_skills(
                FakeEnv(), None, tmp_path / "new", record=False, clip_ids=ids
            )
    assert not (tmp_path / "new").exists()


def test_rotation_of_body_does_not_count_as_arm_motion():
    state = {
        "root_position": torch.zeros(1, 3),
        "root_linear_velocity": torch.tensor([[1.0, 0.0, 0.0]]),
        "root_angular_velocity": torch.tensor([[0.0, 0.0, 2.0]]),
        "point_position": torch.zeros(1, 5, 3),
        "point_velocity": torch.zeros(1, 5, 3),
    }
    state["point_position"][:, 1:3, 0] = 0.5
    state["point_velocity"][:, 1:3, :2] = torch.tensor([1.0, 1.0])
    torch.testing.assert_close(
        evaluate._arm_velocity(state), torch.zeros(1, 2, 3)
    )


@pytest.mark.parametrize("video_failure", [False, True])
def test_recording_closes_and_keeps_terminal_frame(
    tmp_path, monkeypatch, video_failure
):
    frames, closed = [], []

    class FakeVideo:
        def __init__(self, *args):
            pass

        def add(self, actual, reference, frame, seconds, metrics):
            frames.append((frame, seconds))

        def close(self):
            closed.append(True)

        def verify(self):
            assert closed
            if video_failure:
                raise RuntimeError("truncated video")
            return {"frames": len(frames)}

    monkeypatch.setattr(evaluate, "_Video", FakeVideo)
    if video_failure:
        with pytest.raises(RuntimeError, match="truncated video"):
            evaluate.evaluate_skills(FakeEnv(), None, tmp_path / "run")
    else:
        evaluate.evaluate_skills(FakeEnv(), None, tmp_path / "run")
    assert closed == [True]
    assert [frame for frame, _ in frames] == [-2, 0, 2, 4, 6, 7]
    report = json.loads((tmp_path / "run/summary.json").read_text())
    assert report["state"] == ("failed" if video_failure else "completed")
    assert (tmp_path / "run/00-test/states.npz").is_file()
    if video_failure:
        assert (
            "video_encoding_or_validation_failure"
            in report["clips"][0]["failure_flags"]
        )


def test_nonfinite_metric_cannot_mark_evaluation_complete(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        evaluate, "_metrics", lambda *_: {"invalid": np.array(float("nan"))}
    )
    with pytest.raises(RuntimeError, match="Nonfinite"):
        evaluate.evaluate_skills(
            FakeEnv(), None, tmp_path / "run", record=False
        )
    report = json.loads((tmp_path / "run/summary.json").read_text())
    assert report["state"] == "failed"
    assert report["clips"][0]["failure_flags"] == ["evaluation_exception"]


@pytest.mark.parametrize("skill", ["punch", "locomotion"])
def test_punch_form_metrics_saved_by_phase_only_for_punch(tmp_path, skill):
    class FormErrorEnv(FakeEnv):
        def reference(self):
            result = super().reference()
            result["joint_position"][..., self.shoulder_indices] += 0.2
            result["joint_position"][..., self.elbow_indices] += 0.4
            return result

    result = evaluate.evaluate_skills(
        FormErrorEnv(skill=skill), None, tmp_path / "run", record=False
    )
    metrics = result["clips"][0]["metrics"]
    names = (
        "shoulder_error_rad",
        "elbow_error_rad",
        "wrist_relative_error_m",
        "wrist_relative_rotation_rad",
        "wrist_relative_velocity_mps",
    )
    with np.load(tmp_path / "run/00-test/states.npz") as data:
        for name in names:
            assert ("metric_" + name in data) == (skill == "punch")
        for phase in ("entry", "source_motion", "settle"):
            assert ("shoulder_error_rad" in metrics[phase]) == (
                skill == "punch"
            )
            if skill == "punch":
                assert metrics[phase]["shoulder_error_rad"][
                    "mean"
                ] == pytest.approx(0.2)
                assert metrics[phase]["elbow_error_rad"][
                    "mean"
                ] == pytest.approx(0.4)
                assert all(
                    metrics[phase][name]["samples"] > 0 for name in names
                )
