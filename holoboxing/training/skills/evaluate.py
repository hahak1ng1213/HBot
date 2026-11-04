"""Continuous physical skill evaluation, separate from combat and learning.

Zero residual evaluates the frozen lower controller following the reference.
A teacher evaluates a learned reference-conditioned residual policy. Neither
kind is autonomous competition-policy evaluation or kinematic source replay.
"""

from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

from holoboxing.data.common import sha256
from holoboxing.training.skills.env import (
    punch_tracking_metrics,
    tracking_metrics,
)


def _array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


def _write(path, report):
    path.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"samples": 0, "mean": None, "p95": None, "max": None}
    if not np.isfinite(values).all():
        raise ValueError("Cannot report nonfinite skill metrics")
    return {
        "samples": len(values),
        "mean": np.mean(values, axis=0).tolist(),
        "p95": np.quantile(values, 0.95, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
    }


def _arm_velocity(state):
    offsets = state["point_position"][:, 1:3] - state["root_position"][:, None]
    return (
        state["point_velocity"][:, 1:3]
        - state["root_linear_velocity"][:, None]
        - torch.cross(
            state["root_angular_velocity"][:, None].expand_as(offsets),
            offsets,
            dim=-1,
        )
    )


def _metrics(actual, reference, env):
    metrics = tracking_metrics(actual, reference)
    if env.skill == "punch":
        metrics.update(
            punch_tracking_metrics(
                actual, reference, env.shoulder_indices, env.elbow_indices
            )
        )
    metrics.update(
        joint_velocity_error_radps=(
            actual["joint_velocity"] - reference["joint_velocity"]
        )
        .square()
        .mean(-1)
        .sqrt(),
        body_angular_velocity_error_radps=(
            actual["body_angular_velocity"]
            - reference["body_angular_velocity"]
        )
        .square()
        .sum(-1)
        .mean(-1)
        .sqrt(),
        wrist_arm_speed_mps=_arm_velocity(actual).norm(dim=-1),
        reference_wrist_arm_speed_mps=_arm_velocity(reference).norm(dim=-1),
        wrist_velocity_error_mps=(
            _arm_velocity(actual) - _arm_velocity(reference)
        ).norm(dim=-1),
        foot_speed_xy_mps=actual["point_velocity"][:, 3:, :2].norm(dim=-1),
    )
    return {key: _array(value[0]) for key, value in metrics.items()}


class _Video:
    """Render the live simulated robot and optional target markers only."""

    def __init__(self, physics, directory, label, initialization):
        import imageio_ffmpeg
        import mujoco
        from PIL import ImageFont

        self.mujoco = mujoco
        self.codec = imageio_ffmpeg
        self.directory = directory
        self.path = directory / "motion.mp4"
        self.label, self.initialization = label, initialization
        # Renderer and live data must use the same model instance.
        self.model, self.data = physics.models[0], physics.native[0]
        self.model.vis.global_.offwidth = 960
        self.model.vis.global_.offheight = 640
        self.renderer = mujoco.Renderer(self.model, height=640, width=960)
        self.camera = mujoco.MjvCamera()
        self.camera.distance, self.camera.azimuth, self.camera.elevation = (
            3.4,
            135,
            -20,
        )
        try:
            self.font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17
            )
        except OSError:
            self.font = ImageFont.load_default()
        self.writer = imageio_ffmpeg.write_frames(
            str(self.path),
            (960, 640),
            fps=25,
            codec="libx264",
            quality=7,
            pix_fmt_in="rgb24",
            pix_fmt_out="yuv420p",
            macro_block_size=16,
            output_params=["-threads", "1"],
        )
        try:
            self.writer.send(None)
        except BaseException:
            self.renderer.close()
            raise
        self.frames = 0
        self.picture = None

    def add(self, actual, reference, frame, seconds, metrics):
        from PIL import Image, ImageDraw

        self.camera.lookat[:] = _array(actual["root_position"][0])
        self.camera.lookat[2] = 0.8
        self.renderer.update_scene(self.data, camera=self.camera)
        scene = self.renderer.scene
        for slot, position in enumerate(
            _array(reference["point_position"][0])
        ):
            if slot == 0 or scene.ngeom >= scene.maxgeom:
                continue
            self.mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                self.mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.025, 0.025, 0.025]),
                position,
                np.eye(3).flatten(),
                np.array([1, 0.65, 0.05, 0.8], dtype=np.float32),
            )
            scene.ngeom += 1
        picture = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 960, 78), fill=(15, 18, 23))
        lines = (
            f"{self.label} | initialization: {self.initialization}",
            f"t={seconds:.2f}s  source frame={frame}  "
            f"body={float(metrics['body_position_m']):.3f}m  "
            f"joints={float(metrics['joint_error_rad']):.3f}rad",
            "LIVE PHYSICS | orange: reference wrists/ankles | no opponent",
        )
        for y, line in zip((5, 29, 53), lines, strict=True):
            draw.text((12, y), line, font=self.font, fill="white")
        self.writer.send(np.asarray(picture))
        self.frames += 1
        self.picture = picture
        if self.frames == 1:
            picture.save(self.directory / "first.png")

    def close(self):
        try:
            self.writer.close()
        finally:
            self.renderer.close()
        if self.picture is not None:
            self.picture.save(self.directory / "last.png")

    def verify(self):
        actual, seconds = self.codec.count_frames_and_secs(str(self.path))
        if actual != self.frames:
            raise RuntimeError(
                "Video frame count differs from rendered frames"
            )
        return {
            "path": str(self.path),
            "sha256": sha256(self.path),
            "frames": actual,
            "seconds": seconds,
            "fps": 25,
        }


def _summary(arrays, metrics, length, dt):
    phases = arrays["source_frame"]
    active = (phases >= 0) & (phases < length)
    report = {
        "whole_rollout": {},
        "source_motion": {},
        "entry": {},
        "settle": {},
    }
    for name, mask in (
        ("whole_rollout", np.ones(len(phases), bool)),
        ("source_motion", active),
        ("entry", phases < 0),
        ("settle", phases >= length),
    ):
        report[name] = {
            key: _statistics(value[mask]) for key, value in metrics.items()
        }
    contacts = arrays["actual_foot_contact"].astype(bool)
    speed = metrics["foot_speed_xy_mps"]
    report["foot_contact_fraction"] = np.mean(contacts, axis=0).tolist()
    report["contact_conditioned_foot_speed_mps"] = [
        _statistics(speed[contacts[:, foot], foot]) for foot in range(2)
    ]
    report["reference_foot_contact"] = (
        "not supplied by original motion; "
        "no inferred ground-truth contact label"
    )
    report["last_50_frames_root_speed_mps"] = _statistics(
        metrics["root_speed_mps"][-50:]
    )
    report["last_50_frames_foot_speed_xy_mps"] = _statistics(speed[-50:])
    report["actual_root_displacement_m"] = (
        arrays["actual_root_position"][-1] - arrays["actual_root_position"][0]
    ).tolist()
    report["reference_root_displacement_m"] = (
        arrays["reference_root_position"][-1]
        - arrays["reference_root_position"][0]
    ).tolist()
    report["fallen_samples"] = int(arrays["actual_fallen"].sum())
    report["sample_period_s"] = dt
    return report


def evaluate_skills(
    env,
    teacher_or_none,
    output_new,
    *,
    record=True,
    initialization="neutral",
    clip_ids=None,
):
    """Evaluate complete skills; exceptions preserve evidence and raise."""
    if env.num_envs != 1:
        raise ValueError("Skill evaluation requires exactly one environment")
    if initialization not in {"neutral", "reference"}:
        raise ValueError("Choose neutral or reference initialization")
    chosen = (
        list(range(len(env.library.clips)))
        if clip_ids is None
        else list(clip_ids)
    )
    if (
        not chosen
        or len(set(chosen)) != len(chosen)
        or any(
            type(i) is not int or not 0 <= i < len(env.library.clips)
            for i in chosen
        )
    ):
        raise ValueError("clip_ids must be distinct valid integer indices")
    output = Path(output_new).expanduser().absolute()
    output.mkdir(parents=True, exist_ok=False)
    kind = (
        "frozen_lower_reference_baseline"
        if teacher_or_none is None
        else "learned_reference_conditioned_teacher"
    )
    summary = {
        "format": "solo_skill_evaluation_v1",
        "state": "running",
        "kind": kind,
        "initialization": initialization,
        "continuous_without_midclip_reset": True,
        "no_opponent_or_combat_reward": True,
        "kinematic_replay": False,
        "deployable_competition_policy": False,
        "form_certified": False,
        "reference_manifest": env.library.manifest,
        "environment_recipe": env.recipe,
        "clips": [],
    }
    _write(output / "summary.json", summary)
    was_training = getattr(teacher_or_none, "training", None)
    if hasattr(teacher_or_none, "eval"):
        teacher_or_none.eval()
    try:
        for clip_id in chosen:
            clip = env.library.clips[clip_id]
            directory = output / f"{clip_id:02d}-{clip['name']}"
            directory.mkdir(exist_ok=False)
            report = {
                "state": "running",
                "kind": kind,
                "clip_id": clip_id,
                "clip": clip["name"],
                "initialization": initialization,
                "source_frames": int(clip["frames"]),
                "form_certified": False,
            }
            _write(directory / "report.json", report)
            rows, metric_rows, video = [], [], None
            started = time.monotonic()
            error = None
            try:
                with torch.inference_mode():
                    observation = env.start(
                        [clip_id],
                        reset_physics=True,
                        neutral=initialization == "neutral",
                        random_phase=False,
                        yaw=0,
                    )
                    if record:
                        video = _Video(
                            env.physics,
                            directory,
                            f"{clip['name']} | {kind}",
                            initialization,
                        )
                    maximum = (
                        clip["frames"]
                        + (
                            env.entry_steps
                            if initialization == "neutral"
                            else 0
                        )
                        + env.settle_steps
                        + 2
                    )
                    done, completed, failed = False, False, False
                    action = torch.zeros(
                        (1, env.action_dim), device=env.device
                    )
                    for step in range(maximum + 1):
                        actual = env._state
                        reference = {
                            key: value[:, 0]
                            for key, value in env.reference().items()
                        }
                        current = _metrics(actual, reference, env)
                        state_row = {
                            "source_frame": _array(env.frame[0]),
                            "simulation_time_s": np.array(step * env.dt),
                            "action_applied": _array(action[0]),
                            "qpos": env.physics.native[0].qpos.copy(),
                            "qvel": env.physics.native[0].qvel.copy(),
                        }
                        state_row.update(
                            {
                                "actual_" + key: _array(value[0])
                                for key, value in actual.items()
                            }
                        )
                        state_row.update(
                            {
                                "reference_" + key: _array(value[0])
                                for key, value in reference.items()
                            }
                        )
                        if not all(
                            np.isfinite(value).all()
                            for value in (
                                *state_row.values(),
                                *current.values(),
                            )
                        ):
                            raise RuntimeError(
                                "Nonfinite live state, reference or metric"
                            )
                        rows.append(state_row)
                        metric_rows.append(current)
                        if video is not None and (step % 2 == 0 or done):
                            video.add(
                                actual,
                                reference,
                                int(env.frame[0]),
                                step * env.dt,
                                current,
                            )
                        if done:
                            break
                        if step == maximum:
                            raise RuntimeError(
                                "Skill exceeded its complete-clip budget"
                            )
                        action = (
                            torch.zeros((1, env.action_dim), device=env.device)
                            if teacher_or_none is None
                            else teacher_or_none.act(
                                observation, deterministic=True
                            )
                        )
                        observation, _, terminal, info = env.step(action)
                        done = bool(terminal[0])
                        completed, failed = (
                            bool(info["completed"][0]),
                            bool(info["failed"][0]),
                        )
                    report.update(
                        state="completed",
                        frames_executed=len(rows) - 1,
                        simulation_seconds=(len(rows) - 1) * env.dt,
                        end_reason="failure_termination"
                        if failed
                        else "complete_clip_and_settle",
                        completed_clip_and_settle=completed,
                        failure_flags=[
                            "environment_tracking_or_fall_termination"
                        ]
                        if failed
                        else [],
                        successful_motion_learning_claim=False,
                    )
            except BaseException as exception:
                error = exception
                report.update(
                    state="failed",
                    error={
                        "type": type(exception).__name__,
                        "message": str(exception),
                    },
                    failure_flags=["evaluation_exception"],
                    successful_motion_learning_claim=False,
                )
            finally:
                if video is not None:
                    try:
                        video.close()
                        report["video"] = video.verify()
                    except BaseException as exception:
                        error = error or exception
                        report.update(
                            state="failed",
                            video_error=str(exception),
                            successful_motion_learning_claim=False,
                        )
                        report.setdefault("failure_flags", []).append(
                            "video_encoding_or_validation_failure"
                        )
                if rows:
                    arrays = {
                        key: np.stack([row[key] for row in rows])
                        for key in rows[0]
                    }
                    metrics = {
                        key: np.stack([row[key] for row in metric_rows])
                        for key in metric_rows[0]
                    }
                    np.savez_compressed(
                        directory / "states.npz",
                        **arrays,
                        **{"metric_" + k: v for k, v in metrics.items()},
                    )
                    report["states_sha256"] = sha256(directory / "states.npz")
                    report["metrics"] = _summary(
                        arrays, metrics, clip["frames"], env.dt
                    )
                report["wall_seconds"] = time.monotonic() - started
                _write(directory / "report.json", report)
                summary["clips"].append(
                    {**report, "directory": str(directory)}
                )
                summary["state"] = "failed" if error is not None else "running"
                _write(output / "summary.json", summary)
            if error is not None:
                raise error
        summary["state"] = "completed"
        summary["all_clips_completed_without_environment_failure"] = all(
            clip["completed_clip_and_settle"] for clip in summary["clips"]
        )
        _write(output / "summary.json", summary)
        return summary
    finally:
        if was_training is not None and hasattr(teacher_or_none, "train"):
            teacher_or_none.train(was_training)
