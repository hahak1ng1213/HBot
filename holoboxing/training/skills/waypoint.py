"""Bounded physical waypoint demo: scripted steering, learned forward footwork.

This is an experiment above the frozen five-point controller, not a learned
navigation policy. Turns and early stops are untrained target requests. The
live simulator, never a pose assignment, determines whether they work.
"""

from __future__ import annotations
import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.skills.env import quat_mul


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw(quaternion):
    q = torch.as_tensor(quaternion).detach().cpu().numpy().reshape(-1, 4)[0]
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def array(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value).copy()


@dataclass(frozen=True)
class WaypointConfig:
    max_seconds: float = 45.0
    goal_tolerance_m: float = 0.30
    arrival_min_distance_m: float = 0.0
    stop_speed_mps: float = 0.15
    stop_angular_speed_radps: float = 0.30
    max_root_tilt_rad: float = 0.50
    stable_seconds: float = 1.0
    heading_tolerance_rad: float = math.radians(12)
    max_heading_step_rad: float = math.radians(10)
    turn_rate_radps: float = math.radians(12)
    turn_settle_seconds: float = 0.5
    hold_timeout_seconds: float = 3.0
    initial_heading_rad: float = 0.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if not math.isfinite(value) or (
                key not in {"initial_heading_rad", "arrival_min_distance_m"}
                and value <= 0
            ):
                raise ValueError(f"Invalid waypoint setting: {key}")
        if not 0 <= self.arrival_min_distance_m < self.goal_tolerance_m:
            raise ValueError("Arrival minimum must be below the outer radius")
        if self.max_heading_step_rad > math.radians(30):
            raise ValueError("Turn increments must not exceed 30 degrees")
        if self.stable_seconds > self.hold_timeout_seconds:
            raise ValueError(
                "Hold timeout must include the arrival observation window"
            )


def measured_goal(state, goal, guard_offset):
    position = array(state["root_position"])[0, :2]
    delta = np.asarray(goal) - position
    distance = float(np.linalg.norm(delta))
    heading = wrap(yaw(state["root_quaternion"]) + guard_offset)
    bearing = math.atan2(delta[1], delta[0]) if distance > 1e-9 else heading
    q = array(state["root_quaternion"])[0]
    q = q / np.linalg.norm(q)
    tilt = math.acos(float(np.clip(1 - 2 * (q[1] ** 2 + q[2] ** 2), -1, 1)))
    return {
        "distance_m": distance,
        "heading_rad": heading,
        "bearing_rad": bearing,
        "heading_error_rad": wrap(bearing - heading),
        "speed_mps": float(
            np.linalg.norm(array(state["root_linear_velocity"])[0, :2])
        ),
        "angular_speed_radps": float(
            np.linalg.norm(array(state["root_angular_velocity"])[0])
        ),
        "fallen": bool(array(state["fallen"])[0]),
        "root_tilt_rad": tilt,
    }


def capture_hold(state):
    """Capture a fixed world target, including feet; never chase drift."""
    return (
        state["point_position"].detach().clone(),
        state["point_quaternion"].detach().clone(),
    )


def hold_targets(snapshot):
    # Identical poses at all five future times imply a zero-velocity target.
    p, q = snapshot
    return p[:, None].expand(-1, 5, -1, -1), q[:, None].expand(-1, 5, -1, -1)


def turn_targets(snapshot, pivot, elapsed, delta, config, dt=0.02):
    """Coherent yaw transform at a bounded rate; only target poses move."""
    p, q = snapshot
    future = p.new_tensor((1, 2, 3, 4, 5)) * dt + elapsed
    angles = future.mul(config.turn_rate_radps).clamp_max(abs(delta))
    angles *= 1 if delta >= 0 else -1
    c, s = angles.cos()[None, :, None], angles.sin()[None, :, None]
    relative = p[:, None] - pivot[:, None, None]
    x, y, z = relative.expand(-1, 5, -1, -1).unbind(-1)
    targets = torch.stack((c * x - s * y, s * x + c * y, z), -1)
    targets += pivot[:, None, None]
    turn_q = torch.stack(
        (torch.cos(angles / 2), angles * 0, angles * 0, torch.sin(angles / 2)),
        -1,
    )
    rotations = quat_mul(turn_q[None, :, None], q[:, None])
    return targets, rotations


def verify_teacher_contract(env, saved):
    """Keep the original teacher ABI and provenance, including clip one-hot."""
    checks = {
        "reference_manifest": env.library.manifest,
        "environment_recipe": env.recipe,
        "physics_metadata": env.physics.metadata,
        "observation_dim": env.observation_dim,
        "skill": "locomotion",
    }
    if any(saved.get(key) != value for key, value in checks.items()):
        raise ValueError("Teacher data or environment contract differs")
    if env.num_envs != 1 or not env.library.clips:
        raise ValueError("Waypoint demonstration needs one physical world")
    if any(
        tuple(array(c["direction"]).tolist()) != (1.0, 0.0)
        for c in env.library.clips
    ):
        raise ValueError(
            "This demo requires the exact forward-only teacher library"
        )


def run_waypoint(env, teacher, goal, config, *, observer=None):
    """One continuous episode; only its initial ordinary stance is reset.

    Stop initiation uses actual goal distance at every physics control step,
    not reference endpoints. A frozen measured-pose hold is conservative and
    can fail; an upright, low-speed observation window determines arrival.
    """
    goal = np.asarray(goal, dtype=float)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("Expected two finite world goal coordinates")
    if env.num_envs != 1 or not math.isclose(env.dt, 0.02):
        raise ValueError("Use one world at 50 Hz")
    clips = env.library.clips
    # Prefer the shorter whole primitive; retain the training library.
    clip_id = min(range(len(clips)), key=lambda i: clips[i]["frames"])
    clip = clips[clip_id]
    guard_offset = wrap(
        float(clip["guard_heading_world"]) - yaw(clip["root_quaternion"][0])
    )
    root_qi = env.physics.root_qi
    initial_yaw = yaw(env.physics.initial_qpos[root_qi + 3 : root_qi + 7])
    reset_yaw = config.initial_heading_rad - guard_offset - initial_yaw
    env.start(
        [clip_id],
        reset_physics=True,
        neutral=True,
        random_phase=False,
        yaw=reset_yaw,
    )
    teacher.eval()
    state = env.physics.state()
    mode, snapshot, turn_pivot = "decide", None, None
    phase_seconds, turn_delta, stable = 0.0, 0.0, 0
    events, rows = [], []
    arrived, failed, reason = False, False, "time_limit"
    maximum = int(math.floor(config.max_seconds / env.dt))
    required_stable = int(math.ceil(config.stable_seconds / env.dt))
    min_distance = float("inf")
    action = torch.zeros((1, 30), device=env.device)
    started = time.monotonic()
    for step in range(maximum + 1):
        seconds = step * env.dt
        metrics = measured_goal(state, goal, guard_offset)
        min_distance = min(min_distance, metrics["distance_m"])
        if metrics["fallen"]:
            failed, reason = True, "fall"
        elif metrics["distance_m"] < config.arrival_min_distance_m:
            # Forward-only execution cannot honestly repair an overshot band.
            failed, reason = True, "too_close_no_reverse_skill"
        # Check every step; allow stopping before a full primitive ends.
        if (
            not failed
            and mode != "hold"
            and config.arrival_min_distance_m
            <= metrics["distance_m"]
            <= config.goal_tolerance_m
        ):
            mode, snapshot, phase_seconds, stable = (
                "hold",
                capture_hold(state),
                0.0,
                0,
            )
            events.append(
                {"time_s": seconds, "event": "early_goal_hold", **metrics}
            )
        if mode == "hold" and not failed:
            stationary = (
                config.arrival_min_distance_m
                <= metrics["distance_m"]
                <= config.goal_tolerance_m
                and metrics["speed_mps"] <= config.stop_speed_mps
                and metrics["angular_speed_radps"]
                <= config.stop_angular_speed_radps
                and metrics["root_tilt_rad"] <= config.max_root_tilt_rad
                and (
                    config.arrival_min_distance_m == 0
                    or abs(metrics["heading_error_rad"])
                    <= config.heading_tolerance_rad
                )
            )
            stable = stable + 1 if stationary and phase_seconds > 0 else 0
            if stable >= required_stable:
                arrived, reason = True, "measured_stable_arrival"
            elif phase_seconds >= config.hold_timeout_seconds:
                reason = "hold_failed_to_stop_at_goal"
                failed = True
        if not (arrived or failed or step == maximum) and mode == "decide":
            if (
                abs(metrics["heading_error_rad"])
                > config.heading_tolerance_rad
            ):
                mode, phase_seconds = "turn", 0.0
                snapshot = capture_hold(state)
                turn_pivot = state["root_position"].detach().clone()
                turn_delta = float(
                    np.clip(
                        metrics["heading_error_rad"],
                        -config.max_heading_step_rad,
                        config.max_heading_step_rad,
                    )
                )
                events.append(
                    {
                        "time_s": seconds,
                        "event": "scripted_turn",
                        "requested_yaw_delta_rad": turn_delta,
                        **metrics,
                    }
                )
            else:
                mode, phase_seconds = "forward", 0.0
                requested_heading = metrics["heading_rad"] + float(
                    np.clip(
                        metrics["heading_error_rad"],
                        -config.max_heading_step_rad,
                        config.max_heading_step_rad,
                    )
                )
                env.start(
                    [clip_id],
                    reset_physics=False,
                    neutral=False,
                    random_phase=False,
                    yaw=requested_heading - float(clip["guard_heading_world"]),
                )
                events.append(
                    {
                        "time_s": seconds,
                        "event": "learned_forward",
                        "clip": clip["name"],
                        "requested_heading_rad": requested_heading,
                        **metrics,
                    }
                )
        row = {
            "simulation_time_s": np.array(seconds),
            "goal_world_xy": goal.copy(),
            "mode": np.array(mode),
            "distance_m": np.array(metrics["distance_m"]),
            "heading_error_rad": np.array(metrics["heading_error_rad"]),
            "teacher_action_previous": array(action[0]),
            "source_frame": array(env.frame[0]),
        }
        row.update({"actual_" + k: array(v[0]) for k, v in state.items()})
        if hasattr(env.physics, "native"):
            row["qpos"] = env.physics.native[0].qpos.copy()
            row["qvel"] = env.physics.native[0].qvel.copy()
        rows.append(row)
        if observer is not None:
            observer(
                step,
                state,
                mode,
                metrics,
                arrived or failed or step == maximum,
            )
        if arrived or failed or step == maximum:
            break
        # Env.step saves tensors later mutated by Env.start. Inference tensors
        # cannot be mutated outside that context at the next option boundary.
        with torch.no_grad():
            if mode == "forward":
                action = teacher.act(
                    env.get_observations(), deterministic=True
                )
                _, _, _, info = env.step(action)
                state = env.physics.state()
                if bool(info["failed"][0]):
                    failed, reason = True, "forward_tracking_or_fall_failure"
                elif bool(info["completed"][0]):
                    mode = "decide"
            else:
                action = torch.zeros_like(action)
                targets = (
                    hold_targets(snapshot)
                    if mode == "hold"
                    else turn_targets(
                        snapshot,
                        turn_pivot,
                        phase_seconds,
                        turn_delta,
                        config,
                        env.dt,
                    )
                )
                state = env.physics.step(*targets)
                # Keep measured state consistent for the next teacher entry.
                env._state = state
                if (
                    mode == "turn"
                    and phase_seconds + env.dt
                    >= abs(turn_delta) / config.turn_rate_radps
                    + config.turn_settle_seconds
                ):
                    mode = "decide"
        phase_seconds += env.dt
    report = {
        "format": "solo_scripted_waypoint_demo_v1",
        "state": "completed",
        "controller": (
            "scripted measured-goal steering and hold; "
            "frozen learned forward teacher"
        ),
        "learned_navigation": False,
        "turning_and_early_stop_trained": False,
        "goal_world_xy": goal.tolist(),
        "goal_semantics": (
            "opponent-center marker; geometric stand-off only"
            if config.arrival_min_distance_m > 0
            else "waypoint"
        ),
        "arrival_distance_band_m": [
            config.arrival_min_distance_m,
            config.goal_tolerance_m,
        ],
        "initial_distance_m": float(rows[0]["distance_m"]),
        "final_distance_m": metrics["distance_m"],
        "minimum_distance_m": min_distance,
        "final_speed_mps": metrics["speed_mps"],
        "final_angular_speed_radps": metrics["angular_speed_radps"],
        "final_root_tilt_rad": metrics["root_tilt_rad"],
        "final_goal_heading_error_rad": metrics["heading_error_rad"],
        "arrival": arrived and not failed,
        "failed": failed,
        "end_reason": reason,
        "simulation_seconds": float(rows[-1]["simulation_time_s"]),
        "stable_arrival_seconds": stable * env.dt,
        "fallen_samples": sum(bool(row["actual_fallen"]) for row in rows),
        "physical_steps": len(rows) - 1,
        "mid_episode_physics_resets": 0,
        "kinematic_replay": False,
        "competition_submission": False,
        "config": asdict(config),
        "events": events,
        "wall_seconds": time.monotonic() - started,
        "limitations": [
            "Forward teacher remains reference-conditioned.",
            "Turning and early hold are scripted physical candidates, "
            "not learned or certified skills.",
            "Holding a captured swing-foot pose can fail; "
            "no pose or velocity is overwritten to stop.",
            "This single run does not establish generalized navigation "
            "or obstacle avoidance.",
        ],
    }
    return report, {k: np.stack([row[k] for row in rows]) for k in rows[0]}


def make_video(physics, output, goal, tolerance, inner_radius=0.0):
    from holoboxing.training.skills.evaluate import _Video

    class WaypointVideo(_Video):
        def add_live(self, state, seconds, mode, metrics):
            from PIL import Image, ImageDraw

            actual = array(state["root_position"])[0]
            center = (actual[:2] + goal) / 2
            self.camera.lookat[:] = [*center, 0.5]
            self.camera.distance = max(
                3.4, float(np.linalg.norm(actual[:2] - goal)) * 1.35 + 1.8
            )
            self.renderer.update_scene(self.data, camera=self.camera)
            scene = self.renderer.scene
            disks = [(tolerance, 0.012, [0.15, 0.9, 0.35, 0.6])]
            if inner_radius > 0:
                disks.append((inner_radius, 0.025, [0.95, 0.2, 0.15, 0.7]))
            for radius, height, color in disks:
                if scene.ngeom >= scene.maxgeom:
                    break
                self.mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    self.mujoco.mjtGeom.mjGEOM_CYLINDER,
                    np.array([radius, 0.01, 0.0]),
                    np.array([*goal, height]),
                    np.eye(3).ravel(),
                    np.array(color, dtype=np.float32),
                )
                scene.ngeom += 1
            picture = Image.fromarray(self.renderer.render())
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 960, 78), fill=(15, 18, 23))
            for y, line in zip(
                (5, 29, 53),
                (
                    "LIVE PHYSICS | learned footwork + SCRIPTED steering",
                    f"t={seconds:.2f}s  mode={mode}  "
                    f"distance={metrics['distance_m']:.3f}m  "
                    f"speed={metrics['speed_mps']:.3f}m/s",
                    f"center ({goal[0]:.2f}, {goal[1]:.2f}) "
                    f"| green arrival {inner_radius:.2f}-{tolerance:.2f}m "
                    "| red: too close",
                ),
                strict=True,
            ):
                draw.text((12, y), line, font=self.font, fill="white")
            self.writer.send(np.asarray(picture))
            self.frames += 1
            self.picture = picture
            if self.frames == 1:
                picture.save(self.directory / "first.png")

    return WaypointVideo(physics, output, "Waypoint", "ordinary stance")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument(
        "--catalog",
        type=existing_path,
        help="Optional exact pinned catalog; must match checkpoint hash",
    )
    parser.add_argument("--checkpoint", type=existing_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--goal", type=float, nargs=2, required=True, metavar=("X", "Y")
    )
    parser.add_argument("--max-seconds", type=float, default=45.0)
    parser.add_argument("--hold-timeout", type=float, default=6.0)
    parser.add_argument("--initial-heading-deg", type=float, default=0.0)
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        default=0.30,
        help="Outer arrival radius in meters (also for stand-off mode)",
    )
    parser.add_argument(
        "--arrival-min-distance",
        type=float,
        default=0.0,
        help="Inner arrival radius; >0 treats goal as opponent-center marker",
    )
    parser.add_argument("--max-heading-step-deg", type=float, default=10.0)
    parser.add_argument("--turn-rate-deg-s", type=float, default=12.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("--output must be new")
    if not np.isfinite(args.goal).all():
        parser.error("--goal must be finite")
    try:
        args.config = WaypointConfig(
            max_seconds=args.max_seconds,
            hold_timeout_seconds=args.hold_timeout,
            initial_heading_rad=math.radians(args.initial_heading_deg),
            goal_tolerance_m=args.goal_tolerance,
            arrival_min_distance_m=args.arrival_min_distance,
            max_heading_step_rad=math.radians(args.max_heading_step_deg),
            turn_rate_radps=math.radians(args.turn_rate_deg_s),
        )
    except ValueError as error:
        parser.error(str(error))
    return args


def main(argv=None):
    args = parse_arguments(argv)
    from holoboxing.data.common import sha256
    from holoboxing.training.resources import resolve_resources
    from holoboxing.training.skills import reference
    from holoboxing.training.skills.actor import load_teacher
    from holoboxing.training.skills.env import SkillTrackingEnv
    from holoboxing.training.skills.physics import SoloPhysics

    torch.set_num_threads(1)
    physics, video, report, rows = None, None, None, None
    output_created = False
    output = args.output.expanduser().absolute()
    try:
        teacher, saved = load_teacher(args.checkpoint, args.device)
        if args.catalog:
            if (
                sha256(args.catalog)
                != saved["reference_manifest"]["catalog_sha256"]
            ):
                raise ValueError("Catalog differs from teacher training")
            reference.CATALOG = args.catalog
        resources, _, _ = resolve_resources(args.assets, args.lower)
        physics = SoloPhysics(
            SimpleNamespace(resources=resources), args.lower, 1, args.device
        )
        library = reference.MotionLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            skill="locomotion",
            device=args.device,
            locomotion_mode="forward",
        )
        env = SkillTrackingEnv(physics, library)
        verify_teacher_contract(env, saved)
        output.mkdir(parents=True, exist_ok=False)
        output_created = True
        if not args.no_video:
            video = make_video(
                physics,
                output,
                np.asarray(args.goal),
                args.config.goal_tolerance_m,
                args.config.arrival_min_distance_m,
            )

        def observe(step, state, mode, metrics, done):
            if video is not None and (step % 2 == 0 or done):
                video.add_live(state, step * env.dt, mode, metrics)

        report, rows = run_waypoint(
            env, teacher, args.goal, args.config, observer=observe
        )
        report["checkpoint"] = {
            "path": str(args.checkpoint),
            "sha256": sha256(args.checkpoint),
            "update": saved.get("update"),
        }
        report["reference_manifest"] = library.manifest
        report["physics_metadata"] = physics.metadata
        np.savez_compressed(output / "states.npz", **rows)
        report["states_sha256"] = sha256(output / "states.npz")
        if video is not None:
            finished_video = video
            video = None
            finished_video.close()
            report["video"] = finished_video.verify()
        (output / "report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in (
                        "arrival",
                        "failed",
                        "end_reason",
                        "final_distance_m",
                        "simulation_seconds",
                    )
                }
            ),
            flush=True,
        )
    except BaseException as error:
        if output_created:
            (output / "failure.json").write_text(
                json.dumps(
                    {
                        "error": str(error),
                        "type": type(error).__name__,
                        "physical_success": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        raise
    finally:
        try:
            if video is not None:
                video.close()
        finally:
            if physics is not None:
                physics.close()


if __name__ == "__main__":
    main()
