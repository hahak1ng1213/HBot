"""Physical contact test: original 045 against a standing, free-root G1."""

# ruff: noqa: E402

import argparse
import json
import math
import os
import platform
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from holoboxing.evaluation.runtime.contact_geometry import (
    opponent_target_semantic,
    strike_geometry,
)
from holoboxing.evaluation.runtime.hand_hit_rules import target_part
from holoboxing.evaluation.runtime.impulse_hit_rules import (
    RollingNormalImpulse,
)
from holoboxing.experiments.combo45_reference import (
    BASE,
    EVENTS,
    ComboReferenceLibrary,
    sha256,
    write_json,
)
from holoboxing.experiments.impulse_metrics import analyze_impulses
from holoboxing.experiments.provenance import code_identity
from holoboxing.training.cli import add_controller_arguments, existing_path
from holoboxing.training.envs.mujoco_physics import CompetitionPhysics
from holoboxing.training.resources import resolve_resources
from holoboxing.training.skills.env import SkillTrackingEnv
from holoboxing.training.skills.evaluate import _Video
from holoboxing.training.skills.physics import SoloPhysics


def numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
    )


class ImpactPhysics:
    """CompetitionPhysics integrates; solo models supply FK and the lower."""

    def __init__(self, assets, lower):
        self.solo = SoloPhysics(assets, lower, 2, "cuda:0")
        self.pair = CompetitionPhysics(assets, 1, "cuda:0")
        self.pair.set_gains(
            torch.tensor(self.solo.kp), torch.tensor(self.solo.kd)
        )
        self.device, self.num_envs = self.solo.device, 1
        for name in (
            "host",
            "joint_names",
            "body_names",
            "point_names",
            "initial_qpos",
            "qi",
            "vi",
            "root_qi",
            "root_vi",
        ):
            setattr(self, name, getattr(self.solo, name))
        self.models, self.native = self.pair.models, self.pair.native
        self.floor_height = self.pair.floor_height
        self.target_distance, self.target_heading = 0.575, 0.0
        self.source_phase = -20
        self.samples, self.contacts = [], []
        self.impulse_clock = RollingNormalImpulse()
        self.max_official_force_error = 0.0
        self.root_vis = [
            int(
                self.pair.host.jnt_dofadr[
                    self.pair.host.body(prefix + "pelvis").jntadr[0]
                ]
            )
            for prefix in ("p1_", "p2_")
        ]
        self.geom_roles = np.array(
            [
                0
                if self.pair.host.body(int(b)).name.startswith("p1_")
                else 1
                if self.pair.host.body(int(b)).name.startswith("p2_")
                else -1
                for b in self.pair.host.geom_bodyid
            ]
        )
        self.geometry = [
            strike_geometry(self.pair.host, geom)
            for geom in range(self.pair.host.ngeom)
        ]
        self.target_semantics = [
            opponent_target_semantic(self.pair.host, 0, geom)
            for geom in range(self.pair.host.ngeom)
        ]
        self.floors = {self.pair.host.geom("smpl_ring_baseLink").id}
        self.foot_ids = [
            [
                self.pair.host.body(prefix + side + "_ankle_roll_link").id
                for side in ("left", "right")
            ]
            for prefix in ("p1_", "p2_")
        ]
        self._both = None
        self.state()

    def state(self):
        data = self.native[0]
        pair_qi, pair_vi, _ = self.pair.indices
        for role, scratch in enumerate(self.solo.native):
            pqi, pvi = self.pair.root_qi[role], self.root_vis[role]
            scratch.qpos[self.root_qi : self.root_qi + 7] = data.qpos[
                pqi : pqi + 7
            ]
            scratch.qpos[self.root_qi + 2] -= self.floor_height
            scratch.qpos[self.qi] = data.qpos[pair_qi[role]]
            scratch.qvel[self.root_vi : self.root_vi + 6] = data.qvel[
                pvi : pvi + 6
            ]
            scratch.qvel[self.vi] = data.qvel[pair_vi[role]]
            scratch.time = data.time
        self._both = self.solo.state()
        feet = np.zeros((2, 2), bool)
        for ci in range(data.ncon):
            geoms = data.contact[ci].geom
            if not any(int(g) in self.floors for g in geoms):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(self.models[0], data, ci, force)
            if force[0] <= 1:
                continue
            for geom in geoms:
                body = int(self.models[0].geom_bodyid[geom])
                weld = int(self.models[0].body_weldid[body])
                for role in range(2):
                    for foot in range(2):
                        feet[role, foot] |= weld == self.foot_ids[role][foot]
        self._both["foot_contact"] = torch.as_tensor(feet, device=self.device)
        return {key: value[:1] for key, value in self._both.items()}

    def reset(
        self, ids, root_position, root_quaternion, joint_position, **velocities
    ):
        if list(numpy(ids)) != [0]:
            raise ValueError("Only one paired world is supported")
        self.solo.reset(
            [0], root_position, root_quaternion, joint_position, **velocities
        )
        q = self.initial_qpos
        heading = self.target_heading
        target_position = np.array(
            [
                [
                    self.target_distance * math.cos(heading),
                    self.target_distance * math.sin(heading),
                    q[self.root_qi + 2],
                ]
            ]
        )
        rotation = Rotation.from_euler(
            "z", heading + math.pi
        ) * Rotation.from_quat(
            q[self.root_qi + 3 : self.root_qi + 7][[1, 2, 3, 0]]
        )
        quat = rotation.as_quat()[[3, 0, 1, 2]][None]
        self.solo.reset([1], target_position, quat, q[self.qi][None])
        model, data = self.models[0], self.native[0]
        mujoco.mj_resetData(model, data)
        data.qpos[:] = self.pair.initial.qpos
        pqi, pvi, _ = self.pair.indices
        for role, scratch in enumerate(self.solo.native):
            qi, vi = self.pair.root_qi[role], self.root_vis[role]
            data.qpos[qi : qi + 7] = scratch.qpos[
                self.root_qi : self.root_qi + 7
            ]
            data.qpos[qi + 2] += self.floor_height
            data.qpos[pqi[role]] = scratch.qpos[self.qi]
            data.qvel[vi : vi + 6] = scratch.qvel[
                self.root_vi : self.root_vi + 6
            ]
            data.qvel[pvi[role]] = scratch.qvel[self.vi]
        mujoco.mj_forward(model, data)
        self.pair._pull()
        self.pair.targets[0] = torch.as_tensor(
            data.qpos[pqi], device=self.device, dtype=torch.float32
        )
        state = self.state()
        self.hold_p = (
            self._both["point_position"][1:2, None]
            .expand(-1, 5, -1, -1)
            .clone()
        )
        self.hold_q = (
            self._both["point_quaternion"][1:2, None]
            .expand(-1, 5, -1, -1)
            .clone()
        )
        self.samples, self.contacts = [], []
        self.impulse_clock = RollingNormalImpulse()
        self.max_official_force_error = 0.0
        self.initial_interrobot_contacts = [
            float(data.contact[c].dist)
            for c in range(data.ncon)
            if set(self.geom_roles[data.contact[c].geom]) == {0, 1}
        ]
        return state

    def _sample_contacts(self, substep):
        model, data = self.models[0], self.native[0]
        normal, world = np.zeros((2, 2)), np.zeros((2, 2, 3))
        positive_separation = np.zeros((2, 2))
        other_normal = 0.0
        for ci in range(data.ncon):
            contact = data.contact[ci]
            geoms = [int(g) for g in contact.geom]
            if set(self.geom_roles[geoms]) != {0, 1}:
                continue
            local = np.zeros(6)
            mujoco.mj_contactForce(model, data, ci, local)
            if not np.isfinite(local).all():
                raise RuntimeError("Nonfinite contact force")
            force = local[:3] @ contact.frame.reshape(3, 3)
            own_side = 0 if self.geom_roles[geoms[0]] == 0 else 1
            own, other = geoms[own_side], geoms[1 - own_side]
            strike = self.geometry[own]
            value = max(0.0, float(local[0]))
            if strike is None or strike[1] not in ("left_hand", "right_hand"):
                other_normal += value
                continue
            hand = 0 if strike[1] == "left_hand" else 1
            semantic = self.target_semantics[other]
            category = 0 if target_part(semantic[2]) is not None else 1
            target_force = force if own_side == 0 else -force
            normal[hand, category] += value
            world[hand, category] += target_force
            if contact.dist > 0:
                positive_separation[hand, category] += value
            self.contacts.append(
                {
                    "time_s": float(data.time),
                    "source_phase": self.source_phase + (substep + 1) / 4,
                    "hand": ("left", "right")[hand],
                    "target": semantic[2],
                    "target_link": semantic[1],
                    "category": ("legal", "unscored")[category],
                    "normal_force_n": value,
                    "world_force_on_target_n": target_force.tolist(),
                    "contact_position_world_m": contact.pos.tolist(),
                    "separation_m": float(contact.dist),
                    "attacker_geom": own,
                    "target_geom": other,
                }
            )
        official = np.zeros(2)
        for item in self.pair.official_contacts()[0]:
            if item["attacker"] == 0:
                official[
                    0 if item["attacking_body"] == "left_hand" else 1
                ] += item["normal_force_n"]
        error = float(np.max(np.abs(official - normal[:, 0])))
        self.max_official_force_error = max(
            self.max_official_force_error, error
        )
        if not np.allclose(official, normal[:, 0], atol=1e-6, rtol=1e-7):
            raise RuntimeError(
                "Raw force differs from official contact extraction"
            )
        windows, _ = self.impulse_clock.update(
            data.time, self.pair.official_contacts()[0]
        )
        self.samples.append(
            {
                "time_s": float(data.time),
                "normal_force_n": normal,
                "world_force_n": world,
                "phase": self.source_phase + (substep + 1) / 4,
                "nonhand_normal_force_n": other_normal,
                "positive_separation_normal_force_n": positive_separation,
                "official_unconsumed_window_ns": windows[0].copy(),
            }
        )

    @torch.inference_mode()
    def step(self, target_positions, target_quaternions):
        self.state()
        state = self._both
        targets = self.solo.lower(
            reference_position_w=torch.cat((target_positions, self.hold_p)),
            reference_quaternion_wxyz=torch.cat(
                (target_quaternions, self.hold_q)
            ),
            current_reference_root_position_w=state["root_position"],
            reference_root_height=state["root_position"][:, 2:3],
            robot_root_quaternion_wxyz=state["root_quaternion"],
            robot_root_angular_velocity_b=state["root_angular_velocity_local"],
            robot_joint_position=state["joint_position"],
            robot_joint_velocity=state["joint_velocity"],
        )
        self.pair.targets[0] = targets
        for substep in range(4):
            self.pair.substep()
            self._sample_contacts(substep)
        return self.state()

    def close(self):
        self.solo.close()


class ImpactVideo(_Video):
    def add_impact(self, physics, env):
        from PIL import Image, ImageDraw

        state = physics._both
        self.camera.lookat[:] = numpy(state["root_position"]).mean(0)
        self.camera.lookat[2] = 0.8 + physics.floor_height
        self.camera.distance = 3.1
        self.camera.azimuth = math.degrees(physics.target_heading) + 115
        self.renderer.update_scene(self.data, camera=self.camera)
        picture = Image.fromarray(self.renderer.render())
        draw = ImageDraw.Draw(picture)
        draw.rectangle((0, 0, 960, 78), fill=(15, 18, 23))
        force = (
            physics.samples[-1]["normal_force_n"][:, 0]
            if physics.samples
            else np.zeros(2)
        )
        for y, line in zip(
            (5, 29, 53),
            (
                self.label,
                f"t={self.data.time:.2f}s  phase={int(env.frame[0])}  "
                f"legal force L/R={force[0]:.1f}/{force[1]:.1f} N",
                "LIVE CONTACT PHYSICS | red: 045 | blue: standing, no attack",
            ),
            strict=True,
        ):
            draw.text((12, y), line, font=self.font, fill="white")
        self.writer.send(np.asarray(picture))
        self.frames += 1
        self.picture = picture
        if self.frames == 1:
            picture.save(self.directory / "first.png")


def clip_heading(clip, aim_events=None):
    offsets = numpy(
        clip["point_position"][:, 1:3, :2] - clip["root_position"][:, None, :2]
    )
    vectors = []
    start = clip["source_bounds"][0]
    events = EVENTS[clip["name"]] if aim_events is None else aim_events
    for hand, source_frame in events:
        index = 0 if hand == "left" else 1
        frame = source_frame - start
        window = offsets[max(0, frame - 7) : frame + 8, index]
        vector = window[np.argmax(np.linalg.norm(window, axis=-1))]
        vectors.append(vector / np.linalg.norm(vector))
    direction = np.mean(vectors, axis=0)
    return float(np.arctan2(direction[1], direction[0]))


def run_case(
    physics,
    library,
    clip_id,
    distance,
    output,
    *,
    record,
    settle_steps=100,
    aim_events=None,
):
    clip = library.clips[clip_id]
    output.mkdir(exist_ok=False)
    physics.target_distance, physics.target_heading = (
        distance,
        clip_heading(clip, aim_events),
    )
    env = SkillTrackingEnv(
        physics, library, seed=91, entry_steps=20, settle_steps=settle_steps
    )
    rows, video = [], None
    report = {
        "state": "running",
        "clip": clip["name"],
        "source_bounds": clip["source_bounds"],
        "initial_pelvis_xy_distance_m": distance,
        "target_heading_deg": math.degrees(physics.target_heading),
        "heading_selection": (
            "fixed per clip from source wrist extension directions; "
            "no contact-based optimization"
        ),
        "initialization": "neutral",
        "entry_seconds": 0.4,
        "settle_seconds": settle_steps * 0.02,
        "no_midclip_resets": True,
        "aim_events": aim_events,
        "endpoint_hold_is_natural_recovery": False,
        "opponent": (
            "free-root G1, frozen lower holds initial five-point pose, "
            "no attack"
        ),
        "not_official_match_evaluation": True,
        "no_learning": True,
    }
    try:
        with torch.inference_mode():
            env.start([clip_id], reset_physics=True, neutral=True, yaw=0)
            report["initial_interrobot_contact_distances_m"] = (
                physics.initial_interrobot_contacts
            )
            if any(
                distance < -1e-5
                for distance in physics.initial_interrobot_contacts
            ):
                report.update(state="rejected_initial_overlap")
                return report
            if record:
                video = ImpactVideo(
                    physics,
                    output,
                    f"045 {clip['name']} | initial distance {distance:.3f}m",
                    "neutral",
                )
            done = False
            info = {}
            for step in range(clip["frames"] + 20 + settle_steps + 3):
                state = physics._both
                row = {
                    "qpos": physics.native[0].qpos.copy(),
                    "qvel": physics.native[0].qvel.copy(),
                    "time_s": np.array(physics.native[0].time),
                    "source_frame": numpy(env.frame[0]).copy(),
                }
                row.update(
                    {
                        key: numpy(state[key]).copy()
                        for key in (
                            "root_position",
                            "root_quaternion",
                            "root_linear_velocity",
                            "root_angular_velocity",
                            "fallen",
                            "point_position",
                            "point_velocity",
                        )
                    }
                )
                rows.append(row)
                if video is not None and (step % 2 == 0 or done):
                    video.add_impact(physics, env)
                if done:
                    break
                physics.source_phase = int(env.frame[0])
                _, _, terminal, info = env.step(
                    torch.zeros((1, 30), device=physics.device)
                )
                done = bool(terminal[0]) or bool(physics._both["fallen"][1])
            if not done:
                raise RuntimeError("Evaluation exceeded clip budget")
            report.update(
                state="completed",
                completed_clip_and_settle=bool(
                    info.get("completed", torch.tensor([False]))[0]
                ),
                fallen_per_role=np.any(
                    np.stack([r["fallen"] for r in rows]), axis=0
                ).tolist(),
                actual_seconds=float(rows[-1]["time_s"]),
                samples_200hz=len(physics.samples),
                official_contact_normal_sum_max_error_n=physics.max_official_force_error,
            )
            fallen = report["fallen_per_role"]
            report["end_reason"] = (
                "attacker_fall"
                if fallen[0]
                else "defender_fall"
                if fallen[1]
                else "tracking_failure"
                if bool(info.get("failed", [False])[0])
                else "source_end"
                if settle_steps == 0
                else "source_and_endpoint_hold_end"
            )
        arrays = {
            key: np.stack([sample[key] for sample in physics.samples])
            for key in physics.samples[0]
        }
        for scope, mask in (
            ("whole_rollout", np.ones(len(arrays["time_s"]), bool)),
            (
                "source_motion",
                (arrays["phase"] > 0)
                & (arrays["phase"] <= clip["frames"] - 1),
            ),
        ):
            if not mask.any():
                report[scope] = None
                continue
            report[scope] = analyze_impulses(
                arrays["time_s"][mask],
                arrays["normal_force_n"][mask],
                arrays["world_force_n"][mask],
                arrays["phase"][mask],
            )
            report[scope]["official_unconsumed_20ms_peak_ns_per_hand"] = (
                arrays["official_unconsumed_window_ns"][mask].max(0).tolist()
            )
        report["entry_normal_impulse_ns"] = (
            arrays["normal_force_n"][arrays["phase"] <= 0].sum(0) * 0.005
        ).tolist()
        report["nonhand_normal_impulse_ns"] = float(
            arrays["nonhand_normal_force_n"].sum() * 0.005
        )
        report["positive_separation_normal_impulse_ns"] = (
            arrays["positive_separation_normal_force_n"].sum(0) * 0.005
        ).tolist()
        report["pelvis_distance_m_min"] = float(
            np.min(
                [
                    np.linalg.norm(
                        r["root_position"][0, :2] - r["root_position"][1, :2]
                    )
                    for r in rows
                ]
            )
        )
        np.savez_compressed(output / "contacts-200hz.npz", **arrays)
        np.savez_compressed(
            output / "states.npz",
            **{key: np.stack([row[key] for row in rows]) for key in rows[0]},
        )
        with (output / "contacts.jsonl").open("w") as stream:
            for row in physics.contacts:
                stream.write(json.dumps(row) + "\n")
        report["contacts_sha256"] = sha256(output / "contacts-200hz.npz")
        report["states_sha256"] = sha256(output / "states.npz")
    except BaseException as error:
        report.update(
            state="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        if video is not None:
            video.close()
            report["video"] = video.verify()
        write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_controller_arguments(parser)
    parser.add_argument("--source", type=existing_path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(91)
    resources, _, _ = resolve_resources(
        args.assets,
        args.lower,
    )
    physics = ImpactPhysics(
        SimpleNamespace(resources=resources),
        args.lower,
    )
    hashes = {
        name: sha256(BASE / "holoboxing/training/skills" / (name + ".py"))
        for name in ("actor", "train", "env", "physics", "reference")
    }
    report = {
        "state": "running",
        "code": code_identity(),
        "host": platform.node(),
        "pid": os.getpid(),
        "python": sys.executable,
        "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "frozen_source_sha256": hashes,
        "lower_and_fk_metadata": physics.solo.metadata,
        "physics_backend": "native MuJoCo 200 Hz; frozen ONNX on GPU",
        "mujoco_version": mujoco.__version__,
        "floor_height_m": physics.floor_height,
        "opponent_total_mass_kg": float(
            sum(
                physics.pair.host.body_mass[b]
                for b in range(physics.pair.host.nbody)
                if physics.pair.host.body(b).name.startswith("p2_")
            )
        ),
        "original_scene_xml_sha256": sha256(resources.scene),
        "script_sha256": sha256(__file__),
        "cases": [],
    }
    try:
        library = ComboReferenceLibrary(
            args.source,
            physics.host,
            physics.joint_names,
            physics.body_names,
            device="cuda:0",
        )
        write_json(args.output / "reference-manifest.json", library.manifest)
        cases = (
            [(0, 0.575)]
            if args.preflight
            else [(i, d) for i in range(3) for d in (0.50, 0.575, 0.65)]
        )
        for clip_id, distance in cases:
            name = f"{clip_id:02d}-{library.names[clip_id]}-d{distance:.3f}"
            print(json.dumps({"starting": name}), flush=True)
            case = run_case(
                physics,
                library,
                clip_id,
                distance,
                args.output / name,
                record=not args.no_video,
            )
            report["cases"].append(case)
            write_json(args.output / "summary.json", report)
            print(
                json.dumps(
                    {
                        "finished": name,
                        "state": case["state"],
                        "fallen": case.get("fallen_per_role"),
                    }
                ),
                flush=True,
            )
        assert hashes == {
            name: sha256(BASE / "holoboxing/training/skills" / (name + ".py"))
            for name in hashes
        }
        report.update(
            state="completed", original_five_source_files_unchanged=True
        )
        write_json(args.output / "summary.json", report)
    except BaseException as error:
        report.update(state="failed", error=str(error))
        write_json(args.output / "summary.json", report)
        raise
    finally:
        physics.close()


if __name__ == "__main__":
    main()
