from __future__ import annotations
import re
from pathlib import Path
from typing import Mapping, Sequence

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F

from holoboxing.training.tasks.combat.mdp.geometry import (
    FIVE_POINT_LINK_NAMES,
    decode_output,
    normalize_quaternion_wxyz,
    quat_apply_wxyz,
    quat_conjugate_wxyz,
)
from holomotion.src.motion_tracking.link_actor_packing import (
    link_actor_layout,
    pack_link_actor_observation,
)
from holomotion.src.motion_tracking.link_fk import SharedLinkFK
from holomotion.src.motion_tracking.link_observation import (
    LinkObservationInput,
    build_link_observation,
)
from holomotion.src.utils.torch_utils import quat_from_matrix


def _values_by_joint_name(
    joint_names: Sequence[str],
    values: Mapping[str, float],
    *,
    regex_keys: bool,
) -> torch.Tensor:
    """Resolve one scalar for every ordered joint name."""

    output = []
    for joint_name in joint_names:
        if regex_keys:
            matches = [
                float(value)
                for pattern, value in values.items()
                if re.fullmatch(pattern, joint_name)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"expected one action scale for joint {joint_name}, "
                    f"found {len(matches)}"
                )
            output.append(matches[0])
        else:
            if joint_name not in values:
                raise ValueError(f"missing default angle for {joint_name}")
            output.append(float(values[joint_name]))
    return torch.tensor(output, dtype=torch.float32)


def _policy_joint_order(
    config_joint_names: Sequence[str],
    scene_joint_names: Sequence[str],
) -> tuple[str, ...]:
    """Validate the simulator joint set without changing its action order.

    The config and exported ONNX can list joints in different orders. The
    controller checks scene order against ONNX metadata separately; reordering
    scene tensors into config order corrupts proprioception and actions.
    """

    configured = tuple(config_joint_names)
    scene = tuple(scene_joint_names)
    if (
        len(scene) != len(configured)
        or len(scene) != len(set(scene))
        or set(scene) != set(configured)
    ):
        raise ValueError("scene joints must exactly match saved robot joints")
    return scene


def expand_five_point_reference(
    position: torch.Tensor,
    quaternion_wxyz: torch.Tensor,
    command_link_names: Sequence[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand `[B,F,5]` poses to the masked HoloMotion link schema."""

    if position.ndim != 4 or position.shape[2:] != (5, 3):
        raise ValueError("position must have shape [B,F,5,3]")
    batch, frames = position.shape[:2]
    if quaternion_wxyz.shape != (batch, frames, 5, 4):
        raise ValueError("quaternion_wxyz must have shape [B,F,5,4]")
    if not bool(torch.isfinite(position).all()):
        raise ValueError("five-point positions must be finite")
    quaternion_wxyz = normalize_quaternion_wxyz(quaternion_wxyz)
    command_names = tuple(command_link_names)
    if len(command_names) != len(set(command_names)):
        raise ValueError("command link names must be unique")
    missing = sorted(set(FIVE_POINT_LINK_NAMES) - set(command_names))
    if missing:
        raise ValueError(f"command schema misses five-point links: {missing}")

    links = len(command_names)
    full_position = position.new_zeros((batch, frames, links, 3))
    full_quaternion = position.new_zeros((batch, frames, links, 4))
    full_quaternion[..., 0] = 1.0
    link_mask = torch.zeros(links, dtype=torch.bool, device=position.device)
    for source, name in enumerate(FIVE_POINT_LINK_NAMES):
        destination = command_names.index(name)
        full_position[:, :, destination] = position[:, :, source]
        full_quaternion[:, :, destination] = quaternion_wxyz[:, :, source]
        link_mask[destination] = True
    return full_position, full_quaternion, link_mask


def projected_gravity_from_root(
    root_quaternion_wxyz: torch.Tensor,
) -> torch.Tensor:
    """Return the unit gravity vector in the measured pelvis frame."""

    root_quaternion_wxyz = normalize_quaternion_wxyz(root_quaternion_wxyz)
    gravity = root_quaternion_wxyz.new_tensor((0.0, 0.0, -1.0))
    gravity = gravity.expand(root_quaternion_wxyz.shape[0], -1)
    return quat_apply_wxyz(quat_conjugate_wxyz(root_quaternion_wxyz), gravity)


def controller_parameters(metadata):
    """Read the deployment values embedded in the public ONNX."""
    names = tuple(metadata["joint_names"].split(","))
    if len(names) != 29 or len(set(names)) != 29:
        raise ValueError("Expected 29 unique ONNX joint names")
    values = []
    for key in ("default_joint_pos", "action_scale"):
        value = torch.tensor(
            [float(v) for v in metadata[key].split(",")], dtype=torch.float32
        )
        if value.shape != (29,) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid ONNX controller metadata: {key}")
        if key == "action_scale" and not (value > 0).all():
            raise ValueError("ONNX action scales must be positive")
        values.append(value)
    return names, *values


class FrozenHoloMotionV2Controller:
    """Batched deterministic v2 controller with explicit recurrent state."""

    def __init__(
        self,
        *,
        config_path: str | Path,
        lower_session,
        robot_file_path: str | Path,
        batch_size: int,
        scene_joint_names: Sequence[str],
        device: torch.device | str,
        fk_cuda_graph: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.config_path = str(Path(config_path).resolve(strict=True))
        self.robot_file_path = str(Path(robot_file_path).resolve(strict=True))
        self.config = OmegaConf.load(self.config_path)
        self.command_link_names = tuple(
            self.config.link_observation.link_names
        )
        self.frame_offsets = tuple(self.config.link_observation.frame_offsets)
        if self.frame_offsets != (1, 2, 3, 4, 5):
            raise ValueError("boxing requires HoloMotion frame offsets 1..5")
        if not bool(self.config.link_observation.include_error_terms):
            raise ValueError("saved boxing controller requires link errors")

        self.scene_joint_names = tuple(scene_joint_names)
        self.actor_joint_names = _policy_joint_order(
            self.config.robot.dof_names, self.scene_joint_names
        )
        metadata = lower_session.metadata
        names, joint_position, action_scale = controller_parameters(metadata)
        if names != self.actor_joint_names:
            raise ValueError("ONNX and simulation joint orders differ")
        self.default_joint_position = joint_position.to(self.device)
        self.action_scale = action_scale.to(self.device)

        scale_values = OmegaConf.to_container(
            self.config.robot.actuators.action_scale, resolve=True
        )
        config_action_scale = _values_by_joint_name(
            self.actor_joint_names,
            scale_values,
            regex_keys=True,
        )
        if not torch.allclose(
            config_action_scale,
            self.action_scale.cpu(),
            atol=5.0e-4,
            rtol=0.0,
        ):
            raise ValueError(
                "deployment action scale differs from saved config"
            )

        self.fk = SharedLinkFK(
            self.robot_file_path,
            self.device,
            joint_names=self.actor_joint_names,
        )
        missing_bodies = sorted(
            set(self.command_link_names) - set(self.fk.body_names)
        )
        if missing_bodies:
            raise ValueError(
                f"robot file misses command links: {missing_bodies}"
            )
        self.command_body_indices = torch.tensor(
            [
                self.fk.body_names.index(name)
                for name in self.command_link_names
            ],
            dtype=torch.long,
            device=self.device,
        )
        self._fk_call = self.fk
        if fk_cuda_graph and self.device.type == "cuda":
            from holoboxing.training.envs.cuda_graph import InferenceGraph

            self._fk_call = InferenceGraph(self.fk)

        layout = link_actor_layout(
            self.config.link_observation,
            self.config.modules.actor.obs_schema,
        )
        self.actor_observation_dim = int(layout["actor_dim"])
        if (
            self.actor_observation_dim != 1378
            or lower_session.batch_size != self.batch_size
        ):
            raise ValueError(
                "ONNX and controller observation/batch dimensions differ"
            )
        self.module = lower_session
        self.context_length = int(metadata["rope_max_seq_len"])
        if self.context_length < 32:
            raise ValueError(
                "Public evaluator requires a 32-frame lower context"
            )
        cache_shape = (1, 2, self.batch_size, 32, 4, 64)
        self.past_key_values = torch.zeros(
            cache_shape, dtype=torch.float32, device=self.device
        )
        self.step_index = torch.zeros(
            self.batch_size, dtype=torch.long, device=self.device
        )
        self.last_action = torch.zeros(
            self.batch_size,
            len(self.actor_joint_names),
            dtype=torch.float32,
            device=self.device,
        )

    def reset(self, batch_indices: torch.Tensor | None = None) -> None:
        """Clear action history and recurrent cache for selected rows."""

        if batch_indices is None:
            batch_indices = torch.arange(self.batch_size, device=self.device)
        batch_indices = torch.as_tensor(
            batch_indices, dtype=torch.long, device=self.device
        )
        self.past_key_values[:, :, batch_indices] = 0.0
        self.step_index[batch_indices] = 0
        self.last_action[batch_indices] = 0.0

    def __call__(
        self,
        *,
        reference_position_w: torch.Tensor,
        reference_quaternion_wxyz: torch.Tensor,
        current_reference_root_position_w: torch.Tensor,
        reference_root_height: torch.Tensor,
        robot_root_quaternion_wxyz: torch.Tensor,
        robot_root_angular_velocity_b: torch.Tensor,
        robot_joint_position: torch.Tensor,
        robot_joint_velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Return absolute scene-ordered joint position targets."""

        expected_batch = self.batch_size
        current_shapes = {
            "current_reference_root_position_w": (
                current_reference_root_position_w,
                (expected_batch, 3),
            ),
            "reference_root_height": (
                reference_root_height,
                (expected_batch, 1),
            ),
            "robot_root_quaternion_wxyz": (
                robot_root_quaternion_wxyz,
                (expected_batch, 4),
            ),
            "robot_root_angular_velocity_b": (
                robot_root_angular_velocity_b,
                (expected_batch, 3),
            ),
            "robot_joint_position": (
                robot_joint_position,
                (expected_batch, 29),
            ),
            "robot_joint_velocity": (
                robot_joint_velocity,
                (expected_batch, 29),
            ),
        }
        for name, (value, shape) in current_shapes.items():
            if value.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")

        reference_position_w = reference_position_w.to(self.device)
        reference_quaternion_wxyz = reference_quaternion_wxyz.to(self.device)
        full_position, full_quaternion, link_mask = (
            expand_five_point_reference(
                reference_position_w,
                reference_quaternion_wxyz,
                self.command_link_names,
            )
        )
        actor_joint_position = robot_joint_position.to(self.device)
        actor_joint_velocity = robot_joint_velocity.to(self.device)
        actual_position, actual_quaternion = self._fk_call(
            actor_joint_position
        )
        actual_position = actual_position.index_select(
            1, self.command_body_indices
        )
        actual_quaternion = actual_quaternion.index_select(
            1, self.command_body_indices
        )
        robot_root_quaternion_wxyz = normalize_quaternion_wxyz(
            robot_root_quaternion_wxyz.to(self.device)
        )
        command = build_link_observation(
            LinkObservationInput(
                reference_link_pos=full_position,
                reference_link_quat_wxyz=full_quaternion,
                reference_root_pos=current_reference_root_position_w.to(
                    self.device
                ),
                robot_root_quat_wxyz=robot_root_quaternion_wxyz,
                robot_link_pos_local=actual_position,
                robot_link_quat_local_wxyz=actual_quaternion,
                frame_offsets=torch.tensor(
                    self.frame_offsets,
                    dtype=torch.float32,
                    device=self.device,
                ),
                link_mask=link_mask,
                time_valid=torch.ones(
                    len(self.frame_offsets),
                    dtype=torch.bool,
                    device=self.device,
                ),
                reference_root_height=reference_root_height.to(self.device),
            ),
            root_link_index=self.command_link_names.index("pelvis"),
            include_error_terms=True,
        )
        proprioception = torch.cat(
            (
                projected_gravity_from_root(robot_root_quaternion_wxyz),
                robot_root_angular_velocity_b.to(self.device),
                actor_joint_position - self.default_joint_position,
                actor_joint_velocity,
                self.last_action,
            ),
            dim=-1,
        )
        observation = pack_link_actor_observation(
            proprioception,
            command,
            include_error_terms=True,
            include_future_link_mask=False,
        )
        with torch.inference_mode():
            result = self.module(
                observation,
                self.past_key_values,
                self.step_index,
            )
        action, present_key_values = result
        if not bool(
            torch.isfinite(action).all()
            and torch.isfinite(present_key_values).all()
        ):
            raise RuntimeError("HoloMotion actor returned non-finite values")
        self.last_action.copy_(action)
        self.past_key_values.copy_(present_key_values)
        self.step_index.add_(1)
        reset = self.step_index >= 32
        self.past_key_values[:, :, reset] = 0
        self.step_index.masked_fill_(reset, 0)
        target_actor_order = (
            self.default_joint_position + self.action_scale * action
        )
        return target_actor_order


__all__ = [
    "FrozenHoloMotionV2Controller",
    "expand_five_point_reference",
    "projected_gravity_from_root",
]


class CommonLower:
    def __init__(self, env, controllers, agents, action_dim=225):
        if action_dim != 225:
            raise ValueError("Only C225 is supported")
        self.action_dim = action_dim
        self.env, self.controllers, self.agents = (
            env,
            controllers,
            tuple(agents),
        )
        if len(self.agents) != 2 or set(controllers) != set(self.agents):
            raise ValueError(
                "Expected the two declared frozen lower controllers"
            )

    def autonomous(self, measured, physical, *, reference_override=None):
        """Only current measurements and already de-normalized upper predictions."""
        count = self.env.num_envs * 2
        if (
            physical.shape != (count, self.action_dim)
            or not torch.isfinite(physical).all()
        ):
            raise ValueError(
                f"Expected finite paired physical {self.action_dim}D references"
            )
        p, r = decode_output(
            measured["position"],
            measured["quaternion"],
            physical,
            action_dim=self.action_dim,
        )
        q = F.normalize(quat_from_matrix(r), dim=-1)
        anchor = measured["position"][:, 0].clone()
        ground = self.env.scene.env_origins[:, 2:3].repeat_interleave(2, dim=0)
        height = anchor[:, 2:3] - ground
        if reference_override is not None:
            p, q, anchor, height = reference_override.apply(
                measured, p, q, anchor, height
            )
        return self._execute(measured, p, q, anchor, height)

    def _execute(self, measured, position, quaternion, anchor, height):
        actions, targets = {}, {}
        for role, name in enumerate(self.agents):
            ids = slice(role, None, 2)
            robot = self.env.scene[name]
            target = self.controllers[name](
                reference_position_w=position[ids],
                reference_quaternion_wxyz=quaternion[ids],
                current_reference_root_position_w=anchor[ids],
                reference_root_height=height[ids],
                robot_root_quaternion_wxyz=measured["quaternion"][ids, 0],
                robot_root_angular_velocity_b=measured["omega_b"][ids],
                robot_joint_position=robot.data.joint_pos,
                robot_joint_velocity=robot.data.joint_vel,
            )
            term = self.env.action_managers[name].get_term("joint_pos")
            actions[name] = (target - term._offset) / term._scale
            targets[name] = target
        return actions, {
            "world_position": position,
            "world_quaternion": quaternion,
            "reference_anchor": anchor,
            "reference_height": height,
            "joint_target": targets,
        }

    def reset(self, done):
        if done.shape != (self.env.num_envs,) or done.dtype != torch.bool:
            raise ValueError("Expected paired reset mask")
        for controller in self.controllers.values():
            controller.reset(done.nonzero().flatten())
