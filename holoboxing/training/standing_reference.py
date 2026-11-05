"""Hold the opponent's neutral world targets until episode reset."""

import torch
from torch.nn import functional as F

from holoboxing.training.tasks.combat.mdp.geometry import decode_output
from holomotion.src.utils.torch_utils import matrix_from_quat, quat_from_matrix


class WorldStandingReference:
    def __init__(
        self,
        neutral,
        num_envs,
        device,
        *,
        role=1,
        wrist_position=None,
        wrist_quaternion=None,
    ):
        self.neutral = torch.as_tensor(neutral, device=device).float()
        if (
            self.neutral.shape != (225,)
            or not torch.isfinite(self.neutral).all()
        ):
            raise ValueError(
                "Standing reference requires a finite neutral 225D pose"
            )
        if role not in (0, 1) or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Invalid standing role or environment count")
        self.role, self.num_envs = role, num_envs
        self.wrist_position = self.wrist_rotation = None
        if (wrist_position is None) != (wrist_quaternion is None):
            raise ValueError("Supply both wrist position and quaternion")
        if wrist_position is not None:
            p = torch.as_tensor(wrist_position, device=device).float()
            q = torch.as_tensor(wrist_quaternion, device=device).float()
            if p.shape != (2, 3) or q.shape != (2, 4):
                raise ValueError("Expected two pelvis-local wrist poses")
            if (
                not torch.isfinite(p).all()
                or not torch.isfinite(q).all()
                or (q.norm(dim=-1) < 1e-6).any()
            ):
                raise ValueError("Invalid wrist pose")
            self.wrist_position = p.clone()
            self.wrist_rotation = matrix_from_quat(F.normalize(q, dim=-1))
        self.pending = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.position = torch.empty(num_envs, 5, 5, 3, device=device)
        self.quaternion = torch.empty(num_envs, 5, 5, 4, device=device)
        self.anchor = torch.empty(num_envs, 3, device=device)
        self.height = torch.empty(num_envs, 1, device=device)

    def reset(self, ids):
        self.pending[ids] = True

    @torch.no_grad()
    def apply(self, measured, position, quaternion, anchor, height):
        ids = self.pending.nonzero().flatten()
        rows = 2 * ids + self.role
        if len(ids):
            p, r = decode_output(
                measured["position"][rows],
                measured["quaternion"][rows],
                self.neutral.expand(len(ids), -1),
                action_dim=225,
            )
            if self.wrist_position is not None:
                # Anchor the guard once in world space, along with the feet.
                # Contact displacement must not recenter the wrist targets.
                p[:, :, 1:3] = p[:, :, :1] + torch.einsum(
                    "ntij,bj->ntbi", r[:, :, 0], self.wrist_position
                )
                r[:, :, 1:3] = r[:, :, :1] @ self.wrist_rotation
            self.position[ids] = p
            self.quaternion[ids] = F.normalize(quat_from_matrix(r), dim=-1)
            self.anchor[ids] = anchor[rows]
            self.height[ids] = height[rows]
            self.pending[ids] = False
        role_rows = slice(self.role, None, 2)
        # Never recenter these targets on a robot displaced by contact.
        for output, fixed in (
            (position, self.position),
            (quaternion, self.quaternion),
            (anchor, self.anchor),
            (height, self.height),
        ):
            output[role_rows] = fixed
        return position, quaternion, anchor, height
