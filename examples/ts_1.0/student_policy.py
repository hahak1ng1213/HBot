"""Self-contained upper-policy distillation; no lower weights or hidden state."""
import torch
from torch import nn
from holoboxing.training.tasks.combat.mdp.geometry import project_xz, rot_xz


class StandardPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_mean', torch.zeros(1095))
        self.register_buffer('input_scale', torch.ones(1095))
        self.register_buffer('output_mean', torch.zeros(225))
        self.register_buffer('output_scale', torch.ones(225))
        self.network = nn.Sequential(nn.Linear(1095, 512), nn.ELU(),
                                     nn.Linear(512, 512), nn.ELU(),
                                     nn.Linear(512, 256), nn.ELU(), nn.Linear(256, 225))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def unprojected(self, obs):
        # Ten measured own states, then five opponent points. Do not learn a
        # time-indexed replay from the clock or fit the referee's score.
        x = ((obs[:, :1095] - self.input_mean) / self.input_scale).clamp(-10, 10)
        return self.baseline(obs) + self.output_mean + self.output_scale * self.network(x)

    @staticmethod
    def baseline(obs):
        # Public own history is in the current root HEADING frame, whereas
        # C225 is in its FULL pelvis frame. Undo the measured root tilt.
        current = obs[:, 972:1080][:, 3:48].reshape(-1, 5, 9)
        inv = project_xz(current[:, 0, 3:]).transpose(-1, -2)
        p = (inv[:, None] @ (current[..., :3]-current[:, :1, :3])[..., None]).squeeze(-1)
        x = (inv[:, None] @ current[..., 3:6, None]).squeeze(-1)
        z = (inv[:, None] @ current[..., 6:9, None]).squeeze(-1)
        return torch.cat((p,x,z),-1)[:,None].expand(-1,5,-1,-1).reshape(-1,225)

    def forward(self, obs):
        p = self.unprojected(obs).reshape(-1, 5, 5, 9)
        return torch.cat((p[..., :3], rot_xz(project_xz(p[..., 3:]))), -1).flatten(1)


def load(path, device='cpu'):
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved.get('format')!='hbot_four_punch_student_v1':
        raise ValueError('Unexpected student checkpoint format')
    model = StandardPolicy()
    model.load_state_dict(saved['model'], strict=True)
    return model.to(device).eval().requires_grad_(False)
