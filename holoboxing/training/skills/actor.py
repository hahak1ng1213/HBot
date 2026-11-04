"""Small residual skill teacher with explicit raw-action likelihoods."""

import hashlib
from pathlib import Path

import torch
from torch import nn


def network(inputs, outputs, hidden_dims):
    layers = []
    for width in hidden_dims:
        layers.extend((nn.Linear(inputs, width), nn.ELU()))
        inputs = width
    layers.append(nn.Linear(inputs, outputs))
    return nn.Sequential(*layers)


class SkillActor(nn.Module):
    def __init__(self, observation_dim, action_dim=30, hidden_dims=(256, 128)):
        super().__init__()
        if (
            type(observation_dim) is not int
            or observation_dim < 1
            or action_dim != 30
            or not hidden_dims
            or any(
                type(width) is not int or width < 1 for width in hidden_dims
            )
        ):
            raise ValueError("Require positive teacher widths and 30 actions")
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)
        self.actor = network(observation_dim, action_dim, hidden_dims)
        self.critic = network(observation_dim, 1, hidden_dims)
        nn.init.zeros_(self.actor[-1].weight)
        nn.init.zeros_(self.actor[-1].bias)
        self.log_std = nn.Parameter(torch.full((action_dim,), -2.0))

    def distribution(self, observations):
        return torch.distributions.Normal(
            self.actor(observations), self.log_std.clamp(-4, -0.5).exp()
        )

    def act(self, observations, deterministic=False):
        return (
            self.actor(observations)
            if deterministic
            else self.distribution(observations).sample()
        )

    def value(self, observations):
        return self.critic(observations).squeeze(-1)

    def evaluate(self, observations, raw_actions):
        # Likelihoods are for the sampled Gaussian action. The environment
        # applies tanh and physical scales after the raw PPO action is stored.
        distribution = self.distribution(observations)
        return (
            distribution.log_prob(raw_actions).sum(-1),
            distribution.entropy().sum(-1),
            self.value(observations),
        )


def implementation_identity():
    """Fingerprint the modules defining teacher training and execution."""
    directory = Path(__file__).resolve().parent
    return {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in (
            "actor.py",
            "train.py",
            "env.py",
            "physics.py",
            "reference.py",
        )
    }


def load_teacher(path, device="cpu"):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("format") != "solo_skill_teacher_v1":
        raise ValueError("Expected a standalone skill teacher checkpoint")
    if saved.get("implementation_sources") != implementation_identity():
        raise ValueError("Teacher implementation differs from its checkpoint")
    model = SkillActor(
        saved["observation_dim"], saved["action_dim"], saved["hidden_dims"]
    ).to(device)
    model.load_state_dict(saved["model"], strict=True)
    if any(
        not torch.isfinite(value).all()
        for value in model.state_dict().values()
    ):
        raise ValueError("Nonfinite teacher checkpoint")
    model.eval()
    return model, saved
