"""Bounded frozen self-play snapshots; collection never trains the opponent."""

import copy

import torch

from holoboxing.training.motion_prior import state_sha256


def cpu_state(module):
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


class OpponentPool:
    def __init__(self, policy, normalizers, *, seed, capacity=8, interval=256):
        if (
            type(capacity) is not int
            or capacity < 2
            or type(interval) is not int
            or interval < 1
        ):
            raise ValueError(
                "Self-play needs a positive interval and at least two slots"
            )
        self.policy, self.normalizers = policy, normalizers
        self.capacity, self.interval = capacity, interval
        self.generator = torch.Generator().manual_seed(seed)
        self.snapshots = []
        self.selected = None
        self.next_refresh = 0

    def _snapshot(self, iteration):
        actor = cpu_state(self.policy.taskpolicy_ego)
        normalizer = cpu_state(self.normalizers["actor_ego"])
        return {
            "iteration": iteration,
            "actor": actor,
            "normalizer": normalizer,
            "sha256": state_sha256(
                {**actor, **{"norm." + k: v for k, v in normalizer.items()}}
            ),
        }

    def add_checkpoint(self, path, identity):
        """Import a trusted, compatible frozen opponent and retain it on eviction."""
        from holoboxing.training.rsl_rl.runners.resume import (
            validate_module_state,
        )
        from holoboxing.training.rsl_rl.modules.normalization import (
            validate_statistics,
        )
        from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
        from holoboxing.evaluation.runtime.model_contract import sha256

        if self.selected is not None:
            raise ValueError("Import opponents before the first rollout")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("identity", {}).get("resources") != identity["resources"]:
            raise ValueError("Opponent resources differ")
        if saved.get("policy_spec") != getattr(
            self.policy, "policy_spec", None
        ):
            raise ValueError(
                "Opponent representation/observation contract differs"
            )
        actor = {
            k.removeprefix("taskpolicy_ego."): v
            for k, v in saved["policy"].items()
            if k.startswith("taskpolicy_ego.")
        }
        norm = {
            k.removeprefix("actor_ego."): v
            for k, v in saved["normalizers"].items()
            if k.startswith("actor_ego.")
        }
        if hasattr(
            self.policy.taskpolicy_opponent, "context_policy"
        ) and not self.policy.policy_spec.get("match_context", False):
            actor.setdefault(
                "context_policy.weight",
                torch.zeros_like(
                    self.policy.taskpolicy_opponent.context_policy.weight
                ).cpu(),
            )
        require_finite((actor, norm), "external opponent")
        validate_module_state(
            self.policy.taskpolicy_opponent, actor, "opponent"
        )
        for (
            name,
            value,
        ) in self.policy.taskpolicy_opponent.state_dict().items():
            if name.startswith("decoder.") and not torch.equal(
                actor[name].to(value), value
            ):
                raise ValueError("Opponent frozen decoder differs")
        validate_module_state(
            self.normalizers["actor_opponent"], norm, "opponent stats"
        )
        validate_statistics(norm)
        if int(norm["count"]) <= 0:
            raise ValueError("Opponent statistics must be fitted")
        digest = state_sha256(
            {**actor, **{"norm." + k: v for k, v in norm.items()}}
        )
        if any(x["sha256"] == digest for x in self.snapshots):
            raise ValueError("Duplicate opponent")
        if len(self.snapshots) >= self.capacity - 2:
            raise ValueError(
                "Reserve two pool slots for initial and recent learner"
            )
        self.snapshots.append(
            dict(
                iteration=0,
                actor=actor,
                normalizer=norm,
                sha256=digest,
                external=True,
                checkpoint_sha256=sha256(path),
            )
        )

    def _install(self):
        snapshot = self.snapshots[self.selected]
        self.policy.taskpolicy_opponent.load_state_dict(
            snapshot["actor"], strict=True
        )
        self.normalizers["actor_opponent"].load_state_dict(
            snapshot["normalizer"], strict=True
        )

    def prepare(self, iteration):
        """Change at update boundaries; the caller resets physical episodes."""
        if self.selected is not None and iteration < self.next_refresh:
            return False
        if (
            not self.snapshots
            or self.snapshots[-1].get("external")
            or self.snapshots[-1]["iteration"] != iteration
        ):
            self.snapshots.append(self._snapshot(iteration))
            if len(self.snapshots) > self.capacity:
                # Retain the initial opponent alongside recent history.
                dynamic = [
                    i
                    for i, x in enumerate(self.snapshots)
                    if not x.get("external")
                ]
                del self.snapshots[dynamic[1]]
        self.selected = int(
            torch.randint(len(self.snapshots), (), generator=self.generator)
        )
        self.next_refresh = iteration + self.interval
        self._install()
        return True

    def state_dict(self):
        return copy.deepcopy(
            {
                "capacity": self.capacity,
                "interval": self.interval,
                "snapshots": self.snapshots,
                "selected": self.selected,
                "next_refresh": self.next_refresh,
                "rng": self.generator.get_state(),
            }
        )

    def validate(self, state):
        from holoboxing.training.rsl_rl.algorithms.ppo import require_finite
        from holoboxing.training.rsl_rl.modules.normalization import (
            validate_statistics,
        )
        from holoboxing.training.rsl_rl.runners.resume import (
            validate_module_state,
        )

        require_finite(state, "self-play pool")
        if (
            state["capacity"] != self.capacity
            or state["interval"] != self.interval
        ):
            raise ValueError("Self-play pool recipe differs")
        snapshots = state["snapshots"]
        selected = state["selected"]
        if not isinstance(snapshots, list) or len(snapshots) > self.capacity:
            raise ValueError("Invalid self-play pool size")
        if selected is None:
            if (
                any(not x.get("external") for x in snapshots)
                or state["next_refresh"] != 0
            ):
                raise ValueError("Invalid uninitialized self-play pool")
        elif type(selected) is not int or not 0 <= selected < len(snapshots):
            raise ValueError("Invalid selected self-play opponent")
        iterations = []
        for snapshot in snapshots:
            validate_module_state(
                self.policy.taskpolicy_opponent,
                snapshot["actor"],
                "pool actor",
            )
            validate_module_state(
                self.normalizers["actor_opponent"],
                snapshot["normalizer"],
                "pool statistics",
            )
            validate_statistics(snapshot["normalizer"])
            digest = state_sha256(
                {
                    **snapshot["actor"],
                    **{
                        "norm." + k: v
                        for k, v in snapshot["normalizer"].items()
                    },
                }
            )
            if snapshot["sha256"] != digest:
                raise ValueError("Self-play snapshot hash differs")
            iteration = snapshot["iteration"]
            if type(iteration) is not int or iteration < 0:
                raise ValueError("Invalid self-play snapshot iteration")
            if not snapshot.get("external"):
                iterations.append(iteration)
        if iterations != sorted(set(iterations)):
            raise ValueError("Self-play snapshot order differs")
        if type(state["next_refresh"]) is not int or state["next_refresh"] < 0:
            raise ValueError("Invalid self-play refresh boundary")
        torch.Generator().set_state(state["rng"])

    def load_state_dict(self, state):
        self.validate(state)
        self.snapshots = copy.deepcopy(state["snapshots"])
        self.selected, self.next_refresh = (
            state["selected"],
            state["next_refresh"],
        )
        self.generator.set_state(state["rng"])
        if self.selected is not None:
            self._install()
