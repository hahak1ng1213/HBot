"""Training-only reset distances, promoted by completed scored bouts."""

from collections import deque

import torch


class DistanceCurriculum:
    distances = (1.0, 1.4, 1.8, None)
    window = 16
    version = "scored_distance_v1"

    def __init__(self, worlds, device):
        self.stage = 0
        self.outcomes = deque(maxlen=self.window)
        self.hits = torch.zeros(worlds, device=device)
        self.fell = torch.zeros(worlds, dtype=torch.bool, device=device)
        self.episode_stage = torch.zeros(
            worlds, dtype=torch.long, device=device
        )

    def reset(self, ids, qpos, root_indices):
        distance = self.distances[self.stage]
        if distance is not None:
            a, b = root_indices
            pa, pb = qpos[ids, a : a + 2].clone(), qpos[ids, b : b + 2].clone()
            direction = torch.nn.functional.normalize(pb - pa, dim=-1)
            center = (pa + pb) / 2
            qpos[ids, a : a + 2] = center - direction * distance / 2
            qpos[ids, b : b + 2] = center + direction * distance / 2
        self.hits[ids] = 0
        self.fell[ids] = False
        self.episode_stage[ids] = self.stage

    def observe(self, hits, fallen, done, invalid):
        self.hits += hits
        self.fell |= fallen | invalid
        # Only complete bouts at the current distance count. Opponent refresh
        # and resume reset unfinished bouts without turning them into successes.
        for i in done.nonzero().flatten().tolist():
            if int(self.episode_stage[i]) == self.stage:
                self.outcomes.append(
                    (bool(self.hits[i] > 0), bool(self.fell[i]))
                )
        if (
            len(self.outcomes) == self.window
            and self.stage < len(self.distances) - 1
        ):
            hits, falls = zip(*self.outcomes)
            if sum(hits) >= 12 and sum(falls) <= 1:
                self.stage += 1
                self.outcomes.clear()

    def state_dict(self):
        # Physical episodes restart on resume; retain completed-bout evidence.
        return dict(
            version=self.version,
            stage=self.stage,
            outcomes=list(self.outcomes),
        )

    def load_state_dict(self, state):
        if state["version"] != self.version or not 0 <= state["stage"] < len(
            self.distances
        ):
            raise ValueError("Incompatible distance curriculum")
        self.stage = state["stage"]
        self.outcomes = deque(state["outcomes"], maxlen=self.window)
