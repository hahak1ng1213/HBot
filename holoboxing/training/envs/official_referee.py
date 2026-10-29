"""Correctness-first batched adapter to the unchanged competition referee.

Transfers legal contact evidence to CPU each substep. Benchmark this reference
implementation before large runs; it intentionally shares the actual referee.
"""

import numpy as np

from holoboxing.evaluation.runtime.rules import MatchStateMachine
from holoboxing.evaluation.runtime.hand_hit_rules import (
    hand_index,
    target_part,
)


class OfficialReferees:
    def __init__(self, worlds):
        self.matches = [MatchStateMachine() for _ in range(worlds)]
        self.samples = [[] for _ in range(worlds)]
        self.ticks = np.zeros(worlds, dtype=np.int64)

    def reset(self, ids):
        for i in ids:
            self.matches[int(i)].reset()
            self.samples[int(i)] = []
            self.ticks[int(i)] = 0

    def sample(self, contacts):
        for i, rows in enumerate(contacts):
            self.ticks[i] += 1
            self.samples[i].append((int(self.ticks[i]) / 200, rows))

    def observe(self, root_pos, fallen):
        results, delta = [], []
        for i, match in enumerate(self.matches):
            before = match.score.copy()
            result = match.observe_frame(
                sim_time_s=int(self.ticks[i]) / 200,
                contacts=[],
                root_pos=root_pos[i],
                fallen=fallen[i],
                physics_samples=self.samples[i],
            )
            results.append(result)
            delta.append(match.score - before)
            self.samples[i] = []
        return results, np.asarray(delta)

    def context(self):
        return np.asarray(
            [
                [
                    [
                        max(0, 60 - int(t) / 200),
                        int(m.score[r] - m.score[1 - r]),
                    ]
                    for r in range(2)
                ]
                for t, m in zip(self.ticks, self.matches, strict=True)
            ],
            dtype=np.float32,
        )


def contact_records(
    model,
    worlds,
    pairs,
    positions,
    forces,
    normals,
    distances,
    count,
    num_worlds,
):
    """Keep each solver contact once; integrate only its positive normal force."""
    result = [[] for _ in range(num_worlds)]
    for c in range(count):
        w = int(worlds[c])
        g0, g1 = (int(x) for x in pairs[c])
        if not 0 <= w < num_worlds or min(g0, g1) < 0:
            continue
        names = [model.body(int(model.geom_bodyid[g])).name for g in (g0, g1)]
        for side in (0, 1):
            own, other = names[side], names[1 - side]
            actor = (
                0
                if own.startswith("p1_")
                else 1
                if own.startswith("p2_")
                else None
            )
            if actor is None or not other.startswith(("p2_", "p1_")[actor]):
                continue
            if hand_index(own) is None or target_part(other) is None:
                continue
            normal_force = float(np.dot(forces[c, :3], normals[c]))
            if not np.isfinite(normal_force):
                raise FloatingPointError("Nonfinite contact evidence")
            result[w].append(
                dict(
                    attacker=actor,
                    defender=1 - actor,
                    attacking_body=own,
                    target_body=other,
                    pair_id=c,
                    physical_contact=bool(distances[c] <= 0),
                    normal_force_n=max(0.0, normal_force),
                    sample_dt_s=0.005,
                    peak_force=float(np.linalg.norm(forces[c, :3])),
                    closing_speed=0.0,
                    position=positions[c].tolist(),
                )
            )
    return result
