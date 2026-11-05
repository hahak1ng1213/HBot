"""Correctness-first batched adapter to the unchanged competition referee.

Transfers legal contact evidence to CPU each substep. Benchmark this reference
implementation before large runs; it intentionally shares the actual referee.
"""

import numpy as np

from holoboxing.evaluation.runtime.contact_geometry import (
    opponent_target_semantic,
    strike_geometry,
)
from holoboxing.evaluation.runtime.hand_hit_rules import target_part
from holoboxing.evaluation.runtime.rules import MatchStateMachine


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

    def contact_metrics(self):
        """Summarize the last observed control frame without changing scoring.

        Counts are legal hand/physics-step samples, not contact points or hits.
        Multiple contact points on the same hand count once, including when
        only one of those points carries the referee's HIT event. Impulse is
        the referee's actual rolling, consumed-on-hit 20 ms normal impulse;
        the peak is over hands and physics substeps in this control frame.
        Call before resetting terminal worlds. Returned arrays own their data.
        """
        metrics = {
            name: np.zeros((len(self.matches), 2), dtype=np.float64)
            for name in (
                "official_legal_contact_samples",
                "official_torso_contact_samples",
                "official_left_hits",
                "official_right_hits",
                "official_subthreshold_contact_samples",
                "official_cooldown_contact_samples",
                "official_contact_impulse_max_ns",
            )
        }
        for world, match in enumerate(self.matches):
            samples = {}
            for row in match.contact_diagnostics:
                if row["hand"] not in ("left", "right"):
                    continue
                key = (row["actor"], row["hand"], row["physics_time_s"])
                samples.setdefault(key, []).append(row)
            for (actor, hand, _), rows in samples.items():
                metrics["official_legal_contact_samples"][world, actor] += 1
                if any(row["body_part"] == "torso" for row in rows):
                    metrics["official_torso_contact_samples"][
                        world, actor
                    ] += 1
                impulse = max(row["normal_impulse_ns"] for row in rows)
                peak = metrics["official_contact_impulse_max_ns"]
                peak[world, actor] = max(peak[world, actor], impulse)
                if any(row["valid"] for row in rows):
                    metrics[f"official_{hand}_hits"][world, actor] += 1
                    continue
                reason = rows[0]["reason"]
                if reason == "impulse_below_threshold":
                    metrics["official_subthreshold_contact_samples"][
                        world, actor
                    ] += 1
                elif reason == "hand_cooldown":
                    metrics["official_cooldown_contact_samples"][
                        world, actor
                    ] += 1
        return metrics

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
    """Deduplicate solver contacts and integrate positive normal force."""
    result = [[] for _ in range(num_worlds)]
    for c in range(count):
        w = int(worlds[c])
        g0, g1 = (int(x) for x in pairs[c])
        if not 0 <= w < num_worlds or min(g0, g1) < 0:
            continue
        geoms = (g0, g1)
        for side in (0, 1):
            strike = strike_geometry(model, geoms[side])
            if strike is None or strike[1] not in ("left_hand", "right_hand"):
                continue
            actor, own = strike
            target = opponent_target_semantic(model, actor, geoms[1 - side])
            if target is None or target_part(target[2]) is None:
                continue
            normal_force = float(np.dot(forces[c, :3], normals[c]))
            if not np.isfinite(normal_force):
                raise FloatingPointError("Nonfinite contact evidence")
            result[w].append(
                dict(
                    attacker=actor,
                    defender=1 - actor,
                    attacking_body=own,
                    target_body=target[2],
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
