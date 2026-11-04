"""Integrate actual per-physics-step fist contacts, independently of scoring.

Rows are END-of-step timestamps. Forces must already be summed over all
solver contact points for each hand/target category, without thresholding.
The vector force must be expressed in world coordinates and act ON the target.
"""

from __future__ import annotations
import math

import numpy as np

HANDS = ("left", "right")
TARGET_CATEGORIES = ("legal", "unscored")


def _rolling_integral(forces, count, dt):
    cumulative = np.r_[0.0, np.cumsum(forces, dtype=np.float64)]
    end = np.arange(1, len(forces) + 1)
    return (cumulative[end] - cumulative[np.maximum(0, end - count)]) * dt


def _contact_spans(contact, max_gap_steps):
    active = np.flatnonzero(contact)
    if not len(active):
        return []
    cuts = np.flatnonzero(np.diff(active) > max_gap_steps + 1) + 1
    return [
        (int(group[0]), int(group[-1])) for group in np.split(active, cuts)
    ]


def analyze_impulses(
    times_s,
    normal_force_n,
    world_force_n,
    phases,
    *,
    dt_s=0.005,
    window_s=0.020,
    max_gap_steps=0,
):
    """Return JSON-serializable totals and contiguous contact episodes.

    Inputs: t[N], normal_force[N,2,2], world_force[N,2,2,3], phases[N].
    Axes are hand=(left,right), target=(legal,unscored).
    Default episodes split on ANY step with zero positive normal force.
    ``max_gap_steps=1`` merges across <=1 absent 5-ms sample at default dt;
    gap duration is disclosed and never adds fictitious contact impulse.

    The rolling-window peak is UNCONSUMED physical impulse. It is not an
    official hit count: scoring consumes windows and applies other rules.
    """
    times = np.asarray(times_s, dtype=np.float64)
    normal = np.asarray(normal_force_n, dtype=np.float64)
    world = np.asarray(world_force_n, dtype=np.float64)
    phase = np.asarray(phases, dtype=np.float64)
    if times.ndim != 1 or not len(times):
        raise ValueError("Require nonempty end-of-step timestamps [N]")
    n = len(times)
    if normal.shape != (n, 2, 2) or world.shape != (n, 2, 2, 3):
        raise ValueError("Require force axes [N,hand,target,(xyz)]")
    if phase.shape != (n,):
        raise ValueError("Require source phase [N]")
    if any(
        not np.isfinite(value).all() for value in (times, normal, world, phase)
    ):
        raise ValueError("Nonfinite contact evidence")
    if (normal < 0).any():
        raise ValueError("Normal force must sum nonnegative contact normals")
    if not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("dt_s must be positive")
    if not math.isfinite(window_s) or window_s < dt_s:
        raise ValueError("window_s must be at least one physical step")
    count = round(window_s / dt_s)
    if not math.isclose(count * dt_s, window_s, abs_tol=1e-10):
        raise ValueError("window_s must contain an integer number of steps")
    if type(max_gap_steps) is not int or max_gap_steps < 0:
        raise ValueError("max_gap_steps must be a nonnegative integer")
    if len(times) > 1 and not np.allclose(
        np.diff(times), dt_s, rtol=1e-6, atol=1e-9
    ):
        raise ValueError(
            "Must sample every physical step at the declared dt_s"
        )
    if (np.linalg.norm(world, axis=-1)[normal == 0] > 1e-7).any():
        raise ValueError("Nonzero contact vector with zero normal force")

    records = []
    for hand, hand_name in enumerate(HANDS):
        for target, target_name in enumerate(TARGET_CATEGORIES):
            forces = normal[:, hand, target]
            vectors = world[:, hand, target]
            contact = forces > 0
            rolling = _rolling_integral(forces, count, dt_s)
            peak_index = int(np.argmax(rolling))
            episodes = []
            for start, end in _contact_spans(contact, max_gap_steps):
                sl = slice(start, end + 1)
                local_peak = int(np.argmax(forces[sl])) + start
                vector_impulse = vectors[sl].sum(axis=0) * dt_s
                active_count = int(contact[sl].sum())
                local_rolling = _rolling_integral(forces[sl], count, dt_s)
                episodes.append(
                    {
                        "index": len(episodes),
                        "first_sample_index": start,
                        "last_sample_index": end,
                        "interval_start_s": float(times[start] - dt_s),
                        "interval_end_s": float(times[end]),
                        "duration_s": (end - start + 1) * dt_s,
                        "positive_force_duration_s": active_count * dt_s,
                        "gap_steps": end - start + 1 - active_count,
                        "first_source_phase": float(phase[start]),
                        "last_source_phase": float(phase[end]),
                        "negative_phase_contact_samples": int(
                            np.count_nonzero(contact[sl] & (phase[sl] < 0))
                        ),
                        "normal_impulse_ns": float(forces[sl].sum() * dt_s),
                        "rolling_window_peak_ns": float(local_rolling.max()),
                        "peak_normal_force_n": float(forces[local_peak]),
                        "peak_normal_force_time_s": float(times[local_peak]),
                        "world_impulse_on_target_ns": vector_impulse.tolist(),
                        "world_impulse_norm_ns": float(
                            np.linalg.norm(vector_impulse)
                        ),
                    }
                )
            vector_impulse = vectors.sum(axis=0) * dt_s
            records.append(
                {
                    "hand": hand_name,
                    "target_category": target_name,
                    "contact_samples": int(contact.sum()),
                    "positive_force_duration_s": int(contact.sum()) * dt_s,
                    "normal_impulse_ns": float(forces.sum() * dt_s),
                    "rolling_window_peak_ns": float(rolling[peak_index]),
                    "rolling_window_peak_time_s": float(times[peak_index])
                    if contact.any()
                    else None,
                    "peak_normal_force_n": float(forces.max()),
                    "world_impulse_on_target_ns": vector_impulse.tolist(),
                    "world_impulse_norm_ns": float(
                        np.linalg.norm(vector_impulse)
                    ),
                    "episodes": episodes,
                }
            )
    return {
        "format": "measured_hand_contact_impulse_v1",
        "sample_count": n,
        "dt_s": dt_s,
        "integration_rule": "sum(force_at_each_physics_step * dt_s)",
        "normal_impulse_definition": (
            "sum positive contact-normal magnitudes before integration; "
            "excludes friction"
        ),
        "world_impulse_definition": (
            "integral of vector force on target in world frame; "
            "includes tangential force"
        ),
        "rolling_window_s": window_s,
        "rolling_window_consumed_by_scoring": False,
        "max_gap_steps": max_gap_steps,
        "max_merged_gap_s": max_gap_steps * dt_s,
        "episode_rule": (
            "positive normal force; split after more than "
            "max_gap_steps absent samples"
        ),
        "episode_window_restarts_at_event_boundary": True,
        "official_effective_hit_count_claimed": False,
        "records": records,
    }
