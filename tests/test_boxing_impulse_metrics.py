"""Analytic checks of substep integration, signs, and episode grouping."""

import unittest

import numpy as np

from holoboxing.experiments.impulse_metrics import analyze_impulses


def packet(n):
    return (
        np.arange(1, n + 1) * 0.005,
        np.zeros((n, 2, 2)),
        np.zeros((n, 2, 2, 3)),
        np.arange(n) // 4,
    )


class ImpulseTests(unittest.TestCase):
    def test_five_ms_peak_not_lost_by_control_frame_sampling(self):
        t, normal, world, phase = packet(4)
        normal[0, 0, 0] = 2000
        world[0, 0, 0, 0] = 2000
        row = analyze_impulses(t, normal, world, phase)["records"][0]
        self.assertEqual(row["normal_impulse_ns"], 10)
        self.assertEqual(row["rolling_window_peak_ns"], 10)
        self.assertEqual(row["peak_normal_force_n"], 2000)
        self.assertEqual(
            normal[-1, 0, 0], 0
        )  # 50-Hz last-step view misses it.

    def test_opposing_vectors_do_not_cancel_normal_impulse(self):
        t, normal, world, phase = packet(2)
        normal[:, 0, 0] = 1000
        world[:, 0, 0, 0] = [1000, -1000]
        row = analyze_impulses(t, normal, world, phase)["records"][0]
        self.assertEqual(row["normal_impulse_ns"], 10)
        self.assertEqual(row["world_impulse_norm_ns"], 0)

    def test_one_absent_step_splits_by_default_or_explicitly_merges(self):
        t, normal, world, phase = packet(3)
        normal[[0, 2], 0, 0] = 100
        world[[0, 2], 0, 0, 0] = 100
        split = analyze_impulses(t, normal, world, phase)["records"][0]
        merged = analyze_impulses(t, normal, world, phase, max_gap_steps=1)[
            "records"
        ][0]
        self.assertEqual(len(split["episodes"]), 2)
        self.assertEqual(len(merged["episodes"]), 1)
        event = merged["episodes"][0]
        self.assertEqual(event["normal_impulse_ns"], 1)
        self.assertEqual(event["positive_force_duration_s"], 0.010)
        self.assertEqual(event["duration_s"], 0.015)
        self.assertEqual(event["gap_steps"], 1)

    def test_window_is_twenty_ms_and_channels_stay_separate(self):
        t, normal, world, phase = packet(5)
        normal[:, 0, 0] = 100
        world[:, 0, 0, 0] = 100
        normal[0, 1, 1] = 500
        world[0, 1, 1, 0] = 500
        records = analyze_impulses(t, normal, world, phase)["records"]
        self.assertEqual(records[0]["normal_impulse_ns"], 2.5)
        self.assertEqual(records[0]["rolling_window_peak_ns"], 2.0)
        self.assertEqual(records[1]["normal_impulse_ns"], 0)
        self.assertEqual(records[2]["normal_impulse_ns"], 0)
        self.assertEqual(records[3]["normal_impulse_ns"], 2.5)

    def test_missing_physics_sample_rejected(self):
        t, normal, world, phase = packet(4)
        with self.assertRaisesRegex(ValueError, "every physical step"):
            analyze_impulses(
                t[[0, 3]], normal[[0, 3]], world[[0, 3]], phase[[0, 3]]
            )


if __name__ == "__main__":
    unittest.main()
