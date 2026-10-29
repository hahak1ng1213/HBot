from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from holoboxing.training.envs.official_referee import (
    OfficialReferees,
    contact_records,
)
from holoboxing.evaluation.matrix import (
    run_matrix,
    validate_plan,
    compare_variants,
)
from holoboxing.evaluation.runtime.model_contract import sha256


def contact(force=250.0, hand="left"):
    return dict(
        attacker=0,
        defender=1,
        attacking_body=hand + "_hand",
        target_body="head",
        peak_force=force,
        closing_speed=0.0,
        normal_force_n=force,
        sample_dt_s=0.005,
        position=[0.0, 0.0, 1.0],
        physical_contact=True,
    )


class RefereeTests(unittest.TestCase):
    def frame(self, ref, rows=(), fallen=(False, False)):
        for _ in range(4):
            ref.sample([list(rows)])
        return ref.observe(np.zeros((1, 2, 3)), np.asarray([fallen]))

    def test_impulse_two_hands_cooldown_and_reset(self):
        ref = OfficialReferees(1)
        _, delta = self.frame(ref, [contact(), contact(hand="right")])
        self.assertEqual(delta.tolist(), [[2, 0]])
        _, delta = self.frame(ref, [contact(), contact(hand="right")])
        self.assertEqual(delta.tolist(), [[0, 0]])
        for _ in range(8):
            self.frame(ref)
        _, delta = self.frame(ref, [contact(), contact(hand="right")])
        self.assertEqual(delta.tolist(), [[2, 0]])
        ref.reset([0])
        self.assertEqual(ref.context().tolist(), [[[60.0, 0.0], [60.0, 0.0]]])

    def test_peak_force_is_not_normal_impulse(self):
        ref = OfficialReferees(1)
        c = contact(1.0)
        c["peak_force"] = 10000
        _, delta = self.frame(ref, [c])
        self.assertEqual(delta.tolist(), [[0, 0]])

    def test_ko_and_simultaneous_ko(self):
        for fallen, winner in [((True, False), 1), ((True, True), None)]:
            ref = OfficialReferees(1)
            for _ in range(150):
                results, _ = self.frame(ref, fallen=fallen)
                self.assertFalse(results[0].match_over)
            results, delta = self.frame(ref, fallen=fallen)
            self.assertTrue(results[0].match_over)
            self.assertEqual(results[0].winner, winner)
            self.assertEqual(delta[0, 1], 1000)

    def test_contact_does_not_end_at_five_seconds(self):
        ref = OfficialReferees(1)
        for _ in range(260):
            results, _ = self.frame(ref, [contact()])
        self.assertFalse(results[0].match_over)

    def test_full_round_timeout(self):
        ref = OfficialReferees(1)
        for _ in range(2999):
            results, _ = self.frame(ref)
        self.assertFalse(results[0].match_over)
        results, _ = self.frame(ref)
        self.assertTrue(results[0].match_over)

    def test_independent_world_reset(self):
        ref = OfficialReferees(2)
        ref.sample([[contact(1000)], []])
        ref.observe(np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=bool))
        ref.reset([1])
        self.assertEqual(ref.matches[0].score.tolist(), [1, 0])
        self.assertEqual(ref.matches[1].score.tolist(), [0, 0])

    def test_contact_mapping_discards_tangential_force_and_illegal_pairs(self):
        names = [
            "p1_left_wrist_yaw_link",
            "p2_torso_link",
            "p1_torso_link",
            "floor",
        ]
        model = SimpleNamespace(
            geom_bodyid=np.arange(4),
            body=lambda i: SimpleNamespace(name=names[i]),
        )
        rows = contact_records(
            model,
            np.array([0, 1, 0]),
            np.array([[0, 1], [0, 2], [0, 3]]),
            np.zeros((3, 3)),
            np.array([[250.0, 999.0, 0.0, 0.0, 0.0, 0.0]] * 3),
            np.array([[1.0, 0.0, 0.0]] * 3),
            np.zeros(3),
            3,
            2,
        )
        self.assertEqual(len(rows[0]), 1)
        self.assertEqual(rows[1], [])
        self.assertEqual(rows[0][0]["normal_force_n"], 250.0)
        ref = OfficialReferees(2)
        for _ in range(4):
            ref.sample(rows)
        _, delta = ref.observe(
            np.zeros((2, 2, 3)), np.zeros((2, 2), dtype=bool)
        )
        self.assertEqual(delta.tolist(), [[1, 0], [0, 0]])


class MatrixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("a", "b", "o"):
            (self.root / name).write_bytes(name.encode())
        self.plan = dict(
            split="validation",
            assets="assets",
            lower="lower",
            candidates=[
                dict(
                    id=n,
                    model=str(self.root / n),
                    seed=1,
                    variant="current",
                    training_opponent_sha256=[],
                )
                for n in ("a", "b")
            ],
            opponents=[dict(id="o", model=str(self.root / "o"))],
        )

    def match(self, red, blue, *args):
        winner = (
            0
            if Path(red).name == "a"
            else 1
            if Path(blue).name == "a"
            else None
        )
        return dict(
            status="PASS",
            summary=dict(winner=winner, final_score=[0, 0], end_reason="TIME"),
        )

    def test_both_sides_and_selection_by_outcome(self):
        report = run_matrix(self.plan, self.root / "run", self.match)
        self.assertEqual(report["selected"], "a")
        self.assertEqual(report["candidates"]["a"]["games"], 2)
        self.assertEqual(report["candidates"]["a"]["win_rate"], 1.0)

    def test_no_test_set_selection(self):
        self.plan["split"] = "test"
        self.assertIsNone(
            run_matrix(self.plan, self.root / "run", self.match)["selected"]
        )

    def test_leaked_opponent_rejected(self):
        self.plan["candidates"][0]["training_opponent_sha256"] = [
            sha256(self.root / "o")
        ]
        with self.assertRaises(ValueError):
            validate_plan(self.plan)

    def test_same_weights_cannot_be_relabelled_as_independent_seeds(self):
        self.plan["candidates"][1]["model"] = self.plan["candidates"][0][
            "model"
        ]
        self.plan["candidates"][1]["seed"] = 2
        with self.assertRaisesRegex(ValueError, "Duplicate candidate"):
            validate_plan(self.plan)

    def test_invalid_results_are_failures(self):
        def broken(*args):
            return dict(
                status="PASS",
                summary=dict(winner=3, final_score=[0, 0], end_reason="TIME"),
            )

        report = run_matrix(self.plan, self.root / "run", broken)
        self.assertIsNone(report["selected"])
        self.assertFalse(report["all_matches_valid"])

    def test_failed_match_disqualifies_candidate(self):
        def broken(*args):
            raise RuntimeError("bad simulation")

        report = run_matrix(self.plan, self.root / "run", broken)
        self.assertIsNone(report["selected"])
        self.assertFalse(report["all_matches_valid"])
        self.assertEqual(report["candidates"]["a"]["failures"], 2)

    def test_ablation_pairs_seeds_and_does_not_inflate_sample_count(self):
        candidates = [
            dict(id=f"{v}{s}", variant=v, seed=s)
            for v in ("a", "b")
            for s in (1, 2, 3)
        ]
        summaries = {
            c["id"]: dict(
                eligible=True, match_points=0.4 if c["variant"] == "a" else 0.6
            )
            for c in candidates
        }
        report = compare_variants(candidates, summaries, "a")["b"]
        self.assertEqual(report["paired_seeds"], [1, 2, 3])
        self.assertAlmostEqual(report["mean_delta"], 0.2)
        summaries["b2"]["eligible"] = False
        report = compare_variants(candidates, summaries, "a")["b"]
        self.assertFalse(report["sufficient_seed_count"])
        self.assertEqual(report["unpaired_or_failed_seeds"], [2])


if __name__ == "__main__":
    unittest.main()
