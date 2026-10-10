"""Focused tests for the budget-policy chronology and timing audit."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

import budget_rl_experiments as budget


class GateChronologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2025-01-01", periods=84 * 24, freq="h")

    def test_main_geometry_has_disjoint_ten_day_gates(self) -> None:
        tune_start, gate_start = budget.calibration_day_boundaries(self.index)
        seq_start, seq_end, deploy_start, deploy_end = budget.split_gate_boundaries(
            self.index, gate_start, embargo_days=1
        )
        self.assertEqual(tune_start // 24, 42)
        self.assertEqual(gate_start // 24, 63)
        self.assertEqual((seq_end - seq_start) // 24, 10)
        self.assertEqual((deploy_start - seq_end) // 24, 1)
        self.assertEqual((deploy_end - deploy_start) // 24, 10)
        self.assertLess(seq_end, deploy_start)

    def test_longer_embargo_is_kept_out_of_both_gates(self) -> None:
        _, gate_start = budget.calibration_day_boundaries(self.index)
        seq_start, seq_end, deploy_start, deploy_end = budget.split_gate_boundaries(
            self.index, gate_start, embargo_days=2
        )
        self.assertEqual((deploy_start - seq_end) // 24, 2)
        self.assertEqual(
            (seq_end - seq_start + deploy_end - deploy_start) // 24,
            19,
        )


class LatencyProfileTests(unittest.TestCase):
    def test_profile_retains_dispersion_and_sample_metadata(self) -> None:
        values = np.arange(400, dtype=float)
        profile = budget.persistence_latency_profile(values)
        self.assertEqual(profile["sample_origins"], 336)
        self.assertEqual(profile["repetitions"], 21)
        self.assertGreaterEqual(profile["median_ms_per_origin"], 0.0)
        self.assertLessEqual(
            profile["q25_ms_per_origin"], profile["q75_ms_per_origin"]
        )


if __name__ == "__main__":
    unittest.main()
