"""Regression tests for short-side acceptance gates G7-short / G12 / G13.

Stage S1a (2026-05-03): gates exist as code and pass-open on long-only
artifacts; flip ``acceptance.short_side.enabled = true`` once short-side
analysis populates the required fields and the gates fire properly.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from kernel.model_acceptance_short import (  # noqa: E402
    build_short_gates_from_config,
    gate_g7_short_oos_ic_floor,
    gate_g12_long_short_parity,
    gate_g13_short_crowdedness,
)


class TestG7ShortFloor(unittest.TestCase):
    def test_skip_open_when_field_missing(self) -> None:
        r = gate_g7_short_oos_ic_floor({"oos_mean_ic": 0.05}, None)
        self.assertTrue(r.passed)
        self.assertIn("skip", r.detail.lower())

    def test_pass_when_above_floor(self) -> None:
        r = gate_g7_short_oos_ic_floor({"short_oos_mean_ic": 0.05}, None, floor=0.02)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.metric, 0.05)

    def test_fail_when_below_floor(self) -> None:
        r = gate_g7_short_oos_ic_floor({"short_oos_mean_ic": 0.01}, None, floor=0.02)
        self.assertFalse(r.passed)

    def test_fail_when_negative(self) -> None:
        # bottom decile actually OUTPERFORMS — shorts would lose money
        r = gate_g7_short_oos_ic_floor({"short_oos_mean_ic": -0.01}, None)
        self.assertFalse(r.passed)

    def test_non_numeric_value_fails(self) -> None:
        r = gate_g7_short_oos_ic_floor({"short_oos_mean_ic": "oops"}, None)
        self.assertFalse(r.passed)


class TestG12LongShortParity(unittest.TestCase):
    def test_skip_when_either_missing(self) -> None:
        r = gate_g12_long_short_parity({"oos_mean_ic": 0.04}, None)
        self.assertTrue(r.passed)
        r2 = gate_g12_long_short_parity({"short_oos_mean_ic": 0.04}, None)
        self.assertTrue(r2.passed)

    def test_symmetric_passes(self) -> None:
        r = gate_g12_long_short_parity(
            {"oos_mean_ic": 0.05, "short_oos_mean_ic": 0.04}, None,
        )
        # asymmetry = |0.05 - 0.04| / max = 0.01 / 0.05 = 0.20 ≤ 0.5
        self.assertTrue(r.passed)

    def test_long_dominant_fails(self) -> None:
        r = gate_g12_long_short_parity(
            {"oos_mean_ic": 0.05, "short_oos_mean_ic": 0.01}, None,
        )
        # asymmetry = 0.04 / 0.05 = 0.80 > 0.5
        self.assertFalse(r.passed)

    def test_short_dominant_fails(self) -> None:
        r = gate_g12_long_short_parity(
            {"oos_mean_ic": 0.005, "short_oos_mean_ic": 0.05}, None,
        )
        self.assertFalse(r.passed)

    def test_both_zero_fails(self) -> None:
        r = gate_g12_long_short_parity(
            {"oos_mean_ic": 0.0, "short_oos_mean_ic": 0.0}, None,
        )
        self.assertFalse(r.passed)

    def test_custom_parity_ratio(self) -> None:
        # tighter threshold — same data fails
        r = gate_g12_long_short_parity(
            {"oos_mean_ic": 0.05, "short_oos_mean_ic": 0.04}, None,
            parity_ratio=0.10,
        )
        self.assertFalse(r.passed)


class TestG13ShortCrowdedness(unittest.TestCase):
    def test_skip_when_field_missing(self) -> None:
        r = gate_g13_short_crowdedness({"short_oos_mean_ic": 0.04}, None)
        self.assertTrue(r.passed)

    def test_pass_when_below_cap(self) -> None:
        r = gate_g13_short_crowdedness(
            {"short_pnl_attribution_high_si": 0.20}, None,
        )
        self.assertTrue(r.passed)

    def test_fail_when_above_cap(self) -> None:
        r = gate_g13_short_crowdedness(
            {"short_pnl_attribution_high_si": 0.65}, None,
        )
        self.assertFalse(r.passed)

    def test_out_of_range_fails(self) -> None:
        r = gate_g13_short_crowdedness(
            {"short_pnl_attribution_high_si": 1.5}, None,
        )
        self.assertFalse(r.passed)

    def test_negative_value_fails(self) -> None:
        r = gate_g13_short_crowdedness(
            {"short_pnl_attribution_high_si": -0.1}, None,
        )
        self.assertFalse(r.passed)


class TestBuilderEnabledFlag(unittest.TestCase):
    def test_disabled_default_skips_open_on_long_only_artifact(self) -> None:
        gates = build_short_gates_from_config({})
        self.assertEqual(len(gates), 3)
        long_only_artifact = {"oos_mean_ic": 0.04}
        for g in gates:
            r = g.check(long_only_artifact, None)
            self.assertTrue(r.passed,
                            f"{g.name} should skip-open on long-only when disabled")

    def test_enabled_with_missing_short_field_fails(self) -> None:
        cfg = {"acceptance": {"short_side": {"enabled": True}}}
        gates = build_short_gates_from_config(cfg)
        long_only_artifact = {"oos_mean_ic": 0.04}
        # G7-short and G12 require short_oos_mean_ic
        g7 = next(g for g in gates if "G7" in g.name)
        g12 = next(g for g in gates if "G12" in g.name)
        g13 = next(g for g in gates if "G13" in g.name)
        r7 = g7.check(long_only_artifact, None)
        r12 = g12.check(long_only_artifact, None)
        r13 = g13.check(long_only_artifact, None)
        self.assertFalse(r7.passed, "G7-short must fail when enabled+missing")
        self.assertFalse(r12.passed, "G12 must fail when enabled+missing")
        self.assertFalse(r13.passed, "G13 must fail when enabled+missing")

    def test_enabled_with_full_short_artifact_passes(self) -> None:
        cfg = {"acceptance": {"short_side": {"enabled": True}}}
        gates = build_short_gates_from_config(cfg)
        full = {
            "oos_mean_ic": 0.05,
            "short_oos_mean_ic": 0.04,
            "short_pnl_attribution_high_si": 0.30,
        }
        results = [g.check(full, None) for g in gates]
        self.assertTrue(all(r.passed for r in results),
                        f"all 3 gates should pass: {[str(r) for r in results]}")

    def test_custom_thresholds_via_config(self) -> None:
        cfg = {"acceptance": {"short_side": {
            "enabled": True,
            "g7_short": {"floor": 0.05},
            "g12": {"parity_ratio": 0.10},
            "g13": {"max_crowded_pct": 0.20},
        }}}
        gates = build_short_gates_from_config(cfg)
        # short_oos_mean_ic = 0.04 < 0.05 floor → fail
        # asymmetry too high → fail
        # crowded 0.30 > 0.20 → fail
        artifact = {
            "oos_mean_ic": 0.05,
            "short_oos_mean_ic": 0.04,
            "short_pnl_attribution_high_si": 0.30,
        }
        results = [g.check(artifact, None) for g in gates]
        self.assertFalse(any(r.passed for r in results),
                         "all 3 should fail with tighter thresholds")


if __name__ == "__main__":
    unittest.main()
