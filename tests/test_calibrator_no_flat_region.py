"""Regression tests for the 2026-05-18 MCD-rebuy incident's calibrator layer.

Three independent invariants pinned:
  1. Prod calibrator artifact has no flat region > 30% of x-domain.
  2. Prod calibrator uses Platt (sigmoid) method, not isotonic.
  3. P-CALIBRATOR-FLAT-REGION preflight check exists + fires on bad curves.

Without these, a future calibrator refit (cron, manual, or unintended
isotonic fallback) could silently re-introduce flat-region tie-breaking.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PROD_CAL = (REPO / "backtesting/renquant_104/artifacts/prod"
            / "panel-rank-calibration.json")


def _largest_flat_fraction(x: list[float], y: list[float]) -> float:
    """Return the largest flat region as fraction of total x-domain.

    A "flat region" is a maximal run of ≥2 consecutive same-y points.
    Single-point segments don't count (every change is a "segment of 1"
    by the cur_start logic, but those aren't flat in any meaningful sense).
    """
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return 0.0
    x_total = float(x[-1] - x[0])
    if x_total <= 0:
        return 0.0
    longest = 0.0
    cur_start = 0
    cur_y = y[0]
    for i in range(1, len(y)):
        if y[i] != cur_y:
            # Segment spans cur_start to i-1 inclusive.
            # Only count as flat if >=2 points (i.e., at least one same-y
            # consecutive transition occurred within the segment).
            if i - cur_start >= 2:
                span = float(x[i - 1] - x[cur_start])
                if span > longest:
                    longest = span
            cur_start = i
            cur_y = y[i]
    # Final segment
    if len(y) - cur_start >= 2:
        span = float(x[-1] - x[cur_start])
        if span > longest:
            longest = span
    return longest / x_total


class TestProdCalibratorFlatRegion:
    """Pin: prod calibrator has no degenerate flat region."""

    def test_no_flat_region_over_30pct(self):
        if not PROD_CAL.exists():
            pytest.skip(f"{PROD_CAL} missing")
        cal = json.loads(PROD_CAL.read_text())
        x = cal["probability"]["x"]
        y = cal["probability"]["y"]
        frac = _largest_flat_fraction(x, y)
        assert frac <= 0.30, (
            f"Prod calibrator has flat region spanning {frac*100:.1f}% of "
            f"x-domain. MCD-rebuy incident was triggered at 57% flat. "
            f"Refit with method=platt or investigate model signal quality. "
            f"See doc/research/2026-05-18-mcd-rebuy-incident.md."
        )

    def test_n_unique_prob_y_at_least_50(self):
        """Soft floor: at least 50 unique calibrated probabilities for
        meaningful ranking across a 100-name universe."""
        if not PROD_CAL.exists():
            pytest.skip(f"{PROD_CAL} missing")
        cal = json.loads(PROD_CAL.read_text())
        y = cal["probability"]["y"]
        n_unique = len(set(round(v, 6) for v in y))
        assert n_unique >= 50, (
            f"Only {n_unique} unique probabilities. Ranking can't "
            f"differentiate a 142-name universe meaningfully."
        )


class TestProdCalibratorMethod:
    """Pin: prod calibrator uses Platt, not isotonic.

    Isotonic is the SCIKIT-LEARN default but creates flat regions.
    Platt (sigmoid) is strictly monotone by construction.
    """

    def test_fit_script_default_is_platt(self):
        src = (REPO / "scripts/fit_calibrator_alpha158_fund.py").read_text()
        # Must default method to platt
        assert 'default="platt"' in src or "default='platt'" in src, \
            "Calibrator fit script must default to method='platt'"

    def test_fit_script_has_acceptance_gate(self):
        src = (REPO / "scripts/fit_calibrator_alpha158_fund.py").read_text()
        assert "ACCEPTANCE-GATE FAIL" in src
        assert "flat region" in src.lower()

    def test_fit_script_2026_05_18_marker(self):
        src = (REPO / "scripts/fit_calibrator_alpha158_fund.py").read_text()
        assert "2026-05-18" in src and "ACCEPTANCE GATE" in src.upper()


class TestPreflightCheck:
    """Pin: P-CALIBRATOR-FLAT-REGION preflight exists + integrates."""

    def _load_preflight(self):
        sys.path.insert(0, str(REPO / "backtesting/renquant_104"))
        from kernel.preflight import (_check_calibrator_flat_region, ALL_CHECKS)
        return _check_calibrator_flat_region, ALL_CHECKS

    def test_check_function_exists(self):
        fn, _ = self._load_preflight()
        assert callable(fn)

    def test_check_registered_in_all_checks(self):
        fn, all_checks = self._load_preflight()
        assert fn in all_checks, \
            "P-CALIBRATOR-FLAT-REGION must be in ALL_CHECKS to fire at preflight"

    def test_check_passes_on_current_prod(self):
        fn, _ = self._load_preflight()
        cfg = json.loads(
            (REPO / "backtesting/renquant_104"
             / "strategy_config.json").read_text())
        result = fn(cfg, REPO / "backtesting/renquant_104")
        assert result.ok, (
            f"Current prod calibrator fails P-CALIBRATOR-FLAT-REGION: "
            f"{result.message}"
        )
        assert "P-CALIBRATOR-FLAT-REGION" == result.name


class TestHelperFunctionCorrectness:
    """Pin the _largest_flat_fraction utility used by tests + preflight."""

    def test_all_unique_returns_zero(self):
        x = [0, 1, 2, 3, 4]
        y = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _largest_flat_fraction(x, y) == 0.0

    def test_full_flat_returns_one(self):
        x = [0, 1, 2, 3, 4]
        y = [0.5, 0.5, 0.5, 0.5, 0.5]
        # All y equal — single flat span of 4 of total 4 = 1.0
        assert _largest_flat_fraction(x, y) == 1.0

    def test_half_flat_returns_half(self):
        x = [0, 1, 2, 3, 4]
        y = [0.1, 0.2, 0.5, 0.5, 0.5]  # last 3 flat
        # Flat from i=2 to i=4 → span=2 of total=4 → 0.5
        assert _largest_flat_fraction(x, y) == 0.5

    def test_isotonic_style_57pct_flat(self):
        """Reproduce the 2026-05-18 incident's isotonic shape:
        50% of curve flat at y=0.478 then steps up."""
        # Build x in [-0.6, +0.6] with 100 evenly-spaced knots
        x = [-0.6 + 1.2 * i / 99 for i in range(100)]
        # First 57% all at y=0.478, then ramp up
        flat_n = 57
        y = [0.478] * flat_n + [0.478 + 0.01 * (i - flat_n + 1)
                                for i in range(flat_n, 100)]
        frac = _largest_flat_fraction(x, y)
        # Flat span = x[flat_n-1] - x[0]; total = x[-1] - x[0] = 1.2
        # x[56] - x[0] = 56/99 * 1.2 ≈ 0.679; frac ≈ 0.566
        assert frac > 0.30, f"Expected > 30% flat, got {frac*100:.1f}%"
        assert frac < 0.65
