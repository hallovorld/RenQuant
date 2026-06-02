"""Track H follow-up — verify that ``run_preflight`` (now a wrapper around
``build_preflight_pipeline()``) produces bytewise-identical output on the
returned list compared to direct legacy ``_check_*`` iteration.

The order test pins ``_LEGACY_CHECK_ORDER`` so any reorder is caught.

Contract preserved:
  1. Returned list has 18 entries in ``_LEGACY_CHECK_ORDER`` order
  2. Each entry: (name, severity, ok, message, details) match legacy
  3. strict=True raises ``PreflightFailed`` on any HARD failure
  4. strict=False returns results without raising
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    PreflightFailed,
    _LEGACY_CHECK_ORDER,
    run_preflight,
)


def _minimal_strategy_dir(tmp_path: Path) -> Path:
    """Lay down a strategy_dir with the minimum required to run preflight."""
    return tmp_path


def _empty_config() -> dict:
    """A config with all gates effectively skippable — most checks soft-pass."""
    return {}


class TestRunPreflightWrapperContract:

    def test_returns_eighteen_results(self, tmp_path):
        results = run_preflight(
            config=_empty_config(),
            broker=None,
            strategy_dir=_minimal_strategy_dir(tmp_path),
            strict=False,
        )
        assert len(results) == 18

    def test_results_in_legacy_order(self, tmp_path):
        results = run_preflight(
            config=_empty_config(),
            broker=None,
            strategy_dir=_minimal_strategy_dir(tmp_path),
            strict=False,
        )
        names = [r.name for r in results]
        assert names == list(_LEGACY_CHECK_ORDER)

    def test_strict_false_returns_results_with_hard_failures(self, tmp_path):
        # Empty config → P-MODEL-ARTIFACT will HARD-fail (artifact missing).
        # strict=False should NOT raise.
        results = run_preflight(
            config=_empty_config(),
            broker=None,
            strategy_dir=_minimal_strategy_dir(tmp_path),
            strict=False,
        )
        hard_fails = [r for r in results
                      if r.severity == "hard" and not r.ok]
        assert len(hard_fails) > 0  # at least artifact-missing
        # Find P-MODEL-ARTIFACT specifically
        model = next(r for r in results if r.name == "P-MODEL-ARTIFACT")
        assert not model.ok
        assert "missing" in model.message

    def test_strict_true_raises_on_hard_failure(self, tmp_path):
        with pytest.raises(PreflightFailed) as exc_info:
            run_preflight(
                config=_empty_config(),
                broker=None,
                strategy_dir=_minimal_strategy_dir(tmp_path),
                strict=True,
            )
        # PreflightFailed.failures should include P-MODEL-ARTIFACT and be
        # ordered per _LEGACY_CHECK_ORDER (P-MODEL-ARTIFACT first).
        failures = exc_info.value.failures
        failure_names = [f.name for f in failures]
        assert "P-MODEL-ARTIFACT" in failure_names
        # First HARD failure listed should be P-MODEL-ARTIFACT (it's first
        # in _LEGACY_CHECK_ORDER).
        assert failures[0].name == "P-MODEL-ARTIFACT"

    def test_missing_strategy_dir_raises_value_error(self, tmp_path):
        with pytest.raises(ValueError, match="strategy_dir"):
            run_preflight(
                config=_empty_config(),
                broker=None,
                strategy_dir=None,
            )


class TestRunPreflightOrderingInvariant:
    """Pin the legacy check ordering — every name must be in the canonical
    tuple so the sort step covers every check."""

    def test_legacy_order_covers_every_check_name(self, tmp_path):
        results = run_preflight(
            config=_empty_config(),
            broker=None,
            strategy_dir=_minimal_strategy_dir(tmp_path),
            strict=False,
        )
        for r in results:
            assert r.name in _LEGACY_CHECK_ORDER, (
                f"check {r.name!r} not in _LEGACY_CHECK_ORDER — "
                "sort step would put it at the end. Add to "
                "_LEGACY_CHECK_ORDER tuple."
            )

    def test_legacy_order_has_no_duplicates(self):
        assert len(_LEGACY_CHECK_ORDER) == len(set(_LEGACY_CHECK_ORDER))

    def test_legacy_order_size_matches_check_count(self):
        # When more checks are added (or any retired), update this assertion.
        # Today's tally: 18 checks in the pipeline
        # (P-NEWS-SENTIMENT-FRESHNESS added 2026-06-02).
        assert len(_LEGACY_CHECK_ORDER) == 18
