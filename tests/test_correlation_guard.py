"""Tests for the correlation-artifact leakage guard.

Pins the 2026-05-10 audit class: `watchlist-correlation.json` is a
static artifact read by both sim and LEAN. Without an `as_of_date`
stamp, a backtest in 2024-01 silently consumes a correlation matrix
computed in 2026 → forward leakage at every bar.

Per CLAUDE.md §5.13.3, `TestCorrelationGuardRegression` is the
AUDIT REGRESSION GUARD that pins the production scenario.
Per CLAUDE.md §5.13.1, `TestCorrelationGuardReadsRealArtifact`
loads the actual on-disk artifact and walks through the parser.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.walk_forward.correlation_guard import (  # noqa: E402
    assert_correlation_no_leakage,
    parse_correlation_artifact,
)


# ─────────────────────────────────────────────────────────────────────────────
# Parser — supports both v1 (flat) and v2 (wrapped) schemas
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationParser:

    def test_v2_schema_extracts_matrix_and_as_of(self):
        raw = {
            "schema_version": 2,
            "as_of_date": "2026-05-07",
            "data_window_start": "2025-11-10",
            "data_window_end": "2026-05-07",
            "matrix": {"AAPL": {"AAPL": 1.0, "MSFT": 0.3}, "MSFT": {"AAPL": 0.3, "MSFT": 1.0}},
        }
        matrix, as_of = parse_correlation_artifact(raw)
        assert as_of == "2026-05-07"
        assert matrix["AAPL"]["MSFT"] == 0.3

    def test_v1_legacy_flat_dict_returns_none_as_of(self):
        raw = {"AAPL": {"AAPL": 1.0, "MSFT": 0.3}, "MSFT": {"AAPL": 0.3, "MSFT": 1.0}}
        matrix, as_of = parse_correlation_artifact(raw)
        assert as_of is None
        assert matrix["AAPL"]["MSFT"] == 0.3

    def test_empty_input_returns_empty_matrix_and_none(self):
        matrix, as_of = parse_correlation_artifact(None)
        assert matrix == {}
        assert as_of is None

    def test_v2_without_matrix_falls_back_to_flat(self):
        # If "matrix" key is missing, treat the whole thing as v1.
        raw = {"AAPL": {"AAPL": 1.0}}
        matrix, as_of = parse_correlation_artifact(raw)
        assert as_of is None
        assert matrix == {"AAPL": {"AAPL": 1.0}}


# ─────────────────────────────────────────────────────────────────────────────
# Leakage detected → raises
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGuardLeakageDetected:

    def test_raises_when_as_of_after_backtest_start(self):
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_correlation_no_leakage(
                "2026-05-07", "2024-01-01", context="unit-test",
            )

    def test_error_message_names_context(self):
        with pytest.raises(ValueError) as excinfo:
            assert_correlation_no_leakage(
                "2026-05-07", "2024-01-01", context="SimAdapter corr=foo.json",
            )
        msg = str(excinfo.value)
        assert "SimAdapter corr=foo.json" in msg
        assert "2026-05-07" in msg
        assert "2024-01-01" in msg

    def test_one_day_after_still_raises(self):
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_correlation_no_leakage(
                "2024-01-02", "2024-01-01", context="unit-test",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Clean / passing cases — no raise
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGuardPasses:

    def test_passes_when_as_of_before_backtest_start(self):
        # No raise expected.
        assert_correlation_no_leakage(
            "2023-12-31", "2024-01-01", context="unit-test",
        )

    def test_passes_when_as_of_equals_backtest_start(self):
        # Boundary: as_of == backtest_start is OK. The correlation
        # reflects data up to and INCLUDING the start bar, but that
        # bar's labels are not "future" relative to the sim window
        # since the sim starts AT that bar. (Stricter than this is
        # arguable but matches conventional walk-forward semantics.)
        assert_correlation_no_leakage(
            "2024-01-01", "2024-01-01", context="unit-test",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Silent skip cases — backward compat + live mode
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGuardSilentSkip:

    def test_legacy_artifact_without_as_of_warns_but_passes(self, caplog):
        # backward compat per §5.13.10: legacy v1 → warn loudly, accept.
        import logging
        with caplog.at_level(logging.WARNING, logger="kernel.walk_forward.correlation_guard"):
            assert_correlation_no_leakage(
                None, "2024-01-01", context="legacy-test",
            )
        assert any("no as_of_date" in r.message for r in caplog.records)

    def test_skips_in_live_mode_even_with_leakage(self):
        # Live runs use the freshest correlation by construction; the
        # guard is a no-op in live mode.
        assert_correlation_no_leakage(
            "2026-05-07", "2024-01-01",
            is_live_mode=True, context="live-test",
        )

    def test_skips_when_backtest_start_is_none(self):
        # Older API surface — caller didn't pass backtest_start.
        # Don't raise; the downstream sim either fails clearly or
        # the leakage path is bypassed entirely.
        assert_correlation_no_leakage(
            "2026-05-07", None, context="no-start-test",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Regression — production scenario from 2026-05-10 audit
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGuardRegression:
    """AUDIT REGRESSION GUARD per CLAUDE.md §5.13.3.

    Pins the exact 2026-05-10 audit class for the correlation artifact:
    production `watchlist-correlation.json` (mtime 2026-05-07, computed
    from data through 2026-05-07) loaded into a backtest covering
    2024-01-01 → 2026-03-26 must raise.
    """

    def test_audit_2026_05_10_correlation_class_blocks(self):
        # Mirrors what would happen if the guard saw the production
        # artifact stamped with its true as_of_date.
        config_backtest_start = "2024-01-01"
        artifact_as_of = "2026-05-07"
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_correlation_no_leakage(
                artifact_as_of,
                config_backtest_start,
                context="SimAdapter corr=watchlist-correlation.json",
            )

    def test_walk_forward_pattern_passes(self):
        # The fix path: correlation computed from data window ending
        # strictly before backtest_start.
        assert_correlation_no_leakage(
            "2023-12-29",
            "2024-01-01",
            context="walk-forward corr",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Real on-disk artifact — §5.13.1 (don't trust synthetic fixtures alone)
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGuardReadsRealArtifact:
    """Per §5.13.1: walks the real prod artifact through the parser so
    we're not lying to ourselves with synthetic fixtures."""

    def test_parses_real_watchlist_correlation_json(self):
        path = _STRATEGY_DIR / "artifacts" / "watchlist-correlation.json"
        if not path.exists():
            pytest.skip(f"production artifact missing at {path}")
        raw = json.loads(path.read_text())
        matrix, as_of = parse_correlation_artifact(raw)
        # Matrix must be non-empty and addressable as ticker → ticker → float.
        assert len(matrix) > 0
        first_t = next(iter(matrix))
        assert isinstance(matrix[first_t], dict)
        # as_of is either None (legacy v1 still on disk) or a parseable string.
        if as_of is not None:
            import pandas as pd
            pd.Timestamp(as_of)  # must coerce

    def test_real_artifact_against_2024_backtest_start_behaves_correctly(self):
        # The real prod artifact today (2026-05-10) is legacy v1: no
        # as_of_date stamp. Guard must WARN and accept — not raise —
        # per the backward-compat rule in §5.13.10. After regen this
        # test will still pass (legacy or fresh-with-as_of_date <
        # 2024-01-01 both accepted).
        path = _STRATEGY_DIR / "artifacts" / "watchlist-correlation.json"
        if not path.exists():
            pytest.skip(f"production artifact missing at {path}")
        raw = json.loads(path.read_text())
        _, as_of = parse_correlation_artifact(raw)
        if as_of is None:
            # Legacy on disk: must NOT raise.
            assert_correlation_no_leakage(
                as_of, "2024-01-01", context="real-artifact-legacy",
            )
        else:
            # Fresh stamped v2: outcome depends on the as_of_date.
            # We just exercise the path; assertions below cover both.
            import pandas as pd
            if pd.Timestamp(as_of) > pd.Timestamp("2024-01-01"):
                with pytest.raises(ValueError):
                    assert_correlation_no_leakage(
                        as_of, "2024-01-01", context="real-artifact-v2",
                    )
            else:
                assert_correlation_no_leakage(
                    as_of, "2024-01-01", context="real-artifact-v2",
                )


# ─────────────────────────────────────────────────────────────────────────────
# Generation-site smoke — CorrelationJob writes v2 schema
# ─────────────────────────────────────────────────────────────────────────────


class TestCorrelationGenerationSiteWritesV2:
    """Pins that CorrelationJob now writes the v2 wrapped schema with
    `as_of_date`. Per §5.13.7, the on-disk artifact is NOT regenerated
    by this test suite (that's an ops step), but the writer code path
    is exercised end-to-end against a tmp_path."""

    def test_correlation_job_emits_as_of_date(self, tmp_path):
        import pandas as pd
        from kernel.pipeline.pp_training import CorrelationJob, TrainingContext

        # Build a tiny synthetic 3-ticker OHLCV the job can consume.
        dates = pd.date_range("2023-01-01", periods=200, freq="B")
        ohlcv = {}
        for t in ("AAA", "BBB", "CCC"):
            ohlcv[t] = pd.DataFrame(
                {"close": pd.Series(range(200), index=dates).astype(float) + 100.0},
                index=dates,
            )

        ctx = TrainingContext(
            config={
                "watchlist": ["AAA", "BBB", "CCC"],
                "_strategy_dir": str(tmp_path),
            },
            ohlcv=ohlcv,
        )
        CorrelationJob().run(ctx)

        out = tmp_path / "artifacts" / "watchlist-correlation.json"
        assert out.exists()
        wrapped = json.loads(out.read_text())
        assert wrapped.get("schema_version") == 2
        assert "as_of_date" in wrapped
        assert "matrix" in wrapped
        assert "data_window_start" in wrapped
        assert "data_window_end" in wrapped
        # Matrix is addressable.
        assert wrapped["matrix"]["AAA"]["AAA"] == pytest.approx(1.0)
        # Parser round-trips correctly.
        matrix, as_of = parse_correlation_artifact(wrapped)
        assert as_of is not None
        assert "AAA" in matrix
