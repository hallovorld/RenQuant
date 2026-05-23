"""Tests for the LEAN-side leakage guard (kernel.walk_forward.lean_guard).

Companion to tests/test_leakage_guard.py — that one tests the underlying
assert_no_leakage helper directly; this one tests the LEAN integration
shape (artifact JSON peek + config-key resolution + LiveMode skip).

Per CLAUDE.md §5.13.3, TestLeanGuardRegression pins the 2026-05-10
audit class invariant.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.walk_forward.lean_guard import assert_lean_panel_no_leakage  # noqa: E402


def _write_artifact(strategy_dir: Path, rel_path: str, meta: dict) -> Path:
    full = strategy_dir / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(json.dumps(meta))
    return full


# ─────────────────────────────────────────────────────────────────────────────
# Leakage detected → raises
# ─────────────────────────────────────────────────────────────────────────────


class TestLeanGuardLeakageDetected:

    def test_raises_when_artifact_trained_after_backtest_start(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2026-05-09", "kind": "panel_ltr_xgboost"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_start": "2026-03-26",
            "backtest_end": "2026-12-31",
        }
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )

    def test_raises_when_trained_date_equals_backtest_start(self, tmp_path):
        # Equality is leakage — model has seen the last bar's label.
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2026-03-26"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_start": "2026-03-26",
            "backtest_end": "2026-12-31",
        }
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )

    def test_error_message_names_lean_context(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2026-05-09"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_start": "2026-03-26",
            "backtest_end": "2026-12-31",
        }
        with pytest.raises(ValueError) as excinfo:
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )
        assert "LEAN backtest panel scorer" in str(excinfo.value)


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward / clean training → passes
# ─────────────────────────────────────────────────────────────────────────────


class TestLeanGuardWalkForward:

    def test_passes_when_trained_strictly_before_backtest_start(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2024-01-01"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_start": "2024-01-02",
            "backtest_end": "2026-03-26",
        }
        # No raise expected.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Silent skip cases (correct behaviour, not leakage)
# ─────────────────────────────────────────────────────────────────────────────


class TestLeanGuardSilentSkip:

    def test_skips_in_live_mode(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2026-05-09"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_end": "2026-03-26",
        }
        # is_live_mode=True → no raise even though leakage would be present
        # in backtest mode.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=True,
        )

    def test_skips_when_panel_scoring_disabled(self, tmp_path):
        config = {
            "ranking": {"panel_scoring": {"enabled": False}},
            "backtest_end": "2026-03-26",
        }
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )

    def test_skips_when_artifact_missing(self, tmp_path):
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/missing.json",
            }},
            "backtest_end": "2026-03-26",
        }
        # LoadScorerTask will fail later with a clearer message; we don't
        # raise here.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )

    def test_raises_when_artifact_has_no_trained_date(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"kind": "legacy_no_metadata"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_end": "2026-03-26",
        }
        with pytest.raises(ValueError, match="missing trained_date"):
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )

    def test_skips_when_artifact_malformed(self, tmp_path):
        full = tmp_path / "artifacts" / "panel-ltr.json"
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text("{not valid json")
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_end": "2026-03-26",
        }
        # Malformed JSON → silent skip; LoadScorerTask will fail later.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )

    def test_skips_when_no_backtest_end_in_config(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {"trained_date": "2026-05-09"},
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            # no backtest_end
        }
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )

    def test_blocks_validation_selection_window(self, tmp_path):
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.json",
            {
                "trained_date": "2026-05-22",
                "effective_train_cutoff_date": "2024-04-08",
                "lookahead_days": 60,
                "split_date_ranges": {
                    "train": {"start": "2020-01-01", "end": "2024-04-08"},
                    "val": {"start": "2024-07-02", "end": "2024-09-30"},
                },
            },
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.json",
            }},
            "backtest_start": "2024-10-01",
            "backtest_end": "2024-12-31",
        }
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Regression guard — pins the exact 2026-05-10 audit class
# ─────────────────────────────────────────────────────────────────────────────


class TestLeanGuardRegression:
    """AUDIT REGRESSION GUARD per CLAUDE.md §5.13.3.

    Pins the 2026-05-10 audit class for the LEAN path: the production
    artifact `panel-ltr.alpha158_fund.json` (trained 2026-05-09, panel_shape
    715629×292×2450) used in a backtest covering 2024-01-01 → 2026-03-26
    must raise. Companion to the SimAdapter regression in
    test_leakage_guard.py.
    """

    def test_audit_2026_05_10_lean_pattern_raises(self, tmp_path):
        # Mirror the exact production scenario the audit caught.
        _write_artifact(
            tmp_path,
            "artifacts/panel-ltr.alpha158_fund.json",
            {
                "trained_date": "2026-05-09",
                "kind": "panel_ltr_xgboost",
                "panel_shape": {"rows": 715629, "tickers": 292, "dates": 2450},
                "best_iter": 100,
                "version": 3,
            },
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/panel-ltr.alpha158_fund.json",
            }},
            "backtest_start": "2024-01-01",
            "backtest_end": "2026-03-26",
        }
        with pytest.raises(ValueError, match="Look-ahead leakage"):
            assert_lean_panel_no_leakage(
                config=config, strategy_dir=tmp_path, is_live_mode=False,
            )

    def test_proper_walk_forward_pattern_passes(self, tmp_path):
        # The walk-forward fix (P1+P2): cutoff strictly < backtest_end.
        _write_artifact(
            tmp_path,
            "artifacts/walkforward/panel-ltr-2024-01-01.json",
            {
                "trained_date": "2024-01-02T03:44:12",
                "cutoff_date": "2024-01-01",
                "kind": "panel_ltr_xgboost",
            },
        )
        config = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/walkforward/panel-ltr-2024-01-01.json",
            }},
            "backtest_start": "2024-01-02",
            "backtest_end": "2026-03-26",
        }
        # Walk-forward pattern — must pass.
        assert_lean_panel_no_leakage(
            config=config, strategy_dir=tmp_path, is_live_mode=False,
        )
