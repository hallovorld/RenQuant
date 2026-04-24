"""Per-model TTL gate — skip per-ticker retraining when the artifact's
trained_date is within `training.model_ttl_days` of today.

Design intent (2026-04-24): user wants per-ticker freshness control
independent of the coarser `training.cadence` weekday gate. E.g.
cadence says "retrain Tue/Thu/Sun" (pipeline-level) but TTL=7 means
"within those runs, still skip tickers whose model is ≤7 days old".
`--force` on train_104.py bypasses both gates.

Failure modes covered:
- TTL=0 → disabled, every ticker retrains (default preserves behavior)
- TTL>0 + fresh artifact → skipped, ttl_skipped=True
- TTL>0 + stale artifact → retrains normally
- TTL>0 + no artifact → retrains (cold path)
- TTL>0 + corrupt artifact → retrains (fail open)
- force_retrain=True → TTL ignored, retrains
"""
from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


def _write_metadata(strategy_dir: Path, ticker: str, trained_date: str | None):
    mp = strategy_dir / "models" / ticker / f"{ticker}-policy-metadata.json"
    mp.parent.mkdir(parents=True, exist_ok=True)
    meta = {"approach": "classification", "sharpe": 1.5}
    if trained_date is not None:
        meta["trained_date"] = trained_date
    mp.write_text(json.dumps(meta))
    return mp


class TestModelIsFresh:
    def test_ttl_disabled_returns_not_fresh(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        _write_metadata(tmp_path, "NVDA", "2026-04-20")
        fresh, reason = _model_is_fresh("NVDA", tmp_path, ttl_days=0)
        assert fresh is False
        assert "disabled" in reason

    def test_fresh_model_skipped(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        today = datetime.date(2026, 4, 24)
        _write_metadata(tmp_path, "NVDA", "2026-04-22")  # 2 days old
        fresh, reason = _model_is_fresh(
            "NVDA", tmp_path, ttl_days=7, today=today,
        )
        assert fresh is True
        assert "fresh" in reason
        assert "age=2d" in reason

    def test_stale_model_retrains(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        today = datetime.date(2026, 4, 24)
        _write_metadata(tmp_path, "NVDA", "2026-04-10")  # 14 days old
        fresh, reason = _model_is_fresh(
            "NVDA", tmp_path, ttl_days=7, today=today,
        )
        assert fresh is False
        assert "stale" in reason

    def test_boundary_ttl_exactly_equals_age(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        today = datetime.date(2026, 4, 24)
        _write_metadata(tmp_path, "NVDA", "2026-04-17")  # 7 days old
        # Age == TTL should be treated as fresh (inclusive bound)
        fresh, _ = _model_is_fresh(
            "NVDA", tmp_path, ttl_days=7, today=today,
        )
        assert fresh is True

    def test_no_artifact_returns_not_fresh(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        fresh, reason = _model_is_fresh("NVDA", tmp_path, ttl_days=7)
        assert fresh is False
        assert "no existing model" in reason

    def test_corrupt_metadata_returns_not_fresh(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        mp = tmp_path / "models" / "NVDA" / "NVDA-policy-metadata.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text("not valid json{{{")
        fresh, reason = _model_is_fresh("NVDA", tmp_path, ttl_days=7)
        assert fresh is False
        assert "parse failed" in reason

    def test_missing_trained_date_returns_not_fresh(self, tmp_path):
        from kernel.pipeline.pp_training import _model_is_fresh
        _write_metadata(tmp_path, "NVDA", None)  # no trained_date
        fresh, reason = _model_is_fresh("NVDA", tmp_path, ttl_days=7)
        assert fresh is False
        assert "no trained_date" in reason

    def test_strategy_dir_none_returns_not_fresh(self):
        from kernel.pipeline.pp_training import _model_is_fresh
        fresh, _ = _model_is_fresh("NVDA", None, ttl_days=7)
        assert fresh is False


class TestChainRespectsTtl:
    """The full _run_ticker_chain should short-circuit when TTL is hit."""

    def test_chain_skips_when_ttl_fresh(self, tmp_path, monkeypatch):
        from kernel.pipeline.pp_training import (
            _run_ticker_chain, TickerTrainingContext,
        )

        today = datetime.date.today().isoformat()
        _write_metadata(tmp_path, "NVDA", today)

        # Sentinel — fail if FeatureJob runs
        import kernel.pipeline.pp_training as mod
        def _boom(tc):
            raise AssertionError("FeatureJob should not run — TTL gate failed")
        monkeypatch.setattr(mod.TickerFeatureJob, "run", lambda self, tc: _boom(tc))

        tc = TickerTrainingContext(
            ticker="NVDA", ohlcv={}, config={"training": {"model_ttl_days": 7}},
            strategy_dir=tmp_path,
        )
        _run_ticker_chain(tc)
        assert tc.ttl_skipped is True
        assert tc.exported is True   # treated as exported for downstream count

    def test_chain_runs_when_force_retrain(self, tmp_path, monkeypatch):
        """force_retrain=True must bypass the TTL gate."""
        from kernel.pipeline.pp_training import (
            _run_ticker_chain, TickerTrainingContext,
        )

        today = datetime.date.today().isoformat()
        _write_metadata(tmp_path, "NVDA", today)

        # Record that FeatureJob was reached — proves TTL was bypassed.
        reached = {"feature": False}
        import kernel.pipeline.pp_training as mod
        def _mark(self, tc):
            reached["feature"] = True
        monkeypatch.setattr(mod.TickerFeatureJob, "run", _mark)

        tc = TickerTrainingContext(
            ticker="NVDA", ohlcv={}, config={
                "training": {"model_ttl_days": 7},
                "_force_retrain": True,
            },
            strategy_dir=tmp_path,
        )
        _run_ticker_chain(tc)
        assert tc.ttl_skipped is False
        assert reached["feature"] is True
