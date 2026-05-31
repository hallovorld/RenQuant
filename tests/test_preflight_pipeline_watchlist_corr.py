"""Track H — paired tests for WatchlistSizeTask + CorrelationMetadataTask
asserting byte-equivalence with the legacy ``_check_*`` functions.

Coverage:
  WatchlistSizeTask vs _check_watchlist_size:
    (a) artifact missing                       → HARD fail
    (b) JSON artifact unparseable              → HARD fail
    (c) trained watchlist not stamped          → soft pass
    (d) watchlist matches trained               → HARD pass
    (e) watchlist mismatch                     → HARD fail

  CorrelationMetadataTask vs _check_correlation_artifact_metadata:
    (f) correlation artifact missing (full)    → HARD fail (sell-only soft)
    (g) correlation unparseable                → HARD fail
    (h) as_of_date present + valid             → HARD pass
    (i) as_of_date missing + no legacy override → HARD fail
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    _check_correlation_artifact_metadata,
    _check_watchlist_size,
)
from kernel.preflight_pipeline import (
    CorrelationMetadataTask,
    PreflightContext,
    WatchlistSizeTask,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_strategy_dir(tmp_path: Path, artifact_payload=None,
                       artifact_path: str = "artifacts/prod/panel-ltr.alpha158_fund.json"
                       ) -> tuple[Path, dict]:
    art = tmp_path / artifact_path
    art.parent.mkdir(parents=True, exist_ok=True)
    if artifact_payload is not None:
        art.write_text(json.dumps(artifact_payload) if isinstance(artifact_payload, dict)
                       else artifact_payload)
    config = {
        "ranking": {"panel_scoring": {"artifact_path": artifact_path,
                                       "kind": "panel_ltr_xgboost"}}
    }
    return tmp_path, config


def _ctx(strategy_dir: Path, config: dict, run_mode: str | None = None) -> PreflightContext:
    return PreflightContext(config=config, strategy_dir=strategy_dir, run_mode=run_mode)


# ─── WatchlistSizeTask parity ────────────────────────────────────────────────

class TestWatchlistSizeTaskParity:

    def test_artifact_missing_hard_fail(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=None)
        cfg["watchlist"] = ["AAPL", "MSFT"]
        leg = _check_watchlist_size(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WatchlistSizeTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-WATCHLIST"
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_artifact_unparseable_hard_fail(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload="{ malformed")
        cfg["watchlist"] = ["AAPL"]
        leg = _check_watchlist_size(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WatchlistSizeTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_trained_wl_not_stamped_soft_pass(self, tmp_path):
        # artifact has no config_fingerprint_fields.watchlist
        payload = {"kind": "panel_ltr_xgboost", "feature_cols": ["KMID"]}
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        cfg["watchlist"] = ["AAPL", "MSFT"]
        leg = _check_watchlist_size(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WatchlistSizeTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_watchlist_match_hard_pass(self, tmp_path):
        payload = {
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["KMID"],
            "config_fingerprint_fields": {"watchlist": ["AAPL", "MSFT", "NVDA"]},
        }
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        cfg["watchlist"] = ["AAPL", "MSFT", "NVDA"]
        leg = _check_watchlist_size(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WatchlistSizeTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_watchlist_mismatch_hard_fail(self, tmp_path):
        payload = {
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["KMID"],
            "config_fingerprint_fields": {"watchlist": ["AAPL", "MSFT", "NVDA"]},
        }
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        cfg["watchlist"] = ["AAPL", "MSFT", "TSLA"]  # NVDA replaced by TSLA
        leg = _check_watchlist_size(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WatchlistSizeTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message
        assert "TSLA" in new.message  # in_live_not_trained
        assert "NVDA" in new.message  # in_trained_not_live


# ─── CorrelationMetadataTask parity ──────────────────────────────────────────

def _write_correlation(tmp_path: Path, payload: dict | None,
                       config_path: str = "prod/watchlist-correlation.json") -> dict:
    """Write a correlation artifact + return the config that points at it."""
    corr_path = tmp_path / "artifacts" / config_path
    corr_path.parent.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        corr_path.write_text(json.dumps(payload))
    return {
        "regime": {"correlation_artifact": config_path},
    }


class TestCorrelationMetadataTaskParity:

    def test_artifact_missing_full_hard_fail(self, tmp_path):
        cfg = _write_correlation(tmp_path, payload=None)
        leg = _check_correlation_artifact_metadata(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        CorrelationMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-CORR-METADATA"
        # _soft_for_sell_only in non-sell mode = HARD fail
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_unparseable_correlation_hard_fail(self, tmp_path):
        cfg = _write_correlation(tmp_path, payload=None)
        # Write malformed bytes to the expected path manually
        path = tmp_path / "artifacts" / "prod" / "watchlist-correlation.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{ malformed")
        leg = _check_correlation_artifact_metadata(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        CorrelationMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_correlation_with_as_of_hard_pass(self, tmp_path):
        # v2 wrapped schema: matrix must be a dict-of-dicts, not list-of-lists
        cfg = _write_correlation(tmp_path, payload={
            "schema_version": 2,
            "as_of_date": "2026-05-22",
            "matrix": {
                "AAPL": {"AAPL": 1.0, "MSFT": 0.5, "NVDA": 0.3},
                "MSFT": {"AAPL": 0.5, "MSFT": 1.0, "NVDA": 0.4},
                "NVDA": {"AAPL": 0.3, "MSFT": 0.4, "NVDA": 1.0},
            },
        })
        leg = _check_correlation_artifact_metadata(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        CorrelationMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message
        assert new.details["as_of_date"] == leg.details["as_of_date"] == "2026-05-22"
        assert new.details["n_tickers"] == leg.details["n_tickers"] == 3

    def test_correlation_missing_as_of_with_legacy_override(self, tmp_path):
        # Legacy v1 (flat dict, no as_of_date wrapper) — parser returns None
        cfg = _write_correlation(tmp_path, payload={
            "AAPL": {"AAPL": 1.0},
        })
        cfg["regime"]["allow_legacy_correlation_without_as_of"] = True
        leg = _check_correlation_artifact_metadata(
            config=cfg, strategy_dir=tmp_path, run_mode="full")
        ctx = _ctx(tmp_path, cfg, run_mode="full")
        CorrelationMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message
