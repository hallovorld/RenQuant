"""Tests for QualityFloorTask conformal Gate B reader (M3).

Covers TEST-3 + STALE-1 from doc/archives/audits/2026-04-28-deep-audit.md.

The static config threshold (default 0.10) must remain the safe fallback;
the conformal artifact only overrides when fully valid.
"""
from __future__ import annotations

import datetime
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.panel_pipeline.task_quality_floor import QualityFloorTask  # noqa: E402


@dataclass
class _StubCtx:
    """Minimal ctx for _gate_b_conformal_tau (only reads ctx.config + regime)."""
    config: dict = field(default_factory=dict)
    regime: str = "BULL_CALM"


def _write_artifact(strategy_dir: Path, body: dict | str) -> Path:
    (strategy_dir / "artifacts").mkdir(parents=True, exist_ok=True)
    p = strategy_dir / "artifacts" / "gate_b_thresholds.json"
    if isinstance(body, str):
        p.write_text(body)
    else:
        p.write_text(json.dumps(body))
    return p


def _ctx(tmp_path: Path, regime: str = "BULL_CALM",
         max_age_days: int | None = 7) -> _StubCtx:
    # 2026-05-11 sim/prod isolation: production lookup path moved to
    # artifacts/prod/. Tests still write to the flat tmp_path/artifacts/
    # for isolation, so explicitly override the artifact path to match.
    cfg = {"_strategy_dir": str(tmp_path)}
    cfg["ranking"] = {"panel_scoring": {"quality_floor": {
        "gate_b_artifact_path": "gate_b_thresholds.json",
    }}}
    if max_age_days is not None:
        cfg["ranking"]["panel_scoring"]["quality_floor"]["edge_sharpe_floor"] = {
            "conformal_max_age_days": max_age_days,
        }
    return _StubCtx(config=cfg, regime=regime)


# ── happy path ──────────────────────────────────────────────────────────────

def test_returns_tau_when_present(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": 0.082, "CHOPPY": 0.158},
    })
    ctx = _ctx(tmp_path, regime="BULL_CALM")
    tau = QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM")
    assert tau == pytest.approx(0.082)


def test_choppy_picks_choppy_threshold(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": 0.082, "CHOPPY": 0.158},
    })
    ctx = _ctx(tmp_path, regime="CHOPPY")
    tau = QualityFloorTask._gate_b_conformal_tau(ctx, "CHOPPY")
    assert tau == pytest.approx(0.158)


# ── safe-fallback paths (return None → caller uses config τ) ─────────────────

def test_returns_none_when_file_missing(tmp_path):
    ctx = _ctx(tmp_path)  # no artifact written
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_regime_missing_in_artifact(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": 0.082},
    })
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BEAR") is None


def test_returns_none_when_regime_is_none(tmp_path):
    assert QualityFloorTask._gate_b_conformal_tau(_ctx(tmp_path), None) is None


def test_returns_none_on_corrupt_json(tmp_path):
    _write_artifact(tmp_path, "not valid json {")
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_thresholds_not_dict(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": "0.10",  # wrong type
    })
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_artifact_root_not_dict(tmp_path):
    _write_artifact(tmp_path, '"a string root"')
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_strategy_dir_missing(tmp_path):
    ctx = _StubCtx(config={"_strategy_dir": ""})  # no abs path
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


# ── value validation ────────────────────────────────────────────────────────

def test_returns_none_when_tau_negative(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": -0.05},
    })
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_tau_above_one(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": 1.5},
    })
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_returns_none_when_tau_nan(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": datetime.datetime.utcnow().isoformat(),
        "thresholds": {"BULL_CALM": float("nan")},
    })
    # JSON serialises NaN as a non-standard float; round-trip loses it.
    # Manually inject string-wrapped NaN to simulate:
    p = tmp_path / "artifacts" / "gate_b_thresholds.json"
    bad = {"fitted_at": datetime.datetime.utcnow().isoformat(),
           "thresholds": {"BULL_CALM": "not_a_number"}}
    p.write_text(json.dumps(bad))
    ctx = _ctx(tmp_path)
    # str → float conversion fails → except path → None
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


# ── STALE-1: max-age check ──────────────────────────────────────────────────

def test_returns_none_when_artifact_older_than_max_age(tmp_path):
    old = (datetime.datetime.utcnow() - datetime.timedelta(days=14)).isoformat()
    _write_artifact(tmp_path, {
        "fitted_at": old,
        "thresholds": {"BULL_CALM": 0.082},
    })
    ctx = _ctx(tmp_path, max_age_days=7)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


def test_within_max_age_returns_tau(tmp_path):
    fresh = (datetime.datetime.utcnow() - datetime.timedelta(days=3)).isoformat()
    _write_artifact(tmp_path, {
        "fitted_at": fresh,
        "thresholds": {"BULL_CALM": 0.082},
    })
    ctx = _ctx(tmp_path, max_age_days=7)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") == pytest.approx(0.082)


def test_max_age_zero_disables_check(tmp_path):
    very_old = (datetime.datetime.utcnow() - datetime.timedelta(days=365)).isoformat()
    _write_artifact(tmp_path, {
        "fitted_at": very_old,
        "thresholds": {"BULL_CALM": 0.082},
    })
    ctx = _ctx(tmp_path, max_age_days=0)
    # max_age_days=0 means "no check"
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") == pytest.approx(0.082)


def test_unparseable_fitted_at_returns_none(tmp_path):
    _write_artifact(tmp_path, {
        "fitted_at": "not-an-iso-date",
        "thresholds": {"BULL_CALM": 0.082},
    })
    ctx = _ctx(tmp_path)
    assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None
