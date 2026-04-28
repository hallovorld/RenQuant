"""Tests for ApplyNGBoostTask feature-drift detector + CRIT-1 fail-safe.

Covers TEST-2 from doc/archives/audits/2026-04-28-deep-audit.md.

Background — the 2026-04-27 incident: deployed NGBoost head was trained
with 184 features (including 156 macro cols); inference panel only
produced 28 → 84.8% zero-fill → corrupted σ → all candidates
underperformed Gate B → 0 buys all day. Single buried log.warning.

Fix landed in this audit:
  1. >5% missing → log.error + skip NGBoost scoring.
  2. CRIT-1: when scoring is skipped, stamp NaN μ/σ on candidates AND
     clear ctx.candidates so Gate B / downstream don't silently admit
     unscored buys.

These tests guard those invariants.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask  # noqa: E402


# ── stubs ───────────────────────────────────────────────────────────────────

@dataclass
class _StubCand:
    ticker: str
    rank_score: float | None = None
    panel_score: float | None = None
    mu: float | None = None
    sigma: float | None = None


@dataclass
class _StubHolding:
    mu: float | None = None
    sigma: float | None = None
    rank_score: float | None = None
    panel_score: float | None = None


@dataclass
class _StubCtx:
    config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    holdings: dict = field(default_factory=dict)
    _panel_matrix: pd.DataFrame | None = None
    _ngboost_head: object = None
    counters: dict = field(default_factory=dict)


class _StubHead:
    """Minimal NGBoost head — exposes feature_cols + predict_distribution."""
    def __init__(self, feature_cols, mus=None, sigmas=None):
        self.feature_cols = list(feature_cols)
        self._mus = mus or {}
        self._sigmas = sigmas or {}

    def predict_distribution(self, X: pd.DataFrame):
        idx = X.index
        mu_series    = pd.Series([self._mus.get(t, 0.01)    for t in idx], index=idx)
        sigma_series = pd.Series([self._sigmas.get(t, 0.05) for t in idx], index=idx)
        return {"mu": mu_series, "sigma": sigma_series}


def _build_ctx(
    candidate_tickers: list[str],
    head_feats: list[str],
    panel_cols: list[str],
    holdings: dict | None = None,
    drift_pct: float | None = None,
    score_mode: str = "additive",
) -> _StubCtx:
    cfg = {
        "ranking": {"panel_scoring": {
            "ngboost": {
                "enabled":  True,
                "score_mode": score_mode,
                **({"max_feature_drift_pct": drift_pct} if drift_pct is not None else {}),
            },
        }},
    }
    cands = [_StubCand(ticker=t) for t in candidate_tickers]
    holdings = holdings or {}

    # Build panel matrix indexed by ticker (cands ∪ holdings)
    all_tickers = candidate_tickers + list(holdings.keys())
    rows = {col: [1.0] * len(all_tickers) for col in panel_cols}
    X = pd.DataFrame(rows, index=all_tickers)

    head = _StubHead(feature_cols=head_feats)
    return _StubCtx(
        config=cfg,
        candidates=cands,
        holdings=holdings,
        _panel_matrix=X,
        _ngboost_head=head,
    )


# ── happy path ──────────────────────────────────────────────────────────────

class TestNoDrift:
    def test_no_warning_when_features_match(self, caplog):
        feats = ["a", "b", "c"]
        ctx = _build_ctx(["AAPL"], feats, feats)
        ApplyNGBoostTask().run(ctx)
        # candidate gets the predicted μ/σ; not None
        assert ctx.candidates[0].mu == 0.01
        assert ctx.candidates[0].sigma == 0.05
        # No drift counter incremented
        assert ctx.counters.get("ngb_drift_fail", 0) == 0
        # No drift warning text in log
        assert not any("MISSING" in r.message or "missing" in r.message
                       for r in caplog.records)


# ── soft warning path: drift under threshold ───────────────────────────────

class TestDriftUnderThreshold:
    def test_zero_fills_one_missing_of_27(self, caplog):
        # 1/27 = 3.7%, default threshold 5%, so soft path
        head_feats = [f"f{i}" for i in range(27)]
        panel_cols = head_feats[:-1]   # 26 cols, missing f26
        ctx = _build_ctx(["AAPL"], head_feats, panel_cols)
        ApplyNGBoostTask().run(ctx)
        # candidate scored normally
        assert ctx.candidates[0].mu == 0.01
        assert ctx.candidates[0].sigma == 0.05
        # candidates pool intact
        assert len(ctx.candidates) == 1
        # warning logged
        assert any("missing" in r.message.lower() and "filling with 0.0" in r.message
                   for r in caplog.records)


# ── HARD FAIL path: drift over threshold ────────────────────────────────────

class TestDriftOverThreshold:
    def test_skips_when_50pct_missing(self):
        head_feats = [f"f{i}" for i in range(10)]
        panel_cols = head_feats[:5]   # 5/10 missing = 50%
        ctx = _build_ctx(["AAPL", "MSFT"], head_feats, panel_cols)
        ApplyNGBoostTask().run(ctx)
        # CRIT-1: candidates were stamped with NaN, then list cleared
        assert ctx.candidates == []
        # Drift counter incremented
        assert ctx.counters.get("ngb_drift_fail") == 1

    def test_crit1_candidates_stamped_nan_before_clear(self):
        # CRIT-1 regression: even though list is cleared, the cand objects
        # we held externally must have NaN μ/σ so any downstream task that
        # received them via cached references rejects them.
        head_feats = [f"f{i}" for i in range(10)]
        panel_cols = head_feats[:3]   # 7/10 missing = 70%
        cands_external = [_StubCand(ticker="AAPL"), _StubCand(ticker="MSFT")]
        ctx = _StubCtx(
            config={"ranking": {"panel_scoring": {"ngboost": {
                "enabled": True, "score_mode": "additive",
            }}}},
            candidates=cands_external,
            holdings={},
            _panel_matrix=pd.DataFrame(
                {c: [1.0, 1.0] for c in panel_cols},
                index=["AAPL", "MSFT"],
            ),
            _ngboost_head=_StubHead(feature_cols=head_feats),
        )
        ApplyNGBoostTask().run(ctx)
        # ctx.candidates is empty post-fail
        assert ctx.candidates == []
        # but the original cand objects are NaN-stamped (Gate B will reject)
        import math
        for c in cands_external:
            assert c.mu is not None
            assert c.sigma is not None
            assert math.isnan(c.mu)
            assert math.isnan(c.sigma)

    def test_threshold_configurable(self):
        # operator override max_feature_drift_pct=0.50 → 40% missing is OK
        head_feats = [f"f{i}" for i in range(10)]
        panel_cols = head_feats[:6]   # 4/10 = 40%
        ctx = _build_ctx(
            ["AAPL"], head_feats, panel_cols, drift_pct=0.50,
        )
        ApplyNGBoostTask().run(ctx)
        # Soft path: candidates kept, scored
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].mu == 0.01

    def test_threshold_override_strict(self):
        # operator override drift_pct=0.01 → ANY missing fails
        head_feats = [f"f{i}" for i in range(10)]
        panel_cols = head_feats[:9]   # 1/10 = 10%, threshold 1% → fail
        ctx = _build_ctx(
            ["AAPL"], head_feats, panel_cols, drift_pct=0.01,
        )
        ApplyNGBoostTask().run(ctx)
        assert ctx.candidates == []


# ── disabled-flag path ──────────────────────────────────────────────────────

class TestDisabled:
    def test_no_op_when_ngboost_disabled(self):
        ctx = _build_ctx(["AAPL"], ["a"], ["a"])
        ctx.config["ranking"]["panel_scoring"]["ngboost"]["enabled"] = False
        ApplyNGBoostTask().run(ctx)
        # untouched — no μ/σ written, no clear, no counter
        assert ctx.candidates[0].mu is None
        assert ctx.counters.get("ngb_drift_fail", 0) == 0
