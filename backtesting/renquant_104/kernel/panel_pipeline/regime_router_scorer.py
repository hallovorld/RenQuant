"""Regime-conditional model router — implements PRIME DIRECTIVE for model selection.

Per 2026-05-19 Phase 0 finding: XGB is catastrophic in COVID-style crash
(cut1 bull_ic=-0.27); HF PatchTST is catastrophic in 2024-unwind (cut5
bull_ic=-0.03). They fail in DIFFERENT regimes, so the right strategy
is ensemble routing: use HF when crash detected, XGB elsewhere.

Empirical evidence (Phase 0, all 3 walk-forward cuts):
  cut1_covid    XGB -0.27 / HF +0.107  → use HF
  cut3_inflpk   XGB +0.22 / HF +0.100  → use XGB
  cut5_unwind   XGB +0.085 / HF +0.016 → use XGB

Default routing rule (overrideable via config):
  BEAR    → HF (XGB collapses in COVID-style crashes)
  CHOPPY  → HF (XGB ranks chaotically; HF more stable)
  BULL_*  → XGB (XGB strong in normal/inflation regimes)
  UNKNOWN → default (typically XGB; safer choice for ambiguous state)

References:
  - CLAUDE.md PRIME DIRECTIVE (regime-conditional everything)
  - [[feedback_response_function_not_detector]] — response function matters
    more than detector quality
  - 2026-05-14 shorts analysis: pooled-mean hides regime-specific behavior
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.panel_pipeline.regime_router_scorer")

# Default routing — derived from Phase 0 empirical results (3 cuts)
DEFAULT_ROUTING = {
    "BEAR":          "hf_patchtst",   # HF +0.107 vs XGB -0.06 in COVID BEAR
    "CHOPPY":        "hf_patchtst",   # HF more stable in chaos
    "BULL_CALM":     "xgb",            # XGB strong in normal
    "BULL_VOLATILE": "xgb",            # XGB +0.22 in inflpk, +0.085 in unwind
    "BULL_STRONG":   "xgb",            # phantom label, but default to XGB
}


class RegimeRouterScorer:
    """Dispatches scoring to a per-regime scorer.

    Interface-compatible with other scorers — exposes feature_cols (UNION),
    seq_len (max across scorers, used by panel-history loaders), and
    score_with_history(panel_history, target_tickers) which inspects
    current regime from ctx and routes to that regime's scorer.

    Attrs:
      scorers: dict[regime_label, scorer]
      routing: dict[regime_label, scorer_key]
      default_scorer_key: fallback only if regime is not in routing
      requires_history: True if ANY scorer requires history
    """

    def __init__(self, scorers: dict[str, object],
                 routing: Optional[dict[str, str]] = None,
                 default_scorer_key: str = "xgb"):
        if not scorers:
            raise ValueError("RegimeRouterScorer needs ≥1 scorer")
        if default_scorer_key not in scorers:
            raise ValueError(f"default_scorer_key {default_scorer_key} not in "
                              f"scorers {list(scorers)}")
        self.scorers = scorers
        self.routing = dict(routing or DEFAULT_ROUTING)
        self.default_scorer_key = default_scorer_key
        missing_routes = {
            regime: scorer_key
            for regime, scorer_key in self.routing.items()
            if scorer_key not in scorers
        }
        if missing_routes:
            raise ValueError(
                "RegimeRouterScorer routing references missing scorer(s): "
                f"{missing_routes}; loaded={list(scorers)}"
            )
        # Union of feature_cols (model-specific subsets filtered at score time)
        feat_set: set[str] = set()
        for s in scorers.values():
            feat_set.update(getattr(s, "feature_cols", []))
        self.feature_cols = sorted(feat_set)
        # Max seq_len so panel history loader fetches enough
        self.seq_len = max((getattr(s, "seq_len", 1) for s in scorers.values()),
                           default=1)
        # Any history-requiring scorer → router requires history
        self.requires_history = any(
            getattr(s, "requires_history", False) for s in scorers.values())
        self.metadata = {
            "routing": dict(self.routing),
            "scorer_keys": list(scorers.keys()),
            "default_scorer_key": default_scorer_key,
        }

    def _pick_scorer(self, regime: str):
        """Returns the scorer to use for the given detected regime."""
        scorer_key = self.routing.get(regime, self.default_scorer_key)
        scorer = self.scorers.get(scorer_key)
        if scorer is None:
            raise RuntimeError(
                f"RegimeRouter: scorer_key={scorer_key} missing for "
                f"regime={regime}; loaded={list(self.scorers)}"
            )
        return scorer, scorer_key

    def score_with_history(self, panel_history: pd.DataFrame,
                            target_tickers: list[str],
                            current_regime: str = "BULL_CALM"
                            ) -> pd.Series:
        """Route to per-regime scorer. current_regime defaults to BULL_CALM
        if caller can't supply (matches kernel/pipeline/context.py default)."""
        scorer, scorer_key = self._pick_scorer(current_regime)
        log.info("RegimeRouter: regime=%s → scorer=%s", current_regime, scorer_key)
        # Filter panel_history to scorer's expected feature_cols
        feat_cols = list(getattr(scorer, "feature_cols", []))
        if not feat_cols:
            raise RuntimeError(f"scorer {scorer_key} missing feature_cols")
        cols_keep = ["ticker", "date"] + [c for c in feat_cols
                                            if c in panel_history.columns]
        ph = panel_history[cols_keep].copy()
        missing = [c for c in feat_cols if c not in ph.columns]
        if missing:
            raise RuntimeError(
                f"RegimeRouter: scorer {scorer_key} missing feature columns "
                f"{missing[:10]} (n={len(missing)})"
            )
        # Dispatch
        if getattr(scorer, "requires_history", False):
            return scorer.score_with_history(ph, target_tickers)
        else:
            # Non-history scorer (XGB) — only needs latest row per ticker
            latest = ph.sort_values("date").groupby("ticker").tail(1)
            latest = latest.set_index("ticker").loc[
                [t for t in target_tickers if t in latest.index]]
            return scorer.score(latest[feat_cols].fillna(0.0))

    def score(self, feature_matrix: pd.DataFrame,
              current_regime: str = "BULL_CALM") -> pd.Series:
        """Non-history score path. Routes to scorer; that scorer's .score()
        is called for non-history scorers, .score_with_history() raises."""
        scorer, scorer_key = self._pick_scorer(current_regime)
        if getattr(scorer, "requires_history", False):
            raise RuntimeError(
                f"RegimeRouter: regime {current_regime} routes to "
                f"history-requiring scorer {scorer_key}; caller must use "
                f"score_with_history() instead of score().")
        return scorer.score(feature_matrix)


def build_default_router(xgb_scorer, hf_scorer) -> RegimeRouterScorer:
    """Convenience builder using Phase 0 default routing."""
    return RegimeRouterScorer(
        scorers={"xgb": xgb_scorer, "hf_patchtst": hf_scorer},
        routing=DEFAULT_ROUTING,
        default_scorer_key="xgb",
    )


__all__ = ["RegimeRouterScorer", "DEFAULT_ROUTING", "build_default_router"]
