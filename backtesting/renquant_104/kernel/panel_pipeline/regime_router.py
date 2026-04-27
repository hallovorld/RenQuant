"""Regime-conditional panel scorer routing — T2-3 (2026-04-27).

Per `doc/roadmap.md` T2-3 (Tier 2): train SEPARATE panel-LTR models per
macro regime (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR), route at
inference via existing `ctx.regime`. The cross-sectional alpha
patterns differ by regime (Two Sigma 2024 case study), so a single
model averaged across regimes underfits each one.

Different from v1 macro broadcast: v1 tried to inject macro INTO a
single model (failed — see macro-factor-frame-redesign.md). T2-3
keeps the SAME panel features but trains 4 models, one per regime.

Public API
==========

`RegimeRouter(strategy_dir, config)`
    .pick_artifact(regime: str) -> Path     — return panel-ltr-by-regime path
    .has_regime_ensemble() -> bool          — is config enabled + artifacts exist
    .load_scorer_for_regime(regime) -> PanelScorer  — load matching scorer

The default panel-ltr.json artifact is the FALLBACK — used when
regime ensemble is disabled OR a regime-specific artifact is missing.

Status: Phase A — skeleton with config gate + artifact discovery.
Training (4 separate retrain configs) + acceptance gate per-regime
deferred to Phase B.

References
==========
- Two Sigma (2024) "A Machine Learning Approach to Regime Modeling"
- Avramov, Cheng, Metzker (2023) "Machine Learning vs Economic
  Restrictions" (J. Finance) — adding macro as conditioning helps;
  raw macro fails. Confirms split-by-regime over single-model approach.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("kernel.panel_pipeline.regime_router")


# Standard regime labels matching kernel/regime.py output
KNOWN_REGIMES = ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR")


class RegimeRouter:
    """Pick the regime-specific panel artifact, falling back to default.

    Usage::

        router = RegimeRouter(strategy_dir, config)
        if router.has_regime_ensemble():
            scorer = router.load_scorer_for_regime(ctx.regime)
        else:
            scorer = ctx.panel_scorer   # default
    """

    def __init__(self, strategy_dir: Path | str, config: dict[str, Any]):
        self.strategy_dir = Path(strategy_dir)
        self.config = config
        self._cfg = config.get("panel_ltr", {}).get("regime_ensemble", {})
        self._artifacts_dir = self.strategy_dir / "artifacts"

    def has_regime_ensemble(self) -> bool:
        """True iff config enabled AND at least one regime-specific
        artifact exists. Otherwise caller should use the default
        panel-ltr.json path."""
        if not self._cfg.get("enabled", False):
            return False
        # Need at least one regime artifact to call this "ensemble"
        for regime in KNOWN_REGIMES:
            if self._regime_artifact_path(regime).exists():
                return True
        return False

    def pick_artifact(self, regime: str) -> Path:
        """Return the artifact path for `regime`. Falls back to the
        default panel-ltr.json if a regime-specific file doesn't exist
        (so missing-artifact scenarios degrade to default-model
        behavior rather than crashing)."""
        if regime not in KNOWN_REGIMES:
            log.warning("RegimeRouter.pick_artifact: unknown regime %r — "
                        "using default panel-ltr.json", regime)
            return self._default_artifact_path()
        regime_path = self._regime_artifact_path(regime)
        if regime_path.exists():
            return regime_path
        # Audit 2nd-round #6 fix (2026-04-27): elevate fallback to WARNING.
        # When operator enabled regime ensemble but artifacts missing,
        # silent INFO log can hide degradation. Now: WARN every fallback.
        log.warning(
            "RegimeRouter: regime ensemble ENABLED but no artifact for "
            "%s at %s — falling back to default panel-ltr.json. Run "
            "per-regime training before relying on routed scores.",
            regime, regime_path,
        )
        return self._default_artifact_path()

    def load_scorer_for_regime(self, regime: str) -> Any:
        """Load and return the PanelScorer for `regime`."""
        path = self.pick_artifact(regime)
        # Lazy import to avoid pulling panel_pipeline at module-import time
        from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
        return PanelScorer.load(path)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _regime_artifact_path(self, regime: str) -> Path:
        """artifacts/panel-ltr.regime-{REGIME_LOWER}.json"""
        return self._artifacts_dir / f"panel-ltr.regime-{regime.lower()}.json"

    def _default_artifact_path(self) -> Path:
        return self._artifacts_dir / "panel-ltr.json"

    def inventory(self) -> dict[str, dict]:
        """Diagnostics: list which regime artifacts exist + their mtime/size.
        Useful for `model_dashboard.py` integration."""
        out: dict[str, dict] = {}
        for regime in KNOWN_REGIMES:
            p = self._regime_artifact_path(regime)
            out[regime] = {
                "path":   str(p),
                "exists": p.exists(),
                "size":   p.stat().st_size if p.exists() else None,
                "mtime":  p.stat().st_mtime if p.exists() else None,
            }
        return out


__all__ = ["RegimeRouter", "KNOWN_REGIMES"]
