"""Config / model consistency guard.

Prevents the recurring class of bugs where strategy_config.json drifts
out of sync with the trained panel-ltr.json + ngboost-head.json:

  2026-04-27 (NGBoost feature drift)  config disabled macro but the head
                                       was trained with 184 macro features
                                       → silent zero-fill → 0 buys.
  2026-04-27 (rank:ndcg config flip)   config said rank:ndcg but the model
                                       was trained pairwise → IC collapse.
  2026-04-28 (watchlist 227 mismatch)  config locked 227 watchlist but
                                       auto-revert only restored the 103
                                       model → 124 tickers without per-ticker
                                       artifacts.

This module computes a fingerprint of the MODEL-RELEVANT config fields
that train_104.py embeds into the artifact at training time, and that
RunnerAdapter verifies at inference startup. Mismatch = HARD FAIL with a
clear remediation message.

The model-relevant fields:

  1. watchlist            — set of tickers the panel was built over.
                            Cross-section z-scoring is universe-relative;
                            adding/removing tickers changes the panel.
  2. panel_ltr.lookahead_days     — labels are forward-N-day returns.
                                   10d ≠ 20d ≠ 60d.
  3. panel_ltr.xgb_params.objective — rank:pairwise vs rank:ndcg trains
                                       different weight surfaces.
  4. panel_ltr.asset_embeddings.enabled — adds 16 emb_* columns.
  5. sector_map / sector_etf_map — sector-neutralized features,
                                    relative-strength context, and QP sector
                                    caps are undefined without consistent
                                    sector metadata.

These are NECESSARY but not sufficient — feature engineering drift (e.g.
macro v3 → macro disabled) is caught separately by the M3 drift detector
in ApplyNGBoostTask. This guard catches CONFIG-side regressions that
slip past the feature-level check.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

log = logging.getLogger("kernel.config_consistency")


def _model_relevant_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Project a config to its model-affecting subset.

    Order-stable so the resulting hash is deterministic across runs.

    2026-05-04 invariant fix per CLAUDE.md §5.3 (incident: panel-ltr.json
    trained 2026-05-03 with hourly+minute features → 2026-05-04 user
    mandate disabled hourly+minute → daily-only data path no longer
    produces those columns → DriftGuardTask fail-safed every bar →
    strategy made 0 trades for 252 days). The training resolution and
    intraday-bar flags must be part of the fingerprint so any drift
    between artifact and runtime data path triggers a loud
    ConfigModelMismatch instead of a silent runtime degradation.
    """
    panel = config.get("panel_ltr", {}) or {}
    xgb_params = panel.get("xgb_params", {}) or {}
    emb_cfg = panel.get("asset_embeddings", {}) or {}
    hourly_cfg = panel.get("hourly", {}) or {}
    minute_cfg = panel.get("minute", {}) or {}
    watchlist = sorted(config.get("watchlist", []) or [])
    benchmark = config.get("benchmark", "SPY")
    raw_sector_map = config.get("sector_map", {}) or {}
    sector_map = {
        ticker: raw_sector_map.get(ticker)
        for ticker in watchlist
        if ticker != benchmark
    }
    used_sectors = sorted(
        {
            sector
            for sector in sector_map.values()
            if isinstance(sector, str) and sector
        }
    )
    raw_sector_etf_map = config.get("sector_etf_map", {}) or {}
    sector_etf_map = {
        sector: raw_sector_etf_map.get(sector)
        for sector in used_sectors
    }
    return {
        # Sorted set of tickers — order doesn't matter for the panel.
        "watchlist": watchlist,
        "lookahead_days":      int(panel.get("lookahead_days", 10)),
        "objective":           str(xgb_params.get("objective", "rank:pairwise")),
        "asset_embeddings":    bool(emb_cfg.get("enabled", False)),
        "training_resolution": str(panel.get("training_resolution", "daily")),
        "hourly_enabled":      bool(hourly_cfg.get("enabled", False)),
        "minute_enabled":      bool(minute_cfg.get("enabled", False)),
        "sector_map":          sector_map,
        "sector_etf_map":      sector_etf_map,
    }


def fingerprint_config(config: dict[str, Any]) -> str:
    """Return a short SHA256-based fingerprint of model-relevant config fields."""
    sub = _model_relevant_fields(config)
    blob = json.dumps(sub, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.sha256(blob).hexdigest()
    return f"sha256:{h[:16]}"


class ConfigModelMismatch(Exception):
    """Raised when artifact's stored fingerprint != live config fingerprint."""


def assert_consistent(
    config: dict[str, Any],
    artifact: dict[str, Any],
    *,
    artifact_label: str = "panel-ltr",
    strict: bool = True,
) -> None:
    """Verify model artifact was trained with config that matches the live one.

    Read ``artifact["config_fingerprint"]`` (added by train_104 SaveArtifactTask)
    and compare to ``fingerprint_config(config)``. When fingerprints differ:

    - ``strict=True`` (default): raise ConfigModelMismatch with the field-level
      diff so the operator can fix immediately.
    - ``strict=False``: log.error but continue (only for migration windows
      where artifacts pre-date this guard).

    Artifacts WITHOUT a stored fingerprint are treated as unknown. In strict
    mode this fails closed; in non-strict migration mode it logs a warning and
    continues.
    """
    live_fp = fingerprint_config(config)
    stored = artifact.get("config_fingerprint")
    if stored is None:
        msg = (
            "Config-consistency: artifact "
            f"{artifact_label} has no fingerprint. Live fingerprint={live_fp}. "
            "Strict full/buy paths require stamped config/sector metadata; "
            "retrain or promote a stamped artifact."
        )
        if strict:
            raise ConfigModelMismatch(msg)
        log.warning(
            "%s",
            msg,
        )
        return
    if stored == live_fp:
        log.info("Config-consistency: %s OK  fp=%s", artifact_label, live_fp)
        return

    # Mismatch — produce field-by-field diff for the error message.
    live_sub = _model_relevant_fields(config)
    stored_sub = artifact.get("config_fingerprint_fields") or {}
    diff_lines = []
    for key in sorted(set(live_sub) | set(stored_sub)):
        live_v = live_sub.get(key)
        stored_v = stored_sub.get(key)
        if live_v != stored_v:
            # Truncate long lists in display
            def _disp(v):
                if isinstance(v, list) and len(v) > 5:
                    return f"[{len(v)} items: {v[:3]}…]"
                return v
            diff_lines.append(f"  {key}: live={_disp(live_v)!r}  stored={_disp(stored_v)!r}")

    msg = (
        f"Config-consistency MISMATCH for {artifact_label}\n"
        f"  Live config fingerprint:    {live_fp}\n"
        f"  Artifact stored fingerprint: {stored}\n"
        f"\nField-level differences:\n" + "\n".join(diff_lines) + "\n"
        f"\nRESOLUTION:\n"
        f"  (a) Retrain the model: python scripts/train_104.py --skip-baseline "
        f"--skip-recalibrate --force\n"
        f"  (b) Restore matching strategy_config.json from the checkpoint that "
        f"matches the artifact (e.g. artifacts/checkpoint_*/strategy_config.json)\n"
        f"  (c) Bypass via runner --skip-config-consistency (DANGEROUS — "
        f"silently produces miscalibrated trades)"
    )
    if strict:
        raise ConfigModelMismatch(msg)
    log.error(msg)
