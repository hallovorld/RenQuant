#!/usr/bin/env python
"""Stage-1 panel-LTR acceptance driver for renquant_103.

Loads the strategy's watchlist + SPY + sector ETFs, builds per-ticker
labelled feature frames via training.features, then hands everything to
training_panel.pipeline.train_panel_model. Writes a JSON artifact and
prints the OOS mean-IC (target: ≥ 0.08).

Usage::

    python scripts/train_panel_model.py --strategy renquant_103
    python scripts/train_panel_model.py --strategy renquant_103 --out artifacts/panel-ltr.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-panel")


def _load_ohlcv(fetch, symbols, provider):
    out = {}
    for sym in symbols:
        try:
            df = fetch(sym, provider=provider)
        except Exception as exc:
            log.warning("  %-6s  fetch failed (%s) — skipping", sym, exc)
            continue
        if df is None or df.empty:
            log.warning("  %-6s  empty OHLCV — skipping", sym)
            continue
        out[sym] = df
    return out


def train(strategy: str, out: str | None) -> None:
    strategy_dir = REPO_ROOT / "backtesting" / strategy
    config_path = strategy_dir / "strategy_config.json"

    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)

    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.data import fetch_ohlcv
    from training.features import build_all_training_features
    from training_panel.pipeline import train_panel_model

    config = json.loads(config_path.read_text())
    watchlist = config["watchlist"]
    indicator_spec = config.get("indicator_spec", {})
    sector_map = config.get("sector_map", {})
    sector_etf_map = config.get("sector_etf_map", {})
    benchmark = config.get("benchmark", "SPY")
    provider = config.get("data_src", "yfinance")
    lookahead = int(config.get("model_params", {}).get("lookahead", 5))
    threshold = float(config.get("model_params", {}).get("threshold", 0.03))

    panel_cfg = config.get("panel_ltr", {})
    out_path = Path(out) if out else (
        strategy_dir / "artifacts" / "panel-ltr.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Fetch OHLCV for watchlist + SPY + sector ETFs ─────────────────────
    log.info("Fetching OHLCV for %d watchlist tickers + %s + %d sector ETFs …",
             len(watchlist), benchmark, len(sector_etf_map))
    needed = set(watchlist) | {benchmark} | set(sector_etf_map.values())
    ohlcv_all = _load_ohlcv(fetch_ohlcv, sorted(needed), provider)
    if benchmark not in ohlcv_all:
        log.error("Benchmark %s OHLCV missing — aborting", benchmark)
        sys.exit(1)

    ohlcv = {t: ohlcv_all[t] for t in watchlist if t in ohlcv_all}
    spy_ohlcv = ohlcv_all[benchmark]
    sector_etf_ohlcv = {
        sec: ohlcv_all[etf] for sec, etf in sector_etf_map.items()
        if etf in ohlcv_all
    }
    log.info("  usable watchlist=%d, sector_etfs=%d",
             len(ohlcv), len(sector_etf_ohlcv))

    # ── Build per-ticker labelled feature frames ──────────────────────────
    log.info("Building per-ticker feature frames …")
    features_ohlcv = dict(ohlcv)
    features_ohlcv[benchmark] = spy_ohlcv
    feature_frames = build_all_training_features(
        watchlist=list(ohlcv.keys()),
        ohlcv=features_ohlcv,
        indicator_spec=indicator_spec,
        lookahead=lookahead,
        threshold=threshold,
    )
    if not feature_frames:
        log.error("No feature frames built — aborting")
        sys.exit(1)

    # ── Train panel model ─────────────────────────────────────────────────
    ticker_sectors = {t: sector_map[t] for t in feature_frames if t in sector_map}
    missing_sectors = [t for t in feature_frames if t not in sector_map]
    if missing_sectors:
        log.warning("Tickers missing from sector_map (skipping): %s", missing_sectors)

    train_cfg = {
        "lookahead_days":      panel_cfg.get("lookahead_days", lookahead),
        "beta_window":         panel_cfg.get("beta_window", 60),
        "min_history_days":    panel_cfg.get("min_history_days", 252),
        "age_warmup_days":     panel_cfg.get("age_warmup_days", 504),
        "cv_n_splits":         panel_cfg.get("cv_n_splits", 5),
        "cv_embargo_days":     panel_cfg.get("cv_embargo_days", lookahead),
        "num_boost_round":     panel_cfg.get("num_boost_round", 400),
        "neutralize_features": panel_cfg.get("neutralize_features", True),
        "nan_prone_cols":      panel_cfg.get("nan_prone_cols", []),
        "xgb_params":          panel_cfg.get("xgb_params", {}),
        "training_notes":      panel_cfg.get(
            "training_notes", f"Stage 1 acceptance run — {pd.Timestamp.today():%Y-%m-%d}"
        ),
    }

    wl_final = [t for t in watchlist if t in feature_frames and t in ticker_sectors]
    log.info("Training panel-LTR on %d tickers (lookahead=%d, min_history=%d, cv=%d) …",
             len(wl_final), train_cfg["lookahead_days"],
             train_cfg["min_history_days"], train_cfg["cv_n_splits"])

    summary = train_panel_model(
        watchlist=wl_final,
        feature_frames=feature_frames,
        ohlcv=ohlcv,
        spy_ohlcv=spy_ohlcv,
        sector_etf_ohlcv=sector_etf_ohlcv,
        ticker_sectors=ticker_sectors,
        listing_dates=None,
        config=train_cfg,
        out_path=out_path,
    )

    # ── Report ────────────────────────────────────────────────────────────
    log.info("─" * 60)
    log.info("  Artifact:     %s", summary["artifact_path"])
    log.info("  Panel rows:   %d",    summary["panel_metadata"]["n_rows"])
    log.info("  Unique dates: %d",    summary["panel_metadata"]["n_dates"])
    log.info("  Tickers:      %d",    summary["panel_metadata"]["n_tickers"])
    log.info("  Features:     %d",    len(summary["feature_cols"]))
    log.info("  OOS mean-IC:  %+.4f", summary["mean_ic"])
    log.info("  Per-fold IC:  %s",
             ", ".join(f"{v:+.4f}" for v in summary["per_fold_ic"]))
    target = 0.08
    verdict = "PASS" if summary["mean_ic"] >= target else "BELOW target"
    log.info("  Target ≥ %.2f → %s", target, verdict)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_103")
    p.add_argument("--out", default=None,
                   help="Output artifact path (default: <strategy>/artifacts/panel-ltr.json)")
    args = p.parse_args()
    train(args.strategy, args.out)


if __name__ == "__main__":
    main()
