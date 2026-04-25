#!/usr/bin/env python
"""Export the BuildPanelTask output (in-memory `ctx.panel` DataFrame) to
the flat CSV shape that the Rust `train-panel` binary consumes.

This bridges the Python panel-build pipeline (PanelDataJob → PanelFeatureJob
→ PanelAssemblyJob.BuildPanelTask) to the Rust trainer, so we can A/B
the same input across both backends. Same columns, same labels, same
date ordering.

Output CSV format (consumed by rust/transformer_scorer/src/dataset.rs):

    date,ticker,<feature_cols...>,label

with NaN labels left as empty strings (boundary lookahead rows).

Usage:
    python scripts/export_panel_to_csv.py \\
        --strategy renquant_104 \\
        --output   /tmp/panel.csv \\
        --no-shuffle      # preserve panel build order

Read 'feature_cols' from the saved panel-ltr.json artifact when it
exists (so the Rust trainer sees exactly the same feature set the
Python XGBoost trainer trained on, in the same order).

Cost: rebuilds the panel from scratch (~3 minutes on the 99-ticker
watchlist). Run once per training-data refresh, then keep the CSV
around for repeated Rust experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", default="renquant_104")
    parser.add_argument("--output",   required=True, type=Path)
    parser.add_argument(
        "--feature-cols-from-artifact", action="store_true",
        help="Use the feature_cols field of the existing panel-ltr.json "
             "instead of recomputing from build (cheap fast path when the "
             "panel build itself is unchanged but the CSV needs rewriting).",
    )
    args = parser.parse_args()

    strategy_dir = REPO / "backtesting" / args.strategy
    if not strategy_dir.exists():
        sys.exit(f"strategy dir not found: {strategy_dir}")
    sys.path.insert(0, str(strategy_dir))

    # ── Build the panel via the actual production pipeline ────────────
    print(f"[export] strategy={args.strategy}  rebuilding panel via PanelTrainingJob …")
    from training_panel.context import PanelTrainingContext   # noqa: PLC0415
    from training_panel.pp_panel_training import (             # noqa: PLC0415
        PanelDataJob, PanelFeatureJob, PanelAssemblyJob,
    )

    cfg_path = strategy_dir / "strategy_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("_strategy_dir", str(strategy_dir))

    watchlist = cfg["watchlist"]
    sectors   = cfg.get("sector_etf_map", {})
    benchmark = cfg.get("benchmark", "SPY")

    ctx = PanelTrainingContext(
        config=cfg, watchlist=watchlist,
        ohlcv={}, sector_etf_ohlcv={}, ticker_sectors={},
        listing_dates=None,
    )

    PanelDataJob().run(ctx)
    PanelFeatureJob().run(ctx)
    PanelAssemblyJob().run(ctx)

    panel = ctx.panel
    if panel is None or panel.empty:
        sys.exit("[export] panel is empty after PanelAssemblyJob; aborting")

    feature_cols = list(ctx.feature_cols)
    if args.feature_cols_from_artifact:
        sidecar = strategy_dir / "artifacts" / "panel-ltr.json"
        if sidecar.exists():
            saved_cols = json.loads(sidecar.read_text()).get("feature_cols")
            if saved_cols:
                feature_cols = list(saved_cols)
                print(f"[export] using {len(feature_cols)} feature_cols from {sidecar}")

    # ── Layout the CSV ────────────────────────────────────────────────
    # Required columns; missing → fill NaN.
    ordered = ["date", "ticker"] + feature_cols + ["label"]
    missing = [c for c in ordered if c not in panel.columns]
    if missing:
        print(f"[export] WARNING: panel is missing columns {missing} — filled NaN")
        for c in missing:
            panel[c] = float("nan")

    out_df = panel[ordered].copy()
    # Ensure date is a string YYYY-MM-DD (the Rust loader parses ISO).
    out_df["date"] = out_df["date"].astype(str).str.slice(0, 10)
    out_df = out_df.sort_values(["date", "ticker"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False, na_rep="")

    n_dates = out_df["date"].nunique()
    n_rows  = len(out_df)
    n_tickers_avg = n_rows / n_dates if n_dates > 0 else 0
    print(
        f"[export] wrote {args.output} — {n_dates} dates × ~{n_tickers_avg:.1f} tickers/date "
        f"= {n_rows} rows, {len(feature_cols)} features"
    )
    print(f"[export] feature_cols: {feature_cols[:5]}{'…' if len(feature_cols) > 5 else ''}")


if __name__ == "__main__":
    main()
