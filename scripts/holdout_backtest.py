#!/usr/bin/env python
"""B2 — Hold-out backtest (single-cut OOS sanity check).

The cheap-and-honest companion to the production-mirroring walk-forward
runner (B1). Trains the FullTrainingPipeline ONCE with sample_end =
train_end, then runs the sim from sim_start onward against the
freshly-trained-on-cutoff artifacts.

Why this is the lowest-cost honest OOS measure
----------------------------------------------
The production training stack reads sample_start / sample_end from the
strategy config. Pre-fix, every per-ticker model on disk had
`live_train_end = today`, so the entire backtest window sat INSIDE the
training set — every "OOS" APY/Sharpe number reported was pure in-sample.

This script enforces a hard cut: train ends BEFORE sim begins, with no
overlap. Result is a strict lower bound for walk-forward (production
retrains see more data; hold-out doesn't), useful as the gating "does
the strategy framework even make sense?" check before investing 47h in
the full B1 walk-forward. See `doc/roadmap.md §B2`.

Production safety
-----------------
The active strategy artifacts (panel-ltr.json, ngboost-head.json, the
per-ticker models, etc.) are ISOLATED via `snapshot_artifacts_ctx` —
training writes to the snapshot copy, never to production. Live runner
reads production artifacts and is unaffected by this script.

Usage
-----

    # Train on data through 2024-12-31, sim Jan 2025 → today
    python scripts/holdout_backtest.py \\
        --train-end 2024-12-31 \\
        --sim-start 2025-01-02 \\
        --sim-end   2026-04-30

    # Quick sanity (~30 min): use the default cutoff (1 year ago)
    python scripts/holdout_backtest.py

The output JSON is written to `data/holdout_results/<train_end>.json`
(plus a stdout summary).

Exit codes
----------
  0  — sim ran to completion (regardless of whether APY was good)
  1  — invalid args / config not found
  2  — training failed
  3  — sim failed
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("b2-holdout")


def _validate_dates(train_end: str, sim_start: str, sim_end: str) -> None:
    """Enforce the hard-cut invariant: train_end < sim_start ≤ sim_end.

    Without this, an off-by-one or a typo can silently put the training
    set inside the OOS window — which is exactly the bug B2 is meant to
    catch. Fail loud at parse time.
    """
    try:
        te = _dt.date.fromisoformat(train_end)
        ss = _dt.date.fromisoformat(sim_start)
        se = _dt.date.fromisoformat(sim_end)
    except ValueError as exc:
        raise SystemExit(f"holdout_backtest: invalid date — {exc}")

    if te >= ss:
        raise SystemExit(
            f"holdout_backtest: train_end ({train_end}) must be STRICTLY "
            f"BEFORE sim_start ({sim_start}). The whole point of B2 is "
            f"non-overlapping train/test windows."
        )
    if ss > se:
        raise SystemExit(
            f"holdout_backtest: sim_start ({sim_start}) must be ≤ "
            f"sim_end ({sim_end})."
        )


def _train_on_snapshot(snap_dir: Path, config: dict, strategy: str) -> None:
    """Run FullTrainingPipeline against the snapshot dir, with sample_end
    fixed to the hold-out train_end. Caller has already set
    config["sample_end"]; we just plumb it into the existing pipeline.
    """
    from kernel.pipeline.pp_training_full import (   # noqa: PLC0415
        FullTrainingContext, FullTrainingPipeline,
    )
    log.info("Training: sample_start=%s sample_end=%s",
             config.get("sample_start"), config.get("sample_end"))
    ctx = FullTrainingContext(
        config=config,
        strategy=strategy,
        strategy_dir=snap_dir,
        # B2 always retrains the full chain; cadence guard isn't relevant
        # to a one-shot hold-out experiment.
        force_retrain=True,
    )
    FullTrainingPipeline().run(ctx)


def _run_sim_on_snapshot(snap_dir: Path, config: dict, sim_start: str,
                          sim_end: str, initial_cash: float) -> dict:
    """Run the sim against the snapshot's artifacts; return key metrics."""
    from kernel.data import fetch_ohlcv   # noqa: PLC0415
    from sim.runner import run_backtest    # noqa: PLC0415

    config["_strategy_dir"]   = str(snap_dir)
    config["initial_cash"]    = initial_cash
    config["backtest_start"]  = sim_start
    config["backtest_end"]    = sim_end

    log.info("Sim: %s → %s on snapshot artifacts", sim_start, sim_end)

    benchmark = config.get("benchmark", "SPY")
    spy_df    = fetch_ohlcv(benchmark)
    etf_map   = config.get("sector_etf_map", {})
    ohlcv: dict = {benchmark: spy_df}
    for sym in sorted(set(config.get("watchlist", [])) | set(etf_map.values())):
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception as exc:
            log.warning("  %s: %s", sym, exc)

    result = run_backtest(
        config         = config,
        strategy_dir   = snap_dir,
        ohlcv          = ohlcv,
        spy_df         = spy_df,
        sector_etf_map = etf_map,
        snapshot       = False,   # already running inside our own snapshot
    )
    result.print_summary()

    # Risk-adjusted metrics (2026-05-02 §3 instrumentation). NaN signals
    # "not enough data" rather than 0.0 — preserve NaN through serialization
    # so the report doesn't fool a reader into thinking we measured 0.
    import math as _m  # noqa: PLC0415
    def _safe(x: float, scale: float = 1.0) -> float:
        v = float(getattr(result, "apy", 0.0))   # placeholder
        v = float(x)
        return v * scale if _m.isfinite(v) else float("nan")
    return {
        "apy":           float(getattr(result, "apy",        0.0)) * 100,
        "sharpe":        _safe(getattr(result, "sharpe",     float("nan"))),
        "sortino":       _safe(getattr(result, "sortino",    float("nan"))),
        "calmar":        _safe(getattr(result, "calmar",     float("nan"))),
        "max_dd":        _safe(getattr(result, "max_dd",     float("nan")), scale=100),
        "ann_vol":       _safe(getattr(result, "ann_vol",    float("nan")), scale=100),
        "win_rate":      float(getattr(result, "win_rate",   0.0)),
        "total_return":  float(getattr(result, "total_return", 0.0)) * 100,
        "n_buys":        int(len(getattr(result, "buys",  []) or [])),
        "n_sells":       int(len(getattr(result, "sells", []) or [])),
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--strategy-config-name", default="strategy_config.json",
                   help="Config filename (default: strategy_config.json).")
    p.add_argument(
        "--train-end",
        # default = ~1 year before today (gives ~1y OOS window)
        default=(_dt.date.today() - _dt.timedelta(days=365)).isoformat(),
        help="Train data ends at this date (inclusive). Sim starts the "
             "next trading day. Default: 1 year ago today.",
    )
    p.add_argument(
        "--sim-start",
        default=None,
        help="Sim start (ISO). Default: train_end + 1 day.",
    )
    p.add_argument(
        "--sim-end",
        default=_dt.date.today().isoformat(),
        help="Sim end (ISO). Default: today.",
    )
    p.add_argument("--initial-cash", type=float, default=100_000)
    p.add_argument(
        "--output", default=None,
        help="Output JSON path. Default: data/holdout_results/<train_end>.json",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="Reuse whatever artifacts are in the snapshot — useful for "
             "iterating on sim-side changes without re-training.",
    )
    args = p.parse_args()

    if args.sim_start is None:
        # train_end + 1 calendar day; the sim will skip non-trading days.
        te = _dt.date.fromisoformat(args.train_end)
        args.sim_start = (te + _dt.timedelta(days=1)).isoformat()

    _validate_dates(args.train_end, args.sim_start, args.sim_end)

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    cfg_path     = strategy_dir / args.strategy_config_name
    if not cfg_path.exists():
        log.error("Strategy config not found: %s", cfg_path)
        sys.exit(1)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    config = json.loads(cfg_path.read_text())
    config["_strategy_config_name"] = args.strategy_config_name
    # The training stack already consumes config["sample_end"] (see
    # backtesting/renquant_104/kernel/pipeline/pp_training.py:DataFetchTask
    # and training_panel/pp_panel_training.py:FetchPanelDataTask).
    # B2's whole point is to set this BEFORE training so the model never
    # sees the OOS window.
    config["sample_end"] = args.train_end

    from kernel.artifact_snapshot import snapshot_artifacts_ctx   # noqa: PLC0415
    started_at = _dt.datetime.now(_dt.timezone.utc)
    log.info("B2 hold-out backtest: train_end=%s sim=%s → %s",
             args.train_end, args.sim_start, args.sim_end)

    with snapshot_artifacts_ctx(strategy_dir) as snap_dir:
        # Train phase
        if not args.skip_train:
            try:
                _train_on_snapshot(snap_dir, config, args.strategy)
            except Exception as exc:
                log.exception("Training failed: %s", exc)
                sys.exit(2)
        else:
            log.info("--skip-train set: reusing snapshot's existing artifacts")

        # Sim phase
        try:
            metrics = _run_sim_on_snapshot(
                snap_dir, config,
                args.sim_start, args.sim_end, args.initial_cash,
            )
        except Exception as exc:
            log.exception("Sim failed: %s", exc)
            sys.exit(3)

    finished_at = _dt.datetime.now(_dt.timezone.utc)

    # Build the report — every metric is labelled `_holdout` per CLAUDE.md
    # §B3 (reporting separation: never mix in-sample, hold-out, and
    # walk-forward numbers without provenance).
    report = {
        "kind":          "b2_holdout",
        "strategy":      args.strategy,
        "config_name":   args.strategy_config_name,
        "train_end":     args.train_end,
        "sim_start":     args.sim_start,
        "sim_end":       args.sim_end,
        "initial_cash":  args.initial_cash,
        "started_utc":   started_at.isoformat(),
        "finished_utc":  finished_at.isoformat(),
        "wall_seconds":  (finished_at - started_at).total_seconds(),
        # All metrics are HOLD-OUT, never in-sample. The suffix discourages
        # anyone from copy-pasting a number out of context.
        "apy_holdout":          metrics["apy"],
        "sharpe_holdout":       metrics["sharpe"],
        "sortino_holdout":      metrics["sortino"],
        "calmar_holdout":       metrics["calmar"],
        "max_dd_holdout":       metrics["max_dd"],
        "ann_vol_holdout":      metrics["ann_vol"],
        "total_return_holdout": metrics["total_return"],
        "win_rate_holdout":     metrics["win_rate"],
        "n_buys":               metrics["n_buys"],
        "n_sells":              metrics["n_sells"],
    }

    out_path = Path(args.output) if args.output else (
        REPO_ROOT / "data" / "holdout_results" / f"{args.train_end}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print()
    print("=" * 60)
    print(f"  B2 HOLD-OUT (train_end={args.train_end})")
    print("=" * 60)
    import math as _m   # noqa: PLC0415
    def _fmt(v: float, fmt: str = "{:+.3f}") -> str:
        return fmt.format(v) if _m.isfinite(v) else "—"
    print(f"  apy_holdout         {report['apy_holdout']:+.2f}%")
    print(f"  sharpe_holdout      {_fmt(report['sharpe_holdout'])}")
    print(f"  sortino_holdout     {_fmt(report['sortino_holdout'])}")
    print(f"  calmar_holdout      {_fmt(report['calmar_holdout'])}")
    print(f"  max_dd_holdout      {_fmt(report['max_dd_holdout'], '{:+.2f}%')}")
    print(f"  ann_vol_holdout     {_fmt(report['ann_vol_holdout'], '{:+.2f}%')}")
    print(f"  total_return_holdout{report['total_return_holdout']:+.2f}%")
    print(f"  win_rate_holdout    {report['win_rate_holdout']:.0%}")
    print(f"  buys / sells        {report['n_buys']} / {report['n_sells']}")
    print(f"  wall seconds        {report['wall_seconds']:.1f}")
    print(f"  report              {out_path}")
    print("=" * 60)
    print(
        "  NOTE: this is a HOLD-OUT (single-cut) result, NOT walk-forward.\n"
        "  Production retrains see more data → real live performance is\n"
        "  expected to be ≥ this number. Treat this as a strict lower bound."
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
