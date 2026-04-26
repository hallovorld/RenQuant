#!/usr/bin/env python
"""Compare panel-LTR vs per-ticker baseline on portfolio performance (Option B).

Full-pipeline OOS simulation with identical guards on both sides — only
the ranking score source differs:
  • baseline: per-ticker tournament models' calibrated rank_score
  • panel:    cross-sectional PanelScorer output (calibrated the same way
              via Platt on realised forward returns)

Both runs use the SAME per-ticker buy/sell signals (oos_signals), so we
isolate the effect of better cross-sectional ranking.

Usage::
    python scripts/compare_panel_vs_baseline.py --strategy renquant_103
    python scripts/compare_panel_vs_baseline.py --strategy renquant_103 --retrain-panel
    python scripts/compare_panel_vs_baseline.py --strategy renquant_103 --use-cached-baseline
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("compare")


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_ohlcv(fetch, symbols, provider):
    out = {}
    for sym in symbols:
        try:
            df = fetch(sym, provider=provider)
        except Exception as exc:
            log.warning("  %-6s fetch failed (%s)", sym, exc)
            continue
        if df is not None and not df.empty:
            out[sym] = df
    return out


def _build_full_history_frames(ctx, executor_workers: int):
    """Run SectorMomentum + per-ticker Neutralize+Factor on ctx.feature_frames.

    Populates ctx.sector_momentum, ctx.neutralized_frames, ctx.raw_factor_frames.
    No truncation — uses the full OHLCV history.
    """
    from training_panel.context import TickerPanelContext
    from training_panel.pp_panel_training import (
        SectorMomentumTask, TickerPanelNeutralizeJob, TickerPanelFactorJob,
    )

    SectorMomentumTask().run(ctx)
    log.info("Sector momentum built for %d sectors", len(ctx.sector_momentum))

    ticker_ctxs = [
        TickerPanelContext(
            ticker=t, ohlcv=ctx.ohlcv, sector_momentum=ctx.sector_momentum,
            ticker_sectors=ctx.ticker_sectors, config=ctx.config,
        )
        for t in ctx.watchlist if t in ctx.feature_frames
    ]
    for tc in ticker_ctxs:
        tc.feature_frame = ctx.feature_frames[tc.ticker]

    def _chain(tc):
        TickerPanelNeutralizeJob().run(tc)
        TickerPanelFactorJob().run(tc)
        return tc

    with ThreadPoolExecutor(max_workers=executor_workers, thread_name_prefix="cmp") as ex:
        futs = [ex.submit(_chain, tc) for tc in ticker_ctxs]
        done = [f.result() for f in as_completed(futs)]

    ctx.neutralized_frames = {tc.ticker: tc.neutralized_frame for tc in done
                              if tc.neutralized_frame is not None}
    ctx.raw_factor_frames  = {tc.ticker: tc.raw_factor_frame  for tc in done
                              if tc.raw_factor_frame  is not None}
    log.info("Factor frames built for %d tickers", len(ctx.raw_factor_frames))


def _train_panel_pre_cutoff(ctx, oos_cutoff, artifact_out: Path):
    """Assemble full-history panel, then filter to pre-cutoff rows before fitting.

    Factor / neutralization passes are walk-forward safe on their own, so we
    build them over the full history and only gate what the model SEES during
    CV + final fit. This keeps frame building simple while preventing the
    panel model from training on OOS labels.
    """
    from training_panel.pp_panel_training import (
        FactorZScoreTask, LabelsTask, BuildPanelTask, PanelModelJob,
    )

    ctx.config["panel_ltr"]["artifact_path"] = str(artifact_out)
    FactorZScoreTask().run(ctx)
    LabelsTask().run(ctx)
    BuildPanelTask().run(ctx)

    full_rows = len(ctx.panel)
    log.info("Panel pre-filter: dates %s → %s",
             ctx.panel["date"].min(), ctx.panel["date"].max())
    ctx.panel = ctx.panel[ctx.panel["date"] < oos_cutoff].reset_index(drop=True)
    ctx.group_sizes = (
        ctx.panel.groupby("date", sort=True).size().values.astype("int32")
    )
    ctx.panel_metadata = {
        **ctx.panel_metadata,
        "n_rows":    int(len(ctx.panel)),
        "n_tickers": int(ctx.panel["ticker"].nunique()),
        "n_dates":   int(ctx.panel["date"].nunique()),
    }
    log.info("Panel filtered: %d → %d rows (pre-cutoff only)",
             full_rows, len(ctx.panel))

    PanelModelJob().run(ctx)
    log.info("Panel artifact → %s  (mean_ic=%+.4f)",
             artifact_out, ctx.summary["mean_ic"])


def _score_panel_oos(scorer, ctx, oos_dates, nan_prone_cols):
    """For each OOS date, score all tickers. Returns {ticker: Series[date→score]}."""
    from kernel.panel_pipeline.feature_matrix import build_inference_matrix

    per_ticker_scores: dict[str, dict] = {t: {} for t in ctx.neutralized_frames}
    for date in oos_dates:
        X = build_inference_matrix(
            ctx.neutralized_frames, ctx.raw_factor_frames, pd.Timestamp(date),
            feature_cols=scorer.feature_cols, nan_prone_cols=nan_prone_cols,
        )
        if X.empty:
            continue
        scores = scorer.score(X)
        for t, s in scores.items():
            if not pd.isna(s):
                per_ticker_scores[t][date] = float(s)

    return {t: pd.Series(d, name=t).sort_index() for t, d in per_ticker_scores.items() if d}


def _run_or_load_baseline(strategy_dir, config, feature_frames, ohlcv, cache_path,
                           use_cache: bool):
    from training.tournament import run_tournament_all
    if use_cache and cache_path.exists():
        log.info("Loading cached baseline results from %s", cache_path)
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    log.info("Running tournament_all (baseline training) — this takes a while…")
    results = run_tournament_all(
        watchlist=list(feature_frames.keys()),
        feature_frames=feature_frames,
        ohlcv=ohlcv,  # must already include SPY (benchmark)
        config=config,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(results, f)
    log.info("Baseline results cached to %s", cache_path)
    return results


def _fit_panel_calibration(panel_scores: pd.Series, ticker: str, ohlcv, spy_df,
                            lookahead: int, threshold: float):
    """Fit Platt calibration on (panel_score → realised excess return > threshold)."""
    from training.scoring import fit_probability_calibration

    if ticker not in ohlcv or panel_scores.empty:
        return None
    prices = ohlcv[ticker]["close"].reindex(panel_scores.index)
    spy    = spy_df["close"].reindex(panel_scores.index).replace(0, np.nan)
    rel    = (prices / spy).replace([np.inf, -np.inf], np.nan)
    future_rel = rel.shift(-lookahead) / rel - 1.0
    try:
        return fit_probability_calibration(
            panel_scores, future_rel,
            lookahead=lookahead, threshold=threshold, score_kind="probability",
        )
    except Exception as exc:
        log.warning("  %-6s calibration fit failed: %s", ticker, exc)
        return None


def _compute_metrics(sim):
    eq = sim.equity_df["portfolio"].astype(float)
    daily_ret = eq.pct_change().dropna()
    if len(daily_ret) < 2 or daily_ret.std() == 0:
        sharpe = 0.0
    else:
        sharpe = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252))
    running_max = eq.cummax()
    max_dd = float(((eq - running_max) / running_max).min())
    n_buys = len(sim.buys)
    n_days = len(eq)
    turnover = n_buys / max(1, n_days)
    return {
        "final_value": float(sim.final_value),
        "total_return": float(sim.total_return),
        "apy": float(sim.apy),
        "sharpe": sharpe,
        "max_dd": max_dd,
        "n_buys": n_buys,
        "n_sells": len(sim.sells),
        "win_rate": float(sim.win_rate),
        "avg_hold": float(sim.avg_hold),
        "turnover": turnover,
        "exit_reasons": dict(sim.exit_reasons),
    }


def _print_comparison(base, panel):
    def row(label, b, p, fmt="{:>10}"):
        print(f"  {label:<20} " + fmt.format(b) + "  vs  " + fmt.format(p)
              + "  " + fmt.format(p - b if isinstance(p, (int, float)) else ""))

    print("\n" + "═" * 78)
    print(f"  {'Metric':<20} {'Baseline':>10}      {'Panel-LTR':>10}      {'Δ':>10}")
    print("─" * 78)
    row("Final value",  f"${base['final_value']:,.0f}",  f"${panel['final_value']:,.0f}",
        "{:>10}")
    row("Total return", f"{base['total_return']*100:+.1f}%", f"{panel['total_return']*100:+.1f}%",
        "{:>10}")
    row("APY",          f"{base['apy']*100:+.1f}%",       f"{panel['apy']*100:+.1f}%",
        "{:>10}")
    row("Sharpe",       f"{base['sharpe']:+.2f}",         f"{panel['sharpe']:+.2f}",
        "{:>10}")
    row("Max drawdown", f"{base['max_dd']*100:+.1f}%",    f"{panel['max_dd']*100:+.1f}%",
        "{:>10}")
    row("Buys / Sells", f"{base['n_buys']}/{base['n_sells']}",
                        f"{panel['n_buys']}/{panel['n_sells']}", "{:>10}")
    row("Win rate",     f"{base['win_rate']*100:.0f}%",   f"{panel['win_rate']*100:.0f}%",
        "{:>10}")
    row("Avg hold (d)", f"{base['avg_hold']:.0f}",        f"{panel['avg_hold']:.0f}",
        "{:>10}")
    row("Turnover",     f"{base['turnover']:.3f}",        f"{panel['turnover']:.3f}",
        "{:>10}")
    print("═" * 78)


def _save_chart(base_sim, panel_sim, out_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(base_sim.equity_df.index, base_sim.equity_df["portfolio"],
            label="baseline", color="#4a90e2", linewidth=1.5)
    ax.plot(panel_sim.equity_df.index, panel_sim.equity_df["portfolio"],
            label="panel-LTR", color="#e07a5f", linewidth=1.5)
    ax.set_title("Portfolio equity — baseline vs panel-LTR (OOS)")
    ax.set_ylabel("Portfolio value ($)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    log.info("Saved chart → %s", out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_103")
    p.add_argument("--retrain-panel", action="store_true",
                   help="Re-train panel artifact even if one exists")
    p.add_argument("--use-cached-baseline", action="store_true",
                   help="Load baseline results from cache if present")
    p.add_argument("--out", default=None, help="PNG chart output path")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("Strategy config not found: %s", config_path)
        sys.exit(1)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.data     import fetch_ohlcv
    from training.features   import build_all_training_features
    from training.tournament import resolve_oos_cutoff
    from training_panel.context import PanelTrainingContext
    from kernel.panel_pipeline.panel_scorer import PanelScorer
    from sim.runner import run_backtest

    config      = json.loads(config_path.read_text())
    watchlist   = config["watchlist"]
    benchmark   = config.get("benchmark", "SPY")
    provider    = config.get("data_src", "yfinance")
    sector_map  = config.get("sector_map", {})
    sector_etf_map = config.get("sector_etf_map", {})
    indicator_spec = config.get("indicator_spec", {})
    model_params   = config.get("model_params", {})
    lookahead = int(model_params.get("lookahead", 5))
    threshold = float(model_params.get("threshold", 0.03))
    oos_cutoff = resolve_oos_cutoff(config)
    log.info("OOS cutoff: %s", oos_cutoff.date())

    # ── 1. OHLCV ──────────────────────────────────────────────────────────
    needed = set(watchlist) | {benchmark} | set(sector_etf_map.values())
    log.info("Fetching OHLCV for %d symbols …", len(needed))
    ohlcv_all = _load_ohlcv(fetch_ohlcv, sorted(needed), provider)
    if benchmark not in ohlcv_all:
        log.error("Benchmark %s missing — aborting", benchmark)
        sys.exit(1)
    ohlcv = {t: ohlcv_all[t] for t in watchlist if t in ohlcv_all}
    spy_df = ohlcv_all[benchmark]
    sector_etf_ohlcv = {sec: ohlcv_all[etf] for sec, etf in sector_etf_map.items()
                        if etf in ohlcv_all}

    # ── 2. Feature frames ──────────────────────────────────────────────────
    log.info("Building per-ticker feature frames …")
    features_in = dict(ohlcv); features_in[benchmark] = spy_df
    feature_frames = build_all_training_features(
        watchlist=list(ohlcv.keys()), ohlcv=features_in,
        indicator_spec=indicator_spec,
        lookahead=lookahead, threshold=threshold,
    )
    if not feature_frames:
        log.error("No feature frames built — aborting")
        sys.exit(1)

    # ── 3. Panel training context + derived frames ────────────────────────
    ticker_sectors = {t: sector_map[t] for t in feature_frames if t in sector_map}
    panel_cfg = dict(config.get("panel_ltr", {}))
    panel_cfg.setdefault("lookahead_days",      lookahead)
    panel_cfg.setdefault("beta_window",         60)
    panel_cfg.setdefault("min_history_days",    252)
    panel_cfg.setdefault("age_warmup_days",     504)
    panel_cfg.setdefault("cv_n_splits",         5)
    panel_cfg.setdefault("cv_embargo_days",     lookahead)
    panel_cfg.setdefault("num_boost_round",     400)
    panel_cfg.setdefault("neutralize_features", True)
    panel_cfg.setdefault("nan_prone_cols",      [])

    merged_cfg = dict(config)
    merged_cfg["panel_ltr"] = panel_cfg
    ctx = PanelTrainingContext(
        config=merged_cfg, watchlist=list(feature_frames.keys()),
        ohlcv=dict(ohlcv) | {benchmark: spy_df},
        sector_etf_ohlcv=sector_etf_ohlcv,
        ticker_sectors=ticker_sectors,
    )
    ctx.feature_frames = feature_frames
    _build_full_history_frames(ctx, executor_workers=4)

    # DEBUG: sanity-check frame date ranges
    for t in ("MU", "TSM", "OXY"):
        if t in ctx.neutralized_frames:
            nf = ctx.neutralized_frames[t]
            log.info("  neutralized[%s]: %s → %s  (%d rows)",
                     t, nf.index.min(), nf.index.max(), len(nf))
        if t in ctx.raw_factor_frames:
            ff = ctx.raw_factor_frames[t]
            log.info("  raw_factor[%s]:  %s → %s  (%d rows)",
                     t, ff.index.min(), ff.index.max(), len(ff))

    # ── 4. Panel artifact (train or reuse) ─────────────────────────────────
    artifact_path = strategy_dir / "artifacts" / "panel-ltr-cmp.json"
    if args.retrain_panel or not artifact_path.exists():
        _train_panel_pre_cutoff(ctx, oos_cutoff, artifact_path)
    else:
        log.info("Reusing panel artifact: %s", artifact_path)
    scorer = PanelScorer.load(artifact_path)
    log.info("Panel scorer loaded  features=%d  mean_ic=%s",
             len(scorer.feature_cols), scorer.metadata.get("oos_mean_ic"))

    # ── 5. Score panel on OOS dates ───────────────────────────────────────
    oos_dates = [d for d in spy_df.index if d >= oos_cutoff]
    log.info("Scoring panel over %d OOS dates (%s → %s) …",
             len(oos_dates), oos_dates[0].date() if oos_dates else None,
             oos_dates[-1].date() if oos_dates else None)
    panel_score_series = _score_panel_oos(
        scorer, ctx, oos_dates,
        nan_prone_cols=panel_cfg.get("nan_prone_cols", []),
    )
    log.info("Panel scored %d tickers", len(panel_score_series))

    # ── 6. Baseline results (from cache or tournament) ─────────────────────
    cache_path = strategy_dir / "artifacts" / "baseline-results.pkl"
    ohlcv_for_tournament = dict(ohlcv)
    ohlcv_for_tournament[benchmark] = spy_df
    baseline_results = _run_or_load_baseline(
        strategy_dir, config, feature_frames, ohlcv_for_tournament, cache_path,
        use_cache=args.use_cached_baseline,
    )

    # ── 7. Build panel_results = baseline but with panel rank_score ────────
    panel_results = {}
    for ticker, r in baseline_results.items():
        new_r = dict(r)
        if ticker in panel_score_series:
            panel_series = panel_score_series[ticker]
            # Restrict to OOS index the baseline used
            if r.get("oos_raw_scores") is not None:
                panel_series = panel_series.reindex(r["oos_raw_scores"].index)
            new_r["oos_raw_scores"]    = panel_series
            new_r["score_calibration"] = _fit_panel_calibration(
                panel_series.dropna(), ticker, ohlcv, spy_df, lookahead, threshold,
            )
        panel_results[ticker] = new_r

    # ── 8. Run sims ────────────────────────────────────────────────────────
    backtest_start = oos_cutoff.strftime("%Y-%m-%d")
    backtest_end   = spy_df.index[-1].strftime("%Y-%m-%d")
    sim_cfg = dict(config)
    sim_cfg["backtest_start"] = backtest_start
    sim_cfg["backtest_end"]   = backtest_end
    if "initial_cash" not in sim_cfg:
        sim_cfg["initial_cash"] = 100_000

    # Audit fix VALIDATE-SNAPSHOT-OVERRIDE (2026-04-26): snapshot=False
    # so our in-memory config mutations (panel_ltr.artifact_path, etc)
    # don't get silently overwritten by the disk strategy_config.json.
    log.info("Running sim: baseline …")
    base_sim  = run_backtest(
        config=sim_cfg, strategy_dir=strategy_dir, results=baseline_results,
        ohlcv=ohlcv, spy_df=spy_df, sector_etf_map=sector_etf_map,
        snapshot=False,
    )
    log.info("Running sim: panel …")
    panel_sim = run_backtest(
        config=sim_cfg, strategy_dir=strategy_dir, results=panel_results,
        ohlcv=ohlcv, spy_df=spy_df, sector_etf_map=sector_etf_map,
        snapshot=False,
    )

    # ── 9. Metrics + chart ─────────────────────────────────────────────────
    base_m  = _compute_metrics(base_sim)
    panel_m = _compute_metrics(panel_sim)
    _print_comparison(base_m, panel_m)

    print("\n  Exit reasons:")
    print(f"    baseline:  {base_m['exit_reasons']}")
    print(f"    panel-LTR: {panel_m['exit_reasons']}")

    out_path = Path(args.out) if args.out else (
        strategy_dir / "img" / "compare_panel_vs_baseline.png"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _save_chart(base_sim, panel_sim, out_path)


if __name__ == "__main__":
    main()
