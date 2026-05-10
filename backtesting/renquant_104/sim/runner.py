"""run_backtest — drive the OOS portfolio simulation through InferencePipeline.

The sim runs the exact same Jobs and Tasks as LEAN (`main.py`) and the
live runner (`live/runner.py`): `SimAdapter` builds an `InferenceContext`
per bar, `InferencePipeline` processes it, `SimAdapter.commit` applies
the emitted trades to a simulated portfolio. One source of truth for
decision logic across LEAN, live, and sim.

Panel scoring: when the caller provides `panel_feature_frames` +
`panel_factor_frames` (neutralized + raw factor matrices over full
history), `SimAdapter` forwards the `today`-restricted slice to each
per-bar `InferenceContext` so `PanelScoringJob` can apply the panel
scorer in the same way as LEAN's `LeanAdapter`.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

log = logging.getLogger("sim.runner")


@dataclass
class SimResult:
    equity_df:    pd.DataFrame                  # date → portfolio, regime
    trade_log:    list[dict]                    # buy/sell records
    rotation_log: list[dict]                    # ROTATION_TREE/REJECT/EXEC events
    final_value:  float
    total_return: float
    apy:          float
    win_rate:     float
    avg_hold:     float
    avg_pnl:      float
    total_tax:    float
    exit_reasons: dict[str, int]
    rotations:    list[dict]                    # paired sell/buy summary

    # Activity monitoring — see kernel.pipeline.task_monitor.MonitorIdleStreakTask.
    # These are computed post-hoc from trade_log so they always reflect the
    # entire run, even if the pipeline's streak counters reset between bars.
    longest_no_trade_streak:   int = 0          # consecutive days without buy/sell
    longest_no_candidate_streak: int = 0        # read from ctx.counters if present
    first_trade_date:          "str | None" = None
    last_activity_date:        "str | None" = None

    # Risk-adjusted metrics (2026-05-02 §3 instrumentation). All annualized
    # at 252 trading days. NaN signals "not enough data" rather than 0.0.
    # Computed from equity_df["portfolio"] in build_result().
    sharpe:           float = float("nan")     # Sharpe (rf=0)
    sortino:          float = float("nan")     # downside-deviation Sharpe
    calmar:           float = float("nan")     # APY / Max DD
    max_dd:           float = float("nan")     # max peak-to-trough drawdown (positive fraction)
    ann_vol:          float = float("nan")     # annualized return volatility

    @property
    def buys(self) -> list[dict]:
        return [t for t in self.trade_log if t["action"] == "buy"]

    @property
    def sells(self) -> list[dict]:
        return [t for t in self.trade_log if t["action"] == "sell"]

    def print_summary(self) -> None:
        import math as _m   # noqa: PLC0415
        print(f"Simulation complete: {len(self.equity_df)} days")
        print(f"Final value: ${self.final_value:,.0f}  |  "
              f"Return: {self.total_return:.1%}  |  APY: {self.apy:.1%}")
        # Risk-adjusted metrics (2026-05-02). Skip lines whose value is
        # NaN (= "not enough data") to keep the summary clean on degenerate
        # zero-trade sims.
        if _m.isfinite(self.sharpe) or _m.isfinite(self.max_dd):
            sharpe_s  = f"{self.sharpe:+.2f}"  if _m.isfinite(self.sharpe)  else "—"
            sortino_s = f"{self.sortino:+.2f}" if _m.isfinite(self.sortino) else "—"
            calmar_s  = f"{self.calmar:+.2f}"  if _m.isfinite(self.calmar)  else "—"
            mdd_s     = f"{self.max_dd:.1%}"   if _m.isfinite(self.max_dd)  else "—"
            vol_s     = f"{self.ann_vol:.1%}"  if _m.isfinite(self.ann_vol) else "—"
            print(f"Risk: Sharpe={sharpe_s}  Sortino={sortino_s}  "
                  f"Calmar={calmar_s}  MaxDD={mdd_s}  Vol={vol_s}")
        print(f"Trades: {len(self.buys)} buys, {len(self.sells)} sells  |  "
              f"Win rate: {self.win_rate:.0%}")
        if self.sells:
            print(f"Avg hold: {self.avg_hold:.0f}d  |  "
                  f"Avg P&L/trade: {self.avg_pnl:.1%}  |  "
                  f"Total tax: ${self.total_tax:,.0f}")
            print(f"Exit reasons: {self.exit_reasons}")
        if self.longest_no_trade_streak:
            marker = "⚠️  " if self.longest_no_trade_streak > 15 else ""
            print(f"{marker}Longest no-trade streak: {self.longest_no_trade_streak}d"
                  f"  |  first trade: {self.first_trade_date or '—'}")
        if self.rotations:
            print(f"\n── Rotations ({len(self.rotations)}) ──")
            for r in self.rotations:
                print(f"  [{r['date']}] {r['sell']:<5} → {r['buy']:<5}  "
                      f"sold_pnl={r['pnl_pct']:+.1%}  hold={r['hold_days']:>3}d  "
                      f"tax=${r['tax']:>7,.0f}")
        else:
            print("\nNo rotations triggered this run.")


def run_backtest(
    *,
    config:               dict,
    strategy_dir:         Path,
    ohlcv:                dict[str, pd.DataFrame],
    spy_df:               pd.DataFrame,
    sector_etf_map:       dict[str, str],
    initial_cash:         float = 100_000.0,
    fallback_corr:        dict | None = None,
    panel_feature_frames: "dict[str, pd.DataFrame] | None" = None,
    panel_factor_frames:  "dict[str, pd.DataFrame] | None" = None,
    backtest_start:       "str | None" = None,
    backtest_end:         "str | None" = None,
    snapshot:             bool = True,
) -> SimResult:
    """Run the OOS sim through SimAdapter + InferencePipeline.

    ``snapshot`` defaults to True (2026-04-24 policy change): every sim
    call freezes a copy of `strategy_dir` (artifacts + models +
    strategy_config*.json) into a tmp location for the duration of the
    run, then deletes the snapshot. This prevents a concurrent retrain
    (notebook + daily cron + manual script) from mutating the sim's
    view mid-run. Pass ``snapshot=False`` to opt out (e.g. automated
    tests that don't care about isolation, or smoke tests that need
    sub-millisecond startup).
    """
    # Snapshot wrapper — recurse into the same function with snapshot=False
    # so the core body stays linear. kernel/artifact_snapshot.py handles
    # the tmp-dir lifecycle.
    if snapshot:
        from kernel.artifact_snapshot import snapshot_artifacts_ctx  # noqa: PLC0415
        import json as _json
        with snapshot_artifacts_ctx(strategy_dir) as snap_dir:
            cfg = dict(config)
            cfg["_strategy_dir"] = str(snap_dir)
            # Re-load the frozen config if present so the caller's
            # in-memory mutations of `config` don't leak into the run.
            snap_cfg_path = Path(snap_dir) / "strategy_config.json"
            if snap_cfg_path.exists():
                # Audit fix SNAPSHOT-OVERRIDE-WARN (2026-04-26):
                # Detect callers passing in-memory config mutations that
                # diverge from disk — they're about to be silently
                # discarded. Warn loudly so future writers (validate
                # script, notebook A/B cells) don't repeat the bug.
                disk_cfg = _json.loads(snap_cfg_path.read_text())
                _dir_hint_keys_to_skip = {"_strategy_dir", "_strategy_name"}
                divergent: list[str] = []
                for k in set(config.keys()) | set(disk_cfg.keys()):
                    if k in _dir_hint_keys_to_skip:
                        continue
                    if config.get(k) != disk_cfg.get(k):
                        divergent.append(k)
                if divergent:
                    import logging as _logging
                    _logging.getLogger("sim.runner").warning(
                        "snapshot=True is about to OVERRIDE in-memory config "
                        "mutations with disk values. Diverged keys: %s. "
                        "If you're doing A/B with config flips, pass "
                        "snapshot=False instead.",
                        sorted(divergent)[:5],
                    )
                cfg = disk_cfg
                cfg["_strategy_dir"] = str(snap_dir)
            return run_backtest(
                config               = cfg,
                strategy_dir         = Path(snap_dir),
                ohlcv                = ohlcv,
                spy_df               = spy_df,
                sector_etf_map       = sector_etf_map,
                initial_cash         = initial_cash,
                fallback_corr        = fallback_corr,
                panel_feature_frames = panel_feature_frames,
                panel_factor_frames  = panel_factor_frames,
                backtest_start       = backtest_start,
                backtest_end         = backtest_end,
                snapshot             = False,
            )

    from adapters.sim import SimAdapter  # noqa: PLC0415
    from kernel.pipeline.pp_inference import InferencePipeline  # noqa: PLC0415

    # Resolve dates BEFORE adapter construction so the legacy-path
    # leakage guard (Track P2, 2026-05-10) can fire at init time —
    # before a single bar runs. Errors-out if backtest_end is before /
    # equal to the model's trained_date (i.e. the prod model has already
    # seen labels inside the sim window).
    _be = backtest_end or config.get("backtest_end")
    adapter = SimAdapter(
        config               = config,
        strategy_dir         = strategy_dir,
        ohlcv                = ohlcv,
        spy_df               = spy_df,
        sector_etf_map       = sector_etf_map,
        initial_cash         = initial_cash,
        fallback_corr        = fallback_corr,
        panel_feature_frames = panel_feature_frames,
        panel_factor_frames  = panel_factor_frames,
        backtest_end         = _be,
    )

    # DB separation (architecture 2026-04-24): sim runs a TRUNCATE of the
    # decision-trace tables at the start of each backtest so the 100th
    # notebook sim of the day is the only one whose rows survive. Keeps
    # sim_runs.db ephemeral while live/LEAN write to the permanent runs.db.
    if adapter._db is not None:  # noqa: SLF001
        from kernel.persistence import clear_sim_tables  # noqa: PLC0415
        deleted = clear_sim_tables(adapter._db)  # noqa: SLF001
        if deleted:
            log.info("run_backtest: cleared %d stale sim-trace rows", deleted)

    start = backtest_start or config.get("backtest_start")
    end   = backtest_end   or config.get("backtest_end")
    if start is None or end is None:
        raise ValueError("run_backtest needs backtest_start + backtest_end "
                         "(either as args or config keys)")

    bt_dates = spy_df.loc[start:end].index
    pipeline = InferencePipeline()

    log.info("run_backtest: %d bars  models=%d  panel=%s  ngboost=%s",
             len(bt_dates), len(adapter._models),                       # noqa: SLF001
             adapter._panel_scorer is not None,                          # noqa: SLF001
             adapter._ngboost_head is not None)                          # noqa: SLF001

    for today in bt_dates:
        ctx = adapter.make_context(today)
        pipeline.run(ctx)
        adapter.commit(ctx)

    return adapter.build_result()


# Back-compat alias — previous callers used run_backtest_via_pipeline
run_backtest_via_pipeline = run_backtest


__all__ = ["SimResult", "run_backtest", "run_backtest_via_pipeline", "Counter"]
