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

Multi-seed harness (CLAUDE.md §5.13.4 — "Single performance number =
unverified claim"). ``run_backtest`` accepts an optional ``seed`` so a
caller can pin XGBoost / NumPy non-determinism, and
``run_backtest_multi_seed`` runs K sims and aggregates DSR + PBO so the
falsifiability triple (mean ± std, DSR, PBO) is computed and printable.
"""
from __future__ import annotations

import logging
import math
import os
import random as _random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
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

    # Falsifiability layer (CLAUDE.md §5.13.4 — "Single performance number =
    # unverified claim"). DSR corrects raw Sharpe for selection bias across
    # n_trials configurations (Bailey-López de Prado 2014). PBO is the
    # Probability of Backtest Overfitting via CSCV (Bailey et al. 2015) —
    # requires multi_seed_returns to be meaningful; NaN in single-seed mode.
    dsr:                          float = float("nan")
    pbo:                          float = float("nan")
    n_trials:                     int   = 1
    # Benchmark-relative metrics vs SPY (Sharpe 1964 / Treynor-Black 1973).
    # NaN when there are < 30 overlapping observations or no SPY series.
    beta_vs_spy:                  float = float("nan")
    alpha_vs_spy:                 float = float("nan")
    information_ratio_vs_spy:     float = float("nan")

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
        # Falsifiability triple (CLAUDE.md §5.13.4). DSR deflates the raw
        # Sharpe by selection-bias across n_trials configurations; PBO is
        # NaN in single-seed mode (set by run_backtest_multi_seed).
        if (_m.isfinite(self.dsr) or _m.isfinite(self.pbo)
                or _m.isfinite(self.beta_vs_spy)
                or _m.isfinite(self.alpha_vs_spy)
                or _m.isfinite(self.information_ratio_vs_spy)):
            dsr_s = f"{self.dsr:+.4f}" if _m.isfinite(self.dsr) else "—"
            pbo_s = f"{self.pbo:.4f}"  if _m.isfinite(self.pbo) else "—"
            print(f"Falsifiability: DSR={dsr_s} (n_trials={self.n_trials})  "
                  f"PBO={pbo_s}")
            beta_s  = f"{self.beta_vs_spy:+.4f}"     if _m.isfinite(self.beta_vs_spy)              else "—"
            alpha_s = f"{self.alpha_vs_spy:+.2%}/yr" if _m.isfinite(self.alpha_vs_spy)             else "—"
            ir_s    = f"{self.information_ratio_vs_spy:+.4f}" if _m.isfinite(self.information_ratio_vs_spy) else "—"
            print(f"vs SPY: Beta={beta_s}  Alpha={alpha_s}  InfoRatio={ir_s}")
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
    seed:                 Optional[int] = None,
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

    ``seed`` (default None = legacy non-deterministic behavior) pins
    NumPy + stdlib RNGs at sim entry. Per CLAUDE.md §5.13.4 the multi-
    seed harness in :func:`run_backtest_multi_seed` uses this to make K
    sims reproducible-but-differentiated. Single-seed callers can also
    pass it for bit-stable replay.
    """
    # Per §5.13.4: pin RNGs BEFORE recursion or adapter construction so
    # any path (snapshot=True/False, A/B with frozen config) sees the
    # same starting state. Side effect on globals is intentional — the
    # whole point is determinism. Idempotent when seed is None.
    _apply_seed(seed)
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
                seed                 = seed,
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


# ── Multi-seed harness (CLAUDE.md §5.13.4) ───────────────────────────────────
#
# Single-sim numbers are unfalsifiable: the 2026-05-09 +6.77% APY claim and
# its 8-hour-later +1.97% replay both came from one-shot sims. The harness
# below runs K sims with different seeds, stacks their daily returns into a
# (T × K) matrix, and feeds the matrix to ``compute_perf_triple`` so DSR +
# PBO are computed across the seed ensemble. Callers see ``mean ± std`` plus
# a CSCV-based PBO and an ensemble-deflated DSR.


def _apply_seed(seed: Optional[int]) -> None:
    """Pin NumPy + stdlib RNGs (§5.13.4 reproducibility primitive).

    Idempotent when ``seed`` is None — preserves the legacy non-deterministic
    behavior so existing notebook/script callers are unaffected. When a seed
    is provided we also seed Python's ``random`` since some upstream Tasks
    (e.g. tie-breaking in selection) draw from it.
    """
    if seed is None:
        return
    np.random.seed(int(seed))
    _random.seed(int(seed))


def _resolve_seeds(seeds: Union[list[int], int]) -> list[int]:
    """Normalize the ``seeds`` argument of :func:`run_backtest_multi_seed`.

    Accepts an explicit list (used as-is) or an int K (auto-generates
    ``list(range(K))``). Per §5.13.4 the floor is 5 — fewer seeds means
    sharpe_std lacks degrees of freedom to be informative.
    """
    if isinstance(seeds, int):
        if seeds < 1:
            raise ValueError(f"seeds must be >= 1, got {seeds}")
        return list(range(seeds))
    if not isinstance(seeds, (list, tuple)):
        raise TypeError(f"seeds must be int or list, got {type(seeds).__name__}")
    out = [int(s) for s in seeds]
    if not out:
        raise ValueError("seeds list must be non-empty")
    return out


def _stack_returns_matrix(per_seed_results: "list[SimResult]") -> "np.ndarray | None":
    """Stack per-seed daily-return columns into a (T × K) matrix.

    Returns None when there are < 2 seeds OR no seed has ≥ 2 valid bars
    (meaningful PBO requires K ≥ 2 columns and ≥ 2 rows). Aligns columns
    by inner-joining each seed's equity index — sims that diverge in
    bar count (e.g. early termination) get truncated to the common range.
    """
    if len(per_seed_results) < 2:
        return None
    series_list: list[pd.Series] = []
    for res in per_seed_results:
        eq = res.equity_df
        if eq is None or eq.empty or "portfolio" not in eq.columns:
            continue
        rets = eq["portfolio"].pct_change().dropna()
        if len(rets) >= 2:
            series_list.append(rets)
    if len(series_list) < 2:
        return None
    df = pd.concat(series_list, axis=1, join="inner").dropna()
    if df.shape[0] < 2 or df.shape[1] < 2:
        return None
    return df.to_numpy(dtype=float)


def _compute_action_consistency(per_seed_results: "list[SimResult]") -> float:
    """Fraction of trade-bars on which > 50% of seeds agreed on action.

    For each (date, ticker, action) triple we count how many seeds emitted
    it; an entry "agrees" when count > K/2. Returns the fraction of
    entries that agreed, or NaN if there are no trades anywhere. This
    is a coarse-but-cheap proxy for "are these K sims doing the same
    thing under the hood" — a complement to the headline-Sharpe view.
    """
    K = len(per_seed_results)
    if K < 2:
        return float("nan")
    counter: Counter = Counter()
    for res in per_seed_results:
        # De-dupe within a seed: a (date, ticker, action) is at most 1 vote
        # per seed regardless of fill quantity.
        seed_keys = {
            (str(t.get("date")), str(t.get("ticker")), str(t.get("action")))
            for t in (res.trade_log or [])
        }
        counter.update(seed_keys)
    if not counter:
        return float("nan")
    threshold = K / 2.0
    agreed = sum(1 for v in counter.values() if v > threshold)
    return float(agreed / len(counter))


def _aggregate_perf(per_seed_results: "list[SimResult]") -> dict:
    """Compute mean ± std across seeds + DSR/PBO from the K-column matrix.

    Sharpe / APY / Calmar / MaxDD / Sortino get an across-seed mean+std.
    DSR + PBO are computed from the stacked matrix when K ≥ 2; otherwise
    we fall back to the headline seed's own DSR (single-seed mode).
    """
    finite = lambda xs: [x for x in xs if math.isfinite(x)]  # noqa: E731

    def _mean_std(getter):
        vals = finite([getter(r) for r in per_seed_results])
        if not vals:
            return float("nan"), float("nan")
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if len(vals) > 1 else float("nan")
        return m, s

    sharpe_mean, sharpe_std = _mean_std(lambda r: r.sharpe)
    apy_mean,    apy_std    = _mean_std(lambda r: r.apy)
    sortino_mean, sortino_std = _mean_std(lambda r: r.sortino)
    calmar_mean, calmar_std = _mean_std(lambda r: r.calmar)
    mdd_mean,    mdd_std    = _mean_std(lambda r: r.max_dd)

    # DSR + PBO from the stacked seed matrix (CLAUDE.md §5.13.4).
    matrix = _stack_returns_matrix(per_seed_results)
    dsr_val = float("nan")
    pbo_val = float("nan")
    if matrix is not None:
        try:
            from kernel.metrics import compute_perf_triple  # noqa: PLC0415
            # Use the headline seed's return series as the "observed"
            # series; the matrix as the K-trial ensemble. n_trials = K
            # so the deflator scales with how many seeds we ran.
            triple = compute_perf_triple(
                returns=matrix[:, 0],
                n_trials=matrix.shape[1],
                multi_seed_returns=matrix,
            )
            dsr_val = float(triple["dsr"])
            pbo_val = float(triple["pbo"])
        except Exception as exc:  # noqa: BLE001
            log.warning("multi_seed: aggregate perf-triple failed: %s", exc)

    return {
        "sharpe_mean": sharpe_mean, "sharpe_std": sharpe_std,
        "apy_mean": apy_mean,       "apy_std": apy_std,
        "sortino_mean": sortino_mean, "sortino_std": sortino_std,
        "calmar_mean": calmar_mean, "calmar_std": calmar_std,
        "max_dd_mean": mdd_mean,    "max_dd_std": mdd_std,
        "dsr": dsr_val, "pbo": pbo_val,
    }


@dataclass
class MultiSeedSimResult:
    """Aggregate of K sims with mean ± std + DSR + PBO (§5.13.4).

    Holds the per-seed ``SimResult`` list AND the across-seed aggregates.
    The headline sim is ``per_seed_results[0]`` (lowest seed) — when a
    caller wants a single SimResult for downstream code that doesn't yet
    speak multi-seed, they can use ``self.headline``.
    """
    per_seed_results: list[SimResult]
    seeds:            list[int]

    # Mean ± std across seeds (§5.13.4 falsifiability layer).
    sharpe_mean:      float = float("nan")
    sharpe_std:       float = float("nan")
    apy_mean:         float = float("nan")
    apy_std:          float = float("nan")
    sortino_mean:     float = float("nan")
    sortino_std:      float = float("nan")
    calmar_mean:      float = float("nan")
    calmar_std:       float = float("nan")
    max_dd_mean:      float = float("nan")
    max_dd_std:       float = float("nan")

    # DSR (selection-bias correction across K trials) + PBO (CSCV across
    # the K-column return matrix). Both NaN when K < 2.
    dsr:              float = float("nan")
    pbo:              float = float("nan")

    # Coarse "do the seeds agree on what to trade" diagnostic. NaN with K=1.
    majority_vote_action_consistency: float = float("nan")

    @property
    def n_seeds(self) -> int:
        return len(self.per_seed_results)

    @property
    def headline(self) -> SimResult:
        """First-seed result (lowest seed). Used when caller needs a
        single ``SimResult`` for legacy downstream code."""
        if not self.per_seed_results:
            raise RuntimeError("MultiSeedSimResult has no per-seed results")
        return self.per_seed_results[0]

    def print_summary(self) -> None:
        K = self.n_seeds
        print(f"Multi-seed sim complete: K={K} seeds={self.seeds}")
        if K == 1:
            # Degenerate K=1 — defer to the headline SimResult's own
            # per-sim summary.
            self.headline.print_summary()
            return
        # Mean ± std across seeds.
        def _line(label: str, m: float, s: float, fmt: str = "+.2f") -> str:
            if not math.isfinite(m):
                return f"{label}=—"
            std_s = f"±{s:{fmt}}" if math.isfinite(s) else "±—"
            return f"{label}={m:{fmt}} {std_s}"
        print(
            _line("APY", self.apy_mean, self.apy_std, "+.2%") + "  |  "
            + _line("Sharpe", self.sharpe_mean, self.sharpe_std) + "  |  "
            + _line("Sortino", self.sortino_mean, self.sortino_std)
        )
        print(
            _line("Calmar", self.calmar_mean, self.calmar_std) + "  |  "
            + _line("MaxDD", self.max_dd_mean, self.max_dd_std, ".1%")
        )
        dsr_s = f"{self.dsr:+.4f}" if math.isfinite(self.dsr) else "—"
        pbo_s = f"{self.pbo:.4f}"  if math.isfinite(self.pbo) else "—"
        cons = self.majority_vote_action_consistency
        cons_s = f"{cons:.1%}" if math.isfinite(cons) else "—"
        print(f"Falsifiability: DSR={dsr_s}  PBO={pbo_s}  "
              f"AgreeRate={cons_s}")


def _run_one_seed(
    seed: int,
    base_kwargs: dict,
) -> SimResult:
    """Run a single sim at the given seed. Helper for the K-seed loop.

    Kept tiny so the parallel/sequential branches in
    :func:`run_backtest_multi_seed` are visually identical.
    """
    kwargs = dict(base_kwargs)
    kwargs["seed"] = seed
    return run_backtest(**kwargs)


def run_backtest_multi_seed(
    *,
    seeds: Union[list[int], int] = 5,
    parallel: bool = False,
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
) -> MultiSeedSimResult:
    """Run K sims with different seeds and aggregate (§5.13.4).

    ``seeds`` accepts either an int K (auto-generates 0..K-1) or an
    explicit list. Default 5 — the §5.13.4 floor for ``mean ± std``.

    ``parallel=True`` uses ``ThreadPoolExecutor`` (XGBoost releases the
    GIL during prediction). Sequential is the default since it's the
    safe path; parallel is opt-in for callers that have verified their
    SimAdapter is thread-safe in their config.

    Returns a :class:`MultiSeedSimResult` containing per-seed ``SimResult``
    objects plus the aggregate triple.
    """
    seed_list = _resolve_seeds(seeds)
    base_kwargs = dict(
        config               = config,
        strategy_dir         = strategy_dir,
        ohlcv                = ohlcv,
        spy_df               = spy_df,
        sector_etf_map       = sector_etf_map,
        initial_cash         = initial_cash,
        fallback_corr        = fallback_corr,
        panel_feature_frames = panel_feature_frames,
        panel_factor_frames  = panel_factor_frames,
        backtest_start       = backtest_start,
        backtest_end         = backtest_end,
        snapshot             = snapshot,
    )

    if parallel and len(seed_list) > 1:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        with ThreadPoolExecutor(max_workers=min(len(seed_list), 10)) as pool:
            futures = [pool.submit(_run_one_seed, s, base_kwargs)
                       for s in seed_list]
            per_seed_results = [f.result() for f in futures]
    else:
        per_seed_results = [_run_one_seed(s, base_kwargs) for s in seed_list]

    agg = _aggregate_perf(per_seed_results)
    consistency = _compute_action_consistency(per_seed_results)

    return MultiSeedSimResult(
        per_seed_results = per_seed_results,
        seeds            = seed_list,
        sharpe_mean      = agg["sharpe_mean"],
        sharpe_std       = agg["sharpe_std"],
        apy_mean         = agg["apy_mean"],
        apy_std          = agg["apy_std"],
        sortino_mean     = agg["sortino_mean"],
        sortino_std      = agg["sortino_std"],
        calmar_mean      = agg["calmar_mean"],
        calmar_std       = agg["calmar_std"],
        max_dd_mean      = agg["max_dd_mean"],
        max_dd_std       = agg["max_dd_std"],
        dsr              = agg["dsr"],
        pbo              = agg["pbo"],
        majority_vote_action_consistency = consistency,
    )


__all__ = [
    "SimResult",
    "MultiSeedSimResult",
    "run_backtest",
    "run_backtest_multi_seed",
    "run_backtest_via_pipeline",
    "Counter",
]
