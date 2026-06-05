"""WF cut loader — §8 Step 4e (PR follow-up to #131).

Converts a walk-forward decision-trace artifact into a sequence of
:class:`AllocatorReplayBar` consumable by the §8 Step 4 offline A/B
replay harness (``allocator_replay.replay_all``).

The harness math (PR #131) accepts any ``Sequence[AllocatorReplayBar]``
— this module is the data adapter that materialises those bars from
the recorded sim decision-trace + realised forward returns.

Inputs (read-only)
------------------

- ``data/sim_runs.db::score_distribution`` — per-bar (date, ticker, mu,
  sigma, regime) recorded by the prod / sim panel scorer.
- ``data/sim_runs.db::ticker_forward_returns`` — per-bar realised
  forward returns at horizons {1d, 5d, 20d, 60d} relative to
  ``as_of_date``.

Outputs
-------

- ``list[AllocatorReplayBar]`` — one bar per (date, regime) tuple in
  [start_date, end_date]. Each bar carries a :class:`ConstraintSnapshot`
  with the sensible per-regime defaults documented below, the per-
  ticker μ̂/σ̂ vectors, and the realised forward return vector at
  ``fwd_horizon_days``.

Constraint defaults (sensible, not load-bearing)
------------------------------------------------

Per spec for §8 Step 4e — these are NOT prod constraints; they bound
the offline A/B replay to the same hard-constraint family the prod QP
sees. Sector / correlation caps are deliberately omitted (follow-up
work — see CLAUDE.md §3.5 single-source-of-truth on sector data).

- ``max_position_pct`` — 0.15 in ``BULL_CALM``, 0.20 elsewhere. Mirrors
  the per-regime conviction cap range used by prod (golden config
  ``regime_params.<R>.long_short.max_position_pct``) without binding
  to any specific run.
- ``cash_reserve`` — 0.05 (5%). Same as the prod default floor.
- ``dw_max`` — per-asset 0.10 (10% per-bar slippage cap). Wide enough
  that the replay rarely binds; cap-violation accounting is the
  harness's job, not the loader's.
- ``turnover_max`` — 1.0 (100% L1). Loose; lets the replay measure
  what each allocator *wants* to trade, not what fits.
- ``wash_sale_mask`` — all-False. The DB does not carry a stamped
  wash-sale mask; the replay still validates feasibility against
  the snapshot, so omitting the mask is the correct "no information"
  default.

Loud-failure modes
------------------

- DB missing → :class:`SystemExit` with the same message shape as
  ``scripts/measure_mu_hat_autocorrelation_by_regime.py`` (#128
  pattern). The replay harness is run from CLI; SystemExit at
  load-time is the user-friendly path.
- Empty date range → empty list (well-defined, no warning — the
  harness's ``replay_all`` returns empty result dicts and the
  follow-up Step 4c DSR / PBO reporters handle the "n_bars=0" case).

References
----------

- PR #131 — A/B replay harness math (paired daily returns / Sharpe /
  per-regime stratification).
- PR #128 — autocorrelation script's friendly SystemExit pattern.
- CLAUDE.md §7.2 — sanity discipline (this loader does NOT report
  any IC / Sharpe number; it only materialises bars).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from kernel.portfolio_qp.allocator_replay import AllocatorReplayBar
from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot


# Sensible per-regime caps. NOT prod constraints — they bound the
# offline A/B replay's hard-constraint family. The prod golden config
# is the source of truth for live sizing; this is replay defaults.
_MAX_POSITION_PCT_BY_REGIME: dict[str, float] = {
    "BULL_CALM": 0.15,
}
_DEFAULT_MAX_POSITION_PCT = 0.20
_DEFAULT_CASH_RESERVE = 0.05
_DEFAULT_DW_MAX_PER_ASSET = 0.10
_DEFAULT_TURNOVER_MAX = 1.0

# Map fwd_horizon_days → ticker_forward_returns column. Hard fail on
# unsupported horizons rather than silently picking the wrong column.
_FWD_HORIZON_COLUMNS: dict[int, str] = {
    1: "fwd_1d",
    5: "fwd_5d",
    10: "fwd_10d",
    20: "fwd_20d",
    60: "fwd_60d",
}


def _max_position_pct_for_regime(regime: Optional[str]) -> float:
    """Per-regime hard cap (§1 PRIME DIRECTIVE — regime-conditional)."""
    if regime is None:
        return _DEFAULT_MAX_POSITION_PCT
    return _MAX_POSITION_PCT_BY_REGIME.get(regime, _DEFAULT_MAX_POSITION_PCT)


def _build_sector_matrix(
    tickers: Sequence[str],
    sector_map: dict,
    max_per_sector: int,
    per_name_cap: float,
):
    """Mirror BuildSectorConstraintMatrixTask (tasks.py): build the per-cut
    sector indicator + cap vector from today's sector_map (#136 / #154
    Step-4h, Option 2 — snapshot today's map). Returns
    (S (m,n) 0/1, cap_vec (m,), names) or (None, None, None) when no
    sector is mapped (replay then runs sector-blind for that bar, and
    constraint_fidelity flags it).

    Algorithm matches prod: sector cap = max_per_sector * per_name_cap.
    """
    sector_to_idx: dict[str, list[int]] = {}
    for j, t in enumerate(tickers):
        sec = sector_map.get(t)
        if sec and isinstance(sec, str):
            sector_to_idx.setdefault(sec, []).append(j)
    if not sector_to_idx:
        return None, None, None
    names = sorted(sector_to_idx)
    m, n = len(names), len(tickers)
    S = np.zeros((m, n), dtype=float)
    for row, name in enumerate(names):
        for j in sector_to_idx[name]:
            S[row, j] = 1.0
    cap_vec = np.full(m, float(max_per_sector) * float(per_name_cap), dtype=float)
    return S, cap_vec, tuple(names)


def _build_snapshot(
    tickers: Sequence[str],
    regime: Optional[str],
    *,
    sector_map: Optional[dict] = None,
    max_per_sector: int = 0,
) -> ConstraintSnapshot:
    """Build a long-only :class:`ConstraintSnapshot` with replay defaults.

    Sector caps (#136 / #154 Step-4h): when a ``sector_map`` + positive
    ``max_per_sector`` are supplied, populate ``sector_indicator`` /
    ``sector_cap_vec`` from today's map (Option 2 — snapshot today's
    sector_map). This makes the replay sector-cap-aware so
    ``constraint_fidelity.decision_grade`` can be True. When omitted the
    snapshot stays sector-blind and constraint_fidelity flags it (the
    pre-Step-4h behavior, preserved for callers that pass nothing).
    """
    n = len(tickers)
    cap = _max_position_pct_for_regime(regime)
    w_upper = np.full(n, cap, dtype=float)
    sector_indicator = sector_cap_vec = sector_names = None
    if sector_map and max_per_sector > 0:
        sector_indicator, sector_cap_vec, sector_names = _build_sector_matrix(
            tickers, sector_map, max_per_sector, cap,
        )
    return ConstraintSnapshot(
        n=n,
        tickers=tuple(tickers),
        w_current=np.zeros(n, dtype=float),
        w_upper_hard=w_upper.copy(),
        w_upper=w_upper.copy(),
        w_lower=0.0,
        dw_max=np.full(n, _DEFAULT_DW_MAX_PER_ASSET, dtype=float),
        cash_reserve=_DEFAULT_CASH_RESERVE,
        turnover_max=_DEFAULT_TURNOVER_MAX,
        drawdown=0.0,
        drawdown_limit=1.0,
        gross_max=None,
        wash_sale_mask=np.zeros(n, dtype=bool),
        regime=regime,
        sector_indicator=sector_indicator,
        sector_cap_vec=sector_cap_vec,
        sector_names=sector_names,
    )


def _fwd_column(fwd_horizon_days: int) -> str:
    """Resolve the ``ticker_forward_returns`` column name. Loud on miss."""
    col = _FWD_HORIZON_COLUMNS.get(int(fwd_horizon_days))
    if col is None:
        supported = sorted(_FWD_HORIZON_COLUMNS.keys())
        raise ValueError(
            f"fwd_horizon_days={fwd_horizon_days} not supported by the "
            f"ticker_forward_returns table. Supported horizons (days): "
            f"{supported}."
        )
    return col


def load_replay_bars_from_sim_db(
    db_path: "str | Path",
    start_date: str,
    end_date: str,
    *,
    fwd_horizon_days: int = 60,
    cost_per_trade_bps: float = 5.0,
    sector_map: Optional[dict] = None,
    max_per_sector: int = 0,
) -> list[AllocatorReplayBar]:
    """Build a list of :class:`AllocatorReplayBar` from the sim decision trace.

    Parameters
    ----------
    db_path : str | Path
        Path to ``sim_runs.db``. Missing → :class:`SystemExit` with a
        friendly message (the autocorrelation script's #128 pattern).
    start_date, end_date : str
        Inclusive ISO date range (``YYYY-MM-DD``). Bars are emitted
        for every date in this window where the join below produces ≥
        2 rows with both ``mu`` / ``sigma`` populated AND a matching
        forward-return row at ``fwd_horizon_days``. Dates outside the
        DB's coverage simply produce no bars; the harness gracefully
        handles an empty input.
    fwd_horizon_days : int
        Realised-return horizon. Must be one of {1, 5, 10, 20, 60} —
        the columns the ``ticker_forward_returns`` schema currently
        carries. Default 60 days matches the §8 prod label horizon
        (CLAUDE.md §2 — panel-LTR labels are ``fwd_60d``).
    cost_per_trade_bps : float
        Round-trip transaction cost stamped onto every bar. Default
        5 bp matches the §8 Step 4b harness default.

    Returns
    -------
    list[AllocatorReplayBar]
        One bar per date in [start_date, end_date] with ≥ 2 usable
        rows. Ticker order within a bar is deterministic (sorted
        ascending) so paired-daily-returns comparisons are stable
        across runs.

    Raises
    ------
    SystemExit
        DB path does not exist. The replay CLI catches this and
        prints the message; the loader is meant to be used from
        scripts.
    ValueError
        Unsupported ``fwd_horizon_days``.
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise SystemExit(
            f"DB not found: {db_path}\n"
            "Note: data/ is gitignored. Provide an explicit db_path "
            "for an external decision-trace DB, or run a sim cycle "
            "to populate data/sim_runs.db."
        )
    fwd_col = _fwd_column(fwd_horizon_days)

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # Single query — date / ticker / mu / sigma / regime joined to
        # the matching realised return. NULL rows on either side drop
        # naturally via INNER JOIN + IS NOT NULL.
        cur.execute(
            f"""
            SELECT s.date, s.ticker, s.mu, s.sigma, s.regime, t.{fwd_col}
            FROM score_distribution s
            INNER JOIN ticker_forward_returns t
                ON s.date = t.as_of_date AND s.ticker = t.ticker
            WHERE s.date BETWEEN ? AND ?
              AND s.mu IS NOT NULL
              AND s.sigma IS NOT NULL
              AND t.{fwd_col} IS NOT NULL
            ORDER BY s.date ASC, s.ticker ASC
            """,
            (start_date, end_date),
        )
        rows = cur.fetchall()
    finally:
        conn.close()

    # Group by date. Ticker ordering is the SQL ORDER BY ascending —
    # paired-daily-returns analysis depends on stable column order
    # across runs (§7.13 process safety: determinism).
    bars: list[AllocatorReplayBar] = []
    current_date: Optional[str] = None
    tickers: list[str] = []
    mus: list[float] = []
    sigmas: list[float] = []
    fwds: list[float] = []
    regime_at_date: Optional[str] = None

    def _emit() -> None:
        """Flush accumulator → one AllocatorReplayBar."""
        # Minimum 2 names: a single-asset allocator decision is not a
        # meaningful "selection" exercise; the harness's per-regime
        # Sharpe also needs ≥ 2 daily-return samples downstream to be
        # well-defined.
        if current_date is None or len(tickers) < 2:
            return
        snap = _build_snapshot(
            tickers, regime_at_date,
            sector_map=sector_map, max_per_sector=max_per_sector,
        )
        bars.append(
            AllocatorReplayBar(
                bar_date=current_date,
                snap=snap,
                mu=np.asarray(mus, dtype=float),
                sigma=np.asarray(sigmas, dtype=float),
                fwd_return=np.asarray(fwds, dtype=float),
                regime=regime_at_date,
                cost_per_trade_bps=float(cost_per_trade_bps),
            )
        )

    for date_str, ticker, mu, sigma, regime, fwd in rows:
        if date_str != current_date:
            _emit()
            current_date = date_str
            tickers = []
            mus = []
            sigmas = []
            fwds = []
            regime_at_date = regime
        # Regime within a single date must be constant. The
        # score_distribution table stamps one regime per (run_id,
        # date) — if the DB has drift, the FIRST row wins and we
        # tag the bar accordingly. (Cross-run regime drift on the
        # same date is a separate data-quality concern.)
        tickers.append(str(ticker))
        mus.append(float(mu))
        sigmas.append(float(sigma))
        fwds.append(float(fwd))
    _emit()
    return bars


__all__ = ["load_replay_bars_from_sim_db"]
