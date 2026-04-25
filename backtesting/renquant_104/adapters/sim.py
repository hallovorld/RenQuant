"""SimAdapter — bridges a simulated portfolio to InferencePipeline.

Mirror of LeanAdapter / RunnerAdapter, but for the notebook / backtest
simulation path. Instead of talking to LEAN or a broker, SimAdapter owns
the sim's mutable state (cash, holdings, HWM, regime) and emulates
broker actions (execute sell, record buy) inside `commit()`.

Usage::

    adapter = SimAdapter(
        config=config, strategy_dir=STRATEGY_DIR,
        ohlcv=ohlcv, spy_df=spy_df,
        sector_etf_map=SECTOR_ETF, initial_cash=100_000,
        panel_feature_frames=ff, panel_factor_frames=fac,
    )
    for today in bt_dates:
        ctx = adapter.make_context(today)
        InferencePipeline().run(ctx)
        adapter.commit(ctx)
    result = adapter.build_result()

This path runs the *exact same* Jobs and Tasks as LEAN + the live runner,
so any decision added to `InferencePipeline` is automatically picked up
by the notebook simulation too. No more drift.
"""
from __future__ import annotations

import datetime
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("adapters.sim")


class SimAdapter:
    """Translate between a simulated portfolio and InferenceContext."""

    def __init__(
        self,
        *,
        config: dict,
        strategy_dir: Path,
        ohlcv: dict[str, pd.DataFrame],
        spy_df: pd.DataFrame,
        sector_etf_map: dict[str, str],
        initial_cash: float,
        fallback_corr: dict | None = None,
        panel_feature_frames: dict[str, pd.DataFrame] | None = None,
        panel_factor_frames: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        from kernel.regime import RegimeState  # noqa: PLC0415
        from kernel.config import REGIMES      # noqa: PLC0415

        self._config         = dict(config)
        self._config["_strategy_dir"] = str(strategy_dir)
        self._strategy_dir   = Path(strategy_dir)
        self._ohlcv          = ohlcv
        self._spy_df         = spy_df
        self._sector_etf_map = sector_etf_map

        # ── Load per-ticker policy artifacts (same path as live/runner) ─────
        self._models = self._load_models()

        # ── Load regime, correlation, earnings artifacts ────────────────────
        self._gmm, self._earnings, self._corr = self._load_artifacts(fallback_corr)

        # ── Panel scorer + NGBoost head (preloaded so PanelScoringJob's
        #    LoadScorerTask / LoadNGBoostTask short-circuit) ─────────────────
        self._panel_scorer  = self._try_load_panel_scorer()
        self._ngboost_head  = self._try_load_ngboost_head()
        self._panel_feature_frames = panel_feature_frames
        self._panel_factor_frames  = panel_factor_frames

        # ── Persistent sim state (emulates broker / LEAN Portfolio) ─────────
        self._cash           = float(initial_cash)
        self._initial_cash   = float(initial_cash)
        self._hwm            = float(initial_cash)
        self._skip_buys      = False
        self._holdings: dict[str, Any] = {}        # ticker → HoldingState
        self._pos_shares: dict[str, float] = {}    # ticker → shares count
        self._last_sell_date: dict[str, pd.Timestamp] = {}   # ticker → date
        self._regime_state   = RegimeState()
        self._regime_counts  = {r: 0 for r in REGIMES}
        # Monitor: persist MonitorIdleStreakTask's streak counters across bars.
        self._monitor_state: dict = {}
        # Rotation V1 persistence gate (2026-04-24): list of per-bar
        # proposed (sell, buy) pair sets, oldest first. Only populated
        # when `rotation.persistence_bars > 0`. Capped at that window.
        self._rotation_proposals: list = []

        # SPY returns buffer (last 100) + previous close for daily return calc
        self._spy_returns: list[float] = []
        self._spy_prev_close: float | None = None

        # ── Logs collected across the run ───────────────────────────────────
        self._equity_curve: list[dict]  = []
        self._trade_log:    list[dict]  = []
        self._rotation_log: list[dict]  = []

        # ── Optional SQLite decision-trace ──────────────────────────────────
        # sim writes to a SEPARATE DB (persistence.sim_db_path, default
        # data/sim_runs.db) so notebook experimentation doesn't pollute
        # the live decision-audit statistics in data/runs.db. The sim DB
        # is TRUNCATEd at the start of every run_backtest (sim.runner)
        # so only the most-recent notebook sim's rows remain.
        from kernel.persistence import get_connection  # noqa: PLC0415
        self._db = get_connection(
            config, strategy_dir=self._strategy_dir, role="sim",
        )

        # Feature cache optimization (2026-04-24): pre-compute per-ticker
        # full-range feature frames ONCE here instead of rebuilding per
        # bar in TickerSellJob/CandidateJob. 5-8x sim speedup on the
        # 570-bar 27-mo window × 42-ticker panel.
        #
        # ✅ Equivalence VERIFIED 2026-04-24: kernel.indicators.
        # build_spy_context_series replaced the scalar-broadcast
        # build_spy_context, which had been the lookahead source. Now
        # cached.loc[:t].iloc[-1] == build_feature_frame(ohlcv[:t]).iloc[-1]
        # for every bar t. See tests/test_feature_cache.py::TestEquivalence.
        #
        # Flag-gated: `sim.feature_cache_enabled: true` (default true —
        # 5-8x sim speedup). Set false to disable for debugging.
        self._feature_cache: dict = {}
        if config.get("sim", {}).get("feature_cache_enabled", True):
            self._build_feature_cache()

        log.info(
            "SimAdapter init: models=%d  gmm=%s  corr=%s  earnings=%s  "
            "panel_scorer=%s  ngboost_head=%s  feature_cache=%d tickers",
            len(self._models), self._gmm is not None, bool(self._corr),
            bool(self._earnings), self._panel_scorer is not None,
            self._ngboost_head is not None, len(self._feature_cache),
        )

    def _build_feature_cache(self) -> None:
        """One-shot: build full-range feature frame per watchlist ticker.

        Uses the same `build_feature_frame` the per-bar task would call,
        but on the FULL OHLCV range once. Per-bar tasks then slice by
        `today` instead of re-running the indicator pipeline 570×42 times.
        """
        from kernel.indicators import build_feature_frame  # noqa: PLC0415

        spy_df = self._ohlcv.get("SPY")
        if spy_df is None:
            log.warning("Feature cache: SPY OHLCV missing — skipping build")
            return

        spec    = self._config.get("indicator_spec", {})
        vol_win = int(self._config.get("regime", {}).get("vol_realized_window", 20))

        built = 0
        for ticker, df in self._ohlcv.items():
            if ticker == "SPY" or df is None or df.empty:
                continue
            frame = build_feature_frame(df, spy_df, spec, vol_win)
            if frame is not None and not frame.empty:
                self._feature_cache[ticker] = frame
                built += 1
        log.info("Feature cache built: %d/%d tickers", built, len(self._ohlcv))

    # ── Artifact loaders ────────────────────────────────────────────────────

    def _load_models(self) -> dict[str, dict]:
        """Run LoadUniverseJob: artifacts + staleness + (conditional) sharpe-floor."""
        from kernel.pipeline.job_universe import UniverseContext, LoadUniverseJob  # noqa: PLC0415
        uctx = UniverseContext(config=self._config, strategy_dir=self._strategy_dir)
        LoadUniverseJob().run(uctx)
        for ticker, reason in uctx.rejections:
            log.debug("SimAdapter: %s rejected — %s", ticker, reason)
        return uctx.loaded_models

    def _load_artifacts(self, fallback_corr):
        from kernel.regime import load_gmm_artifact  # noqa: PLC0415
        artifacts_dir = self._strategy_dir / "artifacts"
        if not artifacts_dir.exists():
            artifacts_dir = self._strategy_dir
        regime_cfg = self._config.get("regime", {})

        earnings_path = artifacts_dir / regime_cfg.get(
            "earnings_artifact", "earnings-calendar.json",
        )
        earnings_cal = {}
        if earnings_path.exists():
            try:
                earnings_cal = json.loads(earnings_path.read_text())
            except Exception as exc:
                log.warning("earnings calendar load failed: %s", exc)

        gmm = load_gmm_artifact(
            artifacts_dir / regime_cfg.get("gmm_artifact", "spy-gmm-regime.json"),
        )

        corr_path = artifacts_dir / regime_cfg.get(
            "correlation_artifact", "watchlist-correlation.json",
        )
        if corr_path.exists():
            corr_dict = json.loads(corr_path.read_text())
        elif fallback_corr is not None:
            corr_dict = fallback_corr
        else:
            corr_dict = {}
        return gmm, earnings_cal, corr_dict

    def _try_load_panel_scorer(self):
        panel_cfg = self._config.get("ranking", {}).get("panel_scoring", {})
        if not panel_cfg.get("enabled", False):
            return None
        path = Path(panel_cfg.get("artifact_path", "artifacts/panel-ltr.json"))
        if not path.is_absolute():
            path = self._strategy_dir / path
        if not path.exists():
            log.warning("SimAdapter: panel artifact not found at %s", path)
            return None
        try:
            from kernel.panel_pipeline import PanelScorer  # noqa: PLC0415
            return PanelScorer.load(path)
        except Exception as exc:
            log.warning("SimAdapter: panel scorer load failed — %s", exc)
            return None

    def _try_load_ngboost_head(self):
        ngb_cfg = (self._config.get("ranking", {})
                              .get("panel_scoring", {})
                              .get("ngboost", {}))
        if not ngb_cfg.get("enabled", False):
            return None
        path = Path(ngb_cfg.get("artifact_path", "artifacts/ngboost-head.json"))
        if not path.is_absolute():
            path = self._strategy_dir / path
        if not path.exists():
            log.warning("SimAdapter: ngboost artifact not found at %s", path)
            return None
        try:
            from training_panel.ngboost_head import NGBoostHead  # noqa: PLC0415
            return NGBoostHead.load(path)
        except Exception as exc:
            log.warning("SimAdapter: ngboost head load failed — %s", exc)
            return None

    # ── Public entry points ─────────────────────────────────────────────────

    def make_context(self, today: pd.Timestamp):
        """Build InferenceContext from current sim state + today's bar."""
        from kernel.pipeline.context import InferenceContext  # noqa: PLC0415

        today_ts = pd.Timestamp(today)
        today_date = today_ts.date() if hasattr(today_ts, "date") else today_ts

        # Update SPY returns buffer
        if today_ts in self._spy_df.index:
            spy_close = float(self._spy_df.loc[today_ts, "close"])
            if self._spy_prev_close is not None and self._spy_prev_close > 0:
                self._spy_returns.append(spy_close / self._spy_prev_close - 1.0)
                if len(self._spy_returns) > 100:
                    self._spy_returns = self._spy_returns[-100:]
            self._spy_prev_close = spy_close

        # Prices for this bar — union of models + sector ETFs
        prices: dict[str, float] = {}
        for t in self._models:
            df = self._ohlcv.get(t)
            if df is not None and today_ts in df.index:
                prices[t] = float(df.loc[today_ts, "close"])
        for _sec, etf in self._sector_etf_map.items():
            df = self._ohlcv.get(etf)
            if df is not None and today_ts in df.index:
                prices[etf] = float(df.loc[today_ts, "close"])
        # Held-position prices (in case a holding isn't in _models — defensives)
        for t in self._holdings:
            df = self._ohlcv.get(t)
            if df is not None and today_ts in df.index:
                prices[t] = float(df.loc[today_ts, "close"])

        pv = self._portfolio_value(prices)

        # last_sell_dates as date objects (pipeline expects datetime.date)
        last_sells_d: dict[str, datetime.date | None] = {}
        for sym, d in self._last_sell_date.items():
            last_sells_d[sym] = d.date() if hasattr(d, "date") else d

        # Truncated OHLCV: each ticker's DataFrame sliced to [:today_ts] so
        # no future bars are visible to the pipeline (replicates LEAN
        # "History(bars up to now)" semantics).
        truncated = {
            t: df.loc[:today_ts] for t, df in self._ohlcv.items()
        }

        ctx = InferenceContext(
            config           = self._config,
            today            = today_date,
            ohlcv            = truncated,
            spy_returns      = list(self._spy_returns),
            models           = self._models,
            gmm              = self._gmm,
            corr_matrix      = self._corr,
            earnings_calendar = self._earnings,
            holdings         = {t: self._holdings[t]
                                for t in list(self._holdings.keys())},
            last_sell_dates  = last_sells_d,
            portfolio_value  = pv,
            cash             = self._cash,
            prices           = prices,
            hwm              = self._hwm,
            skip_buys        = self._skip_buys,
            regime_state     = self._regime_state,
            regime_counts    = self._regime_counts,
            feature_cache    = self._feature_cache,
        )

        # Hand prior streak counters to MonitorIdleStreakTask; it writes back.
        ctx.monitor_state = dict(self._monitor_state)

        # Rotation V1 persistence gate: hand over the last N bars' proposed
        # (sell, buy) pair sets. BuildPairsTask reads via rotation_cfg
        # passthrough. Adapter pushes this bar's proposals in commit().
        ctx.prior_rotation_proposals = list(self._rotation_proposals)

        # Rotation V4 (thesis_symmetric) needs the sim DB to look up
        # candidate scores on each held's entry date.
        if self._db is not None:
            ctx._db = self._db   # noqa: SLF001

        # Preload panel scoring artifacts so PanelScoringJob short-circuits
        # its LoadScorerTask / LoadNGBoostTask.
        if self._panel_scorer is not None:
            ctx._panel_scorer = self._panel_scorer  # noqa: SLF001
        if self._ngboost_head is not None:
            ctx._ngboost_head = self._ngboost_head  # noqa: SLF001
        if self._panel_feature_frames is not None:
            # Slice feature/factor frames to today_ts too (no future leak)
            ctx._panel_feature_frames = {                              # noqa: SLF001
                t: df.loc[:today_ts] for t, df in self._panel_feature_frames.items()
            }
            if self._panel_factor_frames is not None:
                ctx._panel_factor_frames = {                            # noqa: SLF001
                    t: df.loc[:today_ts] for t, df in self._panel_factor_frames.items()
                }

        return ctx

    def commit(self, ctx) -> None:  # noqa: ANN001
        """Apply pipeline outputs to sim state. Mirrors LeanAdapter.commit."""
        # ── Exits ───────────────────────────────────────────────────────────
        today_ts = pd.Timestamp(ctx.today)
        trade_events_this_bar: list[dict] = []
        len_trade_log_before = len(self._trade_log)
        # Track which tickers need FULL liquidation vs partial trim. Only
        # full exits pop from holdings/pos_shares; partial trims update
        # share count in-place (see _apply_sell).
        full_exit_tickers: set[str] = set()
        for ticker, sig in ctx.exits:
            q = getattr(sig, "quantity", None)
            cur = self._pos_shares.get(ticker, 0)
            if q is None or q <= 0 or q >= cur:
                full_exit_tickers.add(ticker)
            self._apply_sell(ticker, sig, today_ts, ctx)

        for ticker in full_exit_tickers:
            self._holdings.pop(ticker, None)
            self._pos_shares.pop(ticker, None)

        exit_tickers = {t for t, _ in ctx.exits}

        # Preserve updated sell_streak / HWM from pipeline's SellJob
        for ticker, hs in ctx.holdings.items():
            if ticker not in exit_tickers:
                self._holdings[ticker] = hs

        # ── Buys ────────────────────────────────────────────────────────────
        for order in ctx.orders:
            self._apply_buy(order, today_ts, ctx)

        # Collect trade events emitted this bar for the persistence trace
        trade_events_this_bar = self._trade_log[len_trade_log_before:]

        # ── Persist cross-bar state from pipeline ───────────────────────────
        self._regime_state  = ctx.regime_state
        self._regime_counts = ctx.regime_counts
        self._hwm           = ctx.hwm
        self._skip_buys     = ctx.skip_buys
        self._monitor_state = dict(getattr(ctx, "monitor_state", {}) or {})

        # Rotation V1 persistence gate (2026-04-24): push this bar's
        # proposed (sell, buy) pair set. Cap the history at 2× the
        # persistence_bars setting (or ≥ 10 if disabled) so memory stays
        # bounded. Only rotations that were actually proposed (pre-gate)
        # matter — post-gate filtering happens in task_rotation, so we
        # stamp ctx.rotations which reflects final pairs.
        pairs_this_bar: set[tuple[str, str]] = {
            (p.sell_ticker, p.buy_ticker) for p in getattr(ctx, "rotations", [])
        }
        persistence_n = int(self._config.get("rotation", {}).get("persistence_bars", 0))
        window = max(persistence_n * 2, 10)
        self._rotation_proposals.append(pairs_this_bar)
        if len(self._rotation_proposals) > window:
            self._rotation_proposals = self._rotation_proposals[-window:]

        # ── Equity curve entry ──────────────────────────────────────────────
        pv = self._portfolio_value(ctx.prices)
        self._equity_curve.append({
            "date": today_ts, "portfolio": pv, "regime": ctx.regime,
        })

        # ── SQLite decision trace ───────────────────────────────────────────
        if self._db is not None:
            from kernel.persistence import (  # noqa: PLC0415
                record_pipeline_run, record_candidate_scores, record_trades,
            )
            run_id = record_pipeline_run(
                self._db,
                run_type        = "sim",
                run_date        = today_ts.date(),
                strategy        = str(self._config.get("model_name", "")),
                regime          = ctx.regime,
                confidence      = float(ctx.confidence) if ctx.confidence is not None else None,
                portfolio_value = pv,
                cash            = self._cash,
                n_candidates    = len(ctx.candidates),
                n_exits         = len(ctx.exits),
                n_rotations     = len(ctx.rotations),
                n_buys          = len(ctx.orders),
            )
            selected_tickers = {o["ticker"] for o in ctx.orders}
            blocked_map = getattr(ctx, "_blocked_by_ticker", None)
            record_candidate_scores(
                self._db, run_id, ctx.candidates, ctx.holdings,
                selected_tickers=selected_tickers,
                blocked_map=blocked_map,
            )
            record_trades(self._db, run_id, trade_events_this_bar)

    # ── Sim-side execution primitives ───────────────────────────────────────

    def _apply_sell(self, ticker: str, sig, today_ts: pd.Timestamp, ctx) -> None:
        """Apply a sell — full liquidation (default) or partial when sig.quantity set.

        When sig.quantity is None or ≥ current shares, sells everything (caller's
        commit() then pops the ticker from holdings/pos_shares). When sig.quantity
        is a positive float < current shares, sells exactly that many shares and
        reduces _pos_shares in place; the caller then skips the pop step.
        """
        from kernel.portfolio import compute_trade_tax  # noqa: PLC0415
        if ticker not in self._holdings or ticker not in self._pos_shares:
            return
        hs = self._holdings[ticker]
        total_shares = self._pos_shares[ticker]

        req_qty = getattr(sig, "quantity", None)
        if req_qty is None or req_qty <= 0 or req_qty >= total_shares:
            sell_shares = total_shares
            is_partial  = False
        else:
            sell_shares = float(req_qty)
            is_partial  = True

        price = ctx.prices.get(ticker)
        if price is None:
            df = self._ohlcv.get(ticker)
            if df is None or today_ts not in df.index:
                return
            price = float(df.loc[today_ts, "close"])

        hold_days = (today_ts.date() - hs.entry_date).days if hs.entry_date else 0
        gross_pnl = sell_shares * (price - hs.entry_price)
        tax_cfg   = ctx.config.get("tax", {})
        tax = compute_trade_tax(
            gross_pnl, hold_days,
            float(tax_cfg.get("short_term_rate", 0.37)),
            float(tax_cfg.get("long_term_rate", 0.20)),
            int(tax_cfg.get("long_term_threshold_days", 365)),
        )
        self._cash += sell_shares * price - tax
        self._last_sell_date[ticker] = today_ts

        if is_partial:
            # Keep the position open with reduced share count. entry_price and
            # entry_date stay — Kelly trims should not reset cost basis / tenure.
            self._pos_shares[ticker] = total_shares - sell_shares
            self._holdings[ticker].shares = total_shares - sell_shares

        self._trade_log.append({
            "action":      "sell",
            "ticker":      ticker,
            "date":        today_ts,
            "price":       price,
            "shares":      sell_shares,
            "pnl_pct":     (price - hs.entry_price) / hs.entry_price
                            if hs.entry_price else 0.0,
            "hold_days":   hold_days,
            "tax":         tax,
            "exit_reason": sig.exit_type,
            "partial":     is_partial,
        })

    def _apply_buy(self, order: dict, today_ts: pd.Timestamp, ctx) -> None:
        from kernel.exits import HoldingState  # noqa: PLC0415
        ticker = order["ticker"]
        shares = order["shares"]
        price  = order["price"]
        invest = shares * price
        if invest > self._cash + 1e-6:
            log.warning("SimAdapter: insufficient cash for %s (need %.2f, have %.2f)",
                        ticker, invest, self._cash)
            return
        self._cash -= invest
        # If this ticker is already held (top-up path), increment shares
        # and adjust avg entry price. Otherwise fresh position.
        if ticker in self._holdings:
            old_shares = float(self._pos_shares.get(ticker, 0))
            new_shares = old_shares + shares
            old_entry  = self._holdings[ticker].entry_price
            new_entry  = (old_entry * old_shares + price * shares) / new_shares if new_shares > 0 else price
            self._holdings[ticker].entry_price = new_entry
            self._holdings[ticker].high_watermark = max(self._holdings[ticker].high_watermark, price)
            self._holdings[ticker].shares = new_shares
            self._pos_shares[ticker] = new_shares
        else:
            self._holdings[ticker] = HoldingState(
                entry_price    = price,
                entry_date     = today_ts.date(),
                high_watermark = price,
                prev_close     = price,
                shares         = shares,
                # Thesis-degradation baseline (Approach A) — snapshot
                # today's decision signals so future rotation checks
                # can compare today's scores to THESE fixed anchors.
                entry_rank_score       = order.get("rank_score"),
                entry_panel_score      = order.get("panel_score"),
                entry_kelly_target_pct = order.get("kelly_target_pct"),
            )
            self._pos_shares[ticker] = shares
        self._trade_log.append({
            "action":    "buy",
            "ticker":    ticker,
            "date":      today_ts,
            "price":     price,
            "shares":    shares,
            "invest":    invest,
            "regime":    order.get("regime"),
            "rank_score": order.get("rank_score"),
            "rs_score":  order.get("rs_score"),
            "sigma":     order.get("sigma"),
            "mu":        order.get("mu"),
            "sigma_mult": order.get("sigma_mult"),
        })

    def _portfolio_value(self, prices: dict[str, float]) -> float:
        total = self._cash
        for t, shares in self._pos_shares.items():
            p = prices.get(t)
            if p is None:
                df = self._ohlcv.get(t)
                if df is not None and not df.empty:
                    p = float(df["close"].iloc[-1])
            if p is not None:
                total += shares * p
        return total

    # ── Summary accessors ───────────────────────────────────────────────────

    def build_result(self):
        """Return a SimResult equivalent to the legacy hand-written runner."""
        from sim.runner import SimResult  # noqa: PLC0415

        equity_df = pd.DataFrame(self._equity_curve).set_index("date") if self._equity_curve \
            else pd.DataFrame(columns=["portfolio", "regime"])
        final_val = float(equity_df["portfolio"].iloc[-1]) if not equity_df.empty else self._initial_cash
        total_ret = final_val / self._initial_cash - 1.0
        n_years = len(equity_df) / 252 if not equity_df.empty else 0
        apy = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0.0

        sells = [t for t in self._trade_log if t["action"] == "sell"]
        wins  = [t for t in sells if t.get("pnl_pct", 0.0) > 0]
        win_rate  = len(wins) / max(1, len(sells))
        avg_hold  = sum(t.get("hold_days", 0) for t in sells) / len(sells) if sells else 0.0
        avg_pnl   = sum(t.get("pnl_pct",   0) for t in sells) / len(sells) if sells else 0.0
        total_tax = sum(t.get("tax",       0) for t in sells)
        exit_reasons = dict(Counter(t.get("exit_reason", "?") for t in sells))

        # Rotation sell/buy pairs (same-day sell with exit_reason=rotation + same-day rotation buy)
        rotations: list[dict] = []
        for s in sells:
            if s.get("exit_reason") != "rotation":
                continue
            sd = s["date"].date() if hasattr(s["date"], "date") else s["date"]
            same_day_buys = [
                b for b in self._trade_log
                if b["action"] == "buy"
                and (b["date"].date() if hasattr(b["date"], "date") else b["date"]) == sd
            ]
            rotations.append({
                "date": sd, "sell": s["ticker"],
                "buy": same_day_buys[0]["ticker"] if same_day_buys else "?",
                "pnl_pct": s.get("pnl_pct", 0.0),
                "hold_days": s.get("hold_days", 0),
                "tax": s.get("tax", 0.0),
            })

        # Activity-monitoring stats: longest run of consecutive trading days
        # without any order (buy or sell). Computed post-hoc from the equity
        # curve + trade log so it always reflects the whole OOS window.
        trade_dates = {
            (t["date"].date() if hasattr(t["date"], "date") else t["date"])
            for t in self._trade_log
        }
        eq_dates = [
            (d.date() if hasattr(d, "date") else d) for d in equity_df.index
        ] if not equity_df.empty else []
        longest_streak = 0
        current_streak = 0
        first_trade: "str | None" = None
        last_activity: "str | None" = None
        for d in eq_dates:
            if d in trade_dates:
                current_streak = 0
                last_activity = str(d)
                if first_trade is None:
                    first_trade = str(d)
            else:
                current_streak += 1
                longest_streak = max(longest_streak, current_streak)

        return SimResult(
            equity_df     = equity_df,
            trade_log     = self._trade_log,
            rotation_log  = self._rotation_log,
            final_value   = final_val,
            total_return  = total_ret,
            apy           = apy,
            win_rate      = win_rate,
            avg_hold      = avg_hold,
            avg_pnl       = avg_pnl,
            total_tax     = total_tax,
            exit_reasons  = exit_reasons,
            rotations     = rotations,
            longest_no_trade_streak     = longest_streak,
            longest_no_candidate_streak = int(self._monitor_state.get("no_candidate_streak", 0)),
            first_trade_date            = first_trade,
            last_activity_date          = last_activity,
        )
