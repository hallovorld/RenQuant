"""Adaptive Regime Multi-Stock Strategy — LEAN entry point.

Thin wrapper: all decision logic lives in kernel/.
LEAN-safe: no common/ imports.  Docker can access kernel/ as a local package.
"""
from AlgorithmImports import *  # noqa: F401,F403
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_config, split_date_parts
from kernel.config       import BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES, artifact_path
from kernel.regime       import RegimeState, detect_regime, load_gmm_artifact
from kernel.models       import load_artifact, score_artifact
from kernel.exits        import HoldingState, compute_exits
from kernel.sizing       import compute_position_size
from kernel.market_gates import check_spy_velocity_crash, check_spy_ema_trend
from kernel.portfolio    import update_drawdown_circuit_breaker, compute_trade_tax
from kernel.selection    import (
    CandidateResult, SelectionContext,
    score_candidates, run_selection_loop,
    is_wash_sale_blocked, is_earnings_blocked,
)

CONFIG = load_config()


class AdaptiveRegimeMultiStockStrategy(QCAlgorithm):

    # ── Initialization ─────────────────────────────────────────────────────────

    def Initialize(self):
        sy, sm, sd = split_date_parts(CONFIG["backtest_start"])
        ey, em, ed = split_date_parts(CONFIG["backtest_end"])
        self.SetStartDate(sy, sm, sd)
        self.SetEndDate(ey, em, ed)
        self.SetCash(CONFIG["initial_cash"])

        self._strategy_dir = Path(__file__).resolve().parent
        self._watchlist    = CONFIG["watchlist"]
        self._benchmark    = CONFIG.get("benchmark", "SPY")

        # ── Symbols ──
        self.symbols: dict[str, Symbol] = {}
        for ticker in self._watchlist:
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

        # Sector ETFs for relative-strength scoring
        self._sector_etf_map     = CONFIG.get("sector_etf_map", {})
        self._sector_etf_symbols: dict[str, Symbol] = {}
        for etf in set(self._sector_etf_map.values()):
            if etf not in self.symbols:
                self._sector_etf_symbols[etf] = self.AddEquity(etf, Resolution.Daily).Symbol

        self._spy_sym = self.AddEquity(self._benchmark, Resolution.Daily).Symbol

        # ── Config ──
        self._max_positions = int(CONFIG.get("max_concurrent_positions", 8))
        self._wash_sale_days = int(CONFIG.get("wash_sale_days", 0))
        self._min_hold_days  = int(CONFIG.get("min_hold_days", 0))
        self._consec_sells   = int(CONFIG.get("consecutive_sell_signals", 3))
        self._sharpe_floor   = float(CONFIG.get("sharpe_floor", 0.8))
        self._staleness_days = int(CONFIG.get("model_staleness_days", 60))
        self._corr_threshold = float(CONFIG.get("regime", {}).get("correlation_guard_threshold", 0.70))
        self._earnings_buf   = int(CONFIG.get("regime", {}).get("earnings_buffer_days", 3))
        self._vol_window     = int(CONFIG.get("regime", {}).get("vol_realized_window", 20))
        self._defensive      = set(CONFIG.get("defensive_tickers", []))
        self._sector_map     = CONFIG.get("sector_map", {})
        self._max_per_sector = int(CONFIG.get("max_positions_per_sector", 0))
        self._tiered_thresholds = CONFIG.get("tiered_thresholds", [])
        self._regime_params  = CONFIG.get("regime_params", {})

        ranking_cfg = CONFIG.get("ranking", {})
        bw = ranking_cfg.get("blend_weights", [0.5, 0.5])
        bt = float(bw[0]) + float(bw[1])
        self._w_rank = float(bw[0]) / bt if bt > 0 else 0.5
        self._w_rs   = float(bw[1]) / bt if bt > 0 else 0.5

        tax_cfg = CONFIG.get("tax", {})
        self._tax_short      = float(tax_cfg.get("short_term_rate", 0.50))
        self._tax_long       = float(tax_cfg.get("long_term_rate", 0.32))
        self._tax_thresh_days = int(tax_cfg.get("long_term_threshold_days", 365))

        # Volume filter
        vf = CONFIG.get("volume_filter", {})
        self._vol_mode       = vf.get("mode", "percentile")
        self._vol_pct_thresh = float(vf.get("percentile_threshold", 85))
        self._vol_lookback   = int(CONFIG.get("volume_zscore_lookback", 20))

        # ── Per-position state ──
        self._holdings: dict[str, HoldingState] = {}   # ticker → HoldingState
        self._last_sell_dates: dict[str, datetime.date | None] = {}

        # ── Regime state ──
        self._regime_state    = RegimeState()
        self._spy_returns: list[float] = []
        self._regime_cfg = CONFIG.get("regime", {})

        # ── Artifacts ──
        self._models: dict[str, dict] = {}
        self._load_all_models()

        self._gmm = load_gmm_artifact(artifact_path(
            self._regime_cfg.get("gmm_artifact", "spy-gmm-regime.json")))
        self._corr = self._load_json_artifact(
            self._regime_cfg.get("correlation_artifact", "watchlist-correlation.json"),
            "Correlation")
        self._earnings = self._load_json_artifact(
            self._regime_cfg.get("earnings_artifact", "earnings-calendar.json"),
            "Earnings")

        # ── Telemetry ──
        self._total_tax        = 0.0
        self._st_trades        = 0
        self._lt_trades        = 0
        self._executed_buys    = 0
        self._executed_sells   = 0
        self._stop_exits       = 0
        self._trail_exits      = 0
        self._sdl_exits        = 0
        self._blocked_wash     = 0
        self._blocked_min_hold = 0
        self._blocked_streak   = 0
        self._sector_blocks    = 0
        self._corr_blocks      = 0
        self._earnings_blocks  = 0
        self._transition_blocks = 0
        self._velocity_blocks  = 0
        self._regime_counts    = {r: 0 for r in REGIMES}
        self._hwm              = float(CONFIG["initial_cash"])
        self._skip_buys        = False
        self._prev_closes: dict[str, float] = {}

        self._setup_charts()
        self.SetWarmUp(90)

    # ── Main event loop ────────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # Update SPY return buffer
        if data.ContainsKey(self._spy_sym):
            spy_close = float(data[self._spy_sym].Close)
            prev = self._prev_closes.get("SPY")
            if prev and prev > 0:
                self._spy_returns.append((spy_close - prev) / prev)
            if len(self._spy_returns) > 100:
                self._spy_returns = self._spy_returns[-100:]

        # Detect regime
        spy_df = self._get_spy_df(60)
        self._regime_state = detect_regime(
            np.array(self._spy_returns),
            spy_df,
            self._gmm,
            self._regime_state,
            CONFIG,
        )
        regime    = self._regime_state.regime
        conf      = self._regime_state.confidence
        regime_p  = self._regime_params.get(regime, {})
        self._regime_counts[regime] = self._regime_counts.get(regime, 0) + 1

        self.Debug(
            f"{self.Time.date()} REGIME={regime} conf={conf:.2f} "
            f"held={[t for t in self._holdings]}"
        )
        self._plot_regime(conf)

        # Portfolio drawdown circuit breaker
        pv = self.Portfolio.TotalPortfolioValue
        self._hwm, self._skip_buys = update_drawdown_circuit_breaker(
            pv, self._hwm, float(regime_p.get("drawdown_halt_pct", 0))
        )

        # ── SELL loop ──
        exit_params = self._build_exit_params(regime_p)
        for ticker in list(self._holdings.keys()):
            if not data.ContainsKey(self.symbols[ticker]):
                continue

            current_price = float(data[self.symbols[ticker]].Close)
            hs = self._holdings[ticker]
            hs.prev_close = self._prev_closes.get(ticker)

            # Get model action for sell-streak check
            features = self._build_feature_frame(ticker, spy_df)
            if features is not None:
                holdings_qty = int(self.Portfolio[self.symbols[ticker]].Quantity)
                sr = score_artifact(self._models[ticker], features.iloc[-1], holdings_qty)
                model_action = sr.signal
            else:
                model_action = "hold"

            sig, hs = compute_exits(current_price, self.Time.date(), model_action, hs, exit_params)
            self._holdings[ticker] = hs   # persist updated state (streak, HWM)

            if not sig.should_exit:
                if model_action != "sell":
                    pass
                elif hs.sell_streak > 0:
                    self._blocked_streak += 1
                    self.Debug(f"{self.Time.date()} {ticker} sell streak {hs.sell_streak}/{self._consec_sells} — waiting")
                continue

            # Count exit types
            if sig.exit_type == "trailing_stop":
                self._trail_exits += 1
            elif sig.exit_type == "stop_loss":
                self._stop_exits += 1
            elif sig.exit_type == "single_day_loss":
                self._sdl_exits += 1
            elif sig.exit_type == "model_sell":
                pass   # counted in executed_sells below

            self._execute_sell(ticker, sig.reason)

        # Update prev closes (after sell, before buy)
        for t in self._models:
            if data.ContainsKey(self.symbols[t]):
                self._prev_closes[t] = float(data[self.symbols[t]].Close)
        if data.ContainsKey(self._spy_sym):
            self._prev_closes["SPY"] = float(data[self._spy_sym].Close)

        # ── BUY phase ──
        held = list(self._holdings.keys())
        open_slots = self._max_positions - len(held)
        if open_slots <= 0 or self._skip_buys:
            self.Plot("Portfolio", "Positions", len(held))
            return

        # Transition uncertainty window
        if self._regime_state.in_transition:
            self._transition_blocks += 1
            self.Plot("Portfolio", "Positions", len(held))
            return

        # BEAR regime — defensives only
        if regime == BEAR:
            self._run_bear_defensives(held, conf)
            self.Plot("Portfolio", "Positions", len(self._holdings))
            return

        # SPY velocity crash filter
        v_halt = float(regime_p.get("spy_velocity_halt_pct", 0.0))
        v_look = int(regime_p.get("spy_velocity_lookback_days", 3))
        if check_spy_velocity_crash(self._spy_returns, v_look, v_halt):
            self._velocity_blocks += 1
            self.Debug(f"{self.Time.date()} SPY velocity crash — blocking buys")
            self.Plot("Portfolio", "Positions", len(held))
            return

        # SPY EMA50 trend gate
        spy_hist = self._get_spy_df(55)
        if spy_hist is not None and check_spy_ema_trend(spy_hist["close"]):
            self.Debug(f"{self.Time.date()} SPY EMA50 gate — blocking buys")
            self.Plot("Portfolio", "Positions", len(held))
            return

        # ── Scan candidates ──
        min_score = float(regime_p.get("min_model_score", 0.10))
        candidates: list[CandidateResult] = []
        today = self.Time.date()

        for ticker in self._models:
            if ticker in held:
                continue
            if not data.ContainsKey(self.symbols[ticker]):
                continue
            if is_earnings_blocked(ticker, today, self._earnings or {}, self._earnings_buf):
                self._earnings_blocks += 1
                continue

            features = self._build_feature_frame(ticker, spy_df)
            if features is None:
                continue

            holdings_qty = 0
            sr = score_artifact(self._models[ticker], features.iloc[-1], holdings_qty)
            self.Debug(
                f"{self.Time.date()} {ticker} action={sr.signal} "
                f"raw={sr.raw_score:.4f} rank={sr.rank_score:.4f}"
            )
            if sr.signal != "buy":
                continue
            if sr.rank_score < min_score:
                continue

            rs_score = self._compute_rs_score(ticker)
            candidates.append(CandidateResult(
                ticker=ticker, raw_score=sr.raw_score,
                rank_score=sr.rank_score, rs_score=rs_score,
                detail=f"raw={sr.raw_score:.3f} rank={sr.rank_score:.3f}",
            ))

        if not candidates:
            self.Plot("Portfolio", "Positions", len(held))
            return

        ranked = score_candidates(candidates, self._w_rank, self._w_rs)

        ctx = SelectionContext(
            today=today,
            held_tickers=held,
            last_sell_dates=self._last_sell_dates,
            earnings_calendar=self._earnings or {},
            corr_matrix=self._corr,
            sector_map=self._sector_map,
            defensive_set=self._defensive,
            wash_sale_days=self._wash_sale_days,
            earnings_buffer=self._earnings_buf,
            corr_threshold=self._corr_threshold,
            max_per_sector=self._max_per_sector,
            tiered_thresholds=self._tiered_thresholds,
            open_slots=open_slots,
        )
        selected, blocks = run_selection_loop(ranked, ctx)

        self._blocked_wash   += blocks["wash_sale"]
        self._sector_blocks  += blocks["sector"]
        self._corr_blocks    += blocks["correlation"]

        for ticker in selected:
            c = next(c for c in ranked if c.ticker == ticker)
            self._execute_buy(ticker, c.rank_score, c.rs_score, c.detail,
                              regime, conf, regime_p)

        self.Plot("Portfolio", "Positions", len(self._holdings))

    # ── BEAR defensive branch ──────────────────────────────────────────────────

    def _run_bear_defensives(self, held: list[str], conf: float) -> None:
        """In BEAR: allow 1 defensive position; pick best by model score."""
        def_held = [t for t in held if t in self._defensive]
        if def_held or self._skip_buys:
            return

        today = self.Time.date()
        bear_candidates: list[CandidateResult] = []
        for ticker in self._defensive:
            if ticker not in self._models or ticker in held:
                continue
            sym = self.symbols.get(ticker)
            if sym is None:
                continue
            if not self.Securities.ContainsKey(sym):
                continue
            if is_wash_sale_blocked(ticker, today, self._last_sell_dates, self._wash_sale_days):
                continue
            features = self._build_feature_frame(ticker, self._get_spy_df(60))
            if features is None:
                continue
            sr = score_artifact(self._models[ticker], features.iloc[-1], 0)
            if sr.signal != "buy":
                continue
            rs = self._compute_rs_score(ticker)
            bear_candidates.append(CandidateResult(
                ticker=ticker, raw_score=sr.raw_score,
                rank_score=sr.rank_score, rs_score=rs,
            ))

        if bear_candidates:
            bear_candidates.sort(key=lambda c: c.rank_score, reverse=True)
            best = bear_candidates[0]
            regime_p = self._regime_params.get(BEAR, {})
            self._execute_buy(best.ticker, best.rank_score, best.rs_score,
                              "bear_defensive", BEAR, conf, regime_p,
                              override_pct=0.15)

    # ── Feature frame ──────────────────────────────────────────────────────────

    def _build_feature_frame(self, ticker: str, spy_df: pd.DataFrame | None) -> pd.DataFrame | None:
        from kernel.indicators import build_feature_frame
        stock_hist = self.History(self.symbols[ticker], 60, Resolution.Daily)
        if stock_hist.empty or len(stock_hist) < 40:
            return None
        stock_rows = stock_hist.loc[self.symbols[ticker]].copy()
        if spy_df is None or len(spy_df) < 40:
            return None
        spec = CONFIG.get("indicator_spec", {})
        return build_feature_frame(stock_rows, spy_df, spec, self._vol_window)

    def _get_spy_df(self, bars: int) -> pd.DataFrame | None:
        h = self.History(self._spy_sym, bars, Resolution.Daily)
        if h.empty:
            return None
        return h.loc[self._spy_sym].copy()

    # ── Relative-strength score ────────────────────────────────────────────────

    def _compute_rs_score(self, ticker: str) -> float:
        """20-day return of stock minus its sector ETF."""
        from kernel.selection import compute_relative_strength
        sector  = self._sector_map.get(ticker, "other")
        etf     = self._sector_etf_map.get(sector)
        if not etf:
            return 0.0
        etf_sym = self._sector_etf_symbols.get(etf) or self.symbols.get(etf)
        if etf_sym is None:
            return 0.0
        sh = self.History(self.symbols[ticker], 21, Resolution.Daily)
        eh = self.History(etf_sym, 21, Resolution.Daily)
        if sh.empty or eh.empty or len(sh) < 21 or len(eh) < 21:
            return 0.0
        sr = sh.loc[self.symbols[ticker]]["close"]
        er = eh.loc[etf_sym]["close"]
        return compute_relative_strength(
            float(sr.iloc[-1] / sr.iloc[0] - 1),
            float(er.iloc[-1] / er.iloc[0] - 1),
        )

    # ── Trade execution ────────────────────────────────────────────────────────

    def _execute_buy(
        self,
        ticker: str,
        rank_score: float,
        rs_score: float,
        detail: str,
        regime: str,
        conf: float,
        regime_p: dict,
        override_pct: float | None = None,
    ) -> None:
        price = float(self.Securities[self.symbols[ticker]].Price)
        pv    = self.Portfolio.TotalPortfolioValue
        cash  = self.Portfolio.Cash

        max_pct    = float(regime_p.get("max_position_pct", 0.30)) * conf
        reserve_pct = float(regime_p.get("cash_reserve_pct", 0.0)) * conf

        _, shares = compute_position_size(pv, cash, max_pct, reserve_pct, price,
                                          override_pct=override_pct)
        if shares < 1:
            self.Debug(f"{self.Time.date()} {ticker} buy skipped — insufficient cash")
            return

        target_pct = (shares * price) / pv
        self.Debug(
            f"{self.Time.date()} {ticker} BUY regime={regime} conf={conf:.2f} "
            f"rank={rank_score:.3f} rs={rs_score:.3f} pct={target_pct:.2%} {detail}"
        )

        self._holdings[ticker] = HoldingState(
            entry_price=price,
            entry_date=self.Time.date(),
            high_watermark=price,
        )
        self._executed_buys += 1
        self.SetHoldings(self.symbols[ticker], target_pct)

    def _execute_sell(self, ticker: str, detail: str) -> None:
        hs         = self._holdings.pop(ticker, None)
        gross_pnl  = self.Portfolio[self.symbols[ticker]].UnrealizedProfit
        days_held  = (self.Time.date() - hs.entry_date).days if hs else 0
        is_lt      = days_held >= self._tax_thresh_days
        tax        = compute_trade_tax(gross_pnl, days_held,
                                       self._tax_short, self._tax_long, self._tax_thresh_days)
        self._total_tax += tax
        if is_lt:
            self._lt_trades += 1
        else:
            self._st_trades += 1
        self.Debug(
            f"{self.Time.date()} {ticker} SELL pnl=${gross_pnl:.2f} held={days_held}d "
            f"tax=${tax:.2f} ({'LT' if is_lt else 'ST'}) {detail}"
        )
        self._last_sell_dates[ticker] = self.Time.date()
        self._executed_sells += 1
        self.Liquidate(self.symbols[ticker])

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_exit_params(self, regime_p: dict) -> dict:
        """Build param dict consumed by kernel.exits.compute_exits."""
        return {
            "trailing_stop_trigger_pct": regime_p.get("trailing_stop_trigger_pct", 0),
            "trailing_stop_trail_pct":   regime_p.get("trailing_stop_trail_pct",   0),
            "stop_loss_pct":             regime_p.get("stop_loss_pct",             0),
            "max_single_day_loss_pct":   regime_p.get("max_single_day_loss_pct",   0),
            "max_hold_days":             regime_p.get("max_hold_days",             0),
            "consecutive_sell_signals":  self._consec_sells,
            "min_hold_days":             self._min_hold_days,
            "lt_hold_gate_days":         int(CONFIG.get("lt_hold_gate_days", 0)),
            "lt_hold_min_gain":          float(CONFIG.get("lt_hold_min_gain", 0.10)),
        }

    def _load_all_models(self) -> None:
        models_dir = self._strategy_dir / "models"
        if not models_dir.exists():
            self.Log("WARNING: models/ not found. Run the notebook to train models.")
            return
        for ticker in self._watchlist:
            artifact = load_artifact(models_dir / ticker, ticker)
            if artifact is None:
                self.Log(f"WARNING: no artifact for {ticker}, skipping")
                continue
            meta = artifact.get("_metadata", {})
            # Staleness check
            trained = meta.get("trained_date")
            if trained and self._staleness_days > 0:
                age = (datetime.now().date() - datetime.strptime(trained, "%Y-%m-%d").date()).days
                if age > self._staleness_days:
                    self.Log(f"WARNING: {ticker} model {age}d old (limit={self._staleness_days}), skipping")
                    continue
            # Sharpe floor
            if self._sharpe_floor > 0 and meta.get("sharpe", 0.0) < self._sharpe_floor:
                self.Log(f"WARNING: {ticker} sharpe={meta.get('sharpe', 0):.3f} below floor, skipping")
                continue
            self._models[ticker] = artifact
        self.Log(f"Loaded {len(self._models)}/{len(self._watchlist)} models: {sorted(self._models)}")

    def _load_json_artifact(self, filename: str, label: str) -> dict | None:
        p = artifact_path(filename)
        if not p.exists():
            self.Log(f"WARNING: {label} artifact {filename} not found")
            return None
        with open(p) as f:
            return json.load(f)

    # ── Charts & end-of-algo ───────────────────────────────────────────────────

    def _setup_charts(self) -> None:
        rc = Chart("Regime")
        rc.AddSeries(Series("State",      SeriesType.Line, ""))
        rc.AddSeries(Series("Confidence", SeriesType.Line, "%"))
        rc.AddSeries(Series("Hurst",      SeriesType.Line, "×100"))
        self.AddChart(rc)
        pc = Chart("Portfolio")
        pc.AddSeries(Series("Positions",  SeriesType.Line, "count"))
        self.AddChart(pc)

    def _plot_regime(self, conf: float) -> None:
        numeric = {BULL_CALM: 3, BULL_VOLATILE: 2, CHOPPY: 1, BEAR: 0}
        hurst_val = 0.0  # proxy; regime.py internally computes but doesn't expose it here
        self.Plot("Regime", "State",      numeric.get(self._regime_state.regime, -1))
        self.Plot("Regime", "Confidence", conf * 100)

    def OnEndOfAlgorithm(self):
        total_days = sum(self._regime_counts.values()) or 1
        pcts = {r: f"{v/total_days:.1%}" for r, v in self._regime_counts.items()}
        stats = {
            "Watchlist":             str(len(self._watchlist)),
            "Active Models":         str(len(self._models)),
            "Executed Buys":         str(self._executed_buys),
            "Executed Sells":        str(self._executed_sells),
            "Stop Loss Exits":        str(self._stop_exits),
            "Single Day Loss Exits":  str(self._sdl_exits),
            "Trailing Stop Exits":    str(self._trail_exits),
            "Blocked Wash Sales":    str(self._blocked_wash),
            "Blocked Min Hold":      str(self._blocked_min_hold),
            "Blocked Sell Streak":   str(self._blocked_streak),
            "Sector Blocks":         str(self._sector_blocks),
            "Corr Guard Blocks":     str(self._corr_blocks),
            "Earnings Blocks":       str(self._earnings_blocks),
            "Transition Blocks":     str(self._transition_blocks),
            "Velocity Blocks":       str(self._velocity_blocks),
            "Total Tax":             f"${self._total_tax:,.2f}",
            "Short-Term Trades":     str(self._st_trades),
            "Long-Term Trades":      str(self._lt_trades),
            "Regime BULL_CALM":      pcts.get(BULL_CALM,     "0%"),
            "Regime BULL_VOL":       pcts.get(BULL_VOLATILE, "0%"),
            "Regime CHOPPY":         pcts.get(CHOPPY,        "0%"),
            "Regime BEAR":           pcts.get(BEAR,          "0%"),
        }
        for k, v in stats.items():
            self.SetRuntimeStatistic(k, v)
        self.Log(
            f"End | models={len(self._models)} buys={self._executed_buys} "
            f"sells={self._executed_sells} tax=${self._total_tax:,.0f} regimes={pcts}"
        )
