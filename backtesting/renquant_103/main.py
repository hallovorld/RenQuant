"""Adaptive Regime Multi-Stock Strategy — LEAN entry point.

Thin wrapper: all decision logic lives in kernel/.
LEAN-safe: no common/ imports.  Docker can access kernel/ as a local package.
"""
from AlgorithmImports import *  # noqa: F401,F403
import json
from datetime import datetime
from pathlib import Path

from config import load_config, split_date_parts
from kernel.config    import BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES, artifact_path
from kernel.regime    import RegimeState, load_gmm_artifact
from kernel.models    import load_artifact
from kernel.exits     import HoldingState
from kernel.sizing    import compute_position_size
from kernel.portfolio import compute_trade_tax

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

        # Expose config dict for LeanAdapter
        self._config = CONFIG

        self._setup_charts()
        self.SetWarmUp(90)

        # ── Adapter + pipeline (share a single InferencePipeline per bar) ──────
        from adapters.lean import LeanAdapter, InferencePipeline as _IP
        self._adapter  = LeanAdapter(self)
        self._pipeline = _IP()

    # ── Main event loop ────────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # Must update SPY buffer before make_context so RegimeJob sees latest returns
        if data.ContainsKey(self._spy_sym):
            spy_close = float(data[self._spy_sym].Close)
            prev = self._prev_closes.get("SPY")
            if prev and prev > 0:
                self._spy_returns.append((spy_close - prev) / prev)
            if len(self._spy_returns) > 100:
                self._spy_returns = self._spy_returns[-100:]

        ctx = self._adapter.make_context(data)
        self._pipeline.run(ctx)
        self._adapter.commit(ctx)

        self.Debug(
            f"{self.Time.date()} REGIME={ctx.regime} conf={ctx.regime_confidence:.2f} "
            f"held={list(self._holdings)}"
        )
        self._plot_regime(ctx.regime_confidence)
        self.Plot("Portfolio", "Positions", len(self._holdings))

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
