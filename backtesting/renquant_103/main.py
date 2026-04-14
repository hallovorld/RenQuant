from AlgorithmImports import *
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_strategy_config, split_date_parts


CONFIG = load_strategy_config()

# ── Regime constants ──────────────────────────────────────────────────────────
BULL_CALM     = "BULL_CALM"
BULL_VOLATILE = "BULL_VOLATILE"
CHOPPY        = "CHOPPY"
BEAR          = "BEAR"
REGIMES       = [BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR]


class AdaptiveRegimeMultiStockStrategy(QCAlgorithm):

    # ── Initialization ────────────────────────────────────────────────────────

    def Initialize(self):
        start_year, start_month, start_day = split_date_parts(CONFIG["backtest_start"])
        end_year,   end_month,   end_day   = split_date_parts(CONFIG["backtest_end"])

        self.SetStartDate(start_year, start_month, start_day)
        self.SetEndDate(end_year,   end_month,   end_day)
        self.SetCash(CONFIG["initial_cash"])

        self.strategy_dir = Path(__file__).resolve().parent
        self.watchlist    = CONFIG["watchlist"]
        self._benchmark   = CONFIG.get("benchmark", "SPY")

        # ── Symbols ──
        self.symbols = {}
        for ticker in self.watchlist:
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

        # Sector ETFs needed for relative-strength ranking
        self._sector_etf_map = CONFIG.get("sector_etf_map", {})
        self._sector_etf_symbols = {}
        for etf in set(self._sector_etf_map.values()):
            if etf not in self.symbols:
                self._sector_etf_symbols[etf] = self.AddEquity(etf, Resolution.Daily).Symbol

        self.spy_symbol = self.AddEquity(self._benchmark, Resolution.Daily).Symbol

        # ── Volume config ──
        self._vol_lookback   = int(CONFIG.get("volume_zscore_lookback", 20))
        vol_filter           = CONFIG.get("volume_filter", {})
        self._vol_mode       = vol_filter.get("mode", "percentile")
        self._vol_pct_thresh = float(vol_filter.get("percentile_threshold", 85))
        self.max_positions   = int(CONFIG.get("max_concurrent_positions", 5))

        # ── Trade constraints ──
        self.wash_sale_days  = int(CONFIG.get("wash_sale_days", 0))
        self.min_hold_days   = int(CONFIG.get("min_hold_days", 0))

        # ── Regime config ──
        regime_cfg = CONFIG.get("regime", {})
        self._hurst_window        = int(regime_cfg.get("hurst_window", 63))
        self._cusum_threshold     = float(regime_cfg.get("cusum_threshold", 3.0))
        self._cusum_drift         = float(regime_cfg.get("cusum_drift", 0.5))
        self._cusum_lookback      = int(regime_cfg.get("cusum_lookback", 20))
        self._transition_bars     = int(regime_cfg.get("transition_uncertainty_bars", 3))
        self._corr_threshold      = float(regime_cfg.get("correlation_guard_threshold", 0.70))
        self._earnings_buffer     = int(regime_cfg.get("earnings_buffer_days", 3))
        self._vol_realized_window = int(regime_cfg.get("vol_realized_window", 20))

        # ── Regime-adaptive params ──
        self._regime_params = CONFIG.get("regime_params", {})

        # ── Per-symbol state ──
        self.entry_times   = {}
        self.entry_prices  = {}
        self.last_sell_times = {}
        self._position_high_watermarks = {}   # ticker → high price for trailing stop
        self._sell_streak  = {}               # ticker → consecutive sell signal count
        self._consecutive_sells_required = int(CONFIG.get("consecutive_sell_signals", 3))

        # ── Regime state ──
        self._current_regime       = BULL_CALM
        self._regime_confidence    = 0.5
        self._transition_countdown = 0
        self._spy_return_buffer    = []       # rolling SPY daily returns for Hurst + CUSUM

        # ── Load artifacts ──
        staleness_days = int(CONFIG.get("model_staleness_days", 60))
        self.models    = {}
        self._load_all_models(staleness_days)

        self._gmm        = self._load_gmm_artifact(regime_cfg.get("gmm_artifact", "spy-gmm-regime.json"))
        self._corr_matrix = self._load_correlation_artifact(regime_cfg.get("correlation_artifact", "watchlist-correlation.json"))
        self._earnings   = self._load_earnings_artifact(regime_cfg.get("earnings_artifact", "earnings-calendar.json"))

        # ── Defensive tickers ──
        self._defensive = set(CONFIG.get("defensive_tickers", []))

        # ── Sector map ──
        self.sector_map             = CONFIG.get("sector_map", {})
        self.max_positions_per_sector = int(CONFIG.get("max_positions_per_sector", 0))

        # ── Tax ──
        tax_cfg               = CONFIG.get("tax", {})
        self.tax_short_rate   = float(tax_cfg.get("short_term_rate", 0.50))
        self.tax_long_rate    = float(tax_cfg.get("long_term_rate", 0.32))
        self.tax_threshold_days = int(tax_cfg.get("long_term_threshold_days", 365))
        self.total_tax        = 0.0
        self.short_term_trades = 0
        self.long_term_trades  = 0

        # ── Telemetry counters ──
        self.executed_buys        = 0
        self.executed_sells       = 0
        self.stop_loss_exits      = 0
        self.trailing_stop_exits  = 0
        self.blocked_wash_sales   = 0
        self.blocked_min_hold     = 0
        self.blocked_streak       = 0   # model said sell but streak not yet met
        self.sector_blocks        = 0
        self.corr_blocks          = 0
        self.earnings_blocks      = 0
        self.transition_blocks    = 0
        self.velocity_blocks      = 0
        self._regime_day_counts   = {r: 0 for r in REGIMES}
        self._high_water_mark     = float(CONFIG["initial_cash"])
        self._skip_buys           = False

        self._setup_charts()
        self.SetWarmUp(90)   # enough for 63-day Hurst + 26-day indicators

    # ── Main event loop ───────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        # ── Update SPY return buffer ──
        if data.ContainsKey(self.spy_symbol):
            spy_close = float(data[self.spy_symbol].Close)
            spy_prev  = self._get_prev_close(self.spy_symbol)
            if spy_prev and spy_prev > 0:
                self._spy_return_buffer.append((spy_close - spy_prev) / spy_prev)
            if len(self._spy_return_buffer) > self._hurst_window + 10:
                self._spy_return_buffer = self._spy_return_buffer[-(self._hurst_window + 10):]

        # ── Layer 1+2+3: Detect regime ──
        self._update_regime()

        # ── Portfolio drawdown circuit breaker (regime-aware threshold) ──
        portfolio_value = self.Portfolio.TotalPortfolioValue
        self._high_water_mark = max(self._high_water_mark, portfolio_value)
        halt_pct = self._rp("drawdown_halt_pct")
        if halt_pct > 0 and self._high_water_mark > 0:
            drawdown = (self._high_water_mark - portfolio_value) / self._high_water_mark
            self._skip_buys = drawdown >= halt_pct

        regime_params = self._regime_params.get(self._current_regime, {})

        # ── SELLS first ──
        held_tickers = [t for t in self.models
                        if self.Portfolio[self.symbols[t]].Quantity > 0]

        for ticker in list(held_tickers):
            if not data.ContainsKey(self.symbols[ticker]):
                continue

            current_price = float(data[self.symbols[ticker]].Close)

            # Update trailing stop high-water mark
            hwm = self._position_high_watermarks.get(ticker, current_price)
            self._position_high_watermarks[ticker] = max(hwm, current_price)

            # Trailing stop (BULL_CALM only)
            ts_trigger = self._rp("trailing_stop_trigger_pct")
            ts_trail   = self._rp("trailing_stop_trail_pct")
            if ts_trigger > 0 and ts_trail > 0 and ticker in self.entry_prices:
                gain = (current_price - self.entry_prices[ticker]) / self.entry_prices[ticker]
                if gain >= ts_trigger:
                    trail_floor = self._position_high_watermarks[ticker] * (1 - ts_trail)
                    if current_price <= trail_floor:
                        self._execute_sell(ticker, f"trailing_stop trail_floor={trail_floor:.2f}")
                        self.trailing_stop_exits += 1
                        held_tickers.remove(ticker)
                        continue

            # Fixed stop-loss
            stop_pct = self._rp("stop_loss_pct")
            if stop_pct > 0 and ticker in self.entry_prices:
                avg_price = self.entry_prices[ticker]
                if avg_price > 0:
                    loss_pct = (avg_price - current_price) / avg_price
                    if loss_pct >= stop_pct:
                        self._execute_sell(ticker, f"stop_loss loss={loss_pct:.1%}")
                        self.stop_loss_exits += 1
                        held_tickers.remove(ticker)
                        continue

            # Max hold forced exit
            max_hold = self._rp("max_hold_days")
            if max_hold > 0 and ticker in self.entry_times:
                days_held = (self.Time.date() - self.entry_times[ticker].date()).days
                if days_held >= max_hold:
                    self._execute_sell(ticker, f"max_hold days={days_held}")
                    held_tickers.remove(ticker)
                    continue

            # Model-driven sell — requires consecutive signals to eliminate noise flips
            features = self._build_feature_frame(ticker)
            if features is None:
                continue
            action, detail = self._choose_action(ticker, features)
            if action == "sell":
                self._sell_streak[ticker] = self._sell_streak.get(ticker, 0) + 1
            else:
                self._sell_streak[ticker] = 0

            if self._sell_streak.get(ticker, 0) >= self._consecutive_sells_required:
                action, _ = self._apply_sell_constraints(ticker, action)
                if action == "sell":
                    self._execute_sell(ticker, detail)
                    self._sell_streak[ticker] = 0
                    held_tickers.remove(ticker)
            elif self._sell_streak.get(ticker, 0) > 0:
                self.blocked_streak += 1

        # ── BUY phase ──
        open_slots = self.max_positions - len(held_tickers)
        if open_slots <= 0 or self._skip_buys:
            self._plot_state(held_tickers)
            return

        # No new buys in BEAR or during transition uncertainty
        if self._current_regime == BEAR or self._transition_countdown > 0:
            if self._transition_countdown > 0:
                self.transition_blocks += 1
                self._transition_countdown -= 1
            self._plot_state(held_tickers)
            return

        # SPY velocity crash filter (per-regime config)
        velocity_halt_pct  = float(regime_params.get("spy_velocity_halt_pct", 0.0))
        velocity_lookback  = int(regime_params.get("spy_velocity_lookback_days", 3))
        if velocity_halt_pct > 0 and len(self._spy_return_buffer) >= velocity_lookback:
            spy_nday = np.prod([1.0 + r for r in self._spy_return_buffer[-velocity_lookback:]]) - 1.0
            if spy_nday < -velocity_halt_pct:
                self.velocity_blocks += 1
                self._plot_state(held_tickers)
                return

        # ── SCAN: model signal is the entry trigger (no separate volume-scan gate) ──
        # Each candidate must: pass earnings filter, have a buy signal from the model,
        # and have model_score >= min_score. This matches the notebook simulation logic.
        min_score = regime_params.get("min_model_score", 0.10)
        scored = []
        for ticker in self.models:
            if ticker in held_tickers:
                continue
            if not data.ContainsKey(self.symbols[ticker]):
                continue

            # Earnings filter
            if self._is_earnings_blocked(ticker):
                self.earnings_blocks += 1
                continue

            # Build features and run model
            features = self._build_feature_frame(ticker)
            if features is None:
                continue

            action, detail = self._choose_action(ticker, features)
            if action != "buy":
                continue

            model_score = self._get_raw_model_score(ticker, features)
            if model_score < min_score:
                continue

            rs_score = self._compute_rs_score(ticker)
            scored.append((ticker, model_score, rs_score, detail))

        if not scored:
            self._plot_state(held_tickers)
            return

        # Normalize and combine
        model_scores = [s[1] for s in scored]
        rs_scores    = [s[2] for s in scored]
        model_min, model_max = min(model_scores), max(model_scores)
        rs_min,    rs_max    = min(rs_scores),    max(rs_scores)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        ranked = sorted(
            scored,
            key=lambda s: 0.5 * norm(s[1], model_min, model_max)
                        + 0.5 * norm(s[2], rs_min,    rs_max),
            reverse=True,
        )

        # ── EXECUTE: correlation-aware greedy selection ──
        for ticker, model_score, rs_score, detail in ranked:
            if open_slots <= 0:
                break

            if self._is_wash_sale_blocked(ticker):
                continue

            # Sector guard
            if self.max_positions_per_sector > 0:
                sector = self.sector_map.get(ticker, "other")
                # Defensives can stack — skip sector guard for them
                if ticker not in self._defensive:
                    sector_count = sum(
                        1 for t in held_tickers
                        if self.sector_map.get(t, "other") == sector
                    )
                    if sector_count >= self.max_positions_per_sector:
                        self.sector_blocks += 1
                        continue

            # Correlation guard
            if not self._passes_correlation_guard(ticker, held_tickers):
                self.corr_blocks += 1
                continue

            self._execute_buy(ticker, model_score, rs_score, detail)
            held_tickers.append(ticker)
            open_slots -= 1

        self._plot_state(held_tickers)

    # ── 3-Layer Regime Detection ──────────────────────────────────────────────

    def _update_regime(self):
        """Compute regime using Hurst (L1) + CUSUM (L2) + GMM (L3)."""
        returns = np.array(self._spy_return_buffer)
        if len(returns) < 30:
            return   # not enough data yet

        # Layer 1 — Hurst exponent
        hurst = self._compute_hurst(returns[-self._hurst_window:])
        if hurst > 0.55:
            hurst_regime = "MOMENTUM"
        elif hurst < 0.45:
            hurst_regime = "REVERSION"
        else:
            hurst_regime = "AMBIGUOUS"

        # Layer 2 — CUSUM changepoint
        cusum_window = returns[-self._cusum_lookback:]
        transition   = self._compute_cusum(cusum_window)
        if transition:
            self._transition_countdown = self._transition_bars
            self.Debug(f"{self.Time.date()} CUSUM changepoint detected — uncertainty window starts")

        # Layer 3 — GMM probabilities
        gmm_probs = self._gmm_predict(returns)

        # Resolve
        dominant_gmm = max(gmm_probs, key=gmm_probs.get)

        # GMM BEAR override: if GMM is strongly bearish, protect
        if gmm_probs.get(BEAR, 0) > 0.5:
            new_regime = BEAR
        elif hurst_regime == "MOMENTUM":
            new_regime = BULL_CALM
        elif hurst_regime == "REVERSION":
            new_regime = CHOPPY
        else:
            # Ambiguous: defer to GMM
            new_regime = dominant_gmm if dominant_gmm != BEAR else BULL_VOLATILE

        # During transition: downgrade confidence
        if self._transition_countdown > 0:
            self._regime_confidence = 0.5
        else:
            self._regime_confidence = gmm_probs.get(new_regime, 0.5)

        if new_regime != self._current_regime:
            self.Debug(
                f"{self.Time.date()} Regime: {self._current_regime} → {new_regime} "
                f"(H={hurst:.3f} GMM={gmm_probs} conf={self._regime_confidence:.2f})"
            )

        self._current_regime = new_regime
        self._regime_day_counts[new_regime] = self._regime_day_counts.get(new_regime, 0) + 1

        # Update regime-adaptive parameters
        self._apply_regime_params()

        # Telemetry
        regime_numeric = {BULL_CALM: 3, BULL_VOLATILE: 2, CHOPPY: 1, BEAR: 0}
        self.Plot("Regime", "State", regime_numeric.get(self._current_regime, -1))
        self.Plot("Regime", "Confidence", self._regime_confidence * 100)
        self.Plot("Regime", "Hurst", hurst * 100)

    def _apply_regime_params(self):
        """Cache current regime params for use in buy/sell logic."""
        self._active_params = self._regime_params.get(self._current_regime, {})

    def _rp(self, key, default=None):
        """Get a regime-adaptive parameter, scaled by confidence."""
        val = self._active_params.get(key, default)
        if val is None:
            return default
        # Position-sizing params scale with confidence; stop/hold don't
        if key in ("max_position_pct", "cash_reserve_pct"):
            return val * self._regime_confidence
        return val

    # ── Layer 1: Hurst Exponent ───────────────────────────────────────────────

    def _compute_hurst(self, returns: np.ndarray) -> float:
        """Rescaled range (R/S) Hurst exponent. Returns H in [0, 1]."""
        n = len(returns)
        if n < 10:
            return 0.5
        max_lag = min(n // 2, 40)
        lags = range(2, max_lag)
        rs_vals = []
        for lag in lags:
            chunks = [returns[i:i + lag] for i in range(0, n - lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean  = chunk.mean()
                devs  = np.cumsum(chunk - mean)
                R     = devs.max() - devs.min()
                S     = chunk.std(ddof=1)
                if S > 0:
                    rs_chunk.append(R / S)
            if rs_chunk:
                rs_vals.append(np.mean(rs_chunk))
        if len(rs_vals) < 2:
            return 0.5
        try:
            lags_used = list(range(2, 2 + len(rs_vals)))
            poly = np.polyfit(np.log(lags_used), np.log(rs_vals), 1)
            return float(np.clip(poly[0], 0.0, 1.0))
        except Exception:
            return 0.5

    # ── Layer 2: CUSUM Changepoint ────────────────────────────────────────────

    def _compute_cusum(self, returns: np.ndarray) -> bool:
        """Return True if a structural break is detected in the return window."""
        if len(returns) < 5:
            return False
        mu    = returns.mean()
        sigma = returns.std(ddof=1)
        if sigma <= 0:
            return False
        s_pos, s_neg = 0.0, 0.0
        for r in returns:
            z    = (r - mu) / sigma
            s_pos = max(0.0, s_pos + z - self._cusum_drift)
            s_neg = max(0.0, s_neg - z - self._cusum_drift)
            if s_pos > self._cusum_threshold or s_neg > self._cusum_threshold:
                return True
        return False

    # ── Layer 3: GMM ─────────────────────────────────────────────────────────

    def _gmm_predict(self, returns: np.ndarray) -> dict:
        """Return P(regime) dict using pre-loaded GMM parameters."""
        if self._gmm is None or len(returns) < self._vol_realized_window + 10:
            # Fallback: uniform distribution
            return {r: 1.0 / len(REGIMES) for r in REGIMES}

        # Build feature vector: [10d_return, 20d_realized_vol, spy_adx, return_autocorr]
        recent = returns[-max(self._vol_realized_window, 11):]
        r10d   = float(np.sum(recent[-10:]))
        vol20  = float(np.std(recent[-self._vol_realized_window:], ddof=1) * np.sqrt(252))

        # ADX proxy from SPY history
        spy_adx = self._get_spy_adx()

        # Return autocorrelation (lag=1) over last 20 days
        if len(recent) >= 12:
            arr  = np.array(recent[-20:]) if len(recent) >= 20 else np.array(recent)
            r_autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0.0
        else:
            r_autocorr = 0.0

        x = np.array([r10d, vol20, spy_adx, r_autocorr])

        # Gaussian log-likelihood per component
        means  = self._gmm["means"]
        covs   = self._gmm["covariances"]
        weights = self._gmm["weights"]
        labels = self._gmm["cluster_labels"]

        log_probs = []
        for k in range(len(means)):
            mu    = np.array(means[k])
            sigma = np.array(covs[k])
            diff  = x - mu
            try:
                sign, logdet = np.linalg.slogdet(sigma)
                inv_s  = np.linalg.inv(sigma)
                mahal  = float(diff @ inv_s @ diff)
                log_p  = -0.5 * (mahal + logdet) + np.log(max(weights[k], 1e-10))
            except Exception:
                log_p = np.log(max(weights[k], 1e-10))
            log_probs.append(log_p)

        log_probs = np.array(log_probs)
        log_probs -= log_probs.max()
        probs = np.exp(log_probs)
        probs /= probs.sum()

        return {label: float(p) for label, p in zip(labels, probs)}

    def _get_spy_adx(self) -> float:
        """Compute ADX(14) on SPY using recent history."""
        hist = self.History(self.spy_symbol, 30, Resolution.Daily)
        if hist.empty or len(hist) < 15:
            return 25.0  # neutral default
        rows = hist.loc[self.spy_symbol].copy()
        high, low, close = rows["high"], rows["low"], rows["close"]
        up   = high.diff()
        down = -low.diff()
        plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        tr = pd.concat([high - low, (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        period = 14
        atr     = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=rows.index).ewm(
            alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        minus_di = 100 * pd.Series(minus_dm, index=rows.index).ewm(
            alpha=1/period, min_periods=period, adjust=False).mean() / atr.replace(0, np.nan)
        dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        return float(adx.iloc[-1]) if not adx.empty else 25.0

    # ── Regime-conditional scan ───────────────────────────────────────────────

    def _regime_scan(self, ticker: str, entry_mode: str,
                     is_defensive: bool) -> tuple[bool, float]:
        """Return (triggered, score) under the current entry mode.

        Defensive tickers (GLD, TLT, XLV, XLU) use inverse logic in BEAR/BULL_VOLATILE
        — they're BUY candidates when the rest of the market is selling off.
        """
        # Counter-cyclical: defensives triggered on BEAR/BULL_VOLATILE by their own strength
        if is_defensive and self._current_regime in (BEAR, BULL_VOLATILE):
            return self._defensive_scan(ticker)

        if entry_mode == "momentum":
            return self._momentum_scan(ticker)
        elif entry_mode == "capitulation":
            return self._capitulation_scan(ticker)
        elif entry_mode == "divergence":
            return self._divergence_scan(ticker)
        else:
            return False, 0.0

    def _momentum_scan(self, ticker: str) -> tuple[bool, float]:
        """Volume spike + up-close + price above EMA(50)."""
        score, triggered = self._compute_volume_score(ticker)
        if not triggered:
            return False, 0.0
        if not self._is_price_up(ticker, min_move=0.0):
            return False, 0.0
        # Trend filter: only enter if stock is in an uptrend (close > 50-day EMA)
        hist50 = self.History(self.symbols[ticker], 51, Resolution.Daily)
        if not hist50.empty and len(hist50) >= 51:
            closes = hist50.loc[self.symbols[ticker]]["close"]
            ema50  = closes.ewm(span=50, adjust=False).mean()
            if closes.iloc[-1] < ema50.iloc[-1]:
                return False, 0.0
        return True, score

    def _capitulation_scan(self, ticker: str) -> tuple[bool, float]:
        """High volume + down-close + in bottom 30% of 5-day range (panic selling)."""
        score, triggered = self._compute_volume_score(ticker)
        if not triggered:
            return False, 0.0
        hist = self.History(self.symbols[ticker], 6, Resolution.Daily)
        if hist.empty or len(hist) < 6:
            return False, 0.0
        rows   = hist.loc[self.symbols[ticker]]
        closes = rows["close"]
        # Must be a down day
        if closes.iloc[-1] >= closes.iloc[-2]:
            return False, 0.0
        # Must be in bottom 30% of the 5-day high-low range
        recent_high = rows["high"].iloc[-5:].max()
        recent_low  = rows["low"].iloc[-5:].min()
        rng = recent_high - recent_low
        if rng <= 0:
            return False, 0.0
        position_in_range = (closes.iloc[-1] - recent_low) / rng
        if position_in_range > 0.30:
            return False, 0.0
        return True, score

    def _divergence_scan(self, ticker: str) -> tuple[bool, float]:
        """Stock outperformed SPY by >1% over last 5 days despite choppy market."""
        stock_hist = self.History(self.symbols[ticker], 6, Resolution.Daily)
        spy_hist   = self.History(self.spy_symbol, 6, Resolution.Daily)
        if stock_hist.empty or spy_hist.empty or len(stock_hist) < 6 or len(spy_hist) < 6:
            return False, 0.0
        stock_ret = (stock_hist.loc[self.symbols[ticker]]["close"].iloc[-1] /
                     stock_hist.loc[self.symbols[ticker]]["close"].iloc[0] - 1)
        spy_ret   = (spy_hist.loc[self.spy_symbol]["close"].iloc[-1] /
                     spy_hist.loc[self.spy_symbol]["close"].iloc[0] - 1)
        outperformance = stock_ret - spy_ret
        if outperformance < 0.01:
            return False, 0.0
        return True, float(outperformance)

    def _defensive_scan(self, ticker: str) -> tuple[bool, float]:
        """Defensives (GLD, TLT, XLU, XLV): triggered when they're showing relative
        strength — i.e., outperforming in a risk-off environment."""
        stock_hist = self.History(self.symbols[ticker], 11, Resolution.Daily)
        spy_hist   = self.History(self.spy_symbol, 11, Resolution.Daily)
        if stock_hist.empty or spy_hist.empty:
            return False, 0.0
        stock_ret = (stock_hist.loc[self.symbols[ticker]]["close"].iloc[-1] /
                     stock_hist.loc[self.symbols[ticker]]["close"].iloc[-10] - 1)
        spy_ret   = (spy_hist.loc[self.spy_symbol]["close"].iloc[-1] /
                     spy_hist.loc[self.spy_symbol]["close"].iloc[-10] - 1)
        outperformance = stock_ret - spy_ret
        # Defensive: triggered when SPY is weak AND defensive is holding up
        if spy_ret > -0.01 or outperformance < 0.02:
            return False, 0.0
        return True, float(outperformance)

    # ── Relative Strength Score ───────────────────────────────────────────────

    def _compute_rs_score(self, ticker: str) -> float:
        """20-day return of stock minus its sector ETF return."""
        sector = self.sector_map.get(ticker, "other")
        etf    = self._sector_etf_map.get(sector)
        if not etf:
            return 0.0

        etf_sym = self._sector_etf_symbols.get(etf) or self.symbols.get(etf)
        if etf_sym is None:
            return 0.0

        stock_hist = self.History(self.symbols[ticker], 21, Resolution.Daily)
        etf_hist   = self.History(etf_sym, 21, Resolution.Daily)
        if stock_hist.empty or etf_hist.empty or len(stock_hist) < 21 or len(etf_hist) < 21:
            return 0.0

        stock_ret = (stock_hist.loc[self.symbols[ticker]]["close"].iloc[-1] /
                     stock_hist.loc[self.symbols[ticker]]["close"].iloc[0] - 1)
        etf_ret   = (etf_hist.loc[etf_sym]["close"].iloc[-1] /
                     etf_hist.loc[etf_sym]["close"].iloc[0] - 1)
        return float(stock_ret - etf_ret)

    # ── Correlation guard ─────────────────────────────────────────────────────

    def _passes_correlation_guard(self, ticker: str, held_tickers: list) -> bool:
        """Return True if ticker is not too correlated with any held position."""
        if self._corr_matrix is None or not held_tickers:
            return True
        for held in held_tickers:
            corr = (self._corr_matrix.get(ticker, {}).get(held)
                    or self._corr_matrix.get(held, {}).get(ticker))
            if corr is not None and abs(corr) >= self._corr_threshold:
                return False
        return True

    # ── Earnings filter ───────────────────────────────────────────────────────

    def _is_earnings_blocked(self, ticker: str) -> bool:
        """Return True if ticker has earnings within ±buffer trading days."""
        if not self._earnings:
            return False
        today = self.Time.date()
        dates = self._earnings.get(ticker, [])
        for d_str in dates:
            try:
                d = datetime.strptime(d_str, "%Y-%m-%d").date()
                if abs((d - today).days) <= self._earnings_buffer:
                    return True
            except ValueError:
                continue
        return False

    # ── Volume scanner ────────────────────────────────────────────────────────

    def _compute_volume_score(self, ticker: str) -> tuple[float, bool]:
        lookback = self._vol_lookback
        history  = self.History(self.symbols[ticker], lookback + 1, Resolution.Daily)
        if history.empty or len(history) < lookback + 1:
            return 0.0, False
        volumes   = history.loc[self.symbols[ticker]]["volume"]
        today_vol = volumes.iloc[-1]
        hist_vol  = volumes.iloc[:-1]
        if self._vol_mode == "percentile":
            rank = (hist_vol < today_vol).sum() / len(hist_vol) * 100
            return rank, rank >= self._vol_pct_thresh
        else:
            mean_vol = hist_vol.mean()
            std_vol  = hist_vol.std()
            if std_vol <= 0:
                return 0.0, False
            z = (today_vol - mean_vol) / std_vol
            return z, z >= float(CONFIG.get("volume_zscore_threshold", 1.5))

    def _is_price_up(self, ticker: str, min_move: float = 0.0) -> bool:
        history = self.History(self.symbols[ticker], 2, Resolution.Daily)
        if history.empty or len(history) < 2:
            return False
        closes = history.loc[self.symbols[ticker]]["close"]
        move   = (closes.iloc[-1] - closes.iloc[-2]) / closes.iloc[-2]
        return move > min_move

    def _get_prev_close(self, symbol) -> float | None:
        hist = self.History(symbol, 2, Resolution.Daily)
        if hist.empty or len(hist) < 2:
            return None
        return float(hist.loc[symbol]["close"].iloc[-2])

    # ── Feature computation ───────────────────────────────────────────────────

    def _build_feature_frame(self, ticker: str):
        """60-day history → indicators + relative features + regime context."""
        stock_hist = self.History(self.symbols[ticker], 60, Resolution.Daily)
        spy_hist   = self.History(self.spy_symbol, 60, Resolution.Daily)
        if stock_hist.empty or spy_hist.empty:
            return None

        stock_rows = stock_hist.loc[self.symbols[ticker]].copy()
        spy_rows   = spy_hist.loc[self.spy_symbol].copy()
        if len(stock_rows) < 40 or len(spy_rows) < 40:
            return None

        stock_ind = self._compute_indicators(stock_rows)
        spy_ind   = self._compute_indicators(spy_rows)
        if stock_ind is None or spy_ind is None:
            return None

        common_idx = stock_ind.index.intersection(spy_ind.index)
        if len(common_idx) < 10:
            return None

        stock_ind = stock_ind.loc[common_idx]
        spy_ind   = spy_ind.loc[common_idx]

        ratio_features = {"rsi", "adx"}
        diff_features  = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

        result = pd.DataFrame(index=common_idx)
        result["close"] = stock_ind["close"]

        for col in ratio_features | diff_features:
            if col in ratio_features:
                result[col] = stock_ind[col] / spy_ind[col].replace(0, np.nan)
            else:
                result[col] = stock_ind[col] - spy_ind[col]

        # Trend features
        close  = stock_ind["close"]
        ema50  = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        result["trend"]      = close / ema50
        result["trend_long"] = close / ema200

        spy_close = spy_ind["close"]
        rel_price = close / spy_close
        result["rel_mom_20d"] = rel_price.pct_change(20)
        result["rel_mom_60d"] = rel_price.pct_change(60)

        # ── Regime context features (new in 103) ──
        spy_returns = spy_close.pct_change().dropna()
        if len(spy_returns) >= self._vol_realized_window:
            result["spy_realized_vol"] = float(
                spy_returns.iloc[-self._vol_realized_window:].std() * np.sqrt(252)
            )
        else:
            result["spy_realized_vol"] = 0.0

        result["spy_adx"]   = self._get_spy_adx()
        result["spy_trend"] = float(spy_close.iloc[-1] / spy_close.ewm(span=50, adjust=False).mean().iloc[-1])

        # Hurst proxy: autocorr of SPY returns (fast approximation)
        if len(spy_returns) >= 12:
            arr = spy_returns.values[-20:]
            result["hurst_proxy"] = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0.0
        else:
            result["hurst_proxy"] = 0.0

        result = result.dropna()
        if result.empty:
            return None
        return result

    def _compute_indicators(self, rows: pd.DataFrame) -> pd.DataFrame | None:
        rows  = rows.copy()
        close  = rows["close"]
        high   = rows["high"]
        low    = rows["low"]
        volume = rows["volume"]

        # RSI
        delta    = close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        avg_loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        rows["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

        # MACD histogram
        ema_fast     = close.ewm(span=12, adjust=False).mean()
        ema_slow     = close.ewm(span=26, adjust=False).mean()
        macd_line    = ema_fast - ema_slow
        rows["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

        # CCI
        tp      = (high + low + close) / 3
        sma20   = tp.rolling(20).mean()
        mad20   = tp.rolling(20).apply(lambda v: np.mean(np.abs(v - v.mean())), raw=True)
        rows["cci"] = (tp - sma20) / (0.015 * mad20.replace(0, np.nan))

        # BBP
        sma20c  = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        rows["bbp"] = (close - sma20c) / (2 * std20.replace(0, np.nan))

        # ADX
        up_move  = high.diff()
        down_move = -low.diff()
        plus_dm  = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=rows.index)
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=rows.index)
        tr = pd.concat([high - low, (high - close.shift()).abs(),
                        (low - close.shift()).abs()], axis=1).max(axis=1)
        atr14    = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
        minus_di = 100 * minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        rows["adx"] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

        # Williams %R
        hh = high.rolling(14).max()
        ll = low.rolling(14).min()
        rows["williams_r"] = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

        # OBV slope
        obv         = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_ema     = obv.ewm(span=20, adjust=False).mean()
        rows["obv_slope"] = obv_ema.diff(5) / obv_ema.shift(5).replace(0, np.nan)

        ind_cols = ["rsi", "macd_hist", "cci", "bbp", "adx", "williams_r", "obv_slope"]
        rows = rows.dropna(subset=ind_cols)
        return rows if not rows.empty else None

    # ── Model prediction ──────────────────────────────────────────────────────

    def _choose_action(self, ticker: str, features: pd.DataFrame) -> tuple:
        score = self._get_raw_model_score(ticker, features)
        model = self.models[ticker]
        if score > model["buy_threshold"]:
            return "buy",  f"score={score:.3f}"
        if score < model["sell_threshold"]:
            return "sell", f"score={score:.3f}"
        return "hold", f"score={score:.3f}"

    def _get_raw_model_score(self, ticker: str, features: pd.DataFrame) -> float:
        """Return continuous model score (classification average or Q-value delta)."""
        model  = self.models[ticker]
        ptype  = model["policy_type"]
        row    = features.iloc[-1]

        if ptype == "classification":
            feat_cols = model["feature_columns"]
            feat_vals = [row.get(c, np.nan) for c in feat_cols]
            if any(np.isnan(v) for v in feat_vals):
                return 0.0
            return self._bag_predict(model["trees"], feat_vals)

        if ptype == "manual":
            return float(self._score_manual_rules(row, model["score_rules"]))

        if ptype == "qlearning":
            holdings = self.Portfolio[self.symbols[ticker]].Quantity
            feat_cols = model["feature_columns"]
            state = self._encode_q_state(
                row, holdings, feat_cols, model["bin_edges"], model["n_bins"])
            q_vals = model["q_table"][state]
            # Score = Q(buy) - Q(hold): positive means buying is better
            return float(q_vals[0] - q_vals[2])

        if ptype == "xgboost":
            feat_cols = model["feature_columns"]
            feat_vals = [float(row.get(c, np.nan)) for c in feat_cols]
            if any(np.isnan(v) for v in feat_vals):
                return 0.0
            p_buy  = self._xgb_predict(model["xgb_buy"],  feat_vals)
            p_sell = self._xgb_predict(model["xgb_sell"], feat_vals)
            return float(p_buy - p_sell)

        return 0.0

    def _traverse_tree(self, tree: list, row: list) -> float:
        idx = 0
        while True:
            feat, split_val, left_off, right_off = tree[idx]
            if feat == -1:
                return split_val
            idx += int(left_off) if row[int(feat)] <= split_val else int(right_off)

    def _bag_predict(self, trees: list, features: list) -> float:
        return sum(self._traverse_tree(t, features) for t in trees) / len(trees)

    def _xgb_predict(self, xgb_json: dict, feat_vals: list) -> float:
        """Pure-Python XGBoost inference from JSON artifact. Returns P ∈ [0, 1].

        Traverses all trees in the gradient boosted ensemble, sums leaf weights,
        then applies sigmoid (binary:logistic objective).
        """
        trees = xgb_json["learner"]["gradient_booster"]["model"]["trees"]
        total = 0.0
        for tree in trees:
            lc = tree["left_children"]
            rc = tree["right_children"]
            sc = tree["split_conditions"]
            si = tree["split_indices"]
            bw = tree["base_weights"]
            node = 0
            while lc[node] != -1:
                fi  = si[node]
                val = feat_vals[fi] if fi < len(feat_vals) else 0.0
                node = lc[node] if val <= sc[node] else rc[node]
            total += bw[node]
        return 1.0 / (1.0 + math.exp(-total))

    def _score_manual_rules(self, row: pd.Series, rules: list) -> int:
        score = 0
        for rule in rules:
            val = row.get(rule["col"])
            if val is None or (isinstance(val, float) and np.isnan(val)):
                continue
            if "buy_below"  in rule and rule["buy_below"]  is not None and val < rule["buy_below"]:
                score += 1
            if "buy_above"  in rule and rule["buy_above"]  is not None and val > rule["buy_above"]:
                score += 1
            if "sell_above" in rule and rule["sell_above"] is not None and val > rule["sell_above"]:
                score -= 1
            if "sell_below" in rule and rule["sell_below"] is not None and val < rule["sell_below"]:
                score -= 1
        return score

    def _encode_q_state(self, row, holdings, feature_columns, bin_edges, n_bins):
        state = 0
        for col in feature_columns:
            val     = row.get(col, 0)
            bin_idx = np.digitize(val, bin_edges[col]) - 1
            bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
            state   = state * n_bins + bin_idx
        holding_bucket = 2 if holdings > 0 else (0 if holdings < 0 else 1)
        return state * 3 + holding_bucket

    # ── Trade execution ───────────────────────────────────────────────────────

    def _execute_buy(self, ticker: str, model_score: float,
                     rs_score: float, detail: str) -> None:
        portfolio_value = self.Portfolio.TotalPortfolioValue
        available_cash  = self.Portfolio.Cash
        cash_reserve    = portfolio_value * self._rp("cash_reserve_pct", 0.0)
        investable      = max(available_cash - cash_reserve, 0)
        max_pct         = self._rp("max_position_pct", 0.30)
        target_pct      = min(max_pct, investable / max(portfolio_value, 1))

        if target_pct < 0.01:
            self.Debug(f"{self.Time.date()} {ticker} buy skipped — insufficient cash")
            return

        self.Debug(
            f"{self.Time.date()} {ticker} BUY regime={self._current_regime} "
            f"conf={self._regime_confidence:.2f} model={model_score:.3f} "
            f"rs={rs_score:.3f} pct={target_pct:.2%} {detail}"
        )
        price = float(self.Securities[self.symbols[ticker]].Price)
        self.entry_times[ticker]                 = self.Time
        self.entry_prices[ticker]                = price
        self._position_high_watermarks[ticker]   = price
        self._sell_streak[ticker]                = 0
        self.executed_buys += 1
        self.SetHoldings(self.symbols[ticker], target_pct)

    def _execute_sell(self, ticker: str, detail: str) -> None:
        gross_pnl  = self.Portfolio[self.symbols[ticker]].UnrealizedProfit
        entry_time = self.entry_times.get(ticker)
        days_held  = (self.Time.date() - entry_time.date()).days if entry_time else 0
        is_lt      = days_held >= self.tax_threshold_days
        tax_rate   = self.tax_long_rate if is_lt else self.tax_short_rate
        tax        = max(gross_pnl, 0) * tax_rate
        self.total_tax += tax
        if is_lt:
            self.long_term_trades += 1
        else:
            self.short_term_trades += 1
        self.Debug(
            f"{self.Time.date()} {ticker} SELL pnl=${gross_pnl:.2f} held={days_held}d "
            f"tax=${tax:.2f} ({'LT' if is_lt else 'ST'}) {detail}"
        )
        self.last_sell_times[ticker] = self.Time
        self.entry_times.pop(ticker, None)
        self.entry_prices.pop(ticker, None)
        self._position_high_watermarks.pop(ticker, None)
        self._sell_streak.pop(ticker, None)
        self.executed_sells += 1
        self.Liquidate(self.symbols[ticker])

    # ── Trade constraints ─────────────────────────────────────────────────────

    def _is_wash_sale_blocked(self, ticker: str) -> bool:
        if self.wash_sale_days <= 0:
            return False
        last_sell = self.last_sell_times.get(ticker)
        if last_sell is None:
            return False
        days = (self.Time.date() - last_sell.date()).days
        if days < self.wash_sale_days:
            self.blocked_wash_sales += 1
            return True
        return False

    def _apply_sell_constraints(self, ticker: str, action: str) -> tuple:
        if action != "sell":
            return action, ""
        if self.min_hold_days > 0 and ticker in self.entry_times:
            days = (self.Time.date() - self.entry_times[ticker].date()).days
            if days < self.min_hold_days:
                self.blocked_min_hold += 1
                return "hold", f"min_hold days={days}"
        return action, ""

    # ── Artifact loading ──────────────────────────────────────────────────────

    def _load_all_models(self, staleness_days: int) -> None:
        models_dir = self.strategy_dir / "models"
        if not models_dir.exists():
            self.Log("WARNING: models/ not found. Run the 103 notebook to train models.")
            return
        for ticker in self.watchlist:
            symbol_dir = models_dir / ticker
            meta_path  = symbol_dir / f"{ticker}-policy-metadata.json"
            if not meta_path.exists():
                self.Log(f"WARNING: No model for {ticker}, skipping")
                continue
            with meta_path.open() as f:
                metadata = json.load(f)
            trained_date = metadata.get("trained_date")
            if trained_date and staleness_days > 0:
                age = (datetime.now().date() -
                       datetime.strptime(trained_date, "%Y-%m-%d").date()).days
                if age > staleness_days:
                    self.Log(f"WARNING: {ticker} model is {age}d old (limit={staleness_days}), skipping")
                    continue
            model_data = self._load_model_artifacts(ticker, metadata, symbol_dir)
            if model_data is not None:
                self.models[ticker] = model_data
        self.Log(f"Loaded {len(self.models)}/{len(self.watchlist)} models: {sorted(self.models.keys())}")

    def _load_model_artifacts(self, ticker, metadata, symbol_dir) -> dict | None:
        ptype         = metadata["policy_type"]
        feature_cols  = metadata.get("feature_columns", [])
        model_data    = {
            "policy_type":     ptype,
            "feature_columns": feature_cols,
            "buy_threshold":   metadata.get("buy_threshold", 0.1),
            "sell_threshold":  metadata.get("sell_threshold", -0.1),
        }
        if ptype == "classification":
            p = symbol_dir / f"{ticker}-rf-trees.json"
            if not p.exists():
                self.Log(f"WARNING: {ticker} trees missing")
                return None
            with p.open() as f:
                model_data["trees"] = json.load(f)
            return model_data
        if ptype == "manual":
            p = symbol_dir / f"{ticker}-manual-rules.json"
            if not p.exists():
                self.Log(f"WARNING: {ticker} manual rules missing")
                return None
            with p.open() as f:
                d = json.load(f)
            model_data["score_rules"]    = d["score_rules"]
            model_data["buy_threshold"]  = d["buy_threshold"]
            model_data["sell_threshold"] = d["sell_threshold"]
            return model_data
        if ptype == "qlearning":
            qp = symbol_dir / f"{ticker}-qtable.json"
            ep = symbol_dir / f"{ticker}-bin-edges.json"
            if not qp.exists() or not ep.exists():
                self.Log(f"WARNING: {ticker} Q-learning artifacts missing")
                return None
            with qp.open() as f:
                model_data["q_table"] = np.array(json.load(f))
            with ep.open() as f:
                model_data["bin_edges"] = {
                    col: np.array(edges) for col, edges in json.load(f).items()
                }
            model_data["n_bins"] = metadata.get("n_bins", 5)
            return model_data
        if ptype == "xgboost":
            artifacts  = metadata.get("artifacts", {})
            buy_path   = symbol_dir / artifacts.get("buy_model",  f"{ticker}-xgb-buy.json")
            sell_path  = symbol_dir / artifacts.get("sell_model", f"{ticker}-xgb-sell.json")
            if not buy_path.exists() or not sell_path.exists():
                self.Log(f"WARNING: {ticker} XGBoost artifacts missing")
                return None
            with buy_path.open() as f:
                model_data["xgb_buy"] = json.load(f)
            with sell_path.open() as f:
                model_data["xgb_sell"] = json.load(f)
            # buy: score > bt, sell: score < -bt  (score = P(buy) - P(sell) ∈ [-1,1])
            bt = metadata.get("buy_threshold", 0.1)
            model_data["buy_threshold"]  = bt
            model_data["sell_threshold"] = -bt
            return model_data
        self.Log(f"WARNING: {ticker} unsupported policy type '{ptype}'")
        return None

    def _load_gmm_artifact(self, filename: str) -> dict | None:
        path = self.strategy_dir / filename
        if not path.exists():
            self.Log(f"WARNING: GMM artifact {filename} not found — regime defaulting to Hurst only")
            return None
        with path.open() as f:
            return json.load(f)

    def _load_correlation_artifact(self, filename: str) -> dict | None:
        path = self.strategy_dir / filename
        if not path.exists():
            self.Log(f"WARNING: Correlation artifact {filename} not found — correlation guard disabled")
            return None
        with path.open() as f:
            return json.load(f)

    def _load_earnings_artifact(self, filename: str) -> dict | None:
        path = self.strategy_dir / filename
        if not path.exists():
            self.Log(f"WARNING: Earnings calendar {filename} not found — earnings filter disabled")
            return None
        with path.open() as f:
            return json.load(f)

    # ── End of algorithm ──────────────────────────────────────────────────────

    def OnEndOfAlgorithm(self):
        total_days = sum(self._regime_day_counts.values()) or 1
        regime_pcts = {r: f"{v/total_days:.1%}" for r, v in self._regime_day_counts.items()}
        runtime = {
            "Watchlist":           str(len(self.watchlist)),
            "Active Models":       str(len(self.models)),
            "Executed Buys":       str(self.executed_buys),
            "Executed Sells":      str(self.executed_sells),
            "Stop Loss Exits":     str(self.stop_loss_exits),
            "Trailing Stop Exits": str(self.trailing_stop_exits),
            "Blocked Wash Sales":  str(self.blocked_wash_sales),
            "Blocked Min Hold":    str(self.blocked_min_hold),
            "Blocked Sell Streak": str(self.blocked_streak),
            "Sector Blocks":       str(self.sector_blocks),
            "Corr Guard Blocks":   str(self.corr_blocks),
            "Earnings Blocks":     str(self.earnings_blocks),
            "Transition Blocks":   str(self.transition_blocks),
            "Velocity Blocks":     str(self.velocity_blocks),
            "Total Tax":           f"${self.total_tax:,.2f}",
            "Short-Term Trades":   str(self.short_term_trades),
            "Long-Term Trades":    str(self.long_term_trades),
            "Regime BULL_CALM":    regime_pcts.get(BULL_CALM, "0%"),
            "Regime BULL_VOL":     regime_pcts.get(BULL_VOLATILE, "0%"),
            "Regime CHOPPY":       regime_pcts.get(CHOPPY, "0%"),
            "Regime BEAR":         regime_pcts.get(BEAR, "0%"),
        }
        for k, v in runtime.items():
            self.SetRuntimeStatistic(k, v)
        self.Log(
            f"End | models={len(self.models)} buys={self.executed_buys} "
            f"sells={self.executed_sells} tax=${self.total_tax:,.0f} "
            f"regimes={regime_pcts}"
        )

    # ── Charts ────────────────────────────────────────────────────────────────

    def _setup_charts(self) -> None:
        regime_chart = Chart("Regime")
        regime_chart.AddSeries(Series("State",      SeriesType.Line, ""))
        regime_chart.AddSeries(Series("Confidence", SeriesType.Line, "%"))
        regime_chart.AddSeries(Series("Hurst",      SeriesType.Line, "×100"))
        self.AddChart(regime_chart)

        port_chart = Chart("Portfolio")
        port_chart.AddSeries(Series("Positions", SeriesType.Line, "count"))
        self.AddChart(port_chart)

    def _plot_state(self, held_tickers: list) -> None:
        self.Plot("Portfolio", "Positions", len(held_tickers))
