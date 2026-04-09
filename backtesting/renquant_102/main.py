from AlgorithmImports import *
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_strategy_config, split_date_parts


CONFIG = load_strategy_config()


class PreTrainedMultiStockStrategy(QCAlgorithm):
	def Initialize(self):
		start_year, start_month, start_day = split_date_parts(CONFIG["backtest_start"])
		end_year, end_month, end_day = split_date_parts(CONFIG["backtest_end"])

		self.SetStartDate(start_year, start_month, start_day)
		self.SetEndDate(end_year, end_month, end_day)
		self.SetCash(CONFIG["initial_cash"])

		self.strategy_dir = Path(__file__).resolve().parent
		self.watchlist = CONFIG["watchlist"]
		self._benchmark_ticker = CONFIG.get("benchmark", "SPY")

		# Add equities for all watchlist stocks + benchmark
		self.symbols = {}
		for ticker in self.watchlist:
			self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol
		self.spy_symbol = self.AddEquity(self._benchmark_ticker, Resolution.Daily).Symbol

		# Volume scanner config — adaptive percentile-based filter
		self.volume_zscore_lookback = int(CONFIG.get("volume_zscore_lookback", 20))
		self.volume_zscore_threshold = float(CONFIG.get("volume_zscore_threshold", 1.5))
		vol_filter = CONFIG.get("volume_filter", {})
		self.volume_filter_mode = vol_filter.get("mode", "percentile")  # "percentile" or "zscore"
		self.volume_percentile_threshold = float(vol_filter.get("percentile_threshold", 85))
		self.max_positions = int(CONFIG.get("max_concurrent_positions", 3))

		# Trade constraints
		self.wash_sale_days = int(CONFIG.get("wash_sale_days", 0))
		self.min_hold_days = int(CONFIG.get("min_hold_days", 0))
		self.max_hold_days = int(CONFIG.get("max_hold_days", 0))
		pos_sizing = CONFIG.get("position_sizing", {})
		self.max_position_pct = float(pos_sizing.get("max_position_pct", 0.33))
		self.cash_reserve_pct = float(pos_sizing.get("cash_reserve_pct", 0.10))

		# Load pre-trained models
		staleness_days = int(CONFIG.get("model_staleness_days", 30))
		self.models = {}
		self._load_all_models(staleness_days)

		# Risk management
		risk_cfg = CONFIG.get("risk", {})
		self.stop_loss_pct = float(risk_cfg.get("stop_loss_pct", 0.0))
		self.drawdown_halt_pct = float(risk_cfg.get("portfolio_drawdown_halt_pct", 0.0))
		self.high_water_mark = float(CONFIG["initial_cash"])
		self._skip_buys = False
		regime_cfg = risk_cfg.get("regime_filter", {})
		self.regime_filter_enabled = bool(regime_cfg.get("enabled", False))
		self.regime_sma_period = int(regime_cfg.get("sma_period", 200))

		# Sector concentration
		self.sector_map = CONFIG.get("sector_map", {})
		self.max_positions_per_sector = int(CONFIG.get("max_positions_per_sector", 0))

		# Per-stock tracking
		self.entry_times = {}
		self.entry_prices = {}
		self.last_sell_times = {}

		# Telemetry
		self.decision_counts = {"buy": 0, "sell": 0, "hold": 0}
		self.executed_buys = 0
		self.executed_sells = 0
		self.blocked_wash_sales = 0
		self.blocked_min_hold = 0
		self.stop_loss_exits = 0
		self.regime_blocks = 0
		self.sector_blocks = 0
		self.volume_scans = 0
		# Tax tracking
		tax_cfg = CONFIG.get("tax", {})
		self.tax_short_rate = float(tax_cfg.get("short_term_rate", 0.50))
		self.tax_long_rate = float(tax_cfg.get("long_term_rate", 0.32))
		self.tax_threshold_days = int(tax_cfg.get("long_term_threshold_days", 365))
		self.total_tax = 0.0
		self.short_term_trades = 0
		self.long_term_trades = 0
		self._setup_telemetry_chart()

		self.SetWarmUp(60)

	def OnData(self, data: Slice):
		if self.IsWarmingUp:
			return

		# Portfolio drawdown circuit breaker — update high-water mark
		portfolio_value = self.Portfolio.TotalPortfolioValue
		self.high_water_mark = max(self.high_water_mark, portfolio_value)
		if self.drawdown_halt_pct > 0 and self.high_water_mark > 0:
			drawdown = (self.high_water_mark - portfolio_value) / self.high_water_mark
			self._skip_buys = drawdown >= self.drawdown_halt_pct
			if self._skip_buys:
				self.Debug(f"{self.Time.date()} drawdown={drawdown:.1%} >= {self.drawdown_halt_pct:.0%}: halting new buys")

		# Step 1: Process SELLS first for all held positions
		held_tickers = [t for t in self.models if self.Portfolio[self.symbols[t]].Quantity > 0]

		for ticker in list(held_tickers):
			if not data.ContainsKey(self.symbols[ticker]):
				continue

			# Stop-loss check
			if self.stop_loss_pct > 0 and ticker in self.entry_prices:
				current_price = float(data[self.symbols[ticker]].Close)
				avg_price = self.entry_prices[ticker]
				if avg_price > 0:
					loss_pct = (avg_price - current_price) / avg_price
					if loss_pct >= self.stop_loss_pct:
						self._execute_sell(ticker, f"stop_loss loss={loss_pct:.1%}")
						self.stop_loss_exits += 1
						held_tickers.remove(ticker)
						continue

			# Check max hold forced sell
			if self.max_hold_days > 0 and ticker in self.entry_times:
				days_held = (self.Time.date() - self.entry_times[ticker].date()).days
				if days_held >= self.max_hold_days:
					self._execute_sell(ticker, f"max_hold days_held={days_held}")
					held_tickers.remove(ticker)
					continue

			features = self._build_feature_frame(ticker)
			if features is None:
				continue

			action, detail = self._choose_action(ticker, features)
			if action == "sell":
				action, constraint_detail = self._apply_sell_constraints(ticker, action)
				if action == "sell":
					self._execute_sell(ticker, detail)
					held_tickers.remove(ticker)
				else:
					self.Debug(f"{self.Time.date()} {ticker} sell blocked: {constraint_detail}")

		# Step 2: Count open slots
		open_slots = self.max_positions - len(held_tickers)
		if open_slots <= 0 or self._skip_buys:
			return

		# Step 3: Regime filter — suppress new entries when SPY is below its 200-day SMA
		if self.regime_filter_enabled and not self._is_bull_regime():
			self.regime_blocks += 1
			self.Debug(f"{self.Time.date()} regime filter: SPY below 200-SMA, skipping buys")
			return

		# Step 4: DETECT — scan volume for non-held stocks (adaptive per-stock)
		self.volume_scans += 1
		candidates = []
		for ticker in self.models:
			if ticker in held_tickers:
				continue
			if not data.ContainsKey(self.symbols[ticker]):
				continue
			score, triggered = self._compute_volume_score(ticker)
			if triggered:
				# Bullish filter: only enter on days where price closed up
				if not self._is_price_up(ticker):
					continue
				candidates.append((ticker, score))

		# Step 5: Rank by volume score descending (highest activity first)
		candidates.sort(key=lambda x: x[1], reverse=True)

		# Step 6: CONFIRM + EXECUTE — apply pre-trained model
		for ticker, zscore in candidates[:open_slots]:
			if self._is_wash_sale_blocked(ticker):
				continue

			# Sector concentration guard
			if self.max_positions_per_sector > 0:
				sector = self.sector_map.get(ticker, "other")
				sector_count = sum(1 for t in held_tickers if self.sector_map.get(t, "other") == sector)
				if sector_count >= self.max_positions_per_sector:
					self.sector_blocks += 1
					self.Debug(f"{self.Time.date()} {ticker} sector={sector} already at {sector_count}/{self.max_positions_per_sector}, skipping")
					continue

			features = self._build_feature_frame(ticker)
			if features is None:
				continue

			action, detail = self._choose_action(ticker, features)
			self.decision_counts[action] += 1

			if action == "buy":
				self._execute_buy(ticker, zscore, detail)
				held_tickers.append(ticker)
				open_slots -= 1
				if open_slots <= 0:
					break

		self._plot_positions()

	def OnEndOfAlgorithm(self):
		runtime_stats = {
			"Watchlist Size": str(len(self.watchlist)),
			"Active Models": str(len(self.models)),
			"Max Positions": str(self.max_positions),
			"Volume Filter Mode": self.volume_filter_mode,
			"Volume Lookback": str(self.volume_zscore_lookback),
			"Volume Threshold": f"P{self.volume_percentile_threshold:.0f}" if self.volume_filter_mode == "percentile" else f"Z{self.volume_zscore_threshold:.1f}",
			"Wash Sale Days": str(self.wash_sale_days),
			"Min Hold Days": str(self.min_hold_days),
			"Max Hold Days": str(self.max_hold_days),
			"Buy Decisions": str(self.decision_counts["buy"]),
			"Sell Decisions": str(self.decision_counts["sell"]),
			"Hold Decisions": str(self.decision_counts["hold"]),
			"Executed Buys": str(self.executed_buys),
			"Executed Sells": str(self.executed_sells),
			"Blocked Wash Sales": str(self.blocked_wash_sales),
			"Blocked Min Hold": str(self.blocked_min_hold),
			"Volume Scans": str(self.volume_scans),
			"Total Tax": f"${self.total_tax:,.2f}",
			"Short-Term Trades": str(self.short_term_trades),
			"Long-Term Trades": str(self.long_term_trades),
			"Tax Rate (ST)": f"{self.tax_short_rate:.0%}",
			"Tax Rate (LT)": f"{self.tax_long_rate:.0%}",
			"Stop Loss Exits": str(self.stop_loss_exits),
			"Regime Blocks": str(self.regime_blocks),
			"Sector Blocks": str(self.sector_blocks),
		}
		for key, value in runtime_stats.items():
			self.SetRuntimeStatistic(key, value)

		model_types = {}
		for ticker, model in self.models.items():
			pt = model["policy_type"]
			model_types[pt] = model_types.get(pt, 0) + 1

		self.Log(
			f"End summary | active_models={len(self.models)} model_types={model_types} "
			f"buys={self.executed_buys} sells={self.executed_sells} "
			f"blocked_wash={self.blocked_wash_sales} blocked_min_hold={self.blocked_min_hold}"
		)

	# ── Model loading ────────────────────────────────────────────────────

	def _load_all_models(self, staleness_days: int) -> None:
		models_dir = self.strategy_dir / "models"
		if not models_dir.exists():
			self.Log("WARNING: models/ directory not found. Run the notebook to train models.")
			return

		for ticker in self.watchlist:
			symbol_dir = models_dir / ticker
			meta_path = symbol_dir / f"{ticker}-policy-metadata.json"
			if not meta_path.exists():
				self.Log(f"WARNING: No model for {ticker}, skipping")
				continue

			with meta_path.open() as f:
				metadata = json.load(f)

			# Check staleness
			trained_date = metadata.get("trained_date")
			if trained_date and staleness_days > 0:
				age = (datetime.now().date() - datetime.strptime(trained_date, "%Y-%m-%d").date()).days
				if age > staleness_days:
					self.Log(f"WARNING: {ticker} model is {age} days old (limit={staleness_days}), skipping")
					continue

			model_data = self._load_model_artifacts(ticker, metadata, symbol_dir)
			if model_data is not None:
				self.models[ticker] = model_data

		n_loaded = len(self.models)
		n_watchlist = len(self.watchlist)
		self.Log(f"Loaded models for {n_loaded}/{n_watchlist} symbols: {sorted(self.models.keys())}")
		if n_loaded == 0:
			self.Log("CRITICAL: No models loaded. Check model staleness and retrain notebook.")
		elif n_loaded < n_watchlist // 2:
			self.Log(f"WARNING: Only {n_loaded}/{n_watchlist} models loaded — check staleness or retrain.")

	def _load_model_artifacts(self, ticker: str, metadata: dict, symbol_dir: Path) -> dict | None:
		policy_type = metadata["policy_type"]
		feature_columns = metadata.get("feature_columns", [])
		model_data = {
			"policy_type": policy_type,
			"feature_columns": feature_columns,
			"buy_threshold": metadata.get("buy_threshold", 0.1),
			"sell_threshold": metadata.get("sell_threshold", -0.1),
		}

		if policy_type == "classification":
			trees_path = symbol_dir / f"{ticker}-rf-trees.json"
			if not trees_path.exists():
				self.Log(f"WARNING: {ticker} trees artifact missing, skipping")
				return None
			with trees_path.open() as f:
				model_data["trees"] = json.load(f)
			return model_data

		if policy_type == "manual":
			rules_path = symbol_dir / f"{ticker}-manual-rules.json"
			if not rules_path.exists():
				self.Log(f"WARNING: {ticker} manual rules artifact missing, skipping")
				return None
			with rules_path.open() as f:
				data = json.load(f)
			model_data["score_rules"] = data["score_rules"]
			model_data["buy_threshold"] = data["buy_threshold"]
			model_data["sell_threshold"] = data["sell_threshold"]
			return model_data

		if policy_type == "qlearning":
			qtable_path = symbol_dir / f"{ticker}-qtable.json"
			edges_path = symbol_dir / f"{ticker}-bin-edges.json"
			if not qtable_path.exists() or not edges_path.exists():
				self.Log(f"WARNING: {ticker} Q-learning artifacts missing, skipping")
				return None
			with qtable_path.open() as f:
				model_data["q_table"] = np.array(json.load(f))
			with edges_path.open() as f:
				model_data["bin_edges"] = {
					col: np.array(edges) for col, edges in json.load(f).items()
				}
			model_data["n_bins"] = metadata.get("n_bins", 5)
			return model_data

		self.Log(f"WARNING: {ticker} unsupported policy type '{policy_type}', skipping")
		return None

	# ── Regime detection ────────────────────────────────────────────────

	def _is_bull_regime(self) -> bool:
		"""Return True if SPY is above its 200-day SMA (bull regime). Fails open on insufficient data."""
		history = self.History(self.spy_symbol, self.regime_sma_period + 1, Resolution.Daily)
		if history.empty or len(history) < self.regime_sma_period:
			return True
		closes = history.loc[self.spy_symbol]["close"]
		sma200 = float(closes.iloc[:-1].mean())
		return float(closes.iloc[-1]) > sma200

	# ── DETECT: Volume scanning ──────────────────────────────────────────

	def _compute_volume_score(self, ticker: str) -> tuple[float, bool]:
		"""Compute volume activity score for a ticker.

		Returns (score, triggered) where:
		  - percentile mode: score = percentile rank of today's volume (0-100),
		    triggered if score >= percentile_threshold.  Each stock is measured
		    against its own recent history, so liquid large-caps and volatile
		    small-caps are held to the same *relative* standard.
		  - zscore mode: score = z-score, triggered if score >= zscore threshold.
		    (legacy behaviour, kept for backward compatibility)
		"""
		lookback = self.volume_zscore_lookback
		history = self.History(self.symbols[ticker], lookback + 1, Resolution.Daily)
		if history.empty or len(history) < lookback + 1:
			return 0.0, False
		volumes = history.loc[self.symbols[ticker]]["volume"]
		today_vol = volumes.iloc[-1]
		hist_vol = volumes.iloc[:-1]

		if self.volume_filter_mode == "percentile":
			# Percentile rank: what fraction of lookback days had lower volume?
			rank = (hist_vol < today_vol).sum() / len(hist_vol) * 100
			return rank, rank >= self.volume_percentile_threshold
		else:
			# Legacy z-score mode
			mean_vol = hist_vol.mean()
			std_vol = hist_vol.std()
			if std_vol <= 0:
				return 0.0, False
			zscore = (today_vol - mean_vol) / std_vol
			return zscore, zscore >= self.volume_zscore_threshold

	def _is_price_up(self, ticker: str) -> bool:
		"""Check if today's close is above yesterday's close (bullish day)."""
		history = self.History(self.symbols[ticker], 2, Resolution.Daily)
		if history.empty or len(history) < 2:
			return False
		closes = history.loc[self.symbols[ticker]]["close"]
		return closes.iloc[-1] > closes.iloc[-2]

	# ── Feature computation ──────────────────────────────────────────────

	def _build_feature_frame(self, ticker: str):
		"""Fetch 60-day history, compute indicators + relative features."""
		stock_history = self.History(self.symbols[ticker], 60, Resolution.Daily)
		spy_history = self.History(self.spy_symbol, 60, Resolution.Daily)

		if stock_history.empty or spy_history.empty:
			return None

		stock_rows = stock_history.loc[self.symbols[ticker]].copy()
		spy_rows = spy_history.loc[self.spy_symbol].copy()

		if len(stock_rows) < 40 or len(spy_rows) < 40:
			return None

		stock_ind = self._compute_indicators(stock_rows)
		spy_ind = self._compute_indicators(spy_rows)

		if stock_ind is None or spy_ind is None:
			return None

		common_idx = stock_ind.index.intersection(spy_ind.index)
		if len(common_idx) < 10:
			return None

		stock_ind = stock_ind.loc[common_idx]
		spy_ind = spy_ind.loc[common_idx]

		ratio_features = {"rsi", "adx"}
		diff_features = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

		result = pd.DataFrame(index=common_idx)
		result["close"] = stock_ind["close"]

		# Relative indicator features (for Classification, Mean Reversion)
		all_feature_cols = ["rsi", "macd_hist", "cci", "bbp", "adx", "williams_r", "obv_slope"]
		for col in all_feature_cols:
			if col in ratio_features:
				result[col] = stock_ind[col] / spy_ind[col].replace(0, np.nan)
			elif col in diff_features:
				result[col] = stock_ind[col] - spy_ind[col]

		# Trend features (for Dual Momentum, Q-Learning)
		close = stock_ind["close"]
		ema50 = close.ewm(span=50, adjust=False).mean()
		ema200 = close.ewm(span=200, adjust=False).mean()
		result["trend"] = close / ema50
		result["trend_long"] = close / ema200

		spy_close = spy_ind["close"]
		rel_price = close / spy_close
		result["rel_mom_20d"] = rel_price.pct_change(20)
		result["rel_mom_60d"] = rel_price.pct_change(60)

		result = result.dropna()
		if result.empty:
			return None

		return result

	def _compute_indicators(self, rows: pd.DataFrame) -> pd.DataFrame | None:
		rows = rows.copy()
		close = rows["close"]
		high = rows["high"]
		low = rows["low"]
		volume = rows["volume"]

		# RSI (period=14)
		delta = close.diff()
		avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
		avg_loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
		rows["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss.replace(0, np.nan)))

		# MACD histogram (fast=12, slow=26, signal=9)
		ema_fast = close.ewm(span=12, adjust=False).mean()
		ema_slow = close.ewm(span=26, adjust=False).mean()
		macd_line = ema_fast - ema_slow
		rows["macd_hist"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

		# CCI (period=20)
		typical_price = (high + low + close) / 3
		cci_sma = typical_price.rolling(20).mean()
		cci_mad = typical_price.rolling(20).apply(
			lambda v: np.mean(np.abs(v - v.mean())), raw=True
		)
		rows["cci"] = (typical_price - cci_sma) / (0.015 * cci_mad.replace(0, np.nan))

		# BBP — Bollinger Band Percentage (period=20)
		sma20 = close.rolling(20).mean()
		std20 = close.rolling(20).std()
		rows["bbp"] = (close - sma20) / (2 * std20.replace(0, np.nan))

		# ADX (period=14)
		up_move = high.diff()
		down_move = -low.diff()
		plus_dm = pd.Series(
			np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=rows.index
		)
		minus_dm = pd.Series(
			np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=rows.index
		)
		tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
		atr14 = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
		plus_di = 100 * plus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
		minus_di = 100 * minus_dm.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean() / atr14.replace(0, np.nan)
		dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
		rows["adx"] = dx.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

		# Williams %R (period=14)
		highest_high = high.rolling(14).max()
		lowest_low = low.rolling(14).min()
		rows["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)

		# OBV slope (signal_period=20)
		obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
		obv_ema = obv.ewm(span=20, adjust=False).mean()
		rows["obv_slope"] = obv_ema.diff(5) / obv_ema.shift(5).replace(0, np.nan)

		indicator_cols = ["rsi", "macd_hist", "cci", "bbp", "adx", "williams_r", "obv_slope"]
		rows = rows.dropna(subset=indicator_cols)
		if rows.empty:
			return None

		return rows

	# ── Model prediction ─────────────────────────────────────────────────

	def _choose_action(self, ticker: str, features: pd.DataFrame) -> tuple:
		"""Apply pre-trained model for this ticker to get buy/sell/hold signal."""
		model = self.models[ticker]
		policy_type = model["policy_type"]
		row = features.iloc[-1]

		if policy_type == "classification":
			feat_cols = model["feature_columns"]
			feat_vals = [row.get(c, np.nan) for c in feat_cols]
			if any(np.isnan(v) for v in feat_vals):
				return "hold", "missing_features"
			score = self._bag_predict(model["trees"], feat_vals)
			if score > model["buy_threshold"]:
				return "buy", f"RF score={score:.3f}"
			if score < model["sell_threshold"]:
				return "sell", f"RF score={score:.3f}"
			return "hold", f"RF score={score:.3f}"

		if policy_type == "manual":
			score = self._score_manual_rules(row, model["score_rules"])
			if score >= model["buy_threshold"]:
				return "buy", f"manual score={score}"
			if score <= model["sell_threshold"]:
				return "sell", f"manual score={score}"
			return "hold", f"manual score={score}"

		if policy_type == "qlearning":
			holdings = self.Portfolio[self.symbols[ticker]].Quantity
			feat_cols = model["feature_columns"]
			state = self._encode_q_state(row, holdings, feat_cols, model["bin_edges"], model["n_bins"])
			action_id = int(np.argmax(model["q_table"][state]))
			action_name = {0: "buy", 1: "sell", 2: "hold"}[action_id]
			return action_name, f"QL state={state} action_id={action_id}"

		return "hold", f"unsupported_policy={policy_type}"

	def _traverse_tree(self, tree: list, row: list) -> float:
		idx = 0
		while True:
			feat, split_val, left_off, right_off = tree[idx]
			if feat == -1:
				return split_val
			idx += int(left_off) if row[int(feat)] <= split_val else int(right_off)

	def _bag_predict(self, trees: list, features: list) -> float:
		preds = [self._traverse_tree(tree, features) for tree in trees]
		return sum(preds) / len(preds)

	def _score_manual_rules(self, row: pd.Series, rules: list) -> int:
		score = 0
		for rule in rules:
			value = row.get(rule["col"])
			if value is None or (isinstance(value, float) and np.isnan(value)):
				continue
			if "buy_below" in rule and rule["buy_below"] is not None and value < rule["buy_below"]:
				score += 1
			if "buy_above" in rule and rule["buy_above"] is not None and value > rule["buy_above"]:
				score += 1
			if "sell_above" in rule and rule["sell_above"] is not None and value > rule["sell_above"]:
				score -= 1
			if "sell_below" in rule and rule["sell_below"] is not None and value < rule["sell_below"]:
				score -= 1
		return score

	def _encode_q_state(self, row: pd.Series, holdings: float,
	                    feature_columns: list, bin_edges: dict, n_bins: int) -> int:
		state = 0
		for col in feature_columns:
			val = row.get(col, 0)
			bin_idx = np.digitize(val, bin_edges[col]) - 1
			bin_idx = int(np.clip(bin_idx, 0, n_bins - 1))
			state = state * n_bins + bin_idx

		holding_bucket = 2 if holdings > 0 else (0 if holdings < 0 else 1)
		return state * 3 + holding_bucket

	# ── Trade execution ──────────────────────────────────────────────────

	def _execute_buy(self, ticker: str, zscore: float, detail: str) -> None:
		portfolio_value = self.Portfolio.TotalPortfolioValue
		available_cash = self.Portfolio.Cash
		cash_reserve = portfolio_value * self.cash_reserve_pct
		investable = max(available_cash - cash_reserve, 0)
		target_pct = min(self.max_position_pct, investable / max(portfolio_value, 1))

		if target_pct < 0.01:
			self.Debug(f"{self.Time.date()} {ticker} buy skipped — insufficient cash")
			return

		self.Debug(f"{self.Time.date()} {ticker} BUY zscore={zscore:.2f} target_pct={target_pct:.4f} {detail}")
		self.entry_times[ticker] = self.Time
		self.entry_prices[ticker] = float(self.Securities[self.symbols[ticker]].Price)
		self.executed_buys += 1
		self.SetHoldings(self.symbols[ticker], target_pct)

	def _execute_sell(self, ticker: str, detail: str) -> None:
		# Track tax on this trade
		gross_pnl = self.Portfolio[self.symbols[ticker]].UnrealizedProfit
		entry_time = self.entry_times.get(ticker)
		days_held = (self.Time.date() - entry_time.date()).days if entry_time else 0
		is_long_term = days_held >= self.tax_threshold_days
		tax_rate = self.tax_long_rate if is_long_term else self.tax_short_rate
		tax = max(gross_pnl, 0) * tax_rate
		self.total_tax += tax
		if is_long_term:
			self.long_term_trades += 1
		else:
			self.short_term_trades += 1
		self.Debug(f"{self.Time.date()} {ticker} SELL pnl=${gross_pnl:.2f} held={days_held}d "
		           f"tax=${tax:.2f} ({'LT' if is_long_term else 'ST'} {tax_rate:.0%}) {detail}")
		self.last_sell_times[ticker] = self.Time
		self.entry_times.pop(ticker, None)
		self.entry_prices.pop(ticker, None)
		self.executed_sells += 1
		self.Liquidate(self.symbols[ticker])

	# ── Trade constraints ────────────────────────────────────────────────

	def _is_wash_sale_blocked(self, ticker: str) -> bool:
		if self.wash_sale_days <= 0:
			return False
		last_sell = self.last_sell_times.get(ticker)
		if last_sell is None:
			return False
		days_since_sell = (self.Time.date() - last_sell.date()).days
		if days_since_sell < self.wash_sale_days:
			self.blocked_wash_sales += 1
			return True
		return False

	def _apply_sell_constraints(self, ticker: str, action: str) -> tuple:
		if action != "sell":
			return action, ""
		if self.min_hold_days > 0 and ticker in self.entry_times:
			days_held = (self.Time.date() - self.entry_times[ticker].date()).days
			if days_held < self.min_hold_days:
				self.blocked_min_hold += 1
				return "hold", f"min_hold days_held={days_held}"
		return action, ""

	# ── Telemetry ────────────────────────────────────────────────────────

	def _setup_telemetry_chart(self) -> None:
		chart = Chart("Portfolio")
		chart.AddSeries(Series("Positions Held", SeriesType.Line, "count"))
		self.AddChart(chart)

	def _plot_positions(self) -> None:
		held_count = sum(1 for t in self.models if self.Portfolio[self.symbols[t]].Quantity > 0)
		self.Plot("Portfolio", "Positions Held", held_count)
