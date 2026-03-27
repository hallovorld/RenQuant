from AlgorithmImports import *
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_strategy_config, split_date_parts


CONFIG = load_strategy_config()


class ZScoreScannerStrategy(QCAlgorithm):
	def Initialize(self):
		start_year, start_month, start_day = split_date_parts(CONFIG["backtest_start"])
		end_year, end_month, end_day = split_date_parts(CONFIG["backtest_end"])

		self.SetStartDate(start_year, start_month, start_day)
		self.SetEndDate(end_year, end_month, end_day)
		self.SetCash(CONFIG["initial_cash"])

		self.strategy_dir = Path(__file__).resolve().parent
		self.watchlist = CONFIG["watchlist"]
		self.benchmark = CONFIG.get("benchmark", "SPY")

		# Add equities for all watchlist stocks + benchmark
		self.symbols = {}
		for ticker in self.watchlist:
			self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol
		self.spy_symbol = self.AddEquity(self.benchmark, Resolution.Daily).Symbol

		# Volume z-score scanner config
		self.volume_zscore_lookback = int(CONFIG.get("volume_zscore_lookback", 15))
		self.volume_zscore_threshold = float(CONFIG.get("volume_zscore_threshold", 2.0))
		self.max_positions = int(CONFIG.get("max_concurrent_positions", 3))

		# Trade constraints
		self.wash_sale_days = int(CONFIG.get("wash_sale_days", 0))
		self.min_hold_days = int(CONFIG.get("min_hold_days", 0))
		self.max_hold_days = int(CONFIG.get("max_hold_days", 0))
		pos_sizing = CONFIG.get("position_sizing", {})
		self.max_position_pct = float(pos_sizing.get("max_position_pct", 0.33))
		self.cash_reserve_pct = float(pos_sizing.get("cash_reserve_pct", 0.10))

		# Load per-stock model artifacts
		self.models = {}
		for ticker in self.watchlist:
			self.models[ticker] = self._load_stock_model(ticker)

		# Per-stock tracking
		self.entry_times = {}
		self.last_sell_times = {}

		# Telemetry
		self.decision_counts = {"buy": 0, "sell": 0, "hold": 0}
		self.executed_buys = 0
		self.executed_sells = 0
		self.blocked_wash_sales = 0
		self.blocked_min_hold = 0
		self.volume_scans = 0
		self._setup_telemetry_chart()

		self.SetWarmUp(60)

	def OnData(self, data: Slice):
		if self.IsWarmingUp:
			return

		# Step 1: Process SELLS first for all held positions
		held_tickers = [t for t in self.watchlist if self.Portfolio[self.symbols[t]].Quantity > 0]

		for ticker in list(held_tickers):
			if not data.ContainsKey(self.symbols[ticker]):
				continue

			# Check max hold forced sell
			if self.max_hold_days > 0 and ticker in self.entry_times:
				days_held = (self.Time.date() - self.entry_times[ticker].date()).days
				if days_held >= self.max_hold_days:
					self._execute_sell(ticker, f"max_hold days_held={days_held}")
					held_tickers.remove(ticker)
					continue

			feature_frame = self._build_feature_frame(ticker)
			if feature_frame is None:
				continue

			action, detail, telemetry = self._choose_action(ticker, feature_frame)

			if action == "sell":
				action, constraint_detail = self._apply_sell_constraints(ticker, action)
				if action == "sell":
					self._execute_sell(ticker, detail)
					held_tickers.remove(ticker)
				else:
					self.Debug(f"{self.Time.date()} {ticker} sell blocked: {constraint_detail}")

		# Step 2: Count open slots
		open_slots = self.max_positions - len(held_tickers)
		if open_slots <= 0:
			return

		# Step 3: DETECT — scan volume z-scores for non-held stocks
		self.volume_scans += 1
		candidates = []
		for ticker in self.watchlist:
			if ticker in held_tickers:
				continue
			if not data.ContainsKey(self.symbols[ticker]):
				continue
			zscore = self._compute_volume_zscore(ticker)
			if zscore >= self.volume_zscore_threshold:
				candidates.append((ticker, zscore))

		# Step 4: Rank by z-score descending (most significant spike first)
		candidates.sort(key=lambda x: x[1], reverse=True)

		# Step 5: CONFIRM — run models on top candidates
		for ticker, zscore in candidates[:open_slots]:
			# Check wash sale constraint before computing features
			if self._is_wash_sale_blocked(ticker):
				continue

			feature_frame = self._build_feature_frame(ticker)
			if feature_frame is None:
				continue

			action, detail, telemetry = self._choose_action(ticker, feature_frame)
			self.decision_counts[action] += 1

			# Step 6: EXECUTE — buy if confirmed
			if action == "buy":
				self._execute_buy(ticker, zscore, detail)
				open_slots -= 1
				if open_slots <= 0:
					break

		self._plot_positions()

	def OnEndOfAlgorithm(self):
		runtime_stats = {
			"Watchlist Size": str(len(self.watchlist)),
			"Max Positions": str(self.max_positions),
			"Z-Score Lookback": str(self.volume_zscore_lookback),
			"Z-Score Threshold": f"{self.volume_zscore_threshold:.1f}",
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
		}
		for key, value in runtime_stats.items():
			self.SetRuntimeStatistic(key, value)

		self.Log(
			f"End summary | watchlist={len(self.watchlist)} max_pos={self.max_positions} "
			f"zscore_lookback={self.volume_zscore_lookback} zscore_threshold={self.volume_zscore_threshold} "
			f"buys={self.executed_buys} sells={self.executed_sells} "
			f"blocked_wash={self.blocked_wash_sales} blocked_min_hold={self.blocked_min_hold}"
		)

	# ── DETECT: Volume z-score scanning ──────────────────────────────────

	def _compute_volume_zscore(self, ticker: str) -> float:
		lookback = self.volume_zscore_lookback
		history = self.History(self.symbols[ticker], lookback + 1, Resolution.Daily)
		if history.empty or len(history) < lookback + 1:
			return 0.0
		volumes = history.loc[self.symbols[ticker]]["volume"]
		today_vol = volumes.iloc[-1]
		hist_vol = volumes.iloc[:-1]
		mean_vol = hist_vol.mean()
		std_vol = hist_vol.std()
		if std_vol <= 0:
			return 0.0
		return (today_vol - mean_vol) / std_vol

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
		self.executed_buys += 1
		self.SetHoldings(self.symbols[ticker], target_pct)

	def _execute_sell(self, ticker: str, detail: str) -> None:
		self.Debug(f"{self.Time.date()} {ticker} SELL {detail}")
		self.last_sell_times[ticker] = self.Time
		self.entry_times.pop(ticker, None)
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

	def _apply_sell_constraints(self, ticker: str, action: str) -> tuple[str, str]:
		if action != "sell":
			return action, ""
		if self.min_hold_days > 0 and ticker in self.entry_times:
			days_held = (self.Time.date() - self.entry_times[ticker].date()).days
			if days_held < self.min_hold_days:
				self.blocked_min_hold += 1
				return "hold", f"min_hold days_held={days_held}"
		return action, ""

	# ── CONFIRM: Feature computation ─────────────────────────────────────

	def _build_feature_frame(self, ticker: str):
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
		if len(common_idx) == 0:
			return None

		stock_ind = stock_ind.loc[common_idx]
		spy_ind = spy_ind.loc[common_idx]

		# Determine features from model metadata
		model = self.models[ticker]
		metadata = model["metadata"]
		feature_columns = metadata["feature_columns"]

		ratio_features = {"rsi", "adx"}
		diff_features = {"macd_hist", "cci", "bbp", "williams_r", "obv_slope"}

		result = pd.DataFrame(index=common_idx)
		for col in feature_columns:
			if col in ratio_features:
				result[col] = stock_ind[col] / spy_ind[col].replace(0, np.nan)
			elif col in diff_features:
				result[col] = stock_ind[col] - spy_ind[col]
			else:
				# Trend/breakout features computed from stock data only
				result[col] = stock_ind.get(col, pd.Series(np.nan, index=common_idx))

		# Add trend features for manual models that need them
		policy_type = metadata.get("policy_type", "classification")
		if policy_type == "manual":
			close = stock_ind["close"]
			ema50 = close.ewm(span=50, adjust=False).mean()
			ema200 = close.ewm(span=200, adjust=False).mean()
			result["trend"] = close / ema50
			result["trend_long"] = close / ema200

			spy_close = spy_ind["close"]
			rel_price = close / spy_close
			result["rel_mom_20d"] = rel_price.pct_change(20)
			result["rel_mom_60d"] = rel_price.pct_change(60)

			# Breakout features
			result["high_20d_break"] = close - close.rolling(20).max()
			result["low_20d_break"] = close - close.rolling(20).min()

		result = result.dropna()
		if result.empty:
			return None

		return result.iloc[-1:]

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

	# ── Model loading and inference ──────────────────────────────────────

	def _load_stock_model(self, ticker: str) -> dict:
		model_name = CONFIG["model_name"]
		artifact_name = f"{model_name}-{ticker}"

		meta_path = self.strategy_dir / f"{artifact_name}-policy-metadata.json"
		if not meta_path.exists():
			raise RuntimeError(
				f"Policy metadata not found for {ticker}: {meta_path}. "
				"Run the training notebook to export model artifacts."
			)
		with meta_path.open() as f:
			metadata = json.load(f)

		policy_type = metadata.get("policy_type", "classification")
		model = {"metadata": metadata, "policy_type": policy_type}

		if policy_type == "classification":
			trees_path = self.strategy_dir / f"{artifact_name}-rf-trees.json"
			if not trees_path.exists():
				raise RuntimeError(f"RF trees not found for {ticker}: {trees_path}")
			with trees_path.open() as f:
				model["trees"] = json.load(f)

		elif policy_type == "manual":
			rules_path = self.strategy_dir / f"{artifact_name}-manual-rules.json"
			if not rules_path.exists():
				raise RuntimeError(f"Manual rules not found for {ticker}: {rules_path}")
			with rules_path.open() as f:
				data = json.load(f)
			model["score_rules"] = data["score_rules"]
			model["buy_threshold"] = data["buy_threshold"]
			model["sell_threshold"] = data["sell_threshold"]

		else:
			raise RuntimeError(f"Unsupported policy type for {ticker}: {policy_type}")

		return model

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

	def _score_manual_rules(self, row: pd.Series, score_rules: list) -> int:
		score = 0
		for rule in score_rules:
			value = row.get(rule["col"])
			if value is None:
				continue
			if "buy_below" in rule and value < rule["buy_below"]:
				score += 1
			if "buy_above" in rule and value > rule["buy_above"]:
				score += 1
			if "sell_above" in rule and value > rule["sell_above"]:
				score -= 1
			if "sell_below" in rule and value < rule["sell_below"]:
				score -= 1
		return score

	def _choose_action(self, ticker: str, feature_frame: pd.DataFrame) -> tuple[str, str, dict]:
		model = self.models[ticker]
		metadata = model["metadata"]
		policy_type = model["policy_type"]
		row = feature_frame.iloc[0]

		if policy_type == "classification":
			features = [row[col] for col in metadata["feature_columns"]]
			score = self._bag_predict(model["trees"], features)
			buy_threshold = metadata.get("buy_threshold", 0.1)
			sell_threshold = metadata.get("sell_threshold", -0.1)
			telemetry = {"score": score, "buy_threshold": buy_threshold, "sell_threshold": sell_threshold}

			if score > buy_threshold:
				return "buy", f"score={score:.4f}", telemetry
			if score < sell_threshold:
				return "sell", f"score={score:.4f}", telemetry
			return "hold", f"score={score:.4f}", telemetry

		if policy_type == "manual":
			score = self._score_manual_rules(row, model["score_rules"])
			buy_threshold = model["buy_threshold"]
			sell_threshold = model["sell_threshold"]
			telemetry = {"score": score, "buy_threshold": buy_threshold, "sell_threshold": sell_threshold}

			if score >= buy_threshold:
				return "buy", f"score={score}", telemetry
			if score <= sell_threshold:
				return "sell", f"score={score}", telemetry
			return "hold", f"score={score}", telemetry

		raise RuntimeError(f"Unsupported policy type: {policy_type}")

	# ── Telemetry ────────────────────────────────────────────────────────

	def _setup_telemetry_chart(self) -> None:
		chart = Chart("Portfolio")
		chart.AddSeries(Series("Positions Held", SeriesType.Line, "count"))
		self.AddChart(chart)

	def _plot_positions(self) -> None:
		held_count = sum(1 for t in self.watchlist if self.Portfolio[self.symbols[t]].Quantity > 0)
		self.Plot("Portfolio", "Positions Held", held_count)
