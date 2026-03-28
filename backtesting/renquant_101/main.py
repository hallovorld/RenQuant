from AlgorithmImports import *
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import load_strategy_config, split_date_parts


CONFIG = load_strategy_config()


class ClassificationNVDAStrategy(QCAlgorithm):
	def Initialize(self):
		start_year, start_month, start_day = split_date_parts(CONFIG["backtest_start"])
		end_year, end_month, end_day = split_date_parts(CONFIG["backtest_end"])

		self.SetStartDate(start_year, start_month, start_day)
		self.SetEndDate(end_year, end_month, end_day)
		self.SetCash(CONFIG["initial_cash"])

		self.symbol = self.AddEquity(CONFIG["stock_symbol"], Resolution.Daily).Symbol
		self.strategy_dir = Path(__file__).resolve().parent
		self.policy_metadata = self._load_policy_metadata()
		self.policy_type = self.policy_metadata["policy_type"]
		self.feature_columns = self.policy_metadata["feature_columns"]
		self.buy_threshold = self.policy_metadata.get("buy_threshold", 0.1)
		self.sell_threshold = self.policy_metadata.get("sell_threshold", -0.1)
		self.wash_sale_days = int(CONFIG.get("wash_sale_days", 0))
		self.min_hold_days = int(CONFIG.get("min_hold_days", 0))
		self.max_hold_days = int(CONFIG.get("max_hold_days", 0))
		pos_sizing = CONFIG.get("position_sizing", {})
		self.max_position_pct = float(pos_sizing.get("max_position_pct", 1.0))
		self.cash_reserve_pct = float(pos_sizing.get("cash_reserve_pct", 0.0))
		self.trees = []
		self.score_rules = []
		self.bin_edges = {}
		self.n_bins = 0
		self.q_table = None
		self._load_policy_artifacts()
		self.last_decision = None
		self.last_sell_time = None
		self.entry_time = None
		self.decision_counts = {"buy": 0, "sell": 0, "hold": 0}
		self.executed_buys = 0
		self.executed_sells = 0
		self.blocked_wash_sales = 0
		self.blocked_min_hold = 0
		# Tax tracking
		tax_cfg = CONFIG.get("tax", {})
		self.tax_short_rate = float(tax_cfg.get("short_term_rate", 0.50))
		self.tax_long_rate = float(tax_cfg.get("long_term_rate", 0.32))
		self.tax_threshold_days = int(tax_cfg.get("long_term_threshold_days", 365))
		self.total_tax = 0.0
		self.short_term_trades = 0
		self.long_term_trades = 0
		self._setup_telemetry_chart()

		self.SetWarmUp(40)

	def OnData(self, data: Slice):
		if self.IsWarmingUp or not data.ContainsKey(self.symbol):
			return

		feature_frame = self._build_feature_frame()
		if feature_frame is None:
			return

		chosen_action, detail, telemetry = self._choose_action(feature_frame)
		raw_action = chosen_action
		chosen_action, constraint_detail = self._apply_trade_constraints(chosen_action)
		if constraint_detail:
			detail = f"{detail} {constraint_detail}"
		self.decision_counts[chosen_action] += 1
		holdings = self.Portfolio[self.symbol].Quantity
		current_state = "long" if holdings > 0 else "flat"
		self._plot_telemetry(telemetry, raw_action, chosen_action)

		if chosen_action != self.last_decision:
			self.Debug(
				f"{self.Time.date()} policy={self.policy_type} action={chosen_action} portfolio={current_state} {detail}"
			)
			self.last_decision = chosen_action

		if chosen_action == "buy" and holdings <= 0:
			portfolio_value = self.Portfolio.TotalPortfolioValue
			available_cash = self.Portfolio.Cash
			cash_reserve = portfolio_value * self.cash_reserve_pct
			investable = max(available_cash - cash_reserve, 0)
			target_pct = min(self.max_position_pct, investable / max(portfolio_value, 1))
			if target_pct < 0.01:
				self.Debug(f"{self.Time.date()} buy skipped — insufficient cash (target_pct={target_pct:.4f})")
			else:
				self.Debug(f"{self.Time.date()} submitting buy order target_pct={target_pct:.4f}")
				self.entry_time = self.Time
				self.executed_buys += 1
				self.SetHoldings(self.symbol, target_pct)
		elif chosen_action == "sell" and holdings > 0:
			# Track tax on this trade
			gross_pnl = self.Portfolio[self.symbol].UnrealizedProfit
			days_held = (self.Time.date() - self.entry_time.date()).days if self.entry_time else 0
			is_long_term = days_held >= self.tax_threshold_days
			tax_rate = self.tax_long_rate if is_long_term else self.tax_short_rate
			tax = max(gross_pnl, 0) * tax_rate
			self.total_tax += tax
			if is_long_term:
				self.long_term_trades += 1
			else:
				self.short_term_trades += 1
			self.Debug(f"{self.Time.date()} SELL pnl=${gross_pnl:.2f} held={days_held}d "
			           f"tax=${tax:.2f} ({'LT' if is_long_term else 'ST'} {tax_rate:.0%})")
			self.last_sell_time = self.Time
			self.entry_time = None
			self.executed_sells += 1
			self.Liquidate(self.symbol)

	def OnEndOfAlgorithm(self):
		runtime_stats = {
			"Policy": self.policy_type,
			"Wash Sale Days": str(self.wash_sale_days),
			"Min Hold Days": str(self.min_hold_days),
			"Max Hold Days": str(self.max_hold_days),
			"Max Position Pct": f"{self.max_position_pct:.0%}",
			"Cash Reserve Pct": f"{self.cash_reserve_pct:.0%}",
			"Buy Decisions": str(self.decision_counts["buy"]),
			"Sell Decisions": str(self.decision_counts["sell"]),
			"Hold Decisions": str(self.decision_counts["hold"]),
			"Executed Buys": str(self.executed_buys),
			"Executed Sells": str(self.executed_sells),
			"Blocked Wash Sales": str(self.blocked_wash_sales),
			"Blocked Min Hold": str(self.blocked_min_hold),
			"Total Tax": f"${self.total_tax:,.2f}",
			"Short-Term Trades": str(self.short_term_trades),
			"Long-Term Trades": str(self.long_term_trades),
			"Tax Rate (ST)": f"{self.tax_short_rate:.0%}",
			"Tax Rate (LT)": f"{self.tax_long_rate:.0%}",
		}
		for key, value in runtime_stats.items():
			self.SetRuntimeStatistic(key, value)

		self.Log(
			"End summary | "
			f"policy={self.policy_type} buys={self.decision_counts['buy']} "
			f"sells={self.decision_counts['sell']} holds={self.decision_counts['hold']} "
			f"executed_buys={self.executed_buys} executed_sells={self.executed_sells} "
			f"blocked_wash_sales={self.blocked_wash_sales} blocked_min_hold={self.blocked_min_hold}"
		)

	def _setup_telemetry_chart(self) -> None:
		chart = Chart("Decision Telemetry")
		chart.AddSeries(Series("Score", SeriesType.Line, "value"))
		chart.AddSeries(Series("Buy Threshold", SeriesType.Line, "value"))
		chart.AddSeries(Series("Sell Threshold", SeriesType.Line, "value"))
		chart.AddSeries(Series("Raw Action", SeriesType.Line, "action"))
		chart.AddSeries(Series("Action", SeriesType.Line, "action"))
		self.AddChart(chart)

	def _plot_telemetry(self, telemetry: dict, raw_action: str, chosen_action: str) -> None:
		if telemetry.get("score") is not None:
			self.Plot("Decision Telemetry", "Score", float(telemetry["score"]))
		if telemetry.get("buy_threshold") is not None:
			self.Plot("Decision Telemetry", "Buy Threshold", float(telemetry["buy_threshold"]))
		if telemetry.get("sell_threshold") is not None:
			self.Plot("Decision Telemetry", "Sell Threshold", float(telemetry["sell_threshold"]))
		self.Plot("Decision Telemetry", "Raw Action", self._encode_action_value(raw_action))
		self.Plot("Decision Telemetry", "Action", self._encode_action_value(chosen_action))

	def _encode_action_value(self, action: str) -> int:
		return {"buy": 1, "hold": 0, "sell": -1}.get(action, 0)

	def _load_policy_metadata(self) -> dict:
		metadata_path = self.strategy_dir / f"{CONFIG['model_name']}-policy-metadata.json"
		if not metadata_path.exists():
			raise RuntimeError(
				"Policy metadata not found. Run the notebook model-training cells to export the model artifacts to the strategy directory."
			)
		with metadata_path.open() as f:
			return json.load(f)

	def _load_policy_artifacts(self) -> None:
		policy_type = self.policy_type

		if policy_type == "classification":
			trees_path = self.strategy_dir / f"{CONFIG['model_name']}-rf-trees.json"
			if not trees_path.exists():
				raise RuntimeError(
					"RF trees artifact not found. Run the notebook model-training cells to export the model artifacts to the strategy directory."
				)
			with trees_path.open() as f:
				self.trees = json.load(f)
			return

		if policy_type == "manual":
			rules_path = self.strategy_dir / f"{CONFIG['model_name']}-manual-rules.json"
			if not rules_path.exists():
				raise RuntimeError(
					"Manual rules artifact not found. Run the notebook model-training cells to export the model artifacts to the strategy directory."
				)
			with rules_path.open() as f:
				data = json.load(f)
			self.score_rules = data["score_rules"]
			self.buy_threshold = data["buy_threshold"]
			self.sell_threshold = data["sell_threshold"]
			return

		if policy_type == "qlearning":
			qtable_path = self.strategy_dir / f"{CONFIG['model_name']}-qtable.json"
			edges_path = self.strategy_dir / f"{CONFIG['model_name']}-bin-edges.json"
			if not qtable_path.exists() or not edges_path.exists():
				raise RuntimeError(
					"Q-learning artifacts not found. Run the notebook model-training cells to export the model artifacts to the strategy directory."
				)
			with qtable_path.open() as f:
				self.q_table = np.array(json.load(f))
			with edges_path.open() as f:
				self.bin_edges = {
					col: np.array(edges) for col, edges in json.load(f).items()
				}
			self.n_bins = self.policy_metadata["n_bins"]
			return

		raise RuntimeError(f"Unsupported policy type: {policy_type}")

	def _traverse_tree(self, tree: list, row: list) -> float:
		idx = 0
		while True:
			feat, split_val, left_off, right_off = tree[idx]
			if feat == -1:
				return split_val
			idx += int(left_off) if row[int(feat)] <= split_val else int(right_off)

	def _bag_predict(self, features: list) -> float:
		preds = [self._traverse_tree(tree, features) for tree in self.trees]
		return sum(preds) / len(preds)

	def _score_manual_rules(self, row: pd.Series) -> int:
		score = 0
		for rule in self.score_rules:
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

	def _encode_q_state(self, row: pd.Series, holdings: float) -> int:
		state = 0
		for col in self.feature_columns:
			bin_idx = np.digitize(row[col], self.bin_edges[col]) - 1
			bin_idx = int(np.clip(bin_idx, 0, self.n_bins - 1))
			state = state * self.n_bins + bin_idx

		holding_bucket = 2 if holdings > 0 else (0 if holdings < 0 else 1)
		return state * 3 + holding_bucket

	def _apply_trade_constraints(self, action: str) -> tuple[str, str]:
		# Max hold: force sell if position held too long
		if self.max_hold_days > 0 and self.Portfolio[self.symbol].Quantity > 0 and self.entry_time is not None:
			days_held = (self.Time.date() - self.entry_time.date()).days
			if days_held >= self.max_hold_days:
				return "sell", f"reason=max_hold days_held={days_held}"

		if action == "buy":
			if self.Portfolio[self.symbol].Quantity > 0:
				return "hold", "reason=already_long"
			if self.last_sell_time is not None:
				days_since_sell = (self.Time.date() - self.last_sell_time.date()).days
				if days_since_sell < self.wash_sale_days:
					self.blocked_wash_sales += 1
					return "hold", f"reason=wash_sale_cooldown days_since_sell={days_since_sell}"
			return action, ""

		if action == "sell":
			if self.Portfolio[self.symbol].Quantity <= 0:
				return "hold", "reason=already_flat"
			if self.entry_time is not None:
				days_held = (self.Time.date() - self.entry_time.date()).days
				if days_held < self.min_hold_days:
					self.blocked_min_hold += 1
					return "hold", f"reason=min_hold days_held={days_held}"
			return action, ""

		return action, ""

	def _build_feature_frame(self):
		history = self.History(self.symbol, 60, Resolution.Daily)
		if history.empty or len(history) < 40:
			return None

		rows = history.loc[self.symbol].copy()
		if len(rows) < 40:
			return None

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

		rows = rows.dropna(subset=self.feature_columns).copy()
		if rows.empty:
			return None

		return rows[self.feature_columns].iloc[-1:]

	def _choose_action(self, feature_frame: pd.DataFrame) -> tuple[str, str, dict]:
		row = feature_frame.iloc[0]
		if self.policy_type == "classification":
			features = row.tolist()
			score = self._bag_predict(features)
			telemetry = {
				"score": score,
				"buy_threshold": self.buy_threshold,
				"sell_threshold": self.sell_threshold,
			}
			if score > self.buy_threshold:
				return "buy", f"score={score:.4f} buy_threshold={self.buy_threshold:.4f}", telemetry
			if score < self.sell_threshold:
				return "sell", f"score={score:.4f} sell_threshold={self.sell_threshold:.4f}", telemetry
			return "hold", f"score={score:.4f} thresholds=({self.sell_threshold:.4f},{self.buy_threshold:.4f})", telemetry

		if self.policy_type == "manual":
			score = self._score_manual_rules(row)
			telemetry = {
				"score": score,
				"buy_threshold": self.buy_threshold,
				"sell_threshold": self.sell_threshold,
			}
			if score >= self.buy_threshold:
				return "buy", f"score={score} buy_threshold={self.buy_threshold}", telemetry
			if score <= self.sell_threshold:
				return "sell", f"score={score} sell_threshold={self.sell_threshold}", telemetry
			return "hold", f"score={score} thresholds=({self.sell_threshold},{self.buy_threshold})", telemetry

		if self.policy_type == "qlearning":
			holdings = self.Portfolio[self.symbol].Quantity
			state = self._encode_q_state(row, holdings)
			action_id = int(np.argmax(self.q_table[state]))
			action_name = {0: "buy", 1: "sell", 2: "hold"}[action_id]
			return action_name, f"state={state} action_id={action_id}", {"score": None, "buy_threshold": None, "sell_threshold": None}

		raise RuntimeError(f"Unsupported policy type: {self.policy_type}")
