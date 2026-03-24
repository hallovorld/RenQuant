from AlgorithmImports import *
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from config import load_strategy_config, split_date_parts


CONFIG = load_strategy_config()


class XGBoostNVDAStrategy(QCAlgorithm):
	def Initialize(self):
		start_year, start_month, start_day = split_date_parts(CONFIG["backtest_start"])
		end_year, end_month, end_day = split_date_parts(CONFIG["backtest_end"])

		self.SetStartDate(start_year, start_month, start_day)
		self.SetEndDate(end_year, end_month, end_day)
		self.SetCash(CONFIG["initial_cash"])

		self.symbol = self.AddEquity(CONFIG["stock_symbol"], Resolution.Daily).Symbol
		self.strategy_dir = Path(__file__).resolve().parent
		self.policy_metadata = self._load_policy_metadata()
		self.state_columns = self.policy_metadata["state_columns"]
		self.action_models = self._load_action_models()
		self.transaction_cost_bps = self.policy_metadata.get("transaction_cost_bps", 5)

		self.SetWarmUp(40)

	def OnData(self, data: Slice):
		if self.IsWarmingUp or not data.ContainsKey(self.symbol):
			return

		feature_frame = self._build_feature_frame()
		if feature_frame is None:
			return

		chosen_action = self._choose_action(feature_frame)
		holdings = self.Portfolio[self.symbol].Quantity

		if chosen_action == "buy" and holdings <= 0:
			self.SetHoldings(self.symbol, 1.0)
		elif chosen_action == "sell" and holdings > 0:
			self.Liquidate(self.symbol)

	def _load_policy_metadata(self) -> dict:
		metadata_path = self.strategy_dir / f"{CONFIG['model_name']}-policy-metadata.json"
		if not metadata_path.exists():
			raise RuntimeError(
				"Policy metadata not found. Run the notebook model-training cells to create the RL artifacts in the strategy directory."
			)

		with metadata_path.open() as metadata_file:
			return json.load(metadata_file)

	def _load_action_models(self) -> dict:
		action_models = {}
		for action_name in ("hold", "buy", "sell"):
			artifact_path = self.strategy_dir / f"{CONFIG['model_name']}-q-{action_name}.json"
			if not artifact_path.exists():
				continue

			model = xgb.XGBRegressor()
			model.load_model(str(artifact_path))
			action_models[action_name] = model

		if "hold" not in action_models:
			raise RuntimeError("The hold action model is required but was not found.")

		return action_models

	def _build_feature_frame(self):
		history = self.History(self.symbol, 60, Resolution.Daily)
		if history.empty or len(history) < 40:
			return None

		rows = history.loc[self.symbol].copy()
		if len(rows) < 40:
			return None

		macd_fast = rows["close"].ewm(span=12, adjust=False).mean()
		macd_slow = rows["close"].ewm(span=26, adjust=False).mean()
		rows["macd_line"] = macd_fast - macd_slow
		rows["macd_signal"] = rows["macd_line"].ewm(span=9, adjust=False).mean()
		rows["macd_hist"] = rows["macd_line"] - rows["macd_signal"]

		delta = rows["close"].diff()
		gains = delta.clip(lower=0)
		losses = -delta.clip(upper=0)
		avg_gain = gains.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
		avg_loss = losses.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
		relative_strength = avg_gain / avg_loss.replace(0, np.nan)
		rows["rsi"] = 100 - (100 / (1 + relative_strength))

		typical_price = (rows["high"] + rows["low"] + rows["close"]) / 3
		cci_mean = typical_price.rolling(20).mean()
		cci_mad = typical_price.rolling(20).apply(
			lambda values: np.mean(np.abs(values - values.mean())), raw=True
		)
		rows["cci"] = (typical_price - cci_mean) / (0.015 * cci_mad)

		rows = rows.dropna().copy()
		if len(rows) < 2:
			return None

		current_row = rows.iloc[-1]
		previous_row = rows.iloc[-2]
		position_flag = 1 if self.Portfolio[self.symbol].Quantity > 0 else 0

		feature_frame = pd.DataFrame(
			[
				{
					"macd_line": current_row["macd_line"],
					"macd_signal": current_row["macd_signal"],
					"macd_hist": current_row["macd_hist"],
					"rsi": current_row["rsi"],
					"cci": current_row["cci"],
					"position_flag": position_flag,
				}
			]
		)

		feature_frame["buy_signal"] = int(
			(current_row["macd_line"] > current_row["macd_signal"])
			and (previous_row["macd_line"] <= previous_row["macd_signal"])
			and (current_row["rsi"] > 50)
		)
		feature_frame["sell_signal"] = int(
			(current_row["macd_line"] < current_row["macd_signal"])
			and (previous_row["macd_line"] >= previous_row["macd_signal"])
			and (current_row["rsi"] < 50)
		)
		return feature_frame

	def _choose_action(self, feature_frame: pd.DataFrame) -> str:
		position_flag = int(feature_frame["position_flag"].iloc[0])
		buy_signal = int(feature_frame["buy_signal"].iloc[0])
		sell_signal = int(feature_frame["sell_signal"].iloc[0])
		state_values = feature_frame[self.state_columns]

		scores = {"hold": self.action_models["hold"].predict(state_values)[0]}
		scores["buy"] = -np.inf
		scores["sell"] = -np.inf

		if position_flag == 0 and buy_signal == 1 and "buy" in self.action_models:
			scores["buy"] = self.action_models["buy"].predict(state_values)[0]
		if position_flag == 1 and sell_signal == 1 and "sell" in self.action_models:
			scores["sell"] = self.action_models["sell"].predict(state_values)[0]

		return max(scores, key=scores.get)