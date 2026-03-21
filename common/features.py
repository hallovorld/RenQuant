import pandas as pd

STATE_COLUMNS = ["macd_line", "macd_signal", "macd_hist", "rsi", "cci", "position_flag"]


def add_gate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Add buy_signal and sell_signal columns based on MACD crossover + RSI gate."""
    df = df.copy()
    df["buy_signal"] = (
        (df["macd_line"] > df["macd_signal"])
        & (df["macd_line"].shift(1) <= df["macd_signal"].shift(1))
        & (df["rsi"] > 50)
    )
    df["sell_signal"] = (
        (df["macd_line"] < df["macd_signal"])
        & (df["macd_line"].shift(1) >= df["macd_signal"].shift(1))
        & (df["rsi"] < 50)
    )
    return df


def build_transitions(
    df: pd.DataFrame,
    transaction_cost: float = 5e-4,
) -> pd.DataFrame:
    """Build state-transition records for FQI from an indicator-enriched OHLCV DataFrame."""
    df = df.copy()
    df["next_return"] = df["close"].pct_change().shift(-1)
    df = df.iloc[:-1].copy()

    records = []
    for row_number in range(len(df) - 1):
        current_row = df.iloc[row_number]
        next_row = df.iloc[row_number + 1]
        next_return = current_row["next_return"]

        for position_flag in (0, 1):
            valid_actions = ["hold"]
            if position_flag == 0 and bool(current_row["buy_signal"]):
                valid_actions.append("buy")
            if position_flag == 1 and bool(current_row["sell_signal"]):
                valid_actions.append("sell")

            for action_name in valid_actions:
                next_position_flag = position_flag
                if action_name == "buy":
                    next_position_flag = 1
                    reward = next_return - transaction_cost
                elif action_name == "hold":
                    reward = next_return if position_flag == 1 else 0.0
                else:  # sell
                    next_position_flag = 0
                    reward = -transaction_cost

                record = {
                    "state_row": row_number,
                    "date": str(df.index[row_number]),
                    "action_name": action_name,
                    "reward": reward,
                    "buy_signal": int(bool(current_row["buy_signal"])),
                    "sell_signal": int(bool(current_row["sell_signal"])),
                    "next_buy_signal": int(bool(next_row["buy_signal"])),
                    "next_sell_signal": int(bool(next_row["sell_signal"])),
                    "next_position_flag": next_position_flag,
                }
                for col in STATE_COLUMNS:
                    if col == "position_flag":
                        record[col] = position_flag
                        record[f"next_{col}"] = next_position_flag
                    else:
                        record[col] = current_row[col]
                        record[f"next_{col}"] = next_row[col]

                records.append(record)

    return pd.DataFrame(records)
