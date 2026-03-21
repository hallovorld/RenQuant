import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from .features import STATE_COLUMNS

_DEFAULT_XGB_PARAMS = dict(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    objective="reg:squarederror",
)


def score_valid_actions(
    feature_frame: pd.DataFrame,
    buy_signal,
    sell_signal,
    position_flag,
    models: dict,
) -> pd.DataFrame:
    """Score all valid actions for each state. Invalid actions receive -inf."""
    zero = np.zeros(len(feature_frame))
    scores = pd.DataFrame(index=feature_frame.index)

    hold_model = models.get("hold")
    buy_model = models.get("buy")
    sell_model = models.get("sell")

    scores["hold"] = zero if hold_model is None else hold_model.predict(feature_frame)
    buy_scores = zero if buy_model is None else buy_model.predict(feature_frame)
    sell_scores = zero if sell_model is None else sell_model.predict(feature_frame)

    scores["buy"] = np.where((position_flag == 0) & (buy_signal == 1), buy_scores, -np.inf)
    scores["sell"] = np.where((position_flag == 1) & (sell_signal == 1), sell_scores, -np.inf)
    return scores


def fitted_q_iteration(
    transitions: pd.DataFrame,
    n_iter: int = 8,
    gamma: float = 0.95,
    state_columns: list[str] | None = None,
    xgb_params: dict | None = None,
) -> dict:
    """Run Fitted Q-Iteration and return a dict of trained XGBRegressor models."""
    if state_columns is None:
        state_columns = STATE_COLUMNS
    if xgb_params is None:
        xgb_params = _DEFAULT_XGB_PARAMS

    action_names = ["hold", "buy", "sell"]
    models = {name: None for name in action_names}
    targets = transitions["reward"].copy()

    for _ in range(n_iter):
        next_features = transitions[[f"next_{c}" for c in state_columns]].copy()
        next_features.columns = state_columns

        next_scores = score_valid_actions(
            next_features,
            transitions["next_buy_signal"],
            transitions["next_sell_signal"],
            transitions["next_position_flag"],
            models,
        )
        max_next_q = next_scores.max(axis=1).replace(-np.inf, 0.0)
        targets = transitions["reward"] + gamma * max_next_q

        for action_name in action_names:
            action_slice = transitions[transitions["action_name"] == action_name]
            if action_slice.empty:
                continue
            model = XGBRegressor(**xgb_params)
            model.fit(action_slice[state_columns], targets.loc[action_slice.index])
            models[action_name] = model

    return models
