"""NGBoost head — Normal(μ, σ) over raw residual forward returns.

Complements PanelLTRModel by producing **both** a location (μ) and a scale
(σ) per candidate. Downstream:

  score = μ − λ·σ          (σ-aware ranking, replaces Gaussianized LTR score
                            when ngboost.enabled is true)
  σ-multiplier             (scales max_position_pct by σ_p50 / σ_i)

Trained on **raw** residuals (`compute_residual_returns` output, not
Gaussianized), so σ is on the return scale and directly consumable for
sizing.

Persistence is a single JSON artifact. The underlying NGBRegressor is
pickled and base64-encoded into the JSON payload so the file remains
self-contained (no separate .pkl alongside) and callers don't have to
invent a second codec path.
"""
from __future__ import annotations

import base64
import json
import logging
import pickle
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ngboost import NGBRegressor
from ngboost.distns import Normal


DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.01,
    "minibatch_frac": 1.0,
    "natural_gradient": True,
    "verbose": False,
    "random_state": 17,
}


class NGBoostHead:
    """NGBoost Normal(μ, σ) regression head.

    API mirrors PanelLTRModel: train / predict / save / load, plus
    `predict_distribution` returning a two-column DataFrame.
    """

    def __init__(self, params: dict | None = None):
        self.params: dict[str, Any] = dict(DEFAULT_PARAMS)
        if params:
            self.params.update(params)
        self.regressor: NGBRegressor | None = None
        self.feature_cols: list[str] = []

    # ── Training ──────────────────────────────────────────────────────────

    def train(
        self,
        panel: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "residual_return_raw",
        sample_weight_col: str | None = "weight",
        *,
        date_col: str | None = "date",
        val_fraction: float = 0.2,
        early_stopping_rounds: int | None = None,
        impute_features: bool = True,
    ) -> dict:
        """Fit NGBRegressor(Normal) on the panel.

        Audit fixes (2026-04-25):
          N-1 / N-13 ─ drop rows with NaN/±inf in any feature column or
                       in the label. Pre-fix, these slipped through and
                       NGBoost either segfaulted or fit on garbage.
          N-22       ─ drop rows whose sample-weight is NaN/non-positive.
          N-2 / N-14 ─ time-ordered train/val split + early stopping on
                       validation NLL. Pre-fix, n_estimators=400 trained
                       to completion regardless of overfit; lr=0.01 ×
                       400 = 4.0 cumulative learning rate easily
                       overfits. Post-fix, NGBoost's native
                       `early_stopping_rounds` halts when val NLL
                       plateaus. The split is by DATE (last 20% of
                       distinct dates as val) so we don't leak future
                       observations into early-stop signal. Set
                       `early_stopping_rounds=None` to disable.
        """
        self.feature_cols = list(feature_cols)
        # Build a clean view: ALWAYS drop rows with NaN/inf in label or
        # bad weight (those can't be imputed). Features get imputed when
        # `impute_features=True` (default), otherwise dropped.
        label_arr = panel[label_col].to_numpy(dtype=float, copy=False)
        keep = np.isfinite(label_arr)
        if sample_weight_col and sample_weight_col in panel.columns:
            w_arr = panel[sample_weight_col].to_numpy(dtype=float, copy=False)
            keep = keep & np.isfinite(w_arr) & (w_arr >= 0.0)
        # Audit fix N-Coverage (2026-04-25): pre-fix, ANY NaN feature in a
        # row dropped that row. With patchy factor coverage (insider trades
        # 44/99 tickers, hourly/minute 83/99), this dropped 86.5% of the
        # panel — biased the fit toward US-domestic large caps with
        # complete coverage. Post-fix: median-fill features per column,
        # train_mu_ic + train_sigma_mean still meaningful since the
        # imputation is neutral (zero-centered features → 0 = neutral).
        if not impute_features:
            feat_arr = panel[feature_cols].to_numpy(dtype=float, copy=False)
            keep = keep & np.isfinite(feat_arr).all(axis=1)
        n_dropped = int(len(panel) - keep.sum())
        if n_dropped:
            import logging  # noqa: PLC0415
            logging.getLogger("ngboost").info(
                "NGBoostHead.train: dropped %d/%d rows (label/weight invalid)",
                n_dropped, len(panel),
            )
        sub = panel.loc[keep]
        if len(sub) < 10:
            raise ValueError(
                f"NGBoostHead.train: too few clean rows ({len(sub)} after "
                f"label/weight drop). Check feature pipeline."
            )

        # ── Audit fix N-2 / N-14: time-ordered train/val split ────────
        # Split by date (last `val_fraction` of distinct dates → val) so
        # NGBoost's early-stop signal is on truly held-out future data.
        # Falls back to row-based split when no date column or only one
        # unique date.
        train_mask = pd.Series(True, index=sub.index)
        do_eval_split = (
            early_stopping_rounds is not None
            and val_fraction is not None
            and 0.0 < val_fraction < 1.0
        )
        if do_eval_split and date_col and date_col in sub.columns:
            dates = pd.to_datetime(sub[date_col]).dt.normalize()
            uniq = np.array(sorted(dates.unique()))
            if len(uniq) >= 5:
                cutoff_idx = int(len(uniq) * (1.0 - val_fraction))
                cutoff = uniq[cutoff_idx]
                train_mask = (dates < cutoff).reindex(sub.index, fill_value=False)
            else:
                do_eval_split = False
        elif do_eval_split:
            # No date column — fall back to last N% of rows.
            n_train = int(len(sub) * (1.0 - val_fraction))
            train_mask = pd.Series(False, index=sub.index)
            train_mask.iloc[:n_train] = True

        sub_train = sub.loc[train_mask]
        sub_val   = sub.loc[~train_mask] if do_eval_split else None

        X_train = sub_train[feature_cols].to_numpy(dtype=float)
        y_train = sub_train[label_col].to_numpy(dtype=float)
        sw_train = None
        if sample_weight_col and sample_weight_col in sub_train.columns:
            sw_train = sub_train[sample_weight_col].to_numpy(dtype=float)

        X_val = y_val = sw_val = None
        if sub_val is not None and len(sub_val) >= 10:
            X_val = sub_val[feature_cols].to_numpy(dtype=float)
            y_val = sub_val[label_col].to_numpy(dtype=float)
            if sample_weight_col and sample_weight_col in sub_val.columns:
                sw_val = sub_val[sample_weight_col].to_numpy(dtype=float)

        # Audit fix N-Coverage: median-fill features. Compute medians on the
        # TRAIN split only (don't leak val into the imputed values).
        # Persist medians on the head so predict-time can use the same
        # imputation. ±inf treated as missing.
        self.feature_medians_ = None
        if impute_features:
            X_train_finite = np.where(np.isfinite(X_train), X_train, np.nan)
            medians = np.nanmedian(X_train_finite, axis=0)
            # If a column is ALL-NaN on train (rare), fall back to 0.0 so
            # the imputed value is neutral on z-scored features.
            medians = np.where(np.isfinite(medians), medians, 0.0)
            self.feature_medians_ = medians.astype(float)
            n_imputed_train = int((~np.isfinite(X_train)).sum())
            X_train = np.where(np.isfinite(X_train), X_train, medians)
            if X_val is not None:
                X_val = np.where(np.isfinite(X_val), X_val, medians)
            if n_imputed_train > 0:
                import logging  # noqa: PLC0415
                logging.getLogger("ngboost").info(
                    "NGBoostHead.train: imputed %d feature cells "
                    "(%.1f%% of %d×%d train matrix) with column medians",
                    n_imputed_train,
                    100.0 * n_imputed_train / max(1, X_train.size),
                    X_train.shape[0], X_train.shape[1],
                )

        self.regressor = NGBRegressor(Dist=Normal, **self.params)
        fit_kwargs: dict[str, Any] = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
        if X_val is not None:
            fit_kwargs["X_val"] = X_val
            fit_kwargs["Y_val"] = y_val
            if sw_val is not None:
                fit_kwargs["val_sample_weight"] = sw_val
            if early_stopping_rounds:
                fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)
        self.regressor.fit(X_train, y_train, **fit_kwargs)

        # Score predictions on the FULL clean set (train + val) so the
        # reported μ̄ / σ̄ describe the whole training distribution
        # NGBoost saw. The early-stopping signal is what changed; the
        # reporting set is unchanged for back-compat.
        X_full = sub[feature_cols].to_numpy(dtype=float)
        y_full = sub[label_col].to_numpy(dtype=float)
        preds = self.regressor.pred_dist(X_full)
        # Audit N-4 (2026-04-25): also report fit-time IC of μ̂ vs y so
        # downstream metadata captures one usable signal-quality number
        # per training run (still no CV — see N-17 for full fix).
        try:
            from scipy.stats import spearmanr  # noqa: PLC0415
            rho, _ = spearmanr(preds.loc, y_full)
            train_ic = float(rho) if rho == rho else float("nan")
        except Exception:
            train_ic = float("nan")
        # Audit N-2 / N-14: also report val IC + actual best iteration,
        # so the operator sees how early stopping kicked in.
        val_ic = float("nan")
        if X_val is not None:
            try:
                from scipy.stats import spearmanr  # noqa: PLC0415
                val_preds = self.regressor.pred_dist(X_val)
                rho_v, _ = spearmanr(val_preds.loc, y_val)
                val_ic = float(rho_v) if rho_v == rho_v else float("nan")
            except Exception:
                pass
        # NGBoost stores the actual stopped-at iteration on `best_val_loss_itr`.
        best_iter = getattr(self.regressor, "best_val_loss_itr", None)
        return {
            "n_rows": int(len(y_full)),
            "n_rows_train":   int(len(X_train)),
            "n_rows_val":     int(len(X_val)) if X_val is not None else 0,
            "n_rows_dropped": n_dropped,
            "n_features": int(len(feature_cols)),
            "train_mu_mean":    float(np.mean(preds.loc)),
            "train_sigma_mean": float(np.mean(preds.scale)),
            "train_mu_ic":      train_ic,
            "val_mu_ic":        val_ic,
            "best_iter":        int(best_iter) if best_iter is not None else None,
        }

    # ── Prediction ────────────────────────────────────────────────────────

    def predict_distribution(self, panel: pd.DataFrame) -> pd.DataFrame:
        """Return DataFrame[mu, sigma] indexed like `panel`.

        Audit fix N-5 (2026-04-25): if any input row has NaN/inf in a
        feature column, NGBoost will either error or produce garbage
        predictions. Pre-fix, that exception was swallowed by
        ApplyNGBoostTask → silent NGBoost no-op. Post-fix, NaN-row
        predictions are returned as NaN (downstream can detect + skip),
        and finite rows score normally.

        Audit fix N-Coverage (2026-04-25): if the head was trained with
        `impute_features=True` (default), apply the same median imputation
        at inference so a single missing factor (e.g. insider_net_buy_90d
        for a foreign ticker) doesn't drop the prediction to NaN. The
        medians live on `self.feature_medians_` and are persisted in the
        artifact.
        """
        if self.regressor is None:
            raise RuntimeError("NGBoostHead.predict called before train/load")
        # Audit fix NGB-NEW-1 (2026-04-26 round-3): validate column presence.
        # Pre-fix, missing columns raised cryptic pandas KeyError.
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"NGBoostHead.predict: panel missing required feature "
                f"columns: {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"(model trained on {len(self.feature_cols)} features)."
            )
        X = panel[self.feature_cols].to_numpy(dtype=float, copy=False).copy()
        medians = getattr(self, "feature_medians_", None)
        if medians is not None:
            X = np.where(np.isfinite(X), X, medians)
        finite_mask = np.isfinite(X).all(axis=1)
        out = pd.DataFrame(
            {"mu": np.nan, "sigma": np.nan},
            index=panel.index,
            dtype=float,
        )
        if finite_mask.any():
            d = self.regressor.pred_dist(X[finite_mask])
            mu_arr    = np.asarray(d.loc, dtype=float)
            sigma_arr = np.asarray(d.scale, dtype=float)
            # Audit fix NGB-OVERFLOW (2026-04-26): clamp pathological
            # sigma values that NGBoost can occasionally emit when the
            # gradient drives `scale` toward extreme magnitudes
            # (manifests as `RuntimeWarning: overflow encountered in
            # square` from ngboost.distns.normal:72 — `self.var =
            # self.scale**2` overflows float when scale > ~1e154).
            #
            # Daily-return σ of 8% ≈ 0.08 is typical; even BULL_VOLATILE
            # tails rarely exceed 0.5. A predicted σ > 5.0 is broken
            # output — likely a numeric blow-up. We clamp to [1e-6, 5.0]
            # and log if more than 1% of rows hit the ceiling, which
            # signals an upstream training/feature issue.
            SIGMA_FLOOR, SIGMA_CEIL = 1e-6, 5.0
            n_clipped_high = int(np.sum(sigma_arr > SIGMA_CEIL))
            n_clipped_low  = int(np.sum(sigma_arr < SIGMA_FLOOR))
            n_nan          = int(np.sum(~np.isfinite(sigma_arr)))
            sigma_arr = np.clip(sigma_arr, SIGMA_FLOOR, SIGMA_CEIL)
            sigma_arr = np.where(np.isfinite(sigma_arr),
                                 sigma_arr, SIGMA_CEIL)
            n_total = len(sigma_arr)
            if n_clipped_high + n_nan > max(1, n_total // 100):
                logging.getLogger("ngboost").warning(
                    "NGBoostHead.predict: %d/%d sigma clipped to ceil "
                    "(%.0e), %d to floor, %d non-finite — possible "
                    "upstream blow-up; check feature distribution",
                    n_clipped_high, n_total, SIGMA_CEIL,
                    n_clipped_low, n_nan,
                )
            # Also clamp extreme μ — same logic applies to predicted
            # mean. A daily expected return > 1.0 (100%) is broken.
            MU_CEIL = 1.0
            mu_arr = np.where(np.isfinite(mu_arr), mu_arr, np.nan)
            mu_arr = np.clip(mu_arr, -MU_CEIL, MU_CEIL)
            out.loc[panel.index[finite_mask], "mu"]    = mu_arr
            out.loc[panel.index[finite_mask], "sigma"] = sigma_arr
        return out

    def predict_mu(self, panel: pd.DataFrame) -> pd.Series:
        return self.predict_distribution(panel)["mu"].rename("mu")

    def predict_sigma(self, panel: pd.DataFrame) -> pd.Series:
        return self.predict_distribution(panel)["sigma"].rename("sigma")

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self.regressor is None:
            raise RuntimeError("NGBoostHead.save called before train")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        blob = base64.b64encode(pickle.dumps(self.regressor)).decode("ascii")
        payload: dict[str, Any] = {
            "version": 1,
            "kind": "ngboost_head",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params": self.params,
            "regressor_pickle_b64": blob,
        }
        # Audit fix N-Coverage (2026-04-25): persist median-imputation
        # vector so predict-time matches train-time imputation.
        medians = getattr(self, "feature_medians_", None)
        if medians is not None:
            payload["feature_medians"] = list(map(float, medians))
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        path.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "NGBoostHead":
        path = Path(path)
        payload = json.loads(path.read_text())
        if payload.get("kind") != "ngboost_head":
            raise ValueError(
                f"NGBoostHead.load: artifact at {path} is not an ngboost_head "
                f"(kind={payload.get('kind')!r})",
            )
        head = cls(params=payload.get("params"))
        head.feature_cols = list(payload["feature_cols"])
        head.regressor = pickle.loads(
            base64.b64decode(payload["regressor_pickle_b64"].encode("ascii")),
        )
        # Audit fix N-Coverage (2026-04-25): restore median vector.
        medians = payload.get("feature_medians")
        if medians is not None:
            head.feature_medians_ = np.asarray(medians, dtype=float)
        return head


# ── Scoring helpers ───────────────────────────────────────────────────────────

def combined_score(mu: pd.Series, sigma: pd.Series, lambda_sigma: float) -> pd.Series:
    """score = μ − λ·σ, preserving the (ticker) index of the inputs."""
    return (mu - float(lambda_sigma) * sigma).rename("ngboost_score")


def sigma_sizing_multiplier(
    sigma: pd.Series,
    *,
    floor: float = 0.3,
    ceiling: float = 1.0,
) -> pd.Series:
    """Per-row multiplier = clip(σ_median / σ_i, floor, ceiling).

    High-σ candidates get smaller allocations. The universe median sits
    at 1.0 (no change from baseline sizing). Candidates with σ ≤ median
    are capped at `ceiling` (default 1.0 — never oversize).
    """
    s = sigma.astype(float)
    med = float(s.median())
    if not np.isfinite(med) or med <= 0.0:
        return pd.Series(1.0, index=s.index, name="sigma_mult")
    mult = med / s.replace(0.0, np.nan)
    mult = mult.clip(lower=float(floor), upper=float(ceiling))
    return mult.fillna(1.0).rename("sigma_mult")
