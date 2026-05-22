#!/usr/bin/env python
"""Regime-first raw-signal baseline for renquant_104 panel scorers.

This script isolates the model signal from the production decision tree. It
uses a Qlib-style top-K fixed-hold event study:

  score all names cross-sectionally at date t
  buy the top K at the close
  hold for N trading days
  compare against SPY, the investable universe, bottom K, and controls

It deliberately bypasses QP sizing, top-ups, stop rules, soft exits, and tax
lot logic. If this raw baseline cannot beat controls per regime, the model is
not ready for more complex execution. If it can, the production pipeline is
where signal decay must be attributed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("raw-signal-baseline")


REFERENCE_NOTES = {
    "topk": (
        "Qlib TopkDropoutStrategy: rank by prediction score, hold TopK, "
        "replace low-ranked holdings with high-ranked unheld names."
    ),
    "controls": (
        "CLAUDE.md 5.2: every new number needs A/A, shuffled-label or "
        "shuffle-control, and time-shift placebo before being trusted."
    ),
    "regime": (
        "CLAUDE.md prime directive: report per-regime metrics first; pooled "
        "metrics are secondary."
    ),
}


@dataclass(frozen=True)
class XGBScorer:
    booster: Any
    feature_cols: list[str]
    feature_means: np.ndarray
    feature_stds: np.ndarray
    label_col: str | None
    lookahead_days: int | None
    trained_date: str | None


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def infer_scorer_kind(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix == ".pt":
        return "hf_patchtst"
    payload = json.loads(path.read_text())
    kind = str(payload.get("kind", "")).lower()
    if kind in {"panel_ltr_xgboost", "xgb", "xgboost"}:
        return "xgb"
    raise ValueError(f"Cannot infer scorer kind from {path}")


def load_xgb_scorer(path: Path) -> XGBScorer:
    import xgboost as xgb  # noqa: PLC0415

    payload = json.loads(path.read_text())
    booster = xgb.Booster()
    booster.load_model(bytearray(payload["booster_raw_json"].encode("utf-8")))
    return XGBScorer(
        booster=booster,
        feature_cols=list(payload["feature_cols"]),
        feature_means=np.asarray(payload["feature_means"], dtype=np.float64),
        feature_stds=np.asarray(payload["feature_stds"], dtype=np.float64),
        label_col=payload.get("label_col"),
        lookahead_days=payload.get("lookahead_days"),
        trained_date=payload.get("trained_date"),
    )


def score_xgb_frame(scorer: XGBScorer, frame: pd.DataFrame) -> pd.Series:
    import xgboost as xgb  # noqa: PLC0415

    missing = [c for c in scorer.feature_cols if c not in frame.columns]
    if missing:
        raise ValueError(f"Panel is missing {len(missing)} feature columns: {missing[:8]}")
    X = frame[scorer.feature_cols].fillna(0.0).values.astype(np.float64)
    safe_stds = np.where(scorer.feature_stds > 0, scorer.feature_stds, 1.0)
    Xn = ((X - scorer.feature_means) / safe_stds).clip(-5.0, 5.0)
    dmat = xgb.DMatrix(Xn, feature_names=scorer.feature_cols)
    return pd.Series(scorer.booster.predict(dmat), index=frame.index, name="score")


def score_xgb_panel(panel: pd.DataFrame, scorer: XGBScorer) -> pd.DataFrame:
    scored = panel[["date", "ticker"]].copy()
    scored["score"] = score_xgb_frame(scorer, panel).values
    return scored


def load_hf_patchtst_scorer(path: Path) -> Any:
    from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415

    return HFPatchTSTPanelScorer.load(path)


def score_hf_patchtst_panel(
    panel: pd.DataFrame,
    artifact_path: Path,
    eval_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    scorer = load_hf_patchtst_scorer(artifact_path)
    rows: list[pd.DataFrame] = []
    lookback_days = int((scorer.seq_len + 10) * 3)
    for idx, today in enumerate(eval_dates, start=1):
        if idx == 1 or idx % 25 == 0:
            log.info("HF PatchTST scoring date %d/%d: %s", idx, len(eval_dates), today.date())
        today_frame = panel.loc[panel["date"] == today, ["date", "ticker"]].copy()
        if today_frame.empty:
            continue
        hist = panel[
            (panel["date"] <= today)
            & (panel["date"] >= today - pd.Timedelta(days=lookback_days))
        ]
        scores = scorer.score_with_history(hist, today_frame["ticker"].tolist())
        today_frame["score"] = today_frame["ticker"].map(scores.to_dict())
        rows.append(today_frame.dropna(subset=["score"]))
    if not rows:
        return pd.DataFrame(columns=["date", "ticker", "score"])
    return pd.concat(rows, ignore_index=True)


def load_close_frame(
    tickers: list[str],
    *,
    price_root: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for ticker in sorted(set(tickers)):
        path = price_root / ticker / "1d.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if "close" not in df.columns:
            continue
        s = df["close"].copy()
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.sort_index()
        s = s[(s.index >= start) & (s.index <= end)]
        if not s.empty:
            series[ticker] = s.astype(float)
    if not series:
        raise ValueError(f"No close-price files found under {price_root}")
    prices = pd.DataFrame(series).sort_index()
    prices.index.name = "date"
    return prices


def build_eval_dates(
    panel_dates: list[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_days: int,
    rebalance_days: int,
    max_dates: int | None,
) -> list[pd.Timestamp]:
    price_set = set(pd.to_datetime(price_dates))
    eligible = [
        pd.Timestamp(d)
        for d in panel_dates
        if start <= pd.Timestamp(d) <= end and pd.Timestamp(d) in price_set
    ]
    if not eligible:
        return []
    price_pos = {pd.Timestamp(d): i for i, d in enumerate(price_dates)}
    usable = [
        d for d in eligible
        if price_pos.get(d, 10**12) + hold_days < len(price_dates)
    ]
    stepped = usable[:: max(1, rebalance_days)]
    if max_dates and len(stepped) > max_dates:
        stride = max(1, len(stepped) // max_dates)
        stepped = stepped[::stride][:max_dates]
    return stepped


def build_shift_source_map(
    eval_dates: list[pd.Timestamp],
    price_dates: pd.DatetimeIndex,
    *,
    shift_days: int,
) -> dict[pd.Timestamp, pd.Timestamp]:
    price_pos = {pd.Timestamp(d): i for i, d in enumerate(price_dates)}
    out: dict[pd.Timestamp, pd.Timestamp] = {}
    for date in eval_dates:
        pos = price_pos.get(pd.Timestamp(date))
        if pos is None:
            continue
        src_pos = pos - shift_days
        if src_pos >= 0:
            out[pd.Timestamp(date)] = pd.Timestamp(price_dates[src_pos])
    return out


def attach_regimes(score_frame: pd.DataFrame, spy_path: Path) -> pd.DataFrame:
    if not spy_path.exists():
        out = score_frame.copy()
        out["regime"] = "UNKNOWN"
        return out
    from kernel.regime_labels import compute_spy_regime_labels  # noqa: PLC0415

    regimes = compute_spy_regime_labels(spy_path)
    regimes["date"] = pd.to_datetime(regimes["date"])
    out = score_frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.merge(regimes[["date", "regime"]], on="date", how="left")
    out["regime"] = out["regime"].fillna("UNKNOWN").replace("nan_nan", "UNKNOWN")
    return out


def cross_sectional_ic_by_date(
    score_frame: pd.DataFrame,
    label_frame: pd.DataFrame,
    *,
    label_col: str,
) -> pd.DataFrame:
    merged = score_frame.merge(
        label_frame[["date", "ticker", label_col]],
        on=["date", "ticker"],
        how="inner",
    ).dropna(subset=["score", label_col])
    rows = []
    for date, grp in merged.groupby("date"):
        if len(grp) < 5:
            continue
        ic = grp["score"].corr(grp[label_col], method="spearman")
        if pd.notna(ic) and math.isfinite(float(ic)):
            rows.append({
                "date": pd.Timestamp(date),
                "ic": float(ic),
                "n": int(len(grp)),
                "regime": str(grp.get("regime", pd.Series(["UNKNOWN"])).iloc[0]),
            })
    return pd.DataFrame(rows)


def apply_score_control(
    score_frame: pd.DataFrame,
    *,
    control: str,
    seed: int,
    shift_days: int,
    eval_dates: list[pd.Timestamp] | None = None,
    shift_source_map: dict[pd.Timestamp, pd.Timestamp] | None = None,
    target_regime_map: dict[pd.Timestamp, str] | None = None,
) -> pd.DataFrame:
    out = score_frame.copy()
    eval_set = {pd.Timestamp(d) for d in (eval_dates or [])}
    if eval_set and control in {"actual", "reverse", "shuffle"}:
        out = out[out["date"].map(pd.Timestamp).isin(eval_set)].copy()
    if control == "actual":
        return out
    if control == "reverse":
        out["score"] = -out["score"].astype(float)
        return out
    if control == "shuffle":
        rng = np.random.default_rng(seed)
        pieces = []
        for _, grp in out.groupby("date", sort=False):
            g = grp.copy()
            g["score"] = rng.permutation(g["score"].values)
            pieces.append(g)
        return pd.concat(pieces, ignore_index=True)
    if control == "time_shift":
        if shift_source_map:
            pieces = []
            for target_date, source_date in shift_source_map.items():
                g = out[out["date"].map(pd.Timestamp) == pd.Timestamp(source_date)].copy()
                if g.empty:
                    continue
                g["date"] = pd.Timestamp(target_date)
                if target_regime_map:
                    g["regime"] = target_regime_map.get(pd.Timestamp(target_date), "UNKNOWN")
                pieces.append(g)
            if not pieces:
                return out.iloc[0:0].copy()
            return pd.concat(pieces, ignore_index=True)
        out = out.sort_values(["ticker", "date"]).copy()
        out["score"] = out.groupby("ticker")["score"].shift(shift_days)
        if eval_set:
            out = out[out["date"].map(pd.Timestamp).isin(eval_set)].copy()
        return out.dropna(subset=["score"]).reset_index(drop=True)
    raise ValueError(f"Unknown control: {control}")


def after_tax_return(ret: float, tax_rate: float) -> float:
    if not math.isfinite(ret):
        return float("nan")
    return ret * (1.0 - tax_rate) if ret > 0 else ret


def event_study_topk(
    score_frame: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    spy_col: str,
    hold_days: int,
    top_k: int,
    bottom_k: int,
    tax_rate: float,
) -> pd.DataFrame:
    price_dates = pd.DatetimeIndex(prices.index)
    price_pos = {pd.Timestamp(d): i for i, d in enumerate(price_dates)}
    rows: list[dict[str, Any]] = []
    grouped = score_frame.dropna(subset=["score"]).groupby("date", sort=True)
    for raw_date, grp in grouped:
        date = pd.Timestamp(raw_date)
        pos = price_pos.get(date)
        if pos is None or pos + hold_days >= len(price_dates):
            continue
        exit_date = pd.Timestamp(price_dates[pos + hold_days])
        tickers = [t for t in grp["ticker"].astype(str).tolist() if t in prices.columns]
        if len(tickers) < max(top_k, bottom_k, 2):
            continue
        entry_px = prices.loc[date, tickers]
        exit_px = prices.loc[exit_date, tickers]
        valid = entry_px.notna() & exit_px.notna() & (entry_px > 0)
        valid_tickers = [t for t in tickers if bool(valid.get(t, False))]
        if len(valid_tickers) < max(top_k, bottom_k, 2):
            continue
        g = grp[grp["ticker"].astype(str).isin(valid_tickers)].copy()
        if len(g) < max(top_k, bottom_k, 2):
            continue
        top = g.nlargest(top_k, "score")["ticker"].astype(str).tolist()
        bottom = g.nsmallest(bottom_k, "score")["ticker"].astype(str).tolist()
        universe = g["ticker"].astype(str).tolist()
        returns = (prices.loc[exit_date, universe] / prices.loc[date, universe] - 1.0).astype(float)
        top_ret = float(returns[top].mean())
        bottom_ret = float(returns[bottom].mean()) if bottom else float("nan")
        universe_ret = float(returns.mean())
        if spy_col in prices.columns:
            spy_entry = finite_float(prices.loc[date, spy_col])
            spy_exit = finite_float(prices.loc[exit_date, spy_col])
            spy_ret = spy_exit / spy_entry - 1.0 if spy_entry > 0 else float("nan")
        else:
            spy_ret = float("nan")
        regime = "UNKNOWN"
        if "regime" in g.columns and len(g):
            regime = str(g["regime"].iloc[0])
        rows.append({
            "entry_date": date.date().isoformat(),
            "exit_date": exit_date.date().isoformat(),
            "regime": regime,
            "n_universe": int(len(universe)),
            "top_tickers": top,
            "bottom_tickers": bottom,
            "top_return": top_ret,
            "top_after_tax_return": after_tax_return(top_ret, tax_rate),
            "bottom_return": bottom_ret,
            "universe_return": universe_ret,
            "spy_return": float(spy_ret),
            "alpha_vs_spy": float(top_ret - spy_ret) if math.isfinite(spy_ret) else float("nan"),
            "alpha_vs_universe": float(top_ret - universe_ret),
            "long_short": float(top_ret - bottom_ret) if math.isfinite(bottom_ret) else float("nan"),
            "top_score_mean": float(g[g["ticker"].isin(top)]["score"].mean()),
            "bottom_score_mean": float(g[g["ticker"].isin(bottom)]["score"].mean()) if bottom else float("nan"),
        })
    return pd.DataFrame(rows)


def annualized_stats(returns: pd.Series, *, period_days: int) -> dict[str, float]:
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if r.empty:
        return {
            "n": 0,
            "mean_return": float("nan"),
            "apy": float("nan"),
            "sharpe": float("nan"),
            "win_rate": float("nan"),
        }
    periods_per_year = 252.0 / float(period_days)
    gross = float(np.prod(1.0 + r.values))
    apy = gross ** (periods_per_year / float(len(r))) - 1.0 if gross > 0 else -1.0
    sd = float(r.std(ddof=1)) if len(r) > 1 else float("nan")
    sharpe = float(r.mean() / sd * math.sqrt(periods_per_year)) if sd > 0 else float("nan")
    return {
        "n": int(len(r)),
        "mean_return": float(r.mean()),
        "apy": float(apy),
        "sharpe": sharpe,
        "win_rate": float((r > 0).mean()),
    }


def summarize_events(events: pd.DataFrame, *, period_days: int) -> dict[str, Any]:
    if events.empty:
        pooled = annualized_stats(pd.Series(dtype=float), period_days=period_days)
        pooled.update({
            "after_tax_apy": float("nan"),
            "after_tax_sharpe": float("nan"),
            "mean_spy_return": float("nan"),
            "mean_universe_return": float("nan"),
            "mean_alpha_vs_spy": float("nan"),
            "mean_alpha_vs_universe": float("nan"),
            "mean_long_short": float("nan"),
        })
        return {"per_regime": {}, "pooled": pooled}
    per_regime: dict[str, Any] = {}
    for regime, grp in events.groupby("regime", dropna=False):
        stats = annualized_stats(grp["top_return"], period_days=period_days)
        tax_stats = annualized_stats(grp["top_after_tax_return"], period_days=period_days)
        stats.update({
            "after_tax_apy": tax_stats["apy"],
            "after_tax_sharpe": tax_stats["sharpe"],
            "mean_spy_return": float(grp["spy_return"].mean()),
            "mean_universe_return": float(grp["universe_return"].mean()),
            "mean_alpha_vs_spy": float(grp["alpha_vs_spy"].mean()),
            "mean_alpha_vs_universe": float(grp["alpha_vs_universe"].mean()),
            "mean_long_short": float(grp["long_short"].mean()),
        })
        per_regime[str(regime)] = stats
    pooled = annualized_stats(events["top_return"], period_days=period_days)
    pooled_tax = annualized_stats(events["top_after_tax_return"], period_days=period_days)
    pooled.update({
        "after_tax_apy": pooled_tax["apy"],
        "after_tax_sharpe": pooled_tax["sharpe"],
        "mean_spy_return": float(events["spy_return"].mean()),
        "mean_universe_return": float(events["universe_return"].mean()),
        "mean_alpha_vs_spy": float(events["alpha_vs_spy"].mean()),
        "mean_alpha_vs_universe": float(events["alpha_vs_universe"].mean()),
        "mean_long_short": float(events["long_short"].mean()),
    })
    return {"per_regime": per_regime, "pooled": pooled}


def summarize_ic(ic_df: pd.DataFrame) -> dict[str, Any]:
    if ic_df.empty:
        return {"pooled": {"n": 0, "mean_ic": float("nan")}, "per_regime": {}}
    per_regime = {}
    for regime, grp in ic_df.groupby("regime", dropna=False):
        per_regime[str(regime)] = {
            "n": int(len(grp)),
            "mean_ic": float(grp["ic"].mean()),
            "median_ic": float(grp["ic"].median()),
            "positive_ic_rate": float((grp["ic"] > 0).mean()),
        }
    return {
        "per_regime": per_regime,
        "pooled": {
            "n": int(len(ic_df)),
            "mean_ic": float(ic_df["ic"].mean()),
            "median_ic": float(ic_df["ic"].median()),
            "positive_ic_rate": float((ic_df["ic"] > 0).mean()),
        },
    }


def max_abs_return_diff(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if len(a) != len(b):
        return float("inf")
    cols = ["top_return", "alpha_vs_spy", "long_short"]
    diffs = []
    for col in cols:
        av = pd.to_numeric(a[col], errors="coerce").fillna(0.0).values
        bv = pd.to_numeric(b[col], errors="coerce").fillna(0.0).values
        diffs.append(float(np.max(np.abs(av - bv))) if len(av) else 0.0)
    return float(max(diffs or [0.0]))


def format_pct(value: Any) -> str:
    x = finite_float(value)
    return "nan" if not math.isfinite(x) else f"{x:+.2%}"


def format_num(value: Any) -> str:
    x = finite_float(value)
    return "nan" if not math.isfinite(x) else f"{x:+.3f}"


def markdown_table(summary: dict[str, Any]) -> list[str]:
    rows = [
        "| regime | n | APY | Sharpe | after-tax APY | alpha vs SPY | long-short | win rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for regime, stats in sorted(summary["per_regime"].items()):
        rows.append(
            f"| {regime} | {stats['n']} | {format_pct(stats['apy'])} | "
            f"{format_num(stats['sharpe'])} | {format_pct(stats['after_tax_apy'])} | "
            f"{format_pct(stats['mean_alpha_vs_spy'])} | "
            f"{format_pct(stats['mean_long_short'])} | "
            f"{format_pct(stats['win_rate'])} |"
        )
    p = summary["pooled"]
    rows.append(
        f"| POOLED | {p['n']} | {format_pct(p['apy'])} | {format_num(p['sharpe'])} | "
        f"{format_pct(p['after_tax_apy'])} | {format_pct(p['mean_alpha_vs_spy'])} | "
        f"{format_pct(p['mean_long_short'])} | {format_pct(p['win_rate'])} |"
    )
    return rows


def write_markdown_report(
    path: Path,
    *,
    args: argparse.Namespace,
    scorer_kind: str,
    actual_summary: dict[str, Any],
    control_summaries: dict[str, dict[str, Any]],
    ic_summary: dict[str, Any],
    aa_max_diff: float,
    n_events: int,
) -> None:
    lines = [
        "# RenQuant 104 Raw-Signal Top-K Baseline",
        "",
        f"Produced: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Purpose",
        "",
        "This evaluates the model before QP sizing, top-ups, stop-losses, "
        "soft exits, rotation, and broker/tax lot handling. It is a signal "
        "audit, not a production backtest.",
        "",
        "## Literature / Mature Scheme Anchors",
        "",
        "- Qlib TopkDropoutStrategy: cross-sectional scores choose TopK "
        "holdings and replace low-ranked names with high-ranked names.",
        "- PatchTST, Nie et al. 2023: sequence model remains shadow-only here; "
        "the same top-K evaluator can test it without changing execution.",
        "- Bailey and Lopez de Prado 2014 DSR/PBO: treat good backtests as "
        "suspect until controls reduce false-discovery risk.",
        "- cvxportfolio/Boyd transaction-cost framing: optimizer economics "
        "must include costs; this script intentionally removes optimizer "
        "effects to isolate alpha first.",
        "",
        "## Configuration",
        "",
        f"- scorer_kind: `{scorer_kind}`",
        f"- artifact: `{args.artifact}`",
        f"- panel: `{args.panel}`",
        f"- window: `{args.start}` to `{args.end}`",
        f"- top_k: `{args.top_k}`",
        f"- bottom_k: `{args.bottom_k}`",
        f"- hold_days: `{args.hold_days}`",
        f"- rebalance_days: `{args.rebalance_days}`",
        f"- short-term tax stress rate: `{args.tax_rate}`",
        f"- events evaluated: `{n_events}`",
        "",
        "## Event Geometry",
        "",
        (
            "Non-overlapping event returns: APY and Sharpe are interpretable "
            "as a coarse fixed-hold strategy proxy."
            if args.rebalance_days >= args.hold_days
            else
            "Overlapping event returns: APY and Sharpe are diagnostic only "
            "and are likely overstated versus a self-financing portfolio."
        ),
        "",
        "## Regime-First Results",
        "",
        *markdown_table(actual_summary),
        "",
        "## Controls",
        "",
        f"- A/A max absolute return diff: `{aa_max_diff:.12g}`",
    ]
    for name, summary in control_summaries.items():
        pooled = summary["pooled"]
        lines.append(
            f"- {name}: pooled APY {format_pct(pooled['apy'])}, "
            f"Sharpe {format_num(pooled['sharpe'])}, "
            f"alpha vs SPY {format_pct(pooled['mean_alpha_vs_spy'])}, "
            f"long-short {format_pct(pooled['mean_long_short'])}"
        )
    lines.extend([
        "",
        "## Cross-Sectional IC",
        "",
        f"- pooled mean IC: {format_num(ic_summary['pooled'].get('mean_ic'))}",
        f"- pooled positive IC rate: {format_pct(ic_summary['pooled'].get('positive_ic_rate'))}",
        "",
        "| regime | n days | mean IC | positive IC rate |",
        "|---|---:|---:|---:|",
    ])
    for regime, stats in sorted(ic_summary["per_regime"].items()):
        lines.append(
            f"| {regime} | {stats['n']} | {format_num(stats['mean_ic'])} | "
            f"{format_pct(stats['positive_ic_rate'])} |"
        )
    lines.extend([
        "",
        "## Interpretation Contract",
        "",
        "If actual top-K does not beat shuffle/reverse/time-shift in the "
        "regimes where it trades, the model edge is not trustworthy. If it "
        "does beat controls but production WF still loses, the loss belongs "
        "to downstream sizing, churn, exits, or tax-aware execution.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--panel", default="data/alpha158_291_fundamental_dataset.parquet")
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--kind", choices=["auto", "xgb", "hf_patchtst"], default="auto")
    parser.add_argument("--label", default="fwd_60d_excess")
    parser.add_argument("--start", default="2024-01-02")
    parser.add_argument("--end", default="2026-03-28")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--bottom-k", type=int, default=10)
    parser.add_argument("--hold-days", type=int, default=60)
    parser.add_argument("--rebalance-days", type=int, default=60)
    parser.add_argument("--tax-rate", type=float, default=0.50)
    parser.add_argument("--score-shift-days", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260522)
    parser.add_argument("--max-dates", type=int, default=None)
    parser.add_argument("--price-root", default="data/ohlcv")
    parser.add_argument("--spy", default="SPY")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    args = parser.parse_args()

    artifact_path = REPO / args.artifact
    panel_path = REPO / args.panel
    price_root = REPO / args.price_root
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    score_end = end + pd.Timedelta(days=args.hold_days * 2 + 14)

    scorer_kind = infer_scorer_kind(artifact_path, args.kind)
    log.info("Loading panel: %s", panel_path)
    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"]).dt.tz_localize(None)
    panel = panel[(panel["date"] >= start - pd.Timedelta(days=180)) & (panel["date"] <= score_end)]
    if args.label not in panel.columns:
        raise ValueError(f"Panel is missing label column {args.label}")
    log.info(
        "Panel slice: %s rows, %d tickers, %s to %s",
        f"{len(panel):,}",
        panel["ticker"].nunique(),
        panel["date"].min().date(),
        panel["date"].max().date(),
    )

    tickers = sorted(set(panel["ticker"].astype(str)) | {args.spy})
    prices = load_close_frame(tickers, price_root=price_root, start=start, end=score_end)
    if args.spy not in prices.columns:
        raise ValueError(f"{args.spy} close history is required under {price_root}")
    panel_dates = sorted(pd.Timestamp(d) for d in panel["date"].unique())
    eval_dates = build_eval_dates(
        panel_dates,
        pd.DatetimeIndex(prices.index),
        start=start,
        end=end,
        hold_days=args.hold_days,
        rebalance_days=args.rebalance_days,
        max_dates=args.max_dates,
    )
    if not eval_dates:
        raise ValueError("No eligible evaluation dates after price/hold filters")
    log.info("Eval dates: %d (%s to %s)", len(eval_dates), eval_dates[0].date(), eval_dates[-1].date())
    shift_source_map = build_shift_source_map(
        eval_dates,
        pd.DatetimeIndex(prices.index),
        shift_days=args.score_shift_days,
    )
    score_dates = sorted(set(eval_dates) | set(shift_source_map.values()))
    if len(shift_source_map) < len(eval_dates):
        log.warning(
            "Time-shift control has %d/%d source dates for shift_days=%d",
            len(shift_source_map),
            len(eval_dates),
            args.score_shift_days,
        )

    score_panel = panel[panel["date"].isin(score_dates)].copy()
    eval_panel = panel[panel["date"].isin(eval_dates)].copy()
    if scorer_kind == "xgb":
        xgb_scorer = load_xgb_scorer(artifact_path)
        score_frame = score_xgb_panel(score_panel, xgb_scorer)
        scorer_meta = {
            "kind": "xgb",
            "n_features": len(xgb_scorer.feature_cols),
            "label_col": xgb_scorer.label_col,
            "lookahead_days": xgb_scorer.lookahead_days,
            "trained_date": xgb_scorer.trained_date,
        }
    elif scorer_kind == "hf_patchtst":
        score_frame = score_hf_patchtst_panel(panel, artifact_path, score_dates)
        scorer_meta = {"kind": "hf_patchtst", "artifact": str(artifact_path)}
    else:
        raise ValueError(f"Unsupported scorer kind: {scorer_kind}")

    score_frame = attach_regimes(score_frame, price_root / args.spy / "1d.parquet")
    score_frame = score_frame.dropna(subset=["score"]).reset_index(drop=True)
    log.info("Scored rows: %s", f"{len(score_frame):,}")

    target_regime_map = {
        pd.Timestamp(row["date"]): str(row["regime"])
        for row in score_frame[score_frame["date"].map(pd.Timestamp).isin(set(eval_dates))]
        [["date", "regime"]].drop_duplicates("date").to_dict(orient="records")
    }
    eval_score_frame = score_frame[score_frame["date"].map(pd.Timestamp).isin(set(eval_dates))].copy()
    ic_df = cross_sectional_ic_by_date(eval_score_frame, eval_panel, label_col=args.label)
    ic_summary = summarize_ic(ic_df)

    events_by_control: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for control in ["actual", "shuffle", "reverse", "time_shift"]:
        controlled_scores = apply_score_control(
            score_frame,
            control=control,
            seed=args.seed,
            shift_days=args.score_shift_days,
            eval_dates=eval_dates,
            shift_source_map=shift_source_map,
            target_regime_map=target_regime_map,
        )
        events = event_study_topk(
            controlled_scores,
            prices,
            spy_col=args.spy,
            hold_days=args.hold_days,
            top_k=args.top_k,
            bottom_k=args.bottom_k,
            tax_rate=args.tax_rate,
        )
        events_by_control[control] = events
        summaries[control] = summarize_events(events, period_days=args.rebalance_days)
        log.info(
            "%s: n=%d pooled APY=%s Sharpe=%s alpha_vs_spy=%s",
            control,
            len(events),
            format_pct(summaries[control]["pooled"]["apy"]),
            format_num(summaries[control]["pooled"]["sharpe"]),
            format_pct(summaries[control]["pooled"]["mean_alpha_vs_spy"]),
        )

    aa_events = event_study_topk(
        apply_score_control(
            score_frame,
            control="actual",
            seed=args.seed,
            shift_days=args.score_shift_days,
            eval_dates=eval_dates,
            shift_source_map=shift_source_map,
            target_regime_map=target_regime_map,
        ),
        prices,
        spy_col=args.spy,
        hold_days=args.hold_days,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        tax_rate=args.tax_rate,
    )
    aa_max_diff = max_abs_return_diff(events_by_control["actual"], aa_events)
    log.info("A/A max absolute return diff: %.12g", aa_max_diff)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_json = REPO / (args.out_json or f"artifacts/diagnostics/raw_signal_baseline_{stamp}.json")
    out_md = REPO / (args.out_md or f"doc/research/raw_signal_baseline_{stamp}.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "produced_at": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "references": REFERENCE_NOTES,
        "scorer_meta": scorer_meta,
        "n_scored_rows": int(len(score_frame)),
        "n_eval_dates": int(len(eval_dates)),
        "ic_summary": ic_summary,
        "event_summary": summaries,
        "sanity": {
            "aa_max_abs_return_diff": aa_max_diff,
            "shuffle_seed": args.seed,
            "time_shift_days": args.score_shift_days,
        },
        "events": {
            name: df.to_dict(orient="records")
            for name, df in events_by_control.items()
        },
    }
    out_json.write_text(json.dumps(json_safe(payload), indent=2))
    write_markdown_report(
        out_md,
        args=args,
        scorer_kind=scorer_kind,
        actual_summary=summaries["actual"],
        control_summaries={k: v for k, v in summaries.items() if k != "actual"},
        ic_summary=ic_summary,
        aa_max_diff=aa_max_diff,
        n_events=len(events_by_control["actual"]),
    )
    log.info("Wrote JSON: %s", out_json)
    log.info("Wrote report: %s", out_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
