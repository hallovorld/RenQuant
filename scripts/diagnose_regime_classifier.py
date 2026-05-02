#!/usr/bin/env python
"""Diagnose the renquant_104 regime classifier on SPY history.

Per roadmap §K (P2): the regime classifier (Hurst + CUSUM + GMM) may
mis-fire on sideways markets. CHOPPY is the regime most often mis-
classified — it sits between MOMENTUM and REVERSION on the Hurst axis
and overlaps with GMM's BULL_VOLATILE cluster.

What this script does
---------------------
1. Loads SPY OHLCV from local cache.
2. Walks history bar-by-bar, calling ``detect_regime`` to label each date.
3. Computes a simple "ground-truth-ish" label using forward 60-day
   return + realized vol — not a perfect oracle, but enough to surface
   systematic disagreement.
4. Outputs:
   - Regime distribution (% of days in each regime)
   - Transition matrix (P(regime_t+1 | regime_t))
   - Confusion matrix (classifier vs ground-truth heuristic)
   - Sample of mis-classified dates with context

Production safety: read-only on SPY cache + RegimeState (constructed
locally per call). Never touches live state files.

Usage::

    python scripts/diagnose_regime_classifier.py
    python scripts/diagnose_regime_classifier.py --since 2024-01-01

Saves report to ``data/audit/regime_diagnosis_<run>.json``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("regime-diag")


def _load_spy() -> pd.DataFrame:
    path = REPO_ROOT / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not path.exists():
        log.error("SPY cache missing: %s", path); sys.exit(1)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def _load_config() -> dict:
    cfg_path = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
    return json.loads(cfg_path.read_text())


def _ground_truth_regime(
    fwd_60d_return: float, fwd_60d_realized_vol: float,
    bull_thresh: float = 0.05, bear_thresh: float = -0.05,
    vol_thresh: float = 0.35,
) -> str:
    """Heuristic regime label based on FUTURE 60-day return + vol.

    Imperfect oracle — uses lookahead. Surfaces systematic disagreement
    between the classifier and "what actually happened" rather than
    declaring ground truth.
    """
    if fwd_60d_realized_vol > vol_thresh or fwd_60d_return < bear_thresh:
        return "BEAR"
    if fwd_60d_return > bull_thresh:
        if fwd_60d_realized_vol > 0.20:
            return "BULL_VOLATILE"
        return "BULL_CALM"
    # Range-bound: small absolute return, not vol-driven
    return "CHOPPY"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--since", default="2020-01-01",
                   help="Diagnose from this date onward (default 2020-01-01).")
    p.add_argument("--out", default=None,
                   help="JSON output (default data/audit/regime_diagnosis_<run>.json)")
    args = p.parse_args()

    from kernel.regime import RegimeState, detect_regime, load_gmm_artifact   # noqa: PLC0415
    config = _load_config()

    spy = _load_spy()
    if "close" not in spy.columns:
        log.error("SPY parquet missing 'close' column")
        return 1
    since_ts = pd.Timestamp(args.since)
    spy = spy.loc[spy.index >= since_ts]
    if len(spy) < 100:
        log.error("Not enough SPY history after %s (got %d days)", args.since, len(spy))
        return 1

    closes = spy["close"].astype(float)
    returns = closes.pct_change().fillna(0.0)
    log.info("Diagnosing %d trading days from %s to %s",
             len(spy), spy.index.min().date(), spy.index.max().date())

    # Load GMM artifact (if exists)
    gmm_path = REPO_ROOT / "backtesting" / "renquant_104" / "artifacts" / "spy-gmm-regime.json"
    gmm = load_gmm_artifact(gmm_path) if gmm_path.exists() else None
    log.info("GMM artifact: %s", "loaded" if gmm else "missing → Hurst-only fallback")

    state = RegimeState()
    classifier_labels: list[str] = []
    confidences: list[float] = []
    transitions = []   # (prev, new) list

    for i in range(len(returns)):
        if i < 30:
            classifier_labels.append("(insufficient_data)")
            confidences.append(float("nan"))
            continue
        window = returns.iloc[: i + 1].values
        spy_window = spy.iloc[: i + 1]
        prev = state.regime
        state = detect_regime(window, spy_window, gmm, state, config)
        classifier_labels.append(state.regime)
        confidences.append(state.confidence)
        if state.regime != prev:
            transitions.append((prev, state.regime, str(spy.index[i].date())))

    # Build ground-truth labels via 60d-forward return / vol
    fwd_60d_return: list[float | None] = []
    fwd_60d_vol:    list[float | None] = []
    for i in range(len(returns)):
        if i + 60 >= len(returns):
            fwd_60d_return.append(None)
            fwd_60d_vol.append(None)
            continue
        fwd_window = returns.iloc[i + 1 : i + 61].values
        fwd_ret = float(np.prod(1.0 + fwd_window) - 1.0)
        fwd_vol = float(np.std(fwd_window, ddof=1) * np.sqrt(252))
        fwd_60d_return.append(fwd_ret)
        fwd_60d_vol.append(fwd_vol)

    ground_labels: list[str] = []
    for ret, vol in zip(fwd_60d_return, fwd_60d_vol):
        if ret is None:
            ground_labels.append("(no_fwd_data)")
        else:
            ground_labels.append(_ground_truth_regime(ret, vol))

    # Distribution
    cls_dist = Counter(l for l in classifier_labels if not l.startswith("("))
    gt_dist  = Counter(l for l in ground_labels    if not l.startswith("("))
    n_total = sum(cls_dist.values())

    # Transition matrix
    trans_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for prev, new, _date in transitions:
        trans_matrix[prev][new] += 1

    # Confusion matrix (classifier vs ground truth)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    n_compared = 0
    n_agreed = 0
    for cls, gt in zip(classifier_labels, ground_labels):
        if cls.startswith("(") or gt.startswith("("):
            continue
        confusion[cls][gt] += 1
        n_compared += 1
        if cls == gt:
            n_agreed += 1
    overall_accuracy = n_agreed / max(1, n_compared)

    # Sample mis-classified dates by class (most surprising first)
    mis_samples: dict[str, list[dict]] = defaultdict(list)
    for i, (cls, gt) in enumerate(zip(classifier_labels, ground_labels)):
        if cls.startswith("(") or gt.startswith("(") or cls == gt:
            continue
        if len(mis_samples[f"{cls}→{gt}"]) >= 5:
            continue
        mis_samples[f"{cls}→{gt}"].append({
            "date": str(spy.index[i].date()),
            "classifier": cls,
            "ground_truth": gt,
            "confidence": confidences[i] if not np.isnan(confidences[i]) else None,
            "fwd_60d_return": fwd_60d_return[i],
            "fwd_60d_vol": fwd_60d_vol[i],
        })

    # CHOPPY-specific stats
    choppy_count = cls_dist.get("CHOPPY", 0)
    choppy_durations: list[int] = []
    cur_run = 0
    for label in classifier_labels:
        if label == "CHOPPY":
            cur_run += 1
        else:
            if cur_run > 0:
                choppy_durations.append(cur_run)
            cur_run = 0
    if cur_run > 0:
        choppy_durations.append(cur_run)

    report = {
        "generated_at":    _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "since":           args.since,
        "n_days":          n_total,
        "classifier_dist": dict(cls_dist),
        "classifier_pct":  {k: round(v / n_total * 100, 1) for k, v in cls_dist.items()},
        "ground_dist":     dict(gt_dist),
        "ground_pct":      {k: round(v / max(1, sum(gt_dist.values())) * 100, 1)
                            for k, v in gt_dist.items()},
        "transitions":     {k: dict(v) for k, v in trans_matrix.items()},
        "confusion":       {k: dict(v) for k, v in confusion.items()},
        "overall_accuracy": round(overall_accuracy, 3),
        "choppy_stats": {
            "n_days":         choppy_count,
            "n_episodes":     len(choppy_durations),
            "avg_duration":   round(np.mean(choppy_durations), 1) if choppy_durations else 0,
            "max_duration":   max(choppy_durations) if choppy_durations else 0,
            "median_duration": int(np.median(choppy_durations)) if choppy_durations else 0,
        },
        "mis_classification_samples": dict(mis_samples),
    }

    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "audit"
        / f"regime_diagnosis_{_dt.datetime.now(_dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))

    print()
    print("=" * 70)
    print(f"  REGIME DIAGNOSIS — {args.since} → present  ({n_total} trading days)")
    print("=" * 70)
    print(f"  Classifier distribution:")
    for k, v in sorted(cls_dist.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<16} {v:>5}  ({v/n_total*100:>5.1f}%)")
    print()
    print(f"  Ground-truth (60d-fwd heuristic) distribution:")
    n_gt = sum(gt_dist.values())
    for k, v in sorted(gt_dist.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<16} {v:>5}  ({v/max(1,n_gt)*100:>5.1f}%)")
    print()
    print(f"  Overall classifier vs ground-truth agreement: {overall_accuracy:.1%}")
    print()
    print(f"  CHOPPY stats:")
    print(f"    days       {choppy_count:>5}  ({choppy_count/n_total*100:.1f}%)")
    print(f"    episodes   {len(choppy_durations):>5}")
    print(f"    avg dur    {report['choppy_stats']['avg_duration']:>5}d")
    print(f"    max dur    {report['choppy_stats']['max_duration']:>5}d")
    print()
    print(f"  Top mis-classifications (classifier→ground_truth):")
    counts_by_pair: dict[str, int] = {
        k: sum(v) for k, v in
        ((kk, [confusion[c][gt] for c, gt in [kk.split("→")]]) for kk in mis_samples)
    }
    for pair, samples in sorted(mis_samples.items(),
                                  key=lambda kv: -counts_by_pair.get(kv[0], 0))[:5]:
        n = sum(confusion[pair.split("→")[0]][pair.split("→")[1]] for _ in [0])
        print(f"    {pair:<26} {n:>4} days")
    print()
    print(f"  Report: {out_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
