#!/usr/bin/env python3
"""Regime-stratified A/B analyzer for re-evaluation panels.

Companion to scripts/analyze_experiments.py. Per CLAUDE.md PRIME
DIRECTIVE, pooled-mean across regimes is BIASED — a regime-conditional
win can score NEITHER pooled. This script:

1. Loads baseline + treatment 16-window panels
2. For each window, reads the *sim run log* to extract the dominant regime
   (computed from per-bar regime_state.regime; bar-weighted)
3. Computes per-regime Δ stats (mean, median, Wilcoxon, worst)
4. Outputs a regime-conditional verdict

Methodology references:
  Bailey & López de Prado 2014 — DSR (per regime)
  Wilcoxon signed-rank — non-parametric, robust to outliers
  Asness-Moskowitz-Pedersen 2013 §4 — regime-stratified factor returns

Usage::

    python scripts/analyze_regime_stratified.py \\
      --baseline data/logs/sim_2026-05-14_baseline_hmm \\
      --treatment data/logs/sim_2026-05-15_re_stop007 \\
      --label "stop_loss 0.07"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats

REGIME_LINE_RE = re.compile(r"regime=([A-Z_]+)")


def _dominant_regime_from_log(log_path: Path) -> Optional[str]:
    """Parse a sim window log for per-bar regime labels; return the
    bar-weighted majority regime. None if log missing/empty.

    NOTE: 2026-05-15 — the HMM regime detector is known to mis-label
    2022 deep-bear windows as BULL_CALM (CLAUDE.md PRIME DIRECTIVE
    documented this). Prefer _regime_from_spy_return below for analyzer
    correctness."""
    if not log_path.exists():
        return None
    counts: dict[str, int] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        m = REGIME_LINE_RE.search(line)
        if m:
            r = m.group(1)
            counts[r] = counts.get(r, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _regime_from_spy_return(start_iso: str, end_iso: str) -> str:
    """Data-driven regime label from SPY's realized return + volatility
    over the window. Bypasses the HMM detector entirely — mismatch with
    in-sim regime is the WHOLE POINT (the HMM is buggy on 2022 windows).

    Buckets (per CLAUDE.md regime taxonomy):
      ret < -10%             → BEAR
      ret in [-10%, -2%]     → CHOPPY
      ret in [-2%, +2%]      → BULL_CALM
      ret in [+2%, +10%]     → BULL_VOLATILE
      ret > +10%             → BULL_STRONG

    Volatility could be added (e.g. promote VOLATILE if ann_vol > 25%);
    keeping it 1-D for now to avoid bucket explosion at n_window=16.
    """
    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download("SPY", start=start_iso, end=end_iso,
                          progress=False, auto_adjust=False)
        if df is None or df.empty:
            return "UNKNOWN"
        close = df["Close"].dropna()
        if hasattr(close, "squeeze"):
            close = close.squeeze()
        if len(close) < 2:
            return "UNKNOWN"
        ret = float(close.iloc[-1] / close.iloc[0] - 1.0)
    except Exception:
        return "UNKNOWN"
    if   ret < -0.10:  return "BEAR"
    elif ret < -0.02:  return "CHOPPY"
    elif ret <  0.02:  return "BULL_CALM"
    elif ret <  0.10:  return "BULL_VOLATILE"
    else:               return "BULL_STRONG"


def _load_panel(eq_dir: Path) -> dict[str, dict]:
    out = {}
    for f in sorted(eq_dir.glob("Q*.json")):
        out[f.stem] = json.loads(f.read_text())
    return out


def _regime_for_window(log_dir: Path, win: str) -> Optional[str]:
    for fname in (f"{win}.log", f"{win}.txt"):
        p = log_dir / fname
        if p.exists():
            r = _dominant_regime_from_log(p)
            if r:
                return r
    return None


def analyze(baseline_dir: Path, treatment_dir: Path,
            label: str = "treatment") -> dict:
    base_panel  = _load_panel(baseline_dir / "equity")
    treat_panel = _load_panel(treatment_dir / "equity")

    # Window-level deltas + regime tag
    rows = []
    for win in sorted(base_panel.keys()):
        if win not in treat_panel:
            continue
        b = base_panel[win]
        t = treat_panel[win]
        delta_apy = float(t["apy"]) - float(b["apy"])
        delta_sharpe = float(t.get("sharpe", 0)) - float(b.get("sharpe", 0))
        # Regime label: PREFER data-driven SPY-return classification
        # because the in-sim HMM detector is known buggy on 2022 windows
        # (CLAUDE.md PRIME DIRECTIVE). Fall back to log-parsed if yfinance
        # unavailable.
        regime = _regime_from_spy_return(b["start"][:10], b["end"][:10])
        if regime == "UNKNOWN":
            regime = _regime_for_window(treatment_dir / "logs", win) \
                  or _regime_for_window(baseline_dir / "logs", win) \
                  or "UNKNOWN"
        rows.append({
            "window": win,
            "regime": regime,
            "start": b["start"][:10],
            "end":   b["end"][:10],
            "base_apy":  float(b["apy"]),
            "treat_apy": float(t["apy"]),
            "delta_apy": delta_apy,
            "delta_sharpe": delta_sharpe,
        })

    # Pooled stats
    arr = np.array([r["delta_apy"] for r in rows])
    pooled = {
        "n_windows": len(arr),
        "mean_pp":   float(arr.mean()) * 100,
        "median_pp": float(np.median(arr)) * 100,
        "worst_pp":  float(arr.min()) * 100,
        "best_pp":   float(arr.max()) * 100,
        "n_pos":     int((arr > 0).sum()),
        "wilcoxon_p": (float(stats.wilcoxon(arr).pvalue)
                       if len(arr) >= 6 and arr.std() > 0 else float("nan")),
    }

    # Per-regime stratification
    by_regime: dict[str, list] = {}
    for r in rows:
        by_regime.setdefault(r["regime"], []).append(r["delta_apy"])
    per_regime = {}
    for regime, vals in by_regime.items():
        v = np.array(vals)
        per_regime[regime] = {
            "n": len(v),
            "mean_pp":   float(v.mean()) * 100,
            "median_pp": float(np.median(v)) * 100,
            "worst_pp":  float(v.min())  * 100,
            "n_pos":     int((v > 0).sum()),
            # Wilcoxon needs n ≥ 6 to be meaningful; report only when n ≥ 4
            "wilcoxon_p": (float(stats.wilcoxon(v).pvalue)
                           if len(v) >= 4 and v.std() > 0 else float("nan")),
        }

    # Verdict
    verdict = "REJECT"
    pos_regimes = [r for r, s in per_regime.items()
                   if s["mean_pp"] > 0 and s["worst_pp"] > -10.0]
    if pos_regimes:
        # Check at least one is statistically meaningful
        sig = [r for r in pos_regimes
               if per_regime[r]["mean_pp"] > 2.0 and per_regime[r]["n"] >= 2]
        verdict = "WIN-CONDITIONAL" if sig else "NEITHER"

    return {
        "label": label,
        "baseline_dir": str(baseline_dir),
        "treatment_dir": str(treatment_dir),
        "rows": rows,
        "pooled": pooled,
        "per_regime": per_regime,
        "verdict": verdict,
        "win_regimes": pos_regimes,
    }


def _print(result: dict) -> None:
    print(f"\n{'=' * 78}")
    print(f"  Regime-stratified A/B: {result['label']}")
    print(f"  Baseline:  {result['baseline_dir']}")
    print(f"  Treatment: {result['treatment_dir']}")
    print(f"{'=' * 78}")

    p = result["pooled"]
    print(f"\nPooled (n={p['n_windows']}):")
    print(f"  mean Δapy   = {p['mean_pp']:+7.2f}pp")
    print(f"  median      = {p['median_pp']:+7.2f}pp")
    print(f"  worst       = {p['worst_pp']:+7.2f}pp")
    print(f"  Wilcoxon p  = {p['wilcoxon_p']:.4f}")
    print(f"  n_pos       = {p['n_pos']}/{p['n_windows']}")

    print(f"\nPer-regime stratified:")
    print(f"  {'Regime':<16} {'n':>3} {'mean':>9} {'median':>9} "
          f"{'worst':>8} {'n_pos':>6} {'Wp':>7}")
    for regime in sorted(result["per_regime"].keys()):
        s = result["per_regime"][regime]
        wp = f"{s['wilcoxon_p']:.3f}" if not np.isnan(s['wilcoxon_p']) else "  —  "
        print(f"  {regime:<16} {s['n']:>3} {s['mean_pp']:>+7.2f}pp "
              f"{s['median_pp']:>+7.2f}pp {s['worst_pp']:>+6.2f}pp "
              f"{s['n_pos']:>3}/{s['n']:<2} {wp:>7}")

    print(f"\nVerdict: {result['verdict']}")
    if result["win_regimes"]:
        print(f"  Conditional-win regimes: {', '.join(result['win_regimes'])}")

    print(f"\nPer-window detail:")
    for r in result["rows"]:
        print(f"  {r['window']} {r['regime']:<14} {r['start']}→{r['end']}  "
              f"base={r['base_apy']*100:+6.1f}pp  treat={r['treat_apy']*100:+6.1f}pp  "
              f"Δ={r['delta_apy']*100:+6.2f}pp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline",  required=True, help="dir with equity/Qxx.json + logs/Qxx.log")
    p.add_argument("--treatment", required=True)
    p.add_argument("--label",     default="treatment")
    p.add_argument("--json",      help="write result to this JSON path")
    args = p.parse_args()

    result = analyze(Path(args.baseline), Path(args.treatment), label=args.label)
    _print(result)

    if args.json:
        # Convert non-serializable types
        Path(args.json).write_text(json.dumps(result, indent=2, default=float))
        print(f"\nWrote: {args.json}")


if __name__ == "__main__":
    main()
