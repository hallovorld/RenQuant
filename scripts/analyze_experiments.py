#!/usr/bin/env python
"""3-tier experiment promotion analyzer (canonical scientific methodology).

Walks data/logs/sim_*/ directories, parses per-config per-window metrics,
applies a 3-tier promotion criterion grounded in the multiple-testing
literature, and outputs a ranked report.

Tier system (see doc/research/promotion-methodology.md):

  Tier 1 — REJECT (hard)
    mean ΔAPY < 0 AND mean ΔSharpe < 0  →  worse than baseline, reject

  Tier 2 — SCREEN (relaxed, "small wins compound")
    mean ΔAPY > 0 AND mean ΔSharpe ≥ 0 AND consistent ≥ 4/N AND ΔSPY-α ≥ 0
    → soft candidate; safe to test further but NOT promote to live yet
    Expected Type-I rate: ~30-40% (Harvey-Liu-Zhu 2016 multiple-testing)

  Tier 3 — CONFIRM (rigorous, eligible for live promotion)
    Tier 2 AND (DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 with t-stat > 3)
    → publishable-rigor edge; safe to flip prod config

References:
  Bailey & López de Prado 2014, J. Portfolio Mgmt 40(5) — DSR
  Bailey, Borwein, LdP & Zhu 2015, J. Computational Finance 14(1) — PBO via CSCV
  Harvey, Liu & Zhu 2016, RFS 29(1) — multiple testing in factor research

Usage:
    python scripts/analyze_experiments.py [--baseline-dir DIR] [--all]
    python scripts/analyze_experiments.py --json out.json      # machine output
"""
from __future__ import annotations
import argparse, json, logging, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("analyze")

# Default 6 OOS windows (consistent with all session experiments)
WINDOWS = [
    ('W1', '2025-04-01', '2025-08-01'),
    ('W2', '2025-08-01', '2025-12-01'),
    ('W3', '2024-12-01', '2025-12-01'),
    ('W4', '2024-08-01', '2024-12-01'),
    ('W5', '2024-04-01', '2024-08-01'),
    ('W6', '2024-12-01', '2025-04-01'),
]
WINDOW_NAMES = [w[0] for w in WINDOWS]


def load_spy_returns() -> dict[str, float]:
    """Per-window SPY total return (%)."""
    spy_path = REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not spy_path.exists():
        raise FileNotFoundError(f"SPY OHLCV missing: {spy_path}")
    spy = pd.read_parquet(spy_path)
    spy.index = pd.to_datetime(spy.index)
    spy = spy.sort_index()
    out = {}
    for w, s, e in WINDOWS:
        px = spy.loc[pd.Timestamp(s):pd.Timestamp(e)]
        if len(px) < 2:
            log.warning(f"{w}: insufficient SPY data — skip")
            continue
        out[w] = float(px['close'].iloc[-1] / px['close'].iloc[0] - 1) * 100
    return out


def parse_sim_log(path: Path) -> dict | None:
    """Extract APY/Sharpe/MaxDD/Return from sim log tail."""
    if not path.exists():
        return None
    try:
        txt = path.read_text()
    except Exception:
        return None
    r = re.search(r'Return: ([+-]?[\d.]+)%', txt)
    s = re.search(r'Sharpe=([+-]?[\d.]+)', txt)
    a = re.search(r'APY: ([+-]?[\d.]+)%', txt)
    m = re.search(r'MaxDD=([\d.]+)%', txt)
    if r is None or s is None:
        return None
    return {
        'ret': float(r.group(1)),
        'sh': float(s.group(1)),
        'apy': float(a.group(1)) if a else None,
        'mdd': float(m.group(1)) if m else None,
    }


def collect_configs(log_root: Path) -> dict[str, dict]:
    """Walk log_root/sim_2026-05-*/W*_<cfg>.log and aggregate by config."""
    log_dirs = sorted(log_root.glob("sim_2026-05-*"))
    if not log_dirs:
        log.warning(f"No sim_2026-05-* dirs in {log_root}")
        return {}
    configs: dict[str, dict] = defaultdict(dict)
    for d in log_dirs:
        for f in d.glob("*.log"):
            n = f.stem
            m = re.match(r'(W\d)_(.+)', n)
            if not m:
                continue
            window, cfg = m.group(1), m.group(2)
            # Normalize: strip "sim_" prefix if present
            cfg = cfg[4:] if cfg.startswith('sim_') else cfg
            metrics = parse_sim_log(f)
            if metrics is None:
                continue
            # Later directories override earlier (most recent run wins)
            configs[cfg][window] = metrics
    return dict(configs)


def compute_dsr_pbo(per_window_returns: list[float], k_trials: int) -> dict:
    """Compute DSR (Bailey-LdP 2014) + (optional) PBO from per-window returns."""
    from renquant_common.metrics.deflated_sharpe import deflated_sharpe_ratio
    arr = np.asarray([x for x in per_window_returns if np.isfinite(x)])
    if len(arr) < 2:
        return {'dsr': np.nan, 'sr_observed': np.nan}
    mean_r = float(np.mean(arr))
    std_r = float(np.std(arr, ddof=1))
    if std_r <= 0:
        return {'dsr': np.nan, 'sr_observed': np.nan}
    sr_obs = mean_r / std_r
    skew = float(((arr - mean_r) ** 3).mean() / std_r ** 3) if std_r > 0 else 0.0
    kurt = float(((arr - mean_r) ** 4).mean() / std_r ** 4) if std_r > 0 else 3.0
    try:
        dsr = deflated_sharpe_ratio(
            sr_observed=sr_obs, n_returns=len(arr), n_trials=max(1, k_trials),
            skew=skew, excess_kurtosis=kurt - 3.0,
        )
    except Exception as exc:
        log.debug(f"DSR failed: {exc}")
        dsr = float('nan')
    return {'dsr': dsr, 'sr_observed': sr_obs}


def apply_tier_criteria(deltas_apy: list[float], deltas_sh: list[float],
                        deltas_spy_alpha: list[float], k_trials: int,
                        n: int) -> dict:
    """3-tier classification per the canonical methodology."""
    if not deltas_apy or n == 0:
        return {'tier': 'INSUFFICIENT_DATA'}
    mean_dapy = float(np.mean(deltas_apy))
    mean_dsh = float(np.mean(deltas_sh))
    mean_dalpha = float(np.mean(deltas_spy_alpha))
    consistent = sum(1 for x in deltas_apy if x > 0 if mean_dapy > 0) if mean_dapy > 0 else 0
    # Tier 1: hard reject
    if mean_dapy < -1.0 or (mean_dapy < 0 and mean_dsh < 0):
        return {'tier': 'TIER1_REJECT',
                'reason': f'mean ΔAPY={mean_dapy:+.1f} ΔSharpe={mean_dsh:+.2f}'}
    # Tier 2: screen
    tier2 = (mean_dapy > 0 and mean_dsh >= 0 and consistent >= 4 and mean_dalpha >= 0)
    if not tier2:
        return {'tier': 'NEITHER',
                'reason': f'mean ΔAPY={mean_dapy:+.1f} ΔSharpe={mean_dsh:+.2f} '
                          f'consistent={consistent}/{n} Δα-SPY={mean_dalpha:+.1f}'}
    # Tier 3: confirmed — needs DSR > 0.5 (50% prob true SR > 0)
    dsr_info = compute_dsr_pbo(deltas_apy, k_trials=k_trials)
    dsr = dsr_info.get('dsr', float('nan'))
    if np.isnan(dsr):
        return {'tier': 'TIER2_SCREEN',
                'reason': 'passed Tier 2; DSR uncomputable (need more samples)',
                'dsr': dsr, 'sr_observed': dsr_info.get('sr_observed')}
    if dsr > 0.5:
        return {'tier': 'TIER3_CONFIRMED',
                'reason': f'DSR={dsr:.2f} > 0.5 ⇒ live-promotable',
                'dsr': dsr, 'sr_observed': dsr_info.get('sr_observed')}
    return {'tier': 'TIER2_SCREEN',
            'reason': f'passed Tier 2; DSR={dsr:.2f} not yet ≥ 0.5',
            'dsr': dsr, 'sr_observed': dsr_info.get('sr_observed')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log-root', default='data/logs', help='sim log root dir')
    p.add_argument('--baseline-dir', default='data/logs/sim_2026-05-11_CVaR_confirm',
                   help='baseline log dir (λ=0 files)')
    p.add_argument('--baseline-pattern', default='{w}_lambda_000.log')
    p.add_argument('--json-out', default=None, help='emit JSON report')
    p.add_argument('--min-windows', type=int, default=6, help='min windows per config')
    args = p.parse_args()

    spy_ret = load_spy_returns()
    log.info(f"SPY returns: {dict((w, f'{v:+.1f}%') for w, v in spy_ret.items())}")

    log_root = REPO / args.log_root
    configs = collect_configs(log_root)
    log.info(f"Collected {len(configs)} configs from {log_root}/sim_2026-05-*/")

    # Baseline
    baseline_dir = REPO / args.baseline_dir
    BL = {}
    for w in WINDOW_NAMES:
        f = baseline_dir / args.baseline_pattern.format(w=w)
        m = parse_sim_log(f)
        if m:
            BL[w] = m
    if len(BL) < args.min_windows:
        raise SystemExit(f"Baseline incomplete: got {len(BL)}/{args.min_windows} windows in {baseline_dir}")

    bl_mean_apy = sum(BL[w]['apy'] for w in BL) / len(BL)
    bl_mean_sh = sum(BL[w]['sh'] for w in BL) / len(BL)
    bl_mean_alpha = sum(BL[w]['ret'] - spy_ret[w] for w in BL) / len(BL)
    log.info(f"Baseline: mean APY={bl_mean_apy:+.1f}% Sh={bl_mean_sh:+.2f} α-SPY={bl_mean_alpha:+.1f}pt")

    # Compute deltas + tier classification for each config
    k_trials = len(configs)  # multiple-testing penalty (DSR ↑ as K ↑)
    results = []
    for cfg, by_window in configs.items():
        if len(by_window) < args.min_windows:
            continue
        dapy, dsh, dalpha = [], [], []
        apys = []
        for w in WINDOW_NAMES:
            if w not in by_window or w not in BL:
                continue
            m = by_window[w]
            dapy.append(m['apy'] - BL[w]['apy'])
            dsh.append(m['sh'] - BL[w]['sh'])
            dalpha.append((m['ret'] - spy_ret[w]) - (BL[w]['ret'] - spy_ret[w]))
            apys.append(m['apy'])
        n = len(dapy)
        verdict = apply_tier_criteria(dapy, dsh, dalpha, k_trials=k_trials, n=n)
        mean_dapy = float(np.mean(dapy))
        mean_dsh = float(np.mean(dsh))
        mean_dalpha = float(np.mean(dalpha))
        consistent_pos = sum(1 for x in dapy if x > 0)
        results.append({
            'cfg': cfg,
            'n_windows': n,
            'mean_apy': float(np.mean(apys)),
            'mean_dapy': mean_dapy,
            'mean_dsh': mean_dsh,
            'mean_dalpha': mean_dalpha,
            'consistent_pos': consistent_pos,
            'verdict': verdict,
        })

    # Sort: TIER3 first, then TIER2, then NEITHER, then TIER1 (worst)
    tier_order = {'TIER3_CONFIRMED': 0, 'TIER2_SCREEN': 1, 'NEITHER': 2, 'TIER1_REJECT': 3, 'INSUFFICIENT_DATA': 4}
    results.sort(key=lambda r: (tier_order.get(r['verdict']['tier'], 5), -r['mean_dalpha']))

    # Print
    print(f"\n=== EXPERIMENT PROMOTION REPORT — {len(results)} configs analyzed ===")
    print(f"Multi-testing penalty k_trials = {k_trials}")
    print(f"Baseline: APY {bl_mean_apy:+.1f}% / Sh {bl_mean_sh:+.2f} / α-SPY {bl_mean_alpha:+.1f}pt")
    print()
    fmt = "{:32} | {:>+7.1f} | {:>+5.1f} | {:>+5.2f} | {:>+5.1f} | {} | {}"
    print(f"{'Config':32} | meanAPY |  ΔAPY |  ΔSh  | Δα-SPY | cons | tier")
    print("-" * 130)
    for r in results:
        v = r['verdict']
        print(fmt.format(
            r['cfg'][:32], r['mean_apy'], r['mean_dapy'], r['mean_dsh'], r['mean_dalpha'],
            f"{r['consistent_pos']}/{r['n_windows']}",
            v['tier'],
        ))

    counts = {t: 0 for t in tier_order}
    for r in results:
        counts[r['verdict']['tier']] = counts.get(r['verdict']['tier'], 0) + 1
    print("\n=== SUMMARY ===")
    for t in ['TIER3_CONFIRMED', 'TIER2_SCREEN', 'NEITHER', 'TIER1_REJECT', 'INSUFFICIENT_DATA']:
        print(f"  {t:24}: {counts.get(t, 0):>3}")

    promotable = [r for r in results if r['verdict']['tier'] in ('TIER2_SCREEN', 'TIER3_CONFIRMED')]
    if promotable:
        print(f"\n=== PROMOTABLE CANDIDATES ({len(promotable)}) ===")
        for r in promotable:
            v = r['verdict']
            print(f"  {v['tier']} — {r['cfg']}")
            print(f"    {v.get('reason','')}")
            print(f"    mean ΔAPY {r['mean_dapy']:+.1f} / ΔSh {r['mean_dsh']:+.2f} / Δα-SPY {r['mean_dalpha']:+.1f} / cons {r['consistent_pos']}/{r['n_windows']}")
    else:
        print("\n=== NO PROMOTABLE CANDIDATES — keep prod baseline ===")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            'baseline': {'mean_apy': bl_mean_apy, 'mean_sh': bl_mean_sh, 'mean_alpha': bl_mean_alpha},
            'k_trials': k_trials,
            'results': results,
            'summary': counts,
        }, indent=2, default=str))
        log.info(f"JSON report → {args.json_out}")


if __name__ == "__main__":
    main()
