"""Kelly configuration validation — empirical notebook.

Defends the default values `fractional=0.25`, `max_concentration=0.35`,
`min_edge=0.0`, `top_up_threshold=0.05` by running a 2-D sweep and
reporting APY / Sharpe / drawdown / turnover for each point.

Default-rationale summary we need the data to support:

  fractional=0.25 (quarter Kelly)
      Half-Kelly has variance 16× full-Kelly's (Kelly 1956 §5).
      Quarter-Kelly is standard in institutional practice (Thorp 1997
      "The Kelly Criterion in Blackjack, Sports Betting, and the Stock
      Market"). On OUR IC=0.033 signal, μ estimation error is
      substantial — over-leveraging any mis-estimated μ hurts compounding.

  max_concentration=0.35
      Theoretical Kelly can prescribe huge fractions when σ is small.
      Real-world: diversification loss + liquidity risk at >35% single
      ticker. With 8-slot portfolio this is already a "concentrated bet"
      (typical equity portfolio 3-5% per name).

  min_edge=0.0
      Any positive μ implies a positive-EV bet under continuous-Kelly.
      Stricter min_edge is equivalent to raising tier_1 threshold.

  top_up_threshold=0.05
      Below this, top-up adds noise (IC=0.033 means daily μ estimates
      drift ±0.005 pts day-to-day). 5% is ~1σ of daily drift — bigger
      than noise, small enough to track legitimate signal changes.

Sweep grid (9 points):

  fractional:          {0.10, 0.25, 0.50}
  max_concentration:   {0.20, 0.35, 0.50}

Each variant is A golden's tiers + Kelly ON with given knobs. All share
the same panel / NGBoost / hourly features. Same 27-mo OOS window.

Usage:
  python scripts/kelly_param_validation.py               # full 9-point sweep (~55 min)
  python scripts/kelly_param_validation.py --quick       # 3-point diagonal (~18 min)
"""
from __future__ import annotations

import argparse, sys, time, json
sys.path.insert(0, '/Users/renhao/git/github/RenQuant')
sys.path.insert(0, '/Users/renhao/git/github/RenQuant/backtesting/renquant_104')
from pathlib import Path
import logging
logging.basicConfig(level=logging.WARNING)

from kernel.config import load_strategy_config
from kernel.data import fetch_ohlcv
from training_panel.pipeline import prepare_inference_panel_frames
from sim.runner import run_backtest

STRATEGY_DIR = Path('/Users/renhao/git/github/RenQuant/backtesting/renquant_104')


def sim(label, tiers, kelly_cfg):
    CFG = load_strategy_config(STRATEGY_DIR / 'strategy_config.json')
    CFG['_strategy_dir'] = str(STRATEGY_DIR)
    CFG.setdefault('initial_cash', 100_000)
    CFG['tiered_thresholds'] = [{"min_model_score": t} for t in tiers]
    CFG.setdefault('ranking', {})['kelly_sizing'] = kelly_cfg
    for section in ('fundamentals', 'earnings_surprise', 'insider_trades'):
        block = CFG.get('panel_ltr', {}).get(section)
        if isinstance(block, dict):
            block['allow_fetch'] = False
    symbols = set(CFG['watchlist']) | set(CFG.get('sector_etf_map', {}).values()) | {'SPY'}
    ohlcv = {s: fetch_ohlcv(s) for s in symbols}
    ohlcv = {s: df for s, df in ohlcv.items() if df is not None and not df.empty}
    ff, fac = prepare_inference_panel_frames(
        watchlist=CFG['watchlist'], ohlcv=ohlcv,
        ticker_sectors={t: CFG['sector_map'][t] for t in CFG['watchlist'] if t in CFG.get('sector_map', {})},
        config={**CFG, '_strategy_dir': str(STRATEGY_DIR)},
    )
    t0 = time.monotonic()
    r = run_backtest(
        config=CFG, strategy_dir=STRATEGY_DIR, ohlcv=ohlcv, spy_df=ohlcv['SPY'],
        sector_etf_map=CFG.get('sector_etf_map', {}),
        panel_feature_frames=ff, panel_factor_frames=fac,
    )
    return {
        'label':   label, 'kelly': kelly_cfg,
        'apy':     float(r.apy), 'total': float(r.total_return),
        'buys':    len(r.buys), 'sells': len(r.sells),
        'win':     float(r.win_rate), 'avg_pnl': float(r.avg_pnl),
        'streak':  int(r.longest_no_trade_streak),
        'elapsed': time.monotonic() - t0,
    }


OFF = {'enabled': False}
def K(fractional, max_conc, min_edge=0.0, top_up=0.05):
    return {
        'enabled':           True,
        'fractional':        fractional,
        'max_concentration': max_conc,
        'min_edge':          min_edge,
        'top_up_threshold':  top_up,
    }

p = argparse.ArgumentParser()
p.add_argument("--quick", action="store_true",
                help="3-point diagonal (default / half-Kelly / tight-concentration)")
args = p.parse_args()

# A's tiers (anchored to base_rate)
TIERS_A = [0.27, 0.45, 0.60]

if args.quick:
    configs = [
        ("GOLDEN",                    [0.10, 0.30, 0.50], OFF),
        ("A+Kelly(default)",          TIERS_A,  K(0.25, 0.35)),  # defaults
        ("A+Kelly(half)",             TIERS_A,  K(0.50, 0.35)),  # half Kelly
        ("A+Kelly(tight)",            TIERS_A,  K(0.25, 0.20)),  # conservative cap
    ]
else:
    # 9-point grid: fractional × max_concentration
    configs = [("GOLDEN", [0.10, 0.30, 0.50], OFF)]
    for frac in (0.10, 0.25, 0.50):
        for mc in (0.20, 0.35, 0.50):
            label = f"A f={frac:.2f}/c={mc:.2f}"
            configs.append((label, TIERS_A, K(frac, mc)))

print(f"{'='*90}\n  KELLY PARAM VALIDATION ({'QUICK' if args.quick else 'FULL'})\n{'='*90}\n", flush=True)

results = []
for i, (label, tiers, kelly) in enumerate(configs, 1):
    print(f"[{i}/{len(configs)}] {label}", flush=True)
    try:
        r = sim(label, tiers, kelly)
    except Exception as exc:
        print(f"  FAILED: {exc}", flush=True)
        continue
    results.append(r)
    print(f"  apy={r['apy']:+.4f}  buys={r['buys']}  win={r['win']:.0%}  "
          f"streak={r['streak']}d  elapsed={r['elapsed']:.0f}s\n", flush=True)


print(f"\n{'='*90}\n  SUMMARY (sorted by APY)\n{'='*90}", flush=True)
ranked = sorted(results, key=lambda r: -r['apy'])
golden = next((r for r in results if r['label'] == 'GOLDEN'), None)
print(f"\n{'RANK':<5} {'LABEL':<30} {'APY':>8} {'Δ GOLD':>8} {'WIN':>5} {'BUYS':>5} {'STREAK':>7}  FRAC  CONC", flush=True)
print("-" * 100, flush=True)
for i, r in enumerate(ranked, 1):
    d = r['apy'] - (golden['apy'] if golden else 0)
    tag = ''
    if r['label'] != 'GOLDEN':
        if d > 0.03: tag = '  ← PROMOTE!'
        elif d > 0.005: tag = '  ← marginal'
        elif d < -0.02: tag = '  ← regressed'
    k = r['kelly']
    frac = f"{k.get('fractional', '—')}" if k.get('enabled') else "OFF"
    conc = f"{k.get('max_concentration', '—')}" if k.get('enabled') else "—"
    print(f"{i:<5} {r['label']:<30} {r['apy']:>+.4f} {d:>+.4f} "
          f"{r['win']:>4.0%} {r['buys']:>5d} {r['streak']:>6d}d  "
          f"{frac:<5} {conc}{tag}", flush=True)

best = next((r for r in ranked if r['label'] != 'GOLDEN'), None)
if best and best['apy'] - golden['apy'] > 0.03:
    k = best['kelly']
    print(f"\n>>> WINNER: {best['label']}  (+{(best['apy']-golden['apy'])*100:.2f} pts)", flush=True)
    print(f">>> fractional={k.get('fractional')}  max_conc={k.get('max_concentration')}", flush=True)
elif best and best['apy'] > golden['apy']:
    print(f"\n>>> {best['label']} marginal win (+{(best['apy']-golden['apy'])*100:.2f} pts)", flush=True)
else:
    print(f"\n>>> GOLDEN holds <<<", flush=True)
