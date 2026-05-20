# Sim A/B Results — 2026-04-26 (5th attempt, real numbers)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Window**: 2024-01-01 → 2026-04-26 (27 months OOS)
**Universe**: 100/101 tickers (with fundamentals + insider + hourly bars)

## Results

| Variant | APY | Sharpe | Max DD | Trades | Win | Final |
|---|---:|---:|---:|---:|---:|---:|
| baseline (v4.1 OLD)        | **+0.00%** ⚠️ | +0.000 | 0.00% | 0     | 0%   | $100k |
| **gate-b@0.10 (production)** | **+26.91%** | **+1.474** | 11.07% | 1316  | 78%  | $169.8k |
| gate-b@0.20 (conservative) | +17.08% | +1.413 | 10.07% | 500   | 83%  | $141.9k |
| qp solver alone            | +26.91% | +1.474 | 11.07% | 1316  | 78%  | $169.8k |

## Findings

### ⚠️ baseline = 0 trades — broken comparison

`baseline=True` produced 0 trades over 27 months. Log shows
`NoCandidateAlert: 560 consecutive days with zero candidates`.
Cause: my `--baseline` override (`solver=greedy` + `quality_floor=False`)
is interacting badly with the disk config's panel_scoring settings
that have been tuned for QP+gates. **NOT a true v4.1 reference**
— this version of "OLD greedy" is broken in unexpected ways.

True v4.1 reproduction needs ALL of golden_config_2026-04-23 reverted
(more extensive than just flipping solver+gate flags). Defer to a
later commit that uses `git checkout strategy_config.golden.json@v4.1`
as the actual baseline.

### qp solver alone == gate-b@0.10

These produced identical numbers because `--qp-solver` doesn't
explicitly disable Gate B → disk config's Gate B at 0.10 is
preserved → equivalent to production. To get pure-QP verdict, need
`--qp-solver --no-gate-b` flag (not yet implemented).

### Real conclusion: gate-b 0.10 vs gate-b 0.20

Both have positive APY + ≥1.4 Sharpe. **0.10 wins on APY (+9.83 pt)**
**0.20 wins on win-rate (+5pt) and DD (−1pt)**.

| Metric | 0.10 | 0.20 | Winner |
|---|---|---|---|
| APY | 26.91% | 17.08% | 0.10 (+9.83 pt) |
| Sharpe | 1.474 | 1.413 | 0.10 (+0.061) |
| Max DD | 11.07% | 10.07% | 0.20 (−1.00 pt) |
| Win rate | 77.76% | 82.58% | 0.20 (+4.82 pt) |
| Trades | 1316 | 500 | (0.20 trades 60% less) |

**0.20 = "fewer but better trades"**, **0.10 = "more total volume"**.

## Recommendation

**Keep production at 0.10**. APY + Sharpe both higher than 0.20.
The DD and win-rate advantages of 0.20 are smaller than the APY hit.

**Consider 0.15 as a tune target** — likely captures most of 0.10's
APY with some of 0.20's win-rate improvement.

## Compared to claimed golden v4.1 (+39.82%)

Today's production (gate-b @ 0.10): +26.91% APY.
Golden v4.1 claim: +39.82% APY.

Likely explanation:
- Different OOS window (golden = 2024-01-01 to 2026-04-23, ours = 2024-01-01 to 2026-04-26 → +3 days, similar)
- Different artifact set (we ran with sim's cached artifacts; golden was fresh-trained)
- Different cv splits if calibrators were re-fit

**Action**: do NOT roll back from 0.10. The 26.91 vs 39.82 gap is
likely artifact freshness (today's calibrator pool_ic = 0.001 — see
`doc/components/calibration-saturation.md`). Calibrator quality is
the bottleneck, not the gate threshold.

## Next experiments (queued)

- gate-b @ 0.15 (tune sweep)
- qp + Gate B at 0.10 + qp_signal_decay=0.5 (Stage 2 partial-move)
- qp + Gate B at 0.10 + qp_robust_kappa=0.5 (Stage 5 robust μ)
- qp + Gate B at 0.10 + qp_cvar_lambda=1.0 (Stage 7 CVaR)

Each ~30 min runtime. Would queue in a Sunday-batch sweep tomorrow.
