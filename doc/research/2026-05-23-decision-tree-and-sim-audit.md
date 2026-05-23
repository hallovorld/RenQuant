# RenQuant 104 Decision Tree And Sim Audit — 2026-05-23

## Scope

This note records the clean XGB true-OOS sim rerun after the 2026-05-23
contract and sim hot-path fixes. It is intentionally blunt: the current
decision tree is cleaner, but not yet scientifically sufficient as a
profitable trading system.

Artifacts:

- Equity: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.equity.json`
- Trade report: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.report.md`
- Round trips: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.round_trips.csv`
- Raw trade log: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.trades.json`
- Score diagnostic JSON: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.json`
- Score diagnostic report: `backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.md`

Relevant fixes pushed before this rerun:

- `00ed66b` — speed up and test the 104 sim feature cache path.
- `473e781` — disable missing NGBoost sim overlays and add an artifact contract test.

## Clean XGB True-OOS Result

Window: `2024-07-02` to `2026-02-10`, 404 trading days.

| metric | XGB sim | SPY same window |
|---|---:|---:|
| Total return | +4.65% | +26.07% |
| APY | +2.88% | +15.59% |
| Sharpe | +0.39 | +0.91 |
| Max drawdown | 10.26% | 19.00% |
| Annual vol | 8.14% | 17.53% |

Interpretation: the system reduced beta and drawdown, but did not harvest
enough alpha to compensate for low exposure. The sim reports beta `+0.1377`
and alpha `+0.97%/yr`; the information ratio is negative. This is not a
production-grade result.

## Trade Forensics

Closed round trips:

- Closed gross P&L: `+$756.77`
- Event-level tax: `+$7,227.68`
- Closed net after event-level tax: `-$6,470.91`
- Closed win rate: `32.14%`
- Average closed hold: `75.6d`, median `55.5d`

Open lots at sim end:

- Open gross P&L: `+$11,128.11`
- Mean open P&L: `+15.0%`
- Open lots dominate the positive final NAV.

Tax conclusion:

- Tax is still a major reporting drag under event-level accounting.
- Annual-net tax estimate is much smaller (`~$1.96k` in the report), lifting
  annual-net APY estimate to `~6.3%`.
- Even using annual-net tax, the system remains far below SPY APY/Sharpe, so
  tax is not the only root cause.

## Decision Tree Observations

Gate and optimizer counts from the clean log:

- `404` decision days.
- Regimes: `BULL_CALM=325`, `BEAR=50`, `CHOPPY=20`, `BULL_VOLATILE=9`.
- EMA50 buy block: `40` days.
- Calibrator saturation abstain: `2` days.
- Realized-vol gate: `4,341 / 42,473` candidate checks dropped (`10.2%`).
- Adaptive weak-buy veto: `30,194 / 38,136` candidate checks dropped (`79.2%`).
- Wash-sale drops: `320`.
- QP buys logged: `40`; round-trip report counts `60` buy fills/events.
- QP soft sells suppressed by tax/horizon guards: `442`.
- Missing NGBoost artifact warnings after fix: `0`.

These numbers show a highly restrictive decision tree, but not a complete
buy-kill. It buys, but it buys from a compressed score band and carries low
market exposure.

## Main Root Cause

The most important finding is not simply "tax" or "decision tree too strict."
It is weak realized discrimination at entry.

Closed winners vs closed losers:

| group | n | mean entry rank_score | mean entry μ | mean entry σ | mean pnl_pct |
|---|---:|---:|---:|---:|---:|
| gross winners | 18 | 0.6130 | 0.0297 | 0.1950 | +13.19% |
| gross losers | 38 | 0.6131 | 0.0297 | 0.2542 | -7.24% |

The entry score and μ are almost identical for winners and losers. Sigma is
meaningfully higher for losers. That means the ranking/decision stack is not
converting risk-adjusted quality into entry selection. The decision tree is
filtering many candidates, but among admitted names the score is not
discriminative enough.

The one-off observation is now a repeatable diagnostic:

```bash
python scripts/analyze_trade_score_diagnostics.py \
  --round-trips-csv backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.round_trips.csv \
  --output-json backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.json \
  --output-md backtesting/renquant_104/artifacts/diagnostics/resim_20260523_contractfix_full/xgb_full.score_diagnostics.md
```

Closed-trade diagnostics:

| score | n | Spearman vs P&L | top-bottom P&L spread | winner mean | loser mean |
|---|---:|---:|---:|---:|---:|
| entry_rank_score | 56 | +0.0152 | -0.30% | +0.6130 | +0.6131 |
| entry_mu | 56 | +0.0152 | -0.30% | +0.0297 | +0.0297 |
| entry_sigma | 56 | -0.4043 | -13.43% | +0.1950 | +0.2542 |
| entry_mu_over_sigma | 56 | +0.2646 | +9.56% | +0.1597 | +0.1241 |
| entry_panel_score | 56 | +0.0152 | -0.30% | +0.6130 | +0.6131 |
| entry_kelly_target_pct | 36 | -0.0651 | +2.01% | +0.1165 | +0.1065 |

This is a decision-tree problem, not only a model problem: the model score
slice that actually reaches executed buys is almost flat, while the risk
dimension remains informative. The next controlled fix must therefore act
before QP solve, not only in post-hoc reporting or end-of-sim tax accounting.

## Bugs Fixed During This Pass

1. Sim feature cache duplicated SPY indicator/regime work per ticker.
   Fixed by extracting `assemble_feature_frame_from_indicators()` and
   precomputing SPY indicators/context once. Tests assert cached assembly
   matches the public `build_feature_frame()` path.

2. Alpha158 frame construction was fragmenting pandas DataFrames by repeated
   column insertion. Fixed by collecting Series and constructing the frame once.
   Tests assert full-frame alpha158 cache equals single-bar inference.

3. XGB true-OOS sim had per-regime NGBoost overlays enabled while the side
   NGBoost artifact path did not exist. Fixed by disabling those overlays in
   the sim config and adding a contract test: if any sim can activate NGBoost,
   its artifact must exist.

## Not Yet Scientifically Solved

1. Trade-level score quality is not adequate. Entry `rank_score` and μ do not
   separate winners from losers.

2. The current tax model still supports both event-level and annual-net views.
   Event-level tax is conservative but can misrepresent strategy economics
   when open gains dominate closed losses.

3. QP soft-sell suppression is very frequent (`442`), mostly because tax/horizon
   guards block sells with `expected_loss=0`. This may be correct for tax-aware
   no-trade bands, but it needs an ablation: same μ, same entries, tax soft-sell
   guard off vs on.

4. PatchTST APY/Sharpe is not scientifically available from the static seed44
   artifact for 2024/2025 historical sim. Its strict sidecar says the effective
   selection cutoff plus 60-day label horizon leaks into that period. PatchTST
   should be evaluated by strict IC now, and APY/Sharpe only through a true
   walk-forward PatchTST artifact.

## Literature Anchor

- No-trade bands and transaction-cost-aware portfolio control are grounded in
  Davis and Norman (1990), "Portfolio Selection with Transaction Costs":
  https://pubsonline.informs.org/doi/pdf/10.1287/moor.15.4.676
- Volatility-managed exposure is empirically motivated by Moreira and Muir
  (2017), "Volatility-Managed Portfolios":
  https://ideas.repec.org/a/bla/jfinan/v72y2017i4p1611-1644.html
- The literature on volatility management is mixed out of sample, so any σ
  penalty must be validated regime-by-regime, not assumed:
  https://www.sciencedirect.com/science/article/pii/S0304405X2030132X
- Alpha158-style feature construction follows Qlib benchmark conventions:
  https://github.com/microsoft/qlib/blob/main/examples/benchmarks/README.md

## Next Scientific Fix Plan

1. Add a trade-level score diagnostic test/report that computes realized IC and
   winner/loser separation for `rank_score`, μ, σ, and μ/σ on the executed
   trade set. Promotion should require positive separation, not just pooled IC.

2. Add a risk-adjusted entry option behind a flag: rank by calibrated μ/σ or
   penalize high realized σ. The current evidence supports this because losers
   have materially higher entry σ while μ/rank are indistinguishable.

3. Run a controlled ablation panel:
   - baseline clean XGB
   - volatility/risk-adjusted entry
   - QP tax soft-sell guard off
   - annual-net tax reporting only
   - combinations only after single-factor evidence is positive

4. For PatchTST, do not run static APY/Sharpe. Train or assemble true
   walk-forward PatchTST folds, then evaluate IC and trade simulation using the
   same leakage guards as XGB.
