# RenQuant 104 Pipeline Decision Attribution

Date: 2026-05-22

Scope: attribute the latest 172-feature WF trade losses to concrete
pipeline steps, using the post-fix WF trade traces under:

`backtesting/renquant_104/artifacts/diagnostics/post_fix_20260522/wf_traces_172_sentiment/`

## Bottom Line

The evidence supports the hypothesis that the original cross-sectional
signal is being economically diluted by the decision pipeline. This does
not prove the raw model is strong enough to trade. It does prove that the
current tree converts a weak positive ranking signal into low-quality
orders, excessive short-term realization, and transition losses.

Key distinction:

- Raw scorer/calibrator still shows positive OOS cross-sectional signal:
  `pool_ic ~= 0.115`, `per_date_ic_mean ~= 0.115`.
- Executed portfolio WF is not acceptable:
  mean Sharpe `-1.323`, mean APY `+0.63%`, SPY mean Sharpe `+1.081`,
  0/3 cuts beat SPY Sharpe or APY.
- Trade-level entry score is nearly unable to separate winners from losers:
  losses `avg_rank=0.5878`, wins `avg_rank=0.5884`;
  losses `avg_mu=0.0190`, wins `avg_mu=0.0193`.

That means the signal is not completely absent, but the current entry,
sizing, top-up, and exit logic are not preserving enough economic edge.

## Order Source Attribution

All formal buy orders in the analyzed WF traces came from the QP/top-up
path, not from the legacy SelectionJob path.

| Order type | Count | Invested | Avg rank | Avg mu | Avg sigma |
| --- | ---: | ---: | ---: | ---: | ---: |
| QP_BUY | 109 | 631,260 | 0.5916 | 0.0208 | 0.1987 |
| TOP_UP | 46 | 261,391 | 0.5802 | 0.0156 | 0.1996 |

Pipeline ownership:

- `InferencePipeline` runs `PanelScoringJob -> PanelRankVetoJob -> RankingJob -> JointActionJob` when joint QP is enabled.
- `JointPortfolioQPTask` / `JointPortfolioQPJob` owns QP buys and QP sells.
- `TopUpHeldTask` runs after the phase-3 action jobs and emits `TOP_UP`.
- `SelectionJob` is not the owner of these buy decisions in this trace.

This matters because debugging `SelectionJob` will not explain the
orders that actually lost money here.

## Exit Owner Attribution

Closed/open round trips pooled across the three WF cuts:

| Pipeline owner | Trades | Win rate | Gross PnL | Tax | Net after tax | Avg hold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `TickerSellJob/EvaluateExitsTask: stop_loss` | 12 | 0.0% | -5,482.75 | 0.00 | -5,482.75 | 21.5d |
| `PanelRankVetoJob/CrossSectionalPanelExitTask` | 29 | 55.2% | +1,491.96 | 1,652.20 | -160.24 | 24.3d |
| open | 7 | 0.0% | 0.00 | 0.00 | 0.00 | n/a |
| `JointActionJob/JointPortfolioQPTask: QP sell/close` | 26 | 65.4% | +3,810.30 | 2,162.70 | +1,647.61 | 16.7d |
| `TickerSellJob/PanelConvictionExitTask` | 62 | 71.0% | +7,472.53 | 5,005.27 | +2,467.26 | 25.0d |
| `TickerSellJob/EvaluateExitsTask: max_hold` | 35 | 82.9% | +9,059.03 | 5,096.15 | +3,962.88 | 52.4d |

Interpretation:

1. The largest explicit loser is `EvaluateExitsTask: stop_loss`.
   It is not a tax illusion: 12/12 losers, net `-5,482.75`.
2. Cross-sectional panel exits are not catastrophic gross losers, but they
   are tax-negative after realizing short-term gains.
3. `max_hold` is the best exit class. This is a strong warning that the
   soft exit stack is often cutting positions earlier than the strategy's
   real edge horizon.
4. QP sells are net positive but short hold, tax-heavy. They need an
   explicit after-cost expected-return hurdle before rebalancing.

## Step-Level Failure Diagnosis

### 1. PanelScoringJob / ApplyGlobalCalibrationTask

File: `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py`

Observed problem:

- Raw/calibrated scorer metadata shows positive IC around `0.115`.
- Executed entry scores are tightly compressed:
  all round-trip entry ranks are mostly `0.558..0.635`.
- Winner and loser ranks are almost identical.

This means the calibrated probability is not a sufficient trade-quality
metric. It may preserve some rank order, but it does not carry enough
economic magnitude for the downstream optimizer.

Risk:

- QP treats small expected-return differences as tradable edge.
- A rank/probability value near 0.59 is not automatically an after-cost
  alpha.

### 2. VetoWeakBuysTask

File: `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py`

Observed problem:

- The analyzed WF config still used `buy_floor: adaptive_mean_std_cap`.
- With calibrated scores clustered around `0.55..0.65`, any cap at `0.30`
  becomes a no-op.
- The production config has moved toward uncapped `adaptive_mean_std`, but
  the WF config used for this trace did not match that stricter behavior.

Risk:

- Weak candidates are admitted into QP.
- The floor is expressed on a compressed probability scale, not an
  after-cost expected-return scale.

Required verification:

- Re-run the 3-cut WF with production-equivalent uncapped floor.
- Add a config-drift guard: WF acceptance configs must not silently use a
  weaker buy floor than production.

### 3. JointPortfolioQPTask / JointPortfolioQPJob

Files:

- `backtesting/renquant_104/kernel/pipeline/pp_inference.py`
- `backtesting/renquant_104/kernel/portfolio_qp/job_qp.py`
- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`

Observed problem:

- Every new buy in the trace is `source=qp`, `order_type=QP_BUY`.
- QP buys were all in `BULL_CALM`.
- Average QP buy mu was only about `0.0208`; average sigma was about
  `0.1987`.
- `rotation.joint_actions.qp_tax_aware=false`, so tax is not decision
  logic, but the realized strategy still creates short-term taxable churn.

Risk:

- The QP is mathematically optimizing a portfolio, but the input mu is too
  weak and too compressed.
- Without a hard after-cost edge hurdle, the optimizer can turn weak rank
  differences into real orders.
- Portfolio construction is not the primary alpha source; it can only
  preserve or destroy alpha. Here it is not preserving enough.

### 4. TopUpHeldTask

File: `backtesting/renquant_104/kernel/pipeline/task_topup.py`

Observed problem:

- Top-ups invested `261,391`, a large fraction of total buy notional.
- Top-ups had weaker average rank and mu than fresh QP buys:
  `avg_rank=0.5802`, `avg_mu=0.0156`.
- The top-up conviction floor of `0.20` is meaningless when all calibrated
  ranks cluster around `0.55..0.65`.

Risk:

- The pipeline can add capital to mediocre positions simply because Kelly
  target exceeds current weight.
- A top-up is economically a new buy and must pass the same after-cost
  hurdle as a fresh entry.

### 5. RegimeJob / BuyGatesJob

Files:

- `backtesting/renquant_104/kernel/pipeline/job_regime.py`
- `backtesting/renquant_104/kernel/pipeline/job_gates.py`

Observed problem:

Worst transition cells:

| Transition / owner | Trades | Net |
| --- | ---: | ---: |
| BULL_CALM -> BEAR, stop_loss | 8 | -3,054.18 |
| BULL_CALM -> CHOPPY, stop_loss | 2 | -1,147.98 |
| BULL_CALM -> BULL_CALM, stop_loss | 1 | -953.32 |

Risk:

- Entries are made under BULL_CALM assumptions, but the strategy does not
  sufficiently detect or price transition risk.
- Stop-loss then becomes the emergency exit for a regime mistake, which is
  too late and too expensive.

### 6. PanelConvictionExitTask / CrossSectionalPanelExitTask

Files:

- `backtesting/renquant_104/kernel/pipeline/task_sell.py`
- `backtesting/renquant_104/kernel/pipeline/task_panel_conviction_xs.py`

Observed problem:

- Legacy panel conviction exits are net positive but tax-heavy.
- Cross-sectional panel exits are slightly net negative after tax.
- `max_hold` performs better than soft exits.

Risk:

- Soft exits fire on score degradation before the economic edge has played
  out.
- The exit tree appears calibrated to short-horizon model noise while the
  winning trades often need longer holding time.

## Is The Decision Tree Washing Out Signal?

Yes, in the economic sense:

- The scorer has positive but weak ranking information.
- The pipeline admits too many candidates from a compressed score band.
- QP and top-up convert small score differences into orders.
- Stop-loss absorbs regime-transition mistakes.
- Soft exits realize short-term taxable gains and shorten winners.

But not proven in the model-only sense:

- We still need a clean raw-signal baseline: daily top-K raw panel score,
  equal weight, fixed 20d/40d hold, no QP, no top-up, no soft exit.
- If raw top-K beats the current tree, the decision pipeline is the
  confirmed culprit.
- If raw top-K is also weak, the model/feature stack is not strong enough.

## Required Next Experiments

Run these before changing production risk behavior again:

1. Raw signal baseline:
   - top-K by raw panel score and by calibrated expected_return
   - equal-weight
   - fixed 20d and 40d hold
   - no QP, no top-up, no soft panel exit
   - report pre-tax and after-tax Sharpe/APY vs SPY

2. Decision-tree ablation:
   - current full tree
   - no top-up
   - no soft panel exit
   - hard exits only
   - equal-weight top-K instead of QP
   - QP with after-cost mu hurdle

3. Regime transition test:
   - block or shrink BULL_CALM buys when SPY/market transition risk rises
   - specifically measure BULL_CALM -> BEAR and BULL_CALM -> CHOPPY losses

4. Config parity guard:
   - acceptance/WF config must match production decision semantics for
     buy_floor, QP mu contract, top-up floor, panel exit, and tax mode.

5. Per-order attribution telemetry:
   - every order should record the responsible job/task, pre-veto score,
     post-calibration rank, expected_return, sigma, QP target weight,
     delta weight, blocked reason, and exit owner.

## Theory / Literature Anchors

- Grinold and Kahn, Active Portfolio Management: IC must be converted into
  implementable breadth and after-cost active return; ranking IC alone is
  not a trade.
- Lo, The Statistics of Sharpe Ratios: Sharpe estimates are noisy and need
  out-of-sample, regime-aware comparison.
- Boyd and Vandenberghe, Convex Optimization: QP only optimizes the given
  objective; if mu is weak or mis-scaled, the optimizer can confidently
  select economically bad trades.
- Ledoit and Wolf, 2004 covariance shrinkage: covariance stabilization is
  necessary but cannot create alpha from weak expected returns.
- Davis and Norman transaction-cost/no-trade-band logic: rebalancing must
  overcome friction; otherwise churn consumes weak edge.

