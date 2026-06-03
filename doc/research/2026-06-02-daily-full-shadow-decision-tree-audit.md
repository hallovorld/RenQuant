# Daily full and shadow decision-tree audit: 2026-06-02

**Date**: 2026-06-02
**Status**: RFC for Claude review
**Scope**: Primary full run `2026-06-02-live-237c90a5` and shadow full run `2026-06-02-live-339a8f76`
**Code status**: Analysis and modification plan only. No runtime/config change is made by this document.

Related documents:

- [`2026-06-02-bull-calm-no-signal-diagnostic.md`](./2026-06-02-bull-calm-no-signal-diagnostic.md)
- [`2026-06-02-bull-calm-signal-recovery-plan.md`](./2026-06-02-bull-calm-signal-recovery-plan.md)
- [`2026-06-02-qp-the-3-questions-addendum.md`](./2026-06-02-qp-the-3-questions-addendum.md)
- [`2026-06-02-qp-architecture-review-and-alternatives.md`](./2026-06-02-qp-architecture-review-and-alternatives.md)
- [`../arch/multirepo-sop.md`](../arch/multirepo-sop.md)
- [`../arch/subrepo-operating-model.md`](../arch/subrepo-operating-model.md)

## TL;DR

Three operator-actionable findings on 2026-06-02:

1. Primary buy silence is correct: BULL_CALM admission rejected all scored buy candidates because the artifact has known-weak BULL_CALM evidence. The historical fixture value is 74 blocked candidates.
2. Primary failed to trim overweight ORCL because cap-compliance fallback was disabled in prod policy. This is a config/safety-path miss, not a model miss.
3. The day had live sell-pending activity for META/GILD that the run bundle did not attribute cleanly. "Quiet" means "daily full emitted no final orders", not "the system did nothing".

Today was not quiet because data failed or because the daily full pipeline never found names. The primary run found and scored buy candidates, then blocked them at runtime because `regime_admission` rejected BULL_CALM. After that, QP only had the four existing holdings in exit-only mode and became infeasible under the strict C2 policy.

The shadow run was different by design. It used the PatchTST shadow artifact, disabled runtime regime admission, continued despite hard preflight evidence failures, and generated one hypothetical ORCL trim. That action was readonly and not promotable evidence for live trading.

There were also intraday live sell-pending events for META and GILD earlier on 2026-06-02. So the right diagnosis is not "no system activity". The right diagnosis is:

1. Primary full buy flow is blocked by a real BULL_CALM signal-quality gate.
2. Primary risk-reduction flow did not trim an overweight holding because cap-compliance fallback was not enabled in prod strategy config and may still need a soft-sell/tax-gate bypass review.
3. The final no-trade alert reported only the first-order blocker and hid the secondary QP infeasibility context.
4. Runner-originated sell fills can be mislabeled as external/manual after the broker fill lands, because order lifecycle attribution is incomplete.

## Files Of Record

| Item | Path |
|---|---|
| Primary full log | `logs/daily_104/2026-06-02.log` |
| Shadow full log | `logs/daily_104/2026-06-02_shadow.log` |
| Primary run DB | `data/runs.alpaca.db` |
| Shadow run DB | `data/runs.alpaca_shadow.db` |
| Primary state | `backtesting/renquant_104/live_state.alpaca.json` |
| Shadow state | `backtesting/renquant_104/live_state.alpaca_shadow.json` |
| Primary strategy config | `backtesting/renquant_104/strategy_config.json` |
| Shadow strategy config | `backtesting/renquant_104/strategy_config.shadow.json` |
| Primary artifact | `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` |

## Primary Full Decision Tree

Primary run id: `2026-06-02-live-237c90a5`.

| Stage | Observed data | Verdict |
|---|---:|---|
| Broker/account | Alpaca LIVE account `212830627`; equity about `$11,227`; settled cash about `$6,174` | Broker connected |
| Portfolio metrics | PV `$11,227.6`; daily return `+0.00181`; Sharpe21 `10.74`; max DD252 `-0.0205` | No drawdown halt |
| Preflight | model artifact present; panel contract OK; best iteration `100 >= 5`; config fingerprint matched `sha256:14586756d4f67691`; watchlist `142`; sector map `141` buyable | Preflight did not abort |
| Data freshness | `146` symbols, all `>= 2026-06-02` | Data was fresh |
| Regime | `BULL_CALM`, confidence `0.63`, Hurst `0.74`, Hurst regime `MOM` | Calm bull regime |
| Model loading | Loaded models for `114/142`; 9 missing artifacts; 19 skipped for Sharpe below `0.5` | Partial but enough to score |
| Feature build | Feature cache `145/145`; fundamentals `138/142`; earnings surprise `134/142`; insider `99/142`; factor frames `142` | Feature path worked |
| Sell scan | `0` exits from `4` held | No forced exits |
| Buy scan | `110` raw tickers | Candidate path started |
| Earnings/wash filters | `86` candidates remain; AVGO/HPE dropped for earnings; many recent-loss names dropped by wash-sale | Normal filtering |
| Realized vol gate | Dropped `12/86` over 60 percent vol | Normal risk filtering |
| Scoring | XGB scorer loaded `172` features; scored `78` rows (`74` candidates plus `4` holdings) | Model produced scores |
| Calibration | Global pool IC `0.11494838927084809`; calibrated `74/74` candidates and `4/4` holdings | Calibration path worked |
| Runtime admission | `regime=BULL_CALM decision=BLOCK reason=regime_admission:failed:BULL_CALM candidates_blocked=74 holdings_exit_only=4` | Binding buy blocker |
| Kelly | Candidates `0/0`; holdings `4/4`, non-zero average `5.2%` | No buy names survived |
| QP | `status=infeasible`; `n=4`; `sum(w_current)=0.450`; `cash_slack=0.550`; `per_asset_cap_max=-0.042`; `turnover_max=0.150` | Secondary blocker |
| Order emit | `status=infeasible:infeasible -- skip` | No orders emitted |

Primary top scored buy candidates before admission block:

| Ticker | Rank score | Expected return |
|---|---:|---:|
| CRWD | `0.643885535599091` | `0.0554318586683362` |
| MPWR | `0.628101196891478` | `0.0498322408123257` |
| TXN | `0.620460415159918` | `0.0471267534335169` |
| CVS | `0.615003181668508` | `0.0451962021409143` |
| FTNT | `0.613466065303818` | `0.0446526570834303` |

Primary holdings after scoring:

| Ticker | Qty | Position pct | Rank score | Expected return | Sigma |
|---|---:|---:|---:|---:|---:|
| ORCL | `10` | `0.217814627900502` | `0.493161677929277` | `0.00224664760502835` | `0.621456361264949` |
| MU | `1` | `0.094779846886671` | `0.556374027424599` | `0.0245185795077352` | `0.845842659135923` |
| EQIX | `1` | `0.0954541161625707` | `0.516952612721844` | `0.0106361099338258` | `0.170253383016457` |
| HON | `2` | `0.0419040757254634` | `0.548336744651208` | `0.0216886537792312` | `0.243413936542397` |

Primary pipeline counters:

```json
{
  "no_candidate_streak": 1,
  "no_trade_streak": 0,
  "qp_exit_only_topup_guard": 4,
  "qp_infeasible": 1,
  "regime_admission_blocked": 74,
  "regime_admission_holdings_exit_only": 4,
  "risk_gate_vol_dropped": 12
}
```

Primary alert:

```text
RENQUANT-104 [full] DECISION | no trade (regime_admission_blocked(74)) | regime=BULL_CALM conf=0.63 hurst=0.74 hurst_reg=MOM held=4 eq=$11,227 | SHADOW[hf_patchtst_pt07_strict_seed44] top3=GM/IBM/AMAT top10_intersection_prim=1/10 rho=+0.18 n=74
```

The alert was technically true but incomplete: it explained the primary buy-side no-trade reason, but it did not surface the secondary `qp_infeasible(1)` risk-reduction failure.

## Why Primary Said No Trade

The binding cause was `regime_admission_blocked(74)`. That is not a data outage. It is the intended runtime admission gate rejecting all BULL_CALM buy candidates because the current alpha158+fund GBDT artifact has weak BULL_CALM evidence.

The secondary cause was QP infeasibility after all buy candidates were removed. The QP universe contained only the four holdings. ORCL was above cap, but strict C2 policy plus config defaults resulted in "block QP orders for this bar" instead of emitting a cap-compliance trim.

The final alert only chose `regime_admission_blocked(74)` as the headline reason. For operator debugging, the daily bundle needs to report both the first binding buy blocker and any downstream risk-reduction blocker.

## Artifact, Preflight, And Runtime Admission Mismatch

The primary artifact has mixed status signals:

- Top-level `promotion_status` is `gated_buys`.
- `promotion_gating_reason` says WF gate and panel contract checks failed, and that sells/risk exits run via a separate wrapper.
- Nested `metadata.wf_gate_metadata.passed` is true after relax flags.
- Nested `trade_monotonicity.passed` is false.
- Nested `sanity_regime_ic.passed` is false, including BULL_CALM weakness.

BULL_CALM details from the artifact:

| Check | BULL_CALM value | Meaning |
|---|---:|---|
| Trade monotonicity Spearman | `-0.16739012052024713` | Ranking is inverted on the trade sample |
| BULL_CALM sanity mean IC | `0.006100153912037021` | Below useful threshold |
| BULL_CALM sanity median IC | `-0.007441123162973637` | Negative median |
| BULL_CALM hit rate | `0.4925` | Coin-flip level |
| BULL_CALM n_dates | `400` | Enough data to trust the failure |

Current preflight passed `P-REGIME-IC` because the configured relax/operator path accepts eligible regime metadata for this run. Runtime `RegimeModelAdmissionTask` then correctly fail-closed BULL_CALM buys.

This is confusing but not contradictory if read carefully. It is still bad operator ergonomics. A prod full run should have one explicit field that says whether runtime buys are admitted by regime.

## Shadow Full Decision Tree

Shadow run id: `2026-06-02-live-339a8f76`.

| Stage | Observed data | Verdict |
|---|---:|---|
| Broker mode | `readonly-alpaca` | No live orders possible |
| Preflight WF gate | HARD fail: WF gate metadata absent | Not promotable |
| Preflight regime IC | HARD fail: regime-layered IC/monotonicity evidence absent | Not promotable |
| Config fingerprint | Matched | Shadow config internally consistent |
| Runtime admission | Disabled in `strategy_config.shadow.json` | Diagnostic only |
| Data freshness | `146` symbols, all `>= 2026-06-02` | Data fresh |
| Regime | `BULL_CALM`, confidence `0.63` | Same market state |
| Sell scan | `0` exits from `4` held | No forced exits |
| Buy scan | `110` raw tickers -> `108` after earnings | Shadow did not apply the same primary wash-sale clocks |
| Realized vol gate | Dropped `13/108` | Normal risk filtering |
| PatchTST scoring | Scored `99/99`; assigned `95` candidates plus `4` holdings | Shadow model produced scores |
| Veto weak buys | Dropped `82` below rank floor `0.537` | 13 candidates survived |
| Kelly | Candidates `13/13`, average `6.8%`; holdings `4/4`, average `7.8%` | Sizing path worked |
| QP emit | One ORCL sell, no buys | Hypothetical shadow action |

Shadow ORCL action:

```text
QP_SELL ORCL dW=-0.1491 shares=6 reason=qp_sell
QP_SELL_CREDIT ORCL credited=$1467 buy_cash_left=$9614
JointPortfolioQPJob: buys=0 sells=1
```

Shadow trade row:

| Ticker | Side | Qty | Price | P/L | Exit reason | Source |
|---|---:|---:|---:|---:|---|---|
| ORCL | sell | `6` | `244.542` | `+5.3515%` | `qp_sell` | `JointPortfolioQPJob.EmitOrdersFromQPSolutionTask` |

Shadow alert:

```text
[SHADOW]RENQUANT-104 [full] SHADOW-ACTION | SHADOW/HYPOTHETICAL (no live orders) | EXIT ORCL (qp_sell) P/L=$+74.53 (+5.35%) | regime=BULL_CALM conf=0.63 hurst=0.74 hurst_reg=MOM held=4 eq=$11,225
```

This is useful diagnostic evidence that the portfolio-construction layer wanted to reduce ORCL. It is not permission to mirror the action into live, because shadow was explicitly running with hard preflight failures and runtime admission disabled.

## Intraday Activity On 2026-06-02

The day was not completely inactive. The primary DB contains intraday sell-pending events:

| Run id | Time in log | Ticker | Side/status | Qty | Price | Reason | Order id |
|---|---|---|---:|---:|---|---|
| `2026-06-02-live-c0227705` | about `13:42 PT` | META | `sell_pending` | `1` | `598.04` | `single_day_loss` | `5a62bfd8-bd81-48ee-8455-a9f528013e32` |
| `2026-06-02-live-b7f0e2e1` | about `14:00 PT` | GILD | `sell_pending` | `4` | `127.41` | `single_day_loss` | `755c1c8f-62e8-4c4e-9851-93045cb616fc` |

Later state reconciliation logged disappeared positions as `STATE-EXT-SELL`. That can be correct for genuinely manual/external activity, but these two examples show a likely attribution gap: once a runner-submitted pending order fills and the ticker disappears from broker positions, reconciliation may not retain enough order-lifecycle state to classify the disappearance as runner-originated.

This is an execution/orchestrator audit problem, not an alpha problem.

## Root Cause Findings

### F1. BULL_CALM no-signal is real

The primary model found candidates but runtime admission rejected BULL_CALM buys. This is consistent with the artifact's BULL_CALM evidence: mean IC near zero, negative median IC, coin-flip hit rate, and failed monotonicity.

Fixing the silence by loosening `regime_admission` would be the wrong direction unless a new model/artifact proves BULL_CALM signal quality.

### F2. Preflight and runtime admission semantics are too hard to read

Preflight can pass under relax/operator policy while runtime admission blocks all BULL_CALM buys. Operators see "preflight passed" and then "no trade", but the artifact itself is effectively buy-gated for the current regime.

The run bundle needs an explicit runtime buy-admission status before scoring and before the final alert.

### F3. QP cap-compliance risk reduction did not fire in primary

The codebase already has a cap-compliance fallback concept, but the primary config did not enable `rotation.joint_actions.allow_cap_compliance_sells_on_infeasible`. The primary run therefore kept strict C2 behavior and emitted no QP orders.

There is another review point before enabling it: if cap-compliance fallback is a hard risk-reduction path, it must not be suppressed by ordinary soft-sell gates such as BULL_CALM minimum holding days, tax gates, or Davis-Norman no-trade bands. It should emit only risk-reduction sells, never buys.

### F4. The no-trade alert drops important secondary blockers

The alert headline was `regime_admission_blocked(74)`. That is correct but incomplete. `qp_infeasible(1)` was also operationally important because it explains why ORCL was not trimmed even though shadow QP wanted to trim it.

### F5. Shadow action is useful but not promotable

Shadow generated a hypothetical ORCL trim, but shadow also had hard WF/regime preflight failures and disabled runtime admission. Its output should be labeled as diagnostic in every daily summary, run trace, and comparison metric.

### F6. Runner-originated sell fills can be mislabeled as external/manual

The META/GILD path shows sell-pending orders with broker order ids earlier in the day, followed by state reconciliation that can classify disappeared holdings as external/manual. The system needs durable order lifecycle attribution across pending, filled, canceled, and reconciled states.

## Multi-Repo Modification Plan

The fix must follow the multi-repo ownership model. Do not patch this only in the umbrella repo. The umbrella should keep this RFC, integration harness changes, and final pin updates after subrepo PRs merge.

| PR | Owner repo | Change | Why this repo owns it |
|---|---|---|---|
| A | `renquant-pipeline` | Add an explicit `P-RUNTIME-REGIME-ADMISSION` or equivalent preflight/diagnostic task that uses the same policy source as `RegimeModelAdmissionTask`. Persist `admitted_regimes`, `blocked_regimes`, `current_regime_admitted`, and `current_regime_reason` into the decision trace. | Runtime decision-tree and admission policy live in pipeline. |
| B | `renquant-pipeline` | Harden cap-compliance fallback. When QP is infeasible only because current holdings violate caps, allow a sell-only `cap_compliance_fallback` status to emit cap-reducing sells. Add a review/test path that proves these sells are not blocked by ordinary soft-sell horizon/tax/no-trade gates unless an explicit hard tax policy says so. | QP solving and order-intent generation live in pipeline. |
| C | `renquant-strategy-104` | After B is tested, set `rotation.joint_actions.allow_cap_compliance_sells_on_infeasible=true` in prod/golden strategy config, keep `qp_c2_infeasible_policy=strict`, and add a `_reason` comment field explaining that this permits risk-reduction sells only. | Active strategy policy and thresholds live in strategy-104. |
| D | `renquant-artifacts` | Extend artifact manifest validation so `promotion_status=gated_buys` cannot coexist with ambiguous pass/fail fields. Require explicit `runtime_buy_admission_status`, `blocked_regimes`, and regime monotonicity/sanity status fields. | Artifact registry owns manifest contracts and validation. |
| E | `renquant-model` | Stamp model outputs with runtime admission metadata by regime at publish time. If BULL_CALM fails, publish as `sell_only`, `gated_buys`, or `blocked_regimes:["BULL_CALM"]` with no ambiguity. Continue BULL_CALM signal research under the existing recovery plan. | Model factory produces and publishes model evidence. |
| F | `renquant-execution` | Add a broker order lifecycle/audit record with `order_id`, `origin`, status transitions, fill records, submitted quantity, filled quantity, and terminal state. Origins should distinguish `runner_live_order`, `z9_stop`, `manual`, and `unknown`. | Broker execution and order audit live in execution. |
| G | `renquant-orchestrator` | Build the daily run bundle from pipeline trace plus execution order lifecycle. Final notifications should separate `orders_generated_this_run`, `orders_pending_today`, `broker_fills_today`, `state_reconciliation_events`, and `shadow_hypothetical_actions`. Include secondary blockers such as `qp_infeasible(1)`. | Daily/full orchestration and notifications live in orchestrator. |
| H | `renquant-backtesting` | Add a forensic replay fixture for the 2026-06-02 primary and shadow decision trees. Expected counts should include 74 primary admission blocks, 1 primary QP infeasible, and one shadow hypothetical ORCL sell. | Simulation, replay, and forensics live in backtesting. |
| I | `RenQuant` umbrella | After subrepo PRs merge, advance `subrepos.lock.json`, keep this RFC, and add only integration-level assertions that the pinned assembly reports the right decision summary. | Umbrella owns pins and integration harness, not runtime logic. |

Optional shared-contract PR:

| Owner repo | Change |
|---|---|
| `renquant-common` | If more than one repo needs the same schema, define a small `RuntimeAdmissionStatus`, `DecisionBlockerSummary`, or `OrderLifecycleRecord` contract in common first. Keep it minimal. Do not move pipeline behavior into common. |

## Priority Triage

The table above spans many repos, so this campaign must be staged. P0 closes the 2026-06-02 safety and operator-readability defects. P1 hardens manifests and notifications once P0 behavior is stable. P2 is valuable but separable from the BULL_CALM/QP incident.

| Priority | PR | Why |
|---|---|---|
| P0 | B: `renquant-pipeline` cap-compliance fallback | Direct safety fix for the silent risk-reduction failure. |
| P0 | A: `renquant-pipeline` runtime admission status | Makes the binding buy blocker visible before final alerting and gives tests a stable trace field. |
| P0 | C: `renquant-strategy-104` enable cap-compliance fallback | Closes the F3 config root cause, but only after B is green. |
| P1 | D/E: `renquant-artifacts` + `renquant-model` manifest stamping | Prevents ambiguous artifact status from recurring; important but not required before P0 risk trims. |
| P1 | G: `renquant-orchestrator` notification/run-bundle surfacing | Fixes the hidden secondary blocker and improves operator ergonomics. |
| P2 | F: `renquant-execution` broker order lifecycle | Fixes META/GILD attribution quality, but it predates and is separable from the BULL_CALM admission incident. |
| P2 | H/I: `renquant-backtesting` forensic fixture + umbrella pin | Lands after subrepo behavior settles and pins the final integrated assembly. |

## Proposed Merge Order

1. `renquant-common`, only if shared contracts are needed.
2. `renquant-pipeline` PR A: runtime admission status in decision trace.
3. `renquant-pipeline` PR B: sell-only cap-compliance fallback hardening.
4. `renquant-strategy-104` PR C: enable cap-compliance fallback in prod policy after B is green.
5. `renquant-artifacts` PR D and `renquant-model` PR E: manifest validation and model publish stamping.
6. `renquant-orchestrator` PR G: daily bundle and notification wiring.
7. `renquant-execution` PR F: durable broker order lifecycle records.
8. `renquant-backtesting` PR H: forensic replay/regression fixture.
9. `RenQuant` umbrella PR I: pin updates and integration assertions.

## Acceptance Tests

Minimum review gates before merging the behavioral changes:

1. A replay of the 2026-06-02 primary run reports fresh data, `N` scored buy candidates, `N` `regime_admission:failed:BULL_CALM` blocks, and `qp_infeasible(1)` as a secondary blocker. The historical fixture value is `N=74`, but tests should assert `blocked == scored` rather than hard-code the magic number.
2. With cap-compliance fallback disabled, the same primary fixture emits no orders and the final summary still includes the QP infeasibility context.
3. With cap-compliance fallback enabled, an ORCL-like overweight holding under BULL_CALM admission block emits only a cap-reducing sell intent, no buys, and no top-up.
4. Cap-compliance fallback either bypasses ordinary soft-sell horizon/tax/no-trade gates by explicit policy, or fail-closes with a named hard blocker. This must be tested directly.
5. A shadow run with hard WF/regime evidence failures is labeled `diagnostic` or `non_promotable` in the run bundle and notification, even if it emits a hypothetical action.
6. A runner-submitted sell-pending order that later fills with the same broker `order_id` is reconciled as runner-originated, not `external_or_manual`.
7. An artifact with `promotion_status=gated_buys` and failed BULL_CALM evidence must expose machine-readable blocked regimes. If it does not, artifact validation fails.
8. The umbrella integration test proves `subrepos.lock.json` points at subrepo commits containing the above fixes.

## Claude Review Questions

1. Recommendation: cap-compliance sells should bypass BULL_CALM `min_holding_days`, ordinary tax gates, and no-trade bands because the holding is already known to violate a hard cap. The only allowed block should be an explicit hard policy such as `tax_policy.block_cap_compliance_sells=true`.
2. Should `promotion_status=gated_buys` permit full daily runs, or should it force an explicit sell-only mode until a runtime-admitted regime appears?
3. Recommendation: shadow WF/regime preflight failures should remain HARD and require a named diagnostic override, for example `SHADOW_PREFLIGHT_OVERRIDE=true`. They should not be silently downgraded to SOFT by default; every shadow action must still stamp `non_promotable=true`.
4. Should live runner reconciliation be lifted primarily into `renquant-execution`, `renquant-orchestrator`, or split as execution owns order lifecycle and orchestrator owns run-bundle attribution?
5. Should final notifications always show the top two blockers by stage, or should they show all non-zero blocker counters for daily full runs?

## Immediate Operator Read

For 2026-06-02, the most defensible interpretation is:

- Primary live buy silence was correct because BULL_CALM admission failed.
- Primary lack of ORCL trim is suspicious and needs the cap-compliance fallback/config review.
- Shadow's ORCL trim is a useful clue, not an executable live recommendation.
- The day had live sell-pending activity earlier, so "quiet" means "daily full emitted no final orders", not "the system did nothing".
