# 2026-05-23 Codex WF Diagnostics

Status: not promotable. This note is an audit handoff for future agents. Do
not reinterpret these numbers without re-running the commands below after the
current fixes.

## Scope

Input WF trace:

`backtesting/renquant_104/artifacts/diagnostics/wf_trade_traces/prod_semantic_172_causalcal_softguard_20260523T070241Z`

WF command used a production-semantic config derived from prod, 3 concurrent
cuts, strict gate, persisted trade traces, and a walk-forward manifest scope.

## Verdict

The candidate failed promotion:

| Cut | Sharpe | APY | SPY Sharpe | Delta Sharpe |
|---|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | -0.140 | -1.60% | +1.778 | -1.918 |
| 2024-07-01 to 2025-06-30 | +0.690 | +7.00% | +0.715 | -0.025 |
| 2025-04-01 to 2026-03-28 | -0.060 | -1.00% | +0.749 | -0.809 |

Aggregate:

- Mean Sharpe: +0.163.
- Positive cuts: 1/3.
- Mean SPY Sharpe: +1.081.
- Mean Delta Sharpe: -0.918.
- Cuts beating SPY Sharpe: 0/3.
- Sanity: real IC +0.0750, shuffled IC -0.0020, placebo IC +0.0462. Placebo exceeds the 0.5 x real-IC threshold, so the sanity gate failed.
- Trade gate: BULL_CALM score monotonicity failed.
- Trade contract: 5 buys were missing finite entry mu/sigma in this trace.

## What Was Actually Fixed

1. QP top-up provenance bug.

   Root cause: QP could emit top-up buys for existing holdings that were not in
   `ctx.candidates`. Attribution looked only in candidates, so top-up buys could
   persist `mu`, `sigma`, and rank fields as null. Fix: QP order emission now
   reads from the shared `_qp_mu_source_map`, which includes both candidates and
   holdings.

   Regression: `tests/test_joint_qp_task.py::TestActionDirections::test_positive_mu_on_held_topup_preserves_holding_scores`.

2. Tax report overstatement clarity.

   Root cause: event-level tax debits are a conservative cash-stress model, but
   forensic reports did not show the same-year annual-net tax lens. That made
   `tax > gross` look like a tax allocation bug even when losing rows had zero
   tax. Fix: reports now show event-level tax, annual-net tax estimate,
   overstatement vs annual-net, and annual-net PnL estimate.

   Regression: `tests/test_sim_trade_ledger.py::test_forensic_report_shows_annual_net_tax_overstatement`.

3. PatchTST DOE counting bug.

   Root cause: DOE postprocess used planned design count for DSR/PBO reporting
   even when one design point was missing. Fix: reporting now separates planned
   from evaluated design points and lists missing point ids.

   Corrected result: 8/9 design points evaluated, missing point id 8. Best
   observed point is id 1 with bull-regime IC mean +0.0580 and DSR -0.7015.
   This is positive raw-signal evidence, not a production promotion proof.

   Regression: `tests/test_patchtst_doe_hf.py::TestPostprocessContracts::test_counts_evaluated_points_not_planned_points`.

4. Shadow e2e timeout bug.

   Root cause: PatchTST shadow preflight can load HF checkpoint, build panel
   frames, and enrich fundamentals/earnings/insider context, exceeding the old
   420 second timeout. Fix: default shadow timeout is now 1800 seconds.

   Regression: `tests/test_smoke_test_model.py::TestShadowE2EContract::test_shadow_e2e_default_timeout_covers_full_patchtst_cold_start`.

5. WF regime-context ambiguity.

   Root cause: WF metadata exposed HMM/SPY market-context regime counts, while
   actual trades used production pipeline regime labels. These are different
   taxonomies and should not be mixed when explaining trades. Fix: WF metadata
   now also persists trade buy/sell regime counts, buy source counts, sell exit
   reason counts, and missing buy mu/sigma totals from the production trace.

   Regression: `tests/test_wf_gate_cli_contract.py::test_wf_gate_trade_trace_summary_uses_production_decision_regimes`.

6. Exit counterfactual anchor bug.

   Root cause: the counterfactual replay was documented and interpreted as
   "what if we held longer after this exit fired", but its `hold_20d` /
   `hold_60d` columns were anchored to entry date, not exit date. That is a
   different policy and can even point to a date before the actual exit. Fix:
   the tool now keeps the old entry-age barrier columns and adds explicit
   `post_exit_hold_{N}d_*` columns anchored to the actual exit date.

   Regression: `tests/test_exit_counterfactuals.py::test_counterfactual_rows_has_post_exit_continuation_lens`.

7. `max_hold_days` regime-anchor bug.

   Root cause: sell tasks built all exit params from the current regime. In the
   trace, BULL_CALM entries were later evaluated under CHOPPY, so CHOPPY's
   `max_hold_days=40` forced 132 `max_hold` exits even though BULL_CALM entries
   are configured for `max_hold_days=500`. Fix: fresh buys now stamp
   `entry_regime`; sell contexts keep current-regime risk exits, but anchor
   `max_hold_days` to the entry regime.

   Regressions:
   `tests/test_exit_param_wiring.py::test_make_sell_tctx_anchors_max_hold_to_entry_regime`
   and `tests/test_execution_pipeline.py::TestExecutionPipelineBuysOnly::test_new_buy_creates_holding_state`.

8. Exit-parameter telemetry bug.

   Root cause: the simulator trade log recorded exit parameters from the
   current regime only. After the max-hold anchor fix, this could still make
   decision-tree rows show `exit_max_hold_days=40` for BULL_CALM entries even
   when the sell task had applied the entry-regime value. Fix: `ExitSignal`
   now carries the applied exit-parameter snapshot, and the simulator falls
   back to entry-regime max-hold attribution when a portfolio-level exit does
   not carry a sell-context signal.

   Regressions:
   `tests/test_exit_param_wiring.py::test_evaluate_exits_stamps_applied_exit_params_on_signal`
   and `tests/test_sim_sell_attribution.py::test_apply_sell_logs_entry_regime_max_hold_when_signal_lacks_params`.

9. Same-bar QP + TopUp cash double-spend bug.

   Root cause: QP used a local `buy_cash_left`, then `TopUpHeldTask` reread
   original `ctx.cash` and could emit additional top-ups against cash already
   reserved by QP buys. The simulator rejected later orders with insufficient
   cash warnings; live Alpaca could receive an over-budget basket and rely on
   broker-side rejection. Fix: TopUp now subtracts pending buy notional and
   decrements its own remaining budget across multiple top-ups. Runner commit
   also has a final live-cash ledger that resizes or rejects over-budget buy
   intents before broker submission.

   Regressions:
   `tests/test_kelly_sizing.py::TestTopUpHeldTask::test_topup_uses_cash_after_pending_buy_orders`
   and `tests/test_runner_state_fixes.py::TestRunnerCashBudgetGuard`.

10. PaperBroker negative-cash bug.

    Root cause: paper broker logged insufficient cash but executed the buy
    anyway, allowing negative cash. Fix: over-cash paper buys are rejected.

    Regression:
    `tests/test_audit_round2_fixes.py::TestPaperBrokerCashTracking::test_overcash_buy_rejected`.

11. Alpha158 sim cold-start performance bug.

    Root cause: alpha158 scorers rebuild their real features from OHLCV inside
    ApplyScoresTask, but SimAdapter still paid the cost of building legacy
    panel feature/factor frames just to give the scoring task a ticker index.
    Fix: alpha158 scorers can now use a target-only matrix when legacy frames
    are absent, and SimAdapter skips legacy frame prep when static/WF artifacts
    are all alpha158 scorers.

    Regressions:
    `tests/test_panel_scoring_job.py::TestAlpha158TargetOnlyMatrix` and
    `tests/test_sim_walkforward.py::TestStaticModelBehaviorPreserved::test_alpha158_static_scorer_skips_legacy_panel_frame_prep`.

## Forensic Trade Evidence

Closed round trips: 228.

| Cut | Closed | Win Rate | Avg Hold | Gross PnL | Event Tax | Event Net |
|---|---:|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | 86 | 59.30% | 45.62 | +7043.18 | +8509.49 | -1466.31 |
| 2024-07-01 to 2025-06-30 | 61 | 59.02% | 57.56 | +14869.78 | +11440.69 | +3429.09 |
| 2025-04-01 to 2026-03-28 | 81 | 55.56% | 53.47 | +8136.53 | +9345.09 | -1208.56 |

Tax lens:

| Cut | Gross PnL | Event Tax | Annual-Net Tax Estimate | Event Net | Annual-Net PnL Estimate |
|---|---:|---:|---:|---:|---:|
| 2024-01-02 to 2024-12-31 | +7043.18 | +8509.49 | +3521.59 | -1466.31 | +3521.59 |
| 2024-07-01 to 2025-06-30 | +14869.78 | +11440.69 | +7434.89 | +3429.09 | +7434.89 |
| 2025-04-01 to 2026-03-28 | +8136.53 | +9345.09 | +4068.27 | -1208.56 | +4068.27 |

No losing lot carried tax in the FIFO ledger, and no row had tax greater than
its own positive gross after the previous allocation fix. The event-level model
is still intentionally punitive for cash stress. It should not be used alone to
answer "is the signal economically positive after calendar-year netting?"

Production decision trace:

- Buy regimes: BULL_CALM 253, CHOPPY 2, BULL_VOLATILE 1.
- Sell regimes: CHOPPY 102, BULL_CALM 30, BEAR 28, BULL_VOLATILE 11.
- Buy sources: JointPortfolioQPJob 198, TopUpJob 58.
- Sell reasons: max_hold 91, stop_loss 31, trailing_stop 18, panel_conviction 17, qp_sell 9, single_day_loss 4, qp_close 1.
- Missing buy mu/sigma in this pre-fix trace: 5/5.

Closed-trade attribution:

| Slice | N | Gross PnL | Net PnL After Event Tax | Win Rate | Avg Hold |
|---|---:|---:|---:|---:|---:|
| BULL_CALM entries | 226 | +29665.12 | +842.46 | 57.96% | 51.57 |
| CHOPPY entries | 1 | +945.20 | +472.60 | 100.00% | 108.00 |
| BULL_VOLATILE entries | 1 | -560.84 | -560.84 | 0.00% | 3.00 |

Exit reason economics:

| Exit Reason | N | Gross PnL | Event Net | Win Rate | Avg Hold |
|---|---:|---:|---:|---:|---:|
| stop_loss | 35 | -19859.93 | -19859.93 | 0.00% | 33.71 |
| qp_sell | 13 | -1727.88 | -1727.88 | 0.00% | 27.15 |
| panel_conviction | 24 | -432.19 | -648.68 | 37.50% | 35.67 |
| single_day_loss | 4 | -339.49 | -387.28 | 50.00% | 21.75 |
| qp_close | 1 | -375.39 | -375.39 | 0.00% | 28.00 |
| trailing_stop | 19 | +8471.10 | +3378.28 | 89.47% | 66.84 |
| max_hold | 132 | +44313.27 | +20375.10 | 78.79% | 60.54 |

Interpretation: the current candidate has positive gross closed-trade PnL, but
not enough gross edge to beat SPY, and event-level tax stress overwhelms two of
three cuts. The bigger strategy problem is not only tax. The decision tree buys
almost exclusively in BULL_CALM, then many exits happen after the regime has
degraded to CHOPPY/BEAR/BULL_VOLATILE. BULL_CALM score monotonicity failing
means QP/exits/thresholds are not preserving the model's rank signal reliably.

Additional structural finding after the first patch: every `max_hold` row in
the trace used `exit_max_hold_days=40`, and all 132 such exits were positions
entered under BULL_CALM but evaluated under CHOPPY at exit. That is a
regime-flow bug, not a model signal result. It must be re-simulated after the
entry-regime max-hold fix before drawing another APY/Sharpe conclusion.

Post-fix short smoke (not acceptance): a 2024-01-02 to 2024-06-30 sim with the
entry-regime max-hold fix produced APY +13.4%, Sharpe +1.40, max drawdown
5.0%, and event tax only $54. This is encouraging but not promotable: it is
one half-year slice, it did not complete the strict WF/SPY gate, and it exposed
the QP + TopUp same-bar cash-budget bug fixed above.

Performance issue still open: full strict WF is too slow because the alpha158
panel scoring path repeatedly rebuilds per-bar feature inputs. Two safe
performance fixes are now implemented: alpha158 scorers skip legacy panel frame
prep, and sim adapters share a per-run panel runtime cache so ApplyScoresTask
reuses loaded fundamentals, earnings surprise, and sentiment frames instead of
rereading the same parquet files on every bar. These do not change feature
definitions or scorer normalization. The remaining hotspot is alpha158 rolling
feature recomputation itself; that needs a cached-vs-uncached parity test before
broader vectorization.

## References Used For This Fix

- CLAUDE.md prime directive: RenQuant evaluation must be regime-conditional;
  pooled means are secondary.
- Lopez de Prado, Advances in Financial Machine Learning: purged / embargoed
  walk-forward evaluation is required for finance labels with future horizons.
- Bailey and Lopez de Prado, 2014, Deflated Sharpe Ratio: DSR/PBO are used to
  avoid multiple-testing overstatement.
- IRS Topic 409 and Publication 544: capital gains/losses are classified as
  short-term/long-term and netted on Schedule D economics; event-level sale tax
  debit is not the same as calendar-year net capital gains.

## Reproduction

Target tests:

```bash
.venv/bin/python -m pytest \
  tests/test_joint_qp_task.py::TestActionDirections::test_positive_mu_on_held_topup_preserves_holding_scores \
  tests/test_sim_trade_ledger.py::test_forensic_report_shows_annual_net_tax_overstatement \
  tests/test_joint_qp_task.py \
  tests/test_sim_trade_ledger.py \
  tests/test_patchtst_doe_hf.py \
  tests/test_smoke_test_model.py \
  tests/test_preflight.py \
  tests/test_wf_gate_cli_contract.py -q
```

Last target result: 150 passed.

WF command to re-run after these fixes:

```bash
TS=$(date -u +%Y%m%dT%H%M%SZ)
TRACE_DIR="artifacts/diagnostics/wf_trade_traces/prod_semantic_172_causalcal_softguard_${TS}"
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 NUMEXPR_NUM_THREADS=4 \
.venv/bin/python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.staging.json \
  --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json \
  --derive-config-from-prod --strict --jobs 3 \
  --trace-dir "$TRACE_DIR"
```

Expected after this code fix: trade contract should no longer fail on QP
top-up buys missing finite mu/sigma. This does not imply Sharpe/APY promotion.
Sharpe/APY requires a separate model/decision-tree improvement that passes the
same WF and sanity gates.
