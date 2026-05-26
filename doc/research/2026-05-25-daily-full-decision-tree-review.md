# RenQuant 104 Daily Full Decision Tree Review — 2026-05-25

## Scope

User requested a forced full daily run even though `scripts/daily_104.sh`
skipped because 2026-05-25 was a NYSE holiday. Per the new full-daily mandate,
direct live and shadow runner paths were executed.

## Runs Reviewed

- Live full: `logs/daily_104/2026-05-25_live_user_full_after_fixes.log`
  - Exit code `2`; no orders.
  - Stopped at strict preflight.
- Shadow full: `logs/daily_104/2026-05-25_shadow_user_full_after_qpskip.log`
  - Exit code `0`; readonly broker; no live orders.
  - DB run id: `2026-05-25-live-3c4ef3c8`.

## Fixes Landed While Reviewing

- `11526ab fix live decision diagnostics and feature cache parity`
  - `ntfy` no-trade rollup now reports contract/QP/risk blockers before the
    generic `drawdown_halt`.
  - Live/shadow RunnerAdapter now uses the same run-local causal feature cache
    as SimAdapter.
  - NGBoost fail-closed now clears `_ngboost_head` explicitly.
  - Relevant tests: `94 passed`, then included in `110 passed`.
- `3e73fdd skip qp after panel contract failure`
  - JointPortfolioQPJob skips when panel scoring contract already failed.
  - This preserves the design rule: alpha admission fails first; QP is sizing,
    not an alpha fallback.
  - Relevant tests: `110 passed`.

## Runtime Improvement

Before the cache parity fix, shadow full spent about `502s` in the pipeline and
`473s` in `TickerCandidateJob`.

After the fix:

- Feature cache built for `145/145` tickers.
- Panel frame prep was a cache hit.
- `TickerSellJob`: `0.32s`.
- `TickerCandidateJob`: `1.19s`.
- Full `InferencePipeline`: `2.89s`.

This confirms the previous live/shadow path was not reusing the sim feature
surface and was doing redundant per-ticker rebuild work.

## Live Full Result

Production live full remains correctly fail-closed at preflight. Hard failures:

- `P-PANEL-CONTRACT`: active panel artifact is missing strict contract fields:
  `train_run_id`, `oos_mean_ic`, `oos_std_ic`, `oos_per_fold_ic`, `cv_method`,
  `cv_embargo_days`, plus sentiment runtime gate contract.
- `P-WF-GATE`: active artifact carries failed evidence:
  `wf_sharpe_mean=-1.3233333333333333`, `spy_sharpe_mean=1.0808386653410664`,
  beat SPY Sharpe `0/3`.
- `P-REGIME-IC`: regime sanity IC evidence absent.
- `P-CONFIG-FP`: live fingerprint `sha256:14586756d4f67691` mismatches stored
  `sha256:9333f7bf91d10cc4`; diff fields include `sector_etf_map`,
  `sector_map`.

Conclusion: no live buy should be emitted from the current prod artifact.

## Shadow Decision Tree Result

Run id: `2026-05-25-live-3c4ef3c8`.

Portfolio state:

- Regime: `BULL_CALM`.
- Confidence: `0.5958`.
- Equity: about `$10,829.54`.
- Cash: about `$8,259.87`.
- Exits: `0`.
- Buys: `0`.
- Rotations: `0`.

Counters:

```json
{
  "no_candidate_streak": 1,
  "no_trade_streak": 2,
  "panel_scoring_fail_closed": 91,
  "risk_gate_vol_dropped": 13
}
```

Blocker counts:

- `panel_scorer_config_mismatch`: `91`.
- `risk_gate_vol`: `13`.
- `universe:no_artifact`: `9`.
- `held_no_new_buy`: `6`.
- `earnings_blackout`: `4`.
- Remaining rows are universe Sharpe floor rejects.

Top pre-panel candidates were scanned and persisted, but every one was blocked
by `panel_scorer_config_mismatch`. Examples:

- `NEE`: rank `0.4667`, ER `+0.0839`, XGBoost, utility.
- `MO`: rank `0.4211`, ER `+0.0955`, XGBoost, consumer.
- `CRWD`: rank `0.3484`, ER `+0.0443`, QLearning, software.
- `NVDA`: rank `0.3474`, ER `+0.0456`, Manual, ai_chip.
- `SOFI`: rank `0.3471`, ER `+0.0250`, Manual, finance.

Held positions were not QP-blocked after the fix; they are now stamped
`held_no_new_buy`:

- `FTNT`: rank `0.2437`, ER `+0.0472`, XGBoost, software.
- `EQIX`: rank `0.1269`, ER `-0.0001`, Classification, datacenter_hw.
- `GE`: rank `0.1866`, ER `-0.0016`, Manual, industrial.
- `MU`: rank `0.3012`, ER `+0.0230`, Manual, ai_chip.
- `META`: rank `0.2389`, ER `+0.0366`, QLearning, giant_tech.
- `HON`: rank `0.0715`, ER `-0.0032`, Manual, industrial.

## Root Cause Of `panel_scorer_config_mismatch`

The shadow PatchTST artifact is a `.pt` model:

`artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`

Its sequence sidecars exist, but neither exposes the strict top-level metadata
needed by the live/full contract:

- `config_fingerprint`: missing.
- `oos_mean_ic`: missing.
- `oos_std_ic`: missing.
- `oos_per_fold_ic`: missing.
- `cv_method`: missing.
- `cv_embargo_days`: missing.
- `train_run_id`: missing.

The sidecar does carry training-contract details such as seed, split counts,
embargo days, label column, and `best_val_ic=0.030657318920158994`, but that is
not equivalent to a current config/sector fingerprint or strict WF acceptance
manifest. Therefore the runtime fail-closed behavior is correct.

Do not manually stamp the current fingerprint onto this artifact unless the
training feature/config/sector surface is provably identical. The scientific
fix is to retrain or re-export PatchTST with the strict artifact contract and
then run the same WF/SPY/regime/calibration acceptance checks as XGB.

## Current Assessment

- The current prod model is not live-buy trustable.
- The current PatchTST shadow artifact is not actionable because its model
  contract is incomplete.
- The decision tree persistence is now much more usable: the latest shadow run
  can answer why each candidate/holding was blocked without relying on logs.
- The biggest active blocker is no longer execution speed or QP confusion; it
  is artifact acceptance quality and strict metadata evidence.

## Next Mainline Step

Produce one strict, comparable acceptance pack for PatchTST and XGB:

- stamped artifact fingerprint and field snapshot;
- WF Sharpe/APY with SPY per cut;
- regime-layered IC and monotonicity;
- calibration health;
- decision-tree sim trace with trade P/L attribution.

Only after that can a PatchTST or XGB artifact be promoted into a buy-capable
path.
