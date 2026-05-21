# RenQuant 104 Claude Fix Re-Audit - 2026-05-21

Scope: re-check the recent "fixed" claims against code, artifacts, scripts,
and targeted production-path tests. Docs were used only as pointers, not as
truth. CLAUDE.md principles applied: production path over fixtures, every fix
gets a regression guard, data foundations before model claims.

## Executive Verdict

Claude fixed some real P0s, but the fix set was not complete. Several claims
were only documentation-level or fixture-level true while the production path
still had broken contracts.

This pass repaired the highest-risk remaining issues:

- NGBoost proper training CLI/data contract and stale baseline gate.
- Time-aware data freshness after same-day market close.
- QP full covariance loading from production artifact layout and loaded context.
- QP respect for `rotation.enabled_regimes`.
- Backup push failure-code capture and hard large-file guard.
- DDV attribute naming docs/tests.
- Future-dated `sample_end` no longer shifts tournament OOS cutoff into future.

## Claim Check

| Claim / Area | Re-audit Result | Evidence |
|---|---:|---|
| 3 QP exposure regression tests added | True | `test_vol_target_scales_qp_upper.py`, `test_dd_kelly_scales_qp_upper.py`, `test_vol_target_independent_of_ngb.py`: 11/11 pass |
| FactorZScore leakage fixed | Mostly true | `FactorZScoreTask` now z-scores cross-sectionally per date, not by final-row broadcast |
| Dashboard artifact path fixed | True | `build_dashboard.py` resolves prod panel path via golden config |
| Calibrator default drift fixed | True | script, live config, golden config all use `platt` |
| Shadow artifact paths fixed | True at config/path level | shadow config paths resolve |
| NGBoost proper run/claim is trustworthy | False before this pass | `scripts/train_ngboost_proper.py --help` previously entered training path; dry-run exposed missing feature columns |
| Backup size/push guard fixed | False before this pass | `if ! git push ...; then PUSH_RC=$?` masked rc as 0; >100MB files only warned |
| Data freshness fixed | False before this pass | Friday after close still accepted Thursday cache because logic used sessions strictly before date |
| QP full Sigma fixed | False before this pass | task hardcoded `artifacts/watchlist-correlation.json`; production file is `artifacts/prod/watchlist-correlation.json` and adapters already load `ctx.corr_matrix` |
| QP respects `rotation.enabled_regimes` | False before this pass | legacy RotationJob respected it; JointPortfolioQPJob did not |
| DDV `sue_signal` / `pead_signal` fix fully tested | Partially false before this pass | code used new attrs, but docstring/tests still carried old `sue_score`/`pead_score` names |

## Fixes Made In This Pass

### 1. NGBoost Proper Training Is Now Auditable

File: `scripts/train_ngboost_proper.py`

Before:
- No argparse. `--help` started real training and failed with KeyError.
- Hardcoded stale XGB baseline `0.0294 +/- 0.0029`.
- Missing feature columns failed deep inside pandas indexing.

After:
- Added argparse, `--help`, `--dry-run`, explicit `--seeds`.
- Added `--missing-feature-policy {error,zero}`. Default `error` fails closed.
- Added `--xgb-baseline-mean` and `--xgb-baseline-std`; artifact save is refused without a same-panel baseline unless explicitly bypassed for research.
- Verified current raw-label panel is missing 3 artifact-required features:
  `sentiment_pos_share`, `mean_sentiment`, `n_articles_log`.

Current state:
- `--dry-run` default exits 2 with a clear error.
- `--dry-run --missing-feature-policy zero` exits 0 and validates `Xtr=(551043, 172)`, `Xva=(147066, 172)`.

### 2. Data Freshness Is Time-Aware

Files:
- `backtesting/renquant_104/kernel/data.py`
- `backtesting/renquant_104/kernel/pipeline/task_data_freshness.py`
- `backtesting/renquant_104/kernel/pipeline/context.py`
- `backtesting/renquant_104/adapters/runner.py`
- `tests/test_data_freshness.py`

Before:
- Logic required the last completed NYSE session strictly before `ctx.today`.
- Friday after market close still accepted Thursday data.

After:
- Live `RunnerAdapter` stamps `ctx.run_timestamp`.
- Freshness checks require today's bar after today's NYSE close, and accept yesterday's bar before close.
- Added before-close and after-close regression tests.

### 3. QP Production Path Now Matches Config

Files:
- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
- `backtesting/renquant_104/kernel/portfolio_qp/job_qp.py`
- `tests/test_p0_fixes_regression_guards.py`

Before:
- `ComputeFullSigmaTask` ignored `ctx.corr_matrix` and hardcoded the wrong file path.
- `JointPortfolioQPJob` ignored `rotation.enabled_regimes`.

After:
- QP uses loaded `ctx.corr_matrix` first.
- Artifact fallback resolves `regime.correlation_artifact`, defaulting to `prod/watchlist-correlation.json`.
- QP skips regimes outside `rotation.enabled_regimes`.

### 4. Backup Guard Now Fails Closed

Files:
- `scripts/backup_to_github.sh`
- `tests/test_p0_fixes_regression_guards.py`

Before:
- Push rc could be masked by `if ! git push`.
- >90MB warning existed, but >100MB GitHub hard-block risk did not fail before push.

After:
- Hard fail at `+99M`.
- Capture push output to `PUSH_LOG`; capture real rc directly.

### 5. Future `sample_end` No Longer Pollutes OOS Cutoff

Files:
- `backtesting/renquant_104/training/tournament.py`
- `tests/test_audit_2026_05_04_fixes.py`

Before:
- Active config has `sample_end = 2026-06-30`, which is future-dated on this audit date.
- `resolve_oos_cutoff()` used future `sample_end` as anchor.

After:
- OOS anchor clamps to `min(sample_end, today)`.

## Remaining Risks Not Solved By Code Patch

1. Production panel artifact metadata is still weak:
   - `artifacts/prod/panel-ltr.alpha158_fund.json` has 172 features but no `oos_mean_ic`, `val_ic`, `train_ic`, or metadata keys.
   - It is trained_date `2026-05-18`.
   - This is not an acceptance-grade artifact record.

2. Current raw-label panel and panel artifact disagree on sentiment columns.
   - This blocks default NGBoost proper training by design after this patch.
   - Correct fix is to regenerate the raw-label panel from the same feature pipeline, not silently zero-fill in production.

3. Existing production artifacts may predate some leak fixes.
   - FactorZScore code is fixed, but already-written artifacts need retrain/stamping before claiming the live model is leak-clean.

4. Calibrator saturation guard exists at runtime, but artifact/preflight acceptance is still not a full cross-section saturation audit.

5. QP expected-return contract remains a design risk.
   - The code can still fall back from `mu` to raw `panel_score` unless transform/source config is explicit.
   - This is safer after the covariance/regime fixes, but not a fully principled expected-return model.

## Verification Run

Commands run:

```bash
./.venv/bin/python scripts/train_ngboost_proper.py --help
./.venv/bin/python scripts/train_ngboost_proper.py --dry-run
./.venv/bin/python scripts/train_ngboost_proper.py --dry-run --missing-feature-policy zero
./.venv/bin/pytest tests/test_data_freshness.py tests/test_p0_fixes_regression_guards.py tests/test_eval_drivers_smoke.py tests/test_regime_momentum_and_deep_dd_gates.py tests/test_buy_sell_audit_fixes.py tests/test_portfolio_qp_solver.py tests/acceptance/jobs/test_split_jobs_e2e.py tests/test_audit_2026_05_04_fixes.py::TestTournamentOOSWindow30d tests/test_vol_target_scales_qp_upper.py tests/test_dd_kelly_scales_qp_upper.py tests/test_vol_target_independent_of_ngb.py -q
```

Results:
- Targeted regression suite: 163 passed, 4 skipped.
- Claude's three exposure-scaling tests: 11 passed.
- NGBoost dry-run default: fails closed with missing-feature contract error.
- NGBoost dry-run zero-fill exploratory path: passes input contract validation.

## Recommendation

Do not treat the model as fully trustworthy yet. The execution pipeline is
safer after this pass, but the model artifact itself still lacks acceptance
metadata and the current raw-label data does not match the active feature
contract. The next promotion-quality step is a clean panel rebuild/retrain
that stamps feature fingerprint, split dates, embargo, OOS IC distribution,
calibrator diagnostics, and artifact lineage into the production artifact.
