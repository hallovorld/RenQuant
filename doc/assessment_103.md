# RenQuant-103 Assessment

Date: 2026-04-19

This is the current authoritative assessment for renquant_103. Older review notes were based on intermediate code states and are no longer reliable summaries of the implementation.

## Status

All high-confidence implementation gaps from prior reviews are now closed.

Fixed in the current codebase:

- LEAN applies the saved GMM scaler before inference.
- LEAN and the live runner both reject below-floor models at load time.
- The live runner now performs real regime detection instead of assuming BULL_CALM.
- The live runner now matches LEAN more closely on trailing stop handling, earnings filtering, transition countdown, correlation guard, wash-sale reconciliation, and cash decrementing during the buy loop.
- Live regime inference now uses real SPY ADX, LEAN-style regime resolution, and confidence-scaled sizing.
- CUSUM now uses a prior reference window in common, LEAN, and live code instead of normalizing on the test window itself.
- CHOPPY no longer has an unreachable model-sell path: `max_hold_days` is now above the global minimum-hold plus sell-streak requirement (40d vs 30+3).
- XGBoost signal thresholds now use the net score directly instead of collapsing to near direction-only behavior.
- LEAN `_build_exit_params()` now correctly passes `lt_hold_gate_days` and `lt_hold_min_gain` to `compute_exits()` — the tax-aware hold gate is now active in LEAN (was silently disabled before).
- Notebook training cells (feature building, tournament, export) extracted into `training/` modules — notebook is now a thin orchestrator with no inline training logic.

## Remaining Risks

No remaining high-confidence execution bugs stood up to the latest verification pass. The open concerns are statistical or design-level:

- The OOS sample is still short enough that Sharpe-based model selection is noisy.
- Tournament selection across multiple model families can still inflate apparent edge.
- The 63-bar Hurst estimate is likely too noisy to treat as a robust regime discriminator.
- Calibration is fit on data closely related to model selection, so reported rank quality may be optimistic.
- `blend_weights` are currently `[1.0, 0.0]`, which means relative-strength infrastructure is present but inactive by calibration outcome. Will update after RS bug fix fully propagates.

These are research-validity concerns, not implementation mismatches between notebook, LEAN, and live execution.

## Verification

- Targeted regression suites: 198 passed.
- Full repository test suite: 560 passed, 2 skipped (562 total).

The repo is in a materially better state than earlier review snapshots. The main remaining work is improving statistical validation and following through on `doc/improvement_plan_2026-04-17.md`, not patching core execution logic.
