# RenQuant-103 Assessment

Date: 2026-04-17

This is the current authoritative assessment for renquant_103. Older review notes were based on intermediate code states and are no longer reliable summaries of the implementation.

## Status

The high-confidence implementation gaps from the prior review are now closed.

Fixed in the current codebase:

- LEAN applies the saved GMM scaler before inference.
- LEAN and the live runner both reject below-floor models at load time.
- The live runner now performs real regime detection instead of assuming BULL_CALM.
- The live runner now matches LEAN more closely on trailing stop handling, earnings filtering, transition countdown, correlation guard, wash-sale reconciliation, and cash decrementing during the buy loop.
- Live regime inference now uses real SPY ADX, LEAN-style regime resolution, and confidence-scaled sizing.
- CUSUM now uses a prior reference window in common, LEAN, and live code instead of normalizing on the test window itself.
- CHOPPY no longer has an unreachable model-sell path: `max_hold_days` is now above the global minimum-hold plus sell-streak requirement.
- XGBoost signal thresholds now use the net score directly instead of collapsing to near direction-only behavior.

## Remaining Risks

No remaining high-confidence execution bug stood up to the latest verification pass. The open concerns are mostly statistical or design-level:

- The OOS sample is still short enough that Sharpe-based model selection is noisy.
- Tournament selection across multiple model families can still inflate apparent edge.
- The 63-bar Hurst estimate is likely too noisy to treat as a robust regime discriminator.
- Calibration is still fit on data closely related to model selection, so reported rank quality may be optimistic.
- `blend_weights` are currently `[1.0, 0.0]`, which means relative-strength infrastructure is present but currently inactive by calibration outcome.

These are research-validity concerns, not newly found implementation mismatches between notebook, LEAN, and live execution.

## Verification

- Targeted regression suites: 198 passed.
- Full repository test suite: 441 passed.

The repo is currently in a materially better state than the earlier April 16 review snapshots. The main remaining work is improving statistical validation, not patching core execution logic.
