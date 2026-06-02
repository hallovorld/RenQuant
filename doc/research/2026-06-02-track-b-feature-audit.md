# Track B feature audit — BULL_CALM signal recovery

**Date**: 2026-06-02
**Status**: §7.2.1 R4 audit memo — required before any IC numbers from
the Track B 176-feature variant are reported.
**Parent plan**: [`2026-06-02-bull-calm-signal-recovery-plan.md`](./2026-06-02-bull-calm-signal-recovery-plan.md)
**Owner**: Claude

## Why this memo exists

CLAUDE.md §7.2.1 R4 requires that "when a leak / failure has a hypothesis
list, every hypothesis must get a written audit memo (ruled_in /
ruled_out / inconclusive + evidence + commit SHAs) BEFORE any new
experiment is launched". Track B introduces 4 new features. This memo
documents — per feature — definition, canonical source, causality
argument, expected sign in BULL_CALM, and a hand-computed fixture that
pins the implementation.

The actual walk-forward retrain + per-regime IC evaluation is NOT
launched as part of this PR. Per §7.2#R2, no IC / Sharpe number from
this feature set may be quoted in any commit, doc, or status report
without a companion placebo verdict block.

## Feature audit

### 1. `mom_carry_12_1`

| Field | Value |
|---|---|
| Definition | `close[t-21] / close[t-252] - 1`. 12-month return ending 1 month ago (skipping the 1-month reversal window). |
| Canonical reference | Kelly, B. T., Gu, S., & Xiu, D. (2020). "Empirical Asset Pricing via Machine Learning." *Review of Financial Studies* 33(5), 2223–2273. Equation (4), "MOM12_m" definition; Table 9 documents this factor's IC strength in low-vol regimes. |
| Causality argument | At date `t` the feature uses `close[t-252]` and `close[t-21]`. Both are strictly in the past. The implementation uses `close.shift(21)` and `close.shift(252)` which produces NaN for the first 252 trading days of each ticker — confirming no extrapolation from future data. |
| Expected sign in BULL_CALM | POSITIVE. Standard cross-sectional momentum factor; persistence dominates in calm regimes per Asness et al. 2013 and Kelly-Gu-Xiu Table 9. |
| Hand-computed fixture | With `close = [100, 102, 101, 103, 105, 104]` and shrunken windows `(skip=2, long=4)`, the feature at `t=4` is `close[2]/close[0] - 1 = 101/100 - 1 = 0.01` and at `t=5` is `103/102 - 1 ≈ 0.009803921`. Pinned by `test_mom_carry_12_1_matches_hand_computation`. The production-window causality test (`test_mom_carry_12_1_causal_at_t_does_not_use_future`) perturbs `close[t+1:] *= 1.5` and asserts the value at `t` is unchanged. |

### 2. `beta_dm`

| Field | Value |
|---|---|
| Definition | Daily-rolling 252-day OLS beta of stock daily returns vs SPY daily returns: `beta = cov(r_s, r_m) / var(r_m)` with `ddof=1`. |
| Canonical reference | Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta." *Journal of Financial Economics* 111(1), 1–25. Their BAB factor sorts cross-sectionally on this rolling beta; the low-beta anomaly is strongest in calm markets. |
| Causality argument | The `pandas.rolling(window=252).cov(...) / .var(...)` window at `t` ends at `t`, includes returns `[t-251 .. t]`, and uses no future values. SPY is reindexed onto the stock's calendar with `method="ffill"` and `limit=5` so the alignment is also strictly causal. |
| Expected sign in BULL_CALM | NEGATIVE (the low-beta anomaly). Frazzini-Pedersen show high-beta stocks underperform their CAPM-implied returns in calm bull markets; the canonical BAB factor is long low-beta, short high-beta. |
| Hand-computed fixture | Synthetic stock with `stock_ret = 1.7 * spy_ret + ε` (ε noise σ=0.003) converges in the last 252-day window to `beta ≈ 1.72` (matches `np.cov / np.var` to within `1e-6`). Pinned by `test_beta_dm_matches_rolling_cov_var_definition`. Causality pinned by `test_beta_dm_causal_at_t_does_not_use_future`. |

### 3. `rvar_total`

| Field | Value |
|---|---|
| Definition | Sum of squared daily returns over the last 60 trading days: `(r_t)^2 + (r_{t-1})^2 + ... + (r_{t-59})^2`. |
| Canonical reference | Standard realized-variance proxy used in the low-volatility anomaly literature (Ang-Hodrick-Xing-Zhang 2006; Baker-Bradley-Wurgler 2011; Frazzini-Pedersen 2014). Cleanest realized-vol summary that requires no factor model. |
| Causality argument | `rolling(60).sum()` at `t` sums `[t-59 .. t]` only. The squared-returns transform `(r * r)` is pointwise and trivially causal. The feature warms up over the first 60 trading days per ticker and is well-defined thereafter. |
| Expected sign in BULL_CALM | NEGATIVE (low-vol outperforms in calm regimes). The "low-vol anomaly" is the cross-sectional negative beta of returns on idiosyncratic + total volatility (Baker-Bradley-Wurgler 2011 *Financial Analysts Journal* 67(1)). |
| Hand-computed fixture | With `close = [100, 102, 101, 99, 100, 103, 105]` the daily returns are `[NaN, 0.02, -0.00980, -0.01980, 0.01010, 0.0300, 0.01942]`; the rolling-3 sum-of-squares at idx 4 = `0.00980² + 0.01980² + 0.01010² ≈ 0.000490`. Pinned by `test_rvar_total_matches_hand_sum_of_squared_returns` (using the same `rolling(3).sum()` formula in reverse). Causality pinned by `test_rvar_total_production_window_causal`. Non-negativity pinned by `test_rvar_total_is_non_negative`. |

### 4. `idio_vol_market`

> **Naming note (renquant-base-data #16, 2026-06-02):** the original name
> `idio_vol_3f` was a misnomer. In production, `renquant-base-data` callers
> pass `sector_close=None` (the sector taxonomy lives in the strategy
> layer, not in base-data), so the prod feature is a **SPY + size 2-factor
> residual std** — NOT a 3-factor residual. The honest rename to
> `idio_vol_market` landed in renquant-base-data #16; this memo + the
> Track B constants in `scripts/train_production_model.py` +
> `src/renquant_model_gbdt/panel_data.py` were swapped in the paired
> consumer PRs (RenQuant#120 + renquant-model#29). The 3-factor variant
> remains available to research callers that supply a sector ETF series,
> but production never exercises that path.

| Field | Value |
|---|---|
| Definition | 60-day rolling residual std after OLS regressing stock daily returns on `[1, r_spy, z_size]` (production: `sector_close=None`) — equivalently `[1, r_spy, r_sector, z_size]` when a research caller supplies a sector ETF. `r_spy` is SPY daily return; `r_sector` is the sector ETF daily return (absent in production); `z_size` is the in-window z-score of `log(close * volume + 1)` as a market-cap proxy. |
| Canonical reference | Ang, A., Hodrick, R. J., Xing, Y., & Zhang, X. (2006). "The Cross-Section of Volatility and Expected Returns." *Journal of Finance* 61(1), 259–299. Their IDIO-VOL is residual std from a Fama-French 3-factor regression; section II.B explicitly permits alternate factor sets. Production runs the **2-factor** market+size variant (no sector ETF wired through base-data); this captures the same idio-vol-puzzle signal at lower data cost. |
| Causality argument | Each 60-day window at `t` uses `[t-59 .. t]` returns + SPY returns + (optional sector returns) + size proxy; no point at `> t` enters the regression. The in-window z-score for size uses `rolling(60).mean()` / `rolling(60).std(ddof=1)` ending at `t`, which is strictly causal. The OLS via `np.linalg.lstsq` operates on the window-only matrix. Pinned by `test_idio_vol_market_causal_at_t_does_not_use_future` (the renamed test in renquant-base-data #16). |
| Expected sign in BULL_CALM | NEGATIVE (idio-vol puzzle). Ang et al. 2006 documented a robust negative cross-sectional relation between idio-vol and future returns — strongest in calm-market regimes per their Table 7 sub-sample analysis. The 2-factor (market + size) residual carries the same sign as the canonical 3-factor variant; SMB/HML add specification refinement, not the dominant signal. |
| Hand-computed fixture | When `stock_ret == 1.0 * spy_ret` exactly (perfect linear combo, no idio noise), the 2-factor residual std is ≈ 0 (within `1e-8` floating-point). Pinned by `test_idio_vol_market_returns_zero_resid_for_perfect_linear_combo` (renquant-base-data #16). |

## Hypothesis status table (per §7.2.1 R4)

| Hypothesis | Status | Evidence | Commits |
|---|---|---|---|
| `mom_carry_12_1` is causal | ruled_in | `test_mom_carry_12_1_causal_at_t_does_not_use_future` perturbs `close[t+1:]` and confirms zero impact at `t` | this PR (renquant-base-data feat/bull-calm-track-b-features) |
| `beta_dm` is causal | ruled_in | `test_beta_dm_causal_at_t_does_not_use_future` perturbs both close and SPY at `t+1:` and confirms zero impact at `t` | this PR |
| `rvar_total` is causal | ruled_in | `test_rvar_total_production_window_causal` perturbs `close[t+1:] *= 10` and confirms zero impact at `t` | this PR |
| `idio_vol_market` is causal | ruled_in | `test_idio_vol_market_causal_at_t_does_not_use_future` perturbs all input series at `t+1:` and confirms zero impact at `t` (renamed from `idio_vol_3f` in renquant-base-data #16) | renquant-base-data #16 |
| Track B lifts BULL_CALM mean_ic ≥ +0.020 | UNRESOLVED | retrain + WF gate eval pending (user fires explicitly after PR review) | not yet |

## Sanity contract for the upcoming retrain (per §7.2 + §7.2.1 R2)

When the user fires `train_walkforward_panel.py --include-features
mom_carry_12_1,beta_dm,rvar_total,idio_vol_market`, the following sanity
artifacts MUST appear before any IC number is quoted:

1. **Shuffled-label placebo** on the same 176-feature recipe. IC must be
   within ±2σ of 0.
2. **Time-shift placebo** at shift = 2× label horizon (per PR #31's
   `--label-shift-days 120` for `fwd_60d_excess`). IC must be within ±2σ
   of 0.
3. **A/A re-split** — re-run the WF retrain with the same recipe + seed
   variation and confirm the BULL_CALM IC reproduces within stated CI.
4. **Per-regime IC** (BEAR / BULL_CALM / BULL_VOLATILE / CHOPPY) — quoted
   FIRST, pooled mean_ic SECOND (per §1.3).

Failure of any of 1-3 means STOP and audit before reporting any number.

## Recipe-fingerprint changes

The artifact's `feature_cols` will include the 4 new columns, so the
existing recipe-match check in `scripts/run_wf_gate.py::_recipe_projection`
will naturally distinguish the 176-feature variant from the 172-feature
baseline (different `feature_cols` → different recipe fingerprint).

Additionally, the artifact stamps an explicit `feature_addendum_v1`
field naming which Track B features were enabled, so consumers can
filter manifests without rehashing the full feature list.

## Cross-refs

- Parent plan: [`2026-06-02-bull-calm-signal-recovery-plan.md`](./2026-06-02-bull-calm-signal-recovery-plan.md)
- Parent diagnostic: [`2026-06-02-bull-calm-no-signal-diagnostic.md`](./2026-06-02-bull-calm-no-signal-diagnostic.md)
- CLAUDE.md §1 PRIME DIRECTIVE, §7.2 sanity triad, §7.2.1 R1-R5, §7.10 canonical references
- Memory: [`feedback_research_pipeline_must_gate_with_sanity_triad`](../../memory/feedback_research_pipeline_must_gate_with_sanity_triad.md)
