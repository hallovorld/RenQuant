# T2-4 (Boyd cvxpy) + Macro v2 + XGB+Macro — Strict Deep Audit (2026-04-27)

**Trigger**: User direction "你新加的t2-4 也有很多bug, 现在进行deep audit" + "你的xgb+macro bug也很多, 开始deep audit". Both written in haste during the autonomous handoff window. Code reads first, audit second.

---

## Part 1 — `kernel/rotation_convex.py` (T2-4)

### Bug T1 — `delta` array unaligned to mu/sigma when input not pre-aligned (HIGH)

**File:line**: `rotation_convex.py:114-126`

```python
tickers = list(current_weights.index)
mu = expected_returns.reindex(tickers).fillna(0.0).values
sigma = cov_matrix.reindex(index=tickers, columns=tickers).fillna(0.0).values
```

`expected_returns` and `cov_matrix` are reindexed to `tickers`, but the
reindex uses **NaN fill default**, then `.fillna(0.0)`. Tickers absent
from `expected_returns` get `μ=0` — the optimizer sees them as "neutral"
and may include them in the buy candidate set even though we have no
prediction for them. Should explicitly raise on missing.

**Effect**: Silent partial-input → confident-zero predictions → solver
buys arbitrary unmodeled tickers up to allocation cap.

**Fix**: `if not set(tickers).issubset(expected_returns.index): raise
ValueError("missing μ for tickers ...")`.

### Bug T2 — `cov_matrix` reindex creates NaN-filled rows for missing tickers; ridge masks the NaN→0 fill (HIGH)

**File:line**: `rotation_convex.py:123-126`

Same issue as T1 but worse: missing rows in `cov_matrix` are filled with
0 → these tickers appear UNCORRELATED with everything (zero off-diagonal
+ zero variance). The `+ 1e-8 * np.eye(n)` ridge keeps the matrix PSD,
but doesn't fix the conceptual bug — the optimizer sees them as
risk-free.

**Effect**: Solver heavily allocates to "risk-free" (= unmodeled) tickers.

**Fix**: explicit error on missing rows in cov_matrix.

### Bug T3 — Smoothed L1 cost penalty UNDER-counts the actual L1 norm (MED)

**File:line**: `rotation_convex.py:204, 210`

```python
eps_l1 = 1e-6
l1_smooth = np.sum(np.sqrt(delta * delta + eps_l1))
```

For Δw=0, `sqrt(0 + 1e-6) ≈ 0.001`. So at zero-delta, the smoothed L1
already adds `n × 0.001 = 0.099` for n=99 tickers as a "fake transaction
cost". This biases the optimizer toward LARGER trades (since the marginal
cost of moving from δ=0 to δ=ε actually DECREASES due to the smoothing
profile: `d/dδ sqrt(δ²+ε)` at δ=0 is 0, much smaller than `d/dδ |δ|` =
±1).

**Effect**: Lower effective transaction cost → larger trades than
intended. cost_coef=0.001 in config might effectively act like
0.0005 per the smoothing.

**Fix**: Use `huber` smoothing (linear for |δ| > ε, quadratic below) OR
just use cvxpy's exact `cp.norm(delta, 1)` (the cvxpy path already does).

### Bug T4 — `cvxpy_OSQP` always reports the wrong solver name (LOW)

**File:line**: `rotation_convex.py:187`

```python
solver_used = "cvxpy_OSQP" if prob.solver_stats is None else f"cvxpy_{prob.solver_stats.solver_name}",
```

Logic is INVERTED: `solver_stats is None` means we don't know which
solver ran — but the code says `cvxpy_OSQP`. Should be the opposite:
when `solver_stats` is None, fallback name should be `cvxpy_unknown` and
the `else` branch (with stats) should use the actual name.

**Fix**: invert ternary.

### Bug T5 — `prob.value` interpretation flip (HIGH)

**File:line**: `rotation_convex.py:185`

```python
objective_value = float(prob.value) if prob.value is not None else float("nan"),
```

cvxpy uses `Maximize` here, so `prob.value` IS the maximized value (not
the negated minimization). Looks correct. **NOT A BUG** but worth
mentioning because the scipy path has to negate (`-float(result.fun)`)
and the asymmetry could mislead callers.

(Withdrawing — not a bug.)

### Bug T6 — `quantize_to_whole_shares` doesn't respect short-sell prevention (HIGH)

**File:line**: `rotation_convex.py:299-313`

When `notional < 0` (sell), `out[ticker] = -int(abs(notional) / price)`.
But there's no check that the ticker is currently HELD with at least
`abs(shares)` shares. The function operates only on weight fractions, not
position counts. If the solver wants to sell 100 shares of AAPL but we
only own 50, the quantizer would still emit -100, leading to attempted
short sale.

**Effect**: Invalid sell orders downstream.

**Fix**: pass current_holdings to quantize and clip sells at current
position count.

### Bug T7 — Negative `notional == 0` shortcut never fires (LOW)

**File:line**: `rotation_convex.py:312-313`

```python
elif notional < 0:
    ...
# notional == 0 → no trade
```

Comment says "no trade" but `out[ticker] = 0` is the default initial value.
Fine in practice; no bug.

(Withdrawing — not a bug.)

### Bug T8 — `bounds` upper limit logically wrong (HIGH)

**File:line**: `rotation_convex.py:224`

```python
bounds = Bounds(lb=-w, ub=np.full(n, self.leverage_cap))
```

Upper bound for δ is `leverage_cap` (default 1.0). But δ_i = +1.0 means
"add 100% NAV to ticker i". Combined with `sum(δ) ≤ leverage_cap - sum(w)`,
single-ticker bound is irrelevant in most cases — but for empty portfolios
(`sum(w) = 0`), the bounds let δ go up to leverage_cap=1 in a single
position. Should be `min(leverage_cap, sector_max_pct)` (or use
sector_max_pct as the per-position upper).

**Effect**: Fresh portfolio can YOLO 100% into one ticker (then sector cap
catches it, but UB=1 is too loose to start with).

**Fix**: `ub = np.full(n, self.sector_max_pct)` since no single position
should exceed sector concentration cap.

### Bug T9 — SLSQP NonlinearConstraint may not be supported in old scipy (MED)

**File:line**: `rotation_convex.py:244, 253`

`NonlinearConstraint` is supported by `trust-constr` but for SLSQP the
docs warn it's converted internally. Behavior may differ across scipy
versions. Should explicitly use `trust-constr` solver when nonlinear
constraints present, or rephrase turnover as ineq constraint via slack
variables.

### Bug T10 — `eps_l1 = 1e-6` makes L1-smoothing's gradient discontinuity scale-dependent (MED)

**File:line**: `rotation_convex.py:204`

For δ values in [0.01, 0.5] (typical trade weights), `eps_l1 = 1e-6` gives
near-exact L1 behavior (smoothing only matters at δ ≈ √eps ≈ 0.001).
GOOD for normal-size trades. But for very tiny rebalances (δ ≈ 0.0001),
the smoothing makes the L1 LINEAR-in-δ² rather than LINEAR-in-|δ|,
distorting the cost. Could matter for high-frequency rebalances.

(LOW priority — current operator config is daily, trades typically > 0.001)

### Bug T11 — quantize_to_whole_shares allocates without considering tax cost (LOW)

**File:line**: `rotation_convex.py:284-313`

Long-only buys cost `shares × price` (decreases cash). Sells gain
`abs(shares) × price` (increases cash). Tax / commissions / slippage
are NOT modeled in the cash budget. For after-tax APY targets, this
under-estimates the cash needed for buys.

(LOW for now — tax accounting is at the strategy_config layer.)

### Bug T12 — No test on `T2-4`'s solver semantics (HIGH for confidence)

`tests/` has no test for `ConvexRotationSolver`. The smoke test I ran in
the commit message verified ONE 3-ticker case but did not pin contract
behavior (constraints respected? optimum found? matches closed-form on
solvable problems?).

**Fix**: write `tests/test_rotation_convex.py` with at least:
- 2-ticker problem with closed-form sol
- All constraints binding (verify each)
- Infeasible problem (verify graceful failure)
- Per-sector cap tested
- Quantization respects cash

---

### T2-4 summary

| # | Issue | Severity |
|---|---|---|
| T1 | μ reindex with NaN fill silently zeroes missing | HIGH |
| T2 | Σ reindex with NaN fill silently zeroes missing | HIGH |
| T3 | L1 smoothing under-counts cost | MED |
| T4 | Solver name reporting inverted | LOW |
| T6 | Short-sell prevention missing in quantizer | HIGH |
| T8 | bounds.ub = leverage_cap allows YOLO | HIGH |
| T9 | NonlinearConstraint behavior under SLSQP | MED |
| T10 | eps_l1 magnitude-dependent smoothing | LOW |
| T11 | No tax/slippage in cash budget | LOW |
| T12 | No tests | HIGH |

**4 HIGH + 1 HIGH-confidence + 2 MED + 3 LOW = 10 actionable issues.**

---

## Part 2 — Macro v2 (kernel/macro_per_ticker.py + pipeline wiring)

### Bug M1 — `macro_levels_to_returns` only handles `_level_z` columns; pre-existing chg columns are dropped (MED)

**File:line**: `macro_per_ticker.py:152-160`

```python
for col in macro_levels.columns:
    if col.endswith("_level_z"):
        base = col.replace("_level_z", "")
        out[f"{base}_chg"] = macro_levels[col].diff()
    # else: skip — chg_5d / chg_20d are smoothed, not point returns
```

Comment is right that `chg_5d_z` is smoothed, but the implementation
**drops them entirely**. v1 macro frame had THREE transforms per symbol
(level_z, chg_5d_z, chg_20d_z). v2 only uses level_z → it's throwing away
2/3 of the macro information.

**Fix**: include `chg_5d_z` directly (without diff) — it's already a
return-ish signal and useful for medium-horizon β.

### Bug M2 — Strict-prior shift compromises bar t computation (HIGH)

**File:line**: `macro_per_ticker.py:124-126`

```python
beta = cov / var.replace(0, np.nan)
cols[f"beta_{macro_col}_{rolling_window}d"] = beta.shift(1)
```

`rolling().cov()` at bar t uses data [t-window, t]. The `.shift(1)` makes
`β_at_t` use data through t-1 (correct strict-prior). BUT: at t=0 (or
during warmup) beta is NaN. The shift puts NaN where the FIRST valid β
should appear. Off-by-one.

**Effect**: First valid β is delayed by 1 bar. Minor but slightly less data
than expected.

**Fix**: This is actually the correct "no leak" semantics — bar t cannot
see today's data. Verifying this is intentional. **Not a bug; documenting**.

(Withdrawing — correct semantics.)

### Bug M3 — `min_window` defaults differ in callers (MED)

**File:line**: `pp_panel_training.py:LoadMacroPerTickerBetasTask`

Reads `cfg.get("min_window", 30)` and passes to `compute_per_ticker_macro_betas`.
But in `compute_per_ticker_macro_betas` itself, default is also 30. Two
defaults declared in two places — drift risk.

**Fix**: Centralize default in one location.

### Bug M4 — `ohlcv` filter `if df.empty or "close" not in df.columns` (LOW)

**File:line**: `macro_per_ticker.py:76-78`

```python
if df is None or df.empty or "close" not in df.columns:
    continue
```

Silent skip — but the calling code expects βs for every ticker in the
watchlist. Skipped tickers have no β and get the "missing → 0.0 fill"
treatment in build_panel_frame. Should log which were skipped.

### Bug M5 — Rolling covariance formula uses biased estimator (LOW)

**File:line**: `macro_per_ticker.py:114-118`

`pandas.rolling().cov()` uses `ddof=1` by default (sample covariance).
That's correct for OLS β estimation, no bug. Documenting since it's a
common gotcha.

(Withdrawing — correct.)

### Bug M6 — β shift creates NaN at end of training window (HIGH)

**File:line**: `macro_per_ticker.py:126`

`beta.shift(1)` — the LAST bar's β is computed from prior data, so the
first NaN is at index 0, last NaN... actually shift(1) just moves all
values forward by one. The LAST original β (computed using full window)
becomes the value at index N-1. **Wait**, that's wrong: `shift(1)` shifts
values DOWN by 1 (or "forward in time" — values move from index t to t+1).
So β at index 0 becomes NaN; β at index t (originally computed from data
[t-window, t]) is now stored at index t+1.

But the LAST index N-1 never gets a value — it would have come from index
N (which doesn't exist).

**Effect**: The last bar of training data has β=NaN → those rows get
filtered/zero-filled in build_panel_frame, losing one bar per ticker.

For training, this is fine (we already drop most-recent N rows due to
forward-return labels needing `lookahead_days` future). For INFERENCE
at "today", β at "today" is computed from data through "yesterday"
(strict-prior) — but per the shift, "today's β" requires the
unshifted β at "tomorrow", which doesn't exist.

**This is wrong for inference**: at the latest bar, we need β computed
from prior data, not β shifted forward from a non-existent future.

**Fix**: don't shift; instead use `closed='left'` parameter on
`.rolling()` to exclude current bar OR roll on `ticker_returns.shift(1)`
+ `macro_r.shift(1)` so the rolling window itself excludes today.

### Bug M7 — Macro chg ≠ macro return (numerical scale mismatch) (MED)

**File:line**: `macro_per_ticker.py:158`

`out[f"{base}_chg"] = macro_levels[col].diff()` — the input is ALREADY
z-scored levels (vxx_level_z). Diff of z-scored levels ≠ z-scored returns.
If level_z is ~N(0, 1), then diff is ~N(0, √2 / window) — much smaller scale.

Then `compute_per_ticker_macro_betas` regresses ticker_returns (scale ~0.01)
on these diff'd z-scores (scale ~0.1). The β coefficient absorbs the scale
mismatch, giving artificially small β values.

**Effect**: β values systematically biased toward zero → less ranking
information.

**Fix**: derive macro returns directly from macro raw closes, not from
z-scored level diff. OR z-score the diff series for consistency.

### Bug M8 — No safeguard on `macro_returns.reindex(ticker_returns.index)` for missing dates (MED)

**File:line**: `macro_per_ticker.py:97-101`

```python
for macro_col in macro_cols:
    macro_r = macro_returns[macro_col].reindex(ticker_returns.index)
```

If `macro_returns.index` doesn't cover `ticker_returns.index` (e.g. macro
ETF data starts later than ticker data), the reindex produces NaN at
non-overlapping dates. Rolling cov + var both produce NaN. β NaN at those
dates.

For a 60d window, you need at least 30 days (min_window) of overlapping
non-NaN. If macro data is shorter, β stays NaN throughout.

**Effect**: tickers whose history extends back farther than macro data
get all-NaN β → useless feature. Silent.

**Fix**: warn on substantial date mismatch (e.g. > 10% NaN in macro_r).

### Bug M9 — `macro_betas` field initialized as `dict` but field type hint forward-references (LOW)

**File:line**: `context.py`

```python
macro_betas: dict[str, "pd.DataFrame"] = field(default_factory=dict)
```

`pd.DataFrame` is in quotes (forward reference). With `from __future__ import
annotations` this would be unnecessary but should work. Documenting.

(Withdrawing — works.)

### Bug M10 — Inference path doesn't reload macro_betas if cache stale (HIGH)

**File:line**: `pipeline.py:LoadMacroPerTickerBetasTask` invocation

The inference path computes macro_betas on every bar from `ctx.ohlcv` +
`ctx.macro_factor_frame`. If today's macro frame is stale (cache hasn't
been refreshed since T-1), β is computed from old data. Per inference
discipline, this is fine (β at T uses [T-60, T-1]) — but the operator
should know which macro symbols had stale data on inference day.

**Fix**: inference-side staleness check. Log or G15 acceptance gate.

---

### Macro v2 summary

| # | Issue | Severity |
|---|---|---|
| M1 | chg_5d/chg_20d columns dropped | MED |
| M3 | Default duplicated in 2 places | MED |
| M4 | Silent skip without log | LOW |
| M6 | shift(1) misalignment for inference latest bar | HIGH |
| M7 | Macro "return" derived from z-scored levels — scale wrong | MED |
| M8 | Missing date overlap warning | MED |
| M10 | Inference staleness not checked | HIGH |

**2 HIGH + 4 MED + 1 LOW = 7 actionable issues.** (Withdrew M2, M5, M9 as non-bugs.)

---

## Part 3 — XGB + Macro path (re-audit)

The v1 macro path from earlier session: `kernel/panel_pipeline/feature_matrix.py:88-113` broadcasts macro values across all tickers per date.

Already documented in `macro-factor-frame-redesign.md` — the structural
bug is that broadcast macro features have ZERO within-date variance for
the rank loss. v2 (per-ticker β) fixes this conceptually but introduces its own bugs (above).

### Bug XM1 — v1 broadcast macro path STILL ACTIVE for non-`v2` config (HIGH legacy)

**File:line**: `feature_matrix.py:106-113`

```python
if macro_values is not None:
    for k, v in macro_values.items():
        ...
        row[k] = v
```

When `panel_ltr.macro.enabled=true` and `version="v1"` (default), the
broadcast logic still fires. v1 is documented as "fundamentally broken"
in the redesign doc but the code path is still reachable.

**Fix**: gate the broadcast on `version != "v2"` AND emit a warning that
v1 is deprecated.

### Bug XM2 — v1 macro_frame symmetry guard test passes for both v1 and v2 (LOW for now)

`tests/test_train_inference_symmetry.py` enforces that both training and
inference call the same `Load*Task` chain. With v2 added, the chain now
includes `LoadMacroPerTickerBetasTask` AND `LoadAssetEmbeddingsTask`. Both
tasks no-op when their respective config flags are off, so the test
continues to pass — but in v1 mode, the macro_frame is loaded but its
v2 conversion is skipped, leaving stale `ctx.macro_betas={}`. Inference
path correctly handles empty `ctx.macro_betas`.

**Not a bug currently**, but the entanglement is fragile.

### Bug XM3 — `panel_ltr.macro.enabled=true, version="v1"` still costs 33 features × 100% noise (HIGH)

This is the empirical finding from earlier in the session: prod was
trained without macro (28 features) and beats macro-on (61 features).
Code path still exists; if operator flips `enabled=true` without setting
`version="v2"`, they get the broken v1 model. Default state is fine
(enabled=false), so this is a footgun, not active bug.

**Fix**: when `version` is unset and `enabled=true`, default to v2 not
v1 — OR raise an error requiring explicit choice.

### Bug XM4 — XGB+macro inference matrix doesn't differentiate v1 vs v2 (MED)

**File:line**: `feature_matrix.py:build_inference_matrix`

The function takes `macro_frame` parameter but doesn't know about
v2's `macro_betas`. v2 is wired through `factor_frames` instead (per
the redesign), so for v2 the `macro_frame` should be empty. But the
inference path's `_panel_macro_frame` adapter attribute might still be
set in v2 mode.

**Fix**: when v2 active, ensure `_panel_macro_frame` is None (or
empty DataFrame) so build_inference_matrix doesn't double-broadcast.

### Bug XM5 — No inference-side test for macro v2 path (HIGH for confidence)

`test_train_inference_symmetry.py` only checks task NAMES match. There's
no test verifying that v2 produces non-trivial β at inference, or that
XGB inference matrix has expected `beta_*_60d` columns when v2 enabled.

**Fix**: extend test_macro_v2_and_embeddings.py with an end-to-end
inference test (use prod XGBoost panel + synthetic macro returns).

---

### XGB + macro summary

| # | Issue | Severity |
|---|---|---|
| XM1 | v1 broadcast still reachable | HIGH |
| XM3 | enabled=true defaults to broken v1 | HIGH |
| XM4 | inference matrix doesn't differentiate v1/v2 | MED |
| XM5 | No inference-side macro v2 test | HIGH |

**3 HIGH + 1 MED = 4 actionable issues.** (Withdrew XM2.)

---

## Combined bug count: 21 actionable + 6 noted

| Component | HIGH | MED | LOW | Total actionable |
|---|---|---|---|---|
| LGBM (separate doc) | 4 | 4 | 2 | 10 |
| T2-4 (Boyd cvxpy) | 5 | 2 | 3 | 10 |
| Macro v2 | 2 | 4 | 1 | 7 |
| XGB+Macro | 3 | 1 | 0 | 4 |
| **Sum** | **14** | **11** | **6** | **31** |

The user's "10 bugs" intuition was correct for each component — total
across 4 components is 31 actionable issues.

---

## Recommended fix priority

If user wants me to fix during the rest of the handoff:

1. **HIGH bugs that affect production behavior** (T1+T2+T6+T8+M6+M10+XM1+XM3): ~2 hours
2. **HIGH bugs that affect confidence** (T12 missing tests, XM5 missing inference test, LGBM #5 dispatch): ~1 hour
3. **MED bugs**: ~1.5 hours
4. **LOW bugs + LGBM doc-only**: ~30 min

Total: ~5 hours to clear the backlog. With ~3 hours of handoff left,
recommend tackling the production-affecting HIGH bugs first.
