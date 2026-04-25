# Panel + rotation pipeline self-audit (in-flight, 2026-04-24)

User asked: "你的 panel model pipeline 一定还有 100 个 bug 吧？" Doing a
proactive audit while the 99-ticker FullTrainingPipeline runs in the
background. This doc lists pattern-level concerns and concrete bugs;
each gets either FIXED or queued for the deep audit next session.

## A. PATTERNS THAT ENABLE SILENT BUGS

### A1. Hardcoded column lists in 5 places per feature

Adding a new feature class (e.g. `m_*` minute features) requires
updating:

  1. `*_FEATURE_COLS` constant in `compute_*_features` module
  2. `TickerPanelFactorJob.run` — panel-time append into raw_factor_frame
  3. `FactorZScoreTask.raw_cols` — z-score computation
  4. `FactorZScoreTask` output assembly loop — emit `*_z` cols
  5. INFERENCE path: `prepare_inference_panel_frames` must call the
     matching `LoadXBarsTask`

Already missed places 3+4 (Bug 8) and place 5 (Bug 12). Each silent
no-op cost a real A/B cycle.

**Recommended fix (next session):** central feature-registry. Each
feature class registers `(load_task, compute_fn, output_cols)` once;
all 5 uses derive from the registry.

### A2. try/except as silent ticker-drop

`TickerPanelFeatureJob.run` and `TickerPanelNeutralizeJob.run` both
have `try/except Exception as exc: log.error(...); continue`. If
many tickers fail (e.g. data fetch race, permission error), the panel
silently shrinks. No counter, no alert.

**Recommended fix:** count failures in `ctx.failed_tickers`, alert if
fraction > 10% of watchlist.

### A3. NaN at inference vs train distribution mismatch

`build_inference_matrix` auto-fills NaN for any feature_cols not in
the input (line 99-100). Trees handle NaN via "default direction".
But this assumes NaN is a *frequent* pattern in training. If a
specific ticker has full data in training but a NaN m_* col at
inference (e.g. minute fetch failed for that ticker yesterday), its
panel score routes through the rare-NaN branch — could be very
different from the "real m_*" branch.

**Mitigation:** alert when `(NaN cols × tickers)` exceeds threshold
at inference time. Not a code bug but an operational risk.

### A4. NGBoost feature_cols ≠ Panel feature_cols (potential)

`PanelLTRModel.feature_cols` is saved in `panel-ltr.json`.
`NGBoostHead.feature_cols` is saved in `ngboost-head.json`.
Both are trained on the same panel — by code path, they should be
identical. But there's no invariant check at load time. If a future
refactor diverges them (e.g. NGBoost on a different feature subset),
sigma estimates will be on a different basis than μ.

**Recommended fix:** assert `panel.feature_cols == ngboost.feature_cols`
in `LoadNGBoostTask`.

## B. CONCRETE BUGS (this session, fixed)

| # | Bug | Severity | Fix commit |
|---|-----|----------|------------|
| 1 | V4 thesis_symmetric scoring_mode dead branch | HIGH | `92f7726` |
| 2 | own_momentum dict not populated | HIGH | `92f7726` |
| 3 | IC baseline misreported (0.0326 vs actual 0.0391) | HIGH (claim) | `92f7726` |
| 4 | V2 test fixture used unrealized → tax drag killed signal | MED | inline |
| 5 | panel_exit V2 thresholds too tight → 0 fires (parameter, not code) | LOW | n/a |
| 6 | rotation 0 fires (candidate-supply bottleneck, not code) | LOW | n/a |
| 7 | rotation.bear_only check before regime gate (by design) | LOW | n/a |
| 8 | **m_* minute cols silently dropped in FactorZScoreTask** | **HIGH** | `a55df54` |
| 9 | **fetch_minute_bars skip-cached one-sided check** | **HIGH** | `49d351f` |
| 10 | **EmitRotationsTask non-atomic: sell-then-buy-fail** | **HIGH** | `d2fab30` |
| 11 | RunnerAdapter ctx._db not wired (V4 dead in live) | HIGH | `8564fcb` |
| 12 | **prepare_inference_panel_frames missing LoadMinuteBarsTask** | **HIGH** | `8e0592a` |

12 bugs found + fixed. 7 of the 12 were SILENT (looked correct, ran
clean, but did the wrong thing).

## C. POTENTIAL CONCERNS NOT YET INVESTIGATED (queued for next session)

- C1. **Adapter wire-up parity** — LeanAdapter doesn't have ctx._db
  (intentional, no SQLite in Docker), but V4 will silently fail there
  with the warning we added. Need to either route V4 through a
  cache-able artifact OR document V4 = sim+live only.
- C2. **`max_positions_per_sector=6`** with 24-ticker software bucket
  caps SaaS exposure to 25%. Software is the highest-IC bucket on
  current panel. May be a bottleneck. A/B candidate.
- C3. **OHLCV freshness during retrain** — train uses cache last-fetched
  whenever; sim uses cache as-of-now. If retrain runs BEFORE today's
  bar, training distribution lags sim by 1 day. Worth pinning.
- C4. **Recalibration silent skip** — RecalibrationJob has its own
  try/except. If panel didn't write its artifact (cascading failure),
  recalibration silently produces stale blend weights.
- C5. **rs_score retired but still computed** — `rs_score` is on
  CandidateResult for log readability but always weighted 0. Is
  actually-zero value being used anywhere by accident?
- C6. **NGBoost overflow warnings** — `RuntimeWarning: overflow
  encountered in square` fires every panel retrain. ngboost's σ²
  occasionally numerical-blow-up on extreme μ. Suppressed silently
  via fillna. Worth investigating whether values become inf/nan in
  the saved artifact.
- C7. **Sector momentum residualization** uses `sector_momentum` keyed
  by sector. With our new sub-sectors (giant_tech / ai_chip /
  datacenter_hw / software), each sub-sector's sector_momentum has
  fewer constituents. ai_chip with 18 tickers has decent breadth;
  giant_tech with 8 is okay; datacenter_hw with 10 is fine. But
  "commodity" with 1 (just GLD) → sector momentum is just GLD's
  return → residualization is degenerate (residual ≈ 0). Could be
  numerically unstable.
- C8. **No invariant check** that all watchlist tickers actually
  appear in sector_map at training start. A typo would silently drop
  a ticker from sector_momentum → its rel_mom_*  / trend cols not
  neutralized properly.
- C9. **Earnings filter** uses `earnings-calendar.json` artifact. New
  tickers added today fetched their earnings via
  fetch_earnings_calendar.py — but did that ALSO refresh the artifact?
  If not, new tickers silently bypass the ±3d earnings filter.

## D. CONCERNS USER WILL ASK ABOUT NEXT

- D1. After 99-ticker retrain, expect APY drop possibility (covered in
  earlier message — sector cap, lower-quality tail tickers).
  Mitigation if drop: bump `max_positions_per_sector` from 6 to 8.
- D2. Rotation may FINALLY fire on the 99-ticker panel — more
  candidates above A-gate. V1 / V4 / Sharpe modes ready to test.
- D3. Transformer retry on 99-ticker × hourly+minute panel will have
  ~107k rows. Still under the 200k gate, but closer. Likely still
  underperforms; revisit when watchlist grows further or history
  extends.

## E. WHAT I'M NOT CLAIMING

- 12 fixed bugs is unlikely to be all. The next-session deep audit
  should systematically walk every Task/Job/adapter and verify config
  flag → execution path completeness.
- Performance regressions can hide behind "+9.57 APY" type illusions
  (random-seed-amplified differences). Always re-run with
  snapshot=True isolated runs before promoting.
- The pattern of "5-place feature add" is brittle. Until refactored,
  each feature add is a 5x bug surface.
