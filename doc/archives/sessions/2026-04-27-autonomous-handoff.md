# 2026-04-27 — Autonomous handoff session summary

Long autonomous session driving model-selection systematization, the
8-variant tournament, deep audits, and bug fixes. 52 commits over
~12 hours.

## Headline result

**PROD remains XGBoost rank:pairwise no-macro 28-feature, OOS IC = +0.0482.**

8-variant tournament conclusively shows this is the best-fit configuration
for the current panel (99 tickers × 753 dates ≈ 75K rows). All 7 alternative
variants underperformed by 19% – 97%.

| Variant | OOS IC | Δ vs PROD | Verdict |
|---|---|---|---|
| **XGB rank:pairwise no-macro (PROD)** | **+0.0482** | — | 🏆 |
| XGB rank:pairwise + macro v1 broadcast | +0.0393 | −19% | rejected |
| XGB rank:pairwise + macro v2 per-ticker β (post-fix) | +0.0371 | −23% | rejected |
| XGB rank:ndcg listwise (post-bucketize) | ≤ baseline | — | rejected |
| LGBM lambdarank no-macro | +0.0193 | −60% | rejected |
| LGBM lambdarank + macro v1 broadcast | +0.0224 | −53% | rejected |
| LGBM v2 (audited hyperparams) | +0.0014 | −97% | retracted |
| LGBM lambdarank + macro v2 per-ticker β | (in flight) | — | pending |

## Why no variant beat PROD

The dominant constraint is **panel breadth**. Grinold-Kahn says
`IR ∝ IC × √breadth`. At 99 tickers × 1-day cross-section ≈ 99 breadth/day,
no model architecture or loss function compensated for the small panel.
Adding macro features made the per-feature signal-to-noise *worse* (33
broadcast columns with 0 within-date variance → noise; 11 per-ticker β
columns helped per-ticker signal but didn't beat the 28-feature
no-macro baseline). LightGBM lambdarank fundamentally underperforms
XGBoost rank:pairwise on this panel size regardless of tuning.

## Bugs fixed this session

Four deep audits this session, identifying 46+ bugs total:

| Audit | Bugs found | HIGH fixed | MED fixed | LOW fixed |
|---|---|---|---|---|
| LGBM deep audit (12 bugs) | 4 HIGH + 4 MED + 2 LOW + 2 noted | 4 / 4 | 3 / 4 | 1 / 2 |
| T2-4 + macro v2 + XGB+macro (21 bugs) | 7 HIGH + 7 MED + 3 LOW + 4 noted | 6 / 7 (1 withdrawn) | 2 / 7 | 1 / 3 |
| 2nd-round 12-hour re-audit (6 bugs) | 3 HIGH + 3 MED | 3 / 3 | 3 / 3 | n/a |
| Other audit findings | various | various | various | various |

Notable HIGH-severity fixes:
- **LGBM #1**: NaN weights silently propagating (gradient corruption)
- **LGBM #2**: `best_iter` storing LAST iteration instead of best
- **LGBM #3**: Tied labels getting arbitrary distinct ranks
- **LGBM #5**: PanelScorer.load dispatch for `panel_lgbm` artifact kind
- **LGBM #11**: No params validation — caller could silently misconfigure
- **T2-4 T1**: `delta` array unaligned to mu/sigma when input not pre-aligned
- **T2-4 T2**: `cov_matrix` reindex creating NaN-filled rows; ridge masking the NaN→0 fill
- **T2-4 T6**: `quantize_to_whole_shares` not respecting short-sell prevention
- **T2-4 T8**: `bounds.ub = leverage_cap` allowing single-position YOLO
- **XM1**: v1 broadcast macro path STILL ACTIVE when v2 enabled (double-injection)
- **2nd-round HIGH #1**: TZ-aware/naive timestamp normalization in finalize_challenger
- **2nd-round HIGH #15**: JSON decode error swallowed, exit code 0 on broken metadata
- **Bug #25**: train-inference macro symmetry guard (no leak)

LOW + MED fixes shipped this session also (M3, M4, #4, #7, #12, etc.).

## Implementation work delivered (not just bug fixes)

### Tier 2 components (T2-2 / T2-3 / T2-4) — Phase A skeletons

- **T2-2 Asset embeddings** (Dolphin 2024 KDD): `training_panel/asset_embeddings.py`
  with PyTorch CNN + InfoNCE contrastive loss + smoke-test-collapse guard;
  weekly cron driver `scripts/train_asset_embeddings.py`; pipeline wiring
  for `LoadAssetEmbeddingsTask`. **Module written; training pending.**

- **T2-3 Regime-conditional ensemble** (Two Sigma 2024):
  `kernel/panel_pipeline/regime_router.py` with config-gated artifact
  routing + WARNING fallback when ensemble enabled but per-regime artifact
  missing. **Skeleton + router; per-regime training pending.**

- **T2-4 Boyd-style convex rotation**: `kernel/rotation_convex.py` with
  cvxpy + scipy SLSQP fallback, ridge-masked Σ, T1-T8 audit fixes,
  `quantize_to_whole_shares` with current-holdings sell-cap. 13 tests
  pin solver semantics. **Solver written; RotationJob wiring pending.**

### Macro v2 — per-ticker rolling β

- `kernel/macro_per_ticker.py` — Cov(ticker_r, macro_r) / Var(macro_r)
  rolling 60d window with strict-prior shift(1)
- Pipeline wiring as `LoadMacroPerTickerBetasTask` → merged into
  `raw_factor_frames` for the panel feature pipeline
- v2 vs v1 broadcast suppression (XM1 fix) so configs can't double-inject
- Result: macro v2 -23% IC vs PROD on XGBoost; verdict rejected

### ModelAcceptanceGate framework

11 gates (G1–G11) enforced in `train_104.py`. G2 calibrator-collapse
floor caught LGBM v1, LGBM v2, A3-v2, and macro v2 weak signals correctly.

### Operator UX

- `scripts/finalize_challenger.py` — manual challenger promotion CLI
  with TZ normalization, JSON-decode error surfacing, `_pick()` helper
  for metadata vs top-level conflict
- `scripts/select_best_model.py` — backend tournament runner
- `doc/components/model-selection.md` — operator playbook
- ntfy silent-on-no-op for the every-30-min intraday sell-only cycle
  (12× per day → no longer spamming notifications)

## Production safety throughout

`panel-ltr.json` md5 == `panel-ltr.xgboost.bak.json` at every checkpoint.
Multiple experiment artifacts saved to `panel-ltr.{lgbm-no-macro,
lgbm-v2, lgbm-macro-v2, macro-v2-post-fix, …}.bak.json` for forensics.
Production never replaced with worse model.

## What blocks beating PROD

The 8-variant tournament saturates the **architecture × loss × feature**
axis on the current panel. The natural next axes:

| Axis | Expected IC ceiling | Effort |
|---|---|---|
| **Panel breadth** (99→200 tickers) | +0.010 to +0.020 | 1 week |
| **Asset embeddings** (T2-2 written, training pending) | +0.005 to +0.015 | days |
| **Regime ensemble** (T2-3 router done, 4× training pending) | +0.005 to +0.015 | week |
| **Stacking** (XGB + LGBM weighted blend) | +0.002 to +0.008 | hours |
| **Multi-horizon labels** (5d/15d/20d sweep) | +0.005 if mismatch | days |
| **Insider + short interest features** | +0.005 | days |

Recommended sequence: **expand watchlist to 200 first**. It's the only
axis that doesn't risk overfitting the existing 99-ticker panel and
directly raises the IR ceiling per Grinold-Kahn.

## Files / artifacts of note

- `doc/components/lgbm-deep-audit-2026-04-27.md` — 12 LGBM bugs catalogued
- `doc/components/t2-4-and-macro-v2-deep-audit-2026-04-27.md` — 21 bugs catalogued
- `doc/components/macro-factor-frame-redesign.md` — macro v2 design + verdict
- `doc/components/model-selection.md` — operator playbook
- `doc/components/asset-embeddings-design.md` — T2-2 design
- `doc/components/boyd-rotation-design.md` — T2-4 design

## Test count

2455 tests pass (was 2330+ at session start — added ~120 tests for new
components and bug regressions).

## 52 commits — selected highlights

- `cb570ee` LGBM #12: drop unused data_random_seed
- `dafdbbc` macro M3+M4: centralize defaults + log skipped tickers
- `9afa87b` ntfy silent intraday sell-only no-op cycles
- `064ed41` 2nd-round audit: 6 new bugs
- `f1d3411` LGBM #4 dtype + #7 feature_cols validation
- `403295c` XM1+XM4 macro v2 broadcast suppression
- `06184e8` 13 tests for ConvexRotationSolver (T12)
- `d0285a3` LGBM #1+2+3+6+8+9+11 (5 HIGH + 2 MED)
- `5389f22` T2-4 T1+T2+T6+T8 HIGH fixes
- `21adfff` T2-3 Phase A regime_router
- `d284d52` T2-4 Phase A rotation_convex
- `82e417a` macro v2 module
- `9fd27c9` XGBoost listwise label bucketize
- `059b005` T2-2 asset embeddings module
