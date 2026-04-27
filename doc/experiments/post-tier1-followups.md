# Post-Tier-1 Follow-ups — 2026-04-25 (evening session)

Living tracker for everything caught during the deep audit + B+E + Tier 1
work but NOT fully resolved. Don't lose these.

## Context

This evening session (2026-04-25 PT, post-dinner), we executed:
- 12+ bug fixes across regime confidence / sizing / state reconciliation /
  calibrator config-key (RC-MISMATCH, CONF-MULT, MIN-1-SHARE, STATE-GC,
  UNMANAGED-NTFY, ENTRY-DATE-FROM-FILLS, ENTRY-DATE-BACKFILL, CAL-7-PATH,
  STATE-GC-NEWBUYS, EXITS-FAIL-DB, CONF-MULT-NONE, CAL-7-TIMEOUT, plus 2
  test regressions fixed)
- Tier 1 panel-LTR retrain (drop 13 noise/sparse features, +book_to_price_z:-1
  monotone, training_window 5.0→1.5y)
- Roadmap entries for Tier 2/3/4
- Web research (TGNS, FASCL, contrastive embeddings, Bagnara critical review)

In-flight at handoff: Tier 1 RefreshPanelCalibratorTask (panel-rank-calibration
fit) computing OOS pool_IC.

## Open follow-ups (do AFTER Tier 1 train completes + iter3 demo)

### 🔴 Code/test debt (Agent C audit findings, only PARTIALLY shipped)

| ID | Item | Why deferred | Effort |
|---|------|---|---|
| **DBT-1** | ENTRY-DATE-FROM-FILLS uses `broker.get_filled_orders(limit=100)` — old positions held >100 trades back can't be seeded from broker. Need pagination OR fall through to sentinel more gracefully (currently shouted with warning). | Edge case; affects only long-tenure positions held across many trades. | 1-2 hours |
| **DBT-2** | ENTRY-DATE-FROM-FILLS / BACKFILL has no sell-then-rebuy lifecycle awareness. Uses earliest BUY → re-bought positions get incorrectly extended tenure (bypasses min_hold + LT-tax discount). | Broker fill list doesn't trivially expose "this is a fresh trip vs continuation". Needs match-fill-to-current-position-qty algo. | 1 day |
| **DBT-3** | 8 missing test groups for runner-only fixes: STATE-GC, STATE-GC-NEWBUYS, ENTRY-DATE-FROM-FILLS, ENTRY-DATE-BACKFILL, UNMANAGED-NTFY (live/runner.py side), EXITS-FAIL-DB. Per CLAUDE.md §2 every fix needs a regression test. We covered RC-MISMATCH/CONF-MULT/MIN-1-SHARE in `tests/test_regime_confidence_fix.py` (22 tests) but the runner-side ones rely on smoke testing. | Time pressure during E2E iteration loop. | 1 day for full coverage |
| **DBT-4** | `compute_regime_confidence` can return values < 0 if `gmm_probs[regime]` is negative (shouldn't happen but no defense). Downstream `confidence_to_size_multiplier` clamps to floor 0.5, so impact is bounded; still better to floor at 0 in the source. | Low impact, defense-in-depth. | 30 min |

### 🔴 Operational state

| ID | Item | Why deferred | Effort |
|---|------|---|---|
| **OP-1** | Add `GLD` to `earnings_surprise.skip_tickers` (Agent B audit recommendation — only ticker with 100% NaN on earnings_surprise_cum_z, missed during Tier 1 batch). | Rolled together too tightly, missed in commit. | 30 sec config edit |
| **OP-2** | Stale HWM auto-snap fires every bar — root cause analysis. Either real prior drawdown (legitimate) or buggy fill (need historical investigation). RU-1 fix masks the symptom but root cause unknown. | Requires Alpaca historical equity curve + portfolio-value DB analysis. | 2-3 hours |
| **OP-3** | BA position unmanaged at broker. UNMANAGED-NTFY now surfaces it on phone, but operator must manually decide: hold / sell / add to watchlist. Currently $697 exposure with no stop-loss / trailing-stop logic. | User decision (not code bug). | User input |
| **OP-4** | AMD not in watchlist. User-noticed during deep audit. May be intentional or oversight. | User decision. | User input |
| **OP-5** | NGBoost RuntimeWarning observed during Tier 1 train: `overflow encountered in square self.var = self.scale**2`. Single sample with very large σ scale. Won't affect μ_mean, but could corrupt single-ticker σ → bad Kelly target → wrong sizing. | Need to inspect ngboost-head.json for any inf/NaN σ entries. | 1 hour |

### 🟡 Feature engineering iteration (Agent B Tier-C recommendations)

| ID | Item | Why deferred | Effort |
|---|------|---|---|
| **FE-1** | Re-run FeatureDiagnosticTask on post-2024-only slice for the 11 surviving intraday features (morning_drift_z, afternoon_drift_z, m_*, etc.). Tier 1 training showed several have weak IC; Tier 1 panel restricted to 1.5y already cuts the worst, but ablation could trim more. | Iterative refinement; needs another retrain to verify. | 1-2 retrain cycles |
| **FE-2** | `earnings_surprise_cum_z` has IC=+0.001 and std=0.578 (collapsed). Either drop entirely (Tier-C from Agent B) or fix data infrastructure (yfinance .earnings_dates incomplete coverage). | Need to decide: drop or invest in data. | 30 min decision + retrain |

### 🟢 Validate Tier 1 result (this session, blocking next steps)

| ID | Item | Effort |
|---|------|---|
| **V-1** | Capture Tier 1 OOS pool_IC vs historical 0.066. If ≥0.08, Tier 1 is a confirmed win → consider promoting to golden after sim verification. | Wait for current train to finish (~5-10 min remaining as of writing) |
| **V-2** | Run iter3 (live --once full E2E) with new artifacts. Compare candidate ranks for NVDA/MSFT/MU vs JNJ vs RTX between iter1/iter2/iter3. | 5-10 min compute |
| **V-3** | Sim verification: run `sim/runner.py` 27-mo OOS with new panel + new calibrator. Compare APY vs golden v4.1 (+39.82% APY baseline). Per CLAUDE.md §2a: if APY ≥ baseline, promote; otherwise audit before accepting (§2b). | ~30-60 min sim |

### 📚 Roadmap items (longer horizon, in `doc/roadmap.md`)

- **T2-1** LightGBM LTR backend swap (highest-confidence: +128% IC validated 4-way audit)
- **T2-2** Contrastive asset embeddings as features (Dolphin 2024 KDD)
- **T2-3** Regime-conditional ensemble (Two Sigma 2024)
- **T3-1** TGNS Transformer + GNN (12-22% IC claim, CN A-share validated)
- **T3-2** FASCL contrastive (Feb 2026 paper, code release pending)
- **T4-1** LLM-generated factors (deferred indefinitely)

## Decision tree at handoff

```
Tier 1 train completes
  ↓
Calibrator pool_IC reported
  ├── ≥ +0.08 OOS → V-2 (iter3 demo) → V-3 (sim) → if APY ≥ golden, promote
  ├── ≈ +0.066 (no improvement) → audit per CLAUDE.md §2b before accepting
  └── < +0.05 → revert (training_window too short / overfit) → try training_window=2.0
```

## Where to pick up

1. Read this file's "Open follow-ups" section.
2. Check `/tmp/train_104_TIER1.log` for the calibrator's `pool_IC` line.
3. Check `git log --oneline -20` to see this session's commits.
4. `/tmp/feature_quality_audit.md` and `/tmp/code_audit_2026-04-25.md`
   have full Agent reports if you want to dig deeper.
