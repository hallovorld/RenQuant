# Long-Short Architecture (Phase 2, 2026-05-13) — SHELVED

> **🛑 OBSOLETE 2026-05-17 — SKIP verdict.**
>
> Phase 1 empirical re-test (commit `28251c2`, 2026-05-17) showed:
> - Model bottom decile 60d-ann return = **+0.58%** (POSITIVE)
> - Model top decile = +17.77%
> - All alpha on LONG side; short side has no negative alpha to harvest
> - Kelly-Gu-Xiu 2020 RFS standard: short alpha needs −10% to −15%/yr to justify infrastructure
> - **Verdict: SKIP. Saves 3-4 weeks engineering.**
>
> Roadmap moved long-short to CLOSED/REJECTED. The design below is kept for historical
> reference in case the model class changes and short-side alpha re-emerges.
>
> See `doc/research/failed-experiments-log.md` + `doc/research/2026-05-14-longshort-clean-FINAL.md`.

---

# Original design (historical reference)

Strategy currently long-only. Phase 1 empirical gate showed
cross-sectional L-S spread +29.2%/yr (doc/research/2026-05-13-short-phase1-gate.md)
— but the bottom-decile re-test 2026-05-17 reversed the verdict (above).
This document specifies the architectural changes to enable a
**risk-controlled market-neutral long-short** strategy, should it ever be revived.

## Design principle: long-short as a SCALE on the existing rank

The strategy already produces per-ticker rank scores via panel-LTR.
Currently long = top K by rank. Phase 2 extends to:
- Long top K (positive ranks)
- Short bottom K (negative ranks)
- Sector-neutral net exposure

This is **NOT** a separate short signal. We re-use the same model;
just add the bottom half to the QP feasible region.

## Architectural changes (Phase 2A scope)

### 1. QP optimizer — allow w_lower < 0

**File**: `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
- `ComputeQPConstraintsTask`: read `_qp_w_lower` from config (currently 0)
- Add per-asset short cap: `max_short_pct` regime parameter (default 0)
- Add gross exposure limit: `max_gross_exposure` (default 1.0, max 1.5)
- Add net exposure limit: `max_net_exposure` (default 1.0, neutral=0)

**Config schema (new)**:
```json
"long_short": {
  "enabled": false,        // off by default
  "max_short_pct": 0.05,   // max 5% short per name
  "max_gross_exposure": 1.30,  // 100% long + 30% short
  "max_net_exposure": 0.85,    // beta-neutral leaning
  "sector_neutral": true,
  "max_per_sector_short_pct": 0.10,
}
```

### 2. Sector-neutral hard constraint

**File**: `kernel/portfolio_qp/tasks.py` (new task)
- `BuildSectorNeutralConstraintTask`: enforce `Σ(w_i for i in sector_s) - Σ(market_weight_i for i in sector_s) ∈ [−ε, +ε]` per sector
- Use existing GICS sector mapping
- Reuses sector matrix already built by `BuildSectorConstraintMatrixTask`

### 3. Position-sign-aware sell logic

**File**: `backtesting/renquant_104/kernel/exits.py`
- `check_stop_loss`: currently `price < entry × (1 − stop)` for long. Add: for short positions (`shares < 0`), check `price > entry × (1 + stop)`.
- `check_trailing_stop`: same logic flip
- `check_max_hold_days`: same for both
- `check_single_day_loss`: long = down move, short = up move

### 4. Short candidate selection

**File**: `kernel/pipeline/job_buy.py` (or new short job)
- After ranking, take bottom K candidates (ranked from worst predicted)
- Apply same wash-sale / cooldown / earnings blackout filters
- Output: list of short candidates passed to QP

### 5. Tax accounting (deferred to 2E)

Short positions:
- Always short-term capital gains (§ 1233)
- 30-day wash-sale rule applies symmetrically (§ 1091 + § 1233)
- IRS rules require treating short proceeds as basis

### 6. Borrow fee accounting (deferred to 2F)

Per holding day on short positions:
- `borrow_fee_bps_per_year × days_held / 365` charged to NAV
- Alpaca paper sandbox returns this in account data

## Phase 2A test plan (TODAY)

1. **Unit tests** (per task above):
   - `test_qp_allow_shorts.py` — pin: w_lower < 0 propagates to solver
   - `test_sector_neutral_constraint.py` — pin: violation rejected
   - `test_exits_short.py` — pin: stop-loss fires on UP move for shorts

2. **Smoke sim** (1 window, ~17 min):
   - Config: `sim_long_short_ext.json` with `long_short.enabled=true`,
     gross=1.30, net=1.00, sector_neutral=true
   - Verify: portfolio has both long + short positions, sells trigger
     on right direction, no NaN/inf, equity curve continuous

3. **Full 16-window sim** (~70 min):
   - Same config, all 16 windows
   - Compare paired daily Δ vs baseline (long-only)

4. **Tier verdict + decision**:
   - Tier 3 → can promote (BUT needs Phase 2D-2I first)
   - Tier 2 → continue to Phase 2D engineering
   - REJECT → debug or reassess

## Risk constraints (hard-coded)

1. Max gross exposure: 1.5 (150% gross = max 100% long + 50% short)
2. Max single short position: 5% of NAV
3. Sector net exposure: |sector_net_w − sector_target| < 5%
4. **Live broker safety**: refuse to flip live config without:
   - Tier 3 verdict on sim
   - Phase 2D-2I complete
   - User explicit `--enable-shorts` flag in config
5. **Reg-T 150% margin guard**: drawdown_halt triggers if gross exposure
   × MaxDD > 50% (force-close before margin call)

## What's OUT of Phase 2A scope (today)

- IRC §1233 wash-sale (Phase 2D)
- Short-term tax treatment (Phase 2E)
- Alpaca locate / borrow fees (Phase 2F)
- Reg-T margin guard (Phase 2G)
- Live deployment (Phase 2I, gated on user OK)

These all REQUIRED before live config flip but NOT required for sim
backtesting.

## Engineering acceptance criteria (Phase 2A)

- All new tasks ≤ 50 lines (per §1c)
- Each new feature behind a config flag (off by default)
- Pin regression tests for each invariant
- Smoke test 1 window must pass
- Documentation updated

## References

- Grinold-Kahn 1999 *Active Portfolio Management* §5 — IR_LS = IR_long × √2
- Kelly-Gu-Xiu 2020 *RFS* §3.4 — empirical 2× Sharpe boost
- Fama-French 1993/2015 — long-short factor construction
- IRC §1091 (wash-sale), §1233 (short sale wash-sale)
- Reg-T 12 CFR 220 (margin requirements)
- Alpaca Markets API docs — paper sandbox supports shorts
