# 2026-05-14 — Long-Short Phase 2 master plan + experiment pairing

## Current state inventory

### ✅ Done (Phase 2A — purely code-side, OFF by default)

| Component | File | Status |
|---|---|---|
| `long_short.enabled` config flag | `kernel/portfolio_qp/tasks.py:330` | Wired, default false |
| Negative `_qp_w_lower` allowed | `kernel/portfolio_qp/tasks.py:336-343` | When `long_short.enabled=true` and regime ≠ BEAR |
| Gross-exposure cap (Σ\|wp\| ≤ 1.30) | `kernel/portfolio_qp/qp_solver.py:119, 254-258` | CLARABEL-compatible |
| Short-aware stop loss | `kernel/exits.py:482-490` | Tested 5 invariants |
| Short-aware single-day-loss gap | `kernel/exits.py:556-564` | Tested |
| Short candidate intake hook | `kernel/portfolio_qp/job_qp.py:113` | Reads `ctx.short_candidates` if present |
| G12 long-short parity gate | `kernel/model_acceptance_short.py` | Production-quality |
| BEAR regime skips shorts | `kernel/portfolio_qp/tasks.py:336` | Defensive — no shorts in crash |

Tests passing:
- `tests/test_exits_short_aware.py` (5 tests)
- `tests/test_qp_long_short_phase2a.py`
- `tests/test_model_acceptance_short.py`

### ❌ NOT done (blockers to actually shorting in sim/paper/live)

| Component | Why blocked | Effort |
|---|---|---|
| **ShortCandidateJob** (populates `ctx.short_candidates`) | DOES NOT EXIST — `ctx.short_candidates` is always `None` | 1d code |
| **Full-universe panel scoring** | Current panel scores 103 stocks; need broader (R1K) for short pool quality | 2d code + retrain |
| **§1233 wash-sale on shorts** | IRS rule different than long wash-sale (30-day "covered" lookback) | 1d code |
| **Short-term tax (shorts always ST)** | Even shorts held > 1 yr are ST per IRS | 0.5d code |
| **Alpaca borrow/locate API** | Need to call /v2/positions to check shortable status | 1d code |
| **Reg-T margin checks** | Must verify buying power before short order | 0.5d code |
| **Phase 2C smoke sim** | 1-window short-only decile spread test | 0.5d |
| **Paper test** | 2 weeks observation on Alpaca paper | 14d wallclock |
| **Live flip** | User OK + monitor | 1d |

**Total Phase 2B-2I**: ~7 days code + 14 days paper test = **3 weeks min**.

### Production status

- `long_short.enabled` not present in `strategy_config.json` → defaults to false
- Live strategy is long-only (verified today's GE buy was long-only)
- The Phase 2A code is loaded but dormant

## Experiments that MUST be paired with shorts

| Experiment | Why pair with shorts | Why short-less is meaningless |
|---|---|---|
| **wl174 universe expansion** | Bigger universe → better short candidates (bottom-decile filter) | Long-only wl174 already tested, NEITHER |
| **Sector-neutral** | Can be truly sector-neutral with longs+shorts; long-only sector-neutral degrades to cash | Long-only is just sector cap (which I tested) |
| **Pair trades** | Requires shorts by construction | N/A |
| **Hedge-ratio sweep** | Requires shorts | N/A |
| **GK_conditional regime dispatch** | Could go LONG in bull regime, SHORT bottom-decile in bear regime | Long-only conditional was tested, NEITHER |
| **Vol-target with hedge overlay** | True vol target works via beta-hedge shorts | Long-only vt15 only de-leverages, doesn't hedge |
| **Tail-risk hedging (Q1-style crashes)** | Q1 2022 SPY −50% — shorts could turn baseline's +25pt → +50pt | Long-only just goes to cash |

## Experiments that DON'T benefit from shorts

| Experiment | Why short-irrelevant |
|---|---|
| Friction knobs (κ, min_dw) | Both sides pay same friction |
| EMA50 gate | Macro gate works regardless of direction |
| risk_aversion γ | Symmetric in QP objective |
| max_position_pct | Already tested both sides via gross_max |
| Signal upgrades (LightGBM, alpha158→richer) | Same μ feeds both directions |

## Q1 2022 thought experiment: how shorts would change baseline

Baseline Q01 2022: SPY −50%, baseline −24.76% (alpha +25pt by going mostly cash).

With shorts enabled, in a window where bottom-decile of panel score is reliable:
- Long 70% with current portfolio
- Short 30% in bottom-decile names
- Bottom-decile during 2022 crash (TSLA, NFLX, PYPL etc.) lost 60-80%
- Short-side P&L would have been +18-24% × 0.30 = +5 to +7pt
- New baseline: −24.76 + ~+6 = **−19% APY**, alpha ≈ +31pt
- **Even more dramatic in Q1 2022 alone** (3 months SPY −16% → baseline likely −6 or so → improvement of +12pt)

## Revised experiment plan

### Priority A — Build Phase 2B (1-2 days) → unblocks the right experiments

1. **ShortCandidateJob** — pulls bottom-decile of panel score (mirroring TopCandidateJob)
2. **Wire `ctx.short_candidates` populator** — runs only when `long_short.enabled=true`
3. **§1233 wash-sale guard** — refuse re-short within 30 days of covering a loss
4. **Smoke sim Phase 2C** — single-window short-only decile spread on 2024 data

Phase 2C decision gate:
- If long_top − short_bottom decile spread is reliably positive over 16 windows → proceed to 2D-2I
- If decile spread is < short_borrow_cost → reject shorts, redirect to other tracks

### Priority B — Re-run the "structural" experiments WITH shorts (after 2B+2C pass)

Once shorts work, re-test on 16-window panel:

1. **wl174 + long_short** — broader universe + bottom-decile shorts
2. **GK_conditional + long_short** — regime-aware dispatch with shorts in BEAR-CALM
3. **Sector-neutral + long_short** — true sector-neutrality not possible without shorts
4. **vt15 + beta-hedge shorts** — true vol target via market-beta hedge

Each is a 16-window panel = 80 wallclock min. 4 panels = 5.3h.

### Priority C — Phase 2D-2I (3 weeks min before live)

1. **2D wash-sale + ST tax** — code (1d)
2. **2E Alpaca borrow/locate + Reg-T** — code + paper API test (1.5d)
3. **2F-2G Paper trading** — 2 weeks live observation
4. **2H Live flip** — explicit user OK only

## What about the current sim queue?

3 panels still running (maxpos10, sectorcap4, ntband2x). They're long-only experiments. They'll complete in ~3-4h regardless.

Most likely all 3 NEITHER per the established pattern. Once done, structural pivot opens. The shorts work (Priority A) is the next natural step.

## ASCII timeline

```
[t=0 now]        [+4h]              [+1d]                [+1.5d]              [+1w]                  [+3w]
─────────────────────────────────────────────────────────────────────────────────────────────────────────
maxpos10        analyze final 3 → structural pivot decision
sectorcap4     │
ntband2x       ↓
              [decision]──→ Priority A (Phase 2B): ShortCandidateJob + §1233 + smoke ──→ decision
                                                                                          │
                                                                                          ↓
                                                                                          Priority B: re-run
                                                                                          structural experiments
                                                                                          WITH shorts ──────→
                                                                                                              │
                                                                                                              ↓
                                                                                                              Phase 2D-2I
                                                                                                              code + paper
                                                                                                              + live
```

## Hand-off decisions

1. **No promotes from current 7 panels** (correctness, not script bug — see self-audit doc)
2. **Continue current 3 panels** to completion (~3-4h)
3. **After queue finishes**: build Phase 2B (ShortCandidateJob etc) — 1-2 days code
4. **Phase 2C smoke sim**: decide if shorts add edge
5. **If yes**: re-run structural experiments with shorts (Priority B)
6. **Live shorts**: still requires Phase 2D-2I + user OK (no auto-promote per exclusion list)
