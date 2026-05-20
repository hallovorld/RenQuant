# 2026-05-15 — Session Close (regime architecture + MA200-gate fix)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## Outcome

Panel A regression fully fixed. 17 commits today, ranging from a fresh
architectural mandate (PRIME DIRECTIVE: regime-conditional) through a
4-iteration debug trajectory that landed the MA200-gate fix.

| Run | n | Mean Δ_APY vs GMM | Notes |
|---|---|---|---|
| Original GMM (May 12 baseline) | 16 | 0pt (reference) | mean APY +5.43% |
| Panel A (MA50 fix, broken) | 16 | **−4.10pt** | Q11/Q15/Q10/Q04 catastrophes |
| MA200-gate fix (this session) | 16 | **−0.33pt** (p=0.58, flat) | mean APY +5.11% |

**Net recovery: +3.78pt mean** vs broken Panel A.

## What's in the codebase now (all committed)

### Foundation (Phase 0)
- `kernel/regime_resolver.py` — per-regime knob overlay (PRIME DIRECTIVE pattern)
- `kernel/regime.py + pipeline/task_regime.py` — MA200 gate, direction-aware Hurst
- `kernel/regime_hmm.py` — Hamilton 1989 HMM forward-filter (alternative detector)
- `scripts/train_spy_hmm.py` — HMM trainer (saves to artifacts/sim/spy-hmm-regime.json)

### Shorts (Phase 1 partial)
- `kernel/portfolio_qp/tasks.py::ComputeQPConstraintsTask` — P1a routes
  `long_short.enabled` via per-regime overlay; BEAR-OFFENSIVE hybrid
  (option γ) implemented (hard_bear=True → shorts allowed; default off)
- vectorbt cross-validators in `tests/test_*_pnl_vectorbt_validator.py`
- `kernel/pipeline/task_gates.py::BEARBranchTask` — soft-gate (didn't
  fix Q11 but useful for future regime-conditional precision)

### Safety
- `max_gross_exposure` hardcapped at 1.0 (no-leverage)
- `regime.bear_branch_legacy_mode` escape hatch for soft-gate

### Validation
- 112+ tests across regime suites all passing
- 4 new test files (regime_resolver, regime_hmm, regime_direction_aware
  with MA200, bear_branch_soft_gate)

## What's NOT fixed (acknowledged trade-offs)

1. **Q04 BULL_CALM still −7.64pt vs GMM.** MA200 partially recovered
   (+2.37pt vs MA50). Remaining: 3 BEAR + 14 BULL_VOLATILE labels in
   Q04 cause modest size-mult drag. Could be cleaned up with HMM
   smoothing (Phase 1b).

2. **Q12 BULL_CALM regressed −10.99pt vs MA50.** MA50's helpful BEAR
   detection in Q12 (which gave +11.66pt) is filtered out by MA200.
   Precision/recall trade-off — accept losing marginal-bear wins in
   exchange for eliminating catastrophes.

3. **No alpha YET vs GMM baseline.** The detector now matches baseline;
   it doesn't beat it. The PRIME DIRECTIVE next step is **wiring per-
   regime KNOBS** (P1b-h: max_shorts, kappa, kelly_scale, vol_target,
   defensive_tickers, etc.) to actually convert detector signal into
   alpha. The detector fix is necessary infrastructure but not
   sufficient.

## Lessons (theory-grounded)

1. **Kaminski-Lo 2014** — confirmed empirically. 0/1 hard switching on
   noisy regime detector destroys momentum-period alpha. Q11 −27pt
   was the textbook failure mode.

2. **Garleanu-Pedersen 2013** — partial rebalance is the canonical fix.
   Our MA200 gate effectively damps the regime change frequency
   (achieves Garleanu-Pedersen's persistence requirement via the
   MA200's smoothing).

3. **Hamilton 1989 HMM** — built, ready for Phase 1b. Theoretically
   superior to our heuristic stack (transition_matrix is intrinsic
   smoothing; no MA200 hack needed). Empirically: smoke test showed
   HMM with current 4 features gives 0% BEAR coverage in Q06 — need
   credit-spread feature for full coverage. Punted.

## Tomorrow's priority (per roadmap.md P1b-h)

Per-regime knob wiring, ordered by per-regime spread (Δ_APY when
each regime's value differs from another):

1. `max_position_pct` (already per-regime, just tighten BEAR/CHOPPY)
2. `defensive_tickers` — add SHV/BIL for 2022-style stagflation bears
   (need OHLCV fetch first)
3. `kappa` per regime (Ang-Bekaert 2002 direction-supported)
4. `kelly_scale` per regime (literature: 0.25× canonical sweet spot;
   regime discount = folklore)
5. `vol_target` per regime (Moreira-Muir 2017 continuous formula is
   the canonical; layer regime caps)
6. `stop_loss_pct` per regime (Kaminski-Lo: tight in CHOPPY, wide in
   BULL_VOL)

## All commits this session (chronological)

```
5ee01e5  docs: MA200-gate postmortem — 4-attempt debug trajectory
a87f54a  fix(regime): require BOTH MA50 AND MA200 below for BEAR  ← WINNER
2447dcb  fix(regime): BEARBranchTask soft-gate — Kaminski-Lo 2014 (didn't help)
68db94d  sim config: softbear — P1d range-find (didn't help)
e00d576  docs: detector vs response function — counter-intuitive Panel A finding
2d55c44  fix(regime/hmm): align HMM cluster labels with codebase taxonomy
93c2f5c  sim configs: HMM detector A/B side configs
1285c95  fix(regime): HMM replaces stateless GMM — Hamilton 1989
054a572  P1a: per-regime long_short.enabled overlay + BEAR-OFFENSIVE hybrid
b70f2f6  docs(roadmap): lock BEAR hybrid design decision (option γ)
3d346b4  docs(roadmap): PRIME DIRECTIVE phase plan + P1 per-regime knob wiring
e3fd4a1  docs(CLAUDE.md): PRIME DIRECTIVE — RenQuant is regime-conditional
3925c0d  fix(regime): direction-aware Hurst (MA50 only — root cause of regression)
7f40316  fix(safety): hardcap max_gross_exposure at 1.0 — no leverage authorized
548c76e  test: vectorbt cross-validators for SimAdapter cash flow paths
```

## Production state at session close (2026-05-15 02:35 PT)

- `strategy_config.golden.json`: `long_short.enabled=false` (preserved)
- All launchd plists in PAPER mode (no real money)
- 130 commits ahead of origin/main
- All experiment artifacts in `data/logs/sim_2026-05-15_ma200_gate/`
- Full panel verification: ~5% mean APY, baseline-flat (regression fixed)
