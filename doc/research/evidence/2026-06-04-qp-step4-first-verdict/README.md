# 2026-06-04 — QP §8 Step 4 A/B Replay: FIRST verdict produced

**Milestone**: the QP Step-4 offline allocator A/B replay — diagnosed as
"built but non-functional end-to-end, no verdict exists" in
[`2026-06-04-qp-step4-replay-blocked-no-verdict.md`](../../2026-06-04-qp-step4-replay-blocked-no-verdict.md)
(#204) — **now runs end-to-end and produces a verdict**. This is the first
allocator comparison ever generated for RenQuant.

**Owner**: Claude. **Status**: framework functional; verdict is NOT
promotion-grade (see caveats §3).

---

## 1 · What unblocked it (the four #204 breaks)

| Break | Fix | PR |
|---|---|---|
| B3 — allocators unregistered | register `hybrid_option_f_allocator` + `hard_only_qp_allocator` | pipeline #38 / umbrella #210 |
| Task 4 — crash on 0 bars | emit `invalid_experiment.json` + rc=2 | pipeline #38 / umbrella #210 |
| **B2 — mu/sigma↔fwd never co-occur (load-bearing)** | backfill UNION `score_distribution` (date,ticker) | bt #36 / umbrella #211 |
| B1 — fwd_60d backfill | ran backfill (script already supported 60d) | data step |

**B2 had a second layer**: `scripts/backfill_forward_returns.py` delegates
to the **pinned** subrepo runtime, which was an older commit without the
#36 fix. The backfill only populated the verdict-needed rows when forced
through the fix-containing tree:

```bash
RQ_BACKTESTING_OPS_RUNNER=umbrella .venv/bin/python \
    scripts/backfill_forward_returns.py --source sim --cache-root data/ohlcv
# → 3229 new rows (the score_distribution sim-run dates)
```

After backfill, `mu+sigma+fwd_20d` co-occurrence went **0 → 3052 bars**
(was the load-bearing #204 B2 break).

## 2 · The verdict (fwd_20d, 2024-01-02 → 2026-03-27, 497 bars)

Command:
```bash
PYTHONPATH=backtesting/renquant_104 python -m kernel.portfolio_qp.run_ab_replay \
    --wf-artifact-root data/sim_runs.db \
    --start-cut 2024-01-02 --end-cut 2026-03-27 \
    --out verdict_fwd20.json \
    --allocators equal_weight_top_k,inverse_vol_top_k,fractional_kelly_top_k,hybrid_option_f_allocator,hard_only_qp_allocator \
    --incumbent fractional_kelly_top_k --fwd-horizon-days 20
```

Per-allocator (raw annualized Sharpe — see §3 caveat, treat as RELATIVE
ranking only):

| Allocator | Sharpe (raw) | mean daily ret |
|---|--:|--:|
| **hard_only_qp_allocator** | **8.83** | 0.0342 |
| hybrid_option_f_allocator | 8.75 | 0.0347 |
| fractional_kelly_top_k (incumbent) | 8.41 | 0.0272 |
| inverse_vol_top_k | 8.21 | 0.0266 |
| equal_weight_top_k | 7.36 | 0.0305 |

Paired vs incumbent (`fractional_kelly_top_k`):
- vs **hard_only_qp**: ΔSharpe −5.26, kelly win-rate 0.39 (z −4.98) → hard-only beats kelly
- vs **hybrid_option_f**: ΔSharpe −5.31, kelly win-rate 0.36 (z −6.15) → hybrid beats kelly
- vs inverse_vol: ΔSharpe +1.31, win-rate 0.23 → mixed
- vs equal_weight: ΔSharpe −1.88, win-rate 0.17 → equal-weight noisier

**Directional signal**: the two QP-family allocators (hard-only QP and
Hybrid F) rank above the current `fractional_kelly` incumbent.

## 3 · Why this is NOT a promotion verdict (caveats — §7.2 / §7.4)

`verdict.promotion_candidate = None`, `next_action = iterate`,
`constraint_fidelity.decision_grade = False`. Three reasons this verdict
must NOT drive a config change:

1. **`decision_grade = False` — no sector caps in the replay snapshot.**
   The WF loader cannot reconstruct per-cut `sector_map` (the #136 audit's
   deferred Step-4h work), so sector-cap regressions are invisible. A
   sector-cap-violating allocator could rank high here and the gate would
   miss it. The framework correctly fails closed (`promotion_candidate =
   None`).

2. **The Sharpe magnitudes (7–8.8) are not credible absolute values.**
   The 497 bars are sparse sim-run dates, not contiguous trading days;
   annualizing daily Sharpe on a sparse, autocorrelated sample inflates
   the number. Only the RELATIVE ranking is interpretable, and even that
   needs the §7.2 placebo triad.

3. **No placebo block (§7.2.1 R2).** No shuffle-label / time-shift /
   A-A was run. Per R2, none of these Sharpe numbers may be quoted as a
   finding in a commit/PR/status report without a companion placebo
   verdict. This README quotes them only to document that the framework
   PRODUCES output — not to claim hard-only QP is better.

## 4 · What it takes to make this promotion-grade (ordered)

1. **Step 4h — sector-cap snapshot** (the #136 deferred work): populate
   the replay `ConstraintSnapshot` with per-cut `sector_indicator` /
   `sector_cap_vec` so `decision_grade` can be True.
2. **Placebo triad** on the replay (shuffle-label, time-shift +2×horizon,
   A/A) — the loader / driver already has the `--loader-module` hook to
   inject a placebo bar source.
3. **Contiguous-bar Sharpe** or an explicit sparse-bar correction so the
   absolute magnitudes are interpretable.
4. **DSR / PBO** (already wired in `replay_significance`) reported as
   `Sharpe_raw / DSR / PBO` per §7.3.

## 5 · Bottom line

The original operator question — *"QP 昨天做了一堆修改，现在有什么提升吗?"*
— is now **answerable in principle**: the A/B replay runs and ranks the
allocators. The first directional read is that the QP-family allocators
(hard-only QP, Hybrid F) edge out the current fractional-Kelly incumbent,
**but the verdict is not decision-grade** until the sector-cap fidelity
(§4.1) and placebo battery (§4.2) land. The machinery is no longer
decoration; the measurement just isn't promotion-trustworthy yet.

`verdict_fwd20.json` in this directory is the raw artifact.
