# 2026-05-13 Baseline structural diagnosis — over-trading is the chronic lag

## Question

User flagged: 7 literature-backed candidates all REJECTED across 16-window
paired evaluation. "这不太正常 看看实验设计，看看baseline，是不是有结构问题
或者bug或者false assumption."

## Headline finding

**Baseline strategy has positive per-trade edge (+1.98% avg P&L) but loses
−7.5pt/yr to SPY** because turnover (~1.69 trades/bar) creates ~21% annual
friction drag that erases the edge. The QP solver's transaction-cost
penalty `cost_kappa = 0.0001` is **20-30× too low** vs realized per-trade
friction (~0.20-0.30%), so the QP is structurally incentivised to
over-trade.

## 16-window evidence

| metric | mean |
|---|---:|
| baseline APY | +5.43% |
| SPY APY | +12.96% |
| alpha vs SPY | **−7.53pt/yr** |
| beats SPY | 3 / 16 |
| turnover/bar | **1.69** (≈ 422 trades/yr on 20-name portfolio) |
| tax per quarter | $4,260 (= 4.26% NAV) → **17%/yr** |
| avg P&L per closed trade | **+1.98%** (positive edge) |
| avg hold | 23.9 trading days |
| corr(turnover/bar, alpha) | **−0.43** |
| corr(P&L/trade, alpha) | +0.22 |

Per-trade edge is positive. Alpha is negative. The gap is friction.

## Three compounding mechanisms

### Mechanism 1 — QP under-prices friction (primary)

`qp_cost_kappa = 0.0001` penalises `‖Δw‖₁`. For a typical Δw = 0.075 trade,
QP-penalty = 7.5e-6 of expected-return units. Real per-trade friction
(commission + slippage + occasional tax) ≈ 0.20-0.30%, so real penalty on
the same trade ≈ 1.5e-4 — **20× larger than QP thinks**.

QP sees benefit/cost ratio ≈ 200:1; it trades whenever expected return is
positive. Real ratio is ≈ 6:1; trades only pay off when expected return
exceeds ~0.20%.

**Fix:** bump `qp_cost_kappa` to ~0.002 (matches real per-Δw friction). QP
naturally rejects trades with expected return below threshold.

### Mechanism 2 — EMA50 gate blocks bull-market re-entries (secondary)

EMA50GateTask is hardcoded (no config flag). Across 16 windows:

| segment | n | mean alpha | mean EMA50 blocked% |
|---|---:|---:|---:|
| BEAR (SPY APY < 0) | 5 | +3.24pt | 54% |
| BULL + EMA mostly OFF (blk < 15%) | 5 | −5.75pt | 5% |
| BULL + EMA partial (blk ≥ 15%) | 6 | **−17.99pt** | 31% |

EMA50 helps in pure-bear windows (Q01 +25pt during −50% SPY drop) but
costs ~12pt extra in bull windows that briefly dip below EMA50 then rally
(Q04 −15pt, Q09 −43pt, Q13 −20pt).

**Decision deferred:** EMA50 is mechanism #2, not the primary. Test after
the cost_kappa fix lands.

### Mechanism 3 — no re-entry cooldown → averaging-down on losers

Q14 deep dive: TXN bought 15× (most of any ticker), ended quarter
−14% → contributed −22pt alpha alone. Buy-count-weighted alpha of
picks was **+11.74pt** vs SPY in Q14 — the ranking model picked winners
on average — but the realized portfolio lost 8pt absolute because
capital cycled through losers via repeated averaging-down.

**Decision deferred:** Test after the cost_kappa fix lands. May resolve
naturally once QP stops over-trading.

## Why "all 7 candidates failed" makes sense now

If the baseline is structurally over-trading and bleeding 21% friction
per year, then layering ANY signal-side improvement (universe expansion,
factor tweaks, regime conditioning, horizon swap) onto the same broken
execution layer cannot rescue alpha. The candidates failed for the same
reason baseline fails: their wins are eaten by the same friction.

**The experiment methodology was fine. The baseline is the bug.**

## Proposed fix (theory-aligned per §2a exception)

Single-knob change: `rotation.joint_actions.qp_cost_kappa: 0.0001 → 0.002`.

Per CLAUDE.md §2a exception clause: "theory-aligned wins where predicted
magnitude matches, and mechanism-clean changes with positive margin ship
even at < +2 pt." This is mechanism-clean (cost-penalty matches realized
friction) with a clear predicted magnitude (recover most of the 17%/yr
tax drag plus 5%/yr commission+slippage).

### Predicted outcome

If the fix works as theory predicts:

- Turnover/bar drops from 1.69 to ~0.5-0.8 (QP rejects sub-threshold
  trades that previously crossed friction)
- Tax drag drops from 17%/yr to 6-10%/yr
- Per-trade edge unchanged (+1.98%) but fewer no-edge trades
- **Predicted alpha lift: +8-12pt/yr** (recover ~half the friction drag)

### Verification protocol (DOE-light, single-knob screen)

1. Run 16-window paired-daily sim with `qp_cost_kappa = 0.002` (single
   config flip, no code change).
2. Compute Tier 1/2/3 verdict via `scripts/eval_paired_returns.py`.
3. Cross-check turnover stat — confirm it dropped per theory.
4. If mean Δalpha > 0 AND consistency ≥ 4/16 AND turnover dropped:
   - SCREEN (Tier 2) ⟹ promote to golden (per §2a exception)
   - Pin invariant: `cost_kappa ≥ ~real_friction_estimate`
5. If turnover dropped but alpha did NOT lift:
   - Investigate Mechanism 2 (EMA50) next.

### Range-finding plan (per §5.11)

Three top-down endpoints, single-window range-find first (Q14, the worst
case):

| run | cost_kappa | hypothesis |
|---|---:|---|
| baseline | 0.0001 | current (over-trading) |
| 10× | 0.001 | mild penalty bump |
| 20× | 0.002 | match real friction |
| 50× | 0.005 | over-penalise (sanity check) |

Run all 4 on Q14 only (~10 min total). Plot turnover & alpha vs kappa.
If alpha curve is monotone-improving up to 0.002 then flat or
declining, 0.002 is the optimum. If alpha keeps improving past 0.005,
something else is going on and we re-think.

## Open questions

1. **Is QP's μ unit-consistent with daily returns?** If μ is z-score
   normalised by `ApplyGrinoldKahnTransformTask`, the optimal kappa may
   need a different magnitude. Verify by reading the transform code.
2. **Does `qp_min_dw_pct = 0.02` already mask sub-threshold trades?**
   2% Δw on 7.5% target weight = 27% of position — a coarse filter. If
   most trades are above 2% min_dw, cost_kappa is the binding constraint.
3. **Does fixing cost_kappa break the per-trade +1.98% edge?** If the
   strategy was trading on the marginal edge that disappears under a
   higher kappa, alpha could stagnate. Q14 + multi-window verification
   needed.

## What this is NOT

- NOT a methodology bug. 16-window paired-daily HAC/bootstrap framework
  is valid. The 7 candidates legitimately failed.
- NOT a "no edge" verdict. +1.98%/trade IS edge. We were converting it
  via too many trades.
- NOT a model-quality issue. Panel-LTR ranking picked winners on average
  in Q14 (+11.74pt buy-weighted alpha). The signal is real.
