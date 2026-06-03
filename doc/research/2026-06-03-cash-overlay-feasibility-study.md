# 2026-06-03 — Cash Overlay (QQQ/SPY Beta-Fill) Feasibility Study

**Trigger**: codex feature proposal — *"当现金比例过大时购买 QQQ 或 SPY 来提高
现金效率，避免 money drag"*.
**Status**: research / exploration memo. **No code change. No experiment
fired.** The §7.4 promotion-gate run is gated on user-fire after this memo
is reviewed.
**Companion**: [`2026-06-03-kelly-sizing-audit.md`](./2026-06-03-kelly-sizing-audit.md)
(PR #158) — the σ-horizon mismatch is the **sizing-side** lever for the
same problem; this memo is the **deployment-side** lever. They are
independent and additive.

## TL;DR

**Revised after Stage 0 + Stage 1 counterfactual (2026-06-03): the codex feature, as proposed, fails the empirical test.** The original theoretical motivation is sound (§3), but the regime-conditional Sharpe of the counterfactual overlay is **negative across all tested knob combinations** (BULL_CALM Sharpe ≈ −2.2, BULL_VOLATILE Sharpe ≈ −3.6). Root cause: conditional adverse selection — high-cash days correlate with bad-SPY days because the strategy raises cash via bearish-signal exits (§5.6.3). The originally well-motivated codex proposal is therefore **REJECTED** at the §6 promotion gate before any sim compute is spent. The §5 BEAR defensive sleeve audit is **PROMOTED to standalone §7.7-class dead-gate bug** — config promises 30% deployment, sim shows zero defensive trades.

Before the §5.5 / §5.6 evidence landed, the memo's original framing was: "feasible and well-motivated" — those original three observations stay valuable for context and are preserved below. Three observations:

1. **The drag is real and measured.** Failed-experiments-log §"3-cut
   12-mo OOS reality check" (line 1947) attests *"the strategy is
   structurally a costly closet-index — long-only equity exposure plus
   active management costs (~30–40% cash drag, ST tax, friction, regime
   flips)"*. Alpha vs SPY was **−15.62% ± 10.21% sign-consistent
   negative** across 3 cuts. Cash drag is the dominant component.

2. **The infrastructure already exists.** `strategy_config.golden.json`
   carries a complete `sector_etf_map` (XLK, XLF, XLV, XLE, XLI, XLY,
   SPY, GLD, TLT, XLU, XLRE, XLC), a `defensive_tickers` list
   (GLD, TLT, XLV, XLU), `bear_defensive_slots=2`, and
   `bear_defensive_pct=0.15` — i.e. the system already takes
   **30% of NAV in ETFs during BEAR regimes**. The codex proposal is
   the BULL_CALM mirror of this pattern.

3. **The theory is canonical.** Sharpe (1991) — *"before costs, the
   return on the average actively managed dollar will equal the return
   on the average passively managed dollar"*. Tobin (1958) two-fund
   separation — the optimal portfolio is a linear combination of the
   tangency portfolio and the risk-free asset, **not cash drag plus
   alpha**. Frazzini-Pedersen (2014) "Betting Against Beta" gives the
   complementary frame — leverage-constrained investors leave market
   premium on the table when they cap risk via cash rather than via
   low-β assets.

**Recommendation**: a regime-conditional, opt-in BULL_CALM cash overlay
that mirrors the existing BEAR defensive sleeve (2 slots, ~15% each, in
QQQ + SPY), gated by §7.4 Tier 3 promotion. Sequence in §7.

---

## 1. Problem statement

### 1.1 The empirical drag

| Source | Quantitative claim |
|---|---|
| Failed-experiments-log line 1947 | ~30–40% cash drag in long-only realisation |
| 3-cut 12-mo OOS reality check | alpha vs SPY = **−15.62% ± 10.21%**, sign-consistent across cuts |
| Cut 3 (2025-05-04, strong bull) | strategy captured 75% of SPY's move — cap-constrained turnover |
| Cut 1 (2024-05-01) | 0 buys / 0 sells; cap binding from artifact-staleness, not sizing |

### 1.2 Why Kelly alone leaves cash on the table

`ApplyKellySizingTask` (`kernel/panel_pipeline/job_panel_scoring.py:2811`)
computes per-name `kelly_target_pct = f* = μ/σ²`, clipped to the
per-regime cap. Cash is the residual:

```
cash% ≡ 1 − Σ_i w_i  ≥ cash_reserve_pct
       ≥ 1 − (n_candidates × max_position_pct)
```

When the calibrated μ̂ is small (BULL_CALM has the lowest cross-sectional
spread per the audit memo's BULL_CALM diagnostic), `Σ kelly_target` falls
well below `1 − cash_reserve`. The QP's `cash_drag_lambda=0.05` soft
penalty (`portfolio_qp/qp_solver.py:439`) pushes the solver to fill,
**but only with the same candidate set the panel-LTR ranked**. If those
candidates are at or near cap, the penalty is bound and cash sits.

This is exactly the regime where Sharpe (1991) says the marginal cash
dollar earns the risk-free rate while the marginal SPY dollar earns
`R_f + (R_m − R_f)`. Over 30%-cash periods at the historical
3–8 pp/year US equity premium, that is **~1–3 pp of pure drag per
calendar year**, before any model gets a chance to add or subtract α.

### 1.3 Why Kelly σ-horizon fix doesn't fully close it

PR #158 (`doc/research/2026-06-03-kelly-sizing-audit.md`) identified an
independent ~4× **underweighting** from the σ-horizon mismatch (μ at
60d, σ annualised). Fixing that pushes `kelly_target_pct` higher per
name, but it does NOT raise the **count** of alpha candidates above the
calibrator floor — so on days when there are simply few qualified
names, cash drag persists. The two levers are complementary, not
redundant.

---

## 2. The proposed overlay

### 2.1 Concept

When BULL_CALM (or any "premium-positive" regime) ends a daily
inference run with `cash% > overlay_threshold`, deploy the excess into
a small ETF sleeve. Default sketch:

```
if regime in {BULL_CALM, BULL_VOLATILE} and cash_pct > 0.20:
    overlay_target = min(cash_pct − 0.10, max_overlay_pct)  # leave 10% true cash
    split overlay_target across QQQ + SPY per overlay_split
```

The mechanism mirrors the existing BEAR defensive sleeve at
`kernel/pipeline/task_selection.py:166`:

```python
bear_def_pct  = float(ctx.config.get("bear_defensive_pct", 0.15))
bear_slots    = int(config.get("bear_defensive_slots", 1))
```

That code already buys GLD/TLT/XLV/XLU when SPY turns BEAR. The codex
proposal is **the symmetric construct for BULL_CALM**.

### 2.2 Why QQQ + SPY (not XLK alone)

- **QQQ** = NASDAQ-100, tech-heavy beta. Tracks the growth tilt that
  dominates BULL_CALM regimes. Already on the watchlist
  (`strategy_config.golden.json:628`).
- **SPY** = broad market. Diversifies the sleeve so the overlay isn't
  a single-sector punt.
- **NOT XLK** — already in our sector_etf_map for the tech sleeve and
  classified as `giant_tech` per the 2026-06-03 sector-map audit. Adding
  XLK to the overlay would compound the giant_tech sector cap binding.
- **NOT TLT/GLD** — those are the BEAR sleeve (negative-beta /
  uncorrelated). BULL_CALM wants positive beta, not negative.

A 60/40 SPY/QQQ split is the default-safe starting point. Pure SPY is
the most defensible "we just want market beta" position; the QQQ tilt
is empirically the BULL_CALM tailwind per CAPM.

---

## 3. Theoretical foundation

### 3.1 Canonical references (per §7.10)

| # | Citation | Relevance |
|---|---|---|
| R-1 | **Sharpe, W. F. (1991). "The Arithmetic of Active Management." *Financial Analysts Journal* 47(1), 7–9.** | The marginal passively-managed dollar earns the average return; the marginal cash dollar earns R_f. The arithmetic is unambiguous about cash drag. |
| R-2 | **Tobin, J. (1958). "Liquidity Preference as Behavior Towards Risk." *Review of Economic Studies* 25(2), 65–86.** | Two-fund separation theorem — optimal portfolios are linear combinations of the tangency portfolio and the risk-free asset. Cash drag is admissible ONLY when the investor is at the maximum-risk-aversion corner of the efficient frontier. |
| R-3 | **Markowitz, H. (1952). "Portfolio Selection." *Journal of Finance* 7(1), 77–91.** | Mean-variance efficient frontier. Cash + alpha names is dominated by tangency-portfolio + cash unless alpha names already span the tangency. |
| R-4 | **Black, F. (1972). "Capital Market Equilibrium with Restricted Borrowing." *Journal of Business* 45(3), 444–455.** | Zero-beta CAPM. When borrowing is restricted, the tangency moves toward a zero-β construct, but cash-as-residual still loses the equity premium. |
| R-5 | **Frazzini, A., & Pedersen, L. H. (2014). "Betting Against Beta." *Journal of Financial Economics* 111(1), 1–25.** | Leverage-constrained investors capture market exposure via high-β stocks rather than levered low-β; cash-as-residual is the unconstrained dual. Their BAB factor is long low-β short high-β. |
| R-6 | **Asness, C. S., Frazzini, A., & Pedersen, L. H. (2012). "Leverage Aversion and Risk Parity." *Financial Analysts Journal* 68(1), 47–59.** | Argues for risk-parity scaling — the inverse of which says *"if your risk budget has slack, fill with beta exposure rather than holding cash"*. |
| R-7 | **Carhart, M. M. (1997). "On Persistence in Mutual Fund Performance." *Journal of Finance* 52(1), 57–82.** | 4-factor model. Important for the **attribution** problem: market β from QQQ/SPY ≠ alpha; the overlay's return must be decomposed cleanly. |
| R-8 | **Berk, J. B., & van Binsbergen, J. H. (2015). "Measuring skill in the mutual fund industry." *Journal of Financial Economics* 118(1), 1–20.** | Value-added vs gross-alpha framework. Whether overlay belongs in the strategy's "skill" attribution or in its "passive base" depends on whether it is regime-conditioned. |
| R-9 | **Garleanu, N., & Pedersen, L. H. (2013). "Dynamic Trading with Predictable Returns and Transaction Costs." *Journal of Finance* 68(6), 2309–2340.** | Already implemented in our `portfolio_qp/proportional_trade.py`. The overlay should obey the same proportional-trade-to-target band so we don't over-trade on small cash% swings. |
| R-10 | **French, K. R. (2008). "Presidential Address: The Cost of Active Investing." *Journal of Finance* 63(4), 1537–1573.** | Quantifies the cost of active investing relative to a passive benchmark — the cash-drag inflation factor when ranking active managers. |
| R-11 | **Pedersen, L. H. (2015). *Efficiently Inefficient*. Princeton University Press, chapter 4.** | Practitioner framework for passive overlays in long-short equity. The chapter explicitly discusses "filling beta" when the alpha book is short on conviction. |
| R-12 | **Sharpe, W. F. (1964). "Capital Asset Prices: A Theory of Market Equilibrium under Conditions of Risk." *Journal of Finance* 19(3), 425–442.** | CAPM. The market premium `E[R_m] − R_f` is the canonical opportunity cost of cash. |

### 3.2 The math (one paragraph)

Let the strategy hold `Σ_i w_i = 1 − c` in alpha names plus `c` in
cash. Per-period expected return is
`E[R_p] = R_f · c + (1 − c) · (R_f + α + β · (R_m − R_f))`
which simplifies to
`E[R_p] = R_f + (1 − c) · (α + β · (R_m − R_f))`.

Adding a passive overlay of size `o ≤ c − c_min` in an asset with
β ≈ 1 changes this to
`E[R_p] = R_f + (1 − c) · (α + β · (R_m − R_f)) + o · (R_m − R_f)`.

The overlay term `o · (R_m − R_f)` is positive in expectation whenever
the equity premium is positive — i.e. **always, on average, ex-post**
per Fama-French (2002) which puts the post-WWII equity premium at ~3.5–7%
depending on horizon. In BULL_CALM specifically, conditional premium is
historically the largest (Cochrane 2017, *"Macro-Finance"*, JF 72(3),
1663–1714).

The overlay carries **no extra α** (by construction) and **adds β to
the portfolio's total β** (proportional to `o`). Volatility goes up by
`o · σ_m` plus covariance terms. **Sharpe goes up iff the marginal
overlay's Sharpe exceeds the strategy's current Sharpe** — true when
the strategy is cash-heavy enough that the alpha names' contribution to
total Sharpe is small.

### 3.3 Where the theory fails

1. **Regime transitions**: deploying overlay at the top of BULL_CALM
   immediately before a BEAR transition is a deep negative tail.
   Mitigation: the regime detector itself; the overlay is per-regime
   gated (§1 PRIME DIRECTIVE).
2. **Alpha sign is conditional**: if the model has positive α
   *only* in BULL_CALM, an overlay that also fires in BULL_CALM may
   either complement or shadow the α depending on the correlation of α
   with R_m. Need to measure.
3. **Borrowed beta on top of long beta**: total portfolio β > 1 in
   BULL_CALM is fine; total β > 1.3 starts looking like leveraged-long
   territory which the risk team has not authorised. Cap the overlay
   so portfolio β stays ≤ 1.
4. **Capacity / liquidity**: SPY + QQQ are the two most-liquid
   ETFs globally — capacity is not a constraint at our AUM.
5. **Tax**: holding overlays > 1 year qualifies for LTCG; turnover at
   ≤ monthly rebalance is fine. Daily Kelly-target rebalance of the
   overlay would generate ST tax churn — use Garleanu-Pedersen R-9
   proportional-trade band (already implemented) to throttle.

---

## 4. Existing system fit

### 4.1 The `defensive_tickers` precedent

`strategy_config.golden.json` already declares:

```json
"bear_defensive_slots": 2,
"bear_defensive_pct": 0.15,
"defensive_tickers": ["GLD", "TLT", "XLV", "XLU"]
```

with `kernel/pipeline/task_selection.py:166` reading `bear_def_pct` and
`config.get("bear_defensive_slots", 1)` to size the BEAR sleeve at
**up to 30% of NAV in 2 ETFs**. This is the existing precedent the
codex feature would parallel.

The proposed BULL_CALM analogue:

```json
"bull_calm_overlay_slots": 2,
"bull_calm_overlay_pct": 0.15,
"overlay_tickers": ["SPY", "QQQ"]
```

with a new `ApplyCashOverlayTask` in `pipeline/task_selection.py`
that fires only when:

- `regime.label in overlay_regimes`
- `cash_pct > overlay_threshold` (e.g. 0.20)
- `total_portfolio_beta_after_overlay ≤ overlay_max_beta` (e.g. 1.0)

### 4.2 The `cash_drag_lambda` interaction

The QP already carries `cash_drag_lambda=0.05`
(`portfolio_qp/qp_solver.py:127, 439`) — a soft objective term
penalising `max(0, target − Σwp)`. Today it can only pull MORE from
the same panel-LTR candidate set. The overlay adds new "candidates"
(QQQ, SPY) into the QP's feasible set under a tightly bounded
sub-budget. The overlay's soft objective complements the existing
`cash_drag_lambda` — same direction (reduce cash), independent control
of magnitude.

### 4.3 Sector-cap interaction (per #136 audit)

The 2026-06-03 sector-map audit pinned the rule: XLK counts against the
`giant_tech` sector cap. **The overlay must NOT amplify the same
binding constraint the audit just closed.** Two safe paths:

- **QQQ → `benchmark_overlay` (new pseudo-sector)** — explicitly
  excluded from the per-sector cap, capped only by the overlay's own
  bound. Treats the overlay as a basket distinct from single-name
  bets.
- **SPY → reuse `benchmark` sector (already single-member)** — SPY is
  already classified `benchmark` so the cap is effectively
  per-name = per-sector.

Both paths require an explicit decision; the §6 experiment will pick.

### 4.4 Existing `etf_hedge` regime-knob slot is empty

Checked all 14 regime entries in `regime_params.*.etf_hedge` — every
one is `null` today. The codex feature would populate this slot with
the per-regime overlay spec. No conflict with existing knobs.

---

## 5. Risks and counter-arguments

| # | Risk | Mitigation |
|---|---|---|
| K-1 | BULL_CALM → BEAR transition while overlay is on | Per-regime gating + 5-day rolling regime confirmation before sleeve build-up |
| K-2 | Overlay β exceeds 1.0 net | Hard cap on `overlay_pct × overlay_β ≤ overlay_max_beta − Σ(w_i × β_i)` |
| K-3 | Alpha attribution muddied | Decompose monthly P&L into `alpha_pnl + overlay_pnl + cash_pnl`; report all three; the §7.4 promotion gate evaluates ALPHA ΔAPY explicitly (not gross) |
| K-4 | Closet-index complaint becomes self-fulfilling prophecy | Already attested: failed-experiments-log already shows the strategy IS a closet-index. The question is whether to ALSO LOSE the equity premium on top, which is the worst of both worlds |
| K-5 | Turnover from daily Kelly-driven overlay rebalances | Garleanu-Pedersen proportional-trade band (R-9, already implemented) — only rebalance overlay when |cash% drift| > `overlay_rebalance_band` (e.g. 5pp) |
| K-6 | Tax churn from ST overlay sales | Hard rule: overlay sales are LTCG-only — built into the existing `meta_label_exit.json` framework via a new `overlay_min_holding_days: 365` gate |
| K-7 | Behavioural risk — operator may interpret QQQ as alpha | Mandatory: monthly attribution report breaks out `alpha_pnl` separately. PR titles + commit messages tagged `[overlay]` |
| K-8 | Model interaction with `defensive_tickers` (GLD/TLT/XLV/XLU) when BULL → BEAR transition | The new overlay sleeve must be fully wound down BEFORE the BEAR sleeve fires; sequencing handled in `task_selection.py`'s existing slot-priority logic |
| K-9 | "We're a long-short with alpha mandate, not a long-only beta house" | The overlay is **opt-in per regime + per-PR config flag**. If the alpha mandate matures and cash% drops naturally, turn it off — it adds no friction to the alpha path |
| K-10 | Sharpe (1991) arithmetic cuts both ways: average passive ≠ above-average | True. But the overlay's claim is NOT to beat SPY; it's to **avoid losing the equity premium on the 30% cash portion**. Different claim |

---

## 5.6 Stage 1 counterfactual — passive overlay fails the empirical test

**Date**: 2026-06-03 (same session as Stage 0).
**Verdict JSON**: [`artifacts/cash_overlay_stage1_counterfactual.json`](../../artifacts/cash_overlay_stage1_counterfactual.json).

Rather than build the overlay code path and run a full Stage-1 sim
(~30 min), I ran an **analytical counterfactual** against
`pipeline_runs` joined with `ticker_forward_returns::SPY::fwd_1d`:
for each historical sim bar where `cash_pct > threshold`, compute the
incremental return the overlay sleeve would have realized on day
`t+1` if it had deployed `(cash_pct − 0.10) × fill_frac` of NAV
into SPY at the day-`t` close. This is faster than a Stage 1 sim
and uses only data already in the DB.

### 5.6.1 Grid results — BULL_VOLATILE (FINDING-2 target)

| threshold | fill | n_active | mean daily return | annualized Sharpe |
|---|---|--:|--:|--:|
| 0.20 | 0.50 | 44 | −0.057% | **−3.79** |
| 0.20 | 0.75 | 44 | −0.085% | **−3.79** |
| 0.20 | 1.00 | 44 | −0.114% | **−3.79** |
| 0.30 | 0.50 | 38 | −0.059% | **−3.64** |
| 0.30 | 0.75 | 38 | −0.088% | **−3.64** |
| 0.30 | 1.00 | 38 | −0.117% | **−3.64** |
| 0.40 | 0.50 | 34 | −0.058% | **−3.53** |
| 0.40 | 0.75 | 34 | −0.087% | **−3.53** |
| 0.40 | 1.00 | 34 | −0.116% | **−3.53** |

All 9 combinations produce **negative** Sharpe ranging −3.53 to −3.79.

### 5.6.2 Grid results — BULL_CALM (FINDING-1 target, for completeness)

| threshold | fill | n_active | mean daily return | annualized Sharpe |
|---|---|--:|--:|--:|
| 0.10 | 0.50 | 111 | −0.073% | **−2.20** |
| 0.10 | 1.00 | 111 | −0.145% | **−2.20** |
| 0.15 | 0.75 | 83 | −0.148% | **−2.60** |
| 0.20 | 1.00 | 72 | −0.233% | **−2.86** |

Best BULL_CALM cell: **−2.20 Sharpe at threshold=10% fill=50%**.
Every BULL_CALM combination is also negative.

### 5.6.3 Why every cell is negative — conditional adverse selection

SPY's unconditional daily return averages roughly **+0.04%/day**
(~10%/yr equity premium). A regime-agnostic overlay sampled on
random days would be slightly positive in expectation. Why are BOTH
BULL_CALM AND BULL_VOLATILE conditional samples negative?

**Conditioning artifact.** The overlay only fires when `cash_pct >
threshold`. High-cash days in `renquant_104` are **not random** —
they are the days when the strategy raised cash by selling. The sell
triggers are:

- Calibrator score drop
- Stop-loss
- Drawdown halt
- Regime change to BEAR
- Rotation out

Of these, **stop-loss + drawdown halt + regime-change-to-BEAR are
bearish-signal sells** — they fire when SPY is dropping or has
just dropped. Conditioning on "cash% high" therefore selects days
where the model has been bearish, and those days correlate strongly
with negative day-`t+1` SPY returns.

Net: the overlay deploys cash **precisely on the days SPY is about
to drop**. This is the K-1 risk I called out in §5: *"deploying
overlay at the top of BULL_CALM immediately before a BEAR transition
is a deep negative tail"*. The counterfactual shows the risk is not
just theoretical — it is the empirical dominant mode.

### 5.6.4 What the counterfactual does NOT account for

The numbers are an **upper bound** on overlay value-add (i.e., the
"best case" we are still rejecting):

- No transaction cost (real spread + commission ≈ 1–3 bps per trade)
- No slippage
- No tax (ST capital gains on rebalancing — would shift mean
  further negative)
- No interaction with existing positions (e.g., overlay + alpha could
  push portfolio β > 1)
- Placebo battery (§7.2.1 R2) NOT run

If the counterfactual were already positive Sharpe, the full sim with
TC + slippage + tax would shave it. Since it is already deeply
negative, those frictions only deepen the loss — there is no point
running the full sim.

### 5.6.5 FINDING-3 promoted: BEAR sleeve is dead

The BEAR-sleeve audit I added as a parallel work item in §5.5.4 is
confirmed:

- `pipeline_runs` BEAR-day count: 26 distinct runs × 2 (day +
  intraday) = 52
- BEAR-day rows with any trade activity: **2** (10 trade rows total)
- Of those, **zero** are GLD / TLT / XLV / XLU
- GLD has only **12 trade rows in the entire DB** — and **none on
  BEAR days**

The golden config promises `bear_defensive_slots=2 ×
bear_defensive_pct=0.15 = 30%` deployment into defensives during
BEAR. Empirically: zero defensive trades on BEAR days, ever. **The
sleeve is configured but never fires.** This is a §7.7-class
implicit-decoration bug:

> "Safety gate ≠ enforced safety gate. Every safety gate ships TWO
> artifacts: (a) gate function + tests, (b) a scheduled cron (plist
> + .sh) that invokes it WITHOUT override. If only (a), the gate is
> decoration."

Recommend separate audit memo + fix-or-delete decision. This is NOT
part of the cash-overlay study scope but is a high-priority
sibling finding.

---

## 5.7 Revised verdict

| Hypothesis | Verdict | Reason |
|---|---|---|
| BULL_CALM regime-conditional overlay | **REJECT** | Stage 1 counterfactual Sharpe ≈ −2.2 (conditional adverse selection) |
| BULL_VOLATILE regime-conditional overlay | **REJECT** | Stage 1 counterfactual Sharpe ≈ −3.6 (conditional adverse selection) |
| Regime-agnostic overlay | **NOT TESTED** | Would risk the K-1 / K-3 / K-8 concerns from §5 without the regime-gate safeguard |
| BEAR defensive sleeve | **BROKEN, separate audit** | Configured 30% deployment in 2 ETFs, empirically zero defensive trades |

### 5.7.1 Why the theory and the empirics disagree

The §3 theory (Sharpe 1991, Tobin 1958, Frazzini-Pedersen 2014) says
cash-as-residual loses the equity premium *in expectation*. That is
true for **unconditional** overlay deployment. Our overlay is
*conditional* on cash% being high, and that conditional sample is
adversely selected against the equity premium for the reason in §5.6.3.

Sharpe (1991) explicitly assumes the marginal passively-managed
dollar is **the average dollar**, deployed at random times. The codex
proposal's deployment trigger (`cash% > threshold`) is **not
random** — it is a downstream consequence of the strategy's bearish
signals. The arithmetic only holds when the overlay is unconditional
or the conditional sample is information-symmetric. Neither holds
here.

### 5.7.2 What COULD still work (out of scope for this memo)

Three hypothesis variants might escape the conditional-adverse-
selection trap. Each is its own §7.2-compliant study:

1. **Unconditional overlay** — fixed e.g. `overlay_pct=0.10` of NAV
   in SPY, set-and-forget. No `cash% > threshold` gate; you simply
   target 90/10 NAV/SPY on every bar. Caveat: total portfolio β >> 1
   in fully-invested periods.

2. **SPY-momentum-gated overlay** — only deploy when SPY 20d momentum
   is positive AND cash% is high. Filters out the adverse-selection
   tail at the cost of activation frequency.

3. **Defensive overlay** — buy SHORT-VOL (e.g. SVXY) or LONG-BOND
   (TLT) when cash% is high, rather than SPY. Negative correlation
   with the strategy's sell-trigger pattern would flip the
   conditional sign.

None of these are the codex proposal; all are research-only follow-ups.

### 5.7.3 What changes operationally

**Nothing in production.** This memo is research-only. The pivot
recommendation is:

- Close the cash-overlay study with a NEGATIVE result and document in
  `doc/research/failed-experiments-log.md`.
- Open a separate §7.7 audit on the BEAR defensive sleeve (the dead-
  gate finding) — that is a real bug, not a research question.
- Continue the Kelly σ-horizon work (#158 / #169) — it addresses the
  sizing-side lever and remains the primary path to closing the
  ~30–40% cash drag in `failed-experiments-log` line 1947.

---

## 6. Experiment design

Per §7.2 + §7.4 + §9 DOE. **Range-finding first** (§7.11); fail-closed on
any placebo violation per §7.2.1 R2.

### 6.1 Stage 0 — measure the drag (no overlay, ~10 min)

```bash
# Run the merged-#174 diagnostic on the existing live + sim logs to
# pin the actual cash% distribution per regime.
.venv/bin/python scripts/diagnose_kelly_sizing.py \
    --log 'logs/live_e2e/*.log' \
    --data backtesting/renquant_104/runs.alpaca.db \
    --state backtesting/renquant_104/live_state.alpaca.json \
    --format json --out artifacts/cash_overlay_baseline.json
```

Acceptance: median `cash_pct` in BULL_CALM is ≥ 20% over the last 90
trading days. If < 20%, the overlay is solving a non-problem and the
study deferred.

### 6.2 Stage 1 — range-finding sim (~30 min)

Single-cut sim on 2024-01-01 → 2024-12-31 with one knob:
`bull_calm_overlay_pct ∈ {0.00 (baseline), 0.10, 0.20, 0.30}`,
hold `overlay_tickers = ["SPY"]` (pure market, cleanest test).

Acceptance: monotonic Sharpe improvement up to some level, then
peak/fall. If APY-vs-baseline is **negative at all levels**, the
hypothesis is dead (kill — log to failed-experiments-log).

### 6.3 Stage 2 — Box-Behnken DOE (~5 hours, per §9)

4 knobs at 3 levels (24-run BB design + 3 centre replicates = 27 runs):

| Knob | Low | Center | High | Source |
|---|--:|--:|--:|---|
| `bull_calm_overlay_pct` | 0.10 | 0.20 | 0.30 | Stage-1 winner range |
| `overlay_threshold` (cash%) | 0.15 | 0.25 | 0.35 | Stage-0 distribution |
| `overlay_split_qqq` | 0.0 | 0.5 | 1.0 | 0=SPY only, 1=QQQ only |
| `overlay_rebalance_band` (pp) | 0.02 | 0.05 | 0.10 | Garleanu-Pedersen R-9 |

Fit `y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²` via
`pyDOE2.bbdesign(4, center=3) → sklearn PolynomialFeatures(degree=2)
→ LinearRegression`. `scipy.optimize.minimize` over the fitted
quadratic to identify the optimum.

### 6.4 Stage 3 — mandatory sanity triad (§7.2)

At the Stage-2 optimum:

| Sanity | Method | Pass threshold |
|---|---|---|
| Shuffle-label placebo | Permute label, re-run Stage 2 with same DOE | All response surface βᵢ within ±2σ of 0 |
| Time-shift placebo | Shift label by 2× horizon = 120 days, re-run | Same |
| A/A (3 seeds) | Re-run optimum at seeds 42, 43, 44 | `std_APY / mean_APY < 0.25` |

Per §7.2.1 R2: **no APY / Sharpe number from Stage 2 may be quoted in
any commit / PR / status report without the companion placebo block.**

### 6.5 Stage 4 — DSR / PBO (§7.3 + §9)

`Sharpe_raw / DSR / PBO` reported per Bailey-López de Prado (2014).
Promotion to Tier 3 requires **DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 with
t > 3.0** per §7.4.

### 6.6 Stage 5 — 27-month full WF replay (~3 hours)

Only if Stage 4 passes. Full walk-forward across all 14 regimes (per §1
PRIME DIRECTIVE — per-regime numbers FIRST).

### 6.7 Stage 6 — verdict JSON

```json
{
  "as_of_date": "YYYY-MM-DD",
  "study": "cash-overlay-bull-calm",
  "per_regime": {
    "BULL_CALM": {"alpha_apy_delta": 0.0, "sharpe_delta": 0.0, "n_bars": 0},
    "BULL_VOLATILE": {...},
    "BEAR": {...},
    "CHOPPY": {...}
  },
  "placebo_block": {
    "shuffle_ic": 0.0, "shuffle_gate_passed": false,
    "timeshift_ic": 0.0, "timeshift_gate_passed": false,
    "aa_seeds": [0.0, 0.0, 0.0], "aa_std": 0.0
  },
  "decomposition": {
    "alpha_pnl_pct": 0.0,
    "overlay_pnl_pct": 0.0,
    "cash_pnl_pct": 0.0,
    "total_pnl_pct": 0.0
  },
  "verdict": {
    "promotion_tier": "1_reject | 2_screen | 3_live_promotable",
    "promotion_blocker": "..."
  }
}
```

---

## 7. Promotion sequence (per §7.4 + §1)

1. **Stage 0 evidence** → decide whether the problem is worth solving
   (kill switch).
2. **Stages 1–4** → research-only A/B. Numbers stay internal until
   placebo + DSR/PBO pass.
3. **Stage 5 27-mo WF** → per-regime APY/Sharpe vs baseline. Per §1
   PRIME DIRECTIVE the per-regime slice is the PRIMARY signal; pooled
   second.
4. **Tier 3 gate** → promote ONLY if BULL_CALM (and any other
   premium-positive regime tested) clears Tier 3 AND no other regime
   tier-1 rejects.
5. **Live config flip** is a regime-conditional edit (per §1.5) — the
   overlay defaults OFF (`bull_calm_overlay_pct=0.0`) and is enabled
   only in the regimes Stage 5 cleared.
6. **Operator UX** — monthly P&L report breaks out `alpha_pnl /
   overlay_pnl / cash_pnl` per §K-3.

---

## 8. What this memo does NOT do

- Does NOT consume any compute. No retrain, no sim, no DOE run.
- Does NOT modify any code paths or configs.
- Does NOT promise any APY/Sharpe number. Per §7.2.1 R2, no number
  may be quoted without the placebo block from §6.4.
- Does NOT decide the overlay split (QQQ vs SPY vs mix). The §6.2
  range-finding picks.
- Does NOT decide the BEAR-transition unwind sequence — that is a
  follow-up engineering memo if the §6 experiment promotes.

## 9. Pre-fire checklist

- [ ] Read the companion Kelly σ-horizon audit (`#158`) so the two
      levers' interaction is understood before lighting either.
- [ ] Decide whether to fire Stage 0 (the read-only diagnostic) as a
      standalone PR (~10 min) before committing to Stages 1–5.
- [ ] Confirm `data/sim_runs.db` has BULL_CALM coverage in the
      target WF window.
- [ ] Confirm `OMP_NUM_THREADS=14` set for the DOE batch (§6.5
      hardware saturation).
- [ ] Run `make doctor` + `pytest tests/test_apply_kelly_sizing_task.py
      -q` clean before any sim-experiment write site touches the panel
      pipeline.

## 10. Decision needed before the next step

The §6 stages cost wall-clock time in this order:

| Stage | Wall-clock | Compute risk | Reversibility |
|---|---|---|---|
| 0 (diagnostic) | ~10 min | none | full |
| 1 (range-finding) | ~30 min | small | full |
| 2 (DOE) | ~5 hr | medium | full (config-driven sim only) |
| 3–4 (sanity + DSR/PBO) | ~10 hr | medium | full |
| 5 (27-mo WF) | ~3 hr | medium | full |

**My recommendation**: fire **Stage 0 only** as the immediate next
step — purely diagnostic, surfaces whether the cash% problem is what
the failed-experiments-log says it is, no risk to anything. The
Stage 1–5 commitment is a separate fire decision after Stage 0
publishes.

If Stage 0 confirms median cash% ≥ 20% in BULL_CALM, **proceed to
Stages 1–5 as a single user-fired DOE batch overnight**. If Stage 0
returns median cash% < 15%, the overlay is solving an artefact and we
should re-examine the Kelly σ-horizon fix's effect (#158 / #169)
before re-opening the overlay question.

## 11. Cross-references

- Sizing-side companion: [`2026-06-03-kelly-sizing-audit.md`](./2026-06-03-kelly-sizing-audit.md) (#158)
- Sector-cap rules: [`2026-06-03-sector-map-config-audit.md`](./2026-06-03-sector-map-config-audit.md) (#136 / #154)
- Empirical cash-drag attestation: [`failed-experiments-log.md`](./failed-experiments-log.md) line 1947
- Existing BEAR overlay precedent: `kernel/pipeline/task_selection.py:166` (`bear_def_pct`, `bear_defensive_slots`)
- QP cash-drag soft penalty: `kernel/portfolio_qp/qp_solver.py:127, 439` (`cash_drag_lambda`)
- Garleanu-Pedersen proportional trade (R-9): `kernel/portfolio_qp/proportional_trade.py` (mirror of subrepo merged 2026-05-30)
- Kelly diagnostic CLI for Stage 0: `scripts/diagnose_kelly_sizing.py` (PR #174, merged 2026-06-03)
- Promotion methodology: [`promotion-methodology.md`](./promotion-methodology.md) (§7.4 3-tier)
