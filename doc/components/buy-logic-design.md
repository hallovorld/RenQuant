# Buy Logic Redesign — Literature-Grounded Quality Gates (2026-04-26)

**Trigger**: e2e R6/R7 evidence — buys at μ ≈ +0.008, σ ≈ 0.08
(daily Sharpe of expected return ≈ 0.10), in a saturated calibrator
regime (rank_score collapsed to 6 unique values across 8 candidates).
The current pipeline has **no quality floor** and structurally cannot
say "today nothing is high-conviction; keep cash."

This doc fixes that with a *theory-first* redesign — three gates each
grounded in a different established framework — not heuristic flags.

---

## 1. Diagnosis — what's structurally wrong

The current pipeline answers the question:
> *"Among the candidates that passed per-ticker and panel signals, which subset maximizes net α subject to cash constraint?"*

It does NOT answer:
> *"Is the optimal portfolio weight for this asset materially different from the current weight, accounting for transaction costs and signal decay?"*

These are different questions. The first is a sorted-knapsack on
candidates; the second is a portfolio-level optimization with a
**no-trade region**. The seminal results below say the second is
the right formulation.

### 1.1 Constantinides (1986) — the no-trade band

> Constantinides, G.M. 1986. "Capital Market Equilibrium with Transaction Costs." *Journal of Political Economy* 94(4): 842–862.

Under proportional transaction costs τ, the optimal portfolio policy
under log utility is **not** to rebalance to the Markowitz weight
w*. Instead, there's a no-trade interval [w*-Δ, w*+Δ] around w*. You
trade only when the current weight crosses the band. The band width
Δ scales with τ^(1/3) (Davis-Norman 1990 closed form).

**Implication for our pipeline**: even when a panel signal says "asset
i has positive expected return", it's not optimal to buy unless the
implied weight is **outside** the band around the current weight.
Today we always buy if cash is free and net_α > 0 — we're always
inside the no-trade region but trade anyway, which destroys utility
relative to the optimal policy.

### 1.2 Garleanu & Pedersen (2013) — signal decay × cost

> Gârleanu, N. and Pedersen, L.H. 2013. "Dynamic Trading with Predictable Returns and Transaction Costs." *Journal of Finance* 68(6): 2309–2340.

For an autocorrelated signal that decays at rate `1-φ` per bar, with
quadratic transaction cost coefficient Λ, the optimal position aims
**partially** at the Markowitz target:

```
w_t = (1 - a) · w_{t-1} + a · w_target
```

where the "trading rate" `a` solves a Riccati equation in (φ, Λ,
risk-aversion γ). For high-decay signals (φ ≈ 0), `a` is small — you
barely move toward the target because the signal will be gone before
the cost is amortized. For persistent signals (φ ≈ 1), `a` ≈ 1 —
you trade aggressively.

**Implication for our pipeline**: we currently treat the panel score
as a one-shot signal. We don't measure φ (signal autocorrelation).
For NGBoost μ ≈ +0.008 daily, **what's the half-life?** If 1 day,
the cost-adjusted optimal trade is ~0. If 5 days, larger. We don't
know — we just buy on cross-sectional rank.

### 1.3 Lo (2002) — Sharpe ratio statistics

> Lo, A.W. 2002. "The Statistics of Sharpe Ratios." *Financial Analysts Journal* 58(4): 36–52.

The annualized Sharpe ratio of a strategy with daily edge Sharpe
SR_d (= μ/σ at daily frequency) is `SR_a = SR_d × √252` for IID
returns; with autocorrelation ρ_1 = γ:

```
SR_a = SR_d × √(252) × √( (1+γ)/(1-γ) )
```

A daily edge_Sharpe of 0.10 (R6/R7) → annualized ~1.6 IF perfectly
realized. But **realized SR ≤ predicted SR**: the realization gap
comes from prediction error in μ̂, scoring drift, and ex-post σ
exceeding ex-ante σ. Practitioners commonly require **predicted**
daily edge_Sharpe > 0.20 to clear an annualized realized SR ≥ 1.0.

**Implication for our pipeline**: 0.10 is below this threshold.
**At least one floor must be on edge_Sharpe.**

### 1.4 Grinold (1989) — Fundamental Law

> Grinold, R.C. 1989. "The Fundamental Law of Active Management." *Journal of Portfolio Management* 15(3): 30–37.

```
IR = IC × √breadth
```

Per-ticker model: breadth = 1 (one bet). Panel model: breadth = N_eligible.
**Combining** weakly correlated panel + per-ticker signals gains IR
multiplicatively if the correlation is low, additively if high.

**Implication for our pipeline**: today's gate is "per-ticker AND
panel" (intersection). Treynor-Black framing argues for "weighted
combination" instead — the per-ticker BUY signal carries depth
(model trained on that ticker's history), the panel signal carries
breadth. We should weigh them, not just intersect.

### 1.5 Lopez de Prado (2018) — backtest overfitting

> López de Prado, M. 2018. "The 10 Reasons Most Machine Learning Funds Fail." *Journal of Portfolio Management* 44(6): 120–133.

Probabilistic Sharpe Ratio (PSR):

```
PSR(SR*) = Φ( (SR_obs - SR*) × √(T-1) / √(1 - skew·SR_obs + ((kurt-1)/4)·SR_obs²) )
```

A Sharpe of 1.0 with T=252 bars on a thin signal can have PSR < 0.5
(<50% chance of beating zero). Our σ̂ in NGBoost is heteroskedastic
— skew/kurt corrections matter.

**Implication for our pipeline**: don't trade on a signal whose
PSR (against SR*=0) is < 0.6. We can compute this from the panel's
historical realized excess returns.

### 1.6 Bouchaud-Bonart-Donier-Gould (2018) — square-root impact law

> Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. 2018. *Trades, Quotes and Prices: Financial Markets Under the Microscope.* Cambridge UP. (Esp. Ch. 12 on impact.)

Empirical fact across markets: temporary price impact of a trade of
size Q (in shares) over a participation horizon scales as
`I(Q) ∝ σ_daily × √(Q / V_daily)`. For 1 share of APP at $448 with
ADV ≈ $200M → Q/V ≈ 5×10⁻⁹ → impact in basis points. Negligible at
our size today.

**Implication for our pipeline**: at our scale, slippage is dominated
by half-spread + commission, not size impact. The flat 5-bps
slippage assumption in `task_joint_actions.py` is fine for now. As
NAV grows past ~$10M this assumption breaks; flag for future scaling.

---

## 2. Three gates — each grounded in one of the above

### Gate A: Edge-Sharpe floor (Lo 2002, Grinold-Kahn 1999)

> *"Don't trade unless the predicted instantaneous Sharpe of the edge
> exceeds the threshold required for an annualized realized Sharpe of 1."*

```
edge_sharpe = μ / σ        (NGBoost outputs)
admit(cand) ⇔ edge_sharpe > τ_S
```

Default τ_S = 0.20. Justification:

```
SR_a_realized   ≈   0.5 × SR_a_predicted        (50% slippage typical)
SR_a_predicted  =   SR_d × √252
SR_d            =   τ_S = 0.20 → SR_a_pred = 3.17 → SR_a_real ≈ 1.6
```

Conservative default is τ_S = 0.15 (SR_a_real ≈ 1.2). Aggressive is
0.25 (SR_a_real ≈ 2.0). Tunable per regime — Gate C (below) tightens
this.

**Edge cases**:
- σ ≈ 0 (NGBoost crashes on degenerate variance): treat as "no signal,"
  reject. Already handled by NGBoost's σ floor of 1e-4.
- μ < 0: edge_sharpe < 0 → reject (not a buy candidate).
- μ = NaN: reject (treat NaN as failed-prediction).

### Gate B: Cross-sectional persistence (Garleanu-Pedersen 2013)

> *"Don't trade a signal whose autocorrelation is too low to recover the cost."*

For each candidate, compute:

```
φ_i = correlation(panel_score_t(i), panel_score_{t-1}(i))   over last 60 bars
```

If `φ_i < 0` (mean-reverting) and the signal magnitude doesn't beat
the cost twice (round-trip), reject. The half-life is
`τ_half = -ln(2)/ln(φ)`. Reject if `τ_half < min_hold_days × 0.5`.

**Why this matters more than µ alone**: a μ=+0.008 signal that
persists for 30 bars compounds; the same signal that flips sign next
bar is fool's gold. We have the panel's daily score history in
`score_distribution` (just shipped) — measurable.

Gate B implementation gates on **measured signal persistence per ticker**,
not just current strength.

### Gate C: No-trade region (Constantinides 1986, Davis-Norman 1990)

> *"Don't trade unless the deviation from optimal weight exceeds the band."*

For each candidate `i` with predicted Sharpe-of-edge `SR_i`:

```
target_weight_i = α_i × SR_i / γ                  (Black-Litterman shape)
deviation_i     = target_weight_i - current_weight_i
band_i          = c × (γ × σ_i² × τ_round_trip)^(1/3)    (Davis-Norman 1990)
admit(i)        ⇔  |deviation_i| > band_i
```

where `α_i` is the asset's covariance-with-portfolio scalar, γ is risk
aversion (default 3.0 ≈ moderate), `τ_round_trip` is round-trip
transaction cost (~0.0010 = 10bps for our case), `c` is a model
constant (~1.5).

**For our parameters** (γ=3, σ=0.08, τ=0.001): `band ≈ 1.5 × (3 × 0.0064 × 0.001)^(1/3)` ≈ 1.5 × 0.027 ≈ **4% of NAV**.

So a candidate at current weight=0 with target weight 3% of NAV
would NOT cross the band → no trade. A candidate at weight=0 with
target=8% does cross → trade.

**This naturally prevents "fill empty slots with weak signal"** — if
the signal isn't strong enough to push target weight past the band,
we keep cash.

---

## 3. How the gates compose

```
candidates_pre  = [c for c in ctx.candidates if pass_per_ticker(c)]
candidates_panel = [c for c in candidates_pre if pass_panel_score(c)]
                                                                  # current behaviour ends here

# NEW — order matters: Gate A is cheapest, Gate C uses Gate A output
candidates_A = [c for c in candidates_panel if c.μ/c.σ > τ_S(regime)]      # Gate A
candidates_B = [c for c in candidates_A     if signal_persistent(c)]       # Gate B
candidates_C = [c for c in candidates_B     if past_no_trade_band(c)]      # Gate C

ctx.candidates = candidates_C
```

**Each gate is independently flag-controlled.** With all three off,
behaviour matches today bar-for-bar. The first gate enabled is Gate A
(highest leverage per implementation cost) — that gate alone would
have rejected today's R6/R7 buys (edge_sharpe ≈ 0.10 < 0.20).

---

## 4. Why this is structurally better than naive thresholds

A naive `panel_score > 0.30` floor (the field that's currently `null`):
- ❌ Scale-dependent (today's panel scores range -0.013 to +0.018; a
  fixed 0.30 would block everything).
- ❌ Calibrator-dependent — when calibrator collapses (today's
  problem), threshold becomes meaningless.
- ❌ No grounding in cost / decay / signal quality.

The three gates above:
- ✅ Gate A is scale-invariant (μ/σ is unit-less).
- ✅ Gate B uses observable signal persistence, not assumption.
- ✅ Gate C uses transaction cost explicitly, with closed-form band.

---

## 5. Alternative frameworks considered & why rejected

### 5.1 Black-Litterman with view-confidence (Black-Litterman 1992)

> Black, F. and Litterman, R. 1992. "Global Portfolio Optimization." *Financial Analysts Journal* 48(5): 28–43.

BL combines a prior (CAPM equilibrium) with views (the panel
predictions) weighted by view confidence (which we'd derive from
NGBoost σ). Mathematically beautiful but: (a) requires a prior we
don't have, (b) we'd need a covariance estimate `Σ` that's the panel-
LTR's job, not the gate's. **Defer to a future iteration where we
build a portfolio-optimization layer above the joint action.**

### 5.2 Brandt-Santa-Clara-Valkanov (2009) parametric portfolio policies

> Brandt, M., Santa-Clara, P., Valkanov, R. 2009. "Parametric Portfolio Policies: Exploiting Characteristics in the Cross Section of Equity Returns." *Review of Financial Studies* 22(9): 3411–3447.

Direct optimization of utility from features — skips the
"estimate μ then optimize" two-step. Promising. Reframes the
panel-LTR objective from rank-IC to direct utility. **Future work**;
big architectural change to training, orthogonal to gates.

### 5.3 Robust portfolio choice (Garlappi-Uppal-Wang 2007)

> Garlappi, L., Uppal, R., Wang, T. 2007. "Portfolio Selection with Parameter and Model Uncertainty: A Multi-Prior Approach." *Review of Financial Studies* 20(1): 41–81.

Uses min-max robust optimization over μ within a confidence ellipsoid
defined by σ. Equivalent to a stricter Sharpe floor under a single
view. **Subsumed by Gate A with a more conservative τ_S.**

### 5.4 Optimal stopping — Shiryaev (1978) disorder problem

> Shiryaev, A.N. 1978. *Optimal Stopping Rules.* Springer.

For a signal that may "disappear" at random time, the optimal stop
problem solves for a threshold that balances expected gain vs cost
of waiting. Reduces to a CUSUM-style detector. We already have CUSUM
in regime detection — adding one for buy timing duplicates the
machinery and adds latency. **Out of scope.**

---

## 6. Test plan (~16 tests, paired NB-style)

| File | Tests | Coverage |
|---|---:|---|
| `tests/test_quality_floor_gate_a.py` | 5 | edge_sharpe formula, NaN/0-σ, threshold round-trip |
| `tests/test_quality_floor_gate_b.py` | 4 | persistence calc, half-life threshold, no-history fallback |
| `tests/test_quality_floor_gate_c.py` | 5 | Davis-Norman band formula, deviation calc, edge cases |
| `tests/test_quality_floor_integration.py` | 4 | all-off ⇒ no change; R7-shape inputs ⇒ rejection; blocked_by surfaces; metric monotonicity |

Existing tests `test_panel_alignment.py` (34 tests) and
`test_policy_alignment.py` (235 tests) MUST stay green — quality
floors are additive, not replacements.

---

## 7. Roll-out

| Stage | Scope | Defaults | Sim verdict gate |
|---|---|---|---|
| 0 (this session) | Land Tasks + tests, all 3 gates **off** | 100% behaviour preserved | No regression on golden v4.1 sim |
| 1 (next session) | Enable Gate A, τ_S=0.15 | Conservative | APY ≥ golden − 0.5pt (loose: kept-cash ≠ lost APY) |
| 2 | Tune τ_S via 27-mo OOS sweep | grid {0.10, 0.15, 0.20, 0.25, 0.30} | Pick τ_S maximising Sharpe (not APY — quality gates trade APY for vol reduction) |
| 3 | Enable Gate B (signal persistence) | φ-threshold from data | ≥ golden APY without Gate B noise |
| 4 | Enable Gate C (no-trade band) | Davis-Norman c=1.5, γ=3 | Final stack vs golden |

Per CLAUDE.md §2a, the **promotion criterion shifts from APY to Sharpe**
because the gates' explicit purpose is variance reduction. APY may
drop modestly while realized Sharpe rises — that's the win.

---

## 8. References

- Black, F., Litterman, R. 1992. "Global Portfolio Optimization." *Financial Analysts Journal* 48(5): 28–43.
- Bouchaud, J.-P., et al. 2018. *Trades, Quotes and Prices.* Cambridge UP.
- Brandt, M., Santa-Clara, P., Valkanov, R. 2009. "Parametric Portfolio Policies." *RFS* 22(9): 3411–3447.
- Constantinides, G.M. 1986. "Capital Market Equilibrium with Transaction Costs." *JPE* 94(4): 842–862.
- Davis, M.H.A., Norman, A.R. 1990. "Portfolio Selection with Transaction Costs." *Mathematics of Operations Research* 15(4): 676–713.
- Garlappi, L., Uppal, R., Wang, T. 2007. "Portfolio Selection with Parameter and Model Uncertainty." *RFS* 20(1): 41–81.
- Gârleanu, N., Pedersen, L.H. 2013. "Dynamic Trading with Predictable Returns and Transaction Costs." *JF* 68(6): 2309–2340.
- Grinold, R.C. 1989. "The Fundamental Law of Active Management." *JPM* 15(3): 30–37.
- Grinold, R.C., Kahn, R.N. 1999. *Active Portfolio Management.* 2nd ed. McGraw-Hill.
- Liu, H., Loewenstein, M. 2002. "Optimal Portfolio Selection with Transaction Costs." *RFS* 15(3): 805–835.
- Lo, A.W. 2002. "The Statistics of Sharpe Ratios." *FAJ* 58(4): 36–52.
- López de Prado, M. 2018. *Advances in Financial Machine Learning.* Wiley. (Chs. 7, 14, 16.)
- López de Prado, M. 2018. "The 10 Reasons Most Machine Learning Funds Fail." *JPM* 44(6): 120–133.
- Shiryaev, A.N. 1978. *Optimal Stopping Rules.* Springer.
- Treynor, J.L., Black, F. 1973. "How to Use Security Analysis to Improve Portfolio Selection." *JB* 46(1): 66–86.

---

## 9. Cross-reference

- Calibrator pool collapse — `doc/components/calibration-saturation.md`
- Joint action / net_α — `kernel/pipeline/task_joint_actions.py`
- NGBoost μ,σ — `training_panel/ngboost_head.py`
- Score distribution — `kernel/pipeline/task_score_distribution.py`
- Regime detection — `kernel/pipeline/job_regime.py`
- Implementation will land at —
  `kernel/panel_pipeline/task_quality_floor.py` (new, ~280 LOC)
