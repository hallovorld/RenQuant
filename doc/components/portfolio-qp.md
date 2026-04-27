# Unified Portfolio Action — Joint Buy/Sell/Rotate Optimization (2026-04-26)

**Mandate**: replace the current sorted-knapsack candidate selection +
separate sell loop + bolted-on rotation logic with a **single
portfolio-level convex optimization per bar** that decides the full
vector of position changes — buys, sells, rotations, cash retention —
in one shot. Buy/sell/rotate stop being "three different code paths"
and become the same primitive: a sign on `Δw_i`.

This is the framework production funds use (AQR, Two Sigma, Renaissance
all employ variants). It generalizes Constantinides, Garleanu-Pedersen,
Boyd, and Markowitz into one solver.

---

## 1. Why the current architecture is structurally limiting

**Today** (renquant_104):

```
Phase 1: TickerSellJob          → ctx.exits  (per-ticker stop/trail/SDL/streak)
Phase 2: TickerCandidateJob     → ctx.candidates  (model said BUY)
Phase 3a: PanelScoringJob        → μ, σ, panel_score on cands & holdings
Phase 3b: JointActionTask        → 3-pass greedy on (BUY, SELL_panel, ROTATE)
Phase 3c: SizeAndEmitTask        → Kelly target × confidence-scaled cap
```

What this **cannot** express:

1. **Risk substitution at the portfolio level.** When σ_AAPL rises and
   σ_GOOG falls, the optimal policy is to swap risk *budget*, not
   shares. Three-pass greedy doesn't know the Σ matrix.
2. **Dynamic hedging within the universe.** GOOG long against TSM long
   when their pairwise corr is +0.7 isn't fully diversified — the
   greedy picker doesn't see the joint covariance penalty.
3. **Cost-amortized partial trades.** Garleanu-Pedersen 2013 prescribes
   *partial* moves toward the target each bar (`w_t = (1-a)w_{t-1} +
   a·target`), with `a` derived from cost coefficient × signal decay.
   Today we trade either 0% or 100% of the target.
4. **Drawdown-constrained sizing.** When DD breaches a threshold we
   halt all buys; literature (Grossman-Zhou 1993, Cvitanić-Karatzas
   1995) prescribes *gradually* shrinking risk via γ_eff, not a binary
   switch.
5. **No-trade region.** Per Constantinides 1986 / Davis-Norman 1990,
   the optimal policy under proportional cost has a band around w*.
   Today we always trade if μ > fee.
6. **Joint sell/buy economics.** A "rotation" today emits a SELL_A +
   BUY_B pair. The convex formulation makes this fall out naturally —
   `Δw_A < 0`, `Δw_B > 0` — with the cost penalty applied uniformly.

The fix: replace Phase 3 entirely with a **single convex QP** that
decides the full Δw vector.

---

## 2. The mathematical formulation (Markowitz → Boyd → ours)

### 2.1 Single-period (Markowitz 1952 + Pogue 1970 transaction-cost extension)

```
                     T          T
max   r̂' (w + Δw) - γ (w + Δw) Σ (w + Δw) - φ(Δw)
 Δw

s.t.  1' (w + Δw)  ≤  1 - cash_reserve
      |Δw_i|        ≤  Δw_max,i              (broker slippage cap)
       w_i + Δw_i   ∈  [0, w_cap_i]          (no shorts; per-position cap)
      Σ_i (w + Δw)_i 1{sector(i)=s} ≤ sec_cap_s   ∀s
       w_i + Δw_i   = 0  if i ∈ wash_sale_block
```

Where:
- `w` ∈ ℝⁿ : current weights (computed at decision-time price)
- `Δw` ∈ ℝⁿ : decision variable
- `r̂` ∈ ℝⁿ : predicted excess returns (NGBoost μ vector)
- `Σ` ∈ ℝⁿˣⁿ : forecast covariance (panel residuals + Ledoit-Wolf shrinkage)
- `γ` : risk aversion (default 3.0; modulated by DD and confidence)
- `φ(Δw)` : cost function, see §2.3

### 2.2 Multi-period (Boyd-Busseti-Diamond-Kahn 2017, Gârleanu-Pedersen 2013)

For an autoregressive signal `r̂_{t+1} = (1-φ_decay) r̂_t + ε`,
**myopic** optimization (Markowitz) is suboptimal. The K-step MPC
problem is:

```
                K
max   E_t [  Σ   r̂'_{t+k} (w_{t+k}) - γ w'_{t+k} Σ w_{t+k} - φ(Δw_{t+k})  ]
 Δw_{t..t+K-1}  k=1
```

solved by dynamic programming. The closed form (linear-quadratic-
Gaussian case):

```
Δw_t* = a × (w_target - w_{t-1})
where a ∈ (0,1) solves a 2nd-order matrix Riccati equation in (φ_decay, Λ_cost, γ)
```

For φ_decay → 1 (persistent signal): a → 1 (full move). For φ_decay
→ 0 (one-shot): a → 0 (don't trade because cost won't be amortized).

**Implementation**: solve the 1-step Markowitz QP at each bar, but
compute `r̂` with explicit decay-discounting (`r̂_eff = r̂_panel /
(1 - λ φ_decay)`). One-step but cost-aware. Equivalent to the K=∞
limit of the Boyd MPC for stationary AR(1) signals.

### 2.3 Cost function (Almgren-Chriss 1999/2001 + Bouchaud et al. 2018)

```
φ(Δw_i) = κ_linear × |Δw_i|              (half-spread + commission)
        + κ_quad   × Δw_i²                (quadratic impact)
        + κ_sqrt   × |Δw_i|^(3/2)         (square-root impact)
```

For our scale (NAV ~ $10k, ADV per ticker > $50M):
- Linear dominates: half-spread ≈ 1bp, commission 0bp on Alpaca → κ_linear ≈ 0.0001
- Quadratic ≈ 0 (well below ADV thresholds where quad kicks in)
- Square-root ≈ 0 (kicks in at >0.1% ADV per Bouchaud 2018; we're at ~0.001%)

So at our current size: `φ(Δw) ≈ 0.0001 × |Δw|`. As NAV grows past
$1M, square-root term begins to matter (would re-tune then).

### 2.4 Drawdown constraint (Grossman-Zhou 1993)

> Grossman, S.J., Zhou, Z. 1993. "Optimal Investment Strategies for Controlling Drawdowns." *Mathematical Finance* 3(3): 241–276.

For a target maximum drawdown `α` (fraction of historical high water
mark), the optimal Kelly fraction shrinks as:

```
f_eff = f_kelly × max(0, 1 - DD_current / α)
```

Equivalent in QP form: scale γ by `1 / max(0, 1 - DD/α)`. As DD
approaches the limit, γ → ∞, forcing Δw → 0 (halt buying) and
encouraging |Δw_i| < 0 (sell to reduce risk). Smooth transition,
no binary cliff.

### 2.5 Robust μ under uncertainty (Garlappi-Uppal-Wang 2007)

Replace point estimate `r̂` with worst-case under a confidence
ellipsoid:

```
r̂_robust = r̂ - κ × diag(σ_panel) × ξ
where ξ ∈ ℝⁿ is the worst-case direction in the ε-ball
```

Reduces to subtracting `κ · σ_i` from each μ — a scaled Sharpe-ratio
penalty. For κ = 1, this is the conservative analog of the Sharpe-of-
edge floor (single-asset case). With Σ off-diagonal: it accounts for
joint uncertainty.

---

## 3. Combining buy + sell + rotate as ONE primitive

Current code: 3 separate emitters (`SizeAndEmitTask` for BUYs,
`TickerSellTask` for exits, `EmitRotationsTask` for swaps). The QP
emits a single `Δw` vector — sign of each component IS the action:

| Sign | Magnitude | Action |
|---|---|---|
| `Δw_i > 0` | held | BUY (add) |
| `Δw_i > 0` | not held | BUY (initiate) |
| `Δw_i < 0` | held | SELL (trim or full close if `w_i + Δw_i = 0`) |
| `Δw_i < 0` | not held | rejected by `w_i + Δw_i ≥ 0` constraint (no shorts) |
| `Δw_i = 0` | held | HOLD |
| `Δw_i = 0` | not held | NO-TRADE |
| `Δw_A < 0, Δw_B > 0` | both | ROTATION (A→B) |

**Stop-loss / trailing-stop / single-day-loss are PRE-OPTIMIZATION
constraints**: they fix `w_i = 0` (force exit) before the QP runs.
This is correct per Kaminski-Lo 2014 — stop-losses are *path-dependent
risk controls*, not part of the forward-looking optimization. They're
event-driven; the QP is signal-driven.

**Wash-sale is a ZERO-LOWER-BOUND constraint**: `Δw_i ≥ 0` for
recently-sold tickers. Prevents re-buying.

**Order of execution per bar**:

```
1. Path-dependent exits first  (stop_loss, trailing, sdl, max_hold) → set w_i = 0
2. Compute r̂_robust = r̂ - κ·σ                                       (Garlappi-Uppal-Wang)
3. Compute γ_eff = γ_base × confidence_scale × DD_scale               (Grossman-Zhou)
4. Solve QP                                                           (Markowitz + cost)
5. Apply Constantinides band: zero out Δw_i with |Δw_i| < band_i      (no-trade region)
6. Round to integer shares                                            (broker reality)
7. Emit (Δw, action_dict) to broker layer
```

Stop 5 is critical: without it, the QP emits dust trades (Δw_i =
0.001%) every bar that erode P&L through cost. The Constantinides
band kills these naturally.

---

## 4. Sell logic — what's preserved vs replaced

### 4.1 Preserved (path-dependent, hard rules)

- **Stop-loss** (cumulative drawdown from entry, regime-conditional)
- **Trailing stop** (drawdown from peak position HWM)
- **Single-day loss** (gap-down protection, BULL_CALM only)
- **Max hold days**
- **Wash-sale block**

These are **enforced before the QP** as constraints:

```
if breached_stop_loss(i):  pre_w_i = 0; lock(i) for cooldown_days
if breached_max_hold(i):   pre_w_i = 0
if recently_sold(i):       Δw_i ≥ 0 (no buys this bar)
```

Theory backing: Kaminski-Lo 2014 ("When Do Stop-Loss Rules Stop
Losses?") empirically confirms stop-loss adds value when (a) returns
have momentum, (b) tails are fat, (c) signal regime shifts. All three
hold for our universe — keep them.

### 4.2 Replaced (signal-driven, currently heuristic)

- **3-day sell streak rule** → subsumed by QP. If three consecutive
  bars produce μ_i < 0 + cost-justified |Δw_i|, the QP will naturally
  reduce `w_i` over those bars. Streak-rule was a proxy for signal
  decay — QP handles it directly.
- **Min-hold-days rule** → keeps as a soft constraint (entry cost
  amortization). Implemented as `Δw_i ≥ -ε` until `today - entry_date
  > min_hold_days`. Hong-Stein 1999 ("Unified Theory of Underreaction
  ... Momentum") supports a min hold for noisy momentum signals.

### 4.3 Why this is better

- Today's panel-veto (PanelRankVetoTask, just shipped) blocks model_sell
  when the panel rank is high. **The QP already does this** — if μ_i is
  positive (panel says good), Δw_i won't go negative regardless of the
  per-ticker model. PanelRankVetoTask becomes redundant once QP is in.
- Today's rotation-advantage gate (5% panel-score margin) is a hand-
  chosen threshold. **The QP derives the right margin from the cost
  function** — only rotates when the joint Δw produces utility net of
  2× round-trip cost.

---

## 5. Drawdown controller — replacement for binary breaker

### 5.1 Today

```
if drawdown > drawdown_halt_pct:
    block all new buys for cooldown_days
```

Binary, no gradient. Misses the period between "approaching limit"
and "hit limit."

### 5.2 Proposed (Grossman-Zhou 1993 + Cvitanić-Karatzas 1995)

```
DD_scaler = max(0, 1 - DD / α_max)       # 0 at limit, 1 at no DD
γ_eff     = γ_base / DD_scaler            # γ → ∞ as DD → α
```

This produces:
- Smooth shrink of risk-taking as DD approaches α_max.
- Forces sells (negative Δw) when γ→∞ pushes existing holdings
  outside the QP's variance budget.
- Natural recovery: γ_eff returns to γ_base as DD eases.

Closed-form Kelly equivalent (Davis-Lleo 2008):
```
f_eff = f_kelly × DD_scaler^q          q ≈ 0.5 for log utility
```

### 5.3 Tail-aware extension (Rockafellar-Uryasev 2002)

For fat-tailed asset returns, replace variance penalty with CVaR_α:

```
risk_term = γ × CVaR_α (w + Δw)        # expected loss in worst α% scenarios
```

Approximated as a 2nd-stage LP within the QP framework. Adds compute
(~5× slower QP) but tail-aware. **Defer to Stage 3 below**; start
with variance.

---

## 6. Kelly sizing under the new framework

### 6.1 Theory

Kelly (1956) in single-asset case:
```
f_kelly = μ / σ²
```

In multi-asset case (Browne 1995):
```
f_kelly = γ⁻¹ × Σ⁻¹ × μ
```

This is **already the unconstrained QP solution** when φ=0 (no cost).
With cost, the optimal trades toward this target with rate `a` per
Boyd-Garleanu. So the QP IS Kelly, with cost-aware partial moves.

### 6.2 Practical safeguards

- **Fractional Kelly** (Thorp 2006): use `f = 0.5 × f_kelly` to halve
  variance with only ~25% IR loss. Plain in QP: scale `r̂` by 0.5.
- **Kelly cap** at `min(f_kelly, max_position_pct × DD_scaler)`.

---

## 7. Cross-asset signal combination (Grinold-Kahn IR multiplication)

We have 4 signal sources:
1. Per-ticker model μ̂_per (from XGBoost / QLearning / Manual / Classification)
2. Panel-LTR µ̂_panel (cross-sectional)
3. NGBoost μ̂_ngb (variance-aware)
4. Sector momentum / RS score (already in pipeline)

Optimal combination (Treynor-Black 1973) weights by:
```
w_i ∝ IC_i / σ²(IC_i)
```

Estimated from CPCV folds. Currently we use a simple `panel_score` →
`rank_score` mapping; the unified framework should use **inverse-
variance-weighted** mean across all 4 signals.

---

## 8. Roll-out — phased over multiple commits

Each stage adds one capability; previous stages stay intact.

| Stage | Scope | Default flag | Promotion gate |
|---|---|---|---|
| **0** | Land QP solver as new `JointPortfolioQPTask`, default OFF, validate parity in unit tests | `joint_qp.enabled=false` | All current tests green |
| **1** | Replace JointActionTask under flag — exact greedy parity verified on R7 inputs | OFF | Identical outputs on synthetic test panels |
| **2** | Enable cost-aware MPC (Garleanu-Pedersen partial-move) | OFF | Sim APY ≥ golden − 0.5pt, Sharpe ≥ +0.1 |
| **3** | Constantinides band layer (no-trade region) | OFF | Sim trade frequency ↓ ≥ 30%, Sharpe non-decreasing |
| **4** | Grossman-Zhou DD scaler (replace binary breaker) | OFF | Worst DD ↓ ≥ 20% on golden universe |
| **5** | Garlappi-Uppal-Wang robust μ | OFF | OOS Sharpe ↑ ≥ +0.15 (variance-reduction win) |
| **6** | Treynor-Black signal combination | OFF | OOS IC ↑ ≥ +0.005 (additive over panel-only) |
| **7** | CVaR risk term (Rockafellar-Uryasev) | OFF | Tail-CVaR ↓ ≥ 25% with no APY loss |

Per CLAUDE.md §2a, intermediate stages where the *theory predicts a
specific small-margin win* (not a blind sweep) qualify for the
"theory-clean parity-driven" promotion exception. E.g., Stage 4
(DD scaler) replaces a known-suboptimal binary cliff with a smooth
controller — any positive margin is meaningful evidence.

---

## 9. Implementation surface

```
backtesting/renquant_104/
├── kernel/
│   ├── portfolio_qp/                     ← NEW directory
│   │   ├── __init__.py
│   │   ├── qp_solver.py                  ← cvxpy formulation, ECOS/SCS backends
│   │   ├── covariance.py                 ← Σ from panel residuals + Ledoit-Wolf
│   │   ├── cost_model.py                 ← Almgren-Chriss + Bouchaud impact
│   │   ├── dd_controller.py              ← Grossman-Zhou scaler
│   │   ├── robust_mu.py                  ← Garlappi-Uppal-Wang transformation
│   │   ├── no_trade_band.py              ← Constantinides / Davis-Norman
│   │   └── signal_combiner.py            ← Treynor-Black weighting
│   └── pipeline/
│       └── task_joint_portfolio_qp.py    ← Task wrapper, plugs into InferencePipeline
└── tests/
    ├── test_portfolio_qp_solver.py        ← parity, edge cases, infeasibility handling
    ├── test_dd_controller.py              ← monotone in DD, smooth at limit
    ├── test_no_trade_band.py              ← Davis-Norman closed form match
    ├── test_robust_mu.py                  ← Garlappi-Uppal-Wang sanity
    └── test_qp_vs_greedy_parity.py       ← QP with φ=0 matches greedy on synthetic panels
```

Total: ~1500 LOC + ~800 LOC tests over 8 commits, behind flags. Each
commit is independently shippable (defaults OFF preserve behaviour).

**Dependencies**:
- `cvxpy` (BSD, stable, pip-installable; already a transitive dep via
  scipy). Backends: ECOS for small problems (≤100 assets, our case),
  SCS for larger.
- No new GPU dependency.

---

## 10. References — comprehensive

### Foundational portfolio theory
- Markowitz, H. 1952. "Portfolio Selection." *Journal of Finance* 7(1): 77–91.
- Pogue, G.A. 1970. "An Extension of the Markowitz Portfolio Selection Model to Include Variable Transaction Costs..." *JF* 25(5): 1005–1027.
- Merton, R.C. 1971. "Optimum Consumption and Portfolio Rules in a Continuous-Time Model." *Journal of Economic Theory* 3(4): 373–413.
- Merton, R.C. 1973. "An Intertemporal Capital Asset Pricing Model." *Econometrica* 41(5): 867–887.

### Multi-period / cost-aware
- Constantinides, G.M. 1986. "Capital Market Equilibrium with Transaction Costs." *JPE* 94(4): 842–862.
- Davis, M.H.A., Norman, A.R. 1990. "Portfolio Selection with Transaction Costs." *MOR* 15(4): 676–713.
- Liu, H., Loewenstein, M. 2002. "Optimal Portfolio Selection with Transaction Costs." *RFS* 15(3): 805–835.
- Gârleanu, N., Pedersen, L.H. 2013. "Dynamic Trading with Predictable Returns and Transaction Costs." *JF* 68(6): 2309–2340.
- Boyd, S., Busseti, E., Diamond, S., Kahn, R. 2017. "Multi-Period Trading via Convex Optimization." *Foundations and Trends in Optimization* 3(1): 1–76.

### Drawdown control
- Grossman, S.J., Zhou, Z. 1993. "Optimal Investment Strategies for Controlling Drawdowns." *Mathematical Finance* 3(3): 241–276.
- Cvitanić, J., Karatzas, I. 1995. "On Portfolio Optimization under 'Drawdown' Constraints." *IMA Vol. in Math. and Its Apps.* 65: 35–46.
- Carr, P., Zhang, H., Hadjiliadis, O. 2011. "Maximum Drawdown Insurance." *International Journal of Theoretical and Applied Finance* 14(8): 1195–1230.
- Davis, M.H.A., Lleo, S. 2008. "Risk-Sensitive Benchmarked Asset Management." *Quantitative Finance* 8(4): 415–426.

### Cost / impact
- Almgren, R., Chriss, N. 1999. "Value Under Liquidation." *Risk* 12(12): 61–63.
- Almgren, R., Chriss, N. 2001. "Optimal Execution of Portfolio Transactions." *Journal of Risk* 3(2): 5–39.
- Bouchaud, J.-P., Bonart, J., Donier, J., Gould, M. 2018. *Trades, Quotes and Prices: Financial Markets Under the Microscope.* Cambridge UP.
- Cont, R., Stoikov, S., Talreja, R. 2010. "A Stochastic Model for Order Book Dynamics." *Operations Research* 58(3): 549–563.

### Stop-loss / path-dependent
- Kaminski, K.M., Lo, A.W. 2014. "When Do Stop-Loss Rules Stop Losses?" *Journal of Financial Markets* 18: 234–254.
- Han, Y., Zhou, G., Zhu, Y. 2016. "Taming Momentum Crashes: A Simple Stop-Loss Strategy." *SSRN*.

### Robust optimization / uncertainty
- Garlappi, L., Uppal, R., Wang, T. 2007. "Portfolio Selection with Parameter and Model Uncertainty: A Multi-Prior Approach." *RFS* 20(1): 41–81.
- Black, F., Litterman, R. 1992. "Global Portfolio Optimization." *FAJ* 48(5): 28–43.
- Pástor, Ľ. 2000. "Portfolio Selection and Asset Pricing Models." *JF* 55(1): 179–223.
- Rockafellar, R.T., Uryasev, S. 2000. "Optimization of Conditional Value-at-Risk." *Journal of Risk* 2(3): 21–42.
- Rockafellar, R.T., Uryasev, S. 2002. "Conditional Value-at-Risk for General Loss Distributions." *Journal of Banking & Finance* 26(7): 1443–1471.

### Kelly / sizing
- Kelly, J.L. 1956. "A New Interpretation of Information Rate." *Bell System Technical Journal* 35(4): 917–926.
- Browne, S. 1995. "Optimal Investment Policies for a Firm with a Random Risk Process." *MOR* 20(4): 937–958.
- Thorp, E.O. 2006. "The Kelly Criterion in Blackjack, Sports Betting, and the Stock Market." Ch. 9 in *Handbook of Asset and Liability Management* (Vol. 1), eds. Zenios, Ziemba.

### Active management / signal combination
- Grinold, R.C. 1989. "The Fundamental Law of Active Management." *JPM* 15(3): 30–37.
- Grinold, R.C., Kahn, R.N. 1999. *Active Portfolio Management.* 2nd ed. McGraw-Hill.
- Treynor, J.L., Black, F. 1973. "How to Use Security Analysis to Improve Portfolio Selection." *JB* 46(1): 66–86.
- Lo, A.W. 2002. "The Statistics of Sharpe Ratios." *FAJ* 58(4): 36–52.

### Behavioural / regime / momentum
- Hong, H., Stein, J.C. 1999. "A Unified Theory of Underreaction, Momentum Trading, and Overreaction in Asset Markets." *JF* 54(6): 2143–2184.
- De Bondt, W.F.M., Thaler, R. 1985. "Does the Stock Market Overreact?" *JF* 40(3): 793–805.

### Machine-learning portfolio research (modern)
- López de Prado, M. 2018. *Advances in Financial Machine Learning.* Wiley.
- Gu, S., Kelly, B., Xiu, D. 2020. "Empirical Asset Pricing via Machine Learning." *RFS* 33(5): 2223–2273.
- Chen, Y., Pelger, M., Zhu, J. 2024. "Deep Learning in Asset Pricing." *Management Science* (forthcoming).

### Hierarchical / risk-parity (alternatives surveyed)
- López de Prado, M. 2016. "Building Diversified Portfolios that Outperform Out of Sample." *JPM* 42(4): 59–69.
- Maillard, S., Roncalli, T., Teïletche, J. 2010. "The Properties of Equally Weighted Risk Contribution Portfolios." *JPM* 36(4): 60–70.

### Stylized facts / empirical microstructure
- Cont, R. 2001. "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." *Quantitative Finance* 1: 223–236.
- Hasbrouck, J. 2007. *Empirical Market Microstructure.* Oxford UP.

---

## 11. Cross-reference

- Component fixes (sub-design) — `doc/components/buy-logic-design.md` (3-gate quality floor, kept as a Stage-1 fallback if QP rollout slips)
- Calibrator pool collapse — `doc/components/calibration-saturation.md`
- Current architecture — `doc/arch/decision-graph-103.md`, `doc/arch/strategy-104.md`
- Existing rotation primitive — `kernel/rotation.py`, `kernel/pipeline/task_rotation.py`
- Existing joint action — `kernel/pipeline/task_joint_actions.py`

---

## 12. Bottom line

We're not adding *more flags*. We're replacing a 3-pass greedy heap +
3 separate emitters with a single convex optimization that's the
production-grade form of the same problem. Each named stage above
maps to a peer-reviewed paper from JF/RFS/JPM/FAJ. With all 7 stages
landed and tuned, this is competitive with the architecture used by
mid-tier quant funds for $100M-$1B AUM. At our $10k scale, most of
the cost terms are negligible — but the structural correctness lets
us scale up without rewriting.
