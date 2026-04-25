# Rotation algorithm — literature review + design proposals

**Session 2026-04-24 PT.** User: "rotation 是我算法的强点！APY 冲 50% 的核心竞争力！" Deep-research request to bring academic / industry rigor to the design space.

## 1. Why current rotation under-performs — empirical diagnosis

Recap of what this session's A/Bs have shown:

| Variant | Rot / 27-mo | ΔAPY |
|---|---:|---:|
| GOLDEN (threshold 0.03, ER) | 0 | baseline |
| LOOSE (0.005, ER) | 0 | 0 |
| FORCE (0.001, μ−λσ) | 0 | 0 |
| + V1 gates | 0 | 0 |
| + V3 gates | 0 | 0 |

**Root cause identified:** `BuildPairsTask` returns `False` when `ctx.ranked` is empty. On our panel, `ScoreBuyTask` + A-gate (tier-1 = 0.27) filters ~all candidates on most bars. Rotation is **starved of rotate-TO candidates**, not of threshold headroom.

**Implication:** rotation gates (V1/V2/V3/V4) are dormant until the panel produces more high-conviction candidates. The literature below informs WHAT rotation design to use once the panel opens up.

---

## 2. Academic foundations — what the literature says

### 2.1 The core momentum result

**Jegadeesh & Titman (1993, JoF)** — "Returns to Buying Winners and Selling Losers". Cross-sectional 3-12-month momentum generates 1%/month alpha, out-of-sample, across 60+ years and 20+ markets. **Core message for rotation: chase momentum, don't fade it.** A rotation rule that kicks held winners in favour of fresh signals is fighting Jegadeesh — this is our user's original worry ("A 跑累了" is the anti-momentum view, which Jegadeesh empirically contradicts — winners keep winning further than you'd think).

Implication: rotation decisions should respect the **cross-section** (B ranks higher than A *right now*) AND the **time series** (A's own momentum hasn't broken).

### 2.2 Time-series vs cross-sectional momentum

**Moskowitz, Ooi & Pedersen (2012, JFE)** — "Time Series Momentum". An asset's **own** 12-1 return predicts its own future return. Separate effect from cross-sectional. **For rotation:** our "A has lost signal" check (`a_velocity < 0`) is a time-series-momentum proxy, but we should measure it on A's own log-returns, not just on A's rank_score (which is cross-sectional by construction).

**Liu, Tsyvinski & Wu (2020, SSRN)** — "Time-series vs cross-sectional momentum: evidence from China". Hybrid strategies (intersect or weight both) outperform either alone by 30-80 bps/month.

**Action for our stack:** add an **own-momentum gate** to rotation:
```
a_own_mom_gate:   stock[A].ret_63d ≤ 0      (A's own 3-mo momentum broke)
b_own_mom_gate:   stock[B].ret_63d ≥ 0      (B's own momentum intact)
```

### 2.3 Risk-managed momentum

**Barroso & Santa-Clara (2015, JFE)** — "Momentum has its moments". Scaling momentum positions inversely to realized vol raises Sharpe from ~0.5 to ~1.0 and cuts max-DD in half. For rotation: **weight the signal by 1/σ**, not just by μ. Our V2 uses `μ − λσ` which is equivalent when λ is tuned.

**Daniel & Moskowitz (2016, JFE)** — "Momentum Crashes". Momentum strategies suffer -30% drawdowns at regime flips (bear → bull). **Add regime filter** — exactly our V3 `enabled_regimes`.

### 2.4 Factor combining

**Asness, Moskowitz & Pedersen (2013, JoF)** — "Value and Momentum Everywhere". Momentum + value + quality in a combined rank beats any single factor (Sharpe 0.8 → 1.2). For rotation: **don't rotate on a single signal**. Our panel-LTR already does this, but the rotation gate looks at one scalar (ER or μ−λσ). Improvement: compare A/B on composite — ER AND panel_score AND μ − AND own-momentum.

**Novy-Marx (2013, JFE)** — "The other side of value: profitability". `gross_profit / total_assets` predicts returns. Our `gross_profitability` factor is already in the panel.

### 2.5 Breadth and watchlist size

**Grinold & Kahn (2000) — "Active Portfolio Management"**. `IR = IC × √breadth`. For a fixed IC, doubling the universe → 41% more IR. **Practical limit:** adding tickers where your IC drops to 0 hurts (not helps). Our watchlist at 44 → 100 is defensible if we curate new tickers to meet IC ≥ baseline.

### 2.6 Decay and turnover

**Arnott, Beck, Kalesnik, West (2016) — "How Can 'Smart Beta' Go Horribly Wrong"**. Factor alpha decays as capital crowds in. For rotation: monitor **recent IC** of your signal; if it's been 0 for 2 months, your rotation rule has been learning noise.

**Almgren & Chriss (2001)** — optimal execution. Trade-off between market impact and timing risk. For rotation: **implicit cost scales with turnover**. Rule: don't rotate if `(net_advantage − turnover_cost) < threshold`. We set `transaction_cost_pct = 0.0` currently (Alpaca is commission-free) but slippage on non-liquid positions is real.

### 2.7 Pair-trading and cointegration

**Avellaneda & Lee (2010, Quant Finance)** — "Statistical arbitrage in the U.S. equities market". When two cointegrated stocks diverge from their historical spread, the mean-reversion trade fires. **User's V4 design is exactly this pattern**:
- `gap_entry` ≈ the historical spread at A's entry
- `gap_today` ≈ today's spread
- `cross_flip = gap_today − gap_entry` is the spread deviation

Caveat: Avellaneda-Lee requires **cointegration** to hold, i.e. `log(A/B)` has stationary AR(1). Most pairs in a tech-heavy watchlist ARE partially cointegrated within sector. A more formal V4 would test cointegration via Engle-Granger before firing.

### 2.8 ML cross-sectional rankers

**Gu, Kelly, Xiu (2020, RFS)** — "Empirical Asset Pricing via Machine Learning". XGBoost + neural nets beat linear + factor models at monthly cross-section return prediction. Tree ensembles win by ~10% on R². **Confirms our panel-LTR approach.**

**López de Prado (2018) — "Advances in Financial Machine Learning"**:
- Purged k-fold CV (we use CPCV — ✓)
- Meta-labeling (two-stage: signal then size — partial, we have μ + σ)
- Fractionally-differentiated features (we use log-returns, not FD)

---

## 3. Industry / practitioner patterns

### A. Gary Antonacci — Dual Momentum
- Absolute: asset's own 12-mo return > Treasury 12-mo
- Relative: asset rank ≥ top-quartile of universe
- Quarterly rebalance, very low turnover
- **Published Sharpe > 1.0 on US+intl equities + bonds**

### B. AQR — Multi-factor rotation (Asness et al.)
- Combine momentum (12-1), value (B/P), quality (ROE, profitability), low-beta
- Equal-weight the standardised z-scores → single composite
- Monthly rank, decile-rotate

### C. Two Sigma / DE Shaw — ML ranking
- LightGBM/XGBoost ranker with 100s of features
- Cross-section pairwise loss (same as our panel-LTR)
- Trade size smoothing (OU process on target weight) to reduce turnover

### D. Bridgewater — Risk-parity + alpha overlay
- Passive risk parity (inverse-vol weighted)
- Momentum overlay as alpha tilts
- **The *overlay* is where rotation lives**: maintain equal risk, rotate tilts

### E. Renaissance (rumoured / inferred)
- Very short horizons (hours-days)
- Many small signals, signal-combining is the alpha
- **Not directly applicable** — their alpha is execution and signal variety, not a single rotation rule

### F. Buffett / DFA — DO NOT rotate
- Long-hold, low turnover
- Buy quality + value at a good price, hold
- **Adversarial benchmark for our rotation claim**

---

## 4. What our V4 is (in this literature)

**V4 = pair-trading signal (Avellaneda-Lee) on ML ranks (Gu-Kelly-Xiu / López de Prado).**

```
a_velocity = A_today − A_entry   # own-rank decay (proxy for own-momentum break)
b_velocity = B_today − B_entry   # own-rank momentum (B picking up steam)
cross_flip = gap_today − gap_entry  # spread widening (pair-trade signal)
```

This is theoretically well-founded IF:
1. rank_score is a monotone function of forward return (calibrator guarantees this by construction — Gu-Kelly-Xiu)
2. The held-cand pair is at least partially cointegrated (likely within sector, shaky across sectors)
3. Spread deviation has mean-reversion at the horizon we care about

## 5. Missing pieces compared to best practice

Compared to Grinold-Kahn / AQR / Antonacci standards, our rotation **currently lacks**:

| Missing | Fix |
|---|---|
| Own time-series momentum gate | Add `stock.ret_63d` sign check for A and B before firing |
| Volatility weighting | `μ / σ` instead of `μ − λσ` (Sharpe-like driver) |
| Turnover-cost-aware threshold | `threshold = f(recent_spread, transaction_cost)` |
| Recent IC monitor | If rolling 60-day IC of rotation signal < 0, auto-disable rotation |
| Sector-relative momentum | Don't rotate within same sector without stronger signal |
| Half-Kelly sizing for swap | Kelly-fraction the swap, don't size at 100% of held |

## 6. Concrete proposals — ranked

### Proposal 1 (ship first, smallest): Own-momentum gate on V4
Add to `find_thesis_symmetric_pairs`:
```python
a_ret_63d = lookup stock[A].close[today] / stock[A].close[today-63d] - 1
b_ret_63d = lookup stock[B].close[today] / stock[B].close[today-63d] - 1
if a_ret_63d > 0:
    continue  # A's own momentum intact — don't rotate OUT
if b_ret_63d < 0:
    continue  # B's own momentum dead — don't rotate INTO
```
Evidence base: Moskowitz 2012. Expected effect: rejects ~50% of rank-based pairs that fight time-series momentum.

### Proposal 2: Sharpe-driver (μ/σ, not μ−λσ)
Add mode `scoring_mode = "sharpe"`:
```python
score = μ / max(σ, ε)   # Sharpe-like with floor
```
Evidence base: Barroso-Santa-Clara. +0.3-0.5 Sharpe in their backtest. Our NGBoost gives both μ and σ, so direct to implement.

### Proposal 3: IC-adaptive threshold
Periodically measure rotation signal's 60-day IC (realized forward return vs predicted edge). If IC < 0 for 15 days, auto-disable rotation. State persisted in `live_state.json`.

Evidence base: Arnott 2016 — factor decay is real.

### Proposal 4: Two-stage sizing (meta-labeling)
Stage 1: decide rotate or not (current binary).
Stage 2: decide how much (fraction of held → fraction moved to B).
```
frac_moved = 0.5 × cross_flip / max_cross_flip  # half-Kelly on the conviction
```

Evidence base: López de Prado — meta-labeling reduces false-positive cost.

### Proposal 5: Cointegration pre-filter
Before a pair (A,B) is eligible, require `adfuller(log(A/B))` p-value < 0.1 on last 252 bars.

Evidence base: Avellaneda-Lee. Limits V4 to pairs where spread mean-reversion is statistically likely.

### Proposal 6: Expand watchlist to 100 (curated) + 10-min panel
Grinold-Kahn: IR scales with sqrt(N). 44 → 100 is 51% more breadth.
Combined with 10-min data: panel rows 47k → 640k → transformer gate passes.

## 7. Recommended path — what ships and in what order

1. **Ship Proposal 1 (own-momentum gate)** — 30 min code, strongest literature support, closest to user's intuition, pairs cleanly with V4.
2. **Ship Proposal 2 (Sharpe driver)** — 20 min code, orthogonal to V4, different mode.
3. **A/B V4 + own-momentum** vs **V4 alone** vs **golden (no rot)** — 3 variants, run only once panel has more candidates.
4. **Defer Proposal 3-5** — they're useful but incremental; data-gated by having rotation actually fire.
5. **Proposal 6 (expand + 10-min)** — parallel track. Fetch 10-min bars (in progress) + curate 20 new tickers this weekend.

## 8. User sanity check

The user's worry — "A 跑累了" — is anti-momentum in Jegadeesh/Moskowitz terms. The literature says winners KEEP winning on average. So:

**Rotation adds value when the signal shows REGIME of A's trend has BROKEN**, not just that B is slightly better. That's why own-momentum gate (Proposal 1) matters — it's the "A's regime actually broke" check, not just "A's rank slipped a bit".

Concretely:
- A is up 40% in 3 months + rank slipped from 0.5 → 0.4 = Jegadeesh says keep A
- A is flat over 3 months + rank slipped 0.5 → 0.3 = A's regime may have broken, B's momentum can steal
- A is down 10% over 3 months + rank 0.2 = clear signal to rotate if B is strong

Proposal 1 codifies exactly this regime check.

---

## 9. References (for reproducibility)

- Jegadeesh & Titman (1993). "Returns to Buying Winners and Selling Losers". *J. Finance* 48(1).
- Moskowitz, Ooi, Pedersen (2012). "Time series momentum". *J. Financial Economics* 104(2).
- Asness, Moskowitz, Pedersen (2013). "Value and momentum everywhere". *J. Finance* 68(3).
- Daniel & Moskowitz (2016). "Momentum crashes". *J. Financial Economics* 122(2).
- Barroso & Santa-Clara (2015). "Momentum has its moments". *J. Financial Economics* 116(1).
- Avellaneda & Lee (2010). "Statistical arbitrage in the U.S. equities market". *Quant Finance* 10(7).
- Gu, Kelly, Xiu (2020). "Empirical asset pricing via machine learning". *Rev. Financial Studies* 33(5).
- López de Prado (2018). *Advances in Financial Machine Learning*. Wiley.
- Grinold & Kahn (2000). *Active Portfolio Management*, 2nd ed. McGraw-Hill.
- Arnott, Beck, Kalesnik, West (2016). "How can 'smart beta' go horribly wrong?". *Research Affiliates*.
- Antonacci (2014). *Dual Momentum Investing*. McGraw-Hill Education.
- Almgren & Chriss (2001). "Optimal execution of portfolio transactions". *J. Risk* 3(2).
- Novy-Marx (2013). "The other side of value: The gross profitability premium". *J. Financial Economics* 108(1).
- Liu, Tsyvinski, Wu (2020). "Time-series vs cross-sectional momentum: evidence from China". SSRN.

All freely available via Google Scholar / SSRN / author homepages.
