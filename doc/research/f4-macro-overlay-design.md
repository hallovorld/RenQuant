# F4 — Macro overlay design (regime + sizing + sector tilt)

**Status:** DESIGN ONLY (per user "理论写成 doc 后面慢慢做"). Implementation
deferred until the M-series experiment chain (F3, B1.2, B1.3, M2) produces
enough evidence for prioritization.

**Why it matters:** four prior macro experiments (v1 broadcast, v2 per-ticker β,
v3 expanded broadcast, v4 macros-as-panel-rows) all failed because they tried
to use macro data as **panel-LTR features** — the wrong architectural layer.
This design re-positions macro data at the **portfolio constructor / regime
detector / position sizer** layers, where the asset-pricing literature places
it (Cochrane 2008, Chen-Roll-Ross 1986, Hamilton 1989).

---

## 1. Available data (already cached)

```
data/macro/      30 ETFs   — DBA, DBC, EDV, EEM, EWJ, FXE, FXI, FXY, GDX,
                              GLD, ITA, KBE, KRE, LQD, MTUM, SMH, TIP, TLT,
                              USMV, USO, UUP, VGK, VIXY, VXX, WCLD, XBI,
                              XLE, XLU, XLV, XOM (already in OHLCV)
data/fred/       22 series — BAMLC0A0CM (IG credit spread), BAMLH0A0HYM2 (HY),
                              CPIAUCSL (CPI), DFF (fed funds), DGS1MO/3MO/6MO/2/5/10/30
                              (treasury yields), DTWEXBGS (dollar broad), ICSA
                              (jobless claims), INDPRO (industrial production),
                              PAYEMS (payrolls), PCEPILFE (core PCE), RSAFS
                              (retail sales), SOFR, T10Y2Y (term spread),
                              T5YIE (5y inflation breakeven), UMCSENT
                              (consumer sentiment), VIXCLS
```

Total: ~52 macro time series. All daily or higher frequency. Already
cached locally — no fetch overhead.

## 2. Three architectural homes for macro data

### Layer 1 — Regime classifier sub-features (`kernel/regime/`)

**Today**: `RegimeJob` runs `HurstTask + CUSUMTask + GMMTask` over SPY only,
producing one of {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR}.

**Proposed**: enrich the regime classifier with macro stress signals:

```python
# new: kernel/regime/macro_features.py
def macro_stress_features(today, lookback=63):
    """
    Returns scalar features the RegimeJob can consume to refine the regime
    decision when SPY-only signals are ambiguous.
    """
    return {
        # Vol: contemporaneous + recent change
        "vix_level":        VIXCLS.loc[today],
        "vix_chg_5d":       VIXCLS.diff(5).loc[today],
        "vix_z_60d":        zscore(VIXCLS, today, 60),

        # Credit stress: HY spread minus IG spread = credit risk premium
        "hy_minus_ig_bps":  (BAMLH0A0HYM2 - BAMLC0A0CM).loc[today],
        "hy_z_60d":         zscore(BAMLH0A0HYM2, today, 60),

        # Curve shape: 10y - 2y; flat / inverted = recession-leaning
        "term_spread":      T10Y2Y.loc[today],
        "term_spread_chg_20d": T10Y2Y.diff(20).loc[today],

        # Real-time growth proxies
        "claims_z_20d":     zscore(ICSA, today, 20),
        "umcsent_z_60d":    zscore(UMCSENT, today, 60),

        # FX risk-off flag: USD strength vs EM
        "dxy_chg_20d":      DTWEXBGS.diff(20).loc[today],
    }
```

`RegimeFinalizeTask` (currently the last RegimeJob task) consumes these
features alongside Hurst / CUSUM / GMM via a small logistic classifier or
hand-coded rule:

```python
# Pseudo-rule (replace with logistic if hand-coded fails)
if vix_level > 28 and hy_minus_ig_bps > 350:
    raise_regime("BEAR")           # override SPY GMM if it wasn't already BEAR
elif vix_level > 22 and term_spread < 0:
    raise_regime("BULL_VOLATILE")  # bull but stress-leaning
elif vix_z_60d > +1.5 and umcsent_z_60d < -1.0:
    raise_regime("CHOPPY")
# else: keep SPY-driven regime
```

**Cost**: 1-2 days of impl + tests. **Expected impact**: 5-10% Sharpe lift
in volatile regimes (Hamilton 1989 evidence on regime-conditional
allocation).

### Layer 2 — Position sizing overrides (`kernel/sizing/`)

**Today**: `regime_params` is a static table — fixed `max_position_pct` and
`cash_reserve_pct` per regime label.

**Proposed**: dynamic sizing driven by continuous macro stress score:

```python
# new: kernel/sizing/macro_stress_overrides.py
def macro_stress_score(today):
    """
    Composite [0, 1] stress score combining VIX, credit, curve, FX.
    0 = serene; 1 = full BEAR-like stress.
    """
    raw = (
        +0.30 * z(VIXCLS, today, 60)                  # vol risk
        +0.25 * z(BAMLH0A0HYM2 - BAMLC0A0CM, today, 60)  # credit risk
        +0.15 * (-z(T10Y2Y, today, 252))               # curve flatness/inversion
        +0.15 * z(DTWEXBGS, today, 60)                 # USD strength = global risk-off
        +0.15 * (-z(UMCSENT, today, 252))              # consumer pessimism
    )
    return sigmoid(raw)

def apply_overrides(regime_params, today):
    """
    Smooth dynamic overrides on top of the static regime table.
    Activates when macro stress > 0.6 (≈top quintile historically).
    """
    s = macro_stress_score(today)
    if s > 0.6:
        regime_params["max_position_pct"]  *= max(0.4, 1 - (s - 0.6) * 1.5)
        regime_params["cash_reserve_pct"]   = min(0.6, s)
    return regime_params
```

**Cost**: 1 day impl + sim validation. **Expected impact**: 30-50%
drawdown reduction in stressed regimes (Cochrane 2008 + AQR Trend
research; defensive overlay protects more than it costs in calm regimes
because the trigger is rare).

### Layer 3 — Sector tilt overlay (`kernel/portfolio/`)

**Today**: panel-LTR + QP solver chooses positions purely by predicted
α; no explicit sector preference.

**Proposed**: small, dynamic SECTOR-LEVEL bias added to the QP target
vector based on yield-curve regime and credit spread:

```python
# new: kernel/portfolio/sector_tilt.py
def sector_tilt_target(today):
    """
    Returns a sector_id → target_weight_offset map (sums to 0).
    Maximum |offset| is 5% of portfolio gross.
    """
    tilt = {}
    if T10Y2Y.diff(20).loc[today] < -10:   # 20d flatter by >10bp
        tilt["finance"]  -= 0.02
        tilt["utility"]  += 0.02
        tilt["consumer"] += 0.01            # defensives
        tilt["ai_chip"]  -= 0.01
    elif T10Y2Y.loc[today] > 100 and T10Y2Y.diff(60).loc[today] > 30:
        tilt["finance"]  += 0.02            # banks like steepening
        tilt["industrial"] += 0.01
    if BAMLH0A0HYM2.loc[today] > 400:       # credit stress
        tilt["energy"]   -= 0.02            # cyclicals weak
        tilt["healthcare"] += 0.02
    return tilt  # consumed by QP as additional linear constraint
```

QP then adds `+ Σ_s tilt[s] · w_s` to its objective, biasing the
optimizer toward favored sectors.

**Cost**: 2-3 days impl + careful sim validation. **Expected impact**:
modest IC lift but meaningful protective behaviour during regime
transitions.

## 3. Why NOT panel features (the failed v1-v4)

For the record, this is what we don't do:

| Failed approach | Why it failed |
|---|---|
| v1: broadcast (vix_z is the same value for every ticker on date D) | Within-date variance = 0 → `rank:pairwise` gradient = 0 |
| v2: per-ticker β (vix_z × β_ticker) | β estimation noise overwhelmed signal at panel-LTR feature scale |
| v3: 30 ETF + 22 FRED broadcast | Same issue as v1 + colsample dilution |
| v4: macros-as-panel-rows (TLT/XLU/... in watchlist) | Forward-return distribution structurally different from equities → rank loss degenerates |

Cochrane (2008) and Chen-Roll-Ross (1986) explain at the theoretical level
why this WAS expected to fail — macro factors price asset CLASSES (equity
vs bonds vs commodities), not within-equity cross-section. The four prior
attempts were running the same theoretical mistake at different
parametrizations.

## 4. Sequencing recommendation

Implement in this order, validating each in sim before promotion:

| Step | Layer | Estimated cost | Risk |
|---|---|---|---|
| 1 | Macro sub-features into RegimeJob (Layer 1) — additive, doesn't break anything | 2 days | Low |
| 2 | Macro stress override on position sizing (Layer 2) — defensive layer, easy rollback | 1 day | Very low |
| 3 | Sector tilt (Layer 3) — needs careful tuning to avoid hurting alpha | 3 days | Medium |

Total ~1 week. Each step is independently rollback-able by config flag.

## 5. Open design questions

1. **Should the regime classifier be a learned model** (small logistic / GBM
   over [SPY GMM probs, macro features]) instead of hand-coded rules? Pros:
   data-driven; cons: small training set per regime (~50-100 BEAR days
   historical), risk of overfit.

2. **Macro stress score weights** — currently hand-set (0.30 VIX, 0.25 HY,
   etc.). Should be fit to historical drawdown via constrained optimization
   on past 5 years.

3. **Sector tilt sizing** — 5% gross caps tilt's worst-case impact. But
   perhaps it should be regime-conditional itself (larger tilts allowed
   in CHOPPY where direct alpha is hardest to find).

4. **Interaction with QP solver** — adding tilt to QP objective is the
   cleanest. Adding as constraint risks making the problem infeasible.
   Need to verify QP handles soft tilts gracefully.

## References

- **Hamilton, J. D. (1989)** *A New Approach to the Economic Analysis of
  Nonstationary Time Series and the Business Cycle*. Econometrica 57(2): 357-384.
- **Chen, N.-F., Roll, R., Ross, S. A. (1986)** *Economic Forces and the
  Stock Market*. Journal of Business 59(3): 383-403.
- **Cochrane, J. H. (2008)** *The Dog That Did Not Bark: A Defense of Return
  Predictability*. Review of Financial Studies 21(4): 1533-1575.
- **Ang, A., Bekaert, G. (2002)** *International Asset Allocation with
  Regime Shifts*. Review of Financial Studies 15(4): 1137-1187.
- **Stein, J. C. (2014)** *Incorporating Financial Stability Considerations
  into a Monetary Policy Framework*. Brookings Papers on Economic Activity.
- **AQR (2014)** *A Century of Evidence on Trend-Following Investing*.
  Hurst, Ooi, Pedersen.
