# Macro data expansion — research plan (2026-04-27)

User question (2026-04-27): 还有哪些数据应该放进模型训练里？军工？石油？
借贷利率？美元指数？欧洲股市指数？人民币指数？大宗商品？期货？etf？

**Goal**: Increase macro information content in the panel-LTR feature
space, beyond the current 11-symbol macro frame (VXX/HYG/UUP/DBC/GLD/
TLT/XLV/XLU/KRE/MTUM/USMV).

## Key constraint from prior experiments

**Pure broadcast macro adds 0 within-date variance** (proven empirically:
8-variant tournament showed XGB+macro v1 broadcast OOS IC −19% vs
no-macro). So new macro **must be ingested as per-ticker β / sensitivity
features** (the macro v2 path) — not raw broadcast.

## Paper references (evidence base)

### Foundational

- **Welch & Goyal (2008)** *"A Comprehensive Look at the Empirical Performance of Equity Premium Prediction"* RFS 21(4): 1455–1508 — kitchen-sink macro variables (default spread, term spread, dividend yield, etc.) underperform constant out-of-sample for return prediction. Critical baseline: **just throwing macro data at a model fails**.
- **Gu, Kelly, Xiu (2020)** *"Empirical Asset Pricing via Machine Learning"* RFS 33(5): 2223–2273 — 94 stock + 8 macro features, neural nets > linear. Macro: dividend-price ratio, earnings-price ratio, book-to-market, T-bill yield, term spread, default spread, stock variance, net equity expansion. Strong evidence ML can extract macro signal IF combined with stock-level features.
- **Kelly, Pruitt, Su (2019)** *"Characteristics are Covariances: A Unified Model of Risk and Return"* J. Finance 74(4): 1791–1843 — IPCA framework. Each stock has time-varying β to latent macro factors estimated jointly with the cross-section. **Direct theoretical justification for our macro v2 per-ticker β approach.**
- **Avramov, Cheng, Metzker (2023)** *"Machine Learning vs. Economic Restrictions"* J. Finance — confirms per-stock conditioning works; raw macro features as direct inputs underperform. Validates the v2 redesign.

### Specific data source studies

- **Driesprong, Jacobsen, Maat (2008)** *"Striking Oil: Another Puzzle?"* J. Financial Economics 89(2): 307–327 — oil price changes predict cross-sectional stock returns, especially in non-energy sectors (delayed reaction).
- **Boons (2016)** *"Macroeconomic Risk Factors and the Cross-Section of Stock Returns"* J. Banking & Finance — industrial production growth, default spread, term spread, expected inflation, real consumption growth — all priced in cross-section.
- **Bansal, Yaron (2004)** *"Risks for the Long Run"* J. Finance — long-run consumption growth and consumption variance as state variables. Implies term-spread + inflation expectations matter.
- **Lustig, Verdelhan (2007)** *"The Cross Section of Foreign Currency Risk Premia and Consumption Growth Risk"* AER — currency factors carry equity-relevant macro information.
- **Adrian, Crump, Moench (2015)** *"Regression-based estimation of dynamic asset pricing models"* J. Financial Economics — Treasury yield curve as dynamic macro factor proxy. Proven channel into equity returns.
- **Asness, Moskowitz, Pedersen (2013)** *"Value and Momentum Everywhere"* J. Finance 68(3): 929–985 — global value/momentum across asset classes. Cross-asset signals work cross-sectionally.
- **Frazzini, Pedersen (2014)** *"Betting Against Beta"* J. Financial Economics 111(1): 1–25 — β to market matters; extends to β to other factors.
- **Bekaert, Hodrick, Zhang (2009)** *"International stock return comovements"* J. Finance — international equity correlations + drivers.
- **Cieslak, Povala (2015)** *"Expected returns in Treasury bonds"* RFS — yield-curve based factors price both bonds and stocks.

### China/CNY-specific

- **Burdekin, Tao (2021)** *"China's Renminbi exchange rate vs. US-China interest rate differential"* J. Asian Economics — CNY exposure matters for US equity sectors with China revenue concentration (semiconductors, consumer electronics, industrial commodities).

### Recent ML-finance integrations

- **Liu, Zhou, Li, Yan, Liu (2023)** *"Machine Learning for High-Frequency Cross-Section Stock Return Prediction"* — adds 100+ macro features via gradient boosting, finds **non-linear interactions** between macro and stock-level factors carry the signal (not raw features).
- **Two Sigma 2024** *"A Machine Learning Approach to Regime Modeling"* — 4-state mixture model on macro+style factors gates which strategy is active. Different from feature-based approaches.

## Proposed data sources by tier

### Tier 1 — ETF proxies via existing yfinance pipeline (1-2 days)

Already have infra (`MacroFactorStore` + `compute_per_ticker_macro_betas`).
Just add symbols to config + cache them.

| Category | Symbol | What it captures | Paper reference |
|---|---|---|---|
| **Defense / aerospace** | `ITA` (iShares US Aerospace & Defense) | Geopolitical risk premium; per-ticker β isolates contractors (LMT, RTX, NOC) | Boons 2016 |
| **Oil price** | `USO` (US Oil Fund), `XLE` (Energy SPDR) | Oil-price exposure for industrials, transports, materials | Driesprong et al. 2008 |
| **Semis** | `SMH`, `SOXX` | Semi-cycle exposure for ANET/AMAT/AVGO/etc.; β isolates upstream vs downstream | Frazzini-Pedersen 2014 (factor β) |
| **Banks / financials** | `KBE` (banks), `XLF` | Yield-curve sensitivity, credit cycle | Adrian et al. 2015 |
| **Biotech** | `XBI`, `IBB` | Pharma policy risk, FDA cycle | Sector-specific β |
| **Gold miners** | `GDX`, `GDXJ` | Inflation hedge β | Bansal-Yaron 2004 |
| **Cloud / SaaS** | `WCLD`, `IGV` | Tech subsegment | Sector-specific |
| **China large-cap** | `FXI`, `MCHI` | China revenue exposure (AAPL, TSLA, semis) | Burdekin-Tao 2021 |
| **Emerging markets** | `EEM`, `VWO` | EM growth exposure | Lustig-Verdelhan 2007 |
| **Europe** | `VGK`, `EFA` | European cycle exposure | Bekaert et al. 2009 |
| **Japan** | `EWJ` | JPY/Nikkei exposure | Bekaert et al. 2009 |
| **Currency: EUR/JPY/GBP** | `FXE`, `FXY`, `FXB` | FX exposure for revenue-mix exporters | Lustig-Verdelhan 2007 |
| **Volatility regimes** | `VIXY`, `SVXY` | Vol-regime hedge demand | Bali et al. 2016 |
| **Long-end Treasury** | `EDV` (>20yr), `IEF` (7-10yr) | Term-premium proxy beyond TLT | Adrian et al. 2015 |
| **Investment-grade credit** | `LQD` | Credit spread exposure (with HYG already in panel) | Cieslak-Povala 2015 |
| **Commodities (broader)** | `DBA` (agri), `CPER` (copper), `UNG` (nat gas) | Industrial demand cycle | Boons 2016 |
| **Inflation-protected** | `TIP` | Real-rate exposure | Bansal-Yaron 2004 |

**Implementation**:
- Add to `kernel/macro.py::DEFAULT_MACRO_SYMBOLS`
- Run weekly cron `scripts/refresh_macro_cache.sh` to backfill
- `compute_per_ticker_macro_betas` already handles arbitrary symbol set

**Estimated count**: 11 (current) + 18 (Tier 1) = **29 macro symbols** → with `chg`/`level_z` transforms = ~87 broadcast features → 29 per-ticker β features. The β features are what add cross-sectional signal.

### Tier 2 — FRED API (free Federal Reserve data, 3-5 days)

`fredapi` Python library or `openbb`. Free key from St. Louis Fed.

| Series | Code | What it captures | Frequency |
|---|---|---|---|
| **2Y Treasury yield** | `DGS2` | Short-rate level | Daily |
| **5Y Treasury yield** | `DGS5` | Mid-curve | Daily |
| **10Y Treasury yield** | `DGS10` | Long-rate | Daily |
| **30Y Treasury yield** | `DGS30` | Ultra-long | Daily |
| **10Y-2Y spread** | computed | **Yield-curve slope (recession proxy)** | Daily |
| **Fed funds rate** | `DFF` | Policy rate | Daily |
| **SOFR** | `SOFR` | Funding cost (replaces LIBOR) | Daily |
| **VIX** | `VIXCLS` (or computed) | Equity vol expectations | Daily |
| **MOVE** | (subscription needed) | Treasury vol; proxy via TLT realized | Daily |
| **OAS Investment Grade** | `BAMLC0A0CM` | IG credit spread | Daily |
| **OAS High Yield** | `BAMLH0A0HYM2` | HY credit spread | Daily |
| **Trade-weighted USD** | `DTWEXBGS` | Broad USD strength | Weekly |
| **CPI YoY** | `CPIAUCSL` (computed YoY) | Realized inflation | Monthly |
| **Core PCE YoY** | `PCEPILFE` | Fed's preferred inflation | Monthly |
| **Industrial Production YoY** | `INDPRO` | Real activity | Monthly |
| **Non-farm payrolls** | `PAYEMS` | Labor market | Monthly |
| **Initial unemployment claims** | `ICSA` | High-frequency labor signal | Weekly |
| **Consumer sentiment** | `UMCSENT` | Demand expectations | Monthly |
| **PMI Manufacturing** | `NAPM` | Forward activity | Monthly |
| **Retail sales** | `RSAFS` | Consumer demand | Monthly |
| **5Y breakeven inflation** | `T5YIE` | Inflation expectations | Daily |
| **TED spread** | `TEDRATE` | Bank credit stress (deprecated post-LIBOR) | Daily |

**Implementation**:
- New module `kernel/fred_macro.py` — wrapper around `fredapi.Fred(api_key)`
- Cache to `data/fred/{series}.parquet`
- Forward-fill monthly → daily before β computation
- 22 new series → 22 per-ticker β features
- **Critical**: lag monthly data by 1 release-day to avoid look-ahead

**Estimated incremental**: 29 + 22 = **51 macro symbols/series**

### Tier 3 — Cross-asset / commodity futures (1 week)

| Source | Data |
|---|---|
| **CME futures** (free EOD via Yahoo) | CL=F (crude), GC=F (gold), SI=F (silver), HG=F (copper), NG=F (nat gas), ZN=F (10Y note), 6E=F (EUR), 6J=F (JPY), 6C=F (CAD) |
| **Commitment of Traders** | CFTC weekly large-spec/commercial positioning. Predicts commodity reversals. Free CSV from CFTC. |
| **Real-time treasury yields curves** | FRED already covers; for 1m/3m/6m bills add `DGS1MO`, `DGS3MO`, `DGS6MO` |

**Implementation**: extend `kernel/fred_macro.py` to fetch CFTC + Yahoo futures.
Add to per-ticker β cache.

### Tier 4 — Premium / alt data (research only)

- **Bloomberg / Refinitiv** (subscription): MOVE Treasury vol, OAS by rating bucket, sector earnings revisions.
- **TraderTV / SentimenTrader**: Fed-funds futures-implied rate path, Eurodollar spreads.
- **Earnings revisions** (IBES via FactSet — sub): cross-sectional analyst-revision factor.
- **Options skew** (CBOE skew index, OptionMetrics): tail-risk pricing.
- **Twitter/Reddit sentiment** (already exist; we have insider trades + earnings; add Reddit WSB sentiment scoring via free API `pushshift`).

**Skip Tier 4 until Tier 1+2 saturated.**

## Plan / phases

### Phase 1 (this week)
1. Add Tier 1 ETFs (18 symbols) to `kernel/macro.py::DEFAULT_MACRO_SYMBOLS`
2. Refresh `data/macro/*.parquet` cache (`scripts/refresh_macro_cache.sh`)
3. Run `compute_per_ticker_macro_betas` against expanded set → 29 β features
4. **A/B**: XGB no-macro (PROD ~0.0411) vs XGB + macro v2 with 29 β. Measure post-fix CV IC.
5. Reject any expansion that doesn't beat baseline.

### Phase 2 (next week)
1. Implement `kernel/fred_macro.py` — FRED API + cache
2. Add 22 series → forward-fill to daily → β features
3. **A/B**: XGB + macro v2 (29 ETF β) vs XGB + macro v2 (29 ETF + 22 FRED β). Measure CV IC.
4. Cross-sectionally lag monthly series by 1 release day to avoid look-ahead.

### Phase 3 (research)
1. Per-ticker β to specific commodities (oil for XOM/CVX, copper for FCX, gold for NEM)
2. Time-varying β regime conditioning (β_oil different in BULL_VOLATILE vs CHOPPY)
3. Cross-asset interactions (β_yieldcurve × β_dxy as features per Liu et al. 2023)

## Risks & expected impact

**Risk**: more β features → more dimensions → curse of dimensionality on
75K-row panel. Mitigation: feature selection via per-feature within-date
IC threshold (already in `FeatureDiagnosticTask`). Drop β columns with
|IC| < 0.005.

**Expected IC impact** (educated guesses, post-HIGH-1 baseline 0.0411):
- Tier 1 (18 ETF β): +0.005 to +0.010 (incremental over current 11 macro β)
- Tier 2 (22 FRED β): +0.005 to +0.015 (rates + inflation typically add)
- Combined: **target 0.0470 - 0.0520** post-fix on enlarged macro v2

If we DON'T see +0.005 from Tier 1, that's evidence the panel is
saturated on macro signal — better to spend effort on watchlist
expansion (T2 in main roadmap).

## What we already learned

The 8-variant tournament + post-HIGH-1 fix tells us:
- Pure broadcast macro doesn't work
- Per-ticker β with 11 macros doesn't beat 28-feature no-macro
- We need MORE β features OR different β construction

This plan addresses both: expand to 51+ β features and consider regime-
or interaction-based β formulations in Phase 3.
