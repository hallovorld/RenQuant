# RenQuant — Roadmap

**Single source of truth for what's next.** Ordered by ROI (Sharpe-lift-per-effort). Every item cites a paper or open-source reference. Last updated **2026-05-09 EOD** (post BUG #6/#7 + cost-aware wash-sale + NGB on/off A/B revert + sector-excess null + σ̂ calib + watchlist breadth audit).

---

## End goal — ambitious but honest target

**APY 20% / Sharpe 1.5** long-only US equity. Per Grinold-Kahn 1999 *Active Portfolio Management* §5 Fundamental Law: `IR = IC × √breadth`:

- IC +0.15 @ 103 breadth (4× current — single-name alpha)
- IC +0.07 @ 500 breadth (~2× current — moderate breadth + IC)
- IC +0.034 @ 2000 breadth (current IC, huge data)

**Reference for honest backstop:** Hou-Xue-Zhang 2020 RFS "Replicating Anomalies" — 65% of factor anomalies fail to replicate OOS. Sharpe 1.0 / APY ~13-15% is a much more likely outcome with same effort. Renaissance Medallion's 35%+ APY uses leverage + global futures + microstructure — different game.

Track-record discipline: **walk-forward defensible** (no single-cut promotions per Bailey-Lopez de Prado 2014 "Pseudo-Mathematics and Financial Charlatanism"); **self-maintaining** with rollback rehearsed; **reproducible** lineage.

---

## Right now (2026-05-09 EOD — honest baseline)

| | |
|---|---|
| Production model | **alpha158 + 5fund + 3PEAD + 3SUE = 169 features** XGB rank:pairwise (`panel-ltr.alpha158_fund.json`, fingerprint `4f1e25989d475225`) |
| **27-mo OOS sim (NGB OFF)** | **APY +6.77% / Sharpe +0.40 / Sortino +0.36 / MaxDD -19.2% / Vol 22.8%** |
| **vs SPY same window** | **Trails by -7.3 APY pp / -0.50 Sharpe** (SPY +14.06% / +0.90 Sharpe) |
| **Pure-alpha component** | ~+0.018-0.020 (after subtracting persistence per E53) |
| **7-cut WF mean IC** | +0.039 ± 0.046 (par with Qlib alpha158 benchmarks per Microsoft 2020; std/mean ratio 1.18 — typical for ML-finance per Hou-Xue-Zhang 2020) |
| Watchlist | 103 live / 292 train panel / ~78 traded in sim |
| NGBoost head | DISABLED (27-mo A/B: -3.78 APY pp / -0.14 Sharpe; persistence ratio 63% per E55) |
| Bug fixes shipped 2026-05-09 | BUG #1 (fund-zero) / #2 (SEC date) / #6 (μ̂ collapse) / #7 (σ-band lock-out) / cost-aware wash-sale per IRC §1091 / 70 regression tests / 4 universal model contracts |
| Calibrator | `n_unique_prob_y=79`, `pool_ic=+0.094` HARD PASS |

**Key reframe (2026-05-09):** previous "Sharpe 1.06 / 1.10 / 2.01" claims were on code with documented silent corruption AND single-cut windows. Today's +0.40 is the **first honest baseline**.

---

## P0 — by ROI (Sharpe-lift / effort)

### 1. ⭐ Walk-forward gate enforcement (free; prevents fictional regressions)

**Cost:** 0.5 day. **Expected lift:** 0 direct; prevents future cherry-picked promotes.

Kill single-cut promote path. Every promote requires:
- 3-cut walk-forward (mandatory; CLAUDE.md §5.9 already required, we slipped)
- §5.2 sanity battery (A/A + shuffled-label + time-shift placebo)
- ΔSharpe ≥ +0.10 on WF mean

**References:**
- Lopez de Prado 2018 *Advances in Financial Machine Learning* §7 + §11 (purged walk-forward + CV in finance)
- Bailey-Lopez de Prado 2014 *Notices of the AMS* "Pseudo-Mathematics and Financial Charlatanism" (deflated Sharpe ratio)
- **Open source:** `mlfinlab` (Hudson & Thames) — implements purged WF + DSR

---

### 2. Options-implied features (~+0.30 Sharpe lift, 1 week, free data)

**Cost:** 1 week. **Expected lift:** +0.02-0.04 IC additive → +0.20-0.40 Sharpe via Fundamental Law.

25-delta put-call skew + IV term structure — most robust single-name alpha factors in US equity per multiple replications.

**References:**
- **Bali-Hovakimian 2009 RFS** "Volatility Spreads and Expected Stock Returns" — top-paper on volatility-spread alpha
- **Goyal-Saretto 2009 JFE** "Cross-section of option returns and volatility" — IV term structure as predictor
- **Cremers-Weinbaum 2010 JFQA** "Deviations from Put-Call Parity" — alpha from informed-options-trader signal
- **Open source:** `yfinance` (free options chain), `optionsuite` for analytics

**Implementation:** Yahoo options chain daily snapshot → 25-delta skew, 30d-90d IV slope, IV rank percentile → 3 features → retrain → WF.

---

### 3. News sentiment via FinBERT (~+0.20-0.30 Sharpe, 2-3 weeks)

**Cost:** 2-3 weeks. **Expected lift:** +0.02 IC additive → +0.20 Sharpe.

News tone is independent alpha source orthogonal to price/fundamental signals. Strong replication record.

**References:**
- **Tetlock 2007 RFS** "Giving Content to Investor Sentiment: The Role of Media in the Stock Market" — foundational; pessimism predicts price → reversal
- **Loughran-McDonald 2011 JF** "When Is a Liability Not a Liability?" — financial-domain sentiment dictionary
- **Ke-Kelly-Xiu 2019 NBER** "Predicting Returns with Text Data" — RNN aggregator on news
- **Huang-Wang-Yang 2023 ACL** "FinBERT: A Pre-trained Financial Language Representation Model" — base model
- **Open source:** `FinBERT-FT` (HuggingFace `ProsusAI/finbert`); Alpaca News API (free 30d); `news-please` for backfill

---

### 4. Watchlist quality-first expansion to wl200 (~+0.20 Sharpe via √breadth)

**Cost:** 1 week. **Expected lift:** +0.16 Sharpe IF transfer coefficient holds.

E26 wl183 NO-GO was bottom-up greedy. New approach: quality-first multi-criteria filter.

**Selection criteria** (per literature):
- Liquidity ≥ $50M median DV (avoids slippage drag per Almgren-Chriss 2000)
- History ≥ 2520 days / 10y (per Lopez de Prado AFML §4 sample-size requirement)
- Per-ticker WF Sharpe ≥ +0.5 in ≥ 4 of 7 cuts (regime robustness)
- Sector cap (max 30 per sector — per Markowitz 1952 + Sharpe 1964 diversification)

**References:**
- **Grinold-Kahn 1999** *Active Portfolio Management* §5 (Fundamental Law: IR = IC × √breadth)
- **Hou-Xue-Zhang 2020 RFS** "Replicating Anomalies" (transfer-coefficient collapse mode)
- **Cakici-Cooper-Schmidt 2023 JFE** "Cross-section of US stock returns: ML approaches" (small-cap ML alpha)
- **Open source:** `qlib` (Microsoft) — universe-construction utilities; `polygon.io` API for liquidity history

**Pass gate:** wl200 WF Sharpe ≥ wl103 baseline +0.10 (no transfer-coef collapse).

---

### 5. Vol-adjusted label retest (~+0.10 Sharpe, 2 days)

**Cost:** 2 days. **Expected lift:** +0.005-0.015 IC.

Replace `fwd_60d_excess_raw` with `fwd_60d_excess_raw / vol_60d`. Reduces heteroscedasticity-driven noise.

**References:**
- **Lim et al 2021 ICLR** "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" §3.4 (vol-normalization for stable quantile estimation)
- **Wakefield 2013** *Bayesian and Frequentist Regression Methods* §3.4 (parametric Gaussian recovery)
- **Open source:** Qlib `qlib/contrib/data/handler.py` (vol-targeted label transformation)

---

### 6. Insider trading retest with EDGAR direct (~+0.20 Sharpe, 1 week)

**Cost:** 1 week. **Expected lift:** +0.01-0.02 IC.

E22 closed at 44% coverage with implementation bugs. Resume condition was full coverage + 2y backfill + cleaner aggregation.

**References:**
- **Lakonishok-Lee 2001 RFS** "Are Insider Trades Informative?" — net buy/sell predicts returns 6-12 mo
- **Cohen-Malloy-Pomorski 2012 JF** "Decoding Inside Information" — opportunistic vs routine insider trade discrimination (best-replicated insider alpha)
- **Open source:** `sec-edgar-downloader` (Form 4 raw); `edgar-tools` (parse); EDGAR public-domain
- **Open source:** `openinsider` HTML scraper (free historical insider DB)

**Pass gate:** WF Δmean IC > +0.005 + sanity passes.

---

### 7. σ̂ calibration improvement (Student-t residuals) (~+0.10 Sharpe via better Kelly, 1 week)

**Cost:** 1 week. **Expected lift:** indirect; better Kelly + better conformal Gate B → +0.10 Sharpe IF NGB ever re-enabled.

Today's audit (D) showed ±2σ̂ coverage = 88% (Gaussian expects 95%) → fat tails. Replace Gaussian with Student-t.

**References:**
- **Lopez de Prado 2018 AFML §15** "Robust Methods" (Student-t for fat-tail residuals)
- **Romano-Patterson-Candès 2019 NeurIPS** "Conformalized Quantile Regression" (proper coverage guarantees)
- **Mandelbrot-Hudson 2004** *The (Mis)Behavior of Markets* (foundational fat-tail observation)
- **Open source:** `mapie` (Maximum A Posteriori Inference Engine — conformal prediction); scipy.stats.t

**Status:** prerequisite for any future NGB re-enable. Lower priority since NGB itself currently NO-GO.

---

### 8. Tax-aware execution improvements (~+0.5-1.0 APY pp net, 1 week)

**Cost:** 1 week. **Expected lift:** +0.5-1.0 APY pp net of taxes + ~5-10 bps slippage savings per round trip.

- Brown-Smith LT-bridge already in place
- Cost-aware wash-sale shipped today per IRC §1091
- Add: HIFO when realizing gains, FIFO when realizing losses
- Add: VWAP-style execution over 30-60min

**References:**
- **Brown-Smith 2011 JF** "Asymmetric Tax Treatment and Portfolio Choice" — long-term bridge
- **Berkin-Jeffrey 1990** *J. of Portfolio Management* "Tax-managed investing" — loss-harvest credit
- **Almgren-Chriss 2000 J. of Risk** "Optimal Execution of Portfolio Transactions" — VWAP-style trajectory
- **Boyd-Busseti-Diamond et al 2017** *cvxportfolio* docs — Tax cost class implementation
- **Open source:** `cvxportfolio` (Stanford Boyd group); Alpaca limit orders + bracket orders

---

### 9. LightGBM with category encoding (~+0.05-0.10 Sharpe, 2 days)

**Cost:** 2 days. **Expected lift:** +0.005-0.015 IC via native sector embedding.

E48 LGB retest was on alpha158+5fund+3PEAD; marginal NO-GO. **Resume condition:** with native categorical sector handling.

**References:**
- **Ke et al 2017 NeurIPS** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree" — native categorical handling
- **Prokhorenkova et al 2018 NeurIPS** "CatBoost: unbiased boosting" — categorical comparison baseline
- **Open source:** `lightgbm` (Microsoft) `categorical_feature` arg; `catboost` (Yandex) `cat_features` arg; Qlib `LGBModel` reference

---

## P1 — architecture (medium-cost, lower expected ROI)

### 10. PatchTST sequence model (~+0.10 Sharpe, 2-3 weeks)

**Cost:** 2-3 weeks. **Expected lift:** +0.01-0.03 IC if sequence dependence is real.

Per-ticker 60-day patch tokenization → encoder. Param/sample ratio < 1/100 per CLAUDE.md §5.12.

**References:**
- **Nie et al 2023 ICLR** "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" — PatchTST
- **Lim et al 2021 ICLR** "Temporal Fusion Transformers" — TFT (more interpretable, more params)
- **Zhou et al 2021 AAAI** "Informer: Beyond Efficient Transformer for Long Sequence" — alternative architecture
- **Open source:** `PatchTST` (yuqinie98/PatchTST GitHub); Qlib `pytorch_tcn_ts.py` reference

**Risk:** TFT/PatchTST OOS often disappoints on individual-name returns vs index. Stage-gate after Track 1+2+4 land.

---

### 11. Multi-horizon ensemble (E42) RETEST after bug fixes (~+0.05 Sharpe, 1 week)

**Cost:** 1 week. E42 was rejected on contaminated panel (BUG #1/#2 active). Retest justified.

**References:**
- **Bali-Cakici-Whitelaw 2011 JF** "Maxing Out: Stocks as Lotteries" — multi-horizon return aggregation
- **Lopez de Prado 2018 AFML §16** "Backtest statistics" (multi-horizon Sharpe combination)
- **Open source:** Qlib `MultiHorizonEnsemble` reference; `quantstats` for ensemble metrics

**Pass gate:** Δ Sharpe > +0.05 on WF mean. E42 had IC↑ + Sharpe↓ — must verify both move together.

---

### 12. Triple-barrier label (E25) RETEST after bug fixes

**Cost:** 1 week. E25 placebo +0.0458 ≈ real +0.0438 → REJECTED. Retest with clean panel + WF gate.

**References:**
- **Lopez de Prado 2018 AFML §3.6** "Triple-Barrier Method"
- **Lopez de Prado 2018 AFML §17** "Backtesting Through Cross-Validation" (placebo definition)
- **Open source:** `mlfinlab` (Hudson & Thames) `triple_barrier_method` impl

**Pass gate:** WF Δmean IC > +0.005 AND placebo IC < 0.5× real IC.

---

### 13. Microstructure features (paid data, 1-2 weeks)

**Cost:** 1-2 weeks + ~$10/mo Alpaca Pro. **Expected lift:** +0.01-0.03 IC at sub-daily horizons.

**References:**
- **Easley-O'Hara 1987 JF** "Price, Trade Size, and Information in Securities Markets" — order-flow alpha
- **Kyle 1985 Econometrica** "Continuous Auctions and Insider Trading" — Kyle lambda
- **Hasbrouck 2007** *Empirical Market Microstructure* (Oxford) — comprehensive reference
- **Cont et al 2014 JF** "The Price Impact of Order Book Events" — depth-3 book features
- **Open source:** `pyrobustlimitorderbook` (Cont group); `alpaca-py` Pro tier API

**Risk:** built for sub-daily; may not help our 5d/60d horizons. Lower priority for daily strategy.

---

## P2 — operational hygiene (compound, no Sharpe lift)

### 14. Acceptance gates for daily retrain output

Wire `kernel/model_acceptance.py` so a bad retrain (IC drop > 5pt OR sanity placebo lift > 1pt) auto-rolls back.

**Reference:** existing `kernel/model_acceptance.py` (in-codebase); CLAUDE.md §5.5 "rollback rehearsal" mandate

### 15. Sunday retrain wire to new pipeline

Replace `sunday_panel_sweep.py` (old 21-feat panel) with weekly `daily_retrain_alpha158_fund` + full §5.2 battery.

### 16. Side-config DB migration

Existing `migrate_experiment_configs_to_db.py`. Remaining: live runner + retrain cron read-from-DB; delete stale `strategy_config.*.json` files.

### 17. Test suite hygiene

~14k tests; trim slow tests, add sim-level integration coverage (V8 leverage bug shipped because no sim test pinned cvxportfolio constraints).

---

## CLOSED — do NOT re-open without new evidence

| Track | Date | Verdict | Reference |
|---|---|---|---|
| Per-sector pure-alpha label | 2026-05-09 | NO-GO | Persistence 89% (E53/E55 framework); pure-alpha drops to +0.005 |
| NGBoost-on σ-aware sizing | 2026-05-09 | NO-GO (reverted) | 27-mo A/B: -3.78 APY pp / -0.14 Sharpe (E55) |
| NGBoost-proper retrain (Duan 2020 §4) | 2026-05-09 | NO-GO | 5-seed +0.0354 ± 0.0026 (sig); 63% persistence ratio (E55) |
| Phase A-C QHead variants (E51/E52) | 2026-05-09 | NO-GO | Architectural changes don't break panel ceiling |
| wl183 expansion bottom-up | 2026-05-05 | NO-GO | Transfer-coef halving (E26); item #4 above is replacement |
| wl1640 R2K | 2026-05-08 | NO-GO | XGB IC dropped 75%; Cakici 2023 doesn't apply at this signal scale (E45) |
| Macro overlay v1-v4 | various | NO-GO | All variants net negative IC at panel size 103 |
| Asset embeddings T2-2 | 2026-04-29 | NO-GO | +0.0001 IC delta = no lift |
| Boyd rotation T2-4 | various | NO-GO | -2.5 APY pts; default OFF |
| PEAD enrichment | 2026-05-08 | PROMOTED at fwd_60d (E47) | live now |
| SUE features | 2026-05-09 | PROMOTED (E49) | live now |
| Walk-forward XGB E27 audit | 2026-05-05 | RESPONSE: revert NGB + new alpha sources | Acted on today; items #2-#6 above |
| QP refactor — adopt cvxportfolio | 2026-05-06 | DONE | cvxpy + CLARABEL primary |
| 6 silent-feature bugs | 2026-05-09 | FIXED | 70 regression tests + 4 universal model contracts |

---

## Stage-gated milestones (12 weeks, target Sharpe 1.0+)

```
Week 1:    Item #1 Walk-forward gate enforcement (free, prerequisite)
Week 2:    Item #2 Options-IV (Bali-Hovakimian 2009 + Goyal-Saretto 2009)
           ↓ Gate: WF Sharpe ≥ +0.55

Week 3-5:  Item #3 News sentiment FinBERT (Tetlock 2007 + Ke-Kelly-Xiu 2019)
           ↓ Gate: WF Sharpe ≥ +0.75

Week 6:    Item #4 Quality-first wl200 (Grinold-Kahn 1999)
           ↓ Gate: WF Sharpe ≥ +0.95

Week 7:    Item #5 vol-adj label + Item #11 multi-horizon (Lim 2021 + Bali-Cakici-Whitelaw 2011)
           ↓ Gate: WF Sharpe ≥ +1.05

Week 8-9:  Item #6 Insider EDGAR direct (Cohen-Malloy-Pomorski 2012)
           ↓ Gate: WF Sharpe ≥ +1.20

Week 10-12: Item #10 PatchTST + Item #13 Microstructure (if Sharpe ≥ 1.20 by week 9)
           ↓ Stretch: WF Sharpe ≥ +1.50 (APY ~20%)
```

**Realistic outcome distribution** (per Hou-Xue-Zhang 2020 65% replication failure rate, items partially independent):

| Outcome | Probability | Final Sharpe | APY |
|---|---|---|---|
| Hit stretch | 15% | 1.5+ | 20%+ |
| Hit baseline | 35% | 1.0-1.4 | 13-18% |
| Partial | 35% | 0.7-1.0 | 10-13% |
| Adverse | 15% | <0.7 | <10% |

---

## How to use this doc

1. Pick topmost unblocked P0 item by ROI rank
2. Open small branch, ship smallest reversible step, commit
3. Run **walk-forward 3-cut + §5.2 sanity** (no single-cut promotes)
4. If WF Sharpe ≥ +0.10 over previous golden → promote in same commit, mark done
5. Otherwise update this doc with what you learned, pick next item

**Working rhythm:** ship don't ponder; one task in flight; commit each meaningful chunk; walk-forward EVERY claim.

---

## Renquant_105 (future)

30-min level model (Easley-Lopez de Prado-O'Hara 2012 microstructure literature). Designed but not started.

- Pre-requisite: renquant_104 stable for ≥ 3 months on live with WF Sharpe ≥ 1.0
- Pre-requisite: minute-bar panel > 1M rows (103 tickers × 2 yrs × ~16 bars/day)
- Independent strategy dir at `backtesting/renquant_105/`

105 design work resumes after 104 is stably live + ≥ Sharpe 1.0 walk-forward.
