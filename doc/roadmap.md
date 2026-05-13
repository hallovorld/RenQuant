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

### ★★ Regime-conditional path (2026-05-12 EVENING discovery)

Found via `scripts/eval_regime_stratified.py`: Grinold-Kahn α→μ (commit
`7bc9b56`) is a **regime-conditional winner** — wins +18%/yr in HIGH_CALM
(SPY 60d Sharpe > 1.5, vol pct < 33%, n=123, t=+1.67), loses −32%/yr in
HIGH_SPIKED (SPY 60d Sharpe > 1.5, vol pct > 66%, n=53, t=−1.95). Pooled
verdict is NEITHER because the two regimes cancel; conditional deployment
could extract real edge.

Full findings + forward plan: [`doc/research/2026-05-12-findings-and-next.md`](research/2026-05-12-findings-and-next.md)

**Blockers to acting on this (must be cleared in order):**

| # | Blocker | Effort | Why |
|---|---|---|---|
| **P0-A** | Sticky regime detector — labels 95% BULL_CALM in our 24mo OOS, misses HIGH_SPIKED periods entirely. Fix via SPY trend/vol signals OR replace GMM | 4-6h | Conditional deployment impossible without working regime signal |
| **P0-B** | Extend walkforward manifest 2022-01 → 2024-01, regenerate ~50 cutoffs | 4h | Each regime cell needs n≥200 for Bonferroni-corrected significance |
| **P1** | Add `ranking.X.regime_overrides` config block + reader Task | 2-3h | Currently `regime_params` only governs risk knobs, not ranking |
| **P2** | Re-evaluate all 3 candidates on 36-48mo OOS, regime-stratified | 1 day | Real Tier 3 verdict on conditional deployment |
| **P3** | If GK-in-HIGH_CALM survives Tier 3, flip live config | 1-2h | First regime-conditional production feature |

P0-A and P0-B must precede P1 per §5.13.10 (no dead config paths).

### ★ Methodology lock-in (2026-05-12)

All future variant promotion gates on the **3-tier framework** in
[`doc/research/promotion-methodology.md`](research/promotion-methodology.md):
**Tier 1** REJECT (worse than baseline) · **Tier 2** SCREEN (small consistent edge,
keep iterating) · **Tier 3** CONFIRMED via DSR > 0.5 or PBO < 0.5 or n ≥ 30 t > 3
(live-promotable). Toolchain: `scripts/analyze_experiments.py` walks every
`data/logs/sim_*/W*_<cfg>.log`, applies criteria, emits ranked tier report.
Post-Bug-C re-evaluation (53 configs × 6 windows): 0 Tier 2, 0 Tier 3 →
keep prod baseline; best candidate `E42_fwd60d` (60-day label window) needs
extended walk-forward retest.

### ★ Execution-tactic block (added 2026-05-09 EOD after trade-level audit)

Trade-level audit on Cut 3 (159 closed trades) revealed 3× systematic leakage in exit logic. Highest ROI is fixing **how we exit**, not the model. All 5 fixes implemented as independent config toggles, ablation-tested, WF-validated.

| # | Fix | Reference | Expected impact |
|---|---|---|---|
| 0a | **σ-aware stop_loss** | Wilder 1978 *New Concepts in Technical Trading Systems* §5; Kestner 2003 *Quantitative Trading Strategies* ch.6 | 24 stop_loss exits avg -12.58% (current); ~half over-bleeding due to wrong-vol stops. Estimated +0.5 pp APY |
| 0b | **Time-decayed stop tightening** | Lo 2007 "Adaptive Markets Hypothesis"; Schwager 1992 *New Market Wizards* | After N days held, tighten stop. Catches model-said-exit-but-stop-not-yet bleeders |
| 0c | **Profit-target ladder (1.5σ / 3σ / runner)** | Kestner 2003 ch.7; Murphy 1999 *Technical Analysis*; Schwager *Wizards* | Ladder out 25%/25%/50%. Currently 14 trailing_stop captured +27% all-or-nothing; ladder locks gains progressively |
| 0d | **VWAP execution over MOO** | Almgren-Chriss 2000 J. Risk; Bertsimas-Lo 1998 | -5 to -10 bps slippage savings per RT. Modest but compounds over 300 trades/yr |
| 0e | **single_day_loss σ-aware + unrealized-gate** | Lo-Mamaysky-Wang 2000; observed: 4 SDL exits avg +12.75% (tripping winners) | Stop-out-winners is pure leakage. Add unrealized<0 gate or raise σ multiplier |

**Pass-gate (each):** WF 3-cut Δ Sharpe ≥ +0.05 over current golden + sanity (no exit-reason-specific overfit).

**Ablation order** (cheap-first):
1. Implement all 5 as opt-in config flags (1 day)
2. Single 12-mo window ablation per fix (~30min × 5 = 2.5h)
3. WF 3-cut on winners (~75min × winners)
4. Combine wins, retest combo

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

### 18. ~~Daily dashboard refresh~~ ✅ DONE 2026-05-09

Markdown dashboard at `doc/dashboard.md`, auto-rendered by GitHub. Refreshed by `scripts/build_dashboard.py` wired into `scripts/daily_104.sh` (post-cron, non-fatal).

Sections: portfolio value/P/L/HWM, recent trades, 21d P/L table (deployment-spike-filtered), model health (panel fingerprint + retrain age + latest WF mean IC), top-5 priorities pulled from this roadmap.

**Tests:** `tests/test_dashboard.py` — 12 cases (unit / integration / E2E markdown-validity).

**Manual refresh:** `python scripts/build_dashboard.py --broker alpaca`

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

---

## Bug-C reassessment backlog (added 2026-05-11 PM)

**Context:** Commit `29e34b0` fixed Bug C — `SimAdapter._portfolio_value` omitted T+2 pending settlement balance, creating phantom ±sale_amount returns. The bug inflated Vol by ~75× and MaxDD by ~10×, and via path-dependent vol drag also corrupted compound APY. **Every sim verdict from 2026-04 through 2026-05-11 AM that produced numbers depending on Vol/MaxDD or close-to-noise APY deltas is suspect.**

This backlog catalogs features REJECTED on those suspect numbers. Each item:
1. needs a post-Bug-C re-test before its verdict can be trusted
2. carries a proposed experimental design respecting CLAUDE.md §5.14 DOE methodology
3. is NOT urgent — production is stable post-fix; revisit when bored or hunting for marginal lift

### High-priority revisits

#### B1. Time-decayed stop tightening (Fix #0b)
- **Original verdict (pre-Bug-C):** "actively harmful −4.33 pp APY / −0.20 Sharpe / +23 trades" (see `doc/AUDIT_2026-05-09.md` finding #2)
- **Why suspect:** the −4.33pt was a single-window pre-fix measurement. Sharpe and MaxDD numbers feeding the ablation conclusion were inflated artifacts. The hypothesis (Schwager 1992 / Lo 2007 regime-adaptive stops) is theoretically defensible.
- **Test design:** 3 windows × {stop_decay_days = 30 / 60 / off} = 9 sims. Pass: mean APY ≥ baseline AND Sharpe within 0.10 → revive at decay_days winner.
- **Code path:** `kernel/exits.py` — check if `stop_decay_days` config knob still wired (may have been deleted in cleanup; reviving needs ~20 line patch).

#### B2. SDL skip-if-winner (Fix #0e)
- **Original verdict (pre-Bug-C):** "+0.05 pp APY / +0.00 Sharpe / 1 fewer trade — inside noise floor"
- **Why suspect:** the "noise floor" was inflated by Bug C phantom Vol. Real noise floor is much narrower (post-fix Vol = ~15% vs pre-fix 157%). A +0.05 pp lift that looked like noise might be real signal.
- **Test design:** 3 windows × on/off = 6 sims. Pass: mean APY ≥ +1 pt over baseline AND consistent direction across 2/3 windows.
- **Config knob:** `regime_params.*.sdl_skip_if_unrealized_above` (default 0.0 = always skip; tighter values like 0.05 would be more selective).

#### B3. Multi-horizon ensemble (E42 in roadmap above)
- **Original verdict:** already flagged in roadmap as "RETEST after bug fixes". Bug C is the trigger.
- **Why suspect:** any "no lift" finding before today is contaminated. The pre-fix Vol-distorted Sharpe was the deciding metric.
- **Test design:** roadmap E42 entry already has a plan; bump priority post-Bug-C.

#### B4. Triple-barrier label (E25 in roadmap above)
- Same: roadmap "RETEST after bug fixes" — Bug C is the trigger.

### Medium-priority revisits

#### B5. Trend overlay (R-03)
- **Original verdict:** in mega ablation, "fires but redundant with drawdown halt" — fired 3+ times in W3 log but produced bit-identical numbers to no-trend.
- **Why suspect:** pre-fix drawdown halt was hyperactive (MaxDD inflated → halt fired more often than truly needed). Post-fix drawdown halt fires LESS often (MaxDD truly ~8%), so trend overlay has more ROOM to be the binding signal.
- **Test design:** 3 windows × {trend on / off} with `drawdown_halt_pct` loosened to 0.30 in all regimes (force trend to be the primary regime gate). 6 sims.

#### B6. DD-Kelly scaling (R-04)
- **Original verdict:** "didn't fire in mega" — interpretation was "dead code".
- **Real reason:** doesn't bind when portfolio drawdown < `dd_max` threshold. Post-fix MaxDD is ~8% which is below most reasonable dd_max thresholds → still doesn't bind in current windows.
- **Test design:** 3 windows × {dd_max ∈ {0.05, 0.08, 0.10}}. Smaller dd_max → binds sooner → tests the mechanism. 9 sims.

#### B7. CVaR `qp_cvar_lambda` re-sweep
- **Original verdict:** +0.1pt ± 7.6 pt noise (post-Bug-C, n=3 windows).
- **Why backlog:** the σ is wide. Could resolve with more windows. If signal real, λ optimization would find it.
- **Test design:** Box-Behnken on λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} × 5 windows. 25 sims.

### Low-priority revisits

#### B8. Stop-loss multi-seed L1-L5 ablation (commit `45`)
- **Original verdict:** "all 5 ablations (L1-L5) degraded performance" pre-Bug-C.
- **Why suspect:** ablation deltas were measured against a Bug-C-distorted baseline.
- **Test design:** rerun the 6-arm sweep post-fix on 3 windows = 18 sims.

#### B9. R-01 CVaR / R-05 DD-tight / R-07 robust μ Phase 1 sims (commit `53`)
- Same as B8: pre-Bug-C ablations.

#### B10. Phase 4 Tier-3/4/5 stop-loss final-best configs (commit `63`)
- **Original verdict:** "Tier-N config wins / loses by X pt".
- **Why suspect:** all Tier comparisons used Bug-C-corrupted equity curves.
- **Test design:** rerun the chosen Tier configs vs baseline post-fix on 3 windows.

#### B11. Meta-label E63 — confirm theory-based rejection holds
- **Post-Bug-C verdict:** mean Δ APY = −1.5 pt (within noise). Rejection KEPT on theory (AUC=0.55 random).
- **Why backlog:** if a higher-quality classifier (more events, better features) ever comes along, the framework is still wired. Trigger to retest: when triple-barrier label E25 (B4) or P4.3-style snapshot logger produces > 1500 events with cross-validated AUC > 0.60.

#### B12. The non-reproducible "+6.77% / +1.97%" 27-mo baseline
- **Original verdict:** "non-reproducible across same-day reruns; σ_APY unknown" (CLAUDE.md status).
- **Why backlog:** with Bug C fixed, the 27-mo baseline can now be measured reliably. Worth one clean rerun to establish the long-window reference. 1 sim, ~3 hours.

### Process: how to drain this backlog

1. **One at a time.** No batch — Bug C taught us that contaminated measurements compound across experiments.
2. **§5.13.4 minimum:** mean ± std from ≥ 5 runs (windows OR seeds).
3. **§5.14 DOE:** screening first (single-knob), Box-Behnken only after a knob shows ≥ +0.10 Sharpe range-find.
4. **Document outcome** in `failed-experiments-log.md` whether positive or negative.
5. **Update this backlog entry**: cross off resolved items, surface what was learned.

### What is NOT in this backlog (decisions stand)

- **Meta-label classifier disable** (commit `0cf758d`) — kept disabled on theory (AUC = 0.55 ≈ random). Bug C contaminated the magnitude of harm, not the theoretical basis for rejection.
- **maxpos=8% sweep** — Bug-C-FLIPPED. Pre-fix: looked like winner. Post-fix: clearly loses 8.2 pt. Verdict is now CORRECT direction, not in backlog.
- **Smart orders (Fix #0d)** — orphan code, never wired into prod. Not a measurement issue; architectural.
- **σ-aware stop revival "v6 disaster"** — currently being re-tested as E-σS (Phase 1 screening). Will be resolved tonight, not backlog.


---

## 7-REDO + 4-Backlog experimental plan (2026-05-11 PM)

Following CLAUDE.md §5.11 (range-find first, optimize after) + §5.14 (DOE methodology with mandatory pass criteria).

### Acceptance criteria for every experiment in this plan

Per §5.13.4:
- **PROMOTE**: mean Δ APY ≥ +2.0 pt AND mean Δ Sharpe ≥ +0.10 AND consistent direction in ≥4/6 windows
- **REJECT**: mean Δ APY ∈ [−1, +1] AND |mean Δ Sharpe| < 0.05 → confirmed null
- **INVESTIGATE**: anything else (regime-dependent, mixed) → expand window panel or design BB

### Phase 0 (running now, ETA 23:01 PT): CVaR 5-window confirmation
- 30 sims (5 λ × 6 windows), some redundant for reproducibility
- λ ∈ {0, 0.15, 0.25, 0.35, 0.50}
- Outputs: λ* (optimal), mean Δ vs baseline, DSR/PBO

### Phase 1 (queued, kick off 23:05 PT after Phase 0): config-only feature screening

**Cost:** 18 sims (3 configs × 6 windows), ~30min concurrent wallclock.

| Config | Setting | Hypothesis |
|---|---|---|
| `sim_E43_voltarget_007.json` | `vol_target.enabled=true`, `target_vol=0.07` | At 7%, the vol-target binds always (SPY realized vol always > 7%). Tests if vol-targeting itself adds Sharpe. |
| `sim_B5_trend_isolated.json` | `trend_overlay.enabled=true`, drawdown_halt loosened to 0.30 | Pre-fix trend was subsumed by hyperactive DD halt; isolate trend with halt loosened. |
| `sim_B6_ddkelly_005.json` | `drawdown_scaling.enabled=true`, `dd_max=0.05` | At 5%, DD-Kelly binds early. Tests Grossman-Zhou scaling. |

Launch script staged: `scripts/_phase1_run.sh`

### Phase 2 (overnight, scheduled): heavy retrain experiments

**Cost:** ~12h total CPU. Sequential because each retrain has high RAM footprint.

| Order | Exp | Setup | Runtime | Pass criteria |
|---|---|---|---|---|
| 1 | **E55 NGBoost** | Train sim-side walkforward NGB head (39 cutoffs); copy to sim path; sim with `ngboost.enabled=true` × 6 windows | ~3h | Δ APY ≥ +2.0 vs NGB-off baseline |
| 2 | **E42 multi-horizon** | Train fwd_5d + fwd_20d panel-LTR models; build ensemble scorer; sim × 6 windows | ~3h | Δ Sharpe ≥ +0.10 vs single-horizon (fwd_60d) |
| 3 | **E26 wl183** | Train panel-LTR on wl183 universe; sim × 6 windows | ~4h | Δ APY ≥ +2.0 vs wl103 |
| 4 | **E41 R1K** | Train R1K walkforward panel-LTR; sim × 6 windows | ~5h | Δ Sharpe ≥ +0.10 vs wl103 |

### Phase 3 (free, post-processing): E27 SPY benchmark

**Cost:** seconds. No sim runs.

Post-process EXISTING equity curves vs SPY benchmark for 6 windows:
- Strategy return per window from sim log Final value
- SPY return per window from `data/ohlcv/SPY.parquet`
- Compute alpha = strategy_return − SPY_return
- Pass criteria: mean alpha ≥ +1 pt (the "do we beat SPY?" question)

### Phase 4 (conditional, after Phase 1+2 winners): Box-Behnken optimization

If Phase 1 yields ≥ 2 winners with theoretical interaction (e.g., CVaR + vol-target), launch 3-level Box-Behnken DOE:
- pyDOE2.bbdesign(k_knobs, center=3) × 5 windows
- For 3 knobs: ~75 sims, ~2-3h wallclock
- Fit `y = β₀ + Σβᵢxᵢ + Σβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ²` quadratic response surface
- scipy.optimize.minimize on surface → predicted optimum
- 2-3 confirmation runs at predicted optimum + DSR/PBO

### Phase 5 (skip unless code-revival authorized): B1 / B2 stop-loss revival

These need ~30-line code patches to revive removed knobs:
- B1 time-decayed stop tightening (`stop_decay_days`, removed from kernel/exits.py)
- B2 SDL skip-if-winner (`sdl_skip_if_unrealized_above`, removed)

Both pre-fix verdicts ("−4.33pt" / "noise") were Bug-C contaminated. To resurrect would need:
1. ~30 lines code patch + tests
2. 6-window × on/off A/B
3. ~1.5h total

Defer unless Phase 1-2 prove insufficient.

### Resource budget summary

| Phase | Sims | Wallclock | When |
|---|---|---|---|
| Phase 0 (CVaR) | 30 (running) | 40min | NOW → 23:01 |
| Phase 1 (3 features × 6 win) | 18 | 30min | 23:05 → 23:35 |
| Phase 2.1 E55 retrain + sim | 39 trains + 6 sims | 3h | 23:40 → 02:40 |
| Phase 2.2 E42 retrain + sim | 78 trains + 6 sims | 3h | 02:45 → 05:45 |
| Phase 2.3 E26 retrain + sim | 1 train + 6 sims | 4h | morning |
| Phase 2.4 E41 R1K retrain + sim | 1 large train + 6 sims | 5h | morning-afternoon |
| Phase 3 SPY post-process | 0 sims | seconds | anytime |

### Decision tree at each gate

```
[Phase 0 done] → CVaR λ* found?
  ├ Yes → flip prod config, ship CVaR λ* (commit + tomorrow's live firing uses it)
  └ No  → keep baseline λ=0; CVaR confirmed reject

[Phase 1 done] → Any winner?
  ├ Yes (single) → confirmation 5-window at winner setting + commit
  ├ Yes (multi) → schedule Phase 4 Box-Behnken on top 2
  └ No  → all 3 confirmed inert; close backlog items

[Phase 2.1 E55 done] → NGB on improves?
  ├ Yes → flip prod config; ship σ-aware stop revival (depends on NGB)
  └ No  → keep NGB off; permanently retire σ-aware-stop family

[Phase 2.2-4] → similar gates per experiment

[Phase 3 SPY] → strategy beats SPY?
  ├ Yes → close E27 backlog; rejoice
  └ No  → continue strategy improvement; may rethink universe (E26/E41 win could pivot)
```

