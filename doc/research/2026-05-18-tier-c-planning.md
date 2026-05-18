# Tier C Planning (2026-05-18 NIGHT session output)

After today's Tier A+B sweep + 2 negative findings (C2 insider, B4 multi-
horizon), the remaining Tier C roadmap items are multi-week scope and
require either external data/code subscriptions or significant engineering.
This doc scopes each so the next session can choose where to invest.

---

## C1 — Options-implied features (roadmap #2, ~+0.30 Sharpe, 1 week)

**Hypothesis**: implied volatility (IV) and put/call skew predict 60-day
forward returns beyond what historical price-based features capture.

**Reference work** (must read before implementation):
- Goyal-Saretto 2009 *JFE* "Cross-section of option-implied volatility
  and stock returns" — Δ between implied and realized vol forecasts
  next-month returns; the bigger the gap, the higher the lottery
  premium, the lower the realized return.
- Bali-Hovakimian 2009 *MSci* "Volatility spreads and expected stock
  returns" — call-IV minus put-IV captures price-pressure imbalance.
- An-Ang-Bali-Cakici 2014 *RFS* "Joint cross-section of stocks and
  options" — IV implies the option's information edge; high-IV stocks
  underperform.
- Cremers-Weinbaum 2010 *JFQA* "Deviations from put-call parity" —
  parity violations predict the underlying.

**Data source decision** (BLOCKER — user must choose):
| Option | Coverage | Cost | Setup time | Quality |
|---|---|---|---|---|
| Alpaca Options API (Pro) | US equity options EOD | $10-50/mo | 1 day | Adequate |
| OptionMetrics IvyDB | Historical EOD, since 1996 | $5000+/yr academic | 2-3 days | Gold standard (cited in 90% of academic papers) |
| ORATS Wheel | Per-strike IV + greeks | $99/mo | 1 day | Good for retail |
| Yahoo Finance (yfinance.Ticker.options) | Real-time chains, limited history | Free | 1h | Poor (no history; current only) |

Minimum-viable path: Alpaca Options API ($10/mo paid tier) →
features `iv_30d_call_atm`, `iv_30d_put_atm`, `iv_skew = put_iv -
call_iv`, `iv_term_struct = iv_30d - iv_90d`.

**Engineering breakdown** (after data source):
- Day 1: ingest pipeline (`scripts/fetch_options_iv.py` → `data/
  options_iv_daily.parquet`)
- Day 2: feature engineering (compute the 4 IV features per ticker per
  date, handle missing strikes, ATM interpolation)
- Day 3: integration into `build_alpha158_fund_panel.py` (new
  `iv_*.parquet` left-join, same BUG #2-style date-coverage guard)
- Day 4-5: training + WF eval + §5.2 sanity (placebo with shifted IV)
- Day 6-7: A/B integration with prod via side config + dense panel sim

**Success criteria**: Δ val_IC ≥ +0.01 (single seed) AND placebo
persistence < 70%. If pass, escalate to 5-seed A/A + 16-window WF sim.

**Risk if invested but fails**: ~$50 paid for IV data + 1 week
engineering. Negative outcome would be valuable per the failed-
experiments-log pattern (rules out a recurring hypothesis).

**Status**: BLOCKED on data source decision.

---

## C3 — Watchlist quality-first expansion to wl200 (roadmap #4, ~+0.20 Sharpe, 1 week)

**Hypothesis** (Grinold-Kahn 1999 Fundamental Law of Active Management):
IR ≈ IC × √Breadth. Expanding from 103 → 200 names should lift IR by
√(200/103) ≈ 1.39× IF transfer coefficient holds.

**Reference**:
- Grinold-Kahn 1999 *Active Portfolio Management* §6 — Fundamental Law.
- Kelly-Gu-Xiu 2020 *RFS* §3 — ML alpha scales with universe size up
  to ~500 names where market-impact starts dominating.
- Previous failed attempt: E26 wl183 — IC dropped 44% with bottom-up
  expansion (low-quality names dilute signal). The expansion strategy
  has to be QUALITY-FIRST.

**Quality criteria** (from rejected E26 post-mortem in failed log):
1. Avg daily $ volume ≥ $50M (1.5× median NYSE+NASDAQ)
2. Market cap ≥ $5B (mid-large only)
3. Earnings reporting consistency (≥ 8 of last 8 quarters reported on
   schedule)
4. No SEC enforcement actions / delisting risk in last 5 years
5. SIC sector classification stable (no recent reverse mergers)

**Engineering breakdown**:
- Day 1: source quality data — SimFin / Quandl Sharadar / Alpha
  Vantage. SimFin has free tier with 1500 US equities + fundamentals.
- Day 2: apply 5 filters → arrive at top ~200 candidates
- Day 3: backtest universe with current 169-feature model on 6
  walkforward cuts; verify IC doesn't collapse
- Day 4-5: refit calibrator + acceptance gate on new universe;
  measure pool_IC, sigma_calib
- Day 6-7: dense panel sim on 8-window post-promote A/B

**Success criteria**: post-expansion pool_IC ≥ 95% of current 103-name
pool_IC + 16-window WF Sharpe ≥ baseline. If pool_IC degrades > 5%,
the expansion is filling with low-quality names → reject + iterate
quality filter.

**Risk**: even with quality filters, the cross-sectional alpha may
not transfer. E26 (wl183 bottom-up) lost 44% IC; that's the risk
boundary even with good filters.

**Status**: ready to start (no external blockers); needs full week.

---

## C5 — News sentiment via FinBERT (roadmap #3, ~+0.20-0.30 Sharpe, 2-3 weeks)

**Hypothesis** (Tetlock 2007 + Ke-Kelly-Xiu 2019): news sentiment
encoded by a domain-tuned transformer (FinBERT, Liu-Huang-Zhou 2021
EMNLP) captures information not yet priced over 60-day horizon.

**Reference**:
- Tetlock 2007 *JF* "Giving Content to Investor Sentiment" — Wall
  Street Journal column sentiment predicts S&P returns.
- Ke-Kelly-Xiu 2019 *NBER w26261* "Predicting Returns with Text Data"
  — supervised topic modeling on financial news; trades top-decile
  vs bottom-decile yields 0.4 daily Sharpe.
- Liu-Huang-Zhou 2021 EMNLP "FinBERT: A Pre-trained Financial
  Language Model" — BERT-base fine-tuned on financial text corpus,
  beats vanilla BERT on Financial Phrase Bank by 6pp accuracy.
- Open source: `ProsusAI/finbert` HuggingFace model card (3-class
  sentiment); `yiyanghkust/finbert-tone` (alternative).

**Data source decision** (BLOCKER — user must choose):
| Option | Coverage | Cost | History |
|---|---|---|---|
| Benzinga News API | Full headlines + bodies, real-time | $99-499/mo | 5+ years |
| Polygon.io News API | Tagged tickers, real-time | $200/mo Pro | 5 years |
| Alpaca News (Pro) | Brief headlines per ticker | $10/mo (with Options Pro) | Limited |
| NewsAPI.org | Public web news | $449/mo Business | Brief |
| Common Crawl scrape | Yahoo Finance, MarketWatch | Free | Manual fetch + clean |

Cheapest viable: Alpaca News tier ($10/mo bundled with Options) for
headlines only.

**Engineering breakdown** (3 weeks):
- Week 1: ingest pipeline (`scripts/fetch_news_headlines.py` →
  `data/news_headlines.parquet`). Per-ticker, per-date, headline +
  body if available.
- Week 2: FinBERT inference (`scripts/score_news_sentiment.py`):
  - HuggingFace `transformers` pipeline (CPU OK on M2 Pro for our
    ~100 tickers × ~5 headlines/day = ~500 inferences/day)
  - Output: per-ticker per-date sentiment score in [-1, +1]
  - Aggregation: mean sentiment over 5-day rolling, plus 5-day vol
    of sentiment (Tetlock's "agreement" feature)
- Week 3: integration + A/B as in C1/C3. WF gate + sanity.

**Success criteria**: same as C1 (Δ val_IC ≥ +0.01, placebo < 70%).
News sentiment is uncorrelated with price-based features in
Ke-Kelly-Xiu's reported results, so even small lift may compound
nicely with existing 169-feat panel.

**Status**: BLOCKED on news data source + 3 weeks engineering. Of
the 3 C-tier items, this has highest expected lift (+0.20-0.30 Sharpe
per roadmap) but also highest engineering cost.

---

## Recommended next-session sequencing

If user has budget for ONE C-item this week, by expected ROI:

| Priority | Item | Cost | Expected lift | Why |
|---|---|---|---|---|
| 1 | **C1 Options-IV** | $10-50/mo + 1 week | +0.30 Sharpe | Highest signal density per engineering hour; the Goyal-Saretto / Cremers-Weinbaum signals are repeatedly confirmed in literature |
| 2 | C3 wl200 expansion | Free + 1 week | +0.20 Sharpe | No data subscription; Grinold-Kahn says √breadth is automatic IF transfer coef holds (E26 risk) |
| 3 | C5 FinBERT news | $10-100/mo + 3 weeks | +0.20 Sharpe | Most ambitious; 3 weeks of work for similar lift to C3 |

If user has NO additional budget this week, the highest-ROI use of
session time is:
- Implement C3 (only engineering, no $$$) **OR**
- Investigate what's NOT in this roadmap — the 169-feat panel ceiling
  may be at IC=+0.04 due to fundamental information limits at this
  universe size; structural change (M2-tier 5-min features per
  CLAUDE.md, multi-strategy ensemble) might be needed.
