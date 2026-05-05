# Capability Expansion — Long/Short + Options for renquant_104

**Status**: 🟡 P0 design (2026-05-03, user-spec: "发挥模型全部的能量"). Implementation in stages.
**Scope**: §1–§9 cover long/short. §11 covers options. §10 is the decision log.
**Author**: Initial draft 2026-05-03.
**Related**: [`doc/components/buy-logic.md`](../components/buy-logic.md), [`doc/components/sell-logic.md`](../components/sell-logic.md), [`doc/arch/strategy-104.md`](../arch/strategy-104.md).

---

## 1. Why now

The cross-sectional panel-LTR rank model produces a full distribution of scores per bar — top decile is what we currently buy, but **bottom decile is equally informative** and currently discarded. Allocating to BOTH ends:
- Doubles the alpha lever per bar (long top + short bottom)
- Hedges market beta (gross 130/30 ≈ market-neutral if balanced)
- Decorrelates from existing long-only book

The literature consensus on cross-sectional rank strategies (Asness et al., AQR; Fama-French 3F+; Jegadeesh-Titman 1993): **about half the alpha is on the short side**. By being long-only we voluntarily forfeit ~50% of the available IC.

## 2. Scope of this doc

In: features needed; data sources; pipeline changes; risk management; validation plan; staging.
Out: paper trading rollout schedule; live capital allocation policy. (Both governed separately by `feedback_sharpe_floor.md` + `feedback_after_tax_principle.md`.)

## 3. Model layer

### 3.1 Signal source (no change)

The XGBoost rank:pairwise model already outputs a continuous panel score per ticker per bar; existing top-K selection is just `argsort()[-K:]`. Bottom-K shorts use `argsort()[:K]` of the **same** scores.

### 3.2 Acceptance gates (additive)

- **G7 (OOS IC floor)**: existing rule applies independently to long top-K vs short bottom-K. **Reject** a model where short-side OOS IC is < 0 (the bottom decile must, on average, underperform).
- **NEW G12 (long-short IC parity)**: |long_top_IC − short_bot_IC| ≤ 0.5 × max(long, short). Asymmetric IC ≥ 2× indicates the model is essentially long-only and dressing it up as long-short adds risk without alpha.
- **NEW G13 (short crowdedness)**: short side P&L attribution shouldn't be > 50% concentrated in tickers with short interest > 20% of float. (Crowded shorts → squeeze risk dominates fundamental decay.)

### 3.3 Calibration

Existing global calibrator (NGBoost μ, σ + isotonic) is **direction-agnostic** — operates on rank score, not on P(up). For shorts: `expected_return_short = -μ` (sign flip), `σ` unchanged. No new calibration artifact needed.

## 4. Data layer

### 4.1 Required new data sources

| Data | Use | Source | Cost | Frequency | Status |
|---|---|---|---|---|---|
| **`is_shortable` flag** | Hard filter — drop non-shortable from short candidates | Alpaca asset API (free) | $0 | realtime | 🟢 immediate (cred already wired) |
| **`easy_to_borrow` flag** | Tier-1 vs Tier-2 short candidates | Alpaca asset API (free) | $0 | realtime | 🟢 immediate |
| **Short interest %** | Crowdedness, squeeze risk | FINRA Daily Short Sale (free) or NYSE/Nasdaq Short Interest (free, biweekly) | $0 | biweekly | 🟡 fetcher to write |
| **Days to cover** | Liquidity risk | Computed from short interest + ADV | $0 | biweekly | 🟡 derive locally |
| **Borrow rate** | Carry cost | Alpaca / IEX / Interactive Brokers (varies; some free, some paid) | $0–$200/mo | daily | 🟡 evaluate |
| **Real-time short interest deltas** | Premium signal | S3 Partners, Ortex | $500–$2k/mo | daily | 🔴 evaluate ROI later |
| **Recall events** | Forced cover risk | Broker-specific (no public feed) | n/a | event | 🔴 broker-only |

### 4.2 Free data action items (start now)

- [x] Wire Alpaca creds (already done — live keys in `.env` 2026-05-03).
- [ ] **Write `scripts/fetch_alpaca_shortable.py`** — pull `is_shortable` + `easy_to_borrow` for wl=183, cache to `data/shortable/{date}.parquet`.
- [ ] **Write `scripts/fetch_finra_short_interest.py`** — pull biweekly short interest from FINRA's downloadable CSV (https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data), cache to `data/short_interest/{ticker}.parquet`.
- [ ] Compute days_to_cover = short_interest_shares / 60d_ADV_shares; persist as feature.

### 4.3 Paid data option (defer until R&D justifies)

- **S3 Partners SQ score** — daily short interest with squeeze/crowdedness composite. ~$500/mo per seat. Defer until long-only-style short prototype shows alpha; THEN evaluate whether crowdedness signal is the bottleneck.

## 5. Pipeline layer

### 5.1 New tasks (sequential)

1. **`LoadShortableFlagsTask`** — read latest `data/shortable/*.parquet`, populate `ctx.shortable[ticker] = {"is_shortable": bool, "easy_to_borrow": bool}`. Runs in PanelDataJob phase, after `LoadFundamentalsTask`.

2. **`LoadShortInterestTask`** — read `data/short_interest/*.parquet`, populate `ctx.short_interest[ticker] = {"si_pct": float, "days_to_cover": float, "as_of": date}`.

3. **`ShortCandidateJob`** (parallel, mirrors `TickerCandidateJob`) — task chain:
   - `EarningsFilterTask` (same as long; same buffer)
   - `WashSaleFilterTask` (longs only — short side is symmetric, but irrelevant; skip)
   - `BuildFeaturesTask` (same)
   - `ScoreShortTask` (NEW — model score with sign flipped)
   - `ScoreThresholdTask` (NEW — score < threshold to qualify, mirrors long's > threshold)
   - `IsShortableTask` (NEW — hard filter on `is_shortable=True`)
   - `EasyToBorrowTask` (NEW — preference filter; tier-2 names allowed but tagged)
   - `ShortInterestVetoTask` (NEW — drop if SI% > config.short.crowdedness_veto, default 25%)
   - `BorrowCostTask` (NEW — compute expected return MINUS borrow rate; reject if net ≤ 0)
   - `AssembleShortCandidateTask`

4. **`JointPortfolioQPTask`** (modify existing) — extend to allow negative weights `w_i ∈ [-w_short_max, +w_long_max]`, add gross/net leverage constraints:
   - `sum(|w_i|) ≤ gross_leverage_cap` (default 1.30 for 130/30)
   - `sum(w_i) ∈ [net_min, net_max]` (default [-0.10, +1.10] for ~market-neutral with bias)
   - Per-ticker cap `|w_i| ≤ 0.10` (kept at 10% per name regardless of side)

### 5.2 New sell-side logic

Existing `TickerSellJob` is long-only oriented. Need a parallel `TickerCoverJob` for closing shorts:
- Same model_action signal (just flipped — model says "buy" → cover the short)
- Stop-loss: HARD stop at +X% adverse move (default +20%, tighter than long's −20% because shorts can blow up unbounded)
- Trailing stop on UP move from entry
- Cover into earnings (skip ±N days, same as long blackout)
- Forced cover on broker recall event (rare but real)

### 5.3 Risk gates (extend existing)

- **`PositionConcentrationGateTask`** (already added 2026-05-03): cap `|w_i|` regardless of side.
- **NEW `GrossLeverageGateTask`**: refuse to add if Σ |w_i| would exceed `gross_leverage_cap`.
- **NEW `ShortFloorGateTask`**: refuse to short if expected_return - borrow_rate < `short_min_edge` (default 5%/yr).
- **NEW `BorrowCostStopTask`**: in sell-only path, force-cover any short whose cumulative borrow cost over holding period > entry edge × 0.5 (P&L erosion guard).

## 6. State + persistence

### 6.1 HoldingState extension

Add fields to `kernel/exits.py::HoldingState`:
- `side: Literal["long", "short"] = "long"`
- `borrow_rate_at_entry: float = 0.0`
- `cumulative_borrow_cost: float = 0.0`
- `entry_short_interest: float | None = None` (squeeze risk audit trail)

### 6.2 Tax accounting

Per `feedback_after_tax_principle.md`: ALL P&L reported after-tax.
- Short gains are **always short-term capital gains** (no LTCG eligibility per IRS §1233).
- Wash-sale rule asymmetric — applies to losses on shorts too.
- Need new `kernel/tax.py::compute_short_p_and_l` that nets borrow cost + dividends-paid (short borrower owes dividends on borrowed shares) + ordinary-income-rate tax.

### 6.3 P&L reporting

`live_state.alpaca.json` and `daily_state.csv` need new columns:
- `gross_long_value`, `gross_short_value`, `net_exposure`, `gross_leverage`
- `total_borrow_cost_ytd`, `dividends_paid_ytd_on_shorts`

## 7. Validation plan

### 7.1 Statistical validation (must pass before any live capital)

1. **Long-side IC unchanged**: paired CPCV on production wl=183 with shorts disabled vs enabled — long IC drift |Δ| ≤ 1bp.
2. **Short-side IC > 0**: bottom-decile short OOS IC ≥ +0.02 (matching the hard floor for long).
3. **§5.2 sanity triple on shorts**: A/A test, shuffled-label, time-shift placebo — all 3 must be ≈ 0 (just like long).
4. **Combined long-short Sharpe ≥ 1.0**: per `feedback_sharpe_floor.md`, portfolio-level after-tax Sharpe.
5. **Borrow-cost-aware backtest**: re-sim wl=183 with synthetic ~3% annualized borrow on hard-to-borrow names + 0.5% on easy-to-borrow. Check long-short Sharpe still > long-only Sharpe net of costs.

### 7.2 Risk validation

1. **Squeeze stress test**: synthetic +50% gap-up shock on top-10% short positions. Portfolio drawdown < 15%.
2. **Liquidity stress**: assume cover-day volume = 50% of normal ADV. Cover slippage ≤ 200 bps.
3. **Borrow recall**: simulate 5% random recall rate — auto-cover impact on Sharpe ≤ 0.05.
4. **Margin call simulation**: max gross leverage breach scenario; system auto-reduces.

### 7.3 Pipeline validation

- All 16 freshness gate + 16 risk gate tests still green with shorts enabled.
- New tests: 20+ tests for short-side gates (parallel to existing).
- LEAN backtest config has shorts enabled; backtest replicates live-runner orders ±2%.

## 8. Staging — implementation order

| Stage | Effort | Output | Gate |
|---|---|---|---|
| **S0 — design + free data** (this doc + Alpaca/FINRA fetchers) | 1 day | this doc + 2 fetchers + populated cache | doc reviewed |
| S1 — extend acceptance gates (G7 short, G12, G13) | 2 days | offline analysis with these gates on wl=183 panel | All 3 pass on bottom-decile |
| S2 — implement ShortCandidateJob (no live) | 4 days | sim shows realistic candidate flow | sim green |
| S3 — extend QP for negative weights + leverage caps | 3 days | QP solver passes degenerate + diverse cases | QP tests green |
| S4 — extend HoldingState + tax + P&L reporting | 2 days | unit tests + sim | tests green |
| S5 — backtest validation (B2 hold-out + 27mo OOS) | 1 week | metrics report; ship/no-ship decision | all §7 metrics pass |
| S6 — paper trading 30 days | 30 days | live paper logs + drift report | drift < 50bp |
| S7 — limited live ($1k–$2k notional, 5% gross-short cap) | 2 weeks | live logs | no margin events |
| S8 — full live | n/a | scheduled rollout | S7 pass |

**Total time-to-live: ~6–8 weeks** (excluding paid-data evaluation).

## 9. Open questions

- **Pairs vs unpaired shorts**: Should short basket be matched to long basket sectorwise to neutralize sector beta? Or just take signal as-is and let market beta cancel statistically?
- **Net long bias**: Always 100% long + X% short, or fully market-neutral 1:1? AQR / Two Sigma typically run net 30–50% long with shorts as alpha-add. Decision deferred to S5 backtest.
- **Single-name short cap vs sector cap**: Per memory `project_multi_stock_sizing.md`, max 1/3 per stock applies to longs. Should shorts have a stricter per-name cap (5%? 10%?) given asymmetric loss profile?
- **Earnings — long blackout exists, short blackout?**: Same buffer? Wider for shorts (post-earnings drift can squeeze)?
- **Borrow cost in objective**: Add to QP objective directly, or just as filter? Direct integration enables Pareto-optimal short selection but adds solver complexity.

## 10. Decision log

- 2026-05-03: User escalated to P0 ("model做空功能提上p0"). Design doc started.
- 2026-05-03: Free data action items (Alpaca shortable + FINRA short interest fetchers) prioritized for immediate execution. S3 Partners deferred until S5 results justify spend.
- 2026-05-03: User asked about options ("发挥模型全部的能量，能做期权吗"). Options scope added in §11.
- 2026-05-03: User deferred options ("期权先不考虑吧"). §11 marked DEFERRED; keep as reference, do not start.
- TBD: net-bias policy (S5), per-name short cap (S5).

---

## 11. Options layer (extension) — 🔴 DEFERRED 2026-05-03

> **User decision 2026-05-03**: "期权先不考虑吧". Options work paused; section preserved as design reference for whenever long/short ships and stabilizes. Do not start any options work until user explicitly reopens.

The cross-sectional rank model is fundamentally **directional** — it picks winners and losers. Options are the natural way to express direction with leverage and convexity. **All three usage tiers below sit on TOP of the existing long/short pipeline; none replace the rank-model core.**

### 11.1 Three usage tiers

| Tier | Strategy | Signal mapping | Complexity | When to use |
|---|---|---|---|---|
| **T1 — Covered calls on existing longs** | Sell OTM call on each held long | Hold = sell call | Low | Income harvest on already-held long positions; reduces drawdown by call premium |
| **T2 — Replace directional rank picks with options** | Top-K → buy 60-90d OTM call; Bottom-K → buy OTM put | Long top → call; Short bottom → put | Medium | Higher Sharpe per $ deployed if signal is real; convexity risk if signal is noisy |
| **T3 — Vol trading (straddle / condor)** | Long IV when realized > implied; short IV when implied > realized | NEW IV forecast model required | High | Decoupled from directional model — separate research project |

**Recommendation**: ship T1 first (lowest risk, real income), then T2 in tandem with long/short. T3 is a different research track entirely.

### 11.2 Required new data

| Data | Use | Source | Cost | Frequency | Status |
|---|---|---|---|---|---|
| **Option chain** (strikes, expirations, bid/ask) | Find tradeable contracts | Alpaca paper API (free) / yfinance | $0 | realtime | 🟡 fetcher to write |
| **IV per contract** | Pricing decisions, vol screen | Same chain feed | $0 | realtime | 🟡 derive or pull |
| **Greeks** (Δ, Γ, Θ, ν) | Position sizing + book-level risk | `py_vollib` / Alpaca-provided | $0 | derive | 🟡 lib + helper |
| **Historical option EOD** | Backtesting | ORATS / IVolatility | $150–$2k/mo | daily | 🔴 P2 (defer until T2 ships paper) |
| **Realized vs implied vol** | T3 only — vol-trade signal | Computed from chain + OHLCV | $0 | daily | 🔴 P3 (T3 only) |

### 11.3 Pipeline changes

#### T1 (covered calls) — minimum changes
- New `task_covered_call.py::EmitCoveredCallTask` in inference pipeline AFTER `TopUpHeldTask` / `TrimHeldTask`
- Reads `ctx.holdings`; for each long position, evaluates: 30-day OTM call at delta ~0.20–0.30
- Emits OPTION_SELL_TO_OPEN order via Alpaca options endpoint
- Tracks short-call lifecycle (assignment, buyback, expiration)
- New state field `holding.covered_call_contract` per position

#### T2 (call/put replacement) — major changes
- New `OptionCandidateJob` parallel to `TickerCandidateJob`
- New `task_option_select.py::SelectOptionContractTask` — picks specific (strike, expiry) given underlying signal + IV
- Modified `JointPortfolioQPTask` — option positions enter as nonlinear payoffs; need either (a) Δ-equivalent linearization or (b) MILP relaxation for integer contract sizing
- Modified `HoldingState` — `instrument: Literal["equity","call","put"]`, `strike: float | None`, `expiration: date | None`, `delta_at_entry: float`
- Roll logic: 21-day-to-expiry threshold → close + open same-Δ further-dated; `task_option_roll.py::RollOptionsTask`

#### T3 (vol trading) — separate research project
- Train IV-forecast model (LSTM or HAR-RV variant) — out of scope for this doc
- Straddle / condor selection logic
- Vega-aware position sizing
- ✋ **DO NOT START until T2 has ≥3 months of paper-trading evidence**

### 11.4 Risk management additions

- **Theta decay budget**: aggregate book theta as a daily P&L drag; cap at 50bp/day on net asset value
- **Vega cap**: book vega ≤ 30bp per 1% IV move
- **Gamma cap**: book gamma per $1 underlying move ≤ 100bp on portfolio
- **Per-contract max loss**: 100% of premium paid (long options); HARD stop at 50% premium loss for short options
- **Rolling stop**: if option underlying moves opposite by 1×ATR, close (don't ride to zero)
- **Earnings blackout extends 7 days post-event** for options (premium crush is bigger than for equity)
- **Liquidity floor**: only trade contracts with bid-ask spread ≤ 5% of mid AND open interest ≥ 100

### 11.5 Tax (IRS — different from equity)

- Equity options: short-term gain if held < 1 yr (most directional plays); long-term if held > 1 yr (rare)
- Section 1256 contracts (futures, broad-based index options like SPX): **always 60% LTCG / 40% STCG** regardless of holding period — significant tax advantage
- Short calls / short puts (writing): premium = ordinary income on expiration; assignment changes basis of underlying
- Wash-sale: applies to options too — losing a put position then re-buying same put within 30 days disallowed
- Dividends: long calls have no dividend rights; short calls owe equivalent if short over ex-date (same as equity short)
- New `kernel/tax_options.py` module to handle these. `feedback_after_tax_principle.md` rule applies — always after-tax.

### 11.6 Staging

| Stage | Effort | Output | Gate |
|---|---|---|---|
| **OS0 — feasibility audit** (Alpaca options API, instruments coverage) | 1 day | doc + sample chain pulls | wl=183 has ≥80% liquid contract coverage |
| **OS1 — T1 covered calls in sim** | 1 week | sim P&L includes premium | sim Sharpe lift > 0.1 |
| **OS2 — T1 paper trading 30 days** | 30 days | live paper logs | drift < 50bp |
| **OS3 — T1 limited live ($500 premium notional)** | 2 weeks | live logs | no assignment surprises |
| **OS4 — T2 sim** | 2 weeks | sim Sharpe vs equity-only baseline | net-of-cost lift ≥ +0.2 Sharpe |
| **OS5 — T2 paper 60 days** | 60 days | live paper | greater Sharpe than long-only paper, no greeks blow-ups |
| **OS6 — T2 limited live (5% NAV)** | 30 days | live logs | … |
| **OS7 — T2 full live** | n/a | scheduled | OS6 pass |

**T1 path: ~3 months to live**. **T2 path: ~6 months to live** (predicated on T1 success).

### 11.7 Why this is realistic, not over-reach

The cross-sectional model already produces **calibrated μ + σ per ticker per bar** (NGBoost head). That's exactly the input an option-pricing replacement layer needs:
- Black-Scholes input is (S, K, r, T, σ). We have S (spot), K (we choose strike), r (risk-free, free), T (we choose expiry), σ (NGBoost output).
- The model "knows" what it expects each ticker to do; options just translate that into a payoff with leverage.
- The hard part is **liquidity, fees, and risk management**, NOT the alpha — alpha is already there in the rank score.

**The bottleneck is execution discipline**: theta decay punishes sloppy entries; an alpha edge that's +10 bp/day evaporates fast if you pay 5 bps wide on every option fill. Hence T1 first (passive premium harvest) before T2 (active premium spending).

### 11.8 Open questions

- **Which instruments**: ETF options (SPY, QQQ) for index-level macro overlays, or single-name only?
- **Cash-secured puts for entry**: instead of buying a stock long, sell cash-secured ATM put → if assigned, get long at lower basis; if not, keep premium. Worth a sub-tier?
- **Hedging strategy**: long-only book hedged with index puts during BULL_VOL regime?
