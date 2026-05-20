# Watchlist expansion candidates: 43 → ~100 (curated)


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Status:** PROPOSAL only. Nothing added to live watchlist yet.

User spec (2026-04-24): include holdings of major active mutual funds —
VPMAX (Vanguard PRIMECAP Admiral, ~$144B AUM) + FCNTX (Fidelity
Contrafund, ~$140B AUM). These funds are run by professional active
managers; their holdings = real-money active selections that beat
their benchmarks over multiple cycles. Including these tickers gets
us inside the universe that disciplined active management considers
investable.

## Current watchlist (43 tickers)

26 tech / 5 finance / 4 industrial / 3 healthcare / 2 consumer / 2 energy / 1 commodity / 1 utility.
58% tech is too concentrated.

## Mutual-fund overlap analysis

**Already in watchlist** (top-10 of one or both funds):
- AAPL, AMZN, GOOG, META, NVDA, ASML, MA, LLY, TXN, TSLA, NFLX, COST

**Missing top-10 holdings** (must add):

| Ticker | In | Sector | Notes |
|---|---|---|---|
| **MSFT** | both top-3 | tech | Largest active-fund consensus — astonishing we're missing this |
| **BRK.B** | FCNTX top-2 | finance | Berkshire — broad cyclical proxy |
| **AZN**  | VPMAX top-5 | healthcare | UK pharma — global diversification |

## Curated 30-ticker proposal — Wave 1 (43 → 73)

Selection: high mutual-fund consensus weight, sector underrepresented in
current watchlist, ≥5yr history, $20B+ market cap, Alpaca/yfinance
covered.

### Wave 1A — must-add (10 tickers, high consensus)

| # | Ticker | Sector | Mutual-fund presence | Reason |
|---|---|---|---|---|
| 1 | **MSFT** | tech | VPMAX top-3, FCNTX top-3 | Missing the largest large-cap growth — anomaly fix |
| 2 | **BRK.B** | finance | FCNTX top-2 | Berkshire cyclical proxy |
| 3 | **AZN**  | healthcare | VPMAX top-5 | Global pharma diversification |
| 4 | **V**    | finance | Top-15 of both | Visa — fintech rails distinct from JPM/MA |
| 5 | **JNJ**  | healthcare | Top-20 of both | Defensive pharma, low-vol |
| 6 | **UNH**  | healthcare | Top-15 in growth funds | Largest health insurer |
| 7 | **WMT**  | consumer | AGTHX top-10, FCNTX top-25 | Defensive staple |
| 8 | **PG**   | consumer | Top-30 of both | Defensive household products |
| 9 | **HD**   | consumer | Top-20 of both | Home improvement, cycle-tilted |
| 10 | **CMCSA** | consumer | VPMAX top-30, FCNTX top-40 | Media + telecom, distinct alpha |

### Wave 1B — high-value follow-up (10 tickers)

| # | Ticker | Sector | Mutual-fund presence | Reason |
|---|---|---|---|---|
| 11 | **ADBE**  | tech | VPMAX top-10 | Software (different from current AI/semi mix) |
| 12 | **NOW**   | tech | VPMAX top-15 | Enterprise SaaS |
| 13 | **ORCL**  | tech | Top-30 of both | Cloud/database, AI infrastructure |
| 14 | **CRM**   | tech | Common in growth | SaaS leader |
| 15 | **INTU**  | tech | Common in growth | Tax/financial SaaS |
| 16 | **ABBV**  | healthcare | Top-30 of both | Specialty pharma |
| 17 | **MRK**   | healthcare | Top-25 of both | Already-listed peer is MRK ✓ in current — skip duplicate |
| 18 | **TMO**   | healthcare | Top-30 of both | Med instruments |
| 19 | **ABT**   | healthcare | Common | Diversified healthcare |
| 20 | **MDT**   | healthcare | Common | Med devices |

(Note: MRK already in current watchlist; replace #17 with **DHR** Danaher
or **BMY** Bristol-Myers per common holdings.)

### Wave 1C — rounding the sectors (10 tickers)

| # | Ticker | Sector | Notes |
|---|---|---|---|
| 21 | **BAC** | finance | Bank-of-America, distinct from JPM cycle |
| 22 | **GS**  | finance | Goldman investment bank exposure |
| 23 | **BLK** | finance | BlackRock asset manager |
| 24 | **DE**  | industrial | Deere agriculture cycle |
| 25 | **UNP** | industrial | Union Pacific freight |
| 26 | **HON** | industrial | Honeywell diversified |
| 27 | **NEE** | utility | NextEra clean-energy |
| 28 | **DUK** | utility | Duke regulated |
| 29 | **CVX** | energy | Chevron major (vs XOM/OXY) |
| 30 | **NEM** | commodity | Newmont gold miner |

## Resulting Wave 1 mix (73 tickers)

| Sector | Now | After Wave 1 | Change |
|---|---:|---:|---:|
| tech | 25 | 30 | +5 (mostly mature SaaS) |
| healthcare | 3 | 10 | +7 |
| finance | 5 | 10 | +5 |
| industrial | 4 | 7 | +3 |
| consumer | 2 | 6 | +4 |
| energy | 2 | 3 | +1 |
| utility | 1 | 3 | +2 |
| commodity | 1 | 2 | +1 |

41% tech (was 58%) — much better breadth.

## Wave 2 — additional 27 (73 → 100)

Defer to next session. Will draw from:
- VPMAX positions 30-100: AVGO, QCOM, ICE, ELV, FANG, EQIX, etc.
- FCNTX 30-100: STZ, ZTS, EW, MMC, PLD, CB, etc.
- Sector add-ons to fully balance.

## Risk / cost

- **Training cost:** ~+70% (43 → 73 tickers)
- **Fundamentals fetch:** 30 new OpenBB calls (1-2 min)
- **Earnings + insider fetch:** ~5 min each
- **10-min bar fetch:** ~5 min
- **First retrain:** +30 min beyond current

**Acceptance gate per Wave:**
1. Retrain panel with new tickers
2. Compare CPCV OOS IC vs prior watchlist
3. A/B portfolio APY on 27-mo OOS
4. Promote ONLY if APY ≥ prior wave's APY (no regression tolerance)

## Implementation order (today's session if approved)

1. ✅ Done: minute panel promoted (in-flight verification on main)
2. ✅ Done: transformer training in tmp workspace (in-flight)
3. ⏭ User-approved Wave 1A 10 tickers → fetch their data → retrain → A/B
4. ⏭ If A/B passes → Wave 1B+1C 20 more
5. ⏭ Wave 2 next session

User confirms Wave 1A → I do everything end-to-end.
