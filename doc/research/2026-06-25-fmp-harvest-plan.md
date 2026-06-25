# FMP one-month harvest plan — store everything while the paid window is active

2026-06-25. Operator directive: *"把 FMP 所有可以拿的数据都 download，先不考虑用不用得上，本地先存着，最大程度利用这个月的额度。"* Starter plan upgraded today
unlocks full-symbol + 5-year history. **This is a ONE-TIME harvest** (deep
history doesn't change) — pull it all this month, store locally, then the daily
deltas can come from the free Finnhub cron (#408) and we cancel the paid plan.

## Budget (Starter)
- **300 API calls / min**, **20 GB / 30-day** trailing bandwidth, **5-year** history depth.
- Universe = the **291 alpha158 training tickers** (superset of the 145 watchlist), so
  every harvested series can become a model feature aligned to the training panel.
- ~291 tickers × ~19 endpoints ≈ 5–6k calls ≈ ~18 min at 300/min; measured data ≈ **13 MB**
  (far under 20 GB — bandwidth is not the binding constraint). Throttle 0.2 s (≈300/min),
  bounded retry/backoff on 429/5xx/timeout, manifest-resumable, fail-closed.

## Storage
`data/fmp_harvest/<endpoint>_291.parquet` + a sidecar `<endpoint>_291.manifest.json`,
one tidy frame per endpoint, every row stamped `ticker` + `fetched_at` + `source`. The
**manifest** is the audit record: requested/with_data/no_data/http_error/fetch_error counts,
error samples, endpoint URL template, started/finished, row+ticker counts, and the output
sha256. (Raw, un-joined — feature engineering is a separate, later step.) Keys never
committed; the whole `/data/` tree is already gitignored (`.gitignore:41`), so parquet +
manifest stay local automatically — no new ignore rule needed.

## Endpoints to harvest (priority order)
**A. Analyst (HIGH — feeds the immediate retrain; already pulling):**
- `grades-historical` ✅ (rating distribution, 8y) · `grades-consensus` · `analyst-estimates`
  (EPS/revenue forecasts + the revision signal) · `price-target-consensus` · `price-target-summary`.

**B. Fundamentals (full statements, all periods — 5y):**
- `income-statement` · `balance-sheet-statement` · `cash-flow-statement` · `ratios` ·
  `key-metrics` · `financial-growth` · `enterprise-values` · `historical-market-capitalization`.

**C. Earnings & events:**
- `earnings` (historical surprises) · `earnings-calendar` · `dividends` · `splits`.

**D. Ownership & flow:**
- ~~`institutional-ownership`~~ (verified 2026-06-25 **plan-locked above Starter** — 402
  "Restricted Endpoint"; omitted) · `insider-trading` · `shares-float`.

**E. Sentiment / news (if quota allows):**
- `stock-news` (per ticker, recent) · `historical-social-sentiment`.

**F. Macro (universe-agnostic, a handful of calls):**
- `treasury-rates` · `economic-indicators` (GDP/CPI/unemployment/etc.).

## What rides on this (and what does NOT)
- The **analyst** harvest (A) feeds the **immediate go/no-go ablation + retrain**
  (separate PRs). Everything else (B–F) is stored "just in case" per the directive —
  **no feature/retrain decision rides on B–F yet**; they're raw inventory for future use.
- Discipline unchanged: any of this becoming a model feature still goes feature-eng PR →
  placebo-clean WF validation → promote ([[deployed-but-dark-is-not-done]]). Storing ≠ using.

## What the harvester ships now vs defers
`scripts/fmp_harvest.py` implements **A (analyst), B (fundamentals), C (earnings/events),
D (ownership), and F-treasury** — ~19 endpoints, all per-ticker except treasury. **Deferred
pending this discussion:** E (per-ticker `stock-news` / `historical-social-sentiment`) and
`economic-indicators` — per-ticker news is the one bandwidth-heavy series (could approach
the 20 GB cap) and its modelling value is least clear, so it's opt-in via a follow-up, not
in the default sweep. Flagging explicitly rather than silently dropping it.

## Execution & state
`scripts/fmp_harvest.py --out data/fmp_harvest --rate 0.2` (manifest-resumable: skips an
endpoint only when its parquet AND an `status: ok` manifest both exist; `--only <substr>`
targets one group; fail-closed by default — any http/fetch error exits non-zero unless
`--allow-errors`). Run once this month; then cancel the paid plan and let Finnhub (#408)
carry the free daily deltas.

**Execution state (honest):** under the paid-window time pressure, a first-pass pull of
A–D + treasury already ran (local-only, gitignored, ~13 MB). That output is **NOT
experiment-ready** — it is raw inventory behind this review gate. The canonical, auditable
harvest is the one produced by the hardened script in *this* PR (with per-endpoint
manifests); nothing here feeds a model until it passes a feature-eng PR → placebo-clean WF
([[deployed-but-dark-is-not-done]]). The analyst go/no-go ablation is the only thing already
acted on, and its verdict was **regime-split → no global retrain** (BULL_CALM adds,
BULL_VOLATILE hurts).
