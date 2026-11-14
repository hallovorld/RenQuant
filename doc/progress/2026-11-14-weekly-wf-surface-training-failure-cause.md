# 2026-11-14 — weekly_wf_promote: surface WHY training failed (log + page)

STATUS:   delivered, awaiting review — zero reviews at head.
WHAT:     On a failed Step 3 retrain, `scripts/weekly_wf_promote.sh` now prints the retrain log's last exception (fallback: last 3 non-empty lines) plus the last 15 lines into the weekly log, and the `WEEKLY-FAIL` page carries `Cause: … | log: <path>`. 1 end-to-end test.
WHY/DIR:  `daily_retrain_alpha158_fund.sh` does `exec >> "$LOG" 2>&1`, so the weekly log said only "Training FAILED — production artifact unchanged." On 2026-11-14 the real cause (`RuntimeError: 1/292 panel tickers stale … MSFT(-2s, last 2026-11-11)`, after 22 incremental-fetch timeouts at 04:06 PST) was only in logs/daily_retrain_alpha158_fund/2026-11-14.log, a second file nobody was pointed to.
EVIDENCE: see §4(b) below.
  artifact:      scripts/weekly_wf_promote.sh, tests/test_weekly_wf_promote_rfc210_fallback.py (22 passed).
  prod or exp:   exp — effective at the next weekly run after merge and umbrella sync (merged ≠ deployed).
  existing data: logs/weekly_wf_promote/2026-11-14.log [VERIFIED 2026-11-14 16:xx PST]: ends at "Training FAILED" with no cause; logs/daily_retrain_alpha158_fund/2026-11-14.log: 22 "possibly delisted; no price data found" / "timed out after 30s — returning stale cache" lines, then the RuntimeError.
  best-known?:   yes — anti-vacuity: with the script change stashed the new test fails; with it 22/22 pass.
  scope:         one shell block + one test; no artifact/config/data touched. Does NOT fix the stale-bar cause itself (separate: zero-tolerance freshness vs. an unretried yfinance fetch).
NEXT:     merge + umbrella sync (ask-first); then retry/robustness for the incremental OHLCV fetch.
