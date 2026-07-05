# Fleet lock bump 4 — post-sprint alignment
DATE: 2026-07-05
PR: (this PR)
STATUS: chore
## What
Aligns all 7 drifted subrepo pins to their current origin/main. Key changes
materialized by this bump:
- renquant-pipeline (+21): S5 decision-ledger wiring (#176), S12 corpus
  refresh (#434/#435), campaign fixes (B8 anti-skew, A1 mis-score, A3 preflight)
- renquant-backtesting (+9): S3 placebo gate (#61, completes D1 path),
  campaign fixes (B1 wf-loader, B5 calendar, B6 ntfy)
- renquant-common (+8): M6 fingerprint 0.9.1/0.9.2 (#21/#22), canonical
  calendar (#24), ntfy sender (#23), Py3.9 compat (#25/#26)
- renquant-base-data (+11): FMP fundamentals/estimate/analyst modules
- renquant-orchestrator (+249): entire 3-day sprint delivery
- renquant-execution (+2), renquant-strategy-104 (+2)
## Safety
All flags OFF. No behavior changes until operator enables specific features.
