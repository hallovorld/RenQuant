# kernel/reconciliation — live <-> sim observability

Continuous reconciliation between Alpaca live fills and the SimAdapter
replay. Born out of the 2026-05-09 17-bug audit, where multiple sim/live
divergences (wash-sale logic, σ-aware stops dead in prod, μ̂ collapse
under missing tickers) sat undetected for days because no daily detector
existed.

## What this answers, every day

1. **Did live execution diverge from sim's recorded decision?**
   For each Alpaca fill, look up sim's decision for the same (date,
   ticker). Disagreement on direction (buy/sell/hold) or qty (>5%) gets
   logged as a divergence case.
2. **What was the slippage distribution?**
   p50 / p95 / max in bps, signed so positive = live worse than sim.
3. **Is the model still predictive?**
   Rolling 30d Spearman IC of `candidate_scores.rank_score` vs realized
   `ticker_forward_returns.fwd_5d`.

Output: a single markdown report at `reports/recon_<end_date>.md`.

## How to run

```bash
source .venv/bin/activate
python scripts/reconcile_live_sim.py \
    --broker alpaca \
    --start-date 2026-05-09 --end-date 2026-05-09 \
    --output reports/recon_2026-05-09.md
```

Defaults: `--broker alpaca`, both dates = yesterday,
`--output reports/recon_<end_date>.md`. Override DB paths with
`--live-db` / `--sim-db` if non-canonical.

The script reads two SQLite files (read-only via URI mode) — never
mutates anything. Safe to schedule in cron.

## Suggested daily schedule

```
0 18 * * 1-5  cd $RENQUANT && .venv/bin/python scripts/reconcile_live_sim.py \
              >> logs/reconcile.log 2>&1
```

(Run after the 4pm market close + sim retrain so both DBs are settled.)

## Recommended alarms

Wire these into `scripts/check_retrain_triggers.py` or whatever
M3-Dagster ends up doing for the M5 sensor:

| Metric | Threshold | Action |
| --- | --- | --- |
| `divergence_rate` | > 5% | block promote, page operator |
| `slippage.p95_bps` | > 30 | warn; investigate venue / order type |
| `slippage.max_bps` | > 100 | page; possible bug or fat-finger |
| `rolling_ic.ic` | < 0.005 | model decay alert; consider retrain |
| `rolling_ic.ok` is false | persistently | upstream data pipeline broken |

The CLI prints a one-line summary on stdout for cron-scrape.

## Integration with M3 Dagster (TBD)

M3 owns the orchestrator. Two sensors live there:

1. **DivergenceRate sensor** — reads `metrics["divergence"]["divergence_rate"]`
   from the markdown report (or, better, a sibling `.json` we may add)
   and fires a pager when > 5%.
2. **PromoteGate sensor** — when `train_104.py` is about to promote
   challenger to golden, M3 reads today's reconciliation report and
   blocks promote if `divergence_rate > 5%` OR `slippage.p95_bps > 30`.

Until M3 lands, the report is operator-readable: open in your editor
each morning, scan the **Summary** section, react.

## Module layout (single-responsibility, ≤50-line helpers per CLAUDE.md §1c)

```
live_sim_reconcile.py
├── load_live_fills          # SQLite read of trades joined on pipeline_runs
├── load_sim_decisions       # ditto, run_type='sim'
├── replay_through_sim       # match live fill -> sim decision (or via SimAdapter)
├── compute_slippage         # p50 / p95 / max signed bps
├── compute_decision_divergence
├── compute_rolling_ic       # 30d Spearman; safe-skip if tables missing
├── emit_report              # markdown
└── build_per_day_breakdown  # date-bucketed table
```

## Tests

`tests/test_reconciliation.py` covers:

* synthetic runs.db fixtures for every helper
* same-fills → 0 divergence, 0 slippage
* sim disagrees on 50% of fills → divergence rate ≈ 0.5
* known synthetic spread → slippage p95 calibrated
* synthetic predictions matching realized → high rolling IC
* missing tables / empty DB → emit empty report, no crash
* **end-to-end SimAdapter walk** with a stub adapter (per §5.13.1: tests
  must walk the real prod path, not just hand-built fixtures)

Run: `pytest tests/test_reconciliation.py -v`.

## What this is NOT

* **Not a live runner.** Read-only over both DBs. Never opens a broker
  client, never writes to runs.db.
* **Not a backtest.** It only re-asks "what would sim have done on the
  *exact* day we traded?" — not what sim's PnL would have been over a
  longer window.
* **Not a model trainer.** Rolling IC is observational. If IC degrades,
  the operator (or M3) decides whether to retrain.
