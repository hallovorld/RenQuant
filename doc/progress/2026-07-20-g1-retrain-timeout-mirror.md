# 2026-07-20 — G1: mirror the retrain-timeout fix into the umbrella working-copy config

## Bottom line
The ~18-day 0-buy freeze is a **pipeline-reliability bug**, not a strategy no-trade.
Root cause (one of a stacked set): the umbrella working-copy retrain config
`backtesting/renquant_104/strategy_config.json` — the config `train_104.py`
actually loads — still carries the pre-fix `parallel_ticker_timeout_seconds: 600`
while the pinned `renquant-strategy-104` config was corrected to `2400` on
2026-07-01. The umbrella mirror commit was never made. This PR makes it.

## Evidence (code-traced, [VERIFIED])
- `scripts/weekly_tournament_retrain.sh:97` sets
  `STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"`, and
  its own comment (lines 137–138) states *"train_104.py reads its own umbrella
  strategy config (backtesting/renquant_104/strategy_config.json)"*.
- `scripts/train_104.py:194` loads `config_path = strategy_dir / args.strategy_config_name`
  (`--strategy-config-name` default `strategy_config.json`) → the umbrella copy.
- Verified values: umbrella active + golden = `600`; pinned `renquant-strategy-104`
  config + deployed `.subrepo_runtime` = `2400`.
- Mechanism of harm (verbatim from the pinned config's own note): a 600s cap
  times out the 142-ticker per-ticker tournament at ~67/142 → `ParallelTimeoutError`
  → whole retrain fails silently → per-ticker models never refresh → the 60-day
  universe-staleness gate drops all non-held tickers → ~0 buy candidates.

## Scope — deliberately surgical
Changes **only** `parallel_ticker_timeout_seconds` (600→2400) in both the active
and golden umbrella configs, in lockstep, plus two audit notes. The umbrella
working-copy has **many other** drifts vs the pinned config (watchlist additions,
`intraday_decisioning`/`sleeve` shadow blocks, `sdl_anchor_policy`,
`bear_trend_filter`, `max_hold_days` 40→500, `decision_ledger`, …). Those are a
separate mirror-lag question and are **intentionally not pulled in here** — this
PR is the minimal, behaviour-safe fix for the verified retrain-timeout drift.

## Blast radius
`parallel_ticker_timeout_seconds` governs only the **weekly tournament retrain
job's** per-ticker parallel timeout. It touches no daily-run decision logic and
no order path — it is a strict widening of a timeout for a separate weekly job.

## This is not a full fix on its own
The 63-stale-model freeze had a stacked cause set. This PR removes the
retrain-timeout blocker. Still open (tracked separately): the 2026-07-13
April-baseline `models/` overwrite trap, and the 2026-07-20 panel `mu=None` gap.

## Deploy is a separate, ask-first step
Merging this PR does **not** deploy it — the live daily/weekly jobs read the live
umbrella working checkout. Bringing this live requires, as ask-first machine
landings with full-funnel preflight + containment record: (1) a live-tree sync of
the umbrella working-copy, then (2) a weekly tournament retrain re-run (writes
~63 per-ticker model files). Neither is done in this PR.

## Separately: no proven profit edge from force-deploying the blocked candidates
D-validation (`doc/research/2026-07-20-g1-deploy-would-have-won.md`, horizon
sweep 20→60d) found **no robust edge** for the currently-blocked candidates
(single BULL_CALM regime). Conclusion: **fix the pipeline** (restore the ability
to trade) but do **not** force buys on an unproven edge.
