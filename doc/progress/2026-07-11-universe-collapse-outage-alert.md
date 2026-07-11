# Universe-collapse must page as an OUTAGE, not report as a normal no-trade

STATUS: new (follow-up #2 of the 2026-07-11 META no-buy forensics, orchestrator PR #473)
WHAT: when the per-ticker admission universe collapses — loaded models below a config-keyed
floor fraction of the watchlist (`universe_collapse_floor_frac`, safe default 0.5), or ZERO
loaded on a non-empty watchlist regardless of the floor — the daily decision ntfy TITLE now
carries a `UNIVERSE-OUTAGE` marker (distinct from a plain `DECISION`), the body leads with
`loaded/watchlist` plus per-cause rejection counts (e.g. `stale:live_train_end=133`), the
alert priority is raised to `high` on live runs, the dedup key is split so an outage can never
be cooldown-suppressed by a prior healthy no-trade, and the persisted run bundle records
`universe_collapse: true|false` + the full `universe_health` dict. Observability only — no
trading decision changes.
WHY/DIR: on 2026-07-08 and 2026-07-09 the per-ticker `live_train_end` metadata regressed to a
2026-04 vintage (live-tree `git checkout HEAD -- backtesting/renquant_104/models/`, 07-08
10:23 PT — the committed snapshot 6daabf6 is a stale 2026-04-20/23 vintage; the follow-up
hotfix bumped `trained_date`, which the axis-based freshness gate DELIBERATELY ignores). The
60d gate correctly fail-closed on 133/145 admission models, the buy scan ran with **0 tickers
for two full sessions**, and ntfy reported both days as a normal
`DECISION | no trade (no_candidates)` — a silent availability outage rendered as a market
decision. Forensics: orchestrator `doc/research/2026-07-11-meta-no-buy-forensics.md` §5 + §8.2.
EVIDENCE:
```
artifact:      live/runner.py (_universe_health/_universe_rejection_cause/_universe_floor_frac,
               stamped in _load_strategy_multi, surfaced in _notify_decision, attached to ctx in
               _run_once_multi_pipeline),
               backtesting/renquant_104/kernel/artifact_contract.py (build_run_bundle records
               universe_collapse + universe_health)
prod or exp:   PROD notification/reporting path, but strictly additive observability — no gate,
               order, sizing, or admission behavior reads the new fields; sell-only cycles never
               carry the marker (no buy scan runs there); ctxs without the stamp (sim/native
               paths) are byte-identical to before
tests:         tests/test_universe_collapse_alert.py (15 tests: collapse verdict incl. the exact
               07-08 shape 4/145 + stale:live_train_end=133, zero-loaded-with-zero-floor,
               boundary at floor, cause bucketing for both legacy and axis-gate reason formats,
               config floor clamping so a typo can't disable the page, title marker on full runs
               only, priority bump, dedup-key split, run-bundle persistence + absence);
               affected suites green: test_no_trade_priority, test_runner_trade_ntfy (except one
               pre-existing main failure, see below), test_artifact_contract,
               test_s_frac_stage0_commit_contract, test_runner_preflight_fail_closed,
               test_env_fingerprint — 324 passed, 0 new failures (A/B vs origin/main baseline)
known-broken:  tests/test_runner_trade_ntfy.py::TestSourceLevel::
               test_live_only_wrapper_does_not_duplicate_runner_success_ntfy already fails at
               origin/main (expects "Wrapper success ntfy suppressed" in scripts/live_only_104.sh;
               the string is absent from origin/main AND the live tree) — pre-existing, untouched
```
CONFIG: `universe_collapse_floor_frac` (top-level strategy config key, fraction of watchlist,
clamped to [0,1], default **0.5**; out-of-range/non-numeric falls back to the default so the
page cannot be silently disabled). Calibration: outage days ran at 4/145 ≈ 2.8%; the degraded
pre-outage sessions (partial-retrain corruption, 77 `no_artifact`) ran at 58/145 = 40% — both
correctly below the floor; post-recovery steady state is 125/145 ≈ 86%.
