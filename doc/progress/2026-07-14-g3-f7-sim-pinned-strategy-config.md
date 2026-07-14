# G3 F-7: sim script resolves pinned strategy config

Date: 2026-07-14
PR: fix(sim): resolve strategy config from pinned subrepo (G3 F-7)

## Problem

`scripts/run_sim_104.py` loaded strategy config from the umbrella's local
copy at `backtesting/renquant_104/strategy_config.json`. This copy has
drifted from the pinned `renquant-strategy-104/configs/strategy_config.json`
that the live bridge uses — sim evaluated a different primary scorer kind
(hf_patchtst) than live (xgb), plus missing config sections (deployment
governor, fractional shares, software stops, decision ledger, intraday
decisioning).

Finding F-7 of the 2026-07-04 architecture compliance audit
(`doc/arch/2026-07-04-umbrella-compliance-audit.md`).

## Fix

`run_sim_104.py` now resolves the strategy config from the pinned subrepo
checkout (`../renquant-strategy-104/configs/<name>`) first. Falls back to
the umbrella copy with a warning if the pin is unavailable. Matches the
live bridge's resolution path (`_with_pinned_strategy_config`).

## Scope

Only `run_sim_104.py` is changed. Other scripts (`run_wf_gate.py`,
`analyze_backtest.py`, etc.) still use the umbrella copy and can be
migrated incrementally.
