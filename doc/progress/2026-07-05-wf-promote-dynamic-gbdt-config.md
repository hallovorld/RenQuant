# fix(promote): dynamic GBDT config resolution in weekly_wf_promote

DATE: 2026-07-05

## What changed

`scripts/weekly_wf_promote.sh` hardcoded `strategy_config.shadow.json` as the
GBDT reference config. After the 06-23 operator lineup reversal (XGB back to
primary in `strategy_config.json`), the shadow config became PatchTST, so every
weekly XGB promote failed at the WF gate kind-parity check:

```
scorer kind ('hf_patchtst') does not match candidate kind ('xgb')
```

Replaced the hardcoded path with `_find_gbdt_config()` — scans both
`strategy_config.json` and `strategy_config.shadow.json`, reads their declared
`ranking.panel_scoring.kind`, and picks the one that declares `xgb`. Survives
future primary/shadow swaps without code changes.

Companion: renquant-backtesting#69 (same dynamic approach in the WF gate's
`_resolve_prod_reference_by_kind()`).
