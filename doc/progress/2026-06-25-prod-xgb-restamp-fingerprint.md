# Fix: prod XGB config-fingerprint re-stamp (the prod twin of #410)

2026-06-25.

## Why
The 2026-06-25 live-tree agent-reset incident (postmortem in #412) reverted the uncommitted
re-stamp of the **prod** XGB panel scorer `artifacts/prod/panel-ltr.alpha158_fund.json` from
`f8fb2259` back to a stale `14586756`. A verification daily-full (readonly) then failed
`P-CONFIG-FP` (`live=f8fb2259 stored=14586756`, `diff_fields=['sector_map','watchlist']`) →
`panel_scorer_config_mismatch` cleared all candidates → fail-closed. **Tomorrow's 13:55 run
would have produced zero buys.** This is the prod twin of the shadow re-stamp #410.

## Root cause (same class as #410)
The watchlist grew **142→145** (+CRWV/+RKLB/+SPCX; sector_map +those labels) — purely additive,
no removals — but the prod artifact was stamped against the pre-growth config. The live re-stamp
to `f8fb2259` was uncommitted state, lost in the reset.

## Why re-stamp (not retrain)
Diff is `sector_map` + `watchlist` only; the watchlist change is additive (no removals). The XGB
panel scorer ranks cross-sectionally (no per-ticker embeddings), so new tickers are just extra
candidates scored by the same weights. `_model_relevant_fields` shows no non-sector diff → safe
re-stamp; a retrain for +3 names would be overkill.

## Fix
Re-stamped `config_fingerprint` + `config_fingerprint_fields` using the canonical
`renquant_common.config_consistency.fingerprint_config` over the pinned strategy-104
`strategy_config.json` (the config the live run uses) → `f8fb2259`.

## Verified
Readonly daily-full re-run: `preflight ✓ P-CONFIG-FP fingerprint match f8fb2259`,
`Config-consistency: panel-ltr.alpha158_fund.json OK`, scored 88 tickers, `ConvictionGate …
demeaned xs_mean=+0.0196` (demean active), full pipeline DONE. `system_doctor` ✓ green.

## Note
Committing it (vs leaving it as uncommitted live state, as it was before the incident) is the
durability lesson from #412 — so a future checkout can't silently revert the live prod scorer.
