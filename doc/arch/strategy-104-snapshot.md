# renquant_104 — generated production snapshot

GENERATED FILE — do not hand-edit. Regenerate with:
`python3 scripts/render_strategy_104_snapshot.py`

Rendered directly from the PINNED config + each referenced artifact's own
stamped metadata (never hand-maintained prose) — see amendment A6,
doc/design/2026-07-01-104-105-design-review-amendments.md. This states ONLY
what the pinned config says AS OF the last regeneration (CI-enforced fresh —
see the workflow below) — a current fact, never a historical/promotion claim
("active since <date>", "promoted on <date>"); that narrative, with its own
dating and provenance, belongs in doc/arch/strategy-104.md instead. Fields
with no clean current-state source (a specific WF run's mean IC, a
regime-detector commit hash, etc.) are NOT rendered here either, for the
same reason — they stay as dated narrative in doc/arch/strategy-104.md.

Source config: `backtesting/renquant_104/strategy_config.json`

| | |
|---|---|
| Active model | `kind="hf_patchtst"`; artifact=`../../artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`; trained_date=2026-05-22; effective_selection_cutoff_date=2026-02-10; lookahead_days=60; fingerprint=sha256:f8fb2259b2bf1537 |
| Shadow model | `kind="xgb"`; name=`xgb_alpha158_fund_previous_primary`; artifact=`artifacts/prod/panel-ltr.alpha158_fund.json`; trained_date=2026-05-18; binding data cutoff=unknown; lookahead_days=60; fingerprint=sha256:14586756d4f67691 |
| Watchlist size | 142 tickers |

