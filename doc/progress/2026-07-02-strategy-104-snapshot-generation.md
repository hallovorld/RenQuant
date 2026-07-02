# Machine-generate the strategy-104 production snapshot

STATUS: delivered
WHAT: `doc/arch/strategy-104.md`'s hand-written "Production snapshot" table drifted
from reality (stayed dated 2026-06-08 through later changes; a related design
review round was wasted trusting it — #210 R1). New `scripts/render_strategy_104_snapshot.py`
reads the PINNED `backtesting/renquant_104/strategy_config.json` (the config
`weekly_wf_promote.sh` treats as canonical prod) plus each referenced artifact's
own stamped metadata — inline JSON for GBDT/XGB artifacts, a `<path>.metadata.json`
sidecar for binary checkpoints (hf_patchtst/.pt) — and renders a deterministic
snapshot to `doc/arch/strategy-104-snapshot.md`: active/shadow model kind,
artifact path, trained_date, binding data cutoff (same `_DATA_CUTOFF_FIELDS`
priority convention as `model_freshness_monitor.py`/`shadow_scoring.py`),
lookahead_days, config_fingerprint, watchlist size. A short pointer was added
near the top of `strategy-104.md` directing readers to the generated file.
Fields with no clean current-state source (a specific WF run's mean IC, a
regime-detector commit hash) were deliberately NOT rendered — they stay
hand-written narrative in `strategy-104.md`.
WHY/DIR: amendment A6, `doc/design/2026-07-01-104-105-design-review-amendments.md`
(renquant-orchestrator PR #223) — a hand-maintained "production snapshot" cannot
stay truthful at this change frequency, and staleness here has already cost a
full review round elsewhere.
EVIDENCE:
- Ran the generator against the REAL live config in this repo (read-only):
  `python3 scripts/render_strategy_104_snapshot.py` → `doc/arch/strategy-104-snapshot.md`.
  Output: primary `kind="hf_patchtst"`, artifact
  `../../artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`,
  trained_date=2026-05-22, `effective_selection_cutoff_date=2026-02-10`,
  lookahead_days=60, fingerprint=`sha256:f8fb2259b2bf1537`; shadow
  `kind="xgb"` name=`xgb_alpha158_fund_previous_primary`, artifact
  `artifacts/prod/panel-ltr.alpha158_fund.json`, trained_date=2026-05-18,
  binding data cutoff=unknown (this XGB artifact stamps none of
  `_DATA_CUTOFF_FIELDS`), lookahead_days=60, fingerprint=`sha256:14586756d4f67691`;
  watchlist=142 tickers.
- **Correction to the amendment doc's own premise**: A6 as written cites "the
  operator re-promoted XGB on 2026-06-23" as the reason `strategy-104.md` is
  stale. I checked the ACTUAL current live config directly (both
  `strategy_config.json` and `strategy_config.golden.json`) and both show
  `panel_scoring.kind="hf_patchtst"` as PRIMARY right now, with XGB as the
  shadow/rollback — i.e. the SAME thing `strategy-104.md` already says. I could
  not independently reproduce the cited 2026-06-23 XGB-primary state from
  current config. Either that promotion was later reversed, or the amendment
  doc's own claim was itself based on stale information — either way, this is
  exactly the kind of hand-checked claim the generator now makes unnecessary
  to litigate: the committed `strategy-104-snapshot.md` reflects whatever the
  pinned config actually says, always.
- `python3 scripts/render_strategy_104_snapshot.py --check` → exit 0 against the
  freshly-generated file (proves the check-mode gate works: exact regeneration
  match, not a day-count heuristic).
- `tests/test_render_strategy_104_snapshot.py` (6 new tests, synthetic fixtures
  only — no dependency on real repo config/artifacts existing) →
  `/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest tests/test_render_strategy_104_snapshot.py -q`
  → 6 passed. Covers: inline-JSON primary+shadow with cutoff-field priority,
  binary-checkpoint-via-sidecar metadata, missing-artifact graceful handling,
  markdown rendering, and the `--check` stale/fresh gate itself.
- `py_compile` clean on both new files.
- New `.github/workflows/strategy-104-snapshot-fresh.yml` (mirrors the existing
  `subrepo-pin-ci-green.yml` style) runs `render_strategy_104_snapshot.py --check`
  on every PR/push to main — fails if the committed snapshot doesn't exactly
  match what regenerating from the current pinned config would produce.
NEXT: wiring regeneration into `weekly_wf_promote.sh` itself (so the snapshot
auto-refreshes after every successful weekly promote, not just whenever someone
remembers to re-run the script by hand) was deliberately deferred — that's a
live production promote pipeline and this PR did not want to touch it. Until
that's wired, the CI check (this PR) is what actually enforces freshness: any
PR that changes the pinned config without regenerating the snapshot fails CI.
