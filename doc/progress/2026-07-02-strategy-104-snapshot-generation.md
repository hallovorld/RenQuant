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
- §4(b) evidence block (`doc/AGENT-RETROSPECTIVE.md`):
  ```
  artifact:      scripts/render_strategy_104_snapshot.py, doc/arch/strategy-104-snapshot.md
  prod or exp:   prod (reads the PINNED backtesting/renquant_104/strategy_config.json —
                 the config weekly_wf_promote.sh treats as canonical — read-only; writes
                 only a new doc file, never touches the prod config/artifacts themselves)
  existing data: grepped doc/arch/strategy-104.md's hand-written table (dated 2026-06-08,
                 unchanged since despite later prod/shadow switches) and confirmed via
                 direct read of strategy_config.json / strategy_config.golden.json that
                 the live pinned kind is currently hf_patchtst-primary/xgb-shadow — see the
                 "Correction to the amendment doc's own premise" note below
  best-known?:   this is the first and only current-state generator for this doc; no
                 prior variant to compare against (the thing it replaces is unversioned
                 hand-edited prose, not a competing generator)
  scope:         this is scripts/render_strategy_104_snapshot.py, prod (reads pinned
                 config read-only), vs the prior state of manually-edited, silently-stale
                 markdown with no freshness gate at all
  ```
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
- **Round-2 (Codex CHANGES_REQUESTED) fixes:**
  1. Field extraction is now a STRUCTURAL whitelist, not an emergent property of
     one function's code: new `_ALLOWED_METADATA_FIELDS` constant + `_extract_allowed()`
     — the ONLY point in the module that reads artifact-metadata field *values*;
     every other function receives data only through it. Proven by
     `test_metadata_whitelist_excludes_secrets_credentials_and_free_form_notes`
     (synthetic metadata carrying `api_key`, `broker_credentials`,
     `_local_debug_path`, `internal_notes`, `aws_access_key_id` alongside the
     legitimate fields — asserts none of it reaches the snapshot dict or the
     rendered markdown).
  2. New `_relativize_for_display()`: an artifact path inside the repo root is
     rendered relative to it; one outside is redacted to `<redacted-external-path>/
     <basename>` — never a raw absolute local path. Proven by
     `test_absolute_artifact_path_is_relativized_or_redacted_never_leaked_raw`.
  3. Tightened both the generated file's own header and the `strategy-104.md`
     pointer note to state explicitly that the generated snapshot is a
     **current fact** ("the pinned config says X, as of the last regeneration"),
     never a historical/promotion claim ("active since <date>") — that dated
     narrative stays in `strategy-104.md` only.
  4. Re-ran the generator against the REAL config after these changes — `git diff
     doc/arch/strategy-104-snapshot.md` is EMPTY (the whitelist doesn't drop
     anything the real artifacts actually stamp; this was a defense-in-depth
     hardening, not a behavior change for today's data), and `--check` still
     passes.
- `tests/test_render_strategy_104_snapshot.py` (8 tests total, synthetic fixtures
  only — no dependency on real repo config/artifacts existing) →
  `/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest tests/test_render_strategy_104_snapshot.py -q`
  → 8 passed. Covers: inline-JSON primary+shadow with cutoff-field priority,
  binary-checkpoint-via-sidecar metadata, missing-artifact graceful handling,
  markdown rendering, the `--check` stale/fresh gate, the metadata whitelist
  leak-proof fixture, and the absolute-path redaction fixture.
- `py_compile` clean on both changed files.
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
