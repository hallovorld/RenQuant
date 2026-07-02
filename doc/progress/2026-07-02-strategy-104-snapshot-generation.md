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

## Round 4 (Codex CHANGES_REQUESTED, round-3 follow-up): the real model-promotion path was still uncovered

**Finding.** Round 3 wired `promote_pin.py` (subrepo-pin bump/revert) and `system_doctor.py`
(daily backstop) — real, correct fixes — but explicitly disclosed the remaining gap:
`weekly_wf_promote.sh`, the ACTUAL model-promotion path (retrain → WF gate → swap active
artifact+calibrator), had no inline check. It could successfully promote a new model and
exit 0 while leaving the committed snapshot stale, with only the NEXT daily
`system_doctor` run eventually reporting it — asynchronous, delayed detection, not the
synchronous same-run enforcement the review required. Round 3's own PR body still
described post-promote enforcement as "merely recommended," contradicting what round 3's
code had actually implemented for `promote_pin.py`.

**Fix.** Wired the SAME `promote_pin.check_snapshot_freshness()` (scratch-rendered,
diff-preview, never auto-commits, never reverts the promotion for a stale-snapshot
finding alone) into every script that mutates the active artifact/calibrator/pin state
this snapshot declares:
- `scripts/weekly_wf_promote.sh` — new Step 7, after the dashboard refresh (Step 6) and
  before the final `PASSED`/`WEEKLY-PROMOTE ✓` success signal. On failure: prints the diff
  preview, sends a distinct `WEEKLY-PROMOTE — SNAPSHOT STALE` ntfy alert (so it's
  distinguishable from a genuine promote failure), and `exit 1` — the promotion itself is
  NOT undone.
- `scripts/manual_promote.sh` — the emergency operator path (bypasses the WF gate by
  design). Same check added at the end, same no-revert contract, same `exit 1` on
  staleness (even though this is an interactive script nobody automates on today, failing
  closed costs nothing and is safer if that ever changes).
- `scripts/restamp_prod_fingerprint.py` — re-stamps the active artifact's fingerprint
  fields in place (a sector-map-only legacy repair, no retrain). Same check added right
  before its final `return 0`; returns 1 on staleness without reverting the (already
  verified-consistent) re-stamp.
- `scripts/promote_shadow_patchtst.py` — the SHADOW PatchTST served-pin swap. Same check
  added right after `rep.rc = RC_OK` is set following a real (non-dry-run) swap; sets
  `rep.rc = RC_GATE_FAILED` on staleness and appends the message to `rep.verdict`. This
  scorer moves no capital, but `collect_snapshot()` reads BOTH the active AND shadow
  config, so a stale snapshot doc from a shadow-pin change is still real drift worth
  surfacing.

Searched for other promotion/rollback/re-stamp wrappers touching the same declared state
(`grep`-based sweep across `scripts/` for `promote(`/`def promote`) — these four are the
complete set found; no other wrapper mutates the artifact/calibrator/pin state this
snapshot represents.

**Tests.** New `tests/test_restamp_prod_fingerprint_snapshot_backstop.py` (3 tests, via a
synthetic sector-only-diff fixture + monkeypatched `promote_pin.check_snapshot_freshness`):
proves (a) a stale snapshot fails the run (`rc == 1`) even though the re-stamp itself
still gets applied (no revert), (b) a fresh snapshot succeeds normally, (c) `--dry-run`
never reaches the backstop at all (nothing was actually promoted yet). Existing
`tests/test_promote_pin.py`/`test_system_doctor.py` suites (which already cover
`check_snapshot_freshness` itself end-to-end, real non-mocked regenerate-and-diff) pass
unchanged — 45 tests total across the touched-adjacent suites, all green.

**Honest gap:** `promote_shadow_patchtst.py` has an existing 72-test suite, but none of
those tests exercise a REAL (non-dry-run) successful swap all the way through its several
gates (freshness/parity/smoke-inference/non-degenerate/resource/sanity-floor) — building
that fixture is a substantial undertaking distinct from this fix's scope, so the new
snapshot-backstop code path in that script is verified by manual code-reading + syntax
check, not by an executed test. Flagging this explicitly rather than claiming coverage
that doesn't exist.

**M9/A6 closure status (round 4 claim, corrected below):** with round 4, every real
production path that can change the active model/calibrator/pin state synchronously
FAILS when it would leave the committed snapshot stale — confirmed correct by Codex's
round-5 code inspection. But "correct by inspection" and "closed" are not the same
claim; round 5 below closes that gap.

## Round 5 (Codex review: code-reading is not execution proof)

**Finding.** Round 4's wiring was independently confirmed correct by code inspection —
`weekly_wf_promote.sh`'s Step 7 genuinely runs before the final success notification,
emits a distinct stale-snapshot alert, and exits non-zero. But NONE of `weekly_wf_
promote.sh` or `manual_promote.sh` had ever actually been EXECUTED under test — the
round-4 test file only exercised `restamp_prod_fingerprint.py` with a mocked freshness
function. `promote_shadow_patchtst.py`'s new path was explicitly admitted unexecuted.
These are production-mutating shell/Python entry points; a syntax check proves the
script parses, not that Step 7's actual control flow does what the comments claim.

**Fix.**
- Added minimal, behavior-preserving environment-override hooks to both shell scripts
  (`RQ_WEEKLY_PROMOTE_REPO_DIR`/`_PYTHON`/`_NOTIFY_LOG`/`_LOCK_FILE`,
  `RQ_MANUAL_PROMOTE_REPO_DIR`/`_PYTHON`) — every default is byte-identical to the prior
  hardcoded value, so production invocation (which never sets these) is unchanged.
  `notify()` gained an opt-in log-append hook so a test can assert exactly which
  notifications fired without needing network access or touching the real ntfy topic.
- `tests/_weekly_promote_fixture.py` (new, shared): builds a self-contained fixture repo
  (genuine copies of `render_strategy_104_snapshot.py`/`promote_pin.py`, real minimal
  `kernel.model_acceptance.promote`, trivial stubs for smoke-test/retrain/WF-manifest-
  stamp/WF-gate/dashboard) — every dependency OTHER than the snapshot backstop itself is
  mocked; the backstop's own `check_snapshot_freshness` call is never mocked, it runs
  for real against the fixture's genuinely-rendered committed snapshot.
- `tests/test_weekly_wf_promote_snapshot_backstop.py` (new, 3 tests): runs the REAL
  `weekly_wf_promote.sh` via subprocess through a fully mocked Steps-1-6 promotion.
  Asserts: fresh snapshot → exit 0, exactly one `WEEKLY-PROMOTE ✓` notification, no
  stale alert, lock released; stale snapshot → exit 1, distinct `SNAPSHOT STALE` alert,
  the success notification genuinely never fires (not merely absent alongside a partial
  run); stale snapshot does NOT revert the already-completed promotion (active artifact
  content is the retrain stub's output, unchanged by the later Step-7 failure).
- `tests/test_manual_promote_snapshot_backstop.py` (new, 3 tests): same pattern for the
  interactive `manual_promote.sh` (stdin-fed for its three `read -p` confirmations) —
  fresh/stale/no-revert, mirroring the weekly-promote suite.
- `promote_shadow_patchtst.py`: extracted the inline snapshot-backstop block into a
  named `_apply_snapshot_freshness_backstop(repo, rep)` function — a minimal refactor
  (no behavior change; `run_promote()` now calls it instead of inlining the same code)
  that makes it independently unit-testable via monkeypatching `check_snapshot_freshness`,
  rather than needing to construct every unrelated gate (parity/cutoff/smoke-inference/
  atomic-swap) this script also checks before reaching a successful swap. 3 new tests in
  `tests/test_promote_shadow_patchtst.py`: fresh keeps `rc==RC_OK` and appends the
  message; stale sets `rc==RC_GATE_FAILED` and appends the alert WITHOUT touching
  `promoted_pin`/`superseded_backup` (the already-completed swap); the check is called
  with the exact `repo`/`sys.executable` arguments passed in. 75/75 in that file's full
  suite pass — no regressions from the extraction.
- CI (`.github/workflows/strategy-104-snapshot-fresh.yml`): the `selftest` job
  previously ran ONLY `test_render_strategy_104_snapshot.py` — none of round 3/4's
  claimed "117 tests" or this round's new execution tests were actually invoked by CI.
  Added `bash -n` syntax checks for both modified shell scripts, and wired ALL of
  `test_promote_pin.py`, `test_system_doctor.py`,
  `test_restamp_prod_fingerprint_snapshot_backstop.py`,
  `test_weekly_wf_promote_snapshot_backstop.py`, `test_manual_promote_snapshot_backstop.py`,
  and `test_promote_shadow_patchtst.py` into the same job. Verified locally by running the
  EXACT command CI now runs: 121 passed, matching what a fresh CI run will show.

**Honest gap, closed.** The prior round's flagged gap (`promote_shadow_patchtst.py`'s
path verified by code-reading only) is now covered by 3 real executed unit tests against
the extracted function — not a full end-to-end run of the whole script's gate chain
(building that remains a substantial undertaking distinct from this fix's actual scope),
but a genuine, monkeypatch-isolated test of the exact integration point under review.

**M9/A6 closure status (corrected):** every real production path that can change the
active model/calibrator/pin state now (a) synchronously fails when it would leave the
committed snapshot stale, AND (b) has that behavior proven by an executed test that CI
actually runs — not merely a correct-by-inspection code review. The "deployed but dark"
gap this task exists to close is closed, with the closure itself demonstrated rather
than asserted.

## Round 5.1 (green the round-5 push + close two holes found in review of it)

The round-5 commit was correct in structure but its first CI run went RED and
two real defects surfaced on inspection; fixed forward:

- **CI red root cause (3 failures in `tests/test_promote_pin.py`):** the
  `bump --apply` tests relied on the DEFAULT verify step being skipped
  (`check_conviction_admits.py` absent). On the hosted runner's full checkout
  the script exists, ran against the synthetic lock fixture, failed for lack
  of live data, and auto-reverted the bump before the behavior under test was
  reached (locally it silently passed because that script reads the LIVE
  run bundle and exited 0 — an environment-dependent test, the actual bug).
  All four such invocations now pass an explicit `--verify-cmd true`,
  hermetic in both environments.
- **Real ntfy/notifier leak in the weekly harness:** running the real
  `weekly_wf_promote.sh` under test still executed the real `curl` POST to
  the production `ntfy.sh/renquant` topic (including a fake
  "SNAPSHOT STALE" alert — a false alarm into the live operator channel on
  every CI/local run) and real `terminal-notifier` popups on macOS.
  `tests/_weekly_promote_fixture.py` now installs no-op `curl` /
  `terminal-notifier` PATH shims (`shim_bin_dir()`), prepended to the
  subprocess PATH; assertions observe `RQ_WEEKLY_PROMOTE_NOTIFY_LOG` as
  before.
- **Successful-swap WIRING proof:** the round-5 shadow tests exercise
  `_apply_snapshot_freshness_backstop` in isolation but never prove
  `run_promote()` reaches it on the successful-swap path. New
  `tests/test_promote_shadow_patchtst_snapshot_backstop.py` drives
  `run_promote()` through a REAL executed `--apply` swap (write-new copy,
  config backup, atomic pin rewrite, promote log) with the checker injected
  via the `promote_pin` module seam: stale flips rc after the swap without
  reverting the on-disk pin; fresh keeps `RC_OK`; dry-run never consults the
  checker. Added to the CI suite list.

Exact CI command re-run locally after these fixes: **124 passed**.
