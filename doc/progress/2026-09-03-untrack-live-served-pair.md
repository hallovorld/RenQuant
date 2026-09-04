# Untrack the live-mutated served pair (panel-ltr + rank calibrator)   (PR #TBD)

STATUS:    delivered — repo-hygiene fix in the 2026-08-31 clobber class
           (RenQuant#638's sibling); operator-authorized 2026-09-03.
WHAT:      (1) index deletions of `backtesting/renquant_104/artifacts/prod/
           panel-ltr.alpha158_fund.json` and `…/panel-rank-calibration.json`
           (`git rm --cached`; the files stay on disk) + `.gitignore` rules
           for the pair and the side-files the promote jobs write beside it
           (`.previous.json`, `weekly_*.staging.json`, `weekly_rollback_*`,
           `.json.bak-*`/`.json.accepted_receipt-*`, `monthly_rollback_*`,
           `_rejected_calibrators/`); the three FROZEN calibrator variants
           (`bull_calm`, `recent-12mo`, `pre-2026-05-15-clip`) stay tracked
           and are proven unmatched. (2) `deploy/live_mutated_prod_artifacts.json`
           — a schema-v1 deployment declaration naming exactly the two paths,
           their role and their writers. (3) `scripts/check_config_artifact_paths.py`:
           when a config field does not resolve AND the path is declared in
           that file for THIS strategy dir, the gate reports `ok` with
           `INFO: live-mutated served artifact absent on THIS machine …`
           instead of "does not resolve" — the ledger waiver's shape
           (`machine_produced_ledger_marker`) for a `.json` primary/calibrator;
           an undeclared absent artifact still fails, and a present declared
           artifact is checked in full. 5 new gate tests. (4)
           `tests/test_ac4_p0_store_declaration.py`: the P0 flat-pair
           presence pin becomes "present → is a file; absent → declared
           live-mutated". (5) `tests/acceptance/pipeline/test_pipeline_acceptance.py`:
           `test_required_artifact_set_present` skips (not fails) in a
           checkout where the declared pair is absent. (6)
           `tests/test_live_prod_pair_untracked.py` (5 tests, #638 template):
           index empty for the pair; patterns present; exact live paths +
           side-files ignored; frozen siblings still tracked and unignored;
           the declaration names exactly the untracked pair.
WHY/DIR:   `weekly_wf_promote.sh` (Step 5 / Step 4b via `fallback_pair_promote.py`),
           `manual_promote.sh` and `monthly_calibrator_refresh.sh`
           `os.replace()` the served pair on every promotion. Git-tracked,
           the two files are therefore permanently "modified" on the live
           tree (both since the 2026-08-31 promotion), and every live-tree
           pull either REFUSES on them or — when the working copy happens to
           equal the last commit — silently RESETS them: on 2026-08-31 07:17
           that reset wiped four refit rows from the git-tracked momentum
           ledger (RenQuant#638) and produced two days of scorer-identity
           CRITICAL / momentum DEGRADED pages. The served pair is the file
           the live book actually trades on; a reset there would swap the
           served model without any promotion record. The 2026-06-28
           untrack line (`doc/arch/2026-06-28-untrack-job-outputs-pipeline-fix.md`)
           kept WF calibrators tracked because they are CONSUMED FROM GIT
           by the gate; this pair is consumed from DISK by the live runner
           and only READ from git by CI's config-artifact-path gate — so
           the honest statement for CI becomes "config↔path shape is
           proven here; presence + identity are proven on the serving
           machine (load-time verification, the shadow-scorer sentinel, and
           `render_strategy_104_snapshot.py --check` in `make doctor`)",
           which is what the declaration + INFO waiver say in words.
           Direction: G-D (ops truth; `check_ops_deployment_ready.py` stops
           counting the pair as blocking dirt) and the 08-31 incident's
           permanent fix.
EVIDENCE:  artifact:      live tree `git status --porcelain -- backtesting/renquant_104/artifacts/prod/`: exactly the two pair files ` M` plus 18 untracked promotion side-files [VERIFIED — read-only, 2026-09-03 between 18:10 and 18:20 PDT]; writers: `scripts/fallback_pair_promote.py:41-59`, `scripts/weekly_wf_promote.sh:317-318/400/596/664-666`, `scripts/manual_promote.sh:73`, `scripts/monthly_calibrator_refresh.sh:146/165`, `scripts/fit_calibrator_alpha158_fund.py:334` [VERIFIED — read-only grep by the impact analysis]
           prod or exp:   prod repo hygiene + one CI gate's classification of a declared-absent path; no served file is moved, rewritten, or promoted
           existing data: impact analysis (read-only, every workflow + test that names the pair): no workflow runs `tests/` wholesale; `verify_pinned_declaration` (snapshot CI) reads no artifact; the byte-exact snapshot `--check` runs only where the files remain on disk → no re-render; the ONE real CI regression without (3) is `config-artifact-path-gate.yml` `verify-pinned-paths` (runs when `subrepos.lock.json` changes) failing 4 config fields — hence the declaration + waiver ride in this PR [VERIFIED — analysis completed 2026-09-03 between 18:23 and 18:47 PDT]; tests in the sparse PR clone with `artifacts/prod` checked out (pair absent — CI's condition): `test_check_config_artifact_paths.py` (5 new) + `test_ac4_p0_store_declaration.py` + `test_live_prod_pair_untracked.py` (5 new) + `test_ops_deployment_ready.py` + `test_repo_hygiene_audit.py` = 76 passed, 3 skipped; `tests/acceptance/pipeline/test_pipeline_acceptance.py::test_required_artifact_set_present` SKIPS with the declared-absent message (its sibling smoke test fails in the sparse clone only because `training_panel` is outside the cone — pre-existing) [VERIFIED — 2026-09-03 between 18:40 and 18:47 PDT]
           best-known?:   n/a — hygiene; no model claim
           scope:         "this PR untracks two files, declares them, and teaches one CI gate to report a declared-absent path as INFO; it does not touch the served files, the promote scripts, or any config"
NEXT:      LANDING RUNBOOK (the ff-only deletes or refuses — never plain pull):
           the two live copies are MODIFIED relative to the last commit, so
           `git pull --ff-only` on the live tree will REFUSE ("Your local
           changes … would be overwritten"). Procedure, in order:
           (1) back up both files + sha256 to `data/state_backups/<ts>/`;
           (2) operator-granted, recorded step: `git rm --cached` the two
           paths in the LIVE index (index-only; the working files are
           untouched) — this is the one git action the landing needs and it
           must be in the ff grant; (3) `git pull --ff-only`; (4) verify:
           `git ls-files -- <pair>` empty, `git check-ignore -v <pair>` hits
           the new rules, sha256 of both files unchanged vs (1), `git status`
           no longer lists them; (5) `python scripts/check_ops_deployment_ready.py`
           and `make snapshot-check` green. If anything differs, restore from
           (1) BEFORE the next 13:55 run. Follow-up, same class, not in this
           PR: `artifacts/prod/earnings-calendar.json` is tracked and
           rewritten daily by `scripts/fetch_earnings_calendar.py` (clean
           today only because content matched).
