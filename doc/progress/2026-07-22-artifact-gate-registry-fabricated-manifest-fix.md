# Follow-up: config-artifact-path gate registry required a file that never existed (PR #525 r-fix)

STATUS:    delivered
WHAT:      `scripts/config_artifact_gate_registry.json` (merged via #525,
           commit 4259022) declared `xgb_prod_artifact_manifest.json` as a
           `required: true` profile. Removed that entry. The generic
           `artifact_manifest` shape handler (`collect_paths_artifact_
           manifest` in `scripts/check_config_artifact_paths.py`) stays —
           unit-tested, harmless, ready for whenever a real manifest of that
           shape exists — but is not registered to any profile until one is
           grounded. Updated the two tests that referenced the fabricated
           profile (`test_shipped_registry_declares_expected_profiles`,
           `test_registry_run_validates_multiple_profiles_and_skips_optional`)
           to assert its absence instead.
WHY/DIR:   Found during a final end-to-end verification of #525 against the
           real repo (before I knew codex's merge-loop had already merged it
           independently — this is the same fix, just landing as a small
           follow-up instead of amending the merged PR). `git ls-files` +
           a full-history grep for `xgb_prod_artifact_manifest`,
           `production_primary`, `readonly_shadow` across the whole repo
           returns zero hits outside the gate's own registry/tests — no
           scheduled path anywhere in this codebase produces or loads a file
           by that name. A `required: true` profile pointing at a file
           nothing ever creates makes `verify-pinned-paths`
           (`.github/workflows/config-artifact-path-gate.yml`) fail closed
           **permanently**, on every future `subrepos.lock.json` change,
           regardless of whether the real configured artifact paths are
           correct — the opposite of what a pre-deploy gate should do. This
           was live on `main` (merged, not caught before merge) until this
           fix.
EVIDENCE:  n/a
           (CI gate + unit tests only, no model/data performance claim.)
           `[VERIFIED]` `pytest -q tests/test_check_config_artifact_paths.py`
           with `PYTHONPATH=<renquant-pipeline>/src` (real #211 resolver):
           21 passed. Ran the real gate against the real repo topology
           (`--registry scripts/config_artifact_gate_registry.json
           --configs-dir backtesting/renquant_104 --strategy-dir
           backtesting/renquant_104 --data-root .`): 2 required profiles
           validated, 2 optional profiles correctly skipped as absent
           (`shadow_a`, `shadow_b` — also genuinely absent from this repo,
           confirmed via `git ls-files`), 7 paths checked, exactly 1 FAIL —
           the disclosed `../../` PatchTST escape (unchanged, still open,
           tracked by #525/#524) — exit 1. No fabricated-manifest failure.
NEXT:      #524 (blocked on #525, which is now merged) can rebase and
           demonstrate it runs against this corrected gate. The `../../`
           escape itself is out of scope here (lives in
           renquant-strategy-104, per #525's original disclosure).
