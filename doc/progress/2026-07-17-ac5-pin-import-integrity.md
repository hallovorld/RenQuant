# GOAL-5 AC5 (D1): pin import-integrity sweep as a lock-PR gate

STATUS: delivered
WHAT: `scripts/pin_import_integrity_sweep.py` bootstraps the multirepo
runtime exactly as the daily does against a candidate lock + checkouts,
AST-walks the pinned pipeline for EVERY aliased-namespace import
(module-level and function-local, `kernel.*` and
`renquant_pipeline.kernel.*`), and imports each target post-bootstrap;
`from M import n` names must resolve as attribute or submodule of the
ALIASED M. Failures name the import, the source site, and the fix side.
`.github/workflows/pin-import-integrity.yml` runs it as a required check
on PRs touching subrepos.lock.json (persist-credentials: false on every
checkout). The sweep rewrites lock local_path/remote to its OWN provided
checkouts and runs the pin guard fail-closed (RENQUANT_STRICT_SUBREPO_PATHS)
— during development the resolver silently substituted the dev machine's
real sibling checkouts for the fixture ones; the strict rewrite closes that.
WHY/DIR: GOAL-5 P0 AC5 CI half. Per-repo CI is structurally blind to the
#524 class (both repos green, first post-pin-sync daily died at
MetaLabelVetoTask).
EVIDENCE (acceptance fixture): against {orch pre-#524 bfb935e4, pipeline
7108f514} the sweep exits 1 naming
`renquant_pipeline.kernel.meta_label.task_meta_label_veto` (both call
sites) AND a second latent gap the incident never reached
(`meta_label.job_meta_label_log`, pp_inference:670); against the current
lock combination it exits 0 with 320 targets checked. Automated in
`tests/test_pin_import_integrity_sweep.py` (6/6: 4 AST unit + both
regression directions; regression class skips where sibling clones are
absent — the workflow's live sweep covers CI).
NEXT: none for D1; the machine-side half (dawn funnel preflight) is
renquant-orchestrator #533.
