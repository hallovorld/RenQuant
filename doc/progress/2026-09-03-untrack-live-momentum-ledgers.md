# Untrack the live-mutated momentum shadow ledgers   (PR TBD)

STATUS:    delivered — repo bookkeeping only; the live tree's files are not touched.
WHAT:      `git rm --cached` of `backtesting/renquant_104/artifacts/momentum/`
           (the ledger `momentum_artifact_ledger.jsonl` + the committed
           `2026-08-02/momentum_residual_v0.json`) and `.gitignore` rules for
           `backtesting/*/artifacts/momentum/` and `…/momentum_fast/`. The live
           tree keeps every file on disk; git simply stops owning them.
WHY/DIR:   The Saturday `momentum-train-weekly` job APPENDS refit rows to the
           momentum ledger on the live tree. The ledger was git-tracked (committed
           2026-08-04 with one row); on 2026-08-31 07:17 the live-tree
           `git pull --ff-only` (and the tree-cleaning before it) reset it to the
           committed 1-row version — file mtime 07:17:50 vs the pull's reflog
           07:17:54 [VERIFIED] — wiping the 08-08/15/22/29 rows, while the
           UNTRACKED `momentum_fast` ledger kept all six. Since then two daily
           pages: `scorer-identity CRITICAL` (a1149c → 9aa2d8 same-path
           replacement — a correct detection) and `SHADOW SCORER DEGRADED
           momentum_residual_v0 stale 32d` (the served row is the 08-02 cutoff).
           A file a live job mutates must never be repo content. Direction: G-F
           (every page has a root cause + fix) / G-D (ops truth).
EVIDENCE:  artifact:      live `backtesting/renquant_104/artifacts/momentum/momentum_artifact_ledger.jsonl` (1 row, cutoff 2026-08-02, mtime 2026-08-31 07:17:50); refit dirs 2026-08-08/15/22/29 still present [VERIFIED — ls/stat 2026-09-03]
           prod or exp:   prod ops (git bookkeeping); no trading path, no file content change
           existing data: `logs/rq104/scorer_identity_2026-09-01..03.log` CRITICAL lines; untracked momentum_fast ledger rows 0..5 cutoffs 08-06..08-29 [VERIFIED]
           best-known?:   n/a
           scope:         "stop tracking two live ledgers and their artifact dirs; recovery of the wiped rows is a separate, receipted operation"
NEXT:      (1) rebuild the wiped rows from the surviving dated artifact dirs with the
           trainer's ledger writer + a shadow_receipt so the identity monitor
           legitimizes the change (own PR / recorded operation); (2) decision for
           the operator: the served artifacts under `artifacts/prod/` are ALSO
           git-tracked and live-mutated (the A4-T1 promotion rewrote
           `panel-ltr.alpha158_fund.json` today) — same hazard class; untracking
           them changes what the snapshot renderer and "committed corpora" mean, so
           it is a policy decision, not a quiet PR.
