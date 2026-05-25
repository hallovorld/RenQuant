# Repo Hygiene Ledger — 2026-05-24

This is a mainline workstream, not a side cleanup. The goal is to keep the
research and production tree auditable while preserving evidence artifacts
until they are summarized, reviewed, and either committed deliberately or
archived deliberately.

## Policy

- Do not delete files as part of hygiene audit.
- Do not move raw experiment evidence without review.
- Classify first, summarize second, then promote or archive.
- Keep production artifacts, broker state, strategy configs, and model
  metadata out of unrelated code commits unless the commit explicitly owns
  them.
- Treat untracked executable scripts as suspect until either promoted into
  tested tooling or documented as local scratch.

## Tooling Added

`scripts/audit_repo_hygiene.py` is read-only. It runs `git status`, classifies
dirty paths, and prints JSON or Markdown review queues. Its embedded policy is:

- `delete_files=false`
- `default_action=inventory_only`
- `archive_requires_review=true`

Regression coverage: `tests/test_repo_hygiene_audit.py`.

## Inventory Snapshot

Command:

```bash
.venv/bin/python scripts/audit_repo_hygiene.py --format md
```

Observed dirty entries during the hygiene audit: `364`. This number is a
snapshot, not a source of truth; rerun the command above after any commit.

| Class | Count | Immediate handling |
|---|---:|---|
| `broker_state` | 3 | Do not stage with code. Review only for live-state fixes. |
| `code` | 5 | Promote tested tools or mark scratch. |
| `documentation` | 2 | Stage only docs that belong to the current change. |
| `experiment_or_diagnostic_artifact` | 136 | Summarize evidence in docs before any archive decision. |
| `generated_data_or_logs` | 9 | Ignore from code commits unless a test fixture is intentional. |
| `local_agent_settings` | 2 | Local state; do not stage. |
| `per_ticker_model_artifact` | 133 | Commit only with a training/provenance acceptance record. |
| `production_model_artifact` | 20 | Production-risk queue; never bulk stage. |
| `scratch_code_artifact` | 15 | DOE/scratch scripts; archive or promote after review. |
| `shadow_model_artifact` | 2 | Shadow provenance required before staging. |
| `sim_model_artifact` | 21 | Keep as evidence until linked to a ledger. |
| `strategy_config` | 13 | Require paired config-parity tests before staging. |
| `untracked_uncategorized` | 3 | Inspect manually. |

## Mainline Queue

1. Keep the hygiene audit in every long-running experiment branch: while sims
   or training jobs run, scan dirty code/config/artifact queues instead of
   waiting idle.
2. Promote useful scratch scripts into named, tested tools or archive them
   after review. Current obvious queue: `scripts/_train_BB_*.py`,
   `scripts/_train_fwd20d.py`, `scripts/_train_fwd5d.py`.
   These are now classified as `scratch_code_artifact`, not production code.
   The live no-trade-streak repair tool was promoted only after adding a
   default dry-run contract; `--apply` is now required for state mutation.
3. Create experiment ledgers for raw result directories before cleanup:
   PatchTST, transformer prototypes, exit A/B, WF trade forensics, and
   Qlib baselines.
4. Keep `.gitignore` focused on local runtime stores, DBs, backup snapshots,
   and scratch directories. Do not hide production model/config drift with
   broad artifact ignores.
5. Before every commit, stage by ownership: code/tests/docs for the fix only;
   leave unrelated live state, model artifacts, data zips, and diagnostics
   unstaged.

## Why This Matters

The same failure class kept reappearing: stale side artifacts, old manifests,
and local scratch files made it hard to tell whether a result was production
evidence, diagnostic evidence, or accidental residue. This ledger makes cleanup
part of the trust boundary without destroying the data needed to audit past
claims.
