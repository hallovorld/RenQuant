# Grant C step 1: the pin advance (momentum pipeline landing batch)

STATUS: planned (merges on codex approval; the machine sync is the granted
landing step that follows).
WHAT: subrepos.lock.json advanced for five subrepos via
scripts/refresh_subrepo_lock.py (CI-green validation passed 5/5): model →
e1f83f8c (through #196/#198 momentum TRAIN+evaluator), pipeline → 60871e24
(through #251/#252/#253 wash-sale floor, feature persistence, momentum
handler — collecting the standing #224/#245/#247 backlog), strategy-104 →
3bfd5abc (through #75 retire + #76 wash-sale design), base-data → f8514066,
common → ef7726dd (AC6 R4 schema).
WHY/DIR: operator approved Grant C (recorded on renquant-orchestrator#747)
— the momentum pipeline landing batch, step 1 of 5. Every advanced commit is
a reviewed merge on its repo's main.
EVIDENCE:
  artifact:      subrepos.lock.json (this PR's single functional change)
  prod or exp:   prod — the pin surface the runtime assembly consumes
  existing data: refresh_subrepo_lock.py --json output: 5/5 validated,
                 reason "CI green" per subrepo `[VERIFIED — the run that
                 produced this diff, 2026-08-02]`
  best-known?:   yes — the script's CI-red refusal is the designed guard;
                 no --force used
  scope:         lock file only; no code; the runtime materialization
                 (subrepo_assemble --runtime-root --sync) happens on the
                 operator-granted machine, recorded on #747
NEXT: merge → machine sync (granted) → plist install → first artifact →
s104#77 merge + second s104 pin advance → cleanups (the #747 Grant C list).
