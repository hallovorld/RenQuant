# daily_104 Step 4 retired — the PatchTST shadow e2e was booting a retired scorer to hit a refusal

**Date:** 2026-08-03 · `RenQuant` (umbrella) · GOAL-5 / 104-repair directive

STATUS:    script-only change; no behaviour of prod (Steps 1-3b) or the blend
           lane (Step 5) is touched. Lands on the live machine via the normal
           umbrella working-copy refresh (RenQuant#520 owner).
WHAT:      Deletes the Step 4 block (HF PatchTST primary shadow e2e) from
           `scripts/daily_104.sh`, replacing it with a retirement tombstone.
WHY:       The PatchTST line was RETIRED by the operator-delegated 2026-08-02
           decision (architecture preserved, no successor training), and
           renquant-strategy-104#75 retired the hf_patchtst shadow lane from
           the pinned configs in the same arc. The daily script still booted
           the full InferencePipeline on the retired scorer every session and
           ended at a buy-side preflight refusal.

EVIDENCE:

```
artifact:      scripts/daily_104.sh (Step 4 block removed)
prod or exp:   prod scheduled surface (daily104), buy path unaffected
existing data: logs/daily_104/2026-08-03.log Step 4 outcome:
               "Shadow run blocked by expected buy-side preflight gate
               (non-fatal, rc=1) — see .../2026-08-03_shadow.log"
               — the same outcome every session; the lane's watch was
               already retired from the shadow-scorer sentinel when the
               lane config was retired (orch#761 arc).  [VERIFIED — read
               this session]
scope:         "this is scripts/daily_104.sh ONLY; Steps 1-3b and Step 5
                are byte-identical; no config, artifact, or state change."
```

## What still exists on purpose

- `live_state.alpaca_shadow.json` / `runs_alpaca_shadow.db` / dated
  `*_shadow.log` files stay on disk as history — orphaned by design, no
  cleaner added.
- The in-process per-model comparison segments (`SHADOW[...]` in the prod
  ntfy) are a different mechanism (prod config `shadow_models`) and keep
  running.
- Step 5 (blend lane) is the maintained full-funnel readonly pattern; any
  future second e2e lane should clone its `RENQUANT_READONLY_TAG` isolation
  shape rather than resurrect Step 4 from history.

## Revert

`git revert` of this commit restores the block verbatim; nothing else to
undo (no state, no jobs, no pins touched).
