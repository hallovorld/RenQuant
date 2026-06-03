# 2026-06-03 — Orchestrator bridge runtime-drift incident

**Severity**: SEV-2 (intraday sell automation completely down for ~2h;
no live orders missed because the wrapper failed before any sell was
attempted — fail-safe direction).

**Detected**: 2026-06-03 ~08:24 PDT by operator (cron log inspection
after noticing the lack of intraday activity).

**Resolved**: 2026-06-03 08:30 PDT (`make subrepo-runtime-root` +
manual `bash scripts/intraday_sell_104.sh` verified clean fire).

**Status**: closed; prevention measures in this PR.

---

## 1 · Timeline (PDT)

| Time | Event |
|---|---|
| 06:30 | First scheduled intraday_sell tick. **FAILED**: `renquant-orchestrator: error: argument command: invalid choice: 'live-bridge' (choose from 'daily-contract')`. |
| 06:30 → 08:24 | 11 successive 12-minute ticks failed identically. No notifications fired (the wrapper's notify path is only on the inner runner's nominal completion, not on argparse-exit at startup). |
| 08:24 | Operator surfaced the issue in chat. |
| 08:25 | Agent diagnosed via `tail logs/intraday_104/2026-06-03.log`. |
| 08:27 | Confirmed pin/vendor diff: `subrepos.lock.json` pinned `renquant-orchestrator` at `fa448e1` (the `live-bridge` merge), but `.subrepo_runtime/repos/renquant-orchestrator/` was at `9f05da8` (pre-`live-bridge`). |
| 08:28 | `make subrepo-runtime-root` advanced `.subrepo_runtime/repos/renquant-orchestrator/` → `fa448e1` and `.subrepo_runtime/repos/renquant-pipeline/` → `3288f60`. New assembly stamp `20260603T152935Z`. |
| 08:28 | `scripts/runtime_qp_sanity_check.py` → `OK: 13 runtime symbols present`. (Note: this gate would have PASSED even before the refresh — it doesn't check orchestrator CLI subcommands. See §4.) |
| 08:30 | `bash scripts/intraday_sell_104.sh` manual fire → clean: `SellOnlyPipeline DONE total=0.27s` + state write + alpaca disconnect. |
| 08:36+ | Cron resumed automatically on the next 12-minute tick. |

---

## 2 · Root cause

**Class**: vendored-subrepo runtime drift. Same class as the
2026-06-03 prod-runtime drift documented in
[`doc/ops/2026-06-03-prod-runtime-qpmultirepo-refresh.md`](2026-06-03-prod-runtime-qpmultirepo-refresh.md);
the runbook is
[`doc/ops/subrepo-runtime-refresh-runbook.md`](subrepo-runtime-refresh-runbook.md).

**Specifically**:
- `subrepos.lock.json` was updated by commit `74b76b4 chore(ops): pin
  orchestrator live bridge` earlier today; that commit PINNED the
  orchestrator at `fa448e1` which has the `live-bridge` and
  `daily-bridge` argparse subcommands registered
  ([`cli.py:64–79`](../../../renquant-orchestrator/src/renquant_orchestrator/cli.py)).
- The umbrella scripts `daily_104.sh:295` and `intraday_sell_104.sh:88`
  were updated in the same wave to call `-m renquant_orchestrator
  daily-bridge ...` and `-m renquant_orchestrator live-bridge ...`.
- **But** `make subrepo-runtime-root` (which checks out each pinned
  ref into `.subrepo_runtime/repos/<repo>/`) was not run after the
  pin landed. The vendored copy stayed at `9f05da8` (pre-`live-bridge`).
- First fire after the pin = first intraday tick at 06:30 PDT =
  immediate argparse error inside the inner subprocess.

The lockfile pin was correct. The umbrella code was correct. The
vendored snapshot was stale. This is **exactly** the failure mode the
PR #145 / #147 / #149 / #150 wave was supposed to catch — and it
**would** have caught it if either:
(a) `runtime_qp_sanity_check.py` checked orchestrator CLI subcommands
    (it doesn't — only portfolio_qp module symbols), OR
(b) `intraday_sell_104.sh` ran the sanity check as a pre-flight gate
    (it doesn't — only `daily_104.sh` does, per PR #147).

Both gaps are addressed in this PR.

---

## 3 · Fix executed (recorded for audit)

```bash
# All commands run from /Users/renhao/git/github/RenQuant.
make subrepo-runtime-root      # 9f05da8 → fa448e1 for orchestrator;
                               # 04b0299 → 3288f60 for pipeline
.venv/bin/python scripts/runtime_qp_sanity_check.py   # OK: 13 symbols
set -a; source .env; set +a
bash scripts/intraday_sell_104.sh                      # clean exit 0
```

**No tracked-file change was needed for the fix itself.**
`.subrepo_runtime/` and `.subrepo_assembly/` are gitignored
(machine-local state per `.gitignore`). The pin (`subrepos.lock.json`)
was already correct from `74b76b4`; the fix was the operator running
the apply step.

---

## 4 · Why the existing safety net missed it

| Gate | Why it didn't catch |
|---|---|
| `runtime_qp_sanity_check.py` REQUIRED_SYMBOLS | The 13 symbols are all `renquant_pipeline.kernel.portfolio_qp.*` + `renquant_common.metrics.*`. Orchestrator CLI subcommands aren't in the list. The script returns `OK` even when `python -m renquant_orchestrator live-bridge` would argparse-fail. |
| `daily_104.sh` RUNTIME-SANITY-FAIL gate | Only `daily_104.sh` runs the sanity check as a pre-flight (PR #147). `intraday_sell_104.sh` doesn't — it relied on the inner subprocess to surface failures. The intraday subprocess DID fail (argparse exit 2), but the wrapper just printed `=== intraday_sell FAILED ===` and exited — no actionable error, no runbook pointer, no notification. |
| `subrepos.lock.json` CI pin guard | The pin was correct. The guard checks the LOCKFILE, not the on-disk vendored copy. By design — it's a CI guard, not a runtime guard. |

---

## 5 · Prevention (this PR)

1. **Expand `runtime_qp_sanity_check.py`** to assert orchestrator CLI
   subcommand existence in addition to module symbols. Adds a
   `RuntimeCommand` dataclass + a `--help` subprocess probe that
   verifies the subparser is registered. With this, the next
   `live-bridge` / `daily-bridge` drift will fail the gate loudly,
   pointing at the runbook.

2. **Wire the sanity check into `intraday_sell_104.sh`** as a
   pre-flight, mirroring `daily_104.sh`'s `RUNTIME-SANITY-FAIL` gate.
   With this, the next drift surfaces the runbook pointer in the
   intraday log immediately instead of the cryptic argparse error.

3. **Update `doc/ops/subrepo-runtime-refresh-runbook.md`** §1 to add
   the intraday symptom row and §4 to surface the `RuntimeCommand`
   list alongside `REQUIRED_SYMBOLS`.

Not in this PR (intentionally): an automatic "run
`make subrepo-runtime-root` after every lockfile update" hook. The
operator (or future agent) doing the pin update is the right person to
also run the apply step — automatic apply could mask CI failures,
hide pin/code mismatches, and turn the runtime into a moving target
that contradicts §3.5's single-source-of-truth principle for vendored
state.

---

## 6 · Lessons (process)

1. **Pinning the lock and running the apply are TWO steps.** The
   commit message `chore(ops): pin orchestrator live bridge` (PR
   that produced `74b76b4`) was complete from a tracked-file
   perspective but operationally incomplete — it should either have
   included a checklist line "operator must run
   `make subrepo-runtime-root` before next cron tick" OR the daily
   pin-update PR template should require that step.

2. **Wrappers should surface runbook pointers on failure.** The
   intraday wrapper's "FAILED" message was a dead end; with this
   PR's gate update, it points at
   `doc/ops/subrepo-runtime-refresh-runbook.md` (same pattern that
   PR #155 added to the underlying sanity check).

3. **§3.1 PR-based workflow applies to incident response too.** The
   in-the-moment fix was operationally correct (one make target) and
   touched only gitignored state, but the audit trail
   (incident memo + prevention measures) belongs in a PR. This memo
   IS that PR. Future operational fixes that don't touch tracked
   files still go through a PR documenting the incident + closing
   the gap that allowed it.

---

Agent-Origin: Claude

🤖 Generated with [Claude Code](https://claude.com/claude-code)
