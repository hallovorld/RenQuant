# Portfolio-QP Drift Audit — umbrella vs renquant-pipeline

- **Date**: 2026-06-03
- **Auditor**: Claude (Opus 4.7)
- **Scope**: `backtesting/renquant_104/kernel/portfolio_qp/` (umbrella) vs
  `src/renquant_pipeline/kernel/portfolio_qp/` (pinned subrepo).
- **Trigger**: 6+ recent QP-architecture refactor PRs across both repos
  (umbrella PRs #128, #129, #130; subrepo PRs #19-22 mirrors). §3.5 mandates
  paired umbrella+subrepo PRs while the byte-equivalent mirror invariant
  holds.

## Post-Audit Status Update (2026-06-03)

This memo records the original drift audit. Subsequent PRs closed the
source-tree hazards:

- `renquant-pipeline@main` now contains `davis_norman.py` and
  `proportional_trade.py`.
- The matching subrepo tests now exist:
  `tests/test_davis_norman_band.py` and
  `tests/test_proportional_trade.py`.
- The QP mirror queue has advanced through snapshot/replay/significance,
  hard-only QP, WF replay, and Hybrid Option F.

The remaining P0 is operational, not source-tree parity: refresh
`.subrepo_runtime/repos/renquant-pipeline/` to subrepo `main`, run
paper/shadow smoke, and fail loud if the vendored runtime does not expose
the expected QP modules. See
[`doc/ops/2026-06-03-prod-runtime-qpmultirepo-refresh.md`](../../ops/2026-06-03-prod-runtime-qpmultirepo-refresh.md).

---

## Methodology

1. Directory listing both sides → enumerate per-file presence.
2. For each paired filename: LOC equality, module-level
   `class` / `def` symbols (`grep -E "^(class|def) "`), and import
   statements (`grep -E "^(from|import) "`).
3. Byte-equivalence check after rewriting the EXPECTED import-prefix swap
   `from kernel.` → `from renquant_pipeline.kernel.` (and `import kernel.`
   → `import renquant_pipeline.kernel.`) on the umbrella side, then MD5
   comparing against the raw subrepo file.
4. For umbrella-only and subrepo-only files: inspect call sites and
   resolve whether the asymmetry is harmless or production-risky.
5. Test pairing across `tests/` directories.

The import-prefix swap is the §3.5-blessed difference and is NOT counted
as drift.

---

## Original Per-File Status

| File | Umbrella | Subrepo | Status |
|---|---|---|---|
| `__init__.py` | 0 B | 0 B | paired, identical (empty) |
| `baseline_allocators.py` | 304 LOC | 304 LOC | **paired, byte-equivalent** (after import prefix swap, MD5 `7e0259e8…`) |
| `constraint_snapshot.py` | 275 LOC | 275 LOC | **paired, byte-equivalent** (MD5 `97b7e5d3…`) |
| `cvxportfolio_backend.py` | 344 LOC | 344 LOC | **paired, byte-equivalent** (MD5 `bd881e51…`) |
| `job_qp.py` | 528 LOC | 528 LOC | **paired, byte-equivalent** (MD5 `3e8def38…`) |
| `qp_solver.py` | 613 LOC | 613 LOC | **paired, byte-equivalent** (MD5 `0b9d22e5…`) |
| `signal_combiner.py` | 117 LOC | 117 LOC | **paired, byte-equivalent** (MD5 `902c80cb…`) |
| `task_joint_qp.py` | 55 LOC | 55 LOC | **paired, byte-equivalent** (MD5 `65faf1e6…`) |
| `tasks.py` | 3671 LOC | 3671 LOC | **paired, byte-equivalent** (MD5 `21e4a3a3…`) |
| `davis_norman.py` | 119 LOC | **MISSING** | **drift — subrepo-only mirror never landed** |
| `proportional_trade.py` | 101 LOC | **MISSING** | **drift — subrepo-only mirror never landed** |

Module-level symbol diff for all paired files: identical. Public API
surface (functions/classes that don't start with `_`) is unchanged across
the pair.

Import statements diff (paired files only):

```
baseline_allocators.py:
  from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot
  → from renquant_pipeline.kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot

job_qp.py:
  from kernel.pipeline.atoms import (...)         → from renquant_pipeline.kernel.pipeline.atoms import (...)
  from kernel.pipeline.context import ...         → from renquant_pipeline.kernel.pipeline.context import ...
  from kernel.pipeline.pipeline import Job, Task  → from renquant_pipeline.kernel.pipeline.pipeline import Job, Task
  from kernel.pipeline.task_benchmark_sleeve …    → from renquant_pipeline.kernel.pipeline.task_benchmark_sleeve …

task_joint_qp.py:
  from kernel.pipeline.context import InferenceContext → from renquant_pipeline.kernel.pipeline.context import InferenceContext
  from kernel.pipeline.pipeline import Task            → from renquant_pipeline.kernel.pipeline.pipeline import Task

tasks.py:
  from kernel.pipeline.atoms.ctx_ops import _get_path, _set_path
    → from renquant_pipeline.kernel.pipeline.atoms.ctx_ops import _get_path, _set_path
  from kernel.pipeline.context import InferenceContext → from renquant_pipeline.kernel.pipeline.context import InferenceContext
  from kernel.pipeline.order_attribution import stamp_order_attribution
    → from renquant_pipeline.kernel.pipeline.order_attribution import stamp_order_attribution
  from kernel.pipeline.pipeline import Task            → from renquant_pipeline.kernel.pipeline.pipeline import Task
```

Every diff is the EXPECTED `kernel.*` → `renquant_pipeline.kernel.*`
prefix swap per §3.5 — NOT drift.

`constraint_snapshot.py`, `cvxportfolio_backend.py`, `qp_solver.py`,
`signal_combiner.py` have ZERO import diffs (no cross-kernel imports).

---

## Functional API drift findings

### Finding 1 — `davis_norman.py` missing in subrepo (PRODUCTION-RISK)

Umbrella ships `backtesting/renquant_104/kernel/portfolio_qp/davis_norman.py`
(119 LOC) with public functions:

- `davis_norman_band(...)`
- `davis_norman_band_clamped(...)`
- `round_trip_to_one_way(...)`

The subrepo `tasks.py` references this module via lazy import inside
`_passes_no_trade_band()` (subrepo `tasks.py:2112`):

```python
if band_method == "davis_norman":
    from .davis_norman import davis_norman_band_clamped  # noqa: PLC0415
```

Subrepo path `src/renquant_pipeline/kernel/portfolio_qp/davis_norman.py`
does NOT exist. When the multirepo runtime takes that branch, it raises
`ModuleNotFoundError: No module named
'renquant_pipeline.kernel.portfolio_qp.davis_norman'`.

**Production config status**:

- `strategy_config.golden.json:732` → `"qp_band_method": "davis_norman"`
- `strategy_config.json:732` → `"qp_band_method": "davis_norman"`
- `strategy_config.shadow.json:638` → `"qp_band_method": "davis_norman"`

All three live configs opt into the DN path. The default runner is the
multirepo one (`scripts/daily_104.sh:71,283,484-487` — `RQ_DAILY_RUNNER`
defaults to `multirepo`; shadow uses `scripts/live_multirepo.py`).

**What happens at runtime — actually, today** (per
`memory/project_subrepo_runtime_vendored_snapshot_2026-06-01.md` + this
audit's investigation of `logs/daily_104/2026-06-02_shadow.log`):

The bootstrap resolves `renquant_pipeline` not from the live sibling
clone but from a vendored snapshot at
`.subrepo_runtime/repos/renquant-pipeline/`. That snapshot was last
refreshed BEFORE the 2026-05-30 portfolio_qp campaign — the snapshot's
`portfolio_qp/` directory only contains 6 source files
(`__init__`, `cvxportfolio_backend`, `job_qp`, `qp_solver`,
`signal_combiner`, `task_joint_qp`, `tasks`) and the snapshot's
`tasks.py` is 3419 LOC vs subrepo-main's 3671 LOC.

The snapshot's `tasks.py` predates the DN call site entirely — it has
NO `from .davis_norman import` lazy import. The hardcoded
"min_dw=%.2f%%, factor=%.1fσ — Davis-Norman" log string that the
shadow log emits is there in the snapshot too, but it's a misleading
status label: today's production runtime SILENTLY IGNORES
`qp_band_method=davis_norman` config and always runs the legacy band.

So the current operational state is a TWO-LAYER hazard:

1. **Layer A (vendored snapshot stale)**: Production today runs a
   pre-2026-05-30 portfolio_qp tree. Every config knob added in the
   2026-05-30 campaign (DN, GP partial trade, constraint snapshot
   path, baseline allocators) is silently no-op in the deployed
   runtime. The 2026-05-30 Bug F shadow fix that motivated DN in the
   first place is NOT in production today.
2. **Layer B (sibling subrepo broken)**: The moment a future
   `make subrepo-runtime-root` refresh pulls subrepo `main` into
   `.subrepo_runtime/`, the runtime gains the DN+PT lazy imports
   without the modules — at that point QP solves crash with
   `ModuleNotFoundError`. Refreshing the vendored snapshot WITHOUT
   first mirroring DN/PT is a known-broken state.

Either layer alone is §7.7 "safety gate ≠ enforced safety gate" — the
DN gate is decoration, the umbrella module that implements it is
unreachable.

Origin: umbrella commit `87773e6` (2026-05-30) added `davis_norman.py`
+ enabled it in golden config. No paired subrepo mirror PR was ever
opened. Recent subrepo mirror PRs #19-22 covered Steps 1a/1b/1c +
baseline allocators, but skipped this leaf.

### Finding 2 — `proportional_trade.py` missing in subrepo (PRODUCTION-RISK)

Umbrella ships
`backtesting/renquant_104/kernel/portfolio_qp/proportional_trade.py`
(101 LOC) with public functions:

- `proportional_trade_target(...)`
- `resolve_trade_horizon_days(...)`

Subrepo `tasks.py` references via lazy import inside `ProportionalTradeTask.run`
(subrepo `tasks.py:2688-2691`):

```python
from .proportional_trade import (  # noqa: PLC0415
    proportional_trade_target,
    resolve_trade_horizon_days,
)
```

Subrepo path
`src/renquant_pipeline/kernel/portfolio_qp/proportional_trade.py` does
NOT exist.

**Production status of that branch**: same TWO-LAYER hazard as
Finding 1.

- Layer A (today, vendored snapshot): the snapshot's `job_qp.py` does
  NOT instantiate `ApplyProportionalTradeTask` (no such class in the
  snapshot — verified by listing snapshot's portfolio_qp files: no
  `proportional_trade.py`, and the snapshot's `tasks.py` predates the
  class). The task is currently absent from production runtime.
- Layer B (subrepo-main, after the next `.subrepo_runtime/` refresh):
  subrepo's `job_qp.py:48,502` imports + instantiates
  `ApplyProportionalTradeTask`. Inside `ApplyProportionalTradeTask.run`
  (subrepo `tasks.py:2686-2691`), the lazy `from .proportional_trade
  import (...)` runs BEFORE any horizon-gate short-circuit. Every QP
  solve → `run()` → `ModuleNotFoundError: No module named
  'renquant_pipeline.kernel.portfolio_qp.proportional_trade'`.

The import on line 2688 fires unconditionally before the
`n_days <= 1.0` legacy-passthrough on line 2708 is even evaluated, so
the absence of a >1d-horizon config knob today does NOT mask the bug
post-snapshot-refresh.

Origin: umbrella commit `bfc08b9` (2026-05-30) added
`proportional_trade.py`. No paired subrepo mirror PR was ever opened.

### Finding 3 — Vendored runtime snapshot stale by ≥4 days (PRODUCTION-RISK, distinct from above)

This finding is independent of Findings 1+2 but came out of the same
investigation. The umbrella's `.subrepo_runtime/repos/renquant-pipeline`
snapshot (the path the multirepo bootstrap actually loads, per
`memory/project_subrepo_runtime_vendored_snapshot_2026-06-01.md`) is
behind subrepo-main:

| Subrepo file (under `portfolio_qp/`) | Vendored snapshot | Subrepo `main` HEAD |
|---|---|---|
| `__init__.py` | present | present |
| `cvxportfolio_backend.py` | present | present |
| `job_qp.py` | present (older) | present (528 LOC, ApplyProportionalTradeTask + JointPortfolioQPJob refactor) |
| `qp_solver.py` | present (older) | present (613 LOC, `solve_portfolio_qp_from_snapshot`) |
| `signal_combiner.py` | present | present |
| `task_joint_qp.py` | present | present |
| `tasks.py` | 3419 LOC (older) | 3671 LOC (DN + GP + BuildConstraintSnapshotTask + baseline allocators) |
| `baseline_allocators.py` | **MISSING** | present (subrepo PR #22) |
| `constraint_snapshot.py` | **MISSING** | present (subrepo PR #19) |
| `davis_norman.py` | **MISSING** (also missing from subrepo) | **MISSING** |
| `proportional_trade.py` | **MISSING** (also missing from subrepo) | **MISSING** |

Subrepo-main merged PRs #19-22 today (2026-06-03 mirror chain). The
vendored runtime root has NOT been refreshed, so production today is
running pre-#19 portfolio_qp code.

Operational implications:
- The 2026-05-30 Bug F shadow fix (DN @ 0.5% floor instead of legacy
  2% min_dw) is NOT in production. Shadow's ORCL Δw=0.88% case
  rationale on `_qp_band_method_note_2026-05-30` does not match
  deployed code.
- `solve_portfolio_qp_from_snapshot` (subrepo PR #20) is NOT in
  deployed code; the deployed QP solver path is pre-snapshot-API.
- `BuildConstraintSnapshotTask` (subrepo PR #21, umbrella #129) is
  NOT in the deployed QP job chain.
- The 3 baseline allocators (subrepo PR #22, umbrella #130) are NOT
  available in deployed runtime.

This is the SAME bug class as the §3.5 paired-PR mandate but at the
deployment-snapshot layer rather than the source-tree layer.

### Finding 4 — Test mirror drift

Umbrella has 3 tests pinning the DN/PT modules that have no subrepo
counterparts:

- `tests/test_davis_norman_band.py`
- `tests/test_proportional_trade.py`
- `tests/test_passes_no_trade_band_dn.py`

Subrepo `tests/` carries `test_baseline_allocators.py`,
`test_build_constraint_snapshot_task.py`, `test_constraint_snapshot.py`,
`test_lift_qp_joint.py`, `test_qp_cap_compliance_fallback.py`,
`test_qp_failure_counters.py`. The 3 DN+PT tests should mirror over
together with the source modules. Per §7.1 every fix has at least one
test through the real adapter — those 3 umbrella tests pin the only
guard rail that would have caught Findings 1 and 2.

---

## Action Items

Original source-tree action items #1 and #2 are now closed by follow-up
subrepo mirror PRs. The remaining work is deployment and regression
guarding:

1. **Refresh `.subrepo_runtime/repos/renquant-pipeline/` vendored snapshot**
   (umbrella; `make subrepo-runtime-root` per the memory note).
   - **Priority**: P0 — production is still running pre-2026-05-30
     portfolio_qp code, silently no-oping DN config + the QP architecture
     refactor.
   - Verify post-refresh with a paper/shadow daily run. Grep logs for the
     Davis-Norman path, `BuildConstraintSnapshotTask`, and absence of
     `ModuleNotFoundError`.

2. **Drift regression guard test**
   - Add a test asserting that the set of `.py` files under
     `backtesting/renquant_104/kernel/portfolio_qp/` matches
     `src/renquant_pipeline/kernel/portfolio_qp/` (paired byte-mirror
     invariant). Either a §7.6 integration test in the umbrella repo,
     or a CI workflow diffing the two trees on every PR.
   - Existing precedent: `tests/test_c211_panel_pipeline_lift.py` MD5
     pinning style — extend or add a portfolio_qp counterpart.
   - Same guard would have caught the entire Finding 1+2 class.

3. **§7.6 hardening — fail-fast import at module level (optional)**
   - Promote the two lazy imports in `tasks.py` (subrepo + umbrella) to
     module-level imports. Today they are lazy (`# noqa: PLC0415`); if
     the helper modules exist there's no circular-import cost. Module-
     level imports would catch the missing-mirror class of bug at
     import time instead of latent under a specific config branch.

4. **Audit other umbrella-only `kernel/*` leaves** (out of scope for
   this audit, but recommended follow-up).
   - The pattern that produced Findings 1 + 2 (umbrella-only leaf
     referenced from a subrepo-mirrored caller) probably repeats
     elsewhere. A grep `grep -rn "from \." subrepo/kernel/` cross-
     referenced against subrepo file presence should find any other
     dangling lazy imports.

---

## Conclusion

8 of 10 paired portfolio_qp source files are byte-equivalent (after the
§3.5-blessed import prefix swap), confirming the recent multi-PR refactor
campaign held the mirror invariant on the files it touched.

At the time of the original audit, three independent drift hazards
remained:

- **Source-tree drift** (Findings 1 + 2): two 2026-05-30 umbrella-only
  leaf modules (`davis_norman.py`, `proportional_trade.py`) are
  referenced by subrepo `tasks.py` lazy imports but never mirrored
  into the subrepo. The subrepo-main runtime, if ever loaded directly,
  would hit `ModuleNotFoundError` on every QP solve.
- **Vendored snapshot drift** (Finding 3): the currently-deployed
  `.subrepo_runtime/repos/renquant-pipeline/` is pre-2026-05-30 and
  silently no-ops the DN config + the entire 2026-05-30 QP refactor
  campaign. The Bug F shadow fix that motivated DN is not deployed.
- **Test mirror drift** (Finding 4): the 3 umbrella tests pinning
  DN+PT have no subrepo counterparts, so neither the byte-equivalence
  mirror invariant nor §7.1 paired-test mandate is satisfied.

As of the post-audit update above, the source-tree and paired-test
hazards are closed. The vendored runtime snapshot drift remains and must
be closed by runtime refresh + smoke before production cron relies on
the QP refactor.

The paired-PR mandate (§3.5) only holds if every umbrella `kernel/`
change ships its subrepo counterpart on the same day — these two
landed 4 days ago and have been silently drifting since.

---

*Audit complete. Memo is docs-only; no source code modified. Follow-up
deployment work should prioritize the vendored runtime refresh and
runtime sanity guard.*
