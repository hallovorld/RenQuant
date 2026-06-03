# Subrepo Runtime Refresh — Runbook

**Audience**: anyone on call when `scripts/runtime_qp_sanity_check.py`
exits non-zero, or when `scripts/daily_104.sh` aborts with
`RUNTIME-SANITY-FAIL`.

**Purpose**: refresh the vendored multirepo runtime under
`.subrepo_runtime/repos/` after subrepo PRs merge, so the daily runner
imports the post-2026-05-30 portfolio_qp code paths instead of silently
falling back to legacy.

**Source incidents**:
- PR #145 — runtime drift critical memo
  ([`doc/ops/2026-06-03-prod-runtime-qpmultirepo-refresh.md`](2026-06-03-prod-runtime-qpmultirepo-refresh.md))
- PR #147 — `feat(ops): fail daily run on stale QP multirepo runtime`
- PR #149 / #150 — `feat(ops): include QP runtime sanity in deployment readiness`
- PR #148 / #152 / #153 — Step 4g runtime + gate pins

---

## 1. The failure modes this runbook covers

The vendored snapshot at `.subrepo_runtime/repos/renquant-pipeline/`
can drift behind `origin/main` of the subrepo. When it does, the
`renquant_pipeline.kernel.portfolio_qp.*` modules the prod runtime
imports may:

| Symptom | Failure mode |
|---|---|
| `runtime_qp_sanity_check.py` exits 1 | One or more required QP symbols missing under `RENQUANT_SUBREPO_ROOT` |
| `daily_104.sh` aborts with `RUNTIME-SANITY-FAIL` | Daily wrapper ran the sanity check and saw stale runtime; daily cycle halted before any live order |
| `make ops-deployment-ready` shows `runtime_qp_sanity_ok: false` | Pre-deploy readiness check caught the drift before launchagent install |
| Symbol loaded from umbrella tree instead of `.subrepo_runtime/repos/...` | `_origin_under_runtime` mismatch — pipeline imports resolved via the legacy umbrella fallback, not the vendored snapshot |

Per §3.5, the umbrella tree is a byte-equivalent mirror for now. The
daily runner is supposed to import through the vendored snapshot.
Falling back to umbrella means `RENQUANT_SUBREPO_ROOT` is unset or the
snapshot does not contain the required module.

## 2. Recovery — happy path

Run from the umbrella repo root (`/Users/renhao/git/github/RenQuant`).

```bash
# 1. Sync umbrella + subrepos (§3.2).
git fetch origin
git checkout main && git pull --ff-only origin main

# 2. Refresh the vendored snapshot to the pinned commits in
#    subrepos.lock.json. Writes:
#      - .subrepo_runtime/repos/<repo>/ — git checkout of each pinned ref
#      - .subrepo_assembly/<timestamp>/env.sh
#      - .subrepo_assembly/current.env  ← symlink the daily runner sources
make subrepo-runtime-root

# 3. Re-run the sanity check against the refreshed snapshot.
make subrepo-runtime-sanity
#   ≡  .venv/bin/python scripts/runtime_qp_sanity_check.py

# 4. Confirm full deployment readiness (also runs the sanity check
#    inside RENQUANT_SUBREPO_ROOT, plus path / strict-mode env checks).
make ops-deployment-ready

# 5. Paper-broker smoke before re-arming prod cron.
bash scripts/daily_104.sh --broker alpaca-paper

# 6. Commit the refreshed lock + assembly artifacts (if any changed).
git status --short
git add subrepos.lock.json .subrepo_assembly/current.env  # if changed
git commit -m "chore(ops): refresh vendored subrepo runtime"
# Per §3.1, ship through a PR — never push directly to main.
```

## 3. Recovery — when the lock itself is stale

If `make subrepo-runtime-root` checks out a still-pre-2026-05-30
commit, the lockfile pins are themselves behind subrepo `main`.
Advance them first:

```bash
# 1. Refresh subrepo pins to match each subrepo's origin/main.
.venv/bin/python scripts/refresh_subrepo_lock.py --execute

# 2. Confirm CI pin guard still green.
make subrepo-pin-ci-green

# 3. Then re-run §2 from step 2.
make subrepo-runtime-root
make subrepo-runtime-sanity
```

Open a PR for the `subrepos.lock.json` change. The CI freshness check
(`scripts/check_lock_pins_ci_green.py`) blocks merges that point at
non-existent commits or stale refs.

## 4. Required symbols the sanity check verifies

Authoritative list lives in `scripts/runtime_qp_sanity_check.py`
(`REQUIRED_SYMBOLS`). At time of writing:

| Module | Symbol | Repo | Provenance |
|---|---|---|---|
| `renquant_pipeline.kernel.portfolio_qp.davis_norman` | `davis_norman_band_clamped` | `renquant-pipeline` | Davis-Norman no-trade band path |
| `renquant_pipeline.kernel.portfolio_qp.proportional_trade` | `proportional_trade_target` | `renquant-pipeline` | Partial-horizon proportional trade |
| `renquant_pipeline.kernel.portfolio_qp.constraint_snapshot` | `ConstraintSnapshot` | `renquant-pipeline` | Hard-constraint snapshot contract (PR #126) |
| `renquant_pipeline.kernel.portfolio_qp.qp_solver` | `solve_portfolio_qp_from_snapshot` | `renquant-pipeline` | Snapshot-based solver entry (PR #127) |
| `renquant_pipeline.kernel.portfolio_qp.baseline_allocators` | `hybrid_option_f_allocator` | `renquant-pipeline` | Hybrid Option F A/B candidate (PR #146) |
| `renquant_pipeline.kernel.portfolio_qp.baseline_allocators` | `hard_only_qp_allocator` | `renquant-pipeline` | Hard-only QP A/B baseline (PR #135) |
| `renquant_pipeline.kernel.portfolio_qp.allocator_replay` | `replay_all` | `renquant-pipeline` | Paired offline replay harness (PR #131) |
| `renquant_pipeline.kernel.portfolio_qp.replay_significance` | `compute_significance_verdicts` | `renquant-pipeline` | DSR / PBO correction (PR #132) |

Each new portfolio_qp module that prod depends on adds a row here.
Adding a `RuntimeSymbol` entry is the canonical way to wire a tripwire
for future drift.

## 5. What you must NOT do

| Anti-pattern | Why it bites |
|---|---|
| `export RQ_DAILY_RUNNER=umbrella` to silence the sanity check | The wrapper only invokes the sanity check in the multirepo path. Flipping back to umbrella imports legacy code — the exact drift we are guarding against. Reserve for emergency triage only, and never leave it persistent. |
| `git checkout` inside `.subrepo_runtime/repos/<repo>/` by hand | Skips the assembly env writer; `RENQUANT_SUBREPO_ROOT` + `current.env` keep pointing at the old timestamped assembly. Use `make subrepo-runtime-root`. |
| Commit changes inside `.subrepo_runtime/` | The directory is rebuilt by `subrepo_assemble.py --sync`; manual edits are blown away on the next refresh. |
| Skip the paper smoke (§2 step 5) | A successful sanity check confirms imports resolve; only the smoke confirms the new code paths actually execute on a real bar. Davis-Norman-band log line is the canary. |

## 6. Cross-references

- Drift incident analysis: [`doc/ops/2026-06-03-prod-runtime-qpmultirepo-refresh.md`](2026-06-03-prod-runtime-qpmultirepo-refresh.md)
- Drift audit (umbrella vs pipeline): [`doc/research/evidence/2026-06-03-portfolio-qp-subrepo-drift-audit.md`](../research/evidence/2026-06-03-portfolio-qp-subrepo-drift-audit.md)
- Subrepo operating model: [`doc/arch/subrepo-operating-model.md`](../arch/subrepo-operating-model.md)
- Multirepo SOP: [`doc/arch/multirepo-sop.md`](../arch/multirepo-sop.md)
- Daily wrapper sanity gate: `scripts/daily_104.sh` (search `RUNTIME-SANITY-FAIL`)
- Deployment-readiness gate: `scripts/check_ops_deployment_ready.py::_run_runtime_qp_sanity`
- Strict-mode env: `RENQUANT_STRICT_SUBREPO_PATHS=1`, `RENQUANT_OPS_FAIL_CLOSED=1` (set by `make subrepo-runtime-root` into `.subrepo_assembly/current.env`; readiness check fails if missing)
