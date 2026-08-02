# 2026-08-02 — kernel-parity resolver: refuse trees not at the locked commit

STATUS: complete (fix + 5 unit tests + live-machine end-to-end read; round 2: the env override leg is verified too)

WHAT: `scripts/check_kernel_parity.py::_resolve_pipeline_kernel` trusted the
lock's `local_path` (a mutable developer checkout) without verifying its HEAD,
so a machine whose sibling checkout lagged the pin silently measured a stale
tree. Hardened: every non-override candidate must have HEAD == the locked
pipeline commit; the pin-materialised runtime clone
(`.subrepo_runtime/repos/renquant-pipeline`) is preferred; with no matching
candidate the resolver returns None → exit 3 (the honest skip), never a
wrong-object measurement. Wrong-commit candidates are named on stderr.

WHY/DIR: GOAL-3 (guards-that-validate-the-wrong-object class). Measured
instance, this machine, 2026-08-02: sibling checkout at `a14dad11` vs locked
`60871e24` — the stale tree read `portfolio.py` and
`walk_forward/leakage_guard.py` as CONVERGED while both are genuinely drifted
vs the pin (0/93 allowlist entries are actually converged). A local
`make doctor`-class run would have reported phantom convergence and could
equally hide real new drift.

EVIDENCE:
- artifact: this PR's diff; tests/test_kernel_parity.py (3 new resolver units:
  accepts-at-pin, refuses-wrong-commit with stderr message, prefers-runtime);
  live read-back below
- prod or exp: script + tests only; no production surface touched (the live
  probe was import-and-read, no writes)
- existing data: live-machine end-to-end after the fix: resolver →
  `.subrepo_runtime/repos/renquant-pipeline/.../kernel`; check_parity →
  common 169 / identical 76 / drifted_allowed 93 / drifted_new 0 — matching
  the kernel-parity CI numbers on RenQuant#551
- best-known?: yes — both the defect and the fix were measured on the machine,
  not asserted
- scope: resolver only; comparison logic, allowlist, exit codes unchanged
  (exit 3 semantics now also covers "candidates exist but none at the pin")

Round 2 (codex review): `RENQUANT_PIPELINE_KERNEL_PATH` no longer bypasses the
pin check. `git -C <override-kernel> rev-parse HEAD` walks up to the enclosing
checkout (no layout assumption) and must equal the locked commit; a
wrong-commit override is refused with a named stderr message. No CI carve-out
is needed: kernel-parity-ci reads the pin from subrepos.lock.json and checks
renquant-pipeline out AT that commit, so the verification passes there by
construction. +2 tests (refuses-wrong-commit-override, accepts-CI-shape).

NEXT: none for this fix. Standing GOAL-3 measurement recorded here: allowlist
rot = 0/93 removable as of the 60871e24 pin (honest negative — no cleanup PR
warranted yet).
