# 2026-08-02 — Grant C step (e): second pin advance — the momentum shadow lane goes live

STATUS: complete (lock + snapshot in this PR; machine sync follows the merge
under the standing Grant C authorization)

WHAT: `subrepos.lock.json` advances renquant-strategy-104
`3bfd5abc` → `ce8ad100` (s104#77: the `momentum_residual_v0_shadow`
ledger-pointer entry, merged after the corrected-order gates) and
renquant-pipeline `60871e24` → `dff3cbe3` (pipeline#255: certified-ledger
disappearance is a named load fault). Both pins passed the refresh tool's
CI-green gate. `doc/arch/strategy-104-snapshot.md` regenerated from the new
s104 pin via the pin-aligned mirror assembly (same method as #551): the
machine block now declares the momentum shadow lane (kind
`momentum_residual`, ledger pointer, ledger sha `9aa2d8c9…`).

WHY/DIR: GOAL-7 slice 5, final step of the corrected batch order (orch#759
record): gates #255/#554/#761 merged → step (c) re-executed (job installed,
verified loaded, Sat 05:00) → s104#77 merged → THIS pin advance makes the
serving config reach the machine at the next granted sync.

EVIDENCE:
- artifact: this PR's diff; local proof: `--verify-pinned-declaration` OK,
  mirror `--check` exit 0 (committed snapshot byte-matches the post-sync live
  render), parity pre-check vs pipeline `dff3cbe3`: zero NEW kernel drift
- prod or exp: reviewed surfaces only; the machine sync happens post-merge
  under the standing Grant C authorization with the documented reverts
- existing data: the pointed-at ledger exists on disk (genesis row,
  `artifact_content_sha256 a824c480cd9c…`, 144/144 names)
- best-known?: yes — every check above executed, not asserted
- scope: two pins + the regenerated snapshot; no other lock entry moves

NEXT: post-merge machine sync (umbrella ff-pull + assemble → runtime at the
new pins; snapshot `--check` re-verified on the live tree) → the daily run
serves the momentum lane in shadow (sentinel watching per orch#761) →
follow-ups: retire the s104 `_2026_08_02_pending_first_artifact` narrative
key on the next routine s104 change; orchestrator PENDING-marker cleanup PR
(same three surfaces as the reverted #759 round 1, now legitimate).
