# 2026-07-18 - Correct #502: complete the shadow-evidence tuple

## Problem

RenQuant #502 advanced the pipeline and strategy-104 pins for the small-n
guard shadow evidence, but left `renquant-orchestrator` pinned before #549.
That omitted the required `smalln_guard_suppressed` sentinel and bridge-bundle
eligibility-ledger write from the materialized umbrella assembly.

The #502 snapshot also could not be regenerated from its reviewed tree: it
recorded calibrator fingerprints from local bytes that were not the tracked
`main` artifact. The canonical snapshot must be a deterministic declaration of
the reviewed tree plus its pinned sources.

## Correction

- Advance only the `renquant-orchestrator` lock entry to
  `8c0acd5f58ce54baa7559a057d6dfc22164e3f8b`, the verified merge containing
  #549.
- Regenerate `doc/arch/strategy-104-snapshot.md` from a clean worktree whose
  `.subrepo_runtime/repos` assembly is materialized from the resulting lock.

The tuple after this change is:

| Responsibility | Pinned component |
|---|---|
| Eligibility classification | renquant-pipeline `d32f7017` (#208) |
| Shadow-only guard keys | renquant-strategy-104 `082dccd2` (#61) |
| Sentinel and bundle evidence | renquant-orchestrator `8c0acd5f` (#549) |

## Scope and safety

This is not a production activation. Production and golden strategy configs
remain free of the guard keys. It merely makes the already intended shadow
evidence path a complete, auditable pinned assembly. Operator landing and
shadow-epoch start still require their separate ask-first procedure and the
frozen RFC #204 section 4 verdict; no capital authorization is created here.

## Verification

- `make subrepo-runtime-root`
- `make snapshot`
- `make snapshot-check`
- `python3 scripts/render_strategy_104_snapshot.py --verify-pinned-declaration`
- `make subrepo-pin-ci-green`
- `git diff --check`

All commands were run in an isolated clean worktree. The new snapshot's
calibrator fingerprints are the values produced by the tracked artifact in
that worktree, not copied from the non-reproducible #502 declaration.
