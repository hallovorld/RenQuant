# Progress — Umbrella 104/105 design-compliance audit (findings memo)

Date: 2026-07-04
Type: docs-only audit memo (no code changes)
Deliverable: `doc/arch/2026-07-04-umbrella-compliance-audit.md`

## What was done

Deep design-compliance audit of the umbrella repo (`backtesting/renquant_104/**`,
`scripts/**`, `live/**`) against the agreed charter:
`subrepo-operating-model.md` Universal Rules 1-6, `multirepo-sop.md` §4
placement, `kernel-inventory.md`, CLAUDE.md §3.5 mirror invariant, and the
session-established rules (kernel-alias on live paths, single-impl-imports-only
for fingerprints, flags default-OFF, no selection logic in umbrella scripts).

Method: fresh blob-less clones of the umbrella (`main` @ 79d47da3) and the
pinned `renquant-pipeline` (778983ab per `subrepos.lock.json`) in an isolated
scratchpad; mechanical kernel diff with import-path normalization; entrypoint
tracing of daily/sim/WF-gate/promote paths; targeted greps for duplicated
fingerprint/calendar/eps implementations, dead code, and tracked large files.
No git operations were performed in any primary checkout or the live tree.

## Headline results

30 findings: 6 P0 / 19 P1 / 5 P2 (memo §0 has the ranked table). Key
structural findings:

- Sim + WF gate/promote run the umbrella kernel; live runs the pinned pipeline
  kernel; 78/169 shared kernel files have drifted (bidirectionally) — the
  promote gate evaluates candidates on code live will not run.
- A known silent-mis-score scorer fix (2026-06-15) and the 2026-07-01
  shadow-ntfy feature exist only on the umbrella side and are shadowed by the
  kernel alias on live paths.
- Promote-side and live-side WF/fingerprint stamp verification are two
  different implementations (umbrella loader + `manifest_uri_resolver` vs
  pipeline loader + `fingerprint_dispatch`).
- CLAUDE.md §3.5's byte-equivalence invariant cites a parity test that does not
  exist; no CI enforces the mirror.

## Follow-ups (not in this PR)

Fix work is deliberately excluded; each finding in the memo names the owning
repo and a one-line proposed fix. Highest-priority follow-ups: port the two
stranded umbrella-side fixes to renquant-pipeline (F-1, F-5), unify WF-stamp
verification behind `renquant_common.model_fingerprint` before any
`accept_legacy_stamps` flip (F-2), and route sim/WF-gate entrypoints through
the same pinned-kernel bootstrap as live (F-3).
