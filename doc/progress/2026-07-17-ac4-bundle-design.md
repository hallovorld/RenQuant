# Progress: AC4 artifact-bundle transactionality design (RFC)

STATUS: delivered (design only — no implementation)
WHAT: RFC for transactional serving-pair bundles (manifest + atomic pointer
swap + single writer protocol) replacing four independent per-file writers.
See doc/design/2026-07-17-artifact-bundle-transactionality.md.
WHY/DIR: GOAL-5 P0 AC4 — the 4x-recurring calibrator/scorer pair-orphaning
class (latest: 07-14→16 incident) is structural under per-file mutation;
pair-level atomicity is the fix. Drafted personally per design-review policy.
EVIDENCE: n/a (design; verification plan = kill-injection + concurrency +
incident-replay + live drill, in the doc §4).
NEXT: codex adversarial review; implementation only after design approval
(P0→P3 staged, default-ON at the end, no dark shipping).
