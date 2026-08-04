# 2026-08-04 — pin advance 3: strategy-104 → 547fc49b (blend component pin rotation)

Carries s104#87: both blend shadow profiles rotate `component[0]
expected_content_sha256` from the pre-promotion `04d7a381…` to the promoted
scorer `6461b827ab2339a8` (trained 2026-08-02). Root cause and the standing
consumer-checklist debt are recorded in s104#87's progress doc and
renquant-orchestrator#793.

- `subrepos.lock.json`: renquant-strategy-104 `d84604d7` → `547fc49b` (sha
  read back from the merge API output)
- `doc/arch/strategy-104-snapshot.md`: re-rendered against the fakeroot with
  its s104 checkout at the new pin and live-refreshed artifact mirrors
  (9-line diff: pin row, blend-profile source hashes, fingerprint)

Deploy after merge (deploy batch 3, grants-logged): live pull + runtime s104
sync → manual Step 5/5b rerun → both blend lanes' first REAL decision records
delivered to the operator tonight.

## Round 2 (codex): the committed artifact tree must carry the promoted pair

Codex correctly rejected round 1: the candidate-pin gate compares the s104
profile pin against the COMMITTED production artifact, which was still the
June `04d7a381…` bytes — the 11:31 promotion existed only as a working-tree
mutation on the run machine, so the candidate assembly could not see it.
That is a real multi-repo handoff failure, not CI cosmetics.

Fix in this round: commit the promoted pair itself as the auditable evidence —
`panel-ltr.alpha158_fund.json` (`6461b827ab2339a8`, `trained_date=2026-08-02`,
`metadata.promotion_basis=freshness_fallback_rfc210` — the stamp IS the
provenance) and `panel-rank-calibration.json` (`bce257d19a3ddb54`, expecting
the runtime-legacy scorer fp `d7bddf2a…`, verified matching
`PanelScorer.load` on the live pair). Committed tree, profile pin, and the
served runtime now agree on one identity.

Standing consequence appended to orch#793: every RFC#210 promotion must
COMMIT the swapped pair in the same consumer batch (or A9 finally moves
artifacts out of git) — a working-tree-only promotion is invisible to every
committed-tree consumer by construction.
