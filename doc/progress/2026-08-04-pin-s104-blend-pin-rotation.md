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
