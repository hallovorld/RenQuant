# 2026-08-04 — pin advance: strategy-104 → c8bba9c9 (RCS clf component kind)

Carries s104#91: the F1/F3 clf component's `kind` moves out of the
shadow-model vocabulary (`xgb`) into the blend-component one (omitted =
loader default `panel`). Measured cause: RCS's first execution fail-closed
with `declares unknown kind 'xgb'`, clearing 83 candidates; the guard added
with the fix walks every blend profile (prod included) so the two vocabularies
cannot be conflated again.

Deploy after merge un-blocks RCS: it has both components (prod scorer + slow
momentum ledger + clf leg) and should produce its first REAL 3-component
decision at the next daily. Rf/RCf stay dormant until the 2026-08-08 fast
genesis (playbook orch#795).
