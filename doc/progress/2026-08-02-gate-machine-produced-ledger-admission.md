# 2026-08-02 — gate: machine-produced-ledger admission for the momentum entry

STATUS: complete (gate change + 3 regressions; the s104 marker PR and the
#556 re-pin follow)

WHAT: `_check_ledger_pointer`'s ABSENT branch admits a
`*_machine_produced_ledger` narrative key as INFO — the momentum ledger is
run-surface state published by the weekly job and never committed, so hosted
runners cannot resolve it BY DESIGN. The admission sits INSIDE #554's
momentum-contract narrowing (kind + exact path fire first), is inert when
the ledger resolves (full chain verification unchanged — regression proves a
tampered chain still fails with the marker present), and the fail-closed
default for unmarked absence stays (message now names both markers).

WHY/DIR: RenQuant#556's verify-pinned-paths red, measured: s104#77 only ever
passed hosted CI because the pending-first-artifact marker's absent-branch
admission happened to cover it; s104#78 correctly retired that marker as
semantically false ("not published anywhere" ≠ "published on the serving
machine, invisible to CI"), leaving every future pin bump carrying the
momentum entry CI-red. This gives the true state its own named, bounded
admission. Alternative considered and rejected for now: committing the
weekly publish set to the umbrella (would need automated repo writes from
the train job — a run-surface design change, not a gate patch).

EVIDENCE:
- artifact: this PR's diff; gate tests 36 passed, 1 skipped (+3: absent+marker
  → INFO; marker cannot rescue a non-momentum `.jsonl`; marker inert when the
  ledger resolves — tampered chain still FAILS)
- prod or exp: gate script + tests only
- existing data: the #556 CI log (1/22 fail, exactly the momentum ledger on
  the hosted runner)
- best-known?: yes — the failure and the fix are both mechanically reproduced
  in the regressions
- scope: the absent branch only; resolution, #554 narrowing, chain
  verification, pending-marker semantics unchanged

NEXT: s104 PR adds the marker key to the momentum entry (bounded-set test
idiom) → #556 re-pins to that tip + snapshot regen → machine sync.
