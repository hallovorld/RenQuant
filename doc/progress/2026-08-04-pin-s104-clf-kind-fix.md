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

## Round 2 (codex): the cross-repo ops contract still pinned the UNSAFE text

The pin gate was red for a real contract mismatch, not a cosmetic one:
`scripts/subrepo_ops_contract.py` required
`renquant_strategy_config "$SUBREPO_ROOT" "$cfg_name"` to appear in
`weekly_wf_promote.sh` — the exact call RenQuant#580 deliberately REMOVED
(it resolves through `renquant_subrepo_root`, which defaults to the sibling
developer checkout absent an assembly override, and its fallback chain ended
at the umbrella working copy).

So the contract was enforcing the unsafe behaviour. Updated to assert the SAFE
one instead, with the incident recorded at the point of enforcement:

- required: `candidates=("$pinned_path")`, the `.subrepo_runtime` pinned-path
  assignment, the `no PINNED strategy config declares kind=xgb` refusal text,
  and the `WEEKLY-BLOCKED` alert;
- forbidden (NEW): `candidates=("$multirepo_path"…` and
  `candidates=("$pinned_path" "$workingcopy_path")` — the two unpinned
  reference sources are now banned BY NAME, so re-introducing either fails the
  contract instead of silently recreating the phantom-config incident.

Contract run: `ok: true`, 0 failures.
