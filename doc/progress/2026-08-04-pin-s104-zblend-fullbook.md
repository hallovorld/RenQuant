# 2026-08-04 — pin advance 4: strategy-104 → 2358b56b (OPERATOR OVERRIDE: z-blend full book)

Carries s104#88: prod primary scorer → zblend(reversal + slow momentum), full
book, under the operator's explicit repeated directive and blast-radius choice
(整本切换). Authority, zero-OOS disclosure, rollback, and review condition
live in `configs/zblend_prod_artifact_manifest.json` (s104) and the s104
progress doc; GOAL-9 registration orch#794.

- `subrepos.lock.json`: renquant-strategy-104 `547fc49b` → `2358b56b` (sha
  read back from merge output)
- snapshot re-rendered at the new pin (blend primary now visible in the
  production table)

Pre-merge verification already on record (s104#88): suite 97 passed + readonly
FULL-FUNNEL sim on the exact config (rc=0, zero hard fails, governance PASS,
GOOG+VLO decisions, pending orders excluded).

Deploy after merge = batch 5 (grants-logged): live pull + runtime s104 sync.
Tomorrow 13:55 PT = the first full-book z-blend run.

## Round 2 (codex): the candidate tree proves the ledger leg — gate learns ledger-served primary

Codex correctly blocked round 1: components[1]'s momentum ledger is a
runtime-generated surface, absent from the committed candidate tree. Fixes,
none of them a waiver:

1. **The ledger + genesis artifact are now COMMITTED** (72 KB:
   `momentum_artifact_ledger.jsonl`, 1 chained row, + `2026-08-02/
   momentum_residual_v0.json`) — same resolution shape as the promoted-pair
   commit in #572; the weekly publish appends on the serving machine and each
   Saturday batch must commit the appended row+artifact (#793 checklist).
2. **The gate learns the ledger-served PRIMARY shape** — the ledger branch
   (previously shadow-only) now admits `kind in ("shadow","primary")`, still
   restricted to the #550 momentum contract (declared kind + exact path).
   `_expected` threads the entry's declared kind through (it was dropped,
   which is why the entry read as kind=None).
3. **Recipe-fp validation, single-source**: `expected_config_fingerprint` on
   a ledger component is now VALIDATED against the tail artifact's params via
   the pinned pipeline's own `_params_fingerprint` (imported, never
   vendored; unavailable import fails closed). Byte pins
   (`expected_content_sha256`) stay refused on the append-only surface — the
   round-1 refusal conflated the two; the serving loader REQUIRES the recipe
   fp, so refusing it contradicted the load-time contract.

Gate on the candidate assembly now: **8/8 OK**, components[1] =
`chain=verified … recipe_fp=momentum-v0-fd65161a20b29314 (validated)`.
Gate suite: 45 passed with the pipeline importable; 42 passed + 3 loud skips
without (mirrors the gate's own guarded import). Four new regressions:
valid-fp pass / fp-mismatch fail / byte-pin still refused / undeclared-kind
still refused.

## Round 3 (codex): pinned-path CI can now validate the fp without pandas

The gate imports `renquant_pipeline.momentum_identity.params_fingerprint` —
the stdlib-only PUBLIC contract pipeline#266 created (the scorer aliases the
same function; one implementation; the published literal
`momentum-v0-fd65161a20b29314` is pinned by test there). The pipeline pin in
this PR advances `5f07a4d2` → `ab5db5ab` (also carries pipeline#265's
at-birth fleet broker tags) and the snapshot is re-rendered at that pin.
Candidate-assembly gate at the final pin: 8/8 OK. Gate suite: 45 passed with
the pipeline importable, 42 + 3 loud skips without.
