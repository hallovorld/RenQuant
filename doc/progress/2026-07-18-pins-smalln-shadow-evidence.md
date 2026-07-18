# 2026-07-18 — pins: small-n guard shadow-evidence bump (P0-compliant, supersedes #498)

## What

Bump two pins in `subrepos.lock.json` to current mains (two lines, nothing else):

| subrepo | old pin | new pin |
|---|---|---|
| renquant-pipeline | `7108f514` | `d32f7017ff052faae16668850e6c1b3be1359f08` |
| renquant-strategy-104 | `0d45d960` | `082dccd21584733af57c33e95851bf3a69b21019` |

Plus the machine-regenerated `doc/arch/strategy-104-snapshot.md` (rendered from a
strategy-104 checkout at the new pin; `--check` passes byte-exact).

## Why this is the P0-compliant path (supersedes #498)

The P0 on #498 (review 4727377226) rejected activating
`buy_floor_min_n` / `buy_floor_absolute_smalln` in the production and golden
configs: a degraded cross-section must not become a live-admission relaxation
without evidence that small-n days are healthy. The accepted remediation chain
is now fully merged:

1. **strategy-104 #61** (`082dccd2`) — guard keys relocated to
   `strategy_config.shadow.json` ONLY; prod and golden configs restored
   byte-identical to pre-#60.
2. **pipeline #207** — amendment to RFC #204: eligibility-ledger precondition
   for the guard (small-n must be provably an eligibility outcome, not a
   failure residue, before the branch may relax anything).
3. **pipeline #208** — the eligibility-ledger precondition implemented; guard
   remains relax-only and fail-closed without a healthy ledger verdict.
4. **orchestrator #549** — `smalln_guard_suppressed` LOUD sentinel rule +
   run-bundle eligibility-ledger write, so every suppression/decision is
   observable per run.

## Production inertness (why this pin bump cannot move the live book)

- **Byte-identical prod/golden configs**: at the new strategy-104 pin the prod
  config blob is `03e8b4a30a5f4b476352f4224b7b5db72893b639` and golden is
  `7c075919c4751717b7fa06b20f880d93ec3dd856` — the exact same blob hashes as at
  the current pin `0d45d960` (pre-#60 state). `buy_floor_min_n` /
  `buy_floor_absolute_smalln` occur 0 times in prod and golden, 2 times in
  shadow.
- **Guard inert without keys**: pipeline #205/#208 branch only when the config
  supplies the keys; strategy-104 #61's tests pin prod/golden bit-identity, so
  the production decision path is unchanged by construction.
- **Snapshot proves it at the declaration level**: the regenerated
  `doc/arch/strategy-104-snapshot.md` diff vs main touches ONLY the pin lines,
  the shadow-config source hash, and the derived source-fingerprint lines. The
  pinned prod config hash (`sha256:b23d8215…`) and the Buy floor row —
  `| Buy floor | mode=adaptive_mean_std; min=+0.2000 |` — are unchanged.
  Artifact fingerprints (incl. the prod calibrator `sha256:d2b4d6ab…` canonical
  since #496) are unchanged.

## What the bump enables

- The daily **shadow arm** now runs with the guard keys present in the shadow
  config and the eligibility ledger wired (pipeline #208 + orchestrator #549),
  generating the per-run guard evidence (would-have-relaxed decisions,
  eligibility verdicts, suppression LOUDs) required for the RFC #204 §4
  activation verdict. No §4 verdict, no production keys — that ordering is now
  mechanical.
- The pipeline bump also carries **#206** (public pair-validation API, bundle
  contract phase 2, GOAL-5 AC4) and the #204/#205 guard code itself (inert in
  prod, exercised in shadow).

## Machine landing

Merging this PR does NOT deploy anything. Landing the pins on the operator
machine remains a separate ask-first operator batch, bundled with the two-arm
epoch-5 refreeze (pin-align + freeze regeneration in one supervised action).
