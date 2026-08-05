# 2026-08-04 — the WF gate's production reference: pinned or nothing (orch#799 item 1)

## The measured failure

Two promote runs, same day, byte-identical booster (`43285f13e98f21ac`):
Sharpe 0.6018 → 0.0524, APY +6.31% → −0.02%, buy source SelectionJob+TopUpJob
→ JointPortfolioQPJob, 373 → 104 simulated trades. The IC battery was
identical to 5 decimals and HMM regime counts identical, so the sim was not
stochastic — **it simulated a different strategy**.

Cause chain, each link read back from logs/configs:

1. `_find_gbdt_config()` searches for a config declaring `kind=xgb`, in order:
   pinned → **umbrella working copy**.
2. The full-book z-blend switch (deployed ~18:1x PT) made the PINNED primary
   `kind=blend`; the pinned shadow is `hf_patchtst`. Neither matches an xgb
   candidate any more.
3. The search therefore fell through to
   `backtesting/renquant_104/strategy_config.shadow.json` — the A8 registry's
   known-diverged umbrella WORKING COPY (hf_patchtst-era semantics: QP
   enabled, kelly on, `buy_floor=adaptive_mean_std`, `min_rank_score=0.55`).
4. `--derive-config-from-prod` faithfully derived "production semantics" from
   that stale file, and `config_parity` passed (it does not cover the
   portfolio-construction path). Log line, 20:23: *"Selected kind-matched
   production reference … backtesting/renquant_104/strategy_config.shadow.json"*
   versus 13:03's pinned config.

## Change

The umbrella working copy is removed from the candidate list (still NAMED at
the point of decision so the exclusion is visible, not accidental). When no
PINNED config matches the candidate kind the wrapper now FAILS CLOSED: it
explains the blend-prod situation, names the open decision (orch#799), pages
`WEEKLY-BLOCKED`, states that production is unchanged and RFC#210 freshness
governance is unaffected, and exits 2.

Fail-closed is strictly better than the alternative here: the RFC#210 fallback
block runs on ANY nonzero gate exit, so freshness governance still protects the
book; what disappears is a phantom verdict that looked like model evidence.

## The decision this exposes (NOT decided here)

With prod as a blend, an xgb candidate has no same-kind reference by
construction. Either (a) derive the xgb reference from the blend's
component[0] semantics, or (b) gate blend prods on blend-kind candidates.
Recorded on orch#799; until it lands the weekly promote will block loudly
instead of producing meaningless numbers.

## Verification

- New `tests/test_gate_prod_reference_fail_closed.py`: working copy absent
  from every candidates array, pinned always present, no-match path
  fails closed with the explanation + alert + "Production unchanged", 4 passed.
- RFC#210 fallback + wrapper-guard suites: 22 passed, 1 PRE-EXISTING failure
  (`test_layer3_cuts_match_candidate_artifact_recipe`, cuts fingerprint
  `14586756` vs candidate `f8fb2259`) which fails identically on clean main —
  same config-identity family as this incident and filed as follow-up on #799,
  deliberately not bundled into this fix.
