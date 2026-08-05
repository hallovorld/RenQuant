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

The umbrella working copy is removed from the candidate list — and after
codex round 2, so is the multirepo/sibling path: `renquant_subrepo_root`
defaults to the SIBLING DEVELOPER CHECKOUT absent an assembly override, so a
locally-edited checkout could recreate this exact incident. The ONLY candidate
in BOTH runner modes is now the lock-aligned `.subrepo_runtime` config — what
the daily run actually loads. Both excluded paths stay NAMED at the point of
decision so the exclusions are visible, not accidental. When no
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

## Round 3: the pre-existing layer-3 red, diagnosed and re-derived

The wrapper-guard failure noted above turned out to be diagnosable in one step
and is fixed in this PR rather than deferred:

`config_fingerprint_fields` on BOTH artifacts is exactly
`{watchlist, sector_map}`. The WF cuts carry a **142-name** watchlist; the
candidate carries **145** (CRWV / RKLB / SPCX added later). The fingerprints
differed because the **universe grew**, not because the recipe drifted — and
the real gate agrees: its manifest matching keys on the RECIPE fingerprint
(`sha256:cfdd6cb8e950da0f`) and passed 43/43 rows on the very artifacts this
test called incompatible.

So the fallback was asserting an identity that includes the universe against a
corpus that necessarily predates universe growth: structurally red forever
after any watchlist addition, and silent about the thing it claimed to guard.

Re-derived to what the fallback can honestly support:
- the recipe-bearing axes (`kind`, `feature_cols`) stay binding, asserted above;
- a fingerprint difference must decompose into differing FIELDS (else the
  fingerprint recipe itself changed → regenerate);
- differences confined to `{watchlist, sector_map}` are accepted as expected
  universe drift; ANY other field failing is a real recipe drift;
- **new**: the corpus's watchlist must be a SUBSET of production's — a corpus
  carrying tickers production has DROPPED is caught, a direction the old
  equality check conflated with benign growth.
