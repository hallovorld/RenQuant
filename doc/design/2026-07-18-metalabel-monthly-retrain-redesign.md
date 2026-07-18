# Design: leakage-correct, consumer-gated monthly meta-label retrain

Task refs: session tracker #73; the on-disk diagnosis
(`doc/progress/2026-07-17-metalabel-diagnosis.md`) and the sentinel ack
ledger reference the same work as task #75 — this RFC supersedes both
references (r2 nit).

Date: 2026-07-18
Status: RFC — design review required before any implementation
Owner: drafted personally per design-review policy
Context: com.renquant.monthly-meta-label-retrain has failed on every run
since deployment; the 2026-07-17 failure ("Look-ahead leakage detected
[SimAdapter legacy load (anchor=2026-06-21...)]") is the leakage guard
CORRECTLY refusing a mis-designed job.

## 1. The three findings that shape this design

1. **The guard is right; the job is wrong.** The job replays the training
   window `[T-60d-365d, T-60d]` (2025-05-18 → 2026-05-18 at the last
   firing) through the snapshot sim, which loads the CURRENT production
   scorer via the legacy static path. `_assert_legacy_no_leakage`
   compares the scorer's selection-cutoff anchor (~2026-06-21) + 60
   business days against the sim's first bar (2025-05-18) and raises —
   correctly: the scorer-derived snapshot features (panel_score_current/
   at_entry/delta, rank_among_holdings, and the calibrated μ) would be
   produced by a model that has seen the future of the window it
   replays. The LABELS are clean (realized forward returns for past
   decisions, fully-realized windows by the T-60d buffer); the
   contamination is the FEATURE side. Honest scope (r2 — review P2-3):
   the walk-forward path fixes the SCORER/CALIBRATOR-derived features;
   regime features come from a static GMM sim artifact (2023-12-31) —
   stale-but-not-leaky, declared as-is; the per-ticker tournament heads
   and the ngboost σ head have NO as-of guard and are RESIDUAL as-of
   gaps, disclosed here and inherited as caveats by any future
   consumer re-arm design. This RFC does not claim "every snapshot
   feature" is point-in-time.
2. **The as-of machinery already exists and was the ORIGINAL
   methodology.** `WalkForwardModelLoader.model_as_of(today)` (pipeline
   `kernel/walk_forward/loader.py`) selects, per sim bar, the newest
   corpus vintage with `cutoff_date < today` — the leakage key is
   cutoff_date, not the batch-build wall-clock `trained_date`. The WF
   corpus (`backtesting/renquant_104/artifacts/sim/walkforward_retrains/`,
   indexed by `artifacts/sim/walkforward_manifest.json`) holds 39
   point-in-time vintages, cutoffs 2024-01-01 → 2026-03-09 at 21-day
   cadence (per-vintage metadata verified point-in-time). The deployed
   classifier's own 2026-05-11 validation ran with the walk-forward
   manifest; the monthly job silently dropped it. The fix is a
   RESTORATION, not an invention. Two traps: (a) the prod config's
   `walkforward.manifest_path` points at
   `walkforward_manifest_dropsenti_v3.json`, which DOES NOT EXIST on
   disk — inheriting it fail-crashes; the snapshot config must override
   to the real manifest; (b) the corpus currently ends at cutoff
   2026-03-09, so the trailing ~10 weeks of any recent window would
   silently degrade to a stale vintage.
3. **The consumer is DOUBLY dark.** `ranking.meta_label.enabled=false`
   in the pinned config AND the artifact was removed on 2026-05-11
   (only `meta-label-exit.json.disabled-2026-05-11` remains). The only
   wired consumer, `MetaLabelVetoTask`, is a path-rule EXIT veto in the
   SellOnly path; the "entry filter" from the win-rate research is
   unbuilt. Today a SUCCESSFUL retrain would materialize an artifact
   nothing reads — the exact "deployed-but-dark / inert scaffolding"
   anti-pattern, plus a monthly alarm for a job whose output is unused.

## 2. Design

### 2.1 Consumer gate (the ordering constraint)

The monthly job becomes CONSUMER-GATED: as step 0 it reads the PINNED
strategy config; if `ranking.meta_label.enabled` is false (or the block
is absent), it exits 0 with a single log line
`meta-label consumer dark — retrain skipped by design (see RFC)` and
touches nothing. No training compute, no artifact churn, no alarm, while
the consumer is dark BY DECISION. The launchd job stays installed (and
in `ops/launchd_manifest.json`) so re-arming is a config change, not an
ops change.

Re-arming the consumer is EXPLICITLY OUT OF SCOPE here and requires its
own design PR choosing between: (a) re-enabling the validated exit veto
(fresh gate-passing artifact + shadow verification + pins), or (b) the
entry-filter design from the win-rate research (larger; currently the
operator-endorsed honest lever), or (c) retiring the job and the veto
task outright. Until one of those lands, this job's correct steady
state is "skipped by design" — visible, cheap, honest.

### 2.2 Leakage-correct snapshot sim (restores 2026-05-11 methodology)

When (and only when) the consumer gate passes:

- The snapshot config sets `walkforward.enabled=true` and OVERRIDES
  `walkforward.manifest_path` to the CALIBRATOR-BOUND corpus manifest
  `artifacts/sim/walkforward_manifest_v2_20260602.json` (r2 — review
  P1-1: the plain `walkforward_manifest.json` twin has 0/39
  `calibrator_uri` entries, and with the pinned
  `global_calibration.enabled=true` the `calibrator_as_of` load
  hard-raises at the first scored bar; the v2 manifest carries the same
  39 cutoffs with calibrator bindings, files verified on disk). The
  prod pointer (dead `dropsenti_v3` reference — trap 2a) is never
  inherited.
- **Digest-stamping prerequisite (r2 — review P1-2, the 2026-09-01
  time bomb):** `kernel/manifest_uri_resolver.py` enforces
  `ARTIFACT_DIGEST_REQUIRED_AFTER = 2026-09-01`, and BOTH corpus
  manifests are currently 0/39 `artifact_sha256` — unfixed, this
  redesign re-creates the chronic failure ~6 weeks after landing.
  Implementation therefore includes a mechanical digest-stamping pass
  over the v2 manifest (content sha256 per referenced artifact,
  stamped via the normal reviewed path), and AC-5 proves the resolver
  accepts the stamped manifest with the enforcement date forced past
  2026-09-01 in test.
- **Eligibility predicate stated correctly (r2 — review P2-1):** the
  loader's actual rule is `feature_cutoff + 60 business days < bar`
  (the label-lookahead embargo), NOT bare `cutoff_date < bar`. Every
  bar is thus served by a vintage ≥ 60bd stale BY DESIGN — that is the
  embargo, not a defect; §2.3's freshness contract bounds EXTRA
  staleness beyond it.
- **Scorer-family parity (r2 — review P2-2):** the job asserts the
  corpus vintages' scorer family maps to the snapshot config's pinned
  scorer family (explicit allowlist mapping, e.g.
  `panel_ltr_xgboost ↔ xgb`); mismatch fails closed with a named
  error. This is also the check that surfaces training/serving family
  skew (live tree has drifted to `hf_patchtst`) for any future
  re-armed consumer.
- `fail_on_no_model` stays true: a sim bar with no ELIGIBLE vintage is
  a hard failure, not a silent fallback.
- The legacy static-load path is NOT used; `_assert_legacy_no_leakage`
  never fires because per-bar vintages satisfy the embargo ordering by
  construction (review-verified: the legacy assert lives only on the
  non-WF branch). The guard itself is untouched.

### 2.3 Corpus-coverage contract (trap 2b, r2 math)

Because the eligibility predicate is `feature_cutoff + 60bd < bar`
(§2.2), the newest vintage that can serve the window's LAST bar must
have `feature_cutoff < TRAIN_END − 60bd`. The job therefore asserts,
before running: newest corpus `feature_cutoff ≥ TRAIN_END − 60bd −
35d` (35d = one 21-day cadence step + margin of EXTRA staleness beyond
the structural embargo). Staler ⇒ FAIL CLOSED with
`wf corpus stale for window` naming the newest cutoff — never train on
features from a silently stale vintage.

**Stated plainly (r2 — review P2-4): this assert FAILS TODAY.** The
corpus's newest cutoff is 2026-03-09, and its refresh dependency
(weekly_wf_promote) is chronically rejecting — the known WF-gate root
(G4 research territory). So even with this RFC landed, the §2.2 path
cannot run until the corpus refreshes; the job's steady state remains
the §2.1 consumer-gated skip, which is the honest state. Re-arming the
meta-label consumer is DOUBLY gated: the consumer decision AND corpus
freshness. Neither gate is this RFC's to open.

### 2.4 What is unchanged

Labels (triple-barrier + realized fwd returns on trigger rows), trainer
(PurgedKFold + embargo), health gate (AUC/n_events/balance/features),
atomic artifact swap, schedule (monthly, 03:30 PT), and the leakage
guard itself. No pipeline/backtesting code changes at all — §2.2 is
snapshot-CONFIG construction inside the wrapper script plus the §2.1/
§2.3 wrapper checks.

## 3. Failure-surface disposition

- Current state (consumer dark): job exits 0 "skipped by design" →
  the sentinel ack row for monthly-meta-label-retrain is RETIRED after
  the first green run (the alarm the operator saw twice disappears
  honestly, not by silencing).
- Consumer re-armed later: the job runs the §2.2 path; its failure
  modes are then real failures (corpus stale, vintage gap, health-gate
  reject) and stay LOUD.

## 4. Acceptance criteria

- AC-1: with the consumer dark (current pinned config), the job exits 0
  in <5s, writes no artifact, and the next sentinel scan shows the
  launchd row green with no ack needed.
- AC-2: with a test config enabling the consumer, the snapshot sim runs
  the walk-forward path end-to-end over a test window with NO
  leakage-guard firing, and per-bar selection verifiably satisfies the
  TRUE predicate: every bar's serving vintage has
  `feature_cutoff + 60bd < bar` (r2 — not bare cutoff ordering), with
  calibrator bindings resolving at every scored bar (v2 manifest).
- AC-3: corpus-staleness injection (newest feature_cutoff older than
  `TRAIN_END − 60bd − 35d`) fails closed with the named error.
- AC-4: the dead `dropsenti_v3` inheritance path is proven unreachable
  (snapshot config always carries the explicit override).
- AC-5: the digest-stamped v2 manifest passes the resolver's
  `ARTIFACT_DIGEST_REQUIRED_AFTER` enforcement with the date forced
  past 2026-09-01 in test (no time bomb).
- AC-6: scorer-family mismatch injection (corpus family ≠ pinned
  family) fails closed with the named error.

## 5. Implementation plan (post-approval)

1. Umbrella PR: `scripts/monthly_meta_label_retrain.sh` — consumer gate
   (step 0), walk-forward snapshot config construction (v2 manifest
   override), corpus-coverage assertion (§2.3 math), scorer-family
   parity assertion. Shell-only; no subrepo changes.
2. Digest-stamping pass over the v2 corpus manifest (content sha256 per
   referenced artifact) via the normal reviewed path — MUST land before
   2026-09-01 regardless of this job's state (the resolver enforcement
   hits every consumer of unstamped manifests, not just this one; AC-5).
3. Orchestrator PR: retire the sentinel ack row after the first green
   run; add the staleness + family-mismatch error strings to the
   sentinel's known-loud patterns.
4. The consumer re-arm decision: separate design PR (out of scope),
   tracked as its own task with the (a)/(b)/(c) options above; it
   inherits the §1.1 residual as-of caveats (regime/per-ticker/σ heads)
   and the §2.3 double gate.
