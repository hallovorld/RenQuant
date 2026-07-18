# Design: leakage-correct, consumer-gated monthly meta-label retrain (task #73)

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
   correctly: every snapshot feature (panel_score_current/at_entry/
   delta, rank_among_holdings, μ/σ) would be produced by a model that
   has seen the future of the window it replays. The LABELS are clean
   (realized forward returns for past decisions, fully-realized windows
   by the T-60d buffer); the contamination is the FEATURE side.
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
  `walkforward.manifest_path` to the real corpus manifest
  (`artifacts/sim/walkforward_manifest.json`), never inheriting the
  prod pointer (dead `dropsenti_v3` reference — trap 2a).
- `fail_on_no_model` stays true: a sim bar with no vintage whose
  `cutoff_date < bar` is a hard failure, not a silent fallback.
- The legacy static-load path is NOT used; `_assert_legacy_no_leakage`
  never fires because per-bar vintages satisfy the cutoff ordering by
  construction. The guard itself is untouched.

### 2.3 Corpus-coverage contract (trap 2b)

The job asserts, before running, that the corpus's newest
`cutoff_date ≥ TRAIN_END − max_vintage_staleness` (proposed: 35 days,
one 21-day cadence step + margin). If the corpus is staler, the job
FAILS CLOSED with `wf corpus stale for window` naming the newest cutoff
— it does not train on features from a silently stale vintage. This
makes the WF-corpus refresh (weekly_wf_promote) an explicit dependency;
the corpus being maintained is exactly the condition under which
"as-of" is meaningful for recent windows.

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
  the walk-forward path end-to-end over a 3-month test window with NO
  leakage-guard firing, and per-bar scorer selection is verifiably
  as-of (log shows monotone cutoff_dates ≤ each bar).
- AC-3: corpus-staleness injection (pointing at a manifest whose newest
  cutoff is >35d before TRAIN_END) fails closed with the named error.
- AC-4: the dead `dropsenti_v3` inheritance path is proven unreachable
  (snapshot config always carries the explicit override).

## 5. Implementation plan (post-approval)

1. Umbrella PR: `scripts/monthly_meta_label_retrain.sh` — consumer gate
   (step 0), walk-forward snapshot config construction, corpus-coverage
   assertion. Shell-only; no subrepo changes.
2. Orchestrator PR: retire the sentinel ack row after the first green
   run; add the staleness error string to the sentinel's known-loud
   patterns.
3. The consumer re-arm decision: separate design PR (out of scope),
   tracked as its own task with the (a)/(b)/(c) options above.
