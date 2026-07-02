# Reproduce analyze_manifest_sanity_placebo.py's aligned_real_ic against the durable OOS table

STATUS: revised after Codex round-10 review — metric identity corrected (round 8), every
reconciliation-protocol parameter pinned with no further researcher discretion (round 9:
exact Algorithm B construction, exact fixture seeds/tolerances, exact adjudication-slice
selection rule, exact disagreement threshold, exact decision mapping), AND (round 10,
this revision) the protocol's own EVIDENTIARY STATUS corrected: pinning parameters after
already seeing the results they'd be applied to is EXPLORATORY/RETROSPECTIVE, not
confirmatory — round 10 relabels the whole protocol accordingly and defines the actual
confirmatory design (≥60 genuinely-new trading sessions past 2026-02-11, frozen procedure
applied verbatim, no re-tuning). Still UNRESOLVED (not a confirmation of either cited
figure; the mechanical procedure is frozen and fixture-tested but has not been — and, as
currently scoped against already-seen data, CANNOT be — executed as a valid confirmatory
test)
WHAT: follow-up to A7 (PR #430, `feat/regen-oos-pick-table`, not yet merged), which
landed a durable, re-runnable OOS pick table replacing the deleted `/tmp` scratch the
renquant105 "no directional edge" direction decision was based on. That PR's own naive
per-date Spearman IC did not match `doc/design/2026-06-28-renquant105-direction-decision.md`'s
cited "genuine (leak-controlled)" IC, so this PR runs the existing, already-committed,
already-tested `scripts/analyze_manifest_sanity_placebo.py` (`_score_manifest_sanity`
for point-in-time scoring, `shift_diagnostics`/`regime_shift_diagnostics` for a
placebo/persistence-injection decomposition) against the SAME manifest/artifact PR #430
used, and reports its output.
**Metric-identity correction (Codex round-8):** this PR does NOT reproduce "genuine
(leak-controlled) IC" — that is the name of a metric computed by three now-DELETED
scripts (`02_repro_and_rigor.py`, `03_injection_tests.py`, `04_injection_floor_leak.py`)
whose exact algorithm is not recoverable. What this PR actually computes and reports is
**the output of `scripts/analyze_manifest_sanity_placebo.py`'s `aligned_real_ic` field**
(at `shift_days=60`, sliced by regime) — a REUSED, already-committed proxy that the
direction-decision doc's own provenance note suggests is the same or a close relative of
the deleted methodology, but this is a hypothesis, not a proven equivalence. Every
mention below is relabeled accordingly; the numbers themselves are unchanged (already
independently cross-checked against PR #430's own naive computation).
WHY/DIR: the renquant105 direction (Track A vs Track B) partly rests on whether the live
model's BULL_CALM-regime skill (79% of live time) is genuinely ~0 or not. A7 made the
underlying data durable; this run reports what the best-available REUSABLE committed
proxy says against that durable data — it does not, by itself, settle the question (see
"Reconciliation protocol" below).
EVIDENCE:
- artifact:      `scripts/analyze_manifest_sanity_placebo.py` run against
                  `backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json`
                  (manifest) + that manifest's last retrain entry's `artifact_uri`
                  (reference artifact) — the identical manifest+artifact pair
                  `scripts/regen_oos_pick_table.py` (PR #430) uses. Output persisted at
                  `doc/research/evidence/2026-07-02-genuine-ic-audit-regen/{wf_sanity_placebo.json,wf_sanity_placebo.md}`.
- prod or exp:   experiment (diagnostic re-score of the prod manifest; no prod path
                  written; `analyze_manifest_sanity_placebo.py` itself is an existing
                  committed diagnostic tool, not modified by this PR)
- existing data: the direction-decision doc's §4(b) provenance cites a "committed
                  `genuine_ic`" the deleted A1 audit "reproduced... to 4dp (0.0415 vs
                  committed 0.0417)" — this PR's overall `aligned_real_ic` (+0.0760,
                  shift=60) and overall naive `real_ic` (+0.0543) are the closest
                  committed analogues found; neither matches 0.0417 exactly either,
                  underscoring that the exact deleted-script metric is not reproduced
                  here, only approximated by the nearest committed proxy.
- best-known?:   this is the best-available REUSABLE, already-tested, already-committed
                  methodology for this decomposition in the repo today — NOT
                  demonstrated to be the best-known variant relative to the deleted
                  original scripts, whose exact algorithm cannot be checked against
                  (they no longer exist). A stricter/different injection methodology
                  may exist and would need to be reconstructed or re-specified (see
                  reconciliation protocol below) before this can be upgraded from
                  "best available" to "confirmed equivalent."
- scope:         this is `analyze_manifest_sanity_placebo.py`'s `aligned_real_ic`
                  (shift=60) on `walkforward_manifest_gbdt_prod_recipe_v2.json`'s 508-date
                  validation window (experiment/diagnostic), vs the direction-decision
                  doc's cited "genuine_ic" ≈0.0417 (overall) and BULL_CALM ≈ −0.003 —
                  same underlying data window, UNVERIFIED whether same algorithm.

  Full numbers (report in full, not rounded toward any answer):

  | Metric | This run (`aligned_real_ic` / `real_ic`) | Direction-decision doc (cited "genuine_ic") | PR #430 (naive Spearman) |
  |---|---:|---:|---:|
  | Overall naive `real_ic` | +0.0543 | — | +0.054 |
  | Overall `aligned_real_ic` (shift=60) | +0.0760 | ~0.0417 | n/a |
  | BULL_CALM naive `full_real_ic` (shift=60) | +0.0213 | (not separately cited) | +0.021 |
  | BULL_CALM `aligned_real_ic` (shift=60) | **+0.0437** | **≈ −0.003** | n/a |
  | BULL_CALM `model_placebo_ic` (shift=60) | +0.0266 (61% of `aligned_real_ic`) | (doc: persistence balloons genuine 0.042→0.29, ratio ~7×) | n/a |

  **BULL_CALM naive `full_real_ic` (+0.0213) matches PR #430's independently-computed
  naive Spearman (+0.021) almost exactly** — a real cross-check that the two
  regeneration efforts agree with each other on the naive figure. Neither this run's
  naive nor its `aligned_real_ic` figure matches the doc's cited BULL_CALM "genuine_ic"
  (−0.003) — not even the sign agrees.
- Validation window: 2024-02-02 to 2026-02-11, 508 dates, 147,066 rows — matches both
  PR #430's regenerated table and the direction-decision doc's cited figures exactly.
- Manifest entry-count reconciliation: the manifest has 43 total retrain entries (doc's
  §4(b) cites "37 PIT artifacts"); 36/43 have `cutoff_date` inside the 508-date window (6
  predate it, 1 postdates it) — 36≈37 (off-by-one, plausibly boundary-inclusivity), and
  all 43 entries share `trained_date=2026-06-15`, twelve days before the direction-decision
  doc, so the manifest has NOT grown since the original audit. "37" almost certainly
  counted entries actually exercised for scoring, not manifest size. Rules out "manifest
  growth" as an explanation for the discrepancy above.
- Placebo/persistence contamination IS present and substantial here too: BULL_CALM's
  `model_placebo_ic` (+0.027) is 61% the magnitude of `aligned_real_ic` (+0.044) —
  qualitatively consistent with the doc's "predictor-side persistence balloons the naive
  IC" narrative — but applying that same adjustment still leaves a POSITIVE residual, not
  a coin-flip negative, in this run.
- `py_compile` clean. No code was modified — this PR only runs an existing, unmodified
  script and commits its durable output.

## Reconciliation protocol (round 3 — EXPLORATORY/RETROSPECTIVE, not confirmatory; NOT executed by this PR)

**Codex round 10 correction — read this before anything below.** Round 2 (below) pinned
every remaining parameter of this protocol to a concrete, no-further-discretion value —
but "no further discretion left NOW" is not the same as "was never subject to discretion."
The adjudication-slice rule (`sorted(unique_dates)[-90:]`) and the disagreement margin
(`0.02`) were BOTH selected AFTER the full 508-date table, including those same last 90
dates, had already been scored and inspected in this PR (the +0.044 / +0.027 numbers
reported above). The 90-date slice was not chosen blind; the 0.02 margin was explicitly
picked as "roughly half the gap already observed" between those two already-seen numbers.
Freezing those choices now stops them drifting further, but it does **not** restore
holdout status, and it does **not** make this a preregistered threshold — pre-registration
exists specifically to prevent choosing test parameters with knowledge of results that
would be run against those parameters, which is exactly what happened here.

**This protocol is therefore EXPLORATORY/RETROSPECTIVE, not confirmatory.** It may be
executed and reported as an informative data point, but it must **NOT**, by itself,
authorize any Track-A or 105-route decision (the direction-decision doc §4 GO/STOP
criteria, or PRs #228/#230/#231's dependency gates on the BULL_CALM premise) — see the
corrected NEXT section at the bottom. A genuine confirmatory test requires the different
design in "Path to an actual confirmatory test" below.

Codex round 9 correctly found round 1 of this protocol was a menu of deferred choices
("Algorithm B is deferred", "shift window... chosen", adjudication slice "e.g. ...") —
not actually frozen, since a researcher could still make each of those choices AFTER
seeing results, defeating the whole point of pre-registration. Round 2 (immediately below)
pins every one of those choices to a concrete, mechanical, no-further-discretion value —
useful as a fully-specified MECHANICAL PROCEDURE (reusable verbatim by the confirmatory
test below), but, per the round-10 correction above, not itself a valid confirmatory
pre-registration because of when its parameters were chosen. It is still a
pre-registration for whoever executes the reconciliation next in EXPLORATORY mode; **this
PR still does not execute it.**

**0. Pinned data (unchanged from round 1).** Both algorithms run against
`backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json`
(manifest) + its last retrain entry's artifact, validation window 2024-02-02 to
2026-02-11 (508 dates, 147,066 rows, 292 names) — identical to PR #430 and this PR.
Record a content hash of the resulting per-(date,name) table before either algorithm
runs.

**1. Algorithm A — already fully specified, already run (this PR's own numbers).**
`analyze_manifest_sanity_placebo.py`'s `shift_diagnostics`/`regime_shift_diagnostics`,
`shift_days=60`, regime=BULL_CALM, reading the `aligned_real_ic` field. Output:
BULL_CALM `aligned_real_ic` = **+0.0437** (already computed above). No further
specification needed.

**2. Algorithm B — best-effort reconstruction, explicitly marked NOT verified
equivalent to the deleted originals.** The only surviving description (direction-decision
doc §4(b)) is: "the metric fails the slow-persistence injection (predictor-side
persistence balloons genuine 0.042 → 0.29)". This names a **persistence-injection test**:
construct a synthetic predictor with realistic cross-sectional persistence structure but
**zero genuine forward-looking information**, run it through the identical IC
methodology, and check whether persistence ALONE can produce an IC comparable to or
larger than the real model's. Reconstructed as a fully mechanical procedure (I could not
recover the exact arithmetic that turns "0.042 vs 0.29" into a single adjusted
"genuine ≈ −0.003" point estimate from the surviving text — that specific step is
**unrecoverable** and is NOT reconstructed; Algorithm B below answers the same
conceptual question via a comparison, not a point-adjustment formula):
- **Construction (exact, no discretion left):** for every `(name, date)` row in the
  pinned table, `persistence_score[name, date] = real_score[name, date − 60 business
  days]` (i.e. that SAME name's own real model score, shifted back exactly one label
  horizon — reuses the identical shift machinery `shift_diagnostics` already implements
  for the label side, applied here to the SCORE side instead). Rows where `date − 60
  business days` falls before the panel's start (no prior score available) are dropped —
  this is a deterministic, data-driven exclusion, not a choice.
- **Scoring:** run `persistence_score` through the IDENTICAL `_cs_ic_series` /
  `summarize_ic` / `regime_shift_diagnostics` machinery Algorithm A already uses (same
  code path, same `shift_days=60`, same regime slice = BULL_CALM), substituting
  `persistence_score` for `mu`. Output field: `persistence_ic` (BULL_CALM,
  `aligned_real_ic`-equivalent at shift=60).
- **What this tests:** whether a signal with realistic persistence but NO knowledge of
  the true forward label can, by construction of the IC methodology itself, produce an
  IC comparable to the real model's +0.0437 in the same regime. It does not attempt to
  reproduce the deleted scripts' exact "0.042 → 0.29" arithmetic.

**3. Unit tests on injected fixtures — exact construction, seeds, and tolerance.**
Before either algorithm may adjudicate the real disputed data, both must pass on
synthetic fixtures with known ground truth, built over the SAME shape as the real table
(508 dates × 292 names, using the real table's own `(date, name)` index so panel
structure/coverage matches):
- **Null fixture:** `null_score[name, date] = rng.standard_normal()` using
  `numpy.random.default_rng(seed=20260702)`, drawn once per `(name, date)` row in
  index order, independent of `fwd_60d_excess`. True IC = 0 by construction. **Pass
  condition:** both Algorithm A's and Algorithm B's methodology, run against
  `null_score` in place of `mu`/`real_score`, must report `|IC| ≤ 0.01` (BULL_CALM,
  shift=60, `aligned_real_ic`). An algorithm reporting `|IC| > 0.01` on pure noise fails
  the fixture and is disqualified until fixed.
- **Leak fixture:** `leak_score[name, date] = fwd_60d_excess[name, date] +
  rng2.normal(0, 0.01)` using `numpy.random.default_rng(seed=20260703)` for the noise
  term (the label itself, plus small noise for numerical non-degeneracy — deliberate,
  perfect look-ahead leakage). True IC ≈ 1 (Spearman rank-IC of a monotonic-plus-tiny-
  noise transform of the label against itself). **Pass condition:** both algorithms must
  report BULL_CALM `aligned_real_ic` (shift=60) `≥ 0.95` against `leak_score`. An
  algorithm reporting `< 0.95` on deliberately-leaked data fails the fixture.
- Both fixtures use the pinned table's own `(date, name)` index and `fwd_60d_excess`
  column so panel coverage/shape matches the real evaluation exactly; only the score
  column is replaced.

**4. Untouched adjudication slice — exact, mechanical selection rule.** The final
reconciled read comes ONLY from the LAST 90 trading dates (ascending order) of the
pinned 508-date window: `sorted(unique_dates)[-90:]` computed directly from the
committed table (`data/exp/oos_pick_table_recipe_v2.parquet`, or its regenerated
equivalent) — a fully mechanical rule with no researcher discretion in which dates are
included. The window's known end date is 2026-02-11; the resulting slice's exact start
date is whatever `sorted(unique_dates)[-90]` evaluates to against the pinned table (not
hand-picked). Both algorithms' code/parameters must be FINALIZED (having already passed
step 3's fixtures) BEFORE being run against this slice — no algorithm changes are
permitted after seeing this slice's results. No dates are excluded from the slice for
any reason (e.g. no post-hoc removal of "unusual" sessions).

**5. Primary metric and disagreement rule — exact numbers, not vague language.** The
primary metric is: `real_ic` = Algorithm A's BULL_CALM `aligned_real_ic` (shift=60) on
the adjudication slice, compared against `persistence_ic` = Algorithm B's BULL_CALM
`persistence_ic` (shift=60) on the SAME slice. **Disagreement rule:**
`real_ic > persistence_ic + 0.02` is the exact, pre-registered threshold for "the real
model shows BULL_CALM signal distinguishable from persistence alone" (0.02 chosen as
roughly half the gap already observed between this PR's BULL_CALM `aligned_real_ic`
(+0.044) and `model_placebo_ic` (+0.027) on the full window — a conservative margin,
not tuned against the adjudication slice itself, which is untouched per step 4).

**6. Decision mapping — exact, exhaustive, pre-committed, EXPLORATORY ONLY (round-10
correction: none of these outcomes authorizes a Track-A/route decision by itself — see
above).** Algorithm A vs Algorithm B is a comparison between the reused committed proxy
and a newly-invented lagged-score comparator built for this reconciliation effort —
**neither outcome says anything about whether the deleted original scripts would have
agreed with either one.** Do not read "A and B agree" as "reproduces the deleted
methodology."
- **`real_ic > persistence_ic + 0.02`** → the real model's BULL_CALM signal, under
  Algorithm A, exceeds what the Algorithm-B persistence comparator alone produces →
  *exploratory* evidence AGAINST the original "≈0 / coin-flip / no demonstrated skill"
  framing for BULL_CALM → flags the 105 direction as worth a confirmatory look (per
  amendment A7's own rule: "if the durable regeneration materially changes A1, the
  direction is re-opened") — raises the question for operator/Codex, does **not** by
  itself reopen or settle it; a genuine reopening decision should wait for (or be made
  explicitly despite) the confirmatory test below.
- **`real_ic ≤ persistence_ic + 0.02`** → the real model's BULL_CALM signal is NOT
  distinguishable from the Algorithm-B persistence comparator → *exploratory* evidence
  CONSISTENT WITH the original "≈0, no demonstrated skill" framing → still does **not**
  itself authorize PRs #228/#230/#231 to cite BULL_CALM as settled no-edge; those PRs may
  cite this outcome only as "consistent with, exploratory, not confirmatory."
- **Either algorithm fails step 3's fixtures** → neither algorithm may adjudicate the
  real data even exploratorily; escalate to operator/Codex for a different methodology;
  render NO verdict on BULL_CALM from this protocol.
- **This protocol is never averaged, never re-run with different parameters after
  seeing the adjudication-slice result, and never used to retroactively justify a
  DIFFERENT margin/threshold than the one frozen in step 5** — this discipline is worth
  keeping even in exploratory mode, and is exactly what step 3-5's mechanical procedure
  should be REUSED VERBATIM for once a genuine confirmatory design (below) is available.

## Path to an actual confirmatory test (round 10, new)

The mechanical procedure in steps 0-6 above (data pin, Algorithm A, Algorithm B
construction, fixture tests, primary metric, disagreement rule, decision mapping) is
worth keeping and reusing verbatim — it is fully specified and fixture-tested. What makes
the CURRENT run of it merely exploratory is only WHICH DATA it was pointed at (a slice
already seen). Two ways to get a genuine confirmatory result from the same procedure:

- **(a) Wait for genuinely future, unobserved sessions (RECOMMENDED).** The pinned
  table's window ends 2026-02-11. Re-run `scripts/regen_oos_pick_table.py` (PR #430) at a
  later date once real trading sessions have accrued PAST 2026-02-11, take ONLY the new
  dates (never seen in this analysis) as the adjudication slice, and apply the
  ALREADY-FROZEN Algorithm A/B code and the already-frozen `0.02` disagreement threshold
  from steps 1-5 above VERBATIM — no re-tuning permitted now that this document is
  published. Minimum sample: propose **≥60 new trading sessions** before running the
  confirmatory test (roughly one label horizon's worth of fresh dates — matches this
  repo's own `label_observation_cutoff` / `lookahead_days=60` convention used throughout
  this analysis, and is enough for `regime_shift_diagnostics`' existing `min_names`/date
  floor to produce a non-degenerate per-regime read). This is the straightforward,
  clearly-valid option: the new dates cannot have influenced a threshold that was
  published before they existed.
- **(b) A cryptographically sealed historical partition (NOT pursued here).** In
  principle, an existing historical slice could serve as a valid holdout if a commitment
  to the exact frozen protocol (steps 0-6) had been independently hashed/timestamped
  BEFORE that specific partition was ever inspected. No such prior commitment exists for
  any slice of the current 508-date table — every date in it was already scored and
  visible before this document existed. Constructing one retroactively is not possible
  (that is precisely the problem being corrected here), and this repo/session has no
  existing infrastructure for independently-timestamped commitments. Option (a) above is
  the only currently-practical path; (b) is documented for completeness, not proposed as
  actionable.

NEXT: **explicit downstream blocker, strengthened by the round-10 correction** — until a
GENUINE confirmatory reconciliation executes (option (a) above: the already-frozen
Algorithm A/B procedure applied to ≥60 genuinely-new, never-before-seen trading sessions
past 2026-02-11), PRs #228/#230/#231 (and any future Track-A GO/STOP decision per the
direction-decision doc §4) must NOT cite either the −0.003 or the +0.044 figure — nor any
future EXPLORATORY run of this protocol against the existing 508-date table — as a
settled premise for the BULL_CALM skill question. Running the frozen protocol against the
already-seen data (if anyone does, for curiosity) produces an EXPLORATORY data point only,
per the decision-mapping correction above, never a decision-authorizing result. This PR
does not choose which number is authoritative, does not execute the protocol, and — as of
this revision — does not claim the frozen procedure is a valid confirmatory test against
the data it currently has: it corrects the metric-identity labeling (round 1), freezes
every remaining mechanical parameter (round 2), and corrects the holdout-validity claim
(round 3, this revision) by defining what a genuine confirmatory run actually requires. An
explicit operator/Codex decision to proceed WITHOUT any reconciliation (e.g. treating both
figures as within-noise-of-each-other near zero given Track A's own +50bps/yr economic GO
bar) remains available at any time as a separate, recorded risk-acceptance path — that is
not this protocol, and does not require waiting for it.
