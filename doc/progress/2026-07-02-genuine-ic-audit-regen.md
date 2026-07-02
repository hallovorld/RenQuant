# Reproduce analyze_manifest_sanity_placebo.py's aligned_real_ic against the durable OOS table

STATUS: revised after Codex round-8 review — metric identity corrected, reconciliation
protocol frozen, still UNRESOLVED (not a confirmation of either cited figure)
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

## Reconciliation protocol (frozen spec — NOT executed by this PR)

Per Codex: until this protocol is executed, neither this PR's figures nor the
direction-decision doc's cited figures may be treated as a settled premise. This section
specifies the protocol for whoever executes the reconciliation next; it is a
pre-registration, not an analysis.

1. **Pinned data.** Both candidate algorithms must run against the IDENTICAL table:
   `backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json`
   (manifest) + its last retrain entry's artifact, validation window 2024-02-02 to
   2026-02-11 (508 dates, 147,066 rows) — exactly as used by PR #430 and this PR. Record
   a content hash of the resulting per-(date,name) table before either algorithm runs, so
   both are provably scored against byte-identical input.
2. **Both algorithms specified in writing before either is run:**
   - Algorithm A (already implemented, already committed): `analyze_manifest_sanity_placebo.py`'s
     `shift_diagnostics`/`regime_shift_diagnostics` — `aligned_real_ic` vs
     `model_placebo_ic` vs `label_autocorr_ic` at a chosen shift window. This PR's
     numbers above ARE algorithm A's output; no further work needed to specify it.
   - Algorithm B (NOT implemented — a hypothesis to reconstruct, not assumed): whatever
     `02_repro_and_rigor.py` / `03_injection_tests.py` / `04_injection_floor_leak.py`
     actually did. The direction-decision doc's prose is the only surviving description
     ("predictor-side persistence balloons the naive IC", "fails the slow-persistence
     injection... genuine 0.042 → 0.29"). Before comparing against Algorithm A, write out
     Algorithm B's exact procedure as a testable spec (what is injected, how the
     "genuine" figure is extracted from the injected-vs-real comparison) — do not assume
     it is equivalent to Algorithm A's shift-based placebo comparison; that equivalence
     is unverified.
3. **Unit tests on injected null/leak fixtures.** Before trusting either algorithm's
   verdict on the real, disputed data: construct synthetic fixtures with a KNOWN
   ground-truth answer — (a) a synthetic LEAKY predictor (deliberately correlated with a
   future-shifted label) and confirm each algorithm correctly flags it as contaminated;
   (b) pure noise (predictor uncorrelated with anything) and confirm each algorithm
   correctly reports ~0 with no leak flag. An algorithm that fails either fixture is
   disqualified from adjudicating the real data until fixed.
4. **Untouched adjudication slice.** Reserve a portion of the real data (e.g. a
   contiguous trailing date range, or a held-out regime-slice) that neither algorithm's
   implementation is tuned or debugged against during steps 2-3. The final reconciled
   read comes from running both algorithms ONLY against this untouched slice, after both
   have passed the synthetic fixtures in step 3 — never from re-running against data
   already inspected during development.

NEXT: **explicit downstream blocker** — until the reconciliation protocol above is
executed, PRs #228/#230/#231 (and any future Track-A GO/STOP decision per the
direction-decision doc §4) must NOT cite either the −0.003 or the +0.044/+0.044-aligned
figure as a settled premise for the BULL_CALM skill question. Both are provisional. This
PR does not choose which number is authoritative — it corrects the metric-identity
labeling and freezes the reconciliation spec above; the actual reconciliation (or an
explicit operator/Codex decision to proceed without it, e.g. treating both figures as
within-noise-of-each-other near zero given Track A's own +50bps/yr economic GO bar) is
future work.
