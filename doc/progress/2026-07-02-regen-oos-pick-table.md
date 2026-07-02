# Regenerate the durable OOS pick table — Track A evidence-base prerequisite

STATUS: delivered (committed generator + regenerated durable table; the leak-controlled
"genuine IC" re-audit and Track A's conditional-pick-quality test are explicitly NOT in scope —
see NEXT)
WHAT: `scripts/regen_oos_pick_table.py` — a committed, re-runnable generator that re-scores the
prod GBDT walk-forward manifest (`backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json`,
43 point-in-time artifacts) read-only through the SAME point-in-time manifest contract the walk-forward
gate itself uses (`scripts.run_wf_gate._score_manifest_sanity` / `WalkForwardModelLoader`), and persists
the per-(date, name) result as `data/exp/oos_pick_table_recipe_v2.parquet` — the durable table the
renquant-orchestrator direction-decision doc (`doc/design/2026-06-28-renquant105-direction-decision.md`
§4) said did not yet exist as a committed artifact, and the 2026-07-01 design-review amendments doc
(A7) elevated to "the evidence base of the 105 direction itself."
WHY/DIR: the A1 audit that anchors the entire renquant105 direction (no robust directional edge
surfaced; apparent skill is a thin BEAR-regime slice; BULL_CALM ~79% of live time reads near-zero)
lived only as deleted `/tmp` scratch from unmerged scripts — not re-fetchable by a reviewer, not
re-runnable, and explicitly called out (amendments doc A7) as disproportionately load-bearing for
something nobody could re-check. This PR does not re-run the full leak-injection audit; it lands the
one missing piece both the direction-decision doc and the amendments doc identified as the actual
blocker: a committed generator + a durable, git-tracked output table.
EVIDENCE:
- Ran `scripts/regen_oos_pick_table.py` end-to-end against the real manifest/panel
  (`/Users/renhao/git/github/RenQuant/.venv/bin/python scripts/regen_oos_pick_table.py`, with
  PYTHONPATH including every sibling subrepo's `src/`). Output: `data/exp/oos_pick_table_recipe_v2.parquet`,
  **147,066 rows / 508 dates / 292 names** — this row/date count is an EXACT match to the figure the
  (now-deleted) A1 audit cited ("147,066 OOS rows / 508 dates") and the val_cut (`2024-02-01`) also
  matches exactly, which is strong evidence the panel construction and manifest/scoring methodology
  reproduced here is faithful to what the original audit ran.
- Regime distribution: `BULL_CALM=115,968 (78.9%)`, `BEAR=14,600 (9.9%)`, `CHOPPY=10,950 (7.4%)`,
  `BULL_VOLATILE=5,548 (3.8%)` — the 78.9% BULL_CALM share matches the direction-decision doc's cited
  "~79% of live time" almost exactly.
- `tests/test_regen_oos_pick_table.py` — 8 passed (decile-bucketing correctness/monotonicity/balance/
  fallback, reference-artifact resolution, import sanity). Deliberately does NOT re-run the expensive
  43-artifact scoring pipeline in CI; that was run once, manually, as this PR's own verification (see
  above and below).
- **IMPORTANT DISCREPANCY — reported honestly, not rounded toward the expected value.** I computed a
  NAIVE (raw, non-leak-adjusted) per-date Spearman IC directly on the regenerated table via
  `analyze_manifest_sanity_placebo.summarize_ic` as an independent sanity check. Results:
  - Overall: **mean_ic = +0.0538** (508 dates, 147,066 rows)
  - BEAR: **mean_ic = +0.3467** (50 dates, 14,600 rows)
  - BULL_CALM: **mean_ic = +0.0213** (399 dates, 115,968 rows)
  - BULL_VOLATILE: **mean_ic = +0.0226** (19 dates)
  - CHOPPY: **mean_ic = +0.0261** (40 dates)

  The direction-decision doc cites a DIFFERENT set of numbers for the same manifest: overall
  "genuine (leak-controlled)" IC ≈ 0.0415-0.0417, and **BULL_CALM genuine ≈ −0.003** (a materially
  different sign and magnitude from the +0.0213 I measured). The BEAR n_dates=50 matches exactly, but
  my BEAR IC (+0.3467) is also considerably higher than the doc's cited +0.236.

  The most likely explanation, and the reason I am NOT treating this as a reproduction failure: the
  direction-decision doc's "genuine (leak-controlled)" IC is explicitly NOT the naive per-date Spearman
  IC I computed here — its own text says "genuine (leak-controlled) IC has a CI that includes 0, and it
  is not leak-free — predictor-side persistence balloons the naive IC" (§1, A1 bullet). That
  leak-decomposition (real vs. placebo vs. label-autocorrelation-inflated IC) is exactly what the
  ORIGINAL audit's `02_repro_and_rigor.py` / `03_injection_tests.py` / `04_injection_floor_leak.py`
  scripts did — none of which are reimplemented by this PR. This PR's scope (per the amendments doc A7
  and the direction-decision doc §4 prerequisite framing) was specifically the durable PICK TABLE
  generator, not the full leak-injection audit. The naive IC I measured is a real, useful number
  (row/date-count-verified against the original audit), but it is NOT comparable to the doc's "genuine"
  figure without redoing that leak-adjustment step, which is out of scope here and flagged below.
- `git diff --check` clean.
NEXT: (1) the leak-controlled "genuine IC" re-audit (porting `02_repro_and_rigor.py` /
`03_injection_tests.py` / `04_injection_floor_leak.py`'s methodology as a committed script against
this now-durable table) is required before anyone treats +0.0538 / +0.0213 as confirming OR refuting
the direction-decision doc's −0.003 BULL_CALM claim — right now neither number should be quoted as
supporting or undermining the 105 direction without that step; (2) Track A's conditional-pick-quality
test itself (direction-decision doc §4 — meta-label candidate-quality test, chronological 60/40 split,
pre-registered GO/STOP thresholds) can now run against this table for the conditioning variables that
are `[VERIFIED]` available (regime, cross-sectional dispersion, score margin); the earnings-surprise and
liquidity/volatility conditioning variables still need their own availability check per §4's spec; (3)
if the durable regeneration (once leak-adjusted) materially changes the A1 finding, the direction itself
reopens — that is what falsifiable means (amendments doc A7).
