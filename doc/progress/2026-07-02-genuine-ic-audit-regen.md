# Reproduce genuine leak-controlled IC against the durable OOS table

STATUS: delivered — WITH A SIGNIFICANT UNRESOLVED DISCREPANCY, not a confirmation
WHAT: follow-up to A7 (PR #430, `feat/regen-oos-pick-table`, not yet merged), which
landed a durable, re-runnable OOS pick table replacing the deleted `/tmp` scratch the
entire renquant105 "no directional edge" direction decision was based on. That PR's
own naive per-date Spearman IC did not match `doc/design/2026-06-28-renquant105-direction-decision.md`'s
cited "genuine (leak-controlled)" IC, so this PR reproduces the genuine/leak-controlled
metric properly, using the SAME manifest/artifact PR #430 used, via the existing,
already-committed, already-tested `scripts/analyze_manifest_sanity_placebo.py`
(`_score_manifest_sanity` for point-in-time scoring, `shift_diagnostics`/
`regime_shift_diagnostics` for the placebo/persistence-injection decomposition that
IS the "genuine (leak-controlled)" methodology — the direction-decision doc's own §4(b)
provenance says the deleted A1 audit "reproduced the committed `genuine_ic` to 4dp
(0.0415 vs committed 0.0417)", i.e. this exact machinery already existed and the
scratch audit cross-checked against it).
WHY/DIR: the entire renquant105 direction (Track A vs Track B) hinges on whether the
live model's BULL_CALM-regime skill (79% of live time) is genuinely ~0 ("a coin flip")
or not. A7 made the underlying data durable; this closes the loop by reproducing the
actual genuine-IC number against that durable data, per amendment A7's own mandate
("if the durable regeneration materially changes A1, the direction is re-opened").
EVIDENCE:
- Ran `scripts/analyze_manifest_sanity_placebo.py --artifact
  <manifest's last retrain's artifact_uri> --manifest
  backtesting/renquant_104/artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.json
  --label auto` — the SAME manifest+artifact pair `scripts/regen_oos_pick_table.py`
  (PR #430) uses, via `/Users/renhao/git/github/RenQuant/.venv/bin/python`
  (PYTHONPATH extended with all sibling-repo `src/` dirs — renquant_common,
  renquant_pipeline, renquant_model, etc. — required for `kernel.walk_forward.loader`
  and the regime classifier; not needed by A7 since it only called the lower-level
  `_score_manifest_sanity` directly, but this script's `build_regime_series` pulls in
  more of the stack).
- **Validation window: 2024-02-02 to 2026-02-11, 508 dates, 147,066 rows — EXACTLY
  matches both PR #430's regenerated table and the direction-decision doc's cited
  figures.** Strong confirmation the same underlying data/window is in play.
- **Manifest entry-count reconciliation (resolves one apparent discrepancy):** the
  manifest currently has 43 total retrain entries (doc's §4(b) cites "37 PIT
  artifacts"). Checked directly: only 36 of the 43 have `cutoff_date` inside the
  508-date validation window (2024-02-02 to 2026-02-11); 6 predate it, 1 postdates it.
  36 ≈ 37 (off by one, plausibly a boundary-inclusivity difference). **The manifest has
  NOT grown since the original audit** — all 43 entries share `trained_date=2026-06-15`
  and the file's own mtime is 2026-06-15, twelve days before the direction-decision doc
  (2026-06-28); "37" almost certainly counted entries actually exercised for scoring,
  not the manifest's total size. This rules out "manifest growth" as the explanation
  for what follows.
- **The actual, unresolved numbers (report in full, not rounded toward any answer):**

  | Metric | This reproduction | Direction-decision doc (cited) | PR #430 (naive Spearman) |
  |---|---:|---:|---:|
  | Overall real/naive IC | +0.0543 | ~0.042 ("genuine") | +0.054 |
  | Overall 60d aligned_real_ic (leak-adjusted) | **+0.0760** | ~0.0417 ("genuine_ic", reproduced to 4dp from a committed baseline) | n/a |
  | Overall interpretation | `promotion_evidence: true`, no warning | "CI includes 0... not leak-free" | n/a |
  | BULL_CALM naive IC | +0.0234 | (not separately cited as naive) | +0.021 |
  | BULL_CALM 60d aligned_real_ic (leak-adjusted / "genuine") | **+0.0437** | **≈ −0.003** ("a coin flip") | n/a |
  | BULL_CALM 60d full_real_ic | +0.0213 | — | — |
  | BULL_CALM 60d model_placebo_ic | +0.0266 (61% of aligned_real_ic) | (doc: persistence balloons genuine 0.042→0.29, ratio ~7×) | n/a |

  **My BULL_CALM `full_real_ic` (+0.0213) matches PR #430's independently-computed
  naive Spearman (+0.021) almost exactly — strong internal cross-check that the two
  regeneration efforts agree with each other.** But NEITHER this reproduction's naive
  figure NOR its leak-adjusted `aligned_real_ic` (+0.044) matches the doc's cited
  genuine BULL_CALM figure (−0.003) — not even the SIGN agrees.
- Full JSON + markdown persisted at
  `doc/research/evidence/2026-07-02-genuine-ic-audit-regen/{wf_sanity_placebo.json,wf_sanity_placebo.md}`
  (durable, committed — not `/tmp`).
- **Honest interpretation — this does NOT confirm the -0.003 figure, and does not
  cleanly refute it either:**
  1. The placebo/persistence contamination the doc describes IS present and
     substantial in this reproduction too: BULL_CALM's `model_placebo_ic` (+0.027) is
     61% the magnitude of `aligned_real_ic` (+0.044) — the model's apparent skill
     really is meaningfully inflated by shift-invariant persistence, qualitatively
     consistent with the doc's "predictor-side persistence balloons the naive IC"
     narrative.
  2. But applying that SAME contamination-adjustment logic here still leaves a
     POSITIVE residual (+0.044 aligned, or +0.021 naive), not a coin-flip negative.
  3. `scripts/analyze_manifest_sanity_placebo.py`'s `shift_diagnostics`/
     `regime_shift_diagnostics` (aligned-real vs model-placebo vs label-autocorr) is
     the best-available REUSABLE, already-committed proxy for "genuine (leak-controlled)
     IC" — the direction-decision doc's own reproducibility note strongly implies this
     IS the methodology (or very close to it) the deleted audit cross-checked against.
     But the deleted scripts (`02_repro_and_rigor.py`, `03_injection_tests.py`,
     `04_injection_floor_leak.py`) are gone, and I cannot rule out they implemented a
     MATERIALLY different/stricter injection methodology (e.g. an explicit synthetic-
     persistence injection test, not just a shift-based placebo comparison) that
     would move the BULL_CALM figure further toward (or past) zero. This
     reproduction should be read as "the best current proxy disagrees with the cited
     number," not as definitive proof the original -0.003 was wrong.
- `py_compile` clean. No code was modified — this PR only runs an existing, unmodified
  script and commits its durable output.
NEXT: this is a genuinely load-bearing, unresolved discrepancy for the 105 direction —
recommend an operator/Codex decision on how to proceed, e.g.: (a) treat this
reproduction as sufficient grounds to reopen the BULL_CALM-skill question (per A7's own
"if the durable regeneration materially changes A1, the direction is re-opened" clause),
(b) attempt a closer reconstruction of the original injection-test methodology from the
direction-decision doc's prose description before drawing conclusions either way, or
(c) treat both figures as within-noise-of-each-other near zero and proceed with Track A's
conditional-pick-quality test regardless (direction-decision doc §4), since Track A's own
GO/STOP threshold is calibrated to a much larger economic bar (+50bps/yr annualized,
capital-weighted) that a swing between -0.003 and +0.044 IC likely does not change the
qualitative "thin/near-zero, not a book" characterization either way. Do not treat this
PR as having settled the question.
