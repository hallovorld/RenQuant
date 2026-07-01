# wf-gate opt-in placebo-clean difference test + dual-logging

STATUS: delivered (additive, opt-in, OFF BY DEFAULT — live promotion behaviour UNCHANGED)

WHAT: adds a NEW placebo-evaluation mode to `scripts/run_wf_gate.py` §5.2 sanity
battery. The default `absolute` mode is the current ceiling
`|placebo_ic| < max(0.005, 0.5×|aligned_real_ic|)`. The opt-in `difference` mode
instead requires a genuine edge ABOVE the embargo-leakage placebo floor:
`aligned_real_ic - placebo_ic > margin` (pre-registered margin, default +0.01).
The gate always computes BOTH verdicts and dual-logs them (shadow), regardless of
which is authoritative, and stamps both into `wf_gate_metadata`.

WHY/DIR: the `absolute` ceiling is structurally unsatisfiable for the
daily-sampled 60-day-horizon label. The overlapping label is autocorrelated at the
2×horizon gate shift, so the time-shift placebo carries a ~+0.04 persistence floor
that exceeds `0.5×aligned_real_ic` regardless of model quality — it fails nearly
every run (see `doc/research/2026-06-10-m6-placebo-gate-verdict.md`). Real prod
evidence: `placebo_ic=+0.0402`, `aligned_real_ic=+0.0529` → absolute ceiling
`+0.0265` FAILS, yet the genuine edge above the floor is `+0.0127` (> +0.01) — a
real edge the absolute gate rejects. The difference test operationalises the M6
"genuine_ic_floor" (+0.01 starting value) without relaxing anything by default.

SAFETY (this is a LIVE real-money promotion gate):
- ADDITIVE + OPT-IN + OFF BY DEFAULT. Default is unchanged `absolute` mode.
  Selection is only via `strategy_config.wf_gate.placebo_mode = "difference"` or
  `--placebo-mode difference` (CLI overrides config; default None → config →
  `absolute`). No prod config is changed by this PR, so merging does NOT alter
  which models pass on the current config. `absolute`-mode verdict is proven to
  reproduce the historical `pass_placebo` expression bit-for-bit (3000-case random
  sweep + edge cases).
- DUAL-LOGGED SHADOW: every run computes and logs BOTH the absolute and the
  difference verdict (with numbers), and stamps `sanity_placebo_verdicts`
  (absolute + difference) + `sanity_placebo_mode` + `sanity_placebo_difference_margin`
  into `wf_gate_metadata`. This is the evidence to justify flipping the default
  later.
- SCOPE: only the placebo-evaluation logic (new mode + dual-logging + config/CLI
  plumbing) changed. Model weights, manifest, recipe fingerprint, shuffled-label
  gate, regime-IC gate, and every other criterion are untouched.

EVIDENCE: `tests/test_wf_gate_placebo_mode.py` (21 tests, all pass) — absolute mode
reproduces the current verdict (random sweep + NaN/zero edge cases); difference
mode passes iff `real_ic - placebo_ic > margin` (boundary, unavailable-fail-closed,
real-evidence PASS while absolute FAILs); dual-log emits both verdicts + numbers;
config resolver precedence (default/config/CLI-override/bogus-fallback/missing).
Neighbouring `test_wf_gate_relaxation.py` + `test_wf_gate_cli_contract.py` still
green (72 passed together). (5 failures in `test_wf_gate_recipe_scope.py` /
`test_wf_gate_regime_sanity_metadata.py` are pre-existing in the blobless clone —
they need `data/alpha158_291_fundamental_dataset.parquet`, unrelated to this change;
confirmed identical on baseline.)

NEXT: flipping the AUTHORITATIVE default to `difference` is a SEPARATE future
decision. It requires the dual-logged shadow evidence across real weekly runs plus
Codex/operator sign-off — do NOT enable in a prod `strategy_config` without that.
Optional follow-up (documented, not built here): a "widen the embargo so the
placebo shift clears the 60d label window" alternative; the difference test is the
primary mechanism.
