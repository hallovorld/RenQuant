# wf-gate opt-in placebo-clean difference test + dual-logging

STATUS: delivered (additive, opt-in, OFF BY DEFAULT — live promotion behaviour UNCHANGED)

WHAT: adds a NEW placebo-evaluation mode to `scripts/run_wf_gate.py` §5.2 sanity
battery. The default `absolute` mode is the current ceiling
`|placebo_ic| < max(0.005, 0.5×|aligned_real_ic|)`. The opt-in `difference` mode
requires BOTH pre-registered criteria to hold (combined policy):
  (a) a POSITIVE real-IC floor — `aligned_real_ic > real_ic_floor`
      (default +0.01, the M6 genuine_ic_floor positive value); AND
  (b) an incremental edge above the embargo-leakage placebo floor —
      `aligned_real_ic - placebo_ic > margin` (pre-registered margin, default +0.01).
A non-positive / below-floor real IC FAILS regardless of the difference, and any
invalid / non-finite config (margin, floor) or IC input FAILS CLOSED. The gate
always computes BOTH verdicts and dual-logs them (shadow), regardless of which is
authoritative, and stamps both into `wf_gate_metadata`.

REVIEW FIX (PR #422 CHANGES_REQUESTED, Codex): the first cut used criterion (b)
ALONE, which passes a directionally HARMFUL model whenever the placebo is more
negative than a negative real IC (e.g. real −0.01 − placebo −0.03 = +0.02 >
margin). Adding the pre-registered positive real-IC floor (a) closes that hole:
non-positive real IC can never clear a positive floor. The SAME combined policy is
now applied to the per-regime difference diagnostic (shared helper
`_placebo_difference_pass`), and specified-but-invalid config resolves to a
fail-closed NaN instead of silently falling back to the permissive default.

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
- FAIL CLOSED: non-finite margin/floor, non-finite IC inputs, or a
  specified-but-unparseable config value all FAIL (never silently pass). Invalid
  config is not a permissive experiment.
- DUAL-LOGGED SHADOW: every run computes and logs BOTH the absolute and the
  difference verdict (with numbers incl. real vs floor), and stamps
  `sanity_placebo_verdicts` (absolute + difference) + `sanity_placebo_mode` +
  `sanity_placebo_difference_margin` + `sanity_placebo_real_ic_floor` into
  `wf_gate_metadata`. This is the evidence to justify flipping the default later.
- SCOPE: only the placebo-evaluation logic (new mode + combined floor+margin
  policy + dual-logging + config/CLI plumbing) changed. Model weights, manifest,
  recipe fingerprint, shuffled-label gate, and every other criterion are
  untouched. The regime-IC gate is unchanged in `absolute` mode (bit-identical)
  and applies the same combined policy in `difference` mode.

EVIDENCE: `tests/test_wf_gate_placebo_mode.py` (46 tests, all pass) — absolute mode
reproduces the current verdict bit-for-bit (3000-case random sweep + NaN/zero edge
cases); the combined difference policy: negative real IC fails even when diff >
margin (reviewer's exact example), zero real IC fails, barely-positive-below-floor
fails, positive-real + diff>margin passes, floor boundary is strict, negative
placebo does not rescue a negative real IC, sign reversal fails, non-finite
margin/floor/IC fail closed, and specified-but-invalid config resolves fail-closed;
dual-log emits both verdicts + numbers; config resolver precedence + floor
plumbing (default/config/CLI-override). Neighbouring `test_wf_gate_relaxation.py` +
`test_wf_gate_derive_prod_config_env.py` green. (`test_wf_gate_regime_sanity_metadata.py`
needs Python 3.11+ / xgboost + dataset parquet to run the full battery; it fails
identically on the pre-fix baseline in the blobless 3.9 clone — unrelated to this
change.)

NEXT: flipping the AUTHORITATIVE default to `difference` is a SEPARATE future
decision. Per the reviewer, before enabling this mode in any prod
`strategy_config`, PRE-REGISTER the exact untouched evaluation window, the unit of
resampling (dates, NOT rows), minimum dates / regime coverage, an effect-size CI,
and a one-time decision rule. Dual-logged weekly runs over overlapping 60-day
labels are NOT independent replications and must not be counted as such. It also
requires the dual-logged shadow evidence across real runs plus Codex/operator
sign-off — do NOT enable without that. Optional follow-up (documented, not built
here): a "widen the embargo so the placebo shift clears the 60d label window"
alternative; the difference test is the primary mechanism.
