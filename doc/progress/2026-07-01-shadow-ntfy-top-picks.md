# Shadow ntfy: honest top-N recommendation + disambiguate [SHADOW]-broker vs shadow-model labeling

STATUS: revised after Codex CHANGES_REQUESTED round 7 (PR #426)
WHAT: `shadow_scoring.py`'s per-model top-N ntfy recommendation now binds every pick to an
explicit freshness/coverage/fingerprint/provenance-schema admission verdict, defaults every pick
to NOT ACTIONABLE (an explicit opt-in flag is required to ever render one as a recommendation),
evaluates EVERY present cutoff provenance axis independently (worst tier wins; a
present-but-malformed axis fails closed rather than being silently dropped as though absent),
requires at least one code-guaranteed cutoff field to be present before an artifact can ever be
certified fresh, AND (round 7) requires that field combination to resolve to a NAMED, VERSIONED
recipe/schema — not just "some confirmed field, any combination" — persisted in the run_id/audit
trail.
WHY/DIR: operator misread a `[SHADOW]`-broker ntfy as a PatchTST buy recommendation (see Incident
below) and asked for a genuine, confidence-scored recommendation; seven rounds of Codex review
progressively closed provenance-spoofing gaps that would have let a stale, partially-stamped,
malformed, or contract-drifted artifact still render as an actionable pick.
EVIDENCE:
```
artifact:      backtesting/renquant_104/kernel/panel_pipeline/shadow_scoring.py
prod or exp:   experiment (Stage-1 observability-only; default NOT ACTIONABLE; no order path)
existing data: no prior committed `_compute_admission` behavior to regress against beyond this
               PR's own rounds 1-6 (all still-passing tests below); no external oos_mean_ic/
               training_runs baseline applies to this module (it consumes upstream primary/shadow
               scores, it does not train or score independently)
best-known?:   best-available fail-closed provenance gate for this module as of round 7; the
               provenance-schema/recipe binding is scoped to what this file alone can validate
               (see NEXT — a complete cross-repo fix needs training-side stamping this PR does
               not add)
scope:         this is shadow_scoring.py::_compute_admission, experiment/observability-only, vs
               its own round-6 baseline (70 passing tests) — round 7 adds 4 tests (74 total), no
               regression
```
`tests/test_shadow_scoring.py` — 74/74 passed
(`/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest tests/test_shadow_scoring.py -q`);
`tests/test_runner_trade_ntfy.py` run alongside — 132/133 passed, the one failure
(`test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`) is pre-existing and unrelated
(reconfirmed via `git stash`, see Round 5/6 sections below for detail); `py_compile` clean.
NEXT: a preregistered shadow evaluation (fixed historical sessions: coverage sensitivity, top-N
stability, forward net return, turnover/cost, regime slices) is required before
`shadow_experimental_actionable_display` is ever safe to flip on in a live config — no such
evaluation has run yet, so the safe default (NOT ACTIONABLE) stays load-bearing indefinitely
until one does. Separately (round 7): a COMPLETE provenance-schema fix would additionally stamp
an explicit `provenance_schema_version`/`recipe_id` field on artifacts themselves, at training
time (`hf_patchtst_scorer.py`'s `stamp_artifact_metadata` call, `train_production_model.py::
build_artifact`) — this PR derives an equivalent signal purely from which confirmed cutoff fields
are present (validated against a known-combinations map), which is correct for every recipe this
repo currently produces, but a genuine cross-repo stamping change is a larger, separate follow-up.

2026-07-01.

## Incident
Operator received an ntfy titled `[SHADOW]...` with body mentioning `BUY OXY` and misread it as
"the shadow PatchTST model recommends OXY". Actually: `[SHADOW]` in the TITLE meant "this run
executed via the readonly/shadow BROKER" (`broker.broker_name == "alpaca_shadow"`), completely
independent of which scoring model was primary — the BUY OXY decision was made by the PRODUCTION
XGB model, echoed through the readonly broker. The body's `SHADOW[name]` diagnostic segment
(comparing a DIFFERENT model, e.g. PatchTST, against whatever primary decided) is not itself an
actionable per-ticker recommendation. PatchTST's own view of OXY that day was rank 15/83,
z≈+0.88 — not a top pick, and the operator had no way to see this from the ntfy message.

Operator mandate (verbatim): "shadow的message应该给出带有信心指数的推荐，因为如果我觉得shadow给
的好，我也会下单的" — the shadow ntfy should give a genuine, actionable recommendation with a
confidence indicator, because the operator may act on the shadow model's own pick independently
of primary's decision.

## Changes

1. **Top-N recommendation list** (`backtesting/renquant_104/kernel/panel_pipeline/shadow_scoring.py`).
   Extracted the existing per-shadow-model summary assembly (previously inline inside
   `ApplyShadowScoringTask.run`) into a pure function `_compute_shadow_summary(...)` — same
   behavior, but now unit-testable without mocking MLflow/the model registry/scorer loading.
   Extended it to compute a top-N (default 5, configurable via
   `ranking.panel_scoring.shadow_top_n_picks`, constant `_DEFAULT_TOP_N_PICKS`) picks list per
   shadow model, reusing the SAME score/rank arrays already used for `top10_overlap` /
   `spearman_vs_primary` — no re-scoring. Each pick carries `ticker`, `shadow_score`,
   `shadow_rank`, `shadow_percentile`, `shadow_zscore` (all RELATIVE to today's scored universe —
   there is no fitted probability calibrator for shadow scores, so no probability is ever
   fabricated), `in_primary_topN` (computed here — primary ranks are already available), and
   `in_primary_admitted` (left `None` here — `ApplyShadowScoringTask` runs inside
   `PanelScoringJob`/Phase 3, BEFORE `RankingJob`/`SelectionJob` populate `ctx.orders`, so "did
   primary actually buy this" is not determinable at this point in the pipeline).

2. **ntfy body** (`live/runner.py`, `_notify_decision`). Added a new `SHADOW-PICKS[name]: ...`
   line directly after the existing `SHADOW[name] top3=... top10∩prim=.../10 ρ=... n=...` line
   (kept byte-for-byte, backward compat). New line renders each pick as
   `TICKER(rank R/N, z=+X.XX[, ALSO-BOUGHT])`, always suffixed with
   `[relative rank, not a validated confidence score]`. `ALSO-BOUGHT` is computed at ntfy-render
   time (not in shadow_scoring.py) from `ctx.orders_placed`, since the full pipeline +
   `adapter.commit` have run by then — this is where `in_primary_admitted` actually becomes
   determinable. Also added `_truncate_ntfy_body` / `_NTFY_BODY_MAX_BYTES` (3800): no length-budget
   guard existed before; the new segments (one per configured shadow model) made an unbounded
   body more likely, so bodies are now explicitly UTF-8-safely truncated with a `…[truncated]`
   marker instead of risking a silent cut by ntfy's own ~4096-byte transport limit.

3. **Title disambiguation.** Renamed the readonly-broker ntfy title prefix from `[SHADOW]` to
   `[READONLY]` (`live/runner.py`, `is_shadow = label.startswith("[READONLY]")`). Chose **Option
   A** (rename outright) after a repo-wide grep for the literal `[SHADOW]` title substring found
   exactly one functional consumer — `live/runner.py`'s own `is_shadow` check, i.e. no external
   log parser/dashboard/alerting rule depends on it. Updated the one existing test that hardcoded
   the old literal (`tests/test_runner_trade_ntfy.py::test_shadow_exit_is_marked_hypothetical_not_live_trade`)
   and updated descriptive (non-functional) comments in `scripts/daily_104.sh` and the
   `_2026-05-19_shadow_purpose` doc field in `strategy_config.shadow.json` /
   `strategy_config.sim_patchtst_clean_20260522.json` for consistency.

## Tests
- `tests/test_shadow_scoring.py`: new `TestComputeShadowSummaryTopPicks` (hand-verified fixture —
  10 tickers, rank/percentile/z-score cross-checked against an independent numpy computation;
  `in_primary_admitted` always `None`; `in_primary_topN` verified against a deliberately
  differently-ranked primary fixture) + `TestApplyShadowScoringTaskUsesExtractedHelper` (source
  contract: `run()` calls the extracted helper, top-N is configurable). 22/22 passed.
- `tests/test_runner_trade_ntfy.py`: new `TestShadowTopPicksNtfy` (SHADOW-PICKS rendering, honest
  relative-rank label present, no fabricated `"NN% confidence"` wording, legacy line unchanged,
  ALSO-BOUGHT overlay correct, multi-model + missing-top_picks edge cases),
  `TestNtfyBodyLengthBudget` (truncation helper unit tests + end-to-end bounded-body test),
  `TestTitlePrefixDisambiguation` (`[READONLY]` triggers shadow-broker behavior, stale `[SHADOW]`
  no longer does, title token distinct from body segment labels). 56/57 passed — the one failure
  (`test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`) is a **pre-existing failure on
  `main`**, unrelated to this change (confirmed via `git stash` — `scripts/live_only_104.sh` was
  already refactored to a thin `exec` wrapper and that one test wasn't updated).
- Also ran `tests/test_panel_scoring_job.py -k shadow-or-thirteen`,
  `tests/test_round3_audit_fixes_2026_04_25.py`, `tests/test_audit_2026_04_24_fixes.py`,
  `tests/test_runner_preflight_fail_closed.py` — 4 pre-existing failures there too, confirmed
  identical on `main` via `git stash`, none touch shadow_scoring.py or the runner.py sections
  changed here.
- Did **not** run the full repo test suite (large, needs live data fixtures / broker
  credentials unrelated to this change) — ran the targeted files above plus a compile check
  (`py_compile`) on all touched Python files and `git diff --check` (clean).

## Scope
Observability-only. No order-placement, gate, or primary/shadow-selection logic touched — the
new `in_primary_admitted`/`in_primary_topN`/`ALSO-BOUGHT` fields are read-only overlays computed
from already-decided state (`ctx.orders_placed`), never fed back into any decision.

## Round 2 (Codex CHANGES_REQUESTED on PR #426)

Codex's review flagged that the round-1 change above still crossed the observability-only
boundary in practical effect: it presented raw shadow ranks as an actionable line for an operator
who may manually place capital, without saying whether the shadow artifact/data were fresh or the
scored universe complete. Real known example: PatchTST has been confirmed ~140 days stale in this
codebase; rank 1 of an 83-name censored subset is not comparable to rank 1 of the intended
~292-name watchlist. A z-score of a stale/degenerate model's raw outputs is not a predictive
confidence measure. Separately, `shadow_percentile = rank / n * 100` gave the BEST-ranked name the
LOWEST percentile (rank 1 of 83 → 1.2) — backwards from the conventional "higher percentile =
better" reading.

### Changes

1. **Admission verdict** (`shadow_scoring.py`, new `_compute_admission` + `_freshness_tier`).
   Every `_compute_shadow_summary` call now binds its `top_picks` to:
   - `verdict`: `healthy` / `warn` / `escalate` / `breach` / `unknown`, bucketed from
     `age_days = as_of_date - artifact_meta["trained_date"]` at thresholds warn=28d/escalate=33d/
     breach=35d. Vocabulary mirrors (does not import — separate repo/package)
     renquant-orchestrator's `model_freshness_monitor.py` tier naming + its `SHADOW_POLICY`
     cadence, so an operator who already reads that monitor's alerts sees the same words here.
     Missing/unparseable `trained_date` is `unknown` — fail-closed, never treated as fresh
     (mirrors this repo's own P-MODEL-STALENESS convention: missing provenance is a fail).
   - `coverage` = `n_scored / n_expected` (`n_expected` = `len(config["watchlist"])`; field naming
     mirrors this repo's own `n_have`/`n_expected` preflight convention). `n_expected<=0`
     (watchlist not configured/available) degrades to `coverage=None` rather than a false pass
     or false fail. Default floor `_DEFAULT_MIN_COVERAGE=0.80`, configurable via
     `ranking.panel_scoring.shadow_min_coverage`.
   - `actionable` = verdict in `{healthy, warn}` AND coverage (if known) `>= min_coverage`.
   - `run_id` = `f"{as_of_date}:{name}:{artifact_fingerprint[:12]}"`, `artifact_fingerprint` /
     `trained_date` sourced from `scorer.metadata` (stamped by every registered scorer kind via
     `panel_scorer.stamp_artifact_metadata` at load time — no new plumbing needed).
   - This deliberately does **not** re-check feature-*data* freshness: primary and shadow score
     the same `ctx._panel_matrix`/`panel_history`, already gated upstream by
     `DataFreshnessGateTask` (`kernel/pipeline/task_data_freshness.py`) before Phase 3
     (`PanelScoringJob`) runs. What's new here is shadow-*artifact*-specific: the shadow model's
     own retrain cadence (primary has one; a shadow artifact had none) and its own scored-universe
     coverage (shadow feature_cols/seq_len differ from primary and censor a different subset).
   - `top_picks` are still fully computed even when `actionable=False` — the gate is enforced by
     the ntfy *renderer*, not by discarding the audit trail (`ctx._shadow_summary` / MLflow keep
     the raw ranks either way).

2. **ntfy body gate** (`live/runner.py`, `_notify_decision`). When `admission.actionable` is
   `False` (or the `admission` key is missing entirely — a fail-closed default for e.g. a stale
   cached ctx from before this fix), the `SHADOW-PICKS[name]` line renders
   `NOT ACTIONABLE (<reasons>) [verdict=... run=...]` instead of any per-ticker ranks — the raw
   picks are never shown as if they were current. When actionable, the line is prefixed
   `[verdict cov=n_scored/n_expected run=run_id]` before the existing `TICKER(rank R/N, z=...)`
   list, binding the picks to their provenance inline. The legacy `SHADOW[name] top3=...`
   diagnostic line (round 1) is untouched — it never claimed actionability, so it isn't gated.

3. **`shadow_percentile` direction fixed** (`shadow_scoring.py`). Changed
   `rank / n_universe * 100` to `(n_universe - rank + 1) / n_universe * 100` so rank 1 (best) reads
   as the 100th percentile and worse ranks read lower — the conventional interpretation. Not
   renamed (`shadow_percentile` kept) since the direction is now unambiguous; the field is not
   itself rendered in the compact ntfy line (only rank + z-score are).

4. **Deferred experiment endpoint (explicitly not built here).** This PR still only presents raw
   diagnostic ranks with honest freshness/coverage caveats. It does not attempt to answer "is the
   shadow model's own top-N pick actually good." That question needs a pre-registered experiment:
   shadow top-N forward net return vs. a pre-registered primary/benchmark, common (uncensored)
   universe, fixed holding horizon, real transaction costs, a minimum number of sessions before
   any read, and no cherry-picking among models or N after the fact. Deferred, not built.

### Tests
- `tests/test_shadow_scoring.py`: new `TestFreshnessTier` (tier boundary values, `unknown` on
  missing/NaN age), `TestComputeAdmission` (healthy+full-coverage actionable; the real 140d-stale
  incident → `breach`/not actionable; the real 83/292 censored-universe incident → coverage fail/
  not actionable even though the artifact itself is fresh; missing `trained_date` → `unknown`/not
  actionable; `n_expected<=0` degrades gracefully; coverage fraction computed + surfaced; run_id
  format), `TestComputeShadowSummaryAdmissionIntegration` (admission/actionable/run_id surfaced on
  the summary dict; top_picks still computed when not actionable; omitted admission kwargs default
  fail-closed). Updated `TestComputeShadowSummaryTopPicks` percentile fixture + `_call` helper for
  the fixed direction + new keyword args. 42/42 passed.
- `tests/test_runner_trade_ntfy.py`: new `TestShadowPicksAdmissionGate` (stale-artifact picks NOT
  ACTIONABLE and ranked breakdown absent; incomplete-coverage picks NOT ACTIONABLE with the
  coverage fraction surfaced; missing `admission` key defaults NOT ACTIONABLE; actionable case
  surfaces verdict/coverage/run_id; no confidence/recommendation wording in the NOT ACTIONABLE
  body; legacy `SHADOW[name]` line unaffected). Updated `_shadow_summary_entry` fixture to default
  to an actionable admission verdict so the round-1 rendering tests keep exercising the actionable
  path unchanged. 62/63 passed — the one failure
  (`test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`) is the same pre-existing
  failure on `main` documented in round 1 above (confirmed again via `git stash`).
- Re-ran `tests/test_panel_scoring_job.py`, `tests/test_round3_audit_fixes_2026_04_25.py`,
  `tests/test_audit_2026_04_24_fixes.py`, `tests/test_runner_preflight_fail_closed.py` — same 4
  pre-existing failures as round 1, confirmed identical via `git stash` (none touch the files
  changed in this round either).
- `py_compile` on all touched files + `git diff --check` clean. Did not run the full repo suite
  (same reason as round 1: live data fixtures / broker credentials unrelated to this change).

### Scope
Still observability-only. The admission gate only decides whether the ntfy body *presents* the
already-computed picks — it does not feed into order placement, primary/shadow selection, or any
gate. `_compute_admission` is a pure function (no I/O beyond the `scorer.metadata` dict already
loaded by the existing scorer-loading path).

## Round 3 (Codex #426 review point 1: "trained cutoff" AND "feature-data cutoff" named separately)

Round 2's `_compute_admission` computed the freshness age from `artifact_meta["trained_date"]`
alone. That is run time, not a data-freshness axis — a fresh `trained_date` over stale/absent
DATA must never certify freshness. This codebase already hit exactly that bug class once
(2026-06-15 "model stale-by-split-recipe": a live model looked freshly-trained but was keyed to a
stale val-tail cutoff). Codex's review named "trained cutoff" and "feature-data cutoff" as two
separate provenance items to bind — round 2 only bound the first.

Concretely: `hf_patchtst_scorer.py` (the real PatchTST-shadow path this incident is about) already
stamps `effective_train_cutoff_date` into `scorer.metadata` at load time (via `_coalesce` from the
checkpoint / training contract / legacy sidecar) — that data was already available and unused.

### Changes
1. **`_DATA_CUTOFF_FIELDS` + `_binding_cutoff`** (`shadow_scoring.py`): binding DATA-cutoff field
   priority, most-binding first (`label_observation_cutoff`, `effective_selection_cutoff_date`,
   `effective_train_cutoff_date`, `data_cutoff_date`, `live_train_end`, `cutoff_date`) — mirrors
   the orchestrator's `model_freshness_monitor.py` `DATA_CUTOFF_FIELDS` order.
2. **`_compute_admission` now prefers the binding cutoff over `trained_date`.** `trained_date` is
   used ONLY as a last-resort fallback when no binding cutoff field is present in `artifact_meta`.
   The verdict's `reasons` now name which cutoff field/value drove the age
   (`effective_train_cutoff_date=2024-11-13`, etc.) instead of only `trained_date`. New output
   keys `binding_cutoff` / `binding_cutoff_field` (additive — existing keys unchanged).
3. **Look-ahead guard** (`_freshness_tier`): a NEGATIVE age (a cutoff later than `as_of_date`) now
   fails closed to `breach` instead of silently reading `healthy` — mirrors the orchestrator's own
   look-ahead guard (its docstring cites this as a real prior lesson, PR #211).
4. **Wording** (`live/runner.py`, review point 3 "stop calling the line a recommendation or
   confidence"): the actionable-path trailing tag changed from `[relative rank, not a validated
   confidence score]` (which still used the word "confidence") to `[raw rank (unvalidated, see
   freshness verdict)]`. The stale round-1 comment block describing this as a "genuine, actionable
   recommendation with an HONEST confidence indicator" was also corrected — that framing is
   exactly what rounds 2/3 walked back.

### Tests
- `tests/test_shadow_scoring.py`: new `TestComputeAdmissionBindingDataCutoff` — binding cutoff
  preferred over a recent `trained_date` when the underlying data is old (the real PatchTST shape:
  retrained recently, `effective_train_cutoff_date` ~596d stale → still `breach`); DATA_CUTOFF_FIELDS
  priority order; `trained_date` fallback only when no binding field present; unparseable binding
  field falls back to `trained_date`; look-ahead cutoff → `breach`; no cutoff and no `trained_date`
  → `unknown`. All prior `TestComputeAdmission`/`TestFreshnessTier` cases pass UNCHANGED (they only
  ever passed `trained_date`, so `_binding_cutoff` returns `(None, None)` for them and the fallback
  path is identical to round 2's behavior). 48/48 passed in `test_shadow_scoring.py`.
- `tests/test_runner_trade_ntfy.py`: updated the actionable-path wording assertion + added a
  `"recommend" not in body.lower()` check. 62/63 passed — the one failure
  (`test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`) is the same pre-existing
  failure confirmed via `git stash` in rounds 1 and 2.
- Re-ran `tests/test_panel_scoring_job.py`, `tests/test_round3_audit_fixes_2026_04_25.py`,
  `tests/test_audit_2026_04_24_fixes.py`, `tests/test_runner_preflight_fail_closed.py` — same 6
  pre-existing failures as before this round (confirmed unrelated: none touch the files changed
  here).
- `py_compile` on all touched files + `git diff --check` clean. Did not run the full repo suite
  (same reason as rounds 1-2).

### Scope
Still observability-only — same as rounds 1-2. This round only changes WHICH provenance field the
admission verdict is keyed on (and adds a look-ahead guard); it does not add new gates, new I/O, or
touch order placement / primary selection.

## Round 4 (Codex CHANGES_REQUESTED — 4 fail-closed gaps + a SCOPE NARROWING)

Round 3 fixed field *precedence* (bind the DATA cutoff over `trained_date` when both are present)
but left four fail-closed gaps, and — the main ask of this round — established that the
freshness/coverage thresholds throughout this feature are **unvalidated operational guesses**, not
empirically-grounded bands. Per the review: "Until a preregistered shadow evaluation establishes
minimum coverage and freshness bands on fixed sessions with costs/top-N stability, keep all picks
explicitly NOT ACTIONABLE by default. A config flag may enable experimental display, but the safe
default cannot authorize discretionary capital from warn-tier/raw-rank output."

### The four fail-closed gaps (all fixed, `shadow_scoring.py::_compute_admission`)

1. **GAP 1 — no more `trained_date` fallback.** Round 3 still fell back to `trained_date` when it
   was the *only* field present (no binding DATA cutoff at all) — that reopened the exact
   stale-data spoof round 3 closed for the "both present" case: a fresh `trained_date` over
   genuinely stale/absent DATA. Now a missing/unparseable binding cutoff is `unknown`, full stop.
   `trained_date` is still returned in the `trained_date` field for DISPLAY (process-liveness
   context) only — never used to compute an age or certify actionability.
2. **GAP 2 — `n_expected<=0` now BLOCKS.** Previously degraded to `coverage=None`/"does not
   block" (an unknown/unresolvable universe denominator silently passed). Now `coverage_ok=False`
   and the admission fails closed, surfaced via a `n_expected=... unknown/unresolvable universe
   size` reason.
3. **GAP 3 — missing artifact fingerprint now BLOCKS.** Previously produced `run_id` keyed on a
   `"nofingerprint"` sentinel and proceeded. Immutable artifact identity is mandatory for an
   actionable verdict now — a missing/placeholder fingerprint fails `fingerprint_ok` and blocks.
4. **GAP 4 — horizon-aware age compensation for `label_observation_cutoff`.** For a
   fwd-N-session-label model, a causally valid `label_observation_cutoff` is intentionally
   horizon-lagged (the label needs N sessions forward to be observed, so even a same-day retrain's
   cutoff sits ~N business days behind the raw data frontier by construction) — comparing that RAW
   age directly against the short-window freshness threshold wrongly marked genuinely fresh
   artifacts stale. Reused the horizon-aware age-compensation PATTERN already merged in
   `renquant-orchestrator`'s `model_freshness_monitor.py` this same session (`#423`/`#213` round —
   `_subtract_business_days` / `_expected_lag_calendar_days`, ported here since this module lives
   in a separate Python package/deploy unit, not imported). The RAW `age_days` is never mutated;
   a new, separately-persisted `horizon_compensated_age_days` field (plus `horizon_lag_days`) is
   what the freshness tier is judged against. The lag is keyed on the model's own stamped
   `artifact_meta["lookahead_days"]` when present, falling back to the documented PatchTST fwd_60d
   convention otherwise, and is scoped ONLY to the `label_observation_cutoff` binding field (a
   different binding field with the same stamped `lookahead_days` gets zero compensation — verified
   by `TestHorizonCompensation::test_compensation_scoped_to_label_observation_cutoff_only`). A
   look-ahead cutoff (later than `as_of_date`) is checked against the RAW age *before* any
   compensation and still fails closed regardless of horizon compensation.

### The scope narrowing (main ask)

`_compute_admission` now computes two distinct outcomes:
- `gates_passed` — every fail-closed gate above passing (freshness tier including GAPs 1/4,
  coverage including GAP 2, fingerprint including GAP 3). This is "would be actionable if surfaced."
- `actionable` = `gates_passed AND experimental_actionable_display`. The new opt-in config flag
  `ranking.panel_scoring.shadow_experimental_actionable_display` (constant
  `_DEFAULT_EXPERIMENTAL_ACTIONABLE_DISPLAY = False`, read by `ApplyShadowScoringTask.run` and
  threaded through `_compute_shadow_summary`) defaults OFF. **With the flag unset (the default),
  `actionable` is always `False` — even when every gate passes** — and the admission `reasons` gain
  an explicit "NOT ACTIONABLE by default pending a preregistered shadow evaluation..." entry so an
  operator reading the ntfy body sees *why*. The flag can only ever RAISE an already-`gates_passed`
  verdict to actionable; it is computed identically regardless of the flag, so it never bypasses a
  failed gate (`gates_passed` alone already tells you that).

This converges the feature to the same "Stage-1 operations-only, no execution-quality claim until
preregistered validation" discipline already established this session for the renquant105
architecture (RFC #212 / orchestrator `model_freshness_monitor.py` design doc): observe/collect
first, claim actionability only after validated evidence — cited explicitly as the design rationale
here, not an arbitrary restriction.

**ntfy rendering change** (`live/runner.py`, `_notify_decision`): because `actionable` is now
`False` for effectively every cycle by default, fully suppressing the ranked ticker breakdown in
the NOT ACTIONABLE branch (rounds 2-3's behavior) would make the entire feature permanently dark —
defeating the PR's original observability-only intent ("want to know what shadow will do"). The
diagnostic rank/z-score list is now **always** rendered, in both branches; only the label and
trailing tag differ (`NOT ACTIONABLE (<reasons>) [verdict=... cov=... run=...] <picks>
[diagnostic rank only, not actionable]` vs. the actionable-path `[verdict cov=... run=...] <picks>
[raw rank (unvalidated, see freshness verdict)]`). The gate is still fully enforced — the ranks are
labeled diagnostic-only, never presented as a pick, in the default (and by far most common) case.

### Tests
- `tests/test_shadow_scoring.py`: rewrote `TestComputeAdmission` fixtures to use a binding DATA
  cutoff + fingerprint (GAP 1/3 fixtures no longer rely on `trained_date` alone) and an explicit
  `experimental_actionable_display=True` wherever a test needs to reach the actionable outcome;
  added default-flag-off tests (`test_healthy_full_coverage_without_flag_is_not_actionable_by_default`),
  GAP 2 tests (`test_zero_expected_universe_blocks_not_actionable`,
  `test_negative_expected_universe_blocks_not_actionable`), GAP 3 test
  (`test_missing_fingerprint_blocks_even_when_flag_enabled`). Renamed/flipped the two
  `TestComputeAdmissionBindingDataCutoff` tests that previously asserted the now-removed
  `trained_date` fallback (`test_no_binding_cutoff_is_unknown_trained_date_never_used_gap1`,
  `test_unparseable_binding_cutoff_field_is_unknown_not_trained_date_fallback`). New
  `TestHorizonCompensation` (GAP 4): 60d/20d horizon fresh-retrain reads healthy, default-lookahead
  fallback, genuine staleness still breaches despite widening, compensation scoped only to
  `label_observation_cutoff`, future-cutoff still fails closed regardless of compensation, raw vs.
  horizon-compensated age persisted as separate fields, full actionable-with-flag pipeline test.
  Extended `TestComputeShadowSummaryAdmissionIntegration` with the flag-matrix (gates-pass-but-flag-off,
  flag-on-and-gates-pass, flag-on-but-coverage-gate-fails, flag-on-but-fingerprint-gate-fails).
  New `test_experimental_actionable_display_flag_is_wired` source-contract test. 64/64 passed.
- `tests/test_runner_trade_ntfy.py`: flipped the three `TestShadowPicksAdmissionGate` tests that
  previously asserted the ranked breakdown was ABSENT when not actionable
  (`test_stale_artifact_picks_are_not_actionable_but_ranks_still_shown`,
  `test_incomplete_coverage_picks_are_not_actionable_but_ranks_still_shown`,
  `test_missing_admission_field_defaults_to_not_actionable_but_ranks_still_shown`) — now assert the
  ranks ARE present (round 4 observability requirement) alongside the NOT ACTIONABLE label/reasons.
  Fixed the not-actionable trailing tag wording (`[diagnostic rank only, not actionable]` — avoided
  "recommend*" per the existing `test_not_actionable_body_has_no_confidence_or_recommendation_wording`
  check). 63/64 passed — the one failure
  (`test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`) is the same pre-existing failure
  documented in rounds 1-3 (confirmed again via `git stash`).
- Re-ran `tests/test_panel_scoring_job.py`, `tests/test_round3_audit_fixes_2026_04_25.py`,
  `tests/test_audit_2026_04_24_fixes.py`, `tests/test_runner_preflight_fail_closed.py` — same 6
  pre-existing failures as round 3, confirmed identical via `git stash` (none touch the files
  changed here).
- `git diff --check` clean.
- Environment note: this round's local venv installed `xgboost`/`scipy`/`pandas`/`numpy`/`pytest`/
  `pytest-xdist`/`pytest-env` fresh via `uv` (the checked-in `requirements.lock.txt` pins
  conda-build `file://` paths that don't resolve outside that original build host) plus
  `mlflow==3.12.0` pinned to match this module's own code-comment ("mlflow 3.12.0 already
  installed") — newer mlflow (3.14) defaults to rejecting the filesystem tracking backend used by
  `TestMLflowSetup`/`TestLogShadowRun`, unrelated to this change.

### Scope
Still observability-only. The default now renders every pick NOT ACTIONABLE — a strictly SAFER
default than rounds 1-3, not a new capability. No order-placement, gate, or primary/shadow-selection
logic touched; `_compute_admission` remains a pure function.

## Round 5 (Codex CHANGES_REQUESTED — two provenance gaps in the experimental opt-in)

CI/focused tests green (126 passed, one documented pre-existing failure), round 4's four gates
accepted. Two remaining ways the experimental opt-in could certify unsupported freshness:

1. **Missing label horizon was guessed, not failed closed.** `_expected_lag_calendar_days` silently
   assumed 60 business days whenever `lookahead_days` was missing/unparseable/non-positive — this
   module supports more than one shadow-model kind and cannot prove every artifact uses the
   documented PatchTST fwd-60d convention, so guessing could subtract ~84 calendar days off a
   genuinely unknown-horizon artifact and turn it `healthy`/`gates_passed` (actionable under the
   opt-in flag).
2. **Freshness bound to the first parseable cutoff, not every required axis.** `_binding_cutoff`
   prioritized `label_observation_cutoff` and stopped — a compensated-recent label cutoff could hide
   an older `effective_selection_cutoff_date`/`effective_train_cutoff_date` present on the SAME
   artifact.

### Changes (`shadow_scoring.py::_compute_admission` and helpers)

- **GAP 4b — horizon compensation requires a STAMPED positive horizon.** `_expected_lag_calendar_days`
  now returns `(compensation_lag, diagnostic_lag)`: `compensation_lag` is `None` whenever
  `label_observation_cutoff` needs compensation but `lookahead_days` is missing/invalid — the
  caller (`_axis_verdict`) then forces that axis to `unknown`, never silently applying the
  documented 60-BD default to certify admission. `diagnostic_lag` is ALWAYS computed (using the
  default when the stamped value is missing) and still surfaced via `horizon_lag_days` purely for
  troubleshooting ("what the lag would have been") — never fed into `horizon_compensated_age_days`
  or the tier when the real value is unknown.
- **Worst-axis-wins, not first-axis-wins.** `_binding_cutoff` (single-field priority pick) is
  replaced by `_all_present_cutoffs` (every present `_DATA_CUTOFF_FIELDS` entry, priority order) +
  `_axis_verdict` (one axis's independent age/tier) + a `_TIER_SEVERITY` ranking
  (`healthy < warn < escalate < breach < unknown`). `_compute_admission` evaluates EVERY present
  axis and binds `binding_cutoff`/`binding_cutoff_field`/`age_days`/`horizon_compensated_age_days`/
  the tier to the WORST one; ties between axes in the same tier still favor the more-binding field
  by the original priority order (stable `max()` + priority-ordered iteration). A new
  `cutoffs_evaluated` list in the returned dict persists every present axis's own verdict — not
  just the one that decided the overall outcome — for audit ("which axes did we actually see, and
  what did each say"), and any secondary off-SLA axis gets its own `reasons` entry even when it
  didn't win the `max()`.
- Deliberately did NOT build a per-model-kind "required axis" registry (the review's fallback
  suggestion "if fields are alternatives for a specific recipe, that mapping must be explicit and
  fingerprinted per model kind"): this module currently ships exactly one shadow-model recipe
  (PatchTST), for which every `_DATA_CUTOFF_FIELDS` entry is a legitimate independent freshness axis
  when present — "evaluate every present axis, worst wins" is already correct with no per-recipe
  ambiguity to resolve today. Documented in-code as the point to introduce that registry once a
  second shadow-model kind with genuinely different axis semantics actually ships — building it now
  would be speculative structure with nothing real to validate it against (same discipline as
  [[over-engineering-validation-before-alpha]]).

### Tests
- `TestHorizonCompensation::test_missing_lookahead_fails_closed_not_guessed` (renamed/flipped from
  `test_default_lookahead_used_when_not_stamped`, which asserted the now-removed guessed-default
  behavior): missing `lookahead_days` → `verdict=unknown`, `gates_passed=False`,
  `horizon_compensated_age_days=None`, `horizon_lag_days` still shows the diagnostic default,
  `reasons` names GAP 4b explicitly.
- `TestComputeAdmissionBindingDataCutoff::test_data_cutoff_field_priority_breaks_ties_when_severities_match`
  (renamed/flipped from `test_data_cutoff_field_priority_matches_orchestrator_order`, whose original
  scenario — a stale `effective_train_cutoff_date` present alongside a fresh
  `label_observation_cutoff` — is now CORRECTLY `breach`, not `healthy`; the old assertion described
  exactly the bug this round fixes): priority now only decided ties between axes in the SAME tier.
- New `test_worst_axis_wins_even_when_it_is_not_the_most_binding_field`: the literal scenario from
  the review quote — a horizon-compensated-healthy `label_observation_cutoff` alongside a genuinely
  ancient `effective_train_cutoff_date` (2020-01-01) — asserts the verdict binds to the stale axis
  (`breach`, `gates_passed=False`), while `cutoffs_evaluated` still shows the healthy axis was seen
  and evaluated, not silently dropped.
- `tests/test_shadow_scoring.py` — 65/65 passed (`/Users/renhao/git/github/RenQuant/.venv/bin/python
  -m pytest tests/test_shadow_scoring.py -q`). `tests/test_runner_trade_ntfy.py` run alongside — same
  63/64 with the one pre-existing `test_live_only_wrapper_does_not_duplicate_runner_success_ntfy`
  failure as rounds 1-4 (reconfirmed via `git stash`, fails identically pre- and post- this round's
  diff — the wrapper script it checks is untouched by this change). `py_compile` clean.

### Scope
Unchanged: still observability-only, default NOT ACTIONABLE, `_compute_admission` remains a pure
function. This round only tightens WHICH artifacts can ever reach `gates_passed=True` under the
experimental opt-in — it does not add or remove a code path.

## Round 6 (Codex CHANGES_REQUESTED — one fail-open axis + missing required-provenance semantics)

Round 5's two fixes accepted for VALID inputs. Two remaining gaps in the same provenance surface,
plus a mechanical progress-doc schema fix:

1. **`_all_present_cutoffs()` silently dropped a present-but-unparseable field.** If, e.g.,
   `label_observation_cutoff="bad"` was present while `effective_train_cutoff_date` was recent and
   parseable, the malformed required axis simply disappeared from evaluation — indistinguishable
   from never having been stamped at all — and the artifact could still become `gates_passed`
   under the experimental flag on the strength of the healthy secondary axis alone. "A malformed
   provenance axis is not equivalent to an absent optional axis" (review).
2. **No required-axis semantics — only axes that happen to be present are ever evaluated.** An
   artifact presenting ONLY a weak/legacy cutoff field (e.g. `cutoff_date`), with NEITHER of this
   repo's two actually-guaranteed cutoff fields present at all, could still certify `healthy` on
   the strength of that one weak field.

### Investigation that corrected round 5's own framing

Round 5's comment claimed this module "ships exactly ONE shadow-model recipe (PatchTST)" as the
justification for not building a per-recipe required-axis registry. That claim was **not actually
verified** and turned out to be wrong: `model_registry.py` registers FOUR kinds
(`xgb`/`patchtst`/`hf_patchtst`/`regime_router`), and the **live production** shadow-model config
(`strategy_config.json`/`strategy_config.golden.json` — the actual config behind the `[SHADOW]...BUY
OXY` incident that started this whole PR) configures the shadow slot as `kind="xgb"`, **not
PatchTST**. So the fallback plan sketched in round 5 ("introduce a per-model-kind mapping once a
second kind ships") was already overdue on the facts.

Reading the ACTUAL stamping code (rather than assuming) across both live paths:

- `hf_patchtst_scorer.py`'s `stamp_artifact_metadata` call ALWAYS stamps
  `effective_train_cutoff_date` (from ckpt/contract/sidecar, `_coalesce`) for hf_patchtst/patchtst
  artifacts.
- `train_production_model.py::build_artifact` ALWAYS stamps `label_observation_cutoff`
  (`train["date"].max()`) on BOTH the walk-forward and full-history XGB paths, and
  `effective_train_cutoff_date` additionally on the walk-forward path (Codex #423, merged
  2026-07-01T22:09:36Z — the SAME day as this PR, a genuinely load-bearing companion fix).
- No other `_DATA_CUTOFF_FIELDS` entry (`effective_selection_cutoff_date`, `data_cutoff_date`,
  `live_train_end`, `cutoff_date`) has an equivalent confirmed, code-guaranteed stamping contract
  anywhere in this repo as of this round.

Given that, a per-"kind" mapping would have been the WRONG shape anyway (the walk-forward/
full-history split inside "xgb" alone already produces two different stamped fields, and
`_compute_admission` doesn't even receive `kind`). The required-axis check implemented below is
keyed on the two CONFIRMED fields directly, not on kind — simpler, more robust, and grounded in
what the training code actually does rather than a guess about "the sole recipe."

### Changes (`shadow_scoring.py`)

- `_all_present_cutoffs` → renamed `_all_present_cutoff_fields`: now returns every present
  (truthy) `_DATA_CUTOFF_FIELDS` RAW value, parsed or not — parsing moved into `_axis_verdict`,
  which returns a proper UNKNOWN axis result (never a silent drop) when a value fails to parse.
- New `_CONFIRMED_STAMPED_CUTOFF_FIELDS = ("label_observation_cutoff",
  "effective_train_cutoff_date")`. `_compute_admission` now additionally requires at least one of
  these to be among the axes actually present (whether or not it parsed — an attempted-but-
  malformed stamp still counts as "attempted", already handled by its own UNKNOWN axis result);
  when neither is present, the overall tier is forced to `unknown` regardless of what a weaker
  field says, with a distinct `reasons` entry naming the gap explicitly.
- `cutoffs_evaluated` entries now allow `"cutoff": null` for a malformed axis (previously assumed
  every evaluated axis had successfully parsed a date).

### Tests (`tests/test_shadow_scoring.py`) — new `TestMalformedCutoffAxesAndRequiredProvenance`
- `test_malformed_high_priority_axis_with_healthy_secondary_is_unknown` — malformed
  `label_observation_cutoff` + healthy `effective_train_cutoff_date` → unknown (the exact "mixed
  high-priority + healthy secondary" scenario the review named).
- `test_healthy_compensated_label_with_malformed_secondary_is_unknown` — healthy compensated
  `label_observation_cutoff` + malformed `effective_train_cutoff_date` → unknown (the review's
  second named scenario).
- `test_future_axis_with_malformed_axis_is_unknown_not_breach` — a look-ahead BREACH axis +
  a malformed UNKNOWN axis → unknown wins over breach, not just over healthy (the review's third
  named scenario; also proves `_TIER_SEVERITY`'s `unknown > breach` ordering end-to-end).
- `test_weak_field_alone_cannot_certify_without_a_confirmed_axis` — only `cutoff_date` present,
  fresh, and parseable → still unknown; the axis itself reads `healthy` in `cutoffs_evaluated` but
  the overall verdict does not, proving the required-axis check is a SEPARATE gate from per-axis
  freshness.
- `test_confirmed_axis_present_alongside_weak_field_certifies_normally` — sanity check the new
  gate is additive: a confirmed field present and healthy alongside a weak field still reaches
  `gates_passed`/`actionable` normally.
- `tests/test_shadow_scoring.py` — 70/70 passed. `tests/test_runner_trade_ntfy.py` run alongside —
  132/133, same one pre-existing failure as every prior round (reconfirmed via `git stash`).
  `py_compile` clean.

### Progress-doc schema (mechanical)
The control-plane queue reported this doc's schema as missing literal `STATUS:`/`WHAT:`/
`WHY/DIR:`/`EVIDENCE:`/`NEXT:` fields despite five rounds of detailed narrative content. Added
those fields at the top of this file (see header above); the narrative sections below are
unchanged and remain the durable per-round record.

### Scope
Unchanged: still observability-only, default NOT ACTIONABLE, `_compute_admission` remains a pure
function. This round only tightens WHICH artifacts can ever reach `gates_passed=True` under the
experimental opt-in, plus the mechanical doc-schema fix — neither adds nor removes a code path.

## Round 7 (Codex CHANGES_REQUESTED — bind admission to a provenance-schema/recipe version)

Round 6's malformed-axis and weak-field-only fail-open fixes accepted. One remaining contract
issue, plus the (now-recurring) evidence-schema completeness ask:

1. **"At least one confirmed field present" (round 6) is code-global, not artifact-bound.**
   `_CONFIRMED_STAMPED_CUTOFF_FIELDS` re-evaluates THIS MODULE's current understanding of "which
   fields count as confirmed" against whatever an artifact happens to carry — it cannot prove
   which STAMPING CONTRACT actually produced that artifact. If a future code change to that
   constant (or a future training-script change to what gets stamped) drifts independently of an
   already-produced artifact, the gate would silently keep evaluating it under a rule it was
   never built against.

### Changes (`shadow_scoring.py`)

- New `_PROVENANCE_SCHEMA_VERSION = "v1"` — versions THIS module's admission-contract rules as a
  whole (the GAP 1-4 gates, `_CONFIRMED_STAMPED_CUTOFF_FIELDS`, `_DATA_CUTOFF_FIELDS`,
  `_TIER_SEVERITY`); bump it whenever any of those change in a way that could change which
  artifacts pass.
- New `_RECIPE_SCHEMA_BY_CONFIRMED_FIELDS`: maps the EXACT set of confirmed cutoff axes an
  artifact presents to a named recipe (`walkforward_only_v1` for `effective_train_cutoff_date`
  alone — hf_patchtst/patchtst or an XGB walk-forward artifact; `full_history_only_v1` for
  `label_observation_cutoff` alone — XGB full-history; `walkforward_dual_axis_v1` for both —
  XGB walk-forward, which `train_production_model.py` stamps with both fields). New
  `_resolve_recipe_schema()` looks up this map; an unrecognized combination (e.g. a future third
  confirmed field added to `_CONFIRMED_STAMPED_CUTOFF_FIELDS` without a matching map entry)
  returns `None`.
- `_compute_admission` now treats an unresolved recipe EXACTLY like "no confirmed field present
  at all" — fail-closed to `unknown`, never silently admitted under a combination this version of
  the contract was never validated against. `run_id` now encodes both the schema version and the
  resolved recipe (`{date}:{name}:{fingerprint}:{schema_version}:{recipe}`) so the audit trail can
  distinguish "evaluated under schema v1, recipe walkforward_dual_axis_v1" from any future
  schema/recipe without ambiguity. New `provenance_schema_version`/`resolved_recipe_schema`
  fields on the returned dict.
- **Scoping decision, stated explicitly rather than attempted:** no artifact anywhere in this repo
  stamps an explicit `schema_version`/`recipe_id` field today (checked `hf_patchtst_scorer.py`'s
  `stamp_artifact_metadata` call and `train_production_model.py::build_artifact` — neither writes
  one). Adding that stamp at training time would be the fully "artifact-bound" version of this
  fix, but it is a cross-repo, training-side change out of scope for this shadow-scoring-focused
  PR. The version implemented here derives an equivalent signal purely from which confirmed
  fields are present, validated against an explicit, versioned map — correct for every recipe this
  repo currently produces, and it still closes the specific drift risk Codex named (an
  unrecognized future combination fails closed instead of silently passing).

### Tests
- `TestProvenanceSchemaVersionBinding` (4 new tests): a lone `effective_train_cutoff_date` and a
  dual-field combination both still resolve and certify normally (regression, additive-only
  check); a THIRD confirmed field monkeypatched into `_CONFIRMED_STAMPED_CUTOFF_FIELDS` (simulating
  future code drift) with a combination absent from the recipe map fails closed to `unknown`,
  never silently admitted; direct unit coverage of `_resolve_recipe_schema()` for known and
  unknown combinations.
- Updated 2 existing tests whose assertions hardcoded the OLD `run_id` format
  (`test_run_id_binds_date_name_and_fingerprint`, `test_missing_fingerprint_blocks_even_when_flag_enabled`)
  to the new `{date}:{name}:{fingerprint}:{schema_version}:{recipe}` shape.
- `tests/test_shadow_scoring.py` — 74/74 passed
  (`/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest tests/test_shadow_scoring.py -q`).
  `py_compile` clean.

### Scope
Unchanged: still observability-only, default NOT ACTIONABLE. This round only tightens WHICH
artifacts can ever reach `gates_passed=True` (an unresolved recipe/schema combination now fails
closed too), plus completes the `doc/AGENT-RETROSPECTIVE.md` §4(b) evidence-block schema in the
header above — neither adds nor removes a code path.
