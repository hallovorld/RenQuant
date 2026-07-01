# Shadow ntfy: honest top-N recommendation + disambiguate [SHADOW]-broker vs shadow-model labeling

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
