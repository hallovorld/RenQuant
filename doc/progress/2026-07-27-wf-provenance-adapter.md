# Progress — WF provenance sink wired through the sim adapter (pipeline#215/#216)

**Date:** 2026-07-27. **Type:** sim-only provenance wiring (umbrella step of the
approved contract; design pipeline#215 §3 step 3).
**Pairs with:** renquant-pipeline#215 (design, merged) + #216 (sink/emitters,
merged on pipeline main).

## STATUS:
delivered

## WHAT:
IMPORTANT PIN CAVEAT: the PINNED runtime renquant-pipeline predates #216, so
on today's pins the sink constructor logs a loud warning and returns `None` —
the sim runs byte-identically with ZERO emit. The sim only starts emitting
provenance once the pipeline pin advances past #216; that pin advance ships
with the rerun batch (prereg'd XGB multi-seed), NOT with this PR.

Two-phase `wf_sim_provenance.v1` emit through the sim path. Record
construction, digest grammar, PIT check, and the JSONL sink are IMPORTS ONLY
from `renquant_pipeline.kernel.walk_forward.provenance` (lazy — import runs
only when a sink exists); the umbrella adds wiring, never a fork:

- `kernel/walk_forward/provenance_adapter.py` (NEW):
  `build_wf_provenance_sink(seed=…)` — one sink per `run_backtest`, JSONL at
  `<sim checkout root>/data/wf_provenance/<sim_run_id>.jsonl` (rooted on
  `__file__`'s tree = the checkout the sim code runs from; NEVER
  `strategy_dir`, which under `snapshot=True` is a deleted-at-exit tmpdir, and
  never a hardcoded live-tree path). `sim_run_id = wfsim-<utc>-<uuid8>` —
  minted per sim run (`ctx.run_id` is per-BAR and rides in
  `score_observation_key` instead). `revision_pins =
  capture_revision_pins({umbrella,pipeline,model,backtesting,common,
  artifacts})` from the checkouts the sim actually imports from (module
  resolution first, sibling default fallback). Pre-#216 pin ⇒ warn + `None`.
- `sim/runner.py::run_backtest`: constructs the sink ONLY when
  `walkforward.enabled` (inner, post-snapshot call; one sink per run; each
  multi-seed leg gets its own `sim_run_id` + seed) and passes it to
  `SimAdapter(provenance_sink=…)`.
- `kernel/walk_forward/loader.py` (umbrella `WalkForwardModelLoader`):
  `provenance_sink=` kwarg mirroring the pipeline loader; `entry_as_of` emits
  `fold_resolved` at the resolution seam (per-bar dedup; artifact/calibrator/
  manifest digests over real bytes, cached per path; `fingerprint_schema`
  from `_scorer_claim_for_entry`). Digesting uses the PLAIN bounded resolver
  — digest ENFORCEMENT stays on the load paths, unchanged.
  `fold_record_for(today)` exposes the emitted record for the ctx stamp.
- `adapters/sim.py`: `make_context` (sink present + WF loader) stamps
  `ctx._wf_provenance_sink`, `ctx._wf_active_fold` (the loader's emitted
  fold_resolved dict), `ctx._wf_input_watermark`. `commit()` runs the
  persistence-off leg and the `pipeline_runs` mirror (below).
- `kernel/pipeline/task_score_distribution.py`: post-INSERT
  `score_committed` emit (payload digest over the EXACT insert tuples;
  `artifact_digest` echo from the fold; emit failures propagate). Row
  building extracted to `collect_score_rows` so the persistence-off leg
  digests the identical payload. `emit_unpersisted_wf_score_committed(ctx)`
  emits `persisted:false` from `SimAdapter.commit` when the task did not
  commit (`persistence.enabled=False` / `score_db` off); empty payloads emit
  nothing (no observation to bind — the orphaned fold_resolved correctly
  marks the date incomplete).
- `kernel/persistence.py`: design §2.4 SECONDARY mirror — `pipeline_runs`
  gains `training_cutoff` + `model_content_sha256` (schema +
  `_COLUMN_MIGRATIONS`); `record_pipeline_run` accepts the kwargs; sim
  `commit()` fills them from the ACTIVE FOLD and stashes the full
  fold_resolved record in `run_bundle_json.wf_provenance_fold`. JSONL stays
  the PRIMARY record (`sim_runs.db` is truncated every run).

Semantics stamped (honest sources, per review):
- `input_watermark`: MEASURED — max bar/feature DATE across the surfaces
  actually attached to the ctx (panel history slice, panel feature/factor
  frames, macro frame, truncated OHLCV), mapped to the 16:00
  America/New_York session close (the same daily-bar event-time convention
  as the score_timestamp fallback). So the PIT check
  `input_watermark <= score_timestamp` fails exactly when a served frame
  contains rows after the simulated decision date. `None` when no dated
  surface is attached — never invented.
- `ctx.run_timestamp`: left `None` DELIBERATELY — the sim is bar-date-only
  (no real intra-day decision instant exists), so `score_timestamp` uses the
  documented fallback: 16:00 ET close on the bar date.

## WHY/DIR:
codex reviews on model#64/#65/#66: post-hoc reconstruction of which
fold/artifact scored which date is inadmissible; provenance must persist at
generation time. Pipeline#215/#216 landed the contract + emitters;
`RecordScoreDistributionTask`'s pipeline copy reads
`ctx._wf_provenance_sink`/`_wf_active_fold`/`_wf_input_watermark` — but the
SIM executes the umbrella-local task/loader copies, so without this wiring
the ctx attrs are never stamped and nothing emits. Ground truth found while
implementing (differs from the design's assumption, recorded honestly):
- The sim constructs the UMBRELLA-local `kernel.walk_forward.loader
  .WalkForwardModelLoader` (not the renquant_backtesting subclass of the
  pipeline loader) and runs the UMBRELLA-local
  `kernel.pipeline.task_score_distribution` — so the emit sites needed the
  same construction-time-small changes HERE, importing every load-bearing
  primitive from the pipeline module (no third implementation of any digest/
  record logic; the 12-line `_score_timestamp`/`_fold_field` ctx-fallback
  helpers are mirrored with a KEEP IN SYNC note because importing the
  pipeline task module would drag its whole package in).
- Live-surface delta ZERO: sink construction exists ONLY in
  `sim.runner.run_backtest` behind `walkforward.enabled`; loader/task
  defaults are `None`/no-op; a test asserts the daily entries
  (`live/runner.py`, `adapters/lean.py`, `adapters/runner*.py`, `main.py`)
  never reference the sink.

## EVIDENCE:
15/15 new + 142 passing in the WF/sim-scoped sweep; 4 pre-existing environment
failures baselined at origin/main (see run configuration below).

Run configuration (the pinned runtime pipeline predates #216, so tests were
run against the merged pipeline/common mains exported read-only from the
sibling checkouts' `origin/main` — `git archive` — with the umbrella venv):

```
PYTHONPATH=<scratch>/pipeline-main-export/src:<scratch>/common-main-export/src:\
  backtesting/renquant_104:. \
/Users/renhao/git/github/RenQuant/.venv/bin/python -m pytest \
  tests/test_sim_wf_provenance.py -n 0 -q          # 15 passed
… -m pytest tests/test_sim_*.py tests/test_walkforward_*.py tests/test_wf_*.py \
  tests/test_run_backtest_*.py tests/test_score_distribution.py \
  tests/test_db_separation.py tests/test_manifest_uri_resolver.py \
  tests/test_blocked_by_population.py tests/test_quality_floor_gate_a.py -q
  # 556 passed, 4 failed — the 4 failures REPRODUCE at origin/main
  # (git stash baseline): 3 × missing data/alpha158_291_fundamental_
  # dataset.parquet (data file absent from a fresh worktree), 1 × training-
  # script Args fixture drift (test_train_walkforward_panel_fails_closed_
  # on_partial_manifest) — none touch this diff.
```

pipeline export = origin/main `ac98b50` (#216 merge); common export =
origin/main `591d8f70` (has `walk_forward_fold_selection`, common#33 — the
sibling common checkout sits on a stale feature branch and cannot import the
pipeline-main loader; environment fact, not a diff regression).

New coverage: fold_resolved once-per-bar dedup + digest-over-real-bytes +
fold switch across bars; ctx stamps incl. watermark tz/close semantics and
run_timestamp=None; two-bar persisted pairs with `score_payload_digest`
recomputed over what the DB reads back (the Phase-A verification);
persistence-off `persisted:false` via `commit()` + no-double-emit + empty-
payload no-op; `pipeline_runs` mirror columns insert + old-DB migration;
daily-path never constructs (default-None + source scan of the daily
entries); pre-#216 pin degrades to `None` with the loud warning.

## NEXT:
- renquant-backtesting: plumb the sim seed through its WF drivers
  (`wf_gate/runner.py` legs) so their sims mint sinks with real seeds
  (design §3 step 3, backtesting piece).
- Pipeline pin advance past #216 WITH the rerun batch (prereg'd XGB
  multi-seed) — the moment the sim actually starts emitting; NOT done here.
- model#65 rebase: Phase-A converter consumes the JSONL as the only fold
  identity source (design §2.5); #66 isolation rebases on top.
