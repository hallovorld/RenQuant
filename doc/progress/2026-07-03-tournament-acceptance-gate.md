# Tournament acceptance gate — fail-closed per-ticker writes (campaign A2)

Date: 2026-07-03
Finding: F-17 (P0), umbrella compliance audit (RQ PR #444); campaign plan
orchestrator PR #297 (Group A, fix A2). Deadline-driven: the Sunday
`weekly_tournament_retrain.sh` job fires again 2026-07-05 06:00 PT.

## Problem

`scripts/weekly_tournament_retrain.sh` → `train_104.py --skip-panel --force`
trains the per-ticker buy-admission tournament (RF / Q-learning / per-ticker
XGB / Manual) and wrote it STRAIGHT to production
`backtesting/renquant_104/models/<TICKER>/`:

- `train_104.py` auto-disables `ModelAcceptanceGate` under `--skip-panel`
  (commit `ba44e58`, 2026-05-22 "Harden alpha158 staged retraining gates").
  That was not an oversight in itself: the panel gate evaluates a candidate
  *panel-ltr artifact* against the active one, and `--skip-panel` never
  produces a panel candidate — the gate is structurally inapplicable to
  per-ticker tournament output. The same commit added the refusal of
  acceptance-enabled runs that include `BaselineTournamentJob`, explicitly
  calling the per-ticker exports "still production writes". The gap: nothing
  tournament-shaped ever replaced the panel gate on this path.
- The write itself was in-place and multi-stage per ticker
  (`export_one_model` model save + metadata patch, then
  `retrain_live_models` full re-save + patch, then calibration patch) — a
  mid-run crash could leave a ticker dir torn, and a broken retrain
  (degenerate scores, frozen/regressed data, metric collapse) silently
  replaced every good model in one Sunday pass.

## Fix

New `backtesting/renquant_104/kernel/tournament_acceptance.py` + wiring in
`kernel/pipeline/pp_training.py::_run_ticker_chain` / `_run_gated_export`:

1. **Per-ticker fail-closed verdict BEFORE any write**, computed only from
   what the tournament already produces (result dict + feature frame +
   incumbent policy metadata):
   - T1 model present; T2 train/oos row floors;
   - T3 non-degenerate OOS raw scores (≥10 finite, not constant);
   - T4 data cutoff: not future, ≤45d stale, and **never regressing vs the
     incumbent's `live_train_end`** (same policy as
     `tournament_retrain_marker.py`);
   - T5 metric collapse vs incumbent: reject only when candidate OOS Sharpe
     is BOTH below an absolute floor (−1.0) AND ≥2.0 below the incumbent.
     Honest degradation still ships — a worse fresh Sharpe is admission
     information for `LoadUniverseJob`'s `ranking.universe_floor`; freezing
     stale good-looking metadata would corrupt admission.
   Comparison legs skip-pass with no readable incumbent (nothing to protect).
2. **Staging-then-swap**: accepted candidates are written by the unchanged
   Export/Calibration jobs into `models/.staging/<T>-<uuid>/<T>/` (same
   filesystem), then promoted per-file via `os.replace`, metadata last.
   RF/QL/Manual `save()` embed absolute artifact paths in metadata —
   the promote rewrites the staging prefix to the live prefix so promoted
   bytes match the pre-fix in-place write exactly. Any exception mid-stage
   rejects the ticker and leaves the incumbent byte-untouched.
3. **Loud failure path**: per-ticker `REJECT` warnings, verdict archive under
   `artifacts/_tournament_acceptance_log/`, `TrainingContext.rejected` →
   `FullTrainingContext.baseline_rejected` → one aggregated ntfy WARN from
   `train_104.py` (honors `RENQUANT_NO_NOTIFY=1`). The completion marker
   independently refuses to certify a partially-refreshed corpus, so the
   Sunday wrapper's existing ✗ alert also fires — belt and suspenders.
4. **Default ON, operator-controllable**: `acceptance.tournament.enabled`
   defaults to `acceptance.enabled` (true in prod config) — Sunday's run is
   protected with zero config edits. `--skip-acceptance` disables both gates
   (unchanged operator semantics); knobs under `acceptance.tournament.*`.

## Protection contract (P0, operator-pinned)

A HEALTHY run's outputs are byte-identical to pre-fix. Proven by
`tests/test_tournament_acceptance.py`:
- A/B fixture runs (gate off = pre-fix code path vs gate on) with no
  incumbent, with an incumbent, and through the REAL `ManualModel` save
  contract (absolute-path metadata + the retrain double-write) — byte-equal
  file sets modulo the fixture root prefix.
- Degenerate candidates (constant/NaN scores, regressed cutoff, metric
  collapse) → rejected, incumbent byte- AND mtime-untouched, reason counted.
- Partial-batch isolation: 1 bad ticker of 3 → the other 2 written normally.
- Fail-closed: gate crash or staged-write crash → reject, incumbent kept,
  staging cleaned.

32 new tests; related existing suites (train_104 wiring, model TTL, cadence,
parallel timeout, retrain marker, sim/prod isolation — 98 tests) all green.
Pre-existing failures in `test_audit_2026_04_24_fixes.py` (3) and
`test_correlation_guard.py` (needs pinned `renquant_pipeline` on PYTHONPATH)
reproduce identically on unmodified `main` — unrelated.

## Deploy note

Merged ≠ deployed: Sunday's job runs the LIVE tree at
`/Users/renhao/git/github/RenQuant`. After merge, the live checkout must be
synced (operator-gated action) before 2026-07-05 06:00 PT for the gate to
cover this week's run.

## Follow-ups (out of scope here)

- F-17's delegation half: port the tournament trainer to renquant-model /
  admission to renquant-pipeline and give `weekly_tournament_retrain.sh` a
  multirepo delegate branch like its sibling wrappers.
- The standalone notebook `ExportJob` path (documented "not recommended")
  still writes ungated; the scheduled path is covered.
