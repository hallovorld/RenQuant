# Momentum ledger-pointer rule in the config-artifact gate — slice 4c   (PR #549)

STATUS:    delivered (merged-but-dark by design: the rule is inert until the
           momentum lane's slice-5 grant batch adds a `.jsonl` shadow entry;
           no config on main points at a ledger today)

WHAT:      `scripts/check_config_artifact_paths.py` gains design point 5 — a
           momentum-aware rule for LEDGER-POINTER shadow entries (slice 4c of
           the momentum pipeline; clears the F-2/F-3 blockers recorded in
           renquant-model#197 amendment 2, surfaced by s104#77):

           - A `shadow_models` entry whose `artifact_path` ends in `.jsonl`
             is a LEDGER POINTER (the s104#77 momentum entry pins the
             append-only digest-chained artifact ledger, the one
             cutoff-stable file in the weekly publish set). It is validated
             by chain identity, NOT by the inline `trained_date` +
             `config_fingerprint` a JSONL ledger cannot carry (F-2):
             (a) when the ledger RESOLVES (canonical resolver, unchanged),
                 the full row chain is verified — `row_index` = line number,
                 `prev_row_sha` links, `row_sha` recomputes over the
                 canonical row body — and any defect FAILS naming the row;
             (b) the tail row's dated artifact must exist beside the ledger
                 (`<ledger_dir>/<cutoff_date>/<kind>.json`, the
                 momentum_train_run.py publish layout) with BOTH its
                 self-carried `content_sha256` and the tail row's
                 `artifact_content_sha256` recomputing over its bytes.
           - ABSENT ledger + a `*_pending_first_artifact` narrative key on
             the entry (the bounded pending guard s104#77 ships) -> INFO
             ("pending first artifact — the designed pre-batch state") and
             PASS: CI stays honest pre-batch while the RUN-time resolve does
             the real check (F-3). ABSENT without the marker stays a FAIL —
             the fail-closed default. The marker cannot rescue classic
             (non-`.jsonl`) entries (tested).
           - `expected_content_sha256` / `expected_config_fingerprint` pins
             on a ledger pointer are REFUSED (fail closed), never silently
             ignored — the append-only ledger changes every publish, so a
             file-sha pin would be stale by design; the chain + tail-artifact
             sha are the swap anchors. An existing-but-EMPTY ledger FAILS
             (only ABSENT+marker is the designed pending state).
           - Every other entry kind (primary / calibrator / classic shadow /
             profile) keeps today's behavior byte-identical; the `../../`
             escape lint still runs FIRST, including on `.jsonl` paths.

           DUPLICATION (explicit, cited in the script above
           `_LEDGER_ROW_REQUIRED`): the chain verification re-implements
           renquant-model `src/renquant_model_momentum/ledger.py`
           (`row_sha256_of`, `load_and_verify_ledger`) and the artifact
           content sha re-implements `src/renquant_model_momentum/train.py`
           (`content_sha256_of`) — canonical JSON `sort_keys=True,
           separators=(",", ":"), allow_nan=False`, digest over the object
           WITHOUT its own sha field. The umbrella cannot import the
           model-factory package (consumers consume by artifact_path, never
           by importing the factory), so those few lines are duplicated on
           purpose; if the model recipe ever changes, this gate must change
           with it.

WHY/DIR:   s104#77 (slice 4, frozen under DO-NOT-MERGE until the grant batch)
           pins the momentum ledger as its shadow `artifact_path`. The gate
           as-is fail-closes that entry twice over: no inline scorer identity
           on a JSONL (F-2), and the publish set is job-written machine state
           the committed-tree CI resolve can never see pre-batch (F-3).
           Without 4c, batch step (e) — merge s104#77 + pin advance — would
           trip this gate by construction. Build order per model#197
           amendment 2: slice 3 (evaluator) -> 4b (pipeline handler) -> 4c
           (THIS) -> the one grant batch (install job -> first artifact ->
           merge s104#77 -> pin advance).

EVIDENCE:  `[VERIFIED-now]` `python3 -m pytest -q -o addopts=''
           tests/test_check_config_artifact_paths.py` (the CI unit job's
           exact command, run in this branch's checkout):
           before (main a07e8a3e): 20 passed, 1 skipped;
           after: 31 passed, 1 skipped — +11 new, zero regressions. The
           skip is `test_real_canonical_contract_*` (renquant-pipeline not
           installed — same as the bare unit CI runner). New tests cover:
           ledger-present-valid passes (chain=verified, rows=2);
           chain-tampered FAILS naming "row 1"; tail-artifact-missing FAILS;
           self-carried-sha mismatch FAILS; ledger-row-vs-artifact sha
           mismatch FAILS ("does not vouch for these bytes"); absent+marker
           -> INFO pass (rc 0 through `main()`); absent without marker ->
           fail-closed; empty ledger FAILS; expected-pin on a pointer
           REFUSED; marker does NOT rescue a classic entry; classic-only
           configs byte-identical.
  artifact:       scripts/check_config_artifact_paths.py +
                  tests/test_check_config_artifact_paths.py — no model/data
                  artifact produced (static CI gate change only).
  prod or exp:    exp. The gate rule is inert on every config on `main`
                  today (none has a `.jsonl` shadow `artifact_path`); it
                  only activates when the slice-5 grant batch adds one.
  existing data:  Yes — the pytest run above, against this branch's
                  checkout, is the existing evidence; no new training run
                  or backtest was needed since this is a static-analysis
                  gate over fixtures.
  best-known?:    N/A — this is a new gate rule (design point 5 of
                  `check_config_artifact_paths.py`), not a variant of an
                  existing metric or model; there is no prior IC/Sharpe
                  baseline to compare against.
  scope:          "this is scripts/check_config_artifact_paths.py, exp
                  (static CI gate; no config, workflow, artifact, job, or
                  production state touched), inert until the slice-5 grant
                  batch introduces the first `.jsonl` shadow entry"

NEXT:      The slice-5 grant batch per model#197 amendment 2's build order:
           install the momentum publish job -> produce the first `.jsonl`
           artifact -> merge s104#77 (frozen under DO-NOT-MERGE until then)
           -> advance the pin. This PR (4c) is the last gate block that
           order needs; slice 3 (evaluator) and 4b (pipeline handler)
           already landed.
