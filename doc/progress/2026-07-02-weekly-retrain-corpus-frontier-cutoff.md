# Weekly PatchTST retrain: cutoff derived from the corpus frontier (S12 B3, wrapper half)

STATUS:   BUILT + TESTED — umbrella half of the two-PR B3 fix. Companion orchestrator PR
          (`renquant_orchestrator.patchtst_weekly_cutoff`) merges FIRST; until the
          orchestrator pin/sync advances, this wrapper fail-closes with an explicit
          "orchestrator pin predates the S12 B3 corpus-frontier cutoff derivation" error.
          No retrain was run; no production path was written (tests scaffold a temp repo
          with a shim python).
WHAT:     `scripts/weekly_retrain_patchtst.sh` WEEKLY mode no longer reads `LATEST_CUT`
          from the static source manifest's frozen tail. It now delegates to
          `python -m renquant_orchestrator.patchtst_weekly_cutoff` with:
          `--corpus $REPO_DIR/data/transformer_v4_wl200_clean.parquet` (override
          `RQ_PATCHTST_CORPUS`), `--lower-bound-manifest $SRC_MANIFEST` (the static
          manifest DEMOTED to a lower-bound sanity — never the cutoff source), and
          `--max-staleness-days ${RQ_PATCHTST_CUTOFF_MAX_STALENESS_DAYS:-28}`
          (`RQ_PATCHTST_CUTOFF_ARGS` for deliberate ops overrides). The derivation
          prints ONLY the cutoff on stdout; any refusal (missing/stale/regressed corpus)
          exits non-zero and `set -e` aborts the retrain BEFORE any training. Everything
          else — refresh chain, EFFECTIVE_SRC single-cutoff temp manifest, WF build argv
          (output dir/manifest, cadence 0, seed 44, epochs 5, device cpu), promote chain,
          FULL-manifest mode — is unchanged.
WHY:      S12 diagnosis §4-B3 (renquant-orchestrator
          `doc/research/2026-07-02-s12-panel-refresh-diagnosis.md`): with `LATEST_CUT`
          pinned to `walkforward_manifest_v2_20260602.json` (latest 2026-03-09), even
          after the B1 corpus refresh (#434 + base-data #31) the weekly retrain advances
          exactly once, then re-trains the same cutoff forever, the promote's
          `cutoffs_advance` correctly refuses, and the served pin re-freezes with the
          #213 monitor degrading again. §5.3 step 3 assigns the wrapper fix here; the
          derivation logic itself lives in the orchestrator (this wrapper's contract:
          "no training logic in this wrapper") where it is unit-tested against fixtures.
EVIDENCE: `tests/test_weekly_retrain_patchtst_wiring.py` (5 cases, shim-python temp-repo
          harness): derived cutoff — not the static tail — lands in the effective source
          manifest; WF invocation argv otherwise byte-identical to the historical flag
          set (regression); derivation failure aborts before training; stale orchestrator
          pin aborts with the explicit message; FULL mode still consumes the static
          manifest at cadence 180. Adjacent suites green: `test_transformer_corpus_refresh.py`
          + `test_schedule_doc.py` (39 passed). Ground-truth CLI verification (read-only,
          via the companion module): frozen corpus ⇒ FAIL-CLOSED STALE (58d > 28d); prod
          fund panel (B1 recipe source, frontier 2026-04-02) ⇒ derives 2026-03-30, past
          the frozen 2026-03-09 tail.
LANDING:  B1 (base-data #31 → base-data pin → #434) → B2 (#433) → B3 (orchestrator PR →
          orchestrator pin/sync → THIS PR) → umbrella-ops runs the single landing command
          on the live tree: `bash scripts/weekly_retrain_patchtst.sh` (refresh → derived-
          cutoff WF retrain → validated promote). Rollback: revert this PR — the wrapper
          returns to the static-manifest cutoff (the frozen-tail behavior), nothing else
          moves.
