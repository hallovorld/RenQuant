# Progress: rawlabel is label-clipped — the weekly PatchTST promotion deadlock

STATUS:   delivered (one source flag + 2 regression tests). Unblocks the weekly
          promote path; no capital path touched (the PatchTST lane is shadow-only,
          and `weekly_wf_promote.sh` does not use this source table).

WHAT:     `scripts/promote_shadow_patchtst.py` `DEFAULT_SOURCES` now marks the
          `rawlabel` source `label_clipped: True`, so it is judged from its
          achievable frontier (`cutoff + stamped lookahead_days` trading days)
          like `transformer_panel`, instead of raw calendar age against the 28d
          fast SLA. Comment and the scope-pinning test updated; a second test
          asserts a genuinely frozen build still breaches.

WHY/DIR:  The weekly job (`com.renquant.weekly-retrain-patchtst`, Sat 05:30) has
          been running correctly for months: it trains a fresh fold, fits its
          calibrator, writes the manifest — and then refuses at the freshness
          gate with `rawlabel: cutoff=2026-04-28 age=88d sla=28d OFF-SLA`,
          keeping the old pin. That is why the served PatchTST shadow artifact
          is 622 days stale while a fresh candidate is produced every week.
          The gate's assumption ("a healthy rawlabel KEEPS unlabeled rows, its
          max(date) IS the bar frontier") stopped being true on 2026-07-18 when
          base-data#48 §2.3 dropped the bar-frontier axis extension from the
          single-writer sidecar recipe; and the sidecar's input — the served
          alpha158 fund panel — is itself dropna'd on the forward labels.

EVIDENCE:
artifact:      `data/alpha158_291_fundamental_dataset.parquet` (796M, mtime
               2026-07-26 10:02) and `data/alpha158_291_fundamental_dataset_rawlabel.parquet`
               (801M, mtime 2026-07-25 05:30), read directly with `pandas.read_parquet`
               on 2026-07-28; plus `logs/weekly_retrain_patchtst/stdout.log`
               (2026-07-25 run, 694K).
prod or exp:   prod — both parquets are the live-tree files
               `scripts/promote_shadow_patchtst.py::DEFAULT_SOURCES` reads at
               `data/{name}.parquet` relative to `RENQUANT_REPO_ROOT`; read-only
               measurement, no write to either path or to the log.
existing data: direct read of `alpha158_291_fundamental_dataset.parquet` shows
               max(date)=2026-04-28 and `fwd_5d_excess` / `fwd_20d_excess` /
               `fwd_60d_excess` each carrying 0 NaN — the served fund panel is
               dropna'd on labels at source, so its labeled frontier is
               2026-04-28 even though the file itself was rebuilt 2026-07-26 (2
               days before the read: the build is advancing, not frozen).
               `..._rawlabel.parquet` shows the same max(date)=2026-04-28 with 0
               unlabeled rows in the last 95 days — it inherits the fund panel's
               label frontier, not a raw bar frontier. `logs/weekly_retrain_patchtst/stdout.log`
               (2026-07-25) shows the job trains, saves a calibrator, and writes
               `manifest written (1 rows, 0 failed)`, then
               `promote: refused — NOT FRESH (expected on a stale panel; old pin
               kept)` for rawlabel on the same run where the `transformer_panel`
               line PASSES the freshness gate via the frontier rule
               (`label_clipped: True`) already applied to that source.
best-known?:   n/a — not a model/IC comparison. This is a root-cause diagnosis
               of a freshness-gate rule mismatch (raw-calendar-age vs
               achievable-frontier), not a performance claim; no alternative
               root-cause was found consistent with the same log + parquet
               reads.
scope:         "this is a read-only measurement of the two served parquets
               (`data/alpha158_291_fundamental_dataset.parquet`,
               `data/alpha158_291_fundamental_dataset_rawlabel.parquet`) and the
               2026-07-25 weekly log, prod, explaining why `rawlabel` alone
               fails the freshness gate every week while `transformer_panel`
               passes on the identical cutoff; no model/IC/Sharpe number is
               claimed, so the §4(b) sanity triad does not apply."

Detection preserved: with the flag, 2026-04-28 gives age-beyond-frontier ≈ 7d
(on SLA), while a build stalled at 2026-01-05 still breaches — asserted by the
new test.

Suites: `tests/test_promote_shadow_patchtst.py` 96/96,
`tests/test_promote_shadow_patchtst_snapshot_backstop.py` 3/3.

NEXT:     After merge + deploy, next Saturday's weekly job should promote a fresh
          PatchTST into the shadow lane on its own — the 622d staleness closes
          via the scheduled pipeline rather than a hand landing. Watch the
          2026-08-01 run and the shadow-scorer sentinel's staleness reason.
          Separately queued: the same "28d SLA vs fwd-label horizon" shape in the
          shadow HEALTH record (orch #588 decision A/B) — that path is a
          different consumer and is NOT changed here.
