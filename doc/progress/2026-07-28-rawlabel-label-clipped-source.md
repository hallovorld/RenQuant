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

EVIDENCE: Read-only measurement of the two served parquets on 2026-07-28
          `[VERIFIED — direct parquet read]`:
          - `alpha158_291_fundamental_dataset.parquet`: max(date)=2026-04-28;
            `fwd_5d_excess` / `fwd_20d_excess` / `fwd_60d_excess` each carry
            **0 NaN** — the panel is dropna'd on labels at source. File mtime
            2026-07-26, i.e. rebuilt 2 days before the read: advancing, not frozen.
          - `..._rawlabel.parquet`: max(date)=2026-04-28, 0 unlabeled rows in the
            last 95 days — inherits the label frontier, contradicting the raw-SLA
            assumption.
          - Weekly job log `logs/weekly_retrain_patchtst/stdout.log` (2026-07-25):
            trains + saves calibrator + `manifest written (1 rows, 0 failed)`,
            then `promote: refused — NOT FRESH (expected on a stale panel; old
            pin kept)`, with the transformer_panel line PASSING on the very same
            date via the frontier rule.
          Detection preserved: with the flag, 2026-04-28 gives
          age-beyond-frontier ≈ 7d (on SLA), while a build stalled at 2026-01-05
          still breaches — asserted by the new test. No model/IC/Sharpe number is
          claimed, so the §4(b) sanity triad does not apply.
          Suites: `tests/test_promote_shadow_patchtst.py` 96/96,
          `tests/test_promote_shadow_patchtst_snapshot_backstop.py` 3/3.

NEXT:     After merge + deploy, next Saturday's weekly job should promote a fresh
          PatchTST into the shadow lane on its own — the 622d staleness closes
          via the scheduled pipeline rather than a hand landing. Watch the
          2026-08-01 run and the shadow-scorer sentinel's staleness reason.
          Separately queued: the same "28d SLA vs fwd-label horizon" shape in the
          shadow HEALTH record (orch #588 decision A/B) — that path is a
          different consumer and is NOT changed here.
