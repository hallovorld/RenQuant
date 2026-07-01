# Shadow PatchTST freshness — restore retrain cadence + validated served-pin promote (#212)

STATUS:   PR (umbrella impl of RFC r2 `doc/design/2026-06-30-shadow-scorer-freshness.md`,
          orchestrator PR #212). SHADOW-ONLY — moves no capital. Promote is DRY-RUN by
          default; the served pin changes ONLY on --apply and ONLY through the fail-closed
          gate. INSTALL of the launchd job is operational (a one-line note in the plist),
          NOT done by this PR.
WHAT:     Two compounding shadow freezes the RFC diagnosed (§1.2/§1.3): (a) the PatchTST
          retrain had NO scheduler (last manual run 2026-06-16), and (b) a successful
          retrain writes only the WF corpus — it does NOT advance the served shadow pin
          (`strategy_config.shadow.json` `panel_scoring.artifact_path`), so the model ages
          in place while the retrain "succeeds" (the repo's "merged-is-not-deployed"
          failure). This PR ships both halves.
FILES:    - `scripts/launchd/com.renquant.weekly-retrain-patchtst.plist` — the CADENCE
            (Sat 05:30 PT, mirrors weekly-wf-promote / weekly-fundamental-refresh). Runs
            `weekly_retrain_patchtst.sh`. Freshness is NOT keyed on the schedule (§3.2): a
            run completing on schedule is a LIVENESS signal only.
          - `scripts/promote_shadow_patchtst.py` — the VALIDATED served-pin promote. Pure,
            unit-tested gate logic + atomic write-new-then-swap.
          - `scripts/weekly_retrain_patchtst.sh` — chains the validated promote after a
            clean WF build (RQ_PATCHTST_PROMOTE=0 to disable, =dry for dry-run). NON-FATAL:
            a not-fresh refusal (exit 10) or gate failure (exit 20) never fails the retrain
            job; the safe state (old pin retained) is kept.
GATE:     FAILS CLOSED (keeps old pin) unless ALL hold —
          §3.1 freshness: every recipe-required source on its source-specific SLA (fast axis
            ≤28d #210 ceiling: transformer panel + rawlabel; slow axis: SEC fundamentals on
            ~55d filing SLA) AND the effective train/selection cutoffs ACTUALLY ADVANCE past
            the served pin. A non-advancing recipe/code-fix retrain needs
            `--allow-non-fresh --reason ...` and is LABELED non-fresh (does NOT reset the
            freshness clock).
          §3.4 validation: (1) artifact LOAD + smoke inference, (2) schema/recipe/config-
            fingerprint PARITY stamped from the CURRENT pinned config (reuses
            `stamp_patchtst_fingerprint.py` — reconciles with the `panel_scorer_config_mismatch`
            re-stamp, §3.3), (3) NON-DEGENERATE outputs, (4) RESOURCE bounds, (5) a minimum
            WF/holdout SANITY FLOOR. Then atomic pin swap; superseded artifact + config
            backup retained for rollback; a run-bundle JSON records axes, per-source SLA
            verdicts, gate results, the non-fresh label, and the superseded id (§5).
SCOPE:    The exact served-config file differs by deploy (umbrella working tree carries the
          hf_patchtst pin in `strategy_config.json`; the pinned strategy-104 subrepo in
          `strategy_config.shadow.json`). The promote is config-path-configurable and
          REFUSES (harmless exit 2) unless the target's `panel_scoring.kind == hf_patchtst`,
          so it can never clobber a non-PatchTST (GBDT) pin. Set RQ_PATCHTST_SERVED_CONFIG
          to the authoritative config for the deploy. Phase 1 (the standalone observe-only
          freshness MONITOR) is pipeline/orchestrator-owned and is NOT in this PR.
EVIDENCE: `[VERIFIED — pytest + read-only end-to-end]`
          - 31 unit tests (parse/dotted/source-SLA/cutoff-advance/tier/§3.4 helpers +
            run_promote dry-run: non-hf refusal, not-fresh refusal, fresh-dry-run OK,
            gate-failure on degenerate scores). All pass.
          - Read-only `--check` against the LIVE tree (no --apply): correctly REFUSED
            not-fresh — transformer_panel 2026-02-10 (140d OFF-SLA), rawlabel 2026-02-11
            (139d OFF-SLA), fundamentals 2026-06-26 (4d OK), tier=breach — matching the
            RFC's finding that no pin-churn without a panel refresh can clear breach. With
            --allow-non-fresh the §3.4 gate ran on the REAL candidate .pt: load+smoke
            inference PASS (scored 5 probe tickers, 2.9s / 276MB), non-degenerate PASS,
            parity PASS (fp sha256:14586756…), sanity_floor FAIL-CLOSED (no WF metric on the
            manifest) → kept old pin. The live config/artifact mtimes were unchanged.
NEXT:     Operator: install the plist (one-line note in its header) and set
          RQ_PATCHTST_SERVED_CONFIG for the deploy. Model/base-data: refresh the upstream
          point-in-time panel (`transformer_v4_wl200_clean.parquet` + rawlabel) — the
          load-bearing §3.1 prerequisite — and surface a WF/holdout IC in the manifest so
          the sanity floor has a metric to clear. Pipeline: Phase-1 observe-only monitor.
