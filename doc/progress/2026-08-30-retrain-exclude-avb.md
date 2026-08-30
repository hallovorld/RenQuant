# Weekly promote: exclude AVB (delisted 2026-08-17) from the retrain until the reviewed exclusion registry is pinned

STATUS:    delivered — INTERIM. One default changed in `scripts/weekly_wf_promote.sh`
           (`RETRAIN_EXCLUDE_TICKERS` `IAC` → `IAC,AVB`); no live-tree write, no pin
           change, no config write. The durable fix is renquant-orchestrator#1096
           (a reviewed exclusion registry applied to the universe AND to the panel
           build); once that is pinned and the `-run` checkout is synced, this
           default is dropped (both names).
WHAT:      `scripts/weekly_wf_promote.sh:404-407`
           `RETRAIN_EXCLUDE_TICKERS="${RENQUANT_RETRAIN_EXCLUDE_TICKERS:-IAC,AVB}"`
           (env override unchanged), threaded through `daily_retrain_alpha158_fund.sh`
           to `retrain_alpha158_fund --exclude-tickers` — the bridge designed on
           2026-07-17 for IAC (`doc/progress/2026-07-17-retrain-exclude-iac.md`).
           A three-line comment records the delisting, the evidence and the
           removal condition.
WHY/DIR:   The 2026-08-29/30 weekly promote FAILED at the strict (0.0) freshness
           gate on ONE name: AVB (AvalonBay; Equity Residential merger closed
           2026-08-17; SEC 8-K/Form 25), last bar 2026-08-24, still in `tier_A` of
           the inventory (generated 2026-05-05, no `delisted_tickers` channel) and
           NOT in the served watchlist. The gate is correct; the universe
           declaration is stale. Codex on orch#1096 rejected a heuristic skip and
           named this bridge as the interim: "AVB can use the existing explicit
           exclusion bridge meanwhile." Scope honesty: `--exclude-tickers` removes
           AVB from the freshness accounting only — with the PINNED orchestrator
           the panel build still reads the raw inventory, so AVB's stale rows stay
           in the panel exactly as IAC's have since July; orch#1096 is what removes
           an excluded name from the actual training universe.
EVIDENCE:  artifact:      `scripts/weekly_wf_promote.sh:404-407` (the
                          `RETRAIN_EXCLUDE_TICKERS` default + its removal-
                          condition comment) — the only code change. The
                          tripping run is
                          `logs/daily_retrain_alpha158_fund/2026-08-30.log`
                          lines 3–10. The stale universe declaration is
                          `transformer_universe_inventory.json` (`tier_A`,
                          generated 2026-05-05, no `delisted_tickers`
                          channel) as read by the PINNED
                          `renquant_orchestrator/retrain_alpha158_fund.py`.
           prod or exp:   prod operational path (the launchd weekly promote
                          wrapper). No model, signal, artifact, fingerprint,
                          config, data or live-state file is written; no
                          performance / IC / Sharpe claim is made.
           existing data: `logs/daily_retrain_alpha158_fund/2026-08-30.log`
                          l.3 `$AVB: possibly delisted; no price data found
                          (1d 2026-08-25 -> 2026-08-30)`; l.9
                          `fetch_ohlcv_incremental(AVB) timed out after 30s —
                          returning stale cache (last=2026-08-24)`; l.10
                          `freshness guard TRIPPED: 1/293 panel tickers stale
                          (0.3% > 0.0%; missing=0 future=0); bars lag expected
                          NYSE session 2026-08-28 by >1 sessions. Worst:
                          AVB(-4s). FAILING retrain.`
                          `[VERIFIED — read-only grep of the live log,
                          2026-08-30]`. Delisting record: SEC 8-K/Form 25
                          https://www.sec.gov/Archives/edgar/data/0000915912/000110465926097833/tm2623381d1_8k.htm
                          `[VERIFIED — prior work, orch#1096 investigation]`.
                          AVB absent from the served watchlist (n=145)
                          `[VERIFIED — json read of
                          renquant-strategy-104/configs/strategy_config.json,
                          2026-08-30]`. No umbrella test pins the default
                          (`git grep RETRAIN_EXCLUDE_TICKERS origin/main --
                          tests` → none) and
                          `bash -n scripts/weekly_wf_promote.sh` passes
                          `[VERIFIED — 2026-08-30]`.
           best-known?:   yes, for the INTERIM — it is the exact explicit
                          exclusion bridge designed for IAC on 2026-07-17
                          (`doc/progress/2026-07-17-retrain-exclude-iac.md`)
                          and the bridge codex named on orch#1096 after
                          rejecting a heuristic skip. It is NOT the durable
                          fix: that is the reviewed exclusion registry in
                          orch#1096, which also removes the name from the
                          panel build.
           scope:         "this is `scripts/weekly_wf_promote.sh`
                          `RETRAIN_EXCLUDE_TICKERS` `IAC` → `IAC,AVB`, prod
                          promote wrapper, freshness accounting only — with
                          the PINNED orchestrator AVB's stale rows stay in
                          the panel exactly as IAC's have since July; no
                          performance claim; vs existing best: the
                          2026-08-30 promote FAILED at the strict 0.0 gate on
                          1/293 names, and this lets the next promote proceed
                          past AVB under the unchanged gate."
NEXT:      merge → operator FF-advances the umbrella live tree (landing action) →
           the next Saturday promote (or a manual `weekly_wf_promote.sh`) proceeds
           past AVB under the unchanged strict gate. Drop `IAC,AVB` from this default
           in the same batch that pins orch#1096 + syncs the `-run` checkout.
