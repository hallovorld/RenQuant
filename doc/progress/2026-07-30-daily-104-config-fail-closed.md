# Progress: the daily runner could silently substitute a DIFFERENT strategy config

STATUS:   delivered. Detection + fail-closed. No config file edited (see BLOCKED).

WHAT:     `scripts/daily_104.sh` — the pinned-config resolution branch now fails
          CLOSED by default, and logs the resolved primary scorer kind in BOTH
          branches. `tests/test_daily_104_config_failclosed.py` — 8 tests.

WHY/DIR:  RenQuant#546. The branch used to fall back to the umbrella copy unless
          `RENQUANT_STRICT_SUBREPO_PATHS` or `RENQUANT_OPS_FAIL_CLOSED` was set —
          and BOTH default to `0`. So on the default path the fallback was taken
          and the `ERROR` line never executed, because it sat inside the gate.
          The substitution left no trace in the daily log.

EVIDENCE: `[VERIFIED-now]` the three configs' primary `ranking.panel_scoring.kind`:
            backtesting/renquant_104/strategy_config.golden.json  hf_patchtst
            backtesting/renquant_104/strategy_config.json          hf_patchtst
            PINNED renquant-strategy-104/configs/strategy_config.json  xgb
          with `shadow_models` kinds `["xgb"]`, `["xgb"]`, and
          `["hf_patchtst","xgb"]` respectively. The umbrella pair has the two
          models' roles EXACTLY INVERTED relative to what runs.
          `[VERIFIED-prior]` the served PatchTST checkpoint is frozen at
          2026-05-22 (623 days stale), its sentinel reads
          `state=degraded/status=fault` with launchd exiting 1, and its weekly
          retrain trains then REFUSES to promote. Its scores are intrinsically
          all-negative — the correct gate for it is `raw > -0.198`, not `raw > 0`
          — so installing it as primary while the buy path applies the ordinary
          floor admits NO name: a sell-only book. That is the 2026-07-15 incident
          class, reached without anyone taking an action.
  artifact:       backtesting/renquant_104/strategy_config.golden.json,
                  backtesting/renquant_104/strategy_config.json,
                  renquant-strategy-104/configs/strategy_config.json (pinned
                  sibling clone, resolved via renquant_strategy_config() in
                  scripts/subrepo_env.sh) — the three files the EVIDENCE
                  paragraph above reads `ranking.panel_scoring.kind` /
                  `shadow_models` from.
  prod or exp:    Production script change, behaviour-narrowing: a run that would
                  have silently proceeded on a different config now refuses.
                  No scoring, sizing, admission or gate logic touched.
  existing data:  Yes — configs already on disk. No compute, no spend.
  best-known?:    Yes, and it corrects TWO errors in my own filing of #546.
                  (1) I wrote that the ERROR line printed before the fallback; it
                  did not — the fallback was silent. (2) I proposed asserting the
                  resolved config's primary kind matches the pinned one, or
                  failing that, golden. Both are impossible: the pinned config is
                  unavailable precisely in the branch being guarded, and GOLDEN
                  IS ITSELF INVERTED. Had that proposal been implemented it would
                  have added a check that passes forever, which is worse than no
                  check because it reads as protection.
  scope:          One script, one test file, this doc. No config edited, no pin
                  advanced, no artifact touched.

SCOPE/LIMITS:
          The umbrella runner (`RQ_DAILY_RUNNER=umbrella`) KEEPS the fallback —
          it is the one mode that legitimately has no subrepo runtime — but now
          logs a WARN naming the risk. Every other mode fails closed.
          The fail-closed branch is asserted STRUCTURALLY, on the shell source,
          not behaviourally. A behavioural test would have to drive a production
          trading script past its credentials check, and the failure mode of
          getting that wrong is placing orders. The assertions are written
          against the exact guard SHAPE — which env vars may and may not gate it,
          and that the fallback assignment sits after the exit — so weakening
          the guard still fails them. Stating this rather than implying the
          coverage is stronger than it is.

BLOCKED:  The configs themselves are NOT repaired here. `backtesting/renquant_104/
          strategy_config.json` is also read by the TRAINER (RenQuant#544: the
          trainer and the live runner read different configs), so changing its
          primary `panel_scoring.kind` may change TRAINING rather than only the
          fallback. That is a separate decision with a separate blast radius.
          `test_all_three_configs_agree_on_the_primary_scorer` therefore ships as
          `xfail(strict=True)`: it FAILS today, documents the divergence in CI,
          and when the configs are repaired it XPASSes, turning CI red and
          forcing the marker off. The ratchet is deliberate.

VERIFICATION:
          `python3 -m pytest tests/test_daily_104_config_failclosed.py -q`
          -> 7 passed, 1 xfailed. `bash -n scripts/daily_104.sh` clean.
          Covered: a resolution failure exits non-zero; the exit is NOT gated on
          either default-off env var (the actual regression); the only escape is
          an explicit `RQ_DAILY_RUNNER=umbrella`; the fallback assignment appears
          exactly once and only AFTER the exit; the resolved-kind log line sits
          outside the if/else so it runs in both branches; and the kind
          extraction returns `UNKNOWN` rather than failing on a config missing
          `ranking`, `panel_scoring`, or `kind` — observability must never be
          able to abort a run.

NEXT:     RenQuant#544 owns the config repair. Until then the xfail marker is the
          standing record that production and its own reviewed reference disagree
          about which model is primary.
