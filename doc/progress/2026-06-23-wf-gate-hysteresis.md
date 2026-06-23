# P-WF-GATE hysteresis — forgive the undefined-Sharpe / zero-trades measurement failure

STATUS:   merge-pending (PR). Behind a config flag, DEFAULT OFF — merging changes no behaviour.
WHAT:     WfGateMetadataTask gains an optional, default-off hysteresis: when
          wf_gate.forgive_undefined_sharpe_when_zero_trades = true, a WF failure whose Sharpe is
          NON-finite (the walk-forward backtest produced zero trades) is softened from HARD-block
          to a soft pass (buys allowed this run with a warning). A finite-negative-Sharpe loss is
          NEVER forgiven.
WHY-DIR:  the buy funnel (PR #180) showed P-WF-GATE HARD-blocked 8/8 recent runs (0 buys). The
          2026-06-03 study asked for hysteresis so a transient flicker does not halt production —
          but on reading the gate, the failure modes are real quality signals, so a naive
          "N-consecutive" rule would trade a genuinely-failed model (a footgun the operator warned
          against). This forgives ONLY the measurement-gap mode (Sharpe undefined because the WF
          backtest had zero trades), never a measured loss.
EVIDENCE: 17 tests pass, incl. the safety test test_finite_negative_sharpe_never_forgiven_even_with_flag
          (a -1.323 Sharpe HARD-blocks even with the flag on) + flag-off parity (no behaviour change)
          + an unrelated-non-finite-reason case stays HARD. `[VERIFIED — pytest test_preflight_pipeline_gate]`
NEXT:     operator enables it by adding wf_gate.forgive_undefined_sharpe_when_zero_trades:true to
          strategy_config (a strategy-104 change). Until then, no effect. Pairs with the funnel
          report (#180) to watch the binding constraint after enabling.
