# 2026-05-23 Daily Replay Decision Quality

## Scope

User asked to run a non-trading daily/full-style replay on Saturday
2026-05-23 to inspect decision quality after recent fixes.

No live orders were placed. `readonly-alpaca` was run with
`RENQUANT_NO_NOTIFY=1` and `RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1`.

Raw local artifacts:

- `backtesting/renquant_104/artifacts/diagnostics/daily_replay_20260523/readonly_full_prod.log`
- `backtesting/renquant_104/artifacts/diagnostics/daily_replay_20260523/xgb_wf_recent_after_qpdef_sim.log`
- `backtesting/renquant_104/artifacts/diagnostics/daily_replay_20260523/xgb_wf_recent_after_qpdef_trade_report.md`

## Production Readonly Full

Current production full run did not reach decision execution. Preflight
blocked it correctly:

- `P-WF-GATE`: active production panel artifact carries failed WF evidence:
  `wf_sharpe_mean=-1.3233`, `spy_sharpe_mean=+1.0808`, `0/3` WF cuts beating
  the acceptance gate.
- `P-CONFIG-FP`: initially hard-failed because the 2026-05-18 legacy artifact
  was not stamped with the newly model-relevant `sector_map` /
  `sector_etf_map` fields.

Interpretation: a real daily full should not buy from this production artifact
until either a WF-passing artifact is promoted or the live run is explicitly
isolated as research/shadow.

Follow-up fix: `P-CONFIG-FP` now treats legacy artifacts that lack only the
new sector fingerprint fields as a soft migration warning *only if*
`P-SECTOR-MAP` independently passes. A real stored sector mismatch remains a
hard failure, and incomplete current sector coverage remains a hard failure.
Readonly production preflight after the fix has one hard blocker:

- `P-WF-GATE`: hard fail, still correct.
- `P-CONFIG-FP`: soft migration warning.
- `P-SECTOR-MAP`: pass, `141` buyable tickers and `13` sectors mapped.

## Leak-Safe Recent Sim

Direct sim with `strategy_config.json` was refused by the GMM leakage guard:
production regime artifact `as_of_date=2026-05-22` cannot replay a window
starting `2026-05-01`. This guard is correct.

I then ran the recent window with:

```bash
.venv/bin/python scripts/run_sim_104.py \
  --strategy-config-name strategy_config.codex_xgb_wf_calibrated_qpfix_noshadow_20260523.json \
  --start 2026-05-01 --end 2026-05-22 \
  --no-compare --no-persist
```

This config keeps production-style decision tree/QP/tax/sector semantics, but
uses a walk-forward manifest and sim regime/correlation artifacts to avoid
look-ahead leakage.

After the QP defensive-gate fix below:

- Final value: `$101,080`
- Total return: `+1.08%`
- APY: `+19.78%` over a very short 16-trading-day window
- Sharpe: `+5.16`
- Max drawdown: `0.51%`
- Trades: `5 buys`, `0 sells`
- Open mark-to-market outcomes: `4 wins / 1 loss`

Same-window SPY reference:

- SPY total return: `+3.47%`
- SPY APY: `+71.07%`
- SPY Sharpe: `+5.22`
- SPY max drawdown: `1.93%`

Interpretation: the repaired decision tree is conservative and lower-vol than
SPY, but it underperformed SPY in raw return over this short window. The high
annualized APY/Sharpe should not be over-interpreted because all lots remain
open and the sample is only 16 trading days.

## Decision-Tree Finding Fixed

The replay exposed a real consistency bug:

- Greedy selection already blocks defensive tickers outside BEAR via
  `defensive_non_bear`.
- QP runs before the greedy fallback and was able to emit `QP_BUY` for
  defensive tickers in `BULL_CALM`.
- Evidence: before the fix, QP bought `GLD` in `BULL_CALM`.

Patch:

- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
- `tests/test_joint_qp_task.py`

New behavior:

- Non-BEAR QP buy/top-up for `config.defensive_tickers` is suppressed and
  stamped as `defensive_non_bear`.
- Defensive sells remain allowed in non-BEAR regimes.

Verification:

```bash
.venv/bin/python -m pytest tests/test_joint_qp_task.py -q
# 40 passed

.venv/bin/python -m pytest \
  tests/test_joint_qp_task.py \
  tests/test_buy_emit_contract.py \
  tests/test_defensive_gate.py -q
# 52 passed
```

Replay verification after the fix:

- `QP_BUY_SUPPRESSED GLD defensive_non_bear (regime=BULL_CALM)`
- No `GLD` buy in the resulting trade ledger.

## Current Decision Quality Read

For 2026-05-22 in the repaired replay:

- Regime: `BULL_CALM`
- Confidence: `0.25`
- Candidate scan: `104 candidates from 109 tickers`
- Realized-vol gate: dropped `17`
- Panel scoring: `87/87` candidates scored and calibrated
- Calibration diagnostic: low IQR around `0.045`, diagnostic only
- Adaptive rank floor: about `0.610`
- Floor veto: dropped `73`
- Ranked candidates after floor: `14`
- QP emitted no new buys:
  - defensive `GLD` suppressed
  - 3 trades skipped by no-trade band
  - 15 trades below minimum delta-weight

This is a materially cleaner decision tree than the prior live behavior that
allowed questionable defensive or sector-skewed buys. It still does not prove
the model is economically strong; it shows the execution tree is less likely
to turn weak/compressed scores into unnecessary trades.

## Remaining Concerns

- Production artifact is still not live-trustable because `P-WF-GATE` fails.
- Production config/artifact fingerprint is now classified correctly: current
  sector metadata is complete, but the active legacy artifact still needs a
  retrain/promotion to stamp sector fields.
- Calibrated score IQR remains compressed. This is not a hard failure, but it
  weakens economic discrimination and makes QP rely heavily on small differences
  in expected return.
- Recent-window performance is below SPY on raw return, although lower drawdown
  makes risk-adjusted stats comparable over this short sample.
- `daily_104.sh` now alerts by default if the shadow e2e leg fails or times
  out. Set `RENQUANT_SHADOW_ALERT_NTFY=0` only for explicit quiet runs.
