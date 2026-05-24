# 2026-05-23 Trade Quality Review

## Scope

This note answers the operator question: what trades did the latest e2e-like
runs actually choose, and were those decisions high quality?

Raw sources:

- Formal live full, user-requested direct run:
  `logs/daily_104/2026-05-23_live_full_direct_user_requested.log`
- Live no-WF-gate diagnostic run:
  `logs/daily_104/live_no_wf_gate_once_20260522-180707.log`
- Production decision DB: `data/runs.alpaca.db`
- Shadow PatchTST readonly run:
  `logs/daily_104/2026-05-23_shadow_manual_after_fpfix.log`
- Shadow decision DB: `data/runs.alpaca_shadow.db`

## Production Full Status

User-requested formal live full was run directly at `2026-05-23 16:56 PT`
with:

```bash
.venv/bin/python -m live.runner --strategy renquant_104 --broker alpaca --once
```

It connected to the LIVE Alpaca account, sent ntfy successfully, and placed no
orders because preflight aborted before inference:

- loaded models: `114/142` symbols
- no-artifact skips: `APH`, `ATI`, `BWXT`, `EME`, `GLW`, `GRMN`, `SPY`, `XLI`, `XLY`
- universe-floor skips: 19 models with per-ticker Sharpe below `0.5`
- `P-MODEL-ARTIFACT`: pass, loaded `panel-ltr.alpha158_fund.json`
- `P-PANEL-CONTRACT`: soft legacy stamp warning
- `P-WF-GATE`: hard fail
- `P-BEST-ITER`: pass, `best_iter=100`
- `P-CONFIG-FP`: soft legacy sector-stamp migration warning
- `P-WATCHLIST`: pass, `n=142`
- `P-SECTOR-MAP`: pass, `141` buyable tickers and `13` sectors mapped
- `P-FEATURE-COVER`: pass, `169` NGBoost features, `0` missing
- `P-STATE-FILE`: pass, loaded `live_state.alpaca.json`
- `P-BROKER-CONNECT`: pass, equity `$10,834.91`
- `P-CALIBRATOR-HEALTH`: pass, `pool_ic=0.1149`
- `P-CALIBRATOR-FLAT-REGION`: pass

The blocking WF evidence:

- `wf_sharpe_mean=-1.3233`
- `spy_sharpe_mean=+1.0808`
- `0/3` WF cuts beat SPY

So the active production artifact should not place new buy-side orders. Any
buy-side trade observed from `strategy_config.live_no_wf_gate_once.json` is a
diagnostic bypass, not a trusted production decision.

Formal-full decision-tree conclusion: there is no new candidate/ranking/QP
tree to analyze from production because the run correctly stops at preflight.
The latest production-quality decision is "do not trade this artifact."

## 2026-05-22 No-WF Diagnostic Buys

Run id: `2026-05-22-live-ff554178`.

Decision tree:

- Regime: `BULL_CALM`, confidence `0.60`.
- Buy scan: `80` candidates from `108` tickers after early filters.
- Realized-vol gate: dropped `12/80` above `60%` annualized vol.
- Scored/calibrated: `68` candidates.
- Adaptive floor: `max(0.20, mean+1.00*std)=0.581`, dropped `57`.
- Ranked after floor: `11`.
- QP emitted `3` buys and `0` sells.
- QP skipped `2` trades by no-trade band and `11` below minimum `Δw`.

Executed/bypassed diagnostic buys:

| Ticker | Shares | Price | Rank | Panel | Mu | Sigma | QP target/Δw | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BAC | 13 | 51.80 | 0.603 | 0.178 | 0.0408 | 0.232 | 6.45% | Borderline |
| D | 8 | 67.67 | 0.593 | 0.140 | 0.0375 | 0.263 | 5.03% | Borderline |
| WFC | 7 | 76.40 | 0.609 | 0.203 | 0.0430 | 0.300 | 5.08% | Borderline |

These are not high-conviction trades. They sit just above the adaptive floor,
not near the top of the scored universe. Top candidates like `CRWD`, `HPE`,
`TXN`, `MCD`, and `MPWR` were not bought because the QP considered their
incremental target weights too small after covariance/current-portfolio/cash
constraints.

Quality conclusion: this was an optimizer picking small allocation changes
from marginal positive scores. It is not strong enough evidence to buy with
real money, especially because the underlying production artifact fails WF.

Data-quality finding from this run: selected `BAC`, `D`, `WFC`, and several
other candidates had `sector=null` in `candidate_scores`, despite the current
config having sector coverage. `P-SECTOR-MAP` now verifies current config
coverage, but score persistence still needs a sector-field audit so DB traces
match the optimizer's sector inputs exactly.

## 2026-05-23 PatchTST Shadow E2E

Run id: `2026-05-23-live-d31d6dc1`.

This was readonly-alpaca shadow; no live orders were submitted. It did send an
ntfy line:

- `[SHADOW]RENQUANT-104 [full] SHADOW-ACTION`

Decision tree:

- Preflight passed for shadow:
  - PatchTST checkpoint loaded.
  - `val_ic=+0.0307`, `n_features=172`.
  - `P-SECTOR-MAP` passed: `141` buyable tickers, `13` sectors.
  - `P-CALIBRATOR-HEALTH` passed: `pool_ic=0.1309`.
- Regime: `BULL_CALM`, confidence `0.60`.
- Sell scan: `0` exits from `6` held before panel scoring.
- Buy scan: `103` candidates from `108` tickers.
- Realized-vol gate: dropped `16/103`.
- PatchTST scored `93/93`, then assigned scores to `87` candidates and `6`
  holdings.
- Calibrated `87/87` candidates and `6/6` holdings.
- Adaptive floor: `0.545`, dropped `77`, leaving `10` ranked.
- QP emitted `0` buys and `2` sells.

Top ranked shadow candidates:

| Ticker | Role | Rank | Panel | Mu | Sigma | QP disposition |
|---|---|---:|---:|---:|---:|---|
| ORCL | candidate | 0.679 | -0.044 | 0.0716 | 0.546 | below min Δw |
| SPOT | candidate | 0.608 | -0.105 | 0.0446 | 0.541 | below min Δw |
| MU | holding | 0.603 | -0.109 | 0.0429 | 0.800 | reduce target, not emitted in old run |
| HON | holding | 0.586 | -0.123 | 0.0360 | 0.253 | no-trade band |
| LLY | candidate | 0.583 | -0.125 | 0.0352 | 0.376 | no-trade band |

Shadow exit/trim quality:

| Ticker | Action | Reason | P/L | Hold days | Rank | Panel | Mu | Sigma | Quantity |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| GE | sell | panel_conviction | +2.47% | 9 | 0.428 | -0.248 | -0.0239 | 0.423 | full |
| FTNT | trim | qp_sell | +59.04% | 26 | 0.517 | -0.177 | 0.0102 | 0.525 | 5 |
| GE | trim | qp_sell | +2.47% | 9 | 0.428 | -0.248 | -0.0239 | 0.423 | 2 |

The GE double row exposed a real bug: the cross-sectional panel exit emitted a
full GE liquidation, then QP emitted an additional GE partial sell in the same
bar. That is not acceptable decision quality. The fix is to suppress any QP
buy/sell/top-up for a ticker that already has an exit intent earlier in the
bar. Regression test:

```bash
.venv/bin/python -m pytest \
  tests/test_joint_qp_task.py::TestActionDirections::test_existing_exit_suppresses_qp_duplicate_sell \
  tests/test_joint_qp_task.py::TestActionDirections::test_negative_mu_on_held_emits_sell -q
# 2 passed

.venv/bin/python -m pytest tests/test_joint_qp_task.py -q
# 41 passed
```

## Operator Read

The 2026-05-22 bank/utility buys were weak diagnostic trades and should not be
used as proof that buy quality is fixed. The 2026-05-23 PatchTST shadow is
more informative: it chose no new buys, wanted to harvest/trim winners, and
revealed a duplicate-exit bug that has now been fixed.

Open issues after this review:

- Re-run shadow after the QP duplicate-exit fix and confirm GE appears only
  once.
- Audit why some persisted `candidate_scores.sector` rows are null when
  current sector coverage passes.
- Production buy-side remains blocked until a WF-passing artifact is promoted.
