# renquant_104 Strict Retrain / WF Gate Failure — 2026-05-21

## Verdict

Do not promote the strict-contract retrained production artifact from this run.

The model trained successfully, passed strict contract dry-run, and passed the current acceptance gate, but failed the 3-cut walk-forward gate and failed the placebo sanity threshold. The active production artifact was rolled back to the pre-run backup.

No full daily live run should be executed from the failed artifact.

## Artifact Handling

Pre-run backup:

- `backtesting/renquant_104/artifacts/prod/codex_pre_strict_20260520-231401/panel-ltr.alpha158_fund.json`
- `backtesting/renquant_104/artifacts/prod/codex_pre_strict_20260520-231401/panel-rank-calibration.json`

Failed strict artifact preserved:

- `backtesting/renquant_104/artifacts/prod/codex_failed_wf_20260521-012231/panel-ltr.alpha158_fund.json`
- `backtesting/renquant_104/artifacts/prod/codex_failed_wf_20260521-012231/panel-rank-calibration.json`

Active production path after rollback:

- `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json`
- `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json`

Post-rollback dry-run preflight passed with 0 hard failures. It is legacy-compatible, not strict-contract complete:

- missing `train_run_id`
- missing `oos_mean_ic`
- missing `oos_std_ic`

That is intentional after rollback. Safety beats keeping a strict artifact that failed WF.

## Training Result

Command:

```bash
.venv/bin/python scripts/train_production_model.py \
  --output-path backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --cv-n-splits 3 \
  --cv-embargo-days 60
```

Observed training summary:

- rows: 716,607
- tickers: 292
- date range: 2016-01-04 to 2026-02-10
- label: `fwd_60d_excess`
- 3-fold purged WF CV ICs: `[+0.0879, -0.0023, +0.0465]`
- CV mean IC: `+0.0440`
- CV std IC: `+0.0451`
- final train IC: `+0.1151`
- config/data fingerprint observed in preflight: `sha256:9333f7bf91d10cc4`
- train run id observed after artifact stamp: `f7dc2054`

Strict dry-run passed before WF:

```bash
.venv/bin/python scripts/train_104.py \
  --dry-run \
  --strict-contract \
  --skip-baseline \
  --skip-recalibrate
```

## Calibrator Fixes Applied Before WF

Two calibrator issues were fixed and pushed before the WF gate:

1. Calibrator health functions now accept numpy arrays instead of failing on ambiguous truthiness.
2. Platt expected-return calibrator output is clipped through the same ER bounds as isotonic.

After fix:

- `expected_return.y` max abs: `0.06915`
- probability `y` range: `0.40945` to `0.69124`
- probability knots: 100 unique y values
- strict dry-run hard checks: pass

## Acceptance Gate

Acceptance gate verdict before WF:

- ACCEPT
- hard failures: 0

Caveat: several acceptance checks were skip-pass because prior metadata was missing. This acceptance result was not strong enough to override WF failure.

## WF Gate Result

Command:

```bash
.venv/bin/python scripts/run_wf_gate.py \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --strict
```

3-cut WF result:

| Cut | Window | Sharpe | APY |
| --- | --- | ---: | ---: |
| 1 | 2024-01-02 to 2024-12-31 | `+0.170` | `+6.40%` |
| 2 | 2024-07-01 to 2025-06-30 | `-0.300` | `-0.00%` |
| 3 | 2025-04-01 to 2026-03-28 | `+0.170` | `+6.60%` |

Aggregate:

- mean Sharpe: `+0.013`
- Sharpe std: `0.271`
- mean APY: `+4.33%`
- positive cuts: 2 / 3
- WF verdict: FAIL
- WF reason: mean Sharpe is far below required `+0.40`

Sanity battery:

- real IC: `+0.0775`
- shuffled IC: `-0.0016` passes `|IC| < 0.005`
- placebo IC: `+0.0394`
- placebo threshold: `0.5 × real_ic = +0.0388`
- sanity verdict: FAIL by a narrow margin

The placebo miss is small numerically, but the WF miss is decisive.

## Source Fixes Landed

Relevant commits pushed:

- `48a7350` — stamp production training strict contract and purged WF CV
- `b5f4517` — accept array calibrator curves
- `77be670` — enforce Platt calibrator expected-return bounds
- `6ba6dcb` — repair WF gate runner defaults and historical sim path
- `2321fb8` — audit PatchTST experiment design
- `330d061` — add bounded WF gate parallelism via `--jobs`

## Important Operational Decision

The failed strict artifact was not left active.

Reason: `daily_104.sh` eventually calls the live runner with the real Alpaca broker path. Leaving a WF-failed artifact in `artifacts/prod` creates avoidable live-trading risk.

## Next Work

1. Investigate the 2024-07-01 to 2025-06-30 WF failure. Start with decision-tree counters, candidate score distributions, QP rejections, and regime mix.
2. Investigate placebo IC persistence. It is barely over threshold, but it should not be ignored.
3. Keep the current active artifact on the pre-run backup until a retrained artifact passes WF.
4. Use `scripts/run_wf_gate.py --jobs 3` only when heavy PatchTST jobs are not competing for resources.
5. Do not run full live daily buy/sell from any artifact unless WF passes or the operator explicitly overrides the gate.
