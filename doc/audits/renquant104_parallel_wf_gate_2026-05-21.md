# renquant_104 Parallel WF Gate Probe — 2026-05-21

## Command

```bash
.venv/bin/python scripts/research_acceptance_104.py \
  --target wf-gate \
  --artifact backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json \
  --wf-jobs 3
```

## Concurrency Result

The three walk-forward cuts did run concurrently. `ps` showed three
`run_sim_104.py` child processes under `run_wf_gate.py`, each consuming a
CPU core for the whole run:

- 2024-01-02 to 2024-12-31
- 2024-07-01 to 2025-06-30
- 2025-04-01 to 2026-03-28

Runtime was about 42 minutes wall-clock. The bottleneck is now the
single-cut `run_sim_104.py` simulation path, not the outer orchestrator.

## Bug Found And Fixed

The first parallel run crashed all three cuts before simulation because
`scripts/run_wf_gate.py` did not bootstrap the repo root and strategy dir
onto `sys.path`. `cut_market_context()` imports both top-level research
helpers and strategy-local pipeline modules:

- `kernel.hmm_regime_labels`
- `kernel.regime_labels`
- `kernel.panel_pipeline.panel_scorer`

Fix: `run_wf_gate.py` now inserts both `REPO` and `STRATEGY_DIR`.

## Result Of Manifest-Scoped Probe

After the import fix, the three cuts completed:

| Cut | Sharpe | APY | SPY Sharpe | Delta Sharpe |
| --- | ---: | ---: | ---: | ---: |
| 2024-01-02 to 2024-12-31 | -0.380 | -0.50% | +1.778 | -2.158 |
| 2024-07-01 to 2025-06-30 | -0.360 | -1.00% | +0.715 | -1.075 |
| 2025-04-01 to 2026-03-28 | -0.030 | +3.90% | +0.749 | -0.779 |

WF verdict:

- Mean Sharpe: **-0.257**
- Positive cuts: **0 / 3**
- Beat SPY Sharpe: **0 / 3**
- Beat SPY APY: **0 / 3**

Sanity battery:

- Real IC: **+0.0775**
- Shuffled IC: **-0.0016**, passes
- Placebo IC: **+0.0394** versus threshold `0.5 * real_ic = +0.0388`, fails by **+0.0006**

## Critical Scope Caveat

This probe used `strategy_config.sim_wl200.json`, whose `walkforward.enabled`
path evaluates a walk-forward manifest:

```text
artifacts/sim/walkforward_manifest_merged.json
```

Therefore the probe **did not directly evaluate** the candidate artifact
`artifacts/prod/panel-ltr.alpha158_fund.json`. The gate correctly stamped
`candidate_artifact_used=false`, and model acceptance should reject this
metadata for promotion.

## Follow-Up Fix

The WF gate default strategy config has been changed from
`strategy_config.sim_wl200.json` to `strategy_config.json` so weekly promote
and research acceptance evaluate the actual candidate artifact by default.
The manifest path remains useful for manifest-level research, but it must be
passed explicitly and must not be accepted as candidate-artifact evidence.
