# Checkpoint 2026-04-27 22:28 — post-NGBoost-feature-drift-fix

## What this is

Immutable snapshot of the four files that constitute the renquant_104 "golden" model + config, taken right after the P0 NGBoost retrain that fixed the 2026-04-27 zero-buy incident.

| File | Role |
|---|---|
| `panel-ltr.json` | Panel-LTR XGBoost rank:pairwise, 27 features, OOS IC=+0.0400 (15-fold CPCV) |
| `ngboost-head.json` | NGBoost μ,σ head, feature_cols match panel exactly (0 drift) |
| `strategy_config.json` | Active strategy config (objective=rank:pairwise, asset_embeddings.enabled=false) |
| `strategy_config.golden.json` | Synced to strategy_config.json (identical SHA) |

## SHA256 manifest

See `MANIFEST.sha256`. To verify: `shasum -a 256 -c MANIFEST.sha256`.

## What it fixes

Pre-fix `ngboost-head.json` was trained with 140+ macro feature columns (vxx/hyg/dgs10/cpiaucsl/...) that the current panel pipeline no longer produces (macro disabled). At inference, those columns were silently zero-filled → σ corrupted → `edge_sharpe = (μ - rf) / σ` compressed below the Gate B threshold (τ=0.10) → **all 10 buy candidates rejected on 2026-04-27**.

Post-fix: NGBoost retrained against current panel feature set. 27/27 cols match. No zero-fill.

Backstop: `kernel/panel_pipeline/job_panel_scoring.py::ApplyNGBoostTask` now hard-fails (skips NGBoost scoring, logs ERROR) when missing-cols-pct exceeds `ngboost.max_feature_drift_pct` (default 0.05).

## How to roll BACK to the pre-fix state

```bash
cp backtesting/renquant_104/artifacts/panel-ltr.pre-fix-2026-04-27.bak.json \
   backtesting/renquant_104/artifacts/panel-ltr.json
cp backtesting/renquant_104/artifacts/ngboost-head.pre-fix-2026-04-27.bak.json \
   backtesting/renquant_104/artifacts/ngboost-head.json
# revert strategy_config.json: set objective=rank:ndcg, asset_embeddings.enabled=true
```

## How to restore THIS checkpoint (if live ever drifts again)

```bash
CHK=backtesting/renquant_104/artifacts/checkpoint_2026-04-27_22h28
cp "$CHK/panel-ltr.json"            backtesting/renquant_104/artifacts/panel-ltr.json
cp "$CHK/ngboost-head.json"         backtesting/renquant_104/artifacts/ngboost-head.json
cp "$CHK/strategy_config.json"      backtesting/renquant_104/strategy_config.json
cp "$CHK/strategy_config.golden.json" backtesting/renquant_104/strategy_config.golden.json
shasum -a 256 -c "$CHK/MANIFEST.sha256"   # must report all OK
```

## Also touched this session (not in this checkpoint)

- `kernel/panel_pipeline/job_panel_scoring.py` — added drift detector (P2)
- `scripts/train_asset_embeddings.py` — fixed tz + OHLCV cache fallback bugs
- `archive/ablation_2026-04-27/` — 5 ablation side-configs archived
- `CLAUDE.md` — corrected baseline IC, marked T2-2 NO-GO + macro-as-panel NO-GO
