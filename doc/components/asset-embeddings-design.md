# Asset Embeddings — Design Plan (A2 from Tier 2 roadmap)

**Status**: Plan / not yet implemented (2026-04-27).
**Reference**: Dolphin et al. 2024 KDD ("Contrastive Learning of Asset Embeddings from Financial Time Series", arXiv 2407.18645).
**Roadmap entry**: `doc/roadmap.md` Tier 2 #T2-2 (lowest-risk evidence-backed win).
**Expected impact**: +3 pts F1 on sector classification benchmark, -19% volatility on hedging tasks (paper claims).

---

## Why this is the lowest-risk Tier 2 win

Asset embeddings are a **feature-only addition**:
- No backend swap (still XGBoost ranker on top)
- No retrain orchestration change (just add columns to the panel matrix)
- Embeddings learn from per-asset OHLCV history alone — same data we already cache
- If they help, OOS IC goes up. If they hurt, drop the columns and revert.

Compare to T2-1 (LightGBM swap, requires backend integration), T2-3 (regime-conditional ensemble, 4 separate models), T2-4 (Boyd cvxpy rotation, replaces greedy sorter).

---

## What the paper does

Dolphin et al. trains contrastive embeddings on OHLCV time series:

1. **Anchor / positive / negative sampling**: for each asset, anchor = its own time series; positive = same asset different time window OR a correlated asset; negative = uncorrelated asset.
2. **Encoder**: small temporal CNN (1D convolutions over price/volume/return history). Output = D-dimensional embedding (D ∈ {16, 32, 64}).
3. **Loss**: InfoNCE / triplet loss — embedding distance reflects asset similarity.
4. **Output**: per-asset embedding vector (one per ticker, fixed at "as-of" date).

The embeddings capture **asset-pair relationships** that a per-ticker model cannot see (sector, industry, factor exposure) without manual labeling.

Reported gains:
- Sector classification: +3 F1 points vs baseline correlation features
- Hedging vol reduction: -19% on out-of-sample portfolios
- Cross-asset prediction: improves whenever the downstream model uses pairwise similarity

---

## Implementation plan (3 phases)

### Phase A — Embedding training (offline, weekly cron)

**Module**: `backtesting/renquant_104/training_panel/asset_embeddings.py` (new file, ~300 LoC)

```python
# Sketch
class AssetEmbeddingTrainer:
    """Train D-dim contrastive embeddings on watchlist OHLCV history."""

    def __init__(self, embedding_dim: int = 16, lookback_days: int = 504,
                 negative_sample_pool_size: int = 50):
        ...

    def fit(self, ohlcv: dict[str, pd.DataFrame],
            as_of_date: pd.Timestamp) -> dict[str, np.ndarray]:
        """Returns {ticker: D-dim embedding vector}."""
        # 1. Build (anchor, positive, negative) triplets per ticker:
        #    - anchor: ticker's last `lookback_days` OHLCV → temporal CNN
        #    - positive: same ticker, sliding window from earlier history
        #    - negative: random sample from `negative_sample_pool_size`
        #      tickers with low (<0.3) historical correlation
        # 2. Train via PyTorch / mlx with InfoNCE loss
        # 3. After convergence (~30 epochs), forward-pass each ticker
        #    once to get its embedding
        # 4. Persist to artifacts/asset-embeddings.json:
        #    {ticker: [d_0, d_1, ..., d_15], "trained_date": ...,
        #     "embedding_dim": 16, "loss_history": [...], "panel_shape": ...}

    def smoke_test_collapse(self, embeddings: dict) -> bool:
        """Reject if embeddings collapsed (cosine sim > 0.95 between
        random pairs → all tickers look identical, useless feature)."""
        ...
```

**Driver**: `scripts/train_asset_embeddings.py` (new, ~80 LoC)
- CLI: `--strategy --embedding-dim --lookback-days --epochs --device {mps,cpu}`
- Reads watchlist from strategy_config.json
- Loads OHLCV via `kernel.data.fetch_ohlcv`
- Calls `AssetEmbeddingTrainer.fit()`
- Smoke-test collapse → write artifact OR fail with non-zero exit

**Cron**: weekly Sunday alongside `screen_watchlist.py`

### Phase B — Wire into panel-LTR features

**Module**: `backtesting/renquant_104/training_panel/panel_frame.py` (extend `build_panel_frame`)

```python
def build_panel_frame(
    feature_frames, factor_frames, labels, ticker_sectors, *,
    asset_embeddings: dict[str, np.ndarray] | None = None,  # NEW
    ...,
):
    ...
    # Append per-ticker embedding columns (broadcast — ticker's
    # embedding is constant across its rows in the panel)
    if asset_embeddings is not None:
        for ticker, emb in asset_embeddings.items():
            for i, v in enumerate(emb):
                panel.loc[panel["ticker"] == ticker, f"emb_{i}"] = v
        # Add emb_0..emb_15 to feature_cols_set
        feature_cols_set.update(f"emb_{i}" for i in range(len(next(iter(asset_embeddings.values())))))
```

**Update**:
- `pp_panel_training.py::PanelDataJob` — load `artifacts/asset-embeddings.json` if exists, attach to ctx
- `pp_panel_training.py::BuildPanelTask` — pass `ctx.asset_embeddings` to `build_panel_frame`
- `kernel/panel_pipeline/feature_matrix.py::build_inference_matrix` — append `emb_0..N` columns at inference time (broadcast same value per ticker as training)
- `training_panel/pipeline.py::prepare_inference_panel_frames` — load asset_embeddings, return as 4th tuple element (preserve symmetry guard test)

### Phase C — Acceptance gate validation

Add a new gate G12 that compares pre/post-embedding OOS IC to ensure embeddings actually help:

```python
def _gate_g12_embedding_lift(staging, active, *, min_lift_pct: float = 0.0):
    """G12 (soft): if both prior and new have embedding metadata, the
    new IC must beat prior by at least min_lift_pct (0% default = at
    least non-negative lift). Catches cases where embeddings drift
    becomes net-negative."""
```

Plus: extend `select_best_model.py` to surface "embedding generation" as a candidate naming convention (`panel-ltr.embeddings-d16.bak.json`).

---

## Risk + mitigation

| Risk | Mitigation |
|---|---|
| Embedding training time | Cap at 30 epochs; ~5min on MPS for 99 tickers × 504-day lookback |
| Collapsed embeddings (all tickers similar) | `smoke_test_collapse()` rejects if random-pair cosine > 0.95 |
| Embeddings drift week-to-week (unstable) | Track Frobenius distance between consecutive weeks; alert if > threshold |
| Inference loads stale embeddings | Add staleness check at inference start (if embedding age > 14 days, log warning) |
| Embeddings overfit per-ticker noise (leak future info) | Train on data ending `lookahead_days` BEFORE the panel labels — strict purge |
| 16 extra features dilute prod's existing signal | If A/B shows IC drop, set `panel_ltr.asset_embeddings.enabled: false` (default OFF) |

---

## Acceptance criteria (when to ship)

Before promoting embeddings to default-on:
1. Phase A complete: artifacts/asset-embeddings.json exists, smoke test passes
2. Phase B complete: train_104.py with embeddings produces a valid artifact (G1-G7 pass)
3. A/B sim: OOS IC ≥ prod (no degradation tolerance — these are extra features, must add value)
4. A/B sim: APY ≥ prod (no degradation tolerance)
5. Memory + inference time check: 99 tickers × 16-dim adds 1.5KB to artifact, ~20µs to inference. Acceptable.
6. Operator manual review of correlations: `cosine_similarity` matrix between sector clusters should look sensible (tech tickers cluster, defensives cluster, etc.) — sanity guard against learned noise

---

## Out of scope (Phase D+)

- **Pair trading via embeddings**: extending sell/buy gates to use embedding similarity for hedge candidate selection. Big surface area; defer.
- **Sector reassignment**: using embeddings to override `sector_map`. Conflicts with manually curated sectors; defer.
- **Cross-strategy embeddings**: shared across 103/104. Adds coordination overhead; defer.
- **Online (streaming) embeddings**: incremental updates per bar. Phase A re-trains weekly which is sufficient given OHLCV is daily.

---

## Effort estimate

| Phase | Effort | Risk |
|---|---|---|
| A — embedding trainer + cron | 1 day | Low (well-bounded) |
| B — wire into panel + inference | 1 day | Medium (must preserve train/inference symmetry) |
| C — G12 gate + validation | 0.5 day | Low |
| Total | ~2.5 days | Low overall |

---

## Open decisions (operator pre-implementation)

1. **Embedding dim D**: 16, 32, or 64? Paper recommends 16 for "compact" cases. Start with 16; revisit if A/B shows underfitting.
2. **Architecture**: temporal CNN (paper) or simpler MLP? CNN handles temporal structure better but adds 200 LoC. Start with CNN.
3. **Negative sampling**: use historical correlation < 0.3 as negative criteria, OR random sampling? Correlation-based is paper's choice; random is simpler. Use correlation-based.
4. **Train / re-train cadence**: weekly Sunday vs daily? Weekly should suffice — embeddings are slow-moving; daily adds noise + cost.
