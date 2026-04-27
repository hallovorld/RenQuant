# renquant_104 — Cross-Sectional Transformer Panel Model (Design)

**Status:** design phase. Not enabled in production. A/B against XGBoost LTR before any production switch.
**Owner:** panel-LTR backend ensemble track.
**Related:** [`renquant_104_design.md`](renquant_104_design.md) (base strategy), [`panel_ltr_primer.md`](panel_ltr_primer.md).

---

## 1. Why a transformer (and why not)

### Why
The current XGBoost `rank:pairwise` booster is a tree ensemble whose score for ticker *i* on date *t* depends **only on that row's features** (`f_i,t`). It cannot model:

1. **Cross-ticker context**: tickers are ranked *against each other* per date, but the booster never sees the other tickers on the same bar. Attention across a date-group lets the model learn things like "tech tickers in the same cohort look similar today → sector-wide momentum regime" or "this ticker is the only name in its sector with positive earnings surprise → stronger signal."
2. **Per-ticker history**: XGBoost gets only the latest feature row. A transformer with a short per-ticker window (e.g. 20d) can learn dynamics like "ROE trending up over 3 quarters" or "momentum has been fading despite a still-positive level."
3. **Interactions** between the 25+ features we have — tree ensembles model interactions but only axis-aligned; attention can model arbitrary pairwise interactions more efficiently.

### Why not (yet)
- **Data efficiency**: XGBoost is brutally efficient with our ~80k panel rows. Transformers typically want 10–100× that. We may overfit.
- **Interpretability**: XGBoost's feature-importance + monotone constraints are production-ready audit tools; a transformer is a black box.
- **Inference latency**: LEAN runs per-bar; PyTorch loads + forward on MPS is ~50ms vs XGBoost's ~1ms. Doable but adds friction.
- **We already pushed IC 0.038 → 0.066 with XGBoost + better features.** The honest next frontier is **new data** (analyst revisions, insider trades, options flow), not model capacity.

**Principle:** treat the transformer as an **alternative ranking backend**, not a replacement. If A/B shows +30% OOS IC over XGBoost on the same feature set (→ 0.085+), ship it as a second scorer and ensemble. If not, keep XGBoost primary.

---

## 2. Architecture

### Shape of the data

Each training sample is a **date-group**: all tickers observed on the same bar *t*. The panel today has:

- Per-bar: up to **38 tickers** active (watchlist size)
- Per-ticker-bar: **25 features** (16 neutralized indicators + 9 factor z-scores)
- Labels: Gaussianized β-neutral residual forward return, one scalar per (ticker, t)
- OOS: ~2247 dates × 38 tickers ≈ 80k panel rows (same as XGBoost input)

### Model: date-wise cross-sectional attention

```
              Input: (B, T, F)   where B=batch of dates, T=tickers/date, F=features
                               │
                      ┌────────▼────────┐
                      │ Feature Encoder │    (Linear F → d_model)
                      └────────┬────────┘
                               │
                   Optional: ticker embedding  (learnable, size=|watchlist|)
                   Optional: sector embedding  (size=|sectors|)
                               │
              ┌────────────────▼────────────────┐
              │   N × Transformer Encoder Blocks │   attention within each date-group
              │   - Self-attention (T tickers)   │   → tickers see each other same-bar
              │   - Feed-forward                  │
              │   - LayerNorm + residual          │
              └────────────────┬────────────────┘
                               │
                      ┌────────▼────────┐
                      │  Score Head     │    (Linear d_model → 1, per ticker)
                      └────────┬────────┘
                               │
                               ▼
                         Score per (ticker, date)
```

**Key mechanism**: self-attention operates **within each date-group**. Tickers on the same bar attend to each other; tickers on different bars never interact (identical to XGBoost LTR groups). A padding mask handles variable group sizes (fewer than 38 tickers on days with listings still ramping up).

### Hyperparams (initial)

| Param | Value | Rationale |
|---|---|---|
| d_model | 128 | Small — ~80k rows doesn't need big |
| n_heads | 4 | d_model / 32 |
| n_layers | 3 | Shallow — guard against overfit |
| dropout | 0.3 | Aggressive — limited data |
| feedforward_dim | 256 | 2× d_model |
| max_tickers_per_date | 38 | = watchlist size; pad shorter groups |
| optimizer | AdamW lr=1e-4, wd=1e-4 | Standard |
| loss | ListNet (softmax over scores per group) | Ranking loss, scale-invariant |
| epochs | 50 with early stopping | Short training |
| batch_size | 32 dates | ≈ 1216 panel rows per batch |

**Why ListNet not pairwise**: ListNet (Cao et al., 2007) is continuous, differentiable, and operates on the full within-group score distribution — better fit for transformers than pairwise hinge. Also cheaper: O(T) per group vs O(T²) for pairwise.

### Regularization (multi-layer — panel IC 0.066 means true signal is weak; overfitting kills us)

1. **Label smoothing** on the Gaussianized labels (add small noise σ=0.05 to break ties during training).
2. **Ticker-conditional dropout**: with probability 0.1 per ticker per batch, mask the ticker's features to zero — forces the attention stage to learn robust cross-sectional structure, not ticker-identity shortcuts.
3. **Explicit feature dropout** at input: 0.2 drop rate on the 25 features.
4. **Walk-forward training**: match XGBoost's purged CV splits exactly. Same folds, same embargo.

---

## 3. Training pipeline integration

**Pipeline principle (CLAUDE.md rule 1b):** everything is a Task/Job.

New Jobs + Tasks slot into `training_panel/pp_panel_training.py`'s Phase 4 as an **alternative** to `PanelModelJob`:

```
PanelTrainingPipeline:
  Phase 1: PanelDataJob         (fetch OHLCV, sector mom, fundamentals)  [unchanged]
  Phase 2: PanelFeatureJob      (per-ticker neutralize+factor, parallel) [unchanged]
  Phase 3: PanelAssemblyJob     (z-score, labels, build panel)           [unchanged]
  Phase 4: PanelModelJob        (XGBoost CV + fit + save)                [unchanged]
  Phase 4b: TransformerModelJob (PyTorch fit + save, NEW — when enabled) [NEW]
  Phase 5: PanelNGBoostJob      (NGBoost Normal μ,σ head)                [unchanged]
```

### Phase 4b — new Job + Tasks

```python
# backtesting/renquant_104/training_panel/transformer_model.py   (new file)
class PanelTransformerModel:
    """PyTorch-backed cross-sectional ranker.

    Same interface as PanelLTRModel — train(panel, group_sizes, feature_cols)
    + predict(panel) → pd.Series — so PanelScoringJob can load either.
    """

# backtesting/renquant_104/training_panel/pp_panel_training.py   (additions)
class TransformerCVTask(PanelTask):      # same CV shape as CrossValidateTask
class TransformerFitTask(PanelTask):     # fit on full panel
class SaveTransformerTask(PanelTask):    # serialize .pt + metadata.json

class TransformerModelJob(PanelJob):
    def should_skip(self, ctx):
        cfg = ctx.config.get("panel_ltr", {})
        return str(cfg.get("backend", "xgboost")) != "transformer" and \
               not cfg.get("train_transformer_parallel", False)
    tasks = (TransformerCVTask, TransformerFitTask, SaveTransformerTask)
```

### Config keys

```jsonc
"panel_ltr": {
    "backend": "xgboost",                 // switch: xgboost | transformer
    "train_transformer_parallel": false,  // also train transformer for A/B even when backend=xgboost
    "transformer_params": {
        "d_model": 128, "n_heads": 4, "n_layers": 3,
        "dropout": 0.3, "feedforward_dim": 256,
        "lr": 1e-4, "weight_decay": 1e-4,
        "max_epochs": 50, "patience": 8,
        "device": "mps"                   // auto-detect fallback: mps → cuda → cpu
    }
}
```

### Artifact format

```
artifacts/panel-transformer.pt        # torch state_dict
artifacts/panel-transformer.json       # metadata (feature_cols, hparams, oos IC, train date)
```

Mirrors XGBoost's `panel-ltr.json` pair so `PanelScorer.load()` can dispatch by file extension.

---

## 4. Inference integration

### Scorer dispatch

```python
# kernel/panel_pipeline/panel_scorer.py  (additions)
def load(path: Path) -> "PanelScorer":
    if path.suffix == ".pt" or path.with_suffix(".pt").exists():
        return TransformerPanelScorer.load(path)
    return XGBoostPanelScorer.load(path)   # current
```

`PanelScoringJob` is **unchanged** — it already treats the scorer as an opaque object with `.score(X) → pd.Series`. The transformer scorer exposes the same interface.

### Per-bar inference path

LEAN + live runner + SimAdapter all build the inference feature matrix the same way (`prepare_inference_panel_frames` → `build_inference_matrix`). The transformer's `predict(X)` reshapes `X` into a date-group (single date, multiple tickers), runs through the model, flattens back to a per-ticker Series. No other pipeline change.

### Latency

Measured on M2 Pro MPS: ~30–50ms for a 38-ticker batch, single date. Under LEAN's per-bar budget. Pre-warming the model (one dummy forward on init) avoids the first-bar cold-start tax.

---

## 5. A/B evaluation protocol

Before flipping `backend: "transformer"` in production, train BOTH with `train_transformer_parallel: true` and compare:

| Metric | XGBoost | Transformer | Verdict |
|---|---|---|---|
| OOS mean IC (CPCV 15-split) | required baseline | must beat | +30% absolute IC to flip |
| Per-fold std | | | lower is better |
| OOS q05 (worst 5% of dates) | | | lower (i.e. less negative) is better |
| Training time | ~10s | ~5 min | acceptable trade-off |
| Inference time (38 tickers) | ~1ms | ~40ms | acceptable |
| Feature importance stability | stable (monotone constraints) | — | soft concern |
| Sim ROI on OOS window | baseline | must match or beat | primary ship gate |

**Ship gate:** transformer ships to production iff all of: IC ≥ 1.3× XGBoost, sim ROI ≥ XGBoost, per-fold IC std ≤ XGBoost × 1.2, infers in <100ms per bar.

If the transformer wins decisively, consider **ensemble** (average XGBoost + transformer ranks per bar) rather than replacement — lower variance and keeps the XGBoost monotone-constraint audit trail alive.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfit → stellar training IC, poor OOS | Aggressive dropout + label smoothing + purged CPCV (same splits as XGBoost) |
| Pad mask wrong → attention leaks across dates | Unit test: compare attention patterns on a known pad mask vs manually zeroed group |
| MPS non-determinism | Set torch seed + `torch.use_deterministic_algorithms(True)` where possible on MPS |
| Black-box ranking → hard to audit | Keep XGBoost as primary until confidence established; require transformer + XGBoost agree on ranking direction for each top-slot ticker as a production gate |
| Inference latency spikes during MPS memory pressure | Pre-warm + profile with `torch.profiler`; fall back to CPU if forward > 200ms |

---

## 7. Implementation plan

**Session A (design — this doc):** approved / revised.

**Session B (prototype, ~6–8 hrs):**
1. `training_panel/transformer_model.py` — `PanelTransformerModel` class with `train` / `predict` / `save` / `load`. 200-300 lines.
2. `training_panel/pp_panel_training.py` — `TransformerCVTask`, `TransformerFitTask`, `SaveTransformerTask`, `TransformerModelJob`. Reuse purged-CV splitter.
3. `kernel/panel_pipeline/transformer_scorer.py` — `TransformerPanelScorer` mirroring `PanelScorer` interface.
4. `PanelScorer.load()` dispatch.
5. Config defaults.
6. Tests:
   - Unit: model shape, pad mask, deterministic seed.
   - Integration: `TransformerModelJob` runs end-to-end on synthetic data.
   - A/B: script `scripts/compare_panel_backends.py` runs both on current panel, reports IC + sim ROI side-by-side.

**Session C (A/B + decision):** run the A/B, read the numbers, ship or shelve.

---

## 8. Open questions for user

1. **Do you want MPS deterministic mode on (slower but reproducible)** or non-deterministic (faster, results drift slightly between runs)? Default: deterministic for A/B trust, non-det for production inference.
2. **Ensemble or replace?** My strong recommendation is ensemble (average XGBoost rank + transformer rank) — keeps the XGBoost audit trail and lowers variance. Replace only if ensemble doesn't beat XGBoost alone.
3. **Training cadence?** Transformer training is 30× slower than XGBoost. Safe options: weekly (Sunday retrain slot already exists) or monthly. Default: weekly.
