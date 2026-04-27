# Cross-Sectional Transformer Panel Backend

**Status:** Implemented but **not** in production. Per Bug #24 audit (2026-04-26) the hourly transformer produces NEGATIVE OOS IC vs XGBoost's +0.0482 — currently shelved behind acceptance gates.
**Owner:** panel-LTR backend ensemble track.

## 1. Why a transformer (and why not)

### Why
Today's XGBoost `rank:pairwise` booster is a tree ensemble whose score for ticker *i* on date *t* depends **only on that row's features** (`f_i,t`). It cannot model:

1. **Cross-ticker context** — tickers are ranked *against each other* per date, but the booster never sees the other tickers on the same bar. Attention across a date-group lets the model learn things like "tech tickers in the same cohort look similar today → sector-wide momentum regime" or "this ticker is the only name in its sector with positive earnings surprise → stronger signal."
2. **Per-ticker history** — XGBoost gets only the latest feature row; a transformer with a short per-ticker window (e.g. 20d) can learn dynamics like "ROE trending up over 3 quarters."
3. **Interactions** — tree ensembles model interactions but only axis-aligned; attention is general.

### Why not (yet)
- **Data efficiency** — XGBoost is brutally efficient with our ~80k panel rows. Transformers want 10–100×. We overfit. Chen-Pelger-Zhu 2024 ship gate ≈ 5000 dates for transformer generalization on cross-sectional asset pricing; we have ~2500.
- **Interpretability** — XGBoost feature importance + monotone constraints are production-ready audit tools; transformer is opaque.
- **Inference latency** — LEAN per-bar; PyTorch on MPS ≈50ms vs XGBoost ~1ms. Doable but adds friction.
- **Diminishing returns** — IC 0.038 → 0.066 came from XGBoost + better features. The honest next frontier is **new data** (analyst revisions, options flow, macro), not model capacity.

**Principle:** treat transformer as **alternative ranking backend**, not a replacement. Ensemble preferred over swap.

---

## 2. Architecture

### Shape of the data

Each training sample is a **date-group**: all tickers observed on the same bar *t*.
- Per-bar: up to ~99 tickers (current watchlist)
- Per-ticker-bar: 24 features (16 neutralized indicators + 8 factor z-scores)
- Labels: Gaussianized β-neutral residual forward return
- Daily OOS: ~2500 dates × 99 tickers ≈ 225k panel rows
- Hourly OOS: ~2500 sessions × 7 hours × 99 tickers ≈ 1.5M panel rows

### Model: date-wise cross-sectional attention

```
              Input: (B, T, F)   where B=batch of dates, T=tickers/date, F=features
                               │
                      ┌────────▼────────┐
                      │ Feature Encoder │    (Linear F → d_model)
                      └────────┬────────┘
                               │
              ┌────────────────▼────────────────┐
              │   N × Transformer Encoder Blocks │   self-attention WITHIN date-group
              │   - Multi-head attention (T)     │   → tickers see each other same-bar
              │   - Feed-forward + LayerNorm     │   → never across dates
              └────────────────┬────────────────┘
                               │
                      ┌────────▼────────┐
                      │  Score Head     │    (Linear d_model → 1)
                      └────────┬────────┘
                               ▼
                       Score per (ticker, date)
```

Padding mask handles variable group sizes (auto-bumped per `Bug #23` fix to fit largest train+eval group).

### Hyperparameters (current defaults)

| Param | Value | Rationale |
|---|---|---|
| d_model | 128 | Small panel — don't go big |
| n_heads | 4 | d_model / 32 |
| n_layers | 3 | Shallow — overfit guard |
| dropout | 0.20 | Aggressive |
| feedforward_dim | 512 | 4× d_model |
| max_tickers_per_date | auto-bump (Bug #23) | Was 128 → grows to fit largest group (601 / 616 etc.) |
| optimizer | AdamW lr=1e-4, wd=5e-4 | Standard |
| loss | ListNet | Continuous, scale-invariant ranking loss |
| max_epochs | 30 | Early stop at patience=6 |
| batch_size | 32 dates | |

### Regularization

1. **Label smoothing** σ=0.05 on Gaussianized labels.
2. **Ticker dropout** — mask a random 10% of tickers per batch; forces the attention layer to learn cross-sectional structure not ticker-identity shortcuts.
3. **Feature dropout** 0.2 at input.
4. **Walk-forward CPCV** — same purged + embargoed splits as XGBoost.

---

## 3. Two training resolutions

### 3.1 Daily (default)

Standard panel: 1 row per (ticker, date). Goes through `BuildPanelTask` → `CrossValidateTask` → `FinalFitTask`. Backend selectable via `panel_ltr.backend = "transformer"`. Sunday sweep (April 2026) showed CPCV mean OOS IC = -0.0029 — **shelved on data thinness** (~2500 dates < Chen-Pelger-Zhu threshold).

### 3.2 Hourly (Stage C-2 design)

Each ticker contributes ~7 rows per session × ~2500 sessions = ~15k rows per ticker. Cross-section panel grows from ~225k to ~1.5M rows.

**Activation flag:** `panel_ltr.training_resolution = "hourly"` (default `"daily"`).

**Pipeline dispatch:** `BuildHourlyResolutionPanelTask` replaces `BuildPanelTask` when flag flipped. Per `pp_panel_training.py`:
- Reads from `data/intraday/{SYM}/1h.parquet` cache (populated by `scripts/fetch_hourly_bars.py`)
- Calls `build_hourly_resolution_panel(bars, label_horizon_bars=7, benchmark_bars=spy_hourly)`
- Forward-fills daily fundamentals onto hourly grid (existing pattern in `LoadFundamentalsTask`)
- `group_sizes` keyed by `(date, hour)` — transformer attention happens within each cross-section snapshot

**Coverage:** 83 of 101 watchlist tickers have hourly history. Tickers without it are dropped for hourly training (logged warning).

### 3.3 Stages shipped (April 2026)

| Stage | Status | What |
|---|---|---|
| C-1 | ✅ | `kernel.intraday_wash` + `hourly_resolution_panel` scaffold; 35 unit tests |
| C-2 | ✅ | `BuildHourlyResolutionPanelTask` wired into PanelTrainingPipeline |
| C-3 | ✅ training works | First successful run after Bug #21 (DateTime64), Bug #23 (max_tickers), Bug #24 (timestamp leak), and the silent-kill from Bash background tool. v6 (round-7) produced negative IC due to Bug #24 — **rolled back** via auto-bak |

---

## 4. A/B evaluation + ship gate

| Metric | XGBoost | Transformer | Verdict |
|---|---|---|---|
| OOS mean IC (CPCV 15-split) | +0.0482 | -0.0008 (v6) | rejected by acceptance gate G4 |
| Per-fold std | baseline | similar | not the issue |
| OOS q05 | baseline | worse | concerns |
| Training time | ~10s | ~5–8 min | acceptable trade-off |
| Inference time (~99 tickers) | ~1ms | ~40ms | acceptable |
| Sim ROI | baseline | not yet measured | gate |

**Ship gate:** transformer ships to production iff:
- IC ≥ 1.3× XGBoost (≥+0.063 absolute)
- Sim ROI ≥ XGBoost on 27-mo OOS
- Per-fold IC std ≤ XGBoost × 1.2
- Inference < 100ms per bar

If transformer wins, **prefer ensemble** (avg XGBoost + transformer ranks) — lower variance, keeps audit trail.

---

## 5. Production safety

### 5.1 Acceptance gates (auto-rollback for bad models)

The v6 hourly transformer's NEGATIVE IC would now be **automatically rejected**. Gate G4 (`oos_mean_ic ≥ prior × 0.7`) blocks promotion; prior `panel-ltr.json` is preserved. ntfy alert fires. No live trade uses the bad model. See [`../../backtesting/renquant_104/kernel/model_acceptance.py`](../../backtesting/renquant_104/kernel/model_acceptance.py).

### 5.2 Auto-bak on shim write

When backend=transformer, `SaveArtifactTask` ALSO writes a shim `panel-ltr.json` so `fit_panel_calibrator.py` can find the new artifact. Pre-fix this destroyed the production XGBoost trees. Now (Bug #141 fix): the existing `panel-ltr.json` is auto-backed-up to `panel-ltr.{prev_kind}.bak.json` BEFORE the shim write. Recovery is `cp panel-ltr.xgboost.bak.json panel-ltr.json`.

### 5.3 Fixes shipped 2026-04-26 round-7

| Bug | Fix | Tests |
|---|---|---|
| #21 DateTime64 in spearmanr (FeatureDiagnosticTask) | dtype filter | 7 |
| #23 max_tickers=604 < eval group 616 | auto-bump considers train+eval | 5 |
| #24 timestamp leaked into feature_cols → look-ahead bias → negative IC | dtype filter at feature_cols construction | 5 |
| #141 shim clobbered XGBoost without auto-bak | snapshot prior before shim | 1 (rewrote) |

### 5.4 Detached training (operator note)

Bash-background-task watchdog can kill training silently. Use `nohup ... &` for >10-minute training runs to fully detach.

---

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Overfit (training IC stellar, OOS poor) | Aggressive dropout + label smoothing + purged CPCV (same splits as XGBoost) |
| Pad mask wrong → attention leaks across dates | Unit tests; `Bug #23` fix made auto-bump consider both train+eval |
| MPS non-determinism | `torch.use_deterministic_algorithms(True)` where MPS supports |
| Black-box ranking → hard to audit | Keep XGBoost primary until ensemble track ships |
| Inference latency spikes | Pre-warm + `torch.profiler`; CPU fallback if forward >200ms |
| Hourly bars cover 83/101 tickers (bias) | Drop tickers with insufficient history; log skipped count |

---

## 7. Cross-references

- **Implementation**: [`training_panel/transformer_model.py`](../../backtesting/renquant_104/training_panel/transformer_model.py)
- **Pipeline integration**: [`training_panel/pp_panel_training.py`](../../backtesting/renquant_104/training_panel/pp_panel_training.py) (`BuildHourlyResolutionPanelTask`, `TransformerCVTask`, `TransformerFitTask`, `SaveTransformerTask`)
- **Inference scorer**: [`kernel/panel_pipeline/transformer_scorer.py`](../../backtesting/renquant_104/kernel/panel_pipeline/transformer_scorer.py)
- **Rust scorer (production inference)**: [`rust/transformer_scorer/`](../../rust/transformer_scorer/)
- **Promotion runbook**: [`../ops/transformer-promotion.md`](../ops/transformer-promotion.md)
- **Acceptance gates**: [`../../backtesting/renquant_104/kernel/model_acceptance.py`](../../backtesting/renquant_104/kernel/model_acceptance.py) — protects every retrain
- **Audit history**:
  - [`../archives/audits/2026-04-25-ngboost-tx.md`](../archives/audits/2026-04-25-ngboost-tx.md) (round-3)
  - [`../archives/audits/2026-04-26-transformer.md`](../archives/audits/2026-04-26-transformer.md) (round-6)
- **Companion docs**:
  - [`panel-ltr.md`](panel-ltr.md) — primer for the consuming model
  - [`macro-factor-frame-design.md`](macro-factor-frame-design.md) — cross-asset features (likely ship before transformer)
- **Reference**: Vaswani et al. 2017 (Attention Is All You Need); Cao et al. 2007 (ListNet); Chen-Pelger-Zhu 2024 (Deep Learning in Asset Pricing).
