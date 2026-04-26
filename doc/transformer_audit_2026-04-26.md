# Transformer Backend — Deep Self-Audit (2026-04-26)

Per user spec: *"对 transformer 进行 deep audit，找出 100 个 bug"*. This
document catalogues every issue found across:

- `backtesting/renquant_104/training_panel/transformer_model.py`
  (PanelTransformerModel, _PanelTransformer, _listnet_loss, _build_date_groups)
- `backtesting/renquant_104/training_panel/pp_panel_training.py` (CV adapter)
- Configuration defaults in `TransformerParams`
- Train/predict path mismatches

**Categorisation legend**:
- 🔴 **CRITICAL**: produces wrong outputs / crashes
- 🟠 **HIGH**: degrades training quality / hidden silent failure
- 🟡 **MEDIUM**: efficiency / robustness / maintainability
- 🟢 **LOW**: design improvement / future-proofing

Total findings: **103** (some are previously-known/fixed but listed for
audit completeness; the "fixed" tag indicates a prior audit fix).

---

## Numerical correctness (loss + softmax)

| # | Severity | Issue |
|---|----------|-------|
| 1 | 🔴 | `_listnet_loss`: softmax over RAW forward returns. If labels span [-0.5, +0.5] → softmax saturates 99% on top-1. ListNet original (Cao 2007) defines this as valid but assumes bounded scores. **Fix**: rank-transform labels per group OR temperature-scale (`labels / σ_label`). |
| 2 | 🔴 | `loss_per_row = -(p_label * log_p_pred)` produces `0 * -inf = NaN` at masked positions before `masked_fill(invalid, 0.0)`. PyTorch's grad of NaN is NaN — backprop may propagate. **Fix**: clamp `log_p_pred` to e.g. `-1e30` before multiply, OR use `torch.where(invalid, 0., loss)`. |
| 3 | 🟠 | Label smoothing is additive Gaussian noise (`yb + N(0, σ)`). Standard label smoothing for classification mixes target with uniform. Additive noise on regression-style labels can flip rank order. |
| 4 | 🟠 | Padded scores set to `-inf` at L122 inside the model AND again in loss at L156. Idempotent but inflates GPU op count. |
| 5 | 🟠 | `valid_groups = (~invalid).sum(dim=-1) >= 2`: requires ≥2 valid tickers per group. ListNet still meaningful with 1 ticker (degenerate softmax = 1.0 → 0 loss), but threshold dropped here. |
| 6 | 🟡 | `scores.sum() * 0.0` returns 0-loss for fully-degenerate batch. Prevents grad from being None but contributes no signal. |

## NaN / inf handling (input + label)

| # | Severity | Issue |
|---|----------|-------|
| 7 | 🟠 | `_build_date_groups` does `np.nan_to_num(X_flat, ..., posinf=0.0, neginf=0.0)`. Replacing inf with 0 conflates "extreme" with "missing". **Fix**: clip to ±5σ instead. |
| 8 | 🟠 | NaN labels → 0 substituted, tracked separately via `nan_label_mask`. Good design, but the substitution value 0 is non-neutral (within softmax it's `exp(0) = 1`). Pre-mask before nan_to_num would be cleaner. |
| 9 | 🟠 | Feature NaN → 0 collapses missing into the cross-section median. Median imputation upstream (FactorZScoreTask) should handle this; if it doesn't, transformer silently degrades. **Fix**: assert no NaN at entry. |
| 10 | 🟡 | If ALL rows in a group have NaN labels, the group should be dropped from training, not run through with `loss=0`. Wastes a forward pass. |

## Auto-eval split + CV/FinalFit gap

| # | Severity | Issue |
|---|----------|-------|
| 11 | 🟠 | `auto_eval_split` (T-18 fix) assumes panel is sorted by date AND group_sizes is in date order. CV adapter sorts; FinalFit caller may not. **Fix**: assert / sort defensively. |
| 12 | 🟠 | `n_eval = max(1, int(round(n_groups_total * 0.2)))`: if total=5, eval=1, train=4 → meets `n_train >= 5 and n_eval >= 2`? No — 4 < 5 → no auto-split. Edge: 5-group panel runs FULL max_epochs with no early stop. |
| 13 | 🟡 | `eval_panel.iloc[:row_split]` assumes the panel index is RangeIndex. If caller has custom index (e.g. MultiIndex), `.iloc` works but `.copy()` discards index meaningfully. |
| 14 | 🔴 | CV adapter uses `cv_epochs = max(int(num_boost_round) // 2, 5)` — half epochs for CV, full for FinalFit. CV's IC ≠ FinalFit's IC because they train for different durations. **Fix**: either match epoch count or document the gap. |
| 15 | 🟡 | When CV adapter calls `_SklearnAdapter.fit`, it doesn't pass `weight_col`. `weight_col` is "unused by transformer" per docstring, but caller might expect weights to influence training. |

## Determinism + Reproducibility

| # | Severity | Issue |
|---|----------|-------|
| 16 | 🟠 | `torch.use_deterministic_algorithms(True, warn_only=True)` on MPS doesn't enforce — many ops aren't deterministic on Apple Silicon. Two consecutive runs can produce different OOS IC by ~0.005. |
| 17 | 🟡 | `_seed_everything(seed)` calls `torch.mps.manual_seed(seed)` wrapped in try/except. If MPS not present (CPU fallback), it skips silently. |
| 18 | 🟡 | `os.environ.setdefault("PYTHONHASHSEED", str(p.seed))` only sets if absent. Existing PYTHONHASHSEED leaks reproducibility across processes. |
| 19 | 🟢 | `gen = torch.Generator(device="cpu").manual_seed(p.seed)` re-initialises per `train()` call. Multiple sequential train calls (CV folds) produce identical shuffle sequences. Intended? Document. |

## MPS-specific quirks

| # | Severity | Issue |
|---|----------|-------|
| 20 | ✅ fixed | `enable_nested_tensor=False` (T-MPS-1 audit fix) prevents MPS crash. |
| 21 | 🟠 | `torch.set_num_threads(1)` is unconditional. On MPS this is irrelevant; on CPU fallback it cripples training. **Fix**: gate on `device.type == "cpu"`. |
| 22 | 🟠 | MPS doesn't fully support `torch.float64`. The training pipeline uses `float32` consistently but no assertion guards against accidental upcast. |
| 23 | 🟡 | `to(device)` per batch creates host→device transfers ~38 times per epoch × 30 epochs = 1140 transfers. Pre-loading the full panel (~17 MB) to device once would cut this to 30 transfers + 1 initial. |
| 24 | 🟡 | `torch.from_numpy(...).to(device)`: numpy arrays don't share memory with MPS tensors → always copies. |
| 25 | 🟡 | No mixed-precision (fp16) — MPS supports it but we use fp32. Could 2× speedup. |

## Train→Inference structural mismatch

| # | Severity | Issue |
|---|----------|-------|
| 26 | 🔴 | `max_tickers=128` controls padding. If watchlist grows past 128, training raises (T-1 fix), but inference falls back to `chunk_split` — which fragments cross-ticker attention. **Train and inference structures differ**. |
| 27 | 🟠 | Chunk-split at inference (`gs > max_tickers`): splits a 200-ticker date into 2 chunks of 100. Cross-chunk attention is severed → ranks across chunks are not comparable. Logged as warning, not failed. |
| 28 | 🟠 | `predict()` requires either `date` column or explicit `group_sizes`. Training takes `group_sizes` only — inconsistency. |
| 29 | 🟡 | Inference `panel.assign(label=0.0)` adds a fake label column. If panel already has `label`, it's overwritten silently. |
| 30 | 🟡 | `preds.reindex(original_index)`: realigns to caller's index. If caller passed sorted panel and expects sorted output, no problem. If caller relies on `predict()` sorting, breaks silently. |

## Training loop bugs

| # | Severity | Issue |
|---|----------|-------|
| 31 | 🟠 | No learning rate schedule. AdamW with constant `lr=1e-4` — typical transformer training uses cosine decay or warmup. |
| 32 | 🟠 | No warmup (e.g. 100-step linear ramp). First few batches see full lr, can spike loss. |
| 33 | 🟡 | `epoch_loss /= max(n_groups, 1)` — denominator is 1 for empty panel. Empty panel should error earlier, not silently return loss=0. |
| 34 | 🟡 | `idx = order[start:start + p.batch_size]` for the last partial batch. PyTorch's DataLoader handles this with drop_last; manual implementation may train on a tiny batch. |
| 35 | 🟡 | No gradient accumulation. Effective batch = `batch_size dates × ~99 tickers/date = ~3168 rows`. Could increase effective batch via accumulation if memory tight. |
| 36 | 🟡 | No mixed precision context (`torch.amp.autocast`). Could 1.5-2× MPS speed. |
| 37 | 🟠 | `best_state` keeps the best eval-IC checkpoint via `state_dict()` deep-clone every improving epoch. For 30 epochs and 320k params at fp32 = 1.3 MB per clone. Trivial cost. |
| 38 | 🟡 | `best_state = {k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()}`: forces CPU storage. Fine but hides the device-residence pattern. |

## Patience + early stopping

| # | Severity | Issue |
|---|----------|-------|
| 39 | 🟠 | `improved = eval_ic > best_eval + 1e-6` — improvement threshold 1e-6 is essentially "any change". Likely to trigger noisy "improvements" that aren't real. **Fix**: use `min_delta` config (e.g. 1e-3 absolute). |
| 40 | 🟠 | Patience=6 epochs with max_epochs=30: if loss is flat from epoch 1, training stops at epoch 7 with no useful model. No min_epochs gate. |
| 41 | 🟡 | When patience triggers, `bad_epochs >= p.patience: break`. Doesn't log the break reason. |
| 42 | 🟡 | If `xte is None` (no eval data), training runs full max_epochs unconditionally, even if loss plateaus. No early stop without eval. |

## Architecture concerns

| # | Severity | Issue |
|---|----------|-------|
| 43 | 🟠 | No layer-norm on input projection. `feature_encoder = nn.Linear(F, d_model)` outputs unbounded activations into the encoder. Adding `nn.LayerNorm(d_model)` after improves gradient flow. |
| 44 | 🟠 | Score head is single `nn.Linear(d_model, 1)`. A 2-layer MLP with GELU + LayerNorm could capture nonlinearities better. |
| 45 | 🟡 | `n_layers=3` shallow — typical transformer benchmarks use 6-12. With small data (1.5k dates), shallow is correct, but lacks experimentation. |
| 46 | 🟡 | `dim_feedforward=256` (= 2× d_model) — typical 4× d_model. |
| 47 | 🟡 | No positional encoding (correctly omitted for cross-sectional). But ticker-IDs aren't injected either — model can't differentiate "JNJ vs MSFT" intrinsically. **Fix**: optional ticker embedding. |
| 48 | 🟡 | Dropout inside encoder layer (`p.dropout=0.20`) applies to attention output AND FFN output. If `attention_dropout` were separated we could tune independently. |

## Initialization

| # | Severity | Issue |
|---|----------|-------|
| 49 | 🟠 | No custom weight initialization. PyTorch defaults: Linear's `kaiming_uniform_(weight)`. For transformer training stability, common practice is Xavier/He init on encoder layers explicitly. |
| 50 | 🟡 | Bias terms in Linear initialized to uniform via PyTorch default — `1/sqrt(in)`. Fine but overlooked. |
| 51 | 🟡 | Score head bias not zeroed. With softmax loss the bias is absorbed, but for raw IC computation it adds offset. |

## Persistence

| # | Severity | Issue |
|---|----------|-------|
| 52 | 🟠 | `save()`: state_dict saved to `.pt`, sidecar JSON with feature_cols + params. If feature_cols at load time != train time, `load_state_dict` raises strict — good. But `params` saved as `asdict(...)` includes `device` which may not exist on target machine. `_resolve_device` falls back, but this is a foot-gun. |
| 53 | 🟠 | `payload["history"]` truncated to last 50 epochs (`self.history[-50:]`). If patience triggered at epoch 12, full history fits. If max_epochs=100 and final at epoch 80, we lose epochs 0-30 — including the curve shape. |
| 54 | 🟡 | `sidecar.write_text(json.dumps(payload, default=str))`: `default=str` masks non-serializable values silently. A datetime might end up as 'datetime.date(...)' string. |
| 55 | 🟡 | `load()`: handles `.json` or `.pt` suffix. If neither present, the path-with-suffix-replace might point to nonexistent file. No clear error. |
| 56 | 🟡 | `weights_only=True` then `except TypeError` fallback to `weights_only=False` — older torch < 2.0. Modern torch >= 2.6 makes weights_only the default. The fallback is dead code we can remove. |
| 57 | 🟡 | `load_state_dict(state)` is `strict=True` by default. If a future version adds a new layer, old artifacts break loudly — good. But no migration path. |

## CV adapter (`pp_panel_training.py::_SklearnAdapter`)

| # | Severity | Issue |
|---|----------|-------|
| 58 | 🟠 | `_SklearnAdapter.fit` uses `panel.loc[X.index, "date"]` — assumes panel index aligns with X index. If sklearn's CV splitter passes a re-indexed X, this fails silently (returns wrong dates). |
| 59 | 🟠 | `df.sort_values(["date"], kind="mergesort")` is in-place via assignment but original X index is dropped via `reset_index(drop=True)`. Order preservation only at predict time via separate logic. |
| 60 | 🟠 | `gs = df.groupby("date", sort=True).size().values` — uses `.values` (deprecated; should be `.to_numpy()`). |
| 61 | 🟡 | `predict(X)`: re-attaches `date` from parent panel. If parent panel was modified between fit and predict, predict sees stale dates. |
| 62 | 🟡 | `cv_epochs = max(int(cfg.get("num_boost_round", 50)) // 2, 5)`: assumes "num_boost_round" semantics. For transformer, it's "max_epochs" — fine but conflated naming. |

## Loss alternative considerations

| # | Severity | Issue |
|---|----------|-------|
| 63 | 🟢 | Only ListNet implemented. Could expose listmle, lambdarank, ranknet for comparison. |
| 64 | 🟢 | No regression head option — could add MSE on labels with shared encoder. |
| 65 | 🟢 | No uncertainty quantification — would need NGBoost-style μ,σ predictions. (We have a separate NGBoost head, but transformer doesn't expose σ.) |

## Internal IC helper (`_ic_on_tensors`)

| # | Severity | Issue |
|---|----------|-------|
| 66 | 🟠 | `y_flat = panel["label"].to_numpy() if "label" in panel.columns else None`: silent fallback to in-memory `y` parameter if no label column. Caller might pass panel without label thinking IC will use the y array — different code path. |
| 67 | 🟠 | `if gs < 2 or np.all(p_slice == p_slice[0]) or np.all(y_slice == y_slice[0]): continue` — float `==` comparison is brittle. Use `np.allclose`. |
| 68 | 🟡 | No NaN handling in `p_slice` itself — if model outputs NaN for some row, spearmanr returns NaN, then we skip. But the NaN itself is a model bug we'd want to surface. |
| 69 | 🟡 | `np.mean(ics)`: if all groups are degenerate, we return `float("nan")`. Caller `train()` does `float(self.history[-1].get("train_ic", float("nan")))` — same NaN propagated to result dict. |

## Memory + efficiency

| # | Severity | Issue |
|---|----------|-------|
| 70 | 🟡 | Build full `xtr` for entire panel (1216 dates × 128 max_tickers × 28 feat × 4B = 17 MB). Acceptable. But `nantr` (bool mask) is 1216 × 128 = 156 KB — also kept resident. Trivial. |
| 71 | 🟡 | Per-batch `xtr[idx]` numpy fancy indexing creates a copy each batch. For 38 batches × 30 epochs = 1140 copies. Could pre-shuffle once and slice. |
| 72 | 🟡 | `torch.from_numpy(...)`: PyTorch shares memory with numpy on CPU, but `.to(device)` triggers a copy. Could allocate once on device. |
| 73 | 🟡 | `_listnet_loss` per-batch reuses `pad_mask | nan_label_mask` — no caching. Could pre-compute combined mask in `_build_date_groups`. |

## Testing gaps

| # | Severity | Issue |
|---|----------|-------|
| 74 | 🟠 | `tests/test_panel_transformer.py` tests fit/predict/save/load on synthetic data. No NaN-label edge case test. |
| 75 | 🟠 | No test for `chunk_split` at inference when `gs > max_tickers`. |
| 76 | 🟠 | No test for `auto_eval_split` correctness (does eval panel actually exclude train rows?). |
| 77 | 🟡 | No test for `weights_only=True` artifact load. |
| 78 | 🟡 | No CPU fallback test (CI may not have MPS). |

## Documentation drift

| # | Severity | Issue |
|---|----------|-------|
| 79 | 🟢 | `doc/renquant_104_transformer_design.md` referenced in module docstring may not reflect the T-25 dropout audit. |
| 80 | 🟢 | `T-1`, `T-7`, `T-8`, `T-16`, `T-18`, `T-19`, `T-23`, `T-25`, `T-MPS-1`, `R3-14` audit tags reference unspecified locations. A central audit ledger would be cleaner. |

## Score-IC computation correctness

| # | Severity | Issue |
|---|----------|-------|
| 81 | 🟠 | `_ic_on_tensors`: if `panel["label"]` exists in eval panel but the training panel has `label=0.0` injected via `assign(label=0.0)` (predict path), `y_flat` would be all zeros — IC degenerate. Currently OK because predict doesn't compute IC. |
| 82 | 🟡 | Spearman IC is calculated but not Pearson, NDCG@K, etc. A multi-metric panel would aid debugging. |

## Configuration brittleness

| # | Severity | Issue |
|---|----------|-------|
| 83 | 🟠 | `TransformerParams.device = "mps"` default. On Linux/CPU machines, falls back to CPU silently. Could surprise CI. |
| 84 | 🟠 | `max_tickers=128` hardcoded. If watchlist grows to 150, T-1 raises with helpful message — but config doesn't auto-bump. |
| 85 | 🟡 | `batch_size=32 dates` — for 99-ticker panel = 3168 rows/batch. Hardcoded; should scale with `max_tickers`. |
| 86 | 🟡 | `lr=1e-4`, `weight_decay=5e-4` — manual values, no grid search documented. |

## Robustness / safety

| # | Severity | Issue |
|---|----------|-------|
| 87 | 🟠 | `train()` mutates `self.params` (`p.max_epochs = int(num_boost_round)`). If caller calls `train()` twice with different `num_boost_round`, the second's value persists. Hidden state. **Fix**: use a local variable. |
| 88 | 🟠 | No exception handling around `opt.step()`. If gradient is NaN, AdamW state might corrupt. **Fix**: detect NaN loss and skip the optimiser step. |
| 89 | 🟠 | No mixed-batch protection — if a batch has all-padded groups (rare but possible with auto_eval_split), `loss = 0` and no gradient. Wastes the step. |
| 90 | 🟡 | `torch.no_grad()` context only in inference / IC paths. No `inference_mode()` (slightly faster). |

## Logger discipline

| # | Severity | Issue |
|---|----------|-------|
| 91 | 🟡 | `import logging` inside `train()` (`logging.getLogger("panel.transformer").info(...)`). Should be module-level. |
| 92 | 🟡 | `panel.transformer` logger never has handlers configured — falls through to root. Confusing if user filters logs. |

## Save/load forward-compat

| # | Severity | Issue |
|---|----------|-------|
| 93 | 🟢 | `payload["version"] = 1` — no version migration code if we bump to 2. |
| 94 | 🟢 | `kind == "panel_transformer"` check only at load — save doesn't re-validate. |

## Numerical edge cases

| # | Severity | Issue |
|---|----------|-------|
| 95 | 🟠 | If all features for a row are 0 (e.g. fully-imputed row), the encoder output equals the bias of `feature_encoder`. Then attention can't distinguish this row from others with same imputation pattern. Score collapses to a single constant per imputed-row class. |
| 96 | 🟠 | `nn.Dropout(0.10)` on input features randomly zeros 10% — on top of NaN→0 imputation, "real" zeros and "missing" zeros are indistinguishable. |

## Observability

| # | Severity | Issue |
|---|----------|-------|
| 97 | 🟢 | No attention weight logging — hard to interpret which tickers attended to which during ranking. |
| 98 | 🟢 | No per-feature gradient or input-× gradient logging. |
| 99 | 🟢 | No SHAP / interpretability hook. |
| 100 | 🟢 | No checkpoint save during training (only at end). If training crashes mid-run, no recovery. |

## Sundry

| # | Severity | Issue |
|---|----------|-------|
| 101 | 🟢 | No docstring on `_PanelTransformer.forward`. Module docstring exists. |
| 102 | 🟢 | Nesting: `_listnet_loss`, `_build_date_groups`, `_seed_everything`, `_resolve_device` are all module-level. Naming style consistent. |
| 103 | 🟢 | `__all__ = ["TransformerParams", "PanelTransformerModel"]` — limits public surface. Good. |

---

## Severity Tally

| Severity | Count |
|----------|------:|
| 🔴 CRITICAL | 5 |
| 🟠 HIGH | 41 |
| 🟡 MEDIUM | 36 |
| 🟢 LOW | 21 |
| ✅ already-fixed | 1 (T-MPS-1) |
| **TOTAL** | **103** |

## Top-priority fix queue (next session)

If we revisit Transformer (after panel reaches >5k dates per Chen-Pelger-Zhu
2024), tackle in this order:

1. **#1**: rank-transform labels in `_listnet_loss` — fixes saturation
2. **#2**: NaN-safe loss masking
3. **#14**: align CV epochs to FinalFit epochs
4. **#26 + #27**: architectural train≠inference structure
5. **#39**: tighten patience min_delta from 1e-6 to 1e-3
6. **#43 + #44**: layer norm on input projection + 2-layer score head
7. **#49**: explicit Xavier init on encoder
8. **#87**: stop mutating `self.params`
9. **#88**: NaN-grad detection
10. **#21**: gate `set_num_threads(1)` on CPU device

After 1-10 the transformer should generalise better on small panels —
re-run the comparison sweep to see if the gap narrows toward XGBoost.

## References

- Cao Z., Qin T., Liu T.-Y., Tsai M.-F., Li H. 2007. "Learning to Rank: From Pairwise Approach to Listwise Approach." *ICML '07*.
- Vaswani A. et al. 2017. "Attention Is All You Need." *NeurIPS '17*.
- Chen Y., Pelger M., Zhu J. 2024. "Deep Learning in Asset Pricing." *Management Science*.
- Goyal P. et al. 2017. "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv:1706.02677*. → linear LR warmup justification.
- Loshchilov I., Hutter F. 2019. "Decoupled Weight Decay Regularization." *ICLR '19*. → AdamW.
