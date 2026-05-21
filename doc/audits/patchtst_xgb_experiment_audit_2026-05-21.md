# PatchTST / XGB Experiment Audit — 2026-05-21

## Executive Verdict

Claude Code's PatchTST experiment should be allowed to finish. It is still running and is useful as a directional architecture A/B among three PatchTST variants:

- HF PatchTST baseline
- HF PatchTST + cross-stock attention
- HF PatchTST + FiLM regime conditioning

But the current run is not sufficient to decide whether PatchTST should replace the production XGB panel-LTR model. It lacks a same-window XGB baseline inside the final comparator, and the PatchTST training script still applies label winsorization using full-panel quantiles after split assignment. That is mild label-distribution leakage.

Use this run to answer: "which PatchTST variant is most promising?"

Do not use this run alone to answer: "is PatchTST better than production XGB?"

## Live Run Status Observed

Observed active processes:

- `scripts/eval_hf_trainer_5cut_5seed.py`
- `scripts/eval_hf_cross_stock_5cut_5seed.py`
- `scripts/eval_hf_film_5cut_5seed.py`
- child workers running `scripts/patchtst_hf.py`

Observed completion counts at audit time:

| Arm | Artifact root | Complete summaries |
| --- | --- | ---: |
| HF PatchTST baseline | `artifacts/hf_trainer_5cut_5seed_pt07_clean` | 17 / 25 |
| Cross-stock attention | `artifacts/hf_cross_stock_5cut_5seed_pt07` | 14 / 25 |
| FiLM regime conditioning | `artifacts/hf_film_5cut_5seed_pt07_clean` | 14 / 25 |

Current recommendation: do not kill or restart these jobs unless memory pressure becomes destructive. Let them finish, then treat the aggregate as a PatchTST architecture screen.

## Concurrency Assessment

The experiment is partially concurrent, but not cleanly parallelized inside each driver.

Each of the three driver scripts loops serially over `CUTS × SEEDS`:

- `scripts/eval_hf_trainer_5cut_5seed.py`
- `scripts/eval_hf_cross_stock_5cut_5seed.py`
- `scripts/eval_hf_film_5cut_5seed.py`

The current machine is still running multiple arms concurrently because separate driver processes were launched. That means real concurrency exists at the arm level, but each arm advances one cut/seed at a time.

This is acceptable for a long background experiment on MPS. A naive 25-worker fanout would likely fight over the same GPU/MPS memory and slow down or destabilize the host. The better future design is a bounded scheduler:

- Max 1-2 MPS PatchTST workers at once.
- Resume/skip already completed `(cut, seed)` outputs.
- Separate CPU-heavy XGB baseline workers from MPS-heavy transformer workers.
- Write a single run manifest with PID, command, git commit, dataset fingerprint, and artifact root.

## Design Strengths

The experiment has several good scientific properties:

1. It uses 5 cuts × 5 seeds, not a single lucky validation window.
2. The walk-forward cuts target stressed regimes: COVID, Fed pivot, inflation peak, SVB, and 2024 unwind.
3. The splitter uses a 60-business-day purge/embargo for the 60-day forward label.
4. All three PatchTST arms use the same pt_07 knobs: `lr=1e-4`, `weight_decay=0.3`, `seq_len=24`, `epochs=8`.
5. Variant changes are isolated behind flags: `--cross-stock-attn` and `--film-regime-cond`.
6. Both cross-stock attention and FiLM are initialized as identity-style supersets of the baseline, which is good A/B hygiene.
7. Model selection uses `eval_min_regime_ic`, not only pooled IC. This matches the project principle that calm-regime validation is not enough.
8. HF Trainer uses `load_best_model_at_end=True`, which fixes the old "last epoch saved instead of best epoch" failure mode.

## Design Problems

### P0 — No Same-Window XGB Baseline In Final Comparator

The final comparator `scripts/compare_arch_5cut_5seed.py` reads `aggregate.csv` from the PatchTST arms. It does not include a same-cut, same-seed, same-feature XGB baseline in the active 3-way comparison.

Existing XGB comparison artifacts exist, for example:

- `artifacts/xgb_5seed_wl200_alpha158only`
- `artifacts/patchtst_shadow/xgb_baseline_same_val.json`

But these are not currently integrated into the same 5-cut × 5-seed comparator. Therefore the current experiment can rank PatchTST variants, but cannot prove PatchTST is competitive with production XGB.

Required fix before promotion:

- Add an XGB arm that uses the same dataset, same cuts, same label, same validation windows, same embargo, and same per-regime IC metric.
- Add it to `compare_arch_5cut_5seed.py`.
- Promotion gate should compare PatchTST's mean min-regime IC against XGB's mean min-regime IC, plus paired per-cut significance where possible.

### P0 — Label Winsorization Uses Full Panel Quantiles

In `scripts/patchtst_hf.py`, split assignment happens before preprocessing, but `winsorize_label()` computes quantiles on the whole panel:

```python
lo, hi = panel[label_col].quantile(pct), panel[label_col].quantile(1 - pct)
panel[label_col] = panel[label_col].clip(lower=lo, upper=hi)
```

Because the panel includes train, embargo, validation, and test rows at that point, validation/test label distribution influences the clipping thresholds. This is label-distribution leakage.

Expected fix:

- Compute label clipping thresholds from `split_label == "train"` only.
- Apply those train-only thresholds to train, embargo, validation, and test rows.
- Stamp `label_winsor_source = "train_only"` and the actual `lo/hi` values into each summary.
- Rerun the clean experiment before making promotion decisions.

This issue does not make the current running jobs worthless, but it does make them unsuitable as final acceptance evidence.

### P1 — No Resume/Skip Contract

Each driver reruns all 25 cases if restarted. There is no skip-if-summary-exists behavior and no manifest-level resume.

Expected fix:

- If `hf_patchtst_{cut}_seed{seed}_summary.json` exists and has matching config hash, skip.
- If a worker fails, record return code and log tail in `raw_results.json`.
- Do not overwrite completed results unless `--force` is passed.

### P1 — Failed Early Attempt Pollutes Logs

The cross-stock driver log contains an early `rc=-15` failed attempt before the restarted clean progress. The later progress appears valid, but mixed logs make audit harder.

Expected fix:

- Write each run under a unique `run_id` directory.
- Keep abandoned/restarted runs separate from the accepted run.

### P1 — Regime Labels Are Evaluation Labels, Not The Full Live Regime Stack

The experiment uses HMM/SPY regime labels for per-regime IC and FiLM context. This is fine for an architecture screen, but it is not exactly the live decision tree regime stack.

Expected fix before promotion:

- Run one acceptance pass through the real inference pipeline and inspect daily decision-tree output.
- Confirm shadow PatchTST scores do not create pathological behavior under BULL_CALM, CHOPPY, and BEAR.

## Mid-Run Signal Read

This is not a final result. It is only a mid-run read from logs.

Baseline so far:

- Strong on COVID and SVB cuts.
- Weak/negative on Fed 2022 cut.
- Mixed on inflation-peak cut.

Cross-stock attention so far:

- Similar pattern to baseline.
- Not obviously dominant from the partial logs.

FiLM so far:

- Some improvement on inflation-peak seed 44/45 relative to baseline.
- Still needs full 25/25 results before judging.

Important: a few positive seeds in one stressed cut are not enough. The final statistic should be min-regime IC averaged across all cuts and seeds, with a direct XGB comparator.

## Required Next Steps

1. Let the current jobs finish.
2. Run `scripts/compare_arch_5cut_5seed.py` only after all three `aggregate.csv` files exist.
3. Treat the output as PatchTST architecture selection only.
4. Patch train-only label winsorization.
5. Add same-window XGB baseline to the comparator.
6. Rerun the clean, resumable experiment.
7. Only consider promotion if PatchTST is not materially worse than XGB on min-regime IC and does not produce worse daily decision-tree behavior in shadow.

## Bottom Line

The experiment design is better than a casual one-off backtest: multi-cut, multi-seed, embargo-aware, regime-aware, and controlled across the three PatchTST variants.

But it is not yet a production-grade PatchTST-vs-XGB proof. The two blockers are same-window XGB comparison and train-only label preprocessing. Let it finish, learn from it, then rerun a clean acceptance version before changing the primary model.
