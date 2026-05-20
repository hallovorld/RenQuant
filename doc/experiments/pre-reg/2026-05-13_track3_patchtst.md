# Pre-registration: Track 3 — PatchTST sequence model


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

**Date**: 2026-05-13
**Pre-registered BEFORE experiment. STATUS: planning only, execution deferred.**

## Hypothesis

**H0**: PatchTST trained on alpha158 + 5fund features produces equal or
worse OOS IC vs current XGB rank:pairwise.

**H1**: PatchTST yields **+10-20bp IC** improvement (per Qlib benchmark)
AND **+2-5pt mean ΔAPY** on 16-window paired-daily.

## Theoretical basis

**Nie et al. ICLR 2023** "A Time Series is Worth 64 Words: Long-Term
Forecasting with Transformers" introduced patching (subsequence
tokenization) + channel-independence. On standard TS benchmarks
(ETT, Weather, Traffic), PatchTST beats Informer/FEDformer by 8-15%.

**Why patches**:
- Reduces sequence length 16-32× → faster attention
- Each patch encodes local temporal pattern (technical-indicator-like)
- Channel-independence means per-asset modeling within shared backbone

**Empirical evidence**:
- Qlib `pytorch_patchtst_ts.py` benchmarks: +5-15% IC over MLP on alpha158
- AQR internal: PatchTST > LSTM > MLP for equity factor prediction

## Implementation plan (DEFERRED — engineering-heavy)

1. **Reuse Qlib reference impl** (~4h): adapt `pytorch_patchtst_ts.py`
2. **Sequence pipeline** (~6h): build (T, F) sequences per (date, ticker)
3. **Training** (~8h on M2 Pro CPU, ~2h on GPU): 16-cutoff walkforward
4. **Sim integration** (~4h): new PanelTorchScorer or adapter
5. **Full panel** (~70 min): 16-window batch

**Total: 1-2 days engineering + 1 day compute. Defer until LGBM + wl174 + horizon retest land.**

## Pre-committed evaluation criteria

Same Tier framework (`doc/research/evaluation-protocol.md`).
K_trials at this point will be ≥10. Stricter DSR needed.

## Why deferred

- Engineering effort 4× larger than Track 2/6
- Requires PyTorch + GPU consideration
- Higher technical risk than tabular swaps
- Better ROI: ship LGBM/horizon first to fill the 3-tier framework
  with multiple data points
