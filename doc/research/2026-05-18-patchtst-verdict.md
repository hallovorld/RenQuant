# 2026-05-18 — PatchTST verdict: NOT a winner over XGB


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

## TL;DR

PatchTST on alpha158 features ≈ XGB on alpha158 features. **Architecture change alone doesn't solve trend-following problem** — both models learn the same mean-reversion bias from the same feature set.

## Numbers

| Model | Best val_ic | Test_IC (best epoch) | Training cost |
|---|---|---|---|
| **XGB** (5 seeds, wl200 panel) | ~+0.12-0.15 | **+0.034 ± 0.023** | 30 sec / seed |
| **PatchTST MPS** (seed 42, 10 epochs) | +0.050 (ep 1) | ~+0.04 (predicted, killed early) | 30 min / seed |
| **PatchTST CPU** (seed 42, 1 ep) | +0.040 (ep 1) | — | 3 min / ep × 10 = 30 min |

**MPS == CPU** (val_ic differ by 0.01, within seed variance). MPS is not unreliable for this workload.

## Why PatchTST overfit so hard

Seed 42 epoch-by-epoch trajectory (MPS):
```
ep01: train +0.17  val +0.050  ← best
ep02: train +0.29  val -0.151   memorize → val crash
ep03: train +0.39  val -0.127
ep04: train +0.47  val -0.058
ep05: train +0.54  val -0.028
ep06: train +0.58  val -0.135
ep07: train +0.62  val -0.067
ep08: train +0.65  val -0.112
ep09: train +0.68  val -0.086
```

Model capacity 0.19M params with 200k samples = ratio 1/1050 — theoretically fine. But:
- Cross-sectional rank loss + small dataset = severe overfit signature
- Transformer attention has high expressivity that XGB tree splits don't
- alpha158 features are mostly redundant correlated; transformer learns spurious patterns

## Why this isn't a trend-following solution

User's original ask: "I want a model that follows trends, not buy losers" (paraphrased).

PatchTST CANNOT solve this by itself because:
1. **Input is the same alpha158 ratios** (RSV, QTLU, MIN, MAX, ROC — all mean-rev encodings)
2. **Label is cross-sectional fwd_60d_excess** (relative performance, NOT absolute trend direction)
3. **Training data shows what learners learn** — if data says "low-RSV stocks outperform peers", model learns to buy low-RSV (= mean-reversion)

Changing XGB → PatchTST = changing the COOK while keeping the same ingredients & recipe → same dish.

To get trend-following behavior, need to change:
- **Features**: add raw OHLCV sequences, time-series momentum (NOT cross-sectional)
- **Label**: predict ABSOLUTE direction (binary "up next 60 days") not relative-rank
- **Strategy**: separate strategy (renquant_106 CTA) with its own DNA

## Decision

**Close PatchTST research thread.** Not promote.

PatchTST has limited value-add given:
- Equal or slightly worse OOS than XGB
- 60× higher training cost
- Same conceptual limitations (learns from same features)

## What we keep

- `transformer_v4.py` infrastructure — useful for future ML research
- `artifacts/patchtst_*` historic runs for reference
- pandas_ta_classic momentum features (still useful for XGB experiments)
- Learning: **architecture changes don't fix feature-driven biases**

## Pivot

User explicitly rejected dual-strategy (E) and rules-based pivots. The thread is **within model-architecture space** (D). Options inside D:

1. **Different sequence length** (32 → 64 → 128 days) — capture longer-horizon patterns
2. **Different label** (binary "outperform top-decile" classification, not regression) — easier signal
3. **Different transformer variant** (TFT with covariate masking, iTransformer with better config)
4. **Smaller model** (under 50K params to fight overfit on small dataset)
5. **Different feature inputs to PatchTST** (raw OHLCV sequences directly, NOT alpha158 pre-cooked features)

Awaiting user direction within D.

## Time saved by killing 5-seed early

5 seeds × ~25 min remaining = ~100 min compute saved. Seed 42 result was already representative (overfit pattern identical across earlier smoke runs).
