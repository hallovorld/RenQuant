# Phase C — Temporal Graph Convolution (NN backend)

**Status:** Design only. NOT shipped. Branch will be `exp/graph-attention-nn` when work begins.
**Trigger:** Layer 1 + Layer 2 retrain on wl178 fails to lift CPCV mean_ic ≥ +0.020.
**Reference:** Feng-Chen-He-Yang-Cao 2019 ("Temporal Relational Ranking for Stock Prediction", arXiv:1809.09441, TOIS 2019). OSS: [`fulifeng/Temporal_Relational_Stock_Ranking`](https://github.com/fulifeng/Temporal_Relational_Stock_Ranking).

---

## Why this layer (post Layer 1+2 outcome)

If Layer 1 + Layer 2 (per-sector rank-norm + sector one-hot) doesn't move CPCV mean_ic past noise, the issue isn't representation — it's the model architecture. Tree-based rank-pairwise loss extracts pairwise information from cross-sectional features but cannot encode **per-pair relational** information (ticker A and ticker B share a sector → their forward returns should correlate; tree models can split on individual features but not on pair-level relations).

Feng et al. 2019 established that injecting an explicit graph relation (sector co-membership, supply-chain edges, etc.) via a Temporal Graph Convolution module on top of a per-ticker LSTM beats independent-stock baselines by ~98% return-ratio on NYSE. The architecture explicitly handles cross-sector heterogeneity by routing same-sector pairs through a shared GAT layer that learns sector-conditional representation.

This is the published-canonical fix for the wl178 fitting failure.

---

## Architecture

```
   Per-ticker raw features  (T tickers × D features × N dates)
                             │
                             ▼
      Per-ticker temporal encoder (LSTM or 1D-CNN)
        h_t,i ∈ ℝ^H                           ← per-ticker hidden state at date t
                             │
                             ▼
   Build sector graph G  (T × T binary; edge iff same sector)
                             │
                             ▼
   Temporal Graph Convolution Layer (GAT-style)
     h'_t,i = σ( W_self · h_t,i + Σ_{j ∈ neighbors(i)} α_{ij} · W_neigh · h_t,j )
     where α_{ij} = softmax over neighbors of attention(h_t,i, h_t,j)
                             │
                             ▼
   Per-ticker score head  (linear + dropout)
     score_t,i = w · h'_t,i  + b
                             │
                             ▼
   Pairwise rank loss (LambdaMART / pairwise BCE)  on cross-section per date
```

**Key invariants:**
- Sector graph is RECOMPUTED each bar from current ticker_sectors mapping (handles future ticker additions or sector reclassifications).
- Edge weights are LEARNED via attention, not fixed (Feng's contribution — older work used fixed adjacency).
- Same-ticker self-edge always present (h_self preserved through residual).
- Attention is masked to within-sector neighbors only (prevents the model from learning spurious cross-sector edges).

---

## Why this should work where Layer 1+2 didn't

| Layer | What it adds | Why insufficient on its own (hypothesis) |
|---|---|---|
| L1 | Per-sector rank features `_sr` | Tree splits on individual features; can't encode pair-level info |
| L2 | Binary sector identity `sector_*` | Same — tree splits on identity, but doesn't aggregate across same-sector tickers |
| **L3 (this)** | **Sector co-membership as graph attention** | Lets the model EXPLICITLY aggregate same-sector tickers' representations before scoring — pair-level rather than feature-level interaction |

The fundamental failure mode of Layer 1+2 is that the rank-pairwise loss compares ticker A vs ticker B on their _features_, which can encode "A is in sector X" but not "A and B share sector X, therefore their score difference should be modulated by sector-level state." Graph attention is the canonical encoding of that pair-level relation.

---

## Implementation phases (when work begins)

### C.1 — Module + tests (week 1)
- New file `backtesting/renquant_104/kernel/panel_pipeline/graph_scorer.py`
  - `GraphAttentionScorer` PyTorch module (~300 LoC)
  - `.score(panel_frame, sector_map)` — inference API matching existing `PanelScorer.score`
  - `.train(panel_frame, labels, sector_map, **kwargs)` — training API
  - Artifact format: state_dict + sector_taxonomy + hyperparams
- Tests:
  - Forward-pass invariant: same-input → same-output (determinism with fixed seed)
  - Gradient sanity: backward pass on a 5-ticker synthetic panel
  - Attention masking: cross-sector edges have zero gradient
  - Save/load roundtrip (artifact integrity)

### C.2 — Pipeline integration (week 2)
- New `Task` in `kernel/panel_pipeline/`: `GraphScoreTask` mirrors existing `ApplyScoresTask` but reads from `GraphAttentionScorer` artifact when `panel_ltr.graph.enabled=true`.
- Acceptance gate per-graph-model trained.
- Default OFF.
- B2 hold-out validation comparing graph backend to current XGBoost + Layer 1+2 baseline.

### C.3 — Promote (week 3, conditional)
- Promote criterion: graph backend beats Layer 1+2 baseline by ≥ +2 APY pts on B2 hold-out.
- Update `golden_config_*.md` with graph backend if promoted.
- If not promoted: document in failed-experiments-log; preserve branch as historical artifact.

---

## Hardware considerations

- M2 Pro (current dev machine): GAT module on a 178-ticker × 750-date panel ≈ 134k forward passes per batch. Modest by NN standards. Should fit comfortably in MPS memory.
- Training time estimate: 2-4 hours per epoch on M2 Pro; ~5 epochs for convergence; total ~15-20 hours per full training run. Single B2 hold-out validation ≈ 1 day.
- Inference time: ~0.5s per bar (acceptable for both daily and intraday cadences).

---

## Open questions before implementation

1. **PyTorch vs MLX backend.** PyTorch MPS has documented gaps; MLX is Apple-native but adds a dependency. Default to PyTorch with `torch.compile` and accept the MPS gaps; revisit if performance is a blocker.

2. **Sector taxonomy depth.** Current `ticker_sectors` is GICS sector level (8-13 categories). For wl500, may need GICS sub-industry (~50 categories) for finer relations. Test both; pick by CPCV.

3. **Graph beyond sector.** Feng et al. also use Wikidata supply-chain edges. Out of scope for Phase C.1; possible Phase C.4 if sector-only graph proves insufficient.

4. **Loss function.** Stay with pairwise BCE (matches XGBoost rank:pairwise) for direct comparability with the tree baseline. Listwise (LambdaMART) is a Phase C.5 follow-up.

---

## Trigger conditions (do NOT start Phase C until)

- Layer 1+2 retrain on wl178 has finished and CPCV mean_ic recorded.
- If mean_ic ≥ +0.020: SHIP Layer 1+2; do NOT start Phase C.
- If mean_ic < +0.020: open Phase C with this design doc as the starting point.

Premature Phase C work without the simpler Layer 1+2 baseline first violates "smallest reversible step" (CLAUDE.md execution-cadence memory). NN backends are 10× the work of feature engineering; only justified by evidence the simpler interventions don't suffice.
