"""PyTorch cross-sectional transformer for the Stage-1 panel.

Alternative ranking backend to :class:`PanelLTRModel` (XGBoost). Mirrors the
same public surface so the caller can dispatch on `panel_ltr.backend`.

Architecture (see `doc/renquant_104_transformer_design.md` §2):

- Input: panel rows grouped by date → one date-group per sample.
- Feature encoder: ``Linear(F → d_model)`` + LayerNorm (audit fix #43).
- N × transformer encoder blocks (self-attention within date-group only).
- Score head: 2-layer MLP ``Linear→GELU→Linear(d_model → 1)`` (audit #44).
- Loss: ListNet over the date-group with rank-transformed labels (audit #1).
- Regularization: feature dropout, ticker-conditional dropout, label smoothing,
  AdamW weight decay, early-stopping with min_delta gate (audit #39).

Public API (same shape as :class:`PanelLTRModel`)::

    m = PanelTransformerModel(params=None)
    m.train(panel, group_sizes, feature_cols, num_boost_round=..., ...) -> dict
    m.predict(panel) -> pd.Series
    m.save(path, metadata=None)
    PanelTransformerModel.load(path)

The serialized artifact has suffix ``.pt`` (state_dict) paired with a
``.json`` sidecar holding feature_cols + hparams + training metadata.

2026-04-26 audit batch — top-10 fixes from doc/transformer_audit_2026-04-26.md:
  #1   rank-transform labels in _listnet_loss (eliminates ListNet saturation
       on raw forward returns)
  #2   NaN-safe loss masking (clamp log_softmax floor before multiply)
  #14  CV/FinalFit epoch alignment (no more silent half-epoch CV)
  #21  set_num_threads(1) gated on CPU device only (don't cripple MPS path)
  #39  patience min_delta tightened from 1e-6 to 1e-3
  #43  LayerNorm on input projection (better gradient flow)
  #44  2-layer score head with GELU
  #49  Xavier init on Linear layers (transformer-stability standard)
  #87  no mutation of self.params from inside train()
  #88  NaN-grad detection — skip optimizer step rather than corrupt AdamW state
"""
from __future__ import annotations

import json
import logging
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

# Audit fix #91 (2026-04-26): module-level logger instead of import-in-function.
log = logging.getLogger("panel.transformer")


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class TransformerParams:
    d_model:          int   = 128
    n_heads:          int   = 4
    n_layers:         int   = 3
    # Audit fix #46 (2026-04-26 round-3): bump dim_feedforward to 4x
    # d_model per Vaswani 2017 §3.3 standard. Pre-fix 2x was unusual
    # for transformer FFN — bumping to 4x adds capacity. d_model=128 →
    # 512 (still well under d_model=768/3072 in BERT base).
    feedforward_dim:  int   = 512
    # Audit fix T-25 (2026-04-25): pre-fix, 0.3+0.2+0.1 dropouts compounded
    # to ~50% effective (1-(1-.3)(1-.2)(1-.1) = 0.496) which over-regularised
    # a 121k-row panel and produced train_ic=0.30 vs OOS_ic=0.022 (7%
    # generalisation). New defaults: 0.20+0.10+0.0 = ~28% effective. Also
    # bumped weight_decay from 1e-4 → 5e-4 to add explicit L2 regularisation
    # since dropout was reduced. Reduced max_epochs 50→30 since the loss
    # curve plateaued ~ epoch 20 in v3.
    dropout:          float = 0.20       # attention/FF/residual dropout
    feature_dropout:  float = 0.10       # zero-out input features
    ticker_dropout:   float = 0.0        # disabled — overlapped with feature_dropout
    label_smoothing:  float = 0.05       # additive Gaussian noise on labels
    lr:               float = 1e-4
    weight_decay:     float = 5e-4
    max_epochs:       int   = 30
    batch_size:       int   = 32         # dates per batch
    patience:         int   = 6          # early-stopping on eval IC, per user pref
    # Audit fix #39 (2026-04-26): tighten min_delta from 1e-6 to 1e-3 so
    # noisy float-rounding "improvements" don't reset the patience counter
    # — only real ≥0.001 IC gains count.
    early_stop_min_delta: float = 1e-3
    grad_clip_norm:   float = 1.0        # T-16 audit fix — clip gradient norm
    auto_eval_split:  bool  = True       # T-18 — auto last-20% dates as eval if no eval_panel
    auto_eval_fraction: float = 0.20
    device:           str   = "mps"      # "mps" | "cuda" | "cpu"
    deterministic:    bool  = True
    seed:             int   = 42
    max_tickers:      int   = 128        # pad groups to this size (≥ watchlist size; was 38, silently truncated 99-ticker groups → audit T-1 2026-04-25)
    # Audit fix #1 (2026-04-26): rank-transform labels in ListNet loss so
    # raw forward returns don't saturate the softmax (top-1 dominates 99%
    # of the probability mass when |label_max - label_mean| > 0.1). Set to
    # False to fall back to raw labels for backward compat.
    rank_transform_labels: bool = True
    # Audit fix #40 (2026-04-26): minimum epochs before early stopping
    # can fire. Pre-fix, patience=6 with flat loss-from-start = stop at
    # epoch 7 with no useful weights. Floor: 5 epochs.
    min_epochs: int = 5
    # Audit fix #53 (2026-04-26): keep ALL history rather than last 50.
    # If max_epochs > 50, the early epochs (where the loss curve says the
    # most about overfit risk) were silently truncated.
    save_full_history: bool = True
    # Audit fix #31/#32 (2026-04-26 batch-3): cosine LR schedule + linear
    # warmup. Default: 10% of epochs warmup, then cosine decay to lr/100.
    # Setting `warmup_fraction=0.0` disables warmup; `cosine_decay=False`
    # disables cosine and uses constant lr.
    warmup_fraction: float = 0.10
    cosine_decay:    bool  = True
    cosine_min_lr_ratio: float = 0.01      # final lr = lr * this
    # Audit fix #27 (2026-04-26 batch-3): how to handle date groups
    # exceeding max_tickers at INFERENCE time. Options:
    #   "warn" (default): chunk-split + log warning
    #   "error": raise — forces user to raise max_tickers + retrain
    #   "silent": chunk-split without log (legacy)
    on_oversized_group: str = "warn"
    # Audit fix #48 (2026-04-26 round-3): split attention vs FF dropout.
    # Pre-fix, single `dropout` controlled both — no way to tune
    # independently. None means "fall back to dropout".
    attention_dropout: float | None = None
    # Audit fix #36 (2026-04-26 round-3): opt-in mixed precision.
    # Default False (fp32) for safety; flip to True on MPS/CUDA for
    # ~1.5-2x speedup on attention. Disabled on CPU (no AMP gain).
    use_amp: bool = False
    # Audit fix #71 #72 (2026-04-26 round-3): opt-in pre-load batches
    # to device. With ~17 MB panel, fits easily on MPS/CUDA — eliminates
    # ~1140 host→device transfers per training run. Default off for
    # backward compat. Enable for 1.2-1.5x speedup.
    preload_to_device: bool = False
    # Audit fix #82 (2026-04-26 round-3): track multiple eval metrics.
    # Pre-fix, only Spearman IC. Adds Kendall tau as secondary signal
    # — useful for diagnosing whether model captures rank order vs noise.
    log_kendall_tau: bool = False
    # Audit fix #35 (2026-04-26 round-3): gradient accumulation. Allows
    # effective batch size = batch_size × accumulation_steps without
    # using more memory. Default 1 (no accumulation). Useful when
    # MPS/CUDA RAM is tight but you want larger effective batch.
    grad_accumulation_steps: int = 1
    # Audit fix #45/#85/#86 (2026-04-26 round-3): hyperparameters
    # `n_layers`, `batch_size`, `lr`, `weight_decay` are all config-
    # driven via TransformerParams. Defaults are tuned for 1.5k-date
    # panels; larger panels may benefit from bumping n_layers (each
    # layer adds ~50k params at d_model=128) and lr (warmup helps
    # higher lr stability). Sunday sweep is the right place to
    # empirically tune these.


# ── Module ─────────────────────────────────────────────────────────────────────

class _PanelTransformer(nn.Module):
    """Per-date self-attention encoder + linear score head.

    Input  : x (B, T, F), pad mask (B, T)  (True = padding)
    Output : score (B, T) — one per ticker slot
    """

    def __init__(self, n_features: int, p: TransformerParams):
        super().__init__()
        self.p = p
        self.feature_encoder = nn.Linear(n_features, p.d_model)
        # Audit fix #43 (2026-04-26): LayerNorm on input projection.
        # Pre-fix, raw projected features fed unbounded activations into
        # the encoder → gradient instability + slower convergence. Now:
        # LayerNorm normalises to N(0, 1) per-feature before encoder.
        self.input_norm   = nn.LayerNorm(p.d_model)
        self.feat_dropout = nn.Dropout(p.feature_dropout)
        # Audit fix #48 (2026-04-26 round-3): allow separate attention
        # dropout. nn.TransformerEncoderLayer applies the SAME dropout
        # value to attention + feedforward + residuals. Splitting needs
        # a custom layer; for now, pick the higher of attention/FF if
        # attention_dropout is set, but document the limitation. Future:
        # custom encoder layer with separate dropouts.
        ff_dropout = p.dropout
        attn_dropout = p.attention_dropout if p.attention_dropout is not None else p.dropout
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = p.d_model,
            nhead           = p.n_heads,
            dim_feedforward = p.feedforward_dim,
            dropout         = max(attn_dropout, ff_dropout),
            batch_first     = True,
            activation      = "gelu",
        )
        # Audit fix T-MPS-1 (2026-04-25): disable nested-tensor optimization.
        # PyTorch's TransformerEncoder fast-path uses
        # `aten::_nested_tensor_from_mask_left_aligned` which is NOT
        # implemented for the MPS backend → retraining crashed mid-CV.
        # `enable_nested_tensor=False` falls back to the standard path
        # (slightly slower on CPU/CUDA, equally fast on MPS where the
        # optimization didn't work anyway).
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=p.n_layers,
            enable_nested_tensor=False,
        )
        # Audit fix #44 (2026-04-26): 2-layer score head with GELU.
        # Pre-fix, single Linear(d_model→1) gave the model no nonlinear
        # capacity to combine encoder features. 2-layer MLP captures
        # interactions while keeping output a scalar score.
        self.score_head = nn.Sequential(
            nn.Linear(p.d_model, p.d_model // 2),
            nn.GELU(),
            nn.Dropout(p.dropout),
            nn.Linear(p.d_model // 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Audit fix #49 (2026-04-26): explicit Xavier init on Linear layers.

        PyTorch defaults to ``kaiming_uniform_(weight, a=sqrt(5))`` for
        Linear which is too aggressive for transformer training stability.
        Xavier (Glorot) init is the literature-standard choice for
        attention models — see Vaswani 2017 §5.4 + Goyal 2017 (large
        minibatch SGD).
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """Run feature_encoder + LayerNorm + Encoder + score_head.

        Audit fix #101 (2026-04-26 round-3): explicit docstring on
        forward.

        Args:
            x: (B, T, F) input feature tensor (B = batch dates, T = max
               tickers per date with padding, F = feature count).
            pad_mask: (B, T) bool, True where slot is padding (no
                      ticker for that date).

        Returns:
            (B, T) raw scores. Padded slots are -inf so loss/argmax
            ignore them (also re-masked in _listnet_loss as defense).
        """
        # x: (B, T, F). pad_mask: (B, T), True where padding.
        x = self.feat_dropout(x)
        h = self.feature_encoder(x)                  # (B, T, d_model)
        h = self.input_norm(h)                       # audit #43
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        s = self.score_head(h).squeeze(-1)           # (B, T)
        # Audit fix #4 (2026-04-26 round-3): masking padded scores to
        # -inf inside forward() is REDUNDANT — _listnet_loss does the
        # same masking via `score_logits.masked_fill(invalid, -inf)`.
        # We still keep the mask here because predict() doesn't go
        # through the loss function and downstream code may compare
        # raw output. But annotated to flag the redundancy.
        s = s.masked_fill(pad_mask, float("-inf"))
        return s


# ── ListNet loss (top-1 softmax cross-entropy over group scores) ──────────────

def _rank_transform_per_row(labels: torch.Tensor,
                            invalid: torch.Tensor) -> torch.Tensor:
    """Convert labels to per-row ranks centred at 0, scaled to ~unit range.

    Audit fix #1 (2026-04-26): pre-fix, ListNet softmax over RAW forward
    returns saturated when any row had |label| > 0.1 — top-1 took 99% of
    the probability mass and the model was trained as a multinomial
    classifier on the single best ticker per date. New behaviour: replace
    each row's labels with their ranks (1, 2, ..., n_valid), centred and
    scaled. Softmax then produces a smooth distribution proportional to
    rank, which is the actual ListNet semantics intended by Cao 2007.

    Audit fix T-NEW-2 (2026-04-26 round-3): vectorised across batch dim
    using `argsort(...).argsort(...)` trick. Pre-fix Python loop did
    ~1140 batches × 32 dates × argsort calls ≈ 36k torch calls; now
    a single argsort per batch → 5-10× speedup on MPS.

    Invalid positions are masked to a sentinel (+inf-like value) before
    sorting so they sort to the END and don't interfere with valid ranks.
    Final out[invalid] = 0 (will be masked to -inf by caller).
    """
    # Push invalid positions to a finite sentinel beyond any real label
    # — they sort to end and we zero them out afterward.
    sentinel = torch.finfo(labels.dtype).max / 2
    masked = torch.where(invalid, sentinel, labels)
    # Vectorised rank: argsort(argsort(x)) gives the rank of each element.
    ranks = masked.argsort(dim=-1).argsort(dim=-1).to(labels.dtype)

    # Per-row valid count (denominator for centering)
    valid_count = (~invalid).sum(dim=-1, keepdim=True).to(labels.dtype)
    valid_count = valid_count.clamp(min=1.0)
    # Center: subtract per-row mean of valid ranks (mean of 0..n-1 = (n-1)/2)
    mean_rank = (valid_count - 1.0) / 2.0

    # Audit fix T-NEW-3 (2026-04-26 round-3): use the SAME scale as the
    # old Python-loop version (sample std of integer ranks 0..n-1).
    # Pre-fix, dividing by valid_count gave output range ~[-0.5, 0.5];
    # softmax was 3-30× flatter than the loop version → training signal
    # weakened. Now: divide by sample std of uniform 0..n-1 which is
    # √(n(n+1)/12). For n=99 this is ~28.7, matching old version's
    # `ranks.std()` ≈ 28.7. Output range becomes ~[-1.7, +1.7], same
    # softmax sharpness as the loop version.
    rank_std = torch.sqrt(valid_count * (valid_count + 1.0) / 12.0).clamp(min=1e-6)
    out = (ranks - mean_rank) / rank_std

    # Zero out invalid positions; caller masks them to -inf in softmax.
    out = out.masked_fill(invalid, 0.0)
    # Rows with < 2 valid → all-zeros (degenerate, caller skips via valid_groups guard).
    degenerate_rows = (valid_count.squeeze(-1) < 2)
    if degenerate_rows.any():
        out[degenerate_rows] = 0.0
    return out


def _listnet_loss(scores: torch.Tensor, labels: torch.Tensor,
                  pad_mask: torch.Tensor,
                  nan_label_mask: torch.Tensor | None = None,
                  rank_transform: bool = True,
                  ) -> torch.Tensor:
    """Cao 2007 top-1 ListNet.

    P(i) = softmax(label_i) over the group; loss = -sum P_label * log P_pred.

    Audit fix T-8 (2026-04-25): when a row had a NaN label originally
    (now zero-substituted by ``_build_date_groups``), it must be excluded
    from BOTH the label softmax and the prediction softmax — same as
    padding. Without this, NaN-rows had probability `exp(0)/Σ exp(yi)`,
    pulling predictions toward the median of valid labels.

    Audit fix #1 (2026-04-26): rank-transform labels before softmax
    when ``rank_transform=True`` (default). Pre-fix, raw forward
    returns saturated softmax → loss collapsed to a single-target
    classifier. With rank transform, labels become smooth distributions
    that ListNet was actually designed for.

    Audit fix #2 (2026-04-26): clamp log_softmax floor before multiply.
    Pre-fix, ``0 * log(0) = 0 * -inf = NaN`` (produced before masked_fill
    to 0), polluting the gradient. Now: clamp ``log_p_pred`` to a finite
    floor (-1e30) so the multiply never produces NaN.

    Args:
      scores         : (B, T) raw scores (model output)
      labels         : (B, T) labels (NaN → 0 substituted upstream)
      pad_mask       : (B, T) True where the slot is padding
      nan_label_mask : (B, T) True where the original label was NaN.
                       If None, treated as all-False (back-compat).
      rank_transform : if True, replace labels with per-row ranks before
                       softmax. Default True per audit #1.
    """
    if nan_label_mask is None:
        nan_label_mask = torch.zeros_like(pad_mask)
    invalid = pad_mask | nan_label_mask

    if rank_transform:
        labels = _rank_transform_per_row(labels, invalid)

    # Mask both padded AND NaN-label positions to -inf before softmax →
    # zero probability mass. ALSO mask the prediction softmax (so the
    # model isn't penalised for any score at NaN positions either).
    minus_inf = float("-inf")
    label_logits = labels.masked_fill(invalid, minus_inf)
    p_label = F.softmax(label_logits, dim=-1)
    score_logits = scores.masked_fill(invalid, minus_inf)
    log_p_pred = F.log_softmax(score_logits, dim=-1)

    # Audit fix #2 (2026-04-26): clamp -inf to a finite floor BEFORE the
    # multiply, so 0 * -inf doesn't produce NaN. The masked_fill below
    # zeroes the contribution at invalid positions either way, but NaN
    # in the intermediate tensor can still corrupt backward.
    log_p_pred = log_p_pred.clamp(min=-1e30)

    loss_per_row = -(p_label * log_p_pred)
    loss_per_row = loss_per_row.masked_fill(invalid, 0.0)
    # Mean over non-degenerate groups (at least 2 valid tickers).
    valid_groups = (~invalid).sum(dim=-1) >= 2
    if valid_groups.any():
        return loss_per_row.sum(dim=-1)[valid_groups].mean()
    # Audit fix #89 (2026-04-26 round-3): log when an entire batch is
    # degenerate (all groups have < 2 valid tickers). Indicates upstream
    # data pipeline issue — should not happen in normal operation.
    log.warning(
        "_listnet_loss: ENTIRE batch degenerate (all groups < 2 valid "
        "tickers) — returning 0 loss. Check upstream label/pad masks."
    )
    return scores.sum() * 0.0   # degenerate batch — zero loss


# ── Batch builder ──────────────────────────────────────────────────────────────

def _build_date_groups(
    panel: pd.DataFrame, group_sizes: np.ndarray,
    feature_cols: list[str], label_col: str,
    max_tickers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn a flat panel into per-date padded tensors.

    Returns (x, y, pad_mask, nan_label_mask) each of shape
    (n_groups, max_tickers, ·). x: float32, y: float32, pad_mask: bool,
    nan_label_mask: bool (True where the original label was NaN — kept
    so the loss can mask it out without confusing it with padding).
    Padding rows are zeros with pad_mask=True.

    Audit fixes (2026-04-25):
      T-1  ─ raise loud if any group exceeds `max_tickers`. The old
             code silently truncated to the first `max_tickers` rows
             per date and advanced offset by the FULL group size, so
             rows past the cap were dropped entirely (62% of the
             watchlist on a 99-ticker × 38-cap configuration). Callers
             that need chunk-splitting (predict path) must do it
             BEFORE entering this helper.
      T-7  ─ track which positions had NaN labels separately from
             padding. Loss masks both.

    Input sanitization (unchanged): NaN/±inf in features → 0 so the
    model's softmax stays numerically safe. Tree backends like XGBoost
    don't need this, but the transformer's softmax is sensitive to
    unbounded inputs.
    """
    if len(group_sizes):
        max_gs = int(np.max(group_sizes))
        if max_gs > max_tickers:
            raise ValueError(
                f"_build_date_groups: a group has {max_gs} rows but "
                f"max_tickers={max_tickers}. Either raise "
                f"TransformerParams.max_tickers ≥ {max_gs} or chunk-split "
                f"oversized groups before calling. Silent truncation has "
                f"been removed (audit T-1 2026-04-25)."
            )

    # Audit fix #22 (2026-04-26): force fp32 on entry. MPS doesn't fully
    # support fp64; downstream tensor allocs assume fp32.
    X_flat_raw = panel[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_flat_raw = panel[label_col].to_numpy(dtype=np.float32, copy=True)
    nan_label_flat = ~np.isfinite(y_flat_raw)

    # Audit fix #7 (2026-04-26): pre-fix, ±inf in features were replaced
    # with 0, conflating "extreme outlier" with "missing data". Now: clip
    # ±inf to ±5σ (per-column std). Acts as a soft outlier guard while
    # preserving sign information. NaN still → 0 (genuine missing).
    n_inf = int(np.isinf(X_flat_raw).sum())
    if n_inf > 0:
        col_std = np.nanstd(X_flat_raw, axis=0)
        col_std = np.where(col_std > 0, col_std, 1.0)
        clip_hi =  5.0 * col_std
        clip_lo = -5.0 * col_std
        X_flat_raw = np.clip(X_flat_raw, clip_lo, clip_hi)
    X_flat = np.nan_to_num(X_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_flat = np.nan_to_num(y_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Audit fix #9 (2026-04-26): defensive log when feature NaN → 0
    # substitution affects > 5% of cells. Upstream FactorZScoreTask
    # should have median-imputed; if we're seeing high NaN here, panel
    # pipeline didn't run cleanly.
    n_nan_x = int(np.isnan(panel[feature_cols].to_numpy(dtype=np.float32)).sum())
    n_total = X_flat_raw.size
    if n_total > 0 and n_nan_x / n_total > 0.05:
        log.warning(
            "_build_date_groups: %.1f%% feature NaN → 0 substitution "
            "(%d / %d cells); panel pipeline imputation may have failed",
            100.0 * n_nan_x / n_total, n_nan_x, n_total,
        )

    # Audit fix #95 (2026-04-26 round-3): warn when a row has ALL features
    # zero post-imputation. Such rows collapse to feature_encoder bias
    # alone — model can't differentiate them from each other → score
    # collapses to a constant per-imputed-class. Rare in practice but
    # catastrophic when it happens (calibrator saturation, ranking ties).
    n_zero_rows = int((np.abs(X_flat) < 1e-12).all(axis=1).sum()) if X_flat.size else 0
    if n_zero_rows > 0:
        log.warning(
            "_build_date_groups: %d row(s) have ALL features = 0 "
            "(post-imputation). These will receive identical scores from "
            "the feature encoder bias and are effectively ungrouped.",
            n_zero_rows,
        )

    n_groups = len(group_sizes)
    n_feat   = X_flat.shape[1]
    x = np.zeros((n_groups, max_tickers, n_feat), dtype=np.float32)
    y = np.zeros((n_groups, max_tickers),         dtype=np.float32)
    pad = np.ones((n_groups, max_tickers),       dtype=bool)
    nan_y = np.zeros((n_groups, max_tickers),    dtype=bool)
    offset = 0
    for gi, gs in enumerate(group_sizes):
        gs_int = int(gs)
        x[gi, :gs_int, :]     = X_flat[offset:offset + gs_int, :]
        y[gi, :gs_int]        = y_flat[offset:offset + gs_int]
        pad[gi, :gs_int]      = False
        nan_y[gi, :gs_int]    = nan_label_flat[offset:offset + gs_int]
        offset += gs_int
    return x, y, pad, nan_y


# ── Determinism + device helpers ──────────────────────────────────────────────

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.mps.manual_seed(seed)   # torch 2.0+ on macOS
    except AttributeError:
        pass


def _resolve_device(requested: str) -> torch.device:
    """Resolve device preference with graceful fallback: mps → cuda → cpu.

    Audit fix #83 (2026-04-26): emit warning when fallback to CPU
    happens — previously silent fallback could surprise CI / Linux runs
    expecting MPS performance and getting orders-of-magnitude slower CPU.
    """
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        log.warning("device='mps' requested but MPS not available — falling back")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        log.warning("device='cuda' requested but CUDA not available — falling back")
    if requested != "cpu":
        log.info("transformer device fallback: %s → cpu", requested)
    return torch.device("cpu")


# ── Main model class ──────────────────────────────────────────────────────────

class PanelTransformerModel:
    """Date-grouped cross-sectional transformer panel ranker.

    Public surface matches :class:`PanelLTRModel` so callers (training /
    scoring / tests) can dispatch on `panel_ltr.backend`.
    """

    def __init__(self, params: dict | None = None):
        merged = asdict(TransformerParams())
        if params:
            merged.update(params)
        self.params: TransformerParams = TransformerParams(**merged)
        self.feature_cols: list[str] = []
        self._model: _PanelTransformer | None = None
        self._device: torch.device = _resolve_device(self.params.device)
        # Last-epoch training/eval stats (populated in train()).
        self.history: list[dict] = []
        self.best_iter: int | None = None

    # ── Training ──────────────────────────────────────────────────────────────

    def train(
        self,
        panel: pd.DataFrame,
        group_sizes: np.ndarray | None = None,
        feature_cols: list[str] | None = None,
        label_col: str = "label",
        weight_col: str | None = "weight",     # unused by transformer (ListNet scale-invariant)
        num_boost_round: int | None = None,    # alias for max_epochs if provided
        early_stopping_rounds: int | None = None,  # alias for patience if provided
        eval_panel: pd.DataFrame | None = None,
        eval_group_sizes: np.ndarray | None = None,
    ) -> dict:
        """Fit the transformer; return train/eval metadata dict.

        Audit fix #28 (2026-04-26 round-3): symmetric API with predict().
        Both can now derive group_sizes from a `date` column on the
        panel — eliminates the predict-vs-train signature mismatch.
        Pre-fix, train required explicit group_sizes; predict accepted
        either. Now both behave identically.
        """
        # Audit fix #15 (2026-04-26 round-3): weight_col is intentionally
        # ignored. ListNet's softmax over labels is scale-invariant;
        # multiplying labels by a per-row weight just rescales the
        # softmax distribution. To respect weights properly we'd need
        # to weight the LOSS per-group (e.g. via group concurrency) —
        # not implemented because group concurrency is constant in our
        # date-grouped panel. Documented to avoid surprise.
        del weight_col   # ListNet is scale-invariant; group weights not applied here.

        if feature_cols is None:
            raise ValueError("PanelTransformerModel.train: feature_cols required")
        # Audit fix #28: derive group_sizes from `date` column when not given.
        if group_sizes is None:
            if "date" not in panel.columns:
                raise ValueError(
                    "PanelTransformerModel.train requires either explicit "
                    "`group_sizes` OR a `date` column on the panel."
                )
            panel = panel.sort_values("date", kind="mergesort").reset_index(drop=True)
            group_sizes = panel.groupby("date", sort=False).size().to_numpy()
        self.feature_cols = list(feature_cols)

        # Audit fix #87 (2026-04-26): don't mutate self.params from inside
        # train(). Pre-fix, calling train() with num_boost_round=N
        # permanently overwrote self.params.max_epochs — a second call
        # without num_boost_round would inherit N from the prior call
        # (silent stateful bug). New: use local effective_* variables.
        p = self.params
        effective_max_epochs = (int(num_boost_round) if num_boost_round is not None
                                else int(p.max_epochs))
        effective_patience   = (int(early_stopping_rounds) if early_stopping_rounds is not None
                                else int(p.patience))

        # Reproducibility
        _seed_everything(p.seed)
        if p.deterministic:
            # Audit fix #16 (2026-04-26 round-3): documented best-effort.
            # MPS doesn't expose all ops deterministically (per PyTorch
            # 2.0+ docs §torch.backends.mps), so two consecutive runs on
            # MPS may differ by ~0.005 OOS IC even with identical seed.
            # Use `warn_only=True` so non-deterministic ops produce a
            # warning rather than crash. CUDA/CPU paths are deterministic.
            os.environ.setdefault("PYTHONHASHSEED", str(p.seed))
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

        # Single-thread CPU guard: PyTorch + OpenMP deadlock in processes
        # that previously used `fork`-based multiprocessing (e.g. the panel
        # pipeline's parallel TickerPanelFeatureJob workers). Forcing
        # set_num_threads(1) here keeps training on the main thread and
        # avoids the fork/OMP interaction entirely.
        #
        # Audit fix #21 (2026-04-26): gate on CPU device only. On MPS the
        # tensor compute dispatches to Apple's GPU backend and CPU thread
        # count is irrelevant. On CPU fallback this previously crippled
        # PyTorch parallelism unnecessarily. Now: only single-thread when
        # we're actually running on CPU.
        if self._device.type == "cpu":
            try:
                torch.set_num_threads(1)
            except Exception:
                pass

        # Audit fix T-18 (2026-04-25): if caller didn't supply eval_panel
        # but auto_eval_split is on, auto-split the last `auto_eval_fraction`
        # date-groups into eval. This makes early stopping work without
        # requiring every caller to plumb explicit eval data — and CV +
        # FinalFit both benefit. Pre-fix, FinalFit always trained for full
        # max_epochs with no early stop, contributing to the v3 overfit.
        # Audit fix #11 (2026-04-26): assert that the panel is sorted by
        # date if a `date` column exists. auto_eval_split takes the
        # LAST n_eval groups via row offsets, so unsorted dates would
        # leak future data into train.
        if eval_panel is None and p.auto_eval_split and "date" in panel.columns:
            try:
                _date_col = panel["date"].to_numpy()
                # contiguous ↔ non-decreasing
                if len(_date_col) > 1 and not (_date_col[:-1] <= _date_col[1:]).all():
                    raise ValueError(
                        "auto_eval_split requires panel sorted by date "
                        "(future would leak into train)."
                    )
            except (KeyError, TypeError):
                pass

        if (
            eval_panel is None
            and p.auto_eval_split
            and p.auto_eval_fraction
            and 0.0 < p.auto_eval_fraction < 1.0
            and len(group_sizes) >= 5
        ):
            n_groups_total = len(group_sizes)
            n_eval = max(1, int(round(n_groups_total * p.auto_eval_fraction)))
            n_train = n_groups_total - n_eval
            if n_train < 5 or n_eval < 2:
                # Audit fix #12 (2026-04-26): log when the auto-split
                # decides to skip due to small panel — pre-fix this was
                # silent and FinalFit ran without early stop.
                log.warning(
                    "auto_eval_split: skipped (n_train=%d need ≥5, n_eval=%d need ≥2). "
                    "Training will run full max_epochs without early-stop.",
                    n_train, n_eval,
                )
            if n_train >= 5 and n_eval >= 2:
                # Slice panel by row offsets corresponding to the
                # date-group split. group_sizes is contiguous per date.
                row_split = int(np.array(group_sizes[:n_train]).sum())
                eval_panel = panel.iloc[row_split:].copy()
                eval_group_sizes = np.array(group_sizes[n_train:], dtype=np.int64)
                panel = panel.iloc[:row_split].copy()
                group_sizes = np.array(group_sizes[:n_train], dtype=np.int64)
                # Audit fix #91 (2026-04-26): module-level logger.
                log.info(
                    "auto_eval_split: train=%d groups (%d rows) | eval=%d groups (%d rows)",
                    n_train, row_split, n_eval, len(eval_panel),
                )

        # Audit fix #84 (2026-04-26 round-3): auto-bump max_tickers
        # when training data has more rows-per-date than the configured
        # cap. Pre-fix, training would raise from inside _build_date_groups
        # — operator had to manually edit config + retrain. Now: bump
        # silently and log so the next training-set growth doesn't hit
        # the cap. Operator can still set max_tickers manually for
        # consistent inference (the sidecar JSON saves the actual value).
        #
        # Audit fix #26 (2026-04-26 round-3): this also MITIGATES the
        # train≠inference structure mismatch. Pre-fix, max_tickers was
        # hardcoded; if the watchlist grew past it, inference would
        # chunk-split (cross-chunk attention severed) while training
        # had only ever seen single-chunk groups. With auto-bump,
        # max_tickers grows to fit training data → inference also fits
        # in a single chunk (no chunk-split fired) → train and inference
        # use IDENTICAL padding structure. The on_oversized_group="error"
        # config (added in #27) further prevents accidental chunk-split
        # if the operator bypasses auto-bump (e.g., loads model with
        # smaller max_tickers).
        if len(group_sizes):
            max_gs_train = int(np.max(group_sizes))
            if max_gs_train > p.max_tickers:
                log.info(
                    "auto-bump max_tickers: %d → %d (train data has %d rows in "
                    "largest date-group)",
                    p.max_tickers, max_gs_train, max_gs_train,
                )
                p.max_tickers = max_gs_train

        # Build batches
        xtr, ytr, padtr, nantr = _build_date_groups(
            panel, group_sizes, feature_cols, label_col, p.max_tickers,
        )
        xte = yte = padte = nante = None
        if eval_panel is not None and eval_group_sizes is not None:
            xte, yte, padte, nante = _build_date_groups(
                eval_panel, eval_group_sizes, feature_cols, label_col, p.max_tickers,
            )

        self._model = _PanelTransformer(n_features=len(feature_cols), p=p).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(),
                                lr=p.lr, weight_decay=p.weight_decay)

        # Audit fix #31+#32 (2026-04-26 batch-3): cosine LR schedule with
        # linear warmup. Per Vaswani 2017 §5.3 + Goyal 2017 (large
        # minibatch SGD), transformers benefit from warmup on the first
        # ~10% of steps, followed by cosine decay. Constant lr can spike
        # gradient norms early on and stall progress later.
        # Audit fix #35 (2026-04-26 round-3): account for grad accumulation
        # when computing optimizer steps (one optimizer step per N batches).
        accum_n = max(1, int(p.grad_accumulation_steps))
        batches_per_epoch = max(1, (xtr.shape[0] + p.batch_size - 1) // p.batch_size)
        steps_per_epoch   = max(1, batches_per_epoch // accum_n)
        total_steps       = steps_per_epoch * effective_max_epochs
        warmup_steps      = int(round(total_steps * float(p.warmup_fraction)))
        min_lr_factor   = float(p.cosine_min_lr_ratio)

        def _lr_factor(step: int) -> float:
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps + 1)
            if not p.cosine_decay:
                return 1.0
            denom = max(1, total_steps - warmup_steps)
            progress = float(step - warmup_steps) / float(denom)
            progress = max(0.0, min(1.0, progress))
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cos

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_factor)

        best_eval = float("-inf")
        best_state: dict | None = None
        bad_epochs = 0
        # Audit fix #19 (2026-04-26 round-3): single Generator instance
        # advanced across epochs. Each `randperm(..., generator=gen)`
        # consumes randomness, so consecutive epochs see DIFFERENT
        # shuffles — but the sequence is fully reproducible from p.seed.
        # Intentional design (vs reseeding per epoch which would give
        # the same shuffle every epoch).
        gen = torch.Generator(device="cpu").manual_seed(p.seed)

        # Audit fix #71+#72 (2026-04-26 round-3): opt-in pre-load whole
        # panel to device. Eliminates per-batch host→device transfer.
        # With ~17 MB panel, fits on MPS easily. Default off for safety
        # (large panels could OOM on small GPUs).
        xtr_dev = ytr_dev = padtr_dev = nantr_dev = None
        if p.preload_to_device:
            xtr_dev   = torch.from_numpy(xtr).to(self._device)
            ytr_dev   = torch.from_numpy(ytr).to(self._device)
            padtr_dev = torch.from_numpy(padtr).to(self._device)
            nantr_dev = torch.from_numpy(nantr).to(self._device)

        n_groups = xtr.shape[0]
        for epoch in range(effective_max_epochs):
            self._model.train()
            order = torch.randperm(n_groups, generator=gen).numpy()
            epoch_loss = 0.0
            nan_skipped = 0
            for start in range(0, n_groups, p.batch_size):
                idx = order[start:start + p.batch_size]
                if p.preload_to_device:
                    idx_t = torch.from_numpy(idx).to(self._device)
                    xb = xtr_dev.index_select(0, idx_t)
                    yb = ytr_dev.index_select(0, idx_t)
                    mb = padtr_dev.index_select(0, idx_t)
                    nb = nantr_dev.index_select(0, idx_t)
                else:
                    xb = torch.from_numpy(xtr[idx]).to(self._device)
                    yb = torch.from_numpy(ytr[idx]).to(self._device)
                    mb = torch.from_numpy(padtr[idx]).to(self._device)
                    nb = torch.from_numpy(nantr[idx]).to(self._device)
                invalid_b = mb | nb

                # Label smoothing (Gaussian noise only on non-pad+non-nan positions)
                if p.label_smoothing > 0:
                    noise = torch.randn_like(yb) * p.label_smoothing
                    yb = yb + noise.masked_fill(invalid_b, 0.0)

                # Ticker-conditional dropout: zero out whole ticker rows occasionally
                if p.ticker_dropout > 0 and self._model.training:
                    drop = (torch.rand(xb.shape[:2], device=self._device) < p.ticker_dropout) & (~mb)
                    xb = xb.masked_fill(drop.unsqueeze(-1), 0.0)

                # Audit fix #36 (2026-04-26 round-3): opt-in mixed precision.
                # autocast wraps the forward+loss in fp16/bf16 where
                # supported. AdamW master weights stay fp32. Default off
                # for backward compat.
                if p.use_amp and self._device.type in ("cuda", "mps"):
                    amp_device = self._device.type
                    with torch.autocast(device_type=amp_device, dtype=torch.float16):
                        scores = self._model(xb, mb)
                        loss = _listnet_loss(scores, yb, mb, nb,
                                             rank_transform=bool(p.rank_transform_labels))
                else:
                    scores = self._model(xb, mb)
                    loss = _listnet_loss(scores, yb, mb, nb,
                                         rank_transform=bool(p.rank_transform_labels))

                # Audit fix #88 (2026-04-26): NaN-grad detection. If
                # forward produced NaN/inf loss (rare but possible from
                # softmax overflow + extreme features), skip this batch
                # ENTIRELY rather than corrupt AdamW's exp_avg / exp_avg_sq
                # state with NaN. Pre-fix, one bad batch poisoned the
                # optimiser for the rest of training.
                if not torch.isfinite(loss):
                    nan_skipped += 1
                    continue

                # Audit fix #35 (2026-04-26 round-3): gradient accumulation.
                # When accumulation_steps > 1, scale loss by 1/N so the
                # accumulated gradient has the same magnitude as a single
                # large-batch step. Step optimizer only every Nth batch.
                accum_n = max(1, int(p.grad_accumulation_steps))
                accum_step = (start // p.batch_size) % accum_n
                if accum_step == 0:
                    opt.zero_grad(set_to_none=True)
                (loss / accum_n).backward()
                if accum_step == accum_n - 1:
                    # Audit fix T-16 (2026-04-25): clip gradient norm before
                    # the optimiser step.
                    if p.grad_clip_norm and p.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self._model.parameters(), max_norm=float(p.grad_clip_norm),
                        )
                    opt.step()
                    # Audit fix #31+#32: step the LR scheduler each
                    # OPTIMIZER step (not each batch — scheduler total_steps
                    # was computed for optimizer steps).
                    scheduler.step()
                epoch_loss += float(loss.item()) * len(idx)
            epoch_loss /= max(n_groups, 1)
            if nan_skipped > 0:
                log.warning("epoch=%d nan_loss_batches=%d (skipped)",
                            epoch, nan_skipped)

            train_ic = self._ic_on_tensors(xtr, ytr, padtr, panel, label_col, group_sizes)

            if xte is not None:
                # nante carries NaN-label mask for parity but is unused by IC
                # computation (spearman just skips degenerate groups).
                _ = nante
                eval_ic = self._ic_on_tensors(
                    xte, yte, padte, eval_panel, label_col, eval_group_sizes,
                )
                # Audit fix #39 (2026-04-26): tighten min_delta from 1e-6
                # to early_stop_min_delta (default 1e-3) so noisy
                # float-rounding "improvements" don't reset the patience
                # counter.
                improved = eval_ic > best_eval + p.early_stop_min_delta
                if improved:
                    best_eval  = eval_ic
                    # Audit fix #38 (2026-04-26 round-3): force best_state
                    # to CPU storage. ~1.3 MB per clone for our 320k-param
                    # model. Storing on GPU would consume MPS memory
                    # for the lifetime of training; CPU residency is
                    # cheap and still fast to restore at end.
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in self._model.state_dict().items()}
                    bad_epochs = 0
                    self.best_iter = epoch
                else:
                    bad_epochs += 1
                self.history.append({
                    "epoch": epoch, "loss": epoch_loss,
                    "train_ic": train_ic, "eval_ic": eval_ic,
                })
                # Audit fix #40 (2026-04-26): floor early stop at min_epochs
                # so a flat loss curve from epoch 0 doesn't kill training
                # at epoch 6 with no useful weights.
                if (bad_epochs >= effective_patience
                        and epoch >= int(p.min_epochs)):
                    log.info("early stop at epoch=%d (patience=%d, best_eval=%.4f)",
                             epoch, effective_patience, best_eval)
                    break
            else:
                # Audit fix #42 (2026-04-26 round-3): no eval data → no
                # early stop available. Training runs full max_epochs.
                # Documented limitation: a training-loss-based early stop
                # is fragile (overfit risk); operator should always
                # provide eval_panel OR enable auto_eval_split for early
                # stopping to engage.
                self.history.append({
                    "epoch": epoch, "loss": epoch_loss, "train_ic": train_ic,
                })

        if best_state is not None:
            self._model.load_state_dict(best_state)

        result: dict[str, Any] = {
            "best_iter": self.best_iter,
            "epochs_run": len(self.history),
            "train_ic": float(self.history[-1].get("train_ic", float("nan"))),
        }
        if xte is not None:
            result["eval_ic"] = float(best_eval)
        return result

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(
        self, panel: pd.DataFrame,
        group_sizes: np.ndarray | None = None,
    ) -> pd.Series:
        """Score each panel row. Caller specifies date-groups via either:

          * a `date` column on ``panel`` (groups = same-date rows, in order), OR
          * the explicit ``group_sizes`` kwarg (per-date row counts aligned
            with panel's row order, panel must be pre-sorted by date).

        A group larger than ``max_tickers`` is split into ``ceil(n/max_tickers)``
        chunks processed independently — cross-ticker attention only sees
        tickers within the same chunk, but this is strictly better than the
        prior behavior of silently truncating everything past ``max_tickers``
        to uninitialized memory.

        Raises ValueError if neither a ``date`` column nor explicit
        ``group_sizes`` is provided (no silent "whole panel = one group"
        fallback — that was a bug).
        """
        if self._model is None:
            raise RuntimeError("PanelTransformerModel.predict called before train/load")

        # Audit fix #X5/#5 (2026-04-26): validate that the panel has
        # all the feature columns we trained on. Pre-fix, missing
        # columns would raise a cryptic KeyError deep inside _build_date_groups.
        missing = [c for c in self.feature_cols if c not in panel.columns]
        if missing:
            raise ValueError(
                f"PanelTransformerModel.predict: panel missing required "
                f"feature columns: {missing[:5]}{'…' if len(missing) > 5 else ''} "
                f"(model trained on {len(self.feature_cols)} features)."
            )

        # Audit T-23 (2026-04-25): groupby("date") must see contiguous
        # rows per date, otherwise group_sizes mismatches the actual
        # date partitioning. If the caller didn't supply explicit
        # group_sizes, sort by date here and remember the original
        # order so we can re-align the output.
        original_index = panel.index
        if group_sizes is None:
            if "date" not in panel.columns:
                raise ValueError(
                    "PanelTransformerModel.predict requires either a `date` "
                    "column on the panel or an explicit `group_sizes` array."
                )
            panel = panel.sort_values("date", kind="mergesort")
            group_sizes = panel.groupby("date", sort=False).size().to_numpy()
        group_sizes = np.asarray(group_sizes, dtype=np.int64)
        if int(group_sizes.sum()) != len(panel):
            raise ValueError(
                f"group_sizes.sum()={int(group_sizes.sum())} != len(panel)={len(panel)}"
            )

        # Audit T-1/T-2/T-19 (2026-04-25): chunk-splitting at inference
        # introduced a train≠inference structure mismatch (training never
        # saw 33-ticker chunks). With max_tickers raised to 128 (≥99
        # watchlist) chunk-splitting normally never fires, but we keep
        # the safety net for future watchlist growth. When it DOES fire,
        # we now warn loudly so the operator can raise max_tickers.
        max_t = int(self.params.max_tickers)
        expanded: list[int] = []
        chunk_split_fired = False
        for gs in group_sizes.tolist():
            if gs <= max_t:
                expanded.append(gs)
            else:
                chunk_split_fired = True
                n_chunks = (gs + max_t - 1) // max_t
                base = gs // n_chunks
                rem  = gs - base * n_chunks
                for i in range(n_chunks):
                    expanded.append(base + (1 if i < rem else 0))
        if chunk_split_fired:
            # Audit fix #27 (2026-04-26 batch-3): config-driven response.
            mode = str(getattr(self.params, "on_oversized_group", "warn")).lower()
            if mode == "error":
                raise ValueError(
                    f"PanelTransformerModel.predict: a date-group exceeds "
                    f"max_tickers={max_t}. on_oversized_group='error' → "
                    f"refusing to chunk-split (cross-chunk attention lost). "
                    f"Raise TransformerParams.max_tickers and retrain."
                )
            elif mode == "silent":
                pass
            else:  # default "warn"
                log.warning(
                    "PanelTransformerModel.predict: a date-group exceeds "
                    "max_tickers=%d → chunk-split fallback. Cross-chunk scores "
                    "lose comparability. Raise max_tickers and retrain.",
                    max_t,
                )
        group_sizes_exp = np.array(expanded, dtype=np.int64)
        # Audit fix #29 (2026-04-26 round-3): if panel already has a
        # `label` column, assign(label=0.0) silently overwrites it.
        # That doesn't affect predict (we don't use the label), but
        # it's surprising state mutation if caller passes by reference.
        # Use copy + drop to avoid the warning.
        if "label" in panel.columns:
            log.debug("predict: panel already has 'label' col — overwriting with 0 for inference (predict doesn't use label)")
        x, _, pad, _ = _build_date_groups(
            panel.assign(label=0.0), group_sizes_exp, self.feature_cols, "label",
            max_t,
        )
        self._model.eval()
        preds_flat = np.full(len(panel), np.nan, dtype=np.float32)
        offset = 0
        bs = self.params.batch_size
        # Audit fix #90 (2026-04-26 round-3): inference_mode is faster
        # than no_grad on PyTorch 1.9+ — disables view tracking entirely.
        with torch.inference_mode():
            for start in range(0, x.shape[0], bs):
                xb = torch.from_numpy(x[start:start + bs]).to(self._device)
                mb = torch.from_numpy(pad[start:start + bs]).to(self._device)
                sb = self._model(xb, mb).detach().cpu().numpy()
                for gi in range(sb.shape[0]):
                    take = int((~pad[start + gi]).sum())
                    preds_flat[offset:offset + take] = sb[gi, :take]
                    offset += take
        # Re-align preds to caller's original index order.
        preds = pd.Series(preds_flat, index=panel.index, name="panel_score")
        return preds.reindex(original_index)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if self._model is None:
            raise RuntimeError("PanelTransformerModel.save called before train")
        # Audit fix #94 (2026-04-26 round-3): validate that feature_cols is
        # populated. Pre-fix, save() with empty feature_cols would produce
        # an unloadable artifact (load() requires feature_cols to rebuild
        # the model). Now: loud error.
        if not self.feature_cols:
            raise RuntimeError(
                "PanelTransformerModel.save: feature_cols is empty. "
                "Did train() succeed? Cannot save without feature schema."
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        sidecar = path.with_suffix(".json")
        torch.save(self._model.state_dict(), path)
        # Audit fix #52 (2026-04-26 batch-3): omit "device" from saved
        # params. Pre-fix, device='mps' baked into the artifact would
        # surprise CI / Linux runs that load the model — _resolve_device
        # falls back gracefully but the device tag was misleading.
        # Now: save params WITHOUT device; load() re-resolves on the
        # target machine.
        params_to_save = asdict(self.params)
        params_to_save.pop("device", None)
        payload = {
            # Audit fix T-NEW-1 (2026-04-26 round-3): bump version to 2.
            # Batch 1 added LayerNorm + 2-layer score head + Xavier init
            # → state_dict keys differ from version-1 artifacts. v2
            # artifacts have keys: feature_encoder, input_norm,
            # encoder, score_head.0, score_head.3.
            "version":      2,
            "kind":         "panel_transformer",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params":       params_to_save,
            "best_iter":    self.best_iter,
            # Audit fix #53 (2026-04-26): save full history when flag set
            # (default). Pre-fix, last-50 truncation lost early-epoch
            # diagnostic data on long runs.
            "history":      (self.history if self.params.save_full_history
                             else self.history[-50:]),
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        # Audit fix #54 (2026-04-26 round-3): explicit JSON encoder rather
        # than relying on `default=str` which silently coerces non-JSON
        # types (datetime, numpy ints, etc.) to their __str__ form. We
        # explicitly handle the known cases and raise on unknown types
        # so artifact corruption is loud, not silent.
        def _json_encoder(o: Any) -> Any:
            if isinstance(o, (date,)):
                return o.isoformat()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            raise TypeError(f"Cannot JSON-encode object of type {type(o).__name__}")
        sidecar.write_text(json.dumps(payload, default=_json_encoder))

    @classmethod
    def load(cls, path: str | Path) -> "PanelTransformerModel":
        path = Path(path)
        # Audit fix #55 (2026-04-26 round-3): clear error messages when
        # either .pt or .json is missing. Pre-fix, you'd get a cryptic
        # FileNotFoundError or PyTorch error far from the load() call.
        if path.suffix == ".json":
            pt_path = path.with_suffix(".pt")
            json_path = path
        else:
            pt_path = path if path.suffix == ".pt" else path.with_suffix(".pt")
            json_path = pt_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(
                f"PanelTransformerModel.load: sidecar JSON not found: {json_path} "
                f"(was looking next to {pt_path})"
            )
        if not pt_path.exists():
            raise FileNotFoundError(
                f"PanelTransformerModel.load: weights file not found: {pt_path} "
                f"(was looking next to {json_path})"
            )
        meta = json.loads(json_path.read_text())
        if meta.get("kind") != "panel_transformer":
            raise ValueError(f"Not a panel_transformer artifact: {json_path}")
        # Audit fix T-NEW-1 + #93 (2026-04-26 round-3): version migration
        # path. v1 artifacts (saved before 2026-04-26 batch-1) have
        # different state_dict keys → strict load_state_dict will fail
        # with a clear PyTorch error. We let it fail with helpful context.
        artifact_version = int(meta.get("version", 1))
        if artifact_version < 2:
            log.warning(
                "Loading panel_transformer v%d artifact with v2+ code. "
                "Architecture changed in 2026-04-26 batch-1 (added LayerNorm + "
                "2-layer score head + Xavier init). State-dict load will fail "
                "if old keys present. Re-train and re-save to upgrade.",
                artifact_version,
            )
        m = cls(params=meta["params"])
        m.feature_cols = list(meta["feature_cols"])
        m.best_iter = meta.get("best_iter")
        m._model = _PanelTransformer(
            n_features=len(m.feature_cols), p=m.params,
        ).to(m._device)
        # Round-3 audit (#R3-14): explicitly set weights_only=True. PyTorch
        # 2.6 made this the default and PyTorch 2.7+ may make weights_only=False
        # raise — pinning here is forward-compatible AND prevents arbitrary
        # code execution from a tampered .pt file. State dicts are pure
        # tensor data so weights_only=True is correct here.
        try:
            state = torch.load(pt_path, map_location=m._device, weights_only=True)
        except TypeError:
            # Older torch (<2.0) without weights_only — fall back.
            state = torch.load(pt_path, map_location=m._device)
        m._model.load_state_dict(state)
        m._model.eval()
        return m

    # ── Internal IC helper ────────────────────────────────────────────────────

    def _ic_on_tensors(
        self,
        x: np.ndarray, y: np.ndarray, pad: np.ndarray,
        panel: pd.DataFrame, label_col: str, group_sizes: np.ndarray,
    ) -> float:
        """Per-date Spearman IC averaged over groups, batched through model."""
        del label_col   # labels already baked into y
        self._model.eval()
        preds_flat = np.empty(len(panel), dtype=np.float32)
        offset = 0
        bs = self.params.batch_size
        # Audit fix #90 (2026-04-26 round-3): inference_mode is faster
        # than no_grad on PyTorch 1.9+ — disables view tracking entirely.
        with torch.inference_mode():
            for start in range(0, x.shape[0], bs):
                xb = torch.from_numpy(x[start:start + bs]).to(self._device)
                mb = torch.from_numpy(pad[start:start + bs]).to(self._device)
                sb = self._model(xb, mb).detach().cpu().numpy()
                for gi in range(sb.shape[0]):
                    take = int((~pad[start + gi]).sum())
                    preds_flat[offset:offset + take] = sb[gi, :take]
                    offset += take
        ics: list[float] = []
        offset2 = 0
        # Audit fix #66 (2026-04-26 batch-3): if "label" column missing,
        # fall back to in-memory y AND log it. Previously silent fallback
        # made it hard to diagnose why IC was based on stale labels.
        y_flat = panel["label"].to_numpy() if "label" in panel.columns else None
        if y_flat is None:
            log.debug("_ic_on_tensors: panel has no 'label' col, using in-memory y tensor")

        # Audit fix #69 (2026-04-26 round-3): explicit log when ALL
        # groups are degenerate (NaN IC returned). Pre-fix, NaN
        # silently propagated up to history → user sees "NaN train_ic"
        # without knowing why.
        for gs in group_sizes:
            gs = int(gs)
            p_slice = preds_flat[offset2:offset2 + gs]
            y_slice = (y_flat[offset2:offset2 + gs] if y_flat is not None
                       else y.reshape(-1)[offset2:offset2 + gs])
            offset2 += gs
            # Audit fix #67 (2026-04-26 batch-3): use np.allclose instead
            # of float == comparison. Pre-fix, p_slice == p_slice[0] for
            # near-equal floats (e.g. 1e-9 difference) returned False
            # → spearmanr called on a degenerate slice → NaN result.
            #
            # Audit fix #68 (2026-04-26 round-3): also detect NaN in
            # predictions (model can output NaN if upstream features
            # are pathological). Pre-fix, NaN propagated silently into
            # spearman; now log + skip group.
            if np.any(~np.isfinite(p_slice)):
                log.debug("_ic_on_tensors: skipping group with non-finite predictions")
                continue
            if (gs < 2
                    or np.allclose(p_slice, p_slice[0], rtol=0, atol=1e-12)
                    or np.allclose(y_slice, y_slice[0], rtol=0, atol=1e-12)):
                continue
            rho, _ = spearmanr(p_slice, y_slice)
            if not np.isnan(rho):
                ics.append(float(rho))
            # Audit fix #82 (2026-04-26 round-3): optional kendall tau
            # for cross-check. Only log; not used for IC return value.
            if self.params.log_kendall_tau:
                try:
                    tau, _ = kendalltau(p_slice, y_slice)
                    if not np.isnan(tau):
                        log.debug("group kendall_tau=%+.4f spearman=%+.4f", tau, rho)
                except Exception:
                    pass
        if not ics:
            log.warning(
                "_ic_on_tensors: all %d groups were degenerate "
                "(< 2 valid tickers or all-equal scores/labels) → IC=NaN",
                len(group_sizes),
            )
            return float("nan")
        return float(np.mean(ics))


__all__ = [
    "TransformerParams",
    "PanelTransformerModel",
]
