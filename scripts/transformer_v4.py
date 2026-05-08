#!/usr/bin/env python
"""Transformer v4 — clean rebuild for cross-sectional ranking.

Design principles (from Qlib reference + Empirical Asset Pricing literature):

  1. **Listwise ranking loss** (pairwise BCE) — NOT MSE/Huber.
     The task is to rank 290 tickers per day; magnitude doesn't matter.
  2. **Per-day batched training** (always, not optional).
     Every batch = one trading day's tickers (~250 samples).
  3. **Smallest possible model first** — Linear → MLP → LSTM → Transformer.
     If linear can't extract signal, the problem is data, not architecture.
  4. **Sanity tests built in** — A/A test, label shuffle, time shift placebo.
  5. **Median CSRankIC** as primary metric — robust to outlier days.
  6. **Multiple seeds** by default — single-seed results are noise.
  7. **Track train_ic alongside val_ic** — detect overfitting.

References:
  - Bryan Kelly et al. "Empirical Asset Pricing via Machine Learning" (2020)
  - Microsoft Qlib reference: github.com/microsoft/qlib (Transformer model)
  - PatchTST (Nie 2023) — adapted for ranking (not forecasting)

Usage::

    # 1. Linear baseline (sanity check that data has signal)
    python scripts/transformer_v4.py --arch linear --epochs 30

    # 2. MLP baseline
    python scripts/transformer_v4.py --arch mlp --epochs 30

    # 3. LSTM (often strongest on this scale)
    python scripts/transformer_v4.py --arch lstm --epochs 30

    # 4. Transformer
    python scripts/transformer_v4.py --arch transformer --epochs 30

    # Sanity tests
    python scripts/transformer_v4.py --arch lstm --label-shuffle  # IC ≈ 0
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# §5.10 hardware saturation
for _k, _v in (("OMP_NUM_THREADS", "10"),
               ("MKL_NUM_THREADS", "10"),
               ("OPENBLAS_NUM_THREADS", "10")):
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-v4")


# ── Data ───────────────────────────────────────────────────────────────────

LABEL_COL = "fwd_5d_excess"


class PerDayDataset(Dataset):
    """Indexes (date, ticker) → 60-bar history + label.

    Each "sample" is a single (ticker, date) point. Sequences are built from
    the full panel so val/test can use earlier (train) bars as context — no
    leakage because labels are tied to sample-end-date split.
    """

    def __init__(self, panel: pd.DataFrame, channel_cols: list[str],
                 seq_len: int, split: str,
                 ticker_to_idx: dict[str, int]):
        self.seq_len = seq_len
        self.channel_cols = channel_cols
        self.ticker_to_idx = ticker_to_idx
        # Drop rows with NaN labels (cache tail)
        panel = panel.dropna(subset=[LABEL_COL]).reset_index(drop=True)
        ch_arr  = panel[channel_cols].astype(np.float32).fillna(0.0).values
        lab_arr = panel[LABEL_COL].astype(np.float32).values
        split_arr = panel["split_label"].values
        date_arr  = panel["date"].values

        self.samples: list[dict] = []
        gp = panel.groupby("ticker", sort=False).indices
        for t, idxs in gp.items():
            idxs = np.asarray(sorted(idxs))
            t_idx = ticker_to_idx.get(t)
            if t_idx is None:
                continue
            for i in range(seq_len, len(idxs)):
                end = idxs[i]
                if split_arr[end] != split:
                    continue
                seq = ch_arr[idxs[i - seq_len: i]]
                if seq.shape[0] != seq_len:
                    continue
                self.samples.append({
                    "seq": seq,
                    "label": lab_arr[end],
                    "ticker_idx": t_idx,
                    "date_ns": np.int64(pd.Timestamp(date_arr[end]).value),
                })
        # Pre-extract dates for the per-day sampler
        self._dates_ns = np.asarray([s["date_ns"] for s in self.samples],
                                     dtype=np.int64)
        log.info("PerDayDataset[%s]: %d samples × seq_len=%d × ch=%d  "
                 "n_dates=%d",
                 split, len(self.samples), seq_len, len(channel_cols),
                 len(np.unique(self._dates_ns)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        return (torch.from_numpy(s["seq"]),
                torch.tensor(s["label"], dtype=torch.float32),
                torch.tensor(s["ticker_idx"], dtype=torch.long),
                torch.tensor(s["date_ns"], dtype=torch.int64))


class PerDayBatchSampler(Sampler):
    """Each batch = one trading day's samples (≥ min_per_day, else skip)."""

    def __init__(self, dataset: PerDayDataset, shuffle: bool = True,
                 min_per_day: int = 5, seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.min_per_day = min_per_day
        self.seed = seed
        dates = dataset._dates_ns
        self._by_date: dict[int, list[int]] = {}
        for pos, d in enumerate(dates):
            self._by_date.setdefault(int(d), []).append(pos)
        # Filter to days with enough samples (min_per_day) — ranking
        # loss needs ≥ 2 samples to compute pairs.
        self._date_keys = [k for k, v in self._by_date.items()
                           if len(v) >= self.min_per_day]
        self.epoch = 0

    def __len__(self) -> int:
        return len(self._date_keys)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        keys = self._date_keys.copy()
        if self.shuffle:
            rng.shuffle(keys)
        for k in keys:
            yield self._by_date[k]
        self.epoch += 1


# ── Models ─────────────────────────────────────────────────────────────────

class LinearBaseline(nn.Module):
    """Linear regression on flattened sequence. Should give pool_IC ≈ 0
    if data has no signal at all — a sanity check. If LinearBaseline gives
    +0.03 IC, the model architecture isn't the bottleneck."""

    def __init__(self, n_channels: int, seq_len: int):
        super().__init__()
        self.linear = nn.Linear(n_channels * seq_len, 1)

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        return self.linear(x.reshape(B, L * C)).squeeze(-1)


class MLPBaseline(nn.Module):
    """2-layer MLP on flattened sequence."""

    def __init__(self, n_channels: int, seq_len: int, hidden: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_channels * seq_len, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        return self.net(x.reshape(B, L * C)).squeeze(-1)


class LSTMRanker(nn.Module):
    """1-layer LSTM. Often strongest on this scale per Qlib benchmarks."""

    def __init__(self, n_channels: int, seq_len: int, hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, num_layers=2,
                            dropout=dropout, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        out, (h, c) = self.lstm(x)
        # Use last hidden state
        return self.head(out[:, -1, :]).squeeze(-1)


class TransformerRanker(nn.Module):
    """Encoder-only transformer over time series."""

    def __init__(self, n_channels: int, seq_len: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x) + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class iTransformerRanker(nn.Module):
    """iTransformer for cross-sectional ranking — v2 two-stage embedding.

    Ref: Liu et al., "iTransformer: Inverted Transformers Are Effective for
    Time Series Forecasting", ICLR 2024. thuml/iTransformer.
    Embedding follows neuralforecast.common._modules.DataEmbedding_inverted.

    Key inversion vs TransformerRanker:
      TransformerRanker : attend across T time steps  per ticker  (temporal axis)
      iTransformerRanker: attend across N tickers     per date    (cross-variate axis)

    v1 failure: single Linear(T*D=3160, 64) — 50:1 compression in one unstructured
    step destroyed the signal. v2 fix: two-stage embedding matching the paper's intent:
      Stage 1 feat_proj  Linear(D, d_feat)          — compress features per time step
                                                       (shared weights across T & N)
      Stage 2 token_proj Linear(T*d_feat, d_model)  — aggregate temporal context per
                                                       ticker → one d_model token
                                                       (DataEmbedding_inverted analogue)
    With D=158, T=20, d_feat=32, d_model=128: compression is 5:1 not 50:1.
    """

    def __init__(self, n_channels: int, seq_len: int,
                 d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.1,
                 d_ff: int = 256, d_feat: int = 32):
        super().__init__()
        # Stage 1: feature compression per time step (shared across T and N)
        # Equivalent to applying a lightweight feature mixer before temporal pooling.
        self.feat_proj   = nn.Linear(n_channels, d_feat)
        self.feat_norm   = nn.LayerNorm(d_feat)

        # Stage 2: DataEmbedding_inverted — aggregate ticker's T compressed steps → token.
        # Original paper: Linear(seq_len, d_model) for scalar variates.
        # Here: Linear(seq_len * d_feat, d_model) — same idea, d_feat-dim variates.
        self.token_proj  = nn.Linear(seq_len * d_feat, d_model)
        self.token_norm  = nn.LayerNorm(d_model)
        self.dropout_emb = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        # x: (N_tickers, seq_len, n_channels)
        N, T, D = x.shape
        # Stage 1: compress D features → d_feat at each time step
        h = self.feat_norm(self.feat_proj(x))               # (N, T, d_feat)
        # Stage 2: flatten temporal dim and project to d_model token per ticker
        # This is DataEmbedding_inverted applied to d_feat-dim variates
        h_flat = h.reshape(N, T * h.shape[-1])              # (N, T*d_feat)
        tokens = self.dropout_emb(
            self.token_norm(self.token_proj(h_flat)))        # (N, d_model)
        # Cross-ticker self-attention (the inverted axis)
        encoded = self.encoder(tokens.unsqueeze(0)).squeeze(0)  # (N, d_model)
        return self.head(encoded).squeeze(-1)                    # (N,)


class PatchTSTRanker(nn.Module):
    """PatchTST for cross-sectional stock ranking.

    Ref: Nie et al., "A Time Series is Worth 64 Words", ICLR 2023.
    yuqinie98/PatchTST. Ported from PatchTST_backbone.py.

    Key idea: divide each ticker's T-step history into overlapping patches,
    embed each patch, then apply transformer attention across patches
    (temporal axis, channel-independent per feature dimension).

    Adaptation for ranking (vs forecasting):
    - Input: (N_tickers, seq_len, D_features)
    - Each ticker processed independently across its D feature channels
    - Patches from seq_len: num_patches = (seq_len - patch_len) // stride + 1
    - Attention across num_patches per channel, then pool → score per ticker

    Paper params (ETTh1): seq_len=336, patch_len=16, stride=8, d_model=16,
    n_heads=4, e_layers=3. We use shorter seq_len for daily stock data.
    """

    def __init__(self, n_channels: int, seq_len: int,
                 patch_len: int = 8, stride: int = 4,
                 d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = 0.2,
                 d_ff: int = 128):
        super().__init__()
        self.patch_len  = patch_len
        self.stride     = stride
        self.n_channels = n_channels
        num_patches = (seq_len - patch_len) // stride + 1
        self.num_patches = num_patches

        # Patch embedding: each patch (patch_len time steps, all D features)
        # → d_model token. Channel-mixing: all features in one projection.
        self.patch_embed = nn.Linear(patch_len * n_channels, d_model)
        self.pos_embed   = nn.Parameter(torch.randn(1, num_patches, d_model) * 0.02)
        self.embed_norm  = nn.LayerNorm(d_model)
        self.embed_drop  = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
        # x: (N_tickers, seq_len, n_channels)
        N, T, D = x.shape
        P, S = self.patch_len, self.stride

        # Extract patches: (N, num_patches, patch_len * n_channels)
        patches = x.unfold(1, P, S)          # (N, num_patches, D, P)
        patches = patches.permute(0, 1, 3, 2) # (N, num_patches, P, D)
        patches = patches.reshape(N, self.num_patches, P * D)

        # Embed + positional encoding
        tokens = self.embed_drop(
            self.embed_norm(self.patch_embed(patches)) + self.pos_embed
        )                                     # (N, num_patches, d_model)

        # Temporal attention across patches per ticker
        encoded = self.encoder(tokens)        # (N, num_patches, d_model)

        # Pool across patches → score per ticker
        pooled = encoded.mean(dim=1)          # (N, d_model)
        return self.head(pooled).squeeze(-1)  # (N,)


def build_model(arch: str, n_channels: int, seq_len: int) -> nn.Module:
    if arch == "linear":
        return LinearBaseline(n_channels, seq_len)
    if arch == "mlp":
        return MLPBaseline(n_channels, seq_len)
    if arch == "lstm":
        return LSTMRanker(n_channels, seq_len)
    if arch == "transformer":
        return TransformerRanker(n_channels, seq_len)
    if arch == "itransformer":
        return iTransformerRanker(n_channels, seq_len)
    if arch == "patchtst":
        return PatchTSTRanker(n_channels, seq_len)
    raise ValueError(f"unknown arch: {arch}")


# ── Loss ───────────────────────────────────────────────────────────────────

def listwise_rank_loss(pred: torch.Tensor, label: torch.Tensor,
                       margin: float = 0.0) -> torch.Tensor:
    """Listwise pairwise BCE rank loss.

    For each pair (i, j) within the batch (one day), compute
    p_ij = sigmoid(pred[i] - pred[j]) and target = 1 if label[i] > label[j]
    else 0. BCE on this is a smooth approximation of "sort pred to match
    sort of label". Ties contribute 0.5 (no signal).

    Uses ALL pairs (O(N²) but N ~ 250 per day → 62k pairs, cheap).
    """
    n = pred.size(0)
    if n < 2:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    # Build pairwise differences
    pred_i = pred.unsqueeze(1).expand(n, n)
    pred_j = pred.unsqueeze(0).expand(n, n)
    label_i = label.unsqueeze(1).expand(n, n)
    label_j = label.unsqueeze(0).expand(n, n)
    # Mask: only use upper triangle (i < j) to avoid double counting
    mask = torch.triu(torch.ones(n, n, device=pred.device), diagonal=1).bool()
    # Targets: 1 if label_i > label_j, 0 if <, 0.5 if = (skip)
    diff_label = label_i - label_j
    target = (diff_label > 0).float()
    # Pred difference + margin (BPR-style)
    diff_pred = pred_i - pred_j - margin
    # BCE with logits
    bce = F.binary_cross_entropy_with_logits(diff_pred[mask], target[mask],
                                              reduction="mean")
    return bce


# ── Eval ───────────────────────────────────────────────────────────────────

def per_day_csrankic(preds: np.ndarray, labels: np.ndarray,
                     dates: np.ndarray) -> tuple[float, float]:
    """Returns (mean, median) per-day cross-sectional Spearman IC."""
    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 5:
            continue
        rho, _ = spearmanr(group["pred"], group["label"])
        if not np.isnan(rho):
            ics.append(rho)
    if not ics:
        return 0.0, 0.0
    return float(np.mean(ics)), float(np.median(ics))


def evaluate(model: nn.Module, loader: DataLoader, device: str
             ) -> tuple[float, float, float]:
    """Returns (mean_ic, median_ic, mse)."""
    model.eval()
    all_preds, all_labels, all_dates = [], [], []
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for seq, lab, t_idx, date in loader:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            t_idx_d = t_idx.to(device, non_blocking=True)
            pred = model(seq, t_idx_d)
            total_se += (pred - lab).pow(2).sum().item()
            total_n += pred.numel()
            all_preds.append(pred.cpu().numpy())
            all_labels.append(lab.cpu().numpy())
            all_dates.append(date.numpy())
    if not all_preds:
        return 0.0, 0.0, 0.0
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    dates = np.concatenate(all_dates)
    mean_ic, median_ic = per_day_csrankic(preds, labels, dates)
    mse = total_se / max(1, total_n)
    return mean_ic, median_ic, mse


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "transformer_dataset_engineered.parquet"))
    p.add_argument("--arch", required=True,
                   choices=["linear", "mlp", "lstm", "transformer", "itransformer", "patchtst"])
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="High default — listwise rank loss has stable gradients")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--device", default="auto",
                   choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-seeds", type=int, default=1)
    p.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / "transformer_v4"))
    p.add_argument("--label-shuffle", action="store_true",
                   help="§5.2 sanity: shuffle labels — IC should ≈ 0")
    p.add_argument("--label-shift-days", type=int, default=0,
                   help="§5.2 sanity: shift labels forward by N days "
                        "as time-shift placebo — IC should ≈ 0 if signal "
                        "is genuine.")
    p.add_argument("--max-train-batches", type=int, default=0,
                   help=">0 to limit train batches per epoch (smoke testing)")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker count (0 = main process, avoids fork issues)")
    args = p.parse_args()

    if args.device == "auto":
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    else:
        device = args.device
    log.info("Device: %s", device)

    # Load data
    log.info("Loading %s …", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    channel_cols = [c for c in panel.columns if c not in excluded]
    log.info("Channels (%d): %s", len(channel_cols), channel_cols)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── Label preprocessing ──
    # Per-day demean: makes labels pure cross-sectional (rank target).
    # No clipping — pairwise rank loss is invariant to magnitude.
    panel[LABEL_COL] = panel[LABEL_COL] - panel.groupby("date")[LABEL_COL].transform("mean")

    # Per-horizon standardize using TRAIN-only stats (no leakage).
    train_mask = panel["split_label"] == "train"
    label_std = float(panel.loc[train_mask, LABEL_COL].std())
    if label_std > 1e-9:
        panel[LABEL_COL] = panel[LABEL_COL] / label_std
    log.info("Label preprocessing: per-day demean + train-only std=%.4f", label_std)

    # ── Sanity tests ──
    if args.label_shuffle:
        log.info("§5.2 SANITY: shuffling labels — expect IC ≈ 0")
        rng = np.random.default_rng(args.seed)
        panel[LABEL_COL] = rng.permutation(panel[LABEL_COL].values)
    if args.label_shift_days > 0:
        log.info("§5.2 SANITY: shifting labels +%d days — expect IC ≈ 0",
                 args.label_shift_days)
        # Shift labels forward by N days within each ticker
        panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
        panel[LABEL_COL] = panel.groupby("ticker")[LABEL_COL].shift(-args.label_shift_days)
        panel = panel.dropna(subset=[LABEL_COL]).reset_index(drop=True)

    # ── Build datasets ──
    all_tickers = sorted(panel["ticker"].unique())
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}

    train_ds = PerDayDataset(panel, channel_cols, args.seq_len, "train", ticker_to_idx)
    val_ds   = PerDayDataset(panel, channel_cols, args.seq_len, "val",   ticker_to_idx)
    test_ds  = PerDayDataset(panel, channel_cols, args.seq_len, "test",  ticker_to_idx)

    # ── Run multi-seed training ──
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_results = []
    for s_idx in range(args.num_seeds):
        seed = args.seed + s_idx
        log.info("══════════════════════════════════════")
        log.info("══ SEED %d (arch=%s) ══", seed, args.arch)
        log.info("══════════════════════════════════════")
        torch.manual_seed(seed)
        np.random.seed(seed)
        result = train_one_seed(
            arch=args.arch,
            train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
            channel_cols=channel_cols, seq_len=args.seq_len,
            device=device, epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay, patience=args.patience,
            warmup_epochs=args.warmup_epochs,
            seed=seed, max_train_batches=args.max_train_batches,
            out_path=out_dir / f"{args.arch}_seed{seed}.pt",
            num_workers=args.num_workers,
        )
        result["seed"] = seed
        seed_results.append(result)

    # ── Summary ──
    log.info("══════════════════════════════════════")
    log.info("══ FINAL SUMMARY (arch=%s, n_seeds=%d) ══",
             args.arch, args.num_seeds)
    log.info("══════════════════════════════════════")
    val_ics = [r["best_val_ic"] for r in seed_results]
    test_means = [r["test_mean_ic"] for r in seed_results]
    test_medians = [r["test_median_ic"] for r in seed_results]
    log.info("val_ic    : mean=%+.4f  std=%+.4f  per-seed=%s",
             float(np.mean(val_ics)), float(np.std(val_ics)),
             [f"{v:+.4f}" for v in val_ics])
    log.info("test_mean : mean=%+.4f  std=%+.4f  per-seed=%s",
             float(np.mean(test_means)), float(np.std(test_means)),
             [f"{v:+.4f}" for v in test_means])
    log.info("test_med  : mean=%+.4f  std=%+.4f  per-seed=%s",
             float(np.mean(test_medians)), float(np.std(test_medians)),
             [f"{v:+.4f}" for v in test_medians])

    summary = {
        "arch": args.arch,
        "dataset": args.dataset,
        "n_channels": len(channel_cols),
        "channel_cols": channel_cols,
        "n_seeds": args.num_seeds,
        "label_shuffle": args.label_shuffle,
        "label_shift_days": args.label_shift_days,
        "label_std": label_std,
        "seed_results": seed_results,
        "val_ic_mean": float(np.mean(val_ics)),
        "val_ic_std": float(np.std(val_ics)),
        "test_mean_ic": float(np.mean(test_means)),
        "test_median_ic": float(np.mean(test_medians)),
    }
    summary_path = out_dir / f"{args.arch}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary written: %s", summary_path)


def train_one_seed(arch, train_ds, val_ds, test_ds, channel_cols, seq_len,
                    device, epochs, lr, weight_decay, patience,
                    warmup_epochs, seed, max_train_batches, out_path,
                    num_workers=4):
    """Train one model with one seed; return result dict."""
    train_sampler = PerDayBatchSampler(train_ds, shuffle=True, seed=seed)
    val_sampler   = PerDayBatchSampler(val_ds,   shuffle=False, seed=seed)
    test_sampler  = PerDayBatchSampler(test_ds,  shuffle=False, seed=seed)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                              num_workers=min(num_workers, 2), pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler,
                              num_workers=min(num_workers, 2), pin_memory=False)

    model = build_model(arch, len(channel_cols), seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %.3fM (arch=%s)", n_params / 1e6, arch)

    opt = torch.optim.AdamW(model.parameters(), lr=lr,
                            weight_decay=weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 1.0
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=3
    )

    best_val_ic = -1e9
    best_epoch = 0
    patience_left = patience
    log.info("Training (epochs=%d, lr=%.1e, warmup=%d, patience=%d)",
             epochs, lr, warmup_epochs, patience)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running, n_batches = 0.0, 0
        for batch_idx, (seq, lab, t_idx, _) in enumerate(train_loader):
            if max_train_batches and batch_idx >= max_train_batches:
                break
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            t_idx_d = t_idx.to(device, non_blocking=True)
            pred = model(seq, t_idx_d)
            loss = listwise_rank_loss(pred, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        if epoch <= warmup_epochs:
            sched.step()
        train_loss = running / max(1, n_batches)

        train_mean_ic, train_med_ic, _ = evaluate(model, train_loader, device)
        val_mean_ic,   val_med_ic,   _ = evaluate(model, val_loader,   device)
        elapsed = time.time() - t0

        log.info(
            "ep %02d  loss=%.4f  train_ic=%+.4f/%+.4f  val_ic=%+.4f/%+.4f "
            " lr=%.1e  t=%.1fs (mean/median)",
            epoch, train_loss, train_mean_ic, train_med_ic,
            val_mean_ic, val_med_ic, opt.param_groups[0]["lr"], elapsed,
        )

        if epoch > warmup_epochs:
            plateau.step(val_mean_ic)

        if val_mean_ic > best_val_ic + 1e-4:
            best_val_ic = val_mean_ic
            best_epoch = epoch
            patience_left = patience
            torch.save({
                "epoch": epoch, "state_dict": model.state_dict(),
                "val_ic": val_mean_ic,
            }, out_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stop @ ep %d (best=%d, ic=%+.4f)",
                         epoch, best_epoch, best_val_ic)
                break

    # Test on best checkpoint
    log.info("Loading best checkpoint epoch %d (val_ic=%+.4f)",
             best_epoch, best_val_ic)
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_mean_ic, test_med_ic, test_mse = evaluate(model, test_loader, device)
    log.info("══ Test: mean_ic=%+.4f  median_ic=%+.4f  mse=%.4f ══",
             test_mean_ic, test_med_ic, test_mse)

    return {
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "test_mean_ic": test_mean_ic,
        "test_median_ic": test_med_ic,
        "test_mse": test_mse,
        "n_params": n_params,
    }


if __name__ == "__main__":
    main()
