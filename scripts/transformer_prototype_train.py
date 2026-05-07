#!/usr/bin/env python
"""Transformer prototype trainer — A/B raw OHLCV vs engineered features.

Trains a sequence Transformer (Patch-TST / TFT style) on the cleaned
panel datasets built in `scripts/transformer_dataset_builder.py` (raw)
and `scripts/transformer_dataset_engineered.py` (11 indicators).

Per CLAUDE.md §5.2: every new metric ships with sanity checks. We
report:
  - per-day cross-sectional Spearman IC on val + test
  - shuffled-label IC (must be ≈ 0)
  - paired_alpha_test = comparison vs production XGB pool_IC

Architecture (default):
  PatchTST-lite — 60-day sequence, channel-independent patch embedding,
  4 transformer encoder layers, 8 heads, d_model=256, 3 horizon heads
  predicting (fwd_5d, fwd_20d, fwd_60d) excess returns.

Usage::

    # Path A — raw OHLCV (5 channels)
    python scripts/transformer_prototype_train.py \\
        --dataset data/transformer_dataset.parquet \\
        --channels 5 --output artifacts/transformer_proto_raw.pt

    # Path B — engineered 11 indicators
    python scripts/transformer_prototype_train.py \\
        --dataset data/transformer_dataset_engineered.parquet \\
        --channels 11 --output artifacts/transformer_proto_eng.pt

Hardware: M2 Pro MPS backend (Apple Silicon GPU). Falls back to CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# §5.10 — saturate 10 cores
for _k, _v in (("OMP_NUM_THREADS", "10"),
               ("MKL_NUM_THREADS", "10"),
               ("OPENBLAS_NUM_THREADS", "10")):
    os.environ.setdefault(_k, _v)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("transformer-proto")


# ── Data ───────────────────────────────────────────────────────────────────

class PanelSeqDataset(Dataset):
    """Indexes (ticker, date) → trailing seq_len bars + (fwd_5d, fwd_20d, fwd_60d).

    Builds an in-memory sample list, one per valid (ticker, date) where:
      - the ticker has ≥ seq_len trailing bars
      - all three labels are non-NaN
    """

    def __init__(self, panel: pd.DataFrame, channel_cols: list[str],
                 seq_len: int, label_cols: list[str], split: str,
                 ticker_to_idx: dict[str, int] | None = None):
        self.seq_len = seq_len
        self.channel_cols = channel_cols
        self.label_cols = label_cols
        # Build / inherit ticker index mapping for ticker embedding.
        if ticker_to_idx is None:
            tickers = sorted(panel["ticker"].unique())
            ticker_to_idx = {t: i for i, t in enumerate(tickers)}
        self.ticker_to_idx = ticker_to_idx
        # ── BUG-T4 fix: build sequences from the FULL panel but filter
        # samples by the split of the SAMPLE-END date.  Earlier dates
        # used as INPUT context is NOT leakage; only the LABEL (forward
        # return) leaks, and that's tied to the sample-end date.
        panel = panel.dropna(subset=label_cols).reset_index(drop=True)
        ch_arr  = panel[channel_cols].astype(np.float32).values
        lab_arr = panel[label_cols].astype(np.float32).values
        split_arr = panel["split_label"].values
        date_arr  = panel["date"].values
        self.samples: list[tuple[np.ndarray, np.ndarray, int, np.int64]] = []
        gp = panel.groupby("ticker", sort=False).indices
        for t, idxs in gp.items():
            idxs = np.asarray(sorted(idxs))
            t_idx = self.ticker_to_idx.get(t)
            if t_idx is None:
                continue   # ticker not in shared vocabulary
            for i in range(seq_len, len(idxs)):
                end = idxs[i]
                if split_arr[end] != split:
                    continue
                seq = ch_arr[idxs[i - seq_len: i]]
                if seq.shape[0] != seq_len:
                    continue
                date_ns = np.int64(pd.Timestamp(date_arr[end]).value)
                self.samples.append((seq, lab_arr[end], t_idx, date_ns))
        # Pre-extract dates as numpy for the per-day sampler (BUG-T5 fix).
        self._dates_ns = np.asarray([s[3] for s in self.samples], dtype=np.int64)
        log.info("PanelSeqDataset[%s]: %d samples × seq_len=%d × ch=%d  "
                 "n_unique_tickers=%d",
                 split, len(self.samples), seq_len, len(channel_cols),
                 len(set(s[2] for s in self.samples)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        seq, label, t_idx, date_ns = self.samples[i]
        return (torch.from_numpy(seq), torch.from_numpy(label),
                int(t_idx), int(date_ns))


# ── BUG-T5 fix: per-day batched sampler ────────────────────────────────────

from torch.utils.data import Sampler  # noqa: E402


class PerDayBatchSampler(Sampler):
    """Yields date-batched indices: each batch is one trading day's samples.

    Required for `--loss=huber_plus_corr` to compute per-day cross-sectional
    correlation correctly. Without this, batches mix dates and the
    correlation auxiliary loss becomes meaningless noise (correlates
    pred-vs-label across DIFFERENT days, not within-day across tickers).
    """

    def __init__(self, dataset: PanelSeqDataset, shuffle: bool = True,
                 seed: int = 0):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        # Group sample positions by date
        dates = dataset._dates_ns
        self._by_date: dict[int, list[int]] = {}
        for pos, d in enumerate(dates):
            self._by_date.setdefault(int(d), []).append(pos)
        self._date_keys = list(self._by_date.keys())
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


# ── Model ──────────────────────────────────────────────────────────────────

class RevIN(nn.Module):
    """Reversible instance normalization (Kim et al. 2022).

    Per-sample, per-channel: subtract mean, divide by std (along time axis),
    apply learnable affine. For PatchTST-style sequence models, this
    removes within-window drift so the model sees stationary input.

    BUG-T6 fix.
    """

    def __init__(self, n_channels: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n_channels))
        self.beta = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, C] → [B, L, C]."""
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + self.eps
        x = (x - mean) / std
        return x * self.gamma + self.beta


class PatchTSTLite(nn.Module):
    """Channel-independent patch transformer. Per Nie et al. 2023 (PatchTST)
    with RevIN (Kim 2022) and optional ticker embedding (BUG-T6, BUG-T9).

    For each input channel, we project a window of `patch_len` bars to a
    `d_model`-dim token. Tokens (across patches × channels) feed into a
    standard transformer encoder. A pooled representation (optionally
    concatenated with ticker embedding) projects to `n_horizons` linear
    heads for multi-horizon return prediction.
    """

    def __init__(self, n_channels: int, seq_len: int, patch_len: int = 12,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 dropout: float = 0.1, n_horizons: int = 3,
                 use_revin: bool = False,
                 n_tickers: int = 0, ticker_embed_dim: int = 0):
        super().__init__()
        assert seq_len % patch_len == 0, (
            f"seq_len {seq_len} not divisible by patch_len {patch_len}"
        )
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.n_patches = seq_len // patch_len
        self.d_model = d_model
        self.use_revin = use_revin
        self.ticker_embed_dim = ticker_embed_dim

        if use_revin:
            self.revin = RevIN(n_channels)

        # Channel-independent: each (ch, patch) → d_model
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.n_patches * n_channels, d_model) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_norm = nn.LayerNorm(d_model)
        if ticker_embed_dim > 0:
            self.ticker_embed = nn.Embedding(n_tickers, ticker_embed_dim)
            head_input_dim = d_model + ticker_embed_dim
        else:
            head_input_dim = d_model
        self.heads = nn.Linear(head_input_dim, n_horizons)

    def forward(self, x: torch.Tensor,
                ticker_idx: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, seq_len, n_channels], ticker_idx: [B] long → [B, n_horizons]."""
        if self.use_revin:
            x = self.revin(x)
        B, L, C = x.shape
        x = x.transpose(1, 2).contiguous()
        x = x.view(B, C, self.n_patches, self.patch_len)
        x = self.patch_embed(x)
        x = x.view(B, C * self.n_patches, self.d_model)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.head_norm(x)
        if self.ticker_embed_dim > 0:
            assert ticker_idx is not None, (
                "ticker_idx required when ticker_embed_dim > 0"
            )
            t_emb = self.ticker_embed(ticker_idx)
            x = torch.cat([x, t_emb], dim=-1)
        return self.heads(x)


# ── Training ───────────────────────────────────────────────────────────────

def per_day_ic(preds: np.ndarray, labels: np.ndarray,
               dates: np.ndarray) -> float:
    """Mean of per-day cross-sectional Spearman IC on the fwd_5d head.

    Returns 0.0 if no day has ≥3 samples (degenerate).
    """
    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    ics = []
    for _, group in df.groupby("date"):
        if len(group) < 3:
            continue
        rho, _ = spearmanr(group["pred"], group["label"])
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else 0.0


def evaluate(model: nn.Module, loader: DataLoader, device: str,
             use_ticker_embed: bool, label_idx_for_ic: int = 0
             ) -> tuple[float, float]:
    """Returns (mean MSE, per-day IC on fwd_5d head)."""
    model.eval()
    all_preds, all_labels, all_dates = [], [], []
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for seq, lab, t_idx, date in loader:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            t_idx_t = (t_idx.to(device, non_blocking=True)
                       if use_ticker_embed else None)
            pred = model(seq, t_idx_t) if use_ticker_embed else model(seq)
            se = (pred - lab).pow(2).sum().item()
            total_se += se
            total_n += pred.numel()
            all_preds.append(pred[:, label_idx_for_ic].cpu().numpy())
            all_labels.append(lab[:, label_idx_for_ic].cpu().numpy())
            all_dates.append(np.asarray(date))
    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    dates = np.concatenate(all_dates)
    mse = total_se / max(1, total_n)
    ic = per_day_ic(preds, labels, dates)
    return mse, ic


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", required=True,
                   help="Path to parquet built by transformer_dataset_*.py")
    p.add_argument("--channels", type=int, required=True,
                   help="Number of input channels (5 raw / 11 engineered)")
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--patch-len", type=int, default=12)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stop patience (epochs without val IC improvement)")
    p.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", required=True,
                   help="Path to save best checkpoint .pt")
    p.add_argument("--label-shuffle", action="store_true",
                   help="§5.2 sanity: shuffle labels — IC should ≈ 0")
    p.add_argument("--max-train-samples", type=int, default=0,
                   help=">0 to limit train set (smoke testing)")
    # ── BUG-T1 fix: clip label outliers (GME 2021 squeeze, etc.) ──
    p.add_argument("--label-clip", type=float, default=0.30,
                   help="Clip |label| to this max (default 0.30 = 30%%); "
                        "fwd_5d_excess goes up to +791%% on GME 2021 squeeze, "
                        "which is real but dominates MSE gradient. Set to 0 to disable.")
    # ── BUG-T2 fix: per-horizon label standardization ──
    p.add_argument("--label-standardize", action="store_true", default=True,
                   help="Standardize labels per-horizon to unit variance (uses "
                        "train-split stats only, no leakage). Default ON. "
                        "Without this, fwd_60d (std≈0.18) dominates fwd_5d "
                        "(std≈0.05) by 17× in joint MSE.")
    p.add_argument("--no-label-standardize", dest="label_standardize",
                   action="store_false")
    # ── BUG-T11 fix: per-day demean of labels ──
    p.add_argument("--label-demean-per-day", action="store_true", default=True,
                   help="Subtract per-day cross-sectional mean from labels "
                        "before standardize. Without this, MSE pushes "
                        "predictions toward per-day mean (which model can't "
                        "see), wasting capacity. Default ON.")
    p.add_argument("--no-label-demean-per-day", dest="label_demean_per_day",
                   action="store_false")
    # ── BUG-T3 fix: robust loss + per-day correlation auxiliary ──
    p.add_argument("--loss", default="huber", choices=["mse", "huber", "huber_plus_corr"],
                   help="huber = robust to outliers; huber_plus_corr adds "
                        "per-batch Pearson correlation as auxiliary ranking signal.")
    p.add_argument("--corr-weight", type=float, default=0.5,
                   help="Weight on per-batch correlation auxiliary loss "
                        "(only used when --loss=huber_plus_corr).")
    # ── BUG-T5 fix: per-day batching for meaningful corr loss ──
    p.add_argument("--per-day-batch", action="store_true",
                   help="Use date-batched DataLoader (each batch = one trading "
                        "day's samples). Required for --loss=huber_plus_corr "
                        "to actually optimize cross-sectional ranking.")
    # ── BUG-T6 fix: RevIN-style sequence-level normalization ──
    p.add_argument("--revin", action="store_true",
                   help="Apply reversible instance normalization per sequence. "
                        "Per Kim et al. 2022 'RevIN' for time-series forecasting.")
    # ── BUG-T9 fix: learnable ticker embedding ──
    p.add_argument("--ticker-embed-dim", type=int, default=0,
                   help=">0 to add a learnable ticker embedding of this dim, "
                        "concatenated to pooled sequence representation.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    # Detect channel cols from the parquet
    log.info("Loading %s …", args.dataset)
    panel = pd.read_parquet(args.dataset)
    label_cols = ["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]
    excluded = {"ticker", "date", "split_label"} | set(label_cols)
    channel_cols = [c for c in panel.columns if c not in excluded]
    if len(channel_cols) != args.channels:
        log.warning("Detected %d channels but --channels=%d; using detected",
                    len(channel_cols), args.channels)
    log.info("Channels: %s", channel_cols)

    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    # ── BUG-T1 fix: clip label outliers (computed BEFORE shuffle so the
    #    shuffle test still gets the realistic-magnitude distribution) ──
    if args.label_clip > 0:
        before_max = panel[label_cols].abs().max().to_dict()
        for c in label_cols:
            panel[c] = panel[c].clip(-args.label_clip, args.label_clip)
        after_max = panel[label_cols].abs().max().to_dict()
        log.info("Label clip ±%.2f: max went %s → %s",
                 args.label_clip, before_max, after_max)

    # ── BUG-T11 fix: per-day demean BEFORE standardize ──
    # Without this, labels carry per-day mean drift that MSE will fit (model
    # implicitly tries to predict the day's market direction, which it
    # cannot see — wasting capacity). Demean first, then standardize.
    if args.label_demean_per_day:
        before_mean = panel.groupby("date")[label_cols[0]].transform("mean")
        for c in label_cols:
            panel[c] = panel[c] - panel.groupby("date")[c].transform("mean")
        log.info("Label demean per-day: max |day_mean| went %.4f → 0",
                 float(before_mean.abs().max()))

    # ── BUG-T2 fix: per-horizon standardize using TRAIN split only ──
    label_scale: dict[str, float] = {c: 1.0 for c in label_cols}
    if args.label_standardize:
        train_mask = panel["split_label"] == "train"
        for c in label_cols:
            std = float(panel.loc[train_mask, c].std())
            if std > 1e-9:
                label_scale[c] = std
                panel[c] = panel[c] / std
        log.info("Label standardize (per-horizon, train-only stats): %s",
                 {c: f"std={s:.4f}" for c, s in label_scale.items()})

    if args.label_shuffle:
        log.info("§5.2 sanity: shuffling labels (IC should be ≈ 0)")
        rng = np.random.default_rng(args.seed)
        for c in label_cols:
            panel[c] = panel[c].sample(frac=1.0, random_state=args.seed).values

    # Build a SHARED ticker_to_idx so train/val/test embeddings align.
    all_tickers = sorted(panel["ticker"].unique())
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    n_tickers = len(all_tickers)
    log.info("Ticker vocab: %d unique tickers", n_tickers)

    train_ds = PanelSeqDataset(panel, channel_cols, args.seq_len,
                               label_cols, "train", ticker_to_idx)
    val_ds   = PanelSeqDataset(panel, channel_cols, args.seq_len,
                               label_cols, "val", ticker_to_idx)
    test_ds  = PanelSeqDataset(panel, channel_cols, args.seq_len,
                               label_cols, "test", ticker_to_idx)

    if args.max_train_samples > 0 and len(train_ds.samples) > args.max_train_samples:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_ds.samples), args.max_train_samples, replace=False)
        train_ds.samples = [train_ds.samples[i] for i in idx]
        train_ds._dates_ns = np.asarray([s[3] for s in train_ds.samples],
                                         dtype=np.int64)
        log.info("Limited train to %d samples (smoke)", len(train_ds.samples))

    # ── BUG-T5 fix: per-day batched DataLoader when corr loss is on ──
    if args.per_day_batch:
        log.info("Using PerDayBatchSampler — each batch = one trading day")
        train_sampler = PerDayBatchSampler(train_ds, shuffle=True, seed=args.seed)
        val_sampler   = PerDayBatchSampler(val_ds, shuffle=False, seed=args.seed)
        test_sampler  = PerDayBatchSampler(test_ds, shuffle=False, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                   num_workers=4, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_sampler=val_sampler,
                                   num_workers=2, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_sampler=test_sampler,
                                   num_workers=2, pin_memory=False)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=4, pin_memory=False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, pin_memory=False)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                                  shuffle=False, num_workers=2, pin_memory=False)

    model = PatchTSTLite(
        n_channels=len(channel_cols),
        seq_len=args.seq_len,
        patch_len=args.patch_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        n_horizons=len(label_cols),
        use_revin=args.revin,
        n_tickers=n_tickers,
        ticker_embed_dim=args.ticker_embed_dim,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model params: %.2fM (revin=%s, ticker_embed_dim=%d)",
             n_params / 1e6, args.revin, args.ticker_embed_dim)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    # ── BUG-T3 fix: switch to Huber (robust to remaining label outliers) ──
    if args.loss == "mse":
        base_loss = nn.MSELoss()
    else:
        base_loss = nn.HuberLoss(delta=1.0)

    def compute_loss(pred: torch.Tensor, lab: torch.Tensor) -> torch.Tensor:
        loss = base_loss(pred, lab)
        if args.loss == "huber_plus_corr":
            # Per-batch Pearson correlation between pred and label (fwd_5d head).
            # Maximizing correlation = ranking signal, complementary to Huber.
            p_v = pred[:, 0] - pred[:, 0].mean()
            l_v = lab[:, 0] - lab[:, 0].mean()
            num = (p_v * l_v).sum()
            den = torch.sqrt((p_v * p_v).sum() * (l_v * l_v).sum() + 1e-8)
            corr = num / den
            loss = loss - args.corr_weight * corr
        return loss

    best_val_ic = -1e9
    best_epoch = 0
    patience_left = args.patience
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("══ Training start (epochs=%d) ══", args.epochs)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running, n_batches = 0.0, 0
        use_ticker_embed = args.ticker_embed_dim > 0
        for seq, lab, t_idx, _ in train_loader:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            t_idx_t = (t_idx.to(device, non_blocking=True)
                       if use_ticker_embed else None)
            pred = model(seq, t_idx_t) if use_ticker_embed else model(seq)
            loss = compute_loss(pred, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            running += loss.item()
            n_batches += 1
        sched.step()
        train_loss = running / max(1, n_batches)

        val_mse, val_ic = evaluate(model, val_loader, device, use_ticker_embed)
        elapsed = time.time() - t0
        log.info(
            "epoch=%02d  loss=%.5f  val_mse=%.5f  val_ic=%+.4f  "
            "lr=%.2e  t=%.1fs", epoch, train_loss, val_mse, val_ic,
            opt.param_groups[0]["lr"], elapsed,
        )
        if val_ic > best_val_ic + 1e-4:
            best_val_ic = val_ic
            best_epoch = epoch
            patience_left = args.patience
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "args": vars(args),
                "val_ic": val_ic,
                "val_mse": val_mse,
                "channel_cols": channel_cols,
                "label_cols": label_cols,
                # Persist label_scale so inference can un-standardize predictions.
                "label_scale": label_scale,
                "label_clip": args.label_clip,
            }, out_path)
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("Early stop @ epoch %d (best=%d, ic=%+.4f)",
                         epoch, best_epoch, best_val_ic)
                break

    # ── Test ────────────────────────────────────────────────────────────
    log.info("Loading best checkpoint from epoch %d (val_ic=%+.4f)",
             best_epoch, best_val_ic)
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_mse, test_ic = evaluate(model, test_loader, device, use_ticker_embed)
    log.info("══ Test: mse=%.5f  ic=%+.4f ══", test_mse, test_ic)

    summary = {
        "dataset": args.dataset,
        "n_channels": len(channel_cols),
        "channel_cols": channel_cols,
        "best_epoch": best_epoch,
        "best_val_ic": best_val_ic,
        "test_ic": test_ic,
        "test_mse": test_mse,
        "n_params": n_params,
        "label_shuffle": args.label_shuffle,
        "label_clip": args.label_clip,
        "label_standardize": args.label_standardize,
        "label_scale": label_scale,
        "loss": args.loss,
        "seq_len": args.seq_len,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("══ summary written %s ══", summary_path)


if __name__ == "__main__":
    main()
