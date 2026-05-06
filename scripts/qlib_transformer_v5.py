#!/usr/bin/env python
"""Qlib-faithful Transformer (TransformerModel) baseline.

Source: qlib/contrib/model/pytorch_transformer_ts.py — read 2026-05-06.
Class TransformerModel + Transformer module, lines 25-264 of that file.

## Faithful replication

Architecture (from `class Transformer`, lines 238-264):
  feature_layer  = nn.Linear(d_feat, d_model)
  pos_encoder    = PositionalEncoding(d_model)         # sinusoidal, max_len=1000
  encoder_layer  = nn.TransformerEncoderLayer(d_model, nhead, dropout)
  transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
  decoder_layer  = nn.Linear(d_model, 1)
Forward: src [N,T,F] → feature_layer → transpose to [T,N,F] (NOT batch_first!)
         → pos_encoder → transformer_encoder → decoder_layer(out[-1]) → squeeze

Hyperparameters (from TransformerModel.__init__, lines 26-44):
  d_feat=20, d_model=64, batch_size=8192, nhead=2, num_layers=2, dropout=0
  n_epochs=100, lr=1e-4, reg=1e-3, early_stop=5, loss="mse", optimizer="adam"
  Note: defaults in source. The `Transformer` module class has dropout=0.5
  default, but TransformerModel passes through dropout=0. Both are options.

Loss (lines 82-92):
  mse on (pred[mask], label[mask]) where mask = ~isnan(label)

Training loop (lines 102-115, 137-199):
  AdaM, gradient clip via clip_grad_value_(3.0)
  Best score = highest val metric (= -val_loss = lowest val_mse)
  Early stop after 5 epochs without improvement

## Adaptation to RenQuant
- Input: alpha158_qlib_dataset.parquet, 158 features, sequence-windowed.
- We build sequences of past 60 days × 158 features → predict label_at_t.
- Same MSE + CSZScoreNorm-on-labels pipeline.
- Per-ticker dataset = same PerDayDataset pattern as transformer_v4.

Usage::

    python scripts/qlib_transformer_v5.py
    python scripts/qlib_transformer_v5.py --label fwd_60d_excess
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

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
log = logging.getLogger("qlib-transformer-v5")


# ── Data ───────────────────────────────────────────────────────────────────

class PanelSeqDataset(Dataset):
    """Lazy sequence builder — keeps ONE shared `ch_arr` (the full feature
    matrix) and just stores (start_pos, end_pos) tuples per sample.

    Avoids the 60×158×4×N_samples pre-materialization explosion that
    OOM'd at 60GB on first attempt. With 396k samples × 60 × 158 × 4 = 15GB
    of duplication, we instead store one 105MB ch_arr + 396k × 8 bytes index = 3MB.
    Builds the [seq_len, n_channels] slice on demand in __getitem__.
    """

    def __init__(self, panel: pd.DataFrame, channel_cols: list[str],
                 seq_len: int, label_col: str, split: str):
        self.seq_len = seq_len
        panel = panel.dropna(subset=[label_col]).reset_index(drop=True)
        # Single shared feature matrix — no per-sample copies.
        self.ch_arr  = panel[channel_cols].astype(np.float32).fillna(0.0).values
        self.lab_arr = panel[label_col].astype(np.float32).values
        split_arr = panel["split_label"].values
        date_arr  = panel["date"].values
        # Each sample = (start_idx, end_idx, date_ns) — 8 bytes × 3 each.
        self.indices: list[tuple[int, int, np.int64]] = []
        gp = panel.groupby("ticker", sort=False).indices
        for t, idxs in gp.items():
            idxs = np.asarray(sorted(idxs))
            for i in range(seq_len, len(idxs)):
                end = idxs[i]
                if split_arr[end] != split:
                    continue
                start = idxs[i - seq_len]
                if end - start != seq_len:
                    continue   # gap in ticker's history
                self.indices.append(
                    (int(start), int(end), np.int64(pd.Timestamp(date_arr[end]).value))
                )
        log.info("PanelSeqDataset[%s]: %d samples × seq_len=%d × ch=%d "
                 "(lazy; mem=%.1f MB shared + %.1f MB index)",
                 split, len(self.indices), seq_len, len(channel_cols),
                 self.ch_arr.nbytes / 1e6, len(self.indices) * 24 / 1e6)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        start, end, date_ns = self.indices[i]
        seq = self.ch_arr[start:end]   # view, not copy
        lab = self.lab_arr[end]
        return torch.from_numpy(seq), torch.tensor(lab), torch.tensor(date_ns)


# ── Model (faithful Qlib lines 222-264) ────────────────────────────────────

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding from Qlib pytorch_transformer_ts:222-235."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, N, F]
        return x + self.pe[: x.size(0), :]


class QlibTransformer(nn.Module):
    """Faithful Qlib pytorch_transformer_ts.py:Transformer (lines 238-264)."""

    def __init__(self, d_feat: int, d_model: int = 64, nhead: int = 2,
                 num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                          num_layers=num_layers)
        self.decoder_layer = nn.Linear(d_model, 1)
        self.d_feat = d_feat

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [N, T, F]
        src = self.feature_layer(src)              # [N, T, d_model]
        src = src.transpose(1, 0)                   # → [T, N, d_model]
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src, mask=None)  # [T, N, d_model]
        # Take last timestep, project to scalar (matches Qlib's decoder_layer)
        return self.decoder_layer(out.transpose(1, 0)[:, -1, :]).squeeze(-1)


# ── Eval ───────────────────────────────────────────────────────────────────

def per_day_ic(preds, labels, dates) -> tuple[float, float]:
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


def evaluate(model, loader, device):
    model.eval()
    preds_, labels_, dates_ = [], [], []
    total_se, total_n = 0.0, 0
    with torch.no_grad():
        for seq, lab, date in loader:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            pred = model(seq.float())
            mask = ~torch.isnan(lab)
            if mask.any():
                total_se += ((pred[mask] - lab[mask]) ** 2).sum().item()
                total_n += int(mask.sum().item())
            preds_.append(pred.cpu().numpy())
            labels_.append(lab.cpu().numpy())
            dates_.append(date.numpy())
    preds = np.concatenate(preds_)
    labels = np.concatenate(labels_)
    dates = np.concatenate(dates_)
    mse = total_se / max(1, total_n)
    mean_ic, med_ic = per_day_ic(preds, labels, dates)
    return mean_ic, med_ic, mse


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset",
                   default=str(REPO_ROOT / "data" / "alpha158_qlib_dataset.parquet"))
    p.add_argument("--label", default="fwd_5d_excess",
                   choices=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"])
    p.add_argument("--seq-len", type=int, default=60)
    # Qlib defaults (TransformerModel.__init__)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--nhead", type=int, default=2)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--n-epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--reg", type=float, default=1e-3,
                   help="weight_decay (Qlib name: reg)")
    p.add_argument("--early-stop", type=int, default=5)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output",
                   default=str(REPO_ROOT / "artifacts" / "qlib_transformer_v5.pt"))
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

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    log.info("Loading %s", args.dataset)
    panel = pd.read_parquet(args.dataset)
    excluded = {"ticker", "date", "split_label",
                "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
    channel_cols = [c for c in panel.columns if c not in excluded]
    log.info("Channels (%d) — first 5: %s", len(channel_cols), channel_cols[:5])
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)

    train_ds = PanelSeqDataset(panel, channel_cols, args.seq_len,
                                args.label, "train")
    val_ds   = PanelSeqDataset(panel, channel_cols, args.seq_len,
                                args.label, "val")
    test_ds  = PanelSeqDataset(panel, channel_cols, args.seq_len,
                                args.label, "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, drop_last=False)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size,
                              shuffle=False, num_workers=2, drop_last=False)

    model = QlibTransformer(
        d_feat=len(channel_cols),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model: QlibTransformer  params=%.3fM", n_params / 1e6)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    def loss_fn(pred, lab):
        mask = ~torch.isnan(lab)
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        return ((pred[mask] - lab[mask]) ** 2).mean()

    log.info("Training: epochs=%d  lr=%.1e  weight_decay=%.1e  early_stop=%d",
             args.n_epochs, args.lr, args.reg, args.early_stop)

    best_score = -1e9
    best_epoch = 0
    stop_counter = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.n_epochs + 1):
        t0 = time.time()
        model.train()
        n_batches = 0
        running_loss = 0.0
        for seq, lab, _ in train_loader:
            seq = seq.to(device, non_blocking=True)
            lab = lab.to(device, non_blocking=True)
            pred = model(seq.float())
            loss = loss_fn(pred, lab)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 3.0)  # Qlib uses value-clip
            opt.step()
            running_loss += loss.item()
            n_batches += 1
        train_loss = running_loss / max(1, n_batches)

        train_ic, train_med, train_mse = evaluate(model, train_loader, device)
        val_ic,   val_med,   val_mse   = evaluate(model, val_loader,   device)
        elapsed = time.time() - t0
        log.info(
            "ep %02d  loss=%.5f  train_ic=%+.4f/%+.4f train_mse=%.5f"
            "  val_ic=%+.4f/%+.4f val_mse=%.5f  t=%.1fs",
            epoch, train_loss, train_ic, train_med, train_mse,
            val_ic, val_med, val_mse, elapsed,
        )

        # Qlib's metric_fn = -loss → higher = better, score = -val_mse
        score = -val_mse
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stop_counter = 0
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_ic": val_ic, "val_mse": val_mse}, out_path)
        else:
            stop_counter += 1
            if stop_counter >= args.early_stop:
                log.info("Early stop at epoch %d (best=%d, ic=%+.4f)",
                         epoch, best_epoch, best_score)
                break

    # Test on best
    log.info("Loading best checkpoint @ epoch %d (val_score=%+.4f)",
             best_epoch, best_score)
    ckpt = torch.load(out_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    test_ic, test_med, test_mse = evaluate(model, test_loader, device)
    log.info("══ TEST: mean_ic=%+.4f median_ic=%+.4f mse=%.4f ══",
             test_ic, test_med, test_mse)

    summary = {
        "model": "qlib_transformer_v5",
        "label": args.label,
        "n_features": len(channel_cols),
        "best_epoch": best_epoch,
        "n_params": n_params,
        "args": vars(args),
        "test_mean_ic": test_ic,
        "test_median_ic": test_med,
        "test_mse": test_mse,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Summary written: %s", summary_path)


if __name__ == "__main__":
    main()
