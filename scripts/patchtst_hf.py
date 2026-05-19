#!/usr/bin/env python3
"""PatchTST cross-sectional ranker via HuggingFace transformers.

REPLACES scripts/transformer_v4.py (784 LOC custom). Per 2026-05-18
user mandate "尽量用第三方lib". Uses `transformers.PatchTSTModel`
(backbone) + minimal pairwise-ranking head (Burges 2005 RankNet).

Architecture:
  input  → PatchTSTModel.encoder → last_hidden_state (B, n_patches, d_model)
  pool   → mean over n_patches → (B, d_model)
  head   → Linear(d_model, 1) → score per ticker
  loss   → pairwise BCE on (i, j) pairs within the same day (per-day batch)

References:
  - Nie et al 2023 ICLR "A Time Series is Worth 64 Words" (PatchTST)
  - Burges et al 2005 ICML "Learning to Rank using Gradient Descent" (RankNet)
  - HuggingFace transformers PatchTST docs

Usage::

    .venv/bin/python scripts/patchtst_hf.py \\
        --dataset data/transformer_v4_wl200_clean.parquet \\
        --cut cut1_covid --epochs 5 --device mps --output-dir artifacts/hf_smoke

PRIME DIRECTIVE: pass --cut name from kernel.walk_forward_splits
(default: cut1_covid). DO NOT use 2023-only val (PRIME DIRECTIVE violation).
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from transformers import PatchTSTConfig, PatchTSTModel

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from kernel.walk_forward_splits import (build_default_cuts,
                                          assign_split_column)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patchtst-hf")


class HFPatchTSTRanker(nn.Module):
    """HF PatchTST backbone + minimal ranking head."""

    def __init__(self, cfg: PatchTSTConfig):
        super().__init__()
        self.backbone = PatchTSTModel(cfg)
        self.head = nn.Linear(cfg.d_model, 1)

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(past_values=past_values)
        # last_hidden_state: (B, n_ch, n_patches, d_model) — pool over patches and channels
        h = out.last_hidden_state.mean(dim=(1, 2))  # (B, d_model)
        return self.head(h).squeeze(-1)  # (B,)


def pairwise_rank_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Burges 2005 RankNet pairwise BCE on (i, j) pairs within batch.

    Targets: 1 if label_i > label_j else 0; loss = BCEWithLogits on
    (score_i - score_j)."""
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    mask = torch.triu(torch.ones(n, n, device=scores.device), diagonal=1).bool()
    s_diff = scores.unsqueeze(1) - scores.unsqueeze(0)        # (n, n)
    l_diff = labels.unsqueeze(1) - labels.unsqueeze(0)        # (n, n)
    target = (l_diff > 0).float()
    return F.binary_cross_entropy_with_logits(s_diff[mask], target[mask])


def per_day_csrankic(preds: np.ndarray, labels: np.ndarray,
                     dates: np.ndarray) -> tuple[float, float]:
    df = pd.DataFrame({"pred": preds, "label": labels, "date": dates})
    ics = []
    for _, g in df.groupby("date"):
        if len(g) < 5: continue
        r, _ = spearmanr(g["pred"], g["label"])
        if not np.isnan(r): ics.append(r)
    if not ics: return 0.0, 0.0
    return float(np.mean(ics)), float(np.median(ics))


def load_panel_with_split(dataset_path: Path, cut_name: str,
                          label_col: str) -> tuple[pd.DataFrame, list[str]]:
    panel = pd.read_parquet(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.dropna(subset=[label_col])
    cut = next(c for c in build_default_cuts() if c.name == cut_name)
    panel["split_label"] = assign_split_column(panel, cut)
    feat_cols = [c for c in panel.columns
                 if c not in {"date", "ticker", "split_label",
                              "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
                 and panel[c].dtype.kind in "fiub"]
    log.info("panel %d rows | cut=%s | train=%d val=%d test=%d | n_feat=%d",
             len(panel), cut_name,
             (panel["split_label"] == "train").sum(),
             (panel["split_label"] == "val").sum(),
             (panel["split_label"] == "test").sum(),
             len(feat_cols))
    return panel, feat_cols


def build_per_day_batches(panel: pd.DataFrame, feat_cols: list[str],
                           label_col: str, seq_len: int, split: str
                           ) -> list[dict]:
    """Returns list of per-day batches. Each batch = (B, seq_len, n_ch) +
    labels + dates. Sequence built from ticker's recent seq_len bars
    ending at the sample date."""
    feat_arr = panel[feat_cols].astype(np.float32).fillna(0.0).values
    lab_arr = panel[label_col].astype(np.float32).values
    by_ticker = panel.groupby("ticker", sort=False).indices
    samples_by_date: dict[int, list[dict]] = {}
    for ticker, idxs in by_ticker.items():
        idxs = np.asarray(sorted(idxs))
        for i in range(seq_len, len(idxs)):
            end_pos = idxs[i]
            if panel.iloc[end_pos]["split_label"] != split: continue
            window = feat_arr[idxs[i - seq_len: i]]
            if window.shape[0] != seq_len: continue
            d = panel.iloc[end_pos]["date"]
            samples_by_date.setdefault(d.value, []).append({
                "x": window, "y": lab_arr[end_pos], "date": d,
            })
    batches = []
    for d_ns, samples in samples_by_date.items():
        if len(samples) < 5: continue
        xs = np.stack([s["x"] for s in samples])  # (B, seq_len, n_ch)
        ys = np.array([s["y"] for s in samples], dtype=np.float32)
        dates = np.array([s["date"].value for s in samples])
        batches.append({"x": xs, "y": ys, "date": dates})
    return batches


def train_one(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    panel, feat_cols = load_panel_with_split(
        Path(args.dataset), args.cut, args.label)
    train_b = build_per_day_batches(panel, feat_cols, args.label,
                                      args.seq_len, "train")
    val_b = build_per_day_batches(panel, feat_cols, args.label,
                                    args.seq_len, "val")
    log.info("batches train=%d val=%d", len(train_b), len(val_b))

    cfg = PatchTSTConfig(
        num_input_channels=len(feat_cols),
        context_length=args.seq_len,
        patch_length=args.patch_length,
        patch_stride=args.patch_length,  # non-overlapping patches
        d_model=args.d_model,
        num_attention_heads=args.n_heads,
        num_hidden_layers=args.n_layers,
        ffn_dim=args.d_model * 2,
    )
    device = args.device
    model = HFPatchTSTRanker(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                              weight_decay=args.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("HF PatchTST n_params=%.2fM", n_params / 1e6)

    best_val_ic = -1e9
    for ep in range(args.epochs):
        model.train()
        np.random.shuffle(train_b)
        losses = []
        for b in train_b:
            x = torch.from_numpy(b["x"]).to(device)
            y = torch.from_numpy(b["y"]).to(device)
            opt.zero_grad()
            scores = model(x)
            loss = pairwise_rank_loss(scores, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        all_p, all_y, all_d = [], [], []
        with torch.no_grad():
            for b in val_b:
                x = torch.from_numpy(b["x"]).to(device)
                scores = model(x).cpu().numpy()
                all_p.append(scores); all_y.append(b["y"]); all_d.append(b["date"])
        if all_p:
            val_mean_ic, val_med_ic = per_day_csrankic(
                np.concatenate(all_p), np.concatenate(all_y),
                np.concatenate(all_d))
        else:
            val_mean_ic = val_med_ic = 0.0
        log.info("ep %02d  loss=%.4f  val_ic=%+.4f (med %+.4f)",
                 ep, np.mean(losses), val_mean_ic, val_med_ic)
        if val_mean_ic > best_val_ic + 1e-4:
            best_val_ic = val_mean_ic

    # Dump val predictions for downstream regime-stratified IC
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_p, all_y, all_d = [], [], []
    model.eval()
    with torch.no_grad():
        for b in val_b:
            x = torch.from_numpy(b["x"]).to(device)
            scores = model(x).cpu().numpy()
            all_p.append(scores); all_y.append(b["y"]); all_d.append(b["date"])
    if all_p:
        preds_df = pd.DataFrame({
            "date": pd.to_datetime(np.concatenate(all_d)),
            "pred": np.concatenate(all_p),
            "label": np.concatenate(all_y),
        })
        dump = out_dir / f"hf_patchtst_{args.cut}_seed{args.seed}_val_preds.parquet"
        preds_df.to_parquet(dump, index=False)
        log.info("preds dumped: %s (%d rows)", dump.name, len(preds_df))

    summary = {
        "arch": "hf_patchtst", "cut": args.cut, "seed": args.seed,
        "best_val_ic": best_val_ic, "n_params": n_params,
        "n_features": len(feat_cols),
    }
    (out_dir / f"hf_patchtst_{args.cut}_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--cut", default="cut1_covid",
                   help="Walk-forward cut name from kernel.walk_forward_splits")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--patch-length", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu",
                   choices=["cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default="artifacts/hf_patchtst")
    args = p.parse_args()

    summary = train_one(args)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
