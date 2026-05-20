#!/usr/bin/env python3
"""DLinear cross-sectional baseline — §5.12 must-have for transformer validation.

If PatchTST/HF cannot beat single-matmul DLinear by ≥+0.005 min-regime IC,
the architecture is NOT the bottleneck (labels/features/sample size is).

Reference:
  Zeng et al 2023 AAAI "Are Transformers Effective for Time Series
  Forecasting?" (arXiv 2205.13504). DLinear = series decomposition into
  trend + seasonal, then two parallel Linear layers, summed.

Architecture (adapted for cross-sectional ranking):
  input          : (B, T, F) per-ticker sequence
  decompose      : moving-avg kernel_size=25 → trend; x - trend → seasonal
  per-channel    : Linear(T → 1) on trend, Linear(T → 1) on seasonal
                   sum → (B, F) per-channel summary
  ranking head   : Linear(F → 1) → (B,) score

Loss: torch.nn.functional.margin_ranking_loss (apples-to-apples vs PatchTST).
Training infra: HF Trainer + PerRegimeICCallback (canonical lib, same as
patchtst_hf.py per CLAUDE.md §5.12).
"""
from __future__ import annotations
import argparse
import importlib.util
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
from transformers import Trainer, TrainingArguments

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _load_patchtst_hf_helpers():
    """Reuse preprocessing / dataset / callback / loss from patchtst_hf.py.
    importlib pattern avoids kernel.* namespace conflicts that bit
    HFPatchTSTPanelScorer prior."""
    spec = importlib.util.spec_from_file_location(
        "patchtst_hf_helpers", REPO / "scripts/patchtst_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("dlinear-baseline")


# ─── Model: DLinear ranker ──────────────────────────────────────────────────

class _SeriesDecomp(nn.Module):
    """Moving-average decomposition: trend = avg_pool(x), seasonal = x - trend."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        # Padding to keep length; same as cure-lab implementation
        self.pad = (kernel_size - 1) // 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, F); pool over T dimension per channel
        x_t = x.transpose(1, 2)  # (B, F, T)
        # Reflect-pad both ends so trend has same length as input
        x_padded = F.pad(x_t, (self.pad, self.kernel_size - 1 - self.pad),
                          mode="replicate")
        trend = F.avg_pool1d(x_padded, kernel_size=self.kernel_size, stride=1)
        trend = trend.transpose(1, 2)  # back to (B, T, F)
        seasonal = x - trend
        return seasonal, trend


class DLinearRanker(nn.Module):
    """DLinear adapted for cross-sectional ranking (B,) per ticker."""

    def __init__(self, seq_len: int, n_features: int, kernel_size: int = 25):
        super().__init__()
        self.decompose = _SeriesDecomp(kernel_size)
        # Per-channel: collapse T → 1
        self.linear_seasonal = nn.Linear(seq_len, 1)
        self.linear_trend = nn.Linear(seq_len, 1)
        # Cross-channel head: (B, F) → (B,) ranking score
        self.head = nn.Linear(n_features, 1)

    def forward(self, past_values: torch.Tensor,
                labels: torch.Tensor | None = None,
                dates=None) -> dict:
        seasonal, trend = self.decompose(past_values)
        # (B, T, F) → (B, F, T) → Linear(T → 1) → (B, F, 1) → squeeze → (B, F)
        sea = self.linear_seasonal(seasonal.transpose(1, 2)).squeeze(-1)
        tre = self.linear_trend(trend.transpose(1, 2)).squeeze(-1)
        h = sea + tre  # (B, F)
        score = self.head(h).squeeze(-1)  # (B,)
        return {"score": score}


# ─── Trainer (Margin Ranking only — no distributional head for DLinear) ─────

class DLinearTrainer(Trainer):
    def __init__(self, *args, ranking_margin: float = 0.1, **kw):
        super().__init__(*args, **kw)
        self._ranking_margin = ranking_margin
        self._margin_ranking_loss = _load_patchtst_hf_helpers().margin_ranking_loss

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs["labels"]
        outputs = model(past_values=inputs["past_values"], labels=labels)
        loss = self._margin_ranking_loss(outputs["score"], labels,
                                          margin=self._ranking_margin)
        return (loss, outputs) if return_outputs else loss


# ─── Train entrypoint ───────────────────────────────────────────────────────

def train_one(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    helpers = _load_patchtst_hf_helpers()
    panel, feat_cols = helpers.load_panel_with_split(
        Path(args.dataset), args.cut, args.label,
        val_tail_pct=getattr(args, "val_tail_pct", 0.10))
    train_ds = helpers.PerDayDataset(panel, feat_cols, args.label,
                                       args.seq_len, "train")
    val_ds = helpers.PerDayDataset(panel, feat_cols, args.label,
                                     args.seq_len, "val")
    log.info("days train=%d val=%d", len(train_ds), len(val_ds))

    model = DLinearRanker(args.seq_len, len(feat_cols),
                           kernel_size=args.kernel_size)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("DLinearRanker n_params=%.2fK seq_len=%d n_feat=%d kernel=%d",
             n_params / 1e3, args.seq_len, len(feat_cols), args.kernel_size)

    # Per-regime IC callback (same canonical infra as PatchTST)
    callbacks = []
    metric_for_best = "eval_loss"
    greater_is_better = False
    spy_path = REPO / args.spy_path
    if spy_path.exists():
        from kernel.hmm_regime_labels import compute_hmm_regime_labels  # noqa: PLC0415
        hmm_labels = compute_hmm_regime_labels(spy_path)
        callbacks.append(helpers.PerRegimeICCallback(val_ds, hmm_labels))
        metric_for_best = "eval_min_regime_ic"
        greater_is_better = True
        log.info("PerRegimeICCallback wired (n_labels=%d)", len(hmm_labels))

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    total_steps = args.epochs * max(1, len(train_ds))
    warmup_steps = int(args.warmup_ratio * total_steps)
    training_args = TrainingArguments(
        output_dir=str(out_dir / "_hf_trainer"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler,
        warmup_steps=warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        seed=args.seed, report_to=[], logging_steps=200,
        dataloader_num_workers=0, remove_unused_columns=False,
        use_cpu=(args.device == "cpu"),
    )

    trainer = DLinearTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=helpers.identity_collator, callbacks=callbacks,
        ranking_margin=args.ranking_margin,
    )
    trainer.train()
    final_metrics = trainer.evaluate()
    best_val_ic = float(final_metrics.get("eval_min_regime_ic", float("nan")))
    log.info("FINAL eval %s", {k: f"{v:+.4f}" if isinstance(v, float) else v
                                 for k, v in final_metrics.items()})

    # Dump val predictions
    device = next(model.parameters()).device
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for day in val_ds.days:
            x = day["past_values"].to(device)
            outputs = model(past_values=x)
            for i, d in enumerate(day["dates"]):
                rows.append({"date": pd.Timestamp(d),
                              "pred": float(outputs["score"][i].cpu()),
                              "label": float(day["labels"][i])})
    preds_df = pd.DataFrame(rows)
    preds_df.to_parquet(
        out_dir / f"dlinear_{args.cut}_seed{args.seed}_val_preds.parquet",
        index=False)
    log.info("preds dumped: %d rows", len(preds_df))

    summary = {
        "arch": "dlinear", "cut": args.cut, "seed": args.seed,
        "best_val_ic": best_val_ic, "n_params": n_params,
        "n_features": len(feat_cols),
        "per_regime_ic": {k.removeprefix("eval_ic_"): v
                          for k, v in final_metrics.items()
                          if k.startswith("eval_ic_")},
    }
    (out_dir / f"dlinear_{args.cut}_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--cut", default="cut1_covid")
    p.add_argument("--val-tail-pct", type=float, default=0.10)
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--seq-len", type=int, default=24,
                   help="Match Phase 2 DOE best (24); DLinear handles longer fine")
    p.add_argument("--kernel-size", type=int, default=25,
                   help="Moving-avg window for trend/seasonal decompose")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="DLinear is linear — higher LR than transformer OK")
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", default="cosine")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--ranking-margin", type=float, default=0.1)
    p.add_argument("--spy-path", default="data/ohlcv/SPY/1d.parquet")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--output-dir", default="artifacts/dlinear_baseline")
    args = p.parse_args()
    print(json.dumps(train_one(args), indent=2, default=str))


if __name__ == "__main__":
    main()
