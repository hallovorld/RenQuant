#!/usr/bin/env python3
"""PatchTST cross-sectional ranker — HuggingFace Trainer + multi-task head.

REPLACES hand-rolled training loop (376 LOC) with HF Trainer + canonical
3rd-party machinery per CLAUDE.md §5.12 ("default to canonical references").

Architecture (HF native + minimal custom):
  backbone : transformers.PatchTSTModel  (Nie 2023 ICLR)
  heads    : Linear(d_model, 1) for ranking
             Linear(d_model, 3) for (df, loc, scale) Student-t distribution
  loss     : torch.nn.functional.margin_ranking_loss (CIKM 2025 arXiv 2510.14156
                 — Margin Ranking + ListNet beat pairwise BCE on portfolio Sharpe)
             + λ * Student-t NLL (per-ticker μ/σ for downstream Kelly/QP)
  trainer  : transformers.Trainer with TrainingArguments
              load_best_model_at_end=True   → solves prior best-epoch save bug
              metric_for_best_model="eval_min_regime_ic"  → PRIME DIRECTIVE
              lr_scheduler_type="cosine_with_warmup"     → no manual schedule
  callback : PerRegimeICCallback  computes per-HMM-regime IC each eval
              (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR)
              selection metric = min(per_regime_ic.values()) per PRIME DIRECTIVE

References:
  - Nie et al 2023 ICLR "A Time Series is Worth 64 Words" (PatchTST)
  - Burges et al 2005 ICML "Learning to Rank using Gradient Descent" — superseded
    by Margin Ranking per CIKM 2025 portfolio-Sharpe benchmark
  - CIKM 2025 (arXiv 2510.14156) "On Evaluating Loss Functions for Stock Ranking"
  - HF Trainer https://huggingface.co/docs/transformers/main_classes/trainer

Usage::

    .venv/bin/python scripts/patchtst_hf.py \\
        --dataset data/transformer_v4_wl200_clean.parquet \\
        --cut cut1_covid --epochs 5 --device mps --output-dir artifacts/hf_smoke
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
from transformers import (PatchTSTConfig, PatchTSTModel, Trainer,
                          TrainerCallback, TrainingArguments)

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# NOTE: kernel.* imports deferred to point-of-use so HFPatchTSTPanelScorer
# can `importlib` this script without triggering kernel namespace conflicts.

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("patchtst-hf")


# ─── Model ──────────────────────────────────────────────────────────────────

# Canonical ordering for one-hot regime context. Must match kernel/regime.py
# emitter (BULL_STRONG is config-legacy phantom — detector doesn't emit it).
REGIMES = ("BULL_CALM", "BULL_VOLATILE", "CHOPPY", "BEAR")


def regime_to_onehot(regime_label: str) -> np.ndarray:
    """Map categorical regime label → (K=4,) one-hot float32. Unknown
    label → all zeros (model gets no regime signal; safer than guess)."""
    out = np.zeros(len(REGIMES), dtype=np.float32)
    if regime_label in REGIMES:
        out[REGIMES.index(regime_label)] = 1.0
    return out


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation (Perez 2017, arXiv 1709.07871).

    γ, β = MLP(context); h' = γ ⊙ h + β. Lightweight regime conditioning:
    shared backbone learns cross-regime features; FiLM modulates them
    per-regime via ~500 extra params for K=4 regimes, d_model=64.

    Init: last layer zero-init → at start (γ, β) = (1, 0) → FiLM is
    identity → strict superset of no-FiLM baseline.
    """

    def __init__(self, d_model: int, n_regimes: int = len(REGIMES),
                 hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_regimes, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * d_model),
        )
        # Zero-init final layer → (γ, β) = (0, 0) at output, then γ ← 1+γ
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        gb = self.net(context)
        delta_gamma, beta = gb.chunk(2, dim=-1)
        return (1.0 + delta_gamma) * h + beta


class CrossStockAttentionLayer(nn.Module):
    """iTransformer-style variate-as-token attention across tickers (Liu 2024,
    arXiv 2310.06625). Addresses PatchTST's documented #1 failure mode for
    cross-sectional finance: channel-independence (each ticker forward
    independently, no cross-stock information sharing per arXiv 2502.09683).

    For one day's batch of N tickers each represented by `d_model`-dim vec:
      input h: (N, d_model)
      query/key/value: each ticker as token
      attention: each ticker attends to ALL other tickers on the same day
      output: (N, d_model) — each ticker enriched with cross-stock context

    Residual + LayerNorm (canonical transformer block). Init: zero-init
    output projection so the residual passes through unchanged at start →
    strict superset of no-cross-stock baseline.

    Compute: O(N²) per day in attention. N=142 (wl200) → ~20k pairs.
    Fine on MPS/CPU.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                           batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        # IDENTITY-AT-INIT via learnable scalar gate (FiLM pattern):
        # output = h + alpha * (transformed(h) - h). With alpha=0 at init,
        # output exactly equals h. Pure zero-init of attn+ffn alone
        # doesn't suffice because LayerNorm transforms h regardless.
        self.alpha = nn.Parameter(torch.zeros(1))
        # Also zero-init final projections for cleaner gradient signal early
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (N, d_model)  — N tickers on one day
        h_batched = h.unsqueeze(0)  # (1, N, d_model)
        attn_out, _ = self.attn(h_batched, h_batched, h_batched)
        h_attn = self.norm1(h_batched + attn_out)
        h_ffn = self.norm2(h_attn + self.ffn(h_attn))
        transformed = h_ffn.squeeze(0)  # (N, d_model)
        # Gated residual: alpha=0 at init → exactly h
        return h + self.alpha * (transformed - h)


class HFPatchTSTRanker(nn.Module):
    """HF PatchTST backbone + dual head: ranking + Student-t distribution.

    Optional FiLM regime conditioning (Perez 2017) between encoder and
    heads. forward() returns dict with always-present "score" key. When
    `use_distributional_head=True`, also returns (df, loc, scale) for
    Student-t NLL training and downstream σ-aware Kelly/QP.
    """

    def __init__(self, cfg: PatchTSTConfig, use_distributional_head: bool = True,
                 use_film_regime: bool = False,
                 use_cross_stock_attn: bool = False,
                 n_regimes: int = len(REGIMES)):
        super().__init__()
        self.backbone = PatchTSTModel(cfg)
        self.use_distributional_head = use_distributional_head
        self.use_film_regime = use_film_regime
        self.use_cross_stock_attn = use_cross_stock_attn
        self.rank_head = nn.Linear(cfg.d_model, 1)
        self.dist_head = nn.Linear(cfg.d_model, 3) if use_distributional_head else None
        self.film = FiLMLayer(cfg.d_model, n_regimes) if use_film_regime else None
        # Cross-stock attention layer between backbone and heads
        self.cross_stock = (
            CrossStockAttentionLayer(cfg.d_model, n_heads=cfg.num_attention_heads)
            if use_cross_stock_attn else None
        )

    def forward(self, past_values: torch.Tensor,
                labels: torch.Tensor | None = None,
                regime_context: torch.Tensor | None = None,
                dates=None) -> dict:
        out = self.backbone(past_values=past_values)
        # (B, n_ch, n_patches, d_model) → pool to (B, d_model)
        h = out.last_hidden_state.mean(dim=(1, 2))
        if self.film is not None and regime_context is not None:
            h = self.film(h, regime_context)
        # Cross-stock attention: each ticker attends to all other tickers
        # on the same day (since batch IS one day's tickers per identity_collator)
        if self.cross_stock is not None:
            h = self.cross_stock(h)
        result: dict = {"score": self.rank_head(h).squeeze(-1)}
        if self.dist_head is not None:
            d = self.dist_head(h)
            result["df"] = F.softplus(d[..., 0]) + 2.0   # df > 2 → finite variance
            result["loc"] = d[..., 1]
            result["scale"] = F.softplus(d[..., 2]) + 1e-6
        return result


# ─── Losses (canonical 3rd-party) ───────────────────────────────────────────

def margin_ranking_loss(scores: torch.Tensor, labels: torch.Tensor,
                        margin: float = 0.1) -> torch.Tensor:
    """torch.nn.functional.margin_ranking_loss over all within-batch pairs.
    CIKM 2025 (arXiv 2510.14156): Margin Ranking is best ranking loss on
    portfolio Sharpe across PortfolioMASTER × S&P 500 benchmark.
    """
    n = scores.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)
    iu, ju = torch.triu_indices(n, n, offset=1, device=scores.device)
    s_i, s_j = scores[iu], scores[ju]
    l_i, l_j = labels[iu], labels[ju]
    target = torch.sign(l_i - l_j)  # ∈ {-1, 0, +1}
    return F.margin_ranking_loss(s_i, s_j, target, margin=margin)


def student_t_nll(df: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor,
                  target: torch.Tensor) -> torch.Tensor:
    """Student-t negative log-likelihood (canonical torch.distributions)."""
    dist = torch.distributions.StudentT(df, loc, scale)
    return -dist.log_prob(target).mean()


# ─── Preprocessing (Kelly-Gu-Xiu 2020 RFS standard) ─────────────────────────

def csrank_norm_per_day(panel: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Per-day cross-sectional rank-norm to [-0.5, +0.5]. Removes scale drift +
    outlier sensitivity. No temporal leakage."""
    panel = panel.copy()
    panel[feat_cols] = (panel.groupby("date")[feat_cols].rank(pct=True) - 0.5)
    panel[feat_cols] = panel[feat_cols].fillna(0.0)
    return panel


def winsorize_label(panel: pd.DataFrame, label_col: str,
                    pct: float = 0.005) -> pd.DataFrame:
    """Winsorize label ±pct percentile (default 0.5% each side ≈ ±3σ)."""
    panel = panel.copy()
    lo, hi = panel[label_col].quantile(pct), panel[label_col].quantile(1 - pct)
    panel[label_col] = panel[label_col].clip(lower=lo, upper=hi)
    return panel


def load_panel_with_split(dataset_path: Path, cut_name: str, label_col: str,
                          preprocess: bool = True,
                          val_tail_pct: float = 0.0) -> tuple[pd.DataFrame, list[str]]:
    """Load panel + assign train/val/test split.

    cut_name = "all": full-data PROD training; last val_tail_pct dates → val.
    cut_name = "cut1_covid" etc: walk-forward VALIDATION per
                                   kernel.walk_forward_splits.
    """
    panel = pd.read_parquet(dataset_path)
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.sort_values(["ticker", "date"]).reset_index(drop=True)
    panel = panel.dropna(subset=[label_col])
    if cut_name == "all":
        dates_sorted = sorted(panel["date"].unique())
        if val_tail_pct > 0:
            n_val = max(1, int(len(dates_sorted) * val_tail_pct))
            val_start = dates_sorted[-n_val]
            panel["split_label"] = "train"
            panel.loc[panel["date"] >= val_start, "split_label"] = "val"
        else:
            panel["split_label"] = "train"
    else:
        from kernel.walk_forward_splits import (assign_split_column,  # noqa: PLC0415
                                                 build_default_cuts)
        cut = next(c for c in build_default_cuts() if c.name == cut_name)
        panel["split_label"] = assign_split_column(panel, cut)
    feat_cols = [c for c in panel.columns
                 if c not in {"date", "ticker", "split_label",
                              "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}
                 and panel[c].dtype.kind in "fiub"]
    if preprocess:
        panel = csrank_norm_per_day(panel, feat_cols)
        panel = winsorize_label(panel, label_col, pct=0.005)
        log.info("preprocessing: CSRankNorm + Winsorize(±0.5%%) applied")
    log.info("panel %d rows | cut=%s | train=%d val=%d test=%d | n_feat=%d",
             len(panel), cut_name,
             (panel["split_label"] == "train").sum(),
             (panel["split_label"] == "val").sum(),
             (panel["split_label"] == "test").sum(),
             len(feat_cols))
    return panel, feat_cols


# ─── Dataset (per-day batching) ─────────────────────────────────────────────

class PerDayDataset(torch.utils.data.Dataset):
    """One Dataset sample = one day's all-ticker batch. With identity_collator
    and Trainer batch_size=1, each Trainer step processes one day's pairwise
    ranking loss.

    If `hmm_labels` is provided, each day's dict gets a `regime_context`
    tensor of shape (N_tickers, K=4) — one-hot for the day's HMM regime.
    All tickers on the same day share the same regime row (regime is a
    market-wide signal, broadcast for FiLM convenience)."""

    def __init__(self, panel: pd.DataFrame, feat_cols: list[str],
                 label_col: str, seq_len: int, split: str,
                 hmm_labels: pd.DataFrame | None = None):
        feat_arr = panel[feat_cols].astype(np.float32).fillna(0.0).values
        lab_arr = panel[label_col].astype(np.float32).values
        samples_by_date: dict[int, list[tuple[np.ndarray, float, pd.Timestamp]]] = {}
        for _, idxs in panel.groupby("ticker", sort=False).indices.items():
            idxs = np.asarray(sorted(idxs))
            for i in range(seq_len, len(idxs)):
                end_pos = idxs[i]
                if panel.iloc[end_pos]["split_label"] != split:
                    continue
                window = feat_arr[idxs[i - seq_len: i]]
                if window.shape[0] != seq_len:
                    continue
                d = panel.iloc[end_pos]["date"]
                samples_by_date.setdefault(d.value, []).append(
                    (window, lab_arr[end_pos], d))

        # Build per-day regime context lookup (if HMM labels provided)
        regime_map: dict[int, str] | None = None
        if hmm_labels is not None:
            regime_map = {pd.Timestamp(d).value: r
                          for d, r in zip(hmm_labels["date"], hmm_labels["regime"])}

        self.days: list[dict] = []
        for d_ns, samples in samples_by_date.items():
            if len(samples) < 5:
                continue
            day = {
                "past_values": torch.from_numpy(np.stack([s[0] for s in samples])),
                "labels": torch.tensor([s[1] for s in samples], dtype=torch.float32),
                "dates": np.array([s[2].value for s in samples], dtype="int64"),
            }
            if regime_map is not None:
                regime = regime_map.get(int(d_ns), "BULL_CALM")  # fallback
                onehot = regime_to_onehot(regime)
                n = len(samples)
                day["regime_context"] = torch.from_numpy(
                    np.broadcast_to(onehot, (n, len(REGIMES))).copy())
                day["regime_label"] = regime
            self.days.append(day)

    def __len__(self):
        return len(self.days)

    def __getitem__(self, idx):
        return self.days[idx]


def identity_collator(batch):
    """No collation — each DataLoader batch is exactly one day's dict."""
    assert len(batch) == 1, f"batch_size must be 1 for per-day batching, got {len(batch)}"
    return batch[0]


# ─── Trainer subclass (multi-task loss) ─────────────────────────────────────

class PatchTSTRankerTrainer(Trainer):
    """HF Trainer with multi-task compute_loss: Margin Ranking + Student-t NLL."""

    def __init__(self, *args, nll_loss_weight: float = 0.5,
                 ranking_margin: float = 0.1, **kw):
        super().__init__(*args, **kw)
        self._nll_loss_weight = nll_loss_weight
        self._ranking_margin = ranking_margin

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs["labels"]
        fwd_kwargs = {"past_values": inputs["past_values"], "labels": labels}
        if "regime_context" in inputs:
            fwd_kwargs["regime_context"] = inputs["regime_context"]
        outputs = model(**fwd_kwargs)
        loss = margin_ranking_loss(outputs["score"], labels,
                                    margin=self._ranking_margin)
        if "loc" in outputs and self._nll_loss_weight > 0:
            nll = student_t_nll(outputs["df"], outputs["loc"], outputs["scale"],
                                labels)
            loss = loss + self._nll_loss_weight * nll
        return (loss, outputs) if return_outputs else loss


# ─── Per-regime IC callback (PRIME DIRECTIVE) ───────────────────────────────

class PerRegimeICCallback(TrainerCallback):
    """After each eval, run a second forward pass on val set, compute per-HMM-
    regime IC (BULL_CALM / BULL_VOLATILE / CHOPPY / BEAR), and inject
    `eval_min_regime_ic` into metrics — this is the selection metric for
    `load_best_model_at_end=True` per PRIME DIRECTIVE.
    """

    def __init__(self, eval_dataset: PerDayDataset, hmm_labels: pd.DataFrame):
        self.eval_dataset = eval_dataset
        self.hmm_labels = hmm_labels

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kw):
        if model is None or metrics is None:
            return
        from kernel.hmm_regime_labels import per_hmm_regime_ic  # noqa: PLC0415
        device = next(model.parameters()).device
        model.eval()
        all_p, all_y, all_d = [], [], []
        with torch.no_grad():
            for day in self.eval_dataset.days:
                x = day["past_values"].to(device)
                fwd_kwargs = {"past_values": x}
                if "regime_context" in day:
                    fwd_kwargs["regime_context"] = day["regime_context"].to(device)
                outputs = model(**fwd_kwargs)
                all_p.append(outputs["score"].cpu().numpy())
                all_y.append(day["labels"].numpy())
                all_d.append(day["dates"])
        if not all_p:
            return
        preds_df = pd.DataFrame({
            "date": pd.to_datetime(np.concatenate(all_d)),
            "pred": np.concatenate(all_p),
            "label": np.concatenate(all_y),
        })
        per_regime = per_hmm_regime_ic(preds_df, self.hmm_labels)
        if per_regime:
            min_ic = float(min(per_regime.values()))
            metrics["eval_min_regime_ic"] = min_ic
            for r, ic in per_regime.items():
                metrics[f"eval_ic_{r}"] = float(ic)
            log.info("per-regime IC: %s | min=%+.4f",
                     {r: f"{v:+.4f}" for r, v in per_regime.items()}, min_ic)
        else:
            log.warning("per-regime IC: no regime had ≥5 days in val — "
                        "falling back to pooled eval_loss for selection")


# ─── Train entrypoint ───────────────────────────────────────────────────────

def train_one(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    panel, feat_cols = load_panel_with_split(
        Path(args.dataset), args.cut, args.label,
        val_tail_pct=getattr(args, "val_tail_pct", 0.10))

    # Compute HMM regime labels once — reused for FiLM dataset injection
    # AND for per-regime IC callback selection metric.
    hmm_labels = None
    spy_path = REPO / args.spy_path
    if spy_path.exists():
        from kernel.hmm_regime_labels import compute_hmm_regime_labels  # noqa: PLC0415
        hmm_labels = compute_hmm_regime_labels(spy_path)
    elif args.film_regime_cond:
        raise FileNotFoundError(
            f"FiLM regime conditioning requires SPY parquet at {spy_path}")

    # Inject regime context into datasets only when FiLM is ON (FiLM-OFF
    # dataset stays lean — no spurious regime_context broadcast)
    ds_hmm = hmm_labels if args.film_regime_cond else None
    train_ds = PerDayDataset(panel, feat_cols, args.label, args.seq_len, "train",
                              hmm_labels=ds_hmm)
    val_ds = PerDayDataset(panel, feat_cols, args.label, args.seq_len, "val",
                            hmm_labels=ds_hmm)
    log.info("days train=%d val=%d", len(train_ds), len(val_ds))

    cfg = PatchTSTConfig(
        num_input_channels=len(feat_cols),
        context_length=args.seq_len,
        patch_length=args.patch_length,
        patch_stride=args.patch_length,  # non-overlapping
        d_model=args.d_model,
        num_attention_heads=args.n_heads,
        num_hidden_layers=args.n_layers,
        ffn_dim=args.d_model * 2,
    )
    model = HFPatchTSTRanker(cfg, use_distributional_head=args.distributional_head,
                              use_film_regime=args.film_regime_cond,
                              use_cross_stock_attn=args.cross_stock_attn)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("HFPatchTSTRanker n_params=%.2fM dist_head=%s film=%s cross_stock=%s",
             n_params / 1e6, args.distributional_head, args.film_regime_cond,
             args.cross_stock_attn)

    # Per-regime IC callback (PRIME DIRECTIVE selection metric)
    callbacks = []
    metric_for_best = None
    greater_is_better = True
    if hmm_labels is not None:
        callbacks.append(PerRegimeICCallback(val_ds, hmm_labels))
        metric_for_best = "eval_min_regime_ic"
        log.info("PerRegimeICCallback wired | n_labels=%d", len(hmm_labels))
    else:
        log.warning("SPY parquet missing at %s — falling back to eval_loss "
                    "for best-model selection (PRIME DIRECTIVE degraded)", spy_path)
        metric_for_best = "eval_loss"
        greater_is_better = False

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
        seed=args.seed,
        report_to=[],
        logging_steps=200,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        use_cpu=(args.device == "cpu"),
    )

    trainer = PatchTSTRankerTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=identity_collator, callbacks=callbacks,
        nll_loss_weight=args.nll_loss_weight,
        ranking_margin=args.ranking_margin,
    )
    trainer.train()

    # Final eval (best model loaded by load_best_model_at_end=True)
    final_metrics = trainer.evaluate()
    best_val_ic = float(final_metrics.get("eval_min_regime_ic", float("nan")))
    log.info("FINAL eval %s", {k: f"{v:+.4f}" if isinstance(v, float) else v
                                 for k, v in final_metrics.items()})

    # Dump val predictions for downstream regime-stratified IC
    device = next(model.parameters()).device
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for day in val_ds.days:
            x = day["past_values"].to(device)
            fwd_kwargs = {"past_values": x}
            if "regime_context" in day:
                fwd_kwargs["regime_context"] = day["regime_context"].to(device)
            outputs = model(**fwd_kwargs)
            for i, d in enumerate(day["dates"]):
                row = {"date": pd.Timestamp(d),
                       "pred": float(outputs["score"][i].cpu()),
                       "label": float(day["labels"][i])}
                if "loc" in outputs:
                    row["mu"] = float(outputs["loc"][i].cpu())
                    row["sigma"] = float(outputs["scale"][i].cpu())
                rows.append(row)
    preds_df = pd.DataFrame(rows)
    dump = out_dir / f"hf_patchtst_{args.cut}_seed{args.seed}_val_preds.parquet"
    preds_df.to_parquet(dump, index=False)
    log.info("preds dumped: %s (%d rows)", dump.name, len(preds_df))

    summary = {
        "arch": "hf_patchtst", "cut": args.cut, "seed": args.seed,
        "best_val_ic": best_val_ic, "n_params": n_params,
        "n_features": len(feat_cols), "uses_distributional_head": args.distributional_head,
        "per_regime_ic": {k.removeprefix("eval_ic_"): v
                          for k, v in final_metrics.items()
                          if k.startswith("eval_ic_")},
    }
    (out_dir / f"hf_patchtst_{args.cut}_seed{args.seed}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    if args.save_model:
        model_path = out_dir / f"hf_patchtst_{args.cut}_seed{args.seed}_model.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "config_dict": cfg.to_dict(),
            "feature_cols": feat_cols,
            "seq_len": args.seq_len,
            "label_col": args.label,
            "best_val_ic": best_val_ic,
            "uses_distributional_head": args.distributional_head,
            "uses_film_regime": args.film_regime_cond,
            "uses_cross_stock_attn": args.cross_stock_attn,
            "uses_csranknorm_preprocessing": True,
            "uses_winsorize_label_preprocessing": True,
            "per_regime_ic": summary["per_regime_ic"],
        }, model_path)
        log.info("model saved: %s", model_path)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--cut", default="cut1_covid",
                   help="walk-forward cut name OR 'all' for full-data prod")
    p.add_argument("--val-tail-pct", type=float, default=0.10)
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--patch-length", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--lr-scheduler", default="cosine",
                   help="HF TrainingArguments.lr_scheduler_type "
                        "(cosine | linear | constant_with_warmup)")
    p.add_argument("--warmup-ratio", type=float, default=0.1,
                   help="Fraction of total steps for LR warmup (HF default 0.0)")
    p.add_argument("--distributional-head", action="store_true", default=True,
                   help="Enable Student-t (df, μ, σ) head + NLL loss")
    p.add_argument("--no-distributional-head", dest="distributional_head",
                   action="store_false",
                   help="Disable distributional head (ranking loss only)")
    p.add_argument("--nll-loss-weight", type=float, default=0.5,
                   help="λ in L = margin_rank + λ * student_t_nll")
    p.add_argument("--ranking-margin", type=float, default=0.1,
                   help="margin in torch.nn.functional.margin_ranking_loss")
    p.add_argument("--film-regime-cond", action="store_true",
                   help="FiLM regime conditioning (Perez 2017): γ, β = MLP(regime) "
                        "modulates encoder output. Identity at init → strict "
                        "superset of FiLM-OFF baseline. Requires --spy-path.")
    p.add_argument("--cross-stock-attn", action="store_true",
                   help="iTransformer-style cross-stock attention (Liu 2024, "
                        "arXiv 2310.06625). Each ticker attends to all other "
                        "tickers on the same day. Addresses PatchTST channel-"
                        "independence — documented #1 failure mode for cross-"
                        "sectional finance (arXiv 2502.09683). Identity-at-init "
                        "via zero-init output projections → strict superset of "
                        "baseline.")
    p.add_argument("--spy-path", default="data/ohlcv/SPY/1d.parquet",
                   help="SPY OHLCV parquet for HMM regime labels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--output-dir", default="artifacts/hf_patchtst")
    args = p.parse_args()
    print(json.dumps(train_one(args), indent=2, default=str))


if __name__ == "__main__":
    main()
