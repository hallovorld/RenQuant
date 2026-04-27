"""Asset embeddings — Dolphin et al. 2024 KDD contrastive features.

T2-2 from doc/roadmap.md (Tier 2). Per-asset 16-D embedding vectors
trained via pairwise-correlation contrastive learning on watchlist
OHLCV history. Used as additional per-ticker features in panel-LTR
(NOT broadcast — each ticker gets its own learned embedding, so
within-date variance is preserved for cross-sectional rank loss).

Reference: Dolphin, R., Smyth, B., Dong, R. (2024) "Contrastive
Learning of Asset Embeddings from Financial Time Series", arXiv
2407.18645. Reported gains: +3 F1 sector classification, -19% vol
on hedging tasks.

Public API
==========

`AssetEmbeddingTrainer`
    .__init__(embedding_dim=16, lookback_days=504, ...)
    .fit(ohlcv: dict[ticker, df], as_of_date) -> dict[ticker, np.ndarray]
    .smoke_test_collapse(embeddings) -> bool

Used by `scripts/train_asset_embeddings.py` (weekly cron).

Design
======

For each asset, we generate (anchor, positive, negative) triplets
where:
- anchor = asset's most recent `lookback_days` of returns
- positive = same asset, sliding window from earlier history
- negative = a sampled OTHER asset from the watchlist

A small temporal CNN encoder maps each (T,) returns time series to
a D-dim embedding. InfoNCE loss pulls (anchor, positive) close and
pushes (anchor, negative) apart.

After training, forward-pass each asset once to get its embedding;
persist to `artifacts/asset-embeddings.json`.

Status: skeleton (T+0). Trainer not yet hooked into FullTrainingPipeline.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger("training_panel.asset_embeddings")


@dataclass
class AssetEmbeddingTrainer:
    """Train D-dim contrastive embeddings on watchlist OHLCV history.

    Default config (matches paper's 'Compact' setting):
    - embedding_dim: 16
    - lookback_days: 504 (~2 years of daily bars)
    - encoder_hidden: 64
    - n_epochs: 30
    - batch_size: 64
    - lr: 1e-3
    - margin: 0.2 (for triplet loss; InfoNCE if temperature given)
    """
    embedding_dim:    int = 16
    lookback_days:    int = 504
    encoder_hidden:   int = 64
    n_epochs:         int = 30
    batch_size:       int = 64
    lr:               float = 1e-3
    margin:           float = 0.2
    temperature:      float = 0.1
    negative_pool:    int = 50
    min_corr_threshold: float = 0.3   # negatives have abs(corr) < this
    seed:             int = 42

    # ── Trainer state (filled by .fit) ──
    embeddings: dict[str, np.ndarray] = field(default_factory=dict)
    loss_history: list[float] = field(default_factory=list)
    trained_date: str | None = None

    def fit(
        self,
        ohlcv: dict[str, pd.DataFrame],
        as_of_date: pd.Timestamp,
    ) -> dict[str, np.ndarray]:
        """Train embeddings on watchlist OHLCV up to as_of_date.

        Strict-prior discipline: only data ≤ as_of_date used. The
        lookback window for each ticker is the most recent
        `lookback_days` bars on-or-before as_of_date.

        Returns dict[ticker, np.ndarray of shape (embedding_dim,)].

        Raises RuntimeError if torch is unavailable. Returns {} when
        no ticker has enough history.
        """
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
        except ImportError as exc:
            raise RuntimeError(
                "AssetEmbeddingTrainer requires torch — install with "
                "`pip install torch` or use a different feature path."
            ) from exc

        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)

        # ── Step 1: collect per-ticker returns windows ──
        windows: dict[str, np.ndarray] = {}
        for ticker, df in ohlcv.items():
            if df is None or df.empty or "close" not in df.columns:
                continue
            close = df["close"].astype(float)
            close = close.loc[close.index <= as_of_date]
            if len(close) < self.lookback_days + 30:  # need extra for positives
                continue
            ret = close.pct_change().dropna().values[-self.lookback_days:]
            if len(ret) < self.lookback_days:
                continue
            windows[ticker] = ret.astype(np.float32)

        if len(windows) < 2:
            log.warning("AssetEmbeddingTrainer.fit: only %d tickers with "
                        "enough history — returning empty embeddings",
                        len(windows))
            return {}

        # ── Step 2: precompute correlations for negative-sampling pool ──
        ticker_list = list(windows.keys())
        n = len(ticker_list)
        ret_matrix = np.stack([windows[t] for t in ticker_list], axis=0)  # (n, T)
        # Per-pair Pearson on whatever overlap (here all are same length)
        corr_matrix = np.corrcoef(ret_matrix)  # (n, n)
        # For each ticker, negative candidates = those with abs(corr) < threshold
        neg_pool: dict[str, list[int]] = {}
        for i, t in enumerate(ticker_list):
            mask = (np.abs(corr_matrix[i]) < self.min_corr_threshold)
            mask[i] = False
            candidates = np.where(mask)[0].tolist()
            if len(candidates) < 1:
                # Fallback: pick 5 lowest |corr|
                ord_idx = np.argsort(np.abs(corr_matrix[i]))
                candidates = [j for j in ord_idx if j != i][:5]
            neg_pool[t] = candidates[:self.negative_pool]

        # ── Step 3: build encoder model (small temporal CNN) ──
        class _TempCNN(nn.Module):
            def __init__(self, d_out: int, hidden: int, T: int):
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(1, hidden // 4, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.Conv1d(hidden // 4, hidden // 2, kernel_size=7, padding=3),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool1d(8),
                    nn.Flatten(),
                    nn.Linear((hidden // 2) * 8, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, d_out),
                )

            def forward(self, x):  # x: (B, T)
                x = x.unsqueeze(1)  # (B, 1, T)
                z = self.conv(x)
                return torch.nn.functional.normalize(z, dim=-1)

        device = torch.device("cpu")  # CPU is fine for ~100 tickers × small model
        model = _TempCNN(self.embedding_dim, self.encoder_hidden,
                         self.lookback_days).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # ── Step 4: training loop with InfoNCE loss ──
        ret_tensor = torch.tensor(ret_matrix, device=device, dtype=torch.float32)

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            shuffled = rng.permutation(n)
            for start in range(0, n, self.batch_size):
                batch_idx = shuffled[start:start + self.batch_size]
                if len(batch_idx) < 2:
                    continue
                anchors = ret_tensor[batch_idx]  # (B, T)

                # Positives: shifted window of same ticker (lookback - 30 to -lookback)
                # We don't have separate windows; for skeleton, augment via slight noise
                # — paper uses sliding-window from earlier history. Skeleton uses
                # input + Gaussian noise (σ=0.001) as positive (light augmentation).
                pos_noise = torch.randn_like(anchors) * 0.001
                positives = anchors + pos_noise

                # Negatives: sample from neg_pool
                neg_idx = []
                for i_local, i_global in enumerate(batch_idx):
                    pool = neg_pool[ticker_list[int(i_global)]]
                    if pool:
                        neg_idx.append(int(rng.choice(pool)))
                    else:
                        neg_idx.append(int(i_global))  # degenerate fallback
                negatives = ret_tensor[neg_idx]

                # Embed
                z_a = model(anchors)
                z_p = model(positives)
                z_n = model(negatives)

                # InfoNCE loss
                pos_sim = (z_a * z_p).sum(dim=-1) / self.temperature
                neg_sim = (z_a * z_n).sum(dim=-1) / self.temperature
                # Loss = -log(exp(pos) / (exp(pos) + exp(neg)))
                logits = torch.stack([pos_sim, neg_sim], dim=-1)
                labels = torch.zeros(len(batch_idx), dtype=torch.long, device=device)
                loss = nn.functional.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(1, n_batches)
            self.loss_history.append(avg_loss)
            if (epoch + 1) % 5 == 0:
                log.info("AssetEmbeddingTrainer epoch %d/%d  loss=%.4f",
                         epoch + 1, self.n_epochs, avg_loss)

        # ── Step 5: forward-pass to get final embeddings ──
        model.eval()
        with torch.no_grad():
            embeddings_tensor = model(ret_tensor)  # (n, embedding_dim)

        embeddings_np = embeddings_tensor.cpu().numpy()
        for i, ticker in enumerate(ticker_list):
            self.embeddings[ticker] = embeddings_np[i]
        self.trained_date = pd.Timestamp(as_of_date).date().isoformat()

        log.info("AssetEmbeddingTrainer.fit: trained %d-D embeddings for "
                 "%d tickers, final loss=%.4f", self.embedding_dim,
                 len(self.embeddings), self.loss_history[-1] if self.loss_history else float("nan"))
        return self.embeddings

    def smoke_test_collapse(self, embeddings: dict[str, np.ndarray] | None = None) -> bool:
        """Detect collapsed embeddings (all tickers map to ~same vector).

        Returns True if embeddings look healthy (sufficient diversity),
        False if collapsed (mean pairwise cosine > 0.95 → reject).
        """
        emb = embeddings or self.embeddings
        if len(emb) < 2:
            return False
        mat = np.stack(list(emb.values()), axis=0)  # (n, d)
        # Cosine similarity matrix
        norms = np.linalg.norm(mat, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        normalized = mat / norms
        cos = normalized @ normalized.T
        n = len(mat)
        # Off-diagonal mean
        mask = ~np.eye(n, dtype=bool)
        mean_cos = float(cos[mask].mean())
        log.info("smoke_test_collapse: mean off-diagonal cosine = %.3f "
                 "(reject if > 0.95)", mean_cos)
        return mean_cos < 0.95

    def save(self, path: Path | str) -> None:
        """Persist embeddings + metadata to a JSON artifact."""
        path = Path(path)
        payload: dict[str, Any] = {
            "version":        1,
            "kind":           "asset_embeddings",
            "trained_date":   self.trained_date,
            "embedding_dim":  self.embedding_dim,
            "lookback_days":  self.lookback_days,
            "n_tickers":      len(self.embeddings),
            "embeddings":     {t: e.tolist() for t, e in self.embeddings.items()},
            "loss_history":   self.loss_history,
            "params": {
                "encoder_hidden":      self.encoder_hidden,
                "n_epochs":            self.n_epochs,
                "batch_size":          self.batch_size,
                "lr":                  self.lr,
                "margin":              self.margin,
                "temperature":         self.temperature,
                "negative_pool":       self.negative_pool,
                "min_corr_threshold":  self.min_corr_threshold,
                "seed":                self.seed,
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path | str) -> "AssetEmbeddingTrainer":
        path = Path(path)
        d = json.loads(path.read_text())
        if d.get("kind") != "asset_embeddings":
            raise ValueError(f"AssetEmbeddingTrainer.load: not an asset_embeddings "
                             f"artifact at {path} (kind={d.get('kind')!r})")
        params = d.get("params", {})
        trainer = cls(
            embedding_dim    = int(d.get("embedding_dim", 16)),
            lookback_days    = int(d.get("lookback_days", 504)),
            encoder_hidden   = int(params.get("encoder_hidden", 64)),
            n_epochs         = int(params.get("n_epochs", 30)),
            batch_size       = int(params.get("batch_size", 64)),
            lr               = float(params.get("lr", 1e-3)),
            margin           = float(params.get("margin", 0.2)),
            temperature      = float(params.get("temperature", 0.1)),
            negative_pool    = int(params.get("negative_pool", 50)),
            min_corr_threshold = float(params.get("min_corr_threshold", 0.3)),
            seed             = int(params.get("seed", 42)),
        )
        trainer.embeddings = {t: np.array(v, dtype=np.float32)
                              for t, v in d.get("embeddings", {}).items()}
        trainer.trained_date = d.get("trained_date")
        trainer.loss_history = list(d.get("loss_history", []))
        return trainer


def load_embeddings_for_inference(
    artifact_path: Path | str,
    max_age_days: int = 14,
) -> dict[str, np.ndarray]:
    """Inference-side helper. Loads trained embeddings + warns if stale.

    Returns {} if the artifact is missing — caller treats as no-feature
    case (skip embedding columns in panel matrix). Logs warning if
    artifact age exceeds `max_age_days`.
    """
    path = Path(artifact_path)
    if not path.exists():
        return {}
    trainer = AssetEmbeddingTrainer.load(path)
    if trainer.trained_date:
        try:
            age_days = (
                pd.Timestamp.now(tz="UTC").date()
                - pd.Timestamp(trainer.trained_date).date()
            ).days
            if age_days > max_age_days:
                log.warning("asset embeddings stale: %d days old (>%d threshold). "
                            "Re-run scripts/train_asset_embeddings.py to refresh.",
                            age_days, max_age_days)
        except Exception:
            pass
    return trainer.embeddings


__all__ = [
    "AssetEmbeddingTrainer",
    "load_embeddings_for_inference",
]
