"""Phase C — Graph Attention panel scorer (Feng et al. 2019 TGC).

Alternative ranking backend to PanelLTRModel (XGBoost) and
PanelTransformerModel. Layer 3 of the design-v2 sector-aware
architecture: when same-sector tickers should attend to each other
(pair-level relation), tree models can't encode that — they split on
features. Graph attention does.

Reference
---------
Feng, F., Chen, X., He, X., Yang, S., Cao, Y. (2019). "Temporal
Relational Ranking for Stock Prediction". TOIS, arXiv:1809.09441.
OSS: github.com/fulifeng/Temporal_Relational_Stock_Ranking

Status (2026-05-02)
-------------------
**Scaffold only — training loop NOT shipped.** Cloud GPU integration
(doc/research/cloud-gpu-training-plan.md) hasn't been set up; M2 Pro
MPS has known PyTorch gaps that make NN training unreliable. This
module ships the FORWARD pass + sector-mask correctness + save/load
roundtrip with full test coverage so that, when GPU comes online,
.train() is the only thing left to fill in.

Public API mirrors PanelLTRModel / PanelTransformerModel for drop-in
compatibility — adapter and inference paths can call .score(X) the
same way regardless of backend.

Architecture
------------
                features (T tickers × D features)
                              ↓
                Per-ticker linear encoder (D → H)
                              ↓
                Sector graph attention (T-aware, masked to same sector)
                  h'_i = σ( W_self·h_i + Σ_{j ∈ same_sector(i)} α_ij·W_neigh·h_j )
                              ↓
                Score head (H → 1) per ticker
                              ↓
                Cross-sectional rank loss (training-time)

Sector graph is recomputed each batch from `ticker_sectors`, so
adding/removing tickers between train and inference is safe.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hyperparameters dataclass ─────────────────────────────────────────────────

@dataclass
class GraphAttentionParams:
    """All hyperparameters for the graph-attention scorer.

    Defaults sized for our wl178/wl500 panel (small-to-mid). For
    transformer-tier (wl500+ with 200k+ rows) bump hidden to 64 and
    attention heads to 4.
    """
    n_features:        int = 27
    hidden_dim:        int = 32
    attention_heads:   int = 2
    dropout_p:         float = 0.1
    score_temperature: float = 1.0
    learning_rate:     float = 1e-3
    weight_decay:      float = 1e-5
    seed:              int   = 42


# ── Core nn.Module — the actual GAT panel ────────────────────────────────────

class _PanelGraphAttention(nn.Module):
    """One-layer graph attention over the per-date ticker cross-section,
    with sector-co-membership as the edge mask.

    Forward inputs
    --------------
    x : Tensor of shape (T, D)
        Per-ticker feature row for ONE date. T = number of tickers in
        the cross-section (varies bar-to-bar). D = n_features.

    sector_ids : Tensor of shape (T,) dtype int64
        Integer sector encoding (0..n_sectors-1). Tickers with the same
        sector_id attend to each other; cross-sector attention is masked.

    Forward output
    --------------
    scores : Tensor of shape (T,)
        Predicted score per ticker. Caller applies cross-sectional rank
        loss against gaussianized labels.

    Invariant
    ---------
    For any pair (i, j) where ``sector_ids[i] != sector_ids[j]``, the
    attention weight α_ij is exactly 0 (post-softmax). The self-edge
    α_ii is always present (every ticker attends to itself, even if
    its sector is otherwise unpopulated).
    """

    def __init__(self, params: GraphAttentionParams):
        super().__init__()
        self.params = params
        torch.manual_seed(params.seed)

        H = params.hidden_dim
        self.encoder = nn.Sequential(
            nn.Linear(params.n_features, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Dropout(params.dropout_p),
        )

        # Multi-head attention components
        self.heads = params.attention_heads
        if H % self.heads != 0:
            raise ValueError(
                f"hidden_dim={H} must be divisible by attention_heads="
                f"{self.heads}",
            )
        self.head_dim = H // self.heads
        self.q_proj = nn.Linear(H, H, bias=False)
        self.k_proj = nn.Linear(H, H, bias=False)
        self.v_proj = nn.Linear(H, H, bias=False)
        self.o_proj = nn.Linear(H, H, bias=False)

        # Score head — single scalar per ticker
        self.score_head = nn.Sequential(
            nn.LayerNorm(H),
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Dropout(params.dropout_p),
            nn.Linear(H // 2, 1),
        )

        # Initialize linear layers with Xavier — helps gradient flow in
        # attention (PyTorch default for Linear is Kaiming uniform which
        # tends to over-saturate softmax in early epochs).
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _sector_mask(self, sector_ids: torch.Tensor) -> torch.Tensor:
        """Build the (T, T) attention mask.

        Returns a tensor where True = ALLOWED to attend, False = MASKED.
        Self-edges always True; cross-sector edges always False;
        within-sector edges True.
        """
        # Outer equality: M[i, j] = (sector_ids[i] == sector_ids[j])
        return sector_ids.unsqueeze(0) == sector_ids.unsqueeze(1)

    def forward(
        self,
        x: torch.Tensor,
        sector_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"x must be (T, D), got shape {tuple(x.shape)}")
        if sector_ids.dim() != 1 or sector_ids.shape[0] != x.shape[0]:
            raise ValueError(
                f"sector_ids must be (T,) matching x's first dim "
                f"(T={x.shape[0]}); got shape {tuple(sector_ids.shape)}"
            )

        T = x.shape[0]
        H = self.params.hidden_dim
        nh = self.heads
        hd = self.head_dim

        # Encode
        h = self.encoder(x)                              # (T, H)

        # Multi-head Q/K/V → reshape to (heads, T, head_dim)
        q = self.q_proj(h).view(T, nh, hd).transpose(0, 1)   # (heads, T, hd)
        k = self.k_proj(h).view(T, nh, hd).transpose(0, 1)
        v = self.v_proj(h).view(T, nh, hd).transpose(0, 1)

        # Scaled dot-product attention per head
        # attn_logits: (heads, T, T)
        attn_logits = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd)

        # Sector mask — broadcast to (heads, T, T)
        mask = self._sector_mask(sector_ids).to(attn_logits.device)
        # Mask: True = keep, False = -inf
        attn_logits = attn_logits.masked_fill(
            ~mask.unsqueeze(0), float("-inf"),
        )
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = F.dropout(
            attn_weights, p=self.params.dropout_p, training=self.training,
        )

        # Apply attention: (heads, T, hd) → concat heads → (T, H)
        out = torch.matmul(attn_weights, v)                # (heads, T, hd)
        out = out.transpose(0, 1).contiguous().view(T, H)  # (T, H)
        out = self.o_proj(out)                             # (T, H)

        # Residual + layer-norm fold
        out = out + h

        # Score
        scores = self.score_head(out).squeeze(-1)          # (T,)
        if self.params.score_temperature != 1.0:
            scores = scores / self.params.score_temperature
        return scores


# ── Wrapper — public API matches PanelLTRModel / PanelTransformerModel ────────

class PanelGraphAttentionModel:
    """Drop-in panel ranker using sector-conditioned graph attention.

    Public API
    ----------
    .train(panel_df, ticker_sectors, label_col, ...) — NOT YET WIRED.
        Will be implemented when cloud GPU integration ships
        (cloud-gpu-training-plan.md). Stub raises NotImplementedError
        with a clear message pointing the operator at the design doc.

    .score(X, ticker_sectors) — Forward pass on inference matrix.
        X: pd.DataFrame indexed by ticker, columns = feature_cols.
        ticker_sectors: dict[ticker, sector_name].

    .save(path) / .load(path) — JSON state-dict + hyperparams.

    .feature_cols — list[str] of feature column names the model expects.
    """

    def __init__(self, params: dict | None = None):
        valid = set(GraphAttentionParams.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in (params or {}).items() if k in valid}
        self.params = GraphAttentionParams(**kwargs)
        self._module: _PanelGraphAttention | None = None
        self._feature_cols: list[str] = []
        # Stable sector → int mapping; rebuilt on train, frozen for inference
        self._sector_vocab: dict[str, int] = {}

    # ── Training stub ─────────────────────────────────────────────────────────

    def train(self, *args, **kwargs):  # noqa: D401
        """NOT YET WIRED — see doc/research/cloud-gpu-training-plan.md.

        This stub fails loud rather than silently no-op so a caller that
        accidentally invokes the un-wired training path immediately
        knows the backend isn't ready (vs. mysterious zero-IC artifacts
        from a half-trained model).
        """
        raise NotImplementedError(
            "PanelGraphAttentionModel.train() is scaffolded but not wired. "
            "Training requires cloud GPU integration — see "
            "doc/research/cloud-gpu-training-plan.md. The forward pass + "
            "save/load are tested and work on CPU; only the training loop "
            "(epoch schedule / gradient clipping / LR warmup / mixed "
            "precision) waits on Phase G dispatch infra."
        )

    # ── Inference path (works on CPU) ─────────────────────────────────────────

    def score(
        self,
        X,                                 # pd.DataFrame, ticker-indexed
        ticker_sectors: dict[str, str],
    ):
        """Forward pass on inference matrix. Returns pd.Series indexed
        by ticker."""
        import pandas as pd  # noqa: PLC0415
        if self._module is None:
            raise RuntimeError(
                "PanelGraphAttentionModel.score() called before .load() — "
                "no trained module to evaluate"
            )
        # Slice + reorder X to feature_cols, missing → NaN → 0 (defensive)
        feats = X.reindex(columns=self._feature_cols).fillna(0.0).astype(float)
        tickers = list(feats.index)
        sector_ids = torch.tensor(
            [self._sector_vocab.get(ticker_sectors.get(t, "_unmapped"), 0)
             for t in tickers],
            dtype=torch.int64,
        )
        x = torch.tensor(feats.values, dtype=torch.float32)
        self._module.eval()
        with torch.no_grad():
            scores = self._module(x, sector_ids).cpu().numpy()
        return pd.Series(scores, index=tickers, name="panel_score")

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write JSON: {params, sector_vocab, feature_cols, state_dict_b64}."""
        import base64
        import io
        import torch as _torch  # noqa: PLC0415

        if self._module is None:
            raise RuntimeError("Nothing to save — module not initialized")
        buf = io.BytesIO()
        _torch.save(self._module.state_dict(), buf)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        payload = {
            "schema_version":  1,
            "kind":            "PanelGraphAttentionModel",
            "params":          asdict(self.params),
            "sector_vocab":    self._sector_vocab,
            "feature_cols":    self._feature_cols,
            "state_dict_b64":  b64,
        }
        Path(path).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PanelGraphAttentionModel":
        import base64
        import io
        import torch as _torch  # noqa: PLC0415

        payload = json.loads(Path(path).read_text())
        if payload.get("kind") != "PanelGraphAttentionModel":
            raise ValueError(
                f"Artifact kind mismatch: expected "
                f"'PanelGraphAttentionModel', got {payload.get('kind')!r}"
            )
        m = cls(params=payload.get("params") or {})
        m._sector_vocab = dict(payload.get("sector_vocab") or {})
        m._feature_cols = list(payload.get("feature_cols") or [])
        # Reconstruct module with same hparams + load state
        m._module = _PanelGraphAttention(m.params)
        sd = _torch.load(
            io.BytesIO(base64.b64decode(payload["state_dict_b64"])),
            map_location="cpu",
        )
        m._module.load_state_dict(sd)
        m._module.eval()
        return m

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)


__all__ = [
    "GraphAttentionParams",
    "_PanelGraphAttention",
    "PanelGraphAttentionModel",
]
