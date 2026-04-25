"""PyTorch cross-sectional transformer for the Stage-1 panel.

Alternative ranking backend to :class:`PanelLTRModel` (XGBoost). Mirrors the
same public surface so the caller can dispatch on `panel_ltr.backend`.

Architecture (see `doc/renquant_104_transformer_design.md` §2):

- Input: panel rows grouped by date → one date-group per sample.
- Feature encoder: ``Linear(F → d_model)``.
- N × transformer encoder blocks (self-attention within date-group only).
- Score head: ``Linear(d_model → 1)`` per ticker.
- Loss: ListNet over the date-group.
- Regularization: feature dropout, ticker-conditional dropout, label smoothing,
  AdamW weight decay, early-stopping.

Public API (same shape as :class:`PanelLTRModel`)::

    m = PanelTransformerModel(params=None)
    m.train(panel, group_sizes, feature_cols, num_boost_round=..., ...) -> dict
    m.predict(panel) -> pd.Series
    m.save(path, metadata=None)
    PanelTransformerModel.load(path)

The serialized artifact has suffix ``.pt`` (state_dict) paired with a
``.json`` sidecar holding feature_cols + hparams + training metadata.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Hyperparameters ────────────────────────────────────────────────────────────

@dataclass
class TransformerParams:
    d_model:          int   = 128
    n_heads:          int   = 4
    n_layers:         int   = 3
    feedforward_dim:  int   = 256
    dropout:          float = 0.3        # attention/FF/residual dropout
    feature_dropout:  float = 0.2        # zero-out input features
    ticker_dropout:   float = 0.1        # zero-out whole ticker within group
    label_smoothing:  float = 0.05       # additive Gaussian noise on labels
    lr:               float = 1e-4
    weight_decay:     float = 1e-4
    max_epochs:       int   = 50
    batch_size:       int   = 32         # dates per batch
    patience:         int   = 6          # early-stopping on eval IC, per user pref
    device:           str   = "mps"      # "mps" | "cuda" | "cpu"
    deterministic:    bool  = True
    seed:             int   = 42
    max_tickers:      int   = 128        # pad groups to this size (≥ watchlist size; was 38, silently truncated 99-ticker groups → audit T-1 2026-04-25)


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
        self.feat_dropout    = nn.Dropout(p.feature_dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model         = p.d_model,
            nhead           = p.n_heads,
            dim_feedforward = p.feedforward_dim,
            dropout         = p.dropout,
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
        self.score_head = nn.Linear(p.d_model, 1)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F). pad_mask: (B, T), True where padding.
        x = self.feat_dropout(x)
        h = self.feature_encoder(x)                  # (B, T, d_model)
        h = self.encoder(h, src_key_padding_mask=pad_mask)
        s = self.score_head(h).squeeze(-1)           # (B, T)
        # Push padded scores to -inf so softmax in loss ignores them.
        s = s.masked_fill(pad_mask, float("-inf"))
        return s


# ── ListNet loss (top-1 softmax cross-entropy over group scores) ──────────────

def _listnet_loss(scores: torch.Tensor, labels: torch.Tensor,
                  pad_mask: torch.Tensor,
                  nan_label_mask: torch.Tensor | None = None,
                  ) -> torch.Tensor:
    """Cao 2007 top-1 ListNet.

    P(i) = softmax(label_i) over the group; loss = -sum P_label * log P_pred.

    Audit fix T-8 (2026-04-25): when a row had a NaN label originally
    (now zero-substituted by ``_build_date_groups``), it must be excluded
    from BOTH the label softmax and the prediction softmax — same as
    padding. Without this, NaN-rows had probability `exp(0)/Σ exp(yi)`,
    pulling predictions toward the median of valid labels.

    Args:
      scores         : (B, T) raw scores (model output)
      labels         : (B, T) labels (NaN → 0 substituted upstream)
      pad_mask       : (B, T) True where the slot is padding
      nan_label_mask : (B, T) True where the original label was NaN.
                       If None, treated as all-False (back-compat).
    """
    if nan_label_mask is None:
        nan_label_mask = torch.zeros_like(pad_mask)
    invalid = pad_mask | nan_label_mask
    # Mask both padded AND NaN-label positions to -inf before softmax →
    # zero probability mass. ALSO mask the prediction softmax (so the
    # model isn't penalised for any score at NaN positions either).
    minus_inf = float("-inf")
    label_logits = labels.masked_fill(invalid, minus_inf)
    p_label = F.softmax(label_logits, dim=-1)
    score_logits = scores.masked_fill(invalid, minus_inf)
    log_p_pred = F.log_softmax(score_logits, dim=-1)
    # masked_fill the per-row contribution to 0 at invalid slots so we
    # don't sum NaN×anything (log_softmax of -inf is -inf).
    loss_per_row = -(p_label * log_p_pred)
    loss_per_row = loss_per_row.masked_fill(invalid, 0.0)
    # Mean over non-degenerate groups (at least 2 valid tickers).
    valid_groups = (~invalid).sum(dim=-1) >= 2
    if valid_groups.any():
        return loss_per_row.sum(dim=-1)[valid_groups].mean()
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

    X_flat_raw = panel[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_flat_raw = panel[label_col].to_numpy(dtype=np.float32, copy=True)
    nan_label_flat = ~np.isfinite(y_flat_raw)

    X_flat = np.nan_to_num(X_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y_flat = np.nan_to_num(y_flat_raw, nan=0.0, posinf=0.0, neginf=0.0)

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
    """Resolve device preference with graceful fallback: mps → cuda → cpu."""
    if requested == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
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
        group_sizes: np.ndarray,
        feature_cols: list[str],
        label_col: str = "label",
        weight_col: str | None = "weight",     # unused by transformer (ListNet scale-invariant)
        num_boost_round: int | None = None,    # alias for max_epochs if provided
        early_stopping_rounds: int | None = None,  # alias for patience if provided
        eval_panel: pd.DataFrame | None = None,
        eval_group_sizes: np.ndarray | None = None,
    ) -> dict:
        """Fit the transformer; return train/eval metadata dict."""
        del weight_col   # ListNet is scale-invariant; group weights not applied here.
        self.feature_cols = list(feature_cols)
        p = self.params
        if num_boost_round is not None:
            p.max_epochs = int(num_boost_round)
        if early_stopping_rounds is not None:
            p.patience = int(early_stopping_rounds)

        # Reproducibility
        _seed_everything(p.seed)
        if p.deterministic:
            # Best-effort: MPS doesn't expose all ops deterministically, but
            # torch.use_deterministic_algorithms guards the ones that do.
            os.environ.setdefault("PYTHONHASHSEED", str(p.seed))
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass

        # Single-thread CPU guard: PyTorch + OpenMP deadlock in processes
        # that previously used `fork`-based multiprocessing (e.g. the panel
        # pipeline's parallel TickerPanelFeatureJob workers). Forcing
        # set_num_threads(1) here keeps training on the main thread and
        # avoids the fork/OMP interaction entirely. Performance cost is
        # minimal for our 47k-row panel; MPS isn't affected (dispatches to
        # its own backend). Safe to leave on unconditionally.
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

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

        best_eval = float("-inf")
        best_state: dict | None = None
        bad_epochs = 0
        gen = torch.Generator(device="cpu").manual_seed(p.seed)

        n_groups = xtr.shape[0]
        for epoch in range(p.max_epochs):
            self._model.train()
            order = torch.randperm(n_groups, generator=gen).numpy()
            epoch_loss = 0.0
            for start in range(0, n_groups, p.batch_size):
                idx = order[start:start + p.batch_size]
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

                scores = self._model(xb, mb)
                loss = _listnet_loss(scores, yb, mb, nb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                epoch_loss += float(loss.item()) * len(idx)
            epoch_loss /= max(n_groups, 1)

            train_ic = self._ic_on_tensors(xtr, ytr, padtr, panel, label_col, group_sizes)

            if xte is not None:
                # nante carries NaN-label mask for parity but is unused by IC
                # computation (spearman just skips degenerate groups).
                _ = nante
                eval_ic = self._ic_on_tensors(
                    xte, yte, padte, eval_panel, label_col, eval_group_sizes,
                )
                improved = eval_ic > best_eval + 1e-6
                if improved:
                    best_eval  = eval_ic
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
                if bad_epochs >= p.patience:
                    break
            else:
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
            import logging  # noqa: PLC0415
            logging.getLogger("panel.transformer").warning(
                "PanelTransformerModel.predict: a date-group exceeds "
                "max_tickers=%d → chunk-split fallback. Cross-chunk scores "
                "lose comparability. Raise max_tickers and retrain.",
                max_t,
            )
        group_sizes_exp = np.array(expanded, dtype=np.int64)
        x, _, pad, _ = _build_date_groups(
            panel.assign(label=0.0), group_sizes_exp, self.feature_cols, "label",
            max_t,
        )
        self._model.eval()
        preds_flat = np.full(len(panel), np.nan, dtype=np.float32)
        offset = 0
        bs = self.params.batch_size
        with torch.no_grad():
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
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".pt":
            path = path.with_suffix(".pt")
        sidecar = path.with_suffix(".json")
        torch.save(self._model.state_dict(), path)
        payload = {
            "version":      1,
            "kind":         "panel_transformer",
            "trained_date": str(date.today()),
            "feature_cols": list(self.feature_cols),
            "params":       asdict(self.params),
            "best_iter":    self.best_iter,
            "history":      self.history[-min(len(self.history), 50):],
        }
        if metadata:
            payload.update({k: v for k, v in metadata.items() if k not in payload})
        sidecar.write_text(json.dumps(payload, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "PanelTransformerModel":
        path = Path(path)
        if path.suffix == ".json":
            pt_path = path.with_suffix(".pt")
            json_path = path
        else:
            pt_path = path if path.suffix == ".pt" else path.with_suffix(".pt")
            json_path = pt_path.with_suffix(".json")
        meta = json.loads(json_path.read_text())
        if meta.get("kind") != "panel_transformer":
            raise ValueError(f"Not a panel_transformer artifact: {json_path}")
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
        with torch.no_grad():
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
        y_flat = panel["label"].to_numpy() if "label" in panel.columns else None
        for gs in group_sizes:
            gs = int(gs)
            p_slice = preds_flat[offset2:offset2 + gs]
            y_slice = (y_flat[offset2:offset2 + gs] if y_flat is not None
                       else y.reshape(-1)[offset2:offset2 + gs])
            offset2 += gs
            if gs < 2 or np.all(p_slice == p_slice[0]) or np.all(y_slice == y_slice[0]):
                continue
            rho, _ = spearmanr(p_slice, y_slice)
            if not np.isnan(rho):
                ics.append(float(rho))
        return float(np.mean(ics)) if ics else float("nan")


__all__ = [
    "TransformerParams",
    "PanelTransformerModel",
]
