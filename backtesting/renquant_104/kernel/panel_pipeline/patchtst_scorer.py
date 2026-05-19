"""PatchTST inference-side scorer for the panel-LTR pipeline.

Wires transformer_v4.py::PatchTSTRanker into the prod ApplyScoresTask via
the same .score(matrix) -> Series interface as PanelScorer (XGB) and
TransformerPanelScorer (legacy).

Key difference vs XGB: PatchTST is a SEQUENCE model. It needs the last
`seq_len` (default 32) days of features per ticker, not just today's
snapshot. This scorer:
  1. Loads a state_dict from .pt (transformer_v4.py checkpoint format)
  2. Rebuilds PatchTSTRanker architecture with same params
  3. At score-time: takes today's feature_matrix + reads HISTORICAL
     features from panel parquet for each ticker, stacks into
     (N_tickers, seq_len, n_channels) tensor, predicts.

Reference:
  - transformer_v4.py::PatchTSTRanker (Nie 2023 ICLR PatchTST adaptation)
  - kernel/panel_pipeline/transformer_scorer.py (legacy TransformerPanelScorer)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[4]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

# 2026-05-18 OpenMP fix: xgboost (imported via panel_pipeline.__init__)
# sets OMP threads at import time. PyTorch TransformerEncoder construction
# then segfaults on OMP collision (consistent SIGSEGV on M2 Pro / macOS).
# Force single-thread OMP/MKL when this module loads (callers wanting
# more torch parallelism can re-set after our scorer load).
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

log = logging.getLogger("kernel.panel_pipeline.patchtst_scorer")


class PatchTSTPanelScorer:
    """Single-date inference wrapper around a transformer_v4.py PatchTST artifact.

    Interface-compatible with :class:`PanelScorer`: ``feature_cols`` attr
    + ``score(matrix) -> pd.Series``.

    The score() method needs `panel_history_df` injected (a wider panel
    containing the last `seq_len` dates × N tickers of pre-normalized
    features). At inference time the adapter loads this from the
    alpha158_fund panel parquet.
    """

    def __init__(self, model, feature_cols: list[str],
                 seq_len: int, metadata: Optional[dict] = None):
        self._model = model
        self._model.eval()
        self.feature_cols = list(feature_cols)
        self.seq_len = int(seq_len)
        self.metadata = metadata or {}
        # PanelScorer compatibility: this scorer needs HISTORY, not just snapshot
        self.requires_history = True

    @classmethod
    def load(cls, path: str | Path,
             feature_cols: Optional[list[str]] = None,
             seq_len: int = 32) -> "PatchTSTPanelScorer":
        """Load PatchTST checkpoint. Needs feature_cols passed in since
        transformer_v4.py doesn't save them in the .pt."""
        import torch  # noqa: PLC0415
        from transformer_v4 import PatchTSTRanker  # noqa: PLC0415
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        # Determine n_channels from first layer shape
        # patch_embed.weight: (d_model, patch_len * n_channels)
        # With patch_len=8 default → n_channels = weight.shape[1] // 8
        sd = ckpt["state_dict"]
        patch_embed_w = sd["patch_embed.weight"]
        d_model = int(patch_embed_w.shape[0])
        patch_len_x_chan = int(patch_embed_w.shape[1])
        patch_len = 8  # default in transformer_v4.py
        n_channels = patch_len_x_chan // patch_len

        if feature_cols is None:
            raise ValueError("feature_cols must be passed (transformer_v4 "
                              ".pt doesn't store them). Pass artifact's "
                              "feature_cols list.")
        if len(feature_cols) != n_channels:
            raise ValueError(
                f"feature_cols len {len(feature_cols)} != model n_channels "
                f"{n_channels} (from patch_embed weight)")

        model = PatchTSTRanker(n_channels=n_channels, seq_len=seq_len)
        model.load_state_dict(sd)
        log.info("PatchTSTPanelScorer loaded: n_channels=%d seq_len=%d "
                 "d_model=%d epoch=%d val_ic=%.4f", n_channels, seq_len, d_model,
                 int(ckpt.get("epoch", -1)), float(ckpt.get("val_ic", float("nan"))))
        meta = {
            "epoch": int(ckpt.get("epoch", -1)),
            "val_ic": float(ckpt.get("val_ic", float("nan"))),
            "patch_len": patch_len,
            "d_model": d_model,
        }
        return cls(model=model, feature_cols=feature_cols, seq_len=seq_len,
                   metadata=meta)

    def score_with_history(self, panel_history: pd.DataFrame,
                            target_tickers: list[str]) -> pd.Series:
        """Score given a (ticker, date) panel containing the last `seq_len`
        dates of pre-normalized features per ticker.

        Args:
            panel_history: DataFrame with cols (ticker, date) + self.feature_cols.
                MUST contain ≥ seq_len rows per ticker in target_tickers,
                already pre-normalized (z-scored, matching training panel).
            target_tickers: tickers to score (the candidates for today's bar).

        Returns:
            pd.Series indexed by ticker, name='panel_score'.
        """
        import torch  # noqa: PLC0415

        if not target_tickers:
            return pd.Series([], dtype=float, name="panel_score")

        # Sort each ticker's history ascending, take last seq_len rows
        sequences = []
        valid_tickers = []
        for tkr in target_tickers:
            g = panel_history[panel_history["ticker"] == tkr].sort_values("date")
            if len(g) < self.seq_len:
                log.warning("PatchTST: ticker %s has %d rows, need %d — skip",
                             tkr, len(g), self.seq_len)
                continue
            g = g.tail(self.seq_len)
            arr = g[self.feature_cols].fillna(0.0).values.astype(np.float32)
            sequences.append(arr)
            valid_tickers.append(tkr)

        if not sequences:
            return pd.Series([], dtype=float, name="panel_score")

        # Stack: (N_tickers, seq_len, n_channels)
        X = np.stack(sequences, axis=0)
        x_tensor = torch.from_numpy(X)
        t_idx = torch.zeros(len(valid_tickers), dtype=torch.long)  # dummy

        with torch.no_grad():
            scores = self._model(x_tensor, t_idx).cpu().numpy()

        result = pd.Series(scores, index=valid_tickers, name="panel_score")
        log.info("PatchTSTPanelScorer.score_with_history: scored %d/%d tickers "
                 "(mean=%+.4f std=%.4f)", len(result), len(target_tickers),
                 float(result.mean()), float(result.std()))
        return result

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        """Legacy interface: takes single-snapshot DataFrame.

        PatchTST CAN'T do single-snapshot inference (needs sequence). This
        method raises a clear error to force callers to use the history
        path. The patched ApplyScoresTask will detect requires_history=True
        and route to score_with_history() instead.
        """
        raise NotImplementedError(
            "PatchTSTPanelScorer requires sequence input. Use "
            "score_with_history(panel_history, target_tickers) instead. "
            "If you got this error from ApplyScoresTask, it means the "
            "dispatch path wasn't patched to detect requires_history=True.")


__all__ = ["PatchTSTPanelScorer"]
