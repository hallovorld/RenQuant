"""HF PatchTST inference scorer — loads model trained by scripts/patchtst_hf.py.

Per 2026-05-19 user mandate "shadow promote pt_01". Interface mirrors
PatchTSTPanelScorer (legacy custom-impl) so model_registry can dispatch
either kind via the same API.

Critical inference detail: at training time, features go through
**CSRankNorm per-day** (Kelly-Gu-Xiu 2020). The model expects rank-normalized
inputs in [-0.5, +0.5]. At inference time, panel_history MUST be
CSRankNorm-transformed BEFORE building sequences — otherwise the model
sees out-of-distribution feature scales and produces garbage scores.

Scorer applies CSRankNorm itself (consumer can pass raw features).
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import Optional

# OMP fix — same as patchtst_scorer.py (xgboost ↔ HF torch coexistence)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.panel_pipeline.hf_patchtst_scorer")


def _csrank_norm_per_day(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Same as scripts/patchtst_hf.py::csrank_norm_per_day — consistency at
    inference."""
    df = df.copy()
    df[feat_cols] = (df.groupby("date")[feat_cols].rank(pct=True) - 0.5)
    df[feat_cols] = df[feat_cols].fillna(0.0)
    return df


class HFPatchTSTPanelScorer:
    """Mirror of PatchTSTPanelScorer interface using HF transformers backbone.

    Attrs:
      feature_cols: list[str] — feature columns expected by the model
      seq_len: int — sequence context length
      requires_history: True — must be passed full history (not single snapshot)
    """

    def __init__(self, model, feature_cols: list[str], seq_len: int,
                 metadata: Optional[dict] = None):
        self._model = model
        self._model.eval()
        self.feature_cols = list(feature_cols)
        self.seq_len = int(seq_len)
        self.metadata = metadata or {}
        self.requires_history = True

    @classmethod
    def load(cls, path: str | Path) -> "HFPatchTSTPanelScorer":
        """Load HF PatchTST checkpoint produced by scripts/patchtst_hf.py
        --save-model."""
        import torch  # noqa: PLC0415
        from transformers import PatchTSTConfig  # noqa: PLC0415
        # Import HFPatchTSTRanker from the training script
        import importlib.util  # noqa: PLC0415
        from pathlib import Path as _P  # noqa: PLC0415
        repo = _P(__file__).resolve().parents[4]
        spec = importlib.util.spec_from_file_location(
            "patchtst_hf_mod", repo / "scripts/patchtst_hf.py")
        hf_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(hf_mod)

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PatchTSTConfig(**ckpt["config_dict"])
        model = hf_mod.HFPatchTSTRanker(cfg)
        # SWA-wrapped state has different prefix
        state = ckpt["state_dict"]
        # If saved from AveragedModel (SWA), strip "module." prefix
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()
                     if k != "n_averaged"}
        model.load_state_dict(state)
        model.eval()
        log.info("HFPatchTSTPanelScorer loaded: n_feat=%d seq_len=%d "
                 "val_ic=%.4f swa=%s",
                 len(ckpt["feature_cols"]), ckpt["seq_len"],
                 float(ckpt.get("best_val_ic", float("nan"))),
                 ckpt.get("uses_swa", False))
        return cls(model=model, feature_cols=ckpt["feature_cols"],
                   seq_len=ckpt["seq_len"],
                   metadata={
                       "val_ic": float(ckpt.get("best_val_ic", float("nan"))),
                       "uses_swa": ckpt.get("uses_swa", False),
                       "uses_csranknorm": ckpt.get(
                           "uses_csranknorm_preprocessing", False),
                       "label_col": ckpt.get("label_col"),
                   })

    def score_with_history(self, panel_history: pd.DataFrame,
                            target_tickers: list[str]) -> pd.Series:
        """Score given (ticker, date) panel with ≥ seq_len rows per target ticker.

        CRITICAL: applies CSRankNorm per-day BEFORE building sequences (model
        was trained on rank-normalized features).
        """
        import torch  # noqa: PLC0415

        if not target_tickers:
            return pd.Series([], dtype=float, name="panel_score")

        # Apply CSRankNorm if the model expects it
        if self.metadata.get("uses_csranknorm", True):
            ph = _csrank_norm_per_day(panel_history.copy(), self.feature_cols)
        else:
            ph = panel_history.copy()

        sequences = []
        valid_tickers = []
        for tkr in target_tickers:
            g = ph[ph["ticker"] == tkr].sort_values("date")
            if len(g) < self.seq_len:
                log.warning("HF PatchTST: ticker %s has %d rows, need %d — skip",
                             tkr, len(g), self.seq_len)
                continue
            g = g.tail(self.seq_len)
            arr = g[self.feature_cols].fillna(0.0).values.astype(np.float32)
            sequences.append(arr)
            valid_tickers.append(tkr)

        if not sequences:
            return pd.Series([], dtype=float, name="panel_score")

        # (N_tickers, seq_len, n_channels)
        X = np.stack(sequences, axis=0)
        x_tensor = torch.from_numpy(X)
        with torch.no_grad():
            scores = self._model(x_tensor).cpu().numpy()
        result = pd.Series(scores, index=valid_tickers, name="panel_score")
        log.info("HFPatchTSTPanelScorer.score_with_history: scored %d/%d "
                 "(mean=%+.4f std=%.4f)", len(result), len(target_tickers),
                 float(result.mean()), float(result.std()))
        return result

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:
        raise NotImplementedError(
            "HFPatchTSTPanelScorer requires sequence input. Use "
            "score_with_history(panel_history, target_tickers) instead. "
            "If ApplyScoresTask routed here, dispatch should detect "
            "requires_history=True."
        )


__all__ = ["HFPatchTSTPanelScorer"]
