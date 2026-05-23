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

from kernel.panel_pipeline.panel_scorer import stamp_artifact_metadata

log = logging.getLogger("kernel.panel_pipeline.hf_patchtst_scorer")


def _csrank_norm_per_day(df: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Same as scripts/patchtst_hf.py::csrank_norm_per_day — consistency at
    inference."""
    df = df.copy()
    df[feat_cols] = (df.groupby("date")[feat_cols].rank(pct=True) - 0.5)
    df[feat_cols] = df[feat_cols].fillna(0.0)
    return df


def _load_contract_sidecar(path: Path) -> dict:
    """Load optional metadata sidecar for legacy HF checkpoints.

    Early shadow checkpoints predated the full training contract fields in
    ``scripts/patchtst_hf.py``. A sidecar lets us restamp the provenance
    without mutating the binary Torch checkpoint.
    """
    candidates = [
        path.with_name(path.name + ".metadata.json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.stem + "_summary.json"),
    ]
    import json  # noqa: PLC0415
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text())
        except Exception as exc:  # noqa: BLE001
            log.warning("HF PatchTST metadata sidecar %s failed: %s",
                        candidate, exc)
            continue
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["_contract_sidecar_path"] = str(candidate)
            return payload
    return {}


def _coalesce(*values):
    for value in values:
        if value is not None:
            return value
    return None


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

        path = Path(path)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sidecar = _load_contract_sidecar(path)
        cfg = PatchTSTConfig(**ckpt["config_dict"])
        uses_dist = ckpt.get("uses_distributional_head", False)
        model = hf_mod.HFPatchTSTRanker(cfg, use_distributional_head=uses_dist)
        state = ckpt["state_dict"]
        # Legacy: SWA-wrapped state had "module." prefix (pre-2026-05-19 refactor)
        if any(k.startswith("module.") for k in state):
            state = {k.removeprefix("module."): v for k, v in state.items()
                     if k != "n_averaged"}
        # Legacy: pre-refactor checkpoints had key `head.*` instead of `rank_head.*`
        if any(k.startswith("head.") for k in state) and not any(
                k.startswith("rank_head.") for k in state):
            state = {("rank_head." + k.removeprefix("head.")) if k.startswith("head.") else k: v
                     for k, v in state.items()}
        model.load_state_dict(state, strict=False)
        model.eval()
        log.info("HFPatchTSTPanelScorer loaded: n_feat=%d seq_len=%d "
                 "val_ic=%.4f dist_head=%s",
                 len(ckpt["feature_cols"]), ckpt["seq_len"],
                 float(ckpt.get("best_val_ic", float("nan"))), uses_dist)
        ckpt_contract = ckpt.get("training_contract", {}) or {}
        sidecar_contract = sidecar.get("training_contract", {}) or {}
        contract = dict(sidecar_contract)
        contract.update({k: v for k, v in ckpt_contract.items()
                         if v is not None})
        split_ranges = _coalesce(
            ckpt.get("split_date_ranges"),
            contract.get("split_date_ranges"),
            sidecar.get("split_date_ranges"),
        )
        validation_end = (
            (split_ranges.get("val") or {}).get("end")
            if isinstance(split_ranges, dict) else None
        )
        metadata = stamp_artifact_metadata({
                       "val_ic": float(ckpt.get("best_val_ic", float("nan"))),
                       "uses_distributional_head": uses_dist,
                       "uses_csranknorm": ckpt.get(
                           "uses_csranknorm_preprocessing", False),
                       "label_col": ckpt.get("label_col"),
                       "trained_date": _coalesce(
                           ckpt.get("trained_date"),
                           contract.get("trained_date"),
                           sidecar.get("trained_date"),
                       ),
                       "effective_train_cutoff_date": _coalesce(
                           ckpt.get("effective_train_cutoff_date"),
                           contract.get("effective_train_cutoff_date"),
                           sidecar.get("effective_train_cutoff_date"),
                       ),
                       "effective_selection_cutoff_date": _coalesce(
                           ckpt.get("effective_selection_cutoff_date"),
                           contract.get("effective_selection_cutoff_date"),
                           sidecar.get("effective_selection_cutoff_date"),
                           validation_end,
                       ),
                       "lookahead_days": _coalesce(
                           ckpt.get("lookahead_days"),
                           contract.get("lookahead_days"),
                           sidecar.get("lookahead_days"),
                       ),
                       "split_date_ranges": split_ranges,
                       "config_fingerprint": _coalesce(
                           ckpt.get("config_fingerprint"),
                           (contract.get("config_contract", {}) or {}).get(
                               "config_fingerprint"
                           )
                       ),
                       "config_fingerprint_fields": _coalesce(
                           ckpt.get("config_fingerprint_fields"),
                           (contract.get("config_contract", {}) or {}).get(
                               "config_fingerprint_fields"
                           )
                       ),
                       "trained_watchlist_n": _coalesce(
                           ckpt.get("trained_watchlist_n"),
                           (contract.get("config_contract", {}) or {}).get(
                               "trained_watchlist_n"
                           )
                       ),
                       "training_contract": contract,
                       "contract_sidecar_path": sidecar.get(
                           "_contract_sidecar_path"),
                       "per_regime_ic": ckpt.get("per_regime_ic", {}),
                   }, path)
        return cls(model=model, feature_cols=ckpt["feature_cols"],
                   seq_len=ckpt["seq_len"], metadata=metadata)

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
            outputs = self._model(x_tensor)
        # New API (post 2026-05-19 HF Trainer refactor): forward returns dict
        if isinstance(outputs, dict):
            scores = outputs["score"].cpu().numpy()
            # Store σ for downstream Kelly/QP if distributional head present
            if "scale" in outputs:
                self._last_sigma = pd.Series(
                    outputs["scale"].cpu().numpy(),
                    index=valid_tickers, name="panel_sigma")
        else:
            # Legacy tensor-output checkpoints (pre-refactor)
            scores = outputs.cpu().numpy()
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
