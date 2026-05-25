#!/usr/bin/env python
"""Fit a global calibrator for an HF PatchTST panel scorer.

Unlike XGB panel artifacts, PatchTST needs sequence input. This script rebuilds
the same CSRankNorm sequence panel used at inference, scores each eligible
(ticker, date), then fits the shared GlobalPanelCalibration on raw forward
excess returns. The saved artifact is stamped with the scorer fingerprint so
runtime cannot silently pair it with a different checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fit-hf-patchtst-calibrator")


def _resolve(raw_path: str | None, default: Path) -> Path:
    if not raw_path:
        return default
    p = Path(raw_path)
    return p if p.is_absolute() else REPO / p


def _artifact_fingerprint(path: Path, metadata: dict | None = None) -> str:
    meta = metadata or {}
    return (
        meta.get("model_content_fingerprint")
        or meta.get("artifact_fingerprint")
        or meta.get("artifact_sha256")
        or meta.get("model_fingerprint")
        or meta.get("fingerprint")
        or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _infer_raw_er_label(label_col: str) -> str:
    return label_col if label_col.endswith("_raw") else f"{label_col}_raw"


def _load_panel_with_raw_label(
    panel_path: Path,
    raw_label_panel_path: Path,
    feature_cols: list[str],
    label_col: str,
    er_label_col: str,
) -> pd.DataFrame:
    needed = ["ticker", "date", label_col, *feature_cols]
    panel = pd.read_parquet(panel_path, columns=[c for c in needed])
    panel["date"] = pd.to_datetime(panel["date"])

    raw = pd.read_parquet(raw_label_panel_path, columns=["ticker", "date", er_label_col])
    raw["date"] = pd.to_datetime(raw["date"])
    merged = panel.merge(raw, on=["ticker", "date"], how="left")
    if merged[er_label_col].notna().sum() == 0:
        raise ValueError(
            f"raw expected-return label {er_label_col!r} has no overlap with "
            f"{panel_path}. Check --raw-label-panel."
        )
    return merged


def _date_window_mask(dates: pd.Series, start: str | None, end: str | None) -> pd.Series:
    mask = pd.Series(True, index=dates.index)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates < pd.Timestamp(end)
    return mask


def _history_start(data_start: str | None, seq_len: int) -> pd.Timestamp | None:
    if not data_start:
        return None
    return pd.Timestamp(data_start) - pd.Timedelta(days=seq_len * 5)


def _score_sequences(
    scorer,
    panel: pd.DataFrame,
    *,
    data_start: str | None,
    data_end: str | None,
    batch_size: int,
) -> pd.DataFrame:
    import torch  # noqa: PLC0415
    from kernel.panel_pipeline.hf_patchtst_scorer import _csrank_norm_per_day  # noqa: PLC0415

    # Keep HF/PyTorch replay conservative by default. Raising torch intra-op
    # threads in the same process that imports xgboost/HF has caused native
    # crashes on Apple Silicon; callers may opt in via RENQUANT_TORCH_THREADS.
    torch.set_num_threads(max(1, int(os.getenv("RENQUANT_TORCH_THREADS", "1"))))
    feature_cols = list(scorer.feature_cols)
    seq_len = int(scorer.seq_len)
    work = panel
    if data_end:
        work = work[work["date"] < pd.Timestamp(data_end)]
    start = _history_start(data_start, seq_len)
    if start is not None:
        work = work[work["date"] >= start]
    log.info(
        "Sequence replay frame rows=%d tickers=%d dates=%s..%s",
        len(work), work["ticker"].nunique(),
        work["date"].min().date(), work["date"].max().date(),
    )
    ph = work[["ticker", "date", *feature_cols]].copy()
    log.info("Applying CSRankNorm to %d feature columns", len(feature_cols))
    ph = _csrank_norm_per_day(ph, feature_cols)
    ph = ph.sort_values(["ticker", "date"])

    seq_batch: list[np.ndarray] = []
    tickers: list[str] = []
    dates_out: list[pd.Timestamp] = []
    chunks: list[pd.DataFrame] = []

    def flush() -> None:
        if not seq_batch:
            return
        x = torch.from_numpy(np.stack(seq_batch, axis=0))
        with torch.no_grad():
            out = scorer._model(x)  # scorer owns the trained HF module.
        if isinstance(out, dict):
            scores = out["score"].detach().cpu().numpy().astype(float)
            mu = out.get("loc")
            sigma = out.get("scale")
            mu_arr = None if mu is None else mu.detach().cpu().numpy().astype(float)
            sg_arr = None if sigma is None else sigma.detach().cpu().numpy().astype(float)
        else:
            scores = out.detach().cpu().numpy().astype(float)
            mu_arr = sg_arr = None
        chunk = pd.DataFrame({
            "ticker": tickers.copy(),
            "date": dates_out.copy(),
            "panel_score": scores.reshape(-1),
        })
        if mu_arr is not None:
            chunk["mu"] = mu_arr.reshape(-1)
        if sg_arr is not None:
            chunk["sigma"] = sg_arr.reshape(-1)
        chunks.append(chunk)
        seq_batch.clear()
        tickers.clear()
        dates_out.clear()

    for ticker, g in ph.groupby("ticker", sort=False):
        g = g.sort_values("date")
        values = g[feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        dates = pd.to_datetime(g["date"]).reset_index(drop=True)
        valid_date = _date_window_mask(dates, data_start, data_end).to_numpy()
        for i in range(seq_len - 1, len(g)):
            if not valid_date[i]:
                continue
            seq_batch.append(values[i - seq_len + 1: i + 1])
            tickers.append(str(ticker))
            dates_out.append(pd.Timestamp(dates.iloc[i]))
            if len(seq_batch) >= batch_size:
                flush()
    flush()
    if not chunks:
        raise ValueError("No PatchTST sequences were scored; check date window.")
    return pd.concat(chunks, ignore_index=True)


def _mean_daily_ic(frame: pd.DataFrame, score_col: str, label_col: str) -> float:
    from scipy.stats import spearmanr  # noqa: PLC0415
    ics: list[float] = []
    for _, g in frame.dropna(subset=[score_col, label_col]).groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g[score_col], g[label_col])
        if np.isfinite(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scorer-artifact", required=True)
    p.add_argument("--panel", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--raw-label-panel", default="data/alpha158_291_fundamental_dataset_rawlabel.parquet")
    p.add_argument("--out", required=True)
    p.add_argument("--label-col", default=None)
    p.add_argument("--er-label-col", default=None)
    p.add_argument("--data-start", default=None)
    p.add_argument("--data-end", default=None)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--method", default="platt", choices=["platt", "isotonic"])
    args = p.parse_args()

    from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415
    from training_panel.global_calibrator import fit_global_calibrator  # noqa: PLC0415

    scorer_path = _resolve(args.scorer_artifact, Path(""))
    panel_path = _resolve(args.panel, REPO / "data" / "transformer_v4_wl200_clean.parquet")
    raw_label_path = _resolve(
        args.raw_label_panel,
        REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet",
    )
    out_path = _resolve(args.out, Path(""))

    log.info("Loading HF PatchTST scorer: %s", scorer_path)
    scorer = HFPatchTSTPanelScorer.load(scorer_path)
    scorer_fp = _artifact_fingerprint(scorer_path, scorer.metadata)
    label_col = args.label_col or scorer.metadata.get("label_col") or "fwd_60d_excess"
    er_label_col = args.er_label_col or _infer_raw_er_label(label_col)

    log.info("Loading panel=%s raw_label_panel=%s", panel_path, raw_label_path)
    panel = _load_panel_with_raw_label(
        panel_path, raw_label_path, scorer.feature_cols, label_col, er_label_col,
    )
    log.info(
        "Panel rows=%d tickers=%d dates=%s..%s",
        len(panel), panel["ticker"].nunique(),
        panel["date"].min().date(), panel["date"].max().date(),
    )

    scored = _score_sequences(
        scorer, panel,
        data_start=args.data_start,
        data_end=args.data_end,
        batch_size=args.batch_size,
    )
    scored = scored.merge(
        panel[["ticker", "date", label_col, er_label_col]],
        on=["ticker", "date"], how="left",
    )
    log.info(
        "Scored rows=%d tickers=%d dates=%s..%s",
        len(scored), scored["ticker"].nunique(),
        scored["date"].min().date(), scored["date"].max().date(),
    )

    panel_scores = {}
    future_returns = {}
    for ticker, g in scored.groupby("ticker"):
        gs = g.sort_values("date").set_index("date")
        panel_scores[ticker] = gs["panel_score"]
        future_returns[ticker] = gs[er_label_col].dropna()

    ic_label = _mean_daily_ic(scored, "panel_score", label_col)
    ic_er = _mean_daily_ic(scored, "panel_score", er_label_col)
    log.info("Daily IC: model_label=%+.4f raw_ER=%+.4f", ic_label, ic_er)
    calib = fit_global_calibrator(
        panel_scores,
        future_returns,
        lookahead_days=60,
        threshold=0.0,
        threshold_mode="crosssectional",
        method=args.method,
        min_rows=1000,
    )
    metadata = {
        "scorer_artifact": str(scorer_path),
        "scorer_artifact_fingerprint": scorer_fp,
        "scorer_model_content_fingerprint": scorer_fp,
        "scorer_val_ic": scorer.metadata.get("val_ic"),
        "scorer_oos_mean_ic": ic_label,
        "scorer_oos_mean_ic_vs_er_label": ic_er,
        "model_label_col": label_col,
        "expected_return_label_col": er_label_col,
        "expected_return_label_contract": "raw_return_units_required",
        "panel_path": str(panel_path),
        "raw_label_panel": str(raw_label_path),
        "method": args.method,
        "calibration_scope": "hf_patchtst_sequence_replay",
        "lookahead_days_used": 60,
    }
    if args.data_start:
        metadata["data_window_start"] = args.data_start
    if args.data_end:
        metadata["data_window_end"] = args.data_end
    log.info("Saving calibrator: %s", out_path)
    calib.save(out_path, metadata=metadata)
    log.info(
        "Saved n=%d pool_IC=%+.4f base_rate=%.3f fingerprint=%s",
        calib.metadata["n_rows"],
        calib.metadata.get("pool_ic") or 0.0,
        calib.metadata.get("prob_base_rate", float("nan")),
        scorer_fp,
    )


if __name__ == "__main__":
    main()
