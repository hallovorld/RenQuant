"""Load a trained meta-label artifact and wrap it as a predictor callable.

The artifact format is produced by ``scripts/_meta_label_train.py``:
    {
      "version": 1,
      "kind":    "meta_label_exit_xgb",
      "feature_cols":     [...],
      "booster_raw_json": <str>,
      "default_threshold": <float>,
      ...
    }

The factory ``load_meta_label_predictor(artifact_path)`` returns a
callable ``predictor(features: dict) -> float in [0, 1]`` that the
:class:`MetaLabelVetoTask` consumes. Missing artifact / corrupt file
yields ``None`` per CLAUDE.md §5.13.10 fallback.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

log = logging.getLogger("kernel.meta_label.predictor")


def load_meta_label_predictor(artifact_path: "str | Path") -> Optional[Callable[[dict], float]]:
    """Load the XGBoost meta-label artifact and return a predictor callable.

    Returns None if the file is missing or malformed (§5.13.10 fallback).
    """
    p = Path(artifact_path)
    if not p.exists():
        log.warning("meta-label artifact missing: %s (veto task will no-op)", p)
        return None
    try:
        art = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.error("meta-label artifact unreadable: %s (%s)", p, exc)
        return None
    if art.get("kind") != "meta_label_exit_xgb":
        log.error("meta-label artifact wrong kind: %s", art.get("kind"))
        return None

    feature_cols = art.get("feature_cols") or []
    if not feature_cols:
        log.error("meta-label artifact has empty feature_cols")
        return None

    import xgboost as xgb  # noqa: PLC0415
    booster = xgb.Booster()
    try:
        booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))
    except Exception as exc:  # noqa: BLE001
        log.error("meta-label booster load failed: %s", exc)
        return None

    def _predict(features: dict) -> float:
        # Order features per the trained model's feature_cols. Missing
        # features → 0.0 (XGB-hist handles missing natively but we
        # pre-fill for safety).
        vec = np.zeros(len(feature_cols), dtype=np.float64)
        for i, col in enumerate(feature_cols):
            v = features.get(col, 0.0)
            try:
                fv = float(v)
                if math.isfinite(fv):
                    vec[i] = fv
            except (TypeError, ValueError):
                vec[i] = 0.0
        dmat = xgb.DMatrix(vec.reshape(1, -1))
        proba = float(booster.predict(dmat)[0])
        return proba

    log.info("Loaded meta-label predictor from %s  features=%d",
             p, len(feature_cols))
    return _predict
