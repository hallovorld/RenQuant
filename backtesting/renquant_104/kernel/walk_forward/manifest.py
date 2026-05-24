"""Walk-forward training manifest — JSON I/O + schema validation.

A manifest pins the entire walk-forward training run that fed a sim:

    {
      "cadence_days": 21,
      "training_window_years": 3.0,
      "retrains": [
        {"cutoff_date": "...", "trained_date": "...", "artifact_uri": "...",
         "calibrator_uri": "..."},
        ...
      ]
    }

Per CLAUDE.md §1c every helper here is single-responsibility and ≤ 50 lines.
The shared `RetrainEntry` is owned by `loader.py` (kept there to avoid an
import cycle: loader imports manifest, not the other way round).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.walk_forward.loader import RetrainEntry


@dataclass
class WalkForwardManifest:
    """All retrain entries plus the cadence metadata that produced them."""

    cadence_days: int
    training_window_years: float
    retrains: list[RetrainEntry] = field(default_factory=list)

    @staticmethod
    def _entry_to_dict(e: RetrainEntry) -> dict[str, Any]:
        row = {
            "cutoff_date": e.cutoff_date.isoformat(),
            "trained_date": e.trained_date.isoformat(),
            "artifact_uri": e.artifact_uri,
            "calibrator_uri": e.calibrator_uri,
            # 2026-05-11 Round 3 audit: persist lookahead_days so
            # the leakage guard's `cutoff + lookahead < today` check
            # survives a manifest round-trip. Default 0 = no
            # forward-label horizon (classification target).
            "lookahead_days": int(e.lookahead_days),
        }
        if e.effective_train_cutoff_date is not None:
            row["effective_train_cutoff_date"] = (
                e.effective_train_cutoff_date.isoformat()
            )
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "cadence_days": int(self.cadence_days),
            "training_window_years": float(self.training_window_years),
            "retrains": [self._entry_to_dict(e) for e in self.retrains],
        }


def _validate_entry(raw: dict, idx: int) -> RetrainEntry:
    """Coerce one JSON row to a RetrainEntry, validating the leakage invariant."""
    for key in ("cutoff_date", "trained_date", "artifact_uri"):
        if key not in raw:
            raise ValueError(f"manifest entry [{idx}] missing key {key!r}")
    cutoff = pd.Timestamp(raw["cutoff_date"])
    trained = pd.Timestamp(raw["trained_date"])
    uri = str(raw["artifact_uri"])
    effective_raw = raw.get("effective_train_cutoff_date")
    effective = pd.Timestamp(effective_raw) if effective_raw else None
    if not uri:
        raise ValueError(f"manifest entry [{idx}] has empty artifact_uri")
    if effective is not None and effective > cutoff:
        raise ValueError(
            f"manifest entry [{idx}] effective_train_cutoff_date "
            f"{effective.isoformat()} > cutoff_date {cutoff.isoformat()}"
        )
    if trained < cutoff:
        # cutoff is the LAST in-sample label; you can't have trained before
        # the cutoff — that means data was used the model wasn't supposed
        # to see (or, more likely, the manifest is mis-built).
        raise ValueError(
            f"manifest entry [{idx}] leakage: trained_date {trained.isoformat()} "
            f"< cutoff_date {cutoff.isoformat()} (training finished BEFORE "
            f"its own labelled-data window ended)."
        )
    # 2026-05-11 Round 3 audit: lookahead_days defaults to 0 for backward
    # compat with v1 manifests, but emit a warning so operators upgrade.
    lookahead = int(raw.get("lookahead_days", 0))
    if lookahead < 0:
        raise ValueError(
            f"manifest entry [{idx}] lookahead_days={lookahead} must be ≥ 0"
        )
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=trained,
        artifact_uri=uri,
        lookahead_days=lookahead,
        calibrator_uri=(
            str(raw.get("calibrator_uri") or raw.get("calibration_uri"))
            if (raw.get("calibrator_uri") or raw.get("calibration_uri"))
            else None
        ),
        effective_train_cutoff_date=effective,
    )


def read_manifest(path: "str | Path") -> WalkForwardManifest:
    """Parse a manifest JSON file into a WalkForwardManifest dataclass.

    Raises FileNotFoundError if the file is missing, ValueError on schema
    issues. Sorts entries by cutoff_date ascending.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"read_manifest: missing {p}")
    payload = json.loads(p.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"read_manifest: expected dict at top level, got {type(payload).__name__}")
    raw_retrains = payload.get("retrains")
    if not isinstance(raw_retrains, list):
        raise ValueError("read_manifest: 'retrains' must be a list")
    entries = [_validate_entry(r, i) for i, r in enumerate(raw_retrains)]
    entries.sort(key=lambda e: e.cutoff_date)
    return WalkForwardManifest(
        cadence_days=int(payload.get("cadence_days", 21)),
        training_window_years=float(payload.get("training_window_years", 3.0)),
        retrains=entries,
    )


def write_manifest(manifest: WalkForwardManifest, path: "str | Path") -> Path:
    """Persist a manifest as pretty-printed JSON, sorted by cutoff_date.

    Returns the resolved Path. Parent directory is created if missing.
    Validates before writing — a manifest with a leakage row will raise.
    """
    # Re-validate: every entry must obey trained_date >= cutoff_date.
    for i, e in enumerate(manifest.retrains):
        if e.trained_date < e.cutoff_date:
            raise ValueError(
                f"write_manifest: entry [{i}] trained_date {e.trained_date.isoformat()} "
                f"< cutoff_date {e.cutoff_date.isoformat()} — refusing to write."
            )
    sorted_entries = sorted(manifest.retrains, key=lambda e: e.cutoff_date)
    payload = WalkForwardManifest(
        cadence_days=manifest.cadence_days,
        training_window_years=manifest.training_window_years,
        retrains=sorted_entries,
    ).to_dict()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return p
