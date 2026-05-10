"""Walk-forward model loader — point-in-time PanelScorer dispatch.

P1 implementation (2026-05-10) — replaces the P2 stub.

Sim binds against `WalkForwardModelLoader.model_as_of(today)` only,
never directly against the artifacts dict. Every call returns the
PanelScorer trained on labels strictly before `today`, eliminating the
look-ahead leakage class documented in CLAUDE.md §5.13.

Contract (DO NOT CHANGE without P1 / P2 sync):

    @dataclass(frozen=True)
    class RetrainEntry:
        cutoff_date: pd.Timestamp
        trained_date: pd.Timestamp
        artifact_uri: str

    class WalkForwardModelLoader:
        def __init__(self, manifest_path: Path) -> None: ...
        def model_as_of(self, today: pd.Timestamp) -> "PanelScorer":
            "Latest retrain with cutoff_date < today.
             Raises ValueError if none."
        def has_walkforward_model(self) -> bool: ...
"""
from __future__ import annotations

import glob
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from kernel.walk_forward.leakage_guard import assert_no_leakage

if TYPE_CHECKING:  # pragma: no cover
    from kernel.panel_pipeline.panel_scorer import PanelScorer


@dataclass(frozen=True)
class RetrainEntry:
    """One row of the walk-forward manifest.

    cutoff_date: last in-sample label date the model was trained on.
                 Strictly less than every sim bar that uses this model.
    trained_date: the wallclock date training finished (used by the
                 leakage guard's defense-in-depth assertion).
    artifact_uri: filesystem path (or future cloud URI) to the
                 PanelScorer-loadable artifact.
    """
    cutoff_date: pd.Timestamp
    trained_date: pd.Timestamp
    artifact_uri: str


def _resolve_manifest_path(raw: "str | Path") -> Path:
    """Resolve an explicit path, OR a glob pattern (newest match wins).

    `manifest_path` is normally a single file. To make manifests easy to
    discover from operator scripts we accept a glob: if the literal path
    doesn't exist AND the string contains glob meta-chars, expand and
    pick the lexicographically last match (typical convention: filenames
    contain ISO timestamps, so last == newest). Returns the path
    unchanged when there's no match — caller's existence check raises.
    """
    p = Path(raw)
    if p.exists():
        return p
    s = str(raw)
    if any(ch in s for ch in "*?["):
        matches = sorted(glob.glob(s))
        if matches:
            return Path(matches[-1])
    return p


def _parse_entry(r: dict) -> RetrainEntry:
    """Build one RetrainEntry from a manifest row, enforcing leakage invariant."""
    cutoff = pd.Timestamp(r["cutoff_date"])
    trained = pd.Timestamp(r["trained_date"])
    if trained < cutoff:
        raise ValueError(
            f"manifest entry leakage: trained_date {trained.isoformat()} "
            f"< cutoff_date {cutoff.isoformat()} — refusing to load."
        )
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=trained,
        artifact_uri=str(r["artifact_uri"]),
    )


class WalkForwardModelLoader:
    """Loads the right retrain artifact for each sim bar.

    Per CLAUDE.md §5.13.5 sim/live both call `model_as_of`; the leakage
    invariant is enforced once here, not duplicated downstream.
    """

    def __init__(self, manifest_path: "str | Path") -> None:
        self._manifest_path = _resolve_manifest_path(manifest_path)
        self._entries: list[RetrainEntry] = []
        self._cache: dict[str, "PanelScorer"] = {}
        if self._manifest_path.exists():
            self._entries = self._parse_manifest(self._manifest_path)

    @staticmethod
    def _parse_manifest(path: Path) -> list[RetrainEntry]:
        payload = json.loads(path.read_text())
        if isinstance(payload, dict):
            rows = payload.get("retrains", [])
        else:
            rows = payload
        if not isinstance(rows, list):
            raise ValueError(
                f"WalkForwardModelLoader: manifest at {path} has 'retrains' "
                f"of type {type(rows).__name__}; expected list."
            )
        out = [_parse_entry(r) for r in rows]
        out.sort(key=lambda e: e.cutoff_date)
        return out

    def has_walkforward_model(self) -> bool:
        """True iff the manifest exists and contains ≥ 1 retrain entry."""
        return len(self._entries) > 0

    def model_as_of(self, today: "pd.Timestamp | str") -> "PanelScorer":
        """Return the latest retrain with cutoff_date < today.

        Per the P1 contract this MUST raise ValueError (not silent skip)
        when no eligible retrain exists — sims must abort loudly rather
        than fall back to the look-ahead default.

        Built-in guards (CLAUDE.md §5.13.5):
            * cutoff_date < today (the primary point-in-time guarantee)
            * trained_date >= cutoff_date (manifest construction invariant)
            * assert_no_leakage(cutoff_date, today): single-source helper
              defense in depth — cutoff_date is the upper exclusive bound
              of training data, NOT the wall-clock trained_date (which is
              the moment the retrain script ran and is always ~"now").
        """
        today_ts = pd.Timestamp(today)
        eligible = [e for e in self._entries if e.cutoff_date < today_ts]
        if not eligible:
            raise ValueError(
                f"WalkForwardModelLoader: no retrain with cutoff_date "
                f"< {today_ts.date().isoformat()} in manifest "
                f"{self._manifest_path} (entries={len(self._entries)}). "
                f"Either the sim window starts before the first retrain "
                f"or the manifest is empty."
            )
        chosen = eligible[-1]
        # Built-in invariants per the contract.
        assert chosen.cutoff_date < today_ts, (
            f"WalkForwardModelLoader internal invariant violated: chosen "
            f"cutoff_date {chosen.cutoff_date.isoformat()} >= today "
            f"{today_ts.isoformat()}"
        )
        assert chosen.trained_date >= chosen.cutoff_date, (
            f"WalkForwardModelLoader internal invariant violated: chosen "
            f"trained_date {chosen.trained_date.isoformat()} < cutoff_date "
            f"{chosen.cutoff_date.isoformat()}"
        )
        # §5.13.5 single-source leakage helper. NOTE: pass cutoff_date,
        # not trained_date. The latter is the wall-clock retrain time
        # (~"now" for all entries) and is unrelated to training-data
        # bounds. cutoff_date is the upper exclusive bound on training
        # data — the real leakage barrier for a walk-forward model.
        # AUDIT 2026-05-10 P3.2 sim crash: prior bug passed trained_date
        # which always fired the guard because trained_date=2026-05-10
        # is never < any pre-2026-05-10 sim bar.
        assert_no_leakage(
            chosen.cutoff_date,
            today_ts,
            context=f"WalkForwardModelLoader.model_as_of("
                    f"today={today_ts.date().isoformat()})",
        )
        if chosen.artifact_uri in self._cache:
            return self._cache[chosen.artifact_uri]
        from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
        scorer = PanelScorer.load(chosen.artifact_uri)
        self._cache[chosen.artifact_uri] = scorer
        return scorer

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def entries(self) -> list[RetrainEntry]:
        """Sorted (ascending cutoff_date) view — read-only convenience."""
        return list(self._entries)
