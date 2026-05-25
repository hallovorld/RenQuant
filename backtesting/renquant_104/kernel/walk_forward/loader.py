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
        calibrator_uri: str | None

    class WalkForwardModelLoader:
        def __init__(self, manifest_path: Path) -> None: ...
        def model_as_of(self, today: pd.Timestamp) -> "PanelScorer":
            "Latest retrain with cutoff_date < today.
             Raises ValueError if none."
        def has_walkforward_model(self) -> bool: ...
"""
from __future__ import annotations

import glob
import hashlib
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
    calibrator_uri: optional filesystem path to the matching
                 GlobalPanelCalibration artifact fitted for this scorer.
    effective_train_cutoff_date: upper exclusive feature-row cutoff when the
                 artifact pre-embargoed labels before selection cutoff. When
                 present, leakage checks use effective_train_cutoff_date +
                 lookahead_days instead of cutoff_date + lookahead_days.
    """
    cutoff_date: pd.Timestamp
    trained_date: pd.Timestamp
    artifact_uri: str
    # 2026-05-11 G1: forward-return horizon used to construct training
    # labels. ``cutoff_date + lookahead_days`` is the upper bound on
    # data the model has "seen" via its labels. Sim bars must satisfy
    # ``today > cutoff_date + lookahead_days`` to avoid leak. Default 0
    # = no forward labels (classification target / point-in-time only).
    lookahead_days: int = 0
    calibrator_uri: str | None = None
    effective_train_cutoff_date: pd.Timestamp | None = None


def _optional_timestamp(raw: object) -> pd.Timestamp | None:
    if raw is None or raw == "":
        return None
    ts = pd.Timestamp(raw)
    if pd.isna(ts):
        return None
    return ts


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
    effective = _optional_timestamp(r.get("effective_train_cutoff_date"))
    if effective is not None and effective > cutoff:
        raise ValueError(
            f"manifest entry leakage: effective_train_cutoff_date "
            f"{effective.isoformat()} > cutoff_date {cutoff.isoformat()}."
        )
    if trained < cutoff:
        raise ValueError(
            f"manifest entry leakage: trained_date {trained.isoformat()} "
            f"< cutoff_date {cutoff.isoformat()} — refusing to load."
        )
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=trained,
        artifact_uri=str(r["artifact_uri"]),
        lookahead_days=int(r.get("lookahead_days", 0)),
        calibrator_uri=(
            str(r.get("calibrator_uri") or r.get("calibration_uri"))
            if (r.get("calibrator_uri") or r.get("calibration_uri"))
            else None
        ),
        effective_train_cutoff_date=effective,
    )


def _normalize_fingerprint(value: str | None) -> str:
    return str(value or "").strip().lower().removeprefix("sha256:")


def _fingerprints_match(expected: str | None, actual: str | None) -> bool:
    """Accept exact matches and historical short-sha prefixes."""
    exp = _normalize_fingerprint(expected)
    act = _normalize_fingerprint(actual)
    if not exp or not act:
        return False
    if exp == act:
        return True
    min_prefix = 12
    return (
        len(exp) >= min_prefix
        and len(act) >= min_prefix
        and (exp.startswith(act) or act.startswith(exp))
    )


def _scorer_fingerprints_from_payload(payload: dict) -> list[str]:
    """Return stable scorer identities stamped in a local artifact JSON."""
    from kernel.panel_pipeline.panel_scorer import model_content_sha256  # noqa: PLC0415

    out: list[str] = []
    try:
        out.append(model_content_sha256(payload))
    except Exception:
        pass
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for source in (payload, metadata):
        for key in (
            "model_content_fingerprint",
            "artifact_fingerprint",
            "artifact_sha256",
            "model_fingerprint",
            "fingerprint",
        ):
            value = source.get(key)
            if value:
                out.append(str(value))
    return list(dict.fromkeys(out))


def _calibrator_scorer_fingerprints(calibrator: object) -> list[str]:
    """Return the scorer identity a calibrator declares it was fitted against."""
    metadata = getattr(calibrator, "metadata", {}) or {}
    out: list[str] = []
    for key in (
        "scorer_model_content_fingerprint",
        "scorer_artifact_fingerprint",
        "scorer_artifact_sha256",
    ):
        value = metadata.get(key)
        if value:
            out.append(str(value))
    return out


def _any_fingerprints_match(expected: list[str], actual: list[str]) -> bool:
    return any(
        _fingerprints_match(exp, act)
        for exp in expected
        for act in actual
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
        self._calibrator_cache: dict[str, object] = {}
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

    @staticmethod
    def _feature_cutoff_date(e: RetrainEntry) -> pd.Timestamp:
        return e.effective_train_cutoff_date or e.cutoff_date

    @staticmethod
    def _safe_last_label_date(e: RetrainEntry) -> pd.Timestamp:
        if e.lookahead_days > 0:
            return (
                WalkForwardModelLoader._feature_cutoff_date(e)
                + pd.tseries.offsets.BDay(e.lookahead_days)
            )
        return WalkForwardModelLoader._feature_cutoff_date(e)

    def entry_as_of(self, today: "pd.Timestamp | str") -> RetrainEntry:
        """Return the latest leakage-safe manifest row for ``today``.

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
        # A fold is eligible only when its feature cutoff plus the forward
        # label horizon is strictly before today. Newer manifests stamp
        # effective_train_cutoff_date when training already pre-embargoed
        # rows before the selection cutoff, avoiding a second lookahead delay.
        eligible = [
            e for e in self._entries
            if self._safe_last_label_date(e) < today_ts
        ]
        if not eligible:
            raise ValueError(
                f"WalkForwardModelLoader: no retrain with cutoff_date / "
                f"feature_cutoff + lookahead_days < "
                f"{today_ts.date().isoformat()} in manifest "
                f"{self._manifest_path} (entries={len(self._entries)}). "
                f"Either the sim window starts before any fold's safe-label "
                f"date or the manifest is empty."
            )
        chosen = eligible[-1]
        # Built-in invariants per the contract.
        assert self._safe_last_label_date(chosen) < today_ts, (
            f"WalkForwardModelLoader internal invariant violated: chosen "
            f"safe_last_label_date {self._safe_last_label_date(chosen).isoformat()} "
            f">= today {today_ts.isoformat()}"
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
        feature_cutoff = self._feature_cutoff_date(chosen)
        # Pass the feature-row cutoff, not the wall-clock trained_date and not
        # the selection cutoff when the artifact declares a pre-embargoed
        # effective_train_cutoff_date.
        assert_no_leakage(
            feature_cutoff,
            today_ts,
            context=f"WalkForwardModelLoader.entry_as_of("
                    f"today={today_ts.date().isoformat()})",
            lookahead_days=chosen.lookahead_days,
        )
        return chosen

    def model_as_of(self, today: "pd.Timestamp | str") -> "PanelScorer":
        """Return the latest retrain scorer for ``today``."""
        chosen = self.entry_as_of(today)
        if chosen.artifact_uri in self._cache:
            return self._cache[chosen.artifact_uri]
        from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
        scorer = PanelScorer.load(chosen.artifact_uri)
        self._cache[chosen.artifact_uri] = scorer
        return scorer

    def calibrator_as_of(self, today: "pd.Timestamp | str"):
        """Return the calibrator matched to ``model_as_of(today)``.

        Walk-forward scorer distributions drift fold by fold. A static
        calibrator is therefore a foreign calibration surface; the manifest
        must bind each scorer artifact to its own point-in-time calibrator.
        """
        chosen = self.entry_as_of(today)
        uri = chosen.calibrator_uri
        if not uri:
            raise ValueError(
                "WalkForwardModelLoader: selected retrain entry has no "
                f"calibrator_uri (cutoff={chosen.cutoff_date.date()}, "
                f"artifact={chosen.artifact_uri}). Rebuild/stamp the "
                "walk-forward manifest with per-fold calibrator artifacts; "
                "do not reuse a static production or sim calibrator."
            )
        if uri in self._calibrator_cache:
            cal = self._calibrator_cache[uri]
            self._assert_calibrator_matches_entry(chosen, cal, uri)
            return cal
        resolved = self._resolve_uri(uri)
        from training_panel.global_calibrator import GlobalPanelCalibration  # noqa: PLC0415
        cal = GlobalPanelCalibration.load(resolved)
        self._assert_calibrator_matches_entry(chosen, cal, resolved)
        self._calibrator_cache[uri] = cal
        return cal

    def _resolve_uri(self, uri: str):
        """Resolve local relative manifest URIs against the manifest folder."""
        if "://" in uri:
            return uri
        p = Path(uri)
        return p if p.is_absolute() else self._manifest_path.parent / p

    def _scorer_fingerprints_for_entry(self, entry: RetrainEntry) -> list[str]:
        """Read the selected fold's local scorer identities without loading it."""
        resolved = self._resolve_uri(entry.artifact_uri)
        if not isinstance(resolved, Path) or not resolved.exists():
            return []
        out: list[str] = []
        if resolved.suffix == ".json":
            payload = json.loads(resolved.read_text())
            if isinstance(payload, dict):
                out.extend(_scorer_fingerprints_from_payload(payload))
        out.append("sha256:" + hashlib.sha256(resolved.read_bytes()).hexdigest())
        return list(dict.fromkeys(out))

    def _assert_calibrator_matches_entry(
        self,
        entry: RetrainEntry,
        calibrator: object,
        calibrator_path: "str | Path",
    ) -> None:
        """Enforce the WF per-fold scorer/calibrator fingerprint contract."""
        scorer_fps = self._scorer_fingerprints_for_entry(entry)
        cal_fps = _calibrator_scorer_fingerprints(calibrator)
        if not scorer_fps or not cal_fps:
            raise ValueError(
                "WalkForwardModelLoader: missing scorer/calibrator fingerprint "
                f"for cutoff={entry.cutoff_date.date()} scorer={entry.artifact_uri} "
                f"calibrator={calibrator_path}. scorer={scorer_fps!r} "
                f"calibrator={cal_fps!r}."
            )
        if not _any_fingerprints_match(cal_fps, scorer_fps):
            raise ValueError(
                "WalkForwardModelLoader: calibrator/scorer fingerprint mismatch "
                f"for cutoff={entry.cutoff_date.date()} scorer={entry.artifact_uri} "
                f"calibrator={calibrator_path}. calibrator={cal_fps} "
                f"scorer={scorer_fps}."
            )

    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    @property
    def entries(self) -> list[RetrainEntry]:
        """Sorted (ascending cutoff_date) view — read-only convenience."""
        return list(self._entries)
