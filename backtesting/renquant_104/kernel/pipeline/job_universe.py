"""LoadUniverseJob — admit tickers into the tradable universe.

Consolidates three previously-duplicated adapter load loops
(LeanAdapter, RunnerAdapter via live/runner._load_strategy_multi,
SimAdapter) into one sequential Task chain so future universe rules
land in exactly one place.

Chain:
    LoadArtifactsTask        walk watchlist, call kernel.models.load_artifact
    FilterStalenessTask      drop artifacts older than model_staleness_days
    FilterUniverseFloorTask  dispatch by ranking.universe_floor.type:
                               - "none"   no filter (default)
                               - "sharpe" metadata.live_holdout_sharpe or .sharpe
                               - "ic"     metadata.panel_oos_ic

New floor types register themselves by adding an entry to FLOOR_EVALUATORS.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

from kernel.config import universe_floor_spec

log = logging.getLogger("kernel.pipeline.universe")


@dataclass
class UniverseContext:
    config:         dict[str, Any]
    strategy_dir:   Path
    loaded_models:  dict[str, dict]          = field(default_factory=dict)
    rejections:     list[tuple[str, str]]    = field(default_factory=list)


class UniverseTask(ABC):
    """Atomic step mutating UniverseContext. Return False to stop the chain."""
    @abstractmethod
    def run(self, uctx: UniverseContext) -> "bool | None": ...
    def should_skip(self, uctx: UniverseContext) -> bool:
        return False
    @property
    def name(self) -> str:
        return type(self).__name__


class LoadArtifactsTask(UniverseTask):
    def run(self, uctx: UniverseContext) -> "bool | None":
        from kernel.models import load_artifact
        models_dir = uctx.strategy_dir / "models"
        if not models_dir.exists():
            log.warning("models/ not found at %s", models_dir)
            return False
        for ticker in uctx.config.get("watchlist", []):
            try:
                art = load_artifact(models_dir / ticker, ticker)
            except Exception as exc:
                log.warning("%s load_artifact failed: %s — rejected", ticker, exc)
                uctx.rejections.append((ticker, f"load_error_{type(exc).__name__}"))
                continue
            if art is None:
                uctx.rejections.append((ticker, "no_artifact"))
                continue
            uctx.loaded_models[ticker] = art
        return True


class FilterStalenessTask(UniverseTask):
    def run(self, uctx: UniverseContext) -> "bool | None":
        staleness_days = int(uctx.config.get("model_staleness_days", 0))
        if staleness_days <= 0:
            return True
        today = date.today()
        stale: list[tuple[str, str]] = []
        for ticker, art in uctx.loaded_models.items():
            meta = art.get("_metadata", {})
            trained = meta.get("trained_date")
            if not trained:
                continue
            try:
                age = (today - datetime.strptime(trained, "%Y-%m-%d").date()).days
            except ValueError:
                continue
            if age > staleness_days:
                stale.append((ticker, f"stale_{age}d_limit_{staleness_days}"))
        for ticker, reason in stale:
            uctx.loaded_models.pop(ticker, None)
            uctx.rejections.append((ticker, reason))
        return True


# ── Floor evaluator registry ──────────────────────────────────────────────────
#
# Each evaluator maps a ticker's artifact metadata → a numeric quality value
# (or None if unavailable). FilterUniverseFloorTask drops a ticker when the
# returned value is below the configured threshold.
#
# To add a new floor type: register an evaluator and the caller sets
# ranking.universe_floor.type to the new name.

def _eval_sharpe(meta: dict) -> "float | None":
    # Prefer live_holdout_sharpe (reflects shipped weights) over tournament
    # sharpe (algorithm-selection metric).
    for key in ("live_holdout_sharpe", "sharpe"):
        v = meta.get(key)
        if v is not None:
            return float(v)
    return None


def _eval_ic(meta: dict) -> "float | None":
    v = meta.get("panel_oos_ic")
    return float(v) if v is not None else None


FLOOR_EVALUATORS: dict[str, Callable[[dict], "float | None"]] = {
    "sharpe": _eval_sharpe,
    "ic":     _eval_ic,
}


class FilterUniverseFloorTask(UniverseTask):
    """Drop tickers whose quality metric (per universe_floor.type) < threshold.

    Missing metric values (`None`) are admitted with a warning — "code-ready"
    for floor types whose metric isn't populated yet. Override this policy by
    changing admit_on_missing to False.
    """
    admit_on_missing: bool = True

    def should_skip(self, uctx: UniverseContext) -> bool:
        floor_type, _ = universe_floor_spec(uctx.config)
        return floor_type == "none"

    def run(self, uctx: UniverseContext) -> "bool | None":
        floor_type, threshold = universe_floor_spec(uctx.config)
        evaluator = FLOOR_EVALUATORS.get(floor_type)
        if evaluator is None:
            log.warning(
                "unknown universe_floor.type=%r (known: %s) — admitting all",
                floor_type, sorted(FLOOR_EVALUATORS.keys()),
            )
            return True
        if threshold <= 0:
            return True
        below: list[tuple[str, str]] = []
        for ticker, art in uctx.loaded_models.items():
            meta = art.get("_metadata", {})
            value = evaluator(meta)
            if value is None:
                if not self.admit_on_missing:
                    below.append((ticker, f"{floor_type}_missing"))
                else:
                    log.warning(
                        "%s %s metric missing — admitting (code-ready)",
                        ticker, floor_type,
                    )
                continue
            if value < threshold:
                below.append(
                    (ticker, f"{floor_type}_{value:.3f}_below_{threshold}")
                )
        for ticker, reason in below:
            uctx.loaded_models.pop(ticker, None)
            uctx.rejections.append((ticker, reason))
        return True


class UniverseJob(ABC):
    @property
    def tasks(self) -> list[UniverseTask]:
        return []
    def run(self, uctx: UniverseContext) -> None:
        for task in self.tasks:
            if task.should_skip(uctx):
                log.debug("[%s] skipped", task.name)
                continue
            if task.run(uctx) is False:
                log.debug("[%s] chain stopped by %s",
                          type(self).__name__, task.name)
                return


class LoadUniverseJob(UniverseJob):
    """Sequential Task chain producing uctx.loaded_models."""
    @property
    def tasks(self) -> list[UniverseTask]:
        return [
            LoadArtifactsTask(),
            FilterStalenessTask(),
            FilterUniverseFloorTask(),
        ]
