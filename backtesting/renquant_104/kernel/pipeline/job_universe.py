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
    # Broker tag for state-file isolation (mirrors InferenceContext.broker_name).
    # None for sim/lean paths; live runs set it from broker.broker_name.
    broker_name:    str | None              = None
    # Authoritative held tickers from the broker. When present, this wins over
    # live_state-derived position_hwm keys so stale state cannot grant held
    # exemptions to flat tickers.
    held_tickers:   set[str] | None          = None
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
        held = _held_tickers_for_context(uctx)
        for ticker, art in uctx.loaded_models.items():
            meta = art.get("_metadata", {})
            trained = meta.get("trained_date")
            if not trained:
                if ticker in held:
                    log.warning(
                        "%s HELD — admitting despite missing trained_date "
                        "(so sell path stays armed)",
                        ticker,
                    )
                    continue
                stale.append((ticker, "trained_date_missing"))
                continue
            try:
                age = (today - datetime.strptime(trained, "%Y-%m-%d").date()).days
            except ValueError:
                if ticker in held:
                    log.warning(
                        "%s HELD — admitting despite invalid trained_date=%r "
                        "(so sell path stays armed)",
                        ticker, trained,
                    )
                    continue
                stale.append((ticker, "trained_date_invalid"))
                continue
            if age > staleness_days:
                if ticker in held:
                    log.warning(
                        "%s HELD — admitting despite stale trained_date=%s "
                        "(age=%dd > limit=%dd, so sell path stays armed)",
                        ticker, trained, age, staleness_days,
                    )
                    continue
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
    # Prefer tournament `sharpe` (full walk-forward OOS, typically ~2yr) over
    # `live_holdout_sharpe`. The holdout Sharpe uses only ~126 trading days,
    # which is too short to be statistically stable: a single volatile stretch
    # flips signs for many tickers. When the holdout Sharpe disagrees sharply
    # with the tournament Sharpe, the gap is noise, not signal.
    for key in ("sharpe", "live_holdout_sharpe"):
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


def _load_held_tickers(
    strategy_dir: Path, broker_name: str | None = None,
) -> set[str]:
    """Read ``live_state.{broker}.json::position_hwm`` → set of currently-held tickers.

    Used by `FilterUniverseFloorTask` to EXEMPT held tickers from the
    quality floor. Rationale: universe_floor is meant to gate OFFENSIVE
    new buys from weak models. For already-held positions, filtering out
    the per-ticker model removes the ONLY source of the
    `model_sell_streak` exit signal — in `task_sell.py::ScoreModelTask`,
    `tc.model is None → model_action = "hold"` forever. The position is
    then stuck until a non-model exit (stop_loss / trailing / max_hold)
    fires, which may never happen for a flat low-vol holding.

    Real incident (2026-04-23): AMZN held at cost $249, model sharpe
    slipped 0.668 → below 1.0 floor → model dropped → AMZN became
    structurally un-exitable via signals.

    2026-04-27: switched to broker-isolated path so paper smoke positions
    don't contaminate alpaca-live admission. Falls back to the legacy
    ``live_state.json`` when the broker-specific file does not yet exist
    (one-time read during migration).
    """
    import json as _j
    from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
    state_file, _legacy = resolve_live_state_read(strategy_dir, broker_name)
    if not state_file.exists():
        return set()
    try:
        data = _j.loads(state_file.read_text())
    except Exception:
        return set()
    # Prefer position_hwm keys (only non-zero positions get entries).
    return set((data.get("position_hwm") or {}).keys())


def _held_tickers_for_context(uctx: UniverseContext) -> set[str]:
    if uctx.held_tickers is not None:
        return {str(t) for t in uctx.held_tickers if t}
    return _load_held_tickers(uctx.strategy_dir, uctx.broker_name)


class FilterUniverseFloorTask(UniverseTask):
    """Drop tickers whose quality metric (per universe_floor.type) < threshold.

    Missing metric values (`None`) fail closed for offensive new-buy names.
    A missing quality metric is missing model evidence, not a weaker fallback.

    **Always exempt (admitted regardless of floor):**

      1. `config.defensive_tickers` — they exist specifically to be
         available when the regime demands them (BEAR / bear_only
         branch). Filtering them out here would make BEAR buys
         structurally impossible.
      2. **Currently-held tickers** (broker snapshot for live, state-file
         fallback for legacy/sim contexts). The
         floor is designed to gate OFFENSIVE new buys from weak models;
         dropping a held position's model kills the `model_sell_streak`
         exit path (ScoreModelTask → tc.model=None → action="hold"
         forever). 2026-04-23 incident: AMZN sharpe=0.668 got filtered,
         turning AMZN into a structurally un-sellable position.
    """
    admit_on_missing: bool = False

    def should_skip(self, uctx: UniverseContext) -> bool:
        floor_type, _ = universe_floor_spec(uctx.config)
        return floor_type == "none"

    def run(self, uctx: UniverseContext) -> "bool | None":
        floor_type, threshold = universe_floor_spec(uctx.config)
        evaluator = FLOOR_EVALUATORS.get(floor_type)
        if evaluator is None:
            raise ValueError(
                f"unknown universe_floor.type={floor_type!r} "
                f"(known: {sorted(FLOOR_EVALUATORS.keys())})"
            )
            return True
        if threshold <= 0:
            return True
        defensives = set(uctx.config.get("defensive_tickers", []) or [])
        held       = _held_tickers_for_context(uctx)
        below: list[tuple[str, str]] = []
        held_admitted: list[tuple[str, float]] = []
        for ticker, art in uctx.loaded_models.items():
            if ticker in defensives:
                continue   # always admit defensives — see class docstring
            if ticker in held:
                # Always admit held positions so model-sell path stays
                # armed. Log sharpe for audit (if sub-floor we're keeping
                # the model anyway but flagging it).
                meta = art.get("_metadata", {})
                v = evaluator(meta)
                if v is not None and v < threshold:
                    held_admitted.append((ticker, v))
                continue
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
        for ticker, v in held_admitted:
            log.warning(
                "%s HELD — admitting despite %s=%.3f < %s (so sell path stays armed)",
                ticker, floor_type, v, threshold,
            )
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


class FilterAutoDropTask(UniverseTask):
    """Drop tickers that have been filtered out for >= N consecutive days.

    User feature 2026-04-24: a ticker that the pipeline filters out (no
    candidate emerges past A-gate / sector / corr / etc) for 3 months
    is functionally dead — kicking it from the watchlist saves training
    compute and panel-feature noise. State is persisted via
    `monitor_state["filter_streaks"]: dict[ticker, int]`. Each bar:

      * if ticker appears in ctx.candidates (passed at least one filter)
        → reset to 0
      * else → increment

    Drop happens at universe-load time when streak >= threshold.

    Config flag: `monitoring.auto_drop_filter_days` (default 0 = off).
    Per CLAUDE.md §2a, this is a defensive cleanup feature, not an alpha
    change — defaults preserve existing behaviour.
    """

    def should_skip(self, uctx: UniverseContext) -> bool:
        threshold = int(uctx.config.get("monitoring", {})
                          .get("auto_drop_filter_days", 0) or 0)
        return threshold <= 0

    def run(self, uctx: UniverseContext) -> "bool | None":
        # Audit fix AUTO-DROP-NULL (Round 2 deep audit, 2026-04-25):
        # pre-fix `int(...get("auto_drop_filter_days", 0))` would raise
        # TypeError if the config has the key explicitly set to null
        # (vs. unset). should_skip uses `or 0` fallback consistently;
        # match it here so explicit-null + explicit-0 + missing-key
        # all behave the same.
        threshold = int(uctx.config.get("monitoring", {})
                          .get("auto_drop_filter_days", 0) or 0)
        # Read streaks from live state file (RunnerAdapter writes this);
        # SimAdapter passes through monitor_state on each bar.
        streaks: dict[str, int] = {}
        if uctx.strategy_dir is not None:
            from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
            ls_path, _legacy = resolve_live_state_read(
                uctx.strategy_dir, uctx.broker_name,
            )
            if ls_path.exists():
                try:
                    import json as _json
                    state = _json.loads(ls_path.read_text())
                    ms    = state.get("monitor_state", {}) or {}
                    streaks = ms.get("filter_streaks", {}) or {}
                except Exception as exc:
                    log.warning("auto_drop: live_state read failed: %s", exc)

        defensives = set(uctx.config.get("defensive_tickers", []) or [])
        held = _held_tickers_for_context(uctx)
        dropped = []
        for ticker, art in list(uctx.loaded_models.items()):
            if ticker in defensives or ticker in held:
                continue
            n = int(streaks.get(ticker, 0))
            if n >= threshold:
                uctx.loaded_models.pop(ticker, None)
                uctx.rejections.append((ticker, f"auto_drop_{n}d_filter_streak"))
                dropped.append((ticker, n))
        if dropped:
            log.warning("auto_drop: %d ticker(s) dropped for filter-streak >= %dd: %s",
                        len(dropped), threshold,
                        ", ".join(f"{t}({n}d)" for t, n in dropped))
        return True


class LoadUniverseJob(UniverseJob):
    """Sequential Task chain producing uctx.loaded_models."""
    @property
    def tasks(self) -> list[UniverseTask]:
        return [
            LoadArtifactsTask(),
            FilterStalenessTask(),
            FilterUniverseFloorTask(),
            FilterAutoDropTask(),
        ]
