"""Per-day per-position snapshot buffer for meta-label training data.

The SnapshotLogger is owned by the adapter (SimAdapter / RunnerAdapter
in training mode). The pipeline's :class:`SnapshotHoldingsTask` calls
``logger.record(row)`` for every (held_ticker, today) tuple per bar.
On adapter teardown, ``logger.dump_to_parquet(path)`` writes the buffer
to disk.

FEATURE_COLUMNS pins the canonical column ordering — the same schema is
consumed by ``scripts/_meta_label_generate.py`` (triple-barrier label
join) and ``scripts/_meta_label_train.py`` (XGBoost feature_cols field
in the artifact). Drift between this list and either downstream script
is a §5.13.13 violation.

References
----------
* López de Prado 2018 *Advances in Financial Machine Learning* ch.20
* doc/research/meta-labeling-exit-policy.md §4 feature taxonomy
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence


# ── Canonical schema (must match downstream training scripts) ─────────
# Grouped by §4 of the meta-labeling design doc.
FEATURE_COLUMNS: tuple[str, ...] = (
    # Row identity
    "date", "ticker",
    # Position state
    "cum_pnl_pct", "peak_gain_pct", "drawdown_from_peak_pct",
    "days_held", "consec_underwater_days",
    "prev_day_return", "gap_open_pct", "realized_vol_20d",
    # Market state
    "spy_5d_ret", "spy_20d_ret", "spy_60d_ret",
    "spy_realized_vol_20d",
    "regime_code",                # 0=BULL_CALM 1=BULL_VOLATILE 2=CHOPPY 3=BEAR
    "regime_just_switched",       # 1 if regime changed in last 5 bars else 0
    "regime_confidence",
    # Model signal
    "panel_score_current", "panel_score_at_entry", "panel_score_delta",
    "panel_score_rank_among_holdings",
    "mu_current", "sigma_current",   # NaN when NGBoost off
    # Portfolio context
    "position_weight", "sector_concentration",
    "portfolio_drawdown_now", "n_concurrent_exits_this_bar",
    # Path-rule signal (what fired this bar)
    "trigger_stop_loss", "trigger_trailing_stop",
    "trigger_single_day_loss", "trigger_max_hold",
    "any_trigger",
    # Outcome stub — populated by triple-barrier labeler (P4.2), kept here
    # so the logger writes a single canonical schema even before labelling.
    "fwd_5d_ret", "fwd_20d_ret",
)


REQUIRED_FIELDS: frozenset = frozenset(("ticker", "date"))


class SnapshotLogger:
    """In-memory per-bar buffer; dumps to parquet on teardown.

    Design constraints:
      * Schema is strict (FEATURE_COLUMNS) — unknown keys raise; missing
        required keys (ticker, date) raise. Other features may be None /
        NaN and the parquet writer keeps them as NaN. Strict-schema
        prevents §5.13.14 ("hardcoded artifact filename" cousin —
        silent column drift between writer and trainer).
      * Append-only — no per-row mutation API. Sim bars are immutable.
      * Cheap construction: empty buffer + frozenset checks; no I/O.
    """

    __slots__ = ("_rows", "_column_set")

    def __init__(self) -> None:
        self._rows: list[dict] = []
        self._column_set: frozenset = frozenset(FEATURE_COLUMNS)

    # ── Public API ─────────────────────────────────────────────────────

    def record(self, row: dict[str, Any]) -> None:
        """Append one (position, day) snapshot row.

        Raises ValueError on unknown / missing-required keys per the
        strict-schema invariant.
        """
        if not isinstance(row, dict):
            raise TypeError(f"row must be dict, got {type(row).__name__}")
        unknown = set(row.keys()) - self._column_set
        if unknown:
            raise ValueError(
                f"unknown keys in snapshot row: {sorted(unknown)} "
                f"(schema = FEATURE_COLUMNS)"
            )
        missing_required = REQUIRED_FIELDS - set(row.keys())
        if missing_required:
            raise ValueError(
                f"missing required keys in snapshot row: "
                f"{sorted(missing_required)}"
            )
        # Defensive copy (caller might mutate the dict)
        self._rows.append(dict(row))

    def n_rows(self) -> int:
        return len(self._rows)

    def dump_to_parquet(self, out_path: "str | Path") -> None:
        """Write the buffer to a parquet file with canonical schema.

        Empty buffer still writes a schema-only parquet so downstream
        joins / readers don't crash when no data was logged.
        """
        import pandas as pd  # noqa: PLC0415
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if self._rows:
            df = pd.DataFrame(self._rows)
            # Reindex to enforce the canonical column order;
            # missing columns become NaN.
            df = df.reindex(columns=list(FEATURE_COLUMNS))
        else:
            df = pd.DataFrame({col: [] for col in FEATURE_COLUMNS})
        df.to_parquet(out, index=False)
