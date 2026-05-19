"""Walk-forward train/val splits for time-series cross-validation.

PRIME DIRECTIVE compliance (CLAUDE.md §🔴 + [[feedback_prime_directive_in_objective_funcs]]):
The val period of EACH cut must contain SPIKED regime days. A model
selection process whose val period only sees calm regimes is blind to
crisis behavior — that's how 2026-05-18 DOE got picked apart for using
2023 (all-calm) val.

Default 5-cut configuration covers known objective vol-spike events:
  cut1  val=2020-Q1   COVID crash         (HIGH_SPIKED, LOW_SPIKED)
  cut2  val=2022-Q1   Fed pivot start     (MED_SPIKED)
  cut3  val=2022-Q4   inflation peak      (HIGH_SPIKED)
  cut4  val=2023-Q1   SVB / banking       (MED_SPIKED)
  cut5  val=2024-Q3   election + Aug carry-trade unwind (HIGH_SPIKED)

Each cut: train = all data BEFORE val_start; val = val_start → val_end.
Test (final OOS) = data AFTER all val periods (post 2025-01-01 or later
configurable).

Reference: López de Prado 2018 *Advances in Financial ML* ch.7
(walk-forward backtesting); Bailey-López de Prado 2014 (DSR over CV
folds for multiple-comparison correction).
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


# Default 5-cut spec — known SPIKED-bearing val windows
DEFAULT_CUTS = [
    ("cut1_covid",    "2018-01-01", "2020-01-01", "2020-04-30"),
    ("cut2_fed",      "2018-01-01", "2022-01-01", "2022-04-30"),
    ("cut3_inflpk",   "2018-01-01", "2022-10-01", "2022-12-31"),
    ("cut4_svb",      "2018-01-01", "2023-01-01", "2023-04-30"),
    ("cut5_unwind",   "2018-01-01", "2024-06-01", "2024-09-30"),
]


@dataclass
class WalkForwardCut:
    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # = val_start (exclusive)
    val_start: pd.Timestamp
    val_end: pd.Timestamp

    def __repr__(self) -> str:
        return (f"WFCut({self.name}: train [{self.train_start.date()}, "
                f"{self.train_end.date()}) val [{self.val_start.date()}, "
                f"{self.val_end.date()}])")


def build_default_cuts() -> list[WalkForwardCut]:
    """Returns 5 walk-forward cuts targeting known SPIKED regime events."""
    cuts = []
    for name, train_start, val_start, val_end in DEFAULT_CUTS:
        cuts.append(WalkForwardCut(
            name=name,
            train_start=pd.Timestamp(train_start),
            train_end=pd.Timestamp(val_start),
            val_start=pd.Timestamp(val_start),
            val_end=pd.Timestamp(val_end),
        ))
    return cuts


def assign_split_column(panel: pd.DataFrame, cut: WalkForwardCut,
                        date_col: str = "date") -> pd.Series:
    """Returns a Series with values train / val / test / oos per panel row.

    train: panel[date_col] < cut.val_start
    val:   cut.val_start ≤ panel[date_col] < cut.val_end
    test:  panel[date_col] ≥ cut.val_end  (held-out for final OOS)
    """
    dates = pd.to_datetime(panel[date_col])
    out = pd.Series("test", index=panel.index, dtype="object")
    out.loc[dates < cut.val_start] = "train"
    mask_val = (dates >= cut.val_start) & (dates < cut.val_end)
    out.loc[mask_val] = "val"
    return out


def verify_regime_coverage(cut: WalkForwardCut, spy_path: Path,
                            require_spiked: bool = True) -> dict[str, int]:
    """Verify a cut's val period contains the regimes we care about.

    Returns dict regime → n_days in val. Raises if require_spiked and no
    SPIKED day present.
    """
    from kernel.regime_labels import compute_spy_regime_labels  # noqa: PLC0415

    regimes = compute_spy_regime_labels(spy_path)
    regimes["date"] = pd.to_datetime(regimes["date"])
    val_mask = ((regimes["date"] >= cut.val_start) &
                (regimes["date"] < cut.val_end))
    val_regimes = regimes.loc[val_mask, "regime"]
    counts = val_regimes.value_counts().to_dict()
    if require_spiked:
        spiked = [r for r in counts if "SPIKED" in str(r)]
        if not spiked:
            raise ValueError(
                f"Cut {cut.name} val window [{cut.val_start.date()}, "
                f"{cut.val_end.date()}) has 0 SPIKED days — PRIME DIRECTIVE "
                f"violation. Regimes present: {sorted(counts)}"
            )
    return counts


__all__ = ["WalkForwardCut", "DEFAULT_CUTS", "build_default_cuts",
           "assign_split_column", "verify_regime_coverage"]
