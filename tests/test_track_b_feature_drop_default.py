"""Track B (post renquant-base-data #16 rename): pin two contracts on
``scripts/train_production_model.py::load_and_slice_panel``:

1. A panel that carries the renamed Track B column ``idio_vol_market`` is
   DROPPED by default (no ``--include-features``). This preserves the
   baseline 172-feature recipe even when the upstream parquet ships with
   Track B columns.
2. ``--include-features`` naming a column NOT present in the panel
   (e.g. the OLD pre-rename name ``idio_vol_3f``) fails LOUDLY via
   ``SystemExit`` — no silent rename translation. The honest design choice
   here is to surface the upstream rename to the operator instead of
   silently producing a baseline-equivalent run, per CLAUDE.md §7.13
   "every fix names the invariant that prevents the entire bug class".

§7.2.1 R3 / §3.5: paired with renquant-model#29 ``TRACK_B_FEATURES`` constant
update. Audit memo: doc/research/2026-06-02-track-b-feature-audit.md.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_production_model import (  # noqa: E402
    TRACK_B_FEATURES,
    load_and_slice_panel,
)


def _synthetic_panel_with_renamed_track_b() -> pd.DataFrame:
    """Tiny panel carrying the post-#16 ``idio_vol_market`` column plus
    a baseline alpha158 column and the fwd_60d_excess label. Mimics the
    shape ``load_and_slice_panel`` parses out of the production parquet.
    """
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    rows = []
    for d in dates:
        for t in ["AAA", "BBB", "CCC"]:
            rows.append({
                "ticker": t,
                "date": d,
                "split_label": "train",
                "alpha158_some_col": 0.1,
                "mom_carry_12_1": 0.02,
                "beta_dm": 0.8,
                "rvar_total": 0.01,
                "idio_vol_market": 0.015,
                "fwd_60d_excess": 0.001,
            })
    return pd.DataFrame(rows)


def test_track_b_constant_uses_renamed_column():
    """The post-#16 ``idio_vol_market`` name MUST be in the constant; the
    old misnomer ``idio_vol_3f`` MUST NOT. Pins the rename at the source of
    truth that ``load_and_slice_panel`` consults.
    """
    assert "idio_vol_market" in TRACK_B_FEATURES
    assert "idio_vol_3f" not in TRACK_B_FEATURES


def test_default_drops_renamed_track_b_column_from_feat_cols():
    """When ``include_features`` is None and the panel parquet carries
    ``idio_vol_market`` (the post-#16 name), the column MUST be dropped from
    ``feat_cols`` so the baseline 172-feature recipe is preserved.
    """
    panel = _synthetic_panel_with_renamed_track_b()
    with patch("scripts.train_production_model.pd.read_parquet", return_value=panel):
        _train, feat_cols, _label = load_and_slice_panel(
            cutoff_date=None,
            watchlist_file=None,
            label_override=None,
            cutoff_embargo_days=None,
            include_features=None,
        )
    assert "idio_vol_market" not in feat_cols, (
        "default (no --include-features) MUST drop the renamed Track B column "
        "to preserve the baseline-172 recipe"
    )
    # Baseline alpha158 column survives.
    assert "alpha158_some_col" in feat_cols


def test_include_features_keeps_renamed_track_b_column():
    """When the operator opts in by passing the new name, the column is
    KEPT in ``feat_cols``.
    """
    panel = _synthetic_panel_with_renamed_track_b()
    with patch("scripts.train_production_model.pd.read_parquet", return_value=panel):
        _train, feat_cols, _label = load_and_slice_panel(
            cutoff_date=None,
            watchlist_file=None,
            label_override=None,
            cutoff_embargo_days=None,
            include_features=list(TRACK_B_FEATURES),
        )
    assert "idio_vol_market" in feat_cols
    assert "mom_carry_12_1" in feat_cols
    assert "beta_dm" in feat_cols
    assert "rvar_total" in feat_cols


def test_include_features_with_stale_pre_rename_name_fails_loudly():
    """The OLD name ``idio_vol_3f`` is no longer in the renquant-base-data
    panel (renamed to ``idio_vol_market`` in #16). Passing the stale name
    via ``--include-features`` MUST raise ``SystemExit`` instead of silently
    producing a baseline-equivalent run that drops every Track B column.
    The error message MUST mention the rename to surface it to the operator.
    """
    panel = _synthetic_panel_with_renamed_track_b()
    with patch("scripts.train_production_model.pd.read_parquet", return_value=panel):
        with pytest.raises(SystemExit) as exc:
            load_and_slice_panel(
                cutoff_date=None,
                watchlist_file=None,
                label_override=None,
                cutoff_embargo_days=None,
                include_features=["mom_carry_12_1", "beta_dm", "rvar_total", "idio_vol_3f"],
            )
    msg = str(exc.value)
    assert "idio_vol_3f" in msg, (
        f"error must name the stale feature; got: {msg!r}"
    )
    # The hint surfaces the upstream rename so the operator can fix the CLI.
    assert "idio_vol_market" in msg, (
        f"error must hint at the renquant-base-data #16 rename to "
        f"'idio_vol_market'; got: {msg!r}"
    )


def test_include_features_with_unrelated_typo_also_fails_loudly():
    """The fail-loud check is general — not just for the known rename. Any
    name not in the panel produces ``SystemExit``. Pins the conservative
    no-silent-translation contract.
    """
    panel = _synthetic_panel_with_renamed_track_b()
    with patch("scripts.train_production_model.pd.read_parquet", return_value=panel):
        with pytest.raises(SystemExit) as exc:
            load_and_slice_panel(
                cutoff_date=None,
                watchlist_file=None,
                label_override=None,
                cutoff_embargo_days=None,
                include_features=["mom_carry_12_1", "not_a_real_feature"],
            )
    assert "not_a_real_feature" in str(exc.value)
