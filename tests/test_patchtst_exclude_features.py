"""Tests for patchtst_hf.load_panel_with_split(--exclude-feature-prefixes).

The WF sanity placebo flags the slow-volatility / drawdown factor family
(STD*/MIN*/IMIN*) as cross-sectional drift surviving a 120d label shift. This
knob ablates that family by feature-name prefix so a retrain can isolate whether
the model's edge is genuine 60d signal or that slow drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import patchtst_hf as m  # noqa: E402


def _panel(tmp_path):
    dates = pd.bdate_range("2024-01-01", periods=40)
    rng = np.random.RandomState(0)
    cols = ("STD60", "STD20", "MIN30", "IMIN10", "MOM12", "RSI14")
    rows = [
        {"date": d, "ticker": t, "fwd_60d_excess": rng.randn(),
         **{c: rng.randn() for c in cols}}
        for t in ("AAA", "BBB", "CCC") for d in dates
    ]
    f = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(f)
    return f


def _feat_cols(f, **kw):
    _, feat_cols = m.load_panel_with_split(
        Path(f), "all", "fwd_60d_excess", preprocess=False, **kw)
    return feat_cols


def test_no_exclusion_keeps_all_features(tmp_path):
    feat = _feat_cols(_panel(tmp_path))
    assert set(feat) == {"STD60", "STD20", "MIN30", "IMIN10", "MOM12", "RSI14"}


def test_exclude_slow_factor_family(tmp_path):
    feat = _feat_cols(_panel(tmp_path),
                      exclude_feature_prefixes=("STD", "MIN", "IMIN"))
    assert set(feat) == {"MOM12", "RSI14"}


def test_exclude_is_prefix_based(tmp_path):
    # "MIN" drops MIN30 AND IMIN10? No — IMIN10 does not start with "MIN".
    feat = _feat_cols(_panel(tmp_path), exclude_feature_prefixes=("MIN",))
    assert "MIN30" not in feat
    assert "IMIN10" in feat  # prefix match is from the start of the name


def test_empty_prefixes_is_noop(tmp_path):
    feat = _feat_cols(_panel(tmp_path), exclude_feature_prefixes=())
    assert len(feat) == 6


def test_excluding_all_features_fails_loud(tmp_path):
    with pytest.raises(ValueError, match="removed ALL features"):
        _feat_cols(_panel(tmp_path),
                   exclude_feature_prefixes=("STD", "MIN", "IMIN", "MOM", "RSI"))


def test_cli_threads_comma_string_to_tuple(tmp_path):
    # The CLI passes a comma string; train_one parses it. Mirror that parse here.
    raw = "STD, MIN ,IMIN"
    parsed = tuple(p.strip() for p in raw.split(",") if p.strip())
    feat = _feat_cols(_panel(tmp_path), exclude_feature_prefixes=parsed)
    assert set(feat) == {"MOM12", "RSI14"}
