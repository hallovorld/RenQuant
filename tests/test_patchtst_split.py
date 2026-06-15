"""Tests for the train/val split window in patchtst_hf.load_panel_with_split.

ROOT-CAUSE GUARD (2026-06-15): val_tail_pct is a fraction of the WHOLE history,
so on a 10-year panel a 0.10 tail reserves ~1 year for validation and pushes
the effective train-cutoff ~14 months into the past — the live pt07 model's
579-day staleness, its placebo/sanity failure, and its negative OOS IC all
trace to this. --val-days sets a FIXED trailing window so the train-cutoff
stays near the data end regardless of history length.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import patchtst_hf as m  # noqa: E402


def _panel_10yr(tmp_path):
    dates = pd.bdate_range("2016-01-04", "2026-02-10")
    rng = np.random.RandomState(0)
    rows = [
        {"date": d, "ticker": t, "feat1": rng.randn(), "fwd_60d_excess": rng.randn()}
        for t in ("AAA", "BBB", "CCC") for d in dates
    ]
    f = tmp_path / "panel.parquet"
    pd.DataFrame(rows).to_parquet(f)
    return f


def _split(f, **kw):
    p, _ = m.load_panel_with_split(Path(f), "all", "fwd_60d_excess",
                                   preprocess=False, **kw)
    val = sorted(p.loc[p["split_label"] == "val", "date"].unique())
    train = sorted(p.loc[p["split_label"] == "train", "date"].unique())
    embargo = sorted(p.loc[p["split_label"] == "embargo", "date"].unique())
    return val, train, embargo


class TestValWindow:
    def test_val_tail_pct_reserves_a_fraction_of_all_history(self, tmp_path):
        # the bug: 10% of ~2630 trading days ≈ 1 year of val.
        val, train, _ = _split(_panel_10yr(tmp_path), val_tail_pct=0.10)
        n_all = len(val) + len(train) + 0  # embargo separate
        assert 230 <= len(val) <= 290           # ~1 year, not a few months
        assert pd.Timestamp(train[-1]).year <= 2024  # train-cutoff pushed back

    def test_val_days_is_a_fixed_window(self, tmp_path):
        val, train, _ = _split(_panel_10yr(tmp_path), val_days=126)
        assert len(val) == 126                  # exactly the fixed window
        # train-cutoff is materially fresher than the 0.10-tail case
        assert pd.Timestamp(train[-1]) >= pd.Timestamp("2025-04-01")

    def test_val_days_overrides_val_tail_pct(self, tmp_path):
        val, _, _ = _split(_panel_10yr(tmp_path), val_days=126, val_tail_pct=0.10)
        assert len(val) == 126

    def test_embargo_sits_between_train_and_val(self, tmp_path):
        val, train, embargo = _split(_panel_10yr(tmp_path), val_days=126,
                                     embargo_days=60)
        assert embargo, "embargo window must be non-empty"
        assert max(train) < min(embargo) <= max(embargo) < min(val)

    def test_no_val_when_neither_set(self, tmp_path):
        val, train, embargo = _split(_panel_10yr(tmp_path))
        assert val == [] and embargo == [] and len(train) > 0

    def test_val_days_capped_at_history_length(self, tmp_path):
        # absurdly large val_days must not crash or take EVERY date (≥1 stays
        # out of val for train/embargo).
        f = _panel_10yr(tmp_path)
        total = len(sorted(pd.read_parquet(f)["date"].unique()))
        val, _, _ = _split(f, val_days=10**9)
        assert 0 < len(val) <= total - 1
