"""Track P3-v2 — train_production_model.py CLI refactor regression tests.

Pins the §5.13.5 invariant (single source of truth: walk-forward and
daily prod retrain both go through train_production_model.py) and the
§5.13.13 invariant (refuse to overwrite production artifact when a
train-cutoff is set).

These tests use heavy monkey-patching to avoid loading the real 715k-row
panel + xgboost training. We monkey-patch the I/O + training entry
points and assert the resolve_paths / load_and_slice_panel /
build_artifact behavior with synthetic data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import train_production_model as TPM  # noqa: E402


# ───── synthetic panel fixture ─────

def _make_synthetic_panel(n_tickers: int = 5, n_dates: int = 40,
                          start: str = "2023-01-01") -> pd.DataFrame:
    """A small alpha158-shaped panel: ticker, date, 3 feat cols, 3 label cols."""
    dates = pd.bdate_range(start, periods=n_dates)
    rows = []
    rng = np.random.default_rng(42)
    for t_idx in range(n_tickers):
        tk = f"T{t_idx:02d}"
        for d in dates:
            rows.append({
                "ticker": tk,
                "date": d,
                "split_label": "train",
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                "earnings_yield": rng.normal(),  # fund col
                "fwd_5d_excess": rng.normal() * 0.01,
                "fwd_20d_excess": rng.normal() * 0.02,
                "fwd_60d_excess": rng.normal() * 0.03,
            })
    return pd.DataFrame(rows)


def _make_synthetic_stats(feat_cols: list[str], tmp_dir: Path) -> None:
    """Stub alpha158_qlib_dataset.stats.json so build_normalization works."""
    stats = {
        "feature_cols": feat_cols,
        "feature_means": [0.0] * len(feat_cols),
        "feature_stds":  [1.0] * len(feat_cols),
    }
    (tmp_dir / "data").mkdir(parents=True, exist_ok=True)
    (tmp_dir / "data" / "alpha158_qlib_dataset.stats.json").write_text(json.dumps(stats))


def _make_synthetic_fund(tickers: list[str], dates: pd.DatetimeIndex,
                         tmp_dir: Path) -> None:
    """Stub data/sec_fundamentals_daily.parquet so build_normalization works."""
    rows = []
    rng = np.random.default_rng(0)
    for tk in tickers:
        for d in dates:
            rows.append({
                "ticker": tk, "date": d,
                "earnings_yield": rng.normal(),
                "book_to_price": rng.normal(),
                "gross_profitability": rng.normal(),
                "roe": rng.normal(),
                "asset_growth": rng.normal(),
            })
    df = pd.DataFrame(rows)
    (tmp_dir / "data").mkdir(parents=True, exist_ok=True)
    df.to_parquet(tmp_dir / "data" / "sec_fundamentals_daily.parquet")


# ───── argument resolution tests (no panel load needed) ─────

class TestProdPathProtection:
    """§5.13.13: --train-cutoff set without walkforward path → reject."""

    def _args(self, **kw) -> argparse.Namespace:
        defaults = {"train_cutoff": None, "output_path": None, "side_label": None}
        defaults.update(kw)
        return argparse.Namespace(**defaults)

    def test_cutoff_without_output_path_rejected(self):
        args = self._args(train_cutoff="2024-01-01")
        with pytest.raises(SystemExit, match="requires --output-path"):
            TPM.resolve_paths(args)

    def test_cutoff_with_prod_output_path_rejected(self):
        args = self._args(
            train_cutoff="2024-01-01",
            output_path="data/panel-ltr-prod-alpha158-fund-fwd60d.json",
            side_label="x",
        )
        with pytest.raises(SystemExit, match="walkforward"):
            TPM.resolve_paths(args)

    def test_cutoff_without_side_label_rejected(self):
        args = self._args(
            train_cutoff="2024-01-01",
            output_path="artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
        )
        with pytest.raises(SystemExit, match="side-label"):
            TPM.resolve_paths(args)

    def test_cutoff_with_walkforward_path_accepted(self):
        args = self._args(
            train_cutoff="2024-01-01",
            output_path="artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
            side_label="walkforward_v2_2024-01-01",
        )
        cutoff, out, is_wf = TPM.resolve_paths(args)
        assert cutoff == pd.Timestamp("2024-01-01")
        assert is_wf is True
        assert "walkforward" in str(out)

    def test_no_cutoff_defaults_to_prod_path(self):
        args = self._args()
        cutoff, out, is_wf = TPM.resolve_paths(args)
        assert cutoff is None
        assert is_wf is False
        assert out == TPM.DEFAULT_OUTPUT


# ───── slicing tests (no xgboost) ─────

class TestCutoffSlicing:
    """With dates 2023-2024, --train-cutoff 2024-01-01 → max date < cutoff."""

    def test_slice_uses_dates_strictly_before_cutoff(self, tmp_path, monkeypatch):
        panel = _make_synthetic_panel(n_tickers=5, n_dates=60, start="2023-11-01")
        full_path = tmp_path / "alpha158_291_fundamental_dataset.parquet"
        panel.to_parquet(full_path)
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data").mkdir(exist_ok=True)
        panel.to_parquet(tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet")

        cutoff = pd.Timestamp("2024-01-01")
        train, feat_cols, _label = TPM.load_and_slice_panel(cutoff)
        assert train["date"].max() < cutoff
        assert len(train) > 0
        # Feature cols exclude the meta + label columns
        for excl in {"ticker", "date", "split_label",
                     "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}:
            assert excl not in feat_cols

    def test_no_cutoff_uses_all_labeled_rows(self, tmp_path, monkeypatch):
        panel = _make_synthetic_panel(n_tickers=5, n_dates=40, start="2023-01-01")
        (tmp_path / "data").mkdir(exist_ok=True)
        panel.to_parquet(tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet")
        monkeypatch.chdir(tmp_path)
        train, _, _ = TPM.load_and_slice_panel(None)
        # All synthetic rows have labels → train == panel (size-wise)
        assert len(train) == len(panel)


# ───── artifact stamping tests ─────

class TestArtifactStampedCutoff:
    """Artifact JSON contains cutoff_date when --train-cutoff was set."""

    def _mini_train_df(self) -> pd.DataFrame:
        panel = _make_synthetic_panel(n_tickers=5, n_dates=30, start="2023-06-01")
        return panel.dropna(subset=["fwd_60d_excess"])

    def test_cutoff_appears_in_artifact(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = self._mini_train_df()
        feat_cols = ["feat_a", "feat_b"]
        mu = np.zeros(len(feat_cols))
        sd = np.ones(len(feat_cols))
        art = TPM.build_artifact(
            booster, feat_cols, mu, sd, train,
            cutoff_date=pd.Timestamp("2024-01-01"),
            side_label="walkforward_v2_2024-01-01",
        )
        assert art["cutoff_date"] == "2024-01-01T00:00:00"
        assert art["side_label"] == "walkforward_v2_2024-01-01"
        assert "panel_shape" in art and isinstance(art["panel_shape"], dict)
        assert art["panel_shape"]["rows"] == len(train)

    def test_no_cutoff_no_cutoff_field(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = self._mini_train_df()
        art = TPM.build_artifact(
            booster, ["feat_a"], np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
        )
        assert "cutoff_date" not in art
        assert "side_label" not in art


class TestSideLabelInArtifact:
    """--side-label appears in training_notes."""

    def test_side_label_in_training_notes(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _make_synthetic_panel(n_tickers=3, n_dates=10).dropna(subset=["fwd_60d_excess"])
        art = TPM.build_artifact(
            booster, ["feat_a"], np.zeros(1), np.ones(1), train,
            cutoff_date=pd.Timestamp("2024-06-01"),
            side_label="walkforward_v2_2024-06-01",
        )
        assert "walkforward_v2_2024-06-01" in art["training_notes"]
        assert "side_label" in art["training_notes"]


class TestBackwardCompat:
    """Invoking with no flags → default output path + no cutoff metadata."""

    def test_default_output_path_unchanged(self):
        args = argparse.Namespace(train_cutoff=None, output_path=None, side_label=None)
        _, out, is_wf = TPM.resolve_paths(args)
        assert is_wf is False
        # backward-compat: default path is data/panel-ltr-prod-alpha158-fund-fwd60d.json
        assert out.name == "panel-ltr-prod-alpha158-fund-fwd60d.json"
        assert out.parent.name == "data"

    def test_walkforward_skip_fingerprint_flag_inferred(self):
        """Walkforward artifacts skip fingerprint stamp (per resolve_paths)."""
        args = argparse.Namespace(
            train_cutoff="2024-01-01",
            output_path="artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
            side_label="walkforward_v2_2024-01-01",
        )
        _, _, is_wf = TPM.resolve_paths(args)
        assert is_wf is True


# ───── audit regression guard (§5.13.3) ─────

class TestAuditP3v2Regression:
    """Pin: --train-cutoff=2024-01-01 produces a smaller panel than full."""

    def test_cutoff_strictly_reduces_dates(self, tmp_path, monkeypatch):
        panel = _make_synthetic_panel(n_tickers=5, n_dates=60, start="2023-11-01")
        (tmp_path / "data").mkdir(exist_ok=True)
        panel.to_parquet(tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet")
        monkeypatch.chdir(tmp_path)
        train_full, _, _ = TPM.load_and_slice_panel(None)
        train_cut, _, _ = TPM.load_and_slice_panel(pd.Timestamp("2024-01-01"))
        assert train_cut["date"].nunique() < train_full["date"].nunique()
        assert len(train_cut) < len(train_full)
        # The bug class this prevents: walkforward training using the
        # FULL panel because the cutoff filter was a no-op.
        assert train_cut["date"].max() < pd.Timestamp("2024-01-01")
