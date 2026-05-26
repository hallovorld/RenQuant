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
        train, feat_cols, _label = TPM.load_and_slice_panel(
            cutoff, cutoff_embargo_days=0,
        )
        assert train["date"].max() < cutoff
        assert len(train) > 0
        # Feature cols exclude the meta + label columns
        for excl in {"ticker", "date", "split_label",
                     "fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"}:
            assert excl not in feat_cols

    def test_default_cutoff_embargo_purges_forward_label_window(self, tmp_path, monkeypatch):
        panel = _make_synthetic_panel(n_tickers=5, n_dates=320, start="2023-01-02")
        (tmp_path / "data").mkdir(exist_ok=True)
        panel.to_parquet(tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet")
        monkeypatch.chdir(tmp_path)

        cutoff = pd.Timestamp("2024-01-02")
        train, _, label = TPM.load_and_slice_panel(cutoff)
        expected_effective = cutoff - pd.offsets.BDay(60)

        assert label == "fwd_60d_excess"
        assert train["date"].max() < expected_effective

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
            train_run_id="abc12345",
        )
        assert art["cutoff_date"] == "2024-01-01T00:00:00"
        assert art["cutoff_embargo_days"] == 60
        assert art["effective_train_cutoff_date"] == (
            pd.Timestamp("2024-01-01") - pd.offsets.BDay(60)
        ).isoformat()
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
            train_run_id="abc12345",
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
            train_run_id="abc12345",
        )
        assert "walkforward_v2_2024-06-01" in art["training_notes"]
        assert "side_label" in art["training_notes"]


class TestStrictContractStamp:
    """Production artifacts must carry machine-checkable OOS evidence."""

    def test_sentiment_gate_zeroes_training_rows_and_stamps_contract(self):
        dates = pd.bdate_range("2024-01-02", periods=2)
        train = pd.DataFrame({
            "ticker": ["AAA", "AAA"],
            "date": dates,
            "mean_sentiment": [0.7, -0.4],
            "n_articles_log": [2.0, 3.0],
            "sentiment_pos_share": [0.8, 0.2],
            "feat_a": [1.0, 2.0],
            "fwd_60d_excess": [0.01, 0.02],
        })
        cfg = {
            "ranking": {
                "panel_scoring": {
                    "sentiment": {
                        "regime_policy": {
                            "BULL_CALM": False,
                            "BEAR": True,
                        }
                    }
                }
            }
        }
        regime_by_date = {
            pd.Timestamp(dates[0]).normalize(): "BULL_CALM",
            pd.Timestamp(dates[1]).normalize(): "BEAR",
        }

        out, meta = TPM.apply_sentiment_training_gate(
            train,
            ["feat_a", "mean_sentiment", "n_articles_log", "sentiment_pos_share"],
            cfg,
            regime_by_date,
        )

        assert out.loc[0, "mean_sentiment"] == 0.0
        assert out.loc[0, "n_articles_log"] == 0.0
        assert out.loc[0, "sentiment_pos_share"] == 0.0
        assert out.loc[1, "mean_sentiment"] == pytest.approx(-0.4)
        assert meta["sentiment_runtime_gate_contract"] == "trained_zeroing"
        assert meta["sentiment_runtime_gate_zeroed_rows"] == 1

    def test_sentiment_gate_zeroes_leading_warmup_missing_regime(self):
        train = pd.DataFrame({
            "ticker": ["AAA"],
            "date": [pd.Timestamp("2024-01-02")],
            "mean_sentiment": [0.7],
            "fwd_60d_excess": [0.01],
        })

        out, meta = TPM.apply_sentiment_training_gate(
            train,
            ["mean_sentiment"],
            {},
            {pd.Timestamp("2024-01-03"): "BEAR"},
        )

        assert out.loc[0, "mean_sentiment"] == 0.0
        assert meta["sentiment_runtime_gate_warmup_zeroed_rows"] == 1
        assert meta["sentiment_runtime_gate_missing_regime_policy"] == "warmup_zero_only"

    def test_sentiment_gate_requires_complete_non_warmup_regime_labels(self):
        train = pd.DataFrame({
            "ticker": ["AAA"],
            "date": [pd.Timestamp("2024-01-04")],
            "mean_sentiment": [0.7],
            "fwd_60d_excess": [0.01],
        })

        with pytest.raises(ValueError, match="missing regime labels"):
            TPM.apply_sentiment_training_gate(
                train,
                ["mean_sentiment"],
                {},
                {pd.Timestamp("2024-01-03"): "BEAR"},
            )

    def test_build_artifact_stamps_strict_contract_fields(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _make_synthetic_panel(n_tickers=3, n_dates=10).dropna(subset=["fwd_60d_excess"])
        cv_result = {
            "cv_method": "purged_walk_forward",
            "cv_n_splits": 3,
            "cv_embargo_days": 60,
            "oos_mean_ic": 0.032,
            "oos_std_ic": 0.004,
            "oos_per_fold_ic": [0.02, 0.03, 0.046],
            "folds": [{"fold": 1, "ic": 0.02}],
        }

        art = TPM.build_artifact(
            booster, ["feat_a"], np.zeros(1), np.ones(1), train,
            cutoff_date=None,
            side_label=None,
            feature_raw_clip_low=[-1.0],
            feature_raw_clip_high=[1.0],
            label_used="fwd_60d_excess",
            train_ic=0.10,
            cv_result=cv_result,
            train_run_id="abc12345",
            sentiment_contract_metadata={
                "sentiment_runtime_gate_contract": "trained_zeroing",
                "sentiment_runtime_gate_zeroed_rows": 12,
            },
        )

        assert art["train_run_id"] == "abc12345"
        assert art["oos_mean_ic"] == pytest.approx(0.032)
        assert art["oos_std_ic"] == pytest.approx(0.004)
        assert art["oos_per_fold_ic"] == [0.02, 0.03, 0.046]
        assert art["cv_method"] == "purged_walk_forward"
        assert art["cv_embargo_days"] == 60
        assert art["training_train_ic"] == pytest.approx(0.10)
        assert art["eval_ic"] == pytest.approx(0.046)
        assert art["feature_raw_clip_low"] == [-1.0]
        assert art["feature_raw_clip_high"] == [1.0]
        assert art["feature_raw_clip_fit_split"] == "train"
        assert art["feature_preprocess_version"] == 2
        assert art["sentiment_runtime_gate_contract"] == "trained_zeroing"
        assert art["sentiment_runtime_gate_zeroed_rows"] == 12

    def test_walkforward_artifact_gets_strict_config_fingerprint(self, tmp_path):
        cfg_path = tmp_path / "strategy_config.wf.json"
        cfg_path.write_text(json.dumps({
            "watchlist": ["AAA", "BBB"],
            "benchmark": "SPY",
            "sector_map": {"AAA": "tech", "BBB": "finance"},
            "sector_etf_map": {"tech": "XLK", "finance": "XLF"},
            "panel_ltr": {
                "lookahead_days": 5,
                "training_resolution": "hourly",
                "hourly": {"enabled": True},
                "minute": {"enabled": True},
                "asset_embeddings": {"enabled": True},
                "xgb_params": {"objective": "rank:ndcg"},
            },
        }))
        art = {"feature_cols": ["feat_a", "feat_b"]}

        fp = TPM.stamp_fingerprint(
            art,
            fingerprint_config_path=str(cfg_path),
            label_used="fwd_60d_excess",
            feat_cols=["feat_a", "feat_b"],
        )

        assert fp.startswith("sha256:")
        assert art["config_fingerprint"] == fp
        fields = art["config_fingerprint_fields"]
        assert fields["watchlist"] == ["AAA", "BBB"]
        assert fields["lookahead_days"] == 60
        assert fields["objective"] == "rank:pairwise"
        assert fields["asset_embeddings"] is False
        assert fields["training_resolution"] == "daily"
        assert fields["hourly_enabled"] is False
        assert fields["minute_enabled"] is False
        assert fields["sector_map"] == {"AAA": "tech", "BBB": "finance"}
        assert fields["sector_etf_map"] == {"finance": "XLF", "tech": "XLK"}

    def test_walk_forward_cv_purges_embargo_before_validation(self, monkeypatch):
        panel = _make_synthetic_panel(n_tickers=8, n_dates=90, start="2023-01-02")
        calls = []

        class FakeBooster:
            def predict(self, _dmatrix):
                return np.arange(len(_dmatrix.get_label() if hasattr(_dmatrix, "get_label") else []))

        def fake_train_xgb(train, feat_cols, label=TPM.LABEL, **_kwargs):
            calls.append(train["date"].max())
            booster = mock.MagicMock()
            booster.predict.side_effect = lambda dmat: np.arange(dmat.num_row())
            return booster, 0.1

        monkeypatch.setattr(TPM, "train_xgb", fake_train_xgb)
        result = TPM.evaluate_walk_forward_cv(
            panel,
            ["feat_a", "feat_b"],
            label="fwd_60d_excess",
            n_splits=2,
            embargo_days=5,
        )

        assert len(result["oos_per_fold_ic"]) == 2
        for fold in result["folds"]:
            train_end = pd.Timestamp(fold["train_end"])
            val_start = pd.Timestamp(fold["val_start"])
            assert train_end < val_start


class TestBackwardCompat:
    """Invoking with no flags → default output path + no cutoff metadata."""

    def test_default_output_path_unchanged(self):
        args = argparse.Namespace(train_cutoff=None, output_path=None, side_label=None)
        _, out, is_wf = TPM.resolve_paths(args)
        assert is_wf is False
        # backward-compat: default path is data/panel-ltr-prod-alpha158-fund-fwd60d.json
        assert out.name == "panel-ltr-prod-alpha158-fund-fwd60d.json"
        assert out.parent.name == "data"

    def test_walkforward_flag_inferred_for_safe_output_path(self):
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
        train_cut, _, _ = TPM.load_and_slice_panel(
            pd.Timestamp("2024-01-01"), cutoff_embargo_days=0,
        )
        assert train_cut["date"].nunique() < train_full["date"].nunique()
        assert len(train_cut) < len(train_full)
        # The bug class this prevents: walkforward training using the
        # FULL panel because the cutoff filter was a no-op.
        assert train_cut["date"].max() < pd.Timestamp("2024-01-01")
