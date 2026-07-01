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
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from scripts import train_production_model as TPM  # noqa: E402
from scripts import restamp_prod_fingerprint as RESTAMP  # noqa: E402
from kernel.model_acceptance import promote  # noqa: E402


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


class TestFullHistoryDataCutoffStamp:
    """#423 — the full-history production panel stamps TWO DISTINCT
    information-set fields so the freshness monitor (orch #213) +
    P-MODEL-STALENESS gate can key staleness on the right axis:

      * ``label_observation_cutoff`` — the fwd-clipped max labeled row. THIS is
        the MODEL-FRESHNESS axis (latest information that affected fitting).
      * ``max_feature_anchor_date`` — the RAW feature/data frontier. This is
        DATA-PIPELINE HEALTH provenance only (leads the label axis by ~60 BD);
        it is NOT model freshness — unused trailing rows cannot refresh weights.

    Neither is wall-clock ``trained_date``; ``effective_selection_cutoff_date``
    is never fabricated; the full-history path never overloads the
    exclusive-bound ``effective_train_cutoff_date`` with the observed-max label;
    and a missing raw frontier stays missing (never backfilled from the label
    cutoff — one field, one meaning).
    """

    def _train_df_with_known_max(self, tail_null: int = 40):
        """Panel (all 2024 dates) whose fwd_60d label is nulled for the last
        ``tail_null`` trading days, so after ``dropna`` the max labeled date is
        a controlled 2024 business day — well before wall-clock
        ``trained_date`` (>= 2025). ``feature_frontier`` is the pre-``dropna``
        panel max (the nulled tail) and leads the labeled max."""
        panel = _make_synthetic_panel(n_tickers=4, n_dates=200, start="2024-01-01")
        dates = pd.Index(sorted(panel["date"].unique()))
        cut = pd.Timestamp(dates[-(tail_null + 1)])
        feature_frontier = pd.Timestamp(dates[-1])
        panel.loc[panel["date"] > cut, "fwd_60d_excess"] = np.nan
        train = panel.dropna(subset=["fwd_60d_excess"])
        assert train["date"].max() == cut  # fixture sanity
        assert feature_frontier > cut       # frontier leads the labeled max
        return train, cut, feature_frontier

    def _build(self, train, max_feature_anchor_date=None):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        return TPM.build_artifact(
            booster, ["feat_a", "feat_b"], np.zeros(2), np.ones(2), train,
            cutoff_date=None, side_label=None, train_run_id="abc12345",
            max_feature_anchor_date=max_feature_anchor_date,
        )

    def test_stamps_label_observation_cutoff_from_max_labeled_date(self):
        train, cut, _ = self._train_df_with_known_max()
        art = self._build(train)
        assert art["label_observation_cutoff"] == cut.isoformat()

    def test_label_observation_cutoff_is_data_not_wall_clock(self):
        """A fresh trained_date over stale labeled data must NOT masquerade as
        fresh: the stamped label cutoff is the DATA max, not today's date."""
        train, cut, _ = self._train_df_with_known_max()
        art = self._build(train)
        stamped = pd.Timestamp(art["label_observation_cutoff"])
        assert stamped == cut
        # Derived from the frame, NOT datetime.now(): a 2024 data cutoff on an
        # artifact trained "today" (>= 2025) proves it is not wall-clock.
        assert stamped < pd.Timestamp(art["trained_date"])
        assert stamped.year == 2024

    def test_feature_anchor_is_distinct_and_leads_label_cutoff(self):
        """``max_feature_anchor_date`` (data-pipeline health provenance) is the
        raw frontier and LEADS the label-observation cutoff (the actual
        model-freshness axis) by the label horizon — the two fields are
        distinct and must never be conflated (Codex #423 round-3 review)."""
        train, cut, frontier = self._train_df_with_known_max()
        art = self._build(train, max_feature_anchor_date=frontier)
        assert art["max_feature_anchor_date"] == frontier.isoformat()
        assert art["label_observation_cutoff"] == cut.isoformat()
        assert (
            pd.Timestamp(art["max_feature_anchor_date"])
            > pd.Timestamp(art["label_observation_cutoff"])
        )

    def test_feature_anchor_omitted_when_absent(self):
        """When no frontier is threaded in (per-regime / legacy callers) the
        field is OMITTED — never silently backfilled from the label cutoff.
        Falling back would give ``max_feature_anchor_date`` one meaning
        (independently observed raw frontier) on one caller path and another
        (copy of the label max) on another, and a caller checking data-pipeline
        health could misread the fallback as a fresh frontier (Codex #423
        round-3 review)."""
        train, cut, _ = self._train_df_with_known_max()
        art = self._build(train)  # no max_feature_anchor_date
        assert "max_feature_anchor_date" not in art
        assert art["label_observation_cutoff"] == cut.isoformat()

    def test_selection_cutoff_never_fabricated_full_history(self):
        """effective_selection_cutoff_date must NOT be copied from the train
        cutoff (Codex #423): omitting it lets lean_guard._selection_anchor fall
        through to the conservative trained_date."""
        train, _, frontier = self._train_df_with_known_max()
        art = self._build(train, max_feature_anchor_date=frontier)
        assert "effective_selection_cutoff_date" not in art

    def test_full_history_omits_exclusive_train_cutoff(self):
        """The full-history path has no --train-cutoff bounding feature rows, so
        the exclusive-bound effective_train_cutoff_date is OMITTED — never
        overloaded with the observed-max label."""
        train, _, frontier = self._train_df_with_known_max()
        art = self._build(train, max_feature_anchor_date=frontier)
        assert "effective_train_cutoff_date" not in art

    def test_walkforward_stamps_exclusive_cutoff_not_selection(self):
        """The walk-forward branch keeps effective_train_cutoff_date as the
        pre-embargo EXCLUSIVE boundary (its documented contract) and still does
        NOT fabricate a selection cutoff. label_observation_cutoff is the
        observed max labeled row on this path too (consistent meaning)."""
        train = _make_synthetic_panel(
            n_tickers=3, n_dates=30, start="2023-06-01"
        ).dropna(subset=["fwd_60d_excess"])
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        art = TPM.build_artifact(
            booster, ["feat_a"], np.zeros(1), np.ones(1), train,
            cutoff_date=pd.Timestamp("2024-01-01"),
            side_label="walkforward_v2_2024-01-01",
            train_run_id="abc12345",
        )
        expected = (pd.Timestamp("2024-01-01") - pd.offsets.BDay(60)).isoformat()
        assert art["effective_train_cutoff_date"] == expected
        assert "effective_selection_cutoff_date" not in art
        assert art["label_observation_cutoff"] == pd.Timestamp(
            train["date"].max()).isoformat()


class TestPromotePreservesDataCutoff:
    """The active-swap promote() must not drop the freshness/provenance fields
    — it copies the whole artifact, so a promoted panel keeps the fields the
    freshness monitor + P-MODEL-STALENESS gate rely on."""

    def test_promote_preserves_freshness_provenance_fields(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RQ_ALLOW_NO_WF", "1")  # bypass WF gate; test swap
        staging = tmp_path / "panel-ltr.staging.json"
        active = tmp_path / "panel-ltr.alpha158_fund.json"
        staging.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["a", "b"],
            "trained_date": "2026-05-18",
            "max_feature_anchor_date": "2026-05-15T00:00:00",
            "label_observation_cutoff": "2024-11-13T00:00:00",
        }))
        promote(staging, active)
        promoted = json.loads(active.read_text())
        assert promoted["max_feature_anchor_date"] == "2026-05-15T00:00:00"
        assert promoted["label_observation_cutoff"] == "2024-11-13T00:00:00"


class TestRestampPreservesDataCutoff:
    """The sector_map re-stamp repair tool rewrites the WHOLE artifact dict
    (mutating only fingerprint fields), so a re-stamp must NOT drop the
    freshness/provenance fields."""

    def test_restamp_preserves_freshness_provenance_fields(self, tmp_path, monkeypatch):
        repo = tmp_path
        strat = repo / "backtesting" / "renquant_104"
        (strat / "artifacts" / "prod").mkdir(parents=True)
        art_rel = "artifacts/prod/panel-ltr.alpha158_fund.json"
        art_path = strat / art_rel
        # Artifact carries the freshness/provenance fields + a legacy
        # fingerprint whose sector_map is None (the exact re-stamp trigger).
        art_path.write_text(json.dumps({
            "kind": "panel_ltr_xgboost",
            "feature_cols": ["a"],
            "trained_date": "2026-05-18",
            "max_feature_anchor_date": "2026-05-15T00:00:00",
            "label_observation_cutoff": "2024-11-13T00:00:00",
            "config_fingerprint": "sha256:old",
            "config_fingerprint_fields": {"watchlist": ["AAA"], "sector_map": None},
        }))
        cfg_path = strat / "strategy_config.json"
        cfg_path.write_text(json.dumps(
            {"ranking": {"panel_scoring": {"artifact_path": art_rel}}}
        ))
        # Redirect module paths to the tmp tree and stub the fingerprint
        # machinery so only sector_map differs (a valid re-stamp).
        monkeypatch.setattr(RESTAMP, "REPO", repo)
        monkeypatch.setattr(RESTAMP, "STRATEGY_DIR", strat)
        monkeypatch.setattr(
            RESTAMP, "_model_relevant_fields",
            lambda cfg: {"watchlist": ["AAA"], "sector_map": {"AAA": "tech"}},
        )
        monkeypatch.setattr(RESTAMP, "fingerprint_config", lambda cfg: "sha256:new")
        monkeypatch.setattr(sys, "argv", ["restamp_prod_fingerprint.py"])
        rc = RESTAMP.main()
        assert rc == 0
        restamped = json.loads(art_path.read_text())
        # Fingerprint was updated (proves the re-stamp ran)...
        assert restamped["config_fingerprint"] == "sha256:new"
        assert restamped["config_fingerprint_fields"]["sector_map"] == {"AAA": "tech"}
        # ...and the freshness/provenance fields survived the rewrite.
        assert restamped["max_feature_anchor_date"] == "2026-05-15T00:00:00"
        assert restamped["label_observation_cutoff"] == "2024-11-13T00:00:00"


class TestModelFreshnessAxisIntegration:
    """INTEGRATION (#423 round-3) — producer + monitor together. Model
    freshness MUST key on ``label_observation_cutoff`` (the latest information
    that actually affected fitting), with the ~60 BD fwd-label horizon lag
    accounted for EXPLICITLY, never on ``max_feature_anchor_date`` (the raw
    feature/data frontier). Keying on the raw frontier would let fresh
    UNLABELED rows — which the model never trained on — make a frozen model
    read healthy: "fresh metadata over stale trained information" under a new
    field name (Codex #423 round-3 review). This class also pins the explicit
    anti-regression the review asked for: appending fresh unlabeled rows
    without changing the labeled training frame must not improve the
    model-freshness read."""

    LOOKAHEAD_BD = 60  # matches fwd_60d_excess

    @classmethod
    def _model_freshness_state(cls, label_cutoff_iso: str, today: pd.Timestamp) -> str:
        """orch #213 prod fast-axis policy keyed on the LABEL axis, with the
        expected fwd-label horizon lag subtracted out explicitly: a labeled
        row can never be more recent than ``today - LOOKAHEAD_BD`` business
        days, so age is measured from THAT expected frontier, not from
        ``today`` directly. HEALTHY <=14 cal days beyond expectation,
        BREACH >28."""
        expected_max = today - pd.offsets.BDay(cls.LOOKAHEAD_BD)
        age = (expected_max - pd.Timestamp(label_cutoff_iso)).days
        if age <= 14:
            return "HEALTHY"
        if age > 28:
            return "BREACH"
        return "WARN"

    @staticmethod
    def _panel_ending(end: pd.Timestamp, n_dates: int, tail_null: int,
                      n_tickers: int = 4) -> pd.DataFrame:
        """alpha158-shaped panel whose FEATURE rows end at ``end`` but whose
        fwd_60d label is nulled for the last ``tail_null`` business days (the
        unobservable-forward-return frontier)."""
        dates = pd.bdate_range(end=end, periods=n_dates)
        rng = np.random.default_rng(7)
        rows = []
        for t in range(n_tickers):
            for d in dates:
                rows.append({
                    "ticker": f"T{t:02d}", "date": d, "split_label": "train",
                    "feat_a": rng.normal(), "feat_b": rng.normal(),
                    "earnings_yield": rng.normal(),
                    "fwd_5d_excess": rng.normal() * 0.01,
                    "fwd_20d_excess": rng.normal() * 0.02,
                    "fwd_60d_excess": rng.normal() * 0.03,
                })
        panel = pd.DataFrame(rows)
        cut = pd.Timestamp(dates[-(tail_null + 1)])
        panel.loc[panel["date"] > cut, "fwd_60d_excess"] = np.nan
        return panel

    def _build_from_panel(self, panel, tmp_path, monkeypatch, booster=None):
        (tmp_path / "data").mkdir(exist_ok=True)
        panel.to_parquet(
            tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet"
        )
        monkeypatch.chdir(tmp_path)
        train, feat_cols, _label, frontier = TPM.load_and_slice_panel(
            None, return_feature_frontier=True,
        )
        if booster is None:
            booster = mock.MagicMock()
            booster.save_raw.return_value = b"{}"
        art = TPM.build_artifact(
            booster, feat_cols,
            np.zeros(len(feat_cols)), np.ones(len(feat_cols)), train,
            cutoff_date=None, side_label=None, train_run_id="int00001",
            max_feature_anchor_date=frontier,
        )
        return art, train

    def test_current_panel_is_healthy_on_label_axis_with_lag_accounted(
            self, tmp_path, monkeypatch):
        today = pd.Timestamp.today().normalize()
        panel = self._panel_ending(end=today, n_dates=200, tail_null=self.LOOKAHEAD_BD)
        art, _train = self._build_from_panel(panel, tmp_path, monkeypatch)
        # Raw frontier ≈ today (data-pipeline health provenance, NOT model
        # freshness)...
        assert pd.Timestamp(art["max_feature_anchor_date"]) >= today - pd.Timedelta(days=3)
        # ...and the LABEL axis, once the expected ~60 BD horizon lag is
        # accounted for explicitly, reads HEALTHY for a genuinely fresh
        # retrain — the monitor must not need the raw frontier to avoid a
        # false BREACH.
        assert self._model_freshness_state(
            art["label_observation_cutoff"], today) == "HEALTHY"

    def test_frozen_panel_breaches_on_label_axis(self, tmp_path, monkeypatch):
        today = pd.Timestamp.today().normalize()
        frozen_end = today - pd.Timedelta(days=400)
        panel = self._panel_ending(end=frozen_end, n_dates=200, tail_null=self.LOOKAHEAD_BD)
        art, _train = self._build_from_panel(panel, tmp_path, monkeypatch)
        assert self._model_freshness_state(
            art["label_observation_cutoff"], today) == "BREACH"

    def test_fresh_unlabeled_rows_do_not_improve_model_freshness(
            self, tmp_path, monkeypatch):
        """Regression for the Codex #423 round-3 review: appending fresh
        UNLABELED feature rows to a frozen panel — without changing the
        labeled training frame the model actually fit on — must NOT move
        ``label_observation_cutoff`` and must NOT improve the model-freshness
        read, even though it correctly advances ``max_feature_anchor_date``
        (a genuine, separate data-pipeline-health signal)."""
        today = pd.Timestamp.today().normalize()
        frozen_end = today - pd.Timedelta(days=400)
        frozen_panel = self._panel_ending(
            end=frozen_end, n_dates=200, tail_null=self.LOOKAHEAD_BD)
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"

        art_frozen, train_frozen = self._build_from_panel(
            frozen_panel, tmp_path, monkeypatch, booster=booster)

        # Simulate the data pipeline catching up to today WITHOUT any new
        # labels existing yet (fresh rows can't have an observable fwd_60d
        # label): append feature-only rows out to `today`, all unlabeled.
        extra_dates = pd.bdate_range(
            start=frozen_end + pd.Timedelta(days=1), end=today)
        rng = np.random.default_rng(11)
        extra_rows = []
        for t in range(4):
            for d in extra_dates:
                extra_rows.append({
                    "ticker": f"T{t:02d}", "date": d, "split_label": "train",
                    "feat_a": rng.normal(), "feat_b": rng.normal(),
                    "earnings_yield": rng.normal(),
                    "fwd_5d_excess": np.nan,
                    "fwd_20d_excess": np.nan,
                    "fwd_60d_excess": np.nan,
                })
        extended_panel = pd.concat(
            [frozen_panel, pd.DataFrame(extra_rows)], ignore_index=True)

        # Rebuild from the SAME (mocked) trained booster — the model itself
        # did not retrain — over the extended panel.
        art_extended, train_extended = self._build_from_panel(
            extended_panel, tmp_path, monkeypatch, booster=booster)

        # The labeled training frame the model actually fit on is UNCHANGED...
        assert len(train_extended) == len(train_frozen)
        assert train_extended["date"].max() == train_frozen["date"].max()
        assert (
            art_extended["label_observation_cutoff"]
            == art_frozen["label_observation_cutoff"]
        )

        # ...so the model-freshness read must be IDENTICAL (still BREACH) —
        # NOT improved to HEALTHY/WARN just because the raw frontier moved.
        state_frozen = self._model_freshness_state(
            art_frozen["label_observation_cutoff"], today)
        state_extended = self._model_freshness_state(
            art_extended["label_observation_cutoff"], today)
        assert state_frozen == state_extended == "BREACH"

        # Meanwhile the raw frontier DID advance (genuine data-pipeline-health
        # signal) — proving the two fields really are decoupled, not that the
        # extension silently no-opped.
        assert (
            pd.Timestamp(art_extended["max_feature_anchor_date"])
            > pd.Timestamp(art_frozen["max_feature_anchor_date"])
        )


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
