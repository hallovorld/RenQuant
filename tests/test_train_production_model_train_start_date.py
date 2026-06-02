"""Codex PR #122 HIGH-blocker regression — train-start lower-bound provenance.

CLAUDE.md §7.6 (artifact-fingerprint identity / data-flow safety):
``--train-start-date`` filters rows in ``load_and_slice_panel()`` but,
before this fix, ``build_artifact()`` only stamped the upper cutoff
(``cutoff_date`` / ``cutoff_embargo_days`` / ``effective_train_cutoff_date``)
and never recorded the lower bound. Two artifacts could therefore share
label / features / config fingerprint while being trained on different row
windows — invisible to gates and audits.

This test suite pins the §7.6 fix:
  1. ``--train-start-date`` flag → artifact stamps ``train_start_date`` +
     ``effective_train_start_date`` + ``train_window`` triplet.
  2. Default path (no flag) → all three fields ABSENT (not stamped as
     ``None``), preserving byte-equivalence with pre-extension artifacts.
  3. Stamped artifact is still consumer-compatible: model loads, inference
     smoke runs, scoring returns finite values.

Audit regression guard — pins the bug class where a recent-history retrain
silently overwrote a full-history production candidate because the artifact
provenance did not record the train-window start.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import train_production_model as TPM  # noqa: E402


def _synthetic_panel(n_tickers: int = 4, n_dates: int = 30,
                     start: str = "2023-01-01") -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=n_dates)
    rng = np.random.default_rng(11)
    rows = []
    for t_idx in range(n_tickers):
        tk = f"T{t_idx:02d}"
        for d in dates:
            rows.append({
                "ticker": tk,
                "date": d,
                "split_label": "train",
                "feat_a": rng.normal(),
                "feat_b": rng.normal(),
                "fwd_5d_excess": rng.normal() * 0.01,
                "fwd_20d_excess": rng.normal() * 0.02,
                "fwd_60d_excess": rng.normal() * 0.03,
            })
    return pd.DataFrame(rows)


def _mini_train() -> pd.DataFrame:
    return _synthetic_panel().dropna(subset=["fwd_60d_excess"])


# ───── stamp ON when flag is set ─────

class TestTrainStartStamped:
    """When train_start_date is set, lower-bound provenance fields exist."""

    def test_train_start_date_stamped_iso(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a", "feat_b"],
            np.zeros(2), np.ones(2), train,
            cutoff_date=None,
            side_label=None,
            train_run_id="ts000001",
            train_start_date="2024-01-01",
        )
        # All three lower-bound provenance fields present
        assert art["train_start_date"] == "2024-01-01T00:00:00"
        assert art["effective_train_start_date"] == "2024-01-01T00:00:00"
        assert isinstance(art["train_window"], dict)
        assert art["train_window"]["start"] == "2024-01-01T00:00:00"
        # No cutoff set → train_window.end is the observed max train date.
        assert art["train_window"]["end"] == pd.Timestamp(
            train["date"].max()).isoformat()

    def test_train_window_end_uses_effective_cutoff_when_set(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=pd.Timestamp("2024-06-01"),
            side_label="walkforward_v2_2024-06-01_tsd",
            cutoff_embargo_days=10,
            train_run_id="ts000002",
            train_start_date="2022-01-01",
        )
        expected_end = (
            pd.Timestamp("2024-06-01") - pd.offsets.BDay(10)
        ).isoformat()
        assert art["train_window"]["start"] == "2022-01-01T00:00:00"
        assert art["train_window"]["end"] == expected_end
        # Upper-bound fields still present (back-compat with prior cutoff
        # behaviour — fix is purely additive).
        assert art["cutoff_date"] == "2024-06-01T00:00:00"
        assert art["effective_train_cutoff_date"] == expected_end

    def test_effective_train_start_today_equals_train_start(self):
        """No lower-bound embargo applies; effective == requested.

        Distinct field name kept for symmetry with
        ``effective_train_cutoff_date`` and future-proofing if a lower-
        bound embargo is ever needed (e.g. min-history burn-in).
        """
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000003",
            train_start_date="2022-06-15",
        )
        assert art["train_start_date"] == art["effective_train_start_date"]


# ───── stamp ABSENT (not None) when no flag — byte-compat guard ─────

class TestDefaultPathByteCompat:
    """Default path → fields ABSENT, not stamped as None."""

    def test_no_train_start_date_no_stamp(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000004",
        )
        # ABSENT — preserves byte-equivalence with pre-extension artifacts.
        assert "train_start_date" not in art
        assert "effective_train_start_date" not in art
        assert "train_window" not in art

    def test_cutoff_only_no_lower_bound_stamp(self):
        """Walk-forward cutoff WITHOUT --train-start-date — only upper-bound
        fields stamped, no lower-bound triplet."""
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=pd.Timestamp("2024-01-01"),
            side_label="walkforward_v2_2024-01-01",
            train_run_id="ts000005",
            # train_start_date intentionally omitted
        )
        assert art["cutoff_date"] == "2024-01-01T00:00:00"
        assert "train_start_date" not in art
        assert "effective_train_start_date" not in art
        assert "train_window" not in art

    def test_default_artifact_json_set_unchanged(self):
        """Diff guard — the set of top-level keys for a default artifact
        must not have gained any of the new lower-bound fields."""
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()
        art = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000006",
        )
        # Round-trip through JSON to catch accidental None stamping (json
        # serializes None as null, which would change the byte stream).
        round_tripped = json.loads(json.dumps(art))
        new_fields = {"train_start_date", "effective_train_start_date",
                      "train_window"}
        assert not (new_fields & set(round_tripped.keys()))


# ───── stamped artifact still loads + scores ─────

class TestStampedArtifactConsumerCompat:
    """A stamped artifact loads + scores correctly (back-compat)."""

    def test_stamped_artifact_inference_smoke_ok(self):
        """Real booster + build_artifact + attach_inference_smoke pipeline."""
        train = _mini_train()
        feat_cols = ["feat_a", "feat_b"]
        # Train a tiny real booster so smoke-test code path is exercised.
        dtrain = xgb.DMatrix(
            train[feat_cols].values,
            label=train["fwd_60d_excess"].values,
        )
        params = {
            "objective": "reg:squarederror",
            "max_depth": 2,
            "eta": 0.1,
            "verbosity": 0,
        }
        booster = xgb.train(params, dtrain, num_boost_round=3)

        art = TPM.build_artifact(
            booster, feat_cols,
            np.zeros(len(feat_cols)), np.ones(len(feat_cols)), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000007",
            train_start_date="2023-01-15",
        )
        # Stamped + still wire-compatible with inference smoke
        assert "train_start_date" in art
        TPM.attach_inference_smoke(art, booster, feat_cols)
        smoke = art["metadata"]["inference_smoke_test"]
        assert smoke["all_finite"] is True
        assert smoke["n"] == 32
        # JSON round-trip preserves the new fields
        serialized = json.dumps(art)
        reloaded = json.loads(serialized)
        assert reloaded["train_start_date"] == "2023-01-15T00:00:00"
        assert reloaded["train_window"]["start"] == "2023-01-15T00:00:00"


# ───── audit regression guard — two artifacts, same recipe, diff windows ─────

class TestSameRecipeDifferentWindowsDistinguishable:
    """§7.6 invariant: full-history vs Track D recent-history retrains must
    be distinguishable via machine-readable artifact metadata (NOT just by
    looking at panel_shape.rows, which is data-dependent and could collide
    by accident).
    """

    def test_two_artifacts_same_recipe_distinct_via_train_window(self):
        booster = mock.MagicMock()
        booster.save_raw.return_value = b"{}"
        train = _mini_train()

        art_full = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000008",
            # full-history: no lower bound
        )
        art_recent = TPM.build_artifact(
            booster, ["feat_a"],
            np.zeros(1), np.ones(1), train,
            cutoff_date=None, side_label=None,
            train_run_id="ts000009",
            train_start_date="2024-01-01",
        )

        # Machine-readable distinguisher present on the recent retrain
        assert "train_window" in art_recent
        # And absent on the full-history retrain — silent reduction of
        # row coverage IS detectable from the artifact alone.
        assert "train_window" not in art_full
