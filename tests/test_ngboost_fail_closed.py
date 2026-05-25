"""NGBoost runtime/preflight fail-closed regression guards."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _cand(ticker: str, *, rank_score: float = 0.5):
    return SimpleNamespace(
        ticker=ticker,
        mu=None,
        sigma=None,
        rank_score=rank_score,
        panel_score=rank_score,
    )


class _Head:
    feature_cols = ["x1", "x2"]

    def __init__(self, mu, sigma):
        self._mu = pd.Series(mu)
        self._sigma = pd.Series(sigma)

    def predict_distribution(self, _X):
        return {"mu": self._mu, "sigma": self._sigma}


def _ctx(*, mode: str = "mu_minus_lambda_sigma", candidates=None, head=None, X=None):
    return SimpleNamespace(
        config={"ranking": {"panel_scoring": {"ngboost": {
            "enabled": True,
            "score_mode": mode,
            "lambda_sigma": 1.0,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}},
        candidates=list(candidates if candidates is not None else [_cand("AAA")]),
        holdings={},
        _ngboost_head=head,
        _panel_matrix=X,
        counters={},
        buy_blocked=False,
        skip_buys=False,
    )


def test_load_ngboost_missing_artifact_blocks_new_buys(tmp_path):
    from kernel.panel_pipeline.job_panel_scoring import LoadNGBoostTask

    ctx = _ctx(candidates=[_cand("AAA"), _cand("BBB")])
    ctx.config["_strategy_dir"] = str(tmp_path)

    ret = LoadNGBoostTask().run(ctx)

    assert ret is False
    assert ctx.candidates == []
    assert ctx.buy_blocked is True
    assert ctx.skip_buys is True
    assert ctx._ngboost_fail_closed_reason == "ngb_artifact_missing"
    assert ctx._blocked_by_ticker == {
        "AAA": "ngb_artifact_missing",
        "BBB": "ngb_artifact_missing",
    }


def test_apply_ngboost_enabled_without_head_blocks_new_buys():
    from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask

    ctx = _ctx(head=None, X=pd.DataFrame({"x1": [1.0], "x2": [2.0]}, index=["AAA"]))

    ret = ApplyNGBoostTask().run(ctx)

    assert ret is False
    assert ctx.candidates == []
    assert ctx._ngboost_fail_closed_reason == "ngb_head_missing"
    assert ctx._blocked_by_ticker["AAA"] == "ngb_head_missing"


def test_apply_ngboost_missing_feature_fails_closed_by_default():
    from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask

    ctx = _ctx(
        head=_Head({"AAA": 0.03}, {"AAA": 0.10}),
        X=pd.DataFrame({"x1": [1.0]}, index=["AAA"]),
    )

    ret = ApplyNGBoostTask().run(ctx)

    assert ret is False
    assert ctx.candidates == []
    assert ctx._ngboost_fail_closed_reason == "ngb_missing_features"
    assert ctx._blocked_by_ticker["AAA"] == "ngb_missing_features"


def test_apply_ngboost_override_mode_requires_full_prediction_coverage():
    from kernel.panel_pipeline.job_panel_scoring import ApplyNGBoostTask

    aaa = _cand("AAA", rank_score=0.80)
    bbb = _cand("BBB", rank_score=0.70)
    ctx = _ctx(
        candidates=[aaa, bbb],
        head=_Head({"AAA": 0.03}, {"AAA": 0.10}),
        X=pd.DataFrame({"x1": [1.0, 2.0], "x2": [2.0, 3.0]}, index=["AAA", "BBB"]),
    )

    ret = ApplyNGBoostTask().run(ctx)

    assert ret is False
    assert ctx.candidates == []
    assert ctx._ngboost_fail_closed_reason == "ngb_prediction_incomplete"
    assert ctx._blocked_by_ticker == {
        "AAA": "ngb_prediction_incomplete",
        "BBB": "ngb_prediction_incomplete",
    }
    assert bbb.rank_score == pytest.approx(0.70)


def test_preflight_active_ngboost_requires_feature_cols_and_run_id(tmp_path):
    from kernel.preflight import (
        _check_artifact_run_id_alignment,
        _check_feature_coverage,
    )

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    panel = artifacts / "panel.json"
    ngb = artifacts / "ngb.json"
    panel.write_text(json.dumps({
        "feature_cols": ["x1", "x2"],
        "train_run_id": "run-a",
    }))
    ngb.write_text(json.dumps({
        "feature_cols": [],
        "train_run_id": None,
    }))
    cfg = {
        "panel_ltr": {"artifact_path": "artifacts/panel.json"},
        "ranking": {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngb.json",
        }}},
    }

    feature = _check_feature_coverage(cfg, tmp_path, run_mode="full")
    run_id = _check_artifact_run_id_alignment(cfg, tmp_path, run_mode="full")

    assert feature.ok is False and feature.severity == "hard"
    assert "feature_cols not stamped" in feature.message
    assert run_id.ok is False and run_id.severity == "hard"
    assert "run_id not stamped" in run_id.message


def test_preflight_active_ngboost_mismatch_soft_only_for_sell_only(tmp_path):
    from kernel.preflight import _check_artifact_run_id_alignment

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "panel.json").write_text(json.dumps({
        "feature_cols": ["x1"],
        "train_run_id": "run-a",
    }))
    (artifacts / "ngb.json").write_text(json.dumps({
        "feature_cols": ["x1"],
        "train_run_id": "run-b",
    }))
    cfg = {
        "panel_ltr": {"artifact_path": "artifacts/panel.json"},
        "ranking": {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngb.json",
        }}},
    }

    full = _check_artifact_run_id_alignment(cfg, tmp_path, run_mode="full")
    sell = _check_artifact_run_id_alignment(cfg, tmp_path, run_mode="sell-only")

    assert full.ok is False and full.severity == "hard"
    assert sell.ok is True and sell.severity == "soft"
