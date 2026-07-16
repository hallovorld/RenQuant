"""Regression guards for renquant_104 artifact/run provenance contracts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))


def _panel_payload(**overrides):
    base = {
        "feature_cols": ["alpha_a", "alpha_b"],
        "trained_date": "2026-05-21",
        "config_fingerprint": "sha256:abcd",
        "train_run_id": "run123",
        "oos_mean_ic": 0.031,
        "oos_std_ic": 0.010,
        "oos_per_fold_ic": [0.02, 0.03, 0.04],
        "lookahead_days": 20,
        "cv_method": "purged_time_series",
        "cv_embargo_days": 20,
        "panel_shape": {"rows": 1000, "tickers": 10, "dates": 100},
    }
    base.update(overrides)
    return base


class TestPanelArtifactContract:
    def test_good_panel_artifact_passes_strict_contract(self):
        from kernel.artifact_contract import validate_panel_artifact_contract

        result = validate_panel_artifact_contract(_panel_payload(), strict=True)

        assert result.ok
        assert result.errors == []
        assert result.details["n_features"] == 2

    def test_missing_oos_evidence_fails_strict_contract(self):
        from kernel.artifact_contract import validate_panel_artifact_contract

        payload = _panel_payload(oos_mean_ic=None)
        result = validate_panel_artifact_contract(payload, strict=True)

        assert not result.ok
        assert "missing oos_mean_ic" in result.errors

    def test_embargo_must_cover_lookahead(self):
        from kernel.artifact_contract import validate_panel_artifact_contract

        result = validate_panel_artifact_contract(
            _panel_payload(lookahead_days=60, cv_embargo_days=10),
            strict=True,
        )

        assert not result.ok
        assert "cv_embargo_days=10 < lookahead_days=60" in result.errors

    def test_non_strict_allows_legacy_artifact_but_warns(self):
        from kernel.artifact_contract import validate_panel_artifact_contract

        payload = _panel_payload(train_run_id=None, oos_mean_ic=None)
        result = validate_panel_artifact_contract(payload, strict=False)

        assert result.ok
        assert any("train_run_id" in w for w in result.warnings)
        assert any("oos_mean_ic" in w for w in result.warnings)


class TestFeatureContract:
    def test_feature_contract_errors_by_default(self):
        from kernel.artifact_contract import validate_feature_contract

        result = validate_feature_contract(["a", "b"], ["a"])

        assert not result.ok
        assert result.details["missing"] == ["b"]

    def test_feature_contract_warn_policy_does_not_fail(self):
        from kernel.artifact_contract import validate_feature_contract

        result = validate_feature_contract(["a", "b"], ["a"], policy="warn")

        assert result.ok
        assert result.warnings


class TestRunBundle:
    def test_run_bundle_hashes_artifacts_and_data_dates(self, tmp_path):
        from kernel.artifact_contract import build_run_bundle, sha256_file

        artifact = tmp_path / "panel.json"
        artifact.write_text(json.dumps(_panel_payload()), encoding="utf-8")
        idx = pd.bdate_range("2026-05-01", periods=3)
        ctx = SimpleNamespace(
            ohlcv={"AAA": pd.DataFrame({"close": [1, 2, 3]}, index=idx)},
            buy_blocked=True,
            skip_buys=False,
            bear_only=False,
            regime="BULL_CALM",
            confidence=0.75,
        )
        config = {
            "watchlist": ["AAA", "BBB"],
            "ranking": {"panel_scoring": {"artifact_path": str(artifact)}},
        }

        bundle = build_run_bundle(
            config,
            STRATEGY_DIR,
            run_id="r1",
            run_type="sim",
            ctx=ctx,
        )

        assert bundle["artifact_hashes"]["panel"] == sha256_file(artifact)
        assert bundle["data_max_dates"]["AAA"] == "2026-05-05"
        assert bundle["pipeline_flags"]["buy_blocked"] is True
        assert bundle["panel_contract"]["ok"] is True

    def test_run_bundle_records_regime_evidence(self, tmp_path):
        from kernel.artifact_contract import build_run_bundle

        ctx = SimpleNamespace(
            ohlcv={},
            buy_blocked=False,
            skip_buys=False,
            bear_only=False,
            regime="BEAR",
            confidence=0.50,
            _regime_evidence={
                "source": "hard_bear",
                "final_regime": "BEAR",
                "hard_bear": True,
                "gmm_probs": {"BEAR": 0.2, "BULL_CALM": 0.8},
                "spy_close": 690.0,
                "spy_ma50": 680.0,
                "spy_ma200": 640.0,
            },
        )
        bundle = build_run_bundle(
            {"watchlist": ["SPY"]},
            STRATEGY_DIR,
            run_id="r-regime",
            run_type="sim",
            ctx=ctx,
        )

        evidence = bundle["regime_evidence"]
        assert evidence["source"] == "hard_bear"
        assert evidence["final_regime"] == "BEAR"
        assert evidence["hard_bear"] is True
        assert evidence["spy_close"] == 690.0


class TestRunBundleTrainingProvenance:
    """training_cutoff + model_content_sha256 extraction (G4 Phase A)."""

    def test_json_panel_artifact_sets_training_metadata(self, tmp_path):
        from kernel.artifact_contract import build_run_bundle

        artifact = tmp_path / "panel.json"
        artifact.write_text(json.dumps(_panel_payload()), encoding="utf-8")
        config = {
            "watchlist": ["AAA"],
            "ranking": {"panel_scoring": {"artifact_path": str(artifact)}},
        }

        bundle = build_run_bundle(
            config, STRATEGY_DIR, run_id="r-tc1", run_type="sim",
        )

        assert bundle["training_cutoff"] == _panel_payload()["trained_date"]
        assert bundle["model_content_sha256"]

    def test_non_json_panel_falls_back_to_active_scorer_metadata(self, tmp_path):
        from kernel.artifact_contract import build_run_bundle

        # PatchTST-style checkpoint: not JSON, so the config-path read yields
        # no payload — provenance must come from the active scorer contract.
        checkpoint = tmp_path / "hf_patchtst_model.pt"
        checkpoint.write_bytes(b"\x80\x02not-json")
        scorer = SimpleNamespace(metadata={
            "trained_date": "2026-05-19",
            "effective_train_cutoff_date": "2026-04-09",
            "model_content_fingerprint": "sha256:feedface",
        })
        ctx = SimpleNamespace(
            ohlcv={}, buy_blocked=False, skip_buys=False, bear_only=False,
            regime="BULL_CALM", confidence=0.9, _panel_scorer=scorer,
        )
        config = {
            "watchlist": ["AAA"],
            "ranking": {"panel_scoring": {"artifact_path": str(checkpoint)}},
        }

        bundle = build_run_bundle(
            config, STRATEGY_DIR, run_id="r-tc2", run_type="live", ctx=ctx,
        )

        assert bundle["training_cutoff"] == "2026-04-09"
        assert bundle["model_content_sha256"] == "sha256:feedface"

    def test_scorer_fallback_prefers_effective_cutoff_over_trained_date(
            self, tmp_path):
        from kernel.artifact_contract import build_run_bundle

        checkpoint = tmp_path / "model.pt"
        checkpoint.write_bytes(b"\x00binary")
        scorer = SimpleNamespace(metadata={"trained_date": "2026-05-19"})
        ctx = SimpleNamespace(
            ohlcv={}, buy_blocked=False, skip_buys=False, bear_only=False,
            regime="BULL_CALM", confidence=0.9, _panel_scorer=scorer,
        )
        config = {
            "watchlist": ["AAA"],
            "ranking": {"panel_scoring": {"artifact_path": str(checkpoint)}},
        }

        bundle = build_run_bundle(
            config, STRATEGY_DIR, run_id="r-tc3", run_type="live", ctx=ctx,
        )

        assert bundle["training_cutoff"] == "2026-05-19"
        assert bundle["model_content_sha256"] is None

    def test_no_panel_and_no_scorer_metadata_is_harmless(self):
        from kernel.artifact_contract import build_run_bundle

        ctx = SimpleNamespace(
            ohlcv={}, buy_blocked=False, skip_buys=False, bear_only=False,
            regime="BULL_CALM", confidence=0.9,
        )

        bundle = build_run_bundle(
            {"watchlist": ["AAA"]}, STRATEGY_DIR,
            run_id="r-tc4", run_type="live", ctx=ctx,
        )

        assert bundle.get("training_cutoff") is None
        assert "run_id" in bundle
