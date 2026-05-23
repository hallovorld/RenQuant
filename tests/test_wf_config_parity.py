from __future__ import annotations

import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from wf_config_parity import evaluate_wf_config_parity  # noqa: E402
from wf_config_builder import build_wf_config_from_prod  # noqa: E402


def _artifact(path: Path, cols: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"kind": "panel_ltr_xgboost", "feature_cols": cols}))
    return path


def _base_config(artifact_path: str, *, kind: str = "xgb") -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": kind,
                "artifact_path": artifact_path,
                "buy_floor": "adaptive_mean_std",
                "rotation_advantage": 0.0,
                "sizing": {"enabled": False, "floor": 0.0, "ceiling": 1.0},
                "sigma_sizing": {"enabled": True, "floor": 0.3, "ceiling": 1.0},
                "ngboost": {
                    "enabled": False,
                    "artifact_path": "ignored/ngboost.json",
                    "score_mode": "additive",
                    "lambda_sigma": 0.0,
                },
            },
            "kelly_sizing": {
                "enabled": True,
                "top_up_threshold": 0.05,
                "topup_conviction_floor": 0.20,
            },
        },
        "rotation": {
            "joint_actions": {
                "enabled": True,
                "solver": "qp",
                "qp_min_dw_pct": 0.02,
                "qp_mu_contract": "strict",
                "_comment": "ignored",
            }
        },
        "regime_params": {
            "BULL_CALM": {"max_position_pct": 0.15, "cash_reserve_pct": 0.0}
        },
        "max_concurrent_positions": 8,
        "max_positions_per_sector": 3,
        "wash_sale_days": 30,
        "tax": {"enabled": True, "short_term_rate": 0.50},
        "defensive_tickers": ["GLD"],
        "sector_map": {"AAPL": "Tech"},
        "tiered_thresholds": [{"min_model_score": 0.20}],
    }


def _write_config(path: Path, cfg: dict) -> Path:
    path.write_text(json.dumps(cfg, indent=2))
    return path


def test_matching_wf_config_with_manifest_passes(tmp_path: Path) -> None:
    prod_art = _artifact(tmp_path / "artifacts/prod/panel.json", ["f1", "f2", "sent"])
    wf_art = _artifact(tmp_path / "artifacts/wf/cut/panel.json", ["f1", "f2", "sent"])
    manifest = tmp_path / "artifacts/sim/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(wf_art), "cutoff_date": "2024-01-01"}]
    }))

    prod_cfg = _base_config(str(prod_art), kind="xgb")
    wf_cfg = _base_config("artifacts/sim/wf-placeholder.json", kind="panel_ltr_xgboost")
    wf_cfg["walkforward"] = {"enabled": True, "manifest_path": str(manifest)}

    result = evaluate_wf_config_parity(
        _write_config(tmp_path / "prod.json", prod_cfg),
        _write_config(tmp_path / "wf.json", wf_cfg),
        candidate_artifact=prod_art,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is True
    assert result["issues"] == []


def test_buy_floor_drift_fails(tmp_path: Path) -> None:
    prod_art = _artifact(tmp_path / "prod_art.json", ["f1", "f2"])
    wf_art = _artifact(tmp_path / "wf_art.json", ["f1", "f2"])
    prod_cfg = _base_config(str(prod_art))
    wf_cfg = _base_config(str(wf_art))
    wf_cfg["ranking"]["panel_scoring"]["buy_floor"] = "adaptive_mean_std_cap"

    result = evaluate_wf_config_parity(
        _write_config(tmp_path / "prod.json", prod_cfg),
        _write_config(tmp_path / "wf.json", wf_cfg),
        candidate_artifact=prod_art,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is False
    assert any(i["path"] == "ranking.panel_scoring.buy_floor" for i in result["issues"])


def test_feature_recipe_drift_fails(tmp_path: Path) -> None:
    prod_art = _artifact(tmp_path / "prod_art.json", ["f1", "f2", "mean_sentiment"])
    wf_art = _artifact(tmp_path / "wf_art.json", ["f1", "f2"])
    prod_cfg = _base_config(str(prod_art))
    wf_cfg = _base_config(str(wf_art))

    result = evaluate_wf_config_parity(
        _write_config(tmp_path / "prod.json", prod_cfg),
        _write_config(tmp_path / "wf.json", wf_cfg),
        candidate_artifact=prod_art,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is False
    feature_issue = next(i for i in result["issues"] if i["path"] == "artifact.feature_cols")
    assert feature_issue["prod_n_features"] == 3
    assert feature_issue["wf_n_features"] == 2
    assert "mean_sentiment" in feature_issue["missing_vs_prod"]


def test_builder_keeps_prod_semantics_but_wf_eval_paths(tmp_path: Path) -> None:
    prod_art = _artifact(tmp_path / "artifacts/prod/panel.json", ["f1", "sent"])
    wf_art = _artifact(tmp_path / "artifacts/wf/cut/panel.json", ["f1", "sent"])
    manifest = tmp_path / "artifacts/sim/manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({
        "retrains": [{"artifact_uri": str(wf_art), "cutoff_date": "2024-01-01"}]
    }))

    prod_cfg = _base_config(str(prod_art), kind="xgb")
    prod_cfg["ranking"]["kelly_sizing"]["realized_vol_floor"] = 0.05
    prod_cfg["rotation"]["joint_actions"]["qp_tax_lot_method"] = "hifo"
    prod_cfg["sector_map"]["BAC"] = "Finance"
    prod_cfg["ranking"]["panel_scoring"]["global_calibration"] = {
        "enabled": True,
        "artifact_path": "artifacts/prod/panel-rank-calibration.json",
    }
    prod_cfg["regime"] = {
        "gmm_artifact": "prod/spy-gmm-regime.json",
    }
    prod_cfg["ranking"]["panel_scoring"]["shadow_models"] = [
        {"name": "diagnostic_shadow", "kind": "hf_patchtst"},
    ]

    base_wf = _base_config("artifacts/old/panel.json", kind="panel_ltr_xgboost")
    base_wf["walkforward"] = {
        "enabled": True,
        "manifest_path": str(manifest),
        "fail_on_no_model": True,
    }
    base_wf["ranking"]["panel_scoring"]["buy_floor"] = "adaptive_mean_std_cap"
    base_wf["rotation"]["joint_actions"]["qp_tax_lot_method"] = "fifo"
    base_wf["sector_map"] = {"AAPL": "Tech"}
    base_wf["ranking"]["panel_scoring"]["global_calibration"] = {
        "enabled": True,
        "artifact_path": "artifacts/sim/panel-rank-calibration.json",
    }
    base_wf["regime"] = {
        "gmm_artifact": "sim/spy-hmm-regime.json",
    }

    built = build_wf_config_from_prod(
        prod_cfg,
        manifest_path=str(manifest),
        base_wf_config=base_wf,
        strategy_dir=tmp_path,
    )
    out = _write_config(tmp_path / "wf_built.json", built)
    prod_path = _write_config(tmp_path / "prod.json", prod_cfg)

    result = evaluate_wf_config_parity(
        prod_path,
        out,
        candidate_artifact=prod_art,
        strategy_dir=tmp_path,
    )

    assert result["passed"] is True
    assert built["ranking"]["panel_scoring"]["buy_floor"] == "adaptive_mean_std"
    assert built["rotation"]["joint_actions"]["qp_tax_lot_method"] == "hifo"
    assert built["sector_map"]["BAC"] == "Finance"
    assert built["walkforward"]["manifest_path"] == str(manifest)
    assert (
        built["ranking"]["panel_scoring"]["global_calibration"]["artifact_path"]
        == "artifacts/sim/panel-rank-calibration.json"
    )
    assert built["regime"]["gmm_artifact"] == "sim/spy-hmm-regime.json"
    assert built["ranking"]["panel_scoring"]["shadow_models"] == []
