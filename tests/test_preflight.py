"""Tests for kernel.preflight — pre-flight smoke test for cron startup.

Catches the class of bugs where artifact / config / state file drifts.
Mandatory at every cron startup; HARD failures raise PreflightFailed
which the runner converts to ntfy + abort.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.preflight import (  # noqa: E402
    PreflightCheck,
    PreflightFailed,
    run_preflight,
    _check_model_artifact,
    _check_panel_artifact_contract,
    _check_wf_gate_metadata,
    _check_best_iter,
    _check_config_fingerprint,
    _check_watchlist_size,
    _check_sector_map_coverage,
    _check_feature_coverage,
    _check_state_file,
    _check_broker_connect,
)
from kernel.config_consistency import (  # noqa: E402
    fingerprint_config, _model_relevant_fields,
)


# ── Fixture: synthetic strategy dir + healthy artifact ─────────────────────

@pytest.fixture
def healthy_setup(tmp_path):
    """Build a config + artifact that should pass every check."""
    cfg = {
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "panel_ltr": {
            "lookahead_days": 10,
            "xgb_params": {"objective": "rank:pairwise"},
            "asset_embeddings": {"enabled": False},
            "min_best_iter": 20,
            "artifact_path": "artifacts/panel-ltr.json",
        },
        "ranking": {"panel_scoring": {
            "enabled": True,
            "artifact_path": "artifacts/panel-ltr.json",
        }},
        "benchmark": "SPY",
        "sector_map": {
            "AAPL": "giant_tech",
            "MSFT": "giant_tech",
            "NVDA": "ai_chip",
        },
        "sector_etf_map": {
            "giant_tech": "XLK",
            "ai_chip": "XLK",
        },
    }
    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    panel = art_dir / "panel-ltr.json"
    panel.write_text(json.dumps({
        "best_iter": 50,
        "oos_mean_ic": 0.045,
        "feature_cols": ["rsi", "macd", "bbp"],
        "config_fingerprint": fingerprint_config(cfg),
        "config_fingerprint_fields": _model_relevant_fields(cfg),
    }))
    return cfg, tmp_path


# ── P-MODEL-ARTIFACT ───────────────────────────────────────────────────────

class TestCheckModelArtifact:
    def test_pass_when_artifact_exists(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_model_artifact(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_artifact_missing(self, tmp_path):
        cfg = {"panel_ltr": {"artifact_path": "artifacts/missing.json"}}
        r = _check_model_artifact(cfg, tmp_path)
        assert not r.ok and r.severity == "hard"
        assert "missing" in r.message.lower()

    def test_fail_on_unreadable_json(self, tmp_path):
        cfg = {"panel_ltr": {"artifact_path": "artifacts/x.json"}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/x.json").write_text("{not json")
        r = _check_model_artifact(cfg, tmp_path)
        assert not r.ok

    def test_sequence_artifact_does_not_parse_checkpoint_as_json(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/patch_model.pt",
        }}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"PK\x03\x04checkpoint")

        r = _check_model_artifact(cfg, tmp_path)

        assert r.ok and r.severity == "hard"
        assert "hf_patchtst checkpoint" in r.message


# ── P-PANEL-CONTRACT ──────────────────────────────────────────────────────

class TestCheckPanelArtifactContract:
    def test_hf_patchtst_binary_uses_summary_sidecar(self, tmp_path):
        """Shadow PatchTST artifacts are .pt checkpoints. Preflight must
        validate the JSON sidecar instead of decoding the binary as UTF-8."""
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        ckpt = art_dir / "hf_patchtst_all_seed44_model.pt"
        ckpt.write_bytes(b"\x80PYTORCH-CHECKPOINT")
        (art_dir / "hf_patchtst_all_seed44_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "cut": "all",
            "seed": 44,
            "best_val_ic": 0.0657,
            "n_features": 172,
        }))
        cfg = {
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/hf_patchtst_all_seed44_model.pt",
            }},
        }

        r = _check_panel_artifact_contract(cfg, tmp_path)

        assert r.ok and r.severity == "hard"
        assert "hf_patchtst checkpoint contract ok" in r.message
        assert r.details["n_features"] == 172

    def test_hf_patchtst_missing_sidecar_fails_hard(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        (art_dir / "hf_patchtst_all_seed44_model.pt").write_bytes(b"\x80PT")
        cfg = {
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/hf_patchtst_all_seed44_model.pt",
            }},
        }

        r = _check_panel_artifact_contract(cfg, tmp_path)

        assert not r.ok and r.severity == "hard"
        assert "summary sidecar missing" in r.message

    def test_strict_hf_patchtst_requires_config_fingerprint(self, tmp_path):
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        (art_dir / "hf_patchtst_all_seed44_model.pt").write_bytes(b"\x80PT")
        (art_dir / "hf_patchtst_all_seed44_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "cut": "all",
            "seed": 44,
            "best_val_ic": 0.0657,
            "n_features": 172,
        }))
        cfg = {
            "preflight": {"artifact_contract": {"strict": True}},
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/hf_patchtst_all_seed44_model.pt",
            }},
        }

        r = _check_panel_artifact_contract(cfg, tmp_path)

        assert not r.ok and r.severity == "hard"
        assert "config_fingerprint missing" in r.message


# ── P-WF-GATE ────────────────────────────────────────────────────────────────

class TestCheckWFGateMetadata:
    def test_failed_wf_metadata_fails_hard(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/panel-ltr.json",
        }}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "metadata": {"wf_gate_metadata": {
                "passed": False,
                "wf_3cut_sharpe_mean": -0.23,
                "spy_sharpe_mean": 1.08,
                "wf_reason": "FAIL: mean Sharpe below floor",
            }},
        }))

        r = _check_wf_gate_metadata(cfg, tmp_path)

        assert not r.ok and r.severity == "hard"
        assert "failed WF gate evidence" in r.message
        assert r.details["passed"] is False

    def test_failed_wf_metadata_allows_sell_only_risk_exits(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/panel-ltr.json",
        }}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "metadata": {"wf_gate_metadata": {
                "passed": False,
                "wf_3cut_sharpe_mean": -0.23,
                "spy_sharpe_mean": 1.08,
                "wf_reason": "FAIL: mean Sharpe below floor",
            }},
        }))

        r = _check_wf_gate_metadata(cfg, tmp_path, run_mode="sell-only")

        assert r.ok and r.severity == "soft"
        assert "sell-only risk exits are allowed" in r.message
        assert r.details["passed"] is False
        assert r.details["run_mode"] == "sell-only"

    def test_passed_wf_metadata_passes_hard(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/panel-ltr.json",
        }}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "metadata": {"wf_gate_metadata": {
                "passed": True,
                "wf_3cut_sharpe_mean": 0.91,
                "spy_sharpe_mean": 0.65,
            }},
        }))

        r = _check_wf_gate_metadata(cfg, tmp_path)

        assert r.ok and r.severity == "hard"
        assert r.details["passed"] is True

    def test_sequence_shadow_skips_wf_gate(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/patch_model.pt",
        }}}

        r = _check_wf_gate_metadata(cfg, tmp_path)

        assert r.ok and r.severity == "soft"
        assert "not applicable" in r.message


# ── P-BEST-ITER (BUG-CV-2 invariant) ───────────────────────────────────────

class TestCheckBestIter:
    def test_pass_when_above_threshold(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_best_iter(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_below_threshold(self, tmp_path):
        """Production was discovered with best_iter=4 today.
        This check refuses to trade on an undertrained model."""
        cfg = {
            "panel_ltr": {
                "min_best_iter": 20,
                "artifact_path": "artifacts/panel-ltr.json",
            },
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
            "oos_mean_ic": 0.04,
        }))
        r = _check_best_iter(cfg, tmp_path)
        assert not r.ok and r.severity == "hard"
        assert "best_iter=4" in r.message
        assert "Retrain required" in r.message

    def test_soft_pass_when_best_iter_missing(self, tmp_path):
        """Legacy artifacts (e.g. transformer) may not stamp best_iter."""
        cfg = {"panel_ltr": {"artifact_path": "artifacts/panel-ltr.json"}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "oos_mean_ic": 0.04,
        }))
        r = _check_best_iter(cfg, tmp_path)
        assert r.ok and r.severity == "soft"

    def test_sequence_artifact_skips_best_iter(self, tmp_path):
        cfg = {"ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/patch_model.pt",
        }}}
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"checkpoint")

        r = _check_best_iter(cfg, tmp_path)

        assert r.ok and r.severity == "soft"
        assert "not applicable" in r.message


# ── P-CONFIG-FP (catches the 24h watchlist-mismatch incident class) ────────

class TestCheckConfigFingerprint:
    def test_pass_when_fingerprints_match(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_config_fingerprint(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_watchlist_drifted(self, healthy_setup):
        cfg, sd = healthy_setup
        # Mutate config so watchlist differs from stored fingerprint
        cfg["watchlist"] = ["AAPL", "MSFT", "NVDA", "TSLA"]   # extra ticker
        r = _check_config_fingerprint(cfg, sd)
        assert not r.ok and r.severity == "hard"
        assert "watchlist" in r.message

    def test_sell_only_soft_passes_fingerprint_drift_for_risk_exits(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["watchlist"] = ["AAPL", "MSFT", "NVDA", "TSLA"]
        r = _check_config_fingerprint(cfg, sd, run_mode="sell-only")
        assert r.ok and r.severity == "soft"
        assert "Sell-only risk exits are allowed" in r.message
        assert "watchlist" in r.message

    def test_fail_when_sector_map_drifted(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["sector_map"]["NVDA"] = "giant_tech"
        r = _check_config_fingerprint(cfg, sd, run_mode="full")
        assert not r.ok and r.severity == "hard"
        assert "sector_map" in r.message

    def test_fail_when_objective_changed(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["panel_ltr"]["xgb_params"]["objective"] = "rank:ndcg"
        r = _check_config_fingerprint(cfg, sd)
        assert not r.ok
        assert "objective" in r.message

    def test_soft_pass_when_no_fingerprint_in_artifact(self, tmp_path):
        """Pre-2026-04-28 artifacts lack fingerprint — soft pass."""
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {"artifact_path": "artifacts/panel-ltr.json"},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 50,
        }))
        r = _check_config_fingerprint(cfg, tmp_path)
        assert r.ok and r.severity == "soft"

    def test_sequence_sidecar_without_fingerprint_soft_passes(self, tmp_path):
        cfg = {
            "watchlist": ["AAPL", "MSFT"],
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/patch_model.pt",
            }},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"checkpoint")
        (tmp_path / "artifacts/patch_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "best_val_ic": 0.03,
            "n_features": 172,
        }))

        r = _check_config_fingerprint(cfg, tmp_path, run_mode="full")

        assert r.ok and r.severity == "soft"
        assert "sequence sidecar lacks fingerprint" in r.message


# ── P-WATCHLIST ────────────────────────────────────────────────────────────

class TestCheckWatchlist:
    def test_pass_when_watchlist_matches(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_watchlist_size(cfg, sd)
        assert r.ok

    def test_fail_when_watchlist_differs(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["watchlist"] = ["AAPL", "MSFT"]   # missing NVDA
        r = _check_watchlist_size(cfg, sd)
        assert not r.ok
        assert "in_trained_not_live" in r.message

    def test_sequence_artifact_soft_passes_unstamped_watchlist(self, tmp_path):
        cfg = {
            "watchlist": ["AAPL", "MSFT"],
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/patch_model.pt",
            }},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"checkpoint")
        (tmp_path / "artifacts/patch_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "best_val_ic": 0.03,
            "n_features": 172,
        }))

        r = _check_watchlist_size(cfg, tmp_path)

        assert r.ok and r.severity == "soft"
        assert "sequence artifact" in r.message

    def test_sequence_artifact_checks_stamped_watchlist(self, tmp_path):
        cfg = {
            "watchlist": ["AAPL", "MSFT"],
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/patch_model.pt",
            }},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"checkpoint")
        (tmp_path / "artifacts/patch_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "best_val_ic": 0.03,
            "n_features": 172,
            "config_fingerprint_fields": {"watchlist": ["AAPL", "MSFT"]},
        }))

        r = _check_watchlist_size(cfg, tmp_path)

        assert r.ok and r.severity == "hard"
        assert "watchlist match" in r.message

    def test_sequence_artifact_fails_stamped_watchlist_mismatch(self, tmp_path):
        cfg = {
            "watchlist": ["AAPL", "MSFT", "NVDA"],
            "ranking": {"panel_scoring": {
                "kind": "hf_patchtst",
                "artifact_path": "artifacts/patch_model.pt",
            }},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/patch_model.pt").write_bytes(b"checkpoint")
        (tmp_path / "artifacts/patch_summary.json").write_text(json.dumps({
            "arch": "hf_patchtst",
            "best_val_ic": 0.03,
            "n_features": 172,
            "config_fingerprint_fields": {"watchlist": ["AAPL", "MSFT"]},
        }))

        r = _check_watchlist_size(cfg, tmp_path)

        assert not r.ok and r.severity == "hard"
        assert "NVDA" in r.message


# ── P-SECTOR-MAP ───────────────────────────────────────────────────────────

class TestCheckSectorMapCoverage:
    def test_pass_when_buyable_watchlist_has_sectors(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_sector_map_coverage(cfg, sd, run_mode="full")
        assert r.ok and r.severity == "hard"
        assert "sector coverage OK" in r.message

    def test_fail_full_when_buyable_ticker_missing_sector(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["watchlist"].append("BAC")
        r = _check_sector_map_coverage(cfg, sd, run_mode="full")
        assert not r.ok and r.severity == "hard"
        assert "BAC" in r.message
        assert r.details["missing_count"] == 1

    def test_sell_only_soft_passes_missing_sector_for_risk_exits(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["watchlist"].append("BAC")
        r = _check_sector_map_coverage(cfg, sd, run_mode="sell-only")
        assert r.ok and r.severity == "soft"
        assert "Sell-only risk exits are allowed" in r.message

    def test_fail_when_sector_lacks_etf_mapping(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["sector_map"]["MSFT"] = "software"
        r = _check_sector_map_coverage(cfg, sd, run_mode="full")
        assert not r.ok and r.severity == "hard"
        assert "software" in r.message


# ── P-FEATURE-COVER ────────────────────────────────────────────────────────

class TestCheckFeatureCoverage:
    def test_skip_when_ngboost_disabled(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_feature_coverage(cfg, sd)
        assert r.ok and r.severity == "soft"

    def test_pass_when_features_overlap(self, healthy_setup):
        cfg, sd = healthy_setup
        cfg["ranking"] = {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}
        # Stamp NGBoost head with same features as panel
        (sd / "artifacts/ngboost-head.json").write_text(json.dumps({
            "feature_cols": ["rsi", "macd", "bbp"],
        }))
        r = _check_feature_coverage(cfg, sd)
        assert r.ok and r.severity == "hard"

    def test_fail_when_too_many_features_missing(self, healthy_setup):
        """Today's macro-drift bug class: NGBoost head has 184 macro
        cols but panel only has 27 → 84.8% missing."""
        cfg, sd = healthy_setup
        cfg["ranking"] = {"panel_scoring": {"ngboost": {
            "enabled": True,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}
        # NGBoost head has 5 features but panel only has 3 → 40% missing
        (sd / "artifacts/ngboost-head.json").write_text(json.dumps({
            "feature_cols": ["rsi", "macd", "bbp", "vxx_z", "hyg_z"],
        }))
        r = _check_feature_coverage(cfg, sd)
        assert not r.ok and r.severity == "hard"
        assert "vxx_z" in r.message or "hyg_z" in r.message

    def test_runs_when_per_regime_overlay_enables_ngb(self, healthy_setup):
        """2026-05-17 fix: per-regime overlay can enable NGB even when
        global flag is False. Preflight must validate features in that
        case too, otherwise CRIT-1 runtime hard-fail fires the first time
        a BEAR/CHOPPY bar lands."""
        cfg, sd = healthy_setup
        cfg["ranking"] = {"panel_scoring": {"ngboost": {
            "enabled": False,
            "artifact_path": "artifacts/ngboost-head.json",
        }}}
        cfg["regime_params"] = {
            "BEAR": {"ngboost": {"enabled": True,
                                  "score_mode": "mu_minus_lambda_sigma",
                                  "lambda_sigma": 1.0}},
        }
        # NGBoost head incompatible with panel (84% missing) → must hard-fail
        (sd / "artifacts/ngboost-head.json").write_text(json.dumps({
            "feature_cols": ["rsi", "macd", "bbp", "vxx_z", "hyg_z"],
        }))
        r = _check_feature_coverage(cfg, sd)
        assert not r.ok and r.severity == "hard", \
            "per-regime overlay should pull preflight into hard-check mode"

    def test_skips_when_no_overlay_activates(self, healthy_setup):
        """REGRESSION GUARD: per-regime entries that don't have
        ngboost.enabled=True must NOT trigger feature-cover check."""
        cfg, sd = healthy_setup
        cfg["regime_params"] = {
            "BEAR": {"stop_loss_pct": 0.07},  # no ngboost overlay
            "CHOPPY": {"ngboost": {"enabled": False}},  # explicit False
        }
        r = _check_feature_coverage(cfg, sd)
        assert r.ok and r.severity == "soft"


# ── P-STATE-FILE ───────────────────────────────────────────────────────────

class TestCheckStateFile:
    def test_skip_when_no_broker_name(self, healthy_setup):
        cfg, sd = healthy_setup
        r = _check_state_file(cfg, sd, None)
        assert r.ok and r.severity == "soft"

    def test_pass_when_state_file_absent(self, healthy_setup):
        """First run: state file doesn't exist yet — soft pass."""
        cfg, sd = healthy_setup
        r = _check_state_file(cfg, sd, "paper")
        assert r.ok and r.severity == "soft"

    def test_pass_when_state_file_valid(self, healthy_setup):
        cfg, sd = healthy_setup
        (sd / "live_state.paper.json").write_text(json.dumps({"x": 1}))
        r = _check_state_file(cfg, sd, "paper")
        assert r.ok and r.severity == "hard"

    def test_fail_on_corrupt_state_file(self, healthy_setup):
        cfg, sd = healthy_setup
        (sd / "live_state.paper.json").write_text("{not json")
        r = _check_state_file(cfg, sd, "paper")
        assert not r.ok and r.severity == "hard"


# ── P-BROKER-CONNECT ───────────────────────────────────────────────────────

class TestCheckBrokerConnect:
    def test_skip_when_no_broker(self):
        r = _check_broker_connect(None)
        assert r.ok and r.severity == "soft"

    def test_pass_when_broker_connects(self):
        broker = MagicMock()
        broker.connect = MagicMock(return_value=None)
        broker.get_account_value = MagicMock(return_value=10000.0)
        r = _check_broker_connect(broker)
        assert r.ok and r.severity == "hard"

    def test_fail_when_broker_raises(self):
        broker = MagicMock()
        broker.connect = MagicMock(side_effect=RuntimeError("API down"))
        r = _check_broker_connect(broker)
        assert not r.ok and r.severity == "hard"


# ── P-CALIBRATOR-HEALTH ────────────────────────────────────────────────────
# 2026-05-05 parity fix: training has a "probability head collapse"
# guard inside fit_global_calibrator. Runtime had no equivalent —
# silently loaded a calibrator with only N unique outputs (N=7 in the
# 2026-05-04 production incident) and ranked all top candidates as
# tied. This check closes the gap.

class TestCheckCalibratorHealth:
    def _setup(self, tmp_path, *, n_unique=20, pool_ic=0.02,
                min_unique_cfg=None):
        from kernel.preflight import _check_calibrator_health
        cfg = {
            "panel_ltr": {},
            "ranking": {"panel_scoring": {"global_calibration": {
                "artifact_path": "artifacts/panel-rank-calibration.json",
            }}},
        }
        if min_unique_cfg is not None:
            cfg["panel_ltr"]["calibrator_health"] = {
                "min_unique_prob_y": min_unique_cfg,
            }
        cal_dir = tmp_path / "artifacts"
        cal_dir.mkdir(exist_ok=True)
        cal_path = cal_dir / "panel-rank-calibration.json"
        cal_path.write_text(json.dumps({
            "metadata": {"n_unique_prob_y": n_unique, "pool_ic": pool_ic}
        }))
        return _check_calibrator_health, cfg, tmp_path

    def test_pass_when_n_unique_above_floor(self, tmp_path):
        check, cfg, sd = self._setup(tmp_path, n_unique=50, pool_ic=0.03)
        r = check(cfg, sd)
        assert r.ok and r.severity == "hard"
        assert "n_unique_prob_y=50" in r.message

    def test_hard_fails_expected_return_flat_plateau(self, tmp_path):
        """P-CALIBRATOR-HEALTH must protect the μ curve, not only prob_y."""
        from kernel.preflight import _check_calibrator_health
        cfg = {
            "panel_ltr": {"calibrator_health": {
                "max_expected_return_flat_fraction": 0.30,
            }},
            "ranking": {"panel_scoring": {"global_calibration": {
                "artifact_path": "artifacts/panel-rank-calibration.json",
            }}},
        }
        (tmp_path / "artifacts").mkdir()
        x = [i / 99 for i in range(100)]
        er_y = [-0.01] * 10 + [i / 1000 for i in range(43)] + [0.20] * 47
        (tmp_path / "artifacts/panel-rank-calibration.json").write_text(json.dumps({
            "metadata": {"n_unique_prob_y": 100, "pool_ic": 0.03},
            "probability": {"x": x, "y": [0.2 + 0.6 * v for v in x]},
            "expected_return": {"x": x, "y": er_y},
        }))

        r = _check_calibrator_health(cfg, tmp_path)

        assert not r.ok and r.severity == "hard"
        assert "expected_return.y has flat region" in r.message

    def test_warn_when_n_unique_below_floor(self, tmp_path):
        # Reproduce the 2026-05-04 incident exactly: n_unique=7
        # 2026-05-05 incident fix: severity downgraded from hard→soft.
        # Calibrator collapse blocks buy quality (top candidates tie) but
        # SHOULD NOT halt sell-only intraday crons — sells use SellGateB +
        # path rules independent of the calibrator. The original hard
        # severity caused 14 consecutive sell-only cron aborts on 2026-05-05
        # before the downgrade. Now: ok=True (soft warn), message stays
        # loud, the operator still sees the WARN, but live ops continue.
        check, cfg, sd = self._setup(tmp_path, n_unique=7, pool_ic=0.02)
        r = check(cfg, sd)
        assert r.ok, "must NOT block — sell-only crons need to keep running"
        assert r.severity == "soft"
        assert "WARN" in r.message
        assert "n_unique_prob_y=7" in r.message
        assert "min_unique_prob_y=10" in r.message
        assert "Sells unaffected" in r.message

    def test_min_unique_configurable(self, tmp_path):
        # n_unique=15 passes the default 10 but should warn at 20.
        # Soft severity (per 2026-05-05 fix), still ok=True.
        check, cfg, sd = self._setup(tmp_path, n_unique=15,
                                      min_unique_cfg=20)
        r = check(cfg, sd)
        assert r.ok and r.severity == "soft"
        assert "WARN" in r.message

    def test_soft_warn_on_negative_pool_ic(self, tmp_path):
        check, cfg, sd = self._setup(tmp_path, n_unique=50, pool_ic=-0.001)
        r = check(cfg, sd)
        # ok=True (soft warn) but flagged
        assert r.ok and r.severity == "soft"
        assert "pool_ic" in r.message and "anti-correlated" in r.message

    def test_legacy_artifact_without_n_unique_soft_skip(self, tmp_path):
        # Pre-2026-05 artifacts didn't stamp n_unique_prob_y. Don't
        # block legacy retrains — soft warn instead.
        from kernel.preflight import _check_calibrator_health
        cfg = {"panel_ltr": {}, "ranking": {"panel_scoring":
                {"global_calibration": {"artifact_path":
                    "artifacts/panel-rank-calibration.json"}}}}
        cal_dir = tmp_path / "artifacts"
        cal_dir.mkdir(exist_ok=True)
        (cal_dir / "panel-rank-calibration.json").write_text(json.dumps({
            "metadata": {"pool_ic": 0.02}  # no n_unique_prob_y
        }))
        r = _check_calibrator_health(cfg, tmp_path)
        assert r.ok and r.severity == "soft"
        assert "legacy artifact" in r.message

    def test_missing_artifact_soft_skip(self, tmp_path):
        from kernel.preflight import _check_calibrator_health
        cfg = {"panel_ltr": {}, "ranking": {"panel_scoring":
                {"global_calibration": {"artifact_path":
                    "artifacts/panel-rank-calibration.json"}}}}
        # No artifact file exists
        r = _check_calibrator_health(cfg, tmp_path)
        assert r.ok and r.severity == "soft"
        assert "absent" in r.message

    def test_check_in_all_checks(self):
        from kernel.preflight import ALL_CHECKS, _check_calibrator_health
        assert _check_calibrator_health in ALL_CHECKS, (
            "P-CALIBRATOR-HEALTH must be in ALL_CHECKS so run_preflight "
            "executes it on every cron tick"
        )


# ── Orchestrator ───────────────────────────────────────────────────────────

class TestRunPreflight:
    def test_strict_mode_raises_on_hard_failure(self, tmp_path):
        """Production undertrained model (best_iter=4) must abort cron."""
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {
                "lookahead_days": 10,
                "xgb_params": {"objective": "rank:pairwise"},
                "asset_embeddings": {"enabled": False},
                "min_best_iter": 20,
                "artifact_path": "artifacts/panel-ltr.json",
            },
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
            "config_fingerprint": fingerprint_config(cfg),
            "config_fingerprint_fields": _model_relevant_fields(cfg),
            "feature_cols": [],
        }))
        with pytest.raises(PreflightFailed, match="best_iter=4"):
            run_preflight(cfg, broker=None, strategy_dir=tmp_path)

    def test_strict_false_returns_results_without_raise(self, tmp_path):
        cfg = {
            "watchlist": ["A"],
            "panel_ltr": {"min_best_iter": 20,
                           "artifact_path": "artifacts/panel-ltr.json"},
        }
        (tmp_path / "artifacts").mkdir()
        (tmp_path / "artifacts/panel-ltr.json").write_text(json.dumps({
            "best_iter": 4,
        }))
        results = run_preflight(cfg, broker=None, strategy_dir=tmp_path, strict=False)
        assert any(not r.ok and r.severity == "hard" for r in results)

    def test_pass_on_healthy_setup(self, healthy_setup):
        cfg, sd = healthy_setup
        results = run_preflight(cfg, broker=None, strategy_dir=sd)
        # All hard checks pass; some soft (NGBoost disabled, state file absent)
        hards = [r for r in results if r.severity == "hard"]
        assert all(r.ok for r in hards), \
            f"hard failures: {[r.message for r in hards if not r.ok]}"

    def test_failed_wf_gate_blocks_full_but_not_sell_only(self, healthy_setup):
        cfg, sd = healthy_setup
        art = sd / cfg["panel_ltr"]["artifact_path"]
        payload = json.loads(art.read_text())
        payload["metadata"] = {"wf_gate_metadata": {
            "passed": False,
            "wf_3cut_sharpe_mean": -1.23,
            "spy_sharpe_mean": 1.08,
            "wf_reason": "FAIL: beat SPY Sharpe 0/3",
        }}
        art.write_text(json.dumps(payload))

        with pytest.raises(PreflightFailed, match="failed WF gate evidence"):
            run_preflight(cfg, broker=None, strategy_dir=sd, run_mode="full")

        results = run_preflight(
            cfg, broker=None, strategy_dir=sd, run_mode="sell-only"
        )
        wf = next(r for r in results if r.name == "P-WF-GATE")
        assert wf.ok and wf.severity == "soft"


# ── PreflightFailed exception ──────────────────────────────────────────────

class TestPreflightFailed:
    def test_message_lists_each_failure(self):
        failures = [
            PreflightCheck("P-X", "hard", False, "reason X"),
            PreflightCheck("P-Y", "hard", False, "reason Y"),
        ]
        e = PreflightFailed(failures)
        assert "P-X" in str(e) and "reason X" in str(e)
        assert "P-Y" in str(e) and "reason Y" in str(e)
        assert "No orders placed" in str(e)
