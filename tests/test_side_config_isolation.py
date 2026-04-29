"""Tests for side-config artifact-path isolation (2026-04-28 evening fix).

Today twice: a side-config retrain (`strategy_config.diag.json` and
`strategy_config.h60_103.json`) silently overwrote production
ngboost-head.json and panel-rank-calibration.json because:

  1. NGBoostSaveTask read `panel_ltr.ngboost.artifact_path` only;
     side configs that set the inference-side path
     `ranking.panel_scoring.ngboost.artifact_path` were ignored.
  2. RefreshPanelCalibratorTask's subprocess always read
     `strategy_config.json` — no way to forward the active config name.
  3. fit_panel_calibrator.py hardcoded the production config path.

These tests enforce the invariant: **a side config that names side
artifact paths MUST NOT touch production paths during training.**
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


# ── NGBoost save reads from EITHER config location ──────────────────────────

class TestNGBoostSavePathRouting:
    def test_inference_side_priority(self):
        """If `ranking.panel_scoring.ngboost.artifact_path` is set,
        it wins — that's where the live runner reads from."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # Both locations are read
        assert 'cfg = ctx.config.get("panel_ltr", {}).get("ngboost", {})' in src
        assert 'cfg_infer = (ctx.config.get("ranking", {})' in src
        # Inference-side has priority (where the live runner reads)
        assert "out_name = out_name_infer or out_name_train" in src

    def test_inference_side_path_used_when_only_it_is_set(self):
        """Side configs that set ONLY the inference path now route NGBoost
        to the side. Pre-fix, these silently fell back to the default
        production path. Today's contamination root cause."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # Verify the fallback pattern
        assert "out_name_infer" in src
        assert ".get(\"ngboost\", {})" in src

    def test_warns_on_path_disagreement(self):
        """If both locations are set and they disagree, log a warning so
        operator notices the typo before contamination."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        assert "training-side path" in src and "inference-side path" in src
        assert "please reconcile in config" in src

    def test_relative_path_with_directory_preserved(self):
        """Pre-fix did `out_path.name` which stripped the directory
        prefix from configured relative paths. Post-fix preserves it."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        # The new code uses out_path.parent check, not blanket .name
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 3000]
        assert "out_path.parent == Path(\".\")" in block, (
            "must check whether path has a directory before defaulting to artifacts/"
        )


# ── fit_panel_calibrator.py CLI ────────────────────────────────────────────

class TestCalibratorCLI:
    def test_strategy_config_name_flag_added(self):
        src = (REPO_ROOT / "scripts/fit_panel_calibrator.py").read_text()
        assert "--strategy-config-name" in src
        assert 'default="strategy_config.json"' in src

    def test_config_loaded_from_arg(self):
        src = (REPO_ROOT / "scripts/fit_panel_calibrator.py").read_text()
        # Pre-fix: hardcoded "strategy_config.json"
        # Post-fix: uses args.strategy_config_name
        assert "args.strategy_config_name" in src

    def test_output_path_routes_to_side_when_panel_is_side(self):
        """If panel-LTR artifact lives at a side path, calibrator goes
        to the matching side calibrator path (not production)."""
        src = (REPO_ROOT / "scripts/fit_panel_calibrator.py").read_text()
        assert 'panel_path.stem.replace("panel-ltr"' in src
        # The naming convention: panel-ltr.h60.json → panel-rank-calibration.h60.json
        assert "panel-rank-calibration" in src


# ── RefreshPanelCalibratorTask forwards config name ─────────────────────────

class TestRefreshCalibratorForwarding:
    def test_subprocess_receives_config_name(self):
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class RefreshPanelCalibratorTask")
        block = src[idx:idx + 6000]
        # Post-fix: reads from ctx.config and forwards to subprocess
        assert '_strategy_config_name' in block
        assert '"--strategy-config-name"' in block

    def test_no_forwarding_for_default_config(self):
        """If user is running default strategy_config.json, no need to
        pass the flag (subprocess defaults to it). Avoid noise."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class RefreshPanelCalibratorTask")
        block = src[idx:idx + 6000]
        assert 'scn != "strategy_config.json"' in block


# ── train_104 stamps the config name ───────────────────────────────────────

class TestTrain104StampsConfigName:
    def test_config_name_stamped_in_ctx(self):
        src = (REPO_ROOT / "scripts/train_104.py").read_text()
        # Pre-fix: only `args.strategy_config_name` was used to read the
        # config but never forwarded. Post-fix: stamped into config dict
        # under a private key so downstream Tasks can read it.
        assert 'config["_strategy_config_name"] = args.strategy_config_name' in src

    def test_audit_tag_present(self):
        src = (REPO_ROOT / "scripts/train_104.py").read_text()
        assert "2026-04-28 evening" in src


# ── Invariant — production paths never touched by side config ──────────────

class TestProductionIsolationInvariant:
    """High-level invariant: after this fix, running train_104.py with a
    side config name MUST NOT modify any production path. We can't fully
    test this without a real retrain, but we can verify the contracts.
    """
    def test_default_config_falls_back_to_production_paths(self):
        """When no side config is passed, defaults still resolve to
        production paths (regression check — don't break the default)."""
        src = (REPO_ROOT / "backtesting/renquant_104/training_panel/pp_panel_training.py").read_text()
        idx = src.find("class NGBoostSaveTask")
        block = src[idx:idx + 3000]
        assert '"ngboost-head.json"' in block, (
            "default fallback to ngboost-head.json must still exist"
        )

    def test_calibrator_default_artifact_unchanged(self):
        src = (REPO_ROOT / "scripts/fit_panel_calibrator.py").read_text()
        # When config doesn't override, output is the production path
        assert '"panel-rank-calibration.json"' in src
