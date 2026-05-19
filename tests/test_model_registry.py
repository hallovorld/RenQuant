"""Regression tests for kernel/panel_pipeline/model_registry.py.

Pin the registry pattern so future model additions follow the same
interface. Verifies:
  1. Built-in handlers (xgb, patchtst) are registered
  2. Each handler exposes scorer_loader + train_cmd
  3. Unknown kind raises clear error
  4. New registrations work via decorator
  5. requires_history flag is correctly set
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting/renquant_104"))


@pytest.fixture(scope="module")
def registry_mod():
    from kernel.panel_pipeline.model_registry import registry, _ModelHandler
    return registry, _ModelHandler


class TestBuiltins:
    def test_xgb_registered(self, registry_mod):
        registry, _ = registry_mod
        h = registry.get("xgb")
        assert h.kind == "xgb"
        assert h.requires_history == False
        assert callable(h.scorer_loader)
        assert callable(h.train_cmd)

    def test_patchtst_registered(self, registry_mod):
        registry, _ = registry_mod
        h = registry.get("patchtst")
        assert h.kind == "patchtst"
        assert h.requires_history == True
        assert callable(h.scorer_loader)
        assert callable(h.train_cmd)

    def test_list_returns_kinds(self, registry_mod):
        registry, _ = registry_mod
        kinds = registry.list()
        assert "xgb" in kinds
        assert "patchtst" in kinds


class TestErrorPath:
    def test_unknown_kind_raises(self, registry_mod):
        registry, _ = registry_mod
        with pytest.raises(ValueError, match="not registered"):
            registry.get("nonexistent_model")

    def test_error_lists_available(self, registry_mod):
        registry, _ = registry_mod
        try:
            registry.get("nonexistent_model")
        except ValueError as exc:
            msg = str(exc)
            assert "xgb" in msg
            assert "patchtst" in msg


class TestExtensibility:
    """Pin the registry's decorator pattern so future model adders use
    the same interface."""

    def test_register_decorator(self, registry_mod):
        registry, _ModelHandler = registry_mod

        @registry.register("fake_lgbm_for_test")
        class FakeLGBMHandler(_ModelHandler):
            requires_history = False
            @classmethod
            def scorer_loader(cls, p, cfg):
                return None
            @classmethod
            def train_cmd(cls, args):
                return ["fake_command"]

        h = registry.get("fake_lgbm_for_test")
        assert h.kind == "fake_lgbm_for_test"
        # Cleanup
        del registry._handlers["fake_lgbm_for_test"]


class TestTrainCmdProduces():
    """train_cmd should produce a runnable shell command list."""

    def test_xgb_train_cmd_has_required_args(self, registry_mod):
        registry, _ = registry_mod
        h = registry.get("xgb")
        # Mock args object
        class Args:
            dataset = "/tmp/test.parquet"
            output = "/tmp/test.json"
            label = "fwd_60d_excess"
            seed = 42
        cmd = h.train_cmd(Args)
        assert "--dataset" in cmd
        assert "--output" in cmd
        assert "--label" in cmd
        assert "--seed" in cmd

    def test_patchtst_train_cmd_has_required_args(self, registry_mod):
        registry, _ = registry_mod
        h = registry.get("patchtst")
        class Args:
            dataset = "/tmp/test.parquet"
            output_dir = "/tmp/test_dir"
            label = "fwd_60d_excess"
            seq_len = 32
            epochs = 10
            num_seeds = 5
            device = "cpu"
        cmd = h.train_cmd(Args)
        assert "--arch" in cmd
        assert "patchtst" in cmd
        assert "--num-seeds" in cmd
        assert "--seq-len" in cmd


class TestInferenceDispatchInLoadScorerTask:
    """Pin: LoadScorerTask reads `kind` from config + dispatches via registry."""

    def test_load_scorer_uses_registry(self):
        src = (REPO / "backtesting/renquant_104/kernel/panel_pipeline"
               / "job_panel_scoring.py").read_text()
        assert "from kernel.panel_pipeline.model_registry import registry" in src
        assert 'panel_cfg.get("kind", "xgb")' in src
        assert "handler.scorer_loader" in src
