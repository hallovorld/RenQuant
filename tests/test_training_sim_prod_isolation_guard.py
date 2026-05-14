"""FullTrainingPipeline must refuse to overwrite prod artifacts from a side config.

§5.13.13 regression guard. The training pipeline's artifact_path default
points at `artifacts/prod/...`. If a sim/research side config forgets to
override panel_ltr.artifact_path, training would silently corrupt the
production model. This pins the guard that raises instead.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_ctx(strategy_dir: Path, config: dict):
    """Minimal ctx fields RunPanelTrainingTask uses before training body."""
    return SimpleNamespace(
        strategy=config.get("model_name", "test"),
        strategy_dir=strategy_dir,
        config=config,
        ohlcv_all={"SPY": None},
        feature_frames={"AAPL": None},  # non-empty so it gets past initial gate
    )


class TestTrainingSimProdIsolation:

    def test_side_config_without_artifact_path_raises(self, tmp_path):
        """Side config (label set) + missing panel_ltr.artifact_path
        → default would be prod → guard must raise."""
        from kernel.pipeline.pp_training_full import RunPanelTrainingTask

        cfg = {
            "_side_config_label": "sim_test",
            "panel_ltr": {},
            "ranking": {"panel_scoring": {}},
            "sectors": {"sector_map": {}, "sector_etfs": {}},
            "sector_map": {},
            "sector_etf_map": {},
        }
        ctx = _make_ctx(tmp_path, cfg)
        with pytest.raises(ValueError, match="prod artifact"):
            RunPanelTrainingTask().run(ctx)

    def test_side_config_with_sim_artifact_path_ok(self, tmp_path):
        """Side config with explicit sim path → guard does NOT raise on
        the prod-path check (downstream errors are fine)."""
        from kernel.pipeline.pp_training_full import RunPanelTrainingTask

        cfg = {
            "_side_config_label": "sim_test",
            "panel_ltr": {
                "artifact_path": "artifacts/sim/panel-ltr.test.json",
            },
            "ranking": {"panel_scoring": {}},
            "sectors": {"sector_map": {}, "sector_etfs": {}},
            "sector_map": {},
            "sector_etf_map": {},
        }
        ctx = _make_ctx(tmp_path, cfg)
        try:
            RunPanelTrainingTask().run(ctx)
        except ValueError as exc:
            assert "prod artifact" not in str(exc), (
                "Should not have hit the prod-path guard when sim path is set"
            )
        except Exception:
            pass  # downstream errors are fine

    def test_prod_config_without_label_uses_default_ok(self, tmp_path):
        """No _side_config_label → prod default is intentional → guard skips."""
        from kernel.pipeline.pp_training_full import RunPanelTrainingTask

        cfg = {
            "panel_ltr": {},
            "ranking": {"panel_scoring": {}},
            "sectors": {"sector_map": {}, "sector_etfs": {}},
            "sector_map": {},
            "sector_etf_map": {},
        }
        ctx = _make_ctx(tmp_path, cfg)
        try:
            RunPanelTrainingTask().run(ctx)
        except ValueError as exc:
            assert "prod artifact" not in str(exc), (
                "Should not refuse prod default when no side config label"
            )
        except Exception:
            pass  # downstream errors are fine
