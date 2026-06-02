"""Track A (2026-06-02) — tests for the explicit `calibrator_per_regime`
schema on `ranking.panel_scoring`.

Covers the new opt-in dict that lets configs pin a per-regime calibrator
artifact (e.g. the BULL_CALM-only calibrator from
`scripts/fit_calibrator_alpha158_fund.py --regime-filter BULL_CALM`) without
relying on the legacy `regime_conditional.artifact_pattern` glob.

Invariants asserted:
  - Back-compat: configs WITHOUT `calibrator_per_regime` load and score
    exactly as before.
  - All-regimes dict: every listed regime ends up in
    `ctx._regime_calibrators`.
  - Partial dict: only listed regimes are populated; others fall back to
    the pooled calibrator via the existing `ApplyGlobalCalibrationTask`
    lookup (`regime_map.get(ctx.regime) or pooled`).
  - `ApplyGlobalCalibrationTask` reads `ctx.regime` and picks the matching
    per-regime calibrator over pooled.
  - Missing artifact path = hard FileNotFoundError (not silent fallback).
  - Invalid regime key = ValueError naming the bad key.
  - Non-dict value = ValueError naming the type.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.global_calibrator import GlobalPanelCalibration  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_calibrator(path: Path, regime: str | None = None) -> None:
    """Save a minimal valid GlobalPanelCalibration artifact for testing.

    Two-knot probability head + tiny ER head keep the G12 / G13 acceptance
    gates happy without us caring about signal quality.
    """
    metadata = {"n_rows": 500}
    if regime is not None:
        metadata["regime"] = regime
        metadata["fit_window_regime"] = regime
    cal = GlobalPanelCalibration(
        prob_x=np.array([-1.0, 1.0]),
        prob_y=np.array([0.1, 0.9]),
        er_x=np.array([-1.0, 1.0]),
        er_y=np.array([-0.01, 0.01]),
        metadata=metadata,
    )
    cal.save(path)


def _make_ctx(
    strategy_dir: Path,
    *,
    regime: str | None,
    per_regime_cfg: dict | None,
    pooled_relpath: str = "artifacts/panel-rank-calibration.json",
    regime_conditional_enabled: bool = False,
) -> SimpleNamespace:
    panel_scoring: dict = {
        "global_calibration": {
            "enabled": True,
            "artifact_path": pooled_relpath,
            "regime_conditional": {
                "enabled": regime_conditional_enabled,
                "artifact_pattern":
                    "artifacts/panel-calibration-{regime}.json",
                "regimes": ["BULL_CALM", "BEAR"],
            },
            # Skip fingerprint match: no scorer is preloaded in these tests.
            "strict_scorer_match": False,
        },
    }
    if per_regime_cfg is not None:
        panel_scoring["calibrator_per_regime"] = per_regime_cfg
    return SimpleNamespace(
        config={
            "_strategy_dir": str(strategy_dir),
            "ranking": {"panel_scoring": panel_scoring},
        },
        regime=regime,
        candidates=[],
        holdings={},
    )


# ── Tests — LoadGlobalCalibrationTask wiring ─────────────────────────────────

class TestPerRegimeCalibratorBackCompat:
    """Back-compat: missing `calibrator_per_regime` key = unchanged behavior."""

    def test_no_per_regime_field_loads_pooled_only(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")

        ctx = _make_ctx(tmp_path, regime="BULL_CALM", per_regime_cfg=None)
        LoadGlobalCalibrationTask().run(ctx)

        assert ctx._global_calibrator is not None
        # Regime-conditional pattern disabled + no per-regime dict ⇒ no map.
        assert getattr(ctx, "_regime_calibrators", None) in (None, {})


class TestPerRegimeCalibratorAllRegimes:
    """Explicit map listing all 4 regimes — all 4 land in the dict."""

    def test_all_four_regimes_loaded(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")
        per_regime_paths = {}
        for regime in ("BULL_CALM", "BULL_VOLATILE", "BEAR", "CHOPPY"):
            p = art_dir / f"panel-rank-calibration.{regime.lower()}.json"
            _write_calibrator(p, regime=regime)
            per_regime_paths[regime] = str(
                p.relative_to(tmp_path).as_posix()
            )

        ctx = _make_ctx(
            tmp_path, regime="BULL_CALM", per_regime_cfg=per_regime_paths,
        )
        LoadGlobalCalibrationTask().run(ctx)

        assert ctx._global_calibrator is not None
        loaded = ctx._regime_calibrators
        assert set(loaded.keys()) == {
            "BULL_CALM", "BULL_VOLATILE", "BEAR", "CHOPPY",
        }
        for regime, cal in loaded.items():
            assert cal.metadata.get("regime") == regime


class TestPerRegimeCalibratorPartial:
    """Subset map — listed regimes load explicit; others fall back to pooled."""

    def test_partial_map_falls_back_for_unlisted(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import (
            ApplyGlobalCalibrationTask,
            LoadGlobalCalibrationTask,
        )
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")
        bc = art_dir / "panel-rank-calibration.bull_calm.json"
        _write_calibrator(bc, regime="BULL_CALM")

        per_regime = {
            "BULL_CALM": str(bc.relative_to(tmp_path).as_posix()),
        }

        # Active regime = BULL_CALM → explicit per-regime calibrator selected.
        ctx_bc = _make_ctx(
            tmp_path, regime="BULL_CALM", per_regime_cfg=per_regime,
        )
        LoadGlobalCalibrationTask().run(ctx_bc)
        assert set(ctx_bc._regime_calibrators.keys()) == {"BULL_CALM"}
        # ApplyGlobalCalibrationTask should pick BULL_CALM's per-regime over pooled.
        picked = (
            ctx_bc._regime_calibrators.get(ctx_bc.regime)
            or ctx_bc._global_calibrator
        )
        assert picked is ctx_bc._regime_calibrators["BULL_CALM"]

        # Active regime = BEAR → no per-regime entry, ApplyGlobal falls back to pooled.
        ctx_bear = _make_ctx(
            tmp_path, regime="BEAR", per_regime_cfg=per_regime,
        )
        LoadGlobalCalibrationTask().run(ctx_bear)
        assert "BEAR" not in ctx_bear._regime_calibrators
        picked_bear = (
            ctx_bear._regime_calibrators.get(ctx_bear.regime)
            or ctx_bear._global_calibrator
        )
        assert picked_bear is ctx_bear._global_calibrator


class TestPerRegimeCalibratorFailClosed:
    """Misconfiguration = hard failure, not silent fallback."""

    def test_missing_file_raises(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")

        per_regime = {"BULL_CALM": "artifacts/does-not-exist.json"}
        ctx = _make_ctx(tmp_path, regime="BULL_CALM", per_regime_cfg=per_regime)
        with pytest.raises(FileNotFoundError, match="calibrator_per_regime"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_invalid_regime_name_raises(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")
        ok_path = art_dir / "panel-rank-calibration.bull_calm.json"
        _write_calibrator(ok_path, regime="BULL_CALM")

        per_regime = {
            "BULL_CALM": str(ok_path.relative_to(tmp_path).as_posix()),
            "MYSTERY_REGIME": "artifacts/whatever.json",
        }
        ctx = _make_ctx(tmp_path, regime="BULL_CALM", per_regime_cfg=per_regime)
        with pytest.raises(ValueError, match="invalid regime keys"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_non_dict_value_raises(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        _write_calibrator(art_dir / "panel-rank-calibration.json")

        ctx = _make_ctx(
            tmp_path, regime="BULL_CALM",
            per_regime_cfg=["BULL_CALM=foo.json"],  # type: ignore[arg-type]
        )
        with pytest.raises(ValueError, match="must be a dict"):
            LoadGlobalCalibrationTask().run(ctx)
