"""Tests for Plan F — regime-conditional global calibrator.

Covers:
  - fit_regime_conditional splits rows by regime label
  - Per-regime isotonic differs when training data differs
  - Sparse regimes (< min_rows) are skipped silently
  - Artifact round-trip preserves regime metadata
  - LoadGlobalCalibrationTask wires both pooled + regime dicts
  - ApplyGlobalCalibrationTask picks the right calibrator by ctx.regime
  - Missing per-regime artifact falls back to pooled
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from training_panel.global_calibrator import (  # noqa: E402
    GlobalPanelCalibration,
    fit_global_calibrator,
    fit_regime_conditional,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _synth_panel(seed: int, n_dates: int = 400, n_tickers: int = 6,
                  return_offset: float = 0.0, noise: float = 0.02):
    """Fabricate (scores, returns) for N tickers across N_dates trading days.

    Returns are structured so that higher score → higher forward return by
    default, with an optional `return_offset` added to shift the map.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    scores:  dict[str, pd.Series] = {}
    returns: dict[str, pd.Series] = {}
    for i in range(n_tickers):
        raw = rng.normal(0, 1, n_dates)
        fwd = 0.01 * raw + return_offset + rng.normal(0, noise, n_dates)
        ticker = f"T{i:02d}"
        scores[ticker]  = pd.Series(raw, index=dates)
        returns[ticker] = pd.Series(fwd, index=dates)
    return dates, scores, returns


def _regime_series(dates: pd.DatetimeIndex,
                    pattern: list[tuple[str, int]]) -> pd.Series:
    """Build a regime label series by repeating `pattern` [(label, days), …]."""
    labels: list[str] = []
    i = 0
    for label, n in pattern:
        labels += [label] * n
    labels = (labels * ((len(dates) // max(len(labels), 1)) + 1))[: len(dates)]
    return pd.Series(labels, index=dates)


# ── fit_regime_conditional ──────────────────────────────────────────────────

class TestFitRegimeConditional:
    def test_splits_by_label(self):
        dates, scores, returns = _synth_panel(seed=1, n_dates=400)
        reg = _regime_series(
            dates,
            [("BULL_CALM", 100), ("BEAR", 100), ("CHOPPY", 100), ("BULL_VOLATILE", 100)],
        )
        out = fit_regime_conditional(
            scores, returns, reg,
            min_rows_per_regime=100,
        )
        assert set(out.keys()) == {"BULL_CALM", "BEAR", "CHOPPY", "BULL_VOLATILE"}
        # Each calibrator should stamp its regime into metadata
        for regime, cal in out.items():
            assert cal.metadata["regime"] == regime
            assert cal.metadata["n_rows"] > 0

    def test_skips_regimes_below_floor(self):
        """A regime with < min_rows_per_regime rows is silently dropped."""
        dates, scores, returns = _synth_panel(seed=2, n_dates=400)
        reg = _regime_series(
            dates, [("BULL_CALM", 395), ("BEAR", 5)],  # BEAR has only 5 × 6 = 30 rows
        )
        out = fit_regime_conditional(
            scores, returns, reg,
            min_rows_per_regime=1000,
        )
        assert "BULL_CALM" in out
        assert "BEAR" not in out

    def test_different_regimes_fit_different_maps(self):
        """Per-regime isotonic outputs differ when the distributions differ."""
        dates1 = pd.bdate_range("2024-01-02", periods=300)
        # Regime A: raw → +return, Regime B: raw → weaker correlation
        rng = np.random.default_rng(42)
        scores: dict[str, pd.Series] = {}
        returns: dict[str, pd.Series] = {}
        for i in range(6):
            raw = rng.normal(0, 1, 300)
            # First 150 days (Regime A) — strong signal
            fwd_a = 0.02 * raw[:150] + rng.normal(0, 0.01, 150)
            # Next 150 days (Regime B) — weak signal + offset
            fwd_b = 0.002 * raw[150:] - 0.005 + rng.normal(0, 0.01, 150)
            fwd = np.concatenate([fwd_a, fwd_b])
            scores[f"T{i}"]  = pd.Series(raw, index=dates1)
            returns[f"T{i}"] = pd.Series(fwd, index=dates1)
        reg = pd.Series(
            ["BULL_CALM"] * 150 + ["BEAR"] * 150, index=dates1,
        )
        out = fit_regime_conditional(
            scores, returns, reg,
            min_rows_per_regime=100, threshold=0.01,
        )
        # Both fit
        assert "BULL_CALM" in out and "BEAR" in out
        # P(outperform) at score=+2 should be higher in BULL_CALM than BEAR
        p_bull = out["BULL_CALM"].calibrate_probability(2.0)
        p_bear = out["BEAR"].calibrate_probability(2.0)
        assert p_bull > p_bear, f"BULL_CALM P={p_bull} should beat BEAR P={p_bear}"

    def test_artifact_round_trip_preserves_regime(self, tmp_path):
        dates, scores, returns = _synth_panel(seed=3, n_dates=300)
        reg = _regime_series(dates, [("BULL_CALM", 300)])
        out = fit_regime_conditional(scores, returns, reg, min_rows_per_regime=100)
        cal = out["BULL_CALM"]
        p = tmp_path / "panel-calibration-BULL_CALM.json"
        cal.save(p)
        loaded = GlobalPanelCalibration.load(p)
        assert loaded.metadata.get("regime") == "BULL_CALM"
        # Same interp surface
        for x in (-2.0, 0.0, 2.0):
            assert cal.calibrate_probability(x) == pytest.approx(
                loaded.calibrate_probability(x),
            )


# ── LoadGlobalCalibrationTask ────────────────────────────────────────────────

class TestLoadRegimeCalibrators:
    def _make_ctx(self, strategy_dir: Path, regime: str | None = None,
                   regime_enabled: bool = True) -> SimpleNamespace:
        return SimpleNamespace(
            config={
                "_strategy_dir": str(strategy_dir),
                "ranking": {"panel_scoring": {"global_calibration": {
                    "enabled": True,
                    "artifact_path": "artifacts/panel-rank-calibration.json",
                    "regime_conditional": {
                        "enabled": regime_enabled,
                        "artifact_pattern":
                            "artifacts/panel-calibration-{regime}.json",
                        "regimes": ["BULL_CALM", "BEAR"],
                    },
                }}},
            },
            regime=regime,
            candidates=[],
            holdings={},
        )

    def _write_calibrator(
        self,
        path: Path,
        regime: str | None,
        scorer_fp: str | None = None,
    ):
        # Minimal valid artifact — 2 knots
        metadata = {"n_rows": 500, "regime": regime} if regime else {"n_rows": 500}
        if scorer_fp:
            metadata["scorer_artifact_fingerprint"] = scorer_fp
        cal = GlobalPanelCalibration(
            prob_x=np.array([-1.0, 1.0]), prob_y=np.array([0.1, 0.9]),
            er_x=np.array([-1.0, 1.0]), er_y=np.array([-0.01, 0.01]),
            metadata=metadata,
        )
        cal.save(path)

    def test_loads_regime_dict_when_files_present(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        self._write_calibrator(art_dir / "panel-rank-calibration.json", None)
        self._write_calibrator(art_dir / "panel-calibration-BULL_CALM.json", "BULL_CALM")
        self._write_calibrator(art_dir / "panel-calibration-BEAR.json", "BEAR")

        ctx = self._make_ctx(tmp_path)
        LoadGlobalCalibrationTask().run(ctx)
        assert ctx._global_calibrator is not None
        assert set(ctx._regime_calibrators.keys()) == {"BULL_CALM", "BEAR"}

    def test_missing_regime_file_falls_back(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        self._write_calibrator(art_dir / "panel-rank-calibration.json", None)
        # Only BULL_CALM present → BEAR should be missing
        self._write_calibrator(art_dir / "panel-calibration-BULL_CALM.json", "BULL_CALM")

        ctx = self._make_ctx(tmp_path)
        LoadGlobalCalibrationTask().run(ctx)
        assert "BULL_CALM" in ctx._regime_calibrators
        assert "BEAR" not in ctx._regime_calibrators

    def test_regime_conditional_disabled_does_not_populate(self, tmp_path):
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        self._write_calibrator(art_dir / "panel-rank-calibration.json", None)
        self._write_calibrator(art_dir / "panel-calibration-BULL_CALM.json", "BULL_CALM")

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        LoadGlobalCalibrationTask().run(ctx)
        assert ctx._global_calibrator is not None
        assert getattr(ctx, "_regime_calibrators", None) in (None, {})

    def test_strict_contract_rejects_foreign_calibrator(self, tmp_path):
        """A scorer must not silently consume another model's calibrator."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        self._write_calibrator(
            art_dir / "panel-rank-calibration.json",
            None,
            scorer_fp="sha256:foreign000000",
        )

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"artifact_fingerprint": "sha256:active111111"}
        )
        with pytest.raises(ValueError, match="fingerprint mismatch"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_strict_contract_accepts_short_artifact_fingerprint(self, tmp_path):
        """Historical artifacts may stamp short scorer sha prefixes."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        full_fp = "sha256:abcdef1234567890deadbeef"
        short_fp = "sha256:abcdef1234567890"
        self._write_calibrator(
            art_dir / "panel-rank-calibration.json",
            None,
            scorer_fp=short_fp,
        )

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"artifact_fingerprint": full_fp}
        )
        LoadGlobalCalibrationTask().run(ctx)
        assert ctx._global_calibrator is not None

    def test_strict_contract_rejects_preloaded_foreign_calibrator(self, tmp_path):
        """WF-preloaded calibrators must obey the same scorer binding."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cal_path = art_dir / "panel-rank-calibration.json"
        self._write_calibrator(cal_path, None, scorer_fp="sha256:foreign000000")

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        ctx._global_calibrator = GlobalPanelCalibration.load(cal_path)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"artifact_fingerprint": "sha256:active111111"}
        )

        with pytest.raises(ValueError, match="fingerprint mismatch"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_strict_contract_rejects_missing_calibrator_fingerprint(self, tmp_path):
        """A newly fitted calibrator must carry scorer_artifact_fingerprint."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        self._write_calibrator(art_dir / "panel-rank-calibration.json", None)

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"artifact_fingerprint": "sha256:active111111"}
        )
        with pytest.raises(ValueError, match="missing scorer/calibrator fingerprint"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_strict_contract_rejects_config_only_preloaded_calibrator(self, tmp_path):
        """Config identity is not enough to bind a WF scorer to a calibrator."""
        from kernel.panel_pipeline.job_panel_scoring import LoadGlobalCalibrationTask
        art_dir = tmp_path / "artifacts"
        art_dir.mkdir()
        cal_path = art_dir / "panel-rank-calibration.json"
        cal = GlobalPanelCalibration(
            prob_x=np.array([-1.0, 1.0]),
            prob_y=np.array([0.25, 0.75]),
            er_x=np.array([-1.0, 1.0]),
            er_y=np.array([-0.01, 0.01]),
            metadata={"config_fingerprint": "sha256:sharedconfig"},
        )
        cal.save(cal_path)

        ctx = self._make_ctx(tmp_path, regime_enabled=False)
        ctx._global_calibrator = GlobalPanelCalibration.load(cal_path)
        ctx._panel_scorer = SimpleNamespace(
            metadata={"config_fingerprint": "sha256:sharedconfig"}
        )

        with pytest.raises(ValueError, match="missing scorer/calibrator fingerprint"):
            LoadGlobalCalibrationTask().run(ctx)

    def test_artifact_fingerprint_is_file_identity_not_config_identity(self, tmp_path):
        """A calibrator must bind to the exact scorer file, not merely any
        model trained under the same strategy_config fingerprint."""
        from kernel.panel_pipeline.panel_scorer import stamp_artifact_metadata
        p = tmp_path / "scorer.json"
        p.write_text("scorer-v1")

        meta = stamp_artifact_metadata(
            {"config_fingerprint": "sha256:config000000000000"},
            p,
        )

        assert meta["config_fingerprint"] == "sha256:config000000000000"
        assert meta["artifact_fingerprint"] == meta["artifact_sha256"]
        assert meta["artifact_fingerprint"] != meta["config_fingerprint"]


# ── ApplyGlobalCalibrationTask dispatch ──────────────────────────────────────

class TestApplyGlobalCalibrationDispatch:
    def _make_ctx(self, regime: str | None,
                   pooled: GlobalPanelCalibration | None,
                   regime_map: dict[str, GlobalPanelCalibration] | None):
        # Stub candidate — ApplyGlobalCalibrationTask only reads panel_score
        # and writes rank_score / expected_return.
        cand = SimpleNamespace(
            ticker="AAPL", role="buy", panel_score=0.5, rank_score=0.0,
            expected_return=0.0,
        )
        ctx = SimpleNamespace(
            config={"ranking": {"panel_scoring": {"global_calibration": {"enabled": True}}}},
            regime=regime,
            candidates=[cand],
            holdings={},
            _global_calibrator=pooled,
            _regime_calibrators=regime_map or {},
        )
        return ctx, cand

    def _cal(self, out_prob: float) -> GlobalPanelCalibration:
        return GlobalPanelCalibration(
            prob_x=np.array([0.0, 1.0]), prob_y=np.array([out_prob, out_prob]),
            er_x=np.array([0.0, 1.0]), er_y=np.array([0.0, 0.0]),
            metadata={},
        )

    def test_uses_regime_calibrator_when_available(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        pooled = self._cal(0.5)
        regime_map = {"BEAR": self._cal(0.8)}
        ctx, cand = self._make_ctx("BEAR", pooled, regime_map)
        ApplyGlobalCalibrationTask().run(ctx)
        assert cand.rank_score == pytest.approx(0.8)

    def test_falls_back_to_pooled_when_regime_missing(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        pooled = self._cal(0.5)
        regime_map = {"BEAR": self._cal(0.8)}  # no BULL_CALM
        ctx, cand = self._make_ctx("BULL_CALM", pooled, regime_map)
        ApplyGlobalCalibrationTask().run(ctx)
        assert cand.rank_score == pytest.approx(0.5)

    def test_falls_back_to_pooled_when_regime_none(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyGlobalCalibrationTask
        pooled = self._cal(0.5)
        ctx, cand = self._make_ctx(None, pooled, {"BEAR": self._cal(0.8)})
        ApplyGlobalCalibrationTask().run(ctx)
        assert cand.rank_score == pytest.approx(0.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
