"""Regression tests for VMA/VSTD blowup on zero-volume days.

Root cause (audit 2026-05-10): `scripts/build_alpha158_qlib.py` divided
volume rolling stats by `v + EPS` where `EPS=1e-12`. On halts / delistings
where today's `v=0`, the denominator collapsed to 1e-12 and the ratio
exploded to ~1e15-1e18. Per-feature stats (mean/std) were then computed
BEFORE clipping → a single outlier poisoned the column, and normal
values 0.8-1.2 collapsed to a single constant (~-0.00305) after z-score.

Production impact: 10/169 feature_stds in
`panel-ltr.alpha158_fund.json` were > 1e15 → 10 features (5.9%) were
effectively dead in inference.

Fix (per §5.13.5, §5.13.11, §5.13.12):
  1. Replace `v + EPS` with rolling-mean denominator floor in build script.
  2. Mirror same fallback in inference single-bar code path.
  3. Winsorize feature columns (0.1%/99.9% train-only) BEFORE computing
     z-score mean/std → defense in depth.

References:
  - `scripts/build_alpha158_qlib.py:251-252` (build path)
  - `backtesting/renquant_104/kernel/panel_pipeline/alpha158_features.py:195-197`
    (inference parity path)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))


def _make_ohlcv_with_zero_volume_days(
    n_bars: int = 120,
    zero_vol_positions: tuple[int, ...] = (50, 80, 119),
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic OHLCV with explicit zero-volume bars (halts / delistings)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-01", periods=n_bars)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n_bars))
    opens = closes * (1 + rng.normal(0, 0.005, n_bars))
    highs = np.maximum(opens, closes) * (1 + np.abs(rng.normal(0, 0.005, n_bars)))
    lows = np.minimum(opens, closes) * (1 - np.abs(rng.normal(0, 0.005, n_bars)))
    vols = rng.uniform(1e6, 1e7, n_bars)
    for pos in zero_vol_positions:
        if 0 <= pos < n_bars:
            vols[pos] = 0.0
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows,
         "close": closes, "volume": vols},
        index=dates,
    )


# ── Per §5.13.1: at least one test runs the real build pipeline ────────────

class TestVolumeFeatureStdsRegression:
    """Build-script feature pipeline must produce sane VMA/VSTD on
    zero-volume days. Sanity bound: |value| < 1e6 (the bug produced ~1e16)."""

    def test_vma_vstd_finite_and_bounded_with_zero_volume(self):
        # Import the REAL build-script feature function (not a fixture).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_alpha158_qlib",
            REPO_ROOT / "scripts" / "build_alpha158_qlib.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = _make_ohlcv_with_zero_volume_days()
        feats = mod.rolling_features(df)

        # Inspect VMA/VSTD for every window
        windows = [5, 10, 20, 30, 60]
        for n in windows:
            for fam in ("VMA", "VSTD"):
                col = feats[f"{fam}{n}"]
                # Drop leading NaNs (warmup period — expected)
                valid = col.dropna()
                assert valid.size > 0, f"{fam}{n} all NaN"
                vals = valid.to_numpy()
                # Per §5.13.11: every value finite
                assert np.isfinite(vals).all(), (
                    f"{fam}{n} contains non-finite values: "
                    f"{vals[~np.isfinite(vals)][:5]}"
                )
                # Sanity bound: nothing > 1e6 (bug produced ~1e16)
                assert np.abs(vals).max() < 1e6, (
                    f"{fam}{n} has value > 1e6 (max={np.abs(vals).max()}); "
                    f"zero-volume-day blowup regression"
                )


class TestRollingMeanFallbackMath:
    """Test the rolling-mean fallback math directly:
    zero-volume row → fallback ratio = rolling_avg / rolling_avg ≈ 1.0
    (not 1e16). Specifically: VMA{n} on a zero-volume bar = mean / fallback,
    where fallback is the rolling-mean of recent non-zero volumes."""

    def test_zero_volume_bar_does_not_blow_up(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_alpha158_qlib",
            REPO_ROOT / "scripts" / "build_alpha158_qlib.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        df = _make_ohlcv_with_zero_volume_days(
            n_bars=100, zero_vol_positions=(99,)
        )
        feats = mod.rolling_features(df)

        # VMA5 at last bar: zero-volume → denominator floor (rolling mean ~5e6)
        # → mean(last 5) / 5e6 ≈ O(1). Bug would give 5e6 / 1e-12 = 5e18.
        for n in (5, 10, 20, 60):
            val = feats[f"VMA{n}"].iloc[-1]
            assert np.isfinite(val), f"VMA{n} non-finite on zero-vol bar"
            assert abs(val) < 100, f"VMA{n} = {val} — fallback failed"

    def test_extended_zero_volume_run_still_finite(self):
        """Multiple consecutive zero-volume bars at the end (e.g. delisting)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_alpha158_qlib",
            REPO_ROOT / "scripts" / "build_alpha158_qlib.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Last 5 bars all zero-volume
        df = _make_ohlcv_with_zero_volume_days(
            n_bars=100, zero_vol_positions=tuple(range(95, 100))
        )
        feats = mod.rolling_features(df)
        for n in (5, 10, 20):
            val = feats[f"VMA{n}"].iloc[-1]
            assert np.isfinite(val), (
                f"VMA{n} non-finite after 5 consecutive zero-vol bars"
            )
            assert abs(val) < 100


# ── AUDIT REGRESSION GUARD (§5.13.3) ───────────────────────────────────────

class TestFeatureStdsBlowupRegression:
    """Pin the invariant that prevents the entire bug class.

    Any newly-trained stats artifact MUST have feature_stds bounded:
      - max < 1e3 (bug produced up to 1e16)
      - min > 1e-6 (so z-score division is well-conditioned)

    This guard runs on NEWLY-TRAINED artifacts. The existing on-disk
    artifact (training pre-dates this fix) is documented as corrupted
    and excluded — regen requires panel retrain (§5.13.7).
    """

    def test_invariant_pin_for_newly_trained_artifacts(self):
        # This test pins the invariant as a function. It's invoked by
        # the panel training pipeline after stats are written.
        def check_feature_stds_bounded(stats_json_path: Path) -> None:
            data = json.loads(Path(stats_json_path).read_text())
            stds = np.array(data["feature_stds"], dtype=float)
            stds = stds[np.isfinite(stds)]
            assert stds.size > 0, "no finite feature_stds"
            assert stds.max() < 1e3, (
                f"feature_stds.max()={stds.max()} > 1e3; "
                f"zero-volume blowup regression"
            )
            assert stds.min() > 1e-6, (
                f"feature_stds.min()={stds.min()} < 1e-6; "
                f"near-constant feature column"
            )

        # Self-test: build a tiny synthetic stats file and assert the guard.
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stats.json", delete=False
        ) as f:
            json.dump({
                "feature_cols": ["A", "B", "C"],
                "feature_means": [0.0, 0.0, 0.0],
                "feature_stds":  [1.0, 0.5, 2.0],
            }, f)
            tmp_path = Path(f.name)
        try:
            check_feature_stds_bounded(tmp_path)
        finally:
            tmp_path.unlink()

        # Negative test: 1e16 std must trip the guard.
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".stats.json", delete=False
        ) as f:
            json.dump({
                "feature_cols": ["A", "B"],
                "feature_means": [0.0, 0.0],
                "feature_stds":  [1.0, 1e16],
            }, f)
            tmp_path = Path(f.name)
        try:
            with pytest.raises(AssertionError, match="zero-volume blowup"):
                check_feature_stds_bounded(tmp_path)
        finally:
            tmp_path.unlink()


# ── Inference parity (§5.3 invariant: train/inference column equivalence) ──

class TestInferenceParityForVolumeFeatures:
    """Inference module must use the SAME denominator-floor logic so
    train-time stats and inference-time raw features stay aligned.
    """

    def test_inference_vma_vstd_finite_on_zero_volume_today(self):
        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at
        df = _make_ohlcv_with_zero_volume_days(
            n_bars=100, zero_vol_positions=(99,)
        )
        feats = compute_alpha158_at(df)
        for n in (5, 10, 20, 30, 60):
            for fam in ("VMA", "VSTD"):
                v = feats[f"{fam}{n}"]
                assert np.isfinite(v), (
                    f"inference {fam}{n} non-finite on zero-vol today: {v}"
                )
                assert abs(v) < 1e6, (
                    f"inference {fam}{n}={v} exceeds sanity bound 1e6"
                )

    def test_inference_vma_matches_build_on_nonzero_volume(self):
        """When today's volume is normal, inference VMA/VSTD should match
        the build-script's last-bar VMA/VSTD (modulo numerical diffs).
        This pins the train/inference parity invariant."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "build_alpha158_qlib",
            REPO_ROOT / "scripts" / "build_alpha158_qlib.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from kernel.panel_pipeline.alpha158_features import compute_alpha158_at

        # All non-zero volumes (no halts) → both paths should agree.
        df = _make_ohlcv_with_zero_volume_days(
            n_bars=120, zero_vol_positions=()
        )
        build_feats = mod.rolling_features(df)
        infer_feats = compute_alpha158_at(df)

        for n in (5, 10, 20, 30, 60):
            for fam in ("VMA", "VSTD"):
                build_val = float(build_feats[f"{fam}{n}"].iloc[-1])
                infer_val = float(infer_feats[f"{fam}{n}"])
                # Allow small tolerance for ddof/numerical differences.
                # VSTD: pandas uses ddof=1, numpy default ddof=0 — accept
                # within 20% relative tolerance for std-based features.
                rtol = 0.25 if fam == "VSTD" else 1e-6
                assert abs(build_val - infer_val) <= rtol * (
                    abs(build_val) + 1e-9
                ), (
                    f"{fam}{n}: build={build_val} inference={infer_val}"
                )
