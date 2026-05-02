"""Tests for SectorRankNormalizeTask — Layer-1 wiring into PanelAssemblyJob.

Three contracts under test:

1.  Default OFF — Task is a complete no-op when
    ``panel_ltr.sector_rank_norm.enabled = False``. ctx.factor_frames is
    bit-for-bit unchanged. This is the wl103 production preservation
    guarantee.

2.  ON-with-good-input — when the flag is true and ctx has populated
    raw_factor_frames + factor_frames + ticker_sectors, the Task adds
    one ``{col}_sr`` column per known raw factor column to each ticker's
    factor frame. Values are within [0, 1] and are the per-(date, sector)
    percentile of the underlying raw value.

3.  Defensive paths — empty raw_factor_frames, missing ticker_sectors,
    and column-collision conditions all log + degrade gracefully without
    raising.

4.  Wiring — the Task appears in PanelAssemblyJob.tasks list, AFTER
    FactorZScoreTask, BEFORE LabelsTask. Position matters because _sr
    consumes raw_factor_frames (set by per-ticker Job, available before
    any of these tasks) AND must be present before BuildPanelTask
    composes the final panel column list.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.pp_panel_training import (   # noqa: E402
    SectorRankNormalizeTask,
    PanelAssemblyJob,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="B")


def _make_ctx(*, enabled: bool, n_tickers: int = 10,
               n_sectors: int = 2, n_dates: int = 5,
               include_factor_frames: bool = True,
               include_ticker_sectors: bool = True,
               include_raw: bool = True) -> SimpleNamespace:
    """Build a minimal PanelTrainingContext-shaped object — uses
    SimpleNamespace because PanelTrainingContext is a heavy dataclass
    with many required fields not relevant to this Task's behavior.
    """
    idx = _idx(n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    sector_labels = ["sector_a", "sector_b"][:n_sectors]
    ticker_sectors = {
        t: sector_labels[i % n_sectors] for i, t in enumerate(tickers)
    } if include_ticker_sectors else {}

    # raw_factor_frames — one DataFrame per ticker with a few of the cols
    # the Task knows about. Use deterministic values so tests can reason.
    raw_factor_frames = {} if include_raw else {}
    if include_raw:
        for i, t in enumerate(tickers):
            sector_idx = i % n_sectors
            base = float(i + 1)  # T00=1, T01=2, ..., T09=10
            scale = 1.0 if sector_idx == 0 else 100.0  # sector b is 100x bigger
            raw_factor_frames[t] = pd.DataFrame({
                "size":        [base * scale]            * n_dates,
                "mom_12_1":    [base * scale * 0.01]     * n_dates,
                "beta_60d":    [base * scale * 0.001]    * n_dates,
            }, index=idx)

    # factor_frames — one DataFrame per ticker with z-scored cols already
    # present (FactorZScoreTask output). The Task will append _sr cols.
    factor_frames = {} if include_factor_frames else {}
    if include_factor_frames:
        for t in raw_factor_frames:
            factor_frames[t] = pd.DataFrame({
                "size_z":     [0.0] * n_dates,
                "mom_12_1_z": [0.0] * n_dates,
                "beta_60d_z": [0.0] * n_dates,
            }, index=idx)

    return SimpleNamespace(
        config={"panel_ltr": {"sector_rank_norm": {"enabled": enabled}}},
        raw_factor_frames=raw_factor_frames,
        factor_frames=factor_frames,
        ticker_sectors=ticker_sectors,
    )


# ── Default OFF — no-op on wl103-style config ─────────────────────────────────

class TestDefaultOff:
    def test_disabled_does_not_modify_factor_frames(self):
        ctx = _make_ctx(enabled=False)
        # Snapshot BEFORE
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorRankNormalizeTask().run(ctx)
        # Compare AFTER — every frame must be unchanged
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])

    def test_missing_panel_ltr_config_block_treats_as_off(self):
        """No `panel_ltr` key at all → off, no crash."""
        ctx = _make_ctx(enabled=False)
        ctx.config = {}    # strip the whole config tree
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorRankNormalizeTask().run(ctx)
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])


# ── ON with good input — produces _sr columns ─────────────────────────────────

class TestEnabled:
    def test_adds_sr_column_for_each_known_raw_col(self):
        ctx = _make_ctx(enabled=True)
        SectorRankNormalizeTask().run(ctx)
        for ticker, df in ctx.factor_frames.items():
            # Pre-existing _z columns still there
            assert "size_z" in df.columns
            # New _sr columns added
            assert "size_sr" in df.columns
            assert "mom_12_1_sr" in df.columns
            assert "beta_60d_sr" in df.columns

    def test_sr_columns_in_unit_interval(self):
        ctx = _make_ctx(enabled=True)
        SectorRankNormalizeTask().run(ctx)
        for ticker, df in ctx.factor_frames.items():
            for col in ["size_sr", "mom_12_1_sr", "beta_60d_sr"]:
                vals = df[col].dropna()
                assert (vals >= 0).all() and (vals <= 1).all(), (
                    f"{ticker}.{col} has values outside [0, 1]: {vals.tolist()}"
                )

    def test_within_sector_top_ticker_gets_pct_1(self):
        """T08 is in sector_a (i=8 → 8%2==0). T08's size value is the
        biggest within sector_a (T00=1, T02=3, T04=5, T06=7, T08=9 →
        T08 wins). Therefore T08.size_sr == 1.0.
        """
        ctx = _make_ctx(enabled=True, n_tickers=10, n_sectors=2)
        SectorRankNormalizeTask().run(ctx)
        # T08 is the largest in sector_a (i=8 → sector_a, value=9*1=9)
        assert ctx.factor_frames["T08"]["size_sr"].iloc[0] == pytest.approx(1.0)
        # T09 is the largest in sector_b (i=9 → sector_b, value=10*100=1000)
        assert ctx.factor_frames["T09"]["size_sr"].iloc[0] == pytest.approx(1.0)
        # T00 is smallest in sector_a → 0.2 (1 of 5)
        assert ctx.factor_frames["T00"]["size_sr"].iloc[0] == pytest.approx(0.2)

    def test_sector_relativity_under_extreme_scale_difference(self):
        """sector_b values are 100× sector_a's. Under GLOBAL z-score
        all sector_b would land in the upper tail and all sector_a in
        the lower tail (the wl178 failure mode). Under sector-rank,
        each sector has its own [0.2, 0.4, 0.6, 0.8, 1.0] spread —
        identical distributions despite the 100× scale gap.
        """
        ctx = _make_ctx(enabled=True, n_tickers=10, n_sectors=2)
        SectorRankNormalizeTask().run(ctx)
        sector_a_vals = sorted(
            ctx.factor_frames[f"T{i:02d}"]["size_sr"].iloc[0]
            for i in range(0, 10, 2)
        )
        sector_b_vals = sorted(
            ctx.factor_frames[f"T{i:02d}"]["size_sr"].iloc[0]
            for i in range(1, 10, 2)
        )
        assert sector_a_vals == pytest.approx(sector_b_vals), (
            "sector-rank-norm must produce IDENTICAL within-sector "
            "percentile distributions regardless of raw scale — that's "
            "the property that lets rank-pairwise loss work on a "
            "heterogeneous panel"
        )


# ── Defensive paths ───────────────────────────────────────────────────────────

class TestDefensive:
    def test_empty_raw_factor_frames_does_not_raise(self):
        ctx = _make_ctx(enabled=True, include_raw=False)
        # Should log and exit cleanly
        SectorRankNormalizeTask().run(ctx)

    def test_missing_ticker_sectors_does_not_raise(self):
        ctx = _make_ctx(enabled=True, include_ticker_sectors=False)
        # Should log warning and exit cleanly without modifying frames
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorRankNormalizeTask().run(ctx)
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])

    def test_collision_guard_does_not_double_apply(self):
        """If a downstream caller pre-populated _sr columns somehow,
        the Task must not silently overwrite them."""
        ctx = _make_ctx(enabled=True)
        # Pre-populate size_sr with a sentinel value
        for t, df in ctx.factor_frames.items():
            df["size_sr"] = 999.0
        SectorRankNormalizeTask().run(ctx)
        # The pre-populated 999s should remain (collision guard fired)
        for t, df in ctx.factor_frames.items():
            assert (df["size_sr"] == 999.0).all(), (
                f"{t}.size_sr was overwritten — collision guard failed"
            )


# ── Wiring — Task placement in PanelAssemblyJob.tasks ─────────────────────────

class TestWiring:
    def test_task_in_panel_assembly_job(self):
        names = [type(t).__name__ for t in PanelAssemblyJob().tasks]
        assert "SectorRankNormalizeTask" in names

    def test_inference_path_includes_sector_rank_norm(self):
        """prepare_inference_panel_frames must call SectorRankNormalizeTask
        AFTER FactorZScoreTask so the inference-time factor_frames carry
        the same _sr columns the trained model expects.

        Without this, panel scoring at inference time would receive
        all-NaN _sr columns (the model's feature_cols list expects them
        but factor_frames don't have them) → drift guard fires → no
        scores. Source-level check — the function body must reference
        SectorRankNormalizeTask.
        """
        from training_panel import pipeline as tp
        import inspect
        src = inspect.getsource(tp.prepare_inference_panel_frames)
        assert "SectorRankNormalizeTask" in src, (
            "prepare_inference_panel_frames must run SectorRankNormalizeTask "
            "after FactorZScoreTask, otherwise inference factor_frames lack "
            "the _sr columns the trained model expects"
        )
        # Also assert order: SectorRankNorm must come AFTER FactorZScore
        # so factor_frames is populated with _z columns it can use.
        i_fz  = src.index("FactorZScoreTask")
        i_srn = src.index("SectorRankNormalizeTask")
        assert i_fz < i_srn, (
            "Source ordering: FactorZScoreTask must call before "
            "SectorRankNormalizeTask in prepare_inference_panel_frames"
        )

    def test_runs_after_factor_zscore_before_labels(self):
        """Order: NeutralizedFeatureZScore → FactorZScore →
        SectorRankNormalize → Labels → BuildHourly → BuildPanel → Diag.

        Position matters: must run after FactorZScoreTask (so factor_frames
        exists with _z columns) and before LabelsTask + BuildPanelTask
        (so _sr columns make it into the final panel feature list).
        """
        names = [type(t).__name__ for t in PanelAssemblyJob().tasks]
        i_fz   = names.index("FactorZScoreTask")
        i_srn  = names.index("SectorRankNormalizeTask")
        i_lab  = names.index("LabelsTask")
        i_bp   = names.index("BuildPanelTask")
        assert i_fz < i_srn < i_lab < i_bp, (
            f"Task order broken: {names}. Required: FactorZ < SectorRankNorm < Labels < BuildPanel"
        )
