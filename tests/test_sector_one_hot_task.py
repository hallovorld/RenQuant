"""Tests for SectorOneHotTask — Layer-2 wiring (sector identity features).

Layer 2 stacks ON TOP of Layer 1 (sector rank-norm). When both flags
are on, the model sees three orthogonal sector encodings:
  * `*_z`  — global cross-sectional z-score (every ticker on same scale per date)
  * `*_sr` — per-(date, sector) percentile rank (sector-relative)
  * `sector_<name>` — binary sector identity (one-hot)

The rank-pairwise loss can then split on whichever encoding extracts
signal best per tree node — the canonical sector-conditioning approach
for tree models per Gu-Kelly-Xiu 2020.

Default OFF — wl103 production preserved bit-for-bit when flag false.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from training_panel.pp_panel_training import (   # noqa: E402
    SectorOneHotTask,
    PanelAssemblyJob,
)


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-01-01", periods=n, freq="B")


def _make_ctx(*, enabled: bool, n_tickers: int = 6, n_sectors: int = 3,
               n_dates: int = 5, ticker_sectors=None,
               max_sectors: int | None = None,
               include_factor_frames: bool = True) -> SimpleNamespace:
    idx = _idx(n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    sector_labels = ["tech", "fin", "energy", "consumer",
                     "healthcare", "industrial"][:n_sectors]
    ts = ticker_sectors if ticker_sectors is not None else {
        t: sector_labels[i % n_sectors] for i, t in enumerate(tickers)
    }
    factor_frames = {} if include_factor_frames else {}
    if include_factor_frames:
        for t in tickers:
            factor_frames[t] = pd.DataFrame({
                "size_z": [0.0] * n_dates,
            }, index=idx)
    cfg: dict = {
        "panel_ltr": {"sector_one_hot": {"enabled": enabled}},
    }
    if max_sectors is not None:
        cfg["panel_ltr"]["sector_one_hot"]["max_sectors"] = max_sectors
    return SimpleNamespace(
        config=cfg,
        factor_frames=factor_frames,
        ticker_sectors=ts,
    )


# ── Default OFF ───────────────────────────────────────────────────────────────

class TestDefaultOff:
    def test_disabled_does_not_modify_factor_frames(self):
        ctx = _make_ctx(enabled=False)
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorOneHotTask().run(ctx)
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])


# ── Enabled — adds sector_<name> columns ──────────────────────────────────────

class TestEnabled:
    def test_each_ticker_gets_one_hot_for_its_sector(self):
        ctx = _make_ctx(enabled=True, n_tickers=6, n_sectors=3)
        SectorOneHotTask().run(ctx)
        # T00 → tech (i=0); T01 → fin; T02 → energy; T03 → tech; ...
        assert ctx.factor_frames["T00"]["sector_tech"].iloc[0] == 1.0
        assert ctx.factor_frames["T00"]["sector_fin"].iloc[0]  == 0.0
        assert ctx.factor_frames["T00"]["sector_energy"].iloc[0] == 0.0
        # T01 → fin
        assert ctx.factor_frames["T01"]["sector_fin"].iloc[0]  == 1.0
        assert ctx.factor_frames["T01"]["sector_tech"].iloc[0] == 0.0

    def test_indicator_constant_across_dates_for_each_ticker(self):
        """Sector membership doesn't change date-to-date — each
        sector_* column should be the same value for every row of a
        ticker."""
        ctx = _make_ctx(enabled=True)
        SectorOneHotTask().run(ctx)
        for t, df in ctx.factor_frames.items():
            for col in [c for c in df.columns if c.startswith("sector_")]:
                vals = df[col].unique()
                assert len(vals) == 1, (
                    f"{t}.{col} should be constant across dates, got {vals}"
                )

    def test_all_distinct_sectors_become_columns(self):
        ctx = _make_ctx(enabled=True, n_sectors=3)
        SectorOneHotTask().run(ctx)
        for t, df in ctx.factor_frames.items():
            assert "sector_tech" in df.columns
            assert "sector_fin" in df.columns
            assert "sector_energy" in df.columns

    def test_unmapped_ticker_gets_all_zero_one_hot(self):
        # T00 is in sector_tech, but T_NEW is unmapped
        ctx = _make_ctx(enabled=True, n_tickers=3, n_sectors=2,
                         ticker_sectors={"T00": "tech", "T01": "fin"})
        # Add T02 with no mapping
        ctx.factor_frames["T02"] = pd.DataFrame(
            {"size_z": [0.0] * 5}, index=_idx(5),
        )
        SectorOneHotTask().run(ctx)
        # T02's sector_* columns should all be 0
        assert ctx.factor_frames["T02"]["sector_tech"].iloc[0] == 0.0
        assert ctx.factor_frames["T02"]["sector_fin"].iloc[0]  == 0.0


# ── Defensive paths ───────────────────────────────────────────────────────────

class TestDefensive:
    def test_empty_factor_frames_does_not_raise(self):
        ctx = _make_ctx(enabled=True, include_factor_frames=False)
        SectorOneHotTask().run(ctx)   # no-op, no crash

    def test_missing_ticker_sectors_does_not_raise(self):
        ctx = _make_ctx(enabled=True, ticker_sectors={})
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorOneHotTask().run(ctx)
        # No mapping → no columns added
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])

    def test_max_sectors_safety_guard(self):
        """Universe with too many distinct sectors → log + skip rather
        than blow up feature_cols. Default cap = 30."""
        ctx = _make_ctx(enabled=True, n_tickers=40, n_sectors=6,
                         max_sectors=4)  # too tight to fit 6 sectors
        snapshot = {t: df.copy() for t, df in ctx.factor_frames.items()}
        SectorOneHotTask().run(ctx)
        # All frames unchanged
        for t, before in snapshot.items():
            pd.testing.assert_frame_equal(before, ctx.factor_frames[t])

    def test_collision_guard_does_not_overwrite(self):
        ctx = _make_ctx(enabled=True, n_sectors=3)
        # Pre-populate sector_tech with sentinel
        for t, df in ctx.factor_frames.items():
            df["sector_tech"] = 999.0
        SectorOneHotTask().run(ctx)
        # Sentinel preserved
        for t, df in ctx.factor_frames.items():
            assert (df["sector_tech"] == 999.0).all()


# ── Wiring (PanelAssemblyJob) ─────────────────────────────────────────────────

class TestWiring:
    def test_task_in_panel_assembly_job(self):
        names = [type(t).__name__ for t in PanelAssemblyJob().tasks]
        assert "SectorOneHotTask" in names

    def test_runs_after_sector_rank_norm_before_labels(self):
        """Layer 2 must run AFTER Layer 1 so factor_frames already has
        _sr columns when one-hot is appended (lets BuildPanelTask see
        all three encodings: _z, _sr, sector_*)."""
        names = [type(t).__name__ for t in PanelAssemblyJob().tasks]
        i_srn = names.index("SectorRankNormalizeTask")
        i_oh  = names.index("SectorOneHotTask")
        i_lab = names.index("LabelsTask")
        assert i_srn < i_oh < i_lab, (
            f"Expected SectorRankNorm < OneHot < Labels, got {names}"
        )

    def test_inference_path_includes_sector_one_hot(self):
        """Inference factor_frames must mirror training — Layer 2 too."""
        from training_panel import pipeline as tp
        import inspect
        src = inspect.getsource(tp.prepare_inference_panel_frames)
        assert "SectorOneHotTask" in src, (
            "prepare_inference_panel_frames must run SectorOneHotTask "
            "after SectorRankNormalizeTask, otherwise inference "
            "factor_frames lack the sector_* columns the trained model "
            "expects when Layer 2 is enabled"
        )
