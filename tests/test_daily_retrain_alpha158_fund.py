"""Unit tests for training_panel.daily_retrain_alpha158_fund.

Covers (per CLAUDE.md §2):
  - Each Task's should_skip cache logic (mtime semantics)
  - Pipeline.run iterates Tasks in order and respects skip
  - Failure inside a Task propagates and is recorded

We test in isolation by stubbing the actual subprocess script calls
(_run_script) and verifying the orchestration logic — same pattern
as the existing kernel/pipeline tests.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))

from training_panel.daily_retrain_alpha158_fund import (  # noqa: E402
    DailyRetrainContext,
    DailyRetrainAlpha158FundPipeline,
    ScanDailyTrainingDataTask,
    BuildAlpha158PanelTask,
    MergeFundFeaturesTask,
    TrainPanelLTRTask,
    RefitCalibratorTask,
    _newest_mtime,
    _resolve_output_override,
    _staging_path,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def ctx(tmp_path):
    """Context whose paths all live under tmp_path."""
    c = DailyRetrainContext(
        repo_dir=tmp_path,
        strategy_dir=tmp_path / "strategy",
        artifacts_dir=tmp_path / "artifacts",
        ohlcv_dir=tmp_path / "ohlcv",
        alpha158_panel=tmp_path / "alpha158.parquet",
        sec_fund_panel=tmp_path / "fund.parquet",
        earnings_surprise_dir=tmp_path / "earnings_surprise",
        news_sentiment_dir=tmp_path / "news_sentiment_alpaca",
        fund_merged_panel=tmp_path / "merged.parquet",
        xgb_artifact_src=tmp_path / "xgb_src.json",
        xgb_artifact_dst=tmp_path / "xgb_dst.json",
        calibrator_artifact=tmp_path / "calib.json",
    )
    c.ohlcv_dir.mkdir(parents=True, exist_ok=True)
    c.data_scan_enabled = False
    return c


def _touch(path: Path, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("stub")
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))


# ── _newest_mtime ────────────────────────────────────────────────────────────

class TestNewestMtime:
    def test_empty(self, tmp_path):
        assert _newest_mtime(tmp_path / "absent.txt") == 0.0

    def test_picks_max_across_files(self, tmp_path):
        a = tmp_path / "a"; b = tmp_path / "b"
        _touch(a, mtime=100); _touch(b, mtime=200)
        assert _newest_mtime(a, b) == 200.0

    def test_recurses_into_directory(self, tmp_path):
        d = tmp_path / "ohlcv"
        d.mkdir()
        _touch(d / "x.parquet", mtime=300)
        _touch(d / "y.parquet", mtime=500)
        assert _newest_mtime(d) == 500.0


def test_staging_path_convention(tmp_path):
    assert _staging_path(tmp_path / "panel-ltr.alpha158_fund.json").name == (
        "panel-ltr.alpha158_fund.staging.json"
    )


def test_shell_wrapper_forwards_pipeline_args():
    """Weekly promote passes unique staging paths through the bash wrapper."""
    wrapper = (REPO / "scripts" / "daily_retrain_alpha158_fund.sh").read_text()
    assert 'training_panel.daily_retrain_alpha158_fund "$@"' in wrapper


def test_output_override_resolves_relative_to_repo(tmp_path):
    out = _resolve_output_override(tmp_path, "backtesting/renquant_104/artifacts/prod/x.json")

    assert out == tmp_path / "backtesting/renquant_104/artifacts/prod/x.json"
    assert out.is_absolute()


# ── BuildAlpha158PanelTask.should_skip ───────────────────────────────────────

class TestBuildAlpha158PanelTaskSkip:
    def test_runs_when_output_missing(self, ctx):
        _touch(ctx.ohlcv_dir / "AAPL" / "1d.parquet", mtime=time.time())
        assert BuildAlpha158PanelTask().should_skip(ctx) is None

    def test_skips_when_output_newer_than_ohlcv(self, ctx):
        _touch(ctx.ohlcv_dir / "AAPL" / "1d.parquet", mtime=100)
        _touch(ctx.alpha158_panel, mtime=200)
        reason = BuildAlpha158PanelTask().should_skip(ctx)
        assert reason is not None
        assert "alpha158 panel newer" in reason

    def test_runs_when_ohlcv_newer_than_output(self, ctx):
        _touch(ctx.alpha158_panel, mtime=100)
        _touch(ctx.ohlcv_dir / "AAPL" / "1d.parquet", mtime=200)
        assert BuildAlpha158PanelTask().should_skip(ctx) is None


# ── MergeFundFeaturesTask.should_skip ────────────────────────────────────────

class TestMergeFundFeaturesTaskSkip:
    def test_runs_when_output_missing(self, ctx):
        _touch(ctx.alpha158_panel, mtime=100)
        _touch(ctx.sec_fund_panel, mtime=100)
        assert MergeFundFeaturesTask().should_skip(ctx) is None

    def test_skips_when_output_newer_than_inputs(self, ctx):
        _touch(ctx.alpha158_panel, mtime=100)
        _touch(ctx.sec_fund_panel, mtime=100)
        _touch(ctx.earnings_surprise_dir / "AAA.parquet", mtime=100)
        _touch(ctx.news_sentiment_dir / "AAA.parquet", mtime=100)
        _touch(ctx.fund_merged_panel, mtime=200)
        assert MergeFundFeaturesTask().should_skip(ctx) is not None

    def test_runs_when_inputs_newer_than_output(self, ctx):
        _touch(ctx.fund_merged_panel, mtime=100)
        _touch(ctx.alpha158_panel, mtime=200)  # alpha158 changed
        _touch(ctx.sec_fund_panel, mtime=100)
        assert MergeFundFeaturesTask().should_skip(ctx) is None

    def test_runs_when_pead_or_sentiment_inputs_newer_than_output(self, ctx):
        """AUDIT REGRESSION GUARD: cached merge must include all sources.

        build_alpha158_fund_panel.py reads data/earnings_surprise for PEAD/SUE
        and data/news_sentiment_alpaca for sentiment. Weekly promote must not
        stamp a fresh model on a stale merged feature panel when those sources
        changed after the prior merge.
        """
        _touch(ctx.alpha158_panel, mtime=100)
        _touch(ctx.sec_fund_panel, mtime=100)
        _touch(ctx.fund_merged_panel, mtime=200)
        _touch(ctx.earnings_surprise_dir / "AAA.parquet", mtime=250)
        assert MergeFundFeaturesTask().should_skip(ctx) is None

        _touch(ctx.fund_merged_panel, mtime=300)
        _touch(ctx.earnings_surprise_dir / "AAA.parquet", mtime=100)
        _touch(ctx.news_sentiment_dir / "AAA.parquet", mtime=350)
        assert MergeFundFeaturesTask().should_skip(ctx) is None

    def test_run_truncates_to_sec_coverage_by_default(self, ctx):
        with patch("training_panel.daily_retrain_alpha158_fund._run_script") as m:
            MergeFundFeaturesTask().run(ctx)
            m.assert_called_once()
        assert "--truncate-to-sec-max" in m.call_args.args[1]


# ── TrainPanelLTRTask output routing ─────────────────────────────────────────

class TestTrainPanelLTRTask:
    def test_skips_when_artifact_newer_than_panel(self, ctx):
        _touch(ctx.fund_merged_panel, mtime=100)
        _touch(ctx.xgb_artifact_dst, mtime=200)
        assert TrainPanelLTRTask().should_skip(ctx) is not None

    def test_runs_when_artifact_older_than_panel(self, ctx):
        _touch(ctx.xgb_artifact_dst, mtime=100)
        _touch(ctx.fund_merged_panel, mtime=200)
        assert TrainPanelLTRTask().should_skip(ctx) is None

    def test_run_invokes_script_with_output_path(self, ctx):
        def fake_run(script, args=None, cwd=None):
            _touch(ctx.xgb_artifact_dst)

        with patch("training_panel.daily_retrain_alpha158_fund._run_script",
                   side_effect=fake_run) as m:
            TrainPanelLTRTask().run(ctx)
            m.assert_called_once()
        called_args = m.call_args.args[1]
        assert "--output-path" in called_args
        assert str(ctx.xgb_artifact_dst) in called_args
        assert ctx.xgb_artifact_dst.exists()

    def test_run_raises_when_script_did_not_produce_artifact(self, ctx):
        # src does NOT exist after _run_script returns
        with patch("training_panel.daily_retrain_alpha158_fund._run_script"):
            with pytest.raises(FileNotFoundError):
                TrainPanelLTRTask().run(ctx)


# ── RefitCalibratorTask ──────────────────────────────────────────────────────

class TestRefitCalibratorTaskSkip:
    def test_skips_when_calibrator_newer_than_xgb(self, ctx):
        _touch(ctx.xgb_artifact_dst, mtime=100)
        _touch(ctx.calibrator_artifact, mtime=200)
        assert RefitCalibratorTask().should_skip(ctx) is not None

    def test_runs_when_xgb_newer_than_calibrator(self, ctx):
        _touch(ctx.calibrator_artifact, mtime=100)
        _touch(ctx.xgb_artifact_dst, mtime=200)
        assert RefitCalibratorTask().should_skip(ctx) is None

    def test_run_pairs_calibrator_with_candidate_scorer(self, ctx):
        with patch("training_panel.daily_retrain_alpha158_fund._run_script") as m:
            RefitCalibratorTask().run(ctx)
            m.assert_called_once()
        called_args = m.call_args.args[1]
        assert "--scorer-artifact" in called_args
        assert str(ctx.xgb_artifact_dst) in called_args
        assert "--out" in called_args
        assert str(ctx.calibrator_artifact) in called_args


# ── Pipeline orchestration ───────────────────────────────────────────────────

class TestPipelineOrchestration:
    def test_runs_all_tasks_in_order(self, ctx):
        order: list[str] = []
        def fake(self_t, ctx_):
            order.append(self_t.name)
        ctx.data_scan_enabled = True
        with patch.object(ScanDailyTrainingDataTask, "run", fake), \
             patch.object(BuildAlpha158PanelTask, "run", fake), \
             patch.object(MergeFundFeaturesTask, "run", fake), \
             patch.object(TrainPanelLTRTask, "run", fake), \
             patch.object(RefitCalibratorTask, "run", fake):
            DailyRetrainAlpha158FundPipeline().run(ctx)
        assert order == [
            "ScanDailyTrainingDataTask",
            "BuildAlpha158PanelTask",
            "MergeFundFeaturesTask",
            "TrainPanelLTRTask",
            "RefitCalibratorTask",
        ]

    def test_skipped_tasks_are_recorded(self, ctx):
        # Stagger mtimes so each output is strictly newer than its input chain.
        _touch(ctx.ohlcv_dir / "AAPL" / "1d.parquet", mtime=100)
        _touch(ctx.alpha158_panel, mtime=200)        # newer than ohlcv
        _touch(ctx.sec_fund_panel, mtime=200)
        _touch(ctx.fund_merged_panel, mtime=300)     # newer than alpha158+fund
        _touch(ctx.xgb_artifact_dst, mtime=400)      # newer than merged
        _touch(ctx.calibrator_artifact, mtime=500)   # newer than xgb
        DailyRetrainAlpha158FundPipeline().run(ctx)
        assert set(ctx.skipped) == {
            "ScanDailyTrainingDataTask",
            "BuildAlpha158PanelTask",
            "MergeFundFeaturesTask",
            "TrainPanelLTRTask",
            "RefitCalibratorTask",
        }

    def test_failure_propagates_and_records(self, ctx):
        with patch.object(BuildAlpha158PanelTask, "should_skip", return_value=None), \
             patch.object(BuildAlpha158PanelTask, "run",
                          side_effect=RuntimeError("kaboom")):
            with pytest.raises(RuntimeError, match="kaboom"):
                DailyRetrainAlpha158FundPipeline().run(ctx)
        assert any("kaboom" in e for e in ctx.errors)


class TestScanDailyTrainingDataTask:
    def test_pipeline_starts_with_data_scan(self):
        assert isinstance(
            DailyRetrainAlpha158FundPipeline().tasks[0],
            ScanDailyTrainingDataTask,
        )

    def test_strict_scan_raises_on_issues_and_writes_report(self, ctx):
        ctx.data_scan_enabled = True
        ctx.watchlist = ["AAA"]
        ctx.data_scan_strict = True

        class Report:
            issues = ["daily OHLCV coverage only 0.0%"]

            def to_dict(self):
                return {"issues": self.issues}

        with patch(
            "training_panel.data_scan.scan_training_inputs",
            return_value=Report(),
        ), patch("training_panel.data_scan.log_scan_summary"), patch(
            "training_panel.data_scan.write_scan_report"
        ) as write_report:
            with pytest.raises(RuntimeError, match="strict=true"):
                ScanDailyTrainingDataTask().run(ctx)

        write_report.assert_called_once()
        assert write_report.call_args.args[1].name == (
            "daily_retrain_training_data_scan.json"
        )
