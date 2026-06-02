"""Regression tests for P-NEWS-SENTIMENT-FRESHNESS (RenQuant#73)."""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY) not in sys.path:
    sys.path.insert(0, str(_STRATEGY))

from kernel.preflight_pipeline.ctx import PreflightContext  # noqa: E402
from kernel.preflight_pipeline.tasks.news_sentiment import (  # noqa: E402
    NewsSentimentFreshnessTask,
)


def _write_artifact(strategy_dir: Path, feature_cols: list[str]) -> str:
    rel = "artifacts/prod/panel.json"
    p = strategy_dir / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "kind": "panel_ltr_xgboost",
        "feature_cols": feature_cols,
    }))
    return rel


def _write_sentiment(sent_dir: Path, symbol: str, day: _dt.date) -> None:
    sent_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": [pd.Timestamp(day)],
        "mean_sentiment": [0.25],
    }).to_parquet(sent_dir / f"{symbol}.parquet")


def _ctx(
    tmp_path: Path,
    *,
    feature_cols: list[str] | None = None,
    sent_dir: Path | None = None,
    enabled: bool = True,
    max_days: int = 5,
    run_mode: str = "full",
) -> PreflightContext:
    rel = _write_artifact(tmp_path, feature_cols or ["KMID", "mean_sentiment"])
    sent_dir = sent_dir or (tmp_path / "sentiment")
    cfg = {
        "ranking": {
            "panel_scoring": {
                "artifact_path": rel,
                "kind": "xgb",
                "sentiment": {
                    "enabled": enabled,
                    "data_dir": str(sent_dir),
                    "max_stale_trading_days": max_days,
                },
            },
        },
    }
    return PreflightContext(config=cfg, strategy_dir=tmp_path, run_mode=run_mode)


def test_sentiment_config_absent_soft_skips(tmp_path):
    result = NewsSentimentFreshnessTask().check(
        PreflightContext(config={}, strategy_dir=tmp_path, run_mode="full")
    )
    assert result.ok
    assert result.severity == "soft"
    assert "disabled/absent" in result.message


def test_active_artifact_without_sentiment_features_soft_skips(tmp_path):
    result = NewsSentimentFreshnessTask().check(
        _ctx(tmp_path, feature_cols=["KMID"], sent_dir=tmp_path / "missing")
    )
    assert result.ok
    assert result.severity == "soft"
    assert "does not declare sentiment" in result.message


def test_missing_sentiment_dir_hard_fails_for_full_run(tmp_path):
    result = NewsSentimentFreshnessTask().check(
        _ctx(tmp_path, sent_dir=tmp_path / "missing", run_mode="full")
    )
    assert not result.ok
    assert result.severity == "hard"
    assert "sentiment dir missing" in result.message
    assert "daily_news_sentiment_refresh.sh" in result.message


def test_stale_sentiment_hard_fails_for_full_run(tmp_path):
    sent_dir = tmp_path / "sentiment"
    _write_sentiment(sent_dir, "AAPL", _dt.date.today() - _dt.timedelta(days=30))
    result = NewsSentimentFreshnessTask().check(
        _ctx(tmp_path, sent_dir=sent_dir, max_days=5, run_mode="full")
    )
    assert not result.ok
    assert result.severity == "hard"
    assert "news sentiment stale" in result.message
    assert result.details["age_trading_days"] > 5


def test_stale_sentiment_soft_warns_for_sell_only(tmp_path):
    sent_dir = tmp_path / "sentiment"
    _write_sentiment(sent_dir, "AAPL", _dt.date.today() - _dt.timedelta(days=30))
    result = NewsSentimentFreshnessTask().check(
        _ctx(tmp_path, sent_dir=sent_dir, max_days=5, run_mode="sell-only")
    )
    assert result.ok
    assert result.severity == "soft"
    assert "sell-only risk exits are allowed" in result.message


def test_fresh_sentiment_hard_passes(tmp_path):
    sent_dir = tmp_path / "sentiment"
    _write_sentiment(sent_dir, "AAPL", _dt.date.today())
    result = NewsSentimentFreshnessTask().check(
        _ctx(tmp_path, sent_dir=sent_dir, max_days=5, run_mode="full")
    )
    assert result.ok
    assert result.severity == "hard"
    assert "news sentiment fresh" in result.message
    assert result.details["latest_date"] == _dt.date.today().isoformat()
