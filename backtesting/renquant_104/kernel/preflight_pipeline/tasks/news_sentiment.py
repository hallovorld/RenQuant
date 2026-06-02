"""P-NEWS-SENTIMENT-FRESHNESS — fail closed on stale scored news.

Issue RenQuant#73: the standalone launchd sentiment cron stopped firing
while reporting no useful failure signal. The daily wrapper now refreshes
sentiment inline, but preflight still needs to surface stale scored-news
parquets before the panel scorer silently emits all-null sentiment features.
"""
from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pandas as pd

from kernel.preflight import PreflightCheck, _soft_for_sell_only  # noqa: PLC0415

from ..base import PreflightTask
from ..ctx import PreflightContext


class NewsSentimentFreshnessTask(PreflightTask):
    """P-NEWS-SENTIMENT-FRESHNESS — max scored sentiment date is recent."""

    check_name = "P-NEWS-SENTIMENT-FRESHNESS"

    def check(self, ctx: PreflightContext) -> PreflightCheck:
        sent_cfg = (
            (ctx.config.get("ranking", {}) or {})
            .get("panel_scoring", {})
            .get("sentiment", {})
            or {}
        )
        if not _sentiment_runtime_enabled(ctx.config, sent_cfg):
            return PreflightCheck(
                self.check_name, "soft", True,
                "sentiment runtime gate disabled/absent; freshness not applicable",
            )
        if not _active_artifact_uses_sentiment(ctx.strategy_dir, ctx.config):
            return PreflightCheck(
                self.check_name, "soft", True,
                "active panel artifact does not declare sentiment features; skip",
            )

        sent_dir = _resolve_sentiment_dir(ctx.strategy_dir, sent_cfg)
        max_days = int(sent_cfg.get("max_stale_trading_days", 5))
        today = _dt.date.today()
        try:
            latest, n_files = _latest_sentiment_date(sent_dir)
        except FileNotFoundError:
            return _soft_for_sell_only(
                self.check_name,
                f"sentiment dir missing: {sent_dir}; run "
                "scripts/daily_news_sentiment_refresh.sh before live buy/full",
                run_mode=ctx.run_mode,
                details={"sentiment_dir": str(sent_dir)},
            )
        except Exception as exc:  # noqa: BLE001
            return _soft_for_sell_only(
                self.check_name,
                f"could not read sentiment freshness from {sent_dir}: {exc}",
                run_mode=ctx.run_mode,
                details={"sentiment_dir": str(sent_dir)},
            )
        if latest is None:
            return _soft_for_sell_only(
                self.check_name,
                f"no readable sentiment parquet dates in {sent_dir}; run "
                "scripts/daily_news_sentiment_refresh.sh before live buy/full",
                run_mode=ctx.run_mode,
                details={"sentiment_dir": str(sent_dir), "n_files": n_files},
            )

        age = _trading_days_since(latest, today)
        details = {
            "sentiment_dir": str(sent_dir),
            "latest_date": latest.isoformat(),
            "age_trading_days": age,
            "max_stale_trading_days": max_days,
            "n_files": n_files,
        }
        if age > max_days:
            return _soft_for_sell_only(
                self.check_name,
                f"news sentiment stale: latest={latest.isoformat()} "
                f"age={age} trading days > max={max_days}; run "
                "scripts/daily_news_sentiment_refresh.sh before live buy/full",
                run_mode=ctx.run_mode,
                details=details,
            )
        return PreflightCheck(
            self.check_name, "hard", True,
            f"news sentiment fresh: latest={latest.isoformat()} "
            f"age={age} trading days <= max={max_days} ({n_files} parquet files)",
            details=details,
        )


def _sentiment_runtime_enabled(config: dict, sent_cfg: dict) -> bool:
    if "enabled" in sent_cfg and bool(sent_cfg.get("enabled")):
        return True
    policy = sent_cfg.get("regime_policy")
    if isinstance(policy, dict) and any(bool(v) for v in policy.values()):
        return True
    regime_params = config.get("regime_params", {}) or {}
    for params in regime_params.values():
        if not isinstance(params, dict):
            continue
        regime_sent = params.get("sentiment")
        if isinstance(regime_sent, dict) and bool(regime_sent.get("enabled")):
            return True
    return False


def _active_artifact_uses_sentiment(strategy_dir: Path, config: dict) -> bool:
    panel_cfg = (
        (config.get("ranking", {}) or {}).get("panel_scoring", {})
        or config.get("panel_ltr", {})
        or {}
    )
    rel = panel_cfg.get("artifact_path")
    if not rel:
        return False
    path = Path(rel)
    if not path.is_absolute():
        path = strategy_dir / path
    if not path.exists() or path.suffix != ".json":
        return False
    try:
        payload = json.loads(path.read_text())
    except Exception:  # noqa: BLE001
        return False
    feature_cols = payload.get("feature_cols") or []
    if not isinstance(feature_cols, list):
        return False
    try:
        from kernel.artifact_contract import SENTIMENT_FEATURE_COLS  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        SENTIMENT_FEATURE_COLS = {
            "mean_sentiment",
            "sentiment_pos_share",
            "sentiment_neg_share",
            "sentiment_dispersion",
            "n_articles_log",
        }
    return any(str(col) in SENTIMENT_FEATURE_COLS for col in feature_cols)


def _resolve_sentiment_dir(strategy_dir: Path, sent_cfg: dict) -> Path:
    raw = (
        sent_cfg.get("data_dir")
        or sent_cfg.get("sentiment_dir")
        or sent_cfg.get("news_sentiment_dir")
    )
    if raw:
        p = Path(raw)
        return p if p.is_absolute() else strategy_dir / p
    if strategy_dir.name == "renquant_104" and strategy_dir.parent.name == "backtesting":
        repo_root = strategy_dir.parents[1]
    else:
        repo_root = strategy_dir
    return repo_root / "data" / "news_sentiment_alpaca"


def _latest_sentiment_date(sent_dir: Path) -> tuple[_dt.date | None, int]:
    if not sent_dir.exists():
        raise FileNotFoundError(sent_dir)
    latest: _dt.date | None = None
    n_files = 0
    for path in sorted(sent_dir.glob("*.parquet")):
        n_files += 1
        d = _max_date_in_parquet(path)
        if d is not None and (latest is None or d > latest):
            latest = d
    return latest, n_files


def _max_date_in_parquet(path: Path) -> _dt.date | None:
    df = pd.read_parquet(path)
    if "date" in df.columns:
        series = pd.to_datetime(df["date"], errors="coerce")
    else:
        series = pd.to_datetime(df.index, errors="coerce")
    series = series.dropna()
    if series.empty:
        return None
    return series.max().date()


def _trading_days_since(start: _dt.date, end: _dt.date) -> int:
    if start >= end:
        return 0
    try:
        from kernel.exits import _is_nyse_trading_day  # noqa: PLC0415
        n = 0
        d = start + _dt.timedelta(days=1)
        while d <= end:
            if _is_nyse_trading_day(d):
                n += 1
            d += _dt.timedelta(days=1)
        return n
    except Exception:  # noqa: BLE001
        return max((end - start).days, 0)
