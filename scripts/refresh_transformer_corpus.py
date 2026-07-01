#!/usr/bin/env python
"""Refresh the PatchTST SHADOW training corpus before the weekly retrain.

WHY (training-data-freeze investigation, fix #4)
------------------------------------------------
The PatchTST shadow model trains on ``data/transformer_v4_wl200_clean.parquet``
(the ``--dataset`` default of ``scripts/train_walkforward_patchtst.py`` and every
PatchTST/xgb baseline). That corpus was frozen at 2026-02-10 for two reasons:

  1. Its builder (``scripts/transformer_dataset_builder.py`` over the transformer
     universe inventory) is on NO refresh cadence — nothing rebuilds it.
  2. It inherits the SAME full-universe OHLCV coverage gap fixed for the alpha158
     panel in orchestrator PR #217/#210: only the ~142-ticker live watchlist gets
     fresh daily ``data/ohlcv/<ticker>/1d.parquet`` bars (a live-path side effect).
     The ~150 extra research tickers in the transformer universe (tier_A + tier_B
     of ``transformer_universe_inventory.json``) have no refresh cadence, so they
     sit frozen; after the correct fwd_60d label clip that surfaces as a frozen
     corpus for those names.

This module wires a fresh-data rebuild into the PatchTST retrain cadence
(``scripts/weekly_retrain_patchtst.sh``), mirroring the orchestrator alpha158 fix:

  1. RefreshTransformerUniverseOhlcvTask — iterate the FULL transformer universe
     (tier_A + tier_B, sourced exactly where the builder reads it) and call the
     incremental (append-merge, non-destructive, timeout-protected) OHLCV fetch
     for each ticker. Resilient: a single ticker's failure / delisting never
     aborts the refresh. Summarizes n_refreshed / n_stale / n_delisted / n_failed.
  2. TransformerUniverseFreshnessGuardTask — after the refresh, compute each
     transformer ticker's RAW OHLCV bar max date; if more than
     ``freshness_max_stale_fraction`` (default 10%) of the universe lags the
     universe frontier by more than ``freshness_stale_after_days`` (default 10
     trading days), emit a LOUD ntfy alert and (per ``freshness_fail_on_stale``,
     default fail-closed) fail or proceed. Reads RAW bars (frontier ~today), NOT
     the built panel (which legitimately ends ~today-60 after the fwd_60d clip),
     so the expected fwd_60d frontier is distinguished from genuine input staleness.
  3. RebuildTransformerCorpusTask — rebuild the transformer panel to a STAGING
     path, then (only if it advances the corpus date frontier + passes a basic
     row/date-count sanity vs the prior corpus) swap it in NON-DESTRUCTIVELY,
     keeping a ``.bak`` of the prior corpus. A regression (staged corpus older /
     materially smaller) never clobbers the served corpus.

After this runs, the existing ``weekly_retrain_patchtst.sh`` WF build + the shadow
promote (PR #419) train on the fresh corpus.

Non-destructive: uses ONLY the incremental append-merge OHLCV primitive; never
overwrites/deletes ``data/ohlcv/``. The model architecture and the fwd_60d label
clip are UNCHANGED. Every network / builder / disk seam is dependency-injected so
this module is unit-testable with mocks/fixtures and no real fetch / rebuild.

RUNTIME WIRING
--------------
``fetch_ohlcv_incremental`` is a base-data primitive
(``renquant_base_data.loaders.data.fetch_ohlcv_incremental``), import-resolved via
the subrepo PYTHONPATH that ``weekly_retrain_patchtst.sh`` already sets up. It is
dependency-injected via ``CorpusRefreshContext.fetch_fn``; when None it resolves
lazily through ``_default_fetch_fn()`` at call time. The corpus builder is
likewise injected via ``CorpusRefreshContext.builder_fn``; when None it invokes
``scripts/transformer_dataset_builder.py`` to the staging path against the same
inventory the builder reads. Tests inject fakes so nothing touches the network or
a production data file.

Usage::

    python scripts/refresh_transformer_corpus.py --repo-dir /path/to/RenQuant
    python scripts/refresh_transformer_corpus.py --no-freshness-fail-on-stale  # warn + proceed
    python scripts/refresh_transformer_corpus.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("refresh-transformer-corpus")

DEFAULT_INVENTORY_FILENAME = "transformer_universe_inventory.json"
DEFAULT_OHLCV_DIRNAME = "ohlcv"
DEFAULT_CORPUS_RELPATH = "transformer_v4_wl200_clean.parquet"
DEFAULT_OHLCV_TIMEOUT_SEC = 30.0
DEFAULT_FRESHNESS_STALE_AFTER_DAYS = 10
DEFAULT_FRESHNESS_MAX_STALE_FRACTION = 0.10
# Staged corpus must retain at least this fraction of the prior corpus's rows to
# be trusted (guards against a truncated / partial rebuild silently shrinking the
# served corpus). A healthy rebuild is >= prior rows (it appends fresher dates).
DEFAULT_MIN_ROW_RATIO = 0.95
DEFAULT_NTFY_TOPIC = "renquant"


def post_ntfy(title: str, body: str, topic: str = DEFAULT_NTFY_TOPIC) -> None:
    """Best-effort ntfy alert. Honors the suite-wide ``RENQUANT_NO_NOTIFY`` flag
    (pytest sets it) so tests never send a live notification. Injected/patched in
    tests to capture the alert."""
    if os.environ.get("RENQUANT_NO_NOTIFY"):
        log.info("[ntfy suppressed] %s: %s", title, body)
        return
    try:
        import urllib.request  # noqa: PLC0415

        req = urllib.request.Request(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={"Title": title, "Priority": "high"},
        )
        urllib.request.urlopen(req, timeout=10)  # noqa: S310
    except Exception as exc:  # pragma: no cover - network best-effort
        log.warning("ntfy post failed: %s", exc)


@dataclass
class CorpusRefreshContext:
    repo_dir: Path

    # ── universe sourcing (mirrors transformer_dataset_builder.py) ──────────
    # Explicit override; when None the universe is sourced from the transformer
    # inventory (tier_A + tier_B) exactly as the builder reads it.
    transformer_universe: Optional[list] = None
    inventory_path: Optional[Path] = None

    # ── full-universe OHLCV refresh ─────────────────────────────────────────
    refresh_ohlcv: bool = True
    # Dependency-injected incremental fetch. When None resolves to the real
    # renquant_base_data.loaders.data.fetch_ohlcv_incremental at runtime.
    fetch_fn: "Callable[..., object] | None" = None
    ohlcv_timeout_sec: float = DEFAULT_OHLCV_TIMEOUT_SEC

    # ── partial-freeze guard ────────────────────────────────────────────────
    # Injectable per-ticker on-disk max-date reader; None → read the parquet.
    ohlcv_max_date_fn: "Callable[[str], object] | None" = None
    freshness_stale_after_days: int = DEFAULT_FRESHNESS_STALE_AFTER_DAYS
    freshness_max_stale_fraction: float = DEFAULT_FRESHNESS_MAX_STALE_FRACTION
    # Fail-closed by default (a partially frozen training universe is a real
    # training-input integrity failure). False → only warn (ntfy) + proceed.
    freshness_fail_on_stale: bool = True

    # ── corpus rebuild + non-destructive, sanity-gated swap ─────────────────
    corpus_path: Optional[Path] = None
    staging_path: Optional[Path] = None
    rebuild_corpus: bool = True
    # Injected panel builder (staging_path, universe) -> None. None → invoke
    # scripts/transformer_dataset_builder.py to the staging path.
    builder_fn: "Callable[[Path, list], None] | None" = None
    # Injected (path) -> (n_rows, max_date|None) reader; None → read the parquet.
    corpus_stats_fn: "Callable[[Path], tuple] | None" = None
    # Staged corpus must advance the date frontier (>= prior max date) ...
    require_date_advance: bool = True
    # ... and keep at least this fraction of the prior corpus's rows.
    min_row_ratio: float = DEFAULT_MIN_ROW_RATIO
    # Fail the retrain when the rebuilt corpus regresses (default, fail-closed);
    # False → warn (ntfy) + keep the prior corpus + proceed.
    swap_fail_on_regression: bool = True

    ntfy_topic: str = DEFAULT_NTFY_TOPIC
    dry_run: bool = False
    quiet: bool = False

    # ── populated at runtime (audit surface) ────────────────────────────────
    ohlcv_max_dates: dict = field(default_factory=dict)
    ohlcv_refresh_summary: dict = field(default_factory=dict)
    freshness_report: dict = field(default_factory=dict)
    swap_report: dict = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.repo_dir / "data"

    @property
    def ohlcv_dir(self) -> Path:
        return self.data_dir / DEFAULT_OHLCV_DIRNAME

    @property
    def resolved_inventory_path(self) -> Path:
        if self.inventory_path is not None:
            return self.inventory_path
        return self.data_dir / DEFAULT_INVENTORY_FILENAME

    @property
    def resolved_corpus_path(self) -> Path:
        if self.corpus_path is not None:
            return self.corpus_path
        return self.data_dir / DEFAULT_CORPUS_RELPATH

    @property
    def resolved_staging_path(self) -> Path:
        if self.staging_path is not None:
            return self.staging_path
        c = self.resolved_corpus_path
        return c.with_name(c.name + ".staging")


# ─────────────────────────── universe sourcing ─────────────────────────────


def _resolve_transformer_universe(ctx: CorpusRefreshContext) -> list:
    """Source the FULL transformer training universe (tier_A + tier_B), NOT just
    the ~142-ticker live watchlist.

    Mirrors ``scripts/transformer_dataset_builder.py``, which reads
    ``tier_A_tickers`` + ``tier_B_tickers`` from ``transformer_universe_inventory
    .json``. An explicit ``ctx.transformer_universe`` wins. Returns a sorted,
    de-duplicated list; an unreadable / missing inventory yields an empty list
    (logged) so the refresh + guard degrade to safe no-ops rather than aborting.
    """
    if ctx.transformer_universe:
        return sorted(dict.fromkeys(str(t) for t in ctx.transformer_universe))
    inv_path = ctx.resolved_inventory_path
    if not inv_path.exists():
        log.warning("transformer universe inventory not found: %s — universe empty", inv_path)
        return []
    try:
        inv = json.loads(inv_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("failed to read transformer universe inventory %s: %s", inv_path, exc)
        return []
    universe = set(inv.get("tier_A_tickers", [])) | set(inv.get("tier_B_tickers", []))
    return sorted(str(t) for t in universe)


# ─────────────────────────── fetch / date helpers ──────────────────────────


def _default_fetch_fn() -> "Callable[..., object]":
    """Resolve the real base-data incremental OHLCV primitive at runtime.

    ``fetch_ohlcv_incremental`` lives in ``renquant-base-data`` and is
    import-resolved via the subrepo PYTHONPATH the retrain sets up. Injected via
    ``CorpusRefreshContext.fetch_fn`` so this module is unit-testable without a
    network fetch. Non-destructive: cache-first, incremental delta only,
    append-merge, timeout-protected.
    """
    from renquant_base_data.loaders.data import fetch_ohlcv_incremental  # noqa: PLC0415

    return fetch_ohlcv_incremental


def _df_max_date(df: "object | None") -> "dt.date | None":
    """Latest bar date of an OHLCV frame (DatetimeIndex or a date column)."""
    if df is None:
        return None
    try:
        import pandas as pd  # noqa: PLC0415

        if getattr(df, "empty", True):
            return None
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            return idx.max().date()
        for col in ("date", "Date", "datetime"):
            if col in getattr(df, "columns", []):
                return pd.to_datetime(df[col]).max().date()
        return pd.to_datetime(idx).max().date()
    except Exception:  # pragma: no cover - defensive; malformed frame
        return None


def _default_ohlcv_max_date(ohlcv_dir: Path, ticker: str) -> "dt.date | None":
    path = ohlcv_dir / ticker / "1d.parquet"
    if not path.exists():
        return None
    try:
        import pandas as pd  # noqa: PLC0415

        return _df_max_date(pd.read_parquet(path))
    except Exception:  # pragma: no cover - defensive; unreadable parquet
        return None


def _resolve_ohlcv_max_date(ctx: CorpusRefreshContext, ticker: str) -> "dt.date | None":
    # Prefer the refresh-captured map (avoids re-reading parquet); otherwise use
    # an injectable reader, defaulting to the on-disk raw OHLCV bars.
    if ticker in ctx.ohlcv_max_dates:
        return ctx.ohlcv_max_dates[ticker]
    if ctx.ohlcv_max_date_fn is not None:
        return ctx.ohlcv_max_date_fn(ticker)
    return _default_ohlcv_max_date(ctx.ohlcv_dir, ticker)


def _frontier(dates) -> "dt.date | None":
    known = [d for d in dates if d is not None]
    return max(known) if known else None


def _trading_days_between(start: "dt.date", end: "dt.date") -> int:
    """Business-day gap (Mon-Fri) between two dates — a holiday-agnostic proxy
    for trading days. Non-negative; 0 when ``start >= end``."""
    if start >= end:
        return 0
    import numpy as np  # noqa: PLC0415

    return int(np.busday_count(start, end))


# ─────────────────────────── corpus stats / build ──────────────────────────


def _default_corpus_stats(path: Path) -> tuple:
    """(n_rows, max_date|None) for a transformer corpus parquet. (0, None) when
    the file is missing / unreadable."""
    if not path.exists():
        return (0, None)
    try:
        import pandas as pd  # noqa: PLC0415

        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return (0, None)
        return (int(len(df)), pd.to_datetime(df["date"]).max().date())
    except Exception:  # pragma: no cover - defensive; unreadable parquet
        try:
            import pandas as pd  # noqa: PLC0415

            df = pd.read_parquet(path)
            return (int(len(df)), _df_max_date(df))
        except Exception:
            return (0, None)


def _resolve_corpus_stats(ctx: CorpusRefreshContext, path: Path) -> tuple:
    if ctx.corpus_stats_fn is not None:
        return ctx.corpus_stats_fn(path)
    return _default_corpus_stats(path)


def _default_build_corpus(ctx: CorpusRefreshContext, staging_path: Path, universe: list) -> None:
    """Rebuild the transformer panel to ``staging_path`` by invoking the existing
    builder against the SAME inventory the corpus is built from.

    RUNTIME WIRING NOTE: the served corpus ``transformer_v4_wl200_clean.parquet``
    is the wl200-clean transformer_v4 corpus. This default wires the canonical
    ``scripts/transformer_dataset_builder.py`` (inventory tier_A+tier_B over
    ``data/ohlcv``) to the staging path; point ``builder_fn`` at the operator's
    exact wl200-clean recipe if it diverges — the injection seam makes that a
    one-line change with no code churn here.
    """
    builder = ctx.repo_dir / "scripts" / "transformer_dataset_builder.py"
    cmd = [
        sys.executable,
        str(builder),
        "--inventory",
        str(ctx.resolved_inventory_path),
        "--ohlcv-dir",
        str(ctx.ohlcv_dir),
        "--output",
        str(staging_path),
    ]
    log.info("rebuilding transformer corpus to staging: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


# ─────────────────────────── tasks ─────────────────────────────────────────


class RefreshTransformerUniverseOhlcvTask:
    """Refresh daily OHLCV bars for the FULL transformer training universe.

    ROOT CAUSE (frozen shadow corpus): only the ~142-ticker live watchlist gets
    fresh bars daily (a live-path side effect). The ~150 extra research tickers
    in the transformer universe (tier_A + tier_B) have no refresh cadence, so the
    corpus froze at 2026-02-10. This iterates the WHOLE transformer universe and
    calls the incremental (append-merge, non-destructive, timeout-protected)
    fetch for each ticker BEFORE the rebuild. Resilient: a single ticker's
    failure / delisting NEVER aborts the refresh — delisted names return their
    stale cache and are counted, not fatal. Records n_refreshed / n_stale /
    n_delisted / n_failed.
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        universe = _resolve_transformer_universe(ctx)
        summary = {
            "n_universe": len(universe),
            "n_refreshed": 0,
            "n_stale": 0,
            "n_delisted": 0,
            "n_failed": 0,
        }
        ctx.ohlcv_refresh_summary = summary
        if not ctx.refresh_ohlcv:
            log.info("OHLCV refresh disabled (refresh_ohlcv=False); skipping")
            return True
        if not universe:
            log.warning("transformer universe empty; nothing to refresh")
            return True
        if ctx.dry_run:
            log.info("[dry-run] would refresh OHLCV for %d transformer tickers", len(universe))
            return True

        fetch_fn = ctx.fetch_fn or _default_fetch_fn()
        max_dates = {}
        failed = set()
        for ticker in universe:
            try:
                df = fetch_fn(ticker, timeout_sec=ctx.ohlcv_timeout_sec)
            except Exception as exc:  # one ticker must never abort the refresh
                failed.add(ticker)
                max_dates[ticker] = None
                log.warning("OHLCV refresh failed for %s: %s", ticker, exc)
                continue
            max_dates[ticker] = _df_max_date(df)
        ctx.ohlcv_max_dates = max_dates

        # Classify against the batch frontier (freshest bar any ticker returned)
        # into disjoint buckets so the counts sum to the universe size.
        frontier = _frontier(max_dates.values())
        for ticker, md in max_dates.items():
            if ticker in failed:
                summary["n_failed"] += 1
            elif md is None:
                summary["n_delisted"] += 1
            elif (
                frontier is not None
                and _trading_days_between(md, frontier) > ctx.freshness_stale_after_days
            ):
                summary["n_stale"] += 1
            else:
                summary["n_refreshed"] += 1
        log.info(
            "OHLCV refresh: universe=%d refreshed=%d stale=%d delisted=%d failed=%d",
            summary["n_universe"],
            summary["n_refreshed"],
            summary["n_stale"],
            summary["n_delisted"],
            summary["n_failed"],
        )
        return True


class TransformerUniverseFreshnessGuardTask:
    """Guard against a *partial* transformer-universe freeze — the silent failure
    mode that let the research tail sit at 2026-02-10 while the ~142-ticker
    watchlist stayed fresh and the watchlist-only scan passed.

    Reads each transformer ticker's RAW OHLCV bar max date — NOT the built
    corpus, which legitimately ends ~today-60 trading days after the (correct)
    fwd_60d label clip. Reading raw bars means an on-frontier universe never trips
    this guard: genuine input staleness (the bars themselves old) is distinguished
    from the expected fwd_60d frontier. A ticker is 'stale' when its newest bar
    lags the universe frontier (the freshest bar any ticker has) by more than
    ``freshness_stale_after_days`` trading days. If more than
    ``freshness_max_stale_fraction`` of the universe is stale, emit a LOUD ntfy
    alert and — per ``freshness_fail_on_stale`` — either fail the retrain
    (default, fail-closed) or proceed with the warning.
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        universe = _resolve_transformer_universe(ctx)
        if not universe:
            log.warning("freshness guard: transformer universe empty; cannot assess — skipping")
            return True
        dates = {t: _resolve_ohlcv_max_date(ctx, t) for t in universe}
        known = {t: d for t, d in dates.items() if d is not None}
        if not known:
            log.warning("freshness guard: no OHLCV max dates resolvable — skipping")
            return True

        frontier = max(known.values())
        stale = {
            t: d
            for t, d in known.items()
            if _trading_days_between(d, frontier) > ctx.freshness_stale_after_days
        }
        missing = {t for t, d in dates.items() if d is None}
        n_stale = len(stale) + len(missing)
        fraction = n_stale / len(universe)
        worst = sorted(
            ((_trading_days_between(d, frontier), t) for t, d in stale.items()),
            reverse=True,
        )[:10]
        report = {
            "as_of_frontier": frontier.isoformat(),
            "n_universe": len(universe),
            "n_stale": n_stale,
            "n_missing": len(missing),
            "stale_fraction": round(fraction, 4),
            "stale_after_days": ctx.freshness_stale_after_days,
            "max_stale_fraction": ctx.freshness_max_stale_fraction,
            "worst_examples": [[lag, t] for lag, t in worst],
        }
        ctx.freshness_report = report

        if fraction <= ctx.freshness_max_stale_fraction:
            log.info(
                "freshness guard OK: %d/%d stale (%.1f%% <= %.1f%%), frontier=%s",
                n_stale,
                len(universe),
                fraction * 100,
                ctx.freshness_max_stale_fraction * 100,
                frontier.isoformat(),
            )
            return True

        worst_str = ", ".join(f"{t}(-{lag}d)" for lag, t in worst[:8])
        title = "RenQuant PatchTST CORPUS-FREEZE"
        body = (
            f"{n_stale}/{len(universe)} transformer tickers stale "
            f"({fraction:.1%} > {ctx.freshness_max_stale_fraction:.0%}); "
            f"bars lag frontier {frontier.isoformat()} by "
            f">{ctx.freshness_stale_after_days} trading days. "
            f"Worst: {worst_str}. "
            f"{'FAILING retrain' if ctx.freshness_fail_on_stale else 'proceeding with warning'}."
        )
        if not ctx.quiet:
            post_ntfy(title, body, ctx.ntfy_topic)
        log.error("freshness guard TRIPPED: %s", body)
        if ctx.freshness_fail_on_stale:
            raise RuntimeError(body)
        return True


class RebuildTransformerCorpusTask:
    """Rebuild the transformer corpus to a STAGING path, then swap it in only if
    it advances the corpus + passes a basic row/date sanity vs the prior corpus.

    Non-destructive: the prior corpus is moved to ``<corpus>.bak`` before the
    staged corpus takes its place, so a bad rebuild is always recoverable. A
    regression (staged corpus older, or materially fewer rows than the prior)
    NEVER clobbers the served corpus — per ``swap_fail_on_regression`` it either
    fails the retrain (default, fail-closed) or warns + keeps the prior corpus.
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        corpus = ctx.resolved_corpus_path
        staging = ctx.resolved_staging_path
        report = {
            "corpus_path": str(corpus),
            "staging_path": str(staging),
            "swapped": False,
        }
        ctx.swap_report = report
        if not ctx.rebuild_corpus:
            log.info("corpus rebuild disabled (rebuild_corpus=False); skipping")
            return True
        if ctx.dry_run:
            log.info("[dry-run] would rebuild transformer corpus to %s and sanity-gate swap", staging)
            return True

        universe = _resolve_transformer_universe(ctx)
        prior_rows, prior_date = _resolve_corpus_stats(ctx, corpus)
        report["prior_rows"] = prior_rows
        report["prior_max_date"] = prior_date.isoformat() if prior_date else None

        # Build to staging (never touches the served corpus).
        staging.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            staging.unlink()
        builder = ctx.builder_fn or (lambda sp, u: _default_build_corpus(ctx, sp, u))
        builder(staging, universe)

        staged_rows, staged_date = _resolve_corpus_stats(ctx, staging)
        report["staged_rows"] = staged_rows
        report["staged_max_date"] = staged_date.isoformat() if staged_date else None

        reasons = self._sanity_reasons(ctx, prior_rows, prior_date, staged_rows, staged_date)
        report["sanity_reasons"] = reasons
        if reasons:
            self._reject(ctx, staging, report, reasons)
            return True

        # Non-destructive swap: prior -> .bak, then staging -> corpus.
        if corpus.exists():
            bak = corpus.with_name(corpus.name + ".bak")
            if bak.exists():
                bak.unlink()
            shutil.move(str(corpus), str(bak))
            report["backup_path"] = str(bak)
        shutil.move(str(staging), str(corpus))
        report["swapped"] = True
        log.info(
            "transformer corpus swapped: %s rows=%d max_date=%s (prior rows=%d max_date=%s, backup=%s)",
            corpus,
            staged_rows,
            report["staged_max_date"],
            prior_rows,
            report["prior_max_date"],
            report.get("backup_path"),
        )
        return True

    @staticmethod
    def _sanity_reasons(ctx, prior_rows, prior_date, staged_rows, staged_date) -> list:
        reasons = []
        if staged_rows <= 0 or staged_date is None:
            reasons.append("staged corpus empty or unreadable")
            return reasons
        if prior_rows > 0 and prior_date is not None:
            if ctx.require_date_advance and staged_date < prior_date:
                reasons.append(
                    f"staged max date {staged_date.isoformat()} < prior {prior_date.isoformat()}"
                )
            if staged_rows < prior_rows * ctx.min_row_ratio:
                reasons.append(
                    f"staged rows {staged_rows} < {ctx.min_row_ratio:.0%} of prior {prior_rows}"
                )
        return reasons

    def _reject(self, ctx, staging, report, reasons) -> None:
        # Leave the served corpus untouched; drop the staged build.
        if staging.exists():
            staging.unlink()
        title = "RenQuant PatchTST CORPUS-REBUILD REJECTED"
        body = (
            "rebuilt transformer corpus rejected (kept prior corpus): "
            + "; ".join(reasons)
            + f". {'FAILING retrain' if ctx.swap_fail_on_regression else 'proceeding with prior corpus'}."
        )
        if not ctx.quiet:
            post_ntfy(title, body, ctx.ntfy_topic)
        log.error("corpus rebuild REJECTED: %s", body)
        if ctx.swap_fail_on_regression:
            raise RuntimeError(body)


# ─────────────────────────── pipeline ──────────────────────────────────────


def build_pipeline() -> list:
    """Ordered tasks: refresh full-universe OHLCV, guard against a partial
    freeze, then rebuild + sanity-gated non-destructive swap."""
    return [
        RefreshTransformerUniverseOhlcvTask(),
        TransformerUniverseFreshnessGuardTask(),
        RebuildTransformerCorpusTask(),
    ]


def run_pipeline(ctx: CorpusRefreshContext) -> None:
    for task in build_pipeline():
        task.run(ctx)


# ─────────────────────────── CLI ───────────────────────────────────────────


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--repo-dir", type=Path, default=REPO_ROOT)
    p.add_argument("--inventory-path", type=Path, default=None)
    p.add_argument("--corpus-path", type=Path, default=None)
    p.add_argument("--staging-path", type=Path, default=None)
    p.add_argument(
        "--transformer-universe-file",
        type=Path,
        default=None,
        help=(
            "Optional JSON: a plain list of tickers OR an inventory object with "
            "tier_A_tickers/tier_B_tickers. Default: "
            "<repo>/data/transformer_universe_inventory.json (what the builder reads)."
        ),
    )
    p.add_argument("--refresh-ohlcv", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--ohlcv-timeout-sec", type=float, default=DEFAULT_OHLCV_TIMEOUT_SEC)
    p.add_argument("--rebuild-corpus", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument(
        "--freshness-stale-after-days",
        type=int,
        default=DEFAULT_FRESHNESS_STALE_AFTER_DAYS,
    )
    p.add_argument(
        "--freshness-max-stale-fraction",
        type=float,
        default=DEFAULT_FRESHNESS_MAX_STALE_FRACTION,
    )
    p.add_argument(
        "--freshness-fail-on-stale",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fail the retrain when the guard trips (default, fail-closed). --no-... only warns.",
    )
    p.add_argument("--require-date-advance", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--min-row-ratio", type=float, default=DEFAULT_MIN_ROW_RATIO)
    p.add_argument(
        "--swap-fail-on-regression",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fail the retrain when the rebuilt corpus regresses (default). --no-... keeps prior + proceeds.",
    )
    p.add_argument("--ntfy-topic", default=DEFAULT_NTFY_TOPIC)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _load_universe_file(path: Path):
    puf = path.expanduser().resolve()
    try:
        payload = json.loads(puf.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"--transformer-universe-file unreadable: {puf}: {exc}")
    if isinstance(payload, list):
        return [str(t) for t in payload], None
    if isinstance(payload, dict):
        return None, puf
    raise SystemExit(f"--transformer-universe-file must be a JSON list or object: {puf}")


def main(argv: "list | None" = None) -> int:
    args = parse_args(argv)
    transformer_universe = None
    inventory_path = args.inventory_path
    if args.transformer_universe_file:
        transformer_universe, inv = _load_universe_file(args.transformer_universe_file)
        if inv is not None:
            inventory_path = inv
    ctx = CorpusRefreshContext(
        repo_dir=args.repo_dir.expanduser().resolve(),
        transformer_universe=transformer_universe,
        inventory_path=inventory_path,
        corpus_path=args.corpus_path,
        staging_path=args.staging_path,
        refresh_ohlcv=args.refresh_ohlcv,
        ohlcv_timeout_sec=args.ohlcv_timeout_sec,
        rebuild_corpus=args.rebuild_corpus,
        freshness_stale_after_days=args.freshness_stale_after_days,
        freshness_max_stale_fraction=args.freshness_max_stale_fraction,
        freshness_fail_on_stale=args.freshness_fail_on_stale,
        require_date_advance=args.require_date_advance,
        min_row_ratio=args.min_row_ratio,
        swap_fail_on_regression=args.swap_fail_on_regression,
        ntfy_topic=args.ntfy_topic,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
    run_pipeline(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
