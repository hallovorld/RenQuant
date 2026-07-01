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
(``scripts/weekly_retrain_patchtst.sh``), mirroring the orchestrator alpha158 fix.

FAIL-CLOSED CONTRACT (Codex review, PR #424)
--------------------------------------------
Every load-bearing input is validated and fails CLOSED on an unassessable state —
a silent skip is never allowed to hand the builder / retrain a degraded corpus:

  * Universe PROVENANCE — a missing / corrupt / empty inventory (or an inventory
    whose digest does not match a bound expected digest) raises, so the refresh /
    rebuild never runs against an empty universe.
  * FRESHNESS — the raw-bar frontier is compared to an INDEPENDENT expected
    completed market session (an exchange-calendar as-of), not only to the
    universe's own max(known dates). A GLOBALLY frozen universe (every ticker
    uniformly stale) trips the guard even though it has zero *relative* staleness;
    unassessable inputs (no resolvable dates) fail closed rather than skip.
  * REBUILD/SWAP — the staged corpus must strictly ADVANCE the frontier (an equal,
    non-advanced corpus is rejected), keep >= ``min_row_ratio`` of the prior rows,
    keep the prior SCHEMA / features / label horizons, and keep >=
    ``min_ticker_coverage_ratio`` of the prior ticker coverage. A wrong builder
    recipe that changes features / universe / schema while still producing a
    plausible row count therefore fails closed instead of silently serving a
    divergent corpus.
  * ATOMIC swap — the prior corpus is backed up by COPY and replaced with a single
    ``os.replace`` (atomic rename on one filesystem) + fsync. The served corpus is
    never moved out of the way, so an interrupted swap can never leave it missing.

The three tasks (refresh → freshness guard → rebuild + atomic swap) are ordered so
the guard runs on freshly-fetched bars and the rebuild only runs behind a green
guard. After this runs, the existing ``weekly_retrain_patchtst.sh`` WF build + the
shadow promote (PR #419) train on the fresh corpus.

Non-destructive: uses ONLY the incremental append-merge OHLCV primitive; never
overwrites/deletes ``data/ohlcv/``. The model architecture and the fwd_60d label
clip are UNCHANGED. Every network / builder / disk seam is dependency-injected so
this module is unit-testable with mocks/fixtures and no real fetch / rebuild.

REFRESH/REBUILD IS PLUMBING, NOT A PROMOTION. This module only assembles a fresh,
integrity-checked corpus. Whether the freshly-trained shadow model is PROMOTED is
still decided downstream by the existing WF gate + shadow replay (PR #419); a
frozen shadow replay (identical model code/seeds, PIT checks, regime IC,
turnover/cost, coverage diagnostics on old vs new corpus) gates any promotion.

RUNTIME WIRING
--------------
``fetch_ohlcv_incremental`` is a base-data primitive
(``renquant_base_data.loaders.data.fetch_ohlcv_incremental``), import-resolved via
the subrepo PYTHONPATH that ``weekly_retrain_patchtst.sh`` already sets up. It is
dependency-injected via ``CorpusRefreshContext.fetch_fn``; when None it resolves
lazily through ``_default_fetch_fn()`` at call time. The corpus builder is
likewise injected via ``CorpusRefreshContext.builder_fn``; when None it invokes
``scripts/transformer_dataset_builder.py`` to the staging path against the same
inventory / labels / integrity-report the builder reads. The staged corpus is
then validated against the served corpus's schema / label / coverage CONTRACT, so
a divergent recipe fails closed rather than silently swapping in. Tests inject
fakes so nothing touches the network or a production data file.

Usage::

    python scripts/refresh_transformer_corpus.py --repo-dir /path/to/RenQuant
    python scripts/refresh_transformer_corpus.py --no-freshness-fail-on-stale  # warn + proceed
    python scripts/refresh_transformer_corpus.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
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
DEFAULT_INTEGRITY_FILENAME = "transformer_data_integrity_report.json"
DEFAULT_LABELS_FILENAME = "transformer_panel_labels.parquet"
DEFAULT_OHLCV_DIRNAME = "ohlcv"
DEFAULT_CORPUS_RELPATH = "transformer_v4_wl200_clean.parquet"
DEFAULT_OHLCV_TIMEOUT_SEC = 30.0
DEFAULT_FRESHNESS_STALE_AFTER_DAYS = 10
DEFAULT_FRESHNESS_MAX_STALE_FRACTION = 0.10
# Staged corpus must retain at least this fraction of the prior corpus's rows to
# be trusted (guards against a truncated / partial rebuild silently shrinking the
# served corpus). A healthy rebuild is >= prior rows (it appends fresher dates).
DEFAULT_MIN_ROW_RATIO = 0.95
# Staged corpus must retain at least this fraction of the prior corpus's distinct
# tickers (a wrong recipe / universe drift silently dropping names fails closed).
DEFAULT_MIN_TICKER_COVERAGE_RATIO = 0.90
# Recorded for audit; the OUTPUT-CONTRACT gate (schema/label/coverage) is what
# actually binds the recipe — a divergent builder output fails the swap closed.
DEFAULT_BUILDER_RECIPE = "transformer_dataset_builder:tier_A+tier_B/raw-OHLCV/fwd_{5,20,60}d/wl200_clean"
DEFAULT_NTFY_TOPIC = "renquant"

# Label columns look like ``fwd_60d_excess`` / ``label_20d`` / ``y_5d``.
_LABEL_HORIZON_RE = re.compile(r"(\d+)")


class CorpusRefreshError(RuntimeError):
    """Fail-closed integrity failure (bad provenance / unassessable freshness /
    interrupted swap). Subclasses RuntimeError so callers can catch either."""


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
    # Fail closed (default) when the required training universe is unresolvable /
    # empty. False → warn + degrade to a safe no-op (explicit ops escape hatch).
    require_universe: bool = True
    # When set, the sourced inventory's sha256 must equal this or the run fails
    # closed (binds the exact universe file that produced the served corpus).
    expected_inventory_digest: Optional[str] = None

    # ── full-universe OHLCV refresh ─────────────────────────────────────────
    refresh_ohlcv: bool = True
    # Dependency-injected incremental fetch. When None resolves to the real
    # renquant_base_data.loaders.data.fetch_ohlcv_incremental at runtime.
    fetch_fn: "Callable[..., object] | None" = None
    ohlcv_timeout_sec: float = DEFAULT_OHLCV_TIMEOUT_SEC

    # ── partial / global freeze guard ───────────────────────────────────────
    # Injectable per-ticker on-disk max-date reader; None → read the parquet.
    ohlcv_max_date_fn: "Callable[[str], object] | None" = None
    freshness_stale_after_days: int = DEFAULT_FRESHNESS_STALE_AFTER_DAYS
    freshness_max_stale_fraction: float = DEFAULT_FRESHNESS_MAX_STALE_FRACTION
    # Independent "expected completed market session" the raw-bar frontier is
    # measured against (catches a GLOBAL freeze that has zero relative staleness).
    # Explicit date wins; else expected_as_of_fn(); else the last business day.
    freshness_as_of: "dt.date | None" = None
    expected_as_of_fn: "Callable[[], dt.date] | None" = None
    # Fail-closed by default (a partially frozen training universe is a real
    # training-input integrity failure). False → only warn (ntfy) + proceed.
    freshness_fail_on_stale: bool = True

    # ── corpus rebuild + non-destructive, sanity-gated swap ─────────────────
    corpus_path: Optional[Path] = None
    staging_path: Optional[Path] = None
    labels_path: Optional[Path] = None
    integrity_report_path: Optional[Path] = None
    rebuild_corpus: bool = True
    builder_recipe: str = DEFAULT_BUILDER_RECIPE
    # Injected panel builder (staging_path, universe) -> None. None → invoke
    # scripts/transformer_dataset_builder.py to the staging path.
    builder_fn: "Callable[[Path, list], None] | None" = None
    # Injected (path) -> (n_rows, max_date|None) reader; None → read the parquet.
    corpus_stats_fn: "Callable[[Path], tuple] | None" = None
    # Injected (path) -> schema dict reader; None → read the parquet schema.
    corpus_schema_fn: "Callable[[Path], dict] | None" = None
    # Staged corpus must strictly ADVANCE the date frontier (equal is rejected) ...
    require_date_advance: bool = True
    # ... keep at least this fraction of the prior corpus's rows ...
    min_row_ratio: float = DEFAULT_MIN_ROW_RATIO
    # ... keep at least this fraction of the prior corpus's distinct tickers ...
    min_ticker_coverage_ratio: float = DEFAULT_MIN_TICKER_COVERAGE_RATIO
    # ... and keep the prior schema / features / label horizons (recipe parity).
    validate_schema: bool = True
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
    universe_provenance: dict = field(default_factory=dict)
    _universe_cache: Optional[list] = field(default=None, repr=False)

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
    def resolved_labels_path(self) -> Path:
        if self.labels_path is not None:
            return self.labels_path
        return self.data_dir / DEFAULT_LABELS_FILENAME

    @property
    def resolved_integrity_report_path(self) -> Path:
        if self.integrity_report_path is not None:
            return self.integrity_report_path
        return self.data_dir / DEFAULT_INTEGRITY_FILENAME

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


# ─────────────────────────── fail-closed helper ────────────────────────────


def _fail_closed(ctx: CorpusRefreshContext, title: str, body: str) -> "None":
    """Emit a LOUD ntfy alert (unless quiet), log, and raise — the single place a
    load-bearing integrity failure turns into a fail-closed abort."""
    if not ctx.quiet:
        post_ntfy(title, body, ctx.ntfy_topic)
    log.error("%s: %s", title, body)
    raise CorpusRefreshError(body)


# ─────────────────────────── universe sourcing ─────────────────────────────


def _digest_universe(universe: list) -> str:
    return hashlib.sha256("\n".join(universe).encode("utf-8")).hexdigest()


def _source_transformer_universe(ctx: CorpusRefreshContext) -> "tuple[list, dict]":
    """Source the FULL transformer training universe (tier_A + tier_B) and record
    provenance. PURE (never raises): the caller decides fail-closed behaviour.

    Mirrors ``scripts/transformer_dataset_builder.py``, which unions
    ``tier_A_tickers`` + ``tier_B_tickers`` from the inventory. An explicit
    ``ctx.transformer_universe`` wins. A missing / corrupt / empty-tiers inventory
    yields an empty list plus a provenance dict flagging why.
    """
    if ctx.transformer_universe:
        uni = sorted(dict.fromkeys(str(t) for t in ctx.transformer_universe))
        return uni, {
            "source": "explicit",
            "inventory_path": None,
            "inventory_digest": _digest_universe(uni),
            "n_universe": len(uni),
            "reason": None,
        }
    inv_path = ctx.resolved_inventory_path
    prov: dict = {
        "source": "inventory",
        "inventory_path": str(inv_path),
        "inventory_digest": None,
        "n_universe": 0,
        "reason": None,
    }
    if not inv_path.exists():
        prov["reason"] = "inventory not found"
        return [], prov
    try:
        raw = inv_path.read_bytes()
    except OSError as exc:
        prov["reason"] = f"inventory unreadable: {exc}"
        return [], prov
    prov["inventory_digest"] = hashlib.sha256(raw).hexdigest()
    try:
        inv = json.loads(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        prov["reason"] = f"inventory corrupt (invalid JSON): {exc}"
        return [], prov
    if not isinstance(inv, dict):
        prov["reason"] = "inventory not a JSON object"
        return [], prov
    universe = set(inv.get("tier_A_tickers", [])) | set(inv.get("tier_B_tickers", []))
    uni = sorted(str(t) for t in universe)
    prov["n_universe"] = len(uni)
    if not uni:
        prov["reason"] = "inventory tier_A_tickers + tier_B_tickers empty"
    return uni, prov


def _resolve_transformer_universe(ctx: CorpusRefreshContext) -> list:
    """Resolve + validate the training universe (cached). Fails CLOSED when the
    universe is required but unresolvable / empty, or when a bound inventory
    digest does not match — required training-universe provenance must fail
    closed, never silently degrade to an empty universe."""
    if ctx._universe_cache is not None:
        return ctx._universe_cache
    universe, prov = _source_transformer_universe(ctx)
    ctx.universe_provenance = prov

    if ctx.expected_inventory_digest and prov.get("inventory_digest") and (
        prov["inventory_digest"] != ctx.expected_inventory_digest
    ):
        _fail_closed(
            ctx,
            "RenQuant PatchTST INVENTORY-DIGEST MISMATCH",
            (
                f"transformer inventory digest {prov['inventory_digest'][:12]} != bound "
                f"{ctx.expected_inventory_digest[:12]} ({prov.get('inventory_path')}); "
                "the universe file that produced the served corpus changed — refusing to "
                "refresh/rebuild against an unverified universe."
            ),
        )

    if not universe:
        if ctx.require_universe:
            _fail_closed(
                ctx,
                "RenQuant PatchTST UNIVERSE-PROVENANCE FAIL",
                (
                    f"transformer training universe unresolvable/empty "
                    f"(source={prov.get('source')}, reason={prov.get('reason')}, "
                    f"inventory={prov.get('inventory_path')}); required training-universe "
                    "provenance failed — refusing to refresh/rebuild on an empty universe."
                ),
            )
        log.warning(
            "transformer universe empty (%s) but require_universe=False — safe no-op",
            prov.get("reason"),
        )
    ctx._universe_cache = universe
    return universe


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


def _default_expected_asof() -> "dt.date":
    """The most recent completed market session, holiday-agnostic proxy: the last
    business day <= today. Independent of the universe's own bars so a GLOBAL
    freeze (every ticker uniformly stale) is detectable."""
    import numpy as np  # noqa: PLC0415

    today = dt.date.today()
    d = np.busday_offset(np.datetime64(today, "D"), 0, roll="backward")
    return dt.date.fromisoformat(str(d))


def _resolve_expected_asof(ctx: CorpusRefreshContext) -> "dt.date":
    if ctx.freshness_as_of is not None:
        return ctx.freshness_as_of
    if ctx.expected_as_of_fn is not None:
        return ctx.expected_as_of_fn()
    return _default_expected_asof()


# ─────────────────────────── corpus stats / schema / build ─────────────────


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


def _label_horizons(columns) -> "frozenset":
    """Forward-label horizons present in a column set, e.g. {5, 20, 60} from
    ``fwd_5d_excess`` / ``fwd_20d_excess`` / ``fwd_60d_excess``."""
    horizons = set()
    for c in columns:
        lc = str(c).lower()
        if "fwd" in lc or "label" in lc or lc.startswith("y_"):
            m = _LABEL_HORIZON_RE.search(lc)
            if m:
                horizons.add(int(m.group(1)))
    return frozenset(horizons)


def _default_corpus_schema(path: Path) -> dict:
    """Cheap output-contract snapshot of a corpus parquet: sorted column names,
    distinct ticker count, and the set of forward-label horizons. Empty snapshot
    when the file is missing / unreadable so a first-time build is unconstrained.
    """
    empty = {"columns": [], "n_tickers": 0, "label_horizons": frozenset()}
    if not path.exists():
        return dict(empty)
    columns: list = []
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415

        columns = list(pq.read_schema(path).names)
    except Exception:  # pragma: no cover - fall back to a pandas read
        try:
            import pandas as pd  # noqa: PLC0415

            columns = list(pd.read_parquet(path).columns)
        except Exception:
            return dict(empty)
    n_tickers = 0
    if "ticker" in columns:
        try:
            import pandas as pd  # noqa: PLC0415

            n_tickers = int(pd.read_parquet(path, columns=["ticker"])["ticker"].nunique())
        except Exception:  # pragma: no cover - defensive
            n_tickers = 0
    return {
        "columns": sorted(str(c) for c in columns),
        "n_tickers": n_tickers,
        "label_horizons": _label_horizons(columns),
    }


def _resolve_corpus_stats(ctx: CorpusRefreshContext, path: Path) -> tuple:
    if ctx.corpus_stats_fn is not None:
        return ctx.corpus_stats_fn(path)
    return _default_corpus_stats(path)


def _resolve_corpus_schema(ctx: CorpusRefreshContext, path: Path) -> dict:
    if ctx.corpus_schema_fn is not None:
        return ctx.corpus_schema_fn(path)
    return _default_corpus_schema(path)


def _default_build_corpus(ctx: CorpusRefreshContext, staging_path: Path, universe: list) -> None:
    """Rebuild the transformer panel to ``staging_path`` by invoking the existing
    builder against the SAME inventory / labels / integrity-report the corpus is
    built from.

    RUNTIME WIRING NOTE: the served corpus ``transformer_v4_wl200_clean.parquet``
    is the wl200-clean transformer_v4 corpus. This default wires the canonical
    ``scripts/transformer_dataset_builder.py`` to the staging path. If the
    operator's exact wl200-clean recipe diverges, its OUTPUT still has to satisfy
    the served corpus's schema / label / coverage contract (the swap gate) or the
    swap fails closed — a silent feature/universe/schema change can never reach
    the served corpus. Point ``builder_fn`` at the exact recipe to also match the
    build side; that is a one-line injection change with no code churn here.
    """
    builder = ctx.repo_dir / "scripts" / "transformer_dataset_builder.py"
    cmd = [
        sys.executable,
        str(builder),
        "--inventory",
        str(ctx.resolved_inventory_path),
        "--integrity-report",
        str(ctx.resolved_integrity_report_path),
        "--labels",
        str(ctx.resolved_labels_path),
        "--ohlcv-dir",
        str(ctx.ohlcv_dir),
        "--output",
        str(staging_path),
    ]
    log.info("rebuilding transformer corpus to staging [%s]: %s", ctx.builder_recipe, " ".join(cmd))
    subprocess.run(cmd, check=True)  # noqa: S603


# ─────────────────────────── atomic swap helpers ───────────────────────────


def _fsync_file(path: Path) -> None:
    """Best-effort fsync of a file's bytes to durable storage."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - not all filesystems support fsync
        pass


def _fsync_dir(path: Path) -> None:
    """Best-effort fsync of a directory entry (so a rename is durable)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:  # pragma: no cover - directory fsync unsupported on some fs
        pass


def _atomic_replace_corpus(staging: Path, corpus: Path) -> "str | None":
    """Atomically replace ``corpus`` with ``staging`` on ONE filesystem.

    The served corpus is NEVER moved out of the way: the prior is backed up by a
    COPY, then a single ``os.replace`` (atomic rename) overwrites the served path.
    If the replace is interrupted, the served corpus is still the intact prior, so
    it can never disappear. Returns the ``.bak`` path (None on a first-time build).
    """
    _fsync_file(staging)  # staged bytes durable before we touch the served path
    bak: "str | None" = None
    if corpus.exists():
        bak_path = corpus.with_name(corpus.name + ".bak")
        if bak_path.exists():
            bak_path.unlink()
        shutil.copy2(str(corpus), str(bak_path))  # COPY (prior stays in place)
        _fsync_file(bak_path)
        bak = str(bak_path)
    os.replace(str(staging), str(corpus))  # atomic on one filesystem
    _fsync_dir(corpus.parent)
    return bak


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

    Fails CLOSED before any fetch if the required training universe is
    unresolvable (bad inventory provenance).
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        universe = _resolve_transformer_universe(ctx)  # fails closed on bad provenance
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
        if not universe:  # only reachable with require_universe=False
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
    """Guard against a transformer-universe freeze — both a PARTIAL freeze (the
    research tail sits at 2026-02-10 while the ~142-ticker watchlist stays fresh,
    which the watchlist-only scan passed) and a GLOBAL freeze (every ticker
    uniformly stale, which a frontier-relative check alone would miss).

    Reads each transformer ticker's RAW OHLCV bar max date — NOT the built
    corpus, which legitimately ends ~today-60 trading days after the (correct)
    fwd_60d label clip. Reading raw bars means an on-frontier universe never trips
    this guard: genuine input staleness (the bars themselves old) is distinguished
    from the expected fwd_60d frontier.

      * PARTIAL: a ticker is stale when its newest bar lags the universe frontier
        (the freshest bar any ticker has) by > ``freshness_stale_after_days``; if
        > ``freshness_max_stale_fraction`` of the universe is stale, trip.
      * GLOBAL: the universe frontier itself is compared to an INDEPENDENT expected
        completed market session; if the frontier lags that as-of by >
        ``freshness_stale_after_days`` the whole universe is frozen, trip.

    Unassessable inputs (no resolvable dates) fail closed rather than skip. On a
    trip: LOUD ntfy alert and — per ``freshness_fail_on_stale`` — fail the retrain
    (default, fail-closed) or proceed with the warning.
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        universe = _resolve_transformer_universe(ctx)  # fails closed on bad provenance
        if not universe:  # only reachable with require_universe=False
            log.warning("freshness guard: transformer universe empty; cannot assess — skipping")
            return True
        dates = {t: _resolve_ohlcv_max_date(ctx, t) for t in universe}
        known = {t: d for t, d in dates.items() if d is not None}
        missing = {t for t, d in dates.items() if d is None}
        if not known:
            # Unassessable input — no resolvable OHLCV dates. Fail CLOSED (never a
            # silent skip): we cannot certify the training inputs are fresh.
            _fail_closed(
                ctx,
                "RenQuant PatchTST CORPUS-FRESHNESS UNASSESSABLE",
                (
                    f"no OHLCV max dates resolvable for any of {len(universe)} transformer "
                    "tickers; cannot assess training-input freshness — refusing to proceed."
                ),
            )

        frontier = max(known.values())
        expected_asof = _resolve_expected_asof(ctx)
        frontier_lag = _trading_days_between(frontier, expected_asof)
        global_frozen = frontier_lag > ctx.freshness_stale_after_days

        stale = {
            t: d
            for t, d in known.items()
            if _trading_days_between(d, frontier) > ctx.freshness_stale_after_days
        }
        n_stale = len(stale) + len(missing)
        fraction = n_stale / len(universe)
        partial_frozen = fraction > ctx.freshness_max_stale_fraction
        worst = sorted(
            ((_trading_days_between(d, frontier), t) for t, d in stale.items()),
            reverse=True,
        )[:10]
        report = {
            "as_of_frontier": frontier.isoformat(),
            "expected_as_of": expected_asof.isoformat(),
            "frontier_lag_days": frontier_lag,
            "global_frozen": global_frozen,
            "partial_frozen": partial_frozen,
            "n_universe": len(universe),
            "n_stale": n_stale,
            "n_missing": len(missing),
            "stale_fraction": round(fraction, 4),
            "stale_after_days": ctx.freshness_stale_after_days,
            "max_stale_fraction": ctx.freshness_max_stale_fraction,
            "worst_examples": [[lag, t] for lag, t in worst],
        }
        ctx.freshness_report = report

        if not (partial_frozen or global_frozen):
            log.info(
                "freshness guard OK: %d/%d stale (%.1f%% <= %.1f%%), frontier=%s lags market as-of %s by %dd",
                n_stale,
                len(universe),
                fraction * 100,
                ctx.freshness_max_stale_fraction * 100,
                frontier.isoformat(),
                expected_asof.isoformat(),
                frontier_lag,
            )
            return True

        parts = []
        if global_frozen:
            parts.append(
                f"universe frontier {frontier.isoformat()} lags expected completed market "
                f"session {expected_asof.isoformat()} by {frontier_lag} > "
                f"{ctx.freshness_stale_after_days} trading days (GLOBAL FREEZE — every ticker "
                "uniformly stale)"
            )
        if partial_frozen:
            worst_str = ", ".join(f"{t}(-{lag}d)" for lag, t in worst[:8])
            parts.append(
                f"{n_stale}/{len(universe)} transformer tickers stale "
                f"({fraction:.1%} > {ctx.freshness_max_stale_fraction:.0%}); bars lag frontier "
                f"{frontier.isoformat()} by >{ctx.freshness_stale_after_days} trading days. "
                f"Worst: {worst_str}"
            )
        title = "RenQuant PatchTST CORPUS-FREEZE"
        body = (
            "; ".join(parts)
            + f". {'FAILING retrain' if ctx.freshness_fail_on_stale else 'proceeding with warning'}."
        )
        if not ctx.quiet:
            post_ntfy(title, body, ctx.ntfy_topic)
        log.error("freshness guard TRIPPED: %s", body)
        if ctx.freshness_fail_on_stale:
            raise CorpusRefreshError(body)
        return True


class RebuildTransformerCorpusTask:
    """Rebuild the transformer corpus to a STAGING path, then swap it in only if
    it strictly advances the corpus + passes a schema/row/date/coverage sanity gate
    vs the prior corpus.

    Non-destructive & ATOMIC: the prior corpus is backed up by COPY to
    ``<corpus>.bak`` and replaced with a single ``os.replace`` (atomic rename on
    one filesystem) + fsync, so the served corpus is never moved out of the way and
    an interrupted swap can never leave it missing. A regression (staged corpus not
    advancing, materially fewer rows, dropped features / changed label horizon, or
    dropped ticker coverage) NEVER clobbers the served corpus — per
    ``swap_fail_on_regression`` it either fails the retrain (default, fail-closed)
    or warns + keeps the prior corpus.
    """

    def run(self, ctx: CorpusRefreshContext) -> bool:
        corpus = ctx.resolved_corpus_path
        staging = ctx.resolved_staging_path
        report = {
            "corpus_path": str(corpus),
            "staging_path": str(staging),
            "builder_recipe": ctx.builder_recipe,
            "swapped": False,
        }
        ctx.swap_report = report
        if not ctx.rebuild_corpus:
            log.info("corpus rebuild disabled (rebuild_corpus=False); skipping")
            return True
        if ctx.dry_run:
            log.info("[dry-run] would rebuild transformer corpus to %s and sanity-gate swap", staging)
            return True

        universe = _resolve_transformer_universe(ctx)  # fails closed on bad provenance
        if not universe:  # only reachable with require_universe=False
            log.warning("transformer universe empty; skipping rebuild")
            return True

        prior_rows, prior_date = _resolve_corpus_stats(ctx, corpus)
        prior_schema = _resolve_corpus_schema(ctx, corpus)
        report["prior_rows"] = prior_rows
        report["prior_max_date"] = prior_date.isoformat() if prior_date else None
        report["prior_n_tickers"] = prior_schema.get("n_tickers")

        # Build to staging (never touches the served corpus).
        staging.parent.mkdir(parents=True, exist_ok=True)
        if staging.exists():
            staging.unlink()
        builder = ctx.builder_fn or (lambda sp, u: _default_build_corpus(ctx, sp, u))
        builder(staging, universe)

        staged_rows, staged_date = _resolve_corpus_stats(ctx, staging)
        staged_schema = _resolve_corpus_schema(ctx, staging)
        report["staged_rows"] = staged_rows
        report["staged_max_date"] = staged_date.isoformat() if staged_date else None
        report["staged_n_tickers"] = staged_schema.get("n_tickers")

        reasons = self._sanity_reasons(
            ctx, prior_rows, prior_date, prior_schema, staged_rows, staged_date, staged_schema
        )
        report["sanity_reasons"] = reasons
        if reasons:
            self._reject(ctx, staging, report, reasons)
            return True

        # Passed the gate → atomic, non-destructive swap.
        try:
            bak = _atomic_replace_corpus(staging, corpus)
        except Exception as exc:  # served corpus preserved (never moved away)
            report["swap_error"] = str(exc)
            # Defensive: if a partial state ever left the served path missing, restore
            # it from the .bak copy before failing closed.
            bak_path = corpus.with_name(corpus.name + ".bak")
            if not corpus.exists() and bak_path.exists():
                shutil.copy2(str(bak_path), str(corpus))
            _fail_closed(
                ctx,
                "RenQuant PatchTST CORPUS-SWAP FAILED",
                (
                    f"atomic corpus swap failed ({exc}); served corpus left intact "
                    f"({corpus}) — retrain aborted. Staged build kept at {staging} for triage."
                ),
            )
        report["backup_path"] = bak
        report["swapped"] = True
        log.info(
            "transformer corpus swapped (atomic): %s rows=%d max_date=%s tickers=%s "
            "(prior rows=%d max_date=%s tickers=%s, backup=%s)",
            corpus,
            staged_rows,
            report["staged_max_date"],
            report["staged_n_tickers"],
            prior_rows,
            report["prior_max_date"],
            report["prior_n_tickers"],
            bak,
        )
        return True

    @staticmethod
    def _sanity_reasons(
        ctx, prior_rows, prior_date, prior_schema, staged_rows, staged_date, staged_schema
    ) -> list:
        reasons = []
        if staged_rows <= 0 or staged_date is None:
            reasons.append("staged corpus empty or unreadable")
            return reasons
        if prior_rows > 0 and prior_date is not None:
            # Strictly ADVANCE — an equal (non-advanced) corpus is not an advance.
            if ctx.require_date_advance and staged_date <= prior_date:
                reasons.append(
                    f"staged max date {staged_date.isoformat()} does not advance prior "
                    f"{prior_date.isoformat()} (<=)"
                )
            if staged_rows < prior_rows * ctx.min_row_ratio:
                reasons.append(
                    f"staged rows {staged_rows} < {ctx.min_row_ratio:.0%} of prior {prior_rows}"
                )
        # Recipe parity: a wrong builder recipe that still produces a plausible row
        # count must not silently change features / universe / schema / label horizon.
        if ctx.validate_schema and prior_schema:
            prior_cols = set(prior_schema.get("columns", []))
            staged_cols = set(staged_schema.get("columns", []))
            dropped = prior_cols - staged_cols
            if dropped:
                reasons.append(
                    f"staged corpus dropped columns (recipe/schema drift): {sorted(dropped)}"
                )
            prior_hz = prior_schema.get("label_horizons") or frozenset()
            staged_hz = staged_schema.get("label_horizons") or frozenset()
            if prior_hz and prior_hz != staged_hz:
                reasons.append(
                    f"label horizon(s) changed (recipe drift): prior {sorted(prior_hz)} -> "
                    f"staged {sorted(staged_hz)}"
                )
            prior_nt = prior_schema.get("n_tickers") or 0
            staged_nt = staged_schema.get("n_tickers") or 0
            if prior_nt > 0 and staged_nt < prior_nt * ctx.min_ticker_coverage_ratio:
                reasons.append(
                    f"staged ticker coverage {staged_nt} < {ctx.min_ticker_coverage_ratio:.0%} "
                    f"of prior {prior_nt}"
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
            raise CorpusRefreshError(body)


# ─────────────────────────── pipeline ──────────────────────────────────────


def build_pipeline() -> list:
    """Ordered tasks: refresh full-universe OHLCV, guard against a partial/global
    freeze, then rebuild + sanity-gated atomic non-destructive swap."""
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
    p.add_argument("--labels-path", type=Path, default=None)
    p.add_argument("--integrity-report-path", type=Path, default=None)
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
    p.add_argument(
        "--require-universe",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Fail closed when the training universe is unresolvable/empty (default).",
    )
    p.add_argument(
        "--expected-inventory-digest",
        default=None,
        help="Bind the sourced inventory's sha256; a mismatch fails closed.",
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
        "--freshness-as-of",
        type=lambda s: dt.date.fromisoformat(s),
        default=None,
        help="ISO date of the expected completed market session (default: last business day).",
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
        "--min-ticker-coverage-ratio", type=float, default=DEFAULT_MIN_TICKER_COVERAGE_RATIO
    )
    p.add_argument(
        "--validate-schema",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Reject a staged corpus that drops features / changes label horizon / coverage.",
    )
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
        labels_path=args.labels_path,
        integrity_report_path=args.integrity_report_path,
        require_universe=args.require_universe,
        expected_inventory_digest=args.expected_inventory_digest,
        corpus_path=args.corpus_path,
        staging_path=args.staging_path,
        refresh_ohlcv=args.refresh_ohlcv,
        ohlcv_timeout_sec=args.ohlcv_timeout_sec,
        rebuild_corpus=args.rebuild_corpus,
        freshness_stale_after_days=args.freshness_stale_after_days,
        freshness_max_stale_fraction=args.freshness_max_stale_fraction,
        freshness_as_of=args.freshness_as_of,
        freshness_fail_on_stale=args.freshness_fail_on_stale,
        require_date_advance=args.require_date_advance,
        min_row_ratio=args.min_row_ratio,
        min_ticker_coverage_ratio=args.min_ticker_coverage_ratio,
        validate_schema=args.validate_schema,
        swap_fail_on_regression=args.swap_fail_on_regression,
        ntfy_topic=args.ntfy_topic,
        dry_run=args.dry_run,
        quiet=args.quiet,
    )
    run_pipeline(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
