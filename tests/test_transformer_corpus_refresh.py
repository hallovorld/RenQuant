"""Tests for the PatchTST shadow training-corpus refresh (training-data-freeze fix #4).

These cover the frozen-corpus root cause: the transformer training universe
(tier_A + tier_B, ~292 tickers) is half-frozen because only the ~142-ticker live
watchlist gets fresh daily bars, so ``transformer_v4_wl200_clean.parquet`` sat at
2026-02-10. The refresh task must iterate the FULL transformer universe (not just
the watchlist), a single ticker's failure / delisting must not abort, the guard
must fire when more than a configurable fraction is stale while staying quiet at
the expected fwd_60d frontier, and the rebuilt corpus must swap in NON-destructively
only when it advances + passes a row/date sanity gate.

All fetch / freshness / builder IO is mocked or uses tmp fixtures — no real network
fetch, no real rebuild, and no production data write ever happens here.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.refresh_transformer_corpus as mod  # noqa: E402


def _ohlcv(end: dt.date, periods: int = 5) -> pd.DataFrame:
    """A small OHLCV frame whose newest bar is ``end`` (DatetimeIndex, as the real
    ``fetch_ohlcv_incremental`` returns)."""
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=periods)
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 100},
        index=idx,
    )


def _ctx(tmp_path: Path, **kw) -> mod.CorpusRefreshContext:
    return mod.CorpusRefreshContext(repo_dir=tmp_path, **kw)


def _corpus_parquet(path: Path, max_date: dt.date, n_rows: int = 100) -> None:
    """Write a tiny transformer-corpus-shaped parquet (has a ``date`` column)."""
    dates = pd.bdate_range(end=pd.Timestamp(max_date), periods=n_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date": dates, "ticker": "AAA", "close": 1.0}).to_parquet(path, index=False)


# ─────────────────────────── refresh task ──────────────────────────────────


def test_refresh_iterates_full_transformer_universe_not_just_watchlist(tmp_path) -> None:
    frontier = dt.date(2026, 6, 30)
    watchlist = ["AAPL", "MSFT"]
    research = ["XYZ", "QRS", "TUV", "WXY"]  # names that had no refresh cadence
    universe = watchlist + research
    calls: list[str] = []

    def fake_fetch(sym, *, timeout_sec=None):
        calls.append(sym)
        return _ohlcv(frontier)

    ctx = _ctx(tmp_path, transformer_universe=universe, fetch_fn=fake_fetch)

    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    # The WHOLE transformer universe is refreshed, not just the live watchlist.
    assert set(calls) == set(universe)
    assert set(research).issubset(set(calls))
    summary = ctx.ohlcv_refresh_summary
    assert summary["n_universe"] == len(universe)
    assert summary["n_refreshed"] == len(universe)
    assert summary["n_failed"] == 0
    assert summary["n_delisted"] == 0


def test_refresh_sources_universe_from_inventory_tier_a_and_b(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "transformer_universe_inventory.json").write_text(
        json.dumps(
            {
                "tier_A_tickers": ["AAPL", "MSFT"],
                "tier_B_tickers": ["XYZ", "QRS", "TUV"],
                "tier_C_tickers": ["NOPE"],
            }
        )
    )
    calls: list[str] = []

    def fake_fetch(sym, *, timeout_sec=None):
        calls.append(sym)
        return _ohlcv(dt.date(2026, 6, 30))

    ctx = _ctx(tmp_path, fetch_fn=fake_fetch)

    mod.RefreshTransformerUniverseOhlcvTask().run(ctx)

    # tier_A + tier_B only — tier_C (skip) is excluded, mirroring the builder.
    assert set(calls) == {"AAPL", "MSFT", "XYZ", "QRS", "TUV"}


def test_refresh_delisted_and_failed_tickers_do_not_abort(tmp_path) -> None:
    frontier = dt.date(2026, 6, 30)
    universe = ["AAPL", "MSFT", "DEAD", "BOOM", "XYZ"]

    def fake_fetch(sym, *, timeout_sec=None):
        if sym == "BOOM":
            raise RuntimeError("network exploded")
        if sym == "DEAD":
            return pd.DataFrame()  # delisted: no bars
        return _ohlcv(frontier)

    ctx = _ctx(tmp_path, transformer_universe=universe, fetch_fn=fake_fetch)

    # A single ticker's failure / delisting must NOT abort the refresh.
    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    s = ctx.ohlcv_refresh_summary
    assert s["n_universe"] == 5
    assert s["n_failed"] == 1
    assert s["n_delisted"] == 1
    assert s["n_refreshed"] == 3
    # counts partition the universe
    assert s["n_refreshed"] + s["n_stale"] + s["n_delisted"] + s["n_failed"] == s["n_universe"]


def test_refresh_dry_run_makes_no_fetch(tmp_path) -> None:
    called: list[str] = []

    def fake_fetch(sym, *, timeout_sec=None):
        called.append(sym)
        return _ohlcv(dt.date(2026, 6, 30))

    ctx = _ctx(tmp_path, transformer_universe=["A", "B"], fetch_fn=fake_fetch, dry_run=True)

    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert called == []
    assert ctx.ohlcv_refresh_summary["n_universe"] == 2


def test_refresh_disabled_skips_fetch(tmp_path) -> None:
    called: list[str] = []

    def fake_fetch(sym, *, timeout_sec=None):
        called.append(sym)
        return _ohlcv(dt.date(2026, 6, 30))

    ctx = _ctx(tmp_path, transformer_universe=["A", "B"], fetch_fn=fake_fetch, refresh_ohlcv=False)

    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert called == []


def test_refresh_empty_universe_is_safe_noop(tmp_path) -> None:
    (tmp_path / "data").mkdir(parents=True)  # no inventory present
    ctx = _ctx(tmp_path)
    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert ctx.ohlcv_refresh_summary["n_universe"] == 0


def test_refresh_resolves_default_fetch_fn_when_not_injected(tmp_path, monkeypatch) -> None:
    """Runtime-wiring seam: with no fetch_fn injected the task resolves the real
    base-data primitive via ``_default_fetch_fn`` (patched here so no import)."""
    calls: list[str] = []

    def fake_fetch(sym, *, timeout_sec=None):
        calls.append(sym)
        return _ohlcv(dt.date(2026, 6, 30))

    monkeypatch.setattr(mod, "_default_fetch_fn", lambda: fake_fetch)
    ctx = _ctx(tmp_path, transformer_universe=["A", "B", "C"])

    mod.RefreshTransformerUniverseOhlcvTask().run(ctx)

    assert set(calls) == {"A", "B", "C"}


# ─────────────────────────── freshness guard ───────────────────────────────


def test_guard_quiet_when_bars_fresh_despite_fwd60d_panel_frontier(tmp_path, monkeypatch) -> None:
    """The guard reads RAW OHLCV bars (frontier ~today). A corpus built from them
    legitimately ends ~60 trading days earlier (fwd_60d clip) — that expected
    frontier must NOT be mistaken for input staleness, so with all raw bars fresh
    the guard stays silent."""
    frontier = dt.date(2026, 6, 30)
    universe = [f"T{i}" for i in range(20)]
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates={t: frontier for t in universe},
        freshness_stale_after_days=10,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert posted == []
    assert ctx.freshness_report["n_stale"] == 0
    assert ctx.freshness_report["as_of_frontier"] == frontier.isoformat()


def test_guard_quiet_below_threshold(tmp_path, monkeypatch) -> None:
    frontier = dt.date(2026, 6, 30)
    frozen = dt.date(2026, 5, 12)  # ~35 trading days behind
    universe = [f"T{i}" for i in range(20)]
    md = {t: frontier for t in universe}
    md["T0"] = frozen  # 1/20 = 5% <= 10%
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates=md,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert posted == []
    assert ctx.freshness_report["n_stale"] == 1


def test_guard_fails_closed_on_partial_freeze(tmp_path, monkeypatch) -> None:
    frontier = dt.date(2026, 6, 30)
    frozen = dt.date(2026, 2, 10)  # the actual frozen-corpus date
    fresh_tickers = [f"F{i}" for i in range(10)]  # watchlist-like, fresh
    frozen_tickers = [f"Z{i}" for i in range(10)]  # research, frozen at 2026-02-10
    universe = fresh_tickers + frozen_tickers
    md = {t: frontier for t in fresh_tickers}
    md.update({t: frozen for t in frozen_tickers})
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates=md,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    with pytest.raises(RuntimeError, match="transformer tickers stale"):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)

    assert len(posted) == 1  # LOUD alert fired
    assert ctx.freshness_report["n_stale"] == 10
    assert ctx.freshness_report["stale_fraction"] == 0.5


def test_guard_proceeds_with_warning_when_fail_disabled(tmp_path, monkeypatch) -> None:
    frontier = dt.date(2026, 6, 30)
    frozen = dt.date(2026, 2, 10)
    universe = [f"T{i}" for i in range(20)]
    md = {t: frontier for t in universe}
    for t in universe[:10]:
        md[t] = frozen
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates=md,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=False,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    # proceeds (returns True) but still alerts loudly
    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert len(posted) == 1


def test_guard_counts_missing_bars_as_stale(tmp_path, monkeypatch) -> None:
    frontier = dt.date(2026, 6, 30)
    universe = [f"T{i}" for i in range(10)]
    md = {t: frontier for t in universe}
    for t in universe[:3]:
        md[t] = None  # no bars at all (never fetched / delisted with no cache)
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates=md,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: None)

    with pytest.raises(RuntimeError):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)
    assert ctx.freshness_report["n_missing"] == 3
    assert ctx.freshness_report["n_stale"] == 3


def test_guard_uses_injected_ohlcv_reader(tmp_path, monkeypatch) -> None:
    frontier = dt.date(2026, 6, 30)
    universe = ["AAA", "BBB", "CCC"]
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_date_fn=lambda t: frontier,
        freshness_fail_on_stale=True,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert posted == []
    assert ctx.freshness_report["as_of_frontier"] == frontier.isoformat()


def test_guard_skips_when_no_dates_resolvable(tmp_path) -> None:
    ctx = _ctx(tmp_path, transformer_universe=["A", "B"], ohlcv_max_date_fn=lambda t: None)
    # cannot assess → soft skip (does not raise)
    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True


# ─────────────────────────── rebuild + swap ────────────────────────────────


def test_rebuild_swaps_in_advancing_corpus_non_destructively(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 2, 10), n_rows=100)  # frozen prior

    def fake_builder(staging_path, universe):
        # a fresher, larger corpus (the successful rebuild)
        _corpus_parquet(staging_path, max_date=dt.date(2026, 6, 30), n_rows=140)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    assert mod.RebuildTransformerCorpusTask().run(ctx) is True
    assert posted == []  # clean swap, no alert
    assert ctx.swap_report["swapped"] is True
    # served corpus now advanced ...
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 6, 30)
    # ... prior corpus preserved as .bak (non-destructive) ...
    bak = corpus.with_name(corpus.name + ".bak")
    assert bak.exists()
    _, bak_date = mod._default_corpus_stats(bak)
    assert bak_date == dt.date(2026, 2, 10)
    # ... and the staging file was consumed.
    assert not ctx.resolved_staging_path.exists()


def test_rebuild_rejects_regressed_corpus_and_keeps_prior(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 6, 30), n_rows=140)  # good prior

    def fake_builder(staging_path, universe):
        # a REGRESSED rebuild: older date + far fewer rows (partial build)
        _corpus_parquet(staging_path, max_date=dt.date(2026, 2, 10), n_rows=40)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    # fail-closed: a regression must NOT clobber the served corpus
    with pytest.raises(RuntimeError, match="rejected"):
        mod.RebuildTransformerCorpusTask().run(ctx)

    assert len(posted) == 1  # LOUD alert fired
    assert ctx.swap_report["swapped"] is False
    # served corpus untouched (still the good prior) ...
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 6, 30)
    # ... no .bak created, staged build dropped.
    assert not corpus.with_name(corpus.name + ".bak").exists()
    assert not ctx.resolved_staging_path.exists()


def test_rebuild_regression_proceeds_when_fail_disabled(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 6, 30), n_rows=140)

    def fake_builder(staging_path, universe):
        _corpus_parquet(staging_path, max_date=dt.date(2026, 2, 10), n_rows=40)

    ctx = _ctx(
        tmp_path,
        transformer_universe=["AAA"],
        builder_fn=fake_builder,
        swap_fail_on_regression=False,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    # proceeds (returns True), keeps prior corpus, still alerts
    assert mod.RebuildTransformerCorpusTask().run(ctx) is True
    assert len(posted) == 1
    assert ctx.swap_report["swapped"] is False
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 6, 30)


def test_rebuild_first_time_no_prior_swaps_in(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"  # does not exist yet

    def fake_builder(staging_path, universe):
        _corpus_parquet(staging_path, max_date=dt.date(2026, 6, 30), n_rows=140)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: None)

    assert mod.RebuildTransformerCorpusTask().run(ctx) is True
    assert ctx.swap_report["swapped"] is True
    assert corpus.exists()
    assert not corpus.with_name(corpus.name + ".bak").exists()  # nothing to back up


def test_rebuild_dry_run_does_not_build_or_swap(tmp_path) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 2, 10), n_rows=100)
    built: list = []

    def fake_builder(staging_path, universe):
        built.append(staging_path)
        _corpus_parquet(staging_path, max_date=dt.date(2026, 6, 30), n_rows=140)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder, dry_run=True)

    assert mod.RebuildTransformerCorpusTask().run(ctx) is True
    assert built == []
    assert ctx.swap_report["swapped"] is False
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 2, 10)  # untouched


def test_rebuild_disabled_skips(tmp_path) -> None:
    ctx = _ctx(tmp_path, rebuild_corpus=False)
    assert mod.RebuildTransformerCorpusTask().run(ctx) is True
    assert ctx.swap_report["swapped"] is False


# ─────────────────────────── end to end ────────────────────────────────────


def test_refresh_then_guard_catches_partial_freeze_end_to_end(tmp_path, monkeypatch) -> None:
    """Refresh the whole universe, then the guard catches the research-ticker
    freeze that the watchlist-only scan silently passed."""
    frontier = dt.date(2026, 6, 30)
    frozen = dt.date(2026, 2, 10)
    watchlist = [f"W{i}" for i in range(8)]
    research = [f"R{i}" for i in range(8)]
    universe = watchlist + research

    def fake_fetch(sym, *, timeout_sec=None):
        # fresh where the live path already refreshes; frozen for the research
        # tail that has no refresh cadence upstream
        return _ohlcv(frontier) if sym in watchlist else _ohlcv(frozen)

    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        fetch_fn=fake_fetch,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))

    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert ctx.ohlcv_refresh_summary["n_stale"] == 8

    with pytest.raises(RuntimeError):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)
    assert len(posted) == 1
    assert ctx.freshness_report["n_stale"] == 8


# ─────────────────────────── helpers / pipeline ─────────────────────────────


def test_default_ohlcv_max_date_reads_parquet(tmp_path) -> None:
    ohlcv_dir = tmp_path / "ohlcv"
    (ohlcv_dir / "AAA").mkdir(parents=True)
    _ohlcv(dt.date(2026, 6, 30)).to_parquet(ohlcv_dir / "AAA" / "1d.parquet")

    assert mod._default_ohlcv_max_date(ohlcv_dir, "AAA") == dt.date(2026, 6, 30)
    assert mod._default_ohlcv_max_date(ohlcv_dir, "MISSING") is None


def test_default_corpus_stats_reads_parquet(tmp_path) -> None:
    corpus = tmp_path / "c.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 6, 30), n_rows=42)
    n_rows, max_date = mod._default_corpus_stats(corpus)
    assert n_rows == 42
    assert max_date == dt.date(2026, 6, 30)
    assert mod._default_corpus_stats(tmp_path / "missing.parquet") == (0, None)


def test_trading_days_between_is_business_day_gap() -> None:
    assert mod._trading_days_between(dt.date(2026, 6, 30), dt.date(2026, 6, 30)) == 0
    assert mod._trading_days_between(dt.date(2026, 7, 1), dt.date(2026, 6, 30)) == 0
    # Mon..Mon following week = 5 business days
    assert mod._trading_days_between(dt.date(2026, 6, 22), dt.date(2026, 6, 29)) == 5
    assert mod._trading_days_between(dt.date(2026, 2, 10), dt.date(2026, 6, 30)) > 10


def test_pipeline_order_is_refresh_guard_rebuild() -> None:
    tasks = [type(t).__name__ for t in mod.build_pipeline()]
    assert tasks == [
        "RefreshTransformerUniverseOhlcvTask",
        "TransformerUniverseFreshnessGuardTask",
        "RebuildTransformerCorpusTask",
    ]
