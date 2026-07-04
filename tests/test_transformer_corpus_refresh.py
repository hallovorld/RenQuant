"""Tests for the PatchTST shadow training-corpus refresh (training-data-freeze fix #4).

These cover the frozen-corpus root cause: the transformer training universe
(tier_A + tier_B, ~292 tickers) is half-frozen because only the ~142-ticker live
watchlist gets fresh daily bars, so ``transformer_v4_wl200_clean.parquet`` sat at
2026-02-10. The refresh task must iterate the FULL transformer universe (not just
the watchlist), a single ticker's failure / delisting must not abort, the guard
must fire when more than a configurable fraction is stale (PARTIAL freeze) OR when
the whole universe is uniformly stale vs an independent market as-of (GLOBAL
freeze) while staying quiet at the expected fwd_60d frontier, and the rebuilt
corpus must swap in ATOMICALLY + NON-destructively only when it strictly advances
+ keeps the prior schema / rows / coverage.

Fail-closed contract (Codex review, PR #424): bad universe provenance
(missing/corrupt/empty inventory), unassessable freshness (no resolvable dates or
a global freeze), a non-advancing / wrong-recipe rebuild, and an interrupted swap
all FAIL CLOSED — never a silent skip or a lost served corpus.

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


def _corpus_parquet_rich(
    path: Path,
    max_date: dt.date,
    n_rows: int,
    features=(),
    labels=(),
    n_tickers: int = 3,
) -> None:
    """Write a corpus parquet with explicit feature + multi-horizon label columns
    and a controllable distinct-ticker count (for schema/coverage/recipe tests)."""
    dates = pd.bdate_range(end=pd.Timestamp(max_date), periods=n_rows)
    tickers = [f"T{i % n_tickers}" for i in range(n_rows)]
    data = {"date": dates, "ticker": tickers, "close": 1.0}
    for f in features:
        data[f] = 1.0
    for lb in labels:
        data[lb] = 0.01
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_parquet(path, index=False)


def _capture_ntfy(monkeypatch) -> list:
    posted: list = []
    monkeypatch.setattr(mod, "post_ntfy", lambda *a, **k: posted.append(a))
    return posted


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
    # provenance recorded for the explicit universe
    assert ctx.universe_provenance["source"] == "explicit"
    assert ctx.universe_provenance["n_universe"] == len(universe)


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
    assert ctx.universe_provenance["source"] == "inventory"
    assert ctx.universe_provenance["inventory_digest"]  # sha256 recorded


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


# ─────────────────── universe provenance: fail closed ───────────────────────


def test_provenance_fails_closed_on_missing_inventory(tmp_path, monkeypatch) -> None:
    (tmp_path / "data").mkdir(parents=True)  # no inventory present
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path)  # require_universe defaults True

    # Required training-universe provenance must fail closed, not silently no-op.
    with pytest.raises(mod.CorpusRefreshError, match="UNIVERSE-PROVENANCE|provenance"):
        mod.RefreshTransformerUniverseOhlcvTask().run(ctx)
    assert len(posted) == 1  # loud alert fired
    assert ctx.universe_provenance["reason"]


def test_provenance_fails_closed_on_corrupt_inventory(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "transformer_universe_inventory.json").write_text("{ this is not valid json ")
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path)

    with pytest.raises(mod.CorpusRefreshError):
        mod.RefreshTransformerUniverseOhlcvTask().run(ctx)
    assert len(posted) == 1
    assert "corrupt" in ctx.universe_provenance["reason"]


def test_provenance_fails_closed_on_empty_tiers(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "transformer_universe_inventory.json").write_text(
        json.dumps({"tier_A_tickers": [], "tier_B_tickers": [], "tier_C_tickers": ["NOPE"]})
    )
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path)

    with pytest.raises(mod.CorpusRefreshError):
        mod.RefreshTransformerUniverseOhlcvTask().run(ctx)
    assert len(posted) == 1


def test_provenance_empty_is_safe_noop_when_require_universe_false(tmp_path) -> None:
    (tmp_path / "data").mkdir(parents=True)  # no inventory present
    ctx = _ctx(tmp_path, require_universe=False)
    # Explicit ops escape hatch: degrade to a safe no-op instead of failing closed.
    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert ctx.ohlcv_refresh_summary["n_universe"] == 0


def test_inventory_digest_binding_mismatch_fails_closed(tmp_path, monkeypatch) -> None:
    data = tmp_path / "data"
    data.mkdir(parents=True)
    (data / "transformer_universe_inventory.json").write_text(
        json.dumps({"tier_A_tickers": ["AAPL"], "tier_B_tickers": ["MSFT"]})
    )
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, expected_inventory_digest="deadbeef" * 8)

    with pytest.raises(mod.CorpusRefreshError, match="inventory digest"):
        mod.RefreshTransformerUniverseOhlcvTask().run(ctx)
    assert len(posted) == 1


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
        freshness_as_of=frontier,  # market as-of == frontier → no global freeze
        freshness_stale_after_days=10,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert posted == []
    assert ctx.freshness_report["n_stale"] == 0
    assert ctx.freshness_report["global_frozen"] is False
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
        freshness_as_of=frontier,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

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
        freshness_as_of=frontier,  # frontier fresh → isolates the PARTIAL trip
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError, match="transformer tickers stale"):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)

    assert len(posted) == 1  # LOUD alert fired
    assert ctx.freshness_report["n_stale"] == 10
    assert ctx.freshness_report["stale_fraction"] == 0.5
    assert ctx.freshness_report["global_frozen"] is False  # partial, not global


def test_guard_fails_closed_on_global_freeze_uniform_stale_bars(tmp_path, monkeypatch) -> None:
    """A GLOBALLY frozen universe has ZERO relative staleness (every ticker equal),
    so a frontier-relative check alone passes. The independent market as-of catches
    it: the frontier itself lags the expected completed session."""
    frozen = dt.date(2026, 2, 10)
    asof = dt.date(2026, 6, 30)  # market moved on; the whole universe did not
    universe = [f"T{i}" for i in range(20)]
    ctx = _ctx(
        tmp_path,
        transformer_universe=universe,
        ohlcv_max_dates={t: frozen for t in universe},  # uniformly stale
        freshness_as_of=asof,
        freshness_stale_after_days=10,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError, match="GLOBAL FREEZE"):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)

    assert len(posted) == 1
    assert ctx.freshness_report["global_frozen"] is True
    # zero RELATIVE staleness — only the independent as-of surfaced it
    assert ctx.freshness_report["n_stale"] == 0
    assert ctx.freshness_report["frontier_lag_days"] > 10


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
        freshness_as_of=frontier,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=False,
    )
    posted = _capture_ntfy(monkeypatch)

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
        freshness_as_of=frontier,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError):
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
        freshness_as_of=frontier,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

    assert mod.TransformerUniverseFreshnessGuardTask().run(ctx) is True
    assert posted == []
    assert ctx.freshness_report["as_of_frontier"] == frontier.isoformat()


def test_guard_fails_closed_when_no_dates_resolvable(tmp_path, monkeypatch) -> None:
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, transformer_universe=["A", "B"], ohlcv_max_date_fn=lambda t: None)
    # Unassessable input (no resolvable dates) must FAIL CLOSED, not silently skip.
    with pytest.raises(mod.CorpusRefreshError, match="UNASSESSABLE|cannot assess"):
        mod.TransformerUniverseFreshnessGuardTask().run(ctx)
    assert len(posted) == 1


# ─────────────────────────── rebuild + swap ────────────────────────────────


def test_rebuild_swaps_in_advancing_corpus_non_destructively(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 2, 10), n_rows=100)  # frozen prior

    def fake_builder(staging_path, universe):
        # a fresher, larger corpus (the successful rebuild)
        _corpus_parquet(staging_path, max_date=dt.date(2026, 6, 30), n_rows=140)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

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
    # ... and the staging file was consumed by the atomic replace.
    assert not ctx.resolved_staging_path.exists()


def test_rebuild_rejects_regressed_corpus_and_keeps_prior(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 6, 30), n_rows=140)  # good prior

    def fake_builder(staging_path, universe):
        # a REGRESSED rebuild: older date + far fewer rows (partial build)
        _corpus_parquet(staging_path, max_date=dt.date(2026, 2, 10), n_rows=40)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

    # fail-closed: a regression must NOT clobber the served corpus
    with pytest.raises(mod.CorpusRefreshError, match="rejected"):
        mod.RebuildTransformerCorpusTask().run(ctx)

    assert len(posted) == 1  # LOUD alert fired
    assert ctx.swap_report["swapped"] is False
    # served corpus untouched (still the good prior) ...
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 6, 30)
    # ... no .bak created, staged build dropped.
    assert not corpus.with_name(corpus.name + ".bak").exists()
    assert not ctx.resolved_staging_path.exists()


def test_rebuild_rejects_equal_non_advancing_frontier(tmp_path, monkeypatch) -> None:
    """require_date_advance means STRICTLY advance — an equal (non-advanced) staged
    corpus must be rejected despite matching rows / schema."""
    same_date = dt.date(2026, 6, 30)
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=same_date, n_rows=140)

    def fake_builder(staging_path, universe):
        _corpus_parquet(staging_path, max_date=same_date, n_rows=140)  # equal frontier

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError, match="rejected"):
        mod.RebuildTransformerCorpusTask().run(ctx)
    assert any("does not advance" in r for r in ctx.swap_report["sanity_reasons"])
    assert len(posted) == 1
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == same_date  # untouched


def test_rebuild_rejects_wrong_recipe_schema_and_label_drift(tmp_path, monkeypatch) -> None:
    """A wrong builder recipe can produce a PLAUSIBLE row count + an advancing date
    while silently dropping features / changing the label horizon. The schema gate
    fails it closed."""
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet_rich(
        corpus,
        max_date=dt.date(2026, 6, 30),
        n_rows=140,
        features=["feat_a", "feat_b"],
        labels=["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"],
        n_tickers=3,
    )

    def fake_builder(staging_path, universe):
        # advancing date + equal rows (passes date/row checks) BUT wrong recipe:
        # dropped feat_b and dropped the fwd_60d label horizon.
        _corpus_parquet_rich(
            staging_path,
            max_date=dt.date(2026, 7, 30),
            n_rows=140,
            features=["feat_a"],
            labels=["fwd_5d_excess", "fwd_20d_excess"],
            n_tickers=3,
        )

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError, match="rejected"):
        mod.RebuildTransformerCorpusTask().run(ctx)
    reasons = ctx.swap_report["sanity_reasons"]
    assert any("dropped columns" in r for r in reasons)
    assert any("label horizon" in r for r in reasons)
    assert len(posted) == 1
    # served corpus untouched (still has feat_b + fwd_60d)
    schema = mod._default_corpus_schema(corpus)
    assert "feat_b" in schema["columns"]
    assert 60 in schema["label_horizons"]


def test_rebuild_rejects_ticker_coverage_drop(tmp_path, monkeypatch) -> None:
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet_rich(
        corpus,
        max_date=dt.date(2026, 6, 30),
        n_rows=140,
        labels=["fwd_60d_excess"],
        n_tickers=10,
    )

    def fake_builder(staging_path, universe):
        # advancing + plausible rows + same columns, but collapsed to 1 ticker
        _corpus_parquet_rich(
            staging_path,
            max_date=dt.date(2026, 7, 30),
            n_rows=140,
            labels=["fwd_60d_excess"],
            n_tickers=1,
        )

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

    with pytest.raises(mod.CorpusRefreshError, match="rejected"):
        mod.RebuildTransformerCorpusTask().run(ctx)
    assert any("ticker coverage" in r for r in ctx.swap_report["sanity_reasons"])
    assert len(posted) == 1


def test_rebuild_swap_interruption_preserves_served_corpus(tmp_path, monkeypatch) -> None:
    """If the atomic replace is interrupted (injected failure), the served corpus is
    never moved out of the way — it must remain the intact prior, not disappear."""
    corpus = tmp_path / "data" / "transformer_v4_wl200_clean.parquet"
    _corpus_parquet(corpus, max_date=dt.date(2026, 2, 10), n_rows=100)

    def fake_builder(staging_path, universe):
        _corpus_parquet(staging_path, max_date=dt.date(2026, 6, 30), n_rows=140)

    ctx = _ctx(tmp_path, transformer_universe=["AAA"], builder_fn=fake_builder)
    posted = _capture_ntfy(monkeypatch)

    def boom(src, dst):
        raise OSError("disk full during rename")

    monkeypatch.setattr(mod.os, "replace", boom)  # interrupt the atomic swap

    with pytest.raises(mod.CorpusRefreshError, match="SWAP FAILED|swap failed"):
        mod.RebuildTransformerCorpusTask().run(ctx)

    assert len(posted) == 1
    assert ctx.swap_report["swapped"] is False
    # served corpus still present + intact (the prior), never lost
    assert corpus.exists()
    _, served_date = mod._default_corpus_stats(corpus)
    assert served_date == dt.date(2026, 2, 10)


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
    posted = _capture_ntfy(monkeypatch)

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
        freshness_as_of=frontier,
        freshness_max_stale_fraction=0.10,
        freshness_fail_on_stale=True,
    )
    posted = _capture_ntfy(monkeypatch)

    assert mod.RefreshTransformerUniverseOhlcvTask().run(ctx) is True
    assert ctx.ohlcv_refresh_summary["n_stale"] == 8

    with pytest.raises(mod.CorpusRefreshError):
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


def test_default_corpus_schema_reads_columns_tickers_and_labels(tmp_path) -> None:
    corpus = tmp_path / "c.parquet"
    _corpus_parquet_rich(
        corpus,
        max_date=dt.date(2026, 6, 30),
        n_rows=30,
        features=["feat_a"],
        labels=["fwd_5d_excess", "fwd_60d_excess"],
        n_tickers=3,
    )
    schema = mod._default_corpus_schema(corpus)
    assert "feat_a" in schema["columns"]
    assert schema["n_tickers"] == 3
    assert schema["label_horizons"] == frozenset({5, 60})
    # missing file → empty (unconstrained) snapshot
    empty = mod._default_corpus_schema(tmp_path / "missing.parquet")
    assert empty["columns"] == [] and empty["n_tickers"] == 0


def test_label_horizons_detects_forward_columns() -> None:
    assert mod._label_horizons(["date", "ticker", "close"]) == frozenset()
    assert mod._label_horizons(["fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"]) == frozenset(
        {5, 20, 60}
    )


def test_expected_asof_default_is_a_business_day(tmp_path) -> None:
    asof = mod._default_expected_asof()
    assert asof.weekday() < 5  # Mon-Fri
    # explicit override wins
    fixed = dt.date(2026, 6, 30)
    ctx = _ctx(tmp_path, freshness_as_of=fixed)
    assert mod._resolve_expected_asof(ctx) == fixed


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
        "RebuildRawLabelSidecarTask",
    ]


# ─────────────────────────── TRUE-recipe default builder (S12 B1) ───────────
# The served corpus is the PROD fund panel subset to the live strategy
# watchlist with labels dropna'd (diagnosis §1) — the default builder_fn must
# resolve the committed base-data recipe, source the watchlist from the PINNED
# strategy config, ignore the OHLCV inventory universe for row selection, and
# FAIL CLOSED (never fall back to the divergent legacy recipe) when the recipe
# is unresolvable. All base-data imports are faked — no real base-data pin,
# panel read, or production write happens here.


class _FakeRecipe:
    """Recording stand-in for renquant_base_data.transformer_corpus."""

    def __init__(self) -> None:
        self.watchlist_calls: list = []
        self.build_calls: list = []

    def load_watchlist(self, strategy_config_path):
        self.watchlist_calls.append(Path(strategy_config_path))
        return ["AAPL", "MSFT"]

    def build_transformer_corpus(self, fund_panel_path, watchlist, output_path, **kw):
        self.build_calls.append((Path(fund_panel_path), list(watchlist), Path(output_path)))
        _corpus_parquet(Path(output_path), max_date=dt.date(2026, 6, 30), n_rows=140)
        return {"n_rows": 140, "n_tickers": 2, "max_date": "2026-06-30"}


def _install_fake_recipe(monkeypatch) -> _FakeRecipe:
    import types

    fake = _FakeRecipe()
    module = types.ModuleType("renquant_base_data.transformer_corpus")
    module.load_watchlist = fake.load_watchlist
    module.build_transformer_corpus = fake.build_transformer_corpus
    pkg = types.ModuleType("renquant_base_data")
    pkg.transformer_corpus = module
    monkeypatch.setitem(sys.modules, "renquant_base_data", pkg)
    monkeypatch.setitem(sys.modules, "renquant_base_data.transformer_corpus", module)
    return fake


def test_default_builder_uses_the_true_base_data_recipe(tmp_path, monkeypatch) -> None:
    fake = _install_fake_recipe(monkeypatch)
    cfg = tmp_path / "strategy_config.json"
    cfg.write_text(json.dumps({"watchlist": ["AAPL", "MSFT"]}))
    panel = tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet"
    staging = tmp_path / "data" / "corpus.staging"
    ctx = _ctx(tmp_path, fund_panel_path=panel, strategy_config_path=cfg)

    mod._default_build_corpus(ctx, staging, ["R0", "R1", "R2"])  # inventory universe

    assert fake.watchlist_calls == [cfg]  # watchlist from the PINNED config...
    assert fake.build_calls == [(panel, ["AAPL", "MSFT"], staging)]
    # ...NOT from the OHLCV inventory universe passed through the seam.
    assert staging.exists()


def test_default_builder_fails_closed_when_recipe_unresolvable(tmp_path, monkeypatch) -> None:
    # Simulate a base-data pin that predates the committed recipe: the import
    # raises, and the builder must fail CLOSED (no legacy-recipe fallback).
    monkeypatch.setitem(sys.modules, "renquant_base_data", None)
    monkeypatch.setitem(sys.modules, "renquant_base_data.transformer_corpus", None)
    ctx = _ctx(tmp_path)

    with pytest.raises(mod.CorpusRefreshError, match="TRUE-recipe builder unresolvable"):
        mod._default_build_corpus(ctx, tmp_path / "corpus.staging", ["AAA"])
    assert not (tmp_path / "corpus.staging").exists()


def test_default_strategy_config_resolution_mirrors_subrepo_env(tmp_path, monkeypatch) -> None:
    ctx = _ctx(tmp_path / "RenQuant")
    # RENQUANT_SUBREPO_ROOT (exported by weekly_retrain_patchtst.sh) wins...
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(tmp_path / "assembly" / "repos"))
    assert ctx.resolved_strategy_config_path == (
        tmp_path / "assembly" / "repos" / "renquant-strategy-104" / "configs" / "strategy_config.json"
    )
    # ...else the sibling-checkout default next to the umbrella repo.
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT")
    assert ctx.resolved_strategy_config_path == (
        tmp_path / "renquant-strategy-104" / "configs" / "strategy_config.json"
    )
    # An explicit override beats both.
    explicit = tmp_path / "elsewhere" / "strategy_config.shadow.json"
    ctx2 = _ctx(tmp_path / "RenQuant", strategy_config_path=explicit)
    assert ctx2.resolved_strategy_config_path == explicit


def test_default_fund_panel_and_recipe_string_point_at_the_prod_panel(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.resolved_fund_panel_path == (
        tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet"
    )
    # The audit string records the TRUE recipe, not the legacy raw-OHLCV one.
    assert "renquant_base_data.transformer_corpus" in mod.DEFAULT_BUILDER_RECIPE
    assert "transformer_dataset_builder" not in mod.DEFAULT_BUILDER_RECIPE


def test_cli_wires_fund_panel_and_strategy_config_paths() -> None:
    args = mod.parse_args(
        [
            "--fund-panel-path", "/x/panel.parquet",
            "--strategy-config-path", "/x/strategy_config.json",
        ]
    )
    assert args.fund_panel_path == Path("/x/panel.parquet")
    assert args.strategy_config_path == Path("/x/strategy_config.json")
