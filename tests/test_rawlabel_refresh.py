"""Tests for the RAWLABEL-sidecar refresh stage (S12 rawlabel gap, B1 pattern).

The served ``alpha158_291_fundamental_dataset_rawlabel.parquet`` — the raw
un-z-scored forward-label file both calibrator fits read — was a one-off
research build with no refresh mechanism, so it froze at 2026-02-11 and the
shadow promote refused (``rawlabel: cutoff=2026-02-11 age=142d sla=28d
OFF-SLA``). ``RebuildRawLabelSidecarTask`` rebuilds it in the SAME weekly
``refresh_transformer_corpus.py`` run as the corpus, from the same prod fund
panel + freshly-refreshed OHLCV, with the SAME staged → sanity-gated → atomic
non-destructive swap (keep-prior-on-reject) discipline.

These tests mirror ``tests/test_transformer_corpus_refresh.py``'s swap-task
coverage: advancing swap, the reject matrix (non-advancing / shrunken /
schema-drifted / coverage-dropped staged builds never clobber the served
sidecar), interrupted-swap preservation, warn-and-proceed mode, first-time
build, dry-run/disabled no-ops, and the fail-closed default builder seam. All
builder / disk IO uses tmp fixtures — no real build or production data write
ever happens here.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.refresh_transformer_corpus as mod  # noqa: E402


def _ctx(tmp_path: Path, **kw) -> mod.CorpusRefreshContext:
    return mod.CorpusRefreshContext(repo_dir=tmp_path, **kw)


def _sidecar_parquet(
    path: Path,
    max_date: dt.date,
    n_rows: int = 100,
    features=("KMID", "ROC60"),
    labels=("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "fwd_60d_excess_raw"),
    n_tickers: int = 3,
) -> None:
    """Write a tiny rawlabel-sidecar-shaped parquet (date/ticker + feature +
    multi-horizon label columns, controllable coverage)."""
    dates = pd.bdate_range(end=pd.Timestamp(max_date), periods=n_rows)
    tickers = [f"T{i % n_tickers}" for i in range(n_rows)]
    data = {"ticker": tickers, "date": dates}
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


def _builder(max_date: dt.date, n_rows: int = 120, calls: "list | None" = None, **sidecar_kw):
    def build(staging: Path) -> None:
        if calls is not None:
            calls.append(staging)
        _sidecar_parquet(staging, max_date=max_date, n_rows=n_rows, **sidecar_kw)

    return build


PRIOR_DATE = dt.date(2026, 2, 11)  # the frozen served vintage
FRESH_DATE = dt.date(2026, 7, 2)  # the bar frontier


# ─────────────────────────── swap happy path ────────────────────────────────


def test_rawlabel_swaps_in_advancing_sidecar_non_destructively(tmp_path, monkeypatch) -> None:
    _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(FRESH_DATE, n_rows=120))
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    assert mod.RebuildRawLabelSidecarTask().run(ctx) is True

    report = ctx.rawlabel_swap_report
    assert report["swapped"] is True
    assert report["prior_max_date"] == PRIOR_DATE.isoformat()
    assert report["staged_max_date"] == FRESH_DATE.isoformat()
    # The served sidecar now IS the staged build; the prior is kept as .bak.
    df = pd.read_parquet(served)
    assert pd.to_datetime(df["date"]).max().date() == FRESH_DATE
    bak = served.with_name(served.name + ".bak")
    assert bak.exists()
    assert pd.to_datetime(pd.read_parquet(bak)["date"]).max().date() == PRIOR_DATE
    # Staging was consumed by the atomic rename.
    assert not ctx.resolved_rawlabel_staging_path.exists()


def test_rawlabel_first_time_no_prior_swaps_in(tmp_path, monkeypatch) -> None:
    _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(FRESH_DATE))
    assert not ctx.resolved_rawlabel_path.exists()

    assert mod.RebuildRawLabelSidecarTask().run(ctx) is True
    assert ctx.rawlabel_swap_report["swapped"] is True
    assert ctx.rawlabel_swap_report["backup_path"] is None
    assert ctx.resolved_rawlabel_path.exists()


def test_rawlabel_default_paths_point_at_the_served_sidecar(tmp_path) -> None:
    ctx = _ctx(tmp_path)
    assert ctx.resolved_rawlabel_path == (
        tmp_path / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet"
    )
    assert ctx.resolved_rawlabel_staging_path == (
        tmp_path
        / "data"
        / "alpha158_291_fundamental_dataset_rawlabel.parquet.staging"
    )
    # The audit string records the committed base-data recipe.
    assert "renquant_base_data.rawlabel_sidecar" in mod.DEFAULT_RAWLABEL_BUILDER_RECIPE
    assert "build_raw_fwd60d_label" not in mod.DEFAULT_RAWLABEL_BUILDER_RECIPE


# ─────────────────────────── reject matrix (keep-prior) ─────────────────────


def _assert_prior_kept(ctx, served: Path) -> None:
    assert pd.to_datetime(pd.read_parquet(served)["date"]).max().date() == PRIOR_DATE
    assert not ctx.resolved_rawlabel_staging_path.exists()
    assert ctx.rawlabel_swap_report["swapped"] is False
    assert ctx.rawlabel_swap_report["sanity_reasons"]


def test_rawlabel_rejects_non_advancing_rebuild(tmp_path, monkeypatch) -> None:
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(PRIOR_DATE))  # equal, not >
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    with pytest.raises(mod.CorpusRefreshError, match="does not advance"):
        mod.RebuildRawLabelSidecarTask().run(ctx)
    _assert_prior_kept(ctx, served)
    assert any("RAWLABEL-REBUILD REJECTED" in t for t, *_ in posted)


def test_rawlabel_rejects_shrunken_rebuild(tmp_path, monkeypatch) -> None:
    _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(FRESH_DATE, n_rows=10))
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    with pytest.raises(mod.CorpusRefreshError, match="rows"):
        mod.RebuildRawLabelSidecarTask().run(ctx)
    _assert_prior_kept(ctx, served)


def test_rawlabel_rejects_schema_and_label_horizon_drift(tmp_path, monkeypatch) -> None:
    _capture_ntfy(monkeypatch)
    # Staged build DROPS the raw label column (a wrong-recipe rebuild that still
    # advances the frontier + keeps row counts) — the contract gate refuses it.
    ctx = _ctx(
        tmp_path,
        rawlabel_builder_fn=_builder(
            FRESH_DATE,
            n_rows=120,
            labels=("fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess"),
        ),
    )
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    with pytest.raises(mod.CorpusRefreshError, match="dropped columns"):
        mod.RebuildRawLabelSidecarTask().run(ctx)
    _assert_prior_kept(ctx, served)


def test_rawlabel_rejects_ticker_coverage_drop(tmp_path, monkeypatch) -> None:
    _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(FRESH_DATE, n_rows=120, n_tickers=1))
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100, n_tickers=3)

    with pytest.raises(mod.CorpusRefreshError, match="coverage"):
        mod.RebuildRawLabelSidecarTask().run(ctx)
    _assert_prior_kept(ctx, served)


def test_rawlabel_regression_proceeds_when_fail_disabled(tmp_path, monkeypatch) -> None:
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(
        tmp_path,
        rawlabel_builder_fn=_builder(PRIOR_DATE),
        swap_fail_on_regression=False,
    )
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    assert mod.RebuildRawLabelSidecarTask().run(ctx) is True  # warn + keep prior
    _assert_prior_kept(ctx, served)
    assert any("RAWLABEL-REBUILD REJECTED" in t for t, *_ in posted)


def test_rawlabel_swap_interruption_preserves_served_sidecar(tmp_path, monkeypatch) -> None:
    posted = _capture_ntfy(monkeypatch)
    ctx = _ctx(tmp_path, rawlabel_builder_fn=_builder(FRESH_DATE))
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    def boom(staging, corpus):
        raise OSError("disk detached mid-replace")

    monkeypatch.setattr(mod, "_atomic_replace_corpus", boom)
    with pytest.raises(mod.CorpusRefreshError, match="swap failed"):
        mod.RebuildRawLabelSidecarTask().run(ctx)
    # The served sidecar is still the intact prior — never moved out of the way.
    assert pd.to_datetime(pd.read_parquet(served)["date"]).max().date() == PRIOR_DATE
    assert any("RAWLABEL-SWAP FAILED" in t for t, *_ in posted)


# ─────────────────────────── no-op modes ─────────────────────────────────────


def test_rawlabel_dry_run_does_not_build_or_swap(tmp_path) -> None:
    calls: list = []
    ctx = _ctx(tmp_path, dry_run=True, rawlabel_builder_fn=_builder(FRESH_DATE, calls=calls))
    served = ctx.resolved_rawlabel_path
    _sidecar_parquet(served, max_date=PRIOR_DATE, n_rows=100)

    assert mod.RebuildRawLabelSidecarTask().run(ctx) is True
    assert calls == []
    assert ctx.rawlabel_swap_report["swapped"] is False
    assert pd.to_datetime(pd.read_parquet(served)["date"]).max().date() == PRIOR_DATE


def test_rawlabel_disabled_skips(tmp_path) -> None:
    calls: list = []
    ctx = _ctx(
        tmp_path, rebuild_rawlabel=False, rawlabel_builder_fn=_builder(FRESH_DATE, calls=calls)
    )
    assert mod.RebuildRawLabelSidecarTask().run(ctx) is True
    assert calls == []
    assert ctx.rawlabel_swap_report["swapped"] is False


# ─────────────────────────── default builder seam ────────────────────────────


def test_default_rawlabel_builder_uses_the_committed_base_data_recipe(
    tmp_path, monkeypatch
) -> None:
    import types

    calls: list = []

    def fake_build(fund_panel_path, ohlcv_dir, output_path, **kw):
        calls.append((Path(fund_panel_path), Path(ohlcv_dir), Path(output_path)))
        _sidecar_parquet(Path(output_path), max_date=FRESH_DATE, n_rows=10)
        return {"n_rows": 10, "max_date": FRESH_DATE.isoformat()}

    module = types.ModuleType("renquant_base_data.rawlabel_sidecar")
    module.build_rawlabel_sidecar = fake_build
    pkg = types.ModuleType("renquant_base_data")
    pkg.rawlabel_sidecar = module
    monkeypatch.setitem(sys.modules, "renquant_base_data", pkg)
    monkeypatch.setitem(sys.modules, "renquant_base_data.rawlabel_sidecar", module)

    panel = tmp_path / "data" / "alpha158_291_fundamental_dataset.parquet"
    staging = tmp_path / "data" / "rawlabel.staging"
    ctx = _ctx(tmp_path, fund_panel_path=panel)

    mod._default_build_rawlabel(ctx, staging)
    # The committed recipe gets the SAME prod fund panel the corpus derives
    # from, plus the freshly-refreshed OHLCV dir, and builds to staging only.
    assert calls == [(panel, tmp_path / "data" / "ohlcv", staging)]
    assert staging.exists()


def test_default_rawlabel_builder_fails_closed_when_recipe_unresolvable(
    tmp_path, monkeypatch
) -> None:
    # A base-data pin that predates the committed recipe: the import raises and
    # the builder must fail CLOSED — never silently leave the sidecar frozen.
    monkeypatch.setitem(sys.modules, "renquant_base_data", None)
    monkeypatch.setitem(sys.modules, "renquant_base_data.rawlabel_sidecar", None)
    ctx = _ctx(tmp_path)

    with pytest.raises(mod.CorpusRefreshError, match="rawlabel-sidecar recipe unresolvable"):
        mod._default_build_rawlabel(ctx, tmp_path / "rawlabel.staging")
    assert not (tmp_path / "rawlabel.staging").exists()


# ─────────────────────────── CLI wiring ──────────────────────────────────────


def test_cli_wires_rawlabel_paths_and_toggle() -> None:
    args = mod.parse_args(
        [
            "--rawlabel-path", "/x/rawlabel.parquet",
            "--rawlabel-staging-path", "/x/rawlabel.staging",
            "--no-rebuild-rawlabel",
        ]
    )
    assert args.rawlabel_path == Path("/x/rawlabel.parquet")
    assert args.rawlabel_staging_path == Path("/x/rawlabel.staging")
    assert args.rebuild_rawlabel is False
    # Defaults: the rawlabel stage is ON — one weekly run refreshes both.
    assert mod.parse_args([]).rebuild_rawlabel is True
