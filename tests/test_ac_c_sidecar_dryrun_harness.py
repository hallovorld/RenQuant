"""AC-C tests: the Saturday-chain dry-run harness proves the weekly deadlock
clears against the migrated canonical candidate, using the REAL refresh guard.

All fixtures are temp/sandbox files; the harness is asserted to REFUSE the live
served path. The injected fixture builder keeps the suite green regardless of the
on-pin ``renquant_base_data`` revision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.ac_c_sidecar_dryrun_harness as harness  # noqa: E402

SENTIMENT = ("sentiment_pos_share", "mean_sentiment", "n_articles_log")
RAW_LABEL = "fwd_60d_excess_raw"


def canon_columns(n: int = 179) -> list:
    fixed_tail = ["split_label", *SENTIMENT, RAW_LABEL]
    n_features = n - 2 - len(fixed_tail)
    features = [f"f{i:03d}" for i in range(n_features)]
    return ["ticker", "date", *features, *fixed_tail]


def truth_frame(cols: list, n_tickers: int = 4, n_dates: int = 30, seed: int = 3):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2026-01-02", periods=n_dates)
    tickers = [f"TIC{i}" for i in range(n_tickers)]
    base = pd.MultiIndex.from_product(
        [tickers, dates], names=["ticker", "date"]
    ).to_frame(index=False)
    n = len(base)
    data = {}
    for c in cols:
        if c == "ticker":
            data[c] = base["ticker"].astype("string")
        elif c == "date":
            data[c] = base["date"]
        elif c == "split_label":
            data[c] = pd.array(["train"] * n, dtype="string")
        else:
            data[c] = rng.normal(size=n)
    return pd.DataFrame(data)[cols]


def write_parquet(df, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def fixture_builder(out_df):
    def build(fund_panel, ohlcv_dir, output_path) -> dict:
        write_parquet(out_df, Path(output_path))
        return {"n_rows": int(len(out_df))}

    return build


def _inputs(tmp_path: Path):
    fp = write_parquet(pd.DataFrame({"x": [1]}), tmp_path / "fund.parquet")
    od = tmp_path / "ohlcv"
    od.mkdir(parents=True, exist_ok=True)
    return fp, od


# ─────────────────────────── unit: real guard ──────────────────────────────


def test_guard_schema_reasons_empty_on_schema_parity(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    prior = write_parquet(truth, tmp_path / "prior.parquet")
    staged = write_parquet(truth, tmp_path / "staged.parquet")
    reasons = harness.guard_schema_reasons(prior, staged, require_date_advance=False)
    assert reasons == []
    assert harness.dropped_columns_fired(reasons) is False


def test_guard_schema_reasons_fires_on_dropped_sentiment(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    prior = write_parquet(truth, tmp_path / "prior.parquet")  # 179, has sentiment
    staged_df = truth[[c for c in cols if c not in SENTIMENT]]  # 176, sentiment-free
    staged = write_parquet(staged_df, tmp_path / "staged.parquet")
    reasons = harness.guard_schema_reasons(prior, staged, require_date_advance=False)
    assert harness.dropped_columns_fired(reasons) is True
    assert any("sentiment" in str(r) for r in reasons)


# ─────────────────────────── retrain-prep admission ────────────────────────


def test_retrain_prep_admissible_on_canonical(tmp_path):
    cols = canon_columns(179)
    cand = write_parquet(truth_frame(cols), tmp_path / "cand.parquet")
    prep = harness.retrain_prep_admissible(cand, cols)
    assert prep["n_cols"] == 179
    assert prep["n_labeled_rows"] > 0
    assert prep["n_extension_rows"] == 0
    assert prep["sentiment_present"] is True


def test_retrain_prep_rejects_sentiment_free(tmp_path):
    cols = canon_columns(179)
    bad = truth_frame(cols)[[c for c in cols if c not in SENTIMENT]]
    cand = write_parquet(bad, tmp_path / "cand.parquet")
    with pytest.raises(harness.AcCHarnessError, match="canonical contract|sentiment"):
        harness.retrain_prep_admissible(cand, cols)


# ─────────────────────────── full AC-C dry-run ─────────────────────────────


def test_ac_c_dryrun_passes_on_migrated_candidate(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    cand = write_parquet(truth, tmp_path / "candidate.parquet")
    fp, od = _inputs(tmp_path)
    report = harness.run_ac_c_dryrun(
        cand,
        fp,
        od,
        build_fn=fixture_builder(truth),  # staged build == canonical 179
        canon_columns=cols,
        sandbox_dir=tmp_path / "sandbox",
    )
    assert report["ok"] is True
    assert report["dropped_columns_fired"] is False
    assert report["guard_reasons"] == []
    assert report["retrain_prep"]["n_labeled_rows"] > 0


def test_ac_c_dryrun_detects_unresolved_deadlock(tmp_path):
    """Negative control: if the staged build still drops sentiment (i.e. the
    canonical builder was NOT deployed), the harness catches the SAME
    'dropped columns' failure via the real guard."""
    cols = canon_columns(179)
    truth = truth_frame(cols)
    cand = write_parquet(truth, tmp_path / "candidate.parquet")
    staged_sentiment_free = truth[[c for c in cols if c not in SENTIMENT]]
    fp, od = _inputs(tmp_path)
    with pytest.raises(harness.AcCHarnessError, match="dropped columns|deadlock"):
        harness.run_ac_c_dryrun(
            cand,
            fp,
            od,
            build_fn=fixture_builder(staged_sentiment_free),
            canon_columns=cols,
            sandbox_dir=tmp_path / "sandbox",
        )


def test_ac_c_dryrun_refuses_live_served_path(tmp_path, monkeypatch):
    cols = canon_columns(179)
    live = tmp_path / "live_served.parquet"
    write_parquet(truth_frame(cols), live)
    monkeypatch.setattr(harness, "LIVE_SERVED_PATH", live)
    fp, od = _inputs(tmp_path)
    with pytest.raises(harness.AcCHarnessError, match="LIVE served"):
        harness.run_ac_c_dryrun(
            live, fp, od, build_fn=fixture_builder(truth_frame(cols)), canon_columns=cols,
            sandbox_dir=tmp_path / "sandbox",
        )


def test_ac_c_dryrun_missing_candidate(tmp_path):
    cols = canon_columns(179)
    fp, od = _inputs(tmp_path)
    with pytest.raises(harness.AcCHarnessError, match="not found"):
        harness.run_ac_c_dryrun(
            tmp_path / "nope.parquet", fp, od,
            build_fn=fixture_builder(truth_frame(cols)), canon_columns=cols,
        )
