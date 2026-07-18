"""AC-D tests for the rawlabel-sidecar canonical migration runbook.

Every test runs against TEMP/SANDBOX files — never the live served sidecar — and
injects a fixture 179-col builder, so the suite is green regardless of whether
the on-pin ``renquant_base_data`` is the stale 176-col revision or the canonical
179-col one.
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

import scripts.migrate_rawlabel_sidecar_to_canonical as mig  # noqa: E402


# ─────────────────────────── fixtures ───────────────────────────────────────

SENTIMENT = ("sentiment_pos_share", "mean_sentiment", "n_articles_log")
RAW_LABEL = "fwd_60d_excess_raw"


def canon_columns(n: int = 179) -> list:
    """A synthetic canonical contract of exactly ``n`` columns: keys, feature
    block, split_label, the three sentiment columns, and the raw label LAST —
    the structure the real 179-col contract has."""
    fixed_tail = ["split_label", *SENTIMENT, RAW_LABEL]
    n_features = n - 2 - len(fixed_tail)  # minus ticker, date
    features = [f"f{i:03d}" for i in range(n_features)]
    return ["ticker", "date", *features, *fixed_tail]


def truth_frame(cols: list, n_tickers: int = 4, n_dates: int = 30, seed: int = 7):
    """A canonical (zero-extension) sidecar frame with unique (ticker, date)
    keys and a finite raw label. Built column-at-once (no fragmentation)."""
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


def add_extension_rows(df, cols, n: int = 5):
    """Append ``n`` bar-frontier extension rows: NEW (ticker, date) keys whose
    every non-key column is NaN."""
    last = pd.to_datetime(df["date"]).max()
    ext_dates = pd.bdate_range(last + pd.Timedelta(days=1), periods=n)
    ext_data = {}
    for c in cols:
        if c == "ticker":
            ext_data[c] = pd.array(["TIC0"] * n, dtype="string")
        elif c == "date":
            ext_data[c] = ext_dates
        elif c == "split_label":
            ext_data[c] = pd.array([None] * n, dtype="string")
        else:
            ext_data[c] = np.full(n, np.nan)
    ext = pd.DataFrame(ext_data)[cols]
    return pd.concat([df, ext], ignore_index=True)


def fixture_builder(out_df):
    """A build_fn that ignores inputs and writes ``out_df`` (an injected
    stand-in for the canonical Stage-1 builder)."""

    def build(fund_panel, ohlcv_dir, output_path) -> dict:
        write_parquet(out_df, Path(output_path))
        return {"n_rows": int(len(out_df)), "n_columns": int(len(out_df.columns))}

    return build


def _inputs(tmp_path: Path):
    """Minimal on-disk fund panel + ohlcv fingerprint inputs (content is
    irrelevant to the injected fixture builder, but the script hashes them)."""
    fp = write_parquet(pd.DataFrame({"x": [1, 2]}), tmp_path / "fund_panel.parquet")
    od = tmp_path / "ohlcv"
    (od / "SPY").mkdir(parents=True)
    write_parquet(pd.DataFrame({"close": [1.0]}), od / "SPY" / "1d.parquet")
    return fp, od


# ─────────────────────────── digest primitives ─────────────────────────────


def test_schema_digest_sensitive_to_order():
    a = mig.schema_digest(["ticker", "date", "x"])
    b = mig.schema_digest(["date", "ticker", "x"])
    assert a != b
    assert mig.schema_digest(["ticker", "date", "x"]) == a


def test_sha256_file_streams(tmp_path):
    p = write_parquet(pd.DataFrame({"a": [1, 2, 3]}), tmp_path / "f.parquet")
    import hashlib

    assert mig.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_count_extension_rows(tmp_path):
    cols = canon_columns()
    df = truth_frame(cols)
    assert mig.count_extension_rows(df) == 0
    df2 = add_extension_rows(df, cols, n=4)
    assert mig.count_extension_rows(df2) == 4


# ─────────────────────────── builder-contract preflight ─────────────────────


def test_builder_contract_preflight_accepts_179():
    mig.builder_contract_preflight(canon_columns(179))  # no raise


def test_builder_contract_preflight_refuses_stale_176():
    # sentiment-free 176-col contract == the stale pre-amendment builder.
    stale = [c for c in canon_columns(179) if c not in SENTIMENT]
    assert len(stale) == 176
    with pytest.raises(mig.MigrationPreflightError, match="176|sentiment"):
        mig.builder_contract_preflight(stale)


def test_builder_contract_preflight_requires_raw_label_last():
    cols = canon_columns(179)
    swapped = cols[:-2] + [cols[-1], cols[-2]]  # raw label no longer last
    with pytest.raises(mig.MigrationPreflightError, match="end with"):
        mig.builder_contract_preflight(swapped)


# ─────────────────────────── dry-run integrity ──────────────────────────────


def test_dry_run_clean_179_to_179(tmp_path):
    """Current-state case: served already 179 canonical; the rebuild reproduces
    it byte-for-content. dry-run verifies + leaves the served file UNTOUCHED."""
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth, tmp_path / "served.parquet")
    served_sha_before = mig.sha256_file(served)
    fp, od = _inputs(tmp_path)

    res = mig.run_migration(
        mode="dry-run",
        served_path=served,
        fund_panel_path=fp,
        ohlcv_dir=od,
        build_fn=fixture_builder(truth),
        canon_columns=cols,
    )
    assert res["swapped"] is False
    assert res["diff"]["retained_columns_checksum_equal"] is True
    assert res["diff"]["retained_column_count"] == 179
    assert res["after"]["n_extension_rows"] == 0
    assert res["before"]["sha256"] == res["after"]["sha256"]  # identical content
    # served file untouched, no candidate left behind.
    assert mig.sha256_file(served) == served_sha_before
    assert not list(tmp_path.glob("served.parquet.candidate-*"))


def test_dry_run_176_to_179_adds_sentiment_drops_extension(tmp_path):
    """Migration case: served is 176-col (sentiment-free) WITH extension rows;
    the canonical rebuild adds sentiment + drops extension. dry-run passes; the
    176 retained columns checksum-match on the shared rows."""
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served_cols = [c for c in cols if c not in SENTIMENT]  # 176
    served_df = add_extension_rows(truth[served_cols], served_cols, n=6)
    served = write_parquet(served_df, tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)

    res = mig.run_migration(
        mode="dry-run",
        served_path=served,
        fund_panel_path=fp,
        ohlcv_dir=od,
        build_fn=fixture_builder(truth),
        canon_columns=cols,
    )
    assert res["swapped"] is False
    assert sorted(res["diff"]["added_columns"]) == sorted(SENTIMENT)
    assert res["diff"]["extension_rows_dropped"] == 6
    assert res["diff"]["retained_column_count"] == 176
    assert res["after"]["n_extension_rows"] == 0
    assert res["before"]["fund_panel_sha256"] == mig.sha256_file(fp)


def test_dry_run_rejects_tampered_retained_column(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth, tmp_path / "served.parquet")
    tampered = truth.copy()
    tampered.loc[0, "f000"] = tampered.loc[0, "f000"] + 12345.0  # change retained data
    fp, od = _inputs(tmp_path)

    with pytest.raises(mig.MigrationIntegrityError, match="retained column"):
        mig.run_migration(
            mode="dry-run",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            build_fn=fixture_builder(tampered),
            canon_columns=cols,
        )
    # fail-closed: no candidate left behind.
    assert not list(tmp_path.glob("served.parquet.candidate-*"))


def test_dry_run_rejects_extension_rows_in_candidate(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth, tmp_path / "served.parquet")
    bad = add_extension_rows(truth, cols, n=3)  # candidate carries extension rows
    fp, od = _inputs(tmp_path)

    with pytest.raises(mig.MigrationIntegrityError, match="extension row"):
        mig.run_migration(
            mode="dry-run",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            build_fn=fixture_builder(bad),
            canon_columns=cols,
        )


def test_dry_run_rejects_fabricated_rows(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth.iloc[:-5].copy(), tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)
    # candidate (truth) has 5 keys absent from the smaller served file.
    with pytest.raises(mig.MigrationIntegrityError, match="fabricate"):
        mig.run_migration(
            mode="dry-run",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            build_fn=fixture_builder(truth),
            canon_columns=cols,
        )


def test_dry_run_rejects_non_canonical_schema(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth, tmp_path / "served.parquet")
    # builder emits a sentiment-free 176 output (non-canonical).
    bad = truth[[c for c in cols if c not in SENTIMENT]].copy()
    fp, od = _inputs(tmp_path)
    with pytest.raises(mig.MigrationIntegrityError, match="canonical contract in order"):
        mig.run_migration(
            mode="dry-run",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            build_fn=fixture_builder(bad),
            canon_columns=cols,
        )


def test_run_migration_refuses_stale_176_canon(tmp_path):
    cols = canon_columns(179)
    served = write_parquet(truth_frame(cols), tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)
    stale = [c for c in cols if c not in SENTIMENT]  # 176
    with pytest.raises(mig.MigrationPreflightError):
        mig.run_migration(
            mode="dry-run",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            build_fn=fixture_builder(truth_frame(cols)),
            canon_columns=stale,
        )


# ─────────────────────────── execute + rollback ─────────────────────────────


def test_execute_atomic_swap_and_containment(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served_cols = [c for c in cols if c not in SENTIMENT]
    served_df = truth[served_cols].copy()
    served = write_parquet(served_df, tmp_path / "served.parquet")
    before_sha = mig.sha256_file(served)
    fp, od = _inputs(tmp_path)
    report_out = tmp_path / "report.json"
    containment_out = tmp_path / "containment.json"

    res = mig.run_migration(
        mode="execute",
        served_path=served,
        fund_panel_path=fp,
        ohlcv_dir=od,
        build_fn=fixture_builder(truth),
        canon_columns=cols,
        report_out=report_out,
        containment_out=containment_out,
        task_ref="task-99",
        owner="hallovorld",
        restore_condition="until first green Saturday retrain",
        run_id="TESTRUN",
    )
    assert res["swapped"] is True
    # served now holds the canonical 179-col bytes.
    served_now = pd.read_parquet(served)
    assert list(served_now.columns) == cols
    assert mig.sha256_file(served) == res["after"]["sha256"]
    # backup holds the exact pre-migration bytes.
    bak = Path(res["backup_path"])
    assert bak.exists()
    assert res["backup_sha256"] == before_sha
    assert mig.sha256_file(bak) == before_sha
    # containment record: revert steps + owner + task + restore condition.
    import json

    cont = json.loads(containment_out.read_text())
    assert cont["owner"] == "hallovorld"
    assert cont["task_ref"] == "task-99"
    assert cont["restore_condition"] == "until first green Saturday retrain"
    assert cont["backup_sha256"] == before_sha
    assert any("--rollback" in step for step in cont["revert_steps"])
    assert report_out.exists()


def test_rollback_restores_exact_backed_up_digest(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served_df = truth[[c for c in cols if c not in SENTIMENT]].copy()
    served = write_parquet(served_df, tmp_path / "served.parquet")
    before_sha = mig.sha256_file(served)
    fp, od = _inputs(tmp_path)

    res = mig.run_migration(
        mode="execute",
        served_path=served,
        fund_panel_path=fp,
        ohlcv_dir=od,
        build_fn=fixture_builder(truth),
        canon_columns=cols,
        task_ref="t",
        owner="o",
        restore_condition="c",
        run_id="RB",
    )
    assert mig.sha256_file(served) != before_sha  # migrated

    rb = mig.run_migration(
        mode="rollback",
        served_path=served,
        fund_panel_path=fp,
        ohlcv_dir=od,
        backup_path=res["backup_path"],
        expected_backup_sha256=res["backup_sha256"],
    )
    assert rb["verified"] is True
    assert rb["restored_sha256"] == before_sha
    assert mig.sha256_file(served) == before_sha  # exact pre-migration bytes back


def test_rollback_refuses_on_hash_mismatch(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    served = write_parquet(truth[[c for c in cols if c not in SENTIMENT]], tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)
    res = mig.run_migration(
        mode="execute", served_path=served, fund_panel_path=fp, ohlcv_dir=od,
        build_fn=fixture_builder(truth), canon_columns=cols,
        task_ref="t", owner="o", restore_condition="c", run_id="RB2",
    )
    migrated_sha = mig.sha256_file(served)
    bak = Path(res["backup_path"])
    # tamper the backup bytes so it no longer matches the recorded digest.
    write_parquet(truth, bak)  # different content
    with pytest.raises(mig.MigrationIntegrityError, match="backup sha256"):
        mig.run_migration(
            mode="rollback",
            served_path=served,
            fund_panel_path=fp,
            ohlcv_dir=od,
            backup_path=bak,
            expected_backup_sha256=res["backup_sha256"],
        )
    # served left untouched (still the migrated bytes).
    assert mig.sha256_file(served) == migrated_sha


def test_rollback_requires_backup_and_expected_sha(tmp_path):
    served = write_parquet(truth_frame(canon_columns()), tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)
    with pytest.raises(mig.MigrationPreflightError, match="requires"):
        mig.run_migration(
            mode="rollback", served_path=served, fund_panel_path=fp, ohlcv_dir=od,
            backup_path=None, expected_backup_sha256=None,
        )


# ─────────────────────────── ordering preflight (§2 hazard) ─────────────────


def test_ordering_preflight_passes_on_canonical_file(tmp_path):
    cols = canon_columns(179)
    served = write_parquet(truth_frame(cols), tmp_path / "served.parquet")
    status = mig.ordering_preflight(served, cols)
    assert status["ok"] is True


def test_ordering_preflight_refuses_when_absent(tmp_path):
    cols = canon_columns(179)
    absent = tmp_path / "not_there.parquet"
    with pytest.raises(mig.MigrationPreflightError, match="ABSENT"):
        mig.ordering_preflight(absent, cols)


def test_ordering_preflight_refuses_non_canonical_file(tmp_path):
    cols = canon_columns(179)
    truth = truth_frame(cols)
    # present but sentiment-free (176) → not canonical.
    served = write_parquet(truth[[c for c in cols if c not in SENTIMENT]], tmp_path / "served.parquet")
    with pytest.raises(mig.MigrationPreflightError, match="not\\s+canonical|canonical contract"):
        mig.ordering_preflight(served, cols)


def test_preflight_mode_via_run_migration(tmp_path):
    cols = canon_columns(179)
    served = write_parquet(truth_frame(cols), tmp_path / "served.parquet")
    fp, od = _inputs(tmp_path)
    res = mig.run_migration(
        mode="preflight", served_path=served, fund_panel_path=fp, ohlcv_dir=od,
        build_fn=fixture_builder(truth_frame(cols)), canon_columns=cols,
    )
    assert res["ok"] is True


# ─────────────────────────── CLI guards ─────────────────────────────────────


def test_cli_execute_requires_containment_metadata(tmp_path):
    served = write_parquet(truth_frame(canon_columns()), tmp_path / "served.parquet")
    rc = mig.main(["--execute", "--served-path", str(served)])
    assert rc == 2  # refuses to execute with no containment metadata


def test_cli_requires_a_mode(tmp_path):
    with pytest.raises(SystemExit):
        mig.parse_args([])  # mutually-exclusive group is required
