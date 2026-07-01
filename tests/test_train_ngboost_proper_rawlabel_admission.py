"""Integration tests for scripts/train_ngboost_proper.py's σ-head _rawlabel
admission gate.

This is the CONSUMER-side enforcement companion to renquant-orchestrator PR
#218 (RefreshSigmaHeadRawLabelTask + assert_rawlabel_admissible). Codex's
review on that PR flagged that defining an admissibility helper in the
orchestrator repo does not, by itself, enforce anything: the real σ-head
training entrypoint has to refuse to consume an inadmissible corpus. That
entrypoint is THIS script (--panel-path defaults to
data/alpha158_291_fundamental_dataset_rawlabel.parquet, the exact corpus the
orchestrator task refreshes; its output artifact is what strategy_config.json
wires into production NGBoost scoring — see backtesting/renquant_104/
artifacts/prod/ngboost-head.alpha158_fund.json, trained by this script).

These tests invoke the ACTUAL main() entrypoint (not a mock) via the same
importlib.spec_from_file_location pattern already used by
tests/test_check_retrain_triggers.py for other scripts/*.py modules (scripts/
is not an importable package), and assert:

  - an INVALID-receipt corpus refuses BEFORE any panel/artifact read, NGBoost
    fit, or artifact write;
  - a corpus with no provenance stamp fails closed;
  - a missing corpus fails closed;
  - a corpus whose provenance source_panel_sha256 no longer matches the
    CURRENT source panel on disk fails closed (drift with no receipt);
  - a corpus whose provenance horizon doesn't match the script's HORIZON
    fails closed;
  - a fully-validated, matching-digest/horizon corpus is ADMITTED and trains
    end-to-end: NGBoost fits and a real artifact is written to disk.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FEATURE_COLS = ["f1", "f2", "f3"]


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "_train_ngboost_proper_admission_test",
        REPO_ROOT / "scripts" / "train_ngboost_proper.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def _build_source_panel(path: Path, *, n_dates: int, n_tickers: int) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    tickers = [f"T{i:02d}" for i in range(n_tickers)]
    rows = [
        {"ticker": t, "date": d, **{f: float(rng.normal()) for f in FEATURE_COLS}}
        for d in dates
        for t in tickers
    ]
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return df


def _build_rawlabel_panel(source: pd.DataFrame, path: Path) -> None:
    rng = np.random.default_rng(7)
    df = source.copy()
    df["fwd_60d_excess_raw"] = rng.normal(scale=0.05, size=len(df))
    df.to_parquet(path, index=False)


def _write_provenance(
    rawlabel_path: Path, *, horizon: int, source_panel_sha256: str, n_rows: int, n_tickers: int
) -> None:
    prov = rawlabel_path.with_name(rawlabel_path.name + ".provenance.json")
    prov.write_text(json.dumps({
        "n_rows": n_rows,
        "n_tickers": n_tickers,
        "finite_fraction": 1.0,
        "horizon": horizon,
        "source_panel_sha256": source_panel_sha256,
        "source_panel_frontier": "2024-01-01",
        "rawlabel": str(rawlabel_path),
        "built_at": "2026-07-01T00:00:00Z",
    }, indent=2))


def _write_invalid_receipt(rawlabel_path: Path, *, reason: str) -> None:
    receipt = rawlabel_path.with_name(rawlabel_path.name + ".INVALID.json")
    receipt.write_text(json.dumps({
        "rawlabel": str(rawlabel_path),
        "panel": "unused",
        "horizon": 60,
        "reason": reason,
        "invalidated_at": "2026-07-01T00:00:00Z",
    }, indent=2))


def _panel_artifact(tmp_path: Path, *, exists: bool = True) -> Path:
    art = tmp_path / "panel-ltr.json"
    if exists:
        art.write_text(json.dumps({"feature_cols": FEATURE_COLS, "config_fingerprint": "sha256:test"}))
    return art


def _run(tmp_path: Path, *, panel_path: Path, source_panel_path: Path, panel_artifact: Path | None = None, extra_args=None):
    mod = _load_module()
    out_path = tmp_path / "out" / "ngb-head.json"
    argv = [
        "--panel-path", str(panel_path),
        "--panel-artifact", str(panel_artifact or _panel_artifact(tmp_path)),
        "--source-panel-path", str(source_panel_path),
        "--output-path", str(out_path),
        *(extra_args or []),
    ]
    rc = mod.main(argv)
    return rc, out_path


def test_invalid_receipt_refuses_before_any_read(tmp_path):
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=6,
    )
    _write_invalid_receipt(rawlabel_path, reason="empty-output-as-failure")

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_invalid_receipt_never_reaches_panel_artifact_read(tmp_path):
    """Proves the gate runs strictly BEFORE --panel-artifact is opened: point
    it at a nonexistent file. If admission ran after opening the artifact,
    this would raise an uncaught FileNotFoundError instead of a controlled
    refusal — i.e. this is the strongest available proof (short of profiling)
    that no downstream read happens on an inadmissible corpus."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_invalid_receipt(rawlabel_path, reason="test-reason")

    rc, out_path = _run(
        tmp_path, panel_path=rawlabel_path, source_panel_path=source_path,
        panel_artifact=tmp_path / "does-not-exist.json",
    )

    assert rc == 3
    assert not out_path.exists()


def test_missing_provenance_fails_closed(tmp_path):
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    # No provenance stamp at all — never validated by the refresh task.

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_missing_rawlabel_corpus_fails_closed(tmp_path):
    source_path = tmp_path / "source.parquet"
    _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"  # never created

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_digest_mismatch_fails_closed(tmp_path):
    """The corpus was built from a DIFFERENT source panel than the one live
    on disk now (e.g. the fund panel advanced after the corpus was stamped,
    with no failure recorded) — must fail closed even with no INVALID.json."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256="sha256:" + "0" * 64,  # deliberately wrong
        n_rows=len(source), n_tickers=6,
    )

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_horizon_mismatch_fails_closed(tmp_path):
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=30,  # script's HORIZON constant is 60
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=6,
    )

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_allow_unadmitted_rawlabel_is_opt_in_and_bypasses_the_gate(tmp_path):
    """The research-only escape hatch defaults OFF; explicitly setting it
    skips admission entirely — execution proceeds straight past where the
    gate would have refused and instead crashes trying to open the
    (deliberately nonexistent) --panel-artifact, proving the gate itself
    (not some other check) is what was bypassed."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_invalid_receipt(rawlabel_path, reason="test-reason")

    mod = _load_module()
    default_args = mod.parse_args([
        "--panel-path", str(rawlabel_path),
        "--source-panel-path", str(source_path),
    ])
    assert default_args.allow_unadmitted_rawlabel is False

    out_path = tmp_path / "out" / "ngb-head.json"
    argv = [
        "--panel-path", str(rawlabel_path),
        "--panel-artifact", str(tmp_path / "does-not-exist.json"),
        "--source-panel-path", str(source_path),
        "--output-path", str(out_path),
        "--allow-unadmitted-rawlabel",
    ]
    with pytest.raises(FileNotFoundError):
        mod.main(argv)

    assert not out_path.exists()


@pytest.mark.slow
def test_valid_matching_receipt_admits_and_trains_end_to_end(tmp_path):
    """A fully-validated, matching-digest/horizon corpus is admitted and the
    REAL entrypoint proceeds all the way through: NGBoost fits and an
    artifact is written to disk."""
    source_path = tmp_path / "source.parquet"
    # n_dates large enough that the HORIZON=60 purge still leaves both a
    # non-trivial train and val split (val_cut_idx - HORIZON > 0).
    source = _build_source_panel(source_path, n_dates=120, n_tickers=10)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=10,
    )

    rc, out_path = _run(
        tmp_path, panel_path=rawlabel_path, source_panel_path=source_path,
        extra_args=["--seeds", "42", "--allow-save-without-baseline"],
    )

    assert rc == 0
    assert out_path.exists()
    artifact = json.loads(out_path.read_text())
    assert artifact["kind"] == "ngboost_head"
    assert artifact["feature_cols"] == FEATURE_COLS
