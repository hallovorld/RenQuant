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
  - a corpus TAMPERED/REPLACED after validation (bytes changed, sidecar left
    as-is, source_panel_sha256 and horizon both still matching) fails closed
    on the rawlabel_sha256 check — this is the specific gap the coordinated
    renquant-orchestrator #218 + RenQuant #427 fix closes: previously only
    the INPUT (source panel) was digest-bound, never the OUTPUT (the corpus
    itself);
  - a provenance sidecar missing rawlabel_sha256 or schema_version entirely
    (a pre-fix producer) fails closed rather than being silently admitted;
  - a provenance sidecar declaring a schema_version this consumer doesn't
    recognize fails closed;
  - a fully-validated, matching-digest/horizon/schema corpus is ADMITTED and
    trains end-to-end: NGBoost fits and a real artifact is written to disk.
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
    rawlabel_path: Path,
    *,
    horizon: int,
    source_panel_sha256: str,
    n_rows: int,
    n_tickers: int,
    rawlabel_sha256: str | None = None,
    schema_version: int | None = 1,
) -> None:
    """Write a provenance sidecar. By default ``rawlabel_sha256`` is computed
    from the CURRENT on-disk ``rawlabel_path`` bytes (i.e. a "correctly
    matching" sidecar, matching what renquant-orchestrator's
    RefreshSigmaHeadRawLabelTask actually stamps post-swap) and
    ``schema_version`` defaults to the current schema (1) — so every existing
    call site that only cares about some OTHER field (horizon, source-panel
    digest, ...) gets a valid rawlabel digest "for free" and isolates the
    failure it's actually testing. Pass ``rawlabel_sha256=`` explicitly to
    simulate a tampered/replaced corpus, or ``schema_version=None`` to omit
    the key entirely (simulating a pre-schema-versioning producer)."""
    prov = rawlabel_path.with_name(rawlabel_path.name + ".provenance.json")
    payload = {
        "n_rows": n_rows,
        "n_tickers": n_tickers,
        "finite_fraction": 1.0,
        "horizon": horizon,
        "source_panel_sha256": source_panel_sha256,
        "source_panel_frontier": "2024-01-01",
        "rawlabel": str(rawlabel_path),
        "built_at": "2026-07-01T00:00:00Z",
        "rawlabel_sha256": (
            rawlabel_sha256 if rawlabel_sha256 is not None else _sha256_file(rawlabel_path)
        ),
    }
    if schema_version is not None:
        payload["schema_version"] = schema_version
    prov.write_text(json.dumps(payload, indent=2))


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


# ─── rawlabel_sha256 / schema_version — closes the Codex #218/#427 gap ──────
# The producer's success provenance recorded source_panel_sha256 (the INPUT
# digest) but never a digest of the VALIDATED RAWLABEL CORPUS itself (the
# OUTPUT this admission gate is meant to protect). A later replacement/edit
# of rawlabel.parquet with the sidecar left intact — and the source panel
# UNCHANGED — was indistinguishable from the originally-validated bytes and
# would have been wrongly admitted by source_panel_sha256 + horizon alone.
# These tests prove that gap is now closed.


def test_tampered_rawlabel_corpus_fails_closed_even_with_matching_source_and_horizon(tmp_path):
    """THE core regression from the review: bytes of the corpus itself are
    changed AFTER validation (replaced/edited/corrupted), the sidecar is left
    completely untouched, source_panel_sha256 and horizon both still match
    the live source panel — only rawlabel_sha256 can catch this."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    # Provenance stamped against the ORIGINAL (validated) corpus bytes —
    # rawlabel_sha256 defaults to the digest of rawlabel_path AT THIS POINT.
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=6,
    )

    # Out-of-band replacement of the corpus file — sidecar (and source panel)
    # left untouched, exactly the scenario the review flagged.
    rawlabel_path.write_bytes(b"TAMPERED-BYTES-NEVER-VALIDATED-BY-THE-REFRESH-TASK")

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_missing_rawlabel_sha256_fails_closed(tmp_path):
    """A provenance sidecar predating the rawlabel_sha256 field (an older
    producer, or a hand-crafted receipt) must not be trusted merely because
    source_panel_sha256 + horizon happen to match — the OUTPUT was never
    bound to a digest at all, so it fails closed rather than being silently
    treated as admitted."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    prov = rawlabel_path.with_name(rawlabel_path.name + ".provenance.json")
    prov.write_text(json.dumps({
        "n_rows": len(source),
        "n_tickers": 6,
        "finite_fraction": 1.0,
        "horizon": 60,
        "source_panel_sha256": _sha256_file(source_path),
        "source_panel_frontier": "2024-01-01",
        "rawlabel": str(rawlabel_path),
        "built_at": "2026-07-01T00:00:00Z",
        "schema_version": 1,
        # rawlabel_sha256 deliberately omitted.
    }, indent=2))

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_schema_version_mismatch_fails_closed(tmp_path):
    """A provenance sidecar declaring an older/different schema_version than
    this consumer understands must not be trusted, even if every field this
    consumer happens to look for is present and matching — schema drift is a
    contract violation independent of any single field's content."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=6,
        schema_version=0,  # not the schema this consumer understands (1)
    )

    rc, out_path = _run(tmp_path, panel_path=rawlabel_path, source_panel_path=source_path)

    assert rc == 3
    assert not out_path.exists()


def test_missing_schema_version_fails_closed(tmp_path):
    """A pre-schema-versioning provenance sidecar (predates this field
    entirely) is treated the same as a recognized-wrong version: untrusted,
    fail closed."""
    source_path = tmp_path / "source.parquet"
    source = _build_source_panel(source_path, n_dates=10, n_tickers=6)
    rawlabel_path = tmp_path / "rawlabel.parquet"
    _build_rawlabel_panel(source, rawlabel_path)
    _write_provenance(
        rawlabel_path, horizon=60,
        source_panel_sha256=_sha256_file(source_path), n_rows=len(source), n_tickers=6,
        schema_version=None,  # omit the key entirely
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
