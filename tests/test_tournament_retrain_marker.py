"""Regression guard for the per-ticker tournament retrain completion marker.

Source incident: Codex CHANGES_REQUESTED on PR #420 (2026-06-30). The first cut
of ``scripts/weekly_tournament_retrain.sh`` certified completion from
process-derived signals — it counted pre-existing ``models/*`` dirs and stamped
``trained_date`` with the wall clock — so a PARTIAL or NO-OP retrain that exits 0
could publish a globally fresh-looking marker.

``scripts/tournament_retrain_marker.py`` re-derives completion from the artifacts
themselves. These tests pin the three failure modes Codex named:

  * stale pre-existing dirs not rewritten  → NOT fresh / NOT certified;
  * one expected ticker not rewritten      → partial/fail (never certified);
  * train exit-0 but partial output        → CLI writes NO marker, exits non-zero.

Plus the positive path (all rewritten → certified success with min/max cutoff)
and the coverage-policy / data-cutoff semantics.

Design: pure filesystem fixtures under ``tmp_path``; per-ticker artifact mtimes
are set explicitly with ``os.utime`` so "rewritten this invocation" is
deterministic and does not depend on wall-clock timing.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "tournament_retrain_marker.py"

LAUNCH = 1_000_000.0  # arbitrary fixed "launch" epoch for deterministic tests


def _load_module():
    spec = importlib.util.spec_from_file_location("tournament_retrain_marker_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _write_ticker(
    models_dir: Path,
    ticker: str,
    *,
    mtime: float,
    live_train_end: str | None = "2026-06-25",
    trained_date: str | None = "2026-06-30",
    extra: dict | None = None,
) -> Path:
    """Create ``models/<T>/<T>-policy-metadata.json`` and force its mtime."""
    sym_dir = models_dir / ticker
    sym_dir.mkdir(parents=True, exist_ok=True)
    meta = sym_dir / f"{ticker}-policy-metadata.json"
    payload: dict = {"model_name": ticker}
    if trained_date is not None:
        payload["trained_date"] = trained_date
    if live_train_end is not None:
        payload["live_train_end"] = live_train_end
    if extra:
        payload.update(extra)
    meta.write_text(json.dumps(payload, indent=2))
    os.utime(meta, (mtime, mtime))
    return meta


# ---------------------------------------------------------------------------
# build_marker_evidence — the pure core
# ---------------------------------------------------------------------------
def test_all_rewritten_certifies_success_with_cutoff_range(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    _write_ticker(models, "BBB", mtime=LAUNCH + 20, live_train_end="2026-06-25")
    _write_ticker(models, "CCC", mtime=LAUNCH + 30, live_train_end="2026-06-22")

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH)

    assert ev["certified"] is True
    assert ev["status"] == "success"
    assert ev["coverage"] == 1.0
    assert ev["succeeded_count"] == 3
    assert ev["min_data_cutoff"] == "2026-06-20"
    assert ev["max_data_cutoff"] == "2026-06-25"
    # artifact digest recorded per ticker (provenance)
    assert all(ev["per_ticker"][t]["digest"] for t in ("AAA", "BBB", "CCC"))
    assert ev["per_ticker"]["AAA"]["cutoff_source"] == "live_train_end"


def test_stale_preexisting_dirs_not_rewritten_are_not_fresh(tmp_path):
    """All expected tickers exist but predate launch → NOT certified (the bug)."""
    models = tmp_path / "models"
    for t in ("AAA", "BBB", "CCC"):
        _write_ticker(models, t, mtime=LAUNCH - 500)  # older than launch

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH)

    assert ev["certified"] is False
    assert ev["status"] == "failed"
    assert ev["succeeded_count"] == 0
    assert ev["stale_count"] == 3
    assert set(ev["sets"]["stale"]) == {"AAA", "BBB", "CCC"}
    assert ev["no_stale_masquerade"] is False
    # a stale, pre-existing population contributes NO fresh data cutoff
    assert ev["min_data_cutoff"] is None
    assert ev["max_data_cutoff"] is None


def test_one_ticker_not_rewritten_is_partial_fail(tmp_path):
    """Two fresh, one stale → zero-stale rule blocks certification even at 2/3."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    _write_ticker(models, "CCC", mtime=LAUNCH - 10)  # not rewritten this run

    # Even a permissive coverage floor cannot certify while a stale dir exists.
    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH, min_coverage=0.5)

    assert ev["certified"] is False
    assert ev["status"] == "failed"
    assert ev["stale_count"] == 1
    assert ev["sets"]["stale"] == ["CCC"]
    assert "CCC" in ev["sets"]["failed"]


def test_orphan_dirs_outside_watchlist_do_not_inflate_coverage(tmp_path):
    """Pre-existing orphan dirs NOT in the frozen watchlist are ignored entirely."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    # 5 stale orphans from prior watchlists — these must not count toward anything.
    for orphan in ("OLD1", "OLD2", "OLD3", "OLD4", "OLD5"):
        _write_ticker(models, orphan, mtime=LAUNCH - 9999)

    ev = mod.build_marker_evidence(models, ["AAA", "BBB"], LAUNCH)

    assert ev["certified"] is True
    assert ev["status"] == "success"
    assert ev["expected_count"] == 2
    assert ev["succeeded_count"] == 2
    assert "OLD1" not in json.dumps(ev["sets"])  # orphans absent from every set


def test_missing_tolerated_under_floor_but_not_at_full_coverage(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    _write_ticker(models, "CCC", mtime=LAUNCH + 10)
    _write_ticker(models, "DDD", mtime=LAUNCH + 10)
    _write_ticker(models, "EEE", mtime=LAUNCH + 10)
    _write_ticker(models, "FFF", mtime=LAUNCH + 10)
    _write_ticker(models, "GGG", mtime=LAUNCH + 10)
    _write_ticker(models, "HHH", mtime=LAUNCH + 10)
    _write_ticker(models, "III", mtime=LAUNCH + 10)
    # JJJ has no dir → missing (e.g. an ETF/benchmark the tournament never trains)
    expected = [f"{c}{c}{c}" for c in "ABCDEFGHIJ"]  # AAA..JJJ (10 names)

    # 9/10 = 0.9 → certified partial under a 0.9 floor
    ev = mod.build_marker_evidence(models, expected, LAUNCH, min_coverage=0.9)
    assert ev["certified"] is True
    assert ev["status"] == "partial"
    assert ev["missing_count"] == 1
    assert ev["sets"]["missing"] == ["JJJ"]

    # same artifacts, strict floor → NOT certified
    ev_strict = mod.build_marker_evidence(models, expected, LAUNCH, min_coverage=1.0)
    assert ev_strict["certified"] is False
    assert ev_strict["status"] == "failed"


def test_rewritten_but_unparseable_blocks_certification(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    # rewritten (fresh mtime) but corrupt JSON
    bad_dir = models / "BBB"
    bad_dir.mkdir(parents=True)
    bad = bad_dir / "BBB-policy-metadata.json"
    bad.write_text("{ this is not json")
    os.utime(bad, (LAUNCH + 10, LAUNCH + 10))

    ev = mod.build_marker_evidence(models, ["AAA", "BBB"], LAUNCH, min_coverage=0.5)
    assert ev["certified"] is False
    assert ev["unparseable_count"] == 1
    assert ev["sets"]["unparseable"] == ["BBB"]


def test_cutoff_falls_back_to_trained_date_when_no_live_train_end(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end=None, trained_date="2026-06-28")
    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH)
    assert ev["certified"] is True
    assert ev["per_ticker"]["AAA"]["cutoff_source"] == "trained_date"
    assert ev["min_data_cutoff"] == "2026-06-28"


def test_empty_expected_universe_refused(tmp_path):
    with pytest.raises(ValueError):
        mod.build_marker_evidence(tmp_path, [], LAUNCH)


def test_bad_min_coverage_refused(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            mod.build_marker_evidence(models, ["AAA"], LAUNCH, min_coverage=bad)


# ---------------------------------------------------------------------------
# CLI (main) — the artifact the bash wrapper actually calls
# ---------------------------------------------------------------------------
def _run_cli(tmp_path, models, expected, *, min_coverage, extra_args=None):
    wl = tmp_path / "expected_watchlist.json"
    wl.write_text(json.dumps({"watchlist": expected}))
    marker = tmp_path / "marker.json"
    args = [
        sys.executable, str(SCRIPT),
        "--models-dir", str(models),
        "--watchlist", str(wl),
        "--launch-epoch", str(LAUNCH),
        "--run-id", "20260630T060000Z",
        "--marker", str(marker),
        "--min-coverage", str(min_coverage),
        "--exit-code", "0",
        "--date", "2026-06-30",
        "--completed-at", "2026-06-30T06:10:00Z",
    ] + (extra_args or [])
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc, marker


def test_cli_exit0_partial_is_not_certified_and_writes_no_marker(tmp_path):
    """train exit-0 but a partial/no-op population → CLI must NOT stamp a marker."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH - 10)  # stale — training did not rewrite it

    proc, marker = _run_cli(tmp_path, models, ["AAA", "BBB"], min_coverage=0.5)

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not marker.exists(), "a non-certified run must leave NO fresh marker"
    assert "NOT CERTIFIED" in proc.stderr


def test_cli_certified_stamps_artifact_derived_marker(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    _write_ticker(models, "BBB", mtime=LAUNCH + 20, live_train_end="2026-06-24")

    proc, marker = _run_cli(tmp_path, models, ["AAA", "BBB"], min_coverage=1.0)

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["status"] == "success"
    assert payload["certified"] is True
    assert payload["run_id"] == "20260630T060000Z"
    # trained_date is artifact-derived (min cutoff), NOT the wall clock
    assert payload["trained_date"] == "2026-06-20"
    assert payload["trained_date_source"] == "min_data_cutoff"
    assert payload["wall_clock_date"] == "2026-06-30"
    assert payload["min_data_cutoff"] == "2026-06-20"
    assert payload["max_data_cutoff"] == "2026-06-24"
    assert payload["scope"] == "cadence_completion_only"


def test_cli_all_stale_fails_and_preserves_prior_marker(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH - 100)
    _write_ticker(models, "BBB", mtime=LAUNCH - 100)

    # a prior (real) marker exists — a failed run must not overwrite it
    wl = tmp_path / "expected_watchlist.json"
    wl.write_text(json.dumps({"watchlist": ["AAA", "BBB"]}))
    marker = tmp_path / "marker.json"
    marker.write_text(json.dumps({"status": "success", "trained_date": "2026-06-01"}))
    args = [
        sys.executable, str(SCRIPT),
        "--models-dir", str(models), "--watchlist", str(wl),
        "--launch-epoch", str(LAUNCH), "--run-id", "rid", "--marker", str(marker),
        "--min-coverage", "0.5",
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode != 0
    # untouched prior marker
    assert json.loads(marker.read_text())["trained_date"] == "2026-06-01"
