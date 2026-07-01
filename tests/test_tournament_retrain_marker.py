"""Regression guard for the per-ticker tournament retrain completion marker.

Source incident: Codex CHANGES_REQUESTED on PR #420 (2026-06-30, two rounds).

Round 1 — the first cut of ``scripts/weekly_tournament_retrain.sh`` certified
completion from process-derived signals — it counted pre-existing
``models/*`` dirs and stamped ``trained_date`` with the wall clock — so a
PARTIAL or NO-OP retrain that exits 0 could publish a globally fresh-looking
marker. Fixed by making the marker artifact- and per-ticker-derived
(``mtime >= launch_epoch`` + a coverage floor).

Round 2 — Codex flagged four residual gaps in that fix, each pinned below:

  * ``mtime >= launch_epoch`` alone does not prove bytes changed — a no-op
    writer / restamp / failed run that rewrites identical bytes with a fresh
    mtime still "succeeded". Fix: a PRE-RUN baseline (digest + cutoff) must be
    captured before launch; a post-run rewrite must prove digest identity
    changed, or carry an EXPLICIT ``no_change_reason``.
  * ``evaluate_ticker`` fell back from ``live_train_end`` to ``trained_date``,
    reintroducing the wall-clock freshness spoof. Fix: fallback removed.
  * ``--exit-code`` was recorded but never enforced. Fix: certification now
    HARD-REQUIRES ``exit_code == 0`` inside :func:`build_marker_evidence`
    itself — artifact freshness can never override a failed training process.
  * The hard-coded 0.90 coverage floor was an unregistered magic number that
    could silently mask up to ~14 missing names. Fix: an explicit
    ``non_trainable`` map (ticker -> justification) of intentionally
    non-trained names (benchmark/sector/defensive ETFs); every OTHER expected
    ticker (the "trainable" set) is required at 100% coverage.

Design: pure filesystem fixtures under ``tmp_path``; per-ticker artifact
mtimes are set explicitly with ``os.utime`` so "rewritten this invocation" is
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


def _digest_of(path: Path) -> str:
    return mod._sha256(path)


# ---------------------------------------------------------------------------
# build_marker_evidence — the pure core (no baseline supplied)
# ---------------------------------------------------------------------------
def test_all_rewritten_certifies_success_with_cutoff_range(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    _write_ticker(models, "BBB", mtime=LAUNCH + 20, live_train_end="2026-06-25")
    _write_ticker(models, "CCC", mtime=LAUNCH + 30, live_train_end="2026-06-22")

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH, exit_code=0)

    assert ev["certified"] is True
    assert ev["status"] == "success"
    assert ev["trainable_coverage"] == 1.0
    assert ev["trainable_succeeded_count"] == 3
    assert ev["min_data_cutoff"] == "2026-06-20"
    assert ev["max_data_cutoff"] == "2026-06-25"
    # artifact digest recorded per ticker (provenance)
    assert all(ev["per_ticker"][t]["digest"] for t in ("AAA", "BBB", "CCC"))
    assert ev["per_ticker"]["AAA"]["cutoff_source"] == "live_train_end"
    # no baseline supplied → every fresh ticker is a "new" first-ever training
    assert all(ev["per_ticker"][t]["baseline_status"] == "new" for t in ("AAA", "BBB", "CCC"))


def test_stale_preexisting_dirs_not_rewritten_are_not_fresh(tmp_path):
    """All expected tickers exist but predate launch → NOT certified (the bug)."""
    models = tmp_path / "models"
    for t in ("AAA", "BBB", "CCC"):
        _write_ticker(models, t, mtime=LAUNCH - 500)  # older than launch

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH, exit_code=0)

    assert ev["certified"] is False
    assert ev["status"] == "failed"
    assert ev["trainable_succeeded_count"] == 0
    assert set(ev["sets"]["trainable_stale"]) == {"AAA", "BBB", "CCC"}
    assert set(ev["sets"]["trainable_blocking"]) == {"AAA", "BBB", "CCC"}
    # a stale, pre-existing population contributes NO fresh data cutoff
    assert ev["min_data_cutoff"] is None
    assert ev["max_data_cutoff"] is None


def test_one_ticker_not_rewritten_is_fail(tmp_path):
    """Two fresh, one stale → 100%-trainable rule blocks certification even at 2/3."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    _write_ticker(models, "CCC", mtime=LAUNCH - 10)  # not rewritten this run

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH, exit_code=0)

    assert ev["certified"] is False
    assert ev["status"] == "failed"
    assert ev["sets"]["trainable_stale"] == ["CCC"]
    assert "CCC" in ev["sets"]["trainable_blocking"]


def test_missing_trainable_ticker_blocks_certification(tmp_path):
    """A trainable ticker with NO artifact at all → not certified (100% required)."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    # CCC has no dir at all

    ev = mod.build_marker_evidence(models, ["AAA", "BBB", "CCC"], LAUNCH, exit_code=0)

    assert ev["certified"] is False
    assert ev["sets"]["trainable_missing"] == ["CCC"]
    assert "CCC" in ev["sets"]["trainable_blocking"]


def test_orphan_dirs_outside_watchlist_do_not_inflate_coverage(tmp_path):
    """Pre-existing orphan dirs NOT in the frozen watchlist are ignored entirely."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH + 10)
    # 5 stale orphans from prior watchlists — these must not count toward anything.
    for orphan in ("OLD1", "OLD2", "OLD3", "OLD4", "OLD5"):
        _write_ticker(models, orphan, mtime=LAUNCH - 9999)

    ev = mod.build_marker_evidence(models, ["AAA", "BBB"], LAUNCH, exit_code=0)

    assert ev["certified"] is True
    assert ev["status"] == "success"
    assert ev["expected_count"] == 2
    assert ev["trainable_succeeded_count"] == 2
    assert "OLD1" not in json.dumps(ev["sets"])  # orphans absent from every set


def test_cutoff_no_longer_falls_back_to_trained_date(tmp_path):
    """Codex review #420 round 2: the trained_date fallback is REMOVED — a
    rewritten artifact with no live_train_end can never certify, regardless
    of how fresh trained_date (wall clock) looks."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end=None, trained_date="2026-06-28")

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0)

    assert ev["certified"] is False
    assert ev["per_ticker"]["AAA"]["state"] == mod.STATE_UNPARSEABLE
    assert "trained_date fallback removed" in ev["per_ticker"]["AAA"]["reason"]
    assert ev["min_data_cutoff"] is None


def test_empty_expected_universe_refused(tmp_path):
    with pytest.raises(ValueError):
        mod.build_marker_evidence(tmp_path, [], LAUNCH, exit_code=0)


# ---------------------------------------------------------------------------
# exit_code hard gate (Codex round 2, point 3)
# ---------------------------------------------------------------------------
def test_nonzero_exit_code_blocks_certification_even_with_perfect_artifacts(tmp_path):
    """Every trainable ticker freshly rewritten and identity-proven — but the
    actual train process failed. Certification must still be refused;
    artifact freshness can never override a nonzero exit code."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    _write_ticker(models, "BBB", mtime=LAUNCH + 20, live_train_end="2026-06-25")

    ev = mod.build_marker_evidence(models, ["AAA", "BBB"], LAUNCH, exit_code=1)

    assert ev["certified"] is False
    assert ev["status"] == "failed"
    assert ev["exit_code_ok"] is False
    assert ev["exit_code"] == 1
    # the artifacts themselves are still reported as fresh — the exit-code
    # gate is independent from (and does not corrupt) the artifact evidence.
    assert ev["trainable_coverage"] == 1.0


def test_zero_exit_code_with_perfect_artifacts_certifies(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0)
    assert ev["certified"] is True
    assert ev["exit_code_ok"] is True


# ---------------------------------------------------------------------------
# pre-run baseline / digest identity (Codex round 2, point 1)
# ---------------------------------------------------------------------------
def test_capture_baseline_snapshots_digest_and_cutoff(tmp_path):
    models = tmp_path / "models"
    meta = _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-01")
    _write_ticker(models, "BBB", mtime=LAUNCH - 100, live_train_end=None, trained_date="2026-06-01")
    # CCC has no artifact at all

    snap = mod.capture_baseline(models, ["AAA", "BBB", "CCC"])

    assert set(snap) == {"AAA"}  # BBB has no live_train_end -> no baseline entry; CCC missing
    assert snap["AAA"]["digest"] == _digest_of(meta)
    assert snap["AAA"]["data_cutoff"] == "2026-06-01"


def test_digest_unchanged_from_baseline_without_reason_blocks_certification(tmp_path):
    """A rewritten artifact whose bytes are IDENTICAL to the pre-run baseline
    (no-op writer / cp -p restamp) with no explicit justification must NOT
    certify — this is the exact spoof Codex flagged: mtime alone proved
    nothing about whether training actually ran."""
    models = tmp_path / "models"
    meta = _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-20")
    baseline = {"AAA": {"digest": _digest_of(meta), "data_cutoff": "2026-06-20"}}

    # "Retrain" rewrites the SAME bytes, just touches mtime forward.
    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0, baseline=baseline)

    assert ev["certified"] is False
    assert ev["per_ticker"]["AAA"]["state"] == mod.STATE_UNVERIFIED_NO_CHANGE
    assert ev["sets"]["trainable_unverified_no_change"] == ["AAA"]
    assert "cannot prove training actually ran" in ev["per_ticker"]["AAA"]["reason"]


def test_digest_unchanged_with_explicit_no_change_reason_certifies(tmp_path):
    """An idempotent re-run that genuinely produced identical output MAY
    certify, but only when the artifact itself carries an explicit,
    non-empty justification — never silently."""
    models = tmp_path / "models"
    meta = _write_ticker(
        models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-20",
        extra={"no_change_reason": "no new trading bar since last run; retrain reproduced identical policy"},
    )
    baseline = {"AAA": {"digest": _digest_of(meta), "data_cutoff": "2026-06-20"}}

    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))  # same bytes, fresh mtime

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0, baseline=baseline)

    assert ev["certified"] is True
    assert ev["per_ticker"]["AAA"]["baseline_status"] == "unchanged_explicit"


def test_digest_changed_from_baseline_certifies(tmp_path):
    models = tmp_path / "models"
    meta = _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-20")
    baseline = {"AAA": {"digest": "deadbeef" * 8, "data_cutoff": "2026-06-15"}}

    # Genuine rewrite: new cutoff, different bytes, fresh mtime.
    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))
    meta.write_text(json.dumps({"model_name": "AAA", "live_train_end": "2026-06-27"}, indent=2))
    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0, baseline=baseline)

    assert ev["certified"] is True
    assert ev["per_ticker"]["AAA"]["baseline_status"] == "changed"
    assert ev["min_data_cutoff"] == "2026-06-27"


def test_no_baseline_entry_treats_rewrite_as_new_first_training(tmp_path):
    """A ticker with no pre-run baseline (first-ever training, or the caller
    never captured one) is accepted without an identity check."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0, baseline={})

    assert ev["certified"] is True
    assert ev["per_ticker"]["AAA"]["baseline_status"] == "new"


def test_cutoff_regression_vs_baseline_blocks_certification(tmp_path):
    """A rewritten artifact whose data cutoff moved BACKWARD relative to the
    pre-run baseline must never certify — the effective training window went
    stale, even though the file was genuinely touched this run."""
    models = tmp_path / "models"
    meta = _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-25")
    baseline = {"AAA": {"digest": "irrelevant-old-digest", "data_cutoff": "2026-06-25"}}

    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))
    meta.write_text(json.dumps({"model_name": "AAA", "live_train_end": "2026-06-10"}, indent=2))
    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))

    ev = mod.build_marker_evidence(models, ["AAA"], LAUNCH, exit_code=0, baseline=baseline)

    assert ev["certified"] is False
    assert ev["per_ticker"]["AAA"]["state"] == mod.STATE_CUTOFF_REGRESSED
    assert ev["sets"]["trainable_cutoff_regressed"] == ["AAA"]


# ---------------------------------------------------------------------------
# non_trainable enumeration (Codex round 2, point 4)
# ---------------------------------------------------------------------------
def test_non_trainable_missing_is_tolerated_but_trainable_missing_is_not(tmp_path):
    models = tmp_path / "models"
    for t in ("AAA", "BBB", "CCC"):
        _write_ticker(models, t, mtime=LAUNCH + 10)
    # SPY (benchmark ETF) is expected but the tournament never trains it.
    ev = mod.build_marker_evidence(
        models, ["AAA", "BBB", "CCC", "SPY"], LAUNCH, exit_code=0,
        non_trainable={"SPY": "benchmark index — not a per-ticker admission candidate"},
    )

    assert ev["certified"] is True
    assert ev["status"] == "success"
    assert ev["trainable_count"] == 3
    assert ev["excluded_count"] == 1
    assert ev["sets"]["excluded_missing"] == ["SPY"]
    assert ev["coverage_policy"]["excluded_tickers"] == {
        "SPY": "benchmark index — not a per-ticker admission candidate"
    }


def test_non_trainable_ticker_not_in_expected_set_refused(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    with pytest.raises(ValueError, match="not present in expected_tickers"):
        mod.build_marker_evidence(
            models, ["AAA"], LAUNCH, exit_code=0,
            non_trainable={"ZZZ": "not even in the watchlist"},
        )


def test_non_trainable_without_justification_refused(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "SPY", mtime=LAUNCH + 10)
    for bad_reason in ("", "   ", None):
        with pytest.raises(ValueError, match="non-empty justification"):
            mod.build_marker_evidence(
                models, ["AAA", "SPY"], LAUNCH, exit_code=0,
                non_trainable={"SPY": bad_reason},
            )


def test_non_trainable_excluding_entire_watchlist_refused(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "SPY", mtime=LAUNCH + 10)
    with pytest.raises(ValueError, match="nothing left to certify"):
        mod.build_marker_evidence(
            models, ["SPY"], LAUNCH, exit_code=0,
            non_trainable={"SPY": "benchmark"},
        )


def test_non_trainable_ticker_that_does_succeed_is_recorded_but_not_required(tmp_path):
    """An excluded ticker CAN still have a fresh artifact (e.g. the
    tournament happens to also train it) — that's fine, just not required."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "SPY", mtime=LAUNCH + 10)

    ev = mod.build_marker_evidence(
        models, ["AAA", "SPY"], LAUNCH, exit_code=0,
        non_trainable={"SPY": "benchmark index"},
    )
    assert ev["certified"] is True
    assert ev["sets"]["excluded_succeeded"] == ["SPY"]


def test_no_hardcoded_coverage_floor_remains(tmp_path):
    """Regression guard: build_marker_evidence must NOT accept a bare
    min_coverage knob — Codex flagged the 0.90 blanket floor as an
    unregistered magic number; the replacement is explicit non_trainable
    enumeration only."""
    import inspect
    params = inspect.signature(mod.build_marker_evidence).parameters
    assert "min_coverage" not in params


# ---------------------------------------------------------------------------
# CLI (main) — the artifact the bash wrapper actually calls
# ---------------------------------------------------------------------------
def _run_cli(tmp_path, models, expected, *, non_trainable=None, baseline=None, exit_code=0, extra_args=None):
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
        "--exit-code", str(exit_code),
        "--date", "2026-06-30",
        "--completed-at", "2026-06-30T06:10:00Z",
    ]
    if non_trainable is not None:
        nt_path = tmp_path / "non_trainable.json"
        nt_path.write_text(json.dumps(non_trainable))
        args += ["--non-trainable", str(nt_path)]
    if baseline is not None:
        b_path = tmp_path / "baseline.json"
        b_path.write_text(json.dumps(baseline))
        args += ["--baseline", str(b_path)]
    args += extra_args or []
    proc = subprocess.run(args, capture_output=True, text=True)
    return proc, marker


def test_cli_exit0_partial_is_not_certified_and_writes_no_marker(tmp_path):
    """train exit-0 but a partial/no-op population → CLI must NOT stamp a marker."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    _write_ticker(models, "BBB", mtime=LAUNCH - 10)  # stale — training did not rewrite it

    proc, marker = _run_cli(tmp_path, models, ["AAA", "BBB"])

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not marker.exists(), "a non-certified run must leave NO fresh marker"
    assert "NOT CERTIFIED" in proc.stderr


def test_cli_certified_stamps_artifact_derived_marker(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    _write_ticker(models, "BBB", mtime=LAUNCH + 20, live_train_end="2026-06-24")

    proc, marker = _run_cli(tmp_path, models, ["AAA", "BBB"])

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
    assert payload["exit_code"] == 0


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
    ]
    proc = subprocess.run(args, capture_output=True, text=True)
    assert proc.returncode != 0
    # untouched prior marker
    assert json.loads(marker.read_text())["trained_date"] == "2026-06-01"


def test_cli_nonzero_exit_code_blocks_certification(tmp_path):
    """Perfect artifacts, but --exit-code 1 (train actually failed) → refused."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")

    proc, marker = _run_cli(tmp_path, models, ["AAA"], exit_code=7)

    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not marker.exists()
    assert "NOT CERTIFIED" in proc.stderr
    assert "exit_code" in proc.stdout  # surfaced in the JSON summary


def test_cli_non_trainable_flag_excludes_benchmark_from_coverage(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10, live_train_end="2026-06-20")
    # SPY intentionally never gets an artifact — it's a benchmark ETF.

    proc, marker = _run_cli(
        tmp_path, models, ["AAA", "SPY"],
        non_trainable={"SPY": "benchmark index — not a per-ticker admission candidate"},
    )

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert marker.exists()
    payload = json.loads(marker.read_text())
    assert payload["certified"] is True
    assert payload["excluded_count"] == 1
    assert payload["sets"]["excluded_missing"] == ["SPY"]


def test_cli_emit_baseline_then_certify_detects_unchanged_no_op(tmp_path):
    """End-to-end: --emit-baseline snapshots BEFORE launch; a 'retrain' that
    only touches mtime (no byte change, no justification) must fail
    certification when the baseline is passed back in."""
    models = tmp_path / "models"
    meta = _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-20")

    wl = tmp_path / "watchlist.json"
    wl.write_text(json.dumps({"watchlist": ["AAA"]}))
    baseline_path = tmp_path / "baseline.json"

    emit_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--models-dir", str(models), "--watchlist", str(wl),
         "--emit-baseline", str(baseline_path)],
        capture_output=True, text=True,
    )
    assert emit_proc.returncode == 0, emit_proc.stdout + emit_proc.stderr
    assert baseline_path.exists()
    baseline_snapshot = json.loads(baseline_path.read_text())
    assert baseline_snapshot["AAA"]["digest"] == _digest_of(meta)

    # "Retrain" only touches mtime — no content change, no justification.
    os.utime(meta, (LAUNCH + 10, LAUNCH + 10))

    marker = tmp_path / "marker.json"
    certify_proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--models-dir", str(models), "--watchlist", str(wl),
         "--baseline", str(baseline_path), "--launch-epoch", str(LAUNCH),
         "--run-id", "rid", "--marker", str(marker), "--exit-code", "0"],
        capture_output=True, text=True,
    )
    assert certify_proc.returncode != 0, certify_proc.stdout + certify_proc.stderr
    assert not marker.exists()
    assert "NOT CERTIFIED" in certify_proc.stderr


def test_cli_emit_baseline_mode_does_not_require_launch_epoch_or_marker(tmp_path):
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH - 100, live_train_end="2026-06-20")
    wl = tmp_path / "watchlist.json"
    wl.write_text(json.dumps({"watchlist": ["AAA"]}))
    out = tmp_path / "baseline.json"

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--models-dir", str(models), "--watchlist", str(wl),
         "--emit-baseline", str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert out.exists()


def test_cli_certify_mode_without_required_flags_errors(tmp_path):
    """Certify mode (no --emit-baseline) must require --launch-epoch/--run-id/--marker."""
    models = tmp_path / "models"
    _write_ticker(models, "AAA", mtime=LAUNCH + 10)
    wl = tmp_path / "watchlist.json"
    wl.write_text(json.dumps({"watchlist": ["AAA"]}))

    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--models-dir", str(models), "--watchlist", str(wl)],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "required unless --emit-baseline" in proc.stderr
