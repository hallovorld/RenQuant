"""Shell-level acceptance tests for the redesigned monthly meta-label retrain.

RFC: ``doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md``
(§2.1 consumer gate, §2.2 walk-forward snapshot config, §2.3 corpus
coverage + scorer-family parity).

Coverage (RFC acceptance criteria, shell-level equivalents):

* AC-1: with the consumer dark (the REAL pinned config, copied into a
  sandbox), the job exits 0 in <5s with the single skip line and writes
  nothing outside ``logs/`` — and sends no ntfy alarm.
* AC-3: corpus-staleness injection fails closed with the named
  ``wf corpus stale for window`` error before any config/sim work.
* AC-4: the constructed snapshot config ALWAYS carries the explicit
  calibrator-bound v2 walkforward override — the dead ``dropsenti_v3``
  prod pointer is unreachable (proven at execution level AND by source
  inspection).
* AC-6: scorer-family mismatch / unmapped-kind injection fails closed
  with the named error.

AC-2 (live walk-forward sim run) and AC-5 (resolver digest acceptance)
are runtime-gated and validated at the first green armed run
post-deploy; the real sim is NEVER executed here — the armed sandbox
installs a stub ``sim_driver`` that refuses to run (exits 3).

Sandbox design (per repo convention, cf. ``test_notify_sh.py``): the
script is COPIED into a fabricated umbrella tree under ``tmp_path`` and
pointed there via ``RQ_META_LABEL_REPO_DIR``; ``curl`` is stubbed on
``PATH`` (no network); the sandbox ``.venv`` python is a shim to
``sys.executable``. The real live tree is only ever read (AC-1 copies
the pinned config), never written, and never used as a working dir.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "monthly_meta_label_retrain.sh"
SUBREPO_ENV = REPO / "scripts" / "subrepo_env.sh"

LIVE_PINNED_CONFIG = Path(
    "/Users/renhao/git/github/RenQuant/.subrepo_runtime/repos/"
    "renquant-strategy-104/configs/strategy_config.json"
)

SKIP_LINE = (
    "meta-label consumer dark — retrain skipped by design "
    "(see doc/design/2026-07-18-metalabel-monthly-retrain-redesign.md)"
)
V2_MANIFEST_NAME = "walkforward_manifest_v2_20260602.json"
STALE_ERR = "wf corpus stale for window"
MISMATCH_ERR = "wf corpus scorer-family mismatch"
UNMAPPED_ERR = "wf corpus scorer-family unmapped"


# ---------------------------------------------------------------------------
# Sandbox helpers
# ---------------------------------------------------------------------------


def _build_sandbox(tmp_path: Path, pinned_cfg: dict) -> tuple[Path, dict[str, str]]:
    """Fabricated umbrella tree + hermetic env for the copied wrapper."""
    sandbox = tmp_path / "umbrella"
    (sandbox / "scripts").mkdir(parents=True)
    shutil.copy2(SCRIPT, sandbox / "scripts" / SCRIPT.name)
    shutil.copy2(SUBREPO_ENV, sandbox / "scripts" / SUBREPO_ENV.name)

    venv_bin = sandbox / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "activate").write_text("", encoding="utf-8")
    shim = venv_bin / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)

    cfg_dir = (
        sandbox / ".subrepo_runtime" / "repos" / "renquant-strategy-104" / "configs"
    )
    cfg_dir.mkdir(parents=True)
    (cfg_dir / "strategy_config.json").write_text(
        json.dumps(pinned_cfg, indent=2), encoding="utf-8"
    )

    (sandbox / "backtesting" / "renquant_104" / "artifacts" / "sim").mkdir(parents=True)
    (sandbox / "data").mkdir()

    stub_bin = tmp_path / "stub_bin"
    stub_bin.mkdir()
    curl_log = tmp_path / "curl_args.log"
    curl = stub_bin / "curl"
    curl.write_text(
        f'#!/bin/sh\nprintf \'%s\\n\' "$@" >> "{curl_log}"\n', encoding="utf-8"
    )
    curl.chmod(0o755)

    home = tmp_path / "home"
    home.mkdir()

    env = {
        "PATH": f"{stub_bin}:/usr/bin:/bin",
        "HOME": str(home),
        "RQ_META_LABEL_REPO_DIR": str(sandbox),
        "RENQUANT_SUBREPO_ROOT": str(sandbox / ".subrepo_runtime" / "repos"),
    }
    return sandbox, env


def _write_stub_subrepos(sandbox: Path) -> None:
    """Importable stubs so the multirepo preflights pass; the stub
    sim_driver REFUSES to run as __main__ (the tests must never execute
    a real sim)."""
    root = sandbox / ".subrepo_runtime" / "repos"

    wf_gate = root / "renquant-backtesting" / "src" / "renquant_backtesting" / "wf_gate"
    wf_gate.mkdir(parents=True)
    (wf_gate.parent / "__init__.py").write_text("", encoding="utf-8")
    (wf_gate / "__init__.py").write_text("", encoding="utf-8")
    (wf_gate / "sim_driver.py").write_text(
        "import sys\n"
        "if __name__ == '__main__':\n"
        "    sys.stderr.write('stub sim_driver: refusing to run a real sim in tests\\n')\n"
        "    sys.exit(3)\n",
        encoding="utf-8",
    )

    model_common = root / "renquant-model" / "src" / "renquant_model_common"
    model_common.mkdir(parents=True)
    (model_common / "__init__.py").write_text("", encoding="utf-8")
    (model_common / "meta_label_exit.py").write_text("", encoding="utf-8")


def _write_manifest(
    sandbox: Path,
    cutoffs: list[dt.date],
    *,
    vintage_kind: str | None = "panel_ltr_xgboost",
) -> None:
    """Fabricated v2 corpus manifest (+ optional vintage artifacts).

    ``effective_train_cutoff_date`` is what the wrapper must read (loader
    parity: feature cutoff), so it is set to the given date; cutoff_date
    is offset later to prove the wrapper does NOT key on it.
    """
    strategy_root = sandbox / "backtesting" / "renquant_104"
    rows = []
    for cutoff in cutoffs:
        uri = f"artifacts/walkforward_v2_20260602/{cutoff.isoformat()}/panel-ltr.json"
        rows.append(
            {
                "artifact_uri": uri,
                "cutoff_date": f"{(cutoff + dt.timedelta(days=84)).isoformat()}T00:00:00",
                "effective_train_cutoff_date": f"{cutoff.isoformat()}T00:00:00",
                "lookahead_days": 60,
                "trained_date": "2026-06-02T00:00:00",
            }
        )
        if vintage_kind is not None:
            artifact = strategy_root / uri
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps({"kind": vintage_kind}), encoding="utf-8"
            )
    manifest = {"cadence_days": 21, "retrains": rows, "training_window_years": 2}
    (strategy_root / "artifacts" / "sim" / V2_MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _armed_config(pinned_family: str = "xgb") -> dict:
    """Minimal consumer-ARMED pinned config carrying the dead prod
    walkforward pointer (the AC-4 inheritance trap)."""
    return {
        "ranking": {
            "meta_label": {"enabled": True, "threshold": 0.5},
            "panel_scoring": {
                "kind": pinned_family,
                "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
            },
        },
        "walkforward": {
            "manifest_path": (
                "/Users/renhao/git/github/RenQuant/backtesting/renquant_104/"
                "artifacts/sim/walkforward_manifest_dropsenti_v3.json"
            )
        },
    }


def _snapshot_tree(sandbox: Path) -> set[str]:
    return {
        str(p.relative_to(sandbox))
        for p in sandbox.rglob("*")
        if p.is_file()
    }


def _run(sandbox: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["/bin/bash", str(sandbox / "scripts" / SCRIPT.name)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


# ---------------------------------------------------------------------------
# AC-1 — consumer gate (§2.1)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ac1_consumer_dark_real_pinned_config_exits_zero_fast_writes_nothing(
    tmp_path: Path,
) -> None:
    """AC-1 with the REAL pinned config (copied; the live tree is only read)."""
    if not LIVE_PINNED_CONFIG.exists():
        pytest.skip(f"pinned runtime config not on this machine: {LIVE_PINNED_CONFIG}")
    real_cfg = json.loads(LIVE_PINNED_CONFIG.read_text(encoding="utf-8"))
    if real_cfg.get("ranking", {}).get("meta_label", {}).get("enabled", False):
        pytest.skip(
            "pinned config has re-armed the meta-label consumer; the "
            "consumer-dark AC-1 path no longer applies (see RFC §2.1)"
        )

    sandbox, env = _build_sandbox(tmp_path, real_cfg)
    before = _snapshot_tree(sandbox)

    t0 = time.monotonic()
    proc = _run(sandbox, env)
    elapsed = time.monotonic() - t0

    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert elapsed < 5.0, f"consumer-dark skip took {elapsed:.1f}s (AC-1 bound: 5s)"
    assert SKIP_LINE in proc.stdout

    new_paths = _snapshot_tree(sandbox) - before
    non_log = {p for p in new_paths if not p.startswith("logs/")}
    assert not non_log, f"consumer-dark run wrote outside logs/: {sorted(non_log)}"
    assert not (tmp_path / "curl_args.log").exists(), (
        "consumer-dark skip must not ntfy — no alarm while dark by design"
    )


@pytest.mark.integration
def test_ac1_gate_treats_absent_meta_label_block_as_dark(tmp_path: Path) -> None:
    """RFC §2.1: enabled=false OR the block entirely absent → skip."""
    cfg = {"ranking": {"panel_scoring": {"kind": "xgb"}}}
    sandbox, env = _build_sandbox(tmp_path, cfg)
    proc = _run(sandbox, env)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert SKIP_LINE in proc.stdout


@pytest.mark.parametrize("malformed_enabled", ["false", "true", 0, 1, None])
def test_ac1_gate_rejects_non_boolean_enabled_without_training(
    tmp_path: Path, malformed_enabled: object
) -> None:
    """Only JSON ``true`` can arm the consumer; malformed config fails closed."""
    cfg = {"ranking": {"meta_label": {"enabled": malformed_enabled}}}
    sandbox, env = _build_sandbox(tmp_path, cfg)

    proc = _run(sandbox, env)

    assert proc.returncode != 0
    assert "must be a JSON boolean" in proc.stderr
    assert "invalid or unreadable for consumer gate" in (
        tmp_path / "curl_args.log"
    ).read_text(encoding="utf-8")
    snapshot = (
        sandbox / "backtesting" / "renquant_104"
        / "strategy_config.sim_monthly_retrain_snapshot.json"
    )
    assert not snapshot.exists(), "malformed gate must stop before snapshot/sim work"


# ---------------------------------------------------------------------------
# AC-3 — corpus-coverage assert (§2.3)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ac3_stale_corpus_fails_closed_with_named_error(tmp_path: Path) -> None:
    sandbox, env = _build_sandbox(tmp_path, _armed_config())
    _write_stub_subrepos(sandbox)
    # Newest feature cutoff pinned far in the past: stale for ANY run date.
    # No vintage artifact files on disk: also proves the staleness assert
    # fires BEFORE scorer-family parity reads the artifacts.
    _write_manifest(
        sandbox,
        [dt.date(2024, 12, 1), dt.date(2025, 1, 5)],
        vintage_kind=None,
    )

    proc = _run(sandbox, env)

    assert proc.returncode != 0
    assert STALE_ERR in proc.stdout, proc.stdout + proc.stderr
    assert "newest cutoff 2025-01-05" in proc.stdout
    # Fails closed BEFORE any snapshot-config / sim work:
    snap_cfg = (
        sandbox / "backtesting" / "renquant_104"
        / "strategy_config.sim_monthly_retrain_snapshot.json"
    )
    assert not snap_cfg.exists()
    # The named error is what reaches the sentinel (ntfy body).
    assert STALE_ERR in (tmp_path / "curl_args.log").read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# AC-6 — scorer-family parity (§2.2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ac6_scorer_family_mismatch_fails_closed(tmp_path: Path) -> None:
    """Corpus family maps cleanly but != pinned family (the hf_patchtst
    training/serving skew the RFC calls out)."""
    sandbox, env = _build_sandbox(tmp_path, _armed_config(pinned_family="hf_patchtst"))
    _write_stub_subrepos(sandbox)
    _write_manifest(sandbox, [dt.date.today()], vintage_kind="panel_ltr_xgboost")

    proc = _run(sandbox, env)

    assert proc.returncode != 0
    assert MISMATCH_ERR in proc.stdout, proc.stdout + proc.stderr
    assert MISMATCH_ERR in (tmp_path / "curl_args.log").read_text(encoding="utf-8")


@pytest.mark.integration
def test_ac6_unmapped_vintage_kind_fails_closed(tmp_path: Path) -> None:
    """A kind outside the explicit allowlist is refused (no fuzzy match)."""
    sandbox, env = _build_sandbox(tmp_path, _armed_config())
    _write_stub_subrepos(sandbox)
    _write_manifest(sandbox, [dt.date.today()], vintage_kind="mystery_model")

    proc = _run(sandbox, env)

    assert proc.returncode != 0
    assert UNMAPPED_ERR in proc.stdout, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# AC-4 — explicit walkforward override (§2.2)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_ac4_snapshot_config_always_carries_explicit_v2_override(
    tmp_path: Path,
) -> None:
    """Armed run up to the sim boundary: the constructed snapshot config
    must carry the explicit v2 override with fail_on_no_model=true, and
    the inherited dead dropsenti_v3 pointer must be gone. The stub
    sim_driver then halts the pipeline (exit 3) — no real sim ever runs."""
    sandbox, env = _build_sandbox(tmp_path, _armed_config())
    _write_stub_subrepos(sandbox)
    _write_manifest(sandbox, [dt.date.today()], vintage_kind="panel_ltr_xgboost")

    proc = _run(sandbox, env)

    # Stub sim exits 3 → wrapper fails closed on "snapshot sim failed".
    assert proc.returncode != 0
    assert "snapshot sim failed" in (tmp_path / "curl_args.log").read_text(
        encoding="utf-8"
    )

    snap_cfg_path = (
        sandbox / "backtesting" / "renquant_104"
        / "strategy_config.sim_monthly_retrain_snapshot.json"
    )
    assert snap_cfg_path.exists(), proc.stdout + proc.stderr
    text = snap_cfg_path.read_text(encoding="utf-8")
    snap_cfg = json.loads(text)

    wf = snap_cfg["walkforward"]
    assert wf["enabled"] is True
    assert wf["fail_on_no_model"] is True
    assert wf["manifest_path"] == str(
        sandbox / "backtesting" / "renquant_104" / "artifacts" / "sim" / V2_MANIFEST_NAME
    )
    assert "dropsenti" not in text, (
        "the dead prod walkforward pointer leaked into the snapshot config"
    )
    # Snapshot-collection semantics preserved:
    assert snap_cfg["meta_label_training"]["enabled"] is True
    assert snap_cfg["ranking"]["meta_label"] == {"enabled": False}
    # The sim was halted by the stub before writing any parquet:
    assert not list((sandbox / "data").glob("*.parquet"))


def test_ac4_source_override_is_wholesale_and_gate_precedes_the_sim() -> None:
    """Source-level rot-guard (convention:
    ``test_monthly_jobs_multirepo_fail_closed.py``): the override is a
    WHOLESALE dict replacement — inheritance of the prod pointer is
    structurally unreachable — and the §2.1 gate sits before any
    sim/train machinery."""
    src = SCRIPT.read_text(encoding="utf-8")

    assert SKIP_LINE in src
    assert src.index(SKIP_LINE) < src.index("renquant_backtesting.wf_gate.sim_driver")

    assert 'src["walkforward"] = {' in src, (
        "walkforward override must be a wholesale replacement, never a "
        "merge/setdefault that could inherit the dead prod pointer"
    )
    assert V2_MANIFEST_NAME in src
    assert '"fail_on_no_model": True' in src
    assert "walkforward_manifest_dropsenti_v3" not in src

    # Named fail-closed errors the sentinel patterns on (RFC §5.3):
    assert STALE_ERR in src
    assert MISMATCH_ERR in src
    assert UNMAPPED_ERR in src

    # Sandboxability override keeps the production default:
    assert 'REPO_DIR="${RQ_META_LABEL_REPO_DIR:-/Users/renhao/git/github/RenQuant}"' in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
