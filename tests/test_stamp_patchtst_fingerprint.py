"""Regression coverage for scripts/stamp_patchtst_fingerprint.py.

The shadow daily leg was failing because seed_44 (and similar recovered
artifacts) shipped without `config_fingerprint` in BOTH locations the
LoadScorerTask consults:

  * top-level `metadata.json::config_fingerprint`
  * `metadata.json::training_contract.config_contract.config_fingerprint`

These tests pin the stamper's invariants:
  1. Dry-run never writes.
  2. --write fills BOTH locations atomically.
  3. Re-running on an already-stamped sidecar is a no-op.
  4. Already-stamped TOP level + missing nested → backfills nested only.
  5. Compatibility check rejects mismatched lookahead_days; --force overrides.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAMPER = REPO / "scripts" / "stamp_patchtst_fingerprint.py"


def _make_artifact(tmp_path: Path, **extra) -> Path:
    p = tmp_path / "demo.metadata.json"
    body = {
        "kind": "hf_patchtst_contract_sidecar",
        "artifact_path": "demo.pt",
        "lookahead_days": 60,
        "feature_count": 172,
        "training_contract": {"label_col": "fwd_60d_excess", "lookahead_days": 60},
    }
    body.update(extra)
    p.write_text(json.dumps(body, indent=2))
    return p


def _make_config(tmp_path: Path, **knobs) -> Path:
    """Build a strategy_config.json shaped the way real renquant_104 ships:
    `panel_ltr` is TOP-LEVEL, not nested under `ranking`. `_model_relevant_fields`
    reads from those top-level paths to compute the fingerprint."""
    p = tmp_path / "strategy_config.demo.json"
    body = {
        "watchlist": ["AAPL", "MSFT"],
        "panel_ltr": {
            "lookahead_days": 60,
            "training_resolution": "daily",
            "xgb_params": {"objective": "rank:pairwise"},
            "asset_embeddings": {"enabled": False},
            "hourly": {"enabled": False},
            "minute": {"enabled": False},
        },
        "sector_map": {},
        "sector_etf_map": {},
    }
    body.update(knobs)
    p.write_text(json.dumps(body, indent=2))
    return p


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(STAMPER), *args],
        cwd=str(REPO), capture_output=True, text=True,
    )


def test_dry_run_never_writes(tmp_path):
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    before = art.read_text()
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg))
    assert r.returncode == 0, r.stderr
    assert "Dry-run" in r.stdout
    assert art.read_text() == before


def test_write_fills_both_locations(tmp_path):
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 0, r.stderr
    body = json.loads(art.read_text())
    assert body["config_fingerprint"].startswith("sha256:")
    cc = body["training_contract"]["config_contract"]
    assert cc["config_fingerprint"] == body["config_fingerprint"]
    assert isinstance(body["config_fingerprint_fields"], dict)


def test_re_run_on_stamped_is_noop(tmp_path):
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    snapshot = art.read_text()
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 0, r.stderr
    assert "nothing to do" in r.stdout
    assert art.read_text() == snapshot


def test_backfill_nested_when_top_already_stamped(tmp_path):
    """Models stamped by an older version of the stamper had only the
    top-level field. Re-running with --write must backfill the nested
    config_contract location WITHOUT erroring out."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    # Stamp top-level only — simulate old stamper output.
    body = json.loads(art.read_text())
    body["config_fingerprint"] = "sha256:placeholder"
    art.write_text(json.dumps(body, indent=2))
    # First read live fingerprint to set top-level to match (so we exercise
    # the backfill branch, not the rewrite branch).
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 0, r.stderr
    body = json.loads(art.read_text())
    cc = body.get("training_contract", {}).get("config_contract", {})
    assert cc.get("config_fingerprint") == body["config_fingerprint"]


def test_lookahead_mismatch_refuses_without_force(tmp_path):
    art = _make_artifact(tmp_path, lookahead_days=20,
                         training_contract={"label_col": "fwd_20d_excess",
                                            "lookahead_days": 20})
    cfg = _make_config(tmp_path)  # live lookahead = 60
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "Compatibility check FAILED" in r.stdout
    # Sidecar untouched.
    body = json.loads(art.read_text())
    assert "config_fingerprint" not in body


def test_force_overrides_compat_check(tmp_path):
    art = _make_artifact(tmp_path, lookahead_days=20,
                         training_contract={"label_col": "fwd_20d_excess",
                                            "lookahead_days": 20})
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--force")
    assert r.returncode == 0, r.stderr
    body = json.loads(art.read_text())
    assert "config_fingerprint" in body
