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


def test_missing_artifact_lookahead_refuses_when_live_has_one(tmp_path):
    """Codex review HIGH: stamper used to silently pass when artifact was
    missing lookahead_days. Now fail-closed — incomplete contract is a real
    incompatibility, not a free stamp."""
    art = _make_artifact(tmp_path)
    body = json.loads(art.read_text())
    body.pop("lookahead_days", None)
    body["training_contract"].pop("lookahead_days", None)
    art.write_text(json.dumps(body, indent=2))
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "artifact=<missing>" in r.stdout
    assert "lookahead_days" in r.stdout
    body = json.loads(art.read_text())
    assert "config_fingerprint" not in body


def test_missing_artifact_label_refuses(tmp_path):
    """Same fail-closed principle for label_col."""
    art = _make_artifact(tmp_path)
    body = json.loads(art.read_text())
    body["training_contract"].pop("label_col", None)
    body.pop("label_col", None)
    art.write_text(json.dumps(body, indent=2))
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "label_col" in r.stdout
    assert "artifact=<missing>" in r.stdout


def test_wrong_same_horizon_label_semantics_refuses(tmp_path):
    """Same horizon but different label target (raw vs excess) is a real
    semantic mismatch — startswith('fwd_') is not strong enough. The default
    expected_label_col derives `fwd_{live_lookahead}d_excess` from the live
    config, so `fwd_60d_raw` is rejected."""
    art = _make_artifact(
        tmp_path,
        training_contract={"label_col": "fwd_60d_raw", "lookahead_days": 60},
    )
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "label_col" in r.stdout
    assert "fwd_60d_excess" in r.stdout
    assert "fwd_60d_raw" in r.stdout


def test_explicit_expected_label_col_overrides_default(tmp_path):
    """Non-standard label still stampable when operator declares it."""
    art = _make_artifact(
        tmp_path,
        training_contract={"label_col": "fwd_60d_simple", "lookahead_days": 60},
    )
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-label-col", "fwd_60d_simple")
    assert r.returncode == 0, r.stderr
    body = json.loads(art.read_text())
    assert body["config_fingerprint"].startswith("sha256:")


def test_feature_count_mismatch_refuses(tmp_path):
    """The docstring claimed feature_count was checked; it now actually is.
    Operator passes --expected-feature-count; artifact feature_count must
    match exactly."""
    art = _make_artifact(tmp_path)  # fixture stamps feature_count=172
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-feature-count", "169")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "feature_count" in r.stdout
    assert "169" in r.stdout and "172" in r.stdout


def test_missing_feature_count_refuses_when_expected_provided(tmp_path):
    """If operator declares --expected-feature-count, artifact must carry one
    too — silent missing is the same defect class as the lookahead case."""
    art = _make_artifact(tmp_path)
    body = json.loads(art.read_text())
    body.pop("feature_count", None)
    art.write_text(json.dumps(body, indent=2))
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-feature-count", "172")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "feature_count" in r.stdout
    assert "artifact=<missing>" in r.stdout


def test_feature_count_match_passes(tmp_path):
    """Happy path: operator-provided expected matches artifact's stamped
    feature_count; stamper proceeds."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-feature-count", "172")
    assert r.returncode == 0, r.stderr
    body = json.loads(art.read_text())
    assert body["config_fingerprint"].startswith("sha256:")


def test_already_stamped_with_wrong_feature_count_refuses(tmp_path):
    """Codex review on #58 HIGH: the early-return path for already-stamped
    sidecars used to bypass _check_compatibility. An artifact that was
    over-stamped by the older (less-strict) version of the stamper must still
    be rejected when re-run with a contract flag that the artifact violates."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    # First stamp without feature-count check (mimics #57-era over-stamping).
    r0 = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r0.returncode == 0, r0.stderr
    # Re-run with a contradictory --expected-feature-count. Must fail closed.
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-feature-count", "169")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "feature_count" in r.stdout
    assert "169" in r.stdout and "172" in r.stdout
    assert "nothing to do" not in r.stdout


def test_already_stamped_with_wrong_label_col_refuses(tmp_path):
    """Same defect class: already-stamped artifact whose label_col contradicts
    --expected-label-col must be rejected, not short-circuit to 'nothing to do'."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    r0 = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r0.returncode == 0, r0.stderr
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--expected-label-col", "fwd_60d_raw")
    assert r.returncode == 3, r.stdout + r.stderr
    assert "label_col" in r.stdout
    assert "nothing to do" not in r.stdout


def test_already_stamped_with_missing_label_col_refuses(tmp_path):
    """Already-stamped artifacts must still fail closed when label_col is
    removed after stamping; the early-return cannot bypass missing contracts."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    r0 = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r0.returncode == 0, r0.stderr
    body = json.loads(art.read_text())
    body["training_contract"].pop("label_col", None)
    body.pop("label_col", None)
    art.write_text(json.dumps(body, indent=2))

    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")

    assert r.returncode == 3, r.stdout + r.stderr
    assert "label_col" in r.stdout
    assert "artifact=<missing>" in r.stdout
    assert "nothing to do" not in r.stdout


def test_already_stamped_force_overrides_post_stamp_check(tmp_path):
    """--force still escapes the post-stamp compatibility refusal — same
    semantics as fresh stamping."""
    art = _make_artifact(tmp_path)
    cfg = _make_config(tmp_path)
    r0 = _run("--artifact-meta", str(art), "--strategy-config", str(cfg), "--write")
    assert r0.returncode == 0, r0.stderr
    r = _run("--artifact-meta", str(art), "--strategy-config", str(cfg),
             "--write", "--force", "--expected-feature-count", "169")
    assert r.returncode == 0, r.stderr
