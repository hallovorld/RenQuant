from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "render_strategy_104_snapshot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("render_strategy_104_snapshot_for_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture_root(mod, tmp_path: Path) -> Path:
    """A minimal but complete source tree in the NEW layout: pinned subrepo
    configs under .subrepo_runtime (with a plain-file .git HEAD), artifacts
    under backtesting/renquant_104, and subrepos.lock.json."""
    root = tmp_path / "repo"
    configs = root / mod.PINNED_CONFIGS_REL
    configs.mkdir(parents=True)
    git_dir = root / mod.PINNED_GIT_DIR_REL
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")
    strategy_dir = root / mod.STRATEGY_DIR_REL

    _write_json(strategy_dir / "artifacts" / "prod" / "primary.json", {
        "trained_date": "2026-06-30",
        "effective_train_cutoff_date": "2026-05-01",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:abc123",
        "label_col": "fwd_60d_excess",
        "feature_cols": ["a", "b"],
        "metadata": {"wf_gate_metadata": {"passed": True, "run_at": "2026-06-30T00:00:00"}},
    })
    _write_json(strategy_dir / "artifacts" / "prod" / "calib.json", {
        "kind": "global_panel_calibration", "trained_date": "2026-07-01",
        "metadata": {"method": "platt", "pool_ic": 0.0993,
                     "scorer_model_content_fingerprint": "sha256:feedface"},
    })
    ckpt = strategy_dir / "artifacts" / "shadow" / "model.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    ckpt.write_bytes(b"not a real checkpoint")
    _write_json(Path(str(ckpt) + ".metadata.json"), {
        "trained_date": "2026-05-22",
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:f8fb2259b2bf1537",
    })

    _write_json(configs / mod.ACTIVE_CONFIG_NAME, {
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "max_concurrent_positions": 8,
        "ranking": {
            "panel_scoring": {
                "kind": "xgb",
                "artifact_path": "artifacts/prod/primary.json",
                "conviction_gate": {"enabled": True, "mu_floor": 0.03},
                "global_calibration": {"artifact_path": "artifacts/prod/calib.json"},
                "shadow_models": [
                    {"name": "old_primary", "kind": "hf_patchtst",
                     "artifact_path": "artifacts/shadow/model.pt"},
                ],
            },
        },
        "rotation": {"panel_buy_top_n": 3},
    })
    _write_json(configs / mod.SHADOW_CONFIG_NAME, {
        "ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/shadow/model.pt",
            "global_calibration": {"artifact_path": "artifacts/prod/calib.json"},
        }},
    })
    _write_json(root / mod.LOCK_FILE_REL, {
        "subrepos": [
            {"name": "renquant-strategy-104", "branch": "main",
             "commit": "c" * 40, "status": "bootstrapped"},
            {"name": "renquant-common", "branch": "main",
             "commit": "a" * 40, "status": "bootstrapped"},
        ],
    })
    return root


def test_collect_snapshot_reads_pinned_config_not_umbrella_working_copy(tmp_path):
    """The A6/M9 root cause: the daily run consumes the PINNED subrepo config;
    the umbrella working copy is a rot vector. The collector must key on the
    pinned config and flag the working copy when it disagrees."""
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    # A stale umbrella working copy that still claims a different active kind:
    _write_json(root / mod.STRATEGY_DIR_REL / mod.ACTIVE_CONFIG_NAME, {
        "ranking": {"panel_scoring": {"kind": "hf_patchtst"}},
    })

    snapshot = mod.collect_snapshot(root)

    assert snapshot["active"]["kind"] == "xgb"  # pinned config wins
    assert any("UMBRELLA WORKING-COPY DRIFT" in w for w in snapshot["warnings"])
    rendered = mod.render_markdown(snapshot, generated_at="TEST")
    assert "UMBRELLA WORKING-COPY DRIFT" in rendered


def test_collect_snapshot_active_shadow_calibrator_and_pins(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)

    snapshot = mod.collect_snapshot(root)

    active = snapshot["active"]
    assert active["kind"] == "xgb"
    assert active["trained_date"] == "2026-06-30"
    assert active["binding_cutoff_field"] == "effective_train_cutoff_date"
    assert active["binding_cutoff"] == "2026-05-01"
    assert active["config_fingerprint"] == "sha256:abc123"
    assert active["feature_count"] == 2
    assert active["wf_gate"]["passed"] is True

    assert snapshot["active_calibrator"]["kind"] == "global_panel_calibration"
    assert snapshot["active_calibrator"]["method"] == "platt"
    assert snapshot["active_calibrator"]["trained_date"] == "2026-07-01"

    assert len(snapshot["in_run_shadows"]) == 1
    shadow = snapshot["in_run_shadows"][0]
    assert shadow["name"] == "old_primary"
    assert shadow["metadata_source"] == "sidecar"
    # effective_selection_cutoff_date outranks effective_train_cutoff_date
    assert shadow["binding_cutoff_field"] == "effective_selection_cutoff_date"
    assert shadow["binding_cutoff"] == "2026-02-10"

    assert snapshot["shadow_e2e"]["kind"] == "hf_patchtst"
    assert snapshot["watchlist_size"] == 3
    assert [p["name"] for p in snapshot["pins"]] == [
        "renquant-common", "renquant-strategy-104",
    ]
    assert snapshot["runtime_checkout_commit"] == "c" * 40
    assert snapshot["lock_strategy_104_pin"] == "c" * 40
    # every source that was read is fingerprinted
    assert all(v is not None for v in snapshot["sources"].values())


def test_render_is_deterministic_and_states_absent_fields(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    first = mod.render_markdown(mod.collect_snapshot(root), generated_at="TEST")
    second = mod.render_markdown(mod.collect_snapshot(root), generated_at="TEST")
    assert first == second
    # label_observation_cutoff is not stamped by any fixture artifact:
    assert "| label_observation_cutoff | unknown (field absent) |" in first
    assert "GENERATED FILE" in first
    assert "`cccccccccccc`" in first  # pin table commit short


def test_pin_drift_between_runtime_checkout_and_lock_is_flagged(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    git_head = root / mod.PINNED_GIT_DIR_REL / "HEAD"
    git_head.write_text("d" * 40 + "\n", encoding="utf-8")

    snapshot = mod.collect_snapshot(root)
    assert any("PIN DRIFT" in w for w in snapshot["warnings"])


def test_pin_drift_fails_closed_in_generation_mode(tmp_path, capsys):
    """Codex round-3 review: PIN DRIFT was only ever a warning baked into the
    rendered doc, never a refusal — the canonical committed snapshot could
    legitimize an unpinned runtime while claiming pin alignment."""
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    (root / mod.PINNED_GIT_DIR_REL / "HEAD").write_text("d" * 40 + "\n", encoding="utf-8")
    out = tmp_path / "snapshot.md"

    rc = mod.main(["--repo-root", str(root), "--output", str(out)])
    assert rc == 1
    assert "PIN DRIFT" in capsys.readouterr().err
    assert not out.exists(), "must refuse to write a canonical snapshot over drifted pins"


def test_pin_drift_fails_closed_in_check_mode(tmp_path, capsys):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    out = tmp_path / "snapshot.md"
    assert mod.main(["--repo-root", str(root), "--output", str(out)]) == 0

    (root / mod.PINNED_GIT_DIR_REL / "HEAD").write_text("d" * 40 + "\n", encoding="utf-8")
    rc = mod.main(["--repo-root", str(root), "--output", str(out), "--check"])
    assert rc == 1
    assert "PIN DRIFT" in capsys.readouterr().err


def test_pin_drift_allowed_in_explicit_diagnostic_mode(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    (root / mod.PINNED_GIT_DIR_REL / "HEAD").write_text("d" * 40 + "\n", encoding="utf-8")
    out = tmp_path / "snapshot.md"

    rc = mod.main(["--repo-root", str(root), "--output", str(out), "--allow-pin-drift"])
    assert rc == 0
    assert out.exists()
    assert "PIN DRIFT" in out.read_text(encoding="utf-8")


def test_collect_snapshot_handles_missing_artifact_metadata_gracefully(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    (root / mod.STRATEGY_DIR_REL / "artifacts" / "prod" / "primary.json").unlink()

    snapshot = mod.collect_snapshot(root)
    active = snapshot["active"]
    assert active["kind"] == "xgb"
    assert active["trained_date"] is None
    assert active["metadata_missing"] is True
    rendered = mod.render_markdown(snapshot, generated_at="TEST")
    assert "unknown (metadata file missing or unreadable)" in rendered


def test_collect_snapshot_handles_missing_pinned_configs_gracefully(tmp_path):
    """No pinned runtime checkout at all (e.g. a bare dev clone): the render
    must still complete with explicit warnings, never crash."""
    mod = _load_module()
    root = tmp_path / "bare"
    root.mkdir()
    snapshot = mod.collect_snapshot(root)
    assert any("active config unreadable or missing" in w for w in snapshot["warnings"])
    rendered = mod.render_markdown(snapshot, generated_at="TEST")
    assert "unknown (field absent)" in rendered


def test_metadata_whitelist_excludes_secrets_credentials_and_free_form_notes(tmp_path):
    """Codex review (PR #429): the renderer must WHITELIST fields, not
    serialize arbitrary artifact metadata. Construct metadata carrying
    obviously sensitive/unapproved keys alongside the legitimate ones and
    prove none of the sensitive content ever reaches the rendered output."""
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    _write_json(root / mod.STRATEGY_DIR_REL / "artifacts" / "prod" / "primary.json", {
        # legitimate, whitelisted fields:
        "trained_date": "2026-06-30",
        "effective_train_cutoff_date": "2026-05-01",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:abc123",
        # everything below must NEVER appear in rendered output:
        "api_key": "sk-secret-do-not-leak-123",
        "broker_credentials": {"user": "alice", "password": "hunter2"},
        "_local_debug_path": "/Users/someone/private/notes.txt",
        "internal_notes": "do not ship this text externally, contains PII",
        "aws_access_key_id": "AKIAABCDEFGHIJKLMNOP",
        "training_notes": "free-form prose that stays out of the snapshot",
        "metadata": {
            "wf_gate_metadata": {"passed": False,
                                 "sanity_manifest_path": "/Users/someone/manifest.json"},
        },
    })

    snapshot = mod.collect_snapshot(root)
    rendered = mod.render_markdown(snapshot, generated_at="TEST")

    assert "| trained_date | 2026-06-30 |" in rendered
    assert "sha256:abc123" in rendered

    sensitive_strings = (
        "sk-secret-do-not-leak-123", "hunter2", "alice",
        "/Users/someone/private/notes.txt", "do not ship this text externally",
        "AKIAABCDEFGHIJKLMNOP", "broker_credentials", "api_key",
        "internal_notes", "aws_access_key_id", "_local_debug_path",
        "free-form prose", "/Users/someone/manifest.json",
    )
    for secret in sensitive_strings:
        assert secret not in rendered, f"leaked into rendered output: {secret!r}"
        assert secret not in str(snapshot), f"leaked into snapshot dict: {secret!r}"


def test_absolute_artifact_path_is_relativized_or_redacted_never_leaked_raw(tmp_path):
    """Codex review (PR #429): the rendered snapshot must never contain an
    absolute local filesystem path. Today's config only stores relative
    paths, but the renderer must defend against a future absolute one."""
    mod = _load_module()
    fake_repo_root = tmp_path / "repo2"
    strategy_dir = fake_repo_root / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)

    inside_artifact = strategy_dir / "artifacts" / "prod" / "primary.json"
    _write_json(inside_artifact, {"trained_date": "2026-06-30"})

    row_inside = mod._describe_model(
        role="active", kind="xgb", artifact_rel=str(inside_artifact),
        strategy_dir=strategy_dir, repo_root=fake_repo_root,
    )
    assert not Path(row_inside["artifact_path"]).is_absolute()
    assert str(fake_repo_root) not in row_inside["artifact_path"]

    outside_artifact = tmp_path / "elsewhere" / "model.json"
    _write_json(outside_artifact, {"trained_date": "2026-06-30"})

    row_outside = mod._describe_model(
        role="active", kind="xgb", artifact_rel=str(outside_artifact),
        strategy_dir=strategy_dir, repo_root=fake_repo_root,
    )
    assert str(tmp_path) not in row_outside["artifact_path"]
    assert row_outside["artifact_path"] == "<redacted-external-path>/model.json"


def test_check_mode_passes_after_regeneration_deterministically(tmp_path):
    """Codex round-3 review: a wall-clock Generated-at line churned the
    committed doc on every regeneration with zero semantic change. It is
    replaced with a deterministic source fingerprint that is part of the
    byte-exact --check comparison (no special-casing needed)."""
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    out = tmp_path / "snapshot.md"

    assert mod.main(["--repo-root", str(root), "--output", str(out)]) == 0
    assert mod.main(["--repo-root", str(root), "--output", str(out), "--check"]) == 0

    # Regenerating again with IDENTICAL sources must byte-for-byte match —
    # no timestamp field left to churn.
    first = out.read_text(encoding="utf-8")
    assert mod.main(["--repo-root", str(root), "--output", str(out)]) == 0
    second = out.read_text(encoding="utf-8")
    assert first == second
    assert mod.SOURCE_FINGERPRINT_PREFIX in first


def test_source_fingerprint_changes_only_when_source_content_changes(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    out = tmp_path / "snapshot.md"
    mod.main(["--repo-root", str(root), "--output", str(out)])
    rendered_before = out.read_text(encoding="utf-8")
    fp_before = next(
        line for line in rendered_before.splitlines()
        if line.startswith(mod.SOURCE_FINGERPRINT_PREFIX)
    )

    config_path = root / mod.PINNED_CONFIGS_REL / mod.ACTIVE_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["ranking"]["panel_scoring"]["kind"] = "hf_patchtst"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    mod.main(["--repo-root", str(root), "--output", str(out)])
    rendered_after = out.read_text(encoding="utf-8")
    fp_after = next(
        line for line in rendered_after.splitlines()
        if line.startswith(mod.SOURCE_FINGERPRINT_PREFIX)
    )
    assert fp_before != fp_after, "source fingerprint must change when source content changes"


def test_check_mode_fails_when_pinned_config_changes(tmp_path, capsys):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    out = tmp_path / "snapshot.md"
    assert mod.main(["--repo-root", str(root), "--output", str(out)]) == 0

    config_path = root / mod.PINNED_CONFIGS_REL / mod.ACTIVE_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["ranking"]["panel_scoring"]["kind"] = "hf_patchtst"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    rc = mod.main(["--repo-root", str(root), "--output", str(out), "--check"])
    assert rc == 1
    assert "STALE" in capsys.readouterr().err


def test_verify_pinned_declaration_matches_and_catches_flips(tmp_path):
    mod = _load_module()
    root = _fixture_root(mod, tmp_path)
    out = tmp_path / "snapshot.md"
    assert mod.main(["--repo-root", str(root), "--output", str(out)]) == 0

    configs_dir = root / mod.PINNED_CONFIGS_REL
    lock_path = root / mod.LOCK_FILE_REL
    assert mod.verify_pinned_declaration(
        snapshot_path=out, configs_dir=configs_dir, lock_path=lock_path) == []

    # 1. lock pin advances without regeneration -> caught
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock["subrepos"][0]["commit"] = "e" * 40
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    problems = mod.verify_pinned_declaration(
        snapshot_path=out, configs_dir=configs_dir, lock_path=lock_path)
    assert any("regenerate the snapshot" in p for p in problems)
    lock["subrepos"][0]["commit"] = "c" * 40
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    # 2. pinned config kind flips without regeneration -> caught
    config_path = configs_dir / mod.ACTIVE_CONFIG_NAME
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["ranking"]["panel_scoring"]["kind"] = "hf_patchtst"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    problems = mod.verify_pinned_declaration(
        snapshot_path=out, configs_dir=configs_dir, lock_path=lock_path)
    assert any("active kind" in p for p in problems)


def test_selftest_passes():
    mod = _load_module()
    assert mod.run_selftest() == 0
