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


def test_collect_snapshot_reads_primary_and_shadow_from_inline_json(tmp_path):
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)

    _write_json(strategy_dir / "artifacts" / "prod" / "primary.json", {
        "trained_date": "2026-06-30",
        "effective_train_cutoff_date": "2026-05-01",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:abc123",
    })
    _write_json(strategy_dir / "artifacts" / "prod" / "shadow.json", {
        "trained_date": "2026-06-01",
        "label_observation_cutoff": "2026-04-08",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:def456",
    })

    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/prod/primary.json",
            "shadow_models": [
                {"name": "old_primary", "kind": "xgb",
                 "artifact_path": "artifacts/prod/shadow.json"},
            ],
        }},
        "watchlist": ["AAPL", "MSFT", "NVDA"],
    })

    snapshot = mod.collect_snapshot(config_path)

    assert snapshot["primary"]["kind"] == "xgb"
    assert snapshot["primary"]["trained_date"] == "2026-06-30"
    assert snapshot["primary"]["binding_cutoff_field"] == "effective_train_cutoff_date"
    assert snapshot["primary"]["binding_cutoff"] == "2026-05-01"
    assert snapshot["primary"]["config_fingerprint"] == "sha256:abc123"

    assert len(snapshot["shadows"]) == 1
    shadow = snapshot["shadows"][0]
    assert shadow["name"] == "old_primary"
    # label_observation_cutoff outranks effective_train_cutoff_date in priority
    assert shadow["binding_cutoff_field"] == "label_observation_cutoff"
    assert shadow["binding_cutoff"] == "2026-04-08"

    assert snapshot["watchlist_size"] == 3


def test_collect_snapshot_reads_binary_artifact_via_metadata_sidecar(tmp_path):
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)

    ckpt = strategy_dir / "artifacts" / "shadow" / "model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"not a real checkpoint")
    _write_json(ckpt.with_suffix(".pt.metadata.json"), {
        "trained_date": "2026-05-22",
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10",
        "lookahead_days": 60,
        "config_fingerprint": "sha256:f8fb2259b2bf1537",
    })

    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/shadow/model.pt",
            "shadow_models": [],
        }},
        "watchlist": [],
    })

    snapshot = mod.collect_snapshot(config_path)
    primary = snapshot["primary"]
    assert primary["kind"] == "hf_patchtst"
    assert primary["metadata_source"] == "sidecar"
    assert primary["trained_date"] == "2026-05-22"
    # effective_selection_cutoff_date outranks effective_train_cutoff_date
    assert primary["binding_cutoff_field"] == "effective_selection_cutoff_date"
    assert primary["binding_cutoff"] == "2026-02-10"
    assert snapshot["shadows"] == []


def test_collect_snapshot_handles_missing_artifact_metadata_gracefully(tmp_path):
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/prod/does_not_exist.json",
            "shadow_models": [],
        }},
        "watchlist": ["AAPL"],
    })

    snapshot = mod.collect_snapshot(config_path)
    primary = snapshot["primary"]
    assert primary["kind"] == "xgb"
    assert primary["trained_date"] is None
    assert primary["binding_cutoff_field"] is None
    assert primary["binding_cutoff"] is None


def test_render_markdown_contains_expected_fields():
    mod = _load_module()
    snapshot = {
        "config_path": "backtesting/renquant_104/strategy_config.json",
        "primary": {
            "role": "primary", "name": None, "kind": "hf_patchtst",
            "artifact_path": "artifacts/shadow/model.pt", "trained_date": "2026-05-22",
            "binding_cutoff_field": "effective_train_cutoff_date",
            "binding_cutoff": "2024-11-13", "lookahead_days": 60,
            "config_fingerprint": "sha256:abc", "metadata_source": "sidecar",
        },
        "shadows": [],
        "watchlist_size": 142,
    }
    rendered = mod.render_markdown(snapshot)
    assert "GENERATED FILE" in rendered
    assert 'kind="hf_patchtst"' in rendered
    assert "trained_date=2026-05-22" in rendered
    assert "effective_train_cutoff_date=2024-11-13" in rendered
    assert "142 tickers" in rendered
    assert "(none configured)" in rendered  # no shadow models


def test_check_mode_fails_when_output_is_stale(tmp_path, capsys):
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {"kind": "xgb", "artifact_path": None,
                                       "shadow_models": []}},
        "watchlist": [],
    })
    output_path = tmp_path / "snapshot.md"
    output_path.write_text("stale content\n", encoding="utf-8")

    rc = mod.main([
        "--strategy-config", str(config_path),
        "--output", str(output_path),
        "--check",
    ])
    assert rc == 1
    assert "STALE" in capsys.readouterr().err


def test_metadata_whitelist_excludes_secrets_credentials_and_free_form_notes(tmp_path):
    """Codex review (PR #429): the renderer must WHITELIST fields, not
    serialize arbitrary artifact metadata. Construct metadata carrying
    obviously sensitive/unapproved keys alongside the legitimate ones and
    prove none of the sensitive content ever reaches the rendered output."""
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)

    _write_json(strategy_dir / "artifacts" / "prod" / "primary.json", {
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
    })

    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {
            "kind": "xgb",
            "artifact_path": "artifacts/prod/primary.json",
            "shadow_models": [],
        }},
        "watchlist": ["AAPL"],
    })

    snapshot = mod.collect_snapshot(config_path)
    rendered = mod.render_markdown(snapshot)

    # legitimate fields present:
    assert "trained_date=2026-06-30" in rendered
    assert "sha256:abc123" in rendered

    # nothing sensitive ever reaches the snapshot dict OR the rendered text:
    sensitive_strings = (
        "sk-secret-do-not-leak-123", "hunter2", "alice",
        "/Users/someone/private/notes.txt", "do not ship this text externally",
        "AKIAABCDEFGHIJKLMNOP", "broker_credentials", "api_key",
        "internal_notes", "aws_access_key_id", "_local_debug_path",
    )
    for secret in sensitive_strings:
        assert secret not in rendered, f"leaked into rendered output: {secret!r}"
    assert str(snapshot["primary"]) .find("hunter2") == -1
    for secret in sensitive_strings:
        assert secret not in str(snapshot), f"leaked into snapshot dict: {secret!r}"


def test_absolute_artifact_path_is_relativized_or_redacted_never_leaked_raw(tmp_path):
    """Codex review (PR #429): the rendered snapshot must never contain an
    absolute local filesystem path. Today's config only stores relative
    paths, but the renderer must defend against a future absolute one."""
    mod = _load_module()
    fake_repo_root = tmp_path / "repo"
    strategy_dir = fake_repo_root / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)

    # An artifact path INSIDE the fake repo root, given as absolute:
    inside_artifact = strategy_dir / "artifacts" / "prod" / "primary.json"
    _write_json(inside_artifact, {"trained_date": "2026-06-30"})

    row_inside = mod._describe_model(
        role="primary", kind="xgb", artifact_rel=str(inside_artifact),
        strategy_dir=strategy_dir, repo_root=fake_repo_root,
    )
    assert not Path(row_inside["artifact_path"]).is_absolute()
    assert str(fake_repo_root) not in row_inside["artifact_path"]

    # An artifact path OUTSIDE the fake repo root entirely (e.g. some other
    # local machine path) must be redacted to just its basename, never
    # echoed as a full local path:
    outside_artifact = tmp_path / "elsewhere" / "model.json"
    outside_artifact.parent.mkdir(parents=True)
    _write_json(outside_artifact, {"trained_date": "2026-06-30"})

    row_outside = mod._describe_model(
        role="primary", kind="xgb", artifact_rel=str(outside_artifact),
        strategy_dir=strategy_dir, repo_root=fake_repo_root,
    )
    assert str(tmp_path) not in row_outside["artifact_path"]
    assert row_outside["artifact_path"] == "<redacted-external-path>/model.json"


def test_check_mode_passes_after_regeneration(tmp_path):
    mod = _load_module()
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    strategy_dir.mkdir(parents=True)
    config_path = strategy_dir / "strategy_config.json"
    _write_json(config_path, {
        "ranking": {"panel_scoring": {"kind": "xgb", "artifact_path": None,
                                       "shadow_models": []}},
        "watchlist": [],
    })
    output_path = tmp_path / "snapshot.md"

    assert mod.main(["--strategy-config", str(config_path), "--output", str(output_path)]) == 0
    assert mod.main([
        "--strategy-config", str(config_path),
        "--output", str(output_path),
        "--check",
    ]) == 0
