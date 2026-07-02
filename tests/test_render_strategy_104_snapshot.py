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
