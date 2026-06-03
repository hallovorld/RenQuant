from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_module():
    path = REPO / "scripts" / "run_kelly_sigma_horizon_ab.py"
    spec = importlib.util.spec_from_file_location("run_kelly_sigma_horizon_ab", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_seeds_accepts_count_and_list() -> None:
    mod = _load_module()

    assert mod.parse_seeds("5") == (0, 1, 2, 3, 4)
    assert mod.parse_seeds("3,7,11") == (3, 7, 11)


def test_build_variants_pairs_real_ab_and_aa_resplit(tmp_path: Path) -> None:
    mod = _load_module()
    base = tmp_path / "base.json"
    treatment = tmp_path / "treatment.json"

    variants = mod.build_variants(
        base_config_path=base,
        treatment_config_path=treatment,
        seeds=(0, 1, 2, 3, 4),
        aa_seed_offset=100,
    )

    assert [variant.name for variant in variants] == [
        "A_golden",
        "B_sigma_horizon_60",
        "AA_golden_resplit",
    ]
    assert variants[0].seeds == variants[1].seeds == (0, 1, 2, 3, 4)
    assert variants[2].config_path == base
    assert variants[2].seeds == (100, 101, 102, 103, 104)


def test_build_plan_keeps_treatment_config_in_output_dir(tmp_path: Path) -> None:
    mod = _load_module()
    args = Namespace(
        base_config=str(tmp_path / "base.json"),
        treatment_config="",
        sigma_horizon_days=60,
        start="2024-01-02",
        end="2026-03-28",
        initial_cash=100_000.0,
        seeds="0,1,2,3,4",
        aa_seed_offset=1000,
        output_dir=str(tmp_path / "out"),
        manifest_path="artifacts/sim/walkforward_manifest.json",
        placebo_json=[],
        execute=False,
    )

    plan = mod.build_plan(args)

    assert plan["mode"] == "dry_run"
    assert plan["treatment_config_path"].startswith(str(tmp_path / "out"))
    assert plan["mandatory_checks"]["multi_seed_floor"] == 5
    assert plan["mandatory_checks"]["real_ab"] == ["A_golden", "B_sigma_horizon_60"]
    assert plan["run_overrides"]["manifest_path"] == "artifacts/sim/walkforward_manifest.json"


def test_default_base_config_manifest_preflight_requires_calibrators() -> None:
    mod = _load_module()
    config_path = mod.STRATEGY_DIR / mod.DEFAULT_BASE_CONFIG
    config = json.loads(config_path.read_text())

    try:
        mod.validate_walkforward_manifest(config, mod.STRATEGY_DIR)
    except FileNotFoundError as exc:
        assert "calibrator_uri missing" in str(exc)
    else:
        raise AssertionError("expected default archived manifest to require calibrators")


def test_manifest_override_is_runtime_only() -> None:
    mod = _load_module()
    config = {"walkforward": {"enabled": True, "manifest_path": "missing.json"}}

    mod.apply_run_overrides(
        config,
        manifest_path="artifacts/sim/walkforward_manifest.json",
    )

    assert config["walkforward"]["manifest_path"] == "artifacts/sim/walkforward_manifest.json"


def test_walkforward_manifest_preflight_rejects_missing_calibrator(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "panel-ltr.json"
    artifact.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01",
            "trained_date": "2024-01-02",
            "artifact_uri": "panel-ltr.json",
        }]
    }))

    try:
        mod.preflight_walkforward_manifest(manifest)
    except FileNotFoundError as exc:
        assert "calibrator_uri missing" in str(exc)
    else:
        raise AssertionError("expected missing calibrator preflight failure")


def test_walkforward_manifest_preflight_uses_manifest_relative_paths(tmp_path: Path) -> None:
    mod = _load_module()
    manifest_dir = tmp_path / "artifacts" / "sim"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01",
            "trained_date": "2024-01-02",
            "artifact_uri": "artifacts/walkforward/fold/panel-ltr.json",
            "calibrator_uri": "artifacts/walkforward/fold/panel-rank-calibration.json",
        }]
    }))

    try:
        mod.preflight_walkforward_manifest(manifest)
    except FileNotFoundError as exc:
        assert "artifacts/sim/artifacts/walkforward" in str(exc)
    else:
        raise AssertionError("expected manifest-relative missing path failure")


def test_materialize_manifest_rewrites_strategy_relative_uris(tmp_path: Path) -> None:
    mod = _load_module()
    strategy_dir = tmp_path / "strategy"
    fold = strategy_dir / "artifacts" / "walkforward_v2" / "2024-01-01"
    fold.mkdir(parents=True)
    (fold / "panel-ltr.json").write_text("{}")
    (fold / "panel-rank-calibration.json").write_text("{}")
    manifest_dir = strategy_dir / "artifacts" / "sim"
    manifest_dir.mkdir(parents=True)
    manifest = manifest_dir / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01",
            "trained_date": "2024-01-02",
            "artifact_uri": "artifacts/walkforward_v2/2024-01-01/panel-ltr.json",
            "calibrator_uri": "artifacts/walkforward_v2/2024-01-01/panel-rank-calibration.json",
        }]
    }))

    resolved = mod.materialize_manifest_for_runner(
        manifest,
        out_dir=tmp_path / "out",
        strategy_dir=strategy_dir,
    )

    payload = json.loads(resolved.read_text())
    row = payload["retrains"][0]
    assert Path(row["artifact_uri"]).is_absolute()
    assert Path(row["calibrator_uri"]).is_absolute()
    assert mod.preflight_walkforward_manifest(resolved)["entries_with_calibrator"] == 1


def test_walkforward_manifest_preflight_rejects_config_mismatch(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "panel-ltr.json"
    artifact.write_text(json.dumps({
        "config_fingerprint": "sha256:notlive",
        "config_fingerprint_fields": {"watchlist": ["OLD"]},
    }))
    calibrator = tmp_path / "panel-rank-calibration.json"
    calibrator.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01",
            "trained_date": "2024-01-02",
            "artifact_uri": artifact.name,
            "calibrator_uri": calibrator.name,
        }]
    }))
    config = {
        "_strategy_dir": str(mod.STRATEGY_DIR),
        "watchlist": ["AAA"],
        "ranking": {"panel_scoring": {"strict_config_consistency": True}},
    }

    try:
        mod.preflight_walkforward_manifest(manifest, config=config)
    except ValueError as exc:
        message = str(exc)
        assert "walkforward artifact config preflight failed" in message
        assert "Config-consistency MISMATCH" in message
        assert "sha256:notlive" in message
    else:
        raise AssertionError("expected artifact config mismatch preflight failure")


def test_walkforward_manifest_preflight_respects_non_strict_config(tmp_path: Path) -> None:
    mod = _load_module()
    artifact = tmp_path / "panel-ltr.json"
    artifact.write_text(json.dumps({
        "config_fingerprint": "sha256:notlive",
        "config_fingerprint_fields": {"watchlist": ["OLD"]},
    }))
    calibrator = tmp_path / "panel-rank-calibration.json"
    calibrator.write_text("{}")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [{
            "cutoff_date": "2024-01-01",
            "trained_date": "2024-01-02",
            "artifact_uri": artifact.name,
            "calibrator_uri": calibrator.name,
        }]
    }))
    config = {
        "_strategy_dir": str(mod.STRATEGY_DIR),
        "watchlist": ["AAA"],
        "ranking": {"panel_scoring": {"strict_config_consistency": False}},
    }

    result = mod.preflight_walkforward_manifest(manifest, config=config)

    assert result["entries_with_calibrator"] == 1
    assert result["artifact_config_checked"] == 0
    assert result["strict_config_consistency"] is False


def test_promotion_verdict_blocks_without_placebo() -> None:
    mod = _load_module()
    metrics = {
        "A_golden": {"apy_mean": 0.10, "sharpe_mean": 1.0},
        "B_sigma_horizon_60": {
            "apy_mean": 0.12,
            "sharpe_mean": 1.2,
            "dsr": 0.7,
            "pbo": 0.4,
        },
        "AA_golden_resplit": {"apy_mean": 0.101, "sharpe_mean": 1.01},
    }

    verdict = mod.promotion_verdict(metrics, {"provided": False, "passed": False})

    assert verdict["tier3_ready"] is False
    assert "shuffle/time-shift placebo evidence missing" in verdict["blocked_reasons"]


def test_promotion_verdict_passes_synthetic_tier3() -> None:
    mod = _load_module()
    metrics = {
        "A_golden": {"apy_mean": 0.10, "sharpe_mean": 1.0},
        "B_sigma_horizon_60": {
            "apy_mean": 0.12,
            "sharpe_mean": 1.2,
            "dsr": 0.7,
            "pbo": 0.4,
        },
        "AA_golden_resplit": {"apy_mean": 0.101, "sharpe_mean": 1.01},
    }
    placebo = {"provided": True, "passed": True, "items": []}

    verdict = mod.promotion_verdict(metrics, placebo)

    assert verdict["tier3_ready"] is True
    assert verdict["blocked_reasons"] == []
    assert verdict["deltas"]["sharpe_lift"] == 0.19999999999999996


def test_load_placebo_evidence_reads_manifest_diagnostic(tmp_path: Path) -> None:
    mod = _load_module()
    path = tmp_path / "placebo.json"
    path.write_text(json.dumps({
        "interpretation": {
            "promotion_evidence": True,
            "aligned_real_60_ic": 0.05,
            "placebo_60_ic": 0.01,
            "label_autocorr_60_ic": 0.02,
        }
    }))

    evidence = mod.load_placebo_evidence([str(path)])

    assert evidence["provided"] is True
    assert evidence["passed"] is True
    assert evidence["items"][0]["placebo_60_ic"] == 0.01


def test_bootstrap_subrepo_imports_adds_runtime_srcs(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    repo = tmp_path / "RenQuant"
    repo.mkdir()
    runtime = tmp_path / "runtime"
    subrepos = []
    for name in ("renquant-common", "renquant-pipeline"):
        path = runtime / name
        (path / "src").mkdir(parents=True)
        subrepos.append({"name": name, "local_path": str(path)})
    (repo / "subrepos.lock.json").write_text(json.dumps({"subrepos": subrepos}))
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_ASSEMBLY_DIR", raising=False)
    monkeypatch.setattr(sys, "path", sys.path.copy())

    root = mod.bootstrap_subrepo_imports(repo)

    assert root == runtime.resolve()
    assert str(runtime / "renquant-pipeline" / "src") in sys.path
    assert str(runtime / "renquant-common" / "src") in sys.path
