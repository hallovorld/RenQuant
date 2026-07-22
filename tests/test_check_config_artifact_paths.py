"""Unit tests for scripts/check_config_artifact_paths.py.

The two mandated poles:
  * a config with a broken ``../../`` shadow path FAILS (the incident class);
  * a config with all-resolving, IDENTIFIED primary/calibrator/shadow paths
    PASSES.

The gate CALLS the canonical renquant-pipeline #211 resolver
(``shadow_health.resolve_artifact_identity``) as a single injected dependency.
These unit tests are hermetic: they inject a FAITHFUL fake of that contract
(same absolute -> strategy_dir -> repo_root resolution + ``sha256:<16hex>``
content digest) so the gate LOGIC is exercised without the pipeline installed.
``test_real_canonical_contract_*`` uses the real module when it is importable
(skipped on a bare runner; exercised in the verify-pinned-paths CI job / locally).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import namedtuple
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "check_config_artifact_paths.py"
)
REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "config_artifact_gate_registry.json"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "check_config_artifact_paths_for_test", SCRIPT
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load_module()


# ── a faithful fake of #211's resolve_artifact_identity ──────────────────────

_FakeIdentity = namedtuple(
    "ArtifactIdentity",
    "ref resolved resolved_path source content_sha256 error",
)


def _fake_resolve_identity(raw, strategy_dir, data_root):
    """Mirror kernel.artifact_resolver: absolute -> strategy_dir -> repo_root,
    existence-checked, content_sha256 = 'sha256:<16hex>' of the resolved file."""
    p = Path(raw)
    if p.is_absolute():
        candidates = [(p, "absolute")]
    else:
        candidates = [
            (Path(strategy_dir) / p, "strategy_dir"),
            (Path(data_root) / p, "repo_root"),
        ]
    for cand, source in candidates:
        if cand.is_file():
            sha = "sha256:" + hashlib.sha256(cand.read_bytes()).hexdigest()[:16]
            return _FakeIdentity(raw, True, str(cand), source, sha, None)
    return _FakeIdentity(raw, False, None, "unresolved", None, f"not found: {raw}")


def _fake_contract():
    return mod.ArtifactContract(
        resolve_identity=_fake_resolve_identity,
        norm_digest=None,
        backend="test-fake(#211-faithful)",
    )


def _fake_import_canonical():
    """Stand in for _import_canonical() so default_contract()/main() run their
    real code paths against the faithful fake — no pipeline needed."""
    return _fake_resolve_identity, None, "test-fake(#211-faithful)"


# ── fixtures: real-shaped artifacts WITH identity metadata ───────────────────


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _scorer_doc(trained_date: str = "2026-06-21") -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "trained_date": trained_date,
        "lookahead_days": 60,
        "config_fingerprint": "sha256:deadbeefcafef00d",
        "feature_cols": ["a", "b", "c"],
    }


def _calibrator_doc(trained_date: str = "2026-05-21") -> dict:
    return {
        "kind": "global_panel_calibration",
        "trained_date": trained_date,
        "version": 1,
    }


def _make_tree(tmp_path: Path):
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    data_root = tmp_path
    _write_json(
        strategy_dir / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json",
        _scorer_doc(),
    )
    _write_json(
        strategy_dir / "artifacts" / "prod" / "panel-rank-calibration.json",
        _calibrator_doc(),
    )
    return strategy_dir, data_root


def _base_config(shadow_path: str) -> dict:
    return {
        "ranking": {
            "panel_scoring": {
                "enabled": True,
                "kind": "xgb",
                "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
                "global_calibration": {
                    "enabled": True,
                    "artifact_path": "artifacts/prod/panel-rank-calibration.json",
                },
                "shadow_models": [
                    {
                        "name": "hf_patchtst_previous_primary",
                        "kind": "hf_patchtst",
                        "artifact_path": shadow_path,
                    }
                ],
            }
        },
        "panel_ltr": {
            "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
        },
    }


def _good_shadow(data_root: Path) -> str:
    rel = "store/patchtst_shadow/seed_44/model.pt"
    (data_root / "store" / "patchtst_shadow" / "seed_44").mkdir(parents=True)
    (data_root / rel).write_bytes(b"\x00\x01")
    _write_json(data_root / (rel + ".metadata.json"), _scorer_doc())
    return rel


# ── the two mandated fixtures ────────────────────────────────────────────────


def test_broken_dotdot_shadow_path_fails(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)
    broken = (
        "../../artifacts/patchtst_shadow/"
        "pt07_strict_trainfit_embargo60_20260522/seed_44/"
        "hf_patchtst_all_seed44_model.pt"
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _base_config(broken))

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    failures = [r for r in results if not r.ok]
    assert failures, "expected the broken ../../ shadow path to FAIL"
    shadow_fail = next(r for r in failures if r.kind == "shadow")
    assert ".." in shadow_fail.raw
    assert "escapes the repo" in shadow_fail.reason
    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 1


def test_all_resolving_identified_paths_pass(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)
    good = _good_shadow(data_root)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _base_config(good))

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    failures = [r for r in results if not r.ok]
    assert not failures, f"expected all paths to resolve+identify, got: {failures}"
    assert {r.kind for r in results} == {"primary", "calibrator", "shadow"}
    assert len(results) == 4
    # canonical content_sha256 + provenance carried through on OK rows.
    assert all("content=sha256:" in r.detail for r in results)

    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 0


# ── point 3: identity + swap-detection, not just existence ───────────────────


def test_resolvable_scorer_without_metadata_fails_closed(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_json(
        strategy_dir / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json",
        {"kind": "panel_ltr_xgboost"},  # no trained_date / fingerprint
    )
    config = _base_config(_good_shadow(data_root))
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if not r.ok)
    assert fail.kind == "primary"
    assert "missing required identity field" in fail.reason


def test_resolvable_calibrator_without_trained_date_fails_closed(
    tmp_path: Path,
) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_json(
        strategy_dir / "artifacts" / "prod" / "panel-rank-calibration.json",
        {"kind": "global_panel_calibration", "version": 1},  # no trained_date
    )
    config = _base_config("x")
    del config["ranking"]["panel_scoring"]["shadow_models"]
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if not r.ok)
    assert fail.kind == "calibrator"
    assert "trained_date" in fail.reason


def test_pinned_content_sha256_mismatch_fails(tmp_path: Path) -> None:
    """A config-pinned expected_content_sha256 that disagrees FAILS (#211)."""
    strategy_dir, data_root = _make_tree(tmp_path)
    good = _good_shadow(data_root)
    config = _base_config(good)
    config["ranking"]["panel_scoring"]["shadow_models"][0][
        "expected_content_sha256"
    ] = "sha256:0000000000000000"  # wrong on purpose
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if not r.ok)
    assert fail.kind == "shadow"
    assert "content_sha256 mismatch" in fail.reason


def test_pinned_content_sha256_match_passes(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    good = _good_shadow(data_root)
    actual = "sha256:" + hashlib.sha256(
        (data_root / good).read_bytes()
    ).hexdigest()[:16]
    config = _base_config(good)
    config["ranking"]["panel_scoring"]["shadow_models"][0][
        "expected_content_sha256"
    ] = actual
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []


# ── point 1: single injected canonical dependency ────────────────────────────


def test_default_contract_hard_fails_without_pipeline(monkeypatch) -> None:
    """With the canonical #211 module unimportable, default_contract() raises —
    the gate refuses to run on a drift-prone fallback."""
    def _boom():
        raise mod.CanonicalContractUnavailable("no pipeline")

    monkeypatch.setattr(mod, "_import_canonical", _boom)
    with pytest.raises(mod.CanonicalContractUnavailable):
        mod.default_contract()


def test_main_hard_fails_without_pipeline(tmp_path, monkeypatch) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _base_config(_good_shadow(data_root)))

    def _boom():
        raise mod.CanonicalContractUnavailable("no pipeline")

    monkeypatch.setattr(mod, "default_contract", _boom)
    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 2  # actionable hard-stop, not a silent pass


def test_injected_resolver_is_honoured(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    good = _good_shadow(data_root)
    calls: list[str] = []

    def spy_resolve(raw, sd, dr):
        calls.append(raw)
        return _fake_resolve_identity(raw, sd, dr)

    injected = mod.ArtifactContract(
        resolve_identity=spy_resolve, norm_digest=None, backend="test-spy"
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _base_config(good))

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, injected
    )
    assert good in calls  # the injected resolver was actually called
    assert [r for r in results if not r.ok] == []


# ── point 4: ../../ escape lint is deterministic (even if it resolves) ───────


def test_escape_fails_even_when_resolver_resolves_it(tmp_path: Path) -> None:
    """A ``../../`` path that the resolver WOULD resolve still FAILS on the
    escape lint (the topology-fragile incident class)."""
    strategy_dir, data_root = _make_tree(tmp_path)

    def always_resolves(raw, sd, dr):
        return _FakeIdentity(raw, True, str(tmp_path / "x"), "strategy_dir",
                             "sha256:1111111111111111", None)

    contract = mod.ArtifactContract(
        resolve_identity=always_resolves, norm_digest=None, backend="test"
    )
    config = _base_config("../../artifacts/x/model.pt")
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, contract
    )
    fail = next(r for r in results if r.kind == "shadow")
    assert not fail.ok
    assert "escapes the repo" in fail.reason


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("../../artifacts/x/model.pt", True),
        ("../artifacts/model.pt", True),
        ("artifacts/../prod/model.pt", True),
        ("artifacts/prod/model.pt", False),
        ("model.pt", False),
    ],
)
def test_escapes_repo(raw: str, expected: bool) -> None:
    assert mod.escapes_repo(raw) is expected


def test_unresolved_path_fails(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config("store/missing/model.pt")  # not created -> unresolved
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.kind == "shadow")
    assert not fail.ok
    assert "does not resolve" in fail.reason


# ── point 2: registry-driven, multi-shape ────────────────────────────────────


def test_manifest_shape_extracts_all_three_paths() -> None:
    manifest = {
        "production_primary": {
            "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
            "global_calibration": {
                "artifact_path": "artifacts/prod/panel-rank-calibration.json"
            },
        },
        "readonly_shadow": {
            "artifact_path": "../../artifacts/patchtst_shadow/x/model.pt"
        },
    }
    paths = mod.collect_paths_artifact_manifest(manifest)
    kinds = {k for _, k, _, _ in paths}
    assert kinds == {"primary", "calibrator", "shadow"}
    fields = {f for f, _, _, _ in paths}
    assert "production_primary.artifact_path" in fields
    assert "readonly_shadow.artifact_path" in fields


def test_shipped_registry_declares_expected_profiles() -> None:
    """Only profiles a real scheduled path actually loads belong here. An
    earlier round declared xgb_prod_artifact_manifest.json as required, but
    no scheduled path in this repo ever loads a file by that name (grepped
    the full checkout + git history: zero hits) -- a required profile that
    can never exist would make this gate permanently fail closed. Removed;
    the artifact_manifest shape handler stays available (test_manifest_
    shape_extracts_all_three_paths) for whenever a real one shows up."""
    profiles = mod.load_registry(REGISTRY)
    files = {p["file"] for p in profiles}
    assert {
        "strategy_config.json",
        "strategy_config.shadow.json",
        "strategy_config.shadow_a.json",
        "strategy_config.shadow_b.json",
    } <= files
    assert "xgb_prod_artifact_manifest.json" not in files
    shapes = {p["shape"] for p in profiles}
    assert shapes <= set(mod.COLLECTORS)


def test_registry_run_validates_multiple_profiles_and_skips_optional(
    tmp_path: Path,
) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    configs_dir = tmp_path / "configs"
    good = _good_shadow(data_root)
    _write_json(configs_dir / "strategy_config.json", _base_config(good))
    shadow_cfg = _base_config(good)
    shadow_cfg["ranking"]["panel_scoring"]["artifact_path"] = (
        "../../artifacts/patchtst_shadow/x/model.pt"  # ../../ as PRIMARY
    )
    _write_json(configs_dir / "strategy_config.shadow.json", shadow_cfg)
    # shadow_a/shadow_b intentionally absent -> optional skip.

    run = mod.check_registry(
        REGISTRY, configs_dir, strategy_dir, data_root, _fake_contract()
    )
    shadow_fail = next(
        r for r in run.results
        if "shadow.json" in r.config and "escapes the repo" in r.reason
    )
    assert shadow_fail.kind == "primary"
    assert any("shadow_a" in s for s in run.skipped_profiles)
    assert any("shadow_b" in s for s in run.skipped_profiles)
    assert len(run.validated_profiles) == 2


def test_required_profile_absent_fails(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    configs_dir = tmp_path / "configs"
    configs_dir.mkdir()
    run = mod.check_registry(
        REGISTRY, configs_dir, strategy_dir, data_root, _fake_contract()
    )
    assert any(
        r.kind == "profile" and "required profile missing" in r.reason
        for r in run.results
    )


# ── real canonical module (#211) when importable ─────────────────────────────


def test_real_canonical_contract_resolves_and_identifies(tmp_path: Path) -> None:
    """When renquant-pipeline #211 is importable, the REAL contract resolves +
    stamps content_sha256. Skipped on a bare runner (unit CI); exercised in the
    verify-pinned-paths job and locally."""
    pytest.importorskip("renquant_pipeline.kernel.panel_pipeline.shadow_health")
    contract = mod.default_contract()
    assert "shadow_health" in contract.backend
    strategy_dir, data_root = _make_tree(tmp_path)
    good = _good_shadow(data_root)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _base_config(good))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, contract
    )
    assert [r for r in results if not r.ok] == []
    assert all("content=sha256:" in r.detail for r in results)
