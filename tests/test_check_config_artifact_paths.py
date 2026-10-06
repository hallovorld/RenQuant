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


# ── 2026-09-03: declared live-mutated served pair (untracked) ────────────────


def _declare_live_mutated(data_root: Path, paths: list[str], *, strategy_dir="backtesting/renquant_104",
                          schema_version: int = 1) -> None:
    _write_json(data_root / "deploy" / "live_mutated_prod_artifacts.json", {
        "schema_version": schema_version, "strategy_dir": strategy_dir,
        "artifacts": [{"path": p, "writers": ["test"]} for p in paths],
    })


def _run_with_pair_absent(tmp_path: Path):
    strategy_dir, data_root = _make_tree(tmp_path)
    for name in ("panel-ltr.alpha158_fund.json", "panel-rank-calibration.json"):
        (strategy_dir / "artifacts" / "prod" / name).unlink()   # fresh checkout: pair absent
    config = _base_config(_good_shadow(data_root))
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    return mod.check_config(config_path, "strategy_config", strategy_dir, data_root, _fake_contract())


def test_declared_live_mutated_pair_absent_passes_with_info(tmp_path: Path) -> None:
    """Hosted CI has no live tree: the untracked served pair is absent BY
    DESIGN and, being declared, reports INFO — every field that points at
    it (primary, panel_ltr, calibrator), while the shadow still runs full."""
    _declare_live_mutated(tmp_path, ["artifacts/prod/panel-ltr.alpha158_fund.json",
                                     "artifacts/prod/panel-rank-calibration.json"])
    results = _run_with_pair_absent(tmp_path)
    pair = [r for r in results if r.kind in ("primary", "calibrator")]
    assert len(pair) == 3 and all(r.ok for r in pair), [(r.field, r.reason) for r in pair]
    assert all("INFO: live-mutated served artifact absent" in r.detail for r in pair)
    shadow = next(r for r in results if r.kind == "shadow")
    assert shadow.ok and "INFO: live-mutated" not in shadow.detail


def test_undeclared_absent_pair_still_fails(tmp_path: Path) -> None:
    results = _run_with_pair_absent(tmp_path)
    pair = [r for r in results if r.kind in ("primary", "calibrator")]
    assert pair and all((not r.ok) and "does not resolve" in r.reason for r in pair)


def test_declaration_waives_only_the_declared_path(tmp_path: Path) -> None:
    _declare_live_mutated(tmp_path, ["artifacts/prod/panel-ltr.alpha158_fund.json"])
    results = _run_with_pair_absent(tmp_path)
    cal = next(r for r in results if r.kind == "calibrator")
    assert not cal.ok and "does not resolve" in cal.reason
    assert all(r.ok for r in results if r.kind == "primary")


def test_declaration_for_another_strategy_dir_or_schema_waives_nothing(tmp_path: Path) -> None:
    pair = ["artifacts/prod/panel-ltr.alpha158_fund.json", "artifacts/prod/panel-rank-calibration.json"]
    other_dir = tmp_path / "other_dir"
    _declare_live_mutated(other_dir, pair, strategy_dir="backtesting/other_strategy")
    assert all(not r.ok for r in _run_with_pair_absent(other_dir) if r.kind in ("primary", "calibrator"))
    other_schema = tmp_path / "other_schema"
    _declare_live_mutated(other_schema, pair, schema_version=2)
    assert all(not r.ok for r in _run_with_pair_absent(other_schema) if r.kind in ("primary", "calibrator"))


def test_declared_pair_present_is_checked_in_full(tmp_path: Path) -> None:
    """On the serving machine the declaration changes nothing: a present
    pair goes through identity + provenance exactly as before (here the
    fixture scorer lacks nothing, so it passes WITHOUT the INFO detail)."""
    _declare_live_mutated(tmp_path, ["artifacts/prod/panel-ltr.alpha158_fund.json",
                                     "artifacts/prod/panel-rank-calibration.json"])
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config(_good_shadow(data_root))
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(config_path, "strategy_config", strategy_dir, data_root, _fake_contract())
    pair = [r for r in results if r.kind in ("primary", "calibrator")]
    assert pair and all(r.ok and "INFO: live-mutated" not in r.detail for r in pair)


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


# ── point 5: momentum ledger pointer (slice 4c; model#197 amendment 2) ───────
#
# Fixture recipes mirror renquant-model's OWNED definitions (ledger.py
# row_sha256_of / train.py content_sha256_of): canonical JSON =
# ``sort_keys=True, separators=(",", ":"), allow_nan=False``; row_sha over the
# row WITHOUT row_sha; content_sha256 over the artifact WITHOUT content_sha256.


def _canon_sha(body: dict) -> str:
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"),
                   allow_nan=False).encode("utf-8")
    ).hexdigest()


def _momentum_artifact_doc(cutoff: str) -> dict:
    doc = {
        "kind": "momentum_residual_v0",
        "schema_version": 1,
        "cutoff_date": cutoff,
        "params": {"params_version": "v0", "window": 231, "skip": 21},
        "scores": {"AAA": 0.11, "BBB": -0.04},
    }
    doc["content_sha256"] = _canon_sha(doc)  # over the doc sans content_sha256
    return doc


def _ledger_row(index: int, prev_sha, cutoff: str, artifact_sha: str) -> dict:
    row = {
        "row_index": index,
        "prev_row_sha": prev_sha,
        "appended_at_utc": f"2026-08-0{index + 1}T12:00:00+00:00",
        "kind": "momentum_residual_v0",
        "cutoff_date": cutoff,
        "params_version": "v0",
        "artifact_content_sha256": artifact_sha,
    }
    row["row_sha"] = _canon_sha(row)  # over the row sans row_sha
    return row


MOMENTUM_LEDGER_REL = "artifacts/momentum/momentum_artifact_ledger.jsonl"


def _write_momentum_publish_set(
    strategy_dir: Path,
    cutoffs: tuple[str, ...] = ("2026-07-25", "2026-08-01"),
    *,
    write_tail_artifact: bool = True,
    root_subdir: str = "momentum",
) -> Path:
    """Build the momentum_train_run.py publish layout under the strategy dir:
    ``artifacts/<root_subdir>/<cutoff>/momentum_residual_v0.json`` per cutoff
    plus the chained ledger beside them. Returns the ledger path."""
    root = strategy_dir / "artifacts" / root_subdir
    rows: list[dict] = []
    prev = None
    for i, cutoff in enumerate(cutoffs):
        doc = _momentum_artifact_doc(cutoff)
        if write_tail_artifact or i < len(cutoffs) - 1:
            _write_json(root / cutoff / "momentum_residual_v0.json", doc)
        row = _ledger_row(i, prev, cutoff, doc["content_sha256"])
        rows.append(row)
        prev = row["row_sha"]
    ledger = root / "momentum_artifact_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(
            json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n"
            for r in rows
        ),
        encoding="utf-8",
    )
    return ledger


def _momentum_entry(*, pending: bool = False) -> dict:
    entry = {
        "name": "momentum_residual_v0_shadow",
        "kind": "momentum_residual",
        "artifact_path": MOMENTUM_LEDGER_REL,
    }
    if pending:
        entry["_2026_08_02_pending_first_artifact"] = (
            "bounded pending guard — first publish rides the slice-5 batch"
        )
    return entry


def _config_with_momentum(
    data_root: Path, *, pending: bool = False
) -> dict:
    """The two-lane shadow state s104#77 ships: the classic blend-era entry
    PLUS the momentum ledger pointer."""
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        _momentum_entry(pending=pending)
    )
    return config


def test_ledger_pointer_present_valid_passes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root))

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []
    momentum = next(r for r in results if r.raw.endswith(".jsonl"))
    assert momentum.kind == "shadow"
    assert "ledger_rows=2" in momentum.detail
    assert "chain=verified" in momentum.detail
    assert "tail_cutoff=2026-08-01" in momentum.detail

    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 0


def test_ledger_pointer_chain_tampered_fails_naming_row(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    ledger = _write_momentum_publish_set(strategy_dir)
    # Rewrite history: edit row 1's cutoff_date WITHOUT resealing row_sha.
    lines = ledger.read_text(encoding="utf-8").splitlines()
    row1 = json.loads(lines[1])
    row1["cutoff_date"] = "2099-01-01"
    lines[1] = json.dumps(row1, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "chain verification FAILED" in fail.reason
    assert "row 1" in fail.reason  # the defect is NAMED at its row
    assert "does not recompute" in fail.reason


def test_ledger_pointer_tail_artifact_missing_fails(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir, write_tail_artifact=False)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "does not exist beside the ledger" in fail.reason
    assert "tail row 1" in fail.reason
    assert "2026-08-01" in fail.reason


def test_ledger_pointer_tail_artifact_selfsha_mismatch_fails(
    tmp_path: Path,
) -> None:
    """Tail artifact edited after writing: self-carried content_sha256 stale."""
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    art = (strategy_dir / "artifacts" / "momentum" / "2026-08-01"
           / "momentum_residual_v0.json")
    doc = json.loads(art.read_text(encoding="utf-8"))
    doc["scores"]["AAA"] = 9.99  # edited; content_sha256 left stale
    _write_json(art, doc)

    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "self-carried content_sha256" in fail.reason
    assert "does not recompute" in fail.reason


def test_ledger_pointer_tail_artifact_ledger_sha_mismatch_fails(
    tmp_path: Path,
) -> None:
    """Tail artifact internally consistent but NOT the bytes the ledger row
    vouches for (a swapped artifact with a recomputed self-sha)."""
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    art = (strategy_dir / "artifacts" / "momentum" / "2026-08-01"
           / "momentum_residual_v0.json")
    doc = json.loads(art.read_text(encoding="utf-8"))
    doc["scores"]["AAA"] = 9.99
    del doc["content_sha256"]
    doc["content_sha256"] = _canon_sha(doc)  # reseal: self-consistent again
    _write_json(art, doc)

    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "the ledger does not vouch for these bytes" in fail.reason
    assert "tail row 1" in fail.reason


def test_ledger_absent_with_marker_info_passes(tmp_path: Path, monkeypatch) -> None:
    """ABSENT ledger + the s104#77 bounded pending marker -> INFO pass (the
    designed pre-batch state)."""
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)  # no publish set written
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root, pending=True))

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []
    momentum = next(r for r in results if r.raw.endswith(".jsonl"))
    assert momentum.ok
    assert momentum.detail.startswith("INFO: pending first artifact")
    assert "the designed pre-batch state" in momentum.detail
    assert "_2026_08_02_pending_first_artifact" in momentum.detail

    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 0


def test_ledger_absent_without_marker_fails_closed(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)  # no publish set written
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root, pending=False))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "does not resolve" in fail.reason
    assert "neither a *_machine_produced_ledger" in fail.reason
    assert "fail-closed" in fail.reason


def test_ledger_pointer_empty_ledger_fails(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    ledger = strategy_dir / "artifacts" / "momentum" / "momentum_artifact_ledger.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("", encoding="utf-8")  # exists, zero rows
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_momentum(data_root, pending=True))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok  # empty-but-present is NOT the designed pending state
    assert "EMPTY" in fail.reason


def test_ledger_pointer_rejects_expected_pins(tmp_path: Path) -> None:
    """A config-pinned expected sha on an append-only ledger pointer is
    refused (fail closed), never silently ignored."""
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config = _config_with_momentum(data_root)
    config["ranking"]["panel_scoring"]["shadow_models"][1][
        "expected_content_sha256"
    ] = "sha256:0000000000000000"
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.raw.endswith(".jsonl"))
    assert not fail.ok
    assert "not supported on a ledger pointer" in fail.reason


def test_pending_marker_does_not_rescue_classic_entries(tmp_path: Path) -> None:
    """The marker only applies to ``.jsonl`` ledger pointers: an unresolved
    CLASSIC shadow artifact carrying the marker still FAILS exactly as before
    (no new bypass for dated artifacts)."""
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config("store/missing/model.pt")  # unresolved classic path
    config["ranking"]["panel_scoring"]["shadow_models"][0][
        "_2026_08_02_pending_first_artifact"
    ] = "marker on a NON-ledger entry — must not rescue it"
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.kind == "shadow")
    assert not fail.ok
    assert "does not resolve" in fail.reason


def test_classic_entries_untouched_by_ledger_rule(tmp_path: Path) -> None:
    """Regression pin for slice 4c: a config with ONLY classic entries yields
    byte-identical results whether or not the ledger rule exists — same kinds,
    same pass set, same reasons on the broken pole."""
    strategy_dir, data_root = _make_tree(tmp_path)
    good_config = _base_config(_good_shadow(data_root))
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, good_config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []
    assert {r.kind for r in results} == {"primary", "calibrator", "shadow"}
    assert len(results) == 4
    assert all("content=sha256:" in r.detail for r in results)

    broken = _base_config("../../artifacts/x/model.pt")
    _write_json(config_path, broken)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    fail = next(r for r in results if r.kind == "shadow")
    assert not fail.ok
    assert "escapes the repo" in fail.reason


def test_non_momentum_jsonl_with_pending_marker_fails_closed(tmp_path: Path) -> None:
    """#550 regression: the ledger-pointer branch is a momentum-only contract.
    An unrelated future JSONL shadow model carrying a pending marker must NOT
    ride the #549 exception through the identity gate."""
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        {
            "name": "future_model_shadow",
            "kind": "future_model",
            "artifact_path": "artifacts/future/future_ledger.jsonl",
            "_2026_09_01_pending_first_artifact": "should not rescue this",
        }
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert bad[0].raw == "artifacts/future/future_ledger.jsonl"
    assert "restricted to the momentum contract" in bad[0].reason
    assert "fails closed" in bad[0].reason


def test_momentum_kind_with_wrong_ledger_path_fails_closed(tmp_path: Path) -> None:
    """#550 regression: even the momentum kind must point at the exact
    published ledger reference — a typo'd path fails closed instead of
    inheriting the pending-marker admission."""
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        {
            "name": "momentum_residual_v0_shadow",
            "kind": "momentum_residual",
            "artifact_path": "artifacts/momentum/momentum_artifact_ledgr.jsonl",
            "_2026_08_02_pending_first_artifact": "typo'd path must not pass",
        }
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert "restricted to the momentum contract" in bad[0].reason


def test_machine_produced_marker_admits_an_absent_ledger_as_info(tmp_path: Path) -> None:
    """s104#78 follow-up: the momentum ledger is run-surface state, never
    committed — hosted runners cannot resolve it BY DESIGN. The
    machine-produced marker downgrades exactly that case to INFO."""
    strategy_dir, data_root = _make_tree(tmp_path)   # no ledger written
    config = _base_config(_good_shadow(data_root))
    entry = _momentum_entry()
    entry["_2026_08_02_machine_produced_ledger"] = (
        "run-surface state published by the weekly job; unresolvable off the "
        "serving machine by design")
    config["ranking"]["panel_scoring"]["shadow_models"].append(entry)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []
    momentum = next(r for r in results if r.raw.endswith(".jsonl"))
    assert "machine-produced ledger absent" in momentum.detail


def test_machine_produced_marker_does_not_rescue_non_momentum_jsonl(tmp_path: Path) -> None:
    """#554's narrowing fires FIRST: the marker cannot smuggle an unrelated
    JSONL entry past the contract check."""
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        {
            "name": "future_model_shadow",
            "kind": "future_model",
            "artifact_path": "artifacts/future/future_ledger.jsonl",
            "_2026_09_01_machine_produced_ledger": "must not rescue this",
        }
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert "restricted to the momentum contract" in bad[0].reason


def test_machine_produced_marker_does_not_skip_verification_when_ledger_resolves(
    tmp_path: Path, monkeypatch
) -> None:
    """On the serving machine the ledger exists — the marker must be inert
    there: full chain verification still runs (a tampered chain still fails)."""
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)
    ledger = _write_momentum_publish_set(strategy_dir)
    # tamper with row 0
    rows = ledger.read_text().strip().splitlines()
    row0 = json.loads(rows[0]); row0["artifact_content_sha256"] = "sha256:" + "0" * 16
    rows[0] = json.dumps(row0, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")
    config = _base_config(_good_shadow(data_root))
    entry = _momentum_entry()
    entry["_2026_08_02_machine_produced_ledger"] = "inert when the ledger resolves"
    config["ranking"]["panel_scoring"]["shadow_models"].append(entry)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert "chain verification FAILED" in bad[0].reason


# --- RenQuant#561: the s104#84 FAST lane's bounded pending admission -------

FAST_LEDGER_REL = "artifacts/momentum_fast/momentum_artifact_ledger.jsonl"


def _fast_entry(*, pending: bool = True) -> dict:
    entry = {
        "name": "momentum_fast_v1_shadow",
        "kind": "momentum_residual",
        "artifact_path": FAST_LEDGER_REL,
    }
    if pending:
        entry["_2026_08_03_pending_first_artifact"] = (
            "declared dormant state — first fast artifact rides the arming "
            "batch (s104#84)"
        )
    return entry


def test_fast_ledger_pending_marker_admits_absent_ledger_as_info(
    tmp_path: Path, monkeypatch
) -> None:
    """The exact s104#84 dormant state: fast path + pending marker + ledger
    absent -> INFO pass, so the arming pin batch can deploy."""
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)  # no fast publish set
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(_fast_entry())
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    assert [r for r in results if not r.ok] == []
    fast = next(r for r in results if r.raw == FAST_LEDGER_REL)
    assert fast.ok
    assert fast.detail.startswith("INFO: pending first artifact")
    assert "_2026_08_03_pending_first_artifact" in fast.detail

    rc = mod.main(
        [
            str(config_path), "--shape", "strategy_config",
            "--strategy-dir", str(strategy_dir), "--data-root", str(data_root),
        ]
    )
    assert rc == 0


def test_fast_ledger_without_pending_marker_fails_closed(tmp_path: Path) -> None:
    """Marker removed (the post-first-publish state): the bounded admission
    ends and the gate fails closed until the full fast contract is widened
    here in a reviewed change."""
    strategy_dir, data_root = _make_tree(tmp_path)
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        _fast_entry(pending=False)
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert bad[0].raw == FAST_LEDGER_REL
    assert "admitted ONLY while" in bad[0].reason
    assert "reviewed change" in bad[0].reason


def test_fast_ledger_without_marker_fails_even_when_ledger_resolves(
    tmp_path: Path,
) -> None:
    """A valid, chain-verified fast publish set does NOT substitute for the
    reviewed widening: with the marker gone, even a resolvable fast ledger
    fails closed (the admission is the PENDING state, not the contract)."""
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir, root_subdir="momentum_fast")
    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(
        _fast_entry(pending=False)
    )
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert bad[0].raw == FAST_LEDGER_REL
    assert "admitted ONLY while" in bad[0].reason


def test_fast_ledger_with_marker_still_verifies_when_it_resolves(
    tmp_path: Path, monkeypatch
) -> None:
    """If the fast ledger appears while the marker is still on the entry
    (machine published before the s104 marker-removal change), the marker
    does not skip verification: a tampered chain still fails."""
    monkeypatch.setattr(mod, "_import_canonical", _fake_import_canonical)
    strategy_dir, data_root = _make_tree(tmp_path)
    ledger = _write_momentum_publish_set(strategy_dir, root_subdir="momentum_fast")
    rows = ledger.read_text().strip().splitlines()
    row0 = json.loads(rows[0])
    row0["artifact_content_sha256"] = "sha256:" + "0" * 16
    rows[0] = json.dumps(row0, sort_keys=True, separators=(",", ":"))
    ledger.write_text("\n".join(rows) + "\n", encoding="utf-8")

    config = _base_config(_good_shadow(data_root))
    config["ranking"]["panel_scoring"]["shadow_models"].append(_fast_entry())
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)

    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    bad = [r for r in results if not r.ok]
    assert len(bad) == 1
    assert bad[0].raw == FAST_LEDGER_REL
    assert "chain verification FAILED" in bad[0].reason


# --- GOAL-9 (RQ#574 review): PRIMARY ledger-served blend component ---------------
#
# The full-book z-blend puts the slow-momentum ledger at panel_scoring
# components[1]. The gate admits it under the SAME ledger contract as the
# shadow branch (chain + tail-artifact identity) PLUS exact recipe-fp
# validation against the pinned pipeline's own _params_fingerprint (the
# serving loader REQUIRES the fp on ledger components; byte pins stay
# refused on the append-only surface).

def _mom_fp(params):
    """Import the pinned pipeline's fp recipe lazily — these tests exercise the
    gate's fp-validation branch, which itself requires the import and fails
    closed without it. Skipping (loudly) when the sibling pipeline checkout is
    absent mirrors that contract instead of failing collection."""
    mi = pytest.importorskip("renquant_pipeline.momentum_identity")
    return mi.params_fingerprint(params)


_FIXTURE_PARAMS = {"params_version": "v0", "window": 231, "skip": 21}


def _config_with_primary_blend(data_root: Path, *, fp=None) -> dict:
    config = _base_config(_good_shadow(data_root))
    ps = config["ranking"]["panel_scoring"]
    ps["kind"] = "blend"
    comp1 = {"kind": "momentum_residual", "artifact_path": MOMENTUM_LEDGER_REL}
    if fp is not None:
        comp1["expected_config_fingerprint"] = fp
    ps["components"] = [
        {"artifact_path": ps["artifact_path"]},
        comp1,
    ]
    return config


def test_primary_ledger_component_with_valid_recipe_fp_passes(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_primary_blend(
        data_root, fp=_mom_fp(_FIXTURE_PARAMS)))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    comp = next(r for r in results if "components[1]" in r.field)
    assert comp.ok, comp.reason
    assert comp.kind == "primary"
    assert "chain=verified" in comp.detail
    assert "(validated)" in comp.detail


def test_primary_ledger_component_recipe_fp_mismatch_fails(tmp_path: Path) -> None:
    pytest.importorskip("renquant_pipeline.momentum_identity")
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, _config_with_primary_blend(
        data_root, fp="momentum-v0-0000000000000000"))
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    comp = next(r for r in results if "components[1]" in r.field)
    assert not comp.ok
    assert "recipe mismatch" in comp.reason


def test_primary_ledger_component_byte_pin_still_refused(tmp_path: Path) -> None:
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config = _config_with_primary_blend(data_root)
    config["ranking"]["panel_scoring"]["components"][1][
        "expected_content_sha256"] = "sha256:0000000000000000"
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    comp = next(r for r in results if "components[1]" in r.field)
    assert not comp.ok
    assert "not supported on a ledger pointer" in comp.reason


def test_primary_ledger_component_without_declared_kind_fails(tmp_path: Path) -> None:
    """The #550 contract still keys on the DECLARED kind: a primary component
    pointing at the ledger without kind=momentum_residual fails closed."""
    strategy_dir, data_root = _make_tree(tmp_path)
    _write_momentum_publish_set(strategy_dir)
    config = _config_with_primary_blend(data_root)
    del config["ranking"]["panel_scoring"]["components"][1]["kind"]
    config_path = tmp_path / "configs" / "strategy_config.json"
    _write_json(config_path, config)
    results = mod.check_config(
        config_path, "strategy_config", strategy_dir, data_root, _fake_contract()
    )
    comp = next(r for r in results if "components[1]" in r.field)
    assert not comp.ok
    assert "restricted to the momentum contract" in comp.reason
