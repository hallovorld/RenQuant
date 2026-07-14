"""G3 F-7: sim must resolve the pinned strategy config, fail closed if it can't.

Regression guards for ``scripts/run_sim_104.py``'s
``resolve_strategy_config()``. Finding F-7 of the 2026-07-04 architecture
compliance audit: sim silently read the umbrella-local
``backtesting/renquant_104/strategy_config.json`` copy, which had drifted
from the pinned ``renquant-strategy-104/configs/strategy_config.json`` that
the live bridge uses — sim evaluated a different primary scorer than live.

Codex review on the first attempt at this fix additionally required: (1) the
resolved pinned config's content must be fingerprinted and recorded, and (2)
the "warn and fall back to an unpinned umbrella copy" behavior must be an
explicit, hard-to-trigger-by-accident local-dev-only escape hatch — never
reachable on the standard/default simulation path.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load_run_sim_104():
    sys.path.insert(0, str(REPO / "scripts"))
    return importlib.import_module("run_sim_104")


def _make_repo_root(tmp_path: Path) -> Path:
    repo_root = tmp_path / "RenQuant"
    (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
    return repo_root


def _expected_fingerprint(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


# --- (a) pinned-first resolution ------------------------------------------


def test_pinned_config_resolves_first_and_is_fingerprinted(tmp_path: Path) -> None:
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    pinned_dir = tmp_path / "renquant-strategy-104" / "configs"
    pinned_dir.mkdir(parents=True)
    pinned_cfg = pinned_dir / "strategy_config.json"
    pinned_cfg.write_text('{"primary_scorer_kind": "xgb"}')

    # A drifted umbrella-local copy exists too — pinned must win, not this.
    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    resolved = mod.resolve_strategy_config(
        strategy_dir=strategy_dir,
        repo_root=repo_root,
        config_name="strategy_config.json",
    )

    assert resolved.source == "pinned"
    assert resolved.path == pinned_cfg
    assert resolved.fingerprint == _expected_fingerprint(pinned_cfg)
    assert resolved.fingerprint.startswith("sha256:")
    # Must NOT be the drifted local copy's content.
    assert resolved.fingerprint != _expected_fingerprint(local_cfg)


# --- (b) standard/default invocation fails closed when pin is missing -----


def test_default_invocation_fails_closed_when_pin_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    monkeypatch.delenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, raising=False)

    # No sibling renquant-strategy-104 checkout at all. A local umbrella
    # copy IS present (this is the drift scenario) but must not be used.
    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    with pytest.raises(mod.StrategyConfigResolutionError, match="G3 F-7"):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
        )


def test_default_invocation_fails_closed_even_with_no_local_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    monkeypatch.delenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, raising=False)

    with pytest.raises(mod.StrategyConfigResolutionError):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
        )


# --- (c) explicit local-dev flag allows warn-and-fallback ------------------


def test_local_dev_escape_hatch_falls_back_with_both_flag_and_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    monkeypatch.setenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, "1")
    with caplog.at_level("WARNING"):
        resolved = mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
            allow_unpinned_local_dev=True,
        )

    assert resolved.source == "unpinned_local_dev_fallback"
    assert resolved.path == local_cfg
    assert resolved.fingerprint == _expected_fingerprint(local_cfg)
    assert any("LOCAL-DEV MODE" in rec.message for rec in caplog.records)


# --- (d) no accidental fallback without BOTH flag and env ------------------


def test_no_accidental_fallback_env_set_but_flag_not_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Env var alone (e.g. leaked into a shared shell/CI env) must not be enough."""
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    monkeypatch.setenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, "1")
    with pytest.raises(mod.StrategyConfigResolutionError):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
            allow_unpinned_local_dev=False,  # flag NOT passed
        )


def test_no_accidental_fallback_flag_passed_but_env_not_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CLI flag alone (e.g. a copy-pasted invocation) must not be enough."""
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    monkeypatch.delenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, raising=False)
    with pytest.raises(mod.StrategyConfigResolutionError):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
            allow_unpinned_local_dev=True,  # env NOT set
        )


def test_no_accidental_fallback_env_set_to_non_one_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env var must be exactly '1', not merely truthy/present."""
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    local_cfg = strategy_dir / "strategy_config.json"
    local_cfg.write_text('{"primary_scorer_kind": "hf_patchtst"}')

    monkeypatch.setenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, "true")
    with pytest.raises(mod.StrategyConfigResolutionError):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.json",
            allow_unpinned_local_dev=True,
        )


# --- experiment-only (non-live-mirrored) side configs are unaffected -------


def test_experiment_local_side_config_resolves_without_pin_or_escape_hatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """research sweep configs (strategy_config.sim_*.json) were never
    published to the pin — they must keep resolving from the umbrella copy
    directly, with no escape hatch required, so the large existing body of
    sim-sweep tooling (run_dense_panel.sh, run_regime_overlay_experiments.sh,
    etc.) is unaffected by the F-7 fail-closed change."""
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"
    monkeypatch.delenv(mod.ALLOW_UNPINNED_LOCAL_DEV_ENV, raising=False)

    side_cfg = strategy_dir / "strategy_config.sim_baseline_hmm.json"
    side_cfg.write_text('{"some_experiment_knob": true}')

    resolved = mod.resolve_strategy_config(
        strategy_dir=strategy_dir,
        repo_root=repo_root,
        config_name="strategy_config.sim_baseline_hmm.json",
    )

    assert resolved.source == "experiment_local"
    assert resolved.path == side_cfg
    assert resolved.fingerprint == _expected_fingerprint(side_cfg)


def test_experiment_local_side_config_still_fails_closed_if_missing(
    tmp_path: Path,
) -> None:
    mod = _load_run_sim_104()
    repo_root = _make_repo_root(tmp_path)
    strategy_dir = repo_root / "backtesting" / "renquant_104"

    with pytest.raises(mod.StrategyConfigResolutionError):
        mod.resolve_strategy_config(
            strategy_dir=strategy_dir,
            repo_root=repo_root,
            config_name="strategy_config.sim_does_not_exist.json",
        )


# --- CLI wiring -------------------------------------------------------------


def test_cli_exposes_suppressed_local_dev_flag() -> None:
    src = (REPO / "scripts/run_sim_104.py").read_text()
    assert '"--allow-unpinned-local-dev"' in src
    assert "argparse.SUPPRESS" in src
    assert "RENQUANT_ALLOW_UNPINNED_LOCAL_DEV" in src
    assert "StrategyConfigResolutionError" in src
