"""P0b umbrella ops check — tests for scripts/check_active_scorer_gated.py.

Covers THIS script's responsibility: resolving `ranking.panel_scoring.artifact_path`
from a strategy config (relative to the strategy dir) and mapping the guard's
verdict to exit codes (0 gated / 1 violation / 2 cannot-determine). The guard's
own gating contract (`assert_artifact_gated`) is tested in renquant-backtesting
(test_promotion_integrity_guard.py); here we inject a controllable stub so the
umbrella test stays self-contained and doesn't depend on the subrepo PYTHONPATH.
"""
from __future__ import annotations

import contextlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

REPO_DIR = Path(__file__).resolve().parent.parent
SCRIPT = REPO_DIR / "scripts" / "check_active_scorer_gated.py"

_STUB_KEYS = (
    "renquant_backtesting",
    "renquant_backtesting.forensics",
    "renquant_backtesting.forensics.model_acceptance",
)


@contextlib.contextmanager
def _load_script_with_stub(stub_fn):
    """Import the script with a stubbed renquant_backtesting guard installed.

    The script imports assert_artifact_gated lazily inside main(), so the stub
    must stay in sys.modules for the duration of the main() call — hence a
    context manager rather than a plain loader. stub_fn(artifact_path) returns a
    wf metadata dict, or raises ValueError to signal an ungated artifact.
    """
    pkg = types.ModuleType("renquant_backtesting")
    forensics = types.ModuleType("renquant_backtesting.forensics")
    accept = types.ModuleType("renquant_backtesting.forensics.model_acceptance")
    accept.assert_artifact_gated = stub_fn
    pkg.forensics = forensics
    forensics.model_acceptance = accept
    saved = {k: sys.modules.get(k) for k in _STUB_KEYS}
    sys.modules.update({
        "renquant_backtesting": pkg,
        "renquant_backtesting.forensics": forensics,
        "renquant_backtesting.forensics.model_acceptance": accept,
    })
    try:
        spec = importlib.util.spec_from_file_location("_check_active_scorer_gated", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _write_config(strategy_dir: Path, name: str, artifact_rel: str, kind: str = "xgb") -> None:
    (strategy_dir / name).write_text(json.dumps({
        "ranking": {"panel_scoring": {"kind": kind, "artifact_path": artifact_rel}}
    }))


def test_gated_scorer_exits_0(tmp_path):
    sd = tmp_path / "strat"; sd.mkdir()
    (sd / "model.json").write_text("{}")
    _write_config(sd, "strategy_config.json", "model.json")
    with _load_script_with_stub(lambda p: {"passed": True, "run_at": "2026-06-08T00:00:00Z"}) as mod:
        rc = mod.main(["--strategy-dir", str(sd), "--config", "strategy_config.json"])
    assert rc == 0


def test_ungated_scorer_exits_1(tmp_path):
    sd = tmp_path / "strat"; sd.mkdir()
    (sd / "model.json").write_text("{}")
    _write_config(sd, "strategy_config.json", "model.json")

    def _raise(p):
        raise ValueError("missing wf_gate_metadata")

    with _load_script_with_stub(_raise) as mod:
        rc = mod.main(["--strategy-dir", str(sd), "--config", "strategy_config.json"])
    assert rc == 1


def test_mixed_one_ungated_exits_1(tmp_path):
    sd = tmp_path / "strat"; sd.mkdir()
    (sd / "good.json").write_text("{}")
    (sd / "bad.json").write_text("{}")
    _write_config(sd, "strategy_config.json", "good.json")
    _write_config(sd, "strategy_config.shadow.json", "bad.json")

    def _stub(p):
        if Path(p).name == "good.json":
            return {"passed": True}
        raise ValueError("passed=False")

    with _load_script_with_stub(_stub) as mod:
        rc = mod.main([
            "--strategy-dir", str(sd),
            "--config", "strategy_config.json",
            "--config", "strategy_config.shadow.json",
        ])
    assert rc == 1


def test_artifact_path_resolved_relative_to_strategy_dir(tmp_path):
    """`../../artifacts/...` style paths resolve like the live PatchTST config."""
    root = tmp_path
    sd = root / "backtesting" / "renquant_104"; sd.mkdir(parents=True)
    art = root / "artifacts" / "m.pt"; art.parent.mkdir(parents=True); art.write_text("x")
    _write_config(sd, "strategy_config.json", "../../artifacts/m.pt", kind="hf_patchtst")
    seen = {}

    def _stub(p):
        seen["path"] = Path(p)
        return {"passed": True}

    with _load_script_with_stub(_stub) as mod:
        rc = mod.main(["--strategy-dir", str(sd), "--config", "strategy_config.json"])
    assert rc == 0
    assert seen["path"] == art.resolve()


def test_missing_config_exits_2(tmp_path):
    sd = tmp_path / "strat"; sd.mkdir()
    with _load_script_with_stub(lambda p: {"passed": True}) as mod:
        rc = mod.main(["--strategy-dir", str(sd), "--config", "nope.json"])
    assert rc == 2


def test_missing_artifact_path_key_exits_2(tmp_path):
    sd = tmp_path / "strat"; sd.mkdir()
    (sd / "strategy_config.json").write_text(json.dumps({"ranking": {"panel_scoring": {"kind": "xgb"}}}))
    with _load_script_with_stub(lambda p: {"passed": True}) as mod:
        rc = mod.main(["--strategy-dir", str(sd), "--config", "strategy_config.json"])
    assert rc == 2


def test_guard_import_failure_exits_2(tmp_path):
    """No subrepo PYTHONPATH → cannot determine state → exit 2, not a false OK."""
    sd = tmp_path / "strat"; sd.mkdir()
    (sd / "model.json").write_text("{}")
    _write_config(sd, "strategy_config.json", "model.json")
    # Load the script WITHOUT injecting the stub module, and ensure the real
    # subrepo module is absent so the import inside main() fails.
    saved = {k: sys.modules.pop(k, None) for k in (
        "renquant_backtesting",
        "renquant_backtesting.forensics",
        "renquant_backtesting.forensics.model_acceptance",
    )}
    try:
        spec = importlib.util.spec_from_file_location("_check_active_scorer_gated_noimp", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # Force the import to fail even if a real subrepo is on sys.path.
        sys.modules["renquant_backtesting"] = None  # type: ignore[assignment]
        rc = mod.main(["--strategy-dir", str(sd), "--config", "strategy_config.json"])
        assert rc == 2
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
