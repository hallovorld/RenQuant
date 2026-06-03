"""Regression tests for the vendored QP runtime sanity guard."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO / "scripts" / "runtime_qp_sanity_check.py"
    spec = importlib.util.spec_from_file_location("runtime_qp_sanity_check_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_pkg(root: Path, rel: str) -> None:
    path = root / rel
    path.mkdir(parents=True, exist_ok=True)
    parts = path.relative_to(root).parts
    cur = root
    for part in parts:
        cur = cur / part
        (cur / "__init__.py").write_text("", encoding="utf-8")


def _write_module(root: Path, rel: str, body: str) -> None:
    path = root / rel
    _write_pkg(root, str(path.parent.relative_to(root)))
    path.write_text(body, encoding="utf-8")


def _fake_runtime_root(tmp_path: Path) -> Path:
    root = tmp_path / "runtime" / "repos"
    pipeline_src = root / "renquant-pipeline" / "src"
    common_src = root / "renquant-common" / "src"

    modules = {
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/davis_norman.py":
            "def davis_norman_band_clamped(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/proportional_trade.py":
            "def proportional_trade_target(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/constraint_snapshot.py":
            "class ConstraintSnapshot: pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/qp_solver.py":
            "def solve_portfolio_qp_from_snapshot(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/baseline_allocators.py":
            "def hybrid_option_f_allocator(): pass\n"
            "def hard_only_qp_allocator(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/allocator_replay.py":
            "def replay_all(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/replay_significance.py":
            "def compute_significance_verdicts(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/wf_replay_loader.py":
            "def load_replay_bars_from_sim_db(): pass\n",
        pipeline_src / "renquant_pipeline/kernel/portfolio_qp/run_ab_replay.py":
            "def run_replay(): pass\n",
        common_src / "renquant_common/metrics/deflated_sharpe.py":
            "def deflated_sharpe_ratio(): pass\n",
        common_src / "renquant_common/metrics/pbo.py":
            "def probability_of_backtest_overfitting(): pass\n",
        common_src / "renquant_common/metrics/hac_se.py":
            "def hac_t_stat(): pass\n",
    }
    for path, body in modules.items():
        _write_module(path.parent.parent.parent.parent, str(path.relative_to(path.parent.parent.parent.parent)), body)
    return root


def _clear_fake_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("renquant_pipeline") or name.startswith("renquant_common"):
            sys.modules.pop(name, None)


def test_runtime_qp_sanity_passes_on_complete_runtime(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    root = _fake_runtime_root(tmp_path)
    old_path = list(sys.path)
    monkeypatch.setattr(sys, "path", old_path.copy())
    _clear_fake_modules()

    failures = mod.check_runtime(root)

    assert failures == []
    out = capsys.readouterr().out
    assert "renquant_pipeline.kernel.portfolio_qp.davis_norman" in out
    assert str(root / "renquant-pipeline" / "src") in out


def test_runtime_qp_sanity_fails_on_missing_step4g_module(tmp_path, monkeypatch):
    mod = _load_module()
    root = _fake_runtime_root(tmp_path)
    missing = (
        root / "renquant-pipeline" / "src" / "renquant_pipeline" /
        "kernel" / "portfolio_qp" / "wf_replay_loader.py"
    )
    missing.unlink()
    monkeypatch.setattr(sys, "path", sys.path.copy())
    _clear_fake_modules()

    failures = mod.check_runtime(root)

    assert any("wf_replay_loader" in f for f in failures)


def test_runtime_qp_sanity_fails_cleanly_on_missing_runtime_root(tmp_path, monkeypatch, capsys):
    mod = _load_module()
    missing_root = tmp_path / "missing" / "repos"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    _clear_fake_modules()

    rc = mod.main(["--runtime-root", str(missing_root)])

    assert rc == 1
    out = capsys.readouterr().out
    assert "runtime repo source missing" in out
    assert "FAIL: stale or incomplete QP multirepo runtime." in out
    assert "Traceback" not in out


def test_daily_104_runs_runtime_qp_sanity_before_daily_runner():
    src = (REPO / "scripts" / "daily_104.sh").read_text(encoding="utf-8")
    guard = '"$PYTHON" "$REPO_DIR/scripts/runtime_qp_sanity_check.py"'
    assert guard in src
    assert "RUNTIME-SANITY-FAIL" in src
    assert src.index(guard) < src.index('RUNNER_ARGS=("$REPO_DIR/scripts/daily_multirepo.py")')
