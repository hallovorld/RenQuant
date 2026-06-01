from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _load_contract_module(monkeypatch, runtime_root: Path):
    scripts_dir = REPO / "scripts"
    monkeypatch.syspath_prepend(str(scripts_dir))
    monkeypatch.setenv("RENQUANT_SUBREPO_ROOT", str(runtime_root))
    for name in list(sys.modules):
        if name == "renquant_orchestrator" or name.startswith("renquant_orchestrator."):
            del sys.modules[name]

    path = scripts_dir / "subrepo_daily_contract.py"
    spec = importlib.util.spec_from_file_location("subrepo_daily_contract_under_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_daily_contract_uses_runtime_root_for_subrepo_paths(monkeypatch, tmp_path, capsys):
    runtime_root = tmp_path / "runtime" / "repos"
    orch_pkg = runtime_root / "renquant-orchestrator" / "src" / "renquant_orchestrator"
    orch_pkg.mkdir(parents=True)
    (orch_pkg / "__init__.py").write_text("", encoding="utf-8")
    (orch_pkg / "cli.py").write_text(
        "import json\n"
        "def main(argv):\n"
        "    print(json.dumps({'argv': argv}, sort_keys=True))\n"
        "    return 0\n",
        encoding="utf-8",
    )
    strategy_dir = runtime_root / "renquant-strategy-104" / "configs"
    strategy_dir.mkdir(parents=True)
    (strategy_dir / "strategy_config.json").write_text("{}", encoding="utf-8")

    module = _load_contract_module(monkeypatch, runtime_root)
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "subrepo_daily_contract.py",
            "--as-of",
            "2026-01-02",
            "--output-dir",
            str(out_dir),
        ],
    )

    assert module.main() == 0
    payload = json.loads(capsys.readouterr().out)
    argv = payload["argv"]
    strategy_idx = argv.index("--strategy-config") + 1
    assert argv[strategy_idx] == str(strategy_dir / "strategy_config.json")
    assert "/Users/renhao/git/github/renquant-strategy-104" not in " ".join(argv)
