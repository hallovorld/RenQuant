"""Regression guards for Python subrepo delegation."""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "subrepo_module_delegate.py"


def _load_delegate_module():
    scripts_dir = str(REPO / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("subrepo_module_delegate_for_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_delegate_exports_pinned_strategy_config(monkeypatch, tmp_path: Path) -> None:
    mod = _load_delegate_module()
    repo_root = tmp_path / "RenQuant"
    runtime = tmp_path / "runtime" / "repos"
    strategy_config = runtime / "renquant-strategy-104" / "configs" / "strategy_config.json"
    fake_pkg = runtime / "fakepkg" / "src" / "fakepkg"
    strategy_config.parent.mkdir(parents=True)
    fake_pkg.mkdir(parents=True)
    strategy_config.write_text("{}", encoding="utf-8")
    (fake_pkg / "__init__.py").write_text("", encoding="utf-8")
    (fake_pkg / "ops.py").write_text(
        "import os\n"
        "import sys\n"
        "from pathlib import Path\n"
        "def main():\n"
        "    Path(os.environ['RQ_CAPTURE']).write_text('\\n'.join([\n"
        "        os.environ.get('RENQUANT_REPO_ROOT', ''),\n"
        "        os.environ.get('RENQUANT_SUBREPO_ROOT', ''),\n"
        "        os.environ.get('RENQUANT_STRATEGY_CONFIG', ''),\n"
        "        str('--repo-root' in sys.argv),\n"
        "    ]), encoding='utf-8')\n"
        "    return 0\n",
        encoding="utf-8",
    )
    capture = tmp_path / "capture.txt"
    monkeypatch.setattr(mod, "resolve_subrepo_root", lambda _repo: runtime)
    monkeypatch.setenv("RQ_CAPTURE", str(capture))
    monkeypatch.delenv("RENQUANT_REPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_SUBREPO_ROOT", raising=False)
    monkeypatch.delenv("RENQUANT_STRATEGY_CONFIG", raising=False)

    rc = mod.delegate_to_subrepo_module(
        "fakepkg.ops",
        [],
        repo_root=repo_root,
        packages=("fakepkg",),
        runner_env="RQ_FAKE_RUNNER",
        strict_env="RQ_FAKE_STRICT",
    )

    assert rc == 0
    repo, root, config, has_repo_root_arg = capture.read_text(encoding="utf-8").splitlines()
    assert repo == str(repo_root)
    assert root == str(runtime)
    assert config == str(strategy_config)
    assert has_repo_root_arg == "True"


def test_delegate_fails_closed_without_strategy_config_when_strict(
    monkeypatch,
    tmp_path: Path,
) -> None:
    mod = _load_delegate_module()
    runtime = tmp_path / "runtime" / "repos"
    (runtime / "fakepkg" / "src").mkdir(parents=True)
    monkeypatch.setattr(mod, "resolve_subrepo_root", lambda _repo: runtime)
    monkeypatch.setenv("RENQUANT_STRICT_SUBREPO_PATHS", "1")
    monkeypatch.setenv("RQ_FAKE_STRICT", "1")

    try:
        mod.delegate_to_subrepo_module(
            "fakepkg.ops",
            [],
            repo_root=tmp_path / "RenQuant",
            packages=("fakepkg",),
            runner_env="RQ_FAKE_RUNNER",
            strict_env="RQ_FAKE_STRICT",
        )
    except FileNotFoundError as exc:
        assert "strategy_config.json unavailable" in str(exc)
    else:  # pragma: no cover - assertion path
        raise AssertionError("strict delegate did not fail closed")
