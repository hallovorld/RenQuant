"""Software-stop registry at the NEUTRAL root — the two umbrella follow-ups
of the orchestrator bootstrap step (orch#1078), plus the pin that makes the
wrapper's seed command real (Codex review on RenQuant#613).

1. ``adapters/software_stops_wiring.py`` + ``RunnerAdapter.__init__``:
   * ``execution.software_stops.enabled`` absent/false -> ``None``
     IMMEDIATELY, read the pipeline's own way (a verbatim mirror of
     ``from_config``'s gate, parity-pinned against the real ``from_config``):
     no orchestrator import, no neutral-root resolution, no log line, no
     disk access — the pre-change inert path, preserved literally (the
     contract module is monkeypatched to RAISE on import and nothing is
     logged);
   * enabled -> the registry is built with ``SoftwareStopRegistry.from_config
     (..., repo_root=<neutral root>)`` where the neutral root is the
     orchestrator LOCATION contract's ``software_stops_registry_root
     (runtime_state_root())`` (``~/.renquant/runtime/software-stops``,
     override ``RENQUANT_RUNTIME_STATE_ROOT``) — the ``--data-root`` the
     liveness pager and the orchestrator seeder resolve against. The
     registry path EQUALS the orchestrator's ``seeded_registry_path(<root>,
     broker)`` (import parity against the real sibling when importable;
     skipped WITH the reason otherwise) and a seed written by the
     orchestrator is read back by the runner's registry;
   * enabled + contract not importable -> ``None`` + one ERROR line, and
     ``from_config`` is NOT called (no cwd fallback).
2. ``scripts/intraday_sell_104.sh``: the seeder runs unconditionally
   BEFORE the runner, with the same ``--broker`` the runner receives, and
   never ``exit``s on its own failure (the sell loop is the live book's exit
   path). Structural pins plus the extracted block executed under bash with
   stub outcomes.
3. ASSEMBLY-LEVEL regression: the orchestrator module at the PINNED runtime
   path (``<RENQUANT_SUBREPO_ROOT>/renquant-orchestrator/src``, the path the
   wrapper puts on PYTHONPATH) implements the ``seed`` contract the wrapper
   invokes — a subprocess ``seed --broker alpaca --data-root <tmp>`` prints
   ``SEEDED:`` then ``EXISTS:`` and exits 0. A pinned module without the
   CLI (exit 0, no verdict — the pre-#1078 revision) FAILS this test.

Runs in the lean ``live-broker-fractional-contract`` CI job (pytest only):
the wiring module imports nothing heavy, the pipeline / orchestrator siblings
are stubbed when absent, and the tests that need the full strategy deps or
the runtime assembly skip there WITH a reason and run locally.
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import subprocess
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters import software_stops_wiring as wiring  # noqa: E402

CONTRACT = "renquant_orchestrator.software_stops_registry_contract"
PIPELINE = "renquant_pipeline.software_stops"
RUNNER_SRC = (_STRATEGY / "adapters" / "runner.py").read_text(encoding="utf-8")
WIRING_SRC = (_STRATEGY / "adapters" / "software_stops_wiring.py").read_text(encoding="utf-8")
INTRADAY_PATH = REPO_ROOT / "scripts" / "intraday_sell_104.sh"
INTRADAY = INTRADAY_PATH.read_text(encoding="utf-8")

SEED_CALL = '"$PYTHON" -m "$SEED_MODULE" seed --broker alpaca'
RUNNER_CALL = '"$PYTHON" "${RUNNER_ARGS[@]}" --strategy renquant_104 --broker alpaca --once'


# ── sibling resolution (the _order_math_owner / test_live_multirepo pattern) ─

def _lock_entry(name: str) -> dict | None:
    lock = REPO_ROOT / "subrepos.lock.json"
    if not lock.exists():
        return None
    return next((e for e in json.loads(lock.read_text())["subrepos"] if e["name"] == name), None)


def _sibling_src(name: str) -> Path | None:
    """``<RENQUANT_SUBREPO_ROOT>/<name>/src`` first (the runtime assembly the
    wrappers export), then ``subrepos.lock.json`` ``local_path``."""
    env_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if env_root:
        cand = Path(env_root) / name / "src"
        if cand.is_dir():
            return cand
    entry = _lock_entry(name)
    if entry:
        cand = Path(entry["local_path"]) / "src"
        if cand.is_dir():
            return cand
    return None


def _real_module(modname: str, sibling: str, *needs: str):
    """The REAL sibling module with every name in ``needs``, or ``None`` with
    a reason string (second tuple item) for a skip."""
    try:
        mod = importlib.import_module(modname)
    except ImportError:
        src = _sibling_src(sibling)
        if src is None:
            return None, f"{modname} not importable and no {sibling} sibling checkout resolvable"
        if str(src) not in sys.path:
            sys.path.append(str(src))
        try:
            mod = importlib.import_module(modname)
        except ImportError as exc:
            return None, f"{modname} not importable from {src}: {exc}"
    missing = [n for n in needs if not hasattr(mod, n)]
    if missing:
        return None, (
            f"{modname} at {getattr(mod, '__file__', '?')} lacks {missing} "
            "(checkout predates the seeder, orch#1078)"
        )
    return mod, ""


def _require_real(modname: str, sibling: str, *needs: str):
    mod, reason = _real_module(modname, sibling, *needs)
    if mod is None:
        pytest.skip(reason)
    return mod


# ── stubs for the lean CI job (no siblings installed) ───────────────────────

def _stub_contract(monkeypatch, state_root: Path) -> types.ModuleType:
    """A stand-in for the orchestrator LOCATION contract with the same two
    names the wiring imports and the same composition
    (``<root>/software-stops``)."""
    mod = types.ModuleType(CONTRACT)

    def runtime_state_root(override=None):
        return Path(override).expanduser() if override is not None else state_root

    def software_stops_registry_root(root):
        return Path(root) / "software-stops"

    mod.runtime_state_root = runtime_state_root
    mod.software_stops_registry_root = software_stops_registry_root
    pkg = sys.modules.get("renquant_orchestrator") or types.ModuleType("renquant_orchestrator")
    monkeypatch.setitem(sys.modules, "renquant_orchestrator", pkg)
    monkeypatch.setitem(sys.modules, CONTRACT, mod)
    return mod


class _Recorder:
    """Wraps a ``from_config`` (real or fake): records every call's kwargs."""

    def __init__(self, impl):
        self.impl = impl
        self.calls: list[dict] = []

    def __call__(self, config, **kwargs):
        self.calls.append(dict(kwargs))
        return self.impl(config, **kwargs)


def _fake_from_config(config, *, broker_name=None, repo_root=None):
    """The pipeline gate's shape: flag absent/false -> None BEFORE repo_root
    is read (mirrors renquant_pipeline.software_stops.from_config)."""
    ss = ((config or {}).get("execution") or {}).get("software_stops") or {}
    if not ss.get("enabled", False):
        return None
    base = Path(ss.get("registry_path", "data/rq105/software_stops.json"))
    if not base.is_absolute():
        base = Path(repo_root) / base if repo_root else base
    tagged = base.with_stem(f"{base.stem}.{broker_name}") if broker_name else base
    return SimpleNamespace(path=tagged, is_armed=lambda: True)


@pytest.fixture
def pipeline_stops(monkeypatch):
    """The REAL pipeline registry module when importable, else a stub with the
    same ``from_config`` gate. Either way ``SoftwareStopRegistry.from_config``
    is wrapped in a recorder the tests read the kwargs from."""
    mod, _reason = _real_module(PIPELINE, "renquant-pipeline", "SoftwareStopRegistry")
    if mod is None:
        mod = types.ModuleType(PIPELINE)
        mod.SoftwareStopRegistry = types.SimpleNamespace()
        mod.DEFAULT_REGISTRY_PATH = "data/rq105/software_stops.json"
        pkg = sys.modules.get("renquant_pipeline") or types.ModuleType("renquant_pipeline")
        monkeypatch.setitem(sys.modules, "renquant_pipeline", pkg)
        monkeypatch.setitem(sys.modules, PIPELINE, mod)
        recorder = _Recorder(_fake_from_config)
        monkeypatch.setattr(mod.SoftwareStopRegistry, "from_config", recorder, raising=False)
        return mod, recorder, False
    recorder = _Recorder(mod.SoftwareStopRegistry.from_config)
    monkeypatch.setattr(mod.SoftwareStopRegistry, "from_config", recorder)
    return mod, recorder, True


@pytest.fixture
def neutral_root(monkeypatch, tmp_path):
    """A tmp neutral runtime-state root: the env override the orchestrator
    contract honours; the stub honours the same variable. Nothing under
    ``~/.renquant`` is ever touched."""
    state_root = tmp_path / "state"
    monkeypatch.setenv("RENQUANT_RUNTIME_STATE_ROOT", str(state_root))
    # cwd is a DIFFERENT tmp dir so a cwd fallback would be visible on disk.
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return state_root


def _contract_for(monkeypatch, state_root: Path):
    mod, _reason = _real_module(CONTRACT, "renquant-orchestrator",
                                "runtime_state_root", "software_stops_registry_root")
    return mod if mod is not None else _stub_contract(monkeypatch, state_root)


def _cfg(enabled: bool | None) -> dict:
    if enabled is None:
        return {"execution": {}}
    return {"execution": {"software_stops": {"enabled": enabled}}}


# Every config shape the gate must treat as OFF (falsy the pipeline's way).
OFF_CONFIGS = [
    None, {}, {"execution": None}, {"execution": {}},
    {"execution": {"software_stops": None}}, {"execution": {"software_stops": {}}},
    _cfg(False), _cfg(None),
    {"execution": {"software_stops": {"enabled": 0}}},
    {"execution": {"software_stops": {"enabled": ""}}},
]
ON_CONFIGS = [_cfg(True), {"execution": {"software_stops": {"enabled": 1}}},
              {"execution": {"software_stops": {"enabled": "yes"}}}]


# ── 1a. the enabled gate: pipeline-owned semantics, read first ───────────────

class TestEnabledGate:
    @pytest.mark.parametrize("cfg", OFF_CONFIGS)
    def test_off_shapes(self, cfg):
        assert wiring.software_stops_enabled(cfg) is False

    @pytest.mark.parametrize("cfg", ON_CONFIGS)
    def test_on_shapes(self, cfg):
        assert wiring.software_stops_enabled(cfg) is True

    def test_parity_with_the_real_pipeline_gate(self, tmp_path):
        """The mirror agrees with ``from_config`` on every shape above: OFF
        shapes -> ``from_config`` returns None with a repo_root that raises
        if touched; ON shapes -> a registry under a tmp root (no file
        written — the registry is created on first write)."""
        mod = _require_real(PIPELINE, "renquant-pipeline", "SoftwareStopRegistry")

        class Explodes:
            def __fspath__(self):
                raise AssertionError("repo_root was touched on the flag-off path")

            def __str__(self):
                raise AssertionError("repo_root was stringified on the flag-off path")

        for cfg in OFF_CONFIGS:
            assert wiring.software_stops_enabled(cfg) is False
            assert mod.SoftwareStopRegistry.from_config(
                cfg, broker_name="paper", repo_root=Explodes(),
            ) is None
        for cfg in ON_CONFIGS:
            assert wiring.software_stops_enabled(cfg) is True
            reg = mod.SoftwareStopRegistry.from_config(cfg, broker_name="paper", repo_root=tmp_path)
            assert reg is not None
            assert Path(reg.path) == tmp_path / "data" / "rq105" / "software_stops.paper.json"
        assert list(tmp_path.iterdir()) == []

    def test_gate_is_read_before_any_import_in_source(self):
        body = WIRING_SRC[WIRING_SRC.index("def build_software_stop_registry"):]
        code = re.sub(r'"""[\s\S]*?"""', "", body)  # drop the docstring
        gate = code.index("if not software_stops_enabled(config):")
        pre = code[:gate]
        assert "import" not in pre and "software_stops_neutral_root()" not in pre
        assert "log." not in pre and "from_config" not in pre
        assert code.index("return None") > gate
        assert code.index("from renquant_pipeline.software_stops import") > gate
        assert code.index("software_stops_neutral_root()") > gate
        assert code.index("repo_root=root") > gate
        # the mirror reads exactly the pipeline's key path
        assert ('((config or {}).get("execution") or {}).get("software_stops") or {}'
                in WIRING_SRC)
        assert 'ss_cfg.get("enabled", False)' in WIRING_SRC


# ── 1b. flag off: the pre-change inert path, literally preserved ────────────

class TestFlagOffByteInert:
    @pytest.mark.parametrize("cfg", OFF_CONFIGS)
    def test_none_without_contract_import_root_log_or_disk(
        self, monkeypatch, neutral_root, pipeline_stops, caplog, cfg,
    ):
        """Disabled/absent -> None with the orchestrator contract UNIMPORTABLE
        (its import would raise), the pipeline constructor never called, no
        log record of any level, nothing under the neutral root or the cwd."""
        monkeypatch.setitem(sys.modules, CONTRACT, None)  # import -> ImportError
        _mod, recorder, _real = pipeline_stops
        with caplog.at_level(logging.DEBUG, logger="adapters.runner"):
            assert wiring.build_software_stop_registry(cfg, "paper") is None
        assert recorder.calls == []
        assert caplog.records == []
        assert not neutral_root.exists()
        assert list(Path.cwd().iterdir()) == []

    def test_none_even_when_the_pipeline_is_unimportable(
        self, monkeypatch, neutral_root, caplog,
    ):
        """Disabled -> the pipeline module is not imported either (the old
        path imported it; the gate now precedes every import)."""
        monkeypatch.setitem(sys.modules, PIPELINE, None)
        monkeypatch.setitem(sys.modules, CONTRACT, None)
        with caplog.at_level(logging.DEBUG, logger="adapters.runner"):
            assert wiring.build_software_stop_registry(_cfg(False), "paper") is None
        assert caplog.records == []

    def test_runner_source_delegates_and_has_no_bare_from_config(self):
        assert "build_software_stop_registry(" in RUNNER_SRC
        assert "SoftwareStopRegistry.from_config(" not in RUNNER_SRC, (
            "runner.py must not construct the registry itself (the neutral-root "
            "wiring is the single construction site)"
        )
        assert "repo_root=root" in WIRING_SRC
        assert "software_stops_registry_root(runtime_state_root())" in WIRING_SRC


# ── 1c. flag on: neutral root parity + fail-closed ──────────────────────────

class TestNeutralRootResolution:
    def test_root_is_the_contracts_composition_under_the_env_override(
        self, monkeypatch, neutral_root,
    ):
        _contract_for(monkeypatch, neutral_root)
        assert wiring.software_stops_neutral_root() == neutral_root / "software-stops"

    def test_root_is_never_cwd_relative(self, monkeypatch, neutral_root):
        _contract_for(monkeypatch, neutral_root)
        root = wiring.software_stops_neutral_root()
        assert root.is_absolute()
        assert Path.cwd() not in root.parents and root != Path.cwd()
        assert REPO_ROOT not in root.parents

    def test_contract_missing_raises_named_error_no_fallback(self, monkeypatch):
        monkeypatch.setitem(sys.modules, CONTRACT, None)  # import -> ImportError
        with pytest.raises(wiring.NeutralRootUnavailable) as ei:
            wiring.software_stops_neutral_root()
        msg = str(ei.value)
        assert CONTRACT in msg and "REFUSING" in msg and "cwd" in msg


class TestFlagOnNeutralRootParity:
    def test_from_config_receives_the_neutral_root(
        self, monkeypatch, neutral_root, pipeline_stops,
    ):
        _contract_for(monkeypatch, neutral_root)
        _mod, recorder, _real = pipeline_stops
        reg = wiring.build_software_stop_registry(_cfg(True), "paper")
        assert reg is not None
        assert recorder.calls == [
            {"broker_name": "paper", "repo_root": neutral_root / "software-stops"},
        ]
        assert list(Path.cwd().iterdir()) == []  # registry is created on first write

    def test_registry_path_equals_orchestrator_seeded_path(
        self, monkeypatch, neutral_root, pipeline_stops,
    ):
        """Flag ON under a tmp neutral root: the runner's registry path is
        byte-equal to the orchestrator's ``seeded_registry_path`` for the
        same root + broker (the checker's composition) — import parity
        against the REAL sibling."""
        contract = _require_real(CONTRACT, "renquant-orchestrator",
                                 "seeded_registry_path", "ensure_registry_seeded",
                                 "runtime_state_root", "software_stops_registry_root")
        _mod, recorder, real = pipeline_stops
        if not real:
            pytest.skip(f"{PIPELINE} not importable — parity needs the real pipeline module")

        reg = wiring.build_software_stop_registry(_cfg(True), "paper")
        assert reg is not None
        data_root = contract.software_stops_registry_root(contract.runtime_state_root())
        assert data_root == neutral_root / "software-stops"
        assert recorder.calls[-1]["repo_root"] == data_root
        expected = contract.seeded_registry_path(data_root, "paper")
        assert Path(reg.path) == expected
        assert expected == neutral_root / "software-stops" / "data" / "rq105" / "software_stops.paper.json"
        # not under the cwd (the pre-fix resolution) and not under the repo
        assert Path.cwd() not in Path(reg.path).parents
        assert REPO_ROOT not in Path(reg.path).parents
        assert list(Path.cwd().iterdir()) == []

    def test_orchestrator_seed_is_read_back_by_the_runner_registry(
        self, monkeypatch, neutral_root, pipeline_stops,
    ):
        """Seed with the orchestrator (the wrapper's step) -> the runner's
        registry at the same root loads that file as armed + empty."""
        contract = _require_real(CONTRACT, "renquant-orchestrator",
                                 "seeded_registry_path", "ensure_registry_seeded")
        _mod, _recorder, real = pipeline_stops
        if not real:
            pytest.skip(f"{PIPELINE} not importable — needs the real registry")
        data_root = neutral_root / "software-stops"
        seeded = contract.ensure_registry_seeded(data_root, "paper")
        assert seeded.exists()

        reg = wiring.build_software_stop_registry(_cfg(True), "paper")
        assert reg is not None
        assert Path(reg.path) == seeded
        assert reg.is_armed() is True and not reg.corrupt
        assert json.loads(seeded.read_text())["stops"] == {}

    def test_flag_on_logs_the_resolved_path(
        self, monkeypatch, neutral_root, pipeline_stops, caplog,
    ):
        _contract_for(monkeypatch, neutral_root)
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            reg = wiring.build_software_stop_registry(_cfg(True), "paper")
        assert reg is not None
        line = next(r for r in caplog.records if "NEUTRAL root" in r.getMessage())
        assert str(neutral_root / "software-stops") in line.getMessage()
        assert "software_stops.paper.json" in line.getMessage()


class TestContractImportFailureFailsClosed:
    @pytest.mark.parametrize("cfg", ON_CONFIGS)
    def test_enabled_none_error_logged_and_from_config_never_called(
        self, monkeypatch, neutral_root, pipeline_stops, caplog, cfg,
    ):
        """Enabled + contract module unimportable -> registry None, ONE ERROR
        line, and the pipeline constructor is NOT invoked — no silent cwd
        fallback."""
        monkeypatch.setitem(sys.modules, CONTRACT, None)
        _mod, recorder, _real = pipeline_stops
        with caplog.at_level(logging.ERROR, logger="adapters.runner"):
            reg = wiring.build_software_stop_registry(cfg, "paper")
        assert reg is None
        assert recorder.calls == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        msg = errors[0].getMessage()
        assert "software-stop registry construction FAILED" in msg
        assert CONTRACT in msg and "NOT armed" in msg and "REFUSING" in msg
        assert list(Path.cwd().iterdir()) == []
        assert not neutral_root.exists()

    def test_pipeline_import_failure_still_fails_closed(
        self, monkeypatch, neutral_root, caplog,
    ):
        _contract_for(monkeypatch, neutral_root)
        monkeypatch.setitem(sys.modules, PIPELINE, None)
        with caplog.at_level(logging.ERROR, logger="adapters.runner"):
            assert wiring.build_software_stop_registry(_cfg(True), "paper") is None
        assert any("construction FAILED" in r.getMessage() for r in caplog.records)


class TestRunnerAdapterWiring:
    """End-to-end through the REAL ``RunnerAdapter.__init__`` (preflight mode:
    no DB, no artifacts). Needs the strategy deps -> skips in the lean CI job,
    runs in the full local suite."""

    @staticmethod
    def _runner_adapter():
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        try:
            from adapters.runner import RunnerAdapter  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 — lean CI: strategy deps absent
            pytest.skip(f"adapters.runner not importable here: {exc}")
        return RunnerAdapter

    def test_init_flag_off_none_without_contract_or_log(
        self, monkeypatch, neutral_root, pipeline_stops, tmp_path, caplog,
    ):
        RunnerAdapter = self._runner_adapter()
        monkeypatch.setitem(sys.modules, CONTRACT, None)  # would raise if imported
        _mod, recorder, _real = pipeline_stops
        with caplog.at_level(logging.DEBUG, logger="adapters.runner"):
            adapter = RunnerAdapter(
                _cfg(False), models={}, broker=SimpleNamespace(broker_name="paper"),
                strategy_dir=tmp_path / "strategy", preflight=True,
                preflight_guard=SimpleNamespace(commit_entered=False),
            )
        assert adapter._software_stops is None
        assert recorder.calls == []
        assert [r for r in caplog.records if "software-stop" in r.getMessage()] == []
        assert not neutral_root.exists()

    def test_init_flag_on_passes_the_neutral_root(
        self, monkeypatch, neutral_root, pipeline_stops, tmp_path,
    ):
        RunnerAdapter = self._runner_adapter()
        _contract_for(monkeypatch, neutral_root)
        _mod, recorder, _real = pipeline_stops
        adapter = RunnerAdapter(
            _cfg(True), models={}, broker=SimpleNamespace(broker_name="paper"),
            strategy_dir=tmp_path / "strategy", preflight=True,
            preflight_guard=SimpleNamespace(commit_entered=False),
        )
        assert adapter._software_stops is not None
        assert recorder.calls == [
            {"broker_name": "paper", "repo_root": neutral_root / "software-stops"},
        ]

    def test_init_contract_missing_leaves_layer_unarmed(
        self, monkeypatch, neutral_root, pipeline_stops, tmp_path, caplog,
    ):
        RunnerAdapter = self._runner_adapter()
        from adapters.commit_contract import software_stops_armed  # noqa: PLC0415
        monkeypatch.setitem(sys.modules, CONTRACT, None)
        _mod, recorder, _real = pipeline_stops
        with caplog.at_level(logging.ERROR, logger="adapters.runner"):
            adapter = RunnerAdapter(
                _cfg(True), models={}, broker=SimpleNamespace(broker_name="paper"),
                strategy_dir=tmp_path / "strategy", preflight=True,
                preflight_guard=SimpleNamespace(commit_entered=False),
            )
        assert adapter._software_stops is None
        assert software_stops_armed(adapter._software_stops) is False
        assert recorder.calls == []
        assert any(CONTRACT in r.getMessage() for r in caplog.records
                   if r.levelno == logging.ERROR)


# ── 2. the sell wrapper: seed unconditionally, before the runner, never exit ─

def _seed_block(src: str) -> str:
    """The seed block: from ``SEED_MODULE=`` through its closing ``fi``."""
    start = src.index("SEED_MODULE=")
    end = src.index("\nfi\n", start) + len("\nfi\n")
    return src[start:end]


def _bash() -> str:
    for cand in ("/bin/bash", "/usr/bin/bash", "/opt/homebrew/bin/bash"):
        if Path(cand).exists():
            return cand
    pytest.skip("bash not available")


class TestIntradaySellWrapperSeedsBeforeRunner:
    def test_seed_call_present_unconditional_and_before_runner(self):
        assert SEED_CALL in INTRADAY
        assert 'SEED_MODULE="renquant_orchestrator.software_stops_registry_contract"' in INTRADAY
        seed_idx = INTRADAY.index(SEED_CALL)
        runner_idx = INTRADAY.index(RUNNER_CALL)
        assert seed_idx < runner_idx
        # after the pinned PYTHONPATH export, the cd, and the pin preflight
        for needle in (
            'export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator',
            'cd "$REPO_DIR"',
            'source "$REPO_DIR/scripts/preflight_pin_align.sh"',
        ):
            assert INTRADAY.index(needle) < seed_idx, needle
        # unconditional: the call is not nested in an if/case of its own
        line = next(l for l in INTRADAY.splitlines() if SEED_CALL in l)
        assert line.startswith("SEED_OUT=$("), line
        # "--sell-only --intraday" is still the runner's invocation
        assert "--sell-only --intraday" in INTRADAY[runner_idx:]

    def test_same_broker_as_the_runner(self):
        seed_broker = re.search(r'seed --broker (\S+)', INTRADAY).group(1)
        runner_broker = re.search(r'--strategy renquant_104 --broker (\S+) --once', INTRADAY).group(1)
        assert seed_broker == runner_broker == "alpaca"
        # the header's documented paper-restore sed (`s/--broker alpaca/…/g`)
        # rewrites BOTH executable lines and nothing else
        code_lines = [l for l in INTRADAY.splitlines() if not l.lstrip().startswith("#")]
        assert sum(l.count("--broker alpaca") for l in code_lines) == 2

    def test_seed_failure_never_exits_and_does_not_set_errexit(self):
        block = _seed_block(INTRADAY)
        assert re.search(r"^\s*exit\b", block, re.M) is None, block
        assert "return" not in block
        between = INTRADAY[INTRADAY.index(SEED_CALL):INTRADAY.index(RUNNER_CALL)]
        assert re.search(r"^\s*exit\b", between, re.M) is None
        assert re.search(r"^\s*set -e", INTRADAY, re.M) is None
        assert "set -uo pipefail" in INTRADAY
        assert "SEED_RC=$?" in block
        assert 'if [ "$SEED_RC" -eq 0 ]' in block
        assert "continuing with the sell pass" in block
        assert 'notify "RenQuant 104 software-stops SEED FAILED"' in block

    def test_bash_syntax(self):
        bash = _bash()
        res = subprocess.run([bash, "-n", str(INTRADAY_PATH)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr

    @pytest.mark.parametrize("mode,rc,expect,notified", [
        ("verdict", 0, "software-stops registry seed OK: SEEDED:", False),
        ("exists", 0, "software-stops registry seed OK: EXISTS:", False),
        ("noverdict", 0, "WARNING: software-stops registry seed exited 0 WITHOUT", False),
        ("usage", 1, "ERROR: software-stops registry seed FAILED (exit 1)", True),
        ("corrupt", 2, "ERROR: software-stops registry seed FAILED (exit 2)", True),
        ("import", 3, "ERROR: software-stops registry seed FAILED (exit 3)", True),
    ])
    def test_block_executes_and_continues_under_every_seeder_outcome(
        self, tmp_path, mode, rc, expect, notified,
    ):
        """Run the extracted block under bash with a stub seeder: the line
        after the block is ALWAYS reached, the verdict is logged, a non-zero
        exit pages but never aborts."""
        bash = _bash()
        block = _seed_block(INTRADAY)
        fake = tmp_path / "fakepy.sh"
        outputs = {
            "verdict": 'echo "INFO seeded"; echo "SEEDED: /r/software_stops.alpaca.json (broker=alpaca)"',
            "exists": 'echo "EXISTS: /r/software_stops.alpaca.json (broker=alpaca)"',
            "noverdict": "true",
            "usage": 'echo "SEED USAGE-ERROR: bad broker" >&2',
            "corrupt": 'echo "SEED CORRUPT: registry exists but is CORRUPT" >&2',
            "import": 'echo "SEED IMPORT-FAIL: no renquant_pipeline" >&2',
        }
        fake.write_text(f"#!/usr/bin/env bash\n{outputs[mode]}\nexit {rc}\n")
        fake.chmod(0o755)
        harness = tmp_path / "run.sh"
        harness.write_text(textwrap.dedent(f"""\
            set -uo pipefail
            PYTHON="{fake}"
            LOG="{tmp_path / 'log'}"
            notify() {{ echo "NOTIFY[$1]"; }}
            {block}
            echo "REACHED-RUNNER"
            """))
        res = subprocess.run([bash, str(harness)], capture_output=True, text=True)
        assert res.returncode == 0, res.stderr
        out = res.stdout
        assert "REACHED-RUNNER" in out
        assert expect in out, out
        assert ("NOTIFY[RenQuant 104 software-stops SEED FAILED]" in out) is notified, out


# ── 3. ASSEMBLY-LEVEL: the PINNED runtime module implements `seed` ──────────

def _runtime_assembly_root() -> Path | None:
    """The runtime assembly root the wrappers export: ``RENQUANT_SUBREPO_ROOT``,
    else the repo's ``.subrepo_assembly/current.env`` export, else
    ``<repo>/.subrepo_runtime/repos`` (scripts/subrepo_env.sh resolution)."""
    env_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if env_root:
        return Path(env_root)
    current = REPO_ROOT / ".subrepo_assembly" / "current.env"
    if current.exists():
        for line in current.read_text().splitlines():
            m = re.match(r"\s*export\s+RENQUANT_SUBREPO_ROOT=['\"]?([^'\"]+)['\"]?\s*$", line)
            if m:
                return Path(m.group(1))
    default = REPO_ROOT / ".subrepo_runtime" / "repos"
    return default if default.is_dir() else None


def _pinned_orchestrator_src() -> Path:
    root = _runtime_assembly_root()
    if root is None:
        pytest.skip("runtime assembly absent (no RENQUANT_SUBREPO_ROOT, "
                    ".subrepo_assembly/current.env or .subrepo_runtime/repos)")
    src = root / "renquant-orchestrator" / "src"
    if not (src / "renquant_orchestrator" / "software_stops_registry_contract.py").exists():
        pytest.skip(f"pinned orchestrator not present at {src}")
    return src


class TestPinnedRuntimeAssemblySeeder:
    """The wrapper invokes ``python -m … seed --broker alpaca`` on the pinned
    PYTHONPATH. This proves the PINNED module implements that contract: a
    revision without the CLI (exit 0, no verdict — pre orch#1078) FAILS."""

    def test_runtime_checkout_matches_the_lock(self):
        src = _pinned_orchestrator_src()
        entry = _lock_entry("renquant-orchestrator")
        assert entry and entry.get("commit"), "subrepos.lock.json lacks the orchestrator pin"
        res = subprocess.run(["git", "-C", str(src.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True)
        if res.returncode != 0:
            pytest.skip(f"cannot read the runtime checkout's HEAD: {res.stderr.strip()}")
        assert res.stdout.strip() == entry["commit"], (
            "runtime assembly drifted from subrepos.lock.json — the seed contract "
            "below is proven against the wrong revision"
        )

    def test_pinned_module_seeds_then_reports_exists(self, tmp_path):
        src = _pinned_orchestrator_src()
        pipeline_src = _sibling_src("renquant-pipeline")
        if pipeline_src is None:
            pytest.skip("pinned renquant-pipeline not resolvable (the seeder imports its schema)")
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([str(src), str(pipeline_src)])
        env["PYTHONDONTWRITEBYTECODE"] = "1"  # never write into the pinned checkout
        env["RENQUANT_NO_NOTIFY"] = "1"
        data_root = tmp_path / "software-stops"
        cmd = [sys.executable, "-B", "-m", "renquant_orchestrator.software_stops_registry_contract",
               "seed", "--broker", "alpaca", "--data-root", str(data_root)]

        first = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert first.returncode == 0, first.stderr
        assert re.search(r"^SEEDED: ", first.stdout, re.M), (
            "pinned orchestrator exited 0 WITHOUT a SEEDED verdict — it does not "
            f"implement the seed contract the wrapper invokes. stdout={first.stdout!r} "
            f"stderr={first.stderr!r}"
        )
        expected = data_root / "data" / "rq105" / "software_stops.alpaca.json"
        assert expected.exists()
        snap = json.loads(expected.read_text())
        assert snap["stops"] == {} and snap["last_evaluated_at"] is None

        before = expected.read_bytes()
        second = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))
        assert second.returncode == 0, second.stderr
        assert re.search(r"^EXISTS: ", second.stdout, re.M), second.stdout
        assert expected.read_bytes() == before  # idempotent, never overwrites
        assert list(tmp_path.iterdir()) == [data_root]  # nothing under the cwd

    def test_pre_seeder_revision_would_fail(self, tmp_path):
        """Negative control: a module with the LOCATION functions but no CLI
        (the shape of the pre-#1078 pin) exits 0 without a verdict — and
        the assertion above catches exactly that."""
        fake = tmp_path / "fake" / "renquant_orchestrator"
        fake.mkdir(parents=True)
        (fake / "__init__.py").write_text("")
        (fake / "software_stops_registry_contract.py").write_text(
            "from pathlib import Path\n"
            "def runtime_state_root(override=None):\n    return Path('~/.renquant/runtime').expanduser()\n"
            "def software_stops_registry_root(root):\n    return root / 'software-stops'\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(tmp_path / "fake")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        res = subprocess.run(
            [sys.executable, "-B", "-m", "renquant_orchestrator.software_stops_registry_contract",
             "seed", "--broker", "alpaca", "--data-root", str(tmp_path / "root")],
            capture_output=True, text=True, env=env, cwd=str(tmp_path),
        )
        assert res.returncode == 0 and res.stdout == ""
        assert re.search(r"^SEEDED: ", res.stdout, re.M) is None
