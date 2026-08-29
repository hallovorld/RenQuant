"""Software-stop registry at the NEUTRAL root — the two umbrella follow-ups
of the orchestrator bootstrap step (orch#1078).

1. ``adapters/software_stops_wiring.py`` + ``RunnerAdapter.__init__``: the
   registry is built with ``SoftwareStopRegistry.from_config(..., repo_root=
   <neutral root>)`` where the neutral root is the orchestrator LOCATION
   contract's ``software_stops_registry_root(runtime_state_root())``
   (``~/.renquant/runtime/software-stops``, override
   ``RENQUANT_RUNTIME_STATE_ROOT``) — the ``--data-root`` the liveness pager
   and the orchestrator seeder resolve against. Never the process cwd.
   * flag off  -> ``_software_stops is None`` and ``from_config`` was still
     invoked with the neutral-root kwarg (the flag gate is from_config's
     own; it returns before touching ``repo_root`` — proved with a
     ``repo_root`` that explodes on use);
   * flag on   -> the registry path EQUALS the orchestrator's
     ``seeded_registry_path(<root>, broker)`` (import parity against the real
     sibling when importable; skipped WITH the reason otherwise) and a seed
     written by the orchestrator is read back by the runner's registry;
   * contract not importable -> ``None`` + one ERROR line, and
     ``from_config`` is NOT called (no cwd fallback), flag on or off.
2. ``scripts/intraday_sell_104.sh``: the seeder runs unconditionally
   BEFORE the runner, with the same ``--broker`` the runner receives, and
   never ``exit``s on its own failure (the sell loop is the live book's exit
   path). Structural pins (the pattern of this repo's other wrapper tests)
   plus the extracted block executed under bash with stub outcomes.

Runs in the lean ``live-broker-fractional-contract`` CI job (pytest only):
the wiring module imports nothing heavy, the pipeline / orchestrator siblings
are stubbed when absent, and the one test that needs the full strategy deps
(``RunnerAdapter.__init__`` end-to-end) skips there and runs locally.
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

def _sibling_src(name: str) -> Path | None:
    """``<RENQUANT_SUBREPO_ROOT>/<name>/src`` first (the runtime assembly the
    wrappers export), then ``subrepos.lock.json`` ``local_path``."""
    env_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
    if env_root:
        cand = Path(env_root) / name / "src"
        if cand.is_dir():
            return cand
    lock = REPO_ROOT / "subrepos.lock.json"
    if lock.exists():
        entry = next(
            (e for e in json.loads(lock.read_text())["subrepos"] if e["name"] == name),
            None,
        )
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
            "(pinned checkout predates the seeder, orch#1078)"
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


# ── 1. the wiring: neutral root, flag-off inert, fail-closed ────────────────

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


class TestFlagOffByteInert:
    @pytest.mark.parametrize("enabled", [None, False])
    def test_registry_none_and_from_config_got_the_neutral_root(
        self, monkeypatch, neutral_root, pipeline_stops, enabled,
    ):
        _contract_for(monkeypatch, neutral_root)
        _mod, recorder, _real = pipeline_stops
        reg = wiring.build_software_stop_registry(_cfg(enabled), "paper")
        assert reg is None
        assert recorder.calls == [
            {"broker_name": "paper", "repo_root": neutral_root / "software-stops"},
        ]
        # byte-inert: nothing written under the neutral root or the cwd
        assert not neutral_root.exists()
        assert list(Path.cwd().iterdir()) == []

    def test_from_config_returns_none_before_touching_repo_root(self, neutral_root):
        """The pipeline's own gate: on enabled absent/false ``from_config``
        returns None BEFORE ``repo_root`` is composed — proved with a
        ``repo_root`` object that raises the moment it is used as a path."""
        mod = _require_real(PIPELINE, "renquant-pipeline", "SoftwareStopRegistry")

        class Explodes:
            def __fspath__(self):
                raise AssertionError("repo_root was touched on the flag-off path")

            def __str__(self):
                raise AssertionError("repo_root was stringified on the flag-off path")

        for cfg in (_cfg(None), _cfg(False), {}, None):
            assert mod.SoftwareStopRegistry.from_config(
                cfg, broker_name="paper", repo_root=Explodes(),
            ) is None

    def test_runner_source_delegates_and_has_no_bare_from_config(self):
        assert "build_software_stop_registry(" in RUNNER_SRC
        assert "SoftwareStopRegistry.from_config(" not in RUNNER_SRC, (
            "runner.py must not construct the registry itself (the neutral-root "
            "wiring is the single construction site)"
        )
        assert "repo_root=root" in WIRING_SRC
        assert "software_stops_registry_root(runtime_state_root())" in WIRING_SRC


class TestFlagOnNeutralRootParity:
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
    @pytest.mark.parametrize("enabled", [None, False, True])
    def test_none_error_logged_and_from_config_never_called(
        self, monkeypatch, neutral_root, pipeline_stops, caplog, enabled,
    ):
        """Contract module unimportable -> registry None, ONE ERROR line, and
        the pipeline constructor is NOT invoked — no silent cwd fallback,
        whatever the flag says."""
        monkeypatch.setitem(sys.modules, CONTRACT, None)
        _mod, recorder, _real = pipeline_stops
        with caplog.at_level(logging.ERROR, logger="adapters.runner"):
            reg = wiring.build_software_stop_registry(_cfg(enabled), "paper")
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

    def test_init_flag_off_none_with_neutral_root_kwarg(
        self, monkeypatch, neutral_root, pipeline_stops, tmp_path,
    ):
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        try:
            from adapters.runner import RunnerAdapter  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001 — lean CI: strategy deps absent
            pytest.skip(f"adapters.runner not importable here: {exc}")
        _contract_for(monkeypatch, neutral_root)
        _mod, recorder, _real = pipeline_stops

        class Guard:
            def __init__(self):
                self.commit_entered = False

        adapter = RunnerAdapter(
            _cfg(False), models={}, broker=SimpleNamespace(broker_name="paper"),
            strategy_dir=tmp_path / "strategy", preflight=True,
            preflight_guard=Guard(),
        )
        assert adapter._software_stops is None
        assert recorder.calls == [
            {"broker_name": "paper", "repo_root": neutral_root / "software-stops"},
        ]

    def test_init_contract_missing_leaves_layer_unarmed(
        self, monkeypatch, neutral_root, pipeline_stops, tmp_path, caplog,
    ):
        pytest.importorskip("pandas")
        pytest.importorskip("numpy")
        try:
            from adapters.runner import RunnerAdapter  # noqa: PLC0415
            from adapters.commit_contract import software_stops_armed  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            pytest.skip(f"adapters.runner not importable here: {exc}")
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


def _bash() -> str:
    for cand in ("/bin/bash", "/usr/bin/bash", "/opt/homebrew/bin/bash"):
        if Path(cand).exists():
            return cand
    pytest.skip("bash not available")
