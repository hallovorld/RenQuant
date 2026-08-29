"""Exit-code classification of scripts/check_readonly_e2e.sh (orch#1066 opt c).

The verify script is exercised end-to-end as a subprocess against a THROWAWAY
repo dir (``RENQUANT_REPO_DIR``) whose "orchestrator" is a stub
``renquant_orchestrator`` package on the script's own PYTHONPATH
(``RENQUANT_SUBREPO_ROOT``). The stub writes whatever log markers a case
needs and exits with a chosen code; the real pipeline is never imported.

Contract under test (script header):
  3 = a panel-scoring artifact the shadow config references is missing
      (named path + attribution-NEUTRAL wording that points the operator at
      the previous pinned assembly), and the funnel is NOT run;
  4 = the funnel ran and its log carries `panel_scorer_load_failed` or the
      `STRUCTURAL_BLOCK — engineering condition` alert;
  0 = committed decision (unchanged);
  1 = crash / no-decision WITHOUT the structural markers (unchanged);
  2 = the shadow config itself is unreadable, OR the pinned pipeline's
      artifact resolver cannot be imported / raises (setup; never green,
      no funnel — there is no local fallback resolver).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "check_readonly_e2e.sh"
SUBREPO_ENV = REPO / "scripts" / "subrepo_env.sh"

SHADOW_NAME = "strategy_config.shadow.json"
ARTIFACT_REF = "artifacts/patchtst_shadow/pt_fake/seed_1/model.pt"
DECISION_LINE = "SHADOW-DECISION: no-trade (stub)"
ALERT_LINE = (
    "FunnelIntegrityAlert: STRUCTURAL_BLOCK — engineering condition suppressed "
    "buy capability; do NOT report this session as a normal no-trade. "
    "fired=['panel_scorer_load_failed']"
)
LOAD_FAILED_LINE = (
    "LoadScorerTask: failed to load hf_patchtst artifact "
    "backtesting/renquant_104/artifacts/patchtst_shadow/pt_fake/seed_1/model.pt "
    "— [Errno 2] No such file; reason=panel_scorer_load_failed"
)

# The stub runner: prints the lines in FAKE_RUNNER_LINES (JSON list), touches
# FAKE_RUNNER_SENTINEL so a test can prove the funnel was (not) invoked, and
# exits FAKE_RUNNER_RC.
# The stub PINNED resolver: a faithful copy of kernel.artifact_resolver.
# locate_artifact's precedence (absolute → strategy_dir → repo_root =
# strategy_dir/../..) that also touches FAKE_RESOLVER_SENTINEL so a test can
# prove the script took the import path rather than its fallback.
STUB_RESOLVER = '''
import os
from pathlib import Path
def _candidates(ref, strategy_dir, repo_root):
    p = Path(ref)
    if p.is_absolute():
        return [(p, "absolute")]
    root = repo_root if repo_root is not None else Path(strategy_dir).parent.parent
    return [(Path(strategy_dir) / p, "strategy_dir"), (root / p, "repo_root")]
def locate_artifact(ref, *, strategy_dir, repo_root=None):
    Path(os.environ["FAKE_RESOLVER_SENTINEL"]).write_text(str(ref))
    cands = _candidates(ref, strategy_dir, repo_root)
    for cand, _src in cands:
        if cand.exists():
            return cand
    return cands[0][0]
'''
BROKEN_RESOLVER = 'raise ImportError("stub: pinned resolver unavailable")\n'
RAISING_RESOLVER = '''
import os
from pathlib import Path
def locate_artifact(ref, *, strategy_dir, repo_root=None):
    Path(os.environ["FAKE_RESOLVER_SENTINEL"]).write_text(str(ref))
    raise RuntimeError("stub: resolver exploded on " + str(ref))
'''
RESOLVER_MODULE = "renquant_pipeline.kernel.artifact_resolver"
PINNED_RESOLVER_LINE = f"preflight resolver: pinned {RESOLVER_MODULE}.locate_artifact"

STUB_MAIN = '''
import json, os, sys
from pathlib import Path
Path(os.environ["FAKE_RUNNER_SENTINEL"]).write_text(" ".join(sys.argv[1:]))
for line in json.loads(os.environ.get("FAKE_RUNNER_LINES", "[]")):
    print(line, flush=True)
raise SystemExit(int(os.environ.get("FAKE_RUNNER_RC", "0")))
'''


def _panel_scoring(kind: str = "hf_patchtst", **extra) -> dict:
    ps = {"enabled": True, "kind": kind, "artifact_path": ARTIFACT_REF}
    ps.update(extra)
    return ps


class Harness:
    def __init__(self, tmp_path: Path, *, resolver: str = STUB_RESOLVER) -> None:
        self.repo = tmp_path / "umbrella"
        self.subrepos = tmp_path / "subrepos"
        self.strategy_dir = self.repo / "backtesting" / "renquant_104"
        self.config_dir = self.subrepos / "renquant-strategy-104" / "configs"
        self.sentinel = tmp_path / "runner.invoked"
        self.resolver_sentinel = tmp_path / "resolver.invoked"
        self.log = tmp_path / "e2e.log"
        self.strategy_dir.mkdir(parents=True)
        self.config_dir.mkdir(parents=True)
        (self.repo / "scripts").mkdir()
        shutil.copy(SUBREPO_ENV, self.repo / "scripts" / "subrepo_env.sh")
        pkg = self.subrepos / "renquant-orchestrator" / "src" / "renquant_orchestrator"
        pkg.mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "__main__.py").write_text(STUB_MAIN)
        # the "pinned pipeline" on the script's PYTHONPATH: only the resolver
        # module exists; shadows any site-packages install because PYTHONPATH
        # precedes it.
        kernel = self.subrepos / "renquant-pipeline" / "src" / "renquant_pipeline" / "kernel"
        kernel.mkdir(parents=True)
        (kernel.parent / "__init__.py").write_text("")
        (kernel / "__init__.py").write_text("")
        (kernel / "artifact_resolver.py").write_text(resolver)

    # -- fixtures -------------------------------------------------------
    def write_config(self, panel_scoring: dict | None) -> Path:
        cfg = {"watchlist": ["AAA"], "ranking": {}}
        if panel_scoring is not None:
            cfg["ranking"]["panel_scoring"] = panel_scoring
        path = self.config_dir / SHADOW_NAME
        path.write_text(json.dumps(cfg))
        return path

    def create_artifact(self, ref: str = ARTIFACT_REF, *, under_repo_root: bool = False) -> Path:
        base = self.repo if under_repo_root else self.strategy_dir
        p = base / ref
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x00fake-model")
        return p

    # -- run ------------------------------------------------------------
    def run(self, *, lines: list[str] = (), rc: int = 0) -> subprocess.CompletedProcess:
        env = {k: v for k, v in os.environ.items()
               if k not in {"PYTHONPATH", "RQ_DAILY_RUNNER", "RENQUANT_ASSEMBLY_DIR",
                            "RENQUANT_SUBREPO_ENV"}}
        env.update({
            "RENQUANT_REPO_DIR": str(self.repo),
            "RENQUANT_PYTHON": sys.executable,
            "RENQUANT_SUBREPO_ROOT": str(self.subrepos),
            "RENQUANT_E2E_LOG": str(self.log),
            "RENQUANT_E2E_TIMEOUT_SEC": "60",
            "FAKE_RUNNER_SENTINEL": str(self.sentinel),
            "FAKE_RESOLVER_SENTINEL": str(self.resolver_sentinel),
            "FAKE_RUNNER_LINES": json.dumps(list(lines)),
            "FAKE_RUNNER_RC": str(rc),
        })
        return subprocess.run(
            ["bash", str(SCRIPT)], env=env, capture_output=True, text=True,
            timeout=120, check=False,
        )

    @property
    def runner_invoked(self) -> bool:
        return self.sentinel.exists()

    @property
    def resolver_invoked(self) -> bool:
        return self.resolver_sentinel.exists()


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.fixture
def harness_no_resolver(tmp_path: Path) -> Harness:
    """Pinned resolver import FAILS → setup error, never green."""
    return Harness(tmp_path, resolver=BROKEN_RESOLVER)


@pytest.fixture
def harness_raising_resolver(tmp_path: Path) -> Harness:
    """Pinned resolver imports but RAISES on call → setup error, never green."""
    return Harness(tmp_path, resolver=RAISING_RESOLVER)


# ── 3: dead leg ──────────────────────────────────────────────────────────────

def test_missing_primary_artifact_exits_3_names_path_and_skips_funnel(harness):
    cfg = harness.write_config(_panel_scoring())
    # the artifact exists NOWHERE (the 2026-08-25 measurement)
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 3, res.stdout + res.stderr
    assert str(harness.strategy_dir / ARTIFACT_REF) in res.stdout
    assert str(harness.repo / ARTIFACT_REF) in res.stdout
    assert "ranking.panel_scoring.artifact_path (kind=hf_patchtst)" in res.stdout
    assert PINNED_RESOLVER_LINE in res.stdout
    assert harness.resolver_invoked, "the verdict must come from the pinned resolver"
    assert (f"DEAD_LEG detected before the funnel in {cfg}; attribute by "
            "comparing against the previous pinned assembly "
            "(scripts/promote_pin.py keeps the backup lock) — see orch#1066") in res.stdout
    # the script must NOT assert the leg's age — it never inspects the previous pin
    assert "pre-existing" not in res.stdout
    assert "not a pin-bump regression" not in res.stdout
    assert not harness.runner_invoked, "the funnel must not run on a dead leg"


def test_primary_artifact_only_at_repo_root_resolves_via_pinned_resolver(harness):
    """renquant-pipeline#301: the primary loader resolves through
    kernel.artifact_resolver.locate_artifact (strategy_dir → repo_root), so a
    copy under the umbrella root IS a live leg — the live layout since the
    pin (RenQuant/artifacts/patchtst_shadow exists, the strategy-dir twin does
    not). A preflight that still mirrored the old strategy_dir-only join
    reported a FALSE dead leg here."""
    harness.write_config(_panel_scoring())
    harness.create_artifact(under_repo_root=True)
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "DEAD_LEG" not in res.stdout
    assert PINNED_RESOLVER_LINE in res.stdout
    assert harness.resolver_invoked
    assert f"-> {harness.repo / ARTIFACT_REF}" in res.stdout
    assert harness.runner_invoked


@pytest.mark.parametrize("artifact_present", [True, False], ids=["file-present", "file-absent"])
def test_resolver_import_failure_exits_2_never_green(harness_no_resolver, artifact_present):
    """No local fallback: even with the file on disk the verify must not go
    green on a verdict the pinned loader did not produce."""
    h = harness_no_resolver
    h.write_config(_panel_scoring())
    if artifact_present:
        h.create_artifact()
    res = h.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 2, res.stdout + res.stderr
    assert f"SETUP: cannot import the pinned resolver {RESOLVER_MODULE}" in res.stdout
    assert "stub: pinned resolver unavailable" in res.stdout
    assert "DEAD_LEG" not in res.stdout
    assert "READONLY_E2E: OK" not in res.stdout
    assert not h.resolver_invoked
    assert not h.runner_invoked


def test_resolver_call_failure_exits_2_with_error_text(harness_raising_resolver):
    h = harness_raising_resolver
    h.write_config(_panel_scoring())
    h.create_artifact()
    res = h.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 2, res.stdout + res.stderr
    assert f"SETUP: pinned resolver {RESOLVER_MODULE}.locate_artifact raised" in res.stdout
    assert f"stub: resolver exploded on {ARTIFACT_REF}" in res.stdout
    assert h.resolver_invoked, "the pinned resolver was the one consulted"
    assert "DEAD_LEG" not in res.stdout
    assert not h.runner_invoked


def test_blend_component_resolves_via_repo_root_fallback(harness):
    """Blend components go through kernel.artifact_resolver (strategy_dir →
    repo_root), so a repo-root copy IS a live leg for them."""
    harness.write_config(_panel_scoring(
        kind="blend",
        artifact_path=None,
        components=[{"artifact_path": "artifacts/prod/panel-ltr.json",
                     "expected_content_sha256": "x", "expected_config_fingerprint": "y"}],
    ))
    harness.create_artifact("artifacts/prod/panel-ltr.json", under_repo_root=True)
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert harness.runner_invoked


def test_missing_blend_component_exits_3(harness):
    harness.write_config(_panel_scoring(
        kind="blend",
        artifact_path=None,
        components=[{"artifact_path": "artifacts/prod/panel-ltr.json"},
                    {"artifact_path": "artifacts/prod/panel-clf.json"}],
    ))
    harness.create_artifact("artifacts/prod/panel-ltr.json")
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 3, res.stdout + res.stderr
    assert "components[1].artifact_path" in res.stdout
    assert str(harness.strategy_dir / "artifacts/prod/panel-clf.json") in res.stdout
    assert str(harness.repo / "artifacts/prod/panel-clf.json") in res.stdout
    assert not harness.runner_invoked


def test_enabled_global_calibration_artifact_is_checked(harness):
    harness.write_config(_panel_scoring(
        global_calibration={"enabled": True,
                            "artifact_path": "artifacts/shadow/calib.json"},
    ))
    harness.create_artifact()
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 3, res.stdout + res.stderr
    assert "global_calibration.artifact_path" in res.stdout
    assert str(harness.strategy_dir / "artifacts/shadow/calib.json") in res.stdout


def test_enabled_global_calibration_at_repo_root_resolves(harness):
    harness.write_config(_panel_scoring(
        global_calibration={"enabled": True,
                            "artifact_path": "artifacts/shadow/calib.json"},
    ))
    harness.create_artifact()
    harness.create_artifact("artifacts/shadow/calib.json", under_repo_root=True)
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert f"global_calibration.artifact_path -> {harness.repo / 'artifacts/shadow/calib.json'}" in res.stdout


def test_disabled_global_calibration_artifact_is_ignored(harness):
    harness.write_config(_panel_scoring(
        global_calibration={"enabled": False,
                            "artifact_path": "artifacts/shadow/calib.json"},
    ))
    harness.create_artifact()
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr


# ── 4: structural block ──────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "lines, rc, marker",
    [
        ([LOAD_FAILED_LINE, "QP skipped"], 1, "panel_scorer_load_failed"),
        ([ALERT_LINE], 0, "STRUCTURAL_BLOCK — engineering condition"),
        ([LOAD_FAILED_LINE, ALERT_LINE], 0, "panel_scorer_load_failed"),
    ],
    ids=["load-failed+rc1", "alert+rc0-no-decision", "both"],
)
def test_structural_markers_exit_4_and_print_marker_line(harness, lines, rc, marker):
    harness.write_config(_panel_scoring())
    harness.create_artifact()
    res = harness.run(lines=lines, rc=rc)
    assert res.returncode == 4, res.stdout + res.stderr
    assert harness.runner_invoked
    assert "structural engineering failure in the shadow scorer chain" in res.stdout
    assert "not a decision outcome" in res.stdout
    assert "whether it predates the bump is not established here" in res.stdout
    # the matching log line is echoed verbatim
    first_marker_line = next(l for l in lines if marker in l)
    assert first_marker_line in res.stdout


def test_markers_with_committed_decision_still_exit_0_with_warn(harness):
    """Exit 4 replaces the generic 1 only; a committed decision keeps 0."""
    harness.write_config(_panel_scoring())
    harness.create_artifact()
    res = harness.run(lines=[ALERT_LINE, DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "WARN — decision committed but the log carries a structural-block marker" in res.stdout


# ── 0 / 1 / 2: unchanged semantics ───────────────────────────────────────────

def test_clean_decision_exits_0(harness):
    harness.write_config(_panel_scoring())
    harness.create_artifact()
    res = harness.run(lines=["pipeline progress ...", DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "READONLY_E2E: OK" in res.stdout
    assert harness.runner_invoked
    # the stub received the real invocation shape
    assert "--broker readonly-alpaca" in harness.sentinel.read_text()
    assert "--strategy-config-name strategy_config.shadow.json" in harness.sentinel.read_text()


def test_crash_without_markers_exits_1(harness):
    harness.write_config(_panel_scoring())
    harness.create_artifact()
    res = harness.run(lines=["Traceback (most recent call last):", "KeyError: 'x'"], rc=1)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "READONLY_E2E: FAIL — runner rc=1" in res.stdout
    assert "exit 4" not in res.stdout


def test_silent_run_without_markers_exits_1(harness):
    harness.write_config(_panel_scoring())
    harness.create_artifact()
    res = harness.run(lines=["pipeline progress only"], rc=0)
    assert res.returncode == 1, res.stdout + res.stderr
    assert "NO committed decision" in res.stdout


def test_unreadable_shadow_config_exits_2(harness):
    # no config written at all
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 2, res.stdout + res.stderr
    assert str(harness.config_dir / SHADOW_NAME) in res.stdout
    assert not harness.runner_invoked


def test_panel_scoring_disabled_skips_preflight(harness):
    harness.write_config({"enabled": False, "artifact_path": ARTIFACT_REF})
    res = harness.run(lines=[DECISION_LINE], rc=0)
    assert res.returncode == 0, res.stdout + res.stderr
    assert harness.runner_invoked
