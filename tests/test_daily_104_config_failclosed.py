"""RenQuant#546: the daily runner must not substitute a DIFFERENT config.

Background. `daily_104.sh` resolves the strategy config from the PINNED
renquant-strategy-104 subrepo. It used to fall back to the umbrella copy unless
one of two env vars was set, and both default to `0` — so the default path
silently substituted a different config and did not log it. Measured 2026-07-30:
the umbrella copy names `hf_patchtst` PRIMARY with xgb shadow, while the pinned
config has exactly the inverse. PatchTST's scores are intrinsically all-negative,
so promoting it to primary fails the ordinary buy floor for every name — a silent
sell-only book, reached with nobody taking an action.

On test design, stated rather than glossed: the fail-closed branch is asserted
STRUCTURALLY, on the shell source. A behavioural test would have to invoke a
production trading script far enough to pass its credentials check, and this
suite will not do that — the failure mode of getting it wrong is placing orders.
The structural assertions below are therefore written against the exact guard
SHAPE (which env vars may and may not gate it), not against a paraphrase, so
they still fail if the guard is weakened.

`test_all_three_configs_agree_on_the_primary_scorer` IS behavioural — it reads
the real config files — and is `xfail(strict=True)` because it fails today. When
someone repairs the configs it will XPASS, which turns CI red and forces the
marker off. That ratchet is the point.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "daily_104.sh"
CFG_DIR = REPO / "backtesting" / "renquant_104"
UMBRELLA_CFG = CFG_DIR / "strategy_config.json"
GOLDEN_CFG = CFG_DIR / "strategy_config.golden.json"
PINNED_CFG = (Path("/Users/renhao/git/github/RenQuant/.subrepo_runtime/repos")
              / "renquant-strategy-104" / "configs" / "strategy_config.json")


def _resolution_block() -> str:
    """The `if ! PROD_STRATEGY_CONFIG=...` block, isolated."""
    src = SCRIPT.read_text(encoding="utf-8")
    start = src.index('if ! PROD_STRATEGY_CONFIG=')
    end = src.index('\nfi', start) + 3
    return src[start:end]


def _primary_kind(path: Path) -> str | None:
    d = json.loads(path.read_text(encoding="utf-8"))
    return ((d.get("ranking") or {}).get("panel_scoring") or {}).get("kind")


def test_resolution_failure_exits_nonzero():
    block = _resolution_block()
    assert "exit 1" in block, "a pinned-config resolution failure must exit"


def test_failclosed_is_not_gated_on_default_off_env_vars():
    """The regression: both gates default to 0, so the guard never fired."""
    block = _resolution_block()
    for var in ("RENQUANT_STRICT_SUBREPO_PATHS", "RENQUANT_OPS_FAIL_CLOSED"):
        assert var not in block, (
            f"{var} defaults to 0; gating fail-closed on it re-opens #546")


def test_the_only_fallback_escape_is_the_umbrella_runner_mode():
    """A fallback is allowed for exactly one documented mode, not by default."""
    block = _resolution_block()
    assert 'RQ_DAILY_RUNNER' in block
    assert '!= "umbrella"' in block, (
        "the exit must be taken UNLESS the runner is explicitly umbrella")
    # The fallback assignment must come AFTER the exit guard, never before it.
    assert block.index("exit 1") < block.index(
        'PROD_STRATEGY_CONFIG="$REPO_DIR/backtesting'), (
        "the umbrella fallback is reachable before the fail-closed exit")


def test_the_umbrella_fallback_is_assigned_exactly_once():
    """More than one assignment site means one of them is unguarded."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert len(re.findall(
        r'PROD_STRATEGY_CONFIG="\$REPO_DIR/backtesting', src)) == 1


def test_the_resolved_scorer_kind_is_logged_in_both_branches():
    """The 2026-07-30 investigation could not answer 'which model was primary on
    day X' from any log. That gap is what this line closes, so it must sit
    OUTSIDE the if/else, after resolution."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "RESOLVED_SCORER_KIND" in src
    i_block_end = src.index('\nfi', src.index('if ! PROD_STRATEGY_CONFIG='))
    assert src.index("RESOLVED_SCORER_KIND") > i_block_end, (
        "the log line must run for both branches, not inside one of them")
    assert "primary panel_scoring.kind=" in src


def test_the_scorer_kind_extraction_handles_a_config_without_the_key(tmp_path):
    """The logging must not be able to abort a run. Mirrors the shell one-liner."""
    import subprocess
    for payload in ({}, {"ranking": {}}, {"ranking": {"panel_scoring": {}}},
                    {"ranking": {"panel_scoring": {"kind": "xgb"}}}):
        p = tmp_path / "c.json"
        p.write_text(json.dumps(payload))
        out = subprocess.run(
            ["python3", "-c",
             'import json,sys;print((json.load(open(sys.argv[1])).get("ranking",{})'
             '.get("panel_scoring",{}) or {}).get("kind","UNKNOWN"))', str(p)],
            capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() in {"UNKNOWN", "xgb"}


@pytest.mark.xfail(strict=True, reason=(
    "FAILS TODAY and that is the point: golden and the umbrella copy both name "
    "hf_patchtst PRIMARY while the pinned config names xgb. Because the drift "
    "guard compares the umbrella copy against golden, and both carry the same "
    "inverted intent, it reports clean forever. Repairing the configs is "
    "blocked on RenQuant#544 (the trainer reads the umbrella copy, so changing "
    "its primary kind may change TRAINING, not just the fallback). When that "
    "lands this test XPASSes, CI goes red, and this marker must be removed."))
def test_all_three_configs_agree_on_the_primary_scorer():
    if not PINNED_CFG.exists():
        pytest.skip("pinned subrepo runtime not present on this machine")
    kinds = {
        "umbrella": _primary_kind(UMBRELLA_CFG),
        "golden": _primary_kind(GOLDEN_CFG),
        "pinned": _primary_kind(PINNED_CFG),
    }
    assert len(set(kinds.values())) == 1, f"primary panel_scoring.kind differs: {kinds}"


def test_the_golden_drift_guard_is_not_what_protects_this():
    """Documents WHY a separate check is needed: the drift guard is non-fatal by
    design, and it compares the umbrella copy against golden — two files that
    agree with each other while both disagree with production."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "strategy_config.golden.json" in src
    if GOLDEN_CFG.exists() and UMBRELLA_CFG.exists():
        assert _primary_kind(GOLDEN_CFG) == _primary_kind(UMBRELLA_CFG), (
            "if these ever diverge the drift guard would catch it and this "
            "test's premise needs revisiting")
