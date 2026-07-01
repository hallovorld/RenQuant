"""Opt-in placebo-clean difference test + dual-logging for the §5.2 WF gate.

`scripts/run_wf_gate.py` gains an ADDITIVE, OPT-IN, OFF-BY-DEFAULT placebo
evaluation mode. The default ``absolute`` mode uses the current ceiling
(|placebo_ic| < max(0.005, 0.5×|aligned_real_ic|)), which is structurally
unsatisfiable for the daily-sampled 60-day label because the overlapping label
carries a ~+0.04 embargo-leakage placebo floor (see
doc/research/2026-06-10-m6-placebo-gate-verdict.md). The opt-in ``difference``
mode instead requires a genuine edge ABOVE that floor:
``aligned_real_ic - placebo_ic > margin`` (pre-registered margin).

These tests pin BOTH:
  * default-strict: ``absolute`` mode reproduces the historical verdict, so
    merging does not change live promotion behaviour; and
  * opt-in: ``difference`` mode passes iff ``real_ic - placebo_ic > margin``,
    and BOTH verdicts are always computed (dual-logging / shadow).

Synthetic IC inputs only — no full gate run required.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_wf_gate  # noqa: E402


# Real prod-artifact evidence (backtesting/renquant_104/artifacts/prod/
# panel-ltr.alpha158_fund.json): the absolute ceiling FAILS while a genuine
# edge exists above the placebo floor.
REAL_PLACEBO_IC = 0.040151728638540385
REAL_ALIGNED_REAL_IC = 0.052921747444921494


def _historical_pass_placebo(placebo_ic: float, aligned_real_ic: float) -> bool:
    """The exact pre-change ``pass_placebo`` expression from run_sanity_battery."""
    return (
        (placebo_ic == placebo_ic)
        and (aligned_real_ic == aligned_real_ic)
        and (
            abs(placebo_ic) < run_wf_gate._placebo_ic_threshold(aligned_real_ic)
            if aligned_real_ic != 0
            else True
        )
    )


# --------------------------------------------------------------------------- #
# Default behaviour: absolute mode reproduces the historical verdict.
# --------------------------------------------------------------------------- #
def test_absolute_mode_reproduces_historical_verdict_random_sweep():
    """Over a random IC sweep, absolute-mode verdict == historical gate verdict."""
    rng = random.Random(1234)
    for _ in range(3000):
        placebo = rng.uniform(-0.25, 0.25)
        real = rng.uniform(-0.25, 0.25)
        ev = run_wf_gate._evaluate_placebo(placebo, real, mode="absolute")
        assert ev["passed"] == _historical_pass_placebo(placebo, real), (
            placebo,
            real,
        )


@pytest.mark.parametrize(
    "placebo, real",
    [
        (float("nan"), 0.05),      # placebo unavailable → fail closed
        (0.05, float("nan")),      # aligned_real unavailable → fail closed
        (0.001, 0.0),              # aligned_real == 0 → auto-pass (historical)
        (0.02, 0.0),               # aligned_real == 0 → auto-pass (historical)
        (0.0026, 0.02),            # |placebo| < 0.005 floor → pass
        (0.006, 0.001),            # |placebo| above floor → fail
        (REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC),  # real evidence → fail
    ],
)
def test_absolute_mode_edge_cases(placebo, real):
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="absolute")
    assert ev["passed"] == _historical_pass_placebo(placebo, real)


def test_default_evaluate_placebo_mode_is_absolute():
    """No explicit mode → absolute is authoritative (default-safe)."""
    ev = run_wf_gate._evaluate_placebo(REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC)
    assert ev["mode"] == "absolute"
    assert ev["passed"] is False  # absolute ceiling fails on the real evidence


def test_unknown_mode_falls_back_to_absolute():
    ev = run_wf_gate._evaluate_placebo(
        REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC, mode="not-a-real-mode"
    )
    assert ev["mode"] == "absolute"
    assert ev["passed"] == _historical_pass_placebo(
        REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC
    )


# --------------------------------------------------------------------------- #
# Opt-in difference mode: pass iff real_ic - placebo_ic > margin.
# --------------------------------------------------------------------------- #
def test_difference_mode_passes_iff_edge_above_margin():
    """Real evidence: real-placebo = +0.0128 > 0.01 margin → PASS (edge above floor)."""
    ev = run_wf_gate._evaluate_placebo(
        REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC, mode="difference", margin=0.01
    )
    assert ev["mode"] == "difference"
    assert ev["passed"] is True
    diff = REAL_ALIGNED_REAL_IC - REAL_PLACEBO_IC
    assert ev["difference"]["difference"] == pytest.approx(diff)
    # And the SAME numbers fail the (unchanged) absolute ceiling in shadow.
    assert ev["absolute"]["passed"] is False


def test_difference_mode_margin_boundary():
    """Strict '>' at the margin: equal fails, just-above passes."""
    real, placebo = 0.06, 0.04  # diff = 0.02 exactly
    at = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=0.02)
    assert at["passed"] is False  # 0.02 > 0.02 is False
    above = run_wf_gate._evaluate_placebo(
        placebo, real, mode="difference", margin=0.0199
    )
    assert above["passed"] is True


def test_difference_mode_fails_when_placebo_matches_real():
    """No edge above the floor (real≈placebo) → difference fails."""
    ev = run_wf_gate._evaluate_placebo(0.04, 0.041, mode="difference", margin=0.01)
    assert ev["passed"] is False


def test_difference_mode_unavailable_fails_closed():
    ev = run_wf_gate._evaluate_placebo(
        float("nan"), 0.05, mode="difference", margin=0.01
    )
    assert ev["difference"]["available"] is False
    assert ev["passed"] is False


# --------------------------------------------------------------------------- #
# Dual-logging / shadow: BOTH verdicts computed regardless of authoritative mode.
# --------------------------------------------------------------------------- #
def test_dual_verdicts_present_in_both_modes():
    for mode in ("absolute", "difference"):
        ev = run_wf_gate._evaluate_placebo(
            REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC, mode=mode, margin=0.01
        )
        assert ev["absolute"]["mode"] == "absolute"
        assert ev["difference"]["mode"] == "difference"
        assert ev["absolute"]["passed"] is False   # ceiling fails on evidence
        assert ev["difference"]["passed"] is True   # difference passes on evidence


def test_dual_log_message_emits_both_verdicts_and_numbers():
    ev = run_wf_gate._evaluate_placebo(
        REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC, mode="difference", margin=0.01
    )
    msg = run_wf_gate._placebo_dual_log_message(ev)
    assert "absolute=FAIL" in msg
    assert "difference=PASS" in msg
    assert "authoritative=difference" in msg
    # numeric evidence surfaced for shadow comparison
    assert "+0.0402" in msg  # |placebo_ic|
    assert "+0.0265" in msg  # absolute threshold
    assert "+0.0128" in msg  # real - placebo
    assert "margin=+0.0100" in msg


# --------------------------------------------------------------------------- #
# Config resolution: opt-in, off-by-default, CLI overrides config.
# --------------------------------------------------------------------------- #
def _write_cfg(tmp_dir: Path, name: str, block: dict | None) -> str:
    cfg: dict = {"ranking": {"panel_scoring": {"enabled": True}}}
    if block is not None:
        cfg["wf_gate"] = block
    (tmp_dir / name).write_text(json.dumps(cfg))
    return name


def test_resolve_placebo_settings_defaults_absolute(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(tmp_path, "strategy_config.p_default.json", None)
    out = run_wf_gate._resolve_placebo_settings(name)
    assert out["mode"] == "absolute"
    assert out["difference_margin"] == run_wf_gate.DEFAULT_PLACEBO_DIFFERENCE_MARGIN
    assert out["source"] == "default"


def test_resolve_placebo_settings_config_opt_in(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.p_diff.json",
        {"placebo_mode": "difference", "placebo_difference_margin": 0.02},
    )
    out = run_wf_gate._resolve_placebo_settings(name)
    assert out["mode"] == "difference"
    assert out["difference_margin"] == 0.02
    assert out["source"] == "config"


def test_resolve_placebo_settings_cli_overrides_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.p_diff2.json",
        {"placebo_mode": "difference"},
    )
    out = run_wf_gate._resolve_placebo_settings(name, cli_mode="absolute")
    assert out["mode"] == "absolute"
    assert out["source"] == "cli"


def test_resolve_placebo_settings_bogus_mode_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path, "strategy_config.p_bogus.json", {"placebo_mode": "weird"}
    )
    out = run_wf_gate._resolve_placebo_settings(name)
    assert out["mode"] == "absolute"


def test_resolve_placebo_settings_missing_config_defaults(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    out = run_wf_gate._resolve_placebo_settings("strategy_config.does_not_exist.json")
    assert out["mode"] == "absolute"
    assert out["difference_margin"] == run_wf_gate.DEFAULT_PLACEBO_DIFFERENCE_MARGIN
