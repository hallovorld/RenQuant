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


# --------------------------------------------------------------------------- #
# Combined policy (review CHANGES_REQUESTED on PR #422): the difference mode
# must require BOTH a pre-registered POSITIVE real-IC floor
# (aligned_real_ic > real_ic_floor) AND the incremental criterion
# (aligned_real_ic - placebo_ic > margin). Non-positive real IC FAILS
# regardless of the difference; invalid/non-finite config FAILS CLOSED.
# --------------------------------------------------------------------------- #
FLOOR = run_wf_gate.DEFAULT_PLACEBO_REAL_IC_FLOOR  # +0.01 (M6 genuine_ic_floor)
MARGIN = run_wf_gate.DEFAULT_PLACEBO_DIFFERENCE_MARGIN  # +0.01


def test_default_real_ic_floor_is_positive_m6_value():
    """The pre-registered floor default is the M6 genuine_ic_floor positive value."""
    assert run_wf_gate.DEFAULT_PLACEBO_REAL_IC_FLOOR == 0.01
    assert run_wf_gate.DEFAULT_PLACEBO_REAL_IC_FLOOR > 0.0


def test_negative_real_ic_fails_even_when_diff_exceeds_margin():
    """Reviewer's exact hole: real=-0.01, placebo=-0.03 → diff=+0.02 > margin,
    yet the deployable signal is directionally HARMFUL → must FAIL."""
    real, placebo = -0.01, -0.03
    assert (real - placebo) > MARGIN  # incremental criterion alone would pass
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is False
    assert ev["difference"]["incremental_ok"] is True   # diff clears margin
    assert ev["difference"]["real_ic_floor_ok"] is False  # but floor blocks it
    # The shared helper (used by both global and regime paths) agrees.
    assert run_wf_gate._placebo_difference_pass(real, placebo, MARGIN, FLOOR) is False


def test_zero_real_ic_fails():
    """Zero real IC is not > a positive floor → FAIL even with a huge diff."""
    real, placebo = 0.0, -0.05
    assert (real - placebo) > MARGIN
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is False
    assert run_wf_gate._placebo_difference_pass(real, placebo, MARGIN, FLOOR) is False


def test_negative_placebo_with_positive_real_above_floor_passes():
    """A genuinely positive real IC above the floor is not penalised by a
    negative placebo — the incremental edge is real → PASS."""
    real, placebo = 0.05, -0.02  # real > floor, diff = 0.07 > margin
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is True
    assert ev["difference"]["real_ic_floor_ok"] is True
    assert ev["difference"]["incremental_ok"] is True


def test_negative_placebo_with_negative_real_fails():
    """Negative placebo does NOT rescue a negative real IC."""
    real, placebo = -0.005, -0.20
    assert (real - placebo) > MARGIN  # big positive diff
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is False


def test_sign_reversal_real_negative_placebo_positive_fails():
    """Sign reversal (real<0, placebo>0): diff is negative AND real below floor
    → FAIL on both sub-criteria."""
    real, placebo = -0.02, 0.01
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is False
    assert ev["difference"]["real_ic_floor_ok"] is False
    assert ev["difference"]["incremental_ok"] is False


def test_barely_positive_real_below_floor_fails():
    """Positive but below the pre-registered floor → FAIL (not just >0)."""
    real, placebo = 0.005, -0.10  # real>0 but < 0.01 floor; diff huge
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is False
    assert ev["difference"]["real_ic_floor_ok"] is False


def test_real_ic_floor_boundary_is_strict():
    """aligned_real_ic == floor FAILS (strict >); just above PASSES."""
    placebo = -0.5  # keep the diff comfortably above margin in both cases
    at = run_wf_gate._evaluate_placebo(
        placebo, FLOOR, mode="difference", margin=MARGIN, real_ic_floor=FLOOR
    )
    assert at["passed"] is False  # FLOOR > FLOOR is False
    above = run_wf_gate._evaluate_placebo(
        placebo, FLOOR + 1e-6, mode="difference", margin=MARGIN, real_ic_floor=FLOOR
    )
    assert above["passed"] is True


def test_positive_real_and_diff_above_margin_passes():
    """Both criteria satisfied → PASS (the intended happy path)."""
    real, placebo = 0.05, 0.02  # real > floor, diff = 0.03 > margin
    ev = run_wf_gate._evaluate_placebo(placebo, real, mode="difference", margin=MARGIN)
    assert ev["passed"] is True


@pytest.mark.parametrize("bad_margin", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_margin_fails_closed(bad_margin):
    """Non-finite margin FAILS CLOSED even on strong real evidence."""
    ev = run_wf_gate._evaluate_placebo(
        0.02, 0.20, mode="difference", margin=bad_margin
    )
    assert ev["passed"] is False
    assert run_wf_gate._placebo_difference_pass(0.20, 0.02, bad_margin, FLOOR) is False


@pytest.mark.parametrize("bad_floor", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_floor_fails_closed(bad_floor):
    """Non-finite real-IC floor FAILS CLOSED (invalid config is not permissive)."""
    ev = run_wf_gate._evaluate_placebo(
        0.02, 0.20, mode="difference", margin=MARGIN, real_ic_floor=bad_floor
    )
    assert ev["passed"] is False
    assert run_wf_gate._placebo_difference_pass(0.20, 0.02, MARGIN, bad_floor) is False


@pytest.mark.parametrize("bad_ic", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_ic_inputs_fail_closed(bad_ic):
    """Non-finite IC inputs (real or placebo) FAIL CLOSED in difference mode."""
    assert run_wf_gate._placebo_difference_pass(bad_ic, 0.0, MARGIN, FLOOR) is False
    assert run_wf_gate._placebo_difference_pass(0.20, bad_ic, MARGIN, FLOOR) is False
    ev = run_wf_gate._evaluate_placebo(bad_ic, 0.20, mode="difference", margin=MARGIN)
    assert ev["passed"] is False
    assert ev["difference"]["available"] is False


def test_real_evidence_still_passes_under_combined_policy():
    """The real prod artifact evidence clears BOTH criteria (regression guard):
    real=+0.0529 > +0.01 floor AND real-placebo=+0.0128 > +0.01 margin."""
    ev = run_wf_gate._evaluate_placebo(
        REAL_PLACEBO_IC, REAL_ALIGNED_REAL_IC, mode="difference", margin=0.01
    )
    assert ev["passed"] is True
    assert ev["difference"]["real_ic_floor_ok"] is True
    assert ev["difference"]["incremental_ok"] is True


def test_absolute_mode_ignores_real_ic_floor():
    """The floor only governs the difference verdict; absolute stays bit-identical."""
    # Negative real IC that the absolute ceiling would (historically) pass via
    # the aligned_real==0-style auto-pass is unaffected; here we just confirm the
    # absolute verdict does not consult the floor.
    ev = run_wf_gate._evaluate_placebo(
        -0.03, -0.01, mode="absolute", margin=MARGIN, real_ic_floor=0.99
    )
    assert ev["mode"] == "absolute"
    assert ev["passed"] == _historical_pass_placebo(-0.03, -0.01)


# --------------------------------------------------------------------------- #
# Config resolution: floor default + opt-in + FAIL-CLOSED on invalid config.
# --------------------------------------------------------------------------- #
def test_resolve_placebo_settings_defaults_include_real_ic_floor(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(tmp_path, "strategy_config.floor_default.json", None)
    out = run_wf_gate._resolve_placebo_settings(name)
    assert out["real_ic_floor"] == run_wf_gate.DEFAULT_PLACEBO_REAL_IC_FLOOR


def test_resolve_placebo_settings_real_ic_floor_from_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.floor_cfg.json",
        {"placebo_mode": "difference", "placebo_real_ic_floor": 0.03},
    )
    out = run_wf_gate._resolve_placebo_settings(name)
    assert out["real_ic_floor"] == 0.03


def test_resolve_placebo_settings_invalid_margin_fails_closed(tmp_path, monkeypatch):
    """A SPECIFIED but unparseable margin resolves to NaN → difference fails closed."""
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.bad_margin.json",
        {"placebo_mode": "difference", "placebo_difference_margin": "not-a-number"},
    )
    out = run_wf_gate._resolve_placebo_settings(name)
    assert not run_wf_gate._is_finite_number(out["difference_margin"])
    ev = run_wf_gate._evaluate_placebo(
        0.02, 0.20, mode=out["mode"], margin=out["difference_margin"],
        real_ic_floor=out["real_ic_floor"],
    )
    assert ev["passed"] is False


def test_resolve_placebo_settings_invalid_floor_fails_closed(tmp_path, monkeypatch):
    """A SPECIFIED non-finite floor resolves to NaN → difference fails closed."""
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.bad_floor.json",
        {"placebo_mode": "difference", "placebo_real_ic_floor": None},
    )
    out = run_wf_gate._resolve_placebo_settings(name)
    assert not run_wf_gate._is_finite_number(out["real_ic_floor"])
    ev = run_wf_gate._evaluate_placebo(
        0.02, 0.20, mode=out["mode"], margin=out["difference_margin"],
        real_ic_floor=out["real_ic_floor"],
    )
    assert ev["passed"] is False


def test_resolve_placebo_settings_cli_floor_overrides_config(tmp_path, monkeypatch):
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _write_cfg(
        tmp_path,
        "strategy_config.floor_cli.json",
        {"placebo_mode": "difference", "placebo_real_ic_floor": 0.03},
    )
    out = run_wf_gate._resolve_placebo_settings(name, cli_real_ic_floor=0.05)
    assert out["real_ic_floor"] == 0.05


# --------------------------------------------------------------------------- #
# Per-regime sanity verdict (PR #422 second-round CHANGES_REQUESTED).
#
# The `difference` mode must replace ONLY the placebo evaluation of the
# per-regime sanity gate; every original quality/coverage condition
# (`mean_ic >= min_mean_ic`, `n_dates >= min_dates`) must still apply in BOTH
# modes. The prior cut assigned regime `passed` from
# `_placebo_difference_pass(aligned_real60, placebo60, margin, floor)` ALONE, so
# a regime could pass with poor/negative full-regime `mean_ic` as long as the
# 60-shift aligned subset cleared the floor+margin. These tests pin the fix.
# --------------------------------------------------------------------------- #
MIN_MEAN_IC = 0.02        # run_sanity_battery per-regime quality floor
MAX_PLACEBO_RATIO = 0.5   # run_sanity_battery per-regime absolute ceiling ratio


def _regime_pass(mean_ic, aligned_real60, placebo60, *, mode, margin=MARGIN,
                 real_ic_floor=FLOOR, min_mean_ic=MIN_MEAN_IC,
                 max_placebo_ratio=MAX_PLACEBO_RATIO):
    return run_wf_gate._regime_sanity_pass(
        mean_ic,
        aligned_real60,
        placebo60,
        mode=mode,
        min_mean_ic=min_mean_ic,
        max_placebo_ratio=max_placebo_ratio,
        margin=margin,
        real_ic_floor=real_ic_floor,
    )


def _historical_regime_pass_absolute(mean_ic, aligned_real60, placebo60,
                                     min_mean_ic=MIN_MEAN_IC,
                                     max_placebo_ratio=MAX_PLACEBO_RATIO):
    """The exact pre-refactor absolute-mode per-regime verdict from the loop in
    run_sanity_battery — the shadow reference guarding live behaviour."""
    try:
        mean_ic_f = float(mean_ic)
    except (TypeError, ValueError):
        mean_ic_f = float("nan")
    placebo_ok = True
    if placebo60 is not None and mean_ic_f == mean_ic_f:
        placebo_ref = mean_ic_f
        try:
            aligned_real60_f = float(aligned_real60)
            if aligned_real60_f == aligned_real60_f:
                placebo_ref = aligned_real60_f
        except (TypeError, ValueError):
            placebo_ref = mean_ic_f
        placebo_ok = abs(float(placebo60)) <= max(
            0.005, max_placebo_ratio * abs(placebo_ref)
        )
    return bool(
        mean_ic_f == mean_ic_f and mean_ic_f >= min_mean_ic and placebo_ok
    )


def test_regime_absolute_mode_reproduces_historical_verdict_random_sweep():
    """Refactor is bit-identical in absolute mode → live behaviour unchanged."""
    rng = random.Random(4321)
    for _ in range(3000):
        mean_ic = rng.uniform(-0.1, 0.1)
        real60 = rng.uniform(-0.1, 0.1)
        placebo60 = rng.uniform(-0.1, 0.1)
        assert _regime_pass(mean_ic, real60, placebo60, mode="absolute") == (
            _historical_regime_pass_absolute(mean_ic, real60, placebo60)
        ), (mean_ic, real60, placebo60)


def test_regime_difference_fails_when_mean_ic_below_min_even_if_placebo_passes():
    """Reviewer's exact regression: aligned_real60 clears the placebo test
    (floor + margin) but the full-regime mean_ic is below min_mean_ic → the
    regime must FAIL. Prior cut passed it."""
    real60, placebo60 = 0.05, 0.02     # real>floor(0.01), diff=0.03>margin(0.01)
    weak_mean_ic = 0.005               # < min_mean_ic (0.02)
    # The placebo sub-verdict alone WOULD pass ...
    assert run_wf_gate._placebo_difference_pass(
        real60, placebo60, MARGIN, FLOOR
    ) is True
    # ... but the full regime verdict must still enforce mean_ic >= min_mean_ic.
    assert _regime_pass(weak_mean_ic, real60, placebo60, mode="difference") is False
    # And absolute mode agrees it fails on the weak mean_ic.
    assert _regime_pass(weak_mean_ic, real60, placebo60, mode="absolute") is False


def test_regime_difference_passes_when_mean_ic_and_placebo_both_clear():
    """Happy path: strong full-regime mean_ic AND placebo difference test clear."""
    real60, placebo60 = 0.05, 0.02
    strong_mean_ic = 0.03              # >= min_mean_ic
    assert _regime_pass(strong_mean_ic, real60, placebo60, mode="difference") is True


def test_regime_difference_negative_mean_ic_fails():
    """A negative full-regime mean_ic fails regardless of the aligned subset."""
    real60, placebo60 = 0.05, 0.02     # placebo sub-verdict would pass
    assert _regime_pass(-0.03, real60, placebo60, mode="difference") is False


def test_regime_difference_nan_or_missing_mean_ic_fails():
    """A NaN / missing / non-numeric mean_ic fails the quality gate in difference
    mode — identically to absolute mode (the mean_ic condition is unchanged; the
    mode replaces ONLY the placebo sub-verdict)."""
    for bad in (float("nan"), None, "n/a"):
        assert _regime_pass(bad, 0.05, 0.02, mode="difference") is False
        # Same quality gate applies in absolute mode → identical verdict.
        assert _regime_pass(bad, 0.05, 0.02, mode="absolute") is False


def test_regime_difference_placebo_subverdict_still_enforced():
    """Strong mean_ic does NOT rescue a failing placebo sub-verdict:
    negative aligned_real60 below the floor (reviewer's global hole, at the
    regime level) → FAIL even though diff > margin and mean_ic is strong."""
    strong_mean_ic = 0.05
    real60, placebo60 = -0.01, -0.03   # diff=+0.02>margin but real60<floor
    assert (real60 - placebo60) > MARGIN
    assert _regime_pass(strong_mean_ic, real60, placebo60, mode="difference") is False


def test_regime_difference_clears_absolute_ceiling_when_edge_is_real():
    """The whole point: a regime whose placebo60 EXCEEDS the absolute ceiling
    still passes in difference mode when mean_ic and the real edge are genuine."""
    mean_ic, real60, placebo60 = 0.05, 0.0529, 0.0402  # real prod-shaped numbers
    # Fails the absolute ceiling (|0.0402| > 0.5*0.0529=0.0265) ...
    assert _regime_pass(mean_ic, real60, placebo60, mode="absolute") is False
    # ... but passes the difference test (real>floor AND diff=0.0127>margin).
    assert _regime_pass(mean_ic, real60, placebo60, mode="difference") is True


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_regime_difference_nonfinite_config_fails_closed(bad):
    """Non-finite margin or floor FAILS the regime CLOSED even on strong inputs."""
    mean_ic, real60, placebo60 = 0.05, 0.20, 0.0
    assert _regime_pass(mean_ic, real60, placebo60, mode="difference", margin=bad) is False
    assert _regime_pass(
        mean_ic, real60, placebo60, mode="difference", real_ic_floor=bad
    ) is False
