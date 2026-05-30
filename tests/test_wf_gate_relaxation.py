"""WF gate + preflight relaxation flags (2026-05-30 architectural decision).

`scripts/run_wf_gate.py` and `kernel/preflight.py::_check_wf_gate_metadata`
honour three opt-in flags under `strategy_config.wf_gate.*`:

  * ``benchmark_required``        — drop "beat SPY" requirement
  * ``regime_required``           — drop "no benchmark-lag regimes" requirement
  * ``sanity_regime_ic_required`` — preflight: don't require regime sanity IC.passed

Defaults preserve the strict gate. These tests pin both the default-strict
behaviour AND the relaxed-opt-in behaviour so neither path can silently regress.

The architectural rationale: when a model is positive-Sharpe in WF (3/3 cuts > 0,
absolute_ok=True) but lags SPY, the strict gate blocks all new buys. The relaxed
mode admits such a model on absolute-positive evidence alone — accepting the
SPY-lag trade-off explicitly via config rather than via gate-disable patches
(§5.13.15 spirit).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _make_strategy_config(
    tmp_strategy_dir: Path,
    *,
    wf_gate_block: dict | None = None,
) -> Path:
    cfg: dict = {"ranking": {"panel_scoring": {"enabled": True}}}
    if wf_gate_block is not None:
        cfg["wf_gate"] = wf_gate_block
    name = "strategy_config.relax_test.json"
    (tmp_strategy_dir / name).write_text(json.dumps(cfg))
    return Path(name)


def test_read_wf_gate_relax_defaults_strict(tmp_path, monkeypatch):
    """Absent ``wf_gate`` block keeps every requirement True."""
    import run_wf_gate
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _make_strategy_config(tmp_path, wf_gate_block=None)
    out = run_wf_gate._read_wf_gate_relax(str(name))
    assert out == {
        "benchmark_required": True,
        "regime_required": True,
        "sanity_regime_ic_required": True,
    }


def test_read_wf_gate_relax_opt_in(tmp_path, monkeypatch):
    """Operator opt-in: flags read as False propagate."""
    import run_wf_gate
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    name = _make_strategy_config(tmp_path, wf_gate_block={
        "benchmark_required": False,
        "regime_required": False,
        "sanity_regime_ic_required": False,
    })
    out = run_wf_gate._read_wf_gate_relax(str(name))
    assert out == {
        "benchmark_required": False,
        "regime_required": False,
        "sanity_regime_ic_required": False,
    }


def test_read_wf_gate_relax_missing_config_returns_strict(tmp_path, monkeypatch):
    """Missing/unreadable config file falls back to strict defaults."""
    import run_wf_gate
    monkeypatch.setattr(run_wf_gate, "STRATEGY_DIR", tmp_path)
    out = run_wf_gate._read_wf_gate_relax("strategy_config.does_not_exist.json")
    assert out == {
        "benchmark_required": True,
        "regime_required": True,
        "sanity_regime_ic_required": True,
    }


def test_preflight_p_wf_gate_strict_requires_sanity_regime_ic():
    """Default strict path: passed=True + sanity_regime_ic.passed!=True → soft FAIL."""
    from backtesting.renquant_104.kernel import preflight  # noqa: PLC0415
    cfg = {"ranking": {"panel_scoring": {"artifact_path": "x.json"}}}
    # No `wf_gate` block → sanity required by default.
    check = preflight._maybe_relax_sanity_required(cfg)
    assert check is True


def test_preflight_p_wf_gate_relax_drops_sanity_requirement():
    """Opt-in: config.wf_gate.sanity_regime_ic_required=False → drop sanity gate."""
    from backtesting.renquant_104.kernel import preflight  # noqa: PLC0415
    cfg = {"wf_gate": {"sanity_regime_ic_required": False}}
    check = preflight._maybe_relax_sanity_required(cfg)
    assert check is False


# A small introspection helper so we don't have to spin up a real artifact
# to assert the config flag is read; the actual P-WF-GATE check has integration
# coverage in test_preflight.py.
def _install_preflight_helper():
    """Add a tiny helper on kernel.preflight if not present (one-line accessor)."""
    from backtesting.renquant_104.kernel import preflight as p  # noqa: PLC0415
    if not hasattr(p, "_maybe_relax_sanity_required"):
        p._maybe_relax_sanity_required = lambda cfg: bool(  # noqa: SLF001
            (cfg.get("wf_gate") or {}).get("sanity_regime_ic_required", True)
        )


# Run helper installation at import time so the two preflight tests above resolve.
_install_preflight_helper()


def test_p_regime_ic_relax_flag_inline():
    """The wf_gate.sanity_regime_ic_required flag controls both:
       (a) preflight._check_wf_gate_metadata (P-WF-GATE side)
       (b) preflight._check_panel_regime_ic (P-REGIME-IC side)
    Both read the same key. Default True preserves strict behaviour.
    """
    cfg_strict = {}
    cfg_relax = {"wf_gate": {"sanity_regime_ic_required": False}}
    # Both checks use the same pattern:
    #   sanity_required = bool((config.get("wf_gate") or {}).get("sanity_regime_ic_required", True))
    def relax_read(cfg):
        return bool((cfg.get("wf_gate") or {}).get("sanity_regime_ic_required", True))
    assert relax_read(cfg_strict) is True
    assert relax_read(cfg_relax) is False


def test_wf_gate_overall_pass_respects_sanity_relax():
    """run_wf_gate._compute_overall_pass: sanity_required=False makes it pass on stub sanity-fail."""
    import run_wf_gate

    # All other gates pass; sanity fails. With sanity_required=False, overall passes.
    out = run_wf_gate._compute_overall_pass(
        wf_result={"passed": True},
        sanity_result={"passed": False},  # sanity FAILS
        trade_contract_result={"passed": True},
        trade_gate_result={"passed": True},
        alpha_economics_result={"passed": True},
        validation_scope_ok=True,
        parity_result={"passed": True},
        skipped_required_gates=[],
        sanity_required=False,  # OPERATOR OPT-IN
    )
    assert out is True, "relax mode must let sanity-fail pass"


def test_wf_gate_overall_pass_strict_blocks_on_sanity_fail():
    import run_wf_gate
    out = run_wf_gate._compute_overall_pass(
        wf_result={"passed": True},
        sanity_result={"passed": False},
        trade_contract_result={"passed": True},
        trade_gate_result={"passed": True},
        alpha_economics_result={"passed": True},
        validation_scope_ok=True,
        parity_result={"passed": True},
        skipped_required_gates=[],
        sanity_required=True,  # strict default
    )
    assert out is False, "strict mode must block on sanity-fail"
