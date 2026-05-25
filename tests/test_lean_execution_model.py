"""LEAN execution-model parity guards."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "backtesting" / "renquant_104" / "main.py"


def test_lean_slippage_uses_shared_half_spread_model() -> None:
    src = MAIN.read_text()

    assert "LeanHalfSpreadSlippageModel" in src
    assert "kernel.execution.slippage" in src
    assert "slip_fill_price(" in src
    assert "VolumeShareSlippageModel" not in src


def test_lean_slippage_wiring_fails_loudly() -> None:
    """Execution parity must not disappear behind a swallowed exception."""
    src = MAIN.read_text()
    set_call = "self.SetSlippageModel(LeanHalfSpreadSlippageModel(slip_cfg))"
    assert set_call in src
    call_pos = src.index(set_call)
    nearby = src[max(0, call_pos - 220): call_pos + len(set_call) + 80]
    assert "except" not in nearby
    assert "pass" not in nearby
