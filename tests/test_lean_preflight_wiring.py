"""Regression guards for LEAN/full preflight fail-closed wiring."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parent.parent
MAIN_SRC = (ROOT / "backtesting/renquant_104/main.py").read_text()


def test_lean_initialize_runs_preflight_before_models_and_pipeline():
    init_idx = MAIN_SRC.index("def Initialize")
    strategy_dir_idx = MAIN_SRC.index("self._strategy_dir", init_idx)
    preflight_idx = MAIN_SRC.index("run_preflight(", init_idx)
    models_idx = MAIN_SRC.index("self._load_all_models()", init_idx)
    pipeline_idx = MAIN_SRC.index("self._pipeline = InferencePipeline()", init_idx)

    assert strategy_dir_idx < preflight_idx < models_idx < pipeline_idx
    assert 'run_mode="full"' in MAIN_SRC[preflight_idx:models_idx]
    assert "strict=True" in MAIN_SRC[preflight_idx:models_idx]
    assert "broker=None" in MAIN_SRC[preflight_idx:models_idx]


def test_lean_preflight_success_stamp_brackets_check():
    init_idx = MAIN_SRC.index("def Initialize")
    preflight_false_idx = MAIN_SRC.index("self._preflight_ok = False", init_idx)
    preflight_call_idx = MAIN_SRC.index("run_preflight(", init_idx)
    preflight_true_idx = MAIN_SRC.index("self._preflight_ok = True", init_idx)

    assert preflight_false_idx < preflight_call_idx < preflight_true_idx


def test_lean_main_does_not_catch_preflight_failure():
    init_idx = MAIN_SRC.index("def Initialize")
    preflight_idx = MAIN_SRC.index("run_preflight(", init_idx)
    models_idx = MAIN_SRC.index("self._load_all_models()", init_idx)
    preflight_block = MAIN_SRC[preflight_idx:models_idx]

    assert "PreflightFailed" not in preflight_block
    assert "except" not in preflight_block


def test_lean_adapter_refuses_buy_orders_without_successful_preflight():
    from adapters.lean import LeanAdapter

    adapter = LeanAdapter.__new__(LeanAdapter)
    adapter._algo = SimpleNamespace(_config={}, _preflight_ok=False)

    ctx = SimpleNamespace(orders=[{"ticker": "AAPL", "shares": 1}])

    with pytest.raises(RuntimeError, match="preflight"):
        adapter.commit(ctx)
