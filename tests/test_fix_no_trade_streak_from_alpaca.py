"""Safety tests for the Alpaca no-trade-streak repair script."""
from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fix_no_trade_streak_from_alpaca.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fix_no_trade_streak_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_default_is_dry_run() -> None:
    mod = _load_module()
    args = mod.build_parser().parse_args([])

    assert args.apply is False
    assert args.dry_run is False
    assert mod._should_write(args) is False


def test_apply_required_for_state_write() -> None:
    mod = _load_module()
    args = mod.build_parser().parse_args(["--apply"])

    assert mod._should_write(args) is True


def test_dry_run_overrides_apply_helper() -> None:
    mod = _load_module()
    args = mod.build_parser().parse_args(["--apply", "--dry-run"])

    assert mod._should_write(args) is False
