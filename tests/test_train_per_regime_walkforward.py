"""Regression test for `_train_regime` sequential-failure crash (Track C).

AUDIT REGRESSION GUARD for codex finding on PR #121:

    MED: _train_regime() crashes on the first failed sequential cutoff.
    `(entries if ok else failures).setdefault(iso, None)` does setdefault
    on a list → AttributeError: 'list' object has no attribute 'setdefault'.

Pins the explicit-branch fix:

    if ok:
        entries[iso] = (str(art), str(cal) if cal else None)
    else:
        failures.append((iso, err))

Without this test, regressing back to the broken setdefault form would
ship silently: the parallel (jobs > 1) branch was OK, so most user
invocations would not surface the bug — only --jobs 1 (the default!).
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "train_per_regime_walkforward.py"


def _load_module():
    """Load the train_per_regime_walkforward script as a module."""
    spec = importlib.util.spec_from_file_location(
        "train_per_regime_walkforward", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["train_per_regime_walkforward"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script_mod():
    return _load_module()


class TestTrainRegimeSequentialFailure:
    """Sequential-mode `_train_regime` handles ok=False without crashing."""

    def _make_args(self) -> argparse.Namespace:
        return argparse.Namespace(jobs=1, label="fwd_20d")

    def test_all_failures_no_attribute_error(self, monkeypatch, script_mod):
        """Every cutoff fails ⇒ failures list populated, entries empty,
        no AttributeError raised.

        Pre-fix: this raised
            AttributeError: 'list' object has no attribute 'setdefault'
        on the FIRST iteration.
        """
        retrain_dates = [
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-22"),
            pd.Timestamp("2024-02-12"),
        ]

        def fake_train_one(cutoff_iso, regime, args, lookahead_days):  # noqa: ARG001
            return False, None, None, f"forced failure for {cutoff_iso}"

        monkeypatch.setattr(script_mod, "_train_one", fake_train_one)

        out = script_mod._train_regime(
            regime="BULL_CALM",
            retrain_dates=retrain_dates,
            args=self._make_args(),
            lookahead_days=20,
        )
        assert out == {}, "no entries should be recorded when every cutoff fails"

    def test_mixed_success_and_failure(self, monkeypatch, script_mod):
        """Sequential mode records successes and skips failures correctly.

        Mirrors the parallel branch's behavior: successes go into entries,
        failures append to the failures list with their err string.
        """
        retrain_dates = [
            pd.Timestamp("2024-01-01"),  # ok
            pd.Timestamp("2024-01-22"),  # fail
            pd.Timestamp("2024-02-12"),  # ok
        ]

        def fake_train_one(cutoff_iso, regime, args, lookahead_days):  # noqa: ARG001
            if cutoff_iso == "2024-01-22":
                return False, None, None, "boom"
            return True, f"/tmp/art-{cutoff_iso}.json", f"/tmp/cal-{cutoff_iso}.json", ""

        monkeypatch.setattr(script_mod, "_train_one", fake_train_one)

        out = script_mod._train_regime(
            regime="BULL_CALM",
            retrain_dates=retrain_dates,
            args=self._make_args(),
            lookahead_days=20,
        )
        assert set(out.keys()) == {"2024-01-01", "2024-02-12"}
        assert out["2024-01-01"] == ("/tmp/art-2024-01-01.json",
                                      "/tmp/cal-2024-01-01.json")
        assert out["2024-02-12"] == ("/tmp/art-2024-02-12.json",
                                      "/tmp/cal-2024-02-12.json")

    def test_first_cutoff_failure_does_not_crash(self, monkeypatch, script_mod):
        """The exact failure mode the codex review found: first cutoff
        ok=False. Pre-fix this raised AttributeError on iteration 1.
        """
        retrain_dates = [pd.Timestamp("2024-01-01")]
        monkeypatch.setattr(
            script_mod, "_train_one",
            lambda iso, regime, args, lookahead: (False, None, None, "first-fail"),
        )
        # No exception = pass
        out = script_mod._train_regime(
            regime="BEAR",
            retrain_dates=retrain_dates,
            args=self._make_args(),
            lookahead_days=20,
        )
        assert out == {}
