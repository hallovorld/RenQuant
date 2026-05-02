"""Tests for scripts/holdout_backtest.py (B2 single-cut hold-out).

Roadmap §B2: cheap honest OOS sanity check before investing in B1
walk-forward. The script's correctness contract:

  * Train window and sim window MUST NOT overlap. The whole point of
    B2 is non-overlapping; an off-by-one would silently put training
    inside OOS — exactly the bug B2 exists to catch.
  * The config["sample_end"] hand-off must reach the existing
    pp_training stack without modification (we're plumbing, not
    rewriting).
  * Production artifacts MUST NOT be touched (script always runs inside
    snapshot_artifacts_ctx).

These tests stay light — full B2 takes ~30 min of real compute and a
unit suite shouldn't run that. Instead we verify (a) the date-validation
guard, (b) the script imports cleanly, (c) the sample_end key is set on
the config before any training runs, (d) the snapshot context is used.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT    = REPO_ROOT / "scripts" / "holdout_backtest.py"


def _load_script_module():
    """Import the script as a module without running main()."""
    spec = importlib.util.spec_from_file_location(
        "holdout_backtest_script", SCRIPT,
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


# ── Script imports cleanly ────────────────────────────────────────────────────

class TestImports:
    def test_script_loads_without_executing_main(self):
        mod = _load_script_module()
        assert hasattr(mod, "main")
        assert hasattr(mod, "_validate_dates")
        assert hasattr(mod, "_train_on_snapshot")
        assert hasattr(mod, "_run_sim_on_snapshot")


# ── Date validation: hard cut between train and sim ───────────────────────────

class TestDateValidation:
    """The whole purpose of B2 is non-overlapping windows. The guard
    must fail loud at parse time on overlap or inversion.
    """

    def test_valid_strict_ordering_passes(self):
        mod = _load_script_module()
        # Should not raise
        mod._validate_dates("2024-12-31", "2025-01-02", "2026-04-30")

    def test_train_end_equal_to_sim_start_rejected(self):
        """Train through 2025-01-01 + sim from 2025-01-01 = overlap on
        the boundary day. That's exactly the bug B2 catches; reject."""
        mod = _load_script_module()
        with pytest.raises(SystemExit) as exc:
            mod._validate_dates("2025-01-01", "2025-01-01", "2026-04-30")
        assert "STRICTLY BEFORE" in str(exc.value)

    def test_train_end_after_sim_start_rejected(self):
        mod = _load_script_module()
        with pytest.raises(SystemExit):
            mod._validate_dates("2025-06-01", "2025-01-01", "2026-04-30")

    def test_sim_start_after_sim_end_rejected(self):
        mod = _load_script_module()
        with pytest.raises(SystemExit):
            mod._validate_dates("2024-12-31", "2026-04-30", "2025-01-02")

    def test_invalid_iso_date_rejected(self):
        mod = _load_script_module()
        with pytest.raises(SystemExit) as exc:
            mod._validate_dates("not-a-date", "2025-01-02", "2026-04-30")
        assert "invalid date" in str(exc.value).lower()

    def test_sim_start_equal_to_sim_end_passes(self):
        """Single-day sim is allowed (degenerate but not wrong)."""
        mod = _load_script_module()
        mod._validate_dates("2024-12-31", "2025-01-02", "2025-01-02")


# ── Provenance: report metrics carry _holdout suffix ──────────────────────────

class TestReportingSeparation:
    """CLAUDE.md §B3 reporting-separation contract: hold-out metrics MUST
    be labelled distinctly from in-sample / walk-forward / live numbers.
    The script enforces this via the `_holdout` suffix in the JSON
    report. Smoke test the convention is in the source.
    """

    def test_metric_keys_carry_holdout_suffix(self):
        src = SCRIPT.read_text()
        for key in ("apy_holdout", "sharpe_holdout", "max_dd_holdout",
                    "win_rate_holdout", "total_return_holdout"):
            assert f'"{key}"' in src, (
                f"report must label {key} with _holdout suffix per "
                f"CLAUDE.md §B3 reporting-separation contract"
            )

    def test_kind_field_marks_run_as_b2_holdout(self):
        src = SCRIPT.read_text()
        assert '"kind":          "b2_holdout"' in src or \
               '"kind": "b2_holdout"' in src, (
            "report must self-identify as kind=b2_holdout for downstream "
            "filtering — never copy-pasted out of context"
        )


# ── Production safety: snapshot is used ───────────────────────────────────────

class TestProductionSafety:
    """The active strategy artifacts MUST NOT be touched by B2. The
    script must use snapshot_artifacts_ctx (which copies + cleans up).
    """

    def test_uses_snapshot_artifacts_ctx(self):
        src = SCRIPT.read_text()
        assert "snapshot_artifacts_ctx" in src, (
            "B2 must isolate from production via snapshot_artifacts_ctx"
        )
        assert "with snapshot_artifacts_ctx(strategy_dir) as snap_dir:" in src, (
            "B2 must use the context-manager form (auto-cleanup on exit)"
        )

    def test_passes_snap_dir_not_strategy_dir_to_train(self):
        src = SCRIPT.read_text()
        # Training and sim must both go through snap_dir, not the live
        # strategy_dir. A regression here would mean the script writes
        # new artifacts on top of production.
        assert "_train_on_snapshot(snap_dir," in src
        assert "_run_sim_on_snapshot(" in src

    def test_force_retrain_set_on_full_training_context(self):
        """B2 is a one-shot — no cadence guard. Without force_retrain,
        the cadence gate could short-circuit the train phase entirely
        (e.g. if today isn't a Tue/Thu/Sun) and the user gets results
        from STALE artifacts that pre-date the train_end cut."""
        src = SCRIPT.read_text()
        assert "force_retrain=True" in src


# ── sample_end plumbing: the ONE config knob the whole story hangs on ─────────

class TestSampleEndPlumbing:
    """B2's correctness depends entirely on `config["sample_end"]`
    reaching the data fetch + panel training tasks BEFORE training
    starts. If this hand-off breaks, the model trains on the OOS
    window and the whole hold-out is in-sample noise.
    """

    def test_sample_end_set_before_snapshot_block(self):
        """The config[sample_end] line MUST appear before the
        `with snapshot_artifacts_ctx(...)` block — otherwise training
        starts with the prior config and ignores the cutoff."""
        src = SCRIPT.read_text()
        sample_idx = src.index('config["sample_end"]')
        snap_idx   = src.index("with snapshot_artifacts_ctx")
        assert sample_idx < snap_idx, (
            "config[sample_end] must be set BEFORE entering the "
            "snapshot context — otherwise training reads a config "
            "without the cutoff"
        )

    def test_sample_end_value_is_train_end(self):
        src = SCRIPT.read_text()
        assert 'config["sample_end"] = args.train_end' in src

    def test_existing_pp_training_consumes_sample_end_key(self):
        """Cross-check the contract: the EXISTING training stack already
        reads `cfg["sample_end"]`. If a future refactor renames the key,
        this test should fail loud so B2 doesn't silently break.
        """
        pp_training_src = (REPO_ROOT / "backtesting" / "renquant_104"
                           / "kernel" / "pipeline" / "pp_training.py").read_text()
        panel_training_src = (REPO_ROOT / "backtesting" / "renquant_104"
                              / "training_panel" / "pp_panel_training.py").read_text()
        assert "sample_end" in pp_training_src, (
            "pp_training.py no longer references sample_end — B2 plumbing broken"
        )
        assert "sample_end" in panel_training_src, (
            "pp_panel_training.py no longer references sample_end — B2 plumbing broken"
        )
