"""Tests for scripts/smoke_test_model.py — daily heartbeat + BUG #6 guard.

The smoke test is the daily pipeline heartbeat (replaces 75-min retrain
post-FIX-C). It must:
  1. Pass on a healthy production artifact
  2. Fail loud when artifact is broken (missing, malformed, schema drift)
  3. Catch BUG #6 class regression — μ̂ collapse where 2 different inputs
     produce identical outputs

Reference: doc/ops/schedule.md (daily cadence), doc/AUDIT_2026-05-09.md FIX-C
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

_SCRIPT = REPO / "scripts" / "smoke_test_model.py"


class TestSmokeRuns:
    """Smoke test runs end-to-end on the actual production artifact."""

    def test_smoke_test_passes_on_production_artifact(self):
        """The committed panel-ltr.alpha158_fund.json must score cleanly."""
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--strategy", "renquant_104"],
            capture_output=True, text=True, timeout=30,
            cwd=str(REPO),
        )
        assert result.returncode == 0, \
            f"Smoke test FAILED on production artifact:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "Smoke test PASS" in result.stderr or "Smoke test PASS" in result.stdout

    def test_smoke_test_fails_on_missing_artifact(self, tmp_path):
        """Smoke test exits 1 if artifact path resolves to nothing."""
        # Build a fake strategy dir with only the config (no artifact)
        fake_strategy = tmp_path / "fake_strategy"
        fake_strategy.mkdir()
        (fake_strategy / "artifacts").mkdir()
        cfg = {
            "ranking": {"panel_scoring": {
                "enabled": True,
                "artifact_path": "artifacts/nonexistent.json",
            }}
        }
        (fake_strategy / "strategy_config.json").write_text(json.dumps(cfg))

        # Stage as a strategy dir under backtesting/
        bt = tmp_path / "backtesting" / "fake_strategy_test"
        bt.parent.mkdir(parents=True)
        bt.symlink_to(fake_strategy)

        # Run smoke test pointing at this fake strategy
        result = subprocess.run(
            [sys.executable, str(_SCRIPT), "--strategy", "fake_strategy_test"],
            capture_output=True, text=True, timeout=10,
            cwd=str(tmp_path),
        )
        # Should fail (no artifact at the configured path)
        assert result.returncode != 0
        assert "FAIL" in (result.stderr + result.stdout)


class TestBug6ClassGuard:
    """Smoke test must catch the μ̂-collapse failure mode.

    The actual collapse-detection logic lives in the script itself (asserts
    abs(score_a - score_b) > 1e-9 for two different random inputs). This test
    verifies the script CONTAINS that assertion — a regression-guard guard.
    """

    def test_script_has_diversity_assertion(self):
        src = _SCRIPT.read_text()
        assert "BUG #6 CLASS" in src or "μ̂-collapse" in src or "μ̂ collapse" in src.lower(), \
            "Smoke test must explicitly check for BUG #6 class (identical " \
            "outputs from different inputs)"
        assert "abs(score_a - score_b)" in src, \
            "Smoke test must compare two synthetic inputs' outputs"

    def test_script_uses_two_random_inputs(self):
        src = _SCRIPT.read_text()
        # The script must use 2+ rows so PanelScorer's diversity guard passes
        # AND we can assert different inputs produce different outputs.
        assert "rng.standard_normal((2," in src or "(2, len(feature_cols))" in src, \
            "Smoke test must use 2 synthetic input rows for diversity assertion"


class TestScheduleDocConsistency:
    """The schedule doc must reference all 3 cron scripts + 3 plists.
    Pin the doc-vs-filesystem invariant — if a plist gets renamed,
    the doc fails loudly instead of going stale."""

    def test_weekly_script_referenced(self):
        doc = (REPO / "doc" / "ops" / "schedule.md").read_text()
        assert "scripts/weekly_wf_promote.sh" in doc
        assert (REPO / "scripts" / "weekly_wf_promote.sh").exists()

    def test_monthly_script_referenced(self):
        doc = (REPO / "doc" / "ops" / "schedule.md").read_text()
        assert "scripts/monthly_calibrator_refresh.sh" in doc
        assert (REPO / "scripts" / "monthly_calibrator_refresh.sh").exists()

    def test_smoke_test_referenced(self):
        doc = (REPO / "doc" / "ops" / "schedule.md").read_text()
        assert "scripts/smoke_test_model.py" in doc
        assert (REPO / "scripts" / "smoke_test_model.py").exists()

    def test_event_scripts_referenced(self):
        doc = (REPO / "doc" / "ops" / "schedule.md").read_text()
        for name in ["event_watchlist_change.sh", "event_sec_schema_change.sh",
                     "manual_promote.sh"]:
            assert name in doc, f"{name} missing from schedule doc"
            assert (REPO / "scripts" / name).exists(), f"{name} missing from scripts/"

    def test_weekly_plist_exists_and_lints(self):
        plist = REPO / "scripts" / "launchd" / "com.renquant.weekly-wf-promote.plist"
        assert plist.exists()
        result = subprocess.run(
            ["plutil", "-lint", str(plist)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"plist lint failed: {result.stdout}"

    def test_monthly_plist_exists_and_lints(self):
        plist = REPO / "scripts" / "launchd" / "com.renquant.monthly-calibrator-refresh.plist"
        assert plist.exists()
        result = subprocess.run(
            ["plutil", "-lint", str(plist)],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"plist lint failed: {result.stdout}"


class TestNoRetrainInDailyShell:
    """AUDIT REGRESSION GUARD: daily_104.sh must NOT call the retrain script."""

    def test_daily_does_not_call_daily_retrain_alpha158_fund(self):
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        # Find any non-comment line referencing the retrain script
        lines = [
            ln for ln in daily.splitlines()
            if "daily_retrain_alpha158_fund" in ln
            and not ln.strip().startswith("#")
        ]
        assert not lines, \
            f"AUDIT REGRESSION (FIX-C): daily_104.sh must not call retrain. " \
            f"Found:\n  " + "\n  ".join(lines)

    def test_daily_does_not_set_rq_allow_no_wf(self):
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        lines = [
            ln for ln in daily.splitlines()
            if "RQ_ALLOW_NO_WF" in ln
            and not ln.strip().startswith("#")
        ]
        assert not lines, \
            f"AUDIT REGRESSION (FIX-C): daily_104.sh must not set " \
            f"RQ_ALLOW_NO_WF. Found:\n  " + "\n  ".join(lines)

    def test_daily_calls_smoke_test(self):
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        assert "smoke_test_model.py" in daily, \
            "daily_104.sh must call smoke_test_model.py post-FIX-C"

    def test_daily_suppresses_inner_preflight_ntfy_for_fallback_probe(self):
        """AUDIT REGRESSION GUARD: full-mode probe is wrapped by daily.

        P-WF-GATE should produce the daily BUY-BLOCKED fallback summary, not
        an additional urgent live.runner ERROR before sell-only completes.
        """
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        assert "RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1" in daily
        assert "Full live trader blocked by buy-side preflight gate" in daily

    def test_shadow_e2e_has_wall_clock_timeout(self):
        """AUDIT REGRESSION GUARD: shadow must not hang the daily script."""
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        assert "SHADOW_TIMEOUT_SEC" in daily
        assert "subprocess.TimeoutExpired" in daily
        assert "SHADOW-TIMEOUT" in daily
        assert "RENQUANT_SHADOW_ALERT_NTFY" in daily
        assert "Shadow timeout ntfy suppressed" in daily

    def test_shadow_e2e_default_timeout_covers_full_patchtst_cold_start(self):
        """HF PatchTST shadow full-e2e needs more than the old 420s cap."""
        import re

        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        match = re.search(r'RENQUANT_SHADOW_TIMEOUT_SEC:-(\d+)', daily)
        assert match, "daily_104.sh must keep an explicit shadow timeout default"
        assert int(match.group(1)) >= 1200

    def test_daily_does_not_double_append_log_after_exec_redirect(self):
        """AUDIT REGRESSION GUARD: exec already redirects stdout to LOG."""
        daily = (REPO / "scripts" / "daily_104.sh").read_text()
        redirected_body = daily.split('exec >> "$LOG" 2>&1', 1)[1]
        assert '| tee -a "$LOG"' not in redirected_body


class TestWeeklyShellInvariants:

    def test_weekly_does_not_set_rq_allow_no_wf(self):
        weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
        lines = [
            ln for ln in weekly.splitlines()
            if "RQ_ALLOW_NO_WF" in ln
            and not ln.strip().startswith("#")
        ]
        assert not lines, \
            f"AUDIT REGRESSION: weekly_wf_promote.sh must NEVER set " \
            f"RQ_ALLOW_NO_WF (the whole point of the weekly path is to " \
            f"NOT bypass the gate). Found:\n  " + "\n  ".join(lines)

    def test_weekly_calls_run_wf_gate(self):
        weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
        assert "run_wf_gate.py" in weekly
        assert "--strict" in weekly, \
            "weekly_wf_promote.sh must run WF gate in strict mode"

    def test_weekly_calls_smoke_first(self):
        weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
        # smoke test must come BEFORE retrain (don't waste 90 min on broken pipeline)
        smoke_idx = weekly.find("smoke_test_model.py")
        retrain_idx = weekly.find("daily_retrain_alpha158_fund.sh")
        assert smoke_idx > 0, "weekly_wf_promote.sh must call smoke test"
        assert retrain_idx > 0, "weekly_wf_promote.sh must call retrain"
        assert smoke_idx < retrain_idx, \
            "smoke test must precede retrain (don't waste compute on broken pipeline)"

    def test_weekly_uses_project_venv_and_detected_threads(self):
        weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
        non_comment = "\n".join(
            line for line in weekly.splitlines()
            if not line.lstrip().startswith("#")
        )
        assert "CONDA_PREFIX" not in non_comment
        assert 'VENV_DIR="$REPO_DIR/.venv"' in weekly
        assert "os.cpu_count()" in weekly
        for var in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            assert f'export {var}="$THREADS"' in weekly

    def test_weekly_trains_to_unique_staging_before_gate(self):
        weekly = (REPO / "scripts" / "weekly_wf_promote.sh").read_text()
        assert "RUN_ID=" in weekly
        assert "STAGING_ART=" in weekly
        assert "STAGING_CAL=" in weekly
        assert "--xgb-artifact-out \"$STAGING_ART\"" in weekly
        assert "--calibrator-out \"$STAGING_CAL\"" in weekly
        assert '--artifact "$STAGING_ART"' in weekly
        assert "bash scripts/daily_retrain_alpha158_fund.sh;" not in weekly
        assert "bash scripts/daily_retrain_alpha158_fund.sh; then" not in weekly
