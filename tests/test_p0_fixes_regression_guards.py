"""Regression guards for 2026-05-20 P0 fixes.

Each test pins ONE invariant from doc/audits/2026-05-20-deep-code-audit.md.
The test would have FAILED before the fix landed → would have prevented
the bug from shipping. Per §5.13.3 ("every fix = regression-guard test
class").

Added 2026-05-20 evening after user observed: my P0-11 shadow config
fix broke daily shadow cron (caught only by next-day cron run, not by
any test). User: "能不能靠谱点！另外这些没有 test guard 吗？！"
"""
from __future__ import annotations
import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


class TestP0_3_4_PlistWeekday:
    """Plist Weekday convention: macOS launchd: 0 AND 7 both = Sunday,
    Mon=1..Sat=6. Pin the weekday integers per cron intent."""

    def test_weekly_wf_promote_runs_saturday_not_sunday(self):
        p = REPO / "scripts/launchd/com.renquant.weekly-wf-promote.plist"
        text = p.read_text()
        # Saturday = 6 (NOT 7 which is Sunday alias)
        match = re.search(r"<key>Weekday</key>\s*<integer>(\d+)</integer>", text)
        assert match is not None, "no Weekday key in weekly-wf-promote.plist"
        weekday = int(match.group(1))
        assert weekday == 6, \
            f"weekly_wf_promote.plist Weekday={weekday}; must be 6 (Saturday). " \
            f"7 = Sunday on macOS launchd → 24h delay vs intended Saturday."

    def test_conditional_retrain_runs_monday_to_friday(self):
        p = REPO / "scripts/launchd/com.renquant.conditional-retrain104.plist"
        text = p.read_text()
        # Mon=1..Fri=5 — extract all Weekday integers
        weekdays = sorted(int(m) for m in re.findall(
            r"<key>Weekday</key>\s*<integer>(\d+)</integer>", text))
        assert weekdays == [1, 2, 3, 4, 5], \
            f"conditional-retrain104 fires on Weekdays {weekdays}; must be " \
            f"[1, 2, 3, 4, 5] (Mon-Fri). 6 = Saturday (no fresh data), " \
            f"0 = Sunday."


class TestP0_5_DashboardArtifactPath:
    """Dashboard must resolve panel-LTR path via golden config, not hardcode."""

    def test_dashboard_uses_canonical_resolver(self):
        src = (REPO / "scripts/build_dashboard.py").read_text()
        assert "_resolve_prod_panel_path" in src, \
            "dashboard must use _resolve_prod_panel_path (canonical "\
            "config-driven resolution per §5.13.14), not hardcoded filename"

    def test_dashboard_does_not_hardcode_legacy_data_path(self):
        """build_dashboard.py must NOT hardcode the legacy
        `data/panel-ltr-prod-alpha158-fund-fwd60d.json` path as the
        primary read target (only as a fallback)."""
        src = (REPO / "scripts/build_dashboard.py").read_text()
        # Find non-fallback occurrences (i.e., not inside try/except FallbackError block)
        # Simpler heuristic: count uses; must NOT be the FIRST `panel_path = REPO_ROOT / ...`
        # statement in section_model_health.
        sec = src.split("def section_model_health")[1].split("def ")[0]
        # The first line that assigns panel_path must call the resolver
        m = re.search(r"panel_path\s*=\s*([^\n]+)", sec)
        assert m is not None
        assert "_resolve_prod_panel_path" in m.group(1), \
            "first panel_path assignment in section_model_health must use resolver"


class TestP0_6_CalibratorMethodDefault:
    """3-site coherence: fit script default + golden + live config all 'platt'."""

    def test_fit_script_default_platt(self):
        src = (REPO / "scripts/fit_panel_calibrator.py").read_text()
        # Find the default in panel_cfg.get
        match = re.search(r'panel_cfg\.get\(\s*"calibration_method"\s*,\s*"(\w+)"',
                          src)
        assert match is not None, "no calibration_method default found"
        assert match.group(1) == "platt", \
            f"script default = {match.group(1)}; must be 'platt' " \
            f"(switched from isotonic 2026-05-18)"

    def test_golden_config_platt(self):
        cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.golden.json").read_text())
        method = cfg.get("panel_ltr", {}).get("calibration_method")
        assert method == "platt", \
            f"golden calibration_method = {method}; must be 'platt'"

    def test_live_config_platt(self):
        cfg = json.loads((REPO / "backtesting/renquant_104/strategy_config.json").read_text())
        method = cfg.get("panel_ltr", {}).get("calibration_method")
        assert method == "platt", \
            f"live calibration_method = {method}; must be 'platt'"


class TestP0_9_BugDSettledCash:
    """Live broker.get_cash() must use non_marginable_buying_power (cash + T+2),
    not account.cash (settled only). Pre-fix: live under-trades vs sim post-sell."""

    def test_get_cash_uses_non_marginable_buying_power(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        # The get_cash method must reference non_marginable_buying_power
        m = re.search(r"def get_cash[^}]*?(?=def |\Z)", src, re.DOTALL)
        assert m is not None
        body = m.group(0)
        assert "non_marginable_buying_power" in body, \
            "get_cash must use non_marginable_buying_power (BUG D fix)"

    def test_get_cash_falls_back_when_field_missing(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        m = re.search(r"def get_cash[^}]*?(?=def |\Z)", src, re.DOTALL)
        body = m.group(0)
        # Fallback path for older alpaca-py without the field
        assert "account.cash" in body, \
            "get_cash must fall back to account.cash if non_marginable_buying_power missing"


class TestP0_10_LiveAccountAssertion:
    """LIVE broker connect must assert account_number matches expected
    (RENQUANT_EXPECTED_LIVE_ACCOUNT env). Per 2026-05-17 e2e mandate."""

    def test_alpaca_connect_logs_account_number(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        assert "account.account_number" in src, \
            "connect() must log account.account_number for auditability"

    def test_alpaca_connect_asserts_expected_live_account_when_env_set(self):
        src = (REPO / "live/alpaca_broker.py").read_text()
        assert "RENQUANT_EXPECTED_LIVE_ACCOUNT" in src, \
            "must read RENQUANT_EXPECTED_LIVE_ACCOUNT env to verify LIVE account id"
        assert "ALPACA LIVE-ACCOUNT MISMATCH" in src, \
            "must raise on account number mismatch (don't silently connect to wrong account)"


class TestP0_11_ShadowConfigPathsExistOnDisk:
    """If shadow config artifact_path doesn't exist on disk, the shadow
    daily cron preflight HARD-fails. This regression bit on 2026-05-20
    when I renamed paths without creating the files."""

    def test_shadow_config_artifact_paths_resolve_to_existing_files(self):
        """Every artifact_path key in shadow config must point to a
        readable file on disk. (Side configs whose paths don't exist
        will silently fail daily shadow cron preflight.)"""
        shadow_cfg_path = REPO / "backtesting/renquant_104/strategy_config.shadow.json"
        if not shadow_cfg_path.exists():
            pytest.skip("shadow config not present")
        cfg = json.loads(shadow_cfg_path.read_text())
        strategy_root = shadow_cfg_path.parent

        # Walk the config tree for artifact_path keys
        missing: list[tuple[str, str]] = []
        def walk(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new = f"{path}.{k}" if path else k
                    if k == "artifact_path" and isinstance(v, str):
                        # Resolve path relative to strategy dir
                        full = (strategy_root / v).resolve()
                        if not full.exists():
                            missing.append((new, v))
                    walk(v, new)
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    walk(v, f"{path}[{i}]")
        walk(cfg)

        assert not missing, (
            f"Shadow config has {len(missing)} artifact_path(s) pointing to "
            f"non-existent files (would break shadow cron preflight):\n  " +
            "\n  ".join(f"{k} = {v}" for k, v in missing)
        )


class TestP0_13_DDVAttrName:
    """DDV reads sue_signal / pead_signal (NOT sue_score / pead_score) —
    must match what ApplyScoresTask actually writes."""

    def test_ddv_reads_correct_attrs(self):
        src = (REPO / "backtesting/renquant_104/kernel/pipeline/task_buy_quality_gates.py").read_text()
        # Should use sue_signal / pead_signal (post-fix names)
        assert '"sue_signal"' in src or "'sue_signal'" in src, \
            "DDV must read 'sue_signal' attr (matches ApplyScoresTask write)"
        assert '"pead_signal"' in src or "'pead_signal'" in src, \
            "DDV must read 'pead_signal' attr"
        # Should NOT use the old wrong names
        assert '"sue_score"' not in src and "'sue_score'" not in src, \
            "DDV must NOT use 'sue_score' (old wrong name); use 'sue_signal'"
        assert '"pead_score"' not in src and "'pead_score'" not in src, \
            "DDV must NOT use 'pead_score' (old wrong name); use 'pead_signal'"


class TestP0_16_AtomicCp:
    """monthly_calibrator_refresh.sh rollback cp must be atomic (cp .tmp + mv)."""

    def test_monthly_calibrator_uses_atomic_cp(self):
        sh = (REPO / "scripts/monthly_calibrator_refresh.sh").read_text()
        # Look for `cp X Y` without `.tmp && mv` for PROD_CAL writes.
        # Any line that does `cp "$ROLLBACK_CAL" "$PROD_CAL"` direct (no .tmp)
        # is the bug. Atomic pattern: `cp ... "$PROD_CAL.tmp" && mv "$PROD_CAL.tmp" "$PROD_CAL"`
        # OR `cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"`
        # Find all cp ... "$PROD_CAL" lines:
        bad = []
        for i, line in enumerate(sh.split("\n"), 1):
            if 'cp ' in line and '"$PROD_CAL"' in line and '.tmp' not in line:
                # exclude `cp X.tmp $PROD_CAL && mv` pattern handled separately
                # exclude comments
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                # Allow if next-line continuation or pattern handled
                bad.append((i, stripped))
        # Also reject cp Y Z (writing to ROLLBACK) without atomic
        for i, line in enumerate(sh.split("\n"), 1):
            if 'cp "$PROD_CAL" "$ROLLBACK_CAL"' in line and '.tmp' not in line:
                stripped = line.strip()
                if not stripped.startswith("#"):
                    bad.append((i, stripped))
        assert not bad, (
            f"monthly_calibrator_refresh.sh has {len(bad)} non-atomic cp "
            f"to/from PROD_CAL (P0-16 regression):\n  " +
            "\n  ".join(f"line {i}: {ln}" for i, ln in bad)
        )


class TestP0_17_BackupSizeGuard:
    """backup_to_github.sh must check file sizes before push (GitHub 100MB limit)."""

    def test_backup_has_100mb_guard(self):
        sh = (REPO / "scripts/backup_to_github.sh").read_text()
        # Look for the find -size +XXM check
        assert "find" in sh and "-size" in sh, \
            "backup script must `find -size` check before push (GitHub 100MB)"
        assert "90M" in sh or "100M" in sh, \
            "backup script must reference 90M or 100M size threshold"

    def test_backup_checks_push_exit_code(self):
        sh = (REPO / "scripts/backup_to_github.sh").read_text()
        # The push must check rc, not silently swallow via tee
        assert "PUSH_RC" in sh or "if ! git push" in sh, \
            "backup script must check git push exit code (was silently swallowed)"
