"""Regression guards for daily_104 shadow alert wiring."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_104 = REPO_ROOT / "scripts" / "daily_104.sh"


def test_shadow_failure_alerts_by_default():
    """Shadow e2e failure must not be silent unless explicitly disabled."""
    script = DAILY_104.read_text()

    assert 'RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "$PYTHON"' in script
    assert '${RENQUANT_SHADOW_ALERT_NTFY:-1}' in script
    assert '${RENQUANT_SHADOW_ALERT_NTFY:-0}' not in script
    assert 'RenQuant 104 SHADOW-FAIL' in script
    assert 'RenQuant 104 SHADOW-TIMEOUT' in script


def test_shadow_buy_side_preflight_blocks_do_not_page_phone():
    """Expected buy-side gate blocks are logged, not repeated as shadow errors."""
    script = DAILY_104.read_text()

    assert "SHADOW_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow preflight-block ntfy suppressed" in script
    assert "P-WF-GATE" in script
    assert "P-RUN-ID" in script
    assert "P-CORR-METADATA" in script
    assert "P-META-LABEL" in script
    assert "P-PREFLIGHT-EXCEPTION" in script


def test_full_preflight_exception_falls_back_to_sell_only():
    """Broken buy-side preflight code must not suppress risk exits."""
    script = DAILY_104.read_text()

    assert "P-PREFLIGHT-IMPORT" in script
    assert "P-PREFLIGHT-EXCEPTION" in script
    assert "rerunning sell-only" in script


def test_buy_blocked_wrapper_alert_has_cooldown():
    """Repeated expected buy-side gate blocks should not page every rerun."""
    script = DAILY_104.read_text()

    assert "BUY_BLOCKED_ALERT_STAMP" in script
    assert "RENQUANT_BUY_BLOCKED_ALERT_COOLDOWN_SEC" in script
    assert "BUY-BLOCKED ntfy suppressed by cooldown" in script
    assert "P-RUN-ID" in script
    assert "P-CORR-METADATA" in script
    assert "P-META-LABEL" in script
