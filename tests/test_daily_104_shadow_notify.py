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
    shadow_idx = script.find("SHADOW_BUY_SIDE_PREFLIGHT_PATTERN=")
    shadow_pattern = script[shadow_idx: script.find("\n", shadow_idx)]
    assert "P-PREFLIGHT-EXCEPTION" not in shadow_pattern
    assert "P-PREFLIGHT-IMPORT" not in shadow_pattern


def test_full_preflight_exception_falls_back_to_sell_only():
    """Broken buy-side preflight code must not suppress risk exits."""
    script = DAILY_104.read_text()

    assert "PREFLIGHT_FALLBACK_PATTERN" in script
    assert "PREFLIGHT_SYSTEM_FAILURE_PATTERN" in script
    assert "PREFLIGHT_SYSTEM_FAILURE=1" in script
    assert "P-PREFLIGHT-IMPORT" in script
    assert "P-PREFLIGHT-EXCEPTION" in script
    assert "rerunning sell-only" in script
    assert "Full run hit preflight system failure; sell-only fallback completed" in script


def test_news_sentiment_refresh_is_timeout_bounded():
    """Non-fatal news sentiment refresh must not block the live trader forever."""
    script = DAILY_104.read_text()

    assert "RENQUANT_DAILY_NEWS_TIMEOUT_SEC:-1200" in script
    assert "run_news_sentiment_refresh" in script
    assert "_kill_process_tree" in script
    assert "sentiment refresh timed out" in script


def test_buy_blocked_wrapper_alert_has_cooldown():
    """Repeated expected buy-side gate blocks should not page every rerun."""
    script = DAILY_104.read_text()

    assert "BUY_BLOCKED_ALERT_STAMP" in script
    assert "RENQUANT_BUY_BLOCKED_ALERT_COOLDOWN_SEC" in script
    assert "BUY-BLOCKED ntfy suppressed by cooldown" in script
    assert "P-RUN-ID" in script
    assert "P-CORR-METADATA" in script
    assert "P-META-LABEL" in script
    buy_idx = script.find("BUY_SIDE_PREFLIGHT_PATTERN=")
    buy_pattern = script[buy_idx: script.find("\n", buy_idx)]
    assert "P-PREFLIGHT-EXCEPTION" not in buy_pattern
    assert "P-PREFLIGHT-IMPORT" not in buy_pattern
