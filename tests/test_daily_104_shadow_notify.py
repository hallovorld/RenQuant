"""Regression guards for daily_104 shadow alert wiring."""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_104 = REPO_ROOT / "scripts" / "daily_104.sh"


def test_legacy_shadow_step4_is_retired_not_half_present():
    """Step 4 (HF PatchTST shadow e2e) was RETIRED 2026-08-03. These were
    the Step-4-era guards; the script dropped the block and the guards went
    stale unnoticed because this file runs in NO CI enumeration (fixed in
    the Step 5b PR by updating them to guard the retirement itself)."""
    script = DAILY_104.read_text()

    marker = "Step 4: RETIRED 2026-08-03"
    assert marker in script
    # No half-present legacy lane: the retired block's invocation env, log
    # var, alert titles, and preflight pattern var must ALL be gone (bare
    # legacy tag only in the retirement narrative, never on an invocation).
    assert 'RENQUANT_READONLY_TAG=alpaca_shadow "$PYTHON"' not in script
    assert 'RenQuant 104 SHADOW-FAIL' not in script
    assert 'RenQuant 104 SHADOW-TIMEOUT' not in script
    assert "SHADOW_BUY_SIDE_PREFLIGHT_PATTERN=" not in script.replace(
        "SHADOW_BLEND_MOM_BUY_SIDE_PREFLIGHT_PATTERN=", "").replace(
        "SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN=", "")


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


# ── Step 5: shadow_blend lane (2026-07-27 operator directive) ──────────────

def test_shadow_blend_step_gated_on_pinned_profile():
    """Step 5 must skip with an INFO line until the blend profile exists
    in the PINNED strategy configs dir — the rail lands before the profile."""
    script = DAILY_104.read_text()

    gate = ('if BLEND_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_blend.json)"; then')
    assert gate in script
    assert "Step 5 shadow-blend skipped" in script
    assert "INFO: strategy_config.shadow_blend.json not present" in script
    # The skip line must be in the gate's else-branch (after the gate).
    assert script.find("Step 5 shadow-blend skipped") > script.find(gate)


def test_shadow_blend_threads_readonly_tag_and_own_log():
    """The blend lane must select its own state lane + log file."""
    script = DAILY_104.read_text()

    assert "RENQUANT_READONLY_TAG=alpaca_shadow_blend" in script
    assert 'SHADOW_BLEND_LOG="$LOG_DIR/${DATE}_shadow_blend.log"' in script
    # The profile reaches the runner via the gate-resolved var (the quoted
    # literal this test once matched left the script when the heredoc took
    # the resolved path; stale-guard fix in the Step 5b PR).
    assert '"--strategy-config-path", "$BLEND_STRATEGY_CONFIG",' in script
    # Tag env is set on the SAME invocation that suppresses inner preflight
    # ntfy (mirrors Step 4's env-prefix form), routed to the blend log.
    assert (
        "RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "
        'RENQUANT_READONLY_TAG=alpaca_shadow_blend "$PYTHON" - <<PY '
        '> "$SHADOW_BLEND_LOG" 2>&1'
    ) in script


def test_shadow_blend_failure_alerts_by_default():
    """Blend-lane FAIL/TIMEOUT must page (non-fatal wrapper alert), with
    distinct titles so blend vs legacy shadow are distinguishable."""
    script = DAILY_104.read_text()

    assert "RenQuant 104 SHADOW-BLEND-FAIL" in script
    assert "RenQuant 104 SHADOW-BLEND-TIMEOUT" in script
    # Same operator kill-switch as the legacy lane; must default to ON.
    blend_idx = script.find("--- Step 5:")
    assert blend_idx > 0
    blend_section = script[blend_idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in blend_section
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-0}" not in blend_section


def test_shadow_blend_buy_side_preflight_blocks_do_not_page_phone():
    """Expected buy-side gate blocks: logged, not paged (mirror of Step 4)."""
    script = DAILY_104.read_text()

    assert "SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-blend preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_BLEND_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID", "P-CORR-METADATA", "P-META-LABEL"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern
    assert "P-PREFLIGHT-IMPORT" not in pattern


def test_shadow_blend_runs_after_step4_retirement_and_is_nonfatal():
    """Step 5 comes after the Step 4 retirement note; it never exits the
    daily on failure. (The old form compared against the literal
    "--- Step 4:" echo, which left the script with the retirement — find()
    returned -1 and the ordering assert passed VACUOUSLY; both anchors are
    now existence-checked first.)"""
    script = DAILY_104.read_text()

    step5 = script.find("--- Step 5:")
    step4_retired = script.find("Step 4: RETIRED 2026-08-03")
    assert step5 > 0 and step4_retired > 0
    assert step5 > step4_retired
    blend_section = script[step5:]
    assert "exit 1" not in blend_section
    assert "readonly-alpaca" in blend_section


# ── Step 5b: shadow_blend_mom lane (GOAL-8 S1, 2026-08-04) ──────────────────

def test_shadow_blend_mom_step_gated_on_pinned_profile():
    """Step 5b must skip with an INFO line until the S1 momentum-blend
    profile exists in the PINNED strategy configs dir — the rail lands
    before the profile, same shape Step 5 itself shipped with."""
    script = DAILY_104.read_text()

    gate = ('if BLEND_MOM_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_blend_momentum.json)"; then')
    assert gate in script
    assert "Step 5b shadow-blend-mom skipped" in script
    assert "INFO: strategy_config.shadow_blend_momentum.json not present" in script
    assert script.find("Step 5b shadow-blend-mom skipped") > script.find(gate)


def test_shadow_blend_mom_threads_readonly_tag_and_own_log():
    """The S1 lane must select its own state lane + log file, disjoint from
    prod, legacy shadow, AND the clf-blend lane."""
    script = DAILY_104.read_text()

    assert "RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom" in script
    assert 'SHADOW_BLEND_MOM_LOG="$LOG_DIR/${DATE}_shadow_blend_mom.log"' in script
    assert '"--strategy-config-path", "$BLEND_MOM_STRATEGY_CONFIG",' in script
    assert (
        "RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "
        'RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom "$PYTHON" - <<PY '
        '> "$SHADOW_BLEND_MOM_LOG" 2>&1'
    ) in script


def test_shadow_blend_mom_failure_alerts_by_default():
    """S1-lane FAIL/TIMEOUT must page with DISTINCT titles."""
    script = DAILY_104.read_text()

    assert "RenQuant 104 SHADOW-BLEND-MOM-FAIL" in script
    assert "RenQuant 104 SHADOW-BLEND-MOM-TIMEOUT" in script
    idx = script.find("--- Step 5b:")
    assert idx > 0
    section = script[idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in section
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-0}" not in section


def test_shadow_blend_mom_buy_side_preflight_blocks_do_not_page_phone():
    script = DAILY_104.read_text()

    assert "SHADOW_BLEND_MOM_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-blend-mom preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_BLEND_MOM_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID", "P-CORR-METADATA", "P-META-LABEL"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern
    assert "P-PREFLIGHT-IMPORT" not in pattern


def test_shadow_blend_mom_runs_after_clf_blend_and_is_nonfatal():
    """Step 5b comes after Step 5 (the clf-blend slot keeps its rail);
    neither exits the daily on failure."""
    script = DAILY_104.read_text()

    step5b = script.find("--- Step 5b:")
    step5 = script.find("--- Step 5:")
    assert step5b > 0 and step5 > 0  # non-vacuous anchors
    assert step5b > step5
    section = script[step5b:]
    assert "exit 1" not in section
    assert "readonly-alpaca" in section
