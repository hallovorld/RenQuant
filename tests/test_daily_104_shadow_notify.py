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


# ── Step 5c: shadow_blend_mom_fast lane (GOAL-9 F2, 2026-08-04) ──────────────
# Codex on RQ#575: the rail clone must carry its OWN static guards mirroring
# the Step-5b set — the 5b tests do not touch the 5c strings.

def test_shadow_blend_mom_fast_step_gated_on_pinned_profile():
    """Step 5c must skip with an INFO line until the F2 fast-blend profile
    exists in the PINNED strategy configs dir (landed s104#89; this guard
    still matters for stale pins/rollbacks)."""
    script = DAILY_104.read_text()

    gate = ('if BLEND_MOM_FAST_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_blend_momentum_fast.json)"; then')
    assert gate in script
    assert "Step 5c shadow-blend-mom-fast skipped" in script
    assert "INFO: strategy_config.shadow_blend_momentum_fast.json not present" in script
    assert script.find("Step 5c shadow-blend-mom-fast skipped") > script.find(gate)


def test_shadow_blend_mom_fast_threads_readonly_tag_and_own_log():
    """The F2 lane selects its own state lane + log file, disjoint from prod,
    legacy shadow, clf-blend AND the slow-mom lane (tag registered at birth,
    pipeline#265)."""
    script = DAILY_104.read_text()

    assert "RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom_fast" in script
    assert ('SHADOW_BLEND_MOM_FAST_LOG='
            '"$LOG_DIR/${DATE}_shadow_blend_mom_fast.log"') in script
    assert '"--strategy-config-path", "$BLEND_MOM_FAST_STRATEGY_CONFIG",' in script
    assert (
        "RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "
        'RENQUANT_READONLY_TAG=alpaca_shadow_blend_mom_fast "$PYTHON" - <<PY '
        '> "$SHADOW_BLEND_MOM_FAST_LOG" 2>&1'
    ) in script


def test_shadow_blend_mom_fast_own_timeout_env():
    """The F2 timeout is its own env var (falls back to the shared shadow
    timeout), never the slow lane's."""
    script = DAILY_104.read_text()
    assert ('SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC='
            '"${RENQUANT_SHADOW_BLEND_MOM_FAST_TIMEOUT_SEC:-'
            '${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"') in script


def test_shadow_blend_mom_fast_failure_alerts_by_default():
    """F2 FAIL/TIMEOUT page with DISTINCT titles — never the slow lane's."""
    script = DAILY_104.read_text()

    assert "RenQuant 104 SHADOW-BLEND-MOM-FAST-FAIL" in script
    assert "RenQuant 104 SHADOW-BLEND-MOM-FAST-TIMEOUT" in script
    idx = script.find("--- Step 5c:")
    assert idx > 0
    section = script[idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in section
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-0}" not in section


def test_shadow_blend_mom_fast_buy_side_preflight_blocks_do_not_page_phone():
    """The dormant path: until the 2026-08-08 fast genesis the blend loader
    fail-closes and the run exits through the buy-side-preflight branch —
    that must stay suppressed (the designed daily record), same pattern set
    as 5b, no exception-class gates swallowed."""
    script = DAILY_104.read_text()

    assert "SHADOW_BLEND_MOM_FAST_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-blend-mom-fast preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_BLEND_MOM_FAST_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID", "P-CORR-METADATA", "P-META-LABEL"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern
    assert "P-PREFLIGHT-IMPORT" not in pattern


def test_shadow_blend_mom_fast_runs_after_slow_mom_and_is_nonfatal():
    """Step 5c comes after Step 5b; the wrapper never exits the daily on
    failure (every branch echoes non-fatal or suppresses)."""
    script = DAILY_104.read_text()

    step5c = script.find("--- Step 5c:")
    step5b = script.find("--- Step 5b:")
    assert step5c > 0 and step5b > 0
    assert step5c > step5b
    section = script[step5c:]
    assert "non-fatal" in section


# ── Step 5d: shadow_blend_rb_mom lane (GOAL-9 F1, 2026-08-04) ─────────────────────

def test_shadow_blend_rb_mom_step_gated_on_pinned_profile():
    script = DAILY_104.read_text()
    gate = ('if BLEND_RB_MOM_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_blend_rb_mom.json)"; then')
    assert gate in script
    assert "Step 5d shadow_blend_rb_mom skipped" in script
    assert "INFO: strategy_config.shadow_blend_rb_mom.json not present" in script
    assert script.find("Step 5d shadow_blend_rb_mom skipped") > script.find(gate)


def test_shadow_blend_rb_mom_threads_readonly_tag_and_own_log():
    script = DAILY_104.read_text()
    assert "RENQUANT_READONLY_TAG=alpaca_shadow_blend_rb_mom" in script
    assert 'SHADOW_BLEND_RB_MOM_LOG="$LOG_DIR/${DATE}_shadow_blend_rb_mom.log"' in script
    assert '"--strategy-config-path", "$BLEND_RB_MOM_STRATEGY_CONFIG",' in script


def test_shadow_blend_rb_mom_own_timeout_env():
    script = DAILY_104.read_text()
    assert ('SHADOW_BLEND_RB_MOM_TIMEOUT_SEC='
            '"${RENQUANT_SHADOW_BLEND_RB_MOM_TIMEOUT_SEC:-'
            '${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"') in script


def test_shadow_blend_rb_mom_failure_alerts_by_default():
    script = DAILY_104.read_text()
    assert "RenQuant 104 SHADOW-BLEND-RB-MOM-FAIL" in script
    assert "RenQuant 104 SHADOW-BLEND-RB-MOM-TIMEOUT" in script
    idx = script.find("--- Step 5d:")
    assert idx > 0
    section = script[idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in section


def test_shadow_blend_rb_mom_preflight_blocks_do_not_page_phone():
    script = DAILY_104.read_text()
    assert "SHADOW_BLEND_RB_MOM_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-blend-rb-mom preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_BLEND_RB_MOM_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern


def test_shadow_blend_rb_mom_ordering_and_nonfatal():
    script = DAILY_104.read_text()
    here = script.find("--- Step 5d:")
    prev = script.find("--- Step 5c:")
    assert here > 0 and prev > 0 and here > prev
    assert "non-fatal" in script[here:]


# ── Step 5e: shadow_blend_rb_fast lane (GOAL-9 F3, 2026-08-04) ─────────────────────

def test_shadow_blend_rb_fast_step_gated_on_pinned_profile():
    script = DAILY_104.read_text()
    gate = ('if BLEND_RB_FAST_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_blend_rb_fast.json)"; then')
    assert gate in script
    assert "Step 5e shadow_blend_rb_fast skipped" in script
    assert "INFO: strategy_config.shadow_blend_rb_fast.json not present" in script
    assert script.find("Step 5e shadow_blend_rb_fast skipped") > script.find(gate)


def test_shadow_blend_rb_fast_threads_readonly_tag_and_own_log():
    script = DAILY_104.read_text()
    assert "RENQUANT_READONLY_TAG=alpaca_shadow_blend_rb_fast" in script
    assert 'SHADOW_BLEND_RB_FAST_LOG="$LOG_DIR/${DATE}_shadow_blend_rb_fast.log"' in script
    assert '"--strategy-config-path", "$BLEND_RB_FAST_STRATEGY_CONFIG",' in script


def test_shadow_blend_rb_fast_own_timeout_env():
    script = DAILY_104.read_text()
    assert ('SHADOW_BLEND_RB_FAST_TIMEOUT_SEC='
            '"${RENQUANT_SHADOW_BLEND_RB_FAST_TIMEOUT_SEC:-'
            '${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"') in script


def test_shadow_blend_rb_fast_failure_alerts_by_default():
    script = DAILY_104.read_text()
    assert "RenQuant 104 SHADOW-BLEND-RB-FAST-FAIL" in script
    assert "RenQuant 104 SHADOW-BLEND-RB-FAST-TIMEOUT" in script
    idx = script.find("--- Step 5e:")
    assert idx > 0
    section = script[idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in section


def test_shadow_blend_rb_fast_preflight_blocks_do_not_page_phone():
    script = DAILY_104.read_text()
    assert "SHADOW_BLEND_RB_FAST_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-blend-rb-fast preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_BLEND_RB_FAST_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern


def test_shadow_blend_rb_fast_ordering_and_nonfatal():
    script = DAILY_104.read_text()
    here = script.find("--- Step 5e:")
    prev = script.find("--- Step 5d:")
    assert here > 0 and prev > 0 and here > prev
    assert "non-fatal" in script[here:]


# ── Step 5f: shadow_vol_window lane (orch#1004 impl PR 2, 2026-08-18) ─────────
# The vol-window license lane (CONFIRMED vol-switch conditional, orch#1003;
# pipeline#294 mechanism; s104#99 profile). Same guard set as 5d/5e: each rail
# clone carries its OWN static guards (the RQ#575 lesson — the 5b tests do not
# touch the 5f strings).

def test_shadow_vol_window_step_gated_on_pinned_profile():
    """Step 5f must skip with an INFO line until the vol-window profile
    exists in the PINNED strategy configs dir — the rail lands before the
    pin advance (a separate operator step), and a missing lane config must
    NEVER hard-fail the daily."""
    script = DAILY_104.read_text()
    gate = ('if VOL_WINDOW_STRATEGY_CONFIG="$(renquant_strategy_config '
            '"$SUBREPO_ROOT" strategy_config.shadow_vol_window.json)"; then')
    assert gate in script
    assert "Step 5f shadow_vol_window skipped" in script
    assert "INFO: strategy_config.shadow_vol_window.json not present" in script
    assert script.find("Step 5f shadow_vol_window skipped") > script.find(gate)


def test_shadow_vol_window_threads_readonly_tag_and_own_log():
    """The vol-window lane selects its own state lane + log file, disjoint
    from prod and every other shadow lane (tag alpaca_shadow_vol_window,
    named by the s104#99 profile's never-submit posture)."""
    script = DAILY_104.read_text()
    assert "RENQUANT_READONLY_TAG=alpaca_shadow_vol_window" in script
    assert 'SHADOW_VOL_WINDOW_LOG="$LOG_DIR/${DATE}_shadow_vol_window.log"' in script
    assert '"--strategy-config-path", "$VOL_WINDOW_STRATEGY_CONFIG",' in script
    assert (
        "RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "
        'RENQUANT_READONLY_TAG=alpaca_shadow_vol_window "$PYTHON" - <<PY '
        '> "$SHADOW_VOL_WINDOW_LOG" 2>&1'
    ) in script


def test_shadow_vol_window_own_timeout_env():
    script = DAILY_104.read_text()
    assert ('SHADOW_VOL_WINDOW_TIMEOUT_SEC='
            '"${RENQUANT_SHADOW_VOL_WINDOW_TIMEOUT_SEC:-'
            '${RENQUANT_SHADOW_TIMEOUT_SEC:-1800}}"') in script


def test_shadow_vol_window_failure_alerts_by_default():
    script = DAILY_104.read_text()
    assert "RenQuant 104 SHADOW-VOL-WINDOW-FAIL" in script
    assert "RenQuant 104 SHADOW-VOL-WINDOW-TIMEOUT" in script
    idx = script.find("--- Step 5f:")
    assert idx > 0
    section = script[idx:]
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-1}" in section
    assert "${RENQUANT_SHADOW_ALERT_NTFY:-0}" not in section


def test_shadow_vol_window_preflight_blocks_do_not_page_phone():
    script = DAILY_104.read_text()
    assert "SHADOW_VOL_WINDOW_BUY_SIDE_PREFLIGHT_PATTERN" in script
    assert "Shadow-vol-window preflight-block ntfy suppressed" in script
    idx = script.find("SHADOW_VOL_WINDOW_BUY_SIDE_PREFLIGHT_PATTERN=")
    pattern = script[idx: script.find("\n", idx)]
    for gate in ("P-WF-GATE", "P-RUN-ID", "P-CORR-METADATA", "P-META-LABEL"):
        assert gate in pattern
    assert "P-PREFLIGHT-EXCEPTION" not in pattern
    assert "P-PREFLIGHT-IMPORT" not in pattern


def test_shadow_vol_window_ordering_and_nonfatal():
    """Step 5f comes after Step 5e and before Step 6 (the fleet sentinel's
    watch set stays Steps 5–5e; the vol-window lane's session accounting is
    the orchestrator readout's parity alarm); the wrapper never exits the
    daily on a 5f failure."""
    script = DAILY_104.read_text()
    here = script.find("--- Step 5f:")
    prev = script.find("--- Step 5e:")
    step6 = script.find("--- Step 6: Fleet lane sentinel")
    assert here > 0 and prev > 0 and step6 > 0
    assert prev < here < step6
    section = script[here:step6]
    assert "non-fatal" in section
    assert "exit 1" not in section
    assert "readonly-alpaca" in section


def test_every_blend_lane_success_echo_names_its_own_profile():
    """Codex on RQ#576: the 'profile found' success echo must carry the
    LANE'S OWN identity — a copied echo naming another lane's profile
    corrupts the per-lane audit trail. Pins all five blend lanes, including
    the latent Step-5c instance this review surfaced."""
    script = DAILY_104.read_text()
    for echo in (
        'echo "shadow_blend profile found at $BLEND_STRATEGY_CONFIG"',
        'echo "shadow_blend_momentum profile found at $BLEND_MOM_STRATEGY_CONFIG"',
        'echo "shadow_blend_momentum_fast profile found at $BLEND_MOM_FAST_STRATEGY_CONFIG"',
        'echo "shadow_blend_rb_mom profile found at $BLEND_RB_MOM_STRATEGY_CONFIG"',
        'echo "shadow_blend_rb_fast profile found at $BLEND_RB_FAST_STRATEGY_CONFIG"',
        # Step 5f (vol-window) is not a blend lane, but the identity rule is
        # the same: the success echo must carry the lane's OWN profile.
        'echo "shadow_vol_window profile found at $VOL_WINDOW_STRATEGY_CONFIG"',
    ):
        assert echo in script, echo
    # and the wrong-identity form appears ONLY for the lane it belongs to
    assert script.count('echo "shadow_blend_momentum profile found') == 1


# ── Step 6: the fleet lane sentinel call (GOAL-1, orch#801) ──────────────────

def test_step6_invokes_the_sentinel_with_the_session_date():
    """The umbrella only CALLS the watcher; the logic lives in orchestrator.
    The session date must be explicit — a post-midnight finish has to classify
    its own session, not 'today'."""
    script = DAILY_104.read_text()
    assert "--- Step 6: Fleet lane sentinel" in script
    assert 'bash "$FLEET_SENTINEL" "$DATE"' in script
    assert "ops/renquant104/fleet_lane_sentinel_daily.sh" in script
    assert "RQ_ORCH_RUN_DIR" in script


def test_step6_runs_after_every_fleet_lane_it_inspects():
    """Daily completion is the trigger precisely because the lanes are above."""
    script = DAILY_104.read_text()
    idx6 = script.find("--- Step 6: Fleet lane sentinel")
    assert idx6 > 0
    for heading in ("--- Step 5:", "--- Step 5b:", "--- Step 5c:",
                    "--- Step 5d:", "--- Step 5e:"):
        assert 0 < script.find(heading) < idx6, heading


def _step6() -> str:
    script = DAILY_104.read_text()
    return script[script.find("--- Step 6: Fleet lane sentinel"):]


def test_step6_finding_is_nonfatal_and_NOT_paged_twice():
    """A watcher must not turn its finding into a failed daily run, and a
    FINDING is already paged by the orchestrator wrapper — Step 6 must not
    duplicate it."""
    section = _step6()
    assert "non-fatal" in section
    assert "it has already paged" in section
    finding_branch = section[section.find("FLEET_SENTINEL_RC=$?"):
                             section.find("# ABSENCE IS ACTIONABLE")]
    assert "notify " not in finding_branch, (
        "the finding branch must not send a duplicate page"
    )


def test_step6_ABSENCE_is_actionable_not_an_info_line(tmp_path=None):
    """codex on RQ#582: a stale/failed run-checkout deploy can REMOVE the
    watcher, and nobody else pages here — the missing component IS the pager.
    Absence must use the daily alert surface."""
    section = _step6()
    absence = section[section.find("# ABSENCE IS ACTIONABLE"):]
    assert "notify " in absence, "absence must page"
    assert "FLEET-SENTINEL-MISSING" in absence
    assert "UNWATCHED" in absence
    assert "sync the orchestrator run checkout" in absence.lower()


def test_step6_pages_in_exactly_one_branch():
    """The two cases are distinct and each has exactly one channel: findings →
    wrapper's page; absence → this page. Never both, never neither."""
    section = _step6()
    assert section.count("notify ") == 1, (
        f"expected exactly one notify in Step 6, found {section.count('notify ')}"
    )
