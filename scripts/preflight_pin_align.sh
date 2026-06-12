#!/usr/bin/env bash
# preflight_pin_align.sh — align subrepo checkouts to the audited pins, fail-closed.
#
# 2026-06-11: the daily silently traded STALE code because nothing verified the
# runtime subrepo checkouts matched subrepos.lock.json (the umbrella was a day
# behind origin/main, so the run used yesterday's pins). Policy (operator
# go-ahead): auto-align each isolated runtime checkout to its PINNED commit,
# then trade; ABORT if a repo is dirty or its pin is unreachable — never trade
# unaudited code.
#
# `subrepo_assemble.py --sync --runtime-root` checks out a clean-but-drifted
# isolated runtime repo to its lock commit and refuses (exit 1) on a dirty repo;
# `--dry-run` skips writing the assembly bundle. When already pinned it is a
# fast no-op (no fetch/checkout). Set RENQUANT_PIN_SYNC_RUNTIME_ROOT= to inspect
# sibling developer worktrees instead.
#
# The umbrella main is NOT auto-pulled — shipping new pins is a DELIBERATE
# deploy, not something a live-trading cron should do to itself with unreviewed
# code. When PREFLIGHT_CHECK_UMBRELLA=1 (set by the once-daily run, not the
# 12-minute intraday loop) we only WARN if local main lags origin/main.
#
# Source this from a trading entrypoint AFTER it has set REPO_DIR, PYTHON and a
# notify() function. On failure it calls `exit 1`, aborting the caller.
# Escape hatch: RENQUANT_SKIP_PIN_SYNC=1 (emergencies only).

if [ "${RENQUANT_SKIP_PIN_SYNC:-0}" != "1" ]; then
    if [ "${PREFLIGHT_CHECK_UMBRELLA:-0}" = "1" ]; then
        git -C "$REPO_DIR" fetch --quiet origin main 2>/dev/null || true
        _local_main="$(git -C "$REPO_DIR" rev-parse main 2>/dev/null || true)"
        _origin_main="$(git -C "$REPO_DIR" rev-parse origin/main 2>/dev/null || true)"
        if [ -n "$_local_main" ] && [ -n "$_origin_main" ] && [ "$_local_main" != "$_origin_main" ]; then
            echo "WARN: umbrella main behind origin/main (local=${_local_main:0:7} origin=${_origin_main:0:7}); newer pins NOT deployed — run uses current local pins"
            notify "RenQuant 104 WARN" "Umbrella main behind origin/main — newer pins NOT deployed (deliberate deploy needed)"
        fi
        unset _local_main _origin_main
    fi
    _pin_sync_args=(--sync --dry-run)
    _pin_sync_runtime_root="${RENQUANT_PIN_SYNC_RUNTIME_ROOT-$REPO_DIR/.subrepo_runtime/repos}"
    if [ -n "$_pin_sync_runtime_root" ]; then
        _pin_sync_args+=(--runtime-root "$_pin_sync_runtime_root")
        echo "Aligning isolated subrepo runtime root to pinned commits (subrepos.lock.json)…"
    else
        echo "Aligning sibling subrepo checkouts to pinned commits (subrepos.lock.json)…"
    fi
    if ! "$PYTHON" "$REPO_DIR/scripts/subrepo_assemble.py" "${_pin_sync_args[@]}"; then
        echo "ABORT: subrepo pins not aligned (a repo is dirty or its pin is unreachable). Refusing to trade unaudited code."
        notify "RenQuant 104 ABORT" "Trading aborted: subrepo pins not aligned (dirty/unreachable). No trade."
        unset _pin_sync_args _pin_sync_runtime_root
        exit 1
    fi
    unset _pin_sync_args _pin_sync_runtime_root
    echo "Subrepo checkouts aligned to pins."
fi
