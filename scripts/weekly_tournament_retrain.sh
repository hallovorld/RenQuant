#!/usr/bin/env bash
# weekly_tournament_retrain.sh — Weekly per-ticker TOURNAMENT retrain.
#
# THE MISSING CADENCE (2026-06-30). The 104 model refresh has three
# populations:
#   1. panel-LTR (alpha158+fund GBDT / PatchTST) — weekly_wf_promote.sh (Sat)
#   2. alpha158 linear head              — retrain-alpha158-linear (daily)
#   3. PER-TICKER TOURNAMENT              — <<< THIS SCRIPT (was UNSCHEDULED)
#
# The per-ticker tournament (RL Q-table + RandomForest + per-ticker XGB, in
# backtesting/renquant_104/models/<TICKER>/) is the population that gates
# UNIVERSE ADMISSION. It is produced by `scripts/train_104.py --skip-panel`
# (FullTrainingPipeline's BaselineTournamentJob). It had NO scheduled job:
# `launchctl list | grep renquant` showed retrain-panel104 (a compat no-op
# delegating to weekly_wf_promote = PANEL only), conditional-retrain104
# (VIX/SPY anomaly → weekly_wf_promote, must NOT call train_104 directly),
# retrain-alpha158-linear, monthly-meta-label-retrain, weekly-wf-promote —
# NONE runs the per-ticker tournament. So it silently aged to 61d and had to
# be hand-retrained on 2026-06-30, after starving the universe of admissions
# (a recurring driver of the no-buys). This script restores a reliable weekly
# cadence and — critically — FAILS LOUDLY (ntfy) on any non-zero exit, so a
# silent failure can never again age the tournament out unnoticed.
#
# Schedule: Sunday 06:00 PT (Weekday=0). Chosen to avoid CPU contention with
# the two Saturday 04:00 core-saturating jobs (weekly_wf_promote ~90 min +
# weekly_fundamental_refresh) and to land before Sunday's lighter jobs
# (retrain-panel104 10:00 no-op, weekly-apy104 12:00, screen-watchlist 12:05).
# Market is closed all weekend → full compute headroom, no live-trade overlap.
# Plist: scripts/launchd/com.renquant.weekly-tournament-retrain.plist
#
# --force is DELIBERATE: FullTrainingPipeline has a training.cadence gate
# (_cadence_allows_today) that silently returns (exit 0, no retrain) when the
# configured cadence weekday does not match today — and the weekday convention
# differs from launchd (Python Mon=0..Sun=6 vs launchd Sun=0..Sat=6). --force
# makes THIS launchd schedule the single source of truth for WHEN the
# tournament retrains, so the job can never silently no-op the way the missing
# cadence did. --skip-panel confines the run to the BaselineTournamentJob
# (panel/calibrator are owned by weekly_wf_promote.sh); with --skip-panel,
# train_104.py auto-disables the ModelAcceptanceGate (there is no candidate
# panel artifact), so the per-ticker exports are direct production writes —
# the intended, WF-ungated refresh path for the tournament.
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
# 2026-05-11 audit M-env: .venv per feedback_python_env.md (matches sibling scripts).
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/weekly_tournament_retrain"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/$DATE.log"

# "last successful tournament retrain" marker — consumed by the model-freshness
# monitor (orchestrator PR #213) so it can compute tournament age WITHOUT
# scanning 100+ per-ticker dirs. Co-located with the artifacts it describes.
# Contract: JSON with `trained_date` (same field the panel artifact exposes),
# `status`, `completed_at`, `run_id`, `command`, `exit_code`, `n_models`, `log`.
# Stamped ONLY on exit 0. On failure it is left untouched so its `trained_date`
# keeps ageing → the monitor alerts too (belt-and-suspenders with the loud ntfy).
MARKER="$REPO_DIR/backtesting/renquant_104/models/.last_tournament_retrain.json"

notify() {
    local title="$1" body="$2"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

# Credentials are not required to train the tournament (it reads local price /
# feature data), but source .env if present for parity with sibling retrain
# wrappers. Non-fatal when absent.
CRED_FILE="$REPO_DIR/.env"
if [ -f "$CRED_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    source "$CRED_FILE"
    set +a
fi

# Resolve the pinned-subrepo runtime PYTHONPATH exactly like daily_104.sh so
# train_104.py imports lock-pinned renquant-* primitives (falls back to sibling
# checkouts when no assembly env is present). train_104.py reads its own
# umbrella strategy config (backtesting/renquant_104/strategy_config.json), so
# we only need the import path here, not a config override.
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export RENQUANT_REPO_ROOT="$REPO_DIR"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

exec >> "$LOG" 2>&1
echo "=== weekly_tournament_retrain started at $(date) (run_id=$RUN_ID) ==="

# ── Lock — prevent concurrent / stacked runs ──────────────────────────────────
# Mirrors weekly_wf_promote.sh: a multi-ticker retrain is long, so a manual
# rerun could stack on the cron. Stale-PID aware (kill -0) so a lock left by a
# SIGKILL / hard reboot does not silently block every future run.
LOCK_FILE="/tmp/renquant_104_weekly_tournament.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING" != "?" ] && [ -n "$EXISTING" ] && ! kill -0 "$EXISTING" 2>/dev/null; then
        echo "Stale lock (PID $EXISTING dead) — clearing."
        rm -f "$LOCK_FILE"
        echo $$ > "$LOCK_FILE"
    else
        echo "Another weekly_tournament_retrain run is active (PID=$EXISTING) — skipping."
        notify "RenQuant 104 SKIP" "Weekly tournament retrain skipped — already running (PID=$EXISTING)"
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

# Saturate this host per CLAUDE.md §5.10 — the tournament fits RandomForest +
# per-ticker XGB across the whole universe. Derive from the live core count so
# the budget travels across Apple Silicon upgrades (mirrors weekly_wf_promote).
THREADS=$("$PYTHON" - <<'PY'
import os
print(os.cpu_count() or 1)
PY
)
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export VECLIB_MAXIMUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
echo "Hardware threads: $THREADS"

cd "$REPO_DIR"

# ── Retrain the per-ticker tournament ─────────────────────────────────────────
# --skip-panel  : only the BaselineTournamentJob (panel/calibrator owned by
#                 weekly_wf_promote.sh); auto-disables acceptance staging.
# --force       : bypass the training.cadence gate — this launchd schedule is
#                 the single source of truth (see header). Without it the run
#                 can silently return on a cadence-weekday mismatch.
# --trigger     : audit tag only; does not change training flow.
echo "--- Retraining per-ticker tournament (train_104.py --skip-panel --force) ---"
if "$PYTHON" scripts/train_104.py --skip-panel --force --trigger weekly_tournament_cadence; then
    RC=0
else
    RC=$?
fi

if [ "$RC" -ne 0 ]; then
    echo "=== weekly_tournament_retrain FAILED at $(date) (rc=$RC) ==="
    # FAIL LOUDLY — a silent failure is exactly what aged the tournament out.
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✗" \
        "Weekly per-ticker tournament retrain FAILED (rc=$RC). Universe admission will drift stale — investigate now. Log: $LOG"
    exit 1
fi

# ── Stamp the "last successful tournament retrain" marker (success only) ───────
N_MODELS=$(ls -d "$REPO_DIR/backtesting/renquant_104/models/"*/ 2>/dev/null | wc -l | tr -d ' ')
COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)
if "$PYTHON" - "$MARKER" "$DATE" "$COMPLETED_AT" "$RUN_ID" "$N_MODELS" "$HOSTNAME_SHORT" \
        "logs/weekly_tournament_retrain/$DATE.log" <<'PY'
import json, sys
marker, date, completed_at, run_id, n_models, host, log_rel = sys.argv[1:8]
payload = {
    "job": "weekly_tournament_retrain",
    "status": "success",
    "trained_date": date,               # matches the panel artifact's field name
    "completed_at": completed_at,       # UTC ISO-8601
    "run_id": run_id,
    "command": "scripts/train_104.py --skip-panel --force",
    "exit_code": 0,
    "n_models": int(n_models),
    "host": host,
    "log": log_rel,
}
with open(marker, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"Stamped marker {marker}: trained_date={date} n_models={n_models}")
PY
then
    echo "Marker stamped."
else
    # Non-fatal: the retrain itself succeeded; only the freshness marker failed
    # to write. Alert so the monitor is not blinded, but do not fail the run.
    echo "WARN: marker stamp failed (retrain itself succeeded)."
    notify "RenQuant 104 TOURNAMENT-MARKER ⚠" \
        "Tournament retrain SUCCEEDED but the freshness marker failed to write ($MARKER). Monitor may false-alarm. Log: $LOG"
fi

echo "=== weekly_tournament_retrain PASSED at $(date) — $N_MODELS ticker models refreshed ==="
notify "RenQuant 104 TOURNAMENT-RETRAIN ✓" \
    "Weekly per-ticker tournament retrained ($N_MODELS ticker models). Universe admission refreshed. Log: $LOG"
