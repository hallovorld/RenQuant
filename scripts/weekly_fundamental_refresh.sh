#!/usr/bin/env bash
# weekly_fundamental_refresh.sh — Pull fresh fundamental + earnings data.
#
# A3 (2026-05-09 audit, Issue #3, opened 2026-05-11):
# `sec_fundamentals_daily.parquet` was last regen'd 2026-05-10 but ended
# at 2026-02-10 because the script ran before the Q1 2026 10-Q wave
# landed at SEC EDGAR (40-45 day filing window after quarter end).
# Same story for `earnings_surprise/*.parquet` (Q4 2025 surprises only).
# Inference at `job_panel_scoring.py:197` forward-fills the latest
# available row silently — so live model has been scoring on
# 90-day-stale fundamentals.
#
# Why weekly: filings trickle in continuously after each quarter-end
# (~40-45 day deadline). Weekly catches the bulk of each wave within
# ~5 days of the deadline without thrashing the SEC API:
#   * Q1 ends Mar 31 → 10-Q deadline mid-May → fetch weekly thereafter
#   * Q2 ends Jun 30 → deadline mid-Aug
#   * Q3 ends Sep 30 → deadline mid-Nov
#   * Q4 ends Dec 31 → 10-K deadline late-Mar (large filers) / mid-Apr
#
# Schedule: Saturdays 04:00 PT (offline hours, before Monday open).
# Plist: scripts/launchd/com.renquant.weekly-fundamental-refresh.plist
#
# Steps:
#   1. SEC EDGAR fundamentals (sec_fundamentals_daily.parquet)
#   2. Extended fundamentals (sec_fundamentals_extended.parquet)
#   3. PEAD/SUE earnings surprises (data/earnings_surprise/*.parquet)
#   4. Freshness gate: panel's last date must be within RECENCY_GATE_DAYS
#      of today; ALERT (but don't fail) if older — this happens naturally
#      between filing waves.
#   5. ntfy summary
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="/Users/renhao/git/github/RenQuant/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/weekly_fundamental_refresh"
NTFY_TOPIC="renquant"
RECENCY_GATE_DAYS=60   # alert if panel >60d stale (1 quarter cycle minus buffer)
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

fail_missing_strategy_config() {
    local strict_env="$1"
    if renquant_strict_enabled "$strict_env"; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable and strict multirepo mode is enabled" \
            | tee -a "$LOG"
    else
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable; scheduled refresh defaults to fail-closed multirepo execution" \
            | tee -a "$LOG"
    fi
}

mark_multirepo_unavailable() {
    local module="$1" strict_env="$2"
    if renquant_strict_enabled "$strict_env"; then
        echo "ERROR: $module unavailable and strict multirepo mode is enabled" \
            | tee -a "$LOG"
    else
        echo "ERROR: $module unavailable; scheduled refresh defaults to fail-closed multirepo execution" \
            | tee -a "$LOG"
    fi
}

cd "$REPO_DIR"
source "$VENV_DIR/bin/activate"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common):${PYTHONPATH:-}"
echo "[$(date '+%H:%M:%S')] Weekly fundamental refresh — $DATE" | tee -a "$LOG"

if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    fail_missing_strategy_config RQ_DATA_REFRESH_STRICT
    exit 1
fi

# ── Steps 1-2: SEC EDGAR fundamentals ───────────────────────────────
echo "[$(date '+%H:%M:%S')] Steps 1-2: SEC EDGAR daily + extended fund refresh …" | tee -a "$LOG"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.sec_fundamentals  # noqa: F401
PY
then
    $PYTHON -m renquant_base_data.sec_fundamentals \
        --mode both \
        --universe "$REPO_DIR/scripts/watchlist_universe.json" \
        --data-dir "$REPO_DIR/data" \
        --end-year "$(date +%Y)" \
        --json >> "$LOG" 2>&1
    STEP1_RC=$?
    STEP2_RC=$STEP1_RC
else
    mark_multirepo_unavailable "renquant_base_data.sec_fundamentals" RQ_DATA_REFRESH_STRICT
    STEP1_RC=1
    STEP2_RC=1
fi

# ── Step 3: PEAD/SUE earnings surprises ─────────────────────────────
echo "[$(date '+%H:%M:%S')] Step 3: PEAD/SUE refresh …" | tee -a "$LOG"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.earnings_surprise_refresh  # noqa: F401
PY
then
    $PYTHON -m renquant_base_data.earnings_surprise_refresh \
        --strategy-config "$STRATEGY_CONFIG" \
        --data-dir "$REPO_DIR/data" \
        --json >> "$LOG" 2>&1
    STEP3_RC=$?
else
    mark_multirepo_unavailable "renquant_base_data.earnings_surprise_refresh" RQ_DATA_REFRESH_STRICT
    STEP3_RC=1
fi

# ── Step 4: freshness gate (info-only — filings have inherent lag) ──
GATE_REPORT=$($PYTHON <<PY 2>>"$LOG"
import pandas as pd, datetime as dt
today = dt.date.today()
gate = $RECENCY_GATE_DAYS

def lag(path, date_col="date"):
    try:
        df = pd.read_parquet(path)
        d = pd.to_datetime(df[date_col]).max().date()
        return (today - d).days, d
    except Exception as e:
        return -1, f"ERR:{e}"

f1_lag, f1_d = lag("data/sec_fundamentals_daily.parquet")
f2_lag, f2_d = lag("data/sec_fundamentals_extended.parquet")

# PEAD: scan a few sample files
import glob
es_files = sorted(glob.glob("data/earnings_surprise/*.parquet"))
es_max_date = None
for f in es_files[:50]:    # sample
    try:
        df = pd.read_parquet(f)
        if "date" in df.columns:
            d = pd.to_datetime(df["date"]).max().date()
        else:
            d = pd.to_datetime(df.index).max().date()
        if es_max_date is None or d > es_max_date:
            es_max_date = d
    except Exception:
        pass
es_lag = (today - es_max_date).days if es_max_date else -1

lines = [
    f"fund_daily:    last={f1_d}  lag={f1_lag}d",
    f"fund_extended: last={f2_d}  lag={f2_lag}d",
    f"pead_sample:   last={es_max_date}  lag={es_lag}d  (sampled {min(50, len(es_files))} files)",
]
status = "OK"
if f1_lag > gate or f2_lag > gate or es_lag > gate:
    status = f"STALE_>{gate}d"
print(status + "|" + " ; ".join(lines))
PY
)
echo "[$(date '+%H:%M:%S')] Freshness: $GATE_REPORT" | tee -a "$LOG"

# ── Step 5: ntfy summary ────────────────────────────────────────────
RCS="step1=$STEP1_RC step2=$STEP2_RC step3=$STEP3_RC"
if [[ $STEP1_RC -ne 0 || $STEP2_RC -ne 0 || $STEP3_RC -ne 0 ]]; then
    notify "DATA REFRESH ✗" "$DATE  $RCS  $GATE_REPORT"
    exit 1
fi
notify "DATA REFRESH ✓" "$DATE  $RCS  $GATE_REPORT"
echo "[$(date '+%H:%M:%S')] DONE" | tee -a "$LOG"
