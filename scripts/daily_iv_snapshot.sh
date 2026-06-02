#!/usr/bin/env bash
# Daily IV snapshot — captures one EOD chain snapshot per watchlist ticker.
#
# Roadmap C1 (2026-05-18): Alpaca Free Options API gives current snapshots
# only (no historical chains). To build a usable IV-feature panel for
# retrain, we accumulate one snapshot per trading day across 103 tickers.
# After ~6 months we'll have enough panel rows to attempt integration.
#
# Cadence rationale (§5.13.6): IV updates throughout the day; the EOD
# value captures the day's terminal market view. One snapshot per day
# adds one row per ticker — that's the maximum information per tick
# the panel can use (since panel features are daily).
#
# Schedule: weekdays 13:30 PT (16:30 ET) = 30min after market close.
# Skips weekends and holidays implicitly (the API returns empty/stale
# data on non-trading days; the per-symbol parquet's dedupe on as_of
# prevents bad rows from polluting history).

set -e
REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
GITHUB_DIR="$(dirname "$REPO_DIR")"

cd "$REPO_DIR"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
mkdir -p logs/iv_snapshot

LOG="logs/iv_snapshot/daily_$(date +%Y%m%d).log"
echo "=== $(date) — Daily IV snapshot starting ===" > "$LOG"

set -a
source .env
set +a

if ! STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    if [ "${RENQUANT_STRICT_SUBREPO_PATHS:-0}" = "1" ] || [ "${RQ_DAILY_IV_STRICT:-0}" = "1" ]; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable" \
            | tee -a "$LOG"
        exit 1
    fi
    STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"
fi

export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-base-data renquant-common):${PYTHONPATH:-}"
if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_base_data.options_iv_refresh  # noqa: F401
PY
then
    "$PYTHON" -u -m renquant_base_data.options_iv_refresh \
        --strategy-config "$STRATEGY_CONFIG" \
        --data-dir "$REPO_DIR/data" \
        --json 2>&1 | tee -a "$LOG"
elif [ "${RQ_DAILY_IV_STRICT:-0}" = "1" ]; then
    echo "ERROR: renquant_base_data.options_iv_refresh unavailable and RQ_DAILY_IV_STRICT=1" \
        | tee -a "$LOG"
    exit 1
else
    echo "WARN: renquant_base_data.options_iv_refresh unavailable; falling back to umbrella script." \
        | tee -a "$LOG"
    "$PYTHON" -u scripts/fetch_options_iv_alpaca.py 2>&1 | tee -a "$LOG"
fi

echo "=== $(date) — Daily IV snapshot done ===" | tee -a "$LOG"
