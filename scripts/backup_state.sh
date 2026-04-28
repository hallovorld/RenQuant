#!/usr/bin/env bash
# Snapshot broker-specific state files to a timestamped backup dir.
#
# Why: the 2026-04-27 incident showed that `live_state.json` is the
# only memory of regime cooldowns, sell streaks, position high-water
# marks, last-sell dates, and entry dates. A corrupt write or paper-
# smoke contamination can lose hours of state. This script makes a
# read-only snapshot per broker, callable from a cron.
#
# Usage:
#     scripts/backup_state.sh                      # default: alpaca + alpaca-paper
#     scripts/backup_state.sh alpaca paper         # specific brokers
#     scripts/backup_state.sh --all                # everything matching live_state.*.json
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRATEGY_DIR="$REPO_ROOT/backtesting/renquant_104"
BACKUP_ROOT="$REPO_ROOT/data/state_backups"
TS="$(date +%Y-%m-%d_%H%M)"

# Compute brokers to back up
brokers=()
if [[ "${1:-}" == "--all" ]]; then
    while IFS= read -r f; do
        # extract <broker> from live_state.<broker>.json
        bn="$(basename "$f" .json)"
        bn="${bn#live_state.}"
        brokers+=("$bn")
    done < <(find "$STRATEGY_DIR" -maxdepth 1 -name "live_state.*.json" 2>/dev/null)
elif [[ $# -eq 0 ]]; then
    brokers=("alpaca" "alpaca_paper")
else
    brokers=("$@")
fi

if [[ ${#brokers[@]} -eq 0 ]]; then
    echo "No brokers selected — nothing to back up."
    exit 0
fi

for broker in "${brokers[@]}"; do
    safe="${broker//-/_}"
    src_state="$STRATEGY_DIR/live_state.${safe}.json"
    src_db="$REPO_ROOT/data/runs.${safe}.db"

    dest_dir="$BACKUP_ROOT/${safe}/${TS}"
    mkdir -p "$dest_dir"

    if [[ -f "$src_state" ]]; then
        cp "$src_state" "$dest_dir/live_state.${safe}.json"
        chmod 444 "$dest_dir/live_state.${safe}.json"
        echo "  ✓ $src_state → $dest_dir/"
    else
        echo "  · $src_state missing — skipped"
    fi

    if [[ -f "$src_db" ]]; then
        # SQLite atomic backup (works even when the live runner has it open).
        sqlite3 "$src_db" ".backup '$dest_dir/runs.${safe}.db'"
        chmod 444 "$dest_dir/runs.${safe}.db"
        echo "  ✓ $src_db → $dest_dir/  (sqlite atomic)"
    else
        echo "  · $src_db missing — skipped"
    fi

    # Manifest with SHA256 for tamper detection
    if compgen -G "$dest_dir/*" > /dev/null; then
        ( cd "$dest_dir" && shasum -a 256 ./*.json ./*.db 2>/dev/null > MANIFEST.sha256 ) || true
        chmod 444 "$dest_dir/MANIFEST.sha256" 2>/dev/null || true
    fi
done

echo
echo "Backups under: $BACKUP_ROOT/<broker>/${TS}/"
echo "Verify with: shasum -a 256 -c <broker>/<ts>/MANIFEST.sha256"
