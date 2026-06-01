#!/usr/bin/env bash
# Install active RenQuant launchd plists into ~/Library/LaunchAgents and
# launchctl-load them. Idempotent: unload+reload existing ones.
#
# 2026-04-24: two new plists shipped this session —
#   com.renquant.conditional-retrain104.plist  (13:10 PT Mon-Fri)
#   com.renquant.screen-watchlist.plist        (Sun 12:05 PT)
#
# Usage:
#   bash scripts/install_launchagents.sh
#   bash scripts/install_launchagents.sh --dry-run
#   bash scripts/install_launchagents.sh --check
#
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR="$REPO_DIR/scripts/launchd"
DEST_DIR="$HOME/Library/LaunchAgents"

DRY_RUN=0
CHECK_ONLY=0
case "${1:-}" in
    --dry-run) DRY_RUN=1 ;;
    --check) CHECK_ONLY=1 ;;
    "") ;;
    *)
        echo "Usage: bash scripts/install_launchagents.sh [--dry-run|--check]" >&2
        exit 2
        ;;
esac

mkdir -p "$DEST_DIR"

plists=("$SRC_DIR"/*.plist "$REPO_DIR/scripts/com.renquant.backup.plist")

if [ "$CHECK_ONLY" = "1" ]; then
    launchagents_rc=0
    subrepo_contract_rc=0
    python3 "$REPO_DIR/scripts/check_launchagents.py" --launchagents-dir "$DEST_DIR" || launchagents_rc=$?
    python3 "$REPO_DIR/scripts/subrepo_ops_contract.py" || subrepo_contract_rc=$?
    if [ "$launchagents_rc" -ne 0 ] || [ "$subrepo_contract_rc" -ne 0 ]; then
        exit 1
    fi
    exit 0
fi

for plist_src in "${plists[@]}"; do
    [ -f "$plist_src" ] || { echo "No plists found under $SRC_DIR"; exit 0; }
    label=$(basename "$plist_src" .plist)
    plist_dst="$DEST_DIR/$(basename "$plist_src")"

    echo "── $label ──"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  would copy: $plist_src → $plist_dst"
        echo "  would: launchctl unload $plist_dst 2>/dev/null || true"
        echo "  would: launchctl load   $plist_dst"
        continue
    fi

    # Unload any existing instance (best effort)
    if [ -f "$plist_dst" ]; then
        launchctl unload "$plist_dst" 2>/dev/null || true
        echo "  unloaded previous instance"
    fi

    cp "$plist_src" "$plist_dst"
    launchctl load "$plist_dst"
    echo "  installed + loaded"
done

echo ""
echo "Done. Inspect with:"
echo "  launchctl list | grep renquant"
echo "  bash scripts/install_launchagents.sh --check"
echo ""
echo "Uninstall one:"
echo "  launchctl unload ~/Library/LaunchAgents/com.renquant.XYZ.plist"
echo "  rm              ~/Library/LaunchAgents/com.renquant.XYZ.plist"
