#!/usr/bin/env bash
# Install every .plist under scripts/launchd/ into ~/Library/LaunchAgents
# and launchctl-load it. Idempotent: unload+reload existing ones.
#
# 2026-04-24: two new plists shipped this session —
#   com.renquant.conditional-retrain104.plist  (13:10 PT Mon-Fri)
#   com.renquant.screen-watchlist.plist        (Sun 12:05 PT)
#
# Usage:
#   bash scripts/install_launchagents.sh
#   bash scripts/install_launchagents.sh --dry-run
#
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
SRC_DIR="$REPO_DIR/scripts/launchd"
DEST_DIR="$HOME/Library/LaunchAgents"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=1
fi

mkdir -p "$DEST_DIR"

for plist_src in "$SRC_DIR"/*.plist; do
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
echo ""
echo "Uninstall one:"
echo "  launchctl unload ~/Library/LaunchAgents/com.renquant.XYZ.plist"
echo "  rm              ~/Library/LaunchAgents/com.renquant.XYZ.plist"
