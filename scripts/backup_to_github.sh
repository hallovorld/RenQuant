#!/usr/bin/env bash
# Cloud backup of critical state files to a GitHub private repo.
#
# What gets backed up (≈46MB total, scales slowly):
#   * data/runs.db                — panel training history (sqlite, 568K)
#   * data/runs.alpaca.db         — Alpaca live broker state (sqlite, 45M)
#   * data/insider_trades/*.parquet — cached SEC Form 4 data (~328K, slow to refetch)
#   * backtesting/renquant_104/live_state.{alpaca,paper}.json — current positions
#   * scripts/stage3_progress.json — Track D Stage 3 batch admission progress
#
# Production artifacts (panel-ltr.json, ngboost-head.json, etc) are already in
# the main RenQuant repo (committed). They don't need this backup.
#
# Usage:
#   bash scripts/backup_to_github.sh             # one-shot backup
#   BACKUP_REMOTE=git@github.com:user/repo.git bash scripts/backup_to_github.sh
#
# Env vars:
#   BACKUP_REMOTE — git URL of the private backup repo (required first time;
#                   cached after first clone in BACKUP_REPO/.git/config)
#   BACKUP_REPO   — local clone path (default: ~/.renquant-state-backup)
#   NTFY_TOPIC    — ntfy.sh topic for failure alerts (default: renquant)
#
# Setup (one-time, operator):
#   1. Create empty PRIVATE repo on GitHub (recommended name: renquant-state-backup).
#      Anywhere is fine — single-developer use, just needs to be private.
#   2. Run: BACKUP_REMOTE=git@github.com:USER/renquant-state-backup.git \
#          bash scripts/backup_to_github.sh
#   3. Verify the first backup pushed cleanly.
#   4. Wire to launchd via com.renquant.backup.plist for periodic backups.
#
# SQLite safety:
#   We use the SQLite online backup API (`.backup` command) which is safe to
#   call concurrently with writers (e.g. Stage 3 retraining writing to runs.db).
#   Falls back to plain `cp` if sqlite3 CLI is unavailable.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GITHUB_DIR="$(cd "$REPO_ROOT/.." && pwd)"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_ROOT"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_ROOT" "$GITHUB_DIR")"
VENV_DIR="$REPO_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
BACKUP_REPO="${BACKUP_REPO:-$HOME/.renquant-state-backup}"
NTFY_TOPIC="${NTFY_TOPIC:-renquant}"
TS_HUMAN="$(date '+%Y-%m-%d %H:%M:%S %Z')"
TS_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

notify_failure() {
    local msg="$1"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "RenQuant backup FAILED" -message "$msg" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: RenQuant backup FAILED" -d "$msg" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

trap 'notify_failure "Script failed at line $LINENO"' ERR

run_multirepo_backup() {
    local orchestrator_src
    orchestrator_src="$(renquant_subrepo_src "$SUBREPO_ROOT" renquant-orchestrator)"
    if PYTHONPATH="$orchestrator_src:${PYTHONPATH:-}" "$PYTHON" - <<'PY'
import renquant_orchestrator.state_backup  # noqa: F401
PY
    then
        PYTHONPATH="$orchestrator_src:${PYTHONPATH:-}" "$PYTHON" -m renquant_orchestrator.state_backup \
            --repo-root "$REPO_ROOT" \
            --backup-repo "$BACKUP_REPO"
        return $?
    fi
    return 127
}

if [ "${RQ_STATE_BACKUP_RUNNER:-multirepo}" != "legacy" ]; then
    if run_multirepo_backup; then
        exit 0
    fi
    BACKUP_RC=$?
    if [ "$BACKUP_RC" -ne 127 ]; then
        notify_failure "Multirepo backup pipeline failed rc=$BACKUP_RC at $TS_ISO"
        exit "$BACKUP_RC"
    fi
    if [ "${RQ_STATE_BACKUP_STRICT:-0}" = "1" ]; then
        notify_failure "renquant_orchestrator.state_backup unavailable and RQ_STATE_BACKUP_STRICT=1"
        echo "ERROR: renquant_orchestrator.state_backup unavailable and RQ_STATE_BACKUP_STRICT=1"
        exit 2
    fi
    echo "WARN: renquant_orchestrator.state_backup unavailable; falling back to legacy shell backup."
fi

# ── 1. Ensure backup repo exists locally ──────────────────────────────────────
if [ ! -d "$BACKUP_REPO/.git" ]; then
    if [ -z "${BACKUP_REMOTE:-}" ]; then
        echo "ERROR: BACKUP_REPO ($BACKUP_REPO) doesn't exist and BACKUP_REMOTE not set." >&2
        echo "First-time setup: BACKUP_REMOTE=<git-url> $0" >&2
        exit 1
    fi
    echo "Cloning backup repo from $BACKUP_REMOTE → $BACKUP_REPO"
    git clone "$BACKUP_REMOTE" "$BACKUP_REPO" || {
        # Empty repo case — clone fails because no commits yet; init manually.
        mkdir -p "$BACKUP_REPO"
        cd "$BACKUP_REPO"
        git init -b main
        git remote add origin "$BACKUP_REMOTE"
        echo "# RenQuant state backup" > README.md
        git add README.md
        git commit -m "init"
        git push -u origin main
        cd "$REPO_ROOT"
    }
fi

cd "$BACKUP_REPO"

# ── 2. Pull latest (in case of multi-host writes — paranoia guard) ────────────
git pull --rebase --autostash 2>&1 | tail -3 || true

# ── 3. Snapshot critical files ────────────────────────────────────────────────
mkdir -p data/insider_trades

# SQLite via online backup API (safe with concurrent writers)
backup_sqlite() {
    local src="$1" dst="$2"
    if [ ! -f "$src" ]; then
        echo "  skip: $src not found"
        return
    fi
    if command -v sqlite3 &>/dev/null; then
        sqlite3 "$src" ".backup '$dst'" 2>&1 \
            && echo "  sqlite3 .backup: $(basename $src) → $(basename $dst)" \
            || { echo "  sqlite3 backup failed; falling back to cp"; cp "$src" "$dst"; }
    else
        cp "$src" "$dst"
        echo "  cp (no sqlite3): $(basename $src) → $(basename $dst)"
    fi
}

echo "Snapshot at $TS_HUMAN"
backup_sqlite "$REPO_ROOT/data/runs.db" "$BACKUP_REPO/data/runs.db"
backup_sqlite "$REPO_ROOT/data/runs.alpaca.db" "$BACKUP_REPO/data/runs.alpaca.db"

# Live state JSON files (small, plain copy is fine)
for f in "$REPO_ROOT/backtesting/renquant_104"/live_state.*.json; do
    [ -f "$f" ] && cp "$f" "$BACKUP_REPO/$(basename $f)"
done

# Insider trades parquets (rsync — incremental, only copies changed)
if [ -d "$REPO_ROOT/data/insider_trades" ]; then
    rsync -aq --delete "$REPO_ROOT/data/insider_trades/" "$BACKUP_REPO/data/insider_trades/"
fi

# Stage 3 progress (running experiment state)
[ -f "$REPO_ROOT/scripts/stage3_progress.json" ] && \
    cp "$REPO_ROOT/scripts/stage3_progress.json" "$BACKUP_REPO/stage3_progress.json"
[ -f "$REPO_ROOT/scripts/stage3_final_watchlist.json" ] && \
    cp "$REPO_ROOT/scripts/stage3_final_watchlist.json" "$BACKUP_REPO/stage3_final_watchlist.json"

# ── 3.5. P0-17 audit: GitHub blocks pushes > 100 MB per file. SQLite db
#        grows ~50 KB/day at wl200 scale; will cross 100 MB sometime
#        in 2026. Hard-fail before git add if a file is already at the
#        block threshold; warn at 90 MB so the operator can migrate to LFS.
TOO_LARGE_FILES=$(find "$BACKUP_REPO" -type f -size +99M 2>/dev/null)
if [ -n "$TOO_LARGE_FILES" ]; then
    notify_failure "GitHub backup blocked: files exceed 99MB:\n$TOO_LARGE_FILES\nMigrate to Git LFS before next backup."
    echo "ERROR: files exceed GitHub 100MB push limit — migrate to Git LFS first:"
    echo "$TOO_LARGE_FILES"
    exit 1
fi

LARGE_FILES=$(find "$BACKUP_REPO" -type f -size +90M 2>/dev/null)
if [ -n "$LARGE_FILES" ]; then
    notify_payload="LARGE_FILES_DETECTED_>90MB:\n$LARGE_FILES\nMigrate to Git LFS before next backup."
    curl -s -d "$notify_payload" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
    echo "WARNING: files near 100MB limit — Git LFS needed: $LARGE_FILES"
fi

# ── 4. Commit + push if changed ───────────────────────────────────────────────
git add -A
if git diff --cached --quiet; then
    echo "No changes since last backup; skipping commit."
    exit 0
fi

git commit -m "backup $TS_ISO" --quiet
# Capture push exit to detect silent rejections (P0-17: 100MB block returns
# non-zero exit but tee swallows in plain pipe; explicit check here).
PUSH_LOG="$(mktemp)"
if git push origin main >"$PUSH_LOG" 2>&1; then
    tail -3 "$PUSH_LOG"
    rm -f "$PUSH_LOG"
else
    PUSH_RC=$?
    tail -20 "$PUSH_LOG"
    rm -f "$PUSH_LOG"
    notify_failure "Backup push FAILED rc=$PUSH_RC at $TS_ISO"
    echo "ERROR: push failed; backup commit is local-only"
    exit "$PUSH_RC"
fi
echo "Backup pushed at $TS_ISO"
