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
# RUN_ID must be a genuinely unique run/attempt id, fresh EVERY invocation
# (Codex review, PR #420 round 3): a plain second-resolution timestamp could
# in principle collide between a manual rerun and the cron firing in the same
# second, and — more importantly — is the exact identity value the marker's
# no-change attestation envelope is bound to (see tournament_retrain_marker.py
# _verify_no_change_receipt), so it must not be guessable/reproducible from
# the artifacts alone. uuidgen (present on macOS by default) is preferred;
# fall back to timestamp+PID+$RANDOM if unavailable.
if command -v uuidgen >/dev/null 2>&1; then
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$(uuidgen)"
else
    RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$-$RANDOM"
fi
LOG="$LOG_DIR/$DATE.log"

# "last successful tournament retrain" marker — consumed by the model-freshness
# monitor (orchestrator PR #213) so it can compute tournament age WITHOUT
# scanning 100+ per-ticker dirs. Co-located with the artifacts it describes.
#
# The marker is stamped by scripts/tournament_retrain_marker.py, which derives
# completion evidence from the ARTIFACTS themselves (Codex review, PR #420,
# three rounds): it freezes the expected watchlist, snapshots a PRE-RUN
# baseline (per-ticker digest + data cutoff) before launch, checks each
# per-ticker metadata was REWRITTEN this invocation (mtime >= LAUNCH_EPOCH,
# not a pre-existing orphan dir) AND that its digest actually changed vs the
# baseline, and requires the data cutoff to be non-regressing. Certification
# also HARD-REQUIRES exit_code == 0 — a train failure can never be overridden
# by artifact freshness. `trained_date` is bound to the min data cutoff
# (artifact-derived), NOT the wall clock. Stamped ONLY when certified; on any
# failure it is left untouched so its data cutoff keeps ageing → the monitor
# alerts too (belt-and-suspenders with the loud ntfy).
#
# Round 3 (Codex review, 2026-07-01): a byte-identical rewrite (idempotent
# retrain that legitimately reproduces identical output) can ONLY certify via
# a SEPARATE, out-of-band no-change attestation envelope
# ($NO_CHANGE_RECEIPTS below) bound to THIS invocation's $RUN_ID + artifact
# digest — an in-payload `no_change_reason` string was replayable (identical
# bytes ⇒ the string is necessarily pre-existing, so touching/re-copying an
# old artifact reproduced it and could pass off a stale corpus as fresh).
# Nothing on this host mints that envelope today (train_104.py does not yet
# emit one), so in practice every byte-identical rewrite currently fails
# certification and must be investigated — the safe, conservative default.
MARKER="$REPO_DIR/backtesting/renquant_104/models/.last_tournament_retrain.json"
MODELS_DIR="$REPO_DIR/backtesting/renquant_104/models"
STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"
EXPECTED_WATCHLIST="$LOG_DIR/$DATE.expected_watchlist.json"
EXPECTED_NON_TRAINABLE="$LOG_DIR/$DATE.expected_non_trainable.json"
BASELINE_FILE="$LOG_DIR/$DATE.pre_run_baseline.json"
# No process on this host writes this file today (see round-3 note above) —
# tournament_retrain_marker.py treats a missing path as "no attestations",
# so it is passed unconditionally and simply has no effect until a future
# training-side change legitimately mints run-bound envelopes here.
NO_CHANGE_RECEIPTS="$LOG_DIR/$DATE.no_change_receipts.json"

# Coverage policy (Codex review, PR #420 round 2): the hard-coded 0.90 floor
# was an unregistered magic number that could silently mask up to ~14 missing
# names. Replaced with an EXPLICIT, justified exclusion list — benchmark /
# sector / defensive ETFs the per-ticker tournament intentionally does not
# admit as portfolio candidates — derived LIVE from strategy_config.json
# (`benchmark`, `sector_etf_map` values, `defensive_tickers`) so it can never
# drift out of sync with watchlist/config edits. Every OTHER expected ticker
# (the trainable set) is required at 100% coverage — zero tolerance.

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

# ── Freeze the expected watchlist BEFORE launch ───────────────────────────────
# The completion marker is scored against THIS frozen set (Codex review, #420):
# the per-ticker tournament trains ctx.watchlist (kernel/pipeline/pp_training.py
# FeatureJob iterates config["watchlist"]). Freeze it now — before training can
# mutate anything — so coverage is measured against the universe the run was
# asked to cover, not against whatever orphan dirs happen to be on disk.
echo "--- Freezing expected watchlist from $STRATEGY_CONFIG ---"
if ! "$PYTHON" - "$STRATEGY_CONFIG" "$EXPECTED_WATCHLIST" <<'PY'
import json, sys
cfg_path, out_path = sys.argv[1:3]
cfg = json.load(open(cfg_path))
wl = cfg.get("watchlist") or []
if not wl:
    raise SystemExit("strategy_config has no watchlist — cannot freeze expected set")
json.dump({"watchlist": sorted(set(wl))}, open(out_path, "w"), indent=2)
print(f"Froze {len(set(wl))} expected tickers → {out_path}")
PY
then
    echo "=== weekly_tournament_retrain FAILED — could not freeze expected watchlist ==="
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✗" \
        "Weekly per-ticker tournament retrain ABORTED — could not freeze expected watchlist from $STRATEGY_CONFIG. Log: $LOG"
    exit 1
fi

# ── Freeze the intentional non-trainable exclusions BEFORE launch ────────────
# Derived LIVE from strategy_config.json — never hand-maintained — so it can
# never silently drift out of sync with watchlist/sector-map edits (Codex
# review, PR #420 round 2: the prior 0.90 blanket floor was an unregistered
# magic number). Each exclusion carries an explicit justification; every OTHER
# expected ticker is the trainable set and is required at 100% coverage.
echo "--- Freezing expected non-trainable exclusions from $STRATEGY_CONFIG ---"
if ! "$PYTHON" - "$STRATEGY_CONFIG" "$EXPECTED_WATCHLIST" "$EXPECTED_NON_TRAINABLE" <<'PY'
import json, sys
cfg_path, wl_path, out_path = sys.argv[1:4]
cfg = json.load(open(cfg_path))
watchlist = set(json.load(open(wl_path))["watchlist"])

benchmark = cfg.get("benchmark")
sector_etfs = set(cfg.get("sector_etf_map", {}).values())
defensive = set(cfg.get("defensive_tickers", []))

reasons: dict[str, str] = {}
if benchmark in watchlist:
    reasons[benchmark] = "benchmark index (strategy_config.benchmark) — regime/relative-strength reference, not a per-ticker tournament admission candidate"
for t in sorted(sector_etfs & watchlist):
    if t not in reasons:
        reasons[t] = "sector ETF (strategy_config.sector_etf_map value) — held for sector exposure/hedging, not a per-ticker tournament admission candidate"
for t in sorted(defensive & watchlist):
    if t not in reasons:
        reasons[t] = "defensive/hedge ETF (strategy_config.defensive_tickers) — regime-defensive sleeve, not a per-ticker tournament admission candidate"

json.dump(reasons, open(out_path, "w"), indent=2)
print(f"Froze {len(reasons)} non-trainable exclusions (of {len(watchlist)} watchlist) → {out_path}")
PY
then
    echo "=== weekly_tournament_retrain FAILED — could not freeze non-trainable exclusions ==="
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✗" \
        "Weekly per-ticker tournament retrain ABORTED — could not freeze non-trainable exclusions from $STRATEGY_CONFIG. Log: $LOG"
    exit 1
fi

# ── Snapshot the PRE-RUN baseline BEFORE launch ───────────────────────────────
# Per-ticker digest + data cutoff for whatever is currently on disk (Codex
# review, PR #420 round 2): mtime >= LAUNCH_EPOCH alone only proves a file was
# touched, not that its bytes changed. The marker compares POST-run artifacts
# against this baseline to require a genuine digest change (or an explicit
# no_change_reason) with a non-regressing cutoff.
echo "--- Snapshotting pre-run baseline (tournament_retrain_marker.py --emit-baseline) ---"
if ! "$PYTHON" scripts/tournament_retrain_marker.py \
        --models-dir "$MODELS_DIR" \
        --watchlist "$EXPECTED_WATCHLIST" \
        --emit-baseline "$BASELINE_FILE"; then
    echo "=== weekly_tournament_retrain FAILED — could not snapshot pre-run baseline ==="
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✗" \
        "Weekly per-ticker tournament retrain ABORTED — could not snapshot pre-run baseline. Log: $LOG"
    exit 1
fi

# ── Retrain the per-ticker tournament ─────────────────────────────────────────
# --skip-panel  : only the BaselineTournamentJob (panel/calibrator owned by
#                 weekly_wf_promote.sh); auto-disables acceptance staging.
# --force       : bypass the training.cadence gate — this launchd schedule is
#                 the single source of truth (see header). Without it the run
#                 can silently return on a cadence-weekday mismatch.
# --trigger     : audit tag only; does not change training flow.
#
# LAUNCH_EPOCH is captured immediately BEFORE launch: every artifact the run
# rewrites gets an mtime >= LAUNCH_EPOCH, so the marker can prove per-ticker
# freshness (rewritten this invocation) vs pre-existing orphan dirs.
echo "--- Retraining per-ticker tournament (train_104.py --skip-panel --force) ---"
LAUNCH_EPOCH=$(date +%s)
if "$PYTHON" scripts/train_104.py --skip-panel --force --trigger weekly_tournament_cadence; then
    RC=0
else
    RC=$?
fi

# NOTE: we deliberately do NOT `exit` here on RC != 0 (Codex review, PR #420
# round 2: "the shell must also pass the actual train exit code into
# certification and certification must require exit_code==0; artifact mtimes
# alone cannot override process failure"). RC is threaded into
# tournament_retrain_marker.py --exit-code below, which enforces exit_code==0
# as its OWN independent, always-checked gate — not just a shell short-circuit
# that a manual/partial invocation could bypass. We still fail loudly either
# way; the log line below records the raw train_104.py outcome immediately.
if [ "$RC" -ne 0 ]; then
    echo "train_104.py exited rc=$RC at $(date) — proceeding to marker stamp so exit_code is bound into certification evidence; certification WILL be refused."
fi

# ── Stamp the artifact-derived completion marker (certified runs only) ─────────
# train_104.py exiting 0 is necessary but NOT sufficient (Codex review, #420): a
# partial / no-op run can still exit 0. tournament_retrain_marker.py re-derives
# completion from the artifacts — per-ticker rewrite proof (mtime >= LAUNCH_EPOCH)
# PLUS digest-identity-vs-baseline / non-regressing cutoff — and independently
# HARD-REQUIRES --exit-code == 0 (round 2: the actual $RC is passed here even
# when non-zero, so a train failure can never be overridden by artifact
# freshness). Stamped ONLY when both the trainable set is 100% covered AND
# exit_code is 0; otherwise we FAIL LOUDLY and leave the prior marker untouched
# (its cutoff keeps ageing).
COMPLETED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
HOSTNAME_SHORT=$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo unknown)
echo "--- Stamping completion marker (tournament_retrain_marker.py) ---"
if "$PYTHON" scripts/tournament_retrain_marker.py \
        --models-dir "$MODELS_DIR" \
        --watchlist "$EXPECTED_WATCHLIST" \
        --non-trainable "$EXPECTED_NON_TRAINABLE" \
        --baseline "$BASELINE_FILE" \
        --no-change-receipts "$NO_CHANGE_RECEIPTS" \
        --launch-epoch "$LAUNCH_EPOCH" \
        --run-id "$RUN_ID" \
        --marker "$MARKER" \
        --exit-code "$RC" \
        --command "scripts/train_104.py --skip-panel --force" \
        --host "$HOSTNAME_SHORT" \
        --log "logs/weekly_tournament_retrain/$DATE.log" \
        --completed-at "$COMPLETED_AT" \
        --date "$DATE"; then
    echo "=== weekly_tournament_retrain PASSED at $(date) — completion CERTIFIED, marker stamped ==="
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✓" \
        "Weekly per-ticker tournament retrain CERTIFIED (see $MARKER). Universe admission refreshed. Log: $LOG"
else
    MRC=$?
    echo "=== weekly_tournament_retrain FAILED at $(date) — completion NOT certified (train rc=$RC, marker rc=$MRC) ==="
    # FAIL LOUDLY. Either train_104 itself failed (rc != 0, now hard-gated
    # inside the marker's own certification logic, not just shell control
    # flow) or it exited 0 but the artifacts do not meet the trainable-100%
    # coverage / identity-change / non-regression policy (partial / stale /
    # no-op / regressed). Either way this is exactly the silent-drift failure
    # this job exists to catch. Prior marker left untouched → monitor ages too.
    notify "RenQuant 104 TOURNAMENT-RETRAIN ✗" \
        "Weekly per-ticker tournament retrain NOT CERTIFIED (train exited $RC, certification rc=$MRC). Universe admission is stale/partial — investigate now. Log: $LOG"
    exit 1
fi
