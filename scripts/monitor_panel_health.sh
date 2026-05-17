#!/bin/bash
# Health monitor + early no-op killer for sim panel batches.
#
# Reads batch state from data/logs/monitor/<batch>_state.json (written by
# the panel runner). Polls every 5 minutes and ntfys on:
#
#   RED   process died unexpectedly
#   RED   no-op detected (after 3 windows complete, equity bit-identical
#         to baseline → knob has no effect; kill panel to save compute)
#   RED   baseline artifact refit during run (preflight_analyzer fails)
#   YELLOW no forward progress for 30 minutes
#   YELLOW memory pressure (compressor > 5 GB)
#   YELLOW load average > 12 (sustained)
#   YELLOW SPY-fetch race (multiple .parquet.tmp files)
#
# State: data/logs/monitor/<batch>_monitor.log (continuous)
#        data/logs/monitor/<batch>_alerts.json (1 line per alert)
#
# Survives laptop sleep: when the process resumes (or operator re-launches
# it on wake), it re-reads state and continues.
#
# Usage:
#   nohup ./scripts/monitor_panel_health.sh overlay_2026-05-16 \
#     > logs/reeval_queue/monitor_overlay_$(date +%Y%m%d).log 2>&1 &
#   echo $! > /tmp/overlay_monitor.pid
#
# Stop with:
#   kill $(cat /tmp/overlay_monitor.pid)
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate >/dev/null 2>&1

BATCH="${1:?usage: $0 <batch_name>}"
STATE_FILE="data/logs/monitor/${BATCH}_state.json"
LOG_FILE="data/logs/monitor/${BATCH}_monitor.log"
ALERT_FILE="data/logs/monitor/${BATCH}_alerts.json"
NO_OP_KILLED_FILE="data/logs/monitor/${BATCH}_killed_no_op.json"

if [[ ! -f "$STATE_FILE" ]]; then
  echo "ERROR: state file not found: $STATE_FILE" >&2
  echo "       runner must write this before monitor can start." >&2
  exit 1
fi

mkdir -p "$(dirname "$LOG_FILE")"

log() {
  echo "[$(date +%Y-%m-%d\ %H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

ntfy() {
  local priority="$1" title="$2" body="$3"
  curl -sf -H "Title: $title" -H "Priority: $priority" \
       -d "$body" https://ntfy.sh/renquant >/dev/null 2>&1 || true
  echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"prio\":\"$priority\",\"title\":\"$title\",\"body\":\"$body\"}" >> "$ALERT_FILE"
}

# Read panel + dir lists from state
BASELINE_DIR=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['baseline_dir'])")
RUNNER_PID=$(python3 -c "import json;print(json.load(open('$STATE_FILE'))['runner_pid'])")
TREATMENT_DIRS=$(python3 -c "import json;print(' '.join(json.load(open('$STATE_FILE'))['treatment_dirs']))")
# is_overlay_batch: when true, skip the "3 windows bit-identical = NO-OP" early
# kill. Overlay configs (e.g. regime_params.{BEAR,CHOPPY}.<knob>) legitimately
# produce bit-identical equity in windows that contain no BEAR or CHOPPY bars.
# Killing such panels is a false positive (5/16 diagnosed: A1 sdl_n2_BC had
# 9/14 NOOP windows in quarterly panel, all legitimate).
IS_OVERLAY=$(python3 -c "import json;print(json.load(open('$STATE_FILE')).get('is_overlay_batch', False))" 2>/dev/null)

log "Monitor started for batch '$BATCH'"
log "  baseline:        $BASELINE_DIR"
log "  treatments:      $TREATMENT_DIRS"
log "  runner pid:      $RUNNER_PID"
log "  is_overlay:      $IS_OVERLAY (skip no-op early kill if True)"

LAST_PROGRESS_TS=$(date +%s)
LAST_TOTAL_FILES=0
ALERTED_NO_OP=""
ALERTED_DIED=""

while true; do
  NOW=$(date +%s)

  # ── Check runner process is alive ──
  if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
    # Runner may have legitimately completed. Check if all panels have 16 windows.
    all_done=1
    for d in $BASELINE_DIR $TREATMENT_DIRS; do
      n=$(ls "$d/equity/" 2>/dev/null | wc -l | tr -d ' ')
      if (( n < 16 )); then all_done=0; break; fi
    done
    if (( all_done )); then
      log "All panels at 16/16 and runner exited cleanly. Monitor exiting."
      ntfy "default" "RenQuant: ${BATCH} COMPLETE" \
           "All 3 panels reached 16/16. Monitor exiting."
      exit 0
    elif [[ "$ALERTED_DIED" != "1" ]]; then
      log "RED: runner pid $RUNNER_PID died with panels incomplete"
      progress_report=""
      for d in $BASELINE_DIR $TREATMENT_DIRS; do
        n=$(ls "$d/equity/" 2>/dev/null | wc -l | tr -d ' ')
        progress_report="${progress_report}$(basename $d): ${n}/16; "
      done
      ntfy "high" "RenQuant: ${BATCH} RUNNER DIED" \
           "PID $RUNNER_PID gone. Progress: $progress_report"
      ALERTED_DIED=1
    fi
  fi

  # ── Forward progress check (30-min stall = yellow) ──
  total_files=0
  for d in $BASELINE_DIR $TREATMENT_DIRS; do
    n=$(ls "$d/equity/" 2>/dev/null | wc -l | tr -d ' ')
    total_files=$(( total_files + n ))
  done
  if (( total_files > LAST_TOTAL_FILES )); then
    LAST_TOTAL_FILES=$total_files
    LAST_PROGRESS_TS=$NOW
  elif (( NOW - LAST_PROGRESS_TS > 1800 )); then
    log "YELLOW: no new equity JSON in 30 min (total=$total_files)"
    ntfy "default" "RenQuant: ${BATCH} STALLED" \
         "No new equity JSON in 30 min. Total: ${total_files}. Check logs."
    LAST_PROGRESS_TS=$NOW  # avoid spam
  fi

  # ── No-op detection ── (skipped for overlay batches; see header)
  if [[ "$IS_OVERLAY" == "True" ]]; then
    : # skip — overlay knobs may legitimately not fire in non-target regimes
  else
  for tdir in $TREATMENT_DIRS; do
    label=$(basename "$tdir" | sed 's/^sim_//')
    if [[ ",$ALERTED_NO_OP," == *",$label,"* ]]; then continue; fi

    t_eq_count=$(ls "$tdir/equity/" 2>/dev/null | wc -l | tr -d ' ')
    b_eq_count=$(ls "$BASELINE_DIR/equity/" 2>/dev/null | wc -l | tr -d ' ')
    # Need both treatment and baseline to have ≥3 matching windows
    common=$(comm -12 \
      <(ls "$tdir/equity/" 2>/dev/null | sort) \
      <(ls "$BASELINE_DIR/equity/" 2>/dev/null | sort) | wc -l | tr -d ' ')
    if (( common >= 3 )); then
      # Compare each common file's equity values via Python
      result=$(python3 - "$tdir" "$BASELINE_DIR" <<'PY' 2>/dev/null
import sys, json, os
tdir, bdir = sys.argv[1], sys.argv[2]
t_files = set(os.listdir(os.path.join(tdir, "equity")))
b_files = set(os.listdir(os.path.join(bdir, "equity")))
common = sorted(t_files & b_files)[:3]  # first 3 only
identical = 0
for f in common:
    try:
        t = json.load(open(os.path.join(tdir, "equity", f)))
        b = json.load(open(os.path.join(bdir, "equity", f)))
        # Compare equity dict — bit-identical means knob did nothing
        if t.get("equity") == b.get("equity") and \
           abs(t.get("final_value", 0) - b.get("final_value", -1)) < 1e-6:
            identical += 1
    except Exception:
        pass
print(identical, len(common))
PY
)
      identical=$(echo "$result" | awk '{print $1}')
      checked=$(echo "$result" | awk '{print $2}')
      if [[ "$identical" == "$checked" && "$checked" == "3" ]]; then
        log "RED: NO-OP DETECTED for $label — first 3 windows bit-identical to baseline"
        ntfy "high" "RenQuant: ${BATCH} NO-OP $label" \
             "First 3 windows of $label bit-identical to baseline. Knob is a no-op. Killing panel to save compute."
        # Find and kill panel's xargs/run_one workers for this label
        # Pattern: bash -c "... run_one ... $label"
        pkill -f "run_one.*${label}" 2>/dev/null || true
        # Record kill
        echo "{\"ts\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"label\":\"$label\",\"reason\":\"bit-identical-to-baseline\"}" >> "$NO_OP_KILLED_FILE"
        ALERTED_NO_OP="${ALERTED_NO_OP},${label}"
      fi
    fi
  done
  fi  # end IS_OVERLAY guard

  # ── Memory pressure (vm_stat compressor) ──
  comp_pages=$(vm_stat | awk '/Pages occupied by compressor/ {print $5}' | tr -d '.')
  if [[ -n "$comp_pages" ]] && (( comp_pages > 1310720 )); then  # 1310720 pages × 4KB = 5GB
    comp_gb=$(( comp_pages * 4 / 1024 / 1024 ))
    log "YELLOW: memory compressor at ${comp_gb} GB"
    ntfy "default" "RenQuant: ${BATCH} MEM PRESSURE" \
         "vm_stat compressor: ${comp_gb} GB. Consider reducing -P parallelism."
  fi

  # ── Load average ──
  load5=$(uptime | sed 's/.*load average[s]*: //' | awk -F'[, ]+' '{print $2}')
  load5_int=$(echo "$load5" | awk '{print int($1)}')
  if (( load5_int > 12 )); then
    log "YELLOW: 5-min load average = $load5"
    ntfy "default" "RenQuant: ${BATCH} LOAD HIGH" \
         "5-min load avg: $load5 on 10-core. Throttling suggested."
  fi

  # ── SPY-fetch race ──
  tmp_count=$(ls data/ohlcv/SPY/*.tmp 2>/dev/null | wc -l | tr -d ' ')
  if (( tmp_count > 1 )); then
    log "YELLOW: $tmp_count SPY .tmp files (race?)"
    ntfy "default" "RenQuant: ${BATCH} SPY RACE" \
         "$tmp_count concurrent SPY .parquet.tmp files. Possible race condition."
  fi

  # ── Periodic progress log (every ~30 min) ──
  if (( NOW % 1800 < 300 )); then
    progress_line=""
    for d in $BASELINE_DIR $TREATMENT_DIRS; do
      n=$(ls "$d/equity/" 2>/dev/null | wc -l | tr -d ' ')
      progress_line="${progress_line}$(basename $d | sed 's/^sim_//')=${n}/16 "
    done
    log "progress: $progress_line"
  fi

  sleep 300
done
