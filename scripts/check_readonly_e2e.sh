#!/usr/bin/env bash
# check_readonly_e2e.sh — gold-standard deploy verify: run a FULL readonly
# daily-full end-to-end and assert it produces a decision (does not crash, does
# not go silent). The heavier companion to check_conviction_admits.py: it
# exercises the WHOLE pipeline code path (panel assembly → scoring → gates → QP →
# sizing → execution-plan), so a broad pin bump (e.g. the orchestrator) that
# breaks any stage is caught here, not in production.
#
# SAFE BY CONSTRUCTION: reuses the daily shadow mechanism — `--broker
# readonly-alpaca` + the shadow config whose broker_name=alpaca_shadow isolates
# ALL state to live_state.alpaca_shadow.json + runs_alpaca_shadow.db. It places
# NO orders and NEVER touches prod state/db. (The shadow scorer differs from
# prod, but the pipeline CODE PATH is shared — which is exactly what a code/pin
# bump verify must exercise; prod-scorer specifics are covered by
# check_conviction_admits.py + the bundle check.)
#
# Intended as the `promote_pin.py --verify-cmd` for BROAD bumps, and a
# `make doctor` deep check. Exit codes:
#   0 = clean decision produced
#   1 = crash / timeout / no-decision (would-not-trade) / isolation breach
#   2 = setup error (repo, subrepo env, or the shadow config itself unreadable)
#   3 = DEAD LEG (orch#1066): a panel-scoring artifact the shadow config
#       references is missing on disk, detected BEFORE the funnel runs; the
#       funnel is NOT run. This code says NOTHING about WHEN the leg died —
#       the preflight inspects only the current pinned assembly. Attribute
#       (pre-existing vs introduced by the bump) by comparing against the
#       previous pinned assembly (scripts/promote_pin.py keeps the backup
#       lock) before treating a red verify either way.
#   4 = STRUCTURAL BLOCK: the funnel ran but its log carries
#       `panel_scorer_load_failed` or the FunnelIntegrityAlert
#       `STRUCTURAL_BLOCK — engineering condition` marker — a structural
#       engineering failure in the shadow scorer chain, not a decision
#       outcome; whether it predates the bump is not established here.
#       Replaces the generic 1 ONLY when the run would otherwise have failed
#       (rc≠0 or no committed decision); a committed decision still exits 0
#       (with a WARN).
set -uo pipefail

REPO_DIR="${RENQUANT_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
PYTHON="${RENQUANT_PYTHON:-$REPO_DIR/.venv/bin/python}"
TIMEOUT_SEC="${RENQUANT_E2E_TIMEOUT_SEC:-1200}"
LOG="${RENQUANT_E2E_LOG:-/tmp/check_readonly_e2e.$$.log}"
SHADOW_CONFIG_NAME="strategy_config.shadow.json"

cd "$REPO_DIR" || { echo "SETUP: cannot cd $REPO_DIR"; exit 2; }
[ -f "$REPO_DIR/.env" ] && { set -a; # shellcheck disable=SC1091
  source "$REPO_DIR/.env"; set +a; }
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh" || { echo "SETUP: subrepo_env"; exit 2; }
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$(dirname "$REPO_DIR")")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

# (0) DEAD-LEG PREFLIGHT (orch#1066). The funnel below loads the shadow
# config's panel-scoring artifacts and, if one is missing, fail-closes with
# `panel_scorer_load_failed` → every buy candidate cleared → STRUCTURAL_BLOCK
# → this verify used to exit the generic 1, indistinguishable from a crash
# (measured 2026-08-25). So resolve those refs FIRST, exactly the way the
# pipeline does, and name the missing path with a distinct exit code before
# spending a funnel run on it. The preflight sees ONLY the current pinned
# assembly, so it cannot and does not say whether the leg was already dead
# before a bump — that attribution is the operator's, against the previous
# pin.
#
# Which config document: mirrors the runner selection — live-bridge (the
# default RQ_DAILY_RUNNER=multirepo) routes renquant_104 config reads to the
# PINNED strategy subrepo (renquant_orchestrator/live_bridge.py
# _with_pinned_strategy_config); the umbrella runner reads the strategy dir.
# Artifact refs resolve against the strategy dir in BOTH modes
# (live/runner.py sets config["_strategy_dir"] = backtesting/renquant_104).
STRATEGY_DIR="$REPO_DIR/backtesting/renquant_104"
if [ "${RQ_DAILY_RUNNER:-multirepo}" = "umbrella" ]; then
    SHADOW_CONFIG="$STRATEGY_DIR/$SHADOW_CONFIG_NAME"
else
    SHADOW_CONFIG="$SUBREPO_ROOT/renquant-strategy-104/configs/$SHADOW_CONFIG_NAME"
fi
"$PYTHON" - "$SHADOW_CONFIG" "$STRATEGY_DIR" "$REPO_DIR" <<'PY'
import json, sys
from pathlib import Path

cfg_path, strategy_dir, repo_root = (Path(a) for a in sys.argv[1:4])
try:
    cfg = json.loads(cfg_path.read_text())
except (OSError, ValueError) as exc:
    print(f"SETUP: dead-leg preflight cannot read shadow config {cfg_path}: {exc}")
    raise SystemExit(2)

ps = ((cfg.get("ranking") or {}).get("panel_scoring") or {})
if ps.get("enabled") is False:
    print(f"[readonly-e2e] preflight: panel_scoring disabled in {cfg_path}; no scorer artifact to check")
    raise SystemExit(0)


def strategy_only(ref):
    # LoadScorerTask._resolve_artifact_path / LoadGlobalCalibrationTask:
    # a relative ref is joined onto config["_strategy_dir"] and NOTHING else
    # (no repo-root fallback) — renquant_pipeline/kernel/panel_pipeline/
    # job_panel_scoring.py.
    p = Path(str(ref))
    return [p] if p.is_absolute() else [strategy_dir / p]


def resolver(ref):
    # blend components go through kernel.artifact_resolver.resolve_artifact:
    # absolute → strategy_dir/ref → repo_root/ref (renquant_pipeline/kernel/
    # panel_pipeline/blend_scorer.py _resolve_component_path).
    p = Path(str(ref))
    return [p] if p.is_absolute() else [strategy_dir / p, repo_root / p]


legs = []  # (config key, ref, candidate paths in resolution order)
if ps.get("artifact_path"):
    legs.append((f"ranking.panel_scoring.artifact_path (kind={ps.get('kind', 'xgb')})",
                 ps["artifact_path"], strategy_only(ps["artifact_path"])))
for i, comp in enumerate(ps.get("components") or []):
    if isinstance(comp, dict) and comp.get("artifact_path"):
        legs.append((f"ranking.panel_scoring.components[{i}].artifact_path "
                     f"(kind={comp.get('kind', 'panel')})",
                     comp["artifact_path"], resolver(comp["artifact_path"])))
gc = ps.get("global_calibration") or {}
if gc.get("enabled") and gc.get("artifact_path"):
    legs.append(("ranking.panel_scoring.global_calibration.artifact_path",
                 gc["artifact_path"], strategy_only(gc["artifact_path"])))

missing = [(key, ref, cands) for key, ref, cands in legs
           if not any(c.is_file() for c in cands)]
if not missing:
    print(f"[readonly-e2e] preflight: {len(legs)} panel-scoring artifact ref(s) in "
          f"{cfg_path.name} resolve to existing files")
    raise SystemExit(0)
for key, ref, cands in missing:
    print(f"READONLY_E2E: DEAD_LEG — {key} = {ref!r} is MISSING; tried "
          + ", ".join(str(c) for c in cands))
print(f"READONLY_E2E: DEAD_LEG — DEAD_LEG detected before the funnel in {cfg_path}; "
      "attribute by comparing against the previous pinned assembly "
      "(scripts/promote_pin.py keeps the backup lock) — see orch#1066")
print("READONLY_E2E: DEAD_LEG — funnel NOT run (exit 3); fixing the config is a "
      "separate reviewed decision (orch#1066 options a/b)")
raise SystemExit(3)
PY
PRE_RC=$?
case "$PRE_RC" in
    0) ;;
    3) exit 3 ;;
    *) echo "SETUP: dead-leg preflight failed (rc=$PRE_RC) on $SHADOW_CONFIG"; exit 2 ;;
esac

# Machine-checked ISOLATION contract: fingerprint the PROD state/db before the
# run; assert UNCHANGED after (a config/broker regression that wrote prod state
# must FAIL this guard, not pass). mtime+size, "MISSING" if absent.
PROD_DB="$REPO_DIR/data/runs.alpaca.db"
SHADOW_DB="$REPO_DIR/data/runs.alpaca_shadow.db"
PROD_STATE="$REPO_DIR/backtesting/renquant_104/live_state.alpaca.json"
fingerprint() { stat -f '%m-%z' "$1" 2>/dev/null || echo "MISSING"; }
PROD_DB_BEFORE="$(fingerprint "$PROD_DB")"
PROD_STATE_BEFORE="$(fingerprint "$PROD_STATE")"
SHADOW_DB_BEFORE="$(fingerprint "$SHADOW_DB")"

echo "[readonly-e2e] running isolated shadow e2e (timeout ${TIMEOUT_SEC}s) → $LOG"
RENQUANT_SUPPRESS_PREFLIGHT_NTFY=1 "$PYTHON" - "$REPO_DIR" "$TIMEOUT_SEC" > "$LOG" 2>&1 <<'PY'
import os, subprocess, sys
repo, timeout = sys.argv[1], float(sys.argv[2])
runner = ([sys.executable, "-m", "live.runner"]
          if os.environ.get("RQ_DAILY_RUNNER", "multirepo") == "umbrella"
          else [sys.executable, "-m", "renquant_orchestrator", "live-bridge", "--repo-dir", repo])
cmd = runner + ["--strategy", "renquant_104", "--broker", "readonly-alpaca",
                "--once", "--strategy-config-name", "strategy_config.shadow.json"]
try:
    raise SystemExit(subprocess.run(cmd, cwd=repo, timeout=timeout).returncode)
except subprocess.TimeoutExpired:
    print("E2E_TIMEOUT", flush=True); raise SystemExit(124)
PY
RC=$?
tail -3 "$LOG" 2>/dev/null

# (1) ISOLATION assertion (machine-checked): prod db/state must be UNCHANGED.
PROD_DB_AFTER="$(fingerprint "$PROD_DB")"
PROD_STATE_AFTER="$(fingerprint "$PROD_STATE")"
SHADOW_DB_AFTER="$(fingerprint "$SHADOW_DB")"
if [ "$PROD_DB_AFTER" != "$PROD_DB_BEFORE" ] || [ "$PROD_STATE_AFTER" != "$PROD_STATE_BEFORE" ]; then
    echo "READONLY_E2E: FAIL — ISOLATION BREACH (prod state changed: db $PROD_DB_BEFORE→$PROD_DB_AFTER, state $PROD_STATE_BEFORE→$PROD_STATE_AFTER)"
    exit 1
fi

# STRUCTURAL-BLOCK classification (orch#1066). Two markers, both emitted by
# renquant_pipeline: `panel_scorer_load_failed` (LoadScorerTask fail-closed →
# per-ticker block reason) and the FunnelIntegrityAlert line
# "STRUCTURAL_BLOCK — engineering condition suppressed buy capability".
# Either one means the run did not fail on a DECISION — it failed on the
# scorer chain — so a would-be exit 1 becomes exit 4 and the marker line is
# printed. Crash/timeout/no-decision WITHOUT these markers stays exit 1. The
# classification is about WHAT failed, not WHEN it started failing.
STRUCTURAL_LINE="$(grep -m1 -F -e 'panel_scorer_load_failed' -e 'STRUCTURAL_BLOCK — engineering condition' "$LOG" 2>/dev/null)"
fail() {  # $1 = the generic failure text
    if [ -n "$STRUCTURAL_LINE" ]; then
        echo "READONLY_E2E: STRUCTURAL — $1"
        echo "READONLY_E2E: STRUCTURAL — structural engineering failure in the shadow scorer chain (buy capability suppressed by an engineering condition, not a decision outcome; whether it predates the bump is not established here) (exit 4); marker: $STRUCTURAL_LINE"
        exit 4
    fi
    echo "READONLY_E2E: FAIL — $1"
    exit 1
}

# Exit code first.
if [ "$RC" -ne 0 ]; then
    fail "runner rc=$RC$( [ "$RC" = 124 ] && echo '/timeout' )"
fi

# (2) DECISION assertion: require DECISION-SPECIFIC evidence (a committed cycle
# decision / persisted gate verdicts), NOT mere pipeline-progress markers.
DECISION=$(grep -cE "ntfy sent:|SHADOW-DECISION|cycle decision|gate_verdicts: wrote|RunnerAdapter\.commit:" "$LOG" 2>/dev/null)
if [ "${DECISION:-0}" -lt 1 ]; then
    fail "ran but produced NO committed decision (silent)"
fi

# The run must actually have EXECUTED in isolation (shadow db advanced).
if [ "$SHADOW_DB_AFTER" = "$SHADOW_DB_BEFORE" ]; then
    echo "READONLY_E2E: WARN — shadow db unchanged; e2e may not have persisted (decision=$DECISION)"
fi
if [ -n "$STRUCTURAL_LINE" ]; then
    echo "READONLY_E2E: WARN — decision committed but the log carries a structural-block marker: $STRUCTURAL_LINE"
fi
echo "READONLY_E2E: OK — isolated readonly pipeline produced a committed decision ($DECISION marker(s)); prod state untouched"
exit 0
