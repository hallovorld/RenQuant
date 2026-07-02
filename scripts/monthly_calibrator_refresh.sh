#!/usr/bin/env bash
# monthly_calibrator_refresh.sh — Re-fit the global panel calibrator monthly.
#
# 2026-05-09 audit FIX-C: separates calibrator refresh (fast, low-risk)
# from model retrain (slow, high-risk, weekly). The calibrator's isotonic
# knot positions can drift as the score distribution shifts (regime
# changes, etc.) even when the underlying XGBoost model is unchanged.
# Monthly refit keeps calibrated probabilities + expected returns aligned
# with current score distribution without touching the model.
#
# Schedule: 1st of every month, 03:00 PT.
# Plist: scripts/launchd/com.renquant.monthly-calibrator-refresh.plist
#
# Steps:
#   1. Smoke test (ensure model still loads — abort if broken)
#   2. Run renquant_model_gbdt.fit_calibrator_alpha158_fund against the
#      active production model — writes to a UNIQUE STAGING PATH
#      ($STAGING_CAL, never PROD_CAL directly)
#   3. Test scorer + new calibrator produces sane (P, E[R]) on synthetic
#      input — abort if calibrator collapsed (n_unique_prob_y < floor) or
#      pool_ic regressed vs baseline. Evaluated against $STAGING_CAL.
#   3b. 2026-07-01 fix: verify the calibrator's stamped fingerprint actually
#      BINDS to what the live runtime will compute for the active scorer
#      (runtime-authoritative `PanelScorer.load` + `_any_fingerprints_match`
#      from renquant-pipeline — the same contract
#      `_assert_calibrator_matches_scorer` enforces at runtime). A calibrator
#      can pass Step 3's quality gate and still be bound to the wrong scorer;
#      this is a separate, additional gate, not a replacement. Also
#      evaluated against $STAGING_CAL.
#   3c. 2026-07-01 review round 2 fix: ATOMIC PUBLISH. Only after 3/3b both
#      pass, scripts/monthly_calibrator_atomic_swap.py re-verifies the
#      staged candidate's digest (TOCTOU guard) and atomically renames it
#      onto PROD_CAL, writing an acceptance receipt that binds the checked
#      scorer identity + candidate digest. Any failure anywhere in 2/3/3b/3c
#      quarantines staging and leaves PROD_CAL byte-identical to how the
#      run started (including the no-prior-calibrator case — PROD_CAL
#      simply continues to not exist).
#   4. ntfy summary — n knots, score → P(out) range
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="$VENV_DIR/bin/python"
LOG_DIR="$REPO_DIR/logs/monthly_calibrator"
NTFY_TOPIC="renquant"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    if command -v terminal-notifier &>/dev/null; then
        terminal-notifier -title "$title" -message "$body" -sound Glass 2>/dev/null || true
    fi
    curl -s -H "Title: $title" -d "$body" "https://ntfy.sh/$NTFY_TOPIC" >/dev/null 2>&1 || true
}

CRED_FILE="$REPO_DIR/.env"
if [ -f "$CRED_FILE" ]; then
    set -a
    source "$CRED_FILE"
    set +a
fi

exec >> "$LOG" 2>&1
echo "=== monthly_calibrator_refresh started at $(date) ==="

LOCK_FILE="/tmp/renquant_104_monthly_cal.lock"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING" != "?" ] && ! kill -0 "$EXISTING" 2>/dev/null; then
        rm -f "$LOCK_FILE"; echo $$ > "$LOCK_FILE"
    else
        echo "Another monthly run is active (PID=$EXISTING) — skipping."
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

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

cd "$REPO_DIR"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
# 2026-07-01: renquant-pipeline added so Step 3b can import the
# runtime-authoritative PanelScorer loader + fingerprint-match helpers
# (kernel/panel_pipeline/{panel_scorer,job_panel_scoring}.py) — the same
# modules daily_104.sh / weekly_wf_promote.sh already put on PYTHONPATH.
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-model renquant-common renquant-base-data renquant-artifacts renquant-pipeline):${PYTHONPATH:-}"
MONTHLY_CALIBRATOR_STRICT=0
if renquant_strict_enabled RQ_MONTHLY_CALIBRATOR_STRICT; then
    MONTHLY_CALIBRATOR_STRICT=1
fi
if ! PROD_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
    echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable; monthly calibrator fails closed"
    notify "RenQuant 104 MONTHLY-ABORT" "Pinned strategy config unavailable; calibrator NOT refreshed. Check $LOG"
    exit 1
fi
echo "Strategy config: $PROD_STRATEGY_CONFIG"
echo "Multirepo fail-closed: enabled (strict=$MONTHLY_CALIBRATOR_STRICT)"

# ── Step 1: Smoke test — abort if model broken ───────────────────────────
echo "--- Step 1: Pre-flight smoke test ---"
if ! "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test FAILED — aborting monthly calibrator refresh."
    notify "RenQuant 104 MONTHLY-ABORT" "Pre-flight smoke test failed; calibrator NOT refreshed. Check $LOG"
    exit 1
fi

# ── Step 2: Re-fit global calibrator (to a STAGING path — never PROD_CAL) ──
# 2026-05-11 sim/prod isolation: explicit --out so the calibrator lands
# under artifacts/prod/ (without --out the script derives a flat-path
# orphan from the panel artifact's stem that prod runner won't read).
#
# 2026-05-17 ACCEPTANCE GATE — backup BEFORE refit + IC regression check
# AFTER. Same bug class as today's Sunday-sweep corruption (NGB
# val_IC=-0.0165 → prod silently). Pre-fix this script had no rollback
# target if the new calibrator regressed.
#
# 2026-07-01 REVIEW FIX ROUND 2 (PR #425 CHANGES_REQUESTED, Codex): fitting
# straight to PROD_CAL meant the LIVE RUNTIME could read an unvalidated (or
# scorer-mismatched) calibrator during the fit-to-validation window —
# rollback-after-exposure is not the same as never publishing an
# unvalidated artifact. And a first-ever fit (no prior calibrator) that
# failed validation left the REJECTED artifact sitting at PROD_CAL, so it
# stayed published in the no-baseline case. Fix: fit_calibrator() now
# writes to a UNIQUE STAGING PATH in the SAME directory as PROD_CAL
# ($STAGING_CAL, same-filesystem — required for Step 3c's atomic rename).
# Every gate below (staged-calibrator smoke check, pool_ic/non-collapse,
# and Step 3b's binding check) runs against staging, never PROD_CAL.
# PROD_CAL is only ever touched once, atomically
# (scripts/monthly_calibrator_atomic_swap.py publish → os.replace, a
# single same-filesystem rename syscall), in Step 3c — AFTER every gate
# has passed. Any failure before Step 3c quarantines staging
# (scripts/monthly_calibrator_atomic_swap.py quarantine) and leaves
# PROD_CAL byte-identical to how the run started, including the
# no-baseline case (PROD_CAL simply continues to not exist).
echo "--- Step 2: Re-fit global calibrator (staging) ---"
PROD_SCORER=$("$PYTHON" - "$PROD_STRATEGY_CONFIG" "$REPO_DIR/backtesting/renquant_104" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
strategy_dir = Path(sys.argv[2])
rel = cfg["ranking"]["panel_scoring"]["artifact_path"]
path = Path(rel)
print(path if path.is_absolute() else strategy_dir / path)
PY
)
PROD_CAL=$("$PYTHON" - "$PROD_STRATEGY_CONFIG" "$REPO_DIR/backtesting/renquant_104" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())
strategy_dir = Path(sys.argv[2])
rel = cfg["ranking"]["panel_scoring"]["global_calibration"]["artifact_path"]
path = Path(rel)
print(path if path.is_absolute() else strategy_dir / path)
PY
)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)_$$
ROLLBACK_CAL="$REPO_DIR/backtesting/renquant_104/artifacts/prod/panel-rank-calibration.monthly_rollback_$DATE.json"
# Unique staging path, SAME DIRECTORY as PROD_CAL — os.replace/rename is
# only atomic within one filesystem; same-directory guarantees that.
STAGING_CAL="${PROD_CAL}.staging-${RUN_ID}.json"
RECEIPT="${PROD_CAL}.accepted_receipt-${RUN_ID}.json"

# Archival snapshot of the CURRENT production calibrator, for operator
# reference only — NOT used for automated rollback. Under the staging
# design PROD_CAL is never modified until Step 3c, so there is nothing to
# roll back to on a gate failure; this is purely a dated audit copy (same
# convention as weekly_wf_promote.sh's Step 2 backup).
BASELINE_POOL_IC="None"
BASELINE_N_UNIQUE=0
if [ -f "$PROD_CAL" ]; then
    # ATOMIC: write to .tmp then mv (POSIX cp is two syscalls;
    # SIGKILL mid-cp → half-written rollback). Audit P0-16.
    cp "$PROD_CAL" "$ROLLBACK_CAL.tmp" && mv "$ROLLBACK_CAL.tmp" "$ROLLBACK_CAL"
    echo "Archival backup (reference only, not used for auto-rollback): $ROLLBACK_CAL"
    BASELINE_POOL_IC=$("$PYTHON" -c "
import json
m = json.load(open('$PROD_CAL'))
print(m.get('metadata', {}).get('pool_ic', 'None'))
" 2>/dev/null || echo "None")
    BASELINE_N_UNIQUE=$("$PYTHON" -c "
import json
m = json.load(open('$PROD_CAL'))
print(m.get('metadata', {}).get('n_unique_prob_y', 0))
" 2>/dev/null || echo "0")
    echo "Baseline: pool_ic=$BASELINE_POOL_IC  n_unique_prob_y=$BASELINE_N_UNIQUE"
else
    echo "No prior calibrator at $PROD_CAL — first-ever fit (no regression baseline)"
fi

fit_calibrator() {
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_model_gbdt.fit_calibrator_alpha158_fund  # noqa: F401
PY
    then
        "$PYTHON" -m renquant_model_gbdt.fit_calibrator_alpha158_fund \
            --data-dir "$REPO_DIR/data" \
            --scorer-artifact "$PROD_SCORER" \
            --out "$STAGING_CAL"
    else
        echo "ERROR: renquant_model_gbdt.fit_calibrator_alpha158_fund unavailable; monthly calibrator fails closed"
        return 1
    fi
}

if ! fit_calibrator; then
    echo "Calibrator fit FAILED — production calibrator untouched (staged-only write)."
    "$PYTHON" scripts/monthly_calibrator_atomic_swap.py quarantine \
        --staging "$STAGING_CAL" --reason "fit_calibrator_alpha158_fund failed" \
        >/dev/null 2>&1 || true
    notify "RenQuant 104 MONTHLY-FAIL" "Calibrator fit failed; production calibrator unchanged. Check $LOG"
    exit 1
fi
if [ ! -f "$STAGING_CAL" ]; then
    echo "fit_calibrator reported success but staging artifact missing: $STAGING_CAL"
    notify "RenQuant 104 MONTHLY-FAIL" "Calibrator fit reported success but staging artifact missing; production calibrator unchanged. Check $LOG"
    exit 1
fi
echo "Staged candidate: $STAGING_CAL"
CANDIDATE_SHA256=$("$PYTHON" scripts/monthly_calibrator_atomic_swap.py sha256 --path "$STAGING_CAL")
echo "Candidate sha256: $CANDIDATE_SHA256"

# ── Step 3: Validate the STAGED calibrator — smoke + non-collapse + IC-regression ─
# A load/score smoke check plus 2 hard quality checks, ALL evaluated
# against $STAGING_CAL — PROD_CAL is not read or written by this step:
#   H0 (2026-07-01 round 2): staged calibrator loads and maps score→(P,E[R])
#       sanely for two distinct synthetic scores (same diversity invariant
#       smoke_test_model.py's scorer check enforces — a collapsed
#       calibrator would map every score to the same output). This
#       REPLACES the old "rerun scripts/smoke_test_model.py" check: that
#       script resolves the calibrator path from strategy_config.json —
#       i.e. PROD_CAL — which is no longer written until Step 3c, so
#       rerunning it here would just be an exact duplicate of Step 1 (same
#       scorer, same still-unchanged PROD_CAL): it would validate nothing
#       about the new candidate. H0 exercises the CANDIDATE directly.
#   H2a (existing): n_unique_prob_y >= 10 (non-collapse, was display-only
#       pre-fix)
#   H2b (existing): pool_ic did not drop > 0.02 vs baseline (regression
#       guard)
# Any failure → quarantine staging (scripts/monthly_calibrator_atomic_swap.py
# quarantine) + ntfy + exit non-zero. PROD_CAL is untouched — it was never
# written to begin with, so there is nothing to roll back.
# References:
#   - Diebold-Mariano 1995 (J. Bus. Econ. Stat.) "Comparing Predictive
#     Accuracy" — framework for forecast-accuracy testing. 0.02 IC drop
#     threshold ≈ 2σ given typical pool_ic std ~0.01; heuristic, not
#     formal DM-test (CLAUDE.md §5.12 — exploratory tune-via-A/B).
#   - n_unique_prob_y ≥ 10: internal "G2 calibrator non-collapse"
#     invariant (kernel/model_acceptance.py:DEFAULT_GATES) — calibrator
#     with fewer than 10 unique buckets degenerates to constant scores
#     → ranking collapse; was display-only pre-fix.
echo "--- Step 3: Validate staged calibrator ---"
STAGING_SMOKE_VERDICT=$("$PYTHON" - "$STAGING_CAL" "$REPO_DIR/backtesting/renquant_104" <<'PY' 2>&1
import sys
from pathlib import Path

cal_path = Path(sys.argv[1])
strategy_dir = Path(sys.argv[2])
sys.path.insert(0, str(strategy_dir))

try:
    import numpy as np
    from training_panel.global_calibrator import GlobalPanelCalibration
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: import — {type(exc).__name__}: {exc}")
    sys.exit(1)

try:
    cal = GlobalPanelCalibration.load(cal_path)
except Exception as exc:  # noqa: BLE001
    print(f"FAIL: staged calibrator load — {type(exc).__name__}: {exc}")
    sys.exit(1)

probs = []
for score in (-1.0, 1.0):
    try:
        p = cal.calibrate_probability(score)
        er = cal.expected_return(score)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: staged calibrator map(score={score}) — {type(exc).__name__}: {exc}")
        sys.exit(1)
    if not (np.isfinite(p) and np.isfinite(er)):
        print(f"FAIL: staged calibrator produced non-finite (score={score} prob={p} er={er})")
        sys.exit(1)
    if p < 0.0 or p > 1.0:
        print(f"FAIL: staged calibrator probability out of [0,1] (score={score} got {p})")
        sys.exit(1)
    probs.append(p)

if abs(probs[0] - probs[1]) < 1e-12:
    print(f"FAIL: staged calibrator produced IDENTICAL probability for two different scores (P={probs[0]}) — collapsed")
    sys.exit(1)

print(f"OK staged calibrator smoke: P(-1.0)={probs[0]:.4f} P(1.0)={probs[1]:.4f}")
PY
)
STAGING_SMOKE_RC=$?
echo "$STAGING_SMOKE_VERDICT"
if [ $STAGING_SMOKE_RC -ne 0 ]; then
    echo "Post-fit smoke test FAILED on staged calibrator — production calibrator untouched."
    "$PYTHON" scripts/monthly_calibrator_atomic_swap.py quarantine \
        --staging "$STAGING_CAL" --reason "post-fit smoke test failed: $STAGING_SMOKE_VERDICT" \
        >/dev/null 2>&1 || true
    notify "RenQuant 104 MONTHLY-FAIL" "Post-fit smoke test failed on staged calibrator; production calibrator unchanged. Check $LOG"
    exit 1
fi

# IC-regression-vs-baseline check + non-collapse gate (staged candidate)
GATE_VERDICT=$("$PYTHON" - "$STAGING_CAL" "$BASELINE_POOL_IC" "$BASELINE_N_UNIQUE" <<'PY'
import json, sys, math
staging_cal = sys.argv[1]
base_ic_str = sys.argv[2]
base_n_uniq_str = sys.argv[3]
m = json.load(open(staging_cal))
md = m.get("metadata", {}) or {}
new_ic = md.get("pool_ic")
new_n_uniq = md.get("n_unique_prob_y", 0)

fails = []
# H2a non-collapse hard guard
try:
    n_uniq = int(new_n_uniq)
    if n_uniq < 10:
        fails.append(f"n_unique_prob_y={n_uniq} < 10 (collapsed)")
except (TypeError, ValueError):
    fails.append(f"n_unique_prob_y={new_n_uniq!r} not int")

# H2b IC regression vs baseline (only if baseline existed)
if base_ic_str != "None":
    try:
        base_ic = float(base_ic_str)
        if new_ic is None or not math.isfinite(float(new_ic)):
            fails.append(f"new pool_ic={new_ic!r} not finite")
        else:
            new_ic = float(new_ic)
            drop = base_ic - new_ic
            if drop > 0.02:
                fails.append(f"pool_ic dropped {base_ic:+.4f} → {new_ic:+.4f} (Δ {-drop:+.4f} > 2pp)")
    except (TypeError, ValueError) as e:
        fails.append(f"baseline pool_ic parse: {e}")

if fails:
    print("FAIL: " + "; ".join(fails))
    sys.exit(1)
print(f"OK pool_ic={new_ic} n_unique={new_n_uniq}")
PY
)
GATE_RC=$?
if [ $GATE_RC -ne 0 ]; then
    echo "ACCEPTANCE GATE FAILED: $GATE_VERDICT"
    echo "Quarantining staged calibrator — production calibrator was NEVER modified."
    "$PYTHON" scripts/monthly_calibrator_atomic_swap.py quarantine \
        --staging "$STAGING_CAL" --reason "acceptance gate: $GATE_VERDICT" \
        >/dev/null 2>&1 || true
    notify "RenQuant 104 MONTHLY-REJECT" "Calibrator REJECTED ($GATE_VERDICT); production calibrator was never modified."
    exit 1
fi
echo "Gate: $GATE_VERDICT"

CAL_INFO=$("$PYTHON" -c "
import json
from pathlib import Path
cfg = json.loads(Path('$PROD_STRATEGY_CONFIG').read_text())
cal_rel = cfg['ranking']['panel_scoring']['global_calibration']['artifact_path']
m = json.loads(Path('$STAGING_CAL').read_text())
n_knots_p = len(m.get('probability', {}).get('x', []))
n_knots_e = len(m.get('expected_return', {}).get('x', []))
md = m.get('metadata', {})
n_uniq = md.get('n_unique_prob_y', '—')
pool_ic = md.get('pool_ic', '—')
print(f'knots: prob={n_knots_p} er={n_knots_e}  n_unique_prob_y={n_uniq}  pool_ic={pool_ic}  (config calibration_path={cal_rel})')
" 2>/dev/null || echo "calibrator info unavailable")
echo "Staged calibrator state: $CAL_INFO"

# ── Step 3b: Scorer/calibrator BINDING check (defense-in-depth) ──────────
# 2026-07-01 incident: this calibrator passed Step 3's pool_ic/n_unique
# quality gate above, then fail-closed the live daily-full at runtime,
# because `_assert_calibrator_matches_scorer` (job_panel_scoring.py)
# rejected it — fit_calibrator_alpha158_fund.py's stamped
# scorer_model_content_fingerprint used a DIFFERENT model_content_sha256
# field-set than the runtime's own check, so a calibrator fit here could
# never bind to the live scorer, by construction. Step 3 never exercised
# that contract, so the mismatch shipped silently.
#
# Root cause is being fixed at the source: renquant-common#18 (canonical
# model_content_sha256) + renquant-pipeline#155 + renquant-model#40 (both
# consumers import the shared function instead of hand-copying it). This
# step is defense-in-depth on top of that fix: it runs the SAME
# runtime-authoritative loader (PanelScorer.load) and match logic
# (_any_fingerprints_match / _fingerprint_values, imported — not
# reimplemented — from renquant_pipeline.kernel.panel_pipeline) so ANY
# future re-divergence, or any other cause of a scorer/calibrator
# mismatch, is caught HERE before publish — not after it blocks live
# trading. Keeps working regardless of which of those three PRs lands
# first (see scripts/verify_calibrator_scorer_binding.py docstring).
#
# This is a DIFFERENT failure mode than Step 3's pool_ic/n_unique_prob_y
# quality-regression gate: a calibrator can have excellent pool_ic and
# still be bound to the wrong scorer — that is exactly what happened
# 2026-07-01. FAILS CLOSED (treated as a gate failure, never a silent
# skip) if the runtime-authoritative loader isn't importable — a check
# that exists-but-skips-silently is exactly the failure mode that let
# today's incident through.
echo "--- Step 3b: Validate scorer/calibrator binding (runtime-authoritative) ---"
BINDING_VERDICT=$("$PYTHON" scripts/verify_calibrator_scorer_binding.py \
    --scorer "$PROD_SCORER" --calibrator "$STAGING_CAL" --json 2>&1)
BINDING_RC=$?
echo "$BINDING_VERDICT"
if [ $BINDING_RC -ne 0 ]; then
    if [ $BINDING_RC -eq 2 ]; then
        BINDING_REASON="binding check could not run (runtime-authoritative loader unavailable — failed CLOSED, not skipped)"
    else
        BINDING_REASON="calibrator/scorer BINDING MISMATCH (a DIFFERENT failure than Step 3's pool_ic/n_unique quality gate, which already passed above — this calibrator is bound to the wrong scorer, not merely lower-quality)"
    fi
    echo "SCORER/CALIBRATOR BINDING GATE FAILED: $BINDING_REASON"
    echo "Quarantining staged calibrator — production calibrator was NEVER modified (staging-only fit)."
    "$PYTHON" scripts/monthly_calibrator_atomic_swap.py quarantine \
        --staging "$STAGING_CAL" --reason "$BINDING_REASON" \
        --scorer-path "$PROD_SCORER" \
        >/dev/null 2>&1 || true
    notify "RenQuant 104 MONTHLY-REJECT" "Calibrator REJECTED: $BINDING_REASON. Production calibrator was never modified."
    exit 1
fi
echo "Binding gate: OK — calibrator fingerprint matches the runtime-computed active scorer fingerprint."

# ── Step 3c: Atomic publish — staging → PROD_CAL, only after every gate passed ──
# 2026-07-01 REVIEW FIX ROUND 2: `atomic_publish` re-verifies
# $CANDIDATE_SHA256 (captured right after Step 2's fit) against the
# staging file's CURRENT bytes immediately before the filesystem swap —
# if anything changed the staging file between the gate checks above and
# this point (TOCTOU), the swap refuses to run and PROD_CAL stays
# untouched. The receipt binds the CHECKED scorer identity/fingerprints
# (from Step 3b's binding verdict) + the exact candidate digest, so what
# lands at PROD_CAL is provably the artifact Step 3/3b evaluated, not
# something that could have been swapped in between.
echo "--- Step 3c: Atomic publish (staging → production) ---"
SCORER_FPS_JSON=$("$PYTHON" -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    d = {}
print(json.dumps(d.get('active_fingerprints', [])))
" "$BINDING_VERDICT" 2>/dev/null || echo "[]")
CAL_FPS_JSON=$("$PYTHON" -c "
import json, sys
try:
    d = json.loads(sys.argv[1])
except Exception:
    d = {}
print(json.dumps(d.get('calibrator_fingerprints', [])))
" "$BINDING_VERDICT" 2>/dev/null || echo "[]")
NEW_POOL_IC=$("$PYTHON" -c "
import json
m = json.load(open('$STAGING_CAL'))
print(m.get('metadata', {}).get('pool_ic', 'None'))
" 2>/dev/null || echo "None")
NEW_N_UNIQUE=$("$PYTHON" -c "
import json
m = json.load(open('$STAGING_CAL'))
print(m.get('metadata', {}).get('n_unique_prob_y', 0))
" 2>/dev/null || echo "0")

if ! "$PYTHON" scripts/monthly_calibrator_atomic_swap.py publish \
    --staging "$STAGING_CAL" --prod "$PROD_CAL" \
    --expected-sha256 "$CANDIDATE_SHA256" \
    --receipt-out "$RECEIPT" \
    --scorer-path "$PROD_SCORER" \
    --scorer-fingerprints-json "$SCORER_FPS_JSON" \
    --calibrator-fingerprints-json "$CAL_FPS_JSON" \
    --pool-ic "$NEW_POOL_IC" --n-unique "$NEW_N_UNIQUE"; then
    echo "ATOMIC PUBLISH FAILED — production calibrator was NOT modified (publish verifies-before-writing)."
    "$PYTHON" scripts/monthly_calibrator_atomic_swap.py quarantine \
        --staging "$STAGING_CAL" --reason "atomic publish failed (see log)" \
        >/dev/null 2>&1 || true
    notify "RenQuant 104 MONTHLY-CRITICAL" "All gates passed but atomic publish failed; production calibrator unchanged. Operator action REQUIRED — check $LOG."
    exit 1
fi
echo "Published: $STAGING_CAL -> $PROD_CAL"
echo "Acceptance receipt: $RECEIPT"

# ── Step 4: Refresh dashboard so monthly cadence is visible ──────────────
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 | tail -5 \
    || echo "dashboard refresh failed (non-fatal)"

echo "=== monthly_calibrator_refresh PASSED at $(date) — $CAL_INFO ==="
notify "RenQuant 104 MONTHLY-CAL ✓" "Calibrator refreshed: $CAL_INFO"
