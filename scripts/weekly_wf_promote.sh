#!/usr/bin/env bash
# weekly_wf_promote.sh — Weekly retrain + walk-forward gate + promote.
#
# 2026-05-09 audit FIX-C: this REPLACES daily auto-promote (which had
# RQ_ALLOW_NO_WF=1 bypass and let single-cut acceptance gates ship
# bad models). Trust boundary: every production promote now passes
# WF 3-cut + §5.2 sanity battery (shuffled-label + time-shift placebo).
#
# Schedule: Saturday 04:00 PT (NYC closed weekend buffer).
# Plist: scripts/launchd/com.renquant.weekly-wf-promote.plist
#
# Steps:
#   1. Smoke test (catch immediate breakage before 90 min train)
#   2. Retrain → produces panel-ltr.staging.alpha158_fund.json
#   3. Run scripts/run_wf_gate.py — 3-cut WF + §5.2 sanity. Historical
#      WF uses a manifest, so the gate first verifies the manifest artifacts
#      match the candidate recipe before stamping wf_gate_metadata.
#   4. _check_wf_gate inside promote() refuses to swap if metadata
#      missing or .passed=False — NO RQ_ALLOW_NO_WF override here
#   5. ntfy alert with verdict + Sharpe / IC numbers
#   6. Refresh dashboard so users see the new model state
#
# Failure modes:
#   - Smoke test fail → exit 1, no train
#   - Training fail → exit 1, prior artifact preserved
#   - WF gate fail → exit 1, prior artifact preserved (gate refuses promote)
#   - Promote fail (e.g. acceptance G1-G11 fail) → prior artifact preserved
set -uo pipefail

# Overridable only for the test harness (tests/test_weekly_wf_promote_
# snapshot_backstop.sh) — every default below is byte-identical to the
# prior hardcoded values, so production behavior is unchanged when these
# env vars are unset.
REPO_DIR="${RQ_WEEKLY_PROMOTE_REPO_DIR:-/Users/renhao/git/github/RenQuant}"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="${RQ_WEEKLY_PROMOTE_PYTHON:-$VENV_DIR/bin/python}"
LOG_DIR="$REPO_DIR/logs/weekly_wf_promote"
NTFY_TOPIC="${RQ_WEEKLY_PROMOTE_NTFY_TOPIC:-renquant}"
mkdir -p "$LOG_DIR"

DATE=$(date +%Y-%m-%d)
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/$DATE.log"

notify() {
    local title="$1" body="$2"
    # Test-only observability hook: when set, notify() ALSO appends "TITLE:
    # body" to this file so a test can assert exactly which notifications
    # fired without needing network access or touching the real ntfy topic.
    # No effect on production (unset by default).
    if [ -n "${RQ_WEEKLY_PROMOTE_NOTIFY_LOG:-}" ]; then
        printf '%s: %s\n' "$title" "$body" >> "$RQ_WEEKLY_PROMOTE_NOTIFY_LOG"
    fi
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

GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export RENQUANT_REPO_ROOT="$REPO_DIR"
WF_GATE_RUNNER="${RQ_WF_GATE_RUNNER:-multirepo}"
if [ "$WF_GATE_RUNNER" = "umbrella" ]; then
    echo "WARN: explicit RQ_WF_GATE_RUNNER=umbrella rollback selected."
    PROD_STRATEGY_CONFIG="$REPO_DIR/backtesting/renquant_104/strategy_config.json"
elif [ "$WF_GATE_RUNNER" = "multirepo" ]; then
    if ! PROD_STRATEGY_CONFIG="$(renquant_strategy_config "$SUBREPO_ROOT" strategy_config.json)"; then
        echo "ERROR: pinned renquant-strategy-104 strategy_config.json unavailable"
        exit 1
    fi
else
    echo "ERROR: unknown RQ_WF_GATE_RUNNER=$WF_GATE_RUNNER (expected multirepo or umbrella)"
    exit 2
fi
export RENQUANT_STRATEGY_CONFIG="$PROD_STRATEGY_CONFIG"

# Resolve the config whose declared panel_scoring.kind matches the GBDT
# candidate this job retrains. After the 06-23 lineup reversal XGB moved to
# strategy_config.json (primary) — the filename no longer implies the kind.
# Scan both configs and pick the one that declares kind=xgb.
#
# Codex review (PR #452 round 1): in umbrella/rollback mode, resolution only
# checked backtesting/renquant_104/<name> — the umbrella WORKING-COPY config.
# render_strategy_104_snapshot.py's own header documents that this exact path
# is NOT what production actually consumes and has gone stale across a prior
# lineup swap (the 2026-06-23 XGB re-promotion): the authoritative source is
# the PIN-ALIGNED runtime checkout at
# .subrepo_runtime/repos/renquant-strategy-104/configs/ (kept in sync with
# subrepos.lock.json). Check the pin-aligned location FIRST — it is what the
# real daily run and the umbrella snapshot-backstop harness populate — falling
# back to the umbrella working copy only if the pin-aligned runtime tree is
# not present at all (e.g. pre-bootstrap).
_find_gbdt_config() {
    for cfg_name in strategy_config.json strategy_config.shadow.json; do
        local candidate pinned_path workingcopy_path
        pinned_path="$REPO_DIR/.subrepo_runtime/repos/renquant-strategy-104/configs/$cfg_name"
        workingcopy_path="$REPO_DIR/backtesting/renquant_104/$cfg_name"
        if [ "$WF_GATE_RUNNER" = "umbrella" ]; then
            candidates=("$pinned_path" "$workingcopy_path")
        else
            local multirepo_path
            multirepo_path="$(renquant_strategy_config "$SUBREPO_ROOT" "$cfg_name" 2>/dev/null)" || multirepo_path=""
            candidates=("$multirepo_path" "$pinned_path" "$workingcopy_path")
        fi
        for candidate in "${candidates[@]}"; do
            [ -n "$candidate" ] || continue
            [ -f "$candidate" ] || continue
            local kind
            kind=$("$PYTHON" -c "
import json, sys
c = json.load(open(sys.argv[1]))
print(c.get('ranking',{}).get('panel_scoring',{}).get('kind',''))
" "$candidate" 2>/dev/null)
            if [ "$kind" = "xgb" ] || [ "$kind" = "panel_ltr_xgboost" ]; then
                echo "$candidate"
                return 0
            fi
        done
    done
    return 1
}
if ! GBDT_PROD_CONFIG="$(_find_gbdt_config)"; then
    echo "ERROR: no strategy config declares kind=xgb; cannot resolve GBDT reference"
    exit 2
fi
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-backtesting renquant-pipeline renquant-common renquant-base-data renquant-artifacts renquant-model renquant-strategy-104 renquant-execution):${PYTHONPATH:-}"

run_wf_gate() {
    if [ "$WF_GATE_RUNNER" = "umbrella" ]; then
        "$PYTHON" scripts/run_wf_gate.py "$@"
        return $?
    fi
    if "$PYTHON" - <<'PY' >/dev/null 2>&1
import renquant_backtesting.wf_gate.runner  # noqa: F401
PY
    then
        "$PYTHON" - <<'PY' >&2
import renquant_backtesting.wf_gate.runner as m
print(f"renquant_backtesting.wf_gate.runner={m.__file__}")
PY
        "$PYTHON" -m renquant_backtesting.wf_gate "$@"
        return $?
    fi
    echo "ERROR: renquant_backtesting.wf_gate unavailable; set RQ_WF_GATE_RUNNER=umbrella for explicit rollback."
    return 1
}

exec >> "$LOG" 2>&1
echo "=== weekly_wf_promote started at $(date) ==="

# Lock — prevent concurrent runs (a 90-min job can stack if the user
# triggers a manual rerun before the previous finishes).
LOCK_FILE="${RQ_WEEKLY_PROMOTE_LOCK_FILE:-/tmp/renquant_104_weekly_wf.lock}"
if ! ( set -C; echo $$ > "$LOCK_FILE" ) 2>/dev/null; then
    EXISTING=$(cat "$LOCK_FILE" 2>/dev/null || echo "?")
    if [ "$EXISTING" != "?" ] && ! kill -0 "$EXISTING" 2>/dev/null; then
        echo "Stale lock (PID $EXISTING dead) — clearing."
        rm -f "$LOCK_FILE"
        echo $$ > "$LOCK_FILE"
    else
        echo "Another weekly run is active (PID=$EXISTING) — skipping."
        notify "RenQuant 104 SKIP" "Weekly WF promote skipped — already running"
        exit 0
    fi
fi
trap "rm -f '$LOCK_FILE'" EXIT

# Saturate this host per CLAUDE.md §5.10; do not carry stale laptop-specific
# constants across Apple Silicon upgrades.
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

# ── Step 1: Smoke test ────────────────────────────────────────────────────
echo "--- Step 1: Pre-flight smoke test ---"
# P0 fix (2026-06-07): the production scorer is a torch (PatchTST) model;
# torch.load segfaults under OMP_NUM_THREADS>=4 (set to 14 above for XGBoost).
# Run the torch smoke test single-threaded; the XGBoost retrain below keeps 14.
if ! OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" scripts/smoke_test_model.py --strategy renquant_104; then
    echo "Smoke test FAILED — aborting weekly promote (no train)."
    notify "RenQuant 104 WEEKLY-ABORT" "Pre-flight smoke test failed; weekly promote skipped. Check $LOG"
    exit 1
fi

# ── Step 2: BACKUP current production before final promote ────────────────
# The retrain below writes to unique staging paths. Active production should
# remain unchanged until the strict WF gate has passed; the backup is retained
# as an explicit rollback target for the final active swap.
# 2026-05-11 sim/prod isolation: prod artifacts moved to artifacts/prod/.
# Before this fix, ACTIVE_ART pointed at the now-empty flat path, so the
# `[ -f "$ACTIVE_ART" ]` backup guard always failed silently and rollback
# never copied anything → §5.5 rehearsal invariant decoration-only.
ART_DIR="$REPO_DIR/backtesting/renquant_104/artifacts/prod"
ACTIVE_ART="$ART_DIR/panel-ltr.alpha158_fund.json"
ACTIVE_CAL="$ART_DIR/panel-rank-calibration.json"
ROLLBACK_ART="$ART_DIR/panel-ltr.alpha158_fund.weekly_rollback_$DATE.json"
ROLLBACK_CAL="$ART_DIR/panel-rank-calibration.weekly_rollback_$DATE.json"
STAGING_ART="$ART_DIR/panel-ltr.alpha158_fund.weekly_${RUN_ID}.staging.json"
STAGING_CAL="$ART_DIR/panel-rank-calibration.weekly_${RUN_ID}.staging.json"

echo "--- Step 2: Backup prior production artifacts (rollback rehearsal) ---"
if [ -f "$ACTIVE_ART" ]; then
    cp "$ACTIVE_ART" "$ROLLBACK_ART"
    echo "Backup model: $ROLLBACK_ART"
fi
if [ -f "$ACTIVE_CAL" ]; then
    cp "$ACTIVE_CAL" "$ROLLBACK_CAL"
    echo "Backup calibrator: $ROLLBACK_CAL"
fi

# ── Step 3: Retrain on the alpha158+fund+sentiment pipeline ─────────────
# Note: this also REFITS the calibrator (per daily_retrain_alpha158_fund.py
# step 4). It must write to staging paths; active prod is swapped only after
# the strict WF gate passes.
echo "--- Step 3: Retrain panel-LTR + calibrator to staging ---"
echo "Staging model: $STAGING_ART"
echo "Staging calibrator: $STAGING_CAL"
# Newly-delisted names whose removal hasn't reached the versioned universe
# inventory yet (retrain_alpha158_fund --exclude-tickers is the designed
# bridge; a single stale name blocks the whole retrain at the 0.0 freshness
# tolerance). IAC: bars ceased 2026-05-12 (2026-07-17 anomaly-retrain
# failure); REMOVE from this default once the inventory prunes it.
RETRAIN_EXCLUDE_TICKERS="${RENQUANT_RETRAIN_EXCLUDE_TICKERS:-IAC}"
if ! bash scripts/daily_retrain_alpha158_fund.sh \
    --xgb-artifact-out "$STAGING_ART" \
    --calibrator-out "$STAGING_CAL" \
    --no-drop-sentiment \
    --exclude-tickers "$RETRAIN_EXCLUDE_TICKERS"; then
    echo "Training FAILED — production artifact unchanged."
    notify "RenQuant 104 WEEKLY-FAIL" "Training failed; production model unchanged. Check $LOG"
    exit 1
fi
if [ ! -f "$STAGING_ART" ] || [ ! -f "$STAGING_CAL" ]; then
    echo "Training FAILED — staging artifacts missing."
    notify "RenQuant 104 WEEKLY-FAIL" "Training finished but staging artifact/calibrator missing. Check $LOG"
    exit 1
fi
echo "Training pipeline finished at $(date)"

# ── Step 3.5: Stamp config fingerprints onto the WF manifest artifacts ────
# 2026-05-27: the calibrated_causal manifest's per-cut scorers were created by
# an ad-hoc path that bypassed train_production_model.py's stamp_fingerprint,
# so config_fingerprint=None. The panel scorer's strict assert_consistent then
# fail-closed EVERY bar (panel_scorer_config_mismatch → "Cleared N buy
# candidate(s)") → zero trades → the gate could never pass ANY model. Stamp the
# manifest before the gate runs. Idempotent: already-stamped artifacts are a
# no-op. This is also the fail-fast recipe validation point: if the manifest
# cuts do not match the freshly trained staging artifact, abort before spending
# time in the WF gate that would fail closed anyway.
# 2026-06-16: the weekly GBDT gate uses the regenerated prod-recipe manifest
# that matches the no-drop-sentiment staging candidate. Keep Step 3.5 stamping
# and Step 4 gate evaluation on the same manifest source of truth.
WF_MANIFEST="artifacts/sim/walkforward_manifest_gbdt_prod_recipe_v2.calibrated.json"
echo "--- Step 3.5: Stamp WF manifest fingerprints ($WF_MANIFEST) ---"
if ! "$PYTHON" scripts/stamp_walkforward_fingerprints.py \
    --manifest "$WF_MANIFEST" \
    --fingerprint-config "$GBDT_PROD_CONFIG" \
    --reference-artifact "$STAGING_ART"; then
    echo "WF manifest stamping/recipe validation FAILED — production unchanged."
    notify "RenQuant 104 WEEKLY-FAIL" \
        "WF manifest recipe validation failed against staged model. Production unchanged. Check $LOG."
    exit 1
fi

# ── Step 4: Run WF gate (3-cut WF + §5.2 sanity battery) ──────────────────
echo "--- Step 4: Walk-forward gate (3-cut + sanity) ---"
if ! RENQUANT_STRATEGY_CONFIG="$GBDT_PROD_CONFIG" run_wf_gate \
    --artifact "$STAGING_ART" \
    --strategy-config strategy_config.sim_wl200_gbdt_prod_recipe_calibrated.json \
    --derive-config-from-prod \
    --strict \
    --jobs 3; then
    echo "WF gate REJECTED staged model — consulting the RFC#210 freshness fallback (backtesting#101/#102)."
    # ── Step 4b: RFC#210 freshness fallback (operator P0, 2026-08-03) ─────
    # The gate criterion is UNTOUCHED. When the gate rejects AND the served
    # model is >28d stale AND the candidate is recent with a non-negative
    # stamped genuine_ic (ordinal/sign only) AND no downward ratchet, the
    # ALREADY-DECIDED freshness governance promotes the candidate, stamped
    # promotion_basis=freshness_fallback_rfc210 so downstream always knows
    # governance-served from gate-passed. Fail-closed on every malformed
    # input, and fail-closed here too: if the pinned runtime predates
    # backtesting#102 the module is absent, and this block REFUSES loudly
    # instead of pretending to have consulted anything.
    FALLBACK_JSON="$LOG_DIR/${RUN_ID}.fallback_verdict.json"
    FALLBACK_PROMOTED=0
    if "$PYTHON" -c "import renquant_backtesting.wf_gate.freshness_fallback" 2>/dev/null; then
        if "$PYTHON" -m renquant_backtesting.wf_gate.freshness_fallback \
            --prod "$ACTIVE_ART" --staging "$STAGING_ART" --stamp \
            > "$FALLBACK_JSON" 2>&1; then
            echo "RFC#210 fallback verdict: FALLBACK_PROMOTE — $FALLBACK_JSON"
            FALLBACK_PROMOTED=1
        else
            echo "RFC#210 fallback verdict: REFUSE — production unchanged. See $FALLBACK_JSON"
            sed -n '1,40p' "$FALLBACK_JSON" || true
        fi
    else
        echo "RFC#210 fallback UNAVAILABLE under the current backtesting pin (predates #102) — treating as REFUSE; advance the pin to arm the fallback."
    fi
    if [ "$FALLBACK_PROMOTED" != "1" ]; then
        echo "WF gate REJECTED staged model — production unchanged."
        notify "RenQuant 104 WEEKLY-REJECT" \
            "Walk-forward gate rejected the staged model. Production unchanged. Check $LOG."
        exit 1
    fi
    # Fallback-specific pair promote: same incoming/replace dance as Step 5,
    # but the license is the promotion_basis STAMP (passed=False by design —
    # Step 5's passed-is-True check must not run on this path).
    if ! "$PYTHON" - <<PY
from pathlib import Path
import json
import os
import shutil
import sys

try:
    from renquant_backtesting.forensics.model_acceptance import promote
except Exception:
    sys.path.insert(0, "backtesting/renquant_104")
    from kernel.model_acceptance import promote

model_src = Path("$STAGING_ART")
model_dst = Path("$ACTIVE_ART")
cal_src = Path("$STAGING_CAL")
cal_dst = Path("$ACTIVE_CAL")

model = json.loads(model_src.read_text())
meta = model.get("metadata") or {}
if meta.get("promotion_basis") != "freshness_fallback_rfc210":
    raise SystemExit(
        "staged artifact lacks the freshness_fallback_rfc210 stamp — the "
        "fallback CLI must have stamped it before this promote may run")
gate = meta.get("wf_gate_metadata") or {}
if gate.get("passed") is not False:
    raise SystemExit(
        f"fallback promote requires an explicitly REJECTED candidate "
        f"(stamped passed=False); got {gate.get('passed')!r}")

if not cal_src.exists():
    raise SystemExit(f"missing staging calibrator: {cal_src}")
cal_payload = json.loads(cal_src.read_text())
if not isinstance(cal_payload, dict):
    raise SystemExit(f"staging calibrator is not a JSON object: {cal_src}")

cal_incoming = cal_dst.with_suffix(".incoming.json")
shutil.copy2(cal_src, cal_incoming)
try:
    promote(model_src, model_dst)
    os.replace(cal_incoming, cal_dst)
except Exception:
    try:
        cal_incoming.unlink()
    except FileNotFoundError:
        pass
    raise
print(f"FALLBACK-promoted {model_src.name} -> {model_dst.name} (rfc210 stamp verified)")
print(f"FALLBACK-promoted {cal_src.name} -> {cal_dst.name}")
PY
    then
        echo "Fallback promote FAILED — production may still be on prior model or .previous rollback target."
        notify "RenQuant 104 WEEKLY-FAIL" \
            "RFC#210 fallback promote failed after gate reject. Check $LOG before trading."
        exit 1
    fi
fi

# ── Step 5: Inspect gate metadata + promote staged pair ───────────────────
# On the RFC#210 fallback path the pair was ALREADY promoted in Step 4b under
# the promotion_basis stamp (passed=False by design), so the gate-passed
# promote below must not run — its passed-is-True license check would refuse
# correctly but noisily. Steps 6/7 (dashboard + snapshot backstop) run for
# BOTH paths.
FALLBACK_PROMOTED="${FALLBACK_PROMOTED:-0}"
GATE_SUMMARY=$("$PYTHON" -c "
import json
m = json.load(open('$STAGING_ART'))
gate = m.get('wf_gate_metadata') or m.get('metadata', {}).get('wf_gate_metadata') or {}
sharpe = gate.get('wf_3cut_sharpe_mean')
apy    = gate.get('wf_3cut_apy_mean')
shuf   = gate.get('sanity_shuffled_ic')
plac   = gate.get('sanity_placebo_ic')
parts = []
if sharpe is not None: parts.append(f'WF Sharpe {sharpe:+.2f}')
if apy is not None:    parts.append(f'APY {apy:+.2f}%')
if shuf is not None:   parts.append(f'shuf_IC {shuf:+.4f}')
if plac is not None:   parts.append(f'placebo_IC {plac:+.4f}')
print('  '.join(parts) if parts else '(no metadata)')
" 2>/dev/null || echo "(metadata parse failed)")
echo "Gate metadata: $GATE_SUMMARY"

if [ "$FALLBACK_PROMOTED" = "1" ]; then
    echo "Skipping gate-passed promote — Step 4b already fallback-promoted the pair."
elif ! "$PYTHON" - <<PY
from pathlib import Path
import json
import os
import shutil
import sys

try:
    from renquant_backtesting.forensics.model_acceptance import promote
    print("promote=renquant_backtesting.forensics.model_acceptance", file=sys.stderr)
except Exception as exc:  # noqa: BLE001
    if os.environ.get("RQ_WF_GATE_RUNNER", "multirepo") != "umbrella":
        raise SystemExit(
            "ERROR: renquant_backtesting.forensics.model_acceptance unavailable; "
            "set RQ_WF_GATE_RUNNER=umbrella for explicit rollback."
        ) from exc
    print(
        f"WARN: explicit umbrella rollback selected; using kernel.model_acceptance ({exc})",
        file=sys.stderr,
    )
    sys.path.insert(0, "backtesting/renquant_104")
    from kernel.model_acceptance import promote

model_src = Path("$STAGING_ART")
model_dst = Path("$ACTIVE_ART")
cal_src = Path("$STAGING_CAL")
cal_dst = Path("$ACTIVE_CAL")

model = json.loads(model_src.read_text())
gate = model.get("wf_gate_metadata") or model.get("metadata", {}).get("wf_gate_metadata") or {}
if gate.get("passed") is not True:
    raise SystemExit(f"staged artifact has no passing wf_gate_metadata: {gate}")

if not cal_src.exists():
    raise SystemExit(f"missing staging calibrator: {cal_src}")
cal_payload = json.loads(cal_src.read_text())
if not isinstance(cal_payload, dict):
    raise SystemExit(f"staging calibrator is not a JSON object: {cal_src}")

cal_incoming = cal_dst.with_suffix(".incoming.json")
shutil.copy2(cal_src, cal_incoming)
try:
    promote(model_src, model_dst)
    os.replace(cal_incoming, cal_dst)
except Exception:
    try:
        cal_incoming.unlink()
    except FileNotFoundError:
        pass
    raise
print(f"Promoted {model_src.name} -> {model_dst.name} via kernel.model_acceptance.promote")
print(f"Promoted {cal_src.name} -> {cal_dst.name}")
PY
then
    echo "Promote FAILED — production may still be on prior model or .previous rollback target."
    notify "RenQuant 104 WEEKLY-FAIL" \
        "Promotion step failed after WF gate. Check $LOG before trading."
    exit 1
fi

# ── Step 6: Refresh dashboard ─────────────────────────────────────────────
"$PYTHON" "$REPO_DIR/scripts/build_dashboard.py" --broker alpaca \
    --out "$REPO_DIR/doc/dashboard.md" 2>&1 | tail -5 \
    || echo "dashboard refresh failed (non-fatal)"

# ── Step 7: strategy-104 snapshot freshness backstop (M9/A6 round 4) ──────
# Codex review (PR #432): the promotion above just changed the active
# artifact/calibrator, exactly the state doc/arch/strategy-104-snapshot.md
# declares — but this script is the REAL model-promotion path and, unlike
# promote_pin.py's bump/revert, had no synchronous check of its own; only
# the NEXT daily system_doctor run would eventually notice drift. Reuse
# promote_pin.py's check_snapshot_freshness (scratch-rendered, diff-preview,
# never auto-commits, never touches the promotion that just succeeded) and
# fail THIS run non-zero before reporting overall success.
echo "--- Step 7: strategy-104 snapshot freshness backstop ---"
if ! "$PYTHON" - <<PY
import sys
sys.path.insert(0, "$REPO_DIR/scripts")
from promote_pin import check_snapshot_freshness
fresh, msg = check_snapshot_freshness("$PYTHON", repo=__import__("pathlib").Path("$REPO_DIR"))
print(msg)
raise SystemExit(0 if fresh else 1)
PY
then
    echo "Snapshot freshness backstop FAILED — model promotion above already"
    echo "completed and is NOT being reverted for this reason alone; only"
    echo "doc/arch/strategy-104-snapshot.md needs a follow-up: run"
    echo "'make snapshot' from $REPO_DIR, review the diff, and commit it."
    notify "RenQuant 104 WEEKLY-PROMOTE — SNAPSHOT STALE" \
        "Model promoted ($GATE_SUMMARY) but doc/arch/strategy-104-snapshot.md is now stale. Run 'make snapshot' and commit. Check $LOG."
    exit 1
fi

if [ "$FALLBACK_PROMOTED" = "1" ]; then
    # Emitter line: the silent-refusal sentinel's weekly-wf-promote lane
    # counts this as an ACTION (paired orchestrator PR extends action_re +
    # the emitter contract to this literal).
    echo "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at $(date) — $GATE_SUMMARY ==="
    notify "RenQuant 104 WEEKLY-FALLBACK-PROMOTE" \
        "Gate rejected all candidates; the RFC#210 freshness fallback promoted the staged model (stamped promotion_basis=freshness_fallback_rfc210). $GATE_SUMMARY Check $LOG."
else
    echo "=== weekly_wf_promote PASSED at $(date) — $GATE_SUMMARY ==="
    notify "RenQuant 104 WEEKLY-PROMOTE ✓" \
        "Walk-forward gate passed. New model promoted to production. $GATE_SUMMARY"
fi
