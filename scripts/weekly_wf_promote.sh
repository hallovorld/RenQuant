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
# orch#799: derive an xgb-shaped production reference from a kind=blend prod
# config's component[0] (the production xgb scorer). Writes a temp config and
# echoes its path; on ANY failure it echoes nothing and returns non-zero so the
# caller fails closed (never fabricates a reference). The derivation and its
# adversarial kind-check (the component artifact must DECLARE xgb) live in
# renquant_backtesting.wf_gate.wf_config_builder.derive_xgb_reference_from_blend
# (unit-tested there); this is only the wiring. Depends on that subrepo being
# pinned with the function — until then the import fails and we fall through to
# the exit-2 fail-closed, which is the pre-existing safe behaviour.
_derive_xgb_ref_from_blend() {
    local blend_cfg="$1" out err bt_pythonpath strat_dir
    out="$(mktemp "${TMPDIR:-/tmp}/gbdt_prod_ref_derived.XXXXXX")" || return 1
    err="$(mktemp "${TMPDIR:-/tmp}/gbdt_prod_ref_err.XXXXXX")" || { rm -f "$out"; return 1; }
    bt_pythonpath="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" \
        renquant-backtesting renquant-common renquant-base-data \
        renquant-artifacts renquant-model renquant-strategy-104 2>/dev/null)"
    # Component artifact paths (e.g. artifacts/prod/panel-ltr.alpha158_fund.json)
    # resolve under the umbrella strategy-104 artifact tree — where they live on
    # disk for the adversarial kind-check to read.
    strat_dir="$REPO_DIR/backtesting/renquant_104"
    if PYTHONPATH="${bt_pythonpath}:${PYTHONPATH:-}" "$PYTHON" -c '
import json, sys
from pathlib import Path
from renquant_backtesting.wf_gate.wf_config_builder import (
    derive_xgb_reference_from_blend,
)
blend = json.load(open(sys.argv[1]))
derived = derive_xgb_reference_from_blend(blend, strategy_dir=Path(sys.argv[3]))
with open(sys.argv[2], "w") as fh:
    json.dump(derived, fh)
'  "$blend_cfg" "$out" "$strat_dir" 2>"$err"; then
        echo "$out"
        rm -f "$err"
        return 0
    fi
    # DO NOT SWALLOW THIS (2026-08-21). The derivation used to run under
    # `2>/dev/null`, so on the one path where it matters — it failed — the
    # reason was destroyed and the caller printed a generic "no kind-matched
    # reference" that says nothing about WHY. Diagnosing the 2026-08-16 failure
    # required re-running the call by hand to see the traceback. Surface it on
    # stderr so the next reader has it in the log.
    echo "derive_xgb_reference_from_blend FAILED for $blend_cfg:" >&2
    sed 's/^/    /' "$err" >&2 2>/dev/null || true
    rm -f "$err" "$out"
    return 1
}

_find_gbdt_config() {
    for cfg_name in strategy_config.json strategy_config.shadow.json; do
        local candidate pinned_path workingcopy_path
        pinned_path="$REPO_DIR/.subrepo_runtime/repos/renquant-strategy-104/configs/$cfg_name"
        workingcopy_path="$REPO_DIR/backtesting/renquant_104/$cfg_name"
        # 2026-08-04 (orch#799, MEASURED): the umbrella WORKING COPY must
        # never serve as the production reference. After the full-book z-blend
        # switch made the pinned primary kind=blend, this search fell through
        # to backtesting/renquant_104/strategy_config.shadow.json — the A8
        # registry's known-diverged working copy (kind=xgb, hf_patchtst-era
        # semantics) — and the gate silently simulated a strategy nobody runs
        # (same model, Sharpe 0.602 -> 0.052, greedy path -> joint QP).
        # The reviewed production surface is the PINNED config; a missing
        # kind match is a REAL state that must fail closed, not be papered
        # over with a stale file.
        # Round 2 (codex on #580): the multirepo path resolved through
        # renquant_subrepo_root defaults to the SIBLING DEVELOPER CHECKOUT
        # when no assembly override is set — a locally-edited checkout could
        # recreate exactly the mismatch this fix eliminates. The ONLY
        # candidate in BOTH runner modes is now the lock-aligned runtime
        # config under .subrepo_runtime (what the daily run actually loads).
        candidates=("$pinned_path")
        # (workingcopy_path and the multirepo/sibling path are intentionally
        # EXCLUDED — named here so the exclusions are visible at the point of
        # decision rather than silently absent.)
        : "${workingcopy_path:?}"
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
            # orch#799: after the 2026-08-04 z-blend switch the pinned primary is
            # kind=blend, so no top-level xgb reference exists. component[0] IS the
            # production xgb scorer — derive an xgb-shaped reference from it so the
            # gate compares the xgb candidate against the live xgb leg (keeping the
            # blend), instead of fail-closing. Fail-safe: a failed derivation falls
            # through to the exit-2 below.
            if [ "$kind" = "blend" ]; then
                local derived_ref
                if derived_ref="$(_derive_xgb_ref_from_blend "$candidate")" \
                    && [ -n "$derived_ref" ] && [ -f "$derived_ref" ]; then
                    echo "$derived_ref"
                    return 0
                fi
            fi
        done
    done
    return 1
}
if ! GBDT_PROD_CONFIG="$(_find_gbdt_config)"; then
    echo "ERROR: no PINNED strategy config declares kind=xgb — cannot resolve a"
    echo "       kind-matched GBDT production reference (orch#799)."
    echo "       This is the EXPECTED state while the pinned primary is a blend:"
    echo "       an xgb candidate has no same-kind production reference, and the"
    echo "       umbrella working copy is NOT an acceptable substitute (A8: known"
    echo "       diverged). The gate refuses rather than simulate a phantom config."
    echo "       NOT a pending decision (corrected 2026-08-21). The orch#799"
    echo "       'blend-prod reference rule' WAS decided and implemented:"
    echo "       _derive_xgb_ref_from_blend (this script, 2f85e0d 2026-08-16) +"
    echo "       derive_xgb_reference_from_blend (renquant-backtesting #112,"
    echo "       2026-08-17). It works — verified by hand and by the 2026-08-20"
    echo "       run, which used a derived reference and reached a verdict."
    echo "       Reaching THIS line therefore means the derivation was tried and"
    echo "       FAILED; its stderr is printed above. Read that, not this."
    notify "RenQuant 104 WEEKLY-BLOCKED" \
        "WF gate cannot run: no pinned kind-matched prod reference for the xgb candidate (prod is a blend). See orch#799. Production unchanged; RFC#210 freshness governance unaffected."
    exit 2
fi
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-backtesting renquant-pipeline renquant-common renquant-base-data renquant-artifacts renquant-model renquant-strategy-104 renquant-execution renquant-orchestrator):${PYTHONPATH:-}"

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

# ── OPERATOR MODE: --promote-staged <RUN_ID> (2026-08-04) ────────────────
# Promote an ALREADY-TRAINED staged pair through the RFC#210 fallback
# WITHOUT retraining. Born from the 2026-08-04 manual promotion (operator
# order "现在就promote到104和105！"): the reviewed mechanism existed only
# inside the scheduled retrain->gate->promote chain, so the operator path
# had to replicate the pair-swap by hand under a grant. This mode makes it
# first-class: the SAME dual-contract arming check, the SAME fallback CLI
# (decide gate: 5 checks incl. no-downward-ratchet), the SAME shared
# pair-promote script, and the SAME sentinel-visible emitter line. The PIT
# freshness guard is not weakened - no training happens here.
if [ "${1:-}" = "--promote-staged" ]; then
    PS_RUN_ID="${2:-}"
    if [ -z "$PS_RUN_ID" ]; then
        echo "usage: $0 --promote-staged <RUN_ID>  (e.g. 20260802T170002Z)"
        exit 2
    fi
    # [codex on #566] The run id is interpolated into three paths; without
    # format validation a traversal-like value escapes ART_DIR/LOG_DIR and
    # turns a promotion command into arbitrary JSON selection/writes.
    # Canonical form ONLY, checked before ANY path is constructed.
    case "$PS_RUN_ID" in
        [0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]T[0-9][0-9][0-9][0-9][0-9][0-9]Z) ;;
        *)
            echo "promote-staged REFUSED: RUN_ID must match YYYYMMDDTHHMMSSZ exactly; got '$PS_RUN_ID'"
            exit 2
            ;;
    esac
    STAGING_ART="$ART_DIR/panel-ltr.alpha158_fund.weekly_${PS_RUN_ID}.staging.json"
    STAGING_CAL="$ART_DIR/panel-rank-calibration.weekly_${PS_RUN_ID}.staging.json"
    if [ ! -f "$STAGING_ART" ] || [ ! -f "$STAGING_CAL" ]; then
        echo "promote-staged REFUSED: staged pair not found for RUN_ID=$PS_RUN_ID"
        exit 1
    fi
    ORCH_RUN_DIR="${RQ_ORCH_RUN_DIR:-/Users/renhao/git/github/renquant-orchestrator-run}"
    EMITTER_CONTRACT="$ORCH_RUN_DIR/ops/renquant104/emitter_contract.json"
    if ! grep -q "weekly_wf_promote FALLBACK-PROMOTED" "$EMITTER_CONTRACT" 2>/dev/null; then
        echo "promote-staged REFUSED: sentinel emitter contract missing the FALLBACK-PROMOTED action line at $EMITTER_CONTRACT"
        exit 1
    fi
    if ! "$PYTHON" -c "import renquant_backtesting.wf_gate.freshness_fallback" 2>/dev/null; then
        echo "promote-staged REFUSED: freshness_fallback not importable under the pinned runtime"
        exit 1
    fi
    # RFC#210 A4-T1 (orch#1110 / bt#128): the ONLY path that consumes the
    # candidate exception is the orchestrator's identify -> atomic consume ->
    # stamp wrapper. The direct `freshness_fallback --stamp` CLI now exits 1
    # for the candidate (it has no ledger), so it must not be called here.
    # The wrapper tees its JSON verdict to $LOG_DIR/<RUN_ID>.a4t1_promote.json
    # and exits 0 iff PROMOTED; anything else leaves production unchanged.
    A4T1_WRAPPER="$SUBREPO_ROOT/repos/renquant-orchestrator/ops/renquant104/a4t1_promote_staged.sh"
    if [ ! -x "$A4T1_WRAPPER" ]; then
        echo "promote-staged REFUSED: a4t1 wrapper missing or not executable under the pinned runtime: $A4T1_WRAPPER"
        exit 1
    fi
    PS_VERDICT="$LOG_DIR/${PS_RUN_ID}.a4t1_promote.json"
    if ! PYTHON="$PYTHON" LOG_DIR="$LOG_DIR" "$A4T1_WRAPPER" "$PS_RUN_ID" "$ACTIVE_ART" "$STAGING_ART"; then
        echo "promote-staged: RFC#210 A4-T1 verdict REFUSED — production unchanged. See $PS_VERDICT"
        sed -n '1,40p' "$PS_VERDICT" 2>/dev/null || true
        exit 1
    fi
    if ! "$PYTHON" scripts/fallback_pair_promote.py \
        "$STAGING_ART" "$ACTIVE_ART" "$STAGING_CAL" "$ACTIVE_CAL"
    then
        echo "promote-staged: pair promote FAILED — check .previous rollback target."
        notify "RenQuant 104 PROMOTE-STAGED-FAIL" "RFC#210 promote-staged failed after verdict PROMOTE. Check $LOG before trading."
        exit 1
    fi
    GATE_SUMMARY=$("$PYTHON" -c "
import json
from pathlib import Path
m = json.loads(Path('$ACTIVE_ART').read_text())
meta = m.get('metadata') or {}
g = meta.get('wf_gate_metadata') or {}
print(f\"trained={m.get('trained_date')} basis={meta.get('promotion_basis')} gate_passed={g.get('passed')} genuine_ic={g.get('sanity_placebo_genuine_ic')}\")" 2>/dev/null || echo "summary unavailable")
    echo "=== weekly_wf_promote FALLBACK-PROMOTED (rfc210) at $(date) — $GATE_SUMMARY ==="
    notify "RenQuant 104 FALLBACK-PROMOTED" "promote-staged $PS_RUN_ID promoted under RFC#210. $GATE_SUMMARY"
    exit 0
fi

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
# AVB: delisted 2026-08-17 (Equity Residential merger; SEC 8-K/Form 25), last
# bar 2026-08-24, vetoed the 2026-08-30 promote — INTERIM until the reviewed
# exclusion registry (renquant-orchestrator#1096) is pinned, then drop both.
RETRAIN_EXCLUDE_TICKERS="${RENQUANT_RETRAIN_EXCLUDE_TICKERS:-IAC,AVB}"
# Off-schedule MANUAL runs during market hours (2026-08-04 incident): the
# wall-clock-derived expected session is only the PREVIOUS day after close,
# so a mid-session run sees today's partial bars as "future" and the
# freshness guard fail-closes the retrain. These envs thread the
# retrainer's OWN deterministic-replay pins (--expected-session / --as-of,
# built for exactly this "freshness must not depend on when the job runs"
# case) through the wrapper. Empty = scheduled behavior, byte-identical.
# They PIN the reference to a real completed session; they never loosen
# tolerances — the guard still measures every ticker against it.
RETRAIN_EXPECTED_SESSION="${RENQUANT_RETRAIN_EXPECTED_SESSION:-}"
RETRAIN_AS_OF="${RENQUANT_RETRAIN_AS_OF:-}"
# Argument ARRAY, not ${var:+...} inline expansion: the inline form yields
# ONE shell word ("--as-of <value>" fused), so the retrainer never sees a
# standalone option (codex on #564). Array elements stay separate argv
# entries; the ${arr[@]+...} guard keeps empty arrays safe under set -u on
# macOS bash 3.2.
RETRAIN_PIN_ARGS=()
if [ -n "$RETRAIN_EXPECTED_SESSION" ]; then
    RETRAIN_PIN_ARGS+=(--expected-session "$RETRAIN_EXPECTED_SESSION")
fi
if [ -n "$RETRAIN_AS_OF" ]; then
    RETRAIN_PIN_ARGS+=(--as-of "$RETRAIN_AS_OF")
fi
if ! bash scripts/daily_retrain_alpha158_fund.sh \
    --xgb-artifact-out "$STAGING_ART" \
    --calibrator-out "$STAGING_CAL" \
    --no-drop-sentiment \
    --exclude-tickers "$RETRAIN_EXCLUDE_TICKERS" \
    ${RETRAIN_PIN_ARGS[@]+"${RETRAIN_PIN_ARGS[@]}"}; then
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
    # ARMING CONTRACT [codex on #559 round 1, second demand]: BOTH sides must
    # be present — the provider (freshness_fallback importable under the
    # pinned runtime) AND the consumer (the orchestrator run checkout's
    # emitter contract carrying the FALLBACK-PROMOTED action line, orch#774).
    # A provider armed without the consumer would let the promotion be
    # recorded as a silent-refusal incident; refuse loudly instead.
    # RQ_ORCH_RUN_DIR is overridable for the test harness only.
    ORCH_RUN_DIR="${RQ_ORCH_RUN_DIR:-/Users/renhao/git/github/renquant-orchestrator-run}"
    EMITTER_CONTRACT="$ORCH_RUN_DIR/ops/renquant104/emitter_contract.json"
    if ! grep -q "weekly_wf_promote FALLBACK-PROMOTED" "$EMITTER_CONTRACT" 2>/dev/null; then
        echo "RFC#210 fallback DISARMED: sentinel emitter contract at $EMITTER_CONTRACT does not carry the FALLBACK-PROMOTED action line (land + sync orchestrator#774 first) — treating as REFUSE."
    elif "$PYTHON" -c "import renquant_backtesting.wf_gate.freshness_fallback" 2>/dev/null; then
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
        # Operator directive 2026-08-04: a reject while the SERVED model is
        # fresh is the healthy steady state of RFC#210 governance (the recipe
        # is chronically placebo-dominated; staleness is bounded at 28d), and
        # it must not be reported with a failure tone or a failure exit.
        # scripts/reject_notify_disposition.py proves that shape from the
        # verdict JSON; anything unproven (missing/malformed verdict, refusal
        # on any other check, prod actually stale, disarmed/unavailable paths
        # that never wrote a verdict) still alarms and exits 1 — fail closed
        # toward attention, never toward silence.
        REJECT_DISPOSITION=$("$PYTHON" scripts/reject_notify_disposition.py "$FALLBACK_JSON" 2>/dev/null \
            || echo "ALARM|disposition helper failed")
        case "$REJECT_DISPOSITION" in
            "CALM_FRESH|"*)
                REJECT_AGE=$(printf '%s' "$REJECT_DISPOSITION" | cut -d'|' -f2)
                REJECT_TRAINED=$(printf '%s' "$REJECT_DISPOSITION" | cut -d'|' -f3)
                echo "Reject disposition: prod FRESH (trained $REJECT_TRAINED, ${REJECT_AGE}d <= 28d SLA) — governance nominal, calm notify, exit 0."
                notify "RenQuant 104 WEEKLY-REJECT (prod fresh — no action)" \
                    "WF gate rejected the candidate (expected for this recipe). Served model is FRESH: trained $REJECT_TRAINED, ${REJECT_AGE}d old of a 28d SLA. Governance nominal; nothing to do."
                exit 0
                ;;
            *)
                echo "Reject disposition: ${REJECT_DISPOSITION} — alarm notify, exit 1."
                notify "RenQuant 104 WEEKLY-REJECT" \
                    "Walk-forward gate rejected the staged model. Production unchanged. ${REJECT_DISPOSITION#ALARM|}. Check $LOG."
                exit 1
                ;;
        esac
    fi
    # Fallback-specific pair promote: same incoming/replace dance as Step 5,
    # but the license is the promotion_basis STAMP (passed=False by design —
    # Step 5's passed-is-True check must not run on this path).
    #
    # Codex review (PR #559, round 1 BLOCKER): calling the shared
    # renquant_backtesting.forensics.model_acceptance.promote() here always
    # raised, because that helper's internal _check_wf_gate() unconditionally
    # refuses any staging artifact stamped passed=False — exactly what this
    # path's license intentionally is. The stamp check two lines above is
    # THIS path's gate; do the atomic file swap directly instead of routing
    # through the gate-passed helper.
    if ! "$PYTHON" scripts/fallback_pair_promote.py \
        "$STAGING_ART" "$ACTIVE_ART" "$STAGING_CAL" "$ACTIVE_CAL"
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
# On the FALLBACK_PROMOTED path, Step 4b's _swap_into_active() already
# unlinked $STAGING_ART (same atomic-swap dance as model_acceptance.promote()).
# Read back from $ACTIVE_ART in that case — it now holds the exact bytes
# copied from staging, so the gate metadata is identical. Reading a gone
# staging path here previously produced "(metadata parse failed)" for every
# fallback promotion (codex review, PR #559 round 2).
GATE_SUMMARY=$("$PYTHON" -c "
import json
from pathlib import Path
staging = Path('$STAGING_ART')
src = staging if staging.exists() else Path('$ACTIVE_ART')
m = json.load(open(src))
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
    # Codex review (PR #559 round 2): on the gate-passed path this stays a
    # hard failure (pinned by test_weekly_wf_promote_snapshot_backstop.py).
    # On the FALLBACK_PROMOTED path it must not be — production was already
    # mutated under the stamped promotion_basis license, and swallowing the
    # FALLBACK-PROMOTED action/notification below behind this exit would
    # leave the orchestrator's silent-refusal sentinel unable to see that an
    # action occurred (it only observes the log/notify contract), i.e. an
    # unobserved production change is worse than a noisy stale-snapshot
    # alert. Fall through instead of exiting so that literal always fires.
    if [ "$FALLBACK_PROMOTED" != "1" ]; then
        exit 1
    fi
    echo "Continuing: the fallback promotion above is licensed independently of this backstop; the FALLBACK-PROMOTED action/notification below must still fire."
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
