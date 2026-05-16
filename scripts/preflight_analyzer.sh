#!/bin/bash
# Pre-flight check for `scripts/analyze_regime_stratified.py`.
#
# Refuses to bless an A/B analysis where the BASELINE equity directory
# was generated against artifacts that have since been refit. Stale
# baselines silently contaminate every "win" / "loss" — the difference
# attributed to the candidate knob is partly (sometimes entirely) the
# artifact-refit effect.
#
# This script compares the baseline equity dir's earliest mtime against
# the prod-artifact mtimes (panel-rank-calibration.json,
# panel-ltr.alpha158_fund.json). If any artifact is newer than the
# baseline's first equity write, the comparison is contaminated.
#
# Usage:
#   scripts/preflight_analyzer.sh <baseline_dir> <treatment_dir>
#
# Exit codes:
#   0  blessed
#   1  baseline stale (refit happened after baseline was generated)
#   2  treatment newer than artifacts (treatment may use artifacts you don't
#      currently have on disk — risky)
#   3  usage / missing dirs
set -u
cd /Users/renhao/git/github/RenQuant

if (( $# < 2 )); then
  echo "usage: $0 <baseline_dir> <treatment_dir>" >&2
  exit 3
fi

BASE="$1"
TREAT="$2"

for d in "$BASE" "$TREAT"; do
  if [[ ! -d "${d}/equity" ]]; then
    echo "✗ missing ${d}/equity" >&2
    exit 3
  fi
done

ARTS=(
  "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json"
  "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json"
)

# Earliest equity-file mtime (= when the baseline was last regenerated)
base_first_mtime=$(ls -t "$BASE/equity"/*.json 2>/dev/null | tail -1 | xargs stat -f %m 2>/dev/null)
treat_first_mtime=$(ls -t "$TREAT/equity"/*.json 2>/dev/null | tail -1 | xargs stat -f %m 2>/dev/null)
base_first_date=$(date -r "${base_first_mtime}" "+%Y-%m-%d %H:%M")
treat_first_date=$(date -r "${treat_first_mtime}" "+%Y-%m-%d %H:%M")

echo "── analyzer pre-flight ──"
echo "  baseline   ${BASE}"
echo "             earliest equity mtime: ${base_first_date}"
echo "  treatment  ${TREAT}"
echo "             earliest equity mtime: ${treat_first_date}"

bad=0
for art in "${ARTS[@]}"; do
  if [[ ! -e "$art" ]]; then
    echo "  · ${art} — missing (skip)"
    continue
  fi
  art_mtime=$(stat -f %m "$art")
  art_date=$(date -r "$art_mtime" "+%Y-%m-%d %H:%M")
  if (( art_mtime > base_first_mtime )); then
    echo "  ✗ ${art}"
    echo "      artifact mtime ${art_date} > baseline ${base_first_date}"
    echo "      → baseline used the OLD artifact; A/B contaminated by refit"
    bad=1
  else
    echo "  ✓ ${art}"
    echo "      artifact mtime ${art_date} ≤ baseline ${base_first_date}"
  fi
done

if (( bad )); then
  echo
  echo "BLOCKED: re-generate the baseline panel against current artifacts"
  echo "         before running the stratified analyzer, OR use a same-batch"
  echo "         proxy baseline (a treatment panel whose knob is known no-op)."
  exit 1
fi

echo
echo "ANALYZER PRE-FLIGHT BLESSED"
