#!/bin/bash
# Pre-flight checklist for a new sim panel config.
#
# Refuses to bless a candidate config unless ALL hold:
#   1. STATIC — validator's path map confirms the knob writes to a config
#      path the kernel actually reads (no DEAD_PATH).
#   2. SMOKE — running both candidate and baseline on a 1-month window
#      produces non-bit-identical equity (knob actually fires in some
#      observable way, on this date range, given current artifacts).
#
# Without both, the panel risks producing baseline-equivalent equity
# (5h of compute proving nothing — the failure pattern from 2026-05-15).
#
# Usage:
#   scripts/preflight_panel.sh <candidate_config_name>
#   scripts/preflight_panel.sh <candidate_config_name> <baseline_config_name>
#
# Exit codes:
#   0  blessed (safe to launch full panel)
#   1  static path check failed (knob path doesn't reach kernel)
#   2  smoke failed (equity bit-identical → knob is a no-op in practice)
#   3  usage error
set -u
cd /Users/renhao/git/github/RenQuant
source .venv/bin/activate >/dev/null 2>&1

if (( $# < 1 )); then
  echo "usage: $0 <candidate_config_name> [<baseline_config_name>]" >&2
  echo "  baseline defaults to strategy_config.sim_baseline_hmm.json" >&2
  exit 3
fi

CANDIDATE="$1"
BASELINE="${2:-strategy_config.sim_baseline_hmm.json}"

echo "── pre-flight: ${CANDIDATE} vs ${BASELINE} ──"

# Step 1: static path validator (~3s)
if ! python scripts/validate_sim_config_active.py \
     --baseline "${BASELINE}" \
     --candidate "${CANDIDATE}" > /tmp/preflight_static.log 2>&1; then
  echo "✗ STATIC path check FAILED — knob path not read by kernel"
  echo "  last 10 lines of validator output:"
  tail -10 /tmp/preflight_static.log | sed 's/^/    /'
  exit 1
fi
echo "✓ STATIC path check ACTIVE"

# Step 2: smoke (1-month sim, ~5-10 min, dominated by data load on first run)
echo "  running smoke (5-10 min) on 2024-04-01..2024-05-01..."
if ! python scripts/validate_sim_config_active.py \
     --baseline "${BASELINE}" \
     --candidate "${CANDIDATE}" \
     --smoke \
     --smoke-start 2024-04-01 --smoke-end 2024-05-01 \
     > /tmp/preflight_smoke.log 2>&1; then
  echo "✗ SMOKE FAILED — equity bit-identical to baseline in test window"
  echo "  candidate either has a no-op knob or its knob doesn't fire in this window"
  echo "  last 15 lines of smoke output:"
  tail -15 /tmp/preflight_smoke.log | sed 's/^/    /'
  echo "  NOTE: 'no fire in this window' (e.g. trailing-stop threshold not hit)"
  echo "        is a legitimate possibility — try a different --smoke-start"
  echo "        range that exercises the knob, OR accept this knob has narrow"
  echo "        regime/date applicability before launching the full panel."
  exit 2
fi
echo "✓ SMOKE shows knob fires (equity differs)"
echo
echo "PRE-FLIGHT BLESSED: ${CANDIDATE} safe to launch as full panel"
