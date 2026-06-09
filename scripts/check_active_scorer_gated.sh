#!/usr/bin/env bash
# check_active_scorer_gated.sh — RFC #259 P0b active-scorer gate check (delegate).
#
# Multi-repo + pipeline mode: this wrapper holds NO gate logic. It resolves the
# umbrella root + the subrepo PYTHONPATH, then DELEGATES to the gate-owner module
# `renquant_backtesting.wf_gate.check_active_scorer` (RenQuant CLAUDE.md §3.5 —
# the gate/acceptance logic lives in renquant-backtesting, never the umbrella).
#
# Asserts every active production config's scorer artifact carries a passing
# wf_gate_metadata, so the config-edit promotion bypass fails here instead
# of silently at runtime (where live P-WF-GATE then blocks all buys).
#
# Usage: bash scripts/check_active_scorer_gated.sh [--config NAME ...] [--strategy S]
# Exit:  0 = all gated · 1 = a scorer ungated · 2 = cannot determine (fail-closed).
set -uo pipefail

REPO_DIR="/Users/renhao/git/github/RenQuant"
PYTHON="$REPO_DIR/.venv/bin/python"

# Multi-repo delegation: correct umbrella root + subrepo PYTHONPATH.
export RENQUANT_REPO_ROOT="$REPO_DIR"
GITHUB_DIR="$(dirname "$REPO_DIR")"
# shellcheck disable=SC1091
source "$REPO_DIR/scripts/subrepo_env.sh"
renquant_load_subrepo_env "$REPO_DIR"
SUBREPO_ROOT="$(renquant_subrepo_root "$REPO_DIR" "$GITHUB_DIR")"
export RENQUANT_SUBREPO_ROOT="$SUBREPO_ROOT"
export PYTHONPATH="$(renquant_subrepo_pythonpath "$SUBREPO_ROOT" renquant-orchestrator renquant-common renquant-base-data renquant-artifacts renquant-model renquant-pipeline renquant-execution renquant-strategy-104 renquant-backtesting):${PYTHONPATH:-}"

if ! "$PYTHON" -c "import renquant_backtesting.wf_gate.check_active_scorer" >/dev/null 2>&1; then
    echo "ERROR: renquant_backtesting.wf_gate.check_active_scorer unavailable (subrepo pin/env)." >&2
    exit 2
fi

exec "$PYTHON" -m renquant_backtesting.wf_gate.check_active_scorer "$@"
