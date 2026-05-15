#!/bin/bash
# Pre-open cancel gate runner — invoked by launchd ~15 min before NYSE open.
# See scripts/preopen_cancel_gate.py for the algorithm + scientific basis.

set -eu
cd /Users/renhao/git/github/RenQuant
mkdir -p logs/preopen_gate

# shellcheck disable=SC1091
source .venv/bin/activate
set -a
# shellcheck disable=SC1091
source .env
set +a

exec python scripts/preopen_cancel_gate.py "$@"
