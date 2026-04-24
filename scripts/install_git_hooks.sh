#!/usr/bin/env bash
# Install git pre-commit hook that runs check_config_drift.py.
#
# 2026-04-24: the AB-trim incident shipped a default-on flag (trim_threshold
# 0.10) that silently regressed APY by 12.7 pts. This pre-commit hook guards
# against future incidents — any commit that touches strategy_config.json
# and drifts from strategy_config.golden.json fails the commit.
#
# Usage:
#   bash scripts/install_git_hooks.sh
#
# Bypasses (use sparingly):
#   git commit --no-verify ...    # skip hooks entirely
#   SKIP_DRIFT_CHECK=1 git commit  # skip just this hook
#
# To uninstall:
#   rm .git/hooks/pre-commit

set -e

REPO_DIR=$(git rev-parse --show-toplevel)
HOOK_PATH="$REPO_DIR/.git/hooks/pre-commit"

mkdir -p "$(dirname "$HOOK_PATH")"

cat > "$HOOK_PATH" <<'HOOK_EOF'
#!/usr/bin/env bash
# RenQuant pre-commit: config-drift guard.
# Runs only when strategy_config.json is part of the staged diff.

if [ "$SKIP_DRIFT_CHECK" = "1" ]; then
    exit 0
fi

REPO_DIR=$(git rev-parse --show-toplevel)
CONFIG_STAGED=$(git diff --cached --name-only | grep -E "strategy_config\.json$" || true)

if [ -z "$CONFIG_STAGED" ]; then
    exit 0
fi

PYTHON="/Users/renhao/miniconda3/envs/renquant/bin/python"
if [ ! -x "$PYTHON" ]; then
    PYTHON=$(which python3 2>/dev/null || which python 2>/dev/null)
fi

if [ -z "$PYTHON" ] || [ ! -x "$PYTHON" ]; then
    echo "pre-commit: Python not found, skipping drift check"
    exit 0
fi

# Infer strategy from path (backtesting/{strategy}/strategy_config.json)
STRATEGIES=$(echo "$CONFIG_STAGED" | sed -E 's|backtesting/([^/]+)/strategy_config\.json|\1|' | sort -u)

FAIL=0
for strat in $STRATEGIES; do
    echo "pre-commit: checking drift for strategy=$strat"
    "$PYTHON" "$REPO_DIR/scripts/check_config_drift.py" --strategy "$strat" || FAIL=1
done

if [ "$FAIL" = "1" ]; then
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "  Config drift detected.  Either:"
    echo "    1. Update strategy_config.golden.json to match (promote)"
    echo "    2. Revert the drifted strategy_config.json"
    echo "    3. SKIP_DRIFT_CHECK=1 git commit  (bypass — use sparingly)"
    echo "────────────────────────────────────────────────────────────"
    exit 1
fi

exit 0
HOOK_EOF

chmod +x "$HOOK_PATH"
echo "Installed pre-commit hook at $HOOK_PATH"
echo ""
echo "The hook runs check_config_drift.py when strategy_config.json is staged."
echo "To uninstall: rm $HOOK_PATH"
echo "To bypass (once): SKIP_DRIFT_CHECK=1 git commit ..."
