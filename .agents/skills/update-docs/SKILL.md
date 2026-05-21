---
name: update-docs
description: Review all documentation files against actual code, fix discrepancies, then commit and push. Use after refactors, renames, or feature additions.
---

# Documentation Synchronization

Review all docs, cross-check against code, fix discrepancies, commit and push.

## Steps

1. **Read all documentation files**: AGENTS.md, README.md, and everything in doc/
2. **Cross-check against code** for each doc:
   - File paths and directory structure match reality (use Glob to verify)
   - Function names, class names, and module references exist (use Grep)
   - Configuration examples match actual config files (read strategy_config.json, config.json)
   - Command-line examples use correct flags and paths
   - Strategy names, model names, and artifact filenames are consistent
   - Feature descriptions match what the code actually does (read main.py, notebooks)
3. **Check for completeness**:
   - New features in code that aren't documented
   - Removed features still mentioned in docs
   - Trading constraints, position sizing, and indicator lists match code
4. **Fix all discrepancies** by editing the doc files
5. **Commit and push** with a descriptive message summarizing what was fixed

## Focus areas for RenQuant

- LEAN main.py capabilities vs what architecture.md claims
- Indicator list in docs vs registered indicators in common/indicators/
- Model types and their parameters vs actual implementations
- strategy_config.json schema vs what docs show
- Notebook cell descriptions vs actual notebook content
- Script docstrings vs actual CLI usage

## If $ARGUMENTS is provided

Focus the review on that specific area (e.g., `/update-docs indicators` or `/update-docs LEAN`).
