# Agent PR control plane — operator runbook

The local-agent replacement for the deleted GitHub-event/CI review
automation. Two agents (Claude Code CLI, Codex CLI) review each other's
PRs, fix their own, and merge their own — driven by
`renquant-orchestrator`, triggered by you or `/loop`. No GitHub Actions, no
cloud-model calls.

Design: [`renquant-orchestrator/doc/cross-repo-control-plane-design.md`](../../../renquant-orchestrator/doc/cross-repo-control-plane-design.md).

## 1 · One-time setup — per-agent tokens

Each agent uses its OWN GitHub token, so reviews/commits/merges are
attributed correctly and GitHub's native "cannot approve your own PR" rule
enforces review separation for free.

```bash
# In Claude's shell environment:
export RENQUANT_CLAUDE_GH_TOKEN=ghp_...claude...
# In Codex's shell environment:
export RENQUANT_CODEX_GH_TOKEN=ghp_...codex...
```

The orchestrator's `--as <agent>` resolves the matching var (falls back to
`GH_TOKEN`). If an agent acts on GitHub via an MCP server instead of `gh`,
configure that server with the same per-agent token; `--token` is the
single override point either way.

## 2 · Invocation surfaces

| | Claude | Codex |
|---|---|---|
| review peer's PRs | `/rq-review` | `repos agent --as codex --workflow review` (per `AGENTS.md`) |
| fix own findings | `/rq-fix` | `... --as codex --workflow fix` |
| merge own approved | `/rq-merge` | `... --as codex --workflow merge --execute` |

Claude's `.claude/skills/rq-{review,fix,merge}` make these one-word slash
commands. Codex's `AGENTS.md` carries the equivalent instructions.

Direct CLI (either agent):
```bash
ORCH="PYTHONPATH=/Users/renhao/git/github/renquant-orchestrator/src python3 -m renquant_orchestrator"
$ORCH repos agent --as claude --workflow review --repo all          # cross-repo review queue
$ORCH repos agent --as claude --workflow merge --repo hallovorld/RenQuant --execute
$ORCH repos status                                                  # branch/dirty/ahead-behind, all repos
$ORCH repos sync                                                    # fetch all; ff-only on clean main
$ORCH repos prs                                                     # open PRs across every repo
```

## 3 · Triggering — manual or /loop

Manual = run the skill / command once. Recurring = wrap in `/loop`:

```
/loop 2h  /rq-review                       # Claude reviews codex's PRs every 2h
/loop 30m /rq-merge                        # Claude merges its approved+green PRs (one repo)
/loop 1h  $ORCH repos sync                 # keep all clones fresh
```
Codex runs its own loops for its three workflows. Start/stop a loop to turn
a workflow on/off. The two agents thus close the
review → fix → re-review → merge cycle on independent, operator-controlled
cadences.

## 4 · Safety (enforced by the orchestrator)

- `merge` only acts on APPROVED-at-head + all-checks-green + no stop label
  (`agent:manual-hold` / `agent:cost-cap` / `agent:rebase-conflict`).
- Cross-repo `merge --repo all --execute` REFUSES without `--allow-all` and
  a bounded `--max-merges` (no silent fan-out).
- A repo with no PR-level CI is **not** mergeable unless `--allow-no-checks`
  (default fails closed).
- `merge` posts a visible `merged by <agent>` audit comment before merging,
  and fails closed if it can't.
- `sync` only fast-forwards a clean `main`; feature/dirty trees are
  fetch-only (never auto-pulled).
- You retain `gh pr merge --admin` for manual override (branch protection
  has `enforce_admins=false`).
