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

Store and load tokens via [`doc/ops/agent-token-storage.md`](agent-token-storage.md).
Do not paste token values into this runbook, shell profiles, PR bodies, comments,
or agent chats. The orchestrator's `--as <agent>` resolves the matching var
(falls back to `GH_TOKEN`). If an agent acts on GitHub via an MCP server instead
of `gh`, configure that server with the same per-agent token; `--token` is the
single override point either way.

### 1.1 · The two tokens must belong to DISTINCT GitHub accounts

GitHub's "cannot approve your own PR" rule — the thing that makes an
`APPROVED` review a genuine second opinion — keys on the **account that owns
the token**, not on the token string. So:

- A second PAT minted from the *same* account (e.g. two `hallovorld` PATs)
  does **not** separate the agents. The author and the reviewer are still the
  same login; `gh pr review --approve` fails with
  `Can not approve your own pull request`, and `gh pr merge --admin` is then
  blocked by `required_approving_review_count: 1`.
- `RENQUANT_CLAUDE_GH_TOKEN` and `RENQUANT_CODEX_GH_TOKEN` must therefore come
  from **two different GitHub accounts**. `hallovorld` may be one of them; a
  third bot account is not required. Current two-account roster:

  | Agent | GitHub login | Role |
  |---|---|---|
  | Claude | `hallovorld` | Claude PR author/reviewer + owner emergency override |
  | Codex | `haorensjtu-dev` or another non-`hallovorld` collaborator | Codex PR author/reviewer |

  Only then does the merge flow through the genuine two-identity path.

The PR creator must match the agent identity. Codex PRs must be opened/pushed by
the Codex login, and Claude PRs by the Claude login. If a Codex PR is opened
with `hallovorld`, then Claude-as-`hallovorld` cannot approve it; GitHub sees
the PR author and reviewer as the same login even if the commits say
`Agent-Origin: Codex`.

**Until both agent tokens map to different logins and PRs are created by the
right login**, agent-authored PRs cannot complete the normal review-gated merge
path. They either remain queued for a valid second account, or the owner
performs the emergency override in §4. That override bypasses the review gate,
so it must be rare, explicitly audited, and never treated as the steady state.
Observed live on 2026-06-05: codex's #216/#217 were verified correct by Claude
comments but could not receive formal approvals under the single `hallovorld`
identity, then landed by owner override with audit comments.

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
- Emergency owner override path (when the PR was created with the wrong login or
  no valid second login exists yet, per §1.1):
  `main` has `enforce_admins=true` + `required_approving_review_count=1`, so a
  bare `gh pr merge --admin` is refused. This is not the normal merge path; use
  it only when the owner has independently verified the PR and the missing
  second account is the only blocker. Before the merge, post an audit comment
  that must name/link the reviewer evidence and the review/comment used as the
  second-opinion basis, and states that the review requirement will be restored
  immediately. Then force-land the verified PR:
  ```bash
  gh api repos/<owner>/<repo>/branches/main/protection/required_pull_request_reviews
  gh api -X PATCH repos/<owner>/<repo>/branches/main/protection/required_pull_request_reviews \
    -F dismiss_stale_reviews=false \
    -F require_code_owner_reviews=false \
    -F require_last_push_approval=false \
    -F required_approving_review_count=0
  gh pr merge <PR#> --repo <owner>/<repo> --merge --admin
  gh api -X PATCH repos/<owner>/<repo>/branches/main/protection/required_pull_request_reviews \
    -F dismiss_stale_reviews=false \
    -F require_code_owner_reviews=false \
    -F require_last_push_approval=false \
    -F required_approving_review_count=1
  gh api repos/<owner>/<repo>/branches/main/protection/required_pull_request_reviews
  ```
  Always restore the review requirement in the same shell/trap-backed step and
  verify `required_approving_review_count: 1` afterward. If restore fails, stop
  all merges until branch protection is back in place. This bypasses the review
  gate — use only for owner-verified PRs while §1.1's identity/PR-actor
  contract is broken.
