# Agent × GitHub automation — unified design

**Status**: design under review · supersedes split drafts in PR #16 (Claude-only) and PR #17 (Codex-only)
**Scope**: all 13 renquant repos (umbrella + 12 subrepos)
**Agents covered**: Claude, Codex, future `agent:*` (e.g. Devin, custom)
**Last updated**: 2026-06-01

---

## 1 · Goal

Three workflows, agent-agnostic — one mechanism, multiple agent identities.

| # | Workflow | Today | Target |
|---|---|---|---|
| **G1** | Agent-authored PRs are clearly attributed | Manual footer / agent identity sometimes forgotten | Every branch + PR + commit an agent touches carries enforced attribution at three layers |
| **G2** | Non-agent (human) PRs get a default agent review | Operator pings an agent in chat; agent reads PR by URL | Each registered agent posts a structured review on every non-self PR within minutes of `opened` / `synchronize` |
| **G3** | When a reviewer leaves findings on agent-owned PRs, the agent fixes them | Operator pastes comment URL into chat; agent reads, fixes, pushes | On `pull_request_review:CHANGES_REQUESTED` (or `@<agent> fix` mention), the agent auto-fixes addressable findings with safety gates |

**Non-goals**:
- Replacing human approval. Auto-merge stays manual per [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict).
- Auto-actioning on every event — §6 safety gates throttle.
- Cross-repo coordination (this doc is per-repo wiring; multi-agent collaboration mandate [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict) remains the source of truth for sync discipline).

---

## 2 · Identity model

**Labels are the source of truth** — authorship is ambiguous (a human may open an agent-assisted PR; automation may push via bot token).

### 2.1 · Canonical labels

```text
agent:claude         # Claude-authored
agent:codex          # Codex-authored
agent:auto-fix       # opted into G3 auto-fix on review (set together with agent:<name>)
agent:manual-hold    # stop ALL agent automation on this PR
agent:needs-review   # force a review even if normally skipped
agent:fix-attempt-1  # auto-fix iteration counter (incremented by G3)
agent:fix-attempt-2
agent:fix-attempt-3  # third strike → G3 disabled until human resets
```

### 2.2 · PR body footer (signed contract)

Every agent-opened PR's body ends with:

```text
Agent-Origin: <Claude|Codex|...>
Agent-Policy: auto-fix-on-review
🤖 Generated with [<Agent Name> Code](<agent docs URL>)
```

### 2.3 · Branch prefix (convenience, not contract)

```text
claude/<task-slug>   # Claude-owned
codex/<task-slug>    # Codex-owned
feat/... fix/... chore/... docs/... bug/...   # standard, human or agent
```

Branch prefix is a **convenience** for humans skimming `git branch -a`, not a gate. Labels remain authoritative.

### 2.4 · Commit message trailer

Every commit Claude touches:

```text
Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

Every commit Codex touches:

```text
Co-Authored-By: Codex <noreply@openai.com>
```

(Existing `CLAUDE.md` Harness already encodes this for Claude. Codex adopts the same pattern with its own email.)

### 2.5 · Why labels (not bot user, not branch prefix)

Bot-user-based identity breaks when a human commits via the agent's identity (or vice versa). Branch prefix breaks if someone (or another agent) reuses the branch. Labels survive squash-merge and are machine-queryable via the GitHub API.

---

## 3 · Building blocks

| Block | Provided by | Notes |
|---|---|---|
| Claude Code GitHub Action | `anthropics/claude-code-action` (official) | Modes: `review`, `fix`, `comment` |
| Codex GitHub Action | `openai/codex-action` (whatever the canonical equivalent is) | Equivalent modes |
| Reusable workflows | GitHub native | One template, multiple agents — see §3.1 |
| Claude Code CLI hooks | `claude-code` (this tool) | `PostToolUse` on `Bash` for G1 client-side enforcement |
| Codex CLI hooks | (equivalent) | Same |
| `gh` CLI | already required by [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) | PR creation + commenting surface |
| GitHub repo / org secrets | GitHub | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |

No new infra. No webhook server. No DB. Stateless.

### 3.1 · Reusable workflow strategy

Each renquant repo references ONE shared workflow that takes `agent: <name>` as input. New agents add a single line, not a whole file:

```yaml
# .github/workflows/agent-review.yml in EACH renquant repo
name: agent-review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
jobs:
  claude-review:
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: claude, model: claude-sonnet-4-6 }
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}

  codex-review:
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: codex, model: gpt-5-codex }
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
```

The template lives ONCE in the umbrella RenQuant repo. Per-repo workflow files are 10-line wrappers. Drift across 13 repos becomes a non-issue.

---

## 4 · Per-goal design

### 4.1 · G1 — Mandatory agent attribution

Three layers, defense-in-depth so a forgotten one doesn't drop attribution:

**Layer A — `CLAUDE.md §3.x` rule** (cross-repo canon):

> All branches, commits, and PRs that an agent (Claude, Codex, …) opens MUST carry the canonical attribution per §2.

**Layer B — Agent-CLI hook** (per-agent, `.claude/settings.json` for Claude, equivalent for Codex):

```jsonc
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "scripts/check_agent_attribution.sh claude"
      }]
    }]
  }
}
```

Script inspects the bash command. If it was `gh pr create` or `git commit` without the canonical attribution string, writes a warning to stderr (the agent reads as feedback and self-corrects on next turn). Optional: exit non-zero to block.

**Layer C — server-side workflow check** (`.github/workflows/agent-attribution-check.yml`):

```yaml
name: agent-attribution-check
on:
  pull_request:
    types: [opened, edited, synchronize, labeled, unlabeled]
jobs:
  check:
    # Only enforce when the PR is labeled as agent-authored.
    if: contains(github.event.pull_request.labels.*.name, 'agent:claude') ||
        contains(github.event.pull_request.labels.*.name, 'agent:codex')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          set -e
          BODY=$(gh pr view ${{ github.event.pull_request.number }} --json body -q .body)
          echo "$BODY" | grep -q "^Agent-Origin: \(Claude\|Codex\)" || {
            echo "::error::PR body missing 'Agent-Origin' footer"; exit 1
          }
          git log --format=%B origin/main..HEAD | grep -q "Co-Authored-By:" || {
            echo "::error::Agent commits missing Co-Authored-By trailer"; exit 1
          }
        env:
          GH_TOKEN: ${{ github.token }}
```

Layer A → convention, Layer B → self-feedback, Layer C → enforced gate. Layer C runs only when an `agent:*` label is present.

### 4.2 · G2 — Auto-review

Single reusable workflow template, per-agent invocation in each repo (see §3.1).

`.github/workflows/_agent-review-template.yml` (umbrella, referenced by all repos):

```yaml
name: agent-review-template
on:
  workflow_call:
    inputs:
      agent: { type: string, required: true }     # "claude" or "codex"
      model: { type: string, required: true }
    secrets:
      api_key: { required: true }

jobs:
  review:
    # Skip self-review loop + drafts + manual-hold + own-agent PRs.
    if: |
      github.event.pull_request.draft == false &&
      !contains(github.event.pull_request.labels.*.name, 'agent:manual-hold') &&
      !contains(github.event.pull_request.labels.*.name, format('agent:{0}', inputs.agent))
    runs-on: ubuntu-latest
    timeout-minutes: 20
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      group: agent-review-${{ inputs.agent }}-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Run agent review
        run: scripts/agent_review.sh ${{ inputs.agent }} ${{ inputs.model }}
        env:
          API_KEY: ${{ secrets.api_key }}
          PR_NUMBER: ${{ github.event.pull_request.number }}
```

`scripts/agent_review.sh` in each repo (or shared via composite action) dispatches to the right agent's runner — `anthropics/claude-code-action@v1` for `claude`, equivalent for `codex`.

Review prompt comes from `.github/agent-review-prompt.md` (per-repo customization point).

**Cost-control mitigations** (mandatory):

- `paths-ignore: ['doc/**', '**/*.md']` on the parent workflow (skip docs-only PRs)
- `concurrency: cancel-in-progress: true` (only review the latest commit when a PR is pushed in quick succession)
- `if: github.event.pull_request.changed_files < 100` (skip megaPRs — human-only review)
- Per-agent model selection (cheaper for review, more capable for fix)

### 4.3 · G3 — Auto-fix on reviewer findings

Same reusable-workflow pattern. Triggered on `pull_request_review:submitted` and `issue_comment:created`.

**Trigger logic** (in template):

```text
skip if PR lacks agent:<name>
skip if PR lacks agent:auto-fix
skip if PR has agent:manual-hold
skip if PR has agent:fix-attempt-3 (third strike)
skip if review/comment from agent's own automation (loop guard)
skip if actor is not trusted (owners, members, write-collaborators)
run if review state == CHANGES_REQUESTED
run if review/comment body starts with `@<agent> fix`
```

Attempt counter via label: workflow adds `agent:fix-attempt-N` after each fix run. Third strike disables G3 until human resets (removes the label).

**Permissions**: `contents: write`, `pull-requests: write` — only fires under the gate above, never on untrusted PR head from a fork.

**`pull_request_target` is NEVER used to check out untrusted code with write secrets** (per Codex doc §"Safety Rules"). Use `pull_request` for the checkout job, `pull_request_target` only for metadata operations (labeling, commenting) where the workspace isn't populated from the PR head.

---

## 5 · Cross-repo rollout

13 repos under [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict): umbrella `RenQuant` + 12 subrepos under `hallovorld/`.

Phased rollout to bound blast radius:

| Phase | Repos | Workflows shipped |
|---|---|---|
| **P0 — Pilot** | `renquant-model` only | G2 + G3 for **both** agents (Claude review + Codex review on every non-agent PR) |
| **P1 — Public protected** | 11 public protected repos | G1 attribution-check + G2 + G3 for both agents |
| **P2 — Private** | `renquant-model-internal-name`, `renquant-state-backup` | Same as P1 |
| **P3 — Canon update** | Umbrella `CLAUDE.md §3.x` | Layer A rule canonicalized; future repos inherit |

P0 runs both agents in parallel — gives us cross-reviewer signal (Claude's review of a Codex finding, Codex's review of a Claude finding) to calibrate prompt quality.

---

## 6 · Costs & quotas

Rough envelope per agent at Opus 4.7 / GPT-5-Codex prices:

| Workflow | Tokens/fire | $/fire | Fires/day | $/month/agent |
|---|---|---|---|---|
| G2 auto-review | 30K–150K | $0.05–$0.50 | 10–30 | $15–$450 |
| G3 auto-fix | 50K–300K | $0.10–$1.50 | 1–5 | $3–$225 |
| G1 attribution check | <1K | <$0.01 | every PR | <$5 |

Two agents → roughly 2× envelope ($30–$900/month for G2 across both, etc.). Mitigations from §4.2 hold the upper bound to ~$200/month/agent in realistic use.

---

## 7 · Safety gates

| Risk | Gate |
|---|---|
| **Self-review loop** | `if: !contains(labels, 'agent:<own>')` — agent skips its own PRs |
| **Cross-agent review loop** | Same gate per agent — Claude reviews `agent:codex`, Codex reviews `agent:claude`, neither reviews own |
| **Force-push race** | `--force-with-lease` (standard); `concurrency: cancel-in-progress: false` for G3 |
| **Runaway autonomy** | Manual merge always required (CLAUDE.md §3.1); `agent:fix-attempt-3` stop; `agent:manual-hold` opt-out; `if: changed_files < 100` |
| **Secret leakage in PR diff** | G2 runs `contents: read`; G3 only with `pull_request_target` for metadata ops; PR head never checked out under write secrets from forks (per Codex doc) |
| **Cost runaway** | Org-level API spend alert; `paths-ignore`; `concurrency: cancel-in-progress`; model downgrade per workflow |
| **Hostile contributor** | G3 trust check (`actor in owners/members/write-collaborators`); G3 only fires on `agent:*`-labeled PR (outside contributor can't trigger autofix) |
| **Misattribution** | Layer C gate fails the PR; `no-claude-attribution` / `no-codex-attribution` label opt-out |

---

## 8 · Open questions

1. **Reusable workflow location**: umbrella `RenQuant/.github/workflows/_agent-review-template.yml` (referenced via `uses:`) vs `hallovorld/.github-actions` separate repo. Umbrella is simpler; separate repo is more reusable across orgs. Recommend umbrella for now.

2. **`AGENTS.md` repo-local separation**: Codex doc proposes per-repo `AGENTS.md` distinct from cross-repo `CLAUDE.md`. Adopting: `CLAUDE.md` = cross-repo agent canon (this doc, §3.1/§3.2 rules, etc.); `AGENTS.md` = repo-specific tests/layout/forbidden imports. **Action**: add `AGENTS.md` template to umbrella, link from `CLAUDE.md §3`.

3. **Model selection per workflow**: G2 review = cheaper (Sonnet / GPT-5-mini), G3 fix = more capable (Opus / GPT-5). Hardcode in the per-repo invocation; revisit per-quarter.

4. **Bot identity for write actions**: PAT (start) vs GitHub App (production). PAT for pilot, migrate to App in P1.

5. **Cross-agent coordination**: when Claude and Codex both auto-fix the same PR (concurrent), who wins? Resolution: G3's `concurrency: cancel-in-progress: false` plus label-based attempt counter (`agent:claude:fix-attempt-N`, `agent:codex:fix-attempt-N`) — they queue rather than race. But should this even happen? Probably not — each PR should be owned by ONE agent's G3 (whichever set `agent:auto-fix` on label-set).

---

## 9 · Decisions needed before P0

| Q | Recommended default |
|---|---|
| Doc location | `doc/ops/agent-automation.md` (user-confirmed 2026-06-01) |
| Identity primary | **Labels** (not branch prefix, not bot user) |
| Reusable workflow location | Umbrella `RenQuant/.github/workflows/_agent-*-template.yml` |
| Bot identity | PAT for P0; GitHub App in P1 |
| G3 trigger | `agent:auto-fix` label + (CHANGES_REQUESTED OR `@<agent> fix`) |
| G2 model | Sonnet 4.6 / GPT-5-mini |
| G3 model | Opus 4.7 / GPT-5-Codex |
| Rollout pilot | `renquant-model` |
| Opt-out | `agent:manual-hold` (universal) + `no-claude-review` / `no-codex-review` (per-agent) labels |
| Max fix attempts | 3 (then label stop) |

---

## 10 · Migration from PRs #16 and #17

- This doc supersedes the design content in both PRs.
- PR #16's `doc/ops/claude-github-automation.md` → delete (content folded here).
- PR #17's `doc/arch/codex-agent-automation.md` → delete (content folded here).
- After this PR lands: close PR #16 and PR #17 as superseded; cherry-pick any unique strengths I missed.

**Strengths preserved from PR #16** (Claude-side):
- §5 cost envelope with concrete mitigations (paths-ignore, concurrency, model selection, file-count gate)
- Three-layer attribution defense (CLAUDE.md rule + hook + server check)
- §7 open questions structure

**Strengths preserved from PR #17** (Codex-side):
- Label-based identity contract (replaces my branch-prefix approach)
- `Agent-Origin` footer (machine-readable structured contract)
- Attempt counter via labels (`agent:fix-attempt-N`)
- `pull_request_target` safety treatment
- Trust-check explicit specification (owners/members/write-collaborators)
- `AGENTS.md` repo-local instruction surface separation
- Bounded retry with `agent:fix-attempt-3` stop

---

## 11 · Next steps

1. **Approve this design** (or push back on §9 defaults).
2. **Pilot PR against `renquant-model`**: adds `.github/workflows/agent-review.yml` (10-line wrapper) + `.github/workflows/agent-attribution-check.yml` + `.github/agent-review-prompt.md` + secrets (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). One PR, both agents wired.
3. **Add the template** to the umbrella at `.github/workflows/_agent-*-template.yml`.
4. **Calibrate** on the first 5 PR cycles in the pilot.
5. **P1 rollout**: 11 cookie-cutter PRs against the public protected repos.
6. **Umbrella `CLAUDE.md §3.x` update** — Layer A rule canonicalized.

---

## Appendix · References

- Claude Code Action: https://github.com/anthropics/claude-code-action
- Codex Action (or equivalent): TBD per OpenAI canonical
- Claude Code hooks: https://docs.claude.com/claude-code/hooks
- [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) — PR-based workflow canon
- [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict) — sync-from-remote mandate
- [`CLAUDE.md §3.4`](../../CLAUDE.md#34--pr-review-protocol) — PR review protocol
- GitHub Actions events: https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows
- GitHub Actions reusable workflows: https://docs.github.com/en/actions/sharing-automations/reusing-workflows
