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
- ~~Replacing human approval. Auto-merge stays manual per `CLAUDE.md §3.1`.~~ **Superseded 2026-06-03**: v2 Phase B introduced gated auto-merge for agent-authored PRs (8-gate `agent-auto-merge` workflow + `agent:auto-merge` opt-in label). See [`agent-automation-v2-design.md`](agent-automation-v2-design.md) and [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) point 5. Human-authored PRs still merge manually; opt-in is required per-PR or per-repo.
- Auto-actioning on every event — §6 safety gates throttle.
- Cross-repo coordination (this doc is per-repo wiring; multi-agent collaboration mandate [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict) remains the source of truth for sync discipline). v2 Phase C adds `agent-paired-merge-gate` for the umbrella+subrepo pairing case specifically.

---

## 2 · Identity model

**Labels are the source of truth** — authorship is ambiguous (a human may open an agent-assisted PR; automation may push via bot token).

### 2.1 · Canonical labels

```text
# Authorship labels — set when the agent OPENS the PR (G1 surface).
# A PR has at most ONE authorship label.
agent:claude         # Claude-authored
agent:codex          # Codex-authored

# Fix-executor labels — DISTINCT from authorship; declare which agent(s)
# are PERMITTED to run G3 auto-fix on this PR. A Claude-authored PR can
# add agent:fix:codex to invite Codex for cross-second-opinion fixes,
# without touching the authorship label (which would conflict with G1).
agent:fix:claude     # Claude G3 may run on this PR
agent:fix:codex      # Codex G3 may run on this PR

# Universal controls.
agent:manual-hold    # stop ALL agent automation on this PR
agent:needs-review   # force a review even if normally skipped

# Per-fix-executor attempt counters — gated by agent:fix:<name>, not
# the authorship label. One executor exhausting its budget MUST NOT
# block another from running on the same PR.
agent:fix:claude:attempt-1
agent:fix:claude:attempt-2
agent:fix:claude:attempt-3   # third strike → Claude G3 disabled
agent:fix:codex:attempt-1
agent:fix:codex:attempt-2
agent:fix:codex:attempt-3    # third strike → Codex G3 disabled
```

**Why authorship vs fix-executor split**: prior drafts conflated
`agent:claude` as both "Claude opened this PR" (G1 attribution gate)
and "Claude may auto-fix this PR" (G3 execution permission). That
made cross-second-opinion impossible: a Claude-authored PR adding
`agent:codex` to invite Codex would gain a second authorship label
(silently picked first by the attribution gate) AND there was no
way to opt Codex into G3 without that double-labelling. Now:
authorship is a single `agent:<name>` label set by G1; G3 execution
permission is a separate `agent:fix:<name>` label set per-executor.

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

| Block | Provided by | What it actually gives us |
|---|---|---|
| Claude Code GitHub Action | `anthropics/claude-code-action` | A prompt-driven runner. NOT mode-based — the workflow passes a prompt + `claude_args`, the action invokes Claude with `Bash`/`Edit`/`gh` tools available. Posting comments, choosing tools, deciding when to push commits is the **workflow's** job, not the action's. |
| Codex GitHub Action | `openai/codex-action` | A `codex exec` wrapper that returns `final-message`. Same shape: it runs Codex against a prompt; the workflow consumes the result and posts/commits/pushes. |
| Reusable workflows | GitHub native | One template, multiple agents — see §3.1 |
| Claude Code CLI hooks | `claude-code` (this tool) | `PostToolUse` on `Bash` for G1 client-side enforcement |
| Codex CLI hooks | Codex CLI native | Same hook surface |
| `gh` CLI | already required by [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) | PR creation + commenting + label mutation. Wrappers call `gh pr review` / `gh pr comment` / `git commit && git push` directly — actions don't auto-comment for us. |
| GitHub repo / org secrets | GitHub | `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` |

**Implication for the design**: there is **no built-in `review` / `fix` / `comment` mode** in either action. Our wrapper workflow owns the orchestration: it prompts the agent with the right context (PR diff, review comments, repo invariants), waits for the agent's response, then decides whether to post a review comment (G2) or push a commit (G3). The action is the LLM dispatch; the wrapper is the policy.

No new infra. No webhook server. No DB. Stateless.

### 3.1 · Reusable workflow strategy

Each renquant repo references ONE shared workflow that takes `agent: <name>` as input. New agents add a single line, not a whole file. **Cost gates from §6 are mandatory in the per-repo wrapper, not advisory** — otherwise the pilot will be easy to copy without the cost controls:

```yaml
# .github/workflows/agent-review.yml in EACH renquant repo
name: agent-review
on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
    # No paths-ignore: branch protection requires review jobs to emit a
    # status on every PR, including docs-only PRs.
concurrency:
  # MANDATORY: only review the latest commit when a PR is pushed
  # in quick succession — older runs cancel.
  group: agent-review-${{ github.event.pull_request.number }}
  cancel-in-progress: true
jobs:
  claude-review:
    # MANDATORY size gate: skip mega-PRs (human-only review territory).
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: claude, model: claude-sonnet-4-6 }
    secrets:
      api_key: ${{ secrets.ANTHROPIC_API_KEY }}

  codex-review:
    if: github.event.pull_request.changed_files < 100
    uses: hallovorld/RenQuant/.github/workflows/_agent-review-template.yml@main
    with: { agent: codex, model: gpt-5-codex }
    secrets:
      api_key: ${{ secrets.OPENAI_API_KEY }}
```

The template lives ONCE in the umbrella RenQuant repo. Per-repo workflow files are ~25-line wrappers (mostly cost gates). Drift across 13 repos becomes a non-issue.

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

The naive form (`git log origin/main..HEAD | grep -q Co-Authored-By:`) is theatrical — `actions/checkout@v4` doesn't fetch `origin/main` by default so the range is empty, and a single matching trailer anywhere passes the grep. Real form: fetch the base, enumerate every commit in the PR, and derive the expected `Agent-Origin` + `Co-Authored-By` value from the actual `agent:*` label so a mislabeled PR fails loud:

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
        with:
          # MUST: fetch full history including base ref so the commit
          # enumeration below has real range. Default checkout depth=1
          # would make `git log ${BASE}..HEAD` empty.
          fetch-depth: 0
          ref: ${{ github.event.pull_request.head.sha }}
      - name: Derive expected agent identity from labels
        id: agent
        env:
          LABELS: ${{ toJson(github.event.pull_request.labels.*.name) }}
        run: |
          set -euo pipefail
          # ENFORCE: §2.1 contract — exactly ONE authorship label.
          # Branching on `if/elif` (first match wins) silently resolves
          # double-labelled PRs as the first agent in the chain, hiding
          # a real misconfig where two agents claim authorship. Count
          # explicitly and fail loud on != 1.
          AUTHORSHIP_COUNT=0
          AGENT_LABEL=""
          EXPECTED_ORIGIN=""
          EXPECTED_TRAILER_PATTERN=""
          if echo "$LABELS" | grep -q '"agent:claude"'; then
            AUTHORSHIP_COUNT=$((AUTHORSHIP_COUNT + 1))
            AGENT_LABEL="agent:claude"
            EXPECTED_ORIGIN="Claude"
            EXPECTED_TRAILER_PATTERN="Co-Authored-By: Claude .*<noreply@anthropic.com>"
          fi
          if echo "$LABELS" | grep -q '"agent:codex"'; then
            AUTHORSHIP_COUNT=$((AUTHORSHIP_COUNT + 1))
            AGENT_LABEL="agent:codex"
            EXPECTED_ORIGIN="Codex"
            EXPECTED_TRAILER_PATTERN="Co-Authored-By: Codex <noreply@openai.com>"
          fi
          if [ "$AUTHORSHIP_COUNT" -eq 0 ]; then
            echo "::error::no agent:* authorship label present; this job should not have fired"
            exit 2
          elif [ "$AUTHORSHIP_COUNT" -gt 1 ]; then
            echo "::error::multiple authorship labels present (count=${AUTHORSHIP_COUNT}); §2.1 mandates exactly one — to invite a second agent's G3, use agent:fix:<name>, not the authorship label"
            exit 1
          fi
          {
            echo "agent_label=${AGENT_LABEL}"
            echo "expected_origin=${EXPECTED_ORIGIN}"
            echo "expected_trailer_pattern=${EXPECTED_TRAILER_PATTERN}"
          } >> "$GITHUB_OUTPUT"
      - name: Validate PR body footer matches label
        env:
          GH_TOKEN: ${{ github.token }}
          EXPECTED_ORIGIN: ${{ steps.agent.outputs.expected_origin }}
        run: |
          set -euo pipefail
          BODY=$(gh pr view ${{ github.event.pull_request.number }} --json body -q .body)
          # MUST cross-check: label is agent:X but body MUST say
          # "Agent-Origin: X" too. Mismatch = mislabeled PR, fail.
          if ! grep -qx "Agent-Origin: ${EXPECTED_ORIGIN}" <<< "$BODY"; then
            echo "::error::PR labeled ${{ steps.agent.outputs.agent_label }} but body missing 'Agent-Origin: ${EXPECTED_ORIGIN}' footer"
            exit 1
          fi
      - name: Validate every commit in PR range has matching trailer
        env:
          EXPECTED_TRAILER_PATTERN: ${{ steps.agent.outputs.expected_trailer_pattern }}
        run: |
          set -euo pipefail
          BASE="${{ github.event.pull_request.base.sha }}"
          HEAD="${{ github.event.pull_request.head.sha }}"
          # Fetch base ref explicitly — checkout@v4 fetched HEAD only.
          git fetch --no-tags origin "${BASE}"
          # Enumerate EACH commit in the range, not just whether ANY
          # commit has a trailer. Mixed-trailer PRs (some Claude, some
          # Codex) fail too — a single PR must be one agent's work.
          missing=()
          for sha in $(git rev-list "${BASE}..${HEAD}"); do
            if ! git show -s --format='%B' "$sha" | grep -Eq "$EXPECTED_TRAILER_PATTERN"; then
              missing+=("$sha")
            fi
          done
          if [ "${#missing[@]}" -gt 0 ]; then
            echo "::error::commits missing expected trailer matching '$EXPECTED_TRAILER_PATTERN':"
            printf '  %s\n' "${missing[@]}"
            exit 1
          fi
```

What this enforces, that the naive form did not:

1. **Fetched base** — `fetch-depth: 0` + explicit `git fetch origin <base.sha>` means `git rev-list base..HEAD` returns real SHAs.
2. **Per-commit enumeration** — every commit in the PR range is checked, not "at least one has some trailer".
3. **Label ↔ Origin ↔ Trailer cross-check** — a PR labeled `agent:codex` with `Agent-Origin: Claude` AND a `Co-Authored-By: Codex` line fails because the body says Claude but label says Codex (and vice versa).
4. **Trailer regex pins identity-email** — `Co-Authored-By: Claude .*<noreply@anthropic.com>` won't pass a generic `Co-Authored-By: Dependabot <...>` trailer.

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

- `concurrency: cancel-in-progress: true` (only review the latest commit when a PR is pushed in quick succession)
- `if: github.event.pull_request.changed_files < 100` (skip megaPRs — human-only review)
- Per-agent model selection (cheaper for review, more capable for fix)

**Fail-closed review prerequisite**: missing `ANTHROPIC_API_KEY` /
`OPENAI_API_KEY` is a workflow failure, not a green skip. A PR must not
merge as "reviewed" when the review job did not have credentials to run.

### 4.3 · G3 — Auto-fix on reviewer findings

Same reusable-workflow pattern. Triggered on `pull_request_review:submitted` and `issue_comment:created`.

#### 4.3.1 · Event-context resolution (do this BEFORE label/branch checks)

`pull_request_review` and `issue_comment` have different event payloads. **Most subtle: `issue_comment` does NOT have `github.event.pull_request`** — only `github.event.issue.pull_request` (which is a partial reference, not the full PR object). Naive workflows that read `github.event.pull_request.labels.*.name` for `issue_comment` events get an empty list and silently skip every fix request.

Resolve the PR object explicitly as the FIRST step:

```yaml
jobs:
  fix:
    runs-on: ubuntu-latest
    steps:
      - name: Resolve PR context for both event shapes
        id: pr
        env:
          GH_TOKEN: ${{ github.token }}
        run: |
          set -euo pipefail
          # Two event shapes converge here:
          #   pull_request_review → event.pull_request.number     (full PR object)
          #   issue_comment       → event.issue.number            (issue ref; .pull_request only if comment is on a PR)
          if [ "${{ github.event_name }}" = "pull_request_review" ]; then
            PR_NUMBER="${{ github.event.pull_request.number }}"
          else
            # issue_comment — verify the issue is actually a PR.
            if [ -z "${{ github.event.issue.pull_request.url }}" ]; then
              echo "issue_comment on non-PR issue; skipping"; exit 0
            fi
            PR_NUMBER="${{ github.event.issue.number }}"
          fi
          # Fetch the full PR object for label / head-ref / draft / etc checks.
          gh pr view "$PR_NUMBER" --json \
            number,labels,headRefName,headRefOid,baseRefName,baseRefOid,isDraft,author \
            > pr.json
          {
            echo "pr_number=$PR_NUMBER"
            echo "labels=$(jq -c '[.labels[].name]' pr.json)"
            echo "head_ref=$(jq -r .headRefName pr.json)"
            echo "head_sha=$(jq -r .headRefOid pr.json)"
            echo "base_ref=$(jq -r .baseRefName pr.json)"
          } >> "$GITHUB_OUTPUT"
      # ... subsequent label / trust / attempt checks use steps.pr.outputs.* ...
```

Without this step, the design will recreate the same event-context bug across both agents. Treat §4.3.1 as mandatory implementation surface, not just documentation.

#### 4.3.2 · Trigger logic (in template, AFTER resolve step)

```text
skip if labels lack agent:fix:<name>                        # explicit per-agent
                                                            # opt-in to G3 — NOT
                                                            # the authorship label
                                                            # (see §2.1 split)
skip if labels include agent:manual-hold                    # universal stop
skip if labels include agent:fix:<name>:attempt-3           # PER-EXECUTOR third
                                                            # strike — Claude's
                                                            # exhausted budget MUST
                                                            # NOT block Codex G3
skip if review/comment from agent's own bot user            # loop guard
skip if actor not in {owners, members, write-collaborators} # trust check
run if review.state == CHANGES_REQUESTED
run if review/comment body starts with `@<name> fix`
```

#### 4.3.3 · Attempt counter (per-executor)

Each fix run increments `agent:fix:<name>:attempt-N` — keyed on the
**fix-executor label**, not the authorship label (per §2.1 split).
Per-executor isolation is mandatory because a single PR can carry
BOTH `agent:fix:claude` and `agent:fix:codex` (cross-second-opinion
is a first-class use case, §8 Q5). A global counter — or one keyed
on authorship — would let one agent burn three attempts and block
the other from ever running, which is wrong.

#### 4.3.4 · Permissions + secret-leakage safety

`contents: write` + `pull-requests: write` — only fires under §4.3.2 gates, never on untrusted PR head from a fork.

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

### 6.1 · Quota exhaustion — fail-closed vs graceful-degrade (2026-06-03)

**Incident**: OpenAI returned `Quota exceeded. Check your plan and
billing details.` mid-session. `openai/codex-action@v1` exited 1, the
`codex-review` job failed, and — because `codex-review / review` is a
**required** status check on `main` — **every** open PR (Claude's and
Codex's alike) was wedged at the review gate until quota refilled. The
operator had to temporarily edit branch protection to land a docs PR.

**Root cause**: the G2 template's only failure mode was fail-closed.
That is the correct default (no review possible ⇒ don't let the PR
merge "as reviewed"), but it conflates two very different events:

| Event | What it means | Right response |
|---|---|---|
| Agent posts `CHANGES_REQUESTED` | Review ran, found a blocker | Block merge (G3 may auto-fix) — job SUCCESS |
| Agent action exits non-zero | Review **could not run** (quota / rate-limit / 5xx / OIDC) | Either fail-closed OR degrade |

Crucially, a review-**found**-blocker is expressed by the agent posting
a `CHANGES_REQUESTED` review, which is a **successful** job. So a job
**failure** in the run-agent step *only ever* means "no valid review
produced" — never "review found a problem". That makes it safe to
treat a job failure as a degradable infra event without suppressing a
real finding.

**Mechanism** (opt-in, default preserves fail-closed):

- Repo variable `AGENT_REVIEW_DEGRADE_ON_INFRA_FAILURE`:
  - **unset / `!= "true"`** (default) → original fail-closed behavior;
    the required review check goes red, nothing merges until a review
    runs.
  - **`"true"`** → the run-agent step's failure is classified as an
    infra failure and the review **degrades**: the required check is
    allowed to pass (so the whole pipeline isn't wedged), but the PR is
    stamped with the `agent:review-degraded` label.
- The Phase B auto-merge gate (G-A3) treats `agent:review-degraded` as
  a **stop label** — a degraded PR can be merged by a **human** who has
  read it, but it never auto-merges "as reviewed". The human is the
  fallback reviewer.
- A sticky PR comment (markered per `agent:HEAD_SHA`) explains the
  degrade and tells the operator to either merge manually after reading,
  or re-run the review once quota recovers and remove the label.

**Tradeoff**: with the variable on, a quota outage no longer wedges the
whole repo, but it does mean a human can merge a PR that an agent never
reviewed. That is strictly weaker than pure fail-closed — hence opt-in.
Turn it on per repo only when the operator accepts "human is the
fallback reviewer during outages"; leave it off to keep the hard
guarantee that nothing merges without an agent review.

Enable with:

```bash
gh variable set AGENT_REVIEW_DEGRADE_ON_INFRA_FAILURE --body "true" --repo <owner>/<repo>
```

---

## 7 · Safety gates

| Risk | Gate |
|---|---|
| **Self-review loop** | `if: !contains(labels, 'agent:<own>')` — agent skips its own PRs |
| **Cross-agent review loop** | Same gate per agent — Claude reviews `agent:codex`, Codex reviews `agent:claude`, neither reviews own |
| **Force-push race** | `--force-with-lease` (standard); `concurrency: cancel-in-progress: false` for G3 |
| **Runaway autonomy** | Manual merge always required (CLAUDE.md §3.1); per-executor `agent:fix:<name>:attempt-3` stop (§4.3.3); `agent:manual-hold` opt-out; `if: changed_files < 100` |
| **Secret leakage in PR diff** | G2 runs `contents: read`; G3 only with `pull_request_target` for metadata ops; PR head never checked out under write secrets from forks (per Codex doc) |
| **Cost runaway** | Org-level API spend alert; `concurrency: cancel-in-progress`; file-count gate; model downgrade per workflow |
| **Hostile contributor** | G3 trust check (`actor in owners/members/write-collaborators`); G3 only fires on `agent:*`-labeled PR (outside contributor can't trigger autofix) |
| **Misattribution** | Layer C gate fails the PR; `no-claude-attribution` / `no-codex-attribution` label opt-out |
| **Quota outage wedges every PR** | Default fail-closed; opt-in `AGENT_REVIEW_DEGRADE_ON_INFRA_FAILURE=true` degrades an infra-failed review to `agent:review-degraded` (auto-merge G-A3 stop; human is fallback reviewer). See §6.1 |

---

## 8 · Open questions

1. **Reusable workflow location**: umbrella `RenQuant/.github/workflows/_agent-review-template.yml` (referenced via `uses:`) vs `hallovorld/.github-actions` separate repo. Umbrella is simpler; separate repo is more reusable across orgs. Recommend umbrella for now.

2. **`AGENTS.md` repo-local separation**: Codex doc proposes per-repo `AGENTS.md` distinct from cross-repo `CLAUDE.md`. Adopting: `CLAUDE.md` = cross-repo agent canon (this doc, §3.1/§3.2 rules, etc.); `AGENTS.md` = repo-specific tests/layout/forbidden imports. **Action**: add `AGENTS.md` template to umbrella, link from `CLAUDE.md §3`.

3. **Model selection per workflow**: G2 review = cheaper (Sonnet / GPT-5-mini), G3 fix = more capable (Opus / GPT-5). Hardcode in the per-repo invocation; revisit per-quarter.

4. **Bot identity for write actions**: PAT (start) vs GitHub App (production). PAT for pilot, migrate to App in P1.

5. **Cross-agent coordination**: when Claude and Codex both auto-fix the same PR (concurrent), who wins? Resolution: G3's `concurrency: cancel-in-progress: false` plus the per-executor attempt counter from §4.3.3 (`agent:fix:claude:attempt-N`, `agent:fix:codex:attempt-N` — keyed on fix-executor labels per §2.1 split, never on authorship and never global) means they queue rather than race AND one executor exhausting its budget doesn't block the other. Cross-second-opinion (e.g. reviewer types `@codex fix` on a Claude-authored PR — which carries `agent:claude` for authorship + `agent:fix:codex` for the invitation) is a first-class use case rather than an edge case.

---

## 9 · Decisions needed before P0

| Q | Recommended default |
|---|---|
| Doc location | `doc/ops/agent-automation.md` (user-confirmed 2026-06-01) |
| Identity primary | **Labels** (not branch prefix, not bot user) |
| Reusable workflow location | Umbrella `RenQuant/.github/workflows/_agent-*-template.yml` |
| Bot identity | PAT for P0; GitHub App in P1 |
| G3 trigger | `agent:fix:<name>` executor-permission label + (CHANGES_REQUESTED OR `@<name> fix`); see §4.3.2 |
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
- §5 cost envelope with concrete mitigations (concurrency, model selection, file-count gate)
- Three-layer attribution defense (CLAUDE.md rule + hook + server check)
- §7 open questions structure

**Strengths preserved from PR #17** (Codex-side):
- Label-based identity contract (replaces my branch-prefix approach)
- `Agent-Origin` footer (machine-readable structured contract)
- Attempt counter via labels (per-executor `agent:fix:<name>:attempt-N` after §2.1 split)
- `pull_request_target` safety treatment
- Trust-check explicit specification (owners/members/write-collaborators)
- `AGENTS.md` repo-local instruction surface separation
- Bounded retry with `agent:fix:<name>:attempt-3` stop (per-executor)

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
