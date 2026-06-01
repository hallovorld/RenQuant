# Claude × GitHub automation — design doc

**Status**: proposal · awaiting approval to ship pilot
**Scope**: all 13 renquant repos (umbrella + 12 subrepos)
**Owner**: Claude (the assistant) + user
**Last updated**: 2026-05-31

---

## 1 · Goal

Automate three workflows that currently require manual operator action:

| # | Workflow | Today | Target |
|---|---|---|---|
| **G1** | Claude-authored PRs are clearly attributed | Manual `🤖 Generated with Claude Code` footer; sometimes forgotten | Every branch + PR + commit Claude touches stamps the attribution at the source — enforced, not relied on |
| **G2** | PRs opened by *other* contributors get a Claude review | Operator pings Claude in chat; Claude reads the PR by URL | Claude posts a structured review on every non-Claude PR within minutes of `opened` / `synchronize` |
| **G3** | When a reviewer leaves findings on *Claude's* PR, Claude fixes them | Operator pastes the comment URL into chat; Claude reads, fixes, pushes | Claude auto-fixes addressable findings on receipt of `pull_request_review:CHANGES_REQUESTED` (or `@claude` mention), with safety gates |

**Non-goals**:
- Replacing human approval. Auto-merge stays manual per [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict).
- Auto-actioning on *every* event — see §6 safety gates.
- Cross-repo coordination (this doc is per-repo wiring; the multi-agent collaboration mandate [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict) remains the source of truth for sync discipline).

---

## 2 · Building blocks

| Block | Provided by | Notes |
|---|---|---|
| Claude Code GitHub Action | `anthropics/claude-code-action` (official) | Action runs Claude on the PR diff + repo context; modes `review`, `fix`, `comment`. |
| Claude Code CLI hook system | `claude-code` (this tool) | `PostToolUse` hooks fire after `Bash` calls — useful for G1 enforcement at branch/PR creation. |
| `gh` CLI | already required by [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) | PR creation + commenting + review surface. |
| `git` notes / commit trailers | git native | Per-commit attribution; survives squash-merge. |
| GitHub repo / org secrets | GitHub | `ANTHROPIC_API_KEY` storage. |
| GitHub Actions concurrency / paths-ignore | GitHub | Cost control + dedup. |

No new infra to host. No webhook server. No DB. All stateless.

---

## 3 · Per-goal design

### 3.1 · G1 — Mandatory Claude attribution

Three layers, defense-in-depth so a forgotten one doesn't drop attribution:

**Layer A — `CLAUDE.md §3.x` rule** (every repo, inherits from umbrella canon):

> All branches, commits, and PRs that Claude opens MUST include the canonical attribution. Format:
> - **Branch name**: prefixed `claude/` (e.g. `claude/fix-foo-bar`) when wholly Claude-authored; standard prefix when human-collaborated.
> - **Commit message**: ends with `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` (already in [`CLAUDE.md §3.x`](../../CLAUDE.md) Harness).
> - **PR body**: ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)` (already in [`CLAUDE.md §3.x`](../../CLAUDE.md) Harness).

**Layer B — `PostToolUse` hook on `Bash`** (in `.claude/settings.json`):

```jsonc
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "scripts/check_claude_attribution.sh"
      }]
    }]
  }
}
```

The script inspects `$CLAUDE_TOOL_INPUT` (the bash command Claude just ran). If it was `gh pr create` or `git commit` without the canonical attribution string, it writes a warning to stderr (which Claude reads as feedback and self-corrects on next turn). Optional: exit non-zero to block.

**Layer C — server-side workflow check** (`.github/workflows/attribution.yml` on PR open):

```yaml
name: attribution-check
on:
  pull_request:
    types: [opened, edited, synchronize]
jobs:
  check:
    if: github.actor == 'claude-code-bot' || contains(github.event.pull_request.body, 'Claude')
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: |
          set -e
          # PR body must contain canonical footer
          if ! gh pr view ${{ github.event.pull_request.number }} --json body -q .body | grep -q "Generated with .*Claude Code"; then
            echo "::error::PR body missing Claude attribution footer"
            exit 1
          fi
          # all new commits on branch must have Co-Authored-By trailer
          git log --format=%B origin/main..HEAD | grep -q "Co-Authored-By: Claude Opus" || {
            echo "::error::Claude commits missing Co-Authored-By trailer"
            exit 1
          }
        env:
          GH_TOKEN: ${{ github.token }}
```

Layer A → social/convention, Layer B → self-feedback, Layer C → enforced gate. Layer C only blocks when the PR is Claude-attributed in the first place — human-authored PRs bypass it.

### 3.2 · G2 — Auto-review PRs from non-Claude authors

Single GitHub Actions workflow, one per repo, triggered on `pull_request: [opened, synchronize, reopened]`. Gated to skip self-review (no infinite loop).

`.github/workflows/claude-review.yml`:

```yaml
name: claude-review
on:
  pull_request:
    types: [opened, synchronize, reopened]
    # Cost control: skip docs-only changes
    paths-ignore:
      - 'doc/**'
      - '**/*.md'
      - '.github/**'
jobs:
  review:
    # Skip Claude's own PRs (no self-review loop) and draft PRs.
    if: |
      github.event.pull_request.draft == false &&
      github.actor != 'claude-code-bot' &&
      !contains(github.event.pull_request.labels.*.name, 'no-claude-review')
    runs-on: ubuntu-latest
    timeout-minutes: 20
    permissions:
      contents: read
      pull-requests: write
    concurrency:
      # Cancel in-flight runs when new commits land — review only the latest.
      group: claude-review-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          mode: review
          model: claude-opus-4-7  # or sonnet for cost reduction
          # Surface the project's review canon (CLAUDE.md §7 engineering principles).
          prompt_file: .github/claude-review-prompt.md
```

`.github/claude-review-prompt.md` (per-repo customization point — most repos can re-export the umbrella's):

```markdown
You are reviewing a PR against this repository. Apply CLAUDE.md §7 engineering
principles (test invariants, sanity discipline, multi-measurement requirement,
single source of truth, data-flow safety, anti-decoration, audit discipline).

For each finding, post a PR comment with: severity (BLOCKER/HIGH/MED/LOW),
location (file:line), evidence (cite the actual code), and the smallest
concrete fix. Reference the CLAUDE.md rule that the finding violates.

Skip: style nits handled by linters, subjective preferences, hypothetical
problems. Focus on: actual bugs, missing tests, data leakage, dead code,
cargo-cult patterns.
```

### 3.3 · G3 — Auto-fix on reviewer findings on Claude's PRs

Triggered on `pull_request_review:submitted` AND `issue_comment:created`. Gated on explicit `@claude` mention in the review/comment body to prevent surprise auto-commits.

`.github/workflows/claude-autofix.yml`:

```yaml
name: claude-autofix
on:
  pull_request_review:
    types: [submitted]
  issue_comment:
    types: [created]
jobs:
  fix:
    # Two gates: (a) the PR was authored by Claude, (b) the review/comment
    # explicitly invites Claude with @claude.
    if: |
      github.event.pull_request.user.login == 'claude-code-bot' &&
      (contains(github.event.review.body, '@claude') ||
       contains(github.event.comment.body, '@claude'))
    runs-on: ubuntu-latest
    timeout-minutes: 30
    permissions:
      contents: write
      pull-requests: write
    concurrency:
      group: claude-autofix-${{ github.event.pull_request.number }}
      cancel-in-progress: false  # do not interrupt an in-flight fix
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.ref }}
          fetch-depth: 0
      - uses: anthropics/claude-code-action@v1
        with:
          anthropic_api_key: ${{ secrets.ANTHROPIC_API_KEY }}
          mode: fix
          # Push fix-commits to the PR branch with --force-with-lease,
          # reply on the original review thread with what changed.
          push_strategy: force-with-lease
          reply_on_thread: true
```

The `@claude` gate is the safety belt. To unconditionally auto-fix `CHANGES_REQUESTED` reviews, drop the gate — but expect noise from drive-by reviewer comments. Conservative default: require explicit invite.

---

## 4 · Cross-repo rollout

13 repos cross §3.1: umbrella `RenQuant` + 12 subrepos under `hallovorld/`.

Phased rollout to bound blast radius:

| Phase | Repos | Workflows shipped |
|---|---|---|
| **P0 — Pilot** | `renquant-model` only | G2 + G3 (lower-risk; private-ish workstream) |
| **P1 — Public protected** | The 11 server-side protected public repos | G1 attribution-check + G2 + G3 |
| **P2 — Private** | `renquant-model`, `renquant-state-backup` | Same as P1 |
| **P3 — Canon** | Umbrella `CLAUDE.md §3.x` | Adds Layer A rule; future repos inherit |

P0 lets us see Claude reviewing its own PRs on `renquant-model` for a week (with the self-review skip removed temporarily) — calibrates prompt quality before rolling to 13 repos.

---

## 5 · Costs & quotas

Rough envelope, Opus 4.7 prices:

| Workflow | Tokens per fire | $ per fire | Fires/day est | $/month |
|---|---|---|---|---|
| G2 auto-review | 30K–150K (depends on diff) | $0.05–$0.50 | 10–30 | $15–$450 |
| G3 auto-fix | 50K–300K (diff + repo context) | $0.10–$1.50 | 1–5 | $3–$225 |
| G1 attribution check | <1K | <$0.01 | every PR | <$5 |

**Mitigations**:
- `paths-ignore: doc/**` + `**/*.md` strips docs-only PRs (large chunk of churn).
- `concurrency: cancel-in-progress: true` for reviews — only the latest commit gets reviewed when a PR is pushed multiple times in quick succession.
- `model: claude-sonnet-4-6` for G2 cuts cost ~3x; Opus 4.7 stays for G3 (autonomy quality matters more there).
- `if: github.event.pull_request.changed_files < 100` skips megaPRs (a separate human-only review gate).

---

## 6 · Safety gates

| Risk | Gate |
|---|---|
| **Self-review loop** (Claude reviews its own PR, replies, infinite) | G2's `github.actor != 'claude-code-bot'` skip; G3's `@claude` mention gate |
| **Force-push race** (Claude auto-fix while user pushes) | `--force-with-lease` (already standard); `concurrency: cancel-in-progress: false` on G3 |
| **Runaway autonomy** | Manual merge always required (CLAUDE.md §3.1); `if: pull_request.changed_files < 100`; per-PR `no-claude-review` label opt-out |
| **Secret leakage** in PR diff | Claude action runs with `permissions: contents: read` for G2; G3 elevates to `contents: write` only on `@claude` invite |
| **Cost runaway** | Org-level Anthropic API spend alert; per-repo `paths-ignore`; `if: changed_files < 100` |
| **Hostile PR contributor** trying to invoke Claude | G2 already runs on every PR. G3's `@claude` gate also checks `pull_request.user.login == 'claude-code-bot'` — only Claude's own PRs are auto-fixable; outsiders can't trigger autofix on others' branches |
| **Misattribution** (Claude attributed when it shouldn't be) | Layer C gate fails the PR; human can override with `no-claude-attribution` label |

---

## 7 · Open questions

1. **Bot identity**: Use a dedicated `claude-code-bot` GitHub user (PAT-based) or a GitHub App? App is cleaner permission-wise; PAT is faster to set up. Pilot with PAT, migrate to App in P1.

2. **Cross-repo coherence**: Should G2 reviews check cross-repo invariants (e.g. byte-equivalence between umbrella and subrepo)? Probably not — that's expensive + better caught by dedicated CI tests. Keep G2 scope to the single PR's diff.

3. **Prompt versioning**: `.github/claude-review-prompt.md` lives in each repo. Drift across 13 repos is likely. Options: (a) symlink to umbrella, (b) `git submodule` of a shared prompts dir, (c) accept drift and reconcile at sweep cadence. Recommend (c) — sweep monthly via the same multi-repo sync pattern in [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict).

4. **Should G3 reviews fire on every CHANGES_REQUESTED?** Conservative default is `@claude` gate (explicit invite). Could relax once we trust the fix-quality. Track: % of auto-fixes that needed a human revert (target < 5%).

5. **Audit log**: Every Claude API call from CI should write an entry somewhere queryable (which PR, which mode, which commit, which findings). MVP: GitHub Actions logs are already queryable via `gh run list`. Phase 2: separate audit JSONL on a longer-retention store.

---

## 8 · Decision needed before shipping

| Q | Default if you don't pick | Where it bites |
|---|---|---|
| Bot identity: PAT or GitHub App? | **PAT** (start of P0) | Migration cost in P1 if we want stricter scoping |
| G3 trigger: `@claude` mention or unconditional on `CHANGES_REQUESTED`? | **`@claude` mention** | Operator has to type `@claude` to invoke; safer first month |
| G2 model: Opus 4.7 or Sonnet 4.6? | **Sonnet 4.6** (cost) | Lower-quality reviews; can upgrade per-repo if needed |
| Roll out to umbrella first or pilot on `renquant-model`? | **Pilot on `renquant-model`** (lowest blast radius) | Slower full rollout; better calibration |
| `no-claude-review` opt-out label vs always-on? | **opt-out via label** | Some repos may not want reviews (e.g. `renquant-state-backup`) |

---

## 9 · Next steps

1. **Approve this design** (or push back on specific sections).
2. **Pilot PR**: open against `renquant-model` adding `.github/workflows/claude-review.yml` + `.github/workflows/claude-autofix.yml` + `.github/claude-review-prompt.md` + `ANTHROPIC_API_KEY` secret. Watch one PR cycle.
3. **Calibrate prompt** based on review quality on the pilot.
4. **P1 rollout**: 11 cookie-cutter PRs against the public protected repos.
5. **Umbrella `CLAUDE.md §3.x` update** — codify the rule once the mechanism is proven.
6. **Phase 2 work**: separate audit JSONL, GitHub App migration, monthly prompt-drift sweep.

---

## Appendix · References

- Claude Code Action: https://github.com/anthropics/claude-code-action
- Claude Code hooks: https://docs.claude.com/claude-code/hooks
- [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict) — PR-based workflow canon (2026-05-30)
- [`CLAUDE.md §3.2`](../../CLAUDE.md#32--sync-from-remote-before-every-task--strict) — sync-from-remote mandate (2026-05-30)
- [`CLAUDE.md §3.4`](../../CLAUDE.md#34--pr-review-protocol) — PR review protocol (chat ↔ PR-comment contract)
