# Agent automation — v2 design (loop closure)

**Status**: design proposal · 2026-06-03 · supersedes nothing (extends
[`agent-automation.md`](agent-automation.md))
**Scope**: all 13 renquant repos
**Trigger**: user observation 2026-06-03 — "目前这个系统并不自动化，
需要我不停地告诉 claude 和 codex 去 review fix approve merge PR".
**Decision needed**: §6 — `Auto-merge stays manual` (today's design)
vs. relax to `auto-merge under gated conditions`. v2 makes this an
explicit operator choice rather than a hardcoded "never".

---

## 1 · TL;DR

The v1 design ([`agent-automation.md`](agent-automation.md)) shipped
three reusable workflows (G1 attribution / G2 review / G3 auto-fix) and
they fire in production today —
[verified 2026-06-03 06:46–06:49Z runs on origin against
`feat/track-b-retrain-bundle` and `codex/delegate-scheduled-live-bridge`].

But the user-facing experience is **still 6 manual touchpoints per PR**.
Today's evidence (PRs #154 and #155 in this session):

| Manual step | Evidence |
|---|---|
| 1. Operator types "review PR X" | Codex review on #154 / #155 came from operator-driven CLI run, posted under `hallovorld` not a bot user, **state `COMMENTED`** despite HIGH/MED findings |
| 2. Operator types "fix codex comments on PR X" | This was the exact words the user used 3× this session; G3 auto-fix gate requires `CHANGES_REQUESTED` review state OR `@<agent> fix` body prefix — neither was present |
| 3. Operator confirms fix is good | No auto-re-review-after-fix loop closure |
| 4. Operator types "approve" / "merge" | PR #154 merged manually at 06:50:02Z by `hallovorld` |
| 5. Operator coordinates paired umbrella + subrepo PRs | PR #146 (umbrella) ↔ PR #28 (renquant-pipeline) — no automated cross-repo gate |
| 6. Operator rebases before merge per §3.2 | No server-side enforcement; agents may forget |

v2 closes 5 of the 6 (auto-review trigger / auto-fix trigger / re-review
after fix / auto-approve when clean / auto-merge gated / cross-repo pair
coordination / server-side §3.2). Step 4 (merge) is the **one explicit
policy decision** in §6.

v2 is **incremental on top of v1**: same G1/G2/G3 templates, same labels,
same `agent:manual-hold` big red button. The new pieces are five small
workflows (`agent-review-classify`, `agent-auto-approve`,
`agent-auto-merge`, `agent-paired-merge-gate`, `agent-pre-merge-rebase`)
plus one default-label policy. No teardown of v1 surface.

---

## 2 · What automation already exists (v1 reality check)

Sources:
[`doc/ops/agent-automation.md`](agent-automation.md),
`.github/workflows/_agent-{review,fix,attribution-check}-template.yml`,
`.github/workflows/agent-{review,autofix,attribution-check}.yml`,
[`doc/ops/agent-automation-rollout-checklist.md`](agent-automation-rollout-checklist.md).

| Capability | Trigger | Surface today | Working? |
|---|---|---|---|
| G1 attribution check | `pull_request: opened/edited/synchronize/labeled/unlabeled` | `.github/workflows/agent-attribution-check.yml` → `_agent-attribution-check-template.yml` | ✅ Verified firing on origin |
| G2 auto-review (Claude + Codex) | `pull_request: opened/synchronize/reopened/ready_for_review` (skip `doc/**`) | `agent-review.yml` → `_agent-review-template.yml`, both agents in parallel | ✅ Verified firing on origin |
| G3 auto-fix | `pull_request_review: submitted` OR `issue_comment: created` | `agent-autofix.yml` → `_agent-fix-template.yml` | ⚠️  Gates rarely match — see §3 |
| Cost gates | `concurrency: cancel-in-progress`, `changed_files < 100`; `paths-ignore` removed after the 2026-06-03 fail-closed review gate because required review statuses must emit on every PR | All three workflows | ✅ |
| Per-executor attempt counter (3-strike) | Labels `agent:fix:<name>:attempt-{1,2,3}` | `_agent-fix-template.yml` | ✅ (when G3 fires) |
| `agent:manual-hold` universal stop | Label check | All workflows | ✅ |
| Loop guard (skip own agent's PRs) | Label check `agent:<self>` | G2 template | ✅ |
| Trust check (actor ∈ {owners, members, write-collaborators}) | `gh api collaborators/<actor>/permission` | G3 template | ✅ |
| Shell-injection-safe event-body handling | `env:` indirection, no `${{ }}` in `run:` for user-controlled fields | G3 template | ✅ |
| Cross-event PR-context resolver | Handles `pull_request_review` vs `issue_comment` shape diff | G3 template `Resolve PR context` step | ✅ |
| Per-agent secret gates | API-key absence → noisy `skip`, not workflow fail | G3 template `secret_gate` step | ✅ |

---

## 3 · Why "用户不停地告诉" — the 6 gaps in concrete form

### 3.1 · Gap 1 — Codex review state is `COMMENTED`, G3 expects `CHANGES_REQUESTED`

`_agent-fix-template.yml:174–183`:

```bash
if [ "$REVIEW_STATE" = "CHANGES_REQUESTED" ]; then
    echo "run=true" >> "$GITHUB_OUTPUT"
elif [[ "$REVIEW_BODY" == "$AT_MENTION"* ]]; then
    echo "run=true" >> "$GITHUB_OUTPUT"
else
    echo "no matching trigger; skipping"
fi
```

PR #154 evidence (`gh pr view 154 --json reviews`): three codex reviews,
**all `state == "COMMENTED"`** despite HIGH/MED findings. Codex's prompt
today doesn't tell it to use `CHANGES_REQUESTED` for blocking findings —
so by default it `COMMENTED`s, and G3 never auto-fires.

Same for PR #155 (1 codex review, COMMENTED, with HIGH paper-broker
finding).

**Fix options** (v2 implements both):

(a) **Server-side classifier** (`agent-review-classify.yml`): a small
workflow that fires on every `pull_request_review:submitted`, reads the
review body, greps for `\bHIGH\b` or `\bMED\b` (the structured
severity tags both agent review prompts emit), and if present and the
review state is `COMMENTED`, **adds the `agent:fix:<author-agent>`
label and dispatches `agent-autofix.yml` via
`workflow_dispatch`**. Bridges the COMMENTED → CHANGES_REQUESTED gap
without forcing every codex review to be a blocker.

(b) **Prompt update**: review prompt explicitly maps severity →
review state (HIGH or MED → `CHANGES_REQUESTED`, LOW or
nit → `COMMENTED`, no findings → `APPROVE`). One-line addition to
`.github/agent-review-prompt.md`; covers the next generation of
reviews without requiring (a) to grep.

(a) is the bridge for already-running reviews; (b) is the prompt-side fix.
Ship both; (a) becomes a no-op when (b) is consistently honored.

### 3.2 · Gap 2 — `agent:fix:<name>` opt-in label is not auto-added

`_agent-fix-template.yml:142–146`:

```bash
if ! echo "$LABELS" | grep -q "\"agent:fix:${AGENT}\""; then
    echo "PR lacks agent:fix:${AGENT}; skipping"
    exit 0
fi
```

PR #154 had only `agent:claude`. No `agent:fix:claude`, no
`agent:fix:codex`. Even if codex review state had been `CHANGES_REQUESTED`,
G3 would still skip.

**Fix**: on PR open, if the PR carries an `agent:<X>` authorship label,
auto-add `agent:fix:<X>` (same-agent self-fix) AND `agent:fix:<other>`
(cross-agent fix invitation, per §2.1 split). User removes one or both
to opt out of either side.

Implementation: `agent-default-labels.yml` workflow on
`pull_request:opened` + `labeled` events. Idempotent.

### 3.3 · Gap 3 — No auto-re-review after fix push

After G3 force-pushes a fix, `pull_request:synchronize` fires, which
already triggers `agent-review.yml`. **This already works**. The codex
re-review that landed on PR #154 (cf48d51 → 5dc5ec8 → c59e21d chain)
demonstrates the loop is closing.

But: today the re-review again returns `COMMENTED`. If the new state
has no findings, the review prompt should output `APPROVE`. Same root
cause as Gap 1 — the review prompt doesn't map state → severity.

**Fix**: same prompt update as §3.1 (b).

### 3.4 · Gap 4 — No auto-approve, no auto-merge

[`agent-automation.md §1`](agent-automation.md#1-goal):
> **Non-goals**: Replacing human approval. Auto-merge stays manual per
> `CLAUDE.md §3.1`.

[`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict):
> 3. Self-merge allowed (solo dev) but the PR IS the audit surface.
>    Use `gh pr merge --merge|--squash <PR#>`.
> 5. After verbal approval: `gh pr merge --merge --delete-branch`

This is a **deliberate design decision**, not an oversight.

User's ask "make it more automatic" makes this decision relevant
again. v2 proposes **gated auto-merge** that respects the spirit of
the §3.1 audit-surface requirement (the PR still exists, still gets
reviewed, still gets attributed) while removing the operator-typing
loop:

| Gate | Required? | Why |
|---|---|---|
| All `agent:*` reviewers have `APPROVE` review on the latest head | YES | Replaces "verbal approval" — `APPROVE` is the audit trail |
| Branch protection `required status checks` all green | YES | Server-side existing gate |
| Author is an `agent:*` label (NOT human) | YES | Humans still merge their own PRs (§3.1 ergonomics) |
| No `agent:manual-hold` | YES | Universal stop |
| Branch up to date with base (§3.2 rebased) | YES | Auto-rebase if needed (§3.5 / Gap 6) |
| No `CHANGES_REQUESTED` reviews on latest head | YES | Self-evident |
| PR carries `agent:auto-merge` label OR repo default `agent.auto_merge.default = true` | YES | Explicit per-PR or per-repo opt-in |
| No `paired:<sister-pr>` label OR the sister PR is already merged | YES | §3.5 / Gap 5 cross-repo gate |

`agent-auto-merge.yml` triggers on `pull_request_review:submitted`
with state=`APPROVED`. If all gates pass, runs `gh pr merge --squash
--delete-branch`. This is opt-in per PR (default OFF) or per repo
(`agent.auto_merge.default`); the operator chooses at any scope.

**Decision needed (§6)**: keep "auto-merge stays manual" as v1 spec,
or relax to "auto-merge under the 8 gates above"?

### 3.5 · Gap 5 — Cross-repo paired PRs are unmanaged

When the same change lives in umbrella + subrepo (Phase 1 byte-equivalent
mirror per [`CLAUDE.md §3.5`](../../CLAUDE.md#35--multi-repo-code-placement)),
both PRs must exist and merge in the right order. Today this is fully
manual: PR #146 (umbrella) + PR #28 (renquant-pipeline) were opened in
sequence by the operator and merged manually.

v2 adds two new labels and one workflow:

| Label | Meaning |
|---|---|
| `paired:<owner>/<repo>#<number>` | Sister PR's full reference |
| `paired-canonical` | This PR is the source of truth in the pair (merge first) |
| `paired-mirror` | This PR mirrors the canonical; wait for canonical to merge |

Workflow `agent-paired-merge-gate.yml` on
`pull_request:labeled,opened,synchronize` + `check_run:completed`:

```text
if labels include `paired-mirror` AND `paired:<X>`:
    fetch sister PR X
    if sister PR is OPEN  → post "🔒 waiting on <X>" comment + add agent:manual-hold
    if sister PR is MERGED → remove agent:manual-hold, allow auto-merge
    if sister PR is CLOSED (not merged) → keep agent:manual-hold, post warning
```

Canonical pair workflow:
1. Operator opens canonical PR with `paired-canonical` + `paired:hallovorld/<other-repo>#<TBD>`.
2. Operator opens mirror PR with `paired-mirror` + `paired:hallovorld/<canonical-repo>#<N>`.
3. Auto-merge per §3.4 fires on canonical first.
4. Mirror's paired-merge-gate detects canonical merged → unhold → mirror auto-merges.
5. Both branches deleted.

Out of scope for v2: **auto-OPENING the sister PR** from one side. That
needs a separate "mirror generator" that walks the diff and lifts the
byte-equivalent piece to the other repo. Doable but bigger; deferred to
v3.

### 3.6 · Gap 6 — §3.2 sync mandate is not enforced server-side

`CLAUDE.md §3.2` mandates `git fetch origin` + `git rebase origin/main`
before opening AND before declaring merge-ready. Agents are supposed to
remember. They sometimes don't, leading to stale PRs.

v2: `agent-pre-merge-rebase.yml` on `pull_request:synchronize` +
`workflow_dispatch` (called by `agent-auto-merge.yml`).

```text
if PR is behind base by ≥ 1 commit:
    attempt `git fetch origin && git rebase origin/<base>`
    if clean → `git push --force-with-lease`
    if conflict → label `agent:rebase-conflict`, post comment, stop
```

`agent:rebase-conflict` is a stop label that blocks auto-merge. Human
resolves the conflict locally and removes the label.

### 3.7 · Gap 7 — Codex review identity (posts as `hallovorld`, not bot)

Today codex reviews post under `hallovorld`'s identity because the
G2 wrapper uses the operator's PAT (`AGENT_GIT_PUSH_TOKEN`). The
attribution v1 trust check passes because `hallovorld` is the
owner, so this isn't a security issue — but it muddies the audit
surface (was this review human or agent?).

Fix: separate `CODEX_REVIEW_BOT_TOKEN` (dedicated bot identity, scoped
read+pr-write) for G2 review-posting, distinct from
`AGENT_GIT_PUSH_TOKEN` (write for G3 fix-pushing). Cosmetic but it
makes the audit trail honest.

Out of v2 scope (no automation logic change); flag for ops.

### 3.8 · Gap 8 — No loop budget, no per-PR cost cap

v1 has per-executor attempt-3 strike. It does NOT have:
- Per-PR cost cap (Claude + Codex spend on a single PR — can run away if
  G3 keeps producing review-able diffs)
- Loop budget on review → fix → review cycle count

v2: track `agent-spend-cents` as a PR label, increment after each G2/G3
run (estimate from token counts), stop all agent workflows when
threshold hit. Label `agent:cost-cap` indicates the budget was reached.

---

## 4 · The v2 target loop

```text
                                Human (operator)
                                       │
                                  opens PR
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-default-labels                    │  ← v2 NEW
              │    add `agent:fix:claude` +              │
              │        `agent:fix:codex`                 │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-attribution-check (v1)            │
              │  agent-review × Claude + Codex (v1)      │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-review-classify                   │  ← v2 NEW
              │    if review COMMENTED with HIGH/MED →   │
              │       dispatch agent-autofix             │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-autofix (v1)                      │
              │    fix → force-push                      │
              │    attempt-N label increments            │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-review × Claude + Codex (v1)      │
              │    (fires on push synchronize)           │
              └──────────────────────────────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                 review APPROVE                review COMMENTED/CHANGES_REQUESTED
                          │                         │
                          ▼                         └─► loop back to autofix
              ┌──────────────────────────────────────────┐
              │  agent-pre-merge-rebase                  │  ← v2 NEW
              │    if behind base → rebase + push        │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-paired-merge-gate                 │  ← v2 NEW
              │    if `paired-mirror` and sister open    │
              │       → hold; else release               │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
              ┌──────────────────────────────────────────┐
              │  agent-auto-merge                        │  ← v2 NEW (opt-in)
              │    8 gates per §3.4 → gh pr merge        │
              └──────────────────────────────────────────┘
                                       │
                                       ▼
                                   MERGED
```

Stop knobs that interrupt the loop at any node:

- `agent:manual-hold` — universal.
- `agent:fix:<name>:attempt-3` — that executor halts.
- `agent:cost-cap` — all workflows halt for this PR.
- `agent:rebase-conflict` — auto-merge blocked, needs human.
- `agent:auto-merge` absent AND repo default OFF → loop ends at the
  approved-but-not-merged state; operator merges manually (current §3.1
  behavior).

---

## 5 · Phased rollout

Phase A is the minimal change that closes 80% of "user 不停地告诉".
Phase B adds the merge decision. C/D follow once A+B are stable.

### Phase A — close the review→fix loop (1 PR, low risk)

Single PR adds:
- `.github/workflows/agent-default-labels.yml` (umbrella + fan-out)
- `.github/workflows/agent-review-classify.yml` (umbrella + fan-out)
- Update `.github/agent-review-prompt.md` to map severity → review state
  (HIGH/MED → CHANGES_REQUESTED, LOW/nit → COMMENTED, none → APPROVE)
- Update `.github/agent-fix-prompt.md` to require severity tags

Test on `renquant-model` pilot first; fan-out via existing
`scripts/fan_out_agent_automation.sh` pattern.

**Acceptance**: a future PR with HIGH/MED findings from Codex review
auto-triggers G3 without operator intervention.

### Phase B — gated auto-merge (decision-dependent)

**Requires §6 decision first.**

If go: add `.github/workflows/agent-auto-merge.yml` +
`agent-pre-merge-rebase.yml`. Default OFF (no `agent.auto_merge.default`,
no `agent:auto-merge` label). Per-PR opt-in for the first 10 PRs to
calibrate.

If no-go: ship Phase A only; merge stays manual; that's a coherent place
to stop.

### Phase C — cross-repo pair gate

`.github/workflows/agent-paired-merge-gate.yml`. Manual sister-PR
linking (operator adds `paired:` labels at PR open time). Auto-sister-PR
generation deferred to v3.

### Phase D — cost cap + loop budget

`agent-spend-cents` label incrementer; `agent:cost-cap` stop label.
Worth doing only after Phase A+B fire enough to produce real cost data.

---

## 6 · Decision needed before Phase B

Today's v1 spec ([`agent-automation.md §1`](agent-automation.md#1-goal))
calls auto-merge a **non-goal**. The user's 2026-06-03 ask says the
manual merge step is one of the painful ones.

Three options:

**A. Keep v1 spec** — merge stays fully manual; v2 ships only Phase A
(review→fix loop closure). User still types `gh pr merge` per PR but
no longer types "fix codex comments". This closes 80% of the burden.

**B. Adopt gated auto-merge** (§3.4) — 8 gates, opt-in per PR or per
repo, `agent:manual-hold` and `agent:auto-merge` absent both halt
the path. Audit surface preserved (PR exists, attributed, approved
reviews are the audit record). This closes 100% of the burden but
relaxes §3.1's "verbal approval" line.

**C. Hybrid — auto-merge only for `agent:claude` AND `agent:codex`
mirror PRs, NOT for human-authored or new-feature PRs**. Surfaces the
risk-tier distinction (subrepo mirrors of an already-approved umbrella
PR are by construction byte-equivalent and lower-risk than novel
agent changes).

Recommend **C** as default for a 4-week trial, with `agent:auto-merge`
label as the explicit per-PR opt-in for non-mirror PRs. Easiest to
revert; cleanest audit story; matches today's risk profile (mirror PRs
are the highest-frequency manual step).

---

## 7 · Safety preservation (what does NOT change)

All v1 safety gates carry forward verbatim:

- §3.1 PR-based workflow — strict; v2 only changes WHO presses merge,
  not whether a PR exists.
- §3.2 sync mandate — server-side enforced by §3.6 pre-merge-rebase.
- §3.5 multi-repo code placement — server-side enforced by §3.5
  paired-merge-gate.
- §7.4 promotion gating — v2 doesn't touch live-flip decisions; those
  remain `agent:manual-hold` territory by default.
- All current cost gates (concurrency, changed_files < 100; no review `paths-ignore`
  because branch protection requires a review status on every PR)
  and per-executor attempt-3 limits.
- `agent:manual-hold` halts the entire v2 loop at any node.

---

## 8 · What this PR does NOT do

- No workflow YAML changes (those land in Phase A's follow-up PR).
- No `CLAUDE.md` §3.1 edits (those land if §6 decision is B or C).
- No fan-out to subrepos (those go through the existing
  `scripts/fan_out_agent_automation.sh` pattern after Phase A
  template lands).

This is **design** + **decision request**, not implementation.

---

## 9 · Open questions

1. Phase B decision (A / B / C in §6) — needs operator answer.
2. Should `agent:cost-cap` halt only LLM workflows (G2/G3) or also
   merge gate (Phase D)? Recommend: halt G2/G3, allow merge through
   (we shouldn't have to spend more money to merge an already-approved
   PR).
3. `agent-paired-merge-gate.yml` (§3.5) currently relies on the
   operator labelling sister PRs by hand. Auto-generating sister PRs
   is plausible (walk byte-equivalent files in subrepo vs umbrella,
   propose mirror diff) but adds complexity. Defer to v3?
4. Should `agent-auto-merge` use `--squash` or `--merge`? Today the
   PR template language is `gh pr merge --merge --delete-branch`.
   Recommend honoring the PR's body if it specifies; default to
   `--merge` to match §3.1 example.

---

## 10 · Migration plan (if Phase A+B adopted)

1. Land THIS PR (design only).
2. Phase A PR: umbrella templates + prompt updates + pilot fan-out.
3. Wait 5 PR cycles. Measure: how often does HIGH/MED → autofix fire,
   how often does APPROVE come back without operator intervention.
4. Phase B PR (only if §6 chose B or C): `agent-auto-merge.yml` +
   `agent-pre-merge-rebase.yml`, default OFF, opt-in for the first 10
   PRs.
5. Phase C PR: paired-merge-gate.
6. Phase D PR: cost cap.
7. Update [`agent-automation.md`](agent-automation.md) §1 non-goals
   list if Phase B lands.
8. Update [`CLAUDE.md §3.1`](../../CLAUDE.md#31--pr-based-workflow--strict)
   point 5 if Phase B lands.

---

## Appendix · Today's evidence (PR #154 + PR #155, 2026-06-03)

PR #154 (`docs/qp-136-review-historical-sector-map`):
- Opened 06:12:24Z by `hallovorld`, label `agent:claude`.
- Codex review #1 06:18:56Z, **state COMMENTED**, 2 MED + 1 LOW.
- G3 did not fire (no `agent:fix:claude` label, review state COMMENTED).
- Operator typed in chat: "fix all comments from codex in PRs created by you".
- Claude pushed fix `5dc5ec8` at 06:35Z.
- Codex re-reviewed at 06:39Z, **state COMMENTED again**.
- Codex pushed its own counter-fix `c59e21d` at 06:45Z (autofix
  via @-mention or label addition).
- Final merge: 06:50:02Z by `hallovorld` manually.

PR #155 (`docs/subrepo-runtime-refresh-runbook`):
- Same pattern. Codex review COMMENTED with HIGH paper-broker finding.
- Operator manually triggered the fix.

The exact words "fix codex comments on PR X" / "merge" appeared **5
times** in the session transcript. Phase A removes 3 of those;
Phase B+C removes the remaining 2.

---

Agent-Origin: Claude

🤖 Generated with [Claude Code](https://claude.com/claude-code)
