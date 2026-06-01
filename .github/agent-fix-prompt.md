# Agent fix prompt — RenQuant default

You are addressing review findings on an open PR you (or another agent
identified as `agent:<name>`) authored. The reviewer's comments are
appended at the bottom of this prompt.

## What to do (in this order)

1. **Read every finding** in the review. Don't skip or reorder.

2. **For each finding**: identify the smallest concrete change that
   resolves it. Read the surrounding code first (`Read`, `git log`),
   then `Edit` / `Write`. NEVER change unrelated code in the same
   pass — keep blast radius small.

3. **Run tests** that cover the changed code path. Each renquant repo
   has a `make test` or `pytest tests/` entry point. If a test exists
   that targets the changed area, run it. If no test exists for what
   you changed, ADD ONE per CLAUDE.md §7.1 (every fix has a paired test).

4. **Leave your edits in the working tree — DO NOT commit or push.**
   The wrapping G3 workflow's "Commit + push fix" step is the ONLY
   commit-and-push authority. It detects your working-tree changes,
   signs the commit with the canonical `Co-Authored-By` trailer for
   your agent identity, and pushes with `--force-with-lease`. If you
   commit yourself, the workflow either double-commits (annoying) or
   silently skips its push (worse — locally committed but never pushed).

## What you MUST NOT do

- **No commits, no pushes** — see step 4. The wrapping workflow owns
  those operations. Your job is the edit + test phase only.
- **No drift fixes**: only address findings explicitly in this review.
  Adjacent code that "could also use cleanup" goes in a separate PR.
- **No untested changes**: every Edit/Write must have a passing test
  that exercises the new behavior, or a CLAUDE.md §7-justifiable
  reason for none.
- **No silent skips**: if a finding can't be addressed (out of scope,
  requires upstream change, etc.), name it in your run output so the
  wrapping workflow's summary comment surfaces it.
- **No new dependencies** unless the reviewer specifically requests
  one. Adding deps is its own PR.

## Tools available

`Bash`, `Edit`, `Write`, `Read`, `gh`, `git` (read-only operations
like `log`, `diff`, `status` — NOT `commit` or `push`). The wrapping
workflow posts a summary comment automatically.

---

## Reviewer feedback to address

(Auto-appended by the workflow.)
