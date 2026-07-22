# Progress — agent PR loop: self-inflicted stdout truncation killed every cycle

**Date:** 2026-07-22
**Goal:** GOAL-5 P0 (daily-run reliability — the automation surface itself).
**Type:** bug fix + regression tests. Local automation only; no production path touched.

## STATUS
`com.renquant.agent-pr-loop` has been firing on schedule (every 300s) but
**failing before it did any work**: every cycle died at the first
`repos agent --as codex --workflow review --repo all` call. No PR was reviewed,
fixed, or merged by the loop. Fixed here; the failing call now parses.

## WHAT
`scripts/agent_pr_loop.py`:
- `_run()` returned `"stdout": proc.stdout[-8000:]` — a log-sized **tail** — and
  `_orch_json()` fed exactly that string to `json.loads()`. Any control-plane
  payload over 8000 chars was therefore parsed mid-document.
- `_run()` now also returns `"stdout_full"` (untruncated); `_orch_json()` parses
  that, falling back to `"stdout"` when absent.
- `_write_status()` strips `stdout_full` recursively before persisting, so
  `logs/agent_pr_loop/status.json` stays bounded (that key never lands on disk).
- `_orch_json()` now raises a diagnostic `RuntimeError` naming the payload length
  and stdout head instead of surfacing a bare `json.JSONDecodeError` — the old
  error text (`Expecting value: line 1 column 1 (char 0)`) pointed nowhere.

## WHY/DIR
The `repos agent --repo all` plan bundle is **16652 chars** today (10 repos,
per-repo instructions + queue + violation rows). It crossed the 8000-char tail
budget at some point and the loop has been dead ever since: the tail happened to
start mid-string, so `json.loads` failed on char 0. The truncation is correct for
*logging* and wrong for *parsing* — so the fix separates the two rather than
raising the cap, which would only move the same cliff further out.

Failure histogram from `logs/agent_pr_loop/launchd_stderr.log` (6145 error lines
total) shows this one bug dominating: `Expecting value: line 1 column 1 (char 0)`
×4640, plus 600+ `Extra data: line 1 column N` — the same truncation landing at
different offsets.

## EVIDENCE
- **Root cause reproduced exactly.** Captured the real command's stdout
  (16652 bytes, `rc=0`, valid JSON standalone), then `json.loads(text[-8000:])`
  → `Expecting value: line 1 column 1 (char 0)` — character-for-character the
  error in `status.json` / launchd stderr. `[VERIFIED]`
- **Live status before the fix:** `status.json` `ok:false`, steps stop at
  `agent-identity → repos-sync → short-term-state-bootstrap`; `codex-review` and
  everything after never appended. Two consecutive cycles (16:35Z, 16:40Z)
  identical. `[VERIFIED]`
- **End-to-end after the fix**, patched module against the live control plane:
  `rc 0, stdout tail len 8000, full len 16652` → `PARSE OK — n_repos 10,
  queue_total 6`. The exact call that killed every cycle now yields a real queue.
  `[VERIFIED]`
- **Tests:** `tests/test_agent_pr_loop.py` — 20 passed. The 4 new tests fail
  against the pre-fix script (verified by stashing the fix and re-running) and
  pass after, so they pin the regression rather than restating the fix.

## NEXT
- **Merged ≠ deployed here.** launchd runs `scripts/agent_pr_loop.sh` out of the
  live umbrella checkout, which currently sits on `feat/config-artifact-path-gate`
  (dirty) — so `repos sync` leaves it fetch-only and this fix does NOT reach the
  running loop on merge. Landing it needs an operator-authorized sync of the live
  tree to `main` (ask-first per the landing-actions rule). `queue_total 6` is
  waiting behind that.
- Watch the first post-merge cycle: `status.json` should reach `ok:true` with
  `codex-review`/`claude-review`/`*-fix`/`*-merge` steps present. If a later step
  fails, the new error text will now name the offending payload instead of a bare
  decode error.
- Follow-up worth considering (not in this PR): the loop overwrites a single
  `status.json`, so a failure mode is only visible until the next tick — there is
  no history to detect "silently failing for thousands of cycles" from (4640 is
  ~16 days at a 5-minute cadence, though the log carries no timestamps to prove
  contiguity). A one-line append to
  a JSONL health log would make this class of rot self-announcing.
