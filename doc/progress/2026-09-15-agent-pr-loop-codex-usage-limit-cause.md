# 2026-09-15 — agent-pr-loop: the codex usage limit was reported as a models-cache error for 12 days

## Conclusion

`scripts/agent_pr_loop.py` reported the wrong cause for every codex failure
since 2026-09-03 and, because the wrong cause matched no non-retryable
marker, re-spawned codex every 300 s instead of recording a quota block.
Two one-line defects, both fixed here with regression tests built from the
verbatim `status.json` output:

1. `_exec_failure_cause` returned the FIRST line of the subprocess output.
   For codex that line is a library tracing log
   (`2026-09-15T22:05:44Z ERROR codex_models_manager::cache: failed to load
   models cache: missing field base_instructions`) that codex prints and
   then recovers from (it re-fetches and rewrites `~/.codex/models_cache.json`
   on every run — the file's mtime advances each cycle). The real verdict,
   `ERROR: You've hit your usage limit ... try again at Oct 3rd, 2026 4:00 PM`,
   sat 15 lines further down the same stderr.
2. `NON_RETRYABLE_MARKERS` carried `"usage limit reached"`; codex says
   `"hit your usage limit"`. Even with the right line the cause would not
   have been classified as non-retryable.

Consequence of (1)+(2) together: `main()` raised `RuntimeError` on the
codex-review step every cycle, so the claude-review, fix and BOTH merge
stages never ran, and the durable log carries 2,583 copies of the
misleading cause `[VERIFIED: grep -c base_instructions
logs/agent_pr_loop/launchd_stderr.log = 2583; first occurrence line 13071
= 2026-09-03T15:02:07Z]`. The DEGRADED page and `agent_inbox` both showed
`com.renquant.agent-pr-loop (last exit 1) [NO DOCUMENTED MEANING ...]`.

## Evidence (§4(b))

- `logs/agent_pr_loop/status.json` (finished_at 2026-09-15T22:05:46Z):
  `steps[codex-review].result.exec.rc = 1`, stderr as quoted in
  `tests/test_agent_pr_loop_quota_block.py::CODEX_USAGE_LIMIT_STDERR`
  (verbatim). Top-level `error` = the tracing line. `[VERIFIED]`
- `logs/agent_pr_loop/agent_quota_block.json` = `{}` — no block was ever
  recorded for codex. `[VERIFIED]`
- `~/.codex/models_cache.json`: both models carry `base_instructions`;
  `client_version 0.142.5`, `fetched_at 2026-09-15T22:05:46Z` — rewritten by
  the very run that logged the load error. The cache line is not the cause.
  `[VERIFIED]`
- Anti-vacuity: the four new tests FAIL against `origin/main`'s script
  (`4 failed, 1 passed` with `SCRIPT` pointed at the unpatched copy) and
  PASS against this branch (`36 passed`, both loop test files). `[VERIFIED]`

## Change

- `_exec_failure_cause`: collect every non-empty line of
  stdout_full/stdout/stderr; return (1) the first line carrying a
  non-retryable marker, else (2) the first line starting with `ERROR:`,
  else (3) the first line that is not a structured tracing-log line
  (`_TRACING_LOG_LINE`), else the first line. The 2026-08-11 claude shape
  (actionable line on stdout, empty stderr) is unchanged and still tested.
- `NON_RETRYABLE_MARKERS` += `"hit your usage limit"`.
- Tests: cause extraction on the verbatim codex stderr; the marker; a
  tracing line followed by a Traceback stays retryable and names the
  Traceback; a single tracing line is still reported (never an empty
  cause); a `main()`-level test that the codex cap degrades the cycle,
  records the usage-limit cause, and still runs both merge stages.

## What this does NOT do

- It does not restore codex. The quota resets 2026-10-03 16:00 PT per the
  CLI's own message; nothing merges before then. After deploy the loop
  will re-probe codex hourly instead of every 300 s, run the claude review
  and both merge stages each cycle, and `status.json` / the DEGRADED page
  will name the usage limit.
- It does not touch the ack ledger; the orchestrator-side ack for
  `agent-pr-loop` exit 1 (until the reset) is a separate PR.

## Deploy

The loop runs from the live umbrella checkout (`scripts/agent_pr_loop.py`
via launchd `com.renquant.agent-pr-loop`, RunAtLoad); it is live on the
next `git pull --ff-only` of the umbrella after merge — no job change.
