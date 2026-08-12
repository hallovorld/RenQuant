# 2026-08-12 — agent-pr-loop: name the failure, stop re-running into a spend cap

STATUS:   FIXED (2026-08-12). `tests/test_agent_pr_loop.py` +
          `tests/test_agent_pr_loop_quota_block.py` -> **27 passed**
          `[VERIFIED — .venv/bin/python -m pytest -q tests/test_agent_pr_loop.py
          tests/test_agent_pr_loop_quota_block.py]`.

WHAT:     Two changes to `scripts/agent_pr_loop.py`, both on the running path.
          (1) `_exec_failure_cause()` lifts the SUBPROCESS's own first output
          line into the RuntimeError, so the durable stderr log names the
          reason instead of the step. (2) An enumerated set of non-retryable
          conditions (spend cap / credit exhaustion) records a per-agent block
          in `logs/agent_pr_loop/agent_quota_block.json`; while that block is
          live the agent's CLI is not spawned, and the block expires after
          `QUOTA_REPROBE_SECONDS = 3600` so a lifted cap is picked up with no
          manual state clearing.

WHY/DIR:  The 2026-08-11 incident. `com.renquant.agent-pr-loop` (every 300s)
          reported `ok:false, error:"claude fix failed"` for hours. The cause
          was in the spawned CLI's stdout — `You've hit your monthly spend
          limit` — and it appeared **0 times** in
          `logs/agent_pr_loop/launchd_stdout.log`, which instead held **471**
          copies of `agent_pr_loop: claude fix failed`
          `[VERIFIED — grep counts on the live logs, 2026-08-12]`. The wrapper
          summarized a subprocess failure and discarded the only actionable
          half.

          The retry behaviour is the second defect: a spend cap cannot be
          cleared by re-running five minutes later, so the loop manufactured
          exactly the repeated-identical-alert noise the operator has objected
          to — 12 per hour, none of them new information.

          Design note, deliberate: the classifier is an ENUMERATED list with a
          **retryable default**. An unrecognised failure behaves exactly as it
          does today, so this can only reduce noise, never swallow a novel
          error. And a block SKIPS one agent rather than aborting the cycle —
          previously the raise killed the whole run, so codex's merges and the
          merge audit were lost too while claude was capped.

EVIDENCE:
  artifact:      `scripts/agent_pr_loop.py` (+4 edit sites),
                 `tests/test_agent_pr_loop_quota_block.py` (7 new tests).
  prod or exp:   prod — the automation wrapper. No strategy, model, artifact,
                 config, or order path is touched; it changes only how a
                 subprocess failure is reported and re-attempted.
  existing data: yes — the live `logs/agent_pr_loop/status.json` and the
                 launchd stdout/stderr logs were read READ-ONLY. The
                 non-ASCII-free spend-limit string in the test is copied
                 verbatim from the 2026-08-12T01:23:27Z status record; nothing
                 was generated.
  best-known?:   yes — lifting the child's own message beats inventing a
                 taxonomy the CLIs do not emit, and a time-boxed suppression
                 window beats a kill switch (self-healing when the cap lifts)
                 and beats a longer launchd interval (which would slow every
                 healthy cycle to fix an unhealthy one).
  scope:         "this is `scripts/agent_pr_loop.py` failure reporting and
                 re-attempt (prod), vs existing best = today's
                 `RuntimeError(f'{agent} {workflow} failed')`, which drops the
                 cause and re-spawns into the same wall every 300s. Measured
                 delta: the error string goes from `claude fix failed` to
                 `claude fix BLOCKED, not retryable: You've hit your monthly
                 spend limit...`, and re-spawn frequency under a live cap goes
                 from 12/hour to 1/hour. It does NOT fix the cap itself —
                 raising the limit or switching models is an operator action."

NEXT: the cap is live right now, so the first cycle after this lands will
record the block and the following eleven will skip cleanly; confirm by
reading `logs/agent_pr_loop/agent_quota_block.json` for an `observations`
count that stops climbing every 5 minutes. Merge is not deploy — this machine
picks the change up on the operator's umbrella sync. Not addressed here: no
alert is raised when a block is first recorded; `status["degraded"]` is
populated for a monitor to consume, and wiring that consumer is a separate
decision.
