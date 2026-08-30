# 2026-08-30 — BUY-BLOCKED alert is urgent and says why; a relaxed gate never prints a bare ✓

**Bottom line:** on Mon 2026-08-31 the served panel artifact
(`artifacts/prod/panel-ltr.alpha158_fund.json`, trained 2026-08-02,
`wf_gate_metadata.passed=false`, `promotion_basis=freshness_fallback_rfc210`,
`fallback_genuine_ic=+0.00289`) turns **29 d > 28 d** (`DEFAULT_MAX_SERVED_AGE_DAYS`
in `renquant_pipeline.kernel.rfc210_license`), the license refuses, P-WF-GATE
hard-fails the 13:55 full run, `daily_104.sh` reruns `--sell-only` and — until
this PR — sent `RenQuant 104 BUY-BLOCKED` / "Full run blocked new buys;
sell-only fallback completed" with plain `curl -d`, **no `Priority:` header
(default 3)** and a 6 h cooldown. The operator would have learned nothing
about which artifact, how old, what the stamp says, what unblocks buys.
[VERIFIED — facts from the served artifact + `daily_104.sh:414-483` on
`origin/main` d237635, read 2026-08-30.]

Now: the wrapper composes the alert through the new read-only helper
`scripts/buy_blocked_reason.py`, posts it via `renquant_common.notify.send`
with **`Priority: urgent`, `Tags: rotating_light,rq104`**, title
`RenQuant 104 BUY-BLOCKED (sell-only fallback)`, falls back to curl **with the
same headers** if the sender is unreachable / the POST fails, and the stamp is
keyed by session **date** (once per session) instead of 6 h. The umbrella
kernel copy's P-REGIME-IC no longer prints `✓ … passed for eligible regimes`
when the pass is only the `sanity_regime_ic_required=false` relaxation.
Pure code + one test list edit in CI; no config, artifact, state or live path
touched.

## What the alert now says (body from the helper, real-shape fixture)

```
New buys BLOCKED by P-WF-GATE; the full run was rerun --sell-only.
Served artifact: trained 2026-08-02 (29d old) — …/artifacts/prod/panel-ltr.alpha158_fund.json
wf_gate_metadata.passed=false promotion_basis=freshness_fallback_rfc210 genuine_ic=+0.00289
License: REFUSED: governance-served artifact aged out: trained 2026-08-02, 29d old > 28d RFC#210 serving SLA
Preflight: ✗ P-WF-GATE: active panel artifact carries failed WF gate evidence: …
WF stamp: FAIL: absolute_ok=True, benchmark_ok=False, regime_ok=False; …
Exits continue via the 06:30-13:00 sell-only loop; risk controls stay armed.
Buys resume only when a candidate is promoted (weekly-wf-promote / retrain-panel104 — weekly-wf-promote last exit status: 1, retrain-panel104 last exit status: 1).
Held: …
Log: …/logs/daily_104/2026-08-31.log
```

* **Artifact facts are READ, never assumed** (`served_summary`): trained date,
  age vs `--today`, `passed`, `promotion_basis`, genuine IC
  (`metadata.fallback_genuine_ic`, then the stamp's
  `sanity_placebo_genuine_ic`, else `n/a`). A bare payload yields all-`None`,
  which the body prints as "trained_date missing / age unknown / null".
* **License verdict** = the RFC #210 evaluator's own `reason` when
  `renquant_pipeline` is importable (it is: `daily_104.sh` exports the pinned
  PYTHONPATH). The 28-day policy stays pipeline-owned; the umbrella does not
  re-encode it. When not importable the body says so instead of guessing.
* **Preflight line** = the last `✗ P-*` line the full run printed, captured by
  the wrapper *before* it deletes `$FULL_RUN_LOG` (the buy-side pattern names
  15 gates; the body names whichever one actually fired). Falls back to the
  last `P-WF-GATE` line of any kind.
* **Promotion-job status** from `launchctl list` (`PID\tLastExit\tLabel`,
  read-only); on 2026-08-30 both `com.renquant.weekly-wf-promote` and
  `com.renquant.retrain-panel104` show last exit **1** [VERIFIED `launchctl
  list`], which is precisely why the operator needs it in the alert.

## Sending

`--send` → `renquant_common.notify.send(title, body, priority="urgent",
tags="rotating_light,rq104")`. Exit **0** = posted **or** deliberately
suppressed by `RENQUANT_NO_NOTIFY` (the wrapper must not curl around a
suppression); **3** = `renquant_common` not importable; **4** = POST failed.
On 3/4 `daily_104.sh` curls with `-H "Priority: urgent" -H "Tags:
rotating_light,rq104"` — the bare default-priority line is gone
(`notify "RenQuant 104 BUY-BLOCKED"` no longer exists in the script; asserted).
stdout is the body (re-sendable), stderr the sender diagnostics; both are
echoed into the daily log.

## Once per session date

`logs/daily_104/.buy_blocked_alert_stamp` now holds the session `DATE`; the
alert fires when the stamp differs. `RENQUANT_BUY_BLOCKED_ALERT_COOLDOWN_SEC`
/ `21600` are removed. The old 6 h window let a 13:55 block re-page at the
next 06:30 sell-only pass and then swallow the rest of that day.

## Preflight truth — where it lives

The daily run executes the **pinned renquant-pipeline** kernel through the
orchestrator daily-bridge (`daily_104.sh:407-410`); the RFC #210 license and
the `✓ P-WF-GATE … governance-served … buys admitted` line come from
`renquant_pipeline/kernel/preflight.py` + `preflight_pipeline/tasks/gate.py`,
NOT from this repo. That fix is the companion PR **renquant-pipeline
`fix/preflight-licensed-relaxed-truth`** (both twins: `LICENSED: WF gate
FAILED, genuine_ic=+0.0029, served age 26d ≤ 28 …` and `RELAXED: sanity IC
failed (…); stamp failed BULL_CALM ρ=0.002; sanity_regime_ic_required=false …`).
It reaches production only with a pin bump (merged ≠ deployed).

This repo's `backtesting/renquant_104/kernel/preflight.py` is the
`RQ_DAILY_RUNNER=umbrella` rollback copy (already on the kernel-parity
known-drift allowlist). It has **no** RFC #210 license path — a
`passed=false` artifact always hard-fails there — so only its P-REGIME-IC
relaxed text needed the same treatment; done here with the same
`_regime_ic_pass_message` builder and the same served-shape test.

## Tests (run with `/Users/renhao/git/github/RenQuant/.venv/bin/python`)

* `tests/test_buy_blocked_reason.py` (new, 16): served-shape fixture (the
  artifact's metadata block copied shape-for-shape), never-invent on bare
  payload, genuine-IC fallback, license verdict with/without the pipeline
  evaluator, `✗` line extraction + gate naming, `launchctl` parsing, full body
  content, unreadable artifact still alerts, CLI `--json` headers, sender
  called with `priority="urgent"` + `tags="rotating_light,rq104"`, exit codes
  3/4/suppressed, **behavioural read-only** (subprocess run + mtime diff of
  the tree), and the wrapper wiring (helper invoked with `--send`, curl
  fallback carries the headers, date stamp, capture-before-delete ordering).
* `tests/test_daily_104_shadow_notify.py`: cooldown guard → once-per-date guard.
* `tests/test_preflight_regime_sanity.py` (+3): relaxed text leads with
  `RELAXED:`, strict config still blocks, genuine pass keeps the plain text.
* CI enumeration: `tests/test_buy_blocked_reason.py` added to
  `strategy-104-snapshot-fresh.yml` (a bare runner suffices — stdlib only).

CI commands run locally for every touched area [VERIFIED 2026-08-30]:
`strategy-104-snapshot-fresh` selftest list **228 passed**;
`readonly-e2e-classification` **19 passed** (`check_readonly_e2e.sh` greps
`P-WF-GATE` by name — unchanged); `kernel-parity-ci` **6 passed** against the
pin afb7362; `operator-notification-contract` **147 passed**; umbrella
preflight suites (`test_preflight_regime_sanity`, `test_wf_gate_relaxation`,
`test_preflight_pipeline_gate`, `test_runner_preflight_dry_run`) **38 passed**.
`tests/test_preflight.py` has 5 pre-existing failures on `origin/main`
(`P-CORR-METADATA … No module named 'renquant_pipeline'`, verified by
stash/unstash) — untouched by this PR.

## Not done / follow-ups

* The pipeline PR must be merged **and pinned** for the licensed/relaxed
  wording to reach the 13:55 run; until then the umbrella-side alert already
  states the truth from the artifact itself.
* The alert does not (and should not) decide anything: whether to re-promote,
  extend the SLA or accept the `freshness_fallback` again is the operator's
  RFC #210 decision.
