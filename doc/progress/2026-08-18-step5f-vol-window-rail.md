# 2026-08-18 — Step 5f: the shadow_vol_window rail (orch#1004 impl PR 2, umbrella half)

STATUS:    delivered — daily-full lane wiring + static guards. NOTHING is
           deployed by this PR: the rail is dormant until the PINNED
           renquant-strategy-104 checkout carries
           `strategy_config.shadow_vol_window.json` (s104#99 is merged; the
           pin advance is a separate operator-gated step), and the live tree
           picks this script up only via the operator-gated deploy path.
WHAT:      `scripts/daily_104.sh` Step 5f — the `shadow_vol_window` e2e lane,
           cloned from Step 5e (Step 5 is the maintained pattern per the
           Step 4 retirement note): profile resolved from the PINNED strategy
           configs via `renquant_strategy_config`, skip-with-INFO when
           absent (a missing lane config NEVER hard-fails the daily),
           `--broker readonly-alpaca` +
           `RENQUANT_READONLY_TAG=alpaca_shadow_vol_window` lane isolation,
           own log (`logs/daily_104/<date>_shadow_vol_window.log`), own
           timeout env (`RENQUANT_SHADOW_VOL_WINDOW_TIMEOUT_SEC`, falls back
           to the shared `RENQUANT_SHADOW_TIMEOUT_SEC:-1800`), distinct
           FAIL/TIMEOUT ntfy titles, buy-side-preflight-block suppression
           (same pattern set, no exception-class gates swallowed), non-fatal
           in every branch. Paired with the renquant-orchestrator AC3
           readout PR (cross-referenced in both PR bodies).
WHY/DIR:   orch#1004 approved design §7 impl PR 2 (lane wiring half):
           the CONFIRMED vol-switch conditional (orch#1003, frozen prereg
           orch#1001) runs shadow-first; the lane's per-session ledger
           (`logs/vol_window_license.jsonl`, pipeline#294) accrues the
           pre-committed activation evidence before any operator ask.
EVIDENCE:  see §4 below.
NEXT:      (1) operator-gated pin advance so the pinned s104 checkout carries
           the profile (rail auto-activates); (2) REQUIRED BEFORE THE LANE'S
           FIRST LIVE SESSION: register `alpaca_shadow_vol_window` in
           pipeline `state_paths.ALLOWED_BROKERS` — see §3; (3) the
           orchestrator readout deploy (plist + manifest entry in the same
           reviewed batch) — documented in the paired PR's progress doc.

## 1. Wiring shape

Step 5f is a verbatim clone of the Step 5e block with lane-specific renames
(`VOL_WINDOW_STRATEGY_CONFIG` / `SHADOW_VOL_WINDOW_LOG` /
`SHADOW_VOL_WINDOW_TIMEOUT_SEC` / `SHADOW_VOL_WINDOW_RC` /
`SHADOW_VOL_WINDOW_BUY_SIDE_PREFLIGHT_PATTERN`, tag
`alpaca_shadow_vol_window`, titles `SHADOW-VOL-WINDOW-{FAIL,TIMEOUT}`),
placed after Step 5e and before Step 6. The Step 6 comment is updated to
state that the fleet sentinel's watch set stays Steps 5–5e and that the
vol-window lane's session accounting is the orchestrator readout's parity
alarm (the readout cross-checks the lane's license ledger against its runs
DB per session — the paired PR).

The lane's decision mechanics live elsewhere by design: the license
(pipeline#294 `vol_window_license.py`, flag-gated, kill switch
`RENQUANT_VOL_WINDOW_LICENSE_DISABLE`) and the lane profile (s104#99).
This wrapper only runs the standard readonly lane around them.

## 2. Guards

`tests/test_daily_104_shadow_notify.py`: 6 new static guards mirroring the
5d/5e set (profile gate + INFO ordering; tag/log/config threading incl. the
exact env-prefixed heredoc invocation line; own timeout env; distinct
FAIL/TIMEOUT titles with the alert-default pinned ON; preflight-block
suppression with no exception-class gates swallowed; 5e→5f→6 ordering +
non-fatal), plus the lane's success echo added to the
one-echo-per-lane-identity guard.

## 3. Known gap, declared: the readonly tag is UNREGISTERED in pipeline

`RENQUANT_READONLY_TAG=alpaca_shadow_vol_window` passes the umbrella-side
prefix validation (`live/broker_readonly.py::validate_readonly_tag` — any
`alpaca_shadow*`), but pipeline's `state_paths.ALLOWED_BROKERS` — the hard
allow-list the state/runs-DB path resolution validates against — does NOT
carry it on pipeline main at this PR's time (`[VERIFIED — read 2026-08-18
from pipeline origin/main 43a66f8 src/renquant_pipeline/state_paths.py:
allowlist ends at the three GOAL-9 tags]`). Until a one-line pipeline PR
registers the tag (the GOAL-9 "registered AT BIRTH" convention that
pipeline#265 set precisely so the allowlist is never the trailing consumer),
the lane's first session would fail at the state write with ValueError — the
exact `alpaca_shadow_blend_mom` session-1 incident recorded in the allowlist
itself. FAILURE MODE IF FORGOTTEN: non-fatal and PAGED (the Step 5f FAIL
branch notifies; prod is untouched) — loud, not silent, but still a wasted
session. The pin advance that activates this rail must carry a pipeline pin
with the tag registered.

## 4. Evidence

(a) Conclusion: the daily-full now carries the vol-window lane as a standard
Step-5x rail — dormant until the pinned profile appears, isolated under its
own tag, non-fatal in every branch — and the addition required nothing
beyond the established Step-5x pattern.

(b)
- `artifact:` none — shell wiring + static guards only. The lane's own
  artifacts (session ledger rows, runs DB) exist only after the
  operator-gated pin advance activates the rail.
- `prod or exp:` neither — wiring for a never-submit shadow lane; nothing
  scheduled/deployed by this PR (merged code reaches the live tree only via
  the operator-gated deploy path; the rail additionally gates on the pinned
  profile's presence).
- `existing data:` pattern provenance: Step 5e block + 5d/5e guard set
  `[VERIFIED — cloned from scripts/daily_104.sh @ abf204d and
  tests/test_daily_104_shadow_notify.py @ abf204d]`; lane tag/profile names
  from s104#99 (merged) `[VERIFIED — read 2026-08-18 from s104 origin/main
  8a395e4 configs/strategy_config.shadow_vol_window.json
  `_shadow_vol_window_profile` note]`; ledger filename from pipeline#294
  (merged) `[VERIFIED — pipeline origin/main 43a66f8 vol_window_license.py
  DEFAULT_LEDGER_RELPATH]`.
- `best-known?:` honest scope — (i) the guards are STATIC string pins on the
  script (the house surface for this file; there is no harness that executes
  daily_104.sh end-to-end in tests); `bash -n` clean. (ii) The §3 tag gap is
  real and blocking for the lane's first session; it is a pipeline-repo
  change this umbrella PR cannot carry. (iii) Dormancy (skip-with-INFO) is
  exercised by the same gate construction as the five existing lanes, which
  ran dormant in production before their profiles landed.
- `scope:` one script block + guards; no launchd change, no pin change, no
  live-tree write, prod lane untouched (Step 3 path byte-identical).

Suite: baseline at origin/main `abf204d` (targeted daily_104 surface:
test_daily_104_shadow_notify / test_daily_104_e2e /
test_daily_104_config_failclosed / test_ops_deployment_ready /
test_broker_readonly_tag / test_blend_kind_umbrella) = 101 passed /
2 skipped / 1 xfailed `[VERIFIED — run 2026-08-18 on the unmodified
worktree]`. After = 107 passed / 2 skipped / 1 xfailed (+6 new guards, all
passing; `bash -n` clean) `[VERIFIED — run 2026-08-18]`.

## 5. Files

- `scripts/daily_104.sh` — Step 5f block + Step 6 comment update.
- `tests/test_daily_104_shadow_notify.py` — 6 new guards + echo-identity
  list entry.
- `doc/progress/2026-08-18-step5f-vol-window-rail.md` — this doc.
