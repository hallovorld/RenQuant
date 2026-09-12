# The shell ntfy sender is capped too — no alert arrives as an attachment   (PR #TBD)

STATUS:    delivered — the shell twin of renquant-common#44.
WHAT:      `scripts/notify.sh::rq_notify` now caps the POSTed body at
           `RQ_NTFY_MAX_BODY_BYTES` (default 3800) UTF-8 BYTES — counted with
           `LC_ALL=C wc -c` and cut with `head -c`, so the limit holds for
           Chinese/emoji bodies rather than counting characters — and appends
           a marker naming exactly how many bytes were dropped and that the
           full text is in the sender's log. A body that fits is sent
           byte-identical. A capped send writes one stderr line
           (`[ntfy body capped] <title>: N -> 3800 bytes`) so the truncation
           is observable in the wrapper's log. POSIX sh, sourceable under
           `set -u` from sh, bash and zsh (the 08-30 caller-shell lesson).
WHY/DIR:   The operator asked (2026-09-12) to remove the ntfy messages "that
           send attachments". Nothing in the fleet attaches a file on purpose;
           ntfy.sh converts any body over its 4096-byte `message-size-limit`
           into a `.txt` attachment, so the phone shows a file icon instead of
           the alert text. renquant-common#44 caps the Python send site that
           every sentinel uses; this is the shell send site (rq105 wrappers
           today, the re-point target for the umbrella's bare-curl scripts).
           Direction: G-A stop noise / G-D ops truth.
EVIDENCE:  artifact:      `logs/rq104/launchd_degradation_sentinel.out` — the `rq104 DEGRADED: 2 issue(s)` body posted 2026-09-11 measures **4,617 bytes** > 4,096 → delivered as an attachment [VERIFIED — measured from the evidence log 2026-09-12]; that sender is Python (renquant-common#44), and `rq_notify` had the identical gap with no measured offender yet — fixed for the same reason before one appears
           prod or exp:   shared shell helper; no production path written; only bodies over 3,800 bytes change (truncated with a marker instead of being turned into attachments by the server)
           existing data: `tests/test_notify_sh.py` **27 passed** (12 new = 4 cases × sh/bash/zsh: a 6,000-byte body lands ≤ 3,800 with the marker and the stderr note; a short body is byte-identical; an 18,000-byte Chinese body is capped by BYTES not characters; `RQ_NTFY_MAX_BODY_BYTES=500` is honoured) [VERIFIED — 2026-09-12, umbrella venv, `-o addopts=''`]; `sh -n`, `bash -n`, `zsh -n` all clean
           best-known?:   n/a — delivery plumbing, no model claim
           scope:         "this caps the body in scripts/notify.sh only; the 17 bare-curl scripts counted by orchestrator ops/blind_notifier_scan.py (12 launchd-scheduled) still post uncapped and are the standing re-point target — not touched here"
NEXT:      Landing needs the codex merge gate (out on quota until 2026-10-03)
           or an operator decision. The daily rq105 wrappers source this file
           from the LIVE umbrella tree, so once merged a live ff-only sync
           makes it effective immediately — no pin advance involved, unlike
           renquant-common#44. The 17 bare-curl scripts should be re-pointed
           to `rq_notify` in a follow-up so one cap covers them all.
