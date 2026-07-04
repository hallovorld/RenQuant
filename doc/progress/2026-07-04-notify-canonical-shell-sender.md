# 2026-07-04 — Canonical shell ntfy sender scripts/notify.sh (campaign B6, audit XC-4)

## What

`scripts/notify.sh` — the fleet's single shell ntfy sender (`rq_notify "Title"
"body" [priority] [tags]`), the shell twin of `renquant_common.notify` (same
contract: `NTFY_TOPIC` env > `$RQ_ROOT/.env` parse > `"renquant"`;
`RENQUANT_NO_NOTIFY` truthy suppresses always; curl `--max-time 5`; never fails
the caller). Sourceable from sh/bash/zsh under `set -u`. `.env` is parsed, not
sourced — no caller-env pollution.

## Why here

The 8 hand-rolled `source .env` + curl blocks in the orchestrator's ops
wrappers (audit #296 XC-4, §4.2) all already anchor on `$RQ_ROOT` for
`.env`/`.venv`/logs — the umbrella is the shared shell-ops home, so the one
canonical sourceable helper lives here (renquant-common has no shell share).
The umbrella's own scripts' hand-rolled ntfy blocks (RQ#444 territory) can
re-point to it in a follow-up wave; this PR adds the canonical only, no
umbrella-script re-points.

## Tests

`tests/test_notify_sh.py` — 15 tests (sh/bash/zsh × suppression, env-var
topic + header mapping, `.env` parse without sourcing, default topic,
never-fail with broken curl). No network: curl stubbed via PATH shim.

## Deploy note

The orchestrator wrapper re-points source `$RQ_ROOT/scripts/notify.sh`
fail-soft (`. ... || true`); until the live umbrella tree syncs this commit, a
failure notification from a re-pointed wrapper is lost (logged 127 in the
wrapper log, job exit codes unaffected). Sync the live tree before or with the
orchestrator ops pin bump.
