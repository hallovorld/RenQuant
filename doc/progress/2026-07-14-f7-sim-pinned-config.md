# F-7: Sim defaults to pinned strategy config

Date: 2026-07-14
PR: g3/f7-sim-pinned-config (umbrella)
Audit finding: F-7 (architecture compliance audit §7)

## Problem

`run_sim_104.py` resolved its strategy config from the umbrella-local copy
(`backtesting/renquant_104/strategy_config.json`), while production live runs
use the pinned subrepo config via `live_bridge._with_pinned_strategy_config()`.
This means sim results are not reproducible against the production config and
config drift between the two copies goes undetected.

## Fix

Added `_resolve_strategy_config()` that reads `subrepos.lock.json`, finds the
`renquant-strategy-104` entry, and resolves the config from
`local_path/configs/<config_name>`. This is the same path that `live_bridge.py`
uses for production runs.

Behavior:
- **Default (no flag)**: resolve from pinned subrepo. Fail closed if lock file
  or config file is missing — no silent fallback.
- **`--dev-config`**: explicitly use the umbrella-local copy. Output is tagged
  `DEV` and marked not comparable to production.
- Config fingerprint (SHA-256 prefix) is logged and recorded in the config dict
  as `_strategy_config_digest`, `_strategy_config_source`, and
  `_strategy_config_path` for downstream audit.

## r2: Pin verification (codex round 1 fix)

The initial implementation resolved the config from the lock entry's
`local_path` but never verified that the checkout's HEAD/remote/clean state
matched the lock commit. An ahead, behind, dirty, or wrong-remote checkout
would be labelled PINNED with its digest recorded as if it were a deployment
pin.

Fix:
- `_verify_pin()` checks HEAD matches the lock commit and working tree is clean
- `_resolve_strategy_config()` calls `_verify_pin()` and fails closed on any
  mismatch — never labels a drifted checkout as PINNED
- Relative `local_path` values in the lock file are resolved against
  `repo_root`, not the caller's cwd

## r3: Canonical vs experiment-local configs (pre-Codex-review, found independently)

Found and fixed before this round of the PR was reviewed: the fail-closed
default (r1/r2) would have broken two currently-working scripts that call
`run_sim_104.py` without `--dev-config`:

- `scripts/_doe_orchestrate_bb.sh` (`--strategy-config-name
  "strategy_config.sim_BB_${i}.json"`)
- `scripts/run_parallel_after_trail015.sh` (`--strategy-config-name
  strategy_config.sim_re_sdl_n2.json`)

Both pass one-off research-sweep config names that only ever exist in the
umbrella-local `backtesting/renquant_104/` directory — verified
`strategy_config.sim_re_sdl_n2.json` exists ONLY there, not in the pinned
`renquant-strategy-104` repo's `configs/`. Neither r1 nor r2 had a fallback
for this case; both scripts would `sys.exit(1)` on their next run.

Fix: `_resolve_strategy_config()` now checks the config name against an
explicit allowlist, `CANONICAL_STRATEGY_CONFIG_NAMES` — the exact, verified
contents of the pinned `renquant-strategy-104/configs/` directory
(`strategy_config.json`, `.golden`, `.shadow`, `.shadow_a`, `.shadow_b`). An
explicit allowlist was chosen over a glob/pattern match because the
research-sweep naming zoo (`strategy_config.sim_*.json`, plus assorted
one-off historical names) is too varied to pattern-match safely, and an
allowlist is easier to audit for drift.

- **Canonical name** (in the allowlist): unchanged — strict PINNED path,
  including r2's HEAD/dirty pin verification; `--dev-config` still overrides
  to umbrella-local (`DEV`).
- **Non-canonical name** (anything else, e.g. `strategy_config.sim_*.json`):
  resolves directly from the umbrella-local `strategy_dir`, exactly like
  `--dev-config`'s behavior, but WITHOUT requiring the flag and WITHOUT
  touching `subrepos.lock.json` / `_verify_pin` at all (there's no pin to
  verify) — these configs were never mirrored into the pin, so they were
  never part of the F-7 parity contract in the first place. Source is
  tagged `"LOCAL"` (distinct from `"DEV"`, which is an explicit override of
  a canonical name).

Also fixed: the config fingerprint was a truncated (16-char), unprefixed hex
digest (`hashlib.sha256(cfg_bytes).hexdigest()[:16]`). Changed to the
`sha256:`-prefixed, full 64-char hexdigest to match the established
full-file-audit-hash convention elsewhere in the system (e.g.
`renquant_common.model_fingerprint.artifact_sha256`).

## Tests

11 tests in `tests/test_resolve_strategy_config.py` (r2):
- `_verify_pin`: clean match, HEAD mismatch, dirty checkout, git failure
- `_resolve_strategy_config`: pinned clean match, HEAD mismatch exits, dirty
  checkout exits, relative local_path, dev_config marked DEV (verify_pin not
  called), missing lock exits, missing config after verify exits

12 more tests in `tests/test_run_sim_104_config_resolution.py` (r3):
- a canonical name (`strategy_config.json`) without `--dev-config` still
  resolves via the PINNED/verified path and fails closed if the pinned copy
  is absent;
- an experiment-local name (`strategy_config.sim_BB_09.json`, plus explicit
  regression pins for the two real affected scripts' exact config names)
  resolves from the umbrella-local copy WITHOUT `--dev-config`, never calls
  `_verify_pin`, and still fails closed if that file is missing;
- `--dev-config` still overrides canonical names to umbrella-local, and is a
  no-op vs. default for non-canonical names;
- the fingerprint is `sha256:`-prefixed with a full 64-char hexdigest;
- `CANONICAL_STRATEGY_CONFIG_NAMES` is cross-checked against the live pinned
  `renquant-strategy-104/configs/` checkout (skipped if that checkout isn't
  present on the host) to catch future drift.
