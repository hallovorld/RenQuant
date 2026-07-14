# G3 F-7: sim script resolves pinned strategy config, fails closed

Date: 2026-07-14 (revised 2026-07-14 after Codex review)
PR: fix(sim): resolve strategy config from pinned subrepo (G3 F-7)

## Problem

`scripts/run_sim_104.py` loaded strategy config from the umbrella's local
copy at `backtesting/renquant_104/strategy_config.json`. This copy has
drifted from the pinned `renquant-strategy-104/configs/strategy_config.json`
that the live bridge uses — sim evaluated a different primary scorer kind
(hf_patchtst) than live (xgb), plus missing config sections (deployment
governor, fractional shares, software stops, decision ledger, intraday
decisioning).

Finding F-7 of the 2026-07-04 architecture compliance audit
(`doc/arch/2026-07-04-umbrella-compliance-audit.md`).

## First attempt + Codex review

The first attempt at this fix made `run_sim_104.py` prefer the pinned
config, falling back to the (possibly drifted) umbrella copy with a log
warning if the pin was unavailable. It also accidentally carried a large
amount of unrelated content (a stale `deploy(pins)` commit + intervening
`main` merges), so the PR diff was not scoped to the resolver.

Codex review (verbatim):

> Blocking: the declared scope is one simulator-config resolver, but the PR
> diff carries a large unrelated deployment/pin history: market-data
> archives, factor/map data, production and walk-forward artifacts,
> manifests, and snapshot documentation. The changed-files list also does
> not contain `run_sim_104.py`, despite that being the stated
> implementation.
>
> Please rebuild this from current main as a clean F-7 PR containing only
> the resolver, focused tests for pinned-first and no-pin behavior, and the
> minimal F-7 documentation. Do not mix mutable data, historical artifacts,
> or pin changes into an architecture repair.
>
> The desired behavior must also be fail-closed for an auditable
> simulation: if a run declares a pin, resolve that exact pinned strategy
> config and record its fingerprint in the run bundle. A warning fallback
> to an unpinned umbrella copy creates an unidentifiable backtest and is
> only acceptable for an explicitly labelled local-development mode, never
> the standard simulation path.

## Root cause of the scope contamination

The branch (`g3/f7-sim-pinned-strategy-config`) was built on top of a stale
branch tip — a `deploy(pins): 2026-07-10 deployment batch` commit
(101 files: backtesting data archives, factor/map files, walk-forward
calibrator snapshots, `subrepos.lock.json`) that was **never merged to
`main`** — plus three intervening `Merge remote-tracking branch 'origin/main'`
commits, instead of being branched from a clean `main`. `gh pr diff`
couldn't even render the full diff (`HTTP 406: diff too large`), and
GitHub's truncated view cut off before reaching `scripts/run_sim_104.py`
alphabetically, which is why Codex's review reported the file as absent —
it genuinely was buried, ~30k lines deep in an unrelated diff.

The actual F-7 change was a single, narrow, correctly-scoped commit
(`scripts/run_sim_104.py` + this progress doc, 46 lines total) sitting on
top of that contaminated history. Fix: cherry-pick only that commit onto a
fresh branch off current `main`, discarding the stale pin/data commits and
merge noise entirely (git history, not file edits — same class of fix
already applied once this session to PR #468).

## Fix (corrected scope + fail-closed design)

`scripts/run_sim_104.py` gained `resolve_strategy_config()`:

- **Standard simulation path (default).** For the config names actually
  published under `renquant-strategy-104/configs/` — the
  live-bridge-mirrored names (`strategy_config.json`, `.golden.json`,
  `.shadow.json`, `.shadow_a.json`, `.shadow_b.json`) — resolve *exactly*
  that pinned file. No silent/warned fallback. If the pin can't be
  resolved, this **fails closed**: raises `StrategyConfigResolutionError`,
  which `main()` turns into `log.error(...)` + `sys.exit(1)`.
- **Fingerprint recorded in the run bundle.** Every resolved config (pinned
  or fallback) is content-fingerprinted as `"sha256:" + hexdigest`,
  matching the convention already used elsewhere in this multi-repo system
  (`renquant_common.model_fingerprint.artifact_sha256`,
  `cost_model.cost_model_content_sha256`). The fingerprint + resolved path
  + source (`pinned` / `experiment_local` / `unpinned_local_dev_fallback`)
  are stamped into `config["_strategy_config_*"]` and into the
  `--equity-json` run-bundle payload (the per-run metadata artifact this
  script already writes for paired-returns analysis).
- **Explicit, two-factor local-dev escape hatch.** Mirrors the existing
  `--allow-legacy-direct-execution` / `RENQUANT_ALLOW_LEGACY_DIRECT_EXECUTION`
  pattern in `scripts/production_runner.py`: a hidden (`argparse.SUPPRESS`)
  CLI flag `--allow-unpinned-local-dev` **and** the environment variable
  `RENQUANT_ALLOW_UNPINNED_LOCAL_DEV=1` must **both** be set for the
  warn-and-fallback path to engage. Either alone is a no-op (still fails
  closed), so the escape hatch cannot be triggered by a stray env var or a
  copy-pasted flag.
- **Umbrella-only experiment configs are unaffected.** ~70 of the 73
  configs under `backtesting/renquant_104/*.json` (the `strategy_config.sim_*.json`
  research-sweep variants used by `run_dense_panel.sh`,
  `run_regime_overlay_experiments.sh`, etc.) were never published to the
  pin and were never meant to be — there is no live counterpart for them to
  drift from. These resolve straight from the umbrella copy
  (`source="experiment_local"`), not subject to the fail-closed pin
  requirement, so none of the existing sim-sweep tooling breaks.

## Tests

`tests/test_run_sim_104_strategy_config_resolution.py` (new, 10 cases):
pinned-first resolution + fingerprint; default invocation fails closed with
no pin (with and without a drifted local copy present); the local-dev
escape hatch falls back only with both the flag and the env var set (three
negative cases: env-only, flag-only, and env set to a non-`"1"` value all
still fail closed); experiment-local side configs resolve without the pin
or escape hatch and still fail closed if genuinely missing; a static check
that the CLI flag is wired and suppressed from `--help`.

## Scope

Only `scripts/run_sim_104.py` + its test file + this doc. `git diff
main...HEAD --name-status` from the rebuilt branch touches exactly 3 files.
Other scripts (`run_wf_gate.py`, `analyze_backtest.py`, etc.) still use the
umbrella copy directly and can be migrated incrementally.
