# 2026-08-29 — Readonly e2e verify names a dead shadow leg (orch#1066 option c)

**Bottom line:** `scripts/check_readonly_e2e.sh` — the gold-standard deploy
verify — no longer collapses a missing panel-scoring artifact into the generic
exit 1. It exits **3** BEFORE the funnel runs, naming the config key, the ref
and every path tried, and exits **4** AFTER the funnel when the log carries
`panel_scorer_load_failed` or the `STRUCTURAL_BLOCK — engineering condition`
alert. Both new codes say WHAT failed; neither says WHEN it started failing.
The preflight inspects only the CURRENT pinned assembly, so it cannot tell a
leg that was already dead from one a bump just killed — the exit-3 message is
attribution-neutral and points the operator at the previous pinned assembly
(`scripts/promote_pin.py` keeps a timestamped backup of `subrepos.lock.json`,
`backup_lock` / `latest_backup` / `revert`). Exit 0 / 1 / 2 keep their meaning
for every other outcome. Pure code: no config changed, no flag flipped.
Measured on the live tree today: exit 3 [VERIFIED, see Evidence]. The config
fix itself (remove/replace the dead leg, or restore the artifact) is a
SEPARATE reviewed config decision — orch#1066 options a/b — and is NOT made
here.

## Review round 1 (Codex, PR #614)

The first version printed "pre-existing dead leg … not a pin-bump regression".
That attribution was unsupported: nothing in the script compares against the
previous pin, so a missing artifact newly introduced by a bump would have been
labelled pre-existing — the exact false assurance a deploy verifier must not
give. Reworked: the diagnostics are unchanged, the attribution claim is gone
from the script, the tests assert the neutral wording (and assert the old
wording is absent), and no automatic baseline comparison was added (not
required; the operator attributes against the backup lock).

## The failure mode

1. `live.runner` auto-selects `strategy_config.shadow.json` for any
   `renquant_104` + `readonly-alpaca` run with no `--strategy-config-name`
   (`live/runner.py:1573-1588` `_resolve_strategy_config_name`; the orchestrator
   live-bridge mirrors it in `renquant_orchestrator/live_bridge.py:74-82` and
   routes the read to the PINNED strategy subrepo at `:97-113`). The verify
   passes the name explicitly (`scripts/check_readonly_e2e.sh`, funnel block)
   — same document either way.
2. That config's primary leg is `ranking.panel_scoring.artifact_path =
   artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`
   with `kind: hf_patchtst` (pinned assembly
   `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json`).
3. The primary loader joins a relative ref onto `config["_strategy_dir"]` and
   nothing else — `renquant_pipeline/kernel/panel_pipeline/job_panel_scoring.py:915-928`
   (`LoadScorerTask._resolve_artifact_path`), called at `:1045`; `_strategy_dir`
   is `<repo>/backtesting/renquant_104` (`live/runner.py:498`). There is NO
   repo-root fallback on this path (only blend components go through
   `kernel.artifact_resolver.resolve_artifact`, which tries strategy_dir then
   repo_root — `blend_scorer.py:324-332`, `artifact_resolver.py:56-59`).
   `RenQuant/backtesting/renquant_104/artifacts/patchtst_shadow/` does not
   exist; a copy under `RenQuant/artifacts/patchtst_shadow/...` does, and is
   never consulted by this loader.
4. `HFPatchTSTPanelScorer.load(p)` raises → `_fail_closed_panel_scoring(ctx,
   "panel_scorer_load_failed")` (`job_panel_scoring.py:1118-1124`) → every
   candidate blocked with reason `panel_scorer_load_failed:...`
   (`renquant_pipeline/panel_scoring.py:220`) → funnel verdict
   `STRUCTURAL_BLOCK` and the alert line
   `FunnelIntegrityAlert: STRUCTURAL_BLOCK — engineering condition suppressed
   buy capability` (`kernel/pipeline/task_funnel_integrity.py:873-879`).
5. The verify saw only "rc≠0 or no committed decision" and printed
   `READONLY_E2E: FAIL` → exit 1 — indistinguishable from a pipeline crash.
   Measured 2026-08-25 (orch#1066).

## What changed (`scripts/check_readonly_e2e.sh`)

* **Preflight, before the funnel (exit 3).** After the subrepo env is
  resolved, a python step reads the shadow config the runner will use
  (multirepo default: `$SUBREPO_ROOT/renquant-strategy-104/configs/strategy_config.shadow.json`;
  `RQ_DAILY_RUNNER=umbrella`: `backtesting/renquant_104/strategy_config.shadow.json`)
  and resolves each panel-scoring artifact ref the way the pipeline does:
  - `ranking.panel_scoring.artifact_path` → `_strategy_dir/ref` only;
  - `ranking.panel_scoring.components[i].artifact_path` → `_strategy_dir/ref`,
    then `repo_root/ref` (resolver order);
  - `ranking.panel_scoring.global_calibration.artifact_path` when
    `enabled` → `_strategy_dir/ref` only (`LoadGlobalCalibrationTask`).
  Absolute refs are taken as-is; `panel_scoring.enabled: false` skips the
  check. Any missing file → each missing key + ref + every path tried is
  printed, then the neutral line
  `DEAD_LEG detected before the funnel in <config>; attribute by comparing
  against the previous pinned assembly (scripts/promote_pin.py keeps the
  backup lock) — see orch#1066`, and the script exits 3 WITHOUT running the
  funnel. The shadow config being unreadable is a setup error → exit 2 (its
  existing meaning).
* **Classification, after the funnel (exit 4).** The log is scanned
  (fixed-string) for `panel_scorer_load_failed` and
  `STRUCTURAL_BLOCK — engineering condition`. Where the script would have
  exited 1 (runner rc≠0, or no committed decision), a hit turns that into
  exit 4, printing "structural engineering failure in the shadow scorer chain
  (buy capability suppressed by an engineering condition, not a decision
  outcome; whether it predates the bump is not established here)" plus the
  first matching log line. A run that still commits a decision keeps exit 0
  and prints a WARN naming the marker. The isolation breach check stays exit
  1 and stays first.
* Header comment documents all five codes and the attribution limit.

### Exit codes, exact meanings

| code | meaning |
|------|---------|
| 0 | clean decision produced (unchanged) |
| 1 | crash / timeout / no committed decision / isolation breach — WITHOUT the structural markers (unchanged) |
| 2 | setup error: repo, subrepo env, or the shadow config itself unreadable |
| 3 | dead leg: a referenced panel-scoring artifact is missing on disk in the CURRENT pinned assembly; funnel NOT run; says nothing about when it went missing |
| 4 | structural block: funnel ran; log carries `panel_scorer_load_failed` or the STRUCTURAL_BLOCK engineering-condition alert — a structural engineering failure in the shadow scorer chain, not a decision outcome; says nothing about when it started |

Callers that treat "nonzero = failed verify" (`scripts/promote_pin.py:220-223`
`--verify-cmd`) are unaffected; 3 and 4 are still failures — they now say
what failed. To attribute a 3 after a bump: `promote_pin.py revert` (or read
the latest `subrepos.lock.json` backup) and re-run the verify on the previous
pin; same exit 3 there = pre-existing, exit 0 there = the bump.

## Evidence

* Live tree, today (read-only; the preflight exits before any runner is
  spawned) [VERIFIED]:
  ```
  READONLY_E2E: DEAD_LEG — ranking.panel_scoring.artifact_path (kind=hf_patchtst) = 'artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt' is MISSING; tried /Users/renhao/git/github/RenQuant/backtesting/renquant_104/artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt
  READONLY_E2E: DEAD_LEG — DEAD_LEG detected before the funnel in /Users/renhao/git/github/RenQuant/.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json; attribute by comparing against the previous pinned assembly (scripts/promote_pin.py keeps the backup lock) — see orch#1066
  READONLY_E2E: DEAD_LEG — funnel NOT run (exit 3); fixing the config is a separate reviewed decision (orch#1066 options a/b)
  EXIT=3
  ```
  The enabled `global_calibration` artifact resolved (only the primary leg is
  dead). Whether it was dead before the last bump is exactly what this script
  does not claim; orch#1066's own reading (artifacts dir unchanged since
  Aug 6) is the operator's evidence, not the verifier's.
* `tests/test_check_readonly_e2e_classification.py` — 15 passed
  (`.venv/bin/python -m pytest -q -o addopts='' tests/test_check_readonly_e2e_classification.py`)
  [VERIFIED]. Drives the script as a subprocess against a throwaway repo dir
  with a stub `renquant_orchestrator` on the script's own PYTHONPATH; covers:
  missing primary → 3 with the named path and the neutral line (and asserts
  "pre-existing" / "not a pin-bump regression" are ABSENT), funnel not
  invoked; primary present only at repo root → still 3 (loader fidelity);
  blend component present only at repo root → passes (resolver fidelity);
  missing blend component → 3; enabled/disabled global_calibration; markers →
  4 (three variants, marker line echoed, neutral wording asserted); markers +
  committed decision → 0 with WARN; clean → 0 with the real CLI shape
  asserted; crash → 1; silent → 1; unreadable config → 2;
  `panel_scoring.enabled=false` → preflight skipped.
* `bash -n scripts/check_readonly_e2e.sh` clean [VERIFIED]. shellcheck's only
  finding (SC2155) is on the pre-existing PYTHONPATH export line.
* `.github/workflows/readonly-e2e-classification.yml` — new pytest-only job
  (shape of `kernel-parity-ci.yml`, without the subrepo checkout) that names
  the test file explicitly; path filters on the script, `subrepo_env.sh`, the
  test and the workflow.

## Not done here (deliberately)

* No edit to `strategy_config.shadow.json`: whether the `hf_patchtst` leg is
  removed/replaced (option a) or the artifact restored (option b) is a config
  decision under review like any other. Until one lands, this verify exits 3
  on the live tree.
* No automatic previous-pin comparison in the verifier (review round 1: not
  required; attribution stays with the operator and the backup lock).
* No change to which config the readonly lane auto-selects.
