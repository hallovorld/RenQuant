# Umbrella 104/105 Design-Compliance Audit — Findings Memo

Date: 2026-07-04
Auditor: Claude (agent), review: haorensjtu-dev
Scope: `backtesting/renquant_104/**`, `scripts/**`, `live/**` at umbrella `main`
(79d47da3) cross-checked against the pinned `renquant-pipeline`
(778983ab, per `subrepos.lock.json`). Docs-only PR — no code changes; fixes are
follow-up work items.

Charter audited against:

- `doc/arch/subrepo-operating-model.md` (roles table, Universal Rules 1-6)
- `doc/arch/multirepo-sop.md` §4 placement rule, `doc/arch/kernel-inventory.md`
- Umbrella `CLAUDE.md` §3.5 (Phase-1 byte-equivalent mirror invariant, paired PRs)
- Session-established rules: kernel-alias rule (live paths route `kernel.*` to the
  pinned pipeline via `live_bridge.bootstrap_multirepo`); single-impl-imports-only
  for hashes/fingerprints; flags default-OFF with byte-inertness tests; no
  selection logic in umbrella scripts (#210 §6)

Severity legend: **P0** = behavior/money risk on live or promote paths;
**P1** = structural debt that will produce a P0; **P2** = cosmetic/doc.

---

## 0. Executive summary

**30 findings: 6 × P0, 19 × P1, 5 × P2.** The systemic conclusion: the
Phase-1 "byte-equivalent mirror" invariant (CLAUDE.md §3.5) is dead in
practice — the umbrella kernel and the pinned pipeline kernel are under
active DUAL maintenance (78 of 169 shared files drifted, bidirectionally,
111 umbrella-kernel commits since 2026-05-25) with no enforcement (the cited
MD5-parity test does not exist). Because live runs the pinned pipeline kernel
while sim/WF-gate/promote run the umbrella kernel, every promotion is
validated on code production will not run, and fixes routinely strand on one
side.

The six P0s:

| # | Finding | One line |
|---|---|---|
| F-1 | Scorer fix stranded in mirror | The 2026-06-15 "silent mis-score" HF-PatchTST fix (cross-stock/FiLM reconstruction + fail-closed weights) exists only umbrella-side; live paths run the unfixed pipeline copy (`load_state_dict(strict=False)`), scoring the shadow model daily |
| F-2 | Three forked `WalkForwardModelLoader`s | Stamp leg (backtesting), sim leg (umbrella, 12-char fuzzy match), live-preflight leg (pipeline, M6 schema dispatch) each verify the WF/fingerprint contract with different code — the generator of the 05-27/06-22/07-01 incident class |
| F-3 | Promote gate validates on stale code | WF-gate sims execute umbrella `task_selection` (frozen 05-24) / `sizing` (frozen 04-25) while live runs the 07-03 pipeline versions — including via the default pinned `renquant_backtesting.wf_gate`, whose `sim_driver` shells into the umbrella kernel |
| F-10 | `model_content_sha256` × 4 | Four independent fingerprint impls; the umbrella production-calibrator fit + WF stamping still use a stale local denylist copy while the verifier imports a different one |
| F-11 | `fingerprint_config` fork | The P-CONFIG-FP hard fail-closed gate is hand-rolled in umbrella but a `renquant_common` import in pipeline — one edit to common splits live vs sim fingerprints |
| F-17 | Ungated weekly tournament | Per-ticker buy-admission models are trained and written straight to production every Sunday with acceptance auto-disabled and no delegation to pinned code |

Cross-cutting root cause: the strangler-fig migration stalled mid-flight —
lifts happened (pipeline, backtesting, execution, common) but the umbrella
originals were neither frozen nor deleted, every entrypoint chooses its own
side, and nothing in CI notices. Recommended sequencing in §7.

---

## 1. Kernel mirror drift inventory (D1)

### 1.1 What actually runs where (ground truth, verified)

| Path | Entrypoint | Kernel executed |
|---|---|---|
| Daily live/paper (scheduled) | `scripts/daily_104.sh` → `daily-bridge` (`RQ_DAILY_RUNNER=multirepo` default, `scripts/daily_104.sh:376-381`) → `renquant_orchestrator.live_bridge.bootstrap_multirepo` | **Pinned `renquant_pipeline.kernel`** for every top-level stem present in the pipeline repo; umbrella `kernel/` for the rest (mixed namespace) |
| Intraday sell loop | `scripts/intraday_sell_104.sh` (multirepo default, enforced by `scripts/subrepo_ops_contract.py`) | same mixed namespace |
| Sim / backtests | `scripts/run_sim_104.py:21-22,74-75` (`sys.path.insert` umbrella root + strategy dir; **no bridge**) | **Umbrella `kernel/` only** |
| WF gate + weekly promote | `scripts/weekly_wf_promote.sh:103-121`: default = pinned `renquant_backtesting.wf_gate` (fail-closed, umbrella `scripts/run_wf_gate.py` only on `RQ_WF_GATE_RUNNER=umbrella` rollback). BUT its `sim_driver.py:75-77,147` puts the umbrella root on `sys.path` and calls umbrella `sim.runner.run_backtest` | Gate runner = pinned backtesting; **the sims it scores run the umbrella `kernel/`**; fingerprint verification uses `renquant_backtesting.walk_forward.loader` (`wf_gate/runner.py:2197`) + `renquant_pipeline` PanelScorer (`runner.py:2243`). Promote step prefers `renquant_backtesting.forensics.model_acceptance`, falls back to umbrella `kernel.model_acceptance` (`weekly_wf_promote.sh:283-296`) |

The alias mechanism (`live_bridge.bootstrap_multirepo`, renquant-orchestrator
`src/renquant_orchestrator/live_bridge.py:259-292`) aliases `kernel.<stem>` →
`renquant_pipeline.kernel.<stem>` for each top-level stem in the pinned pipeline
kernel; only `kernel.preflight` and `kernel.panel_pipeline` fail closed. Every
other module that fails to import from the pinned pipeline is **silently skipped**
(`except Exception: continue`) and falls back to the umbrella copy (finding F-9).

### 1.2 Inventory numbers

Diffed with import-path normalization (`renquant_pipeline.kernel` ⇄ `kernel`):

| Bucket | Count | Meaning |
|---|---|---|
| Common files, byte-identical (normalized) | 91 | true mirror |
| Common files, materially different | **78** (41 with ≥40 differing lines) | drifted mirror |
| Umbrella-only files | **47** | never lifted; run from umbrella on BOTH sim and live paths |
| Pipeline-only files | **24** | live-only features; **structurally absent from sim** |

~30.1k umbrella lines vs ~34.6k pipeline lines across the 78 differing files.
The drift is **bidirectional** — 111 commits have touched the umbrella kernel
since the operating model was adopted (2026-05-25), while the pipeline kernel
advanced independently. This is dual maintenance, not a decaying frozen mirror.

Top drifted files (differing lines, umbrella-lines/pipeline-lines, last touch u/p):

| File | Δ lines | u / p lines | u last | p last | Direction |
|---|---|---|---|---|---|
| `panel_pipeline/job_panel_scoring.py` | 3069 | 1030 / 3861 | 06-12 | 07-03 | pipeline ahead |
| `panel_pipeline/shadow_scoring.py` | 1036 | 1227 / 431 | 07-02 | 06-26 | **umbrella ahead** |
| `pipeline/task_selection.py` | 643 | 341 / 868 | 05-24 | 07-03 | pipeline ahead |
| `persistence.py` | 423 | 2449 / 2586 | 06-14 | 06-29 | pipeline ahead |
| `sizing.py` | 317 | 187 / 468 | **04-25** | 07-03 | pipeline ahead |
| `pipeline/job_universe.py` | 280 | 374 / 592 | 05-25 | 07-01 | pipeline ahead |
| `walk_forward/loader.py` | 273 | 474 / 499 | 07-01 | 07-03 | **both ahead** (disjoint features) |
| `panel_pipeline/panel_scorer.py` | 242 | 396 / 244 | 07-02 | 07-03 | both |
| `portfolio_qp/tasks.py` | 214 | 3742 / 3934 | 06-02 | 07-03 | pipeline ahead |
| `panel_pipeline/hf_patchtst_scorer.py` | 183 | 350 / 335 | 07-02 | 06-01 | **umbrella ahead** |
| `pipeline/pp_inference.py` | 109 | 591 / 672 | 07-03 | 07-03 | both (same-day dual edits) |
| `exits.py` / `pipeline/task_sell.py` / `pipeline/task_rotation.py` / `preflight.py` / `decision_trace.py` | 78-113 each | — | 05-25–06-14 | 06-10–07-03 | pipeline ahead |

Classification of the one-side-only files:

- **Umbrella-only (47)** — the un-lifted remainder of `kernel-inventory.md`
  (metrics/, meta_label/, reconciliation/, registry/, `model_acceptance*.py`,
  `fundamentals.py`, `macro*.py`, `manifest_uri_resolver.py`, `drph.py`, …).
  Not a parity break per se (both sim and live resolve them from umbrella), but
  each is an ownership violation still pending its Track C2-C5 lift.
- **Pipeline-only (24)** — live-only capability sim can never exercise:
  `pipeline/task_parking_sleeve.py`, `pipeline/signal_direction.py`,
  `pipeline/task_data_integrity.py`, `pipeline/task_data_verification.py`,
  `pipeline/task_admission_shadow.py`, `panel_pipeline/fingerprint_dispatch.py`,
  `panel_pipeline/global_calibrator.py`, `panel_pipeline/model_contract.py`,
  `gate_registry.py`, `model_protection.py`, `pit_reader.py`,
  `broker_reconciliation.py`, `config_schema.py`, preflight
  `staleness`/`fundamentals_freshness`/`config_schema` tasks, and 6
  `portfolio_qp/` research modules. Pipeline's `pp_inference.py` wires the new
  tasks; the umbrella `pp_inference.py` that sim runs does not.

### 1.3 Is sim/live parity broken? Yes — by construction

Sim (and, worse, the WF gate that stamps promote decisions) executes the
umbrella kernel; the daily live run executes the pinned pipeline kernel. With
78 files drifted and 24 live-only modules, the sim that validates a candidate
is not running the code that will trade it. Concrete consequences are filed as
findings F-1..F-5 below.

### 1.4 D1 findings

**F-1 (P0) — live runs a known "silent mis-score" scorer bug that was fixed only in the mirror.**
`backtesting/renquant_104/kernel/panel_pipeline/hf_patchtst_scorer.py:83-96,166-172`
(umbrella, fixes of 2026-06-15: reconstruct cross-stock/FiLM layers on load +
fail closed on missing component weights) vs pinned
`renquant_pipeline/kernel/panel_pipeline/hf_patchtst_scorer.py:196`
(`model.load_state_dict(state, strict=False)`, zero `film`/`cross_stock`
references, last touched 2026-06-01). `kernel.panel_pipeline` is force-aliased
to the pipeline on live paths, so any live/shadow HF-PatchTST checkpoint with
cross-stock/FiLM layers is scored by the unfixed loader — the exact
silent-mis-score failure the 06-15 commit fixed, plus the 06-15 fail-closed
missing-weights guard is absent. Rule violated: CLAUDE.md §3.5 paired-PR
byte-equivalence. Fix: port the two 06-15 scorer commits (+ their tests) to
renquant-pipeline and re-pin. Owner: **renquant-pipeline**.

**F-2 (P0) — the WF-stamp contract is verified by THREE forked `WalkForwardModelLoader` implementations, one per leg of the promote chain.**
(a) Stamping/gate leg: pinned `renquant_backtesting.wf_gate` uses
`renquant_backtesting.walk_forward.loader.WalkForwardModelLoader`
(`wf_gate/runner.py:2197`); (b) sim leg: umbrella `adapters/sim.py` binds
models per bar via umbrella `kernel/walk_forward/loader.py:147-214` — a
12-char-prefix fuzzy matcher (`_fingerprints_match`,
`_any_fingerprints_match`) importing the stale local `model_content_sha256`
(F-10 site A) plus the umbrella-only `kernel/manifest_uri_resolver.py`
containment/digest hardening of 2026-07-01; (c) live-preflight leg: P-WF-GATE
verifies the same stamps with the pipeline loader
(`renquant_pipeline/kernel/walk_forward/loader.py:140-199`) routed through the
pipeline-only `panel_pipeline/fingerprint_dispatch.py` M6 schema-dispatch
(versioned `IdentityClaim`/`match_claims` + fail-closed `verify()`), 326-line
diff vs the umbrella copy. Divergent code on every side of one contract —
the structural generator of the recurring stamp-accepted-then-rejected
incident class (2026-05-27 / 06-22 / 07-01). The pipeline module's docstring
claims it is "the WF per-fold contract behind `weekly_wf_promote`", which is
not true today — that path runs the backtesting fork. Rule violated:
single-impl-imports-only; Universal Rule 5. Fix: finish the M6 migration so
stamping, sim binding, and preflight all import ONE loader +
`renquant_common.model_fingerprint`; until then, do not flip
`accept_legacy_stamps` to false while umbrella/backtesting stampers emit
legacy stamps. Owner: **renquant-pipeline (canonical loader), renquant-backtesting + RenQuant umbrella (consumers), renquant-common (fingerprint)**.

**F-3 (P0) — the WF gate / sim validates candidates on ~5-week-stale decision code.**
Sim and `run_wf_gate.py` execute umbrella `pipeline/task_selection.py`
(341 lines, frozen 2026-05-24) and `sizing.py` (187 lines, frozen 2026-04-25)
while live executes the pipeline versions (868 / 468 lines, both current to
2026-07-03, including RC-MISMATCH sizing fixes and flag-gated fractional
sizing). A model that passes the weekly gate was never evaluated under live
selection/sizing semantics; promotion evidence is generated by different code
than production. This holds on the DEFAULT promote path too: the lifted
`renquant_backtesting.wf_gate` shells into `sim_driver.py`, which
`sys.path.insert`s the umbrella root and calls umbrella
`sim.runner.run_backtest` (`wf_gate/sim_driver.py:75-77,147`) — the pinned
gate scores sims executed by the drifted umbrella kernel. Rule violated:
sim/live parity premise of the promote gate (Universal Rule 5) + §3.5 mirror
invariant. Fix: run sim/WF-gate through the same pinned-pipeline bootstrap as
live (extend `bootstrap_multirepo` to the sim/gate entrypoints), or freeze +
re-mirror. Owner: **RenQuant umbrella (entrypoints) + renquant-orchestrator
(bridge) + renquant-backtesting (sim_driver)**.

**F-4 (P1) — 24 live-only pipeline modules are structurally invisible to sim.**
Parking sleeve, signal-direction, data-integrity/verification tasks, admission
shadow, model_protection, gate_registry, pit_reader (list in §1.2) exist only
in the pinned pipeline; umbrella `pp_inference.py` (the sim pipeline) does not
wire them. Sim can neither exercise nor regress live-only behavior (e.g. the
parking sleeve moves real cash daily; no sim ever models it). Rule violated:
Universal Rule 1 premise that sim validates the production pipeline. Fix: after
F-3's bootstrap unification, delete the umbrella copies of lifted stems so sim
imports the pinned pipeline wholesale. Owner: **RenQuant umbrella**.

**F-5 (P1) — the 2026-07-01/02 shadow-ntfy feature is dark on the live path (deployed-but-dark class).**
The feature (9 review rounds, PR #426 series) landed in umbrella
`kernel/panel_pipeline/shadow_scoring.py` (1227 lines, `in_primary_admitted`
producer at line 981) and its consumer in `live/runner.py:891-903` — but on
live runs `kernel.panel_pipeline` is force-aliased to the pipeline, whose
`shadow_scoring.py` (431 lines, last touched 2026-06-26) never produces the
new fields. The runner-side consumer executes against a producer that does not
exist in production. Rule violated: CLAUDE.md §3.5 (change must land in BOTH);
session rule "deployed-but-dark is not done". Fix: port the shadow-ntfy
`shadow_scoring.py` changes to renquant-pipeline and re-pin. Owner:
**renquant-pipeline**.

**F-6 (P1) — dual maintenance is live and un-enforced: the §3.5 byte-equivalence invariant has no test.**
111 commits touched the umbrella kernel since 2026-05-25; some land
umbrella-first (shadow-ntfy, wf-gate hardening, hf scorer fixes), others
pipeline-first (selection, sizing, panel scoring, persistence, fractional).
CLAUDE.md §3.5 says paired PRs "verify MD5 equivalence
(`tests/test_c211_panel_pipeline_lift.py`, etc.)" — that test does not exist in
the repo, and no CI check compares the two kernels (78/169 files differ today).
Fix: add a parity CI job that diffs umbrella `kernel/` against the *pinned*
pipeline kernel (normalized imports) and fails on new drift, with an explicit
allowlist for the 47 un-lifted umbrella-only files. Owner: **RenQuant umbrella**.

**F-8 (P1) — bootstrap alias falls back to umbrella silently for all but two modules.**
`live_bridge.bootstrap_multirepo` (renquant-orchestrator
`src/renquant_orchestrator/live_bridge.py:271-274`): any `kernel.<stem>` whose
pipeline import raises is skipped (`except Exception: continue`) and the
umbrella copy shadows it at import time — only `kernel.preflight` and
`kernel.panel_pipeline` fail closed. A pipeline-side regression (missing dep,
syntax error) silently reverts part of the live run to umbrella code with only
a count in a stderr line. Rule violated: fail-closed doctrine
(`RENQUANT_OPS_FAIL_CLOSED`). Fix: fail closed (or at minimum alert per-module)
when an expected pipeline kernel module fails to import; assert the aliased
module count against the pinned repo's manifest. Owner: **renquant-orchestrator**.

**F-7 (P1) — the umbrella `strategy_config.json` duplicate has drifted from the pinned policy: sim defaults to a different PRIMARY model than live.**
Umbrella `backtesting/renquant_104/strategy_config.json` declares
`ranking.panel_scoring.kind: hf_patchtst` (primary) with an `xgb` shadow; the
pinned `renquant-strategy-104` (`dd337d45`, `configs/strategy_config.json`)
declares `kind: xgb` primary with `hf_patchtst` shadow — the live bridge swaps
in the pinned config (`live_bridge._with_pinned_strategy_config`), but
`scripts/run_sim_104.py` and every sim/analysis script that defaults to the
umbrella file evaluates the wrong primary. Rule violated: strategy policy is
owned by `renquant-strategy-104` (roles table); duplicated policy files drift.
Fix: make sim entrypoints resolve the pinned strategy config by default (same
`_with_pinned_strategy_config` path as live) and mark the umbrella copy
experimental-only. Owner: **RenQuant umbrella**.

Reachability note for F-1: under the pinned live config the buggy pipeline
`hf_patchtst_scorer` scores the **shadow** model on every daily run (shadow
model-comparison evidence + shadow-ntfy picks), not the primary; it becomes the
primary-path scorer the moment PatchTST is re-promoted by pin bump — which is
exactly how the 06-05→06-23 promote flip-flops were executed.

Additional F-2 evidence (verification-fork mechanics): the umbrella loader
matches fingerprints with a 12-char-prefix fuzzy matcher
(`kernel/walk_forward/loader.py:147-214`: `_normalize_fingerprint`,
`_fingerprints_match`, `_any_fingerprints_match`) importing the stale local
`model_content_sha256` (F-10 site A), while the pipeline loader
(`renquant_pipeline/kernel/walk_forward/loader.py:140-199`) routes through
`fingerprint_dispatch` (`IdentityClaim`/`build_claim`/`match_claims` +
fail-closed `verify()` for v1 stamps). The umbrella loader is bound into every
sim bar via `adapters/sim.py` (`model_as_of(today)`) and into
`scripts/run_wf_gate.py` — a stamp/verify mismatch visible to one matcher is
invisible to the other by construction. 326-line diff between the two loaders.

---

## 2. Hand-copied implementations (D4) — the triple-impl class

**F-10 (P0) — `model_content_sha256` still has FOUR independent implementations; the umbrella production-calibrator path uses a stale one.**
Sites: (A) umbrella `kernel/panel_pipeline/panel_scorer.py:43,88,108` — full
local denylist implementation (`_MUTABLE_ARTIFACT_KEYS`,
`_PREDICTIVE_CONTENT_HINTS`, `model_content_sha256`), imports nothing shared;
(B) pipeline `kernel/panel_pipeline/panel_scorer.py:53-59` — imports
`renquant_common.model_fingerprint` (the 2026-07-01 unification,
`renquant-common>=0.8.1` pinned in pipeline `pyproject.toml:12-16`);
(C) umbrella `kernel/panel_pipeline/model_fingerprint.py:38`
`compute_model_fingerprint` — hashes ONLY `booster_raw_json` (a structurally
different field set), self-described "canonical", imported nowhere except its
own test — an unexploded third digest; (D) renquant-model's
`fit_calibrator_alpha158_fund.py` 11-field allowlist, documented as diverged in
umbrella `scripts/verify_calibrator_scorer_binding.py:12-18`. Reachability:
site A is imported by umbrella `kernel/walk_forward/loader.py:169`,
`kernel/panel_pipeline/shadow_scoring.py`, and
`scripts/fit_calibrator_alpha158_fund.py:32` — which writes the **production**
calibrator (`backtesting/renquant_104/artifacts/panel-rank-calibration.json`)
— and `scripts/stamp_walkforward_fingerprints.py:35,114`; meanwhile the
binding verifier checks against the pipeline import, i.e. the stamping path and
its own verifier use different hash algorithms. Rule violated:
single-impl-imports-only (the 05-27/06-22/07-01 incident class). Fix: umbrella
depends on `renquant-common>=0.8.1`; delete sites A and C; import
`renquant_common.model_fingerprint` everywhere. Owner: **renquant-common
(canonical), RenQuant umbrella + renquant-model (consumers)**.

**F-11 (P0) — `fingerprint_config` (P-CONFIG-FP hard gate) is a hand-rolled umbrella module vs a shared import in pipeline.**
Umbrella `kernel/config_consistency.py:51,104,116` is fully self-contained;
pipeline has no local file and imports `renquant_common.config_consistency`
(`preflight.py:812-813`, `preflight_pipeline/tasks/config_fingerprint.py:76-78`).
Umbrella's `preflight_pipeline/tasks/config_fingerprint.py:76` is byte-identical
to pipeline's except the import — proving the fork is ownership-only today, on
a strict-by-default fail-closed gate (`job_panel_scoring.py:228-236`,
`strict_config_consistency=True`) whose whole purpose is preventing silent
no-trade incidents. Any future edit to `renquant_common.config_consistency`
changes live fingerprints while sim/WF-gate keep hashing with the frozen local
copy → guaranteed P-CONFIG-FP mismatch at the next promote. Rule violated:
single-impl-imports-only; kernel-inventory already marked this copy "the
remaining shadow". Fix: delete umbrella `kernel/config_consistency.py`, import
from renquant-common. Owner: **RenQuant umbrella**.

**F-12 (P1) — `kernel/execution/*` fill-quantity/epsilon semantics forked.**
Pipeline added `_POSITION_EPS = 1e-9` (`execution/backend.py:26`) and
`resolve_fill_quantity()` (`execution/types.py`, fractional-share negotiation
with `supports_fractional`); umbrella copies still truncate with bare
`int(intent.shares)` (`execution/backend.py:139`, `backend_sim.py:134,180,182`,
`backend_lean.py:129,154`) with no epsilon. Sim/WF-gate no longer model live
fill/quantity behavior; any fractional-sizing validation in sim is skewed vs
production. (`fees.py`/`slippage.py` remain byte-identical — the sync
discipline is achievable, it just stopped.) Fix: lift `kernel/execution/*` to
one shared package imported by both. Owner: **renquant-execution or
renquant-common**.

**F-13 (P1) — regime thresholds hand-copied and config-blind in the eval path.**
Umbrella top-level `kernel/hmm_regime_labels.py:30-36` hardcodes
`BEAR_VOL_20D_THR=0.35` etc. with a "keep in sync with kernel/regime.py"
comment, while the production detector reads the same values
config-overridably (`kernel/regime.py:425-434`,
`regime_cfg.get("bear_vol_threshold", 0.35)`). Identical today; the first
strategy-config override silently splits eval-IC regime labels from production
regime labels. Fix: import the defaults (or accept `regime_cfg`) instead of a
second copy. Owner: **renquant-common (hmm_regime_labels is its contract)**.

**F-14 (P1) — `kernel/artifact_resolver.py` is a hand-maintained cross-repo mirror with no CI diff check.**
Both copies' docstrings say "keep the two copies byte-similar — divergence
between them is exactly the bug class this module exists to kill"; bodies are
identical today but the docstrings have already been independently rewritten.
The module was created because the umbrella previously had FOUR ad-hoc
resolvers. Fix: promote to a shared import or add a CI byte-diff until the
strangler-fig retirement. Owner: **renquant-common**.

**F-15 (P2) — two independent triple-barrier implementations.**
`kernel/triple_barrier.py` (`TripleBarrierConfig`, alpha/beta convention;
consumed by `training_panel/pp_panel_training.py`) vs
`kernel/meta_label/triple_barrier.py` (`apply_triple_barrier`, pt/sl
convention; consumed by `meta_label/labeler.py`) — same AFML algorithm, no
shared code; a correctness fix to one will not propagate. Fix: one
parameterized primitive. Owner: **renquant-backtesting** (per
kernel-inventory B2).

**F-16 (P2) — three inline `ZoneInfo("America/New_York")` re-derivations despite `live/clock.py` being the declared sole authority.**
`adapters/runner_ext_sell.py:169,207` (cites the canonical pattern, then
re-derives inline), `scripts/check_software_stops_liveness.py:64-65`,
`scripts/deadman_check.py:31,42` (duplicates the `NY` constant verbatim). The
heavier calendar logic (`pandas_market_calendars` session helpers in
`kernel/data.py`) is byte-identical umbrella/pipeline — clean. Fix: import
`live.clock.ny_now()`/`trading_date()`. Owner: **RenQuant umbrella**.

---

## 3. Ownership violations + pipeline-primitive compliance (D2, D3)

Positive control first: most retrain/promote wrappers follow a compliant
delegate-to-pinned-subrepo-first, fail-closed pattern
(`scripts/subrepo_module_delegate.py`; e.g. `retrain_alpha158_linear.sh`,
`weekly_wf_promote.sh`, `stamp_walkforward_fingerprints.py:213-223`,
`monthly_calibrator_atomic_swap.py`). The findings below are the paths that
deviate.

**F-17 (P0) — the weekly per-ticker tournament trains and writes production admission models with no gate and no delegation.**
`scripts/weekly_tournament_retrain.sh` (scheduled Sun 06:00 PT,
`com.renquant.weekly-tournament-retrain.plist`) → `scripts/train_104.py
--skip-panel --force`. Unlike every sibling wrapper it has NO multirepo
delegate branch; it runs umbrella `kernel.pipeline.pp_training_full`
directly. With `--skip-panel`, `train_104.py:244-249` auto-disables
`ModelAcceptanceGate` ("no candidate panel artifact is produced"), and the
per-ticker RF/XGB/Q-learning exports — which gate UNIVERSE BUY-ADMISSION —
are written straight to production `backtesting/renquant_104/models/<TICKER>/`
with no staging and no WF/sanity gate (`train_104.py:250-256` itself calls
these "still production writes"). Rules violated: training internals outside
renquant-model; selection/admission logic outside renquant-pipeline; Universal
Rules 1 and 5 (ungated promotion). Fix: delegate like the other wrappers and
stage tournament output behind an acceptance gate. Owner: **renquant-model
(training) + renquant-pipeline (admission)**.

**F-18 (P1) — pre-open cancel gate: broker mutation in scripts/ with a fail-OPEN fallback.**
`scripts/preopen_cancel_gate.py:240-438` computes an inline overnight-severity
gate and cancels live orders directly via
`TradingClient(...).cancel_order_by_id` (line 367) — broker semantics outside
`live/` and outside renquant-execution. The wrapper prefers
`renquant_execution.preopen_cancel_gate` but on import failure falls back to
the umbrella script with only a WARN (`preopen_cancel_gate.sh:72-79`) unless
`RQ_PREOPEN_GATE_STRICT=1` — the opposite default of every other wrapper.
Not graded P0 because the fallback runs the same gate logic (copy-drift
between the execution and umbrella copies was not demonstrated); the pattern
is the same mirror-drift generator as §1 on an order-cancelling path. Fix:
flip the strict default to fail-closed; finish the renquant-execution port and
delete the umbrella copy. Owner: **renquant-execution**.

**F-19 (P1) — monthly meta-label retrain decides acceptance inline and swaps prod in the same script.**
`scripts/monthly_meta_label_retrain.sh:154-194`: training correctly delegates
to `renquant_model_common.meta_label_exit` (lines 126,139), but the promote
decision is hardcoded thresholds in shell (`auc < 0.52`, `n_events < 100`,
balance ∈ [0.30,0.70], `n_features < 25`) followed by an in-script
`mv "$NEW_ARTIFACT" "$PROD_ARTIFACT"` (line 194) — the artifact drives live
exit vetoes. Same pattern at `scripts/monthly_calibrator_refresh.sh:336-365`
(inline `pool_ic` drop / collapse-floor checks; at least routed through the
shared atomic-swap helper). Rule violated: promote thresholds are acceptance
policy, owned by renquant-model/pipeline gates, not umbrella shell. Fix: move
thresholds into a gate function the scripts call. Owner: **renquant-model**.

**F-20 (P1) — `RunnerAdapter` order-dispatch layer lives in `backtesting/renquant_104/adapters/`, a third location that is neither `live/` nor renquant-execution.**
`adapters/runner.py:1132,1454` (~2,189 lines) computes sell/buy quantities and
calls `broker.place_order(...)` directly on every live cycle
(`live/runner.py:419-420`). Actual broker classes stay in `live/*.py`
(mitigation), but order-construction/dispatch orchestration is umbrella
adapter code — operating-model "Open Migration Work" item 4, still pending.
Note `adapters/` is NOT covered by the kernel alias: this code runs from the
umbrella working tree on live paths, unpinned. Fix: lift the dispatch layer
into renquant-execution (or `live/`). Owner: **renquant-execution**.

**F-21 (P1) — kernel's Task/Job/Pipeline framework is a self-contained re-implementation, not `renquant-common` primitives.**
`backtesting/renquant_104/kernel/pipeline/pipeline.py:1-6,45,63`
("Self-contained: only stdlib. No common/ imports.") is the base framework for
the entire ~56k-line kernel tree, mirrored on the pipeline side rather than
imported from renquant-common (Universal Rule 1). Structural debt, not
behavior risk (the framework works and both sides carry it). Fix: lift the
ABCs to renquant-common and re-export. Owner: **renquant-common +
renquant-pipeline**.

**F-22 (P1) — dormant umbrella duplicates of lifted workflows.**
(a) `scripts/run_wf_gate.py` — a ~2.5k-line re-implementation of the promote
gate kept as the `RQ_WF_GATE_RUNNER=umbrella` rollback
(`weekly_wf_promote.sh:103-121`); (b) `scripts/train_panel_linear.py:141-149`
inline sklearn `.fit()` behind `RQ_ALPHA158_LINEAR_RUNNER=umbrella`;
(c) `scripts/execute_shadow_orders.py:1-40` — a manual operator tool with its
own ad-hoc `ManualExecutionPipeline` Task/Job classes submitting real Alpaca
orders via the SDK, bypassing `live/*.py` brokers (dry-run default,
wash-sale/earnings refusals present). Each is a divergence seed on a
money-adjacent path. Fix: delete after parity verification of the lifted
paths; route manual execution through the shared broker client. Owner:
**renquant-backtesting / renquant-model / renquant-execution respectively**.

**F-23 (P2) — offline research tooling with inline policy (allowed carve-out, noted for the record).**
`scripts/ab_bypass_ticker_gate.py:56-90`, `scripts/validate_buy_logic.py:62-146,219-240`
(inline promote heuristics for A/B sims, never write back to config);
`scripts/screen_watchlist.py:9-19` (scheduled Sun 12:05 PT but advisory-only —
writes a markdown report, never mutates the watchlist);
`scripts/_meta_label_train.py` + `_meta_label_*.sh` (superseded research
trainers, unscheduled); `scripts/production_runner.py` (self-described
deprecated standalone scorer, execution disabled); read-only `TradingClient`
queries in `fetch_alpaca_*.py`/`protective_census.py`. Fix: archive/delete the
superseded ones; optionally relocate watchlist policy to renquant-strategy-104.
Owner: **RenQuant umbrella (cleanup)**.

---

## 4. Dead / shadowed / dark code (D5)

The 78 drifted mirror files of §1 are all dead-on-live by aliasing (known
pattern, covered by F-1..F-6). Beyond those:

**F-24 (P1) — `kernel/intraday_governor.py`: a merged protective-risk primitive with zero production importers.**
249 lines + tests, merged 2026-06-16 ("intraday protective-action governor
primitive (#26)"), never wired into `live/runner.py`, `adapters/runner.py`, or
`kernel/intraday*.py`; its docstring describes an integration that never
happened. A named protection that looks live but is unreachable — false sense
of safety. Fix: wire it or delete it (deployed-but-dark rule). Owner:
**renquant-pipeline** (it is runtime protection).

**F-25 (P1) — challenger comparison is an end-to-end dead feature.**
`kernel/challenger.py` (`log_decision`/`compare_window`) has zero non-test
importers; `scripts/finalize_challenger.py` re-implements its own sqlite
reader instead of importing it; `kernel/pipeline/pp_inference.py:261-274`
self-documents "live wiring not yet [complete] … will not record decisions";
`strategy_config.json` still carries `acceptance.challenger.enabled: false`.
Flag → warning stub → orphaned module. Fix: finish the wiring or delete flag +
module + duplicate reader. Owner: **renquant-backtesting**.

**F-26 (P1) — sim-smoke acceptance gates G9-G11 permanently no-op.**
`kernel/model_acceptance.py` reads `metadata.sim_smoke.{apy,sharpe,turnover_ratio}`
at promote time, but the only writer
(`kernel/sim_smoke.py::add_smoke_metrics_to_artifact`) has zero production
call sites (tests only; `run_smoke_test()` is an explicit operator-wiring
stub). Every real promotion logs "no sim_smoke.apy on staging (skip)" — a
silent skip that reads like a pass. Fix: wire the writer into the weekly
promote chain or remove the gates. Owner: **renquant-backtesting**.

**F-27 (P2) — misc dead code.** `kernel/acceptance_entry_ic.py` (tests-only
importers); `backtesting/renquant_104/_archive/` (205 KB of old strategy
configs, self-documented out-of-scope, inert); dual top-level `./kernel/`
namespace package merged via `pkgutil.extend_path` with
`backtesting/renquant_104/kernel/` (works, already broke once on 2026-05-20,
sys.path-order-fragile — consolidate when convenient). Owner: **RenQuant
umbrella**.

Live-reachability confirmations (not findings): `kernel/env_fingerprint.py`
(via `artifact_contract.build_run_bundle` ← `adapters/runner.py:1946`),
`kernel/registry/mlflow_registry.py` (promote + weekly calibrator refresh),
`kernel/{earnings_surprise,insider_trades,macro_per_ticker}.py` (panel
training tasks) are all reachable — they are ownership debt (→ base-data/
backtesting per kernel-inventory), not dead code.

---

## 5. Data-in-git (D6, Universal Rule 4)

Headline: **172 tracked files > 1 MB totaling ~1.46 GB** (of ~1.7 GB total
tracked blob weight). No `.gitattributes`, no LFS, and no size guard anywhere
(`scripts/audit_repo_hygiene.py` has no large-blob check).

**F-28 (P1, egregious) — ~1.2 GB of per-ticker RL Q-tables committed as raw JSON.**
`backtesting/renquant_10{3,4}/models/**/*-qtable.json` — 984 files (130 over
1 MB), the output surface of the weekly tournament (F-17), with zero
`.gitignore` coverage: the production model store lives in normal git history.
Rule 4 violated directly. Fix: manifest + external store, `git rm --cached`,
add ignore + a pre-commit size guard. Owner: **RenQuant umbrella (store) +
renquant-artifacts (manifest home)**.

**F-29 (P1) — `backtesting/data/**`: 224 MB of LEAN sample market data tracked despite an existing `.gitignore` rule.**
1,116 files committed before the `backtesting/data/` ignore rule was added and
never `git rm --cached` — documented intent vs reality gap. Fix: one-time
untrack commit. Owner: **RenQuant umbrella**.

**F-30 (P1) — `backtesting/renquant_103/artifacts/baseline-results.pkl`: a single 99 MB dataframe pickle in git.** Fix: externalize + manifest pointer or
delete (103 is legacy). Owner: **RenQuant umbrella**.

**F-31 (P2) — small-scale hygiene.** `artifacts/patchtst_shadow/**` model
`.pt`/parquet files (~2 MB — fine today, wrong seed pattern), 35 notebook
OHLCV parquet caches, 3 notebooks with ~7 MB embedded outputs, 16 tracked
PNGs, one `.ipynb_checkpoints` file. The committed prod-artifact JSONs under
`backtesting/renquant_104/artifacts/` (299 files / 57.6 MB, individually
small) match the established convention and are NOT a finding; the 123 dated
`walkforward_*` snapshots deserve a growth watch. Owner: **RenQuant umbrella**.

---

## 6. Compliance scorecard vs the charter

| Charter rule | Verdict |
|---|---|
| Universal Rule 1 (everything a common-primitive pipeline) | PARTIAL — kernel has its own Task/Job framework (F-21); main wrappers are thin delegates (compliant); tournament chain bypasses entirely (F-17) |
| Universal Rule 4 (data by manifest, not git) | VIOLATED at scale (F-28/29/30); prod-JSON convention otherwise respected |
| Universal Rule 5 (promotion via immutable fingerprints) | STRUCTURALLY AT RISK — three forked verifiers (F-2), four fingerprint impls (F-10), gates that silently skip (F-26) |
| CLAUDE.md §3.5 (byte-equivalent mirror + paired PRs + MD5 tests) | VIOLATED — 78/169 files drifted bidirectionally, enforcement test does not exist (F-6) |
| Kernel-alias rule (umbrella kernel is a mirror on live paths) | HOLDS mechanically, but with silent per-module fallback (F-8) and mixed namespace |
| Single-impl-imports-only (fingerprints/hashes) | VIOLATED (F-10/F-11), migration in flight pipeline-side (M6) |
| Flags default-OFF + byte-inertness | Broadly respected (software stops, fractional, hysteresis all flag-OFF); one orphaned flag (F-25) |
| No selection logic in umbrella scripts (#210 §6) | Mostly respected on scheduled paths; tournament admission models are the exception (F-17); research tools carve-out applies (F-23) |

## 7. Recommended fix sequencing (for the follow-up work, not this PR)

1. Port the two stranded umbrella-side fixes to renquant-pipeline (F-1 scorer
   mis-score fix; F-5 shadow-ntfy) and re-pin — small, immediate parity wins.
2. Kill the fingerprint forks: umbrella + backtesting import
   `renquant_common.model_fingerprint` / `config_consistency` /
   `fingerprint_dispatch`-successor before any `accept_legacy_stamps` flip
   (F-2, F-10, F-11).
3. Route sim/WF-gate through the pinned pipeline kernel (extend the bridge to
   `run_sim_104.py` + `wf_gate/sim_driver.py`) so promotion evidence is
   generated by production code (F-3, F-4).
4. Add the kernel-parity CI check with an unlifted-file allowlist (F-6) and
   make the bootstrap alias fail closed per module (F-8).
5. Gate or delegate the tournament chain (F-17); fail-close the pre-open gate
   fallback (F-18).
6. Data-in-git cleanup (F-28/29/30) — mechanical, coordinate with the backup
   jobs.

