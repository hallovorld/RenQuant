# ADR 0001 — renquant105 intraday cross-repo topology + `renquant-strategy-105`

**Date:** 2026-06-27
**Status:** **Proposed** — gated on operator + Codex approval. This ADR **MUST be
merged BEFORE any renquant105 topology is created** (no `renquant-strategy-105`
repo, no pin order, no `# COMPAT-105-SHIM`, no intraday subrepo PRs land until
this is ratified).
**Owner:** umbrella `RenQuant` (cross-repo architecture lives here ONCE).
**Supersedes:** the cross-repo scope previously described inside orchestrator
PR #198. That PR is now re-scoped to ORCHESTRATION and references this ADR as the
authoritative cross-repo contract.

> This is the first Architecture Decision Record in `RenQuant/doc/arch/`. There
> was no prior ADR convention (existing arch docs are flat descriptive files such
> as `strategy-104.md`, `subrepo-operating-model.md`). This ADR establishes the
> `doc/arch/adr/NNNN-<slug>.md` convention; future cross-repo decisions follow it.

Read with, and do **not** duplicate, the canonical operating model
[`doc/arch/subrepo-operating-model.md`](../subrepo-operating-model.md) and the SOP
[`doc/arch/multirepo-sop.md`](../multirepo-sop.md). This ADR only records the 105
*delta* to that operating model; everything it does not change is inherited.

---

## 1. Context

renquant104 (the active daily Panel-LTR cross-sectional strategy, see
[`doc/arch/strategy-104.md`](../strategy-104.md)) is the production topology:
`renquant-base-data` → `renquant-common`/`renquant-pipeline` →
`renquant-model` (factory) → `renquant-artifacts` (registry) →
consumed by `renquant-strategy-104` / `renquant-pipeline` (runtime) /
`renquant-backtesting` / `renquant-orchestrator`, with `renquant-execution`
performing broker actions and `RenQuant` pinning the whole assembly in
`subrepos.lock.json`.

renquant105 is a **proposed intraday (open→close, bounded-turnover) subsystem**
that runs **alongside** 104. Its master design lives in the orchestrator repo at
`doc/design/2026-06-27-renquant105-intraday-system.md` (PR #198) and the
accompanying milestone suite. That design is correctly scoped to **orchestration
responsibilities** (pinning, run-bundle stamping, `--strategy` routing, the
shadow→graduate flow). It also *describes* a cross-repo topology change — a new
`renquant-strategy-105` repo, new artifact contracts spanning
base-data/model/pipeline/execution, a lock/pin migration, and forbidden-import
boundaries. Per the operating model, **cross-repo architecture must live ONCE
under `RenQuant/doc/arch/` and be referenced, not defined inside a subrepo PR**
(orchestrator PR #198 §6.1, "Ownership & authority", finding #8 of Codex's
holistic review). This ADR is that authoritative record.

105 is **not greenfield**: the parked intraday subsystem (Alpaca-IEX fetch,
`*Store` caches, `hourly/minute_features`, the dark websocket consumer, the
reusable fail-closed gate stack) already exists, disabled since 2026-05-04. 105 =
re-activation + an intraday model + tighter gates + a gated buy path, mostly
config/wiring.

**Hard default (operator mandate, inherited verbatim):** live intraday TRADING is
DISABLED at the start. `intraday_buys_enabled=false`; Phases 0–2 place **zero**
live intraday orders (data/train/shadow only); live intraday is **Phase 3 only**,
behind the validation gate + an armed kill-switch. Turning it on is a deliberate,
gated act, never the default.

## 2. Decision

Adopt the renquant105 intraday subsystem as a topology change to the canonical
operating model, **conditional on operator + Codex ratification of this ADR**,
with the following invariants. Until this ADR is merged, the §6 matrix of PR #198
is a *proposal referenced by an orchestration design*, not an executed change:
`renquant-strategy-105` is **not** created, no pin order runs, and 105 stays a
design.

### 2.1 New repo: `renquant-strategy-105` (Proposed, not yet created)

A **new pinned subrepo**, mirroring `renquant-strategy-104`. Registered as
**Proposed** in `subrepos.lock.json` under `pending_subrepos` (the single source
of truth for the repo map); it graduates into `subrepos` with a real commit pin
only when it is created at M0 and integration-tested.

**Owns (and ONLY owns):**
- the 105 strategy config skeleton (policy/thresholds, mirrors 104's config bundle);
- the **point-in-time universe manifest** — one frozen + fingerprinted liquid,
  coverage-gated intraday universe (~40–60 names) per decision date, constructed
  only from information available at that date (no survivorship/look-ahead);
- the strategy **config-fingerprint** emitted into the run/model bundle (the same
  `config_fingerprint` contract 104 uses).

**Must NOT own (hard boundary, same as strategy-104 and as CLAUDE.md sets for the
orchestrator):**
- no model-training internals (those live in `renquant-model`);
- no data ingestion / bar-store internals (those live in `renquant-base-data`);
- no decision-tree / gate-kernel internals (those live in `renquant-pipeline`);
- no broker adapters (those live in `renquant-execution`);
- it is **policy/config-only** and **does not submit orders directly** — the
  orchestrator assembles strategy + data + artifact + pipeline + execution.

### 2.2 Forbidden imports / dependency direction

105 stays strictly within the established boundaries; the dependency arrows do not
change direction:

- `renquant-strategy-105` may import `renquant-common` contracts only. It must NOT
  import `renquant-model` (factory) internals, `renquant-pipeline` kernel
  internals, or `renquant-execution` broker adapters. It consumes models by
  `artifact_path`, never by importing the factory.
- `renquant-orchestrator` keeps its CLAUDE.md boundaries unchanged: **no model
  training internals, no signal/decision-tree internals, no broker adapters** in
  the orchestrator. The 105 `--strategy` route adds wiring, not internals.
- `renquant-pipeline` consumes the base-data intraday loader via the canonical
  contract; it does NOT re-implement ingestion. `renquant-model` consumes the
  base-data session-return surface; it does NOT own data ingestion.
- The umbrella `RenQuant` adds **no logic** — it pins and wires only. Any
  temporary umbrella-side compatibility shim is tagged `# COMPAT-105-SHIM` with a
  removal ticket and deleted once the owning repo's pin lands (never a third copy).

### 2.3 Artifact contracts (fingerprint/handoff chain), consistent with 104

The model→pipeline→execution→backtesting handoff for 105 uses the **same
fingerprint-and-pin discipline as 104**. Each capability has exactly ONE owning
repo; consumers use the canonical contract; the umbrella only pins/wires (no
triplication):

| Capability | Owner repo | Consumer | Umbrella role |
|---|---|---|---|
| intraday bar ingestion + `*Store` (incremental, append-only) | `renquant-base-data` | pipeline (canonical loader) | pin base-data; no logic |
| intraday features + session-horizon (open→close) forward-return surface | `renquant-base-data` (data) → contract consumed by `renquant-pipeline` | model / decision | pin both |
| intraday label (triple-barrier, open→close) + CPCV / embargo-in-bars | `renquant-model` | — (published as artifact) | pin model |
| G1–G8 gate stack + decision-ledger wiring | `renquant-pipeline` kernel | — | pin pipeline |
| 105 config / point-in-time universe manifest / config fingerprint | `renquant-strategy-105` (NEW) | orchestrator bridge | pin strategy-105 |
| broker-contract checks (M0.5: intraday-margin/BP fields, rejection + deficit handling, fail-closed on field migration) | `renquant-execution` | pipeline | pin execution |
| pins, run-bundle stamping, `--strategy` routing, shadow→graduate | `renquant-orchestrator` | — | the wiring layer |

**Fingerprint/handoff flow (mirrors 104):** `renquant-model` trains the intraday
label and **publishes a fingerprinted artifact to `renquant-artifacts`**
(`promotion_status: shadow` until accepted) — the factory never writes into a
consumer. `renquant-strategy-105` emits the universe + config fingerprint →
`renquant-orchestrator` stamps it into the run bundle + the model bundle (same
`config_fingerprint` contract 104 uses) → `renquant-pipeline` preflight asserts
the live feature space + universe match the 105 artifact, **fail-closed on
mismatch**. Live promotion requires the immutable fingerprint set (strategy
config, data, model artifact, calibrator, code commits, acceptance metrics), same
as 104.

### 2.4 Lock / pin migration (`subrepos.lock.json` + promote-pin flow)

- `renquant-strategy-105` is added to the umbrella repo map by **registering it in
  `subrepos.lock.json` under `pending_subrepos`** as Proposed (this ADR's
  companion change). `RENQUANT_REPOS.md` is AUTO-GENERATED from the lock by
  `scripts/sync_subrepo_docs.py` and rendered only from `subrepos[]`, so a Proposed
  entry in `pending_subrepos` introduces **no doctor drift** (verified against
  `subrepo_doctor.py` / `render_repo_registry`). It graduates into `subrepos[]`
  with a real `commit` pin once created at M0 and integration-tested.
- **Pin order (paired-PR matrix):** base-data merges the intraday loader + session
  return-surface contract **first** (with a contract test the pipeline imports) →
  `renquant-pipeline` merges against the pinned base-data → `renquant-model`
  against the pinned data contract → `renquant-strategy-105` + `renquant-orchestrator`
  pin **last**. Each pin advances only after the owning repo's CI is green and the
  umbrella integration check passes; the existing atomic promote-pin flow
  (`refresh_subrepo_lock.py`, CI-green gate) governs every pin advance.
- **The orchestrator bridge routes by `--strategy` → NO orchestrator code change.**
  104 keeps running unchanged; 105 is selected by flag.

### 2.5 Rollback

105 is a strictly additive, default-OFF topology; 104 is untouched and remains the
last-known-good.

- **Un-pin 105:** revert the `subrepos.lock.json` pin advance to the prior
  104-only assembly (last-known-good pin). Because 105 runs behind `--strategy`
  and `intraday_buys_enabled=false`, reverting the pin removes 105 from the
  pinned assembly without touching 104's pins.
- **Disable without un-pinning:** set `intraday_buys_enabled=false` (the default)
  / leave the kill-switch state machine out of `NORMAL`; 105 places no live orders.
- **Compat-shim retirement:** every `# COMPAT-105-SHIM` carries a removal ticket
  and is deleted once the owning repo's pin lands — the umbrella never keeps a
  third copy.
- The umbrella `RenQuant` is the permanent rollback source and is never deleted,
  emptied, or rewritten.

### 2.6 Integration test / CI gate the topology change must pass

Before any 105 pin is treated as production:

- the existing umbrella checks stay green: `make subrepo-doctor` (required files,
  remotes, branch, lock commit, **`RENQUANT_REPOS.md` not drifted from the lock**),
  `make subrepo-test`, `make subrepo-smoke`;
- a **cross-repo integration test proves the 105 contract holds end-to-end**: the
  base-data intraday loader + session-return-surface contract test the pipeline
  imports; the pipeline gate-kernel against the pinned base-data; the model label
  artifact resolvable from `renquant-artifacts`; strategy-105 universe + config
  fingerprint stamped into the run bundle and asserted by pipeline preflight
  (fail-closed on mismatch);
- the milestone gates from the orchestrator design remain authoritative for *alpha*
  promotion (Phase -1 feasibility, M0/M0.5 data+broker contract, M1 frozen-policy
  replay GO bar — placebo-clean OOS IC ≥ 0.03, net Sharpe ≥ 1.0, PSR/DSR ≥ 0.95,
  PBO < 20% — M2 gates+shadow, M3 live monitored). **This ADR ratifies the
  *topology*; it does not lower or replace those quantitative alpha gates.**

## 3. Relationship to orchestrator PR #198

- PR #198 is now **scoped to ORCHESTRATION** (pinning, run-bundle stamping,
  `--strategy` routing, shadow→graduate). It **references this ADR** as the
  authoritative cross-repo contract.
- **This ADR supersedes the cross-repo scope that previously lived in #198.** The
  §6 ownership matrix, the new `renquant-strategy-105` repo, the forbidden-import
  rules, the artifact contracts, and the lock/pin migration are decided **here**,
  in the umbrella, not in the orchestrator PR.
- **This ADR must land FIRST.** Until it is merged: `renquant-strategy-105` is not
  created, no pin order is executed, no intraday subrepo PR lands, and 105 remains
  a design. PR #198 neither opens nor edits the umbrella.

## 4. Consequences

- **Positive:** the cross-repo topology has a single authoritative home; no
  stale-doc drift from duplicating architecture into a subrepo PR; the operator can
  ratify the topology independently of the (larger, evolving) orchestration design;
  104 is fully insulated (additive, default-OFF, flag-routed).
- **Cost / risk:** a new pinned subrepo + paired-PR sequencing is operational
  overhead; it is bounded by the strict pin order and the integration gate, and is
  reversible via §2.5.
- **Out of scope (decided elsewhere):** the *quantitative alpha verdict* for
  intraday trading (UNDETERMINED / marginal per the §A priors in the orchestrator
  design) is settled by the M1 measured policy replay, NOT by this ADR. This ADR
  authorizes the *plumbing* to measure it safely; it does not authorize live
  intraday alpha capital.

## 5. References

- Canonical operating model: [`doc/arch/subrepo-operating-model.md`](../subrepo-operating-model.md)
- SOP: [`doc/arch/multirepo-sop.md`](../multirepo-sop.md)
- 104 strategy architecture: [`doc/arch/strategy-104.md`](../strategy-104.md)
- Orchestrator master design (PR #198): `renquant-orchestrator` →
  `doc/design/2026-06-27-renquant105-intraday-system.md` (esp. §6 / §6.1)
- Repo map source of truth: `RenQuant/subrepos.lock.json` (auto-renders
  `RENQUANT_REPOS.md`)
