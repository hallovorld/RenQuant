# ADR 0001 — renquant105 intraday cross-repo topology (DEFERRED)

**Date:** 2026-06-27
**Status:** **Deferred (Proposed-conditional)** — the cross-repo topology below is
a *conditional* design, NOT an authorized change. It is **PARKED** pending an alpha
GO from the Phase -1 / M1 measured-edge track. Do **not** create
`renquant-strategy-105`, do **not** execute any pin order, and do **not** land any
`# COMPAT-105-SHIM` or intraday subrepo PR until the trigger in §0 is met. This ADR
records the interface contracts so that, *if* an edge materializes, the topology is
already designed and reviewed; it does not by itself ratify creating the repo.
**Owner:** umbrella `RenQuant` (cross-repo architecture lives here ONCE).
**Relationship to orchestrator PR #198:** this ADR is the *independent*
authoritative record of the cross-repo topology/interface contracts. It references
PR #198 only for *non-topological* experiment policy (milestone/phase text); the two
documents do **not** depend on each other being final (see §3).

> This is the first Architecture Decision Record in `RenQuant/doc/arch/`. There
> was no prior ADR convention (existing arch docs are flat descriptive files such
> as `strategy-104.md`, `subrepo-operating-model.md`). This ADR establishes the
> `doc/arch/adr/NNNN-<slug>.md` convention; future cross-repo decisions follow it.

Read with, and do **not** duplicate, the canonical operating model
[`doc/arch/subrepo-operating-model.md`](../subrepo-operating-model.md) and the SOP
[`doc/arch/multirepo-sop.md`](../multirepo-sop.md). This ADR only records the 105
*delta* to that operating model; everything it does not change is inherited.

---

## 0. Status & trigger (why this is Deferred, not Proposed)

**Decision: defer topology creation.** Codex's architecture review (finding #1)
and the Phase -1 measured-edge evidence agree on the same conclusion: **do not
stand up new cross-repo infrastructure before a tradable edge is established.**

- **Phase -1 result (orchestrator PR #199, measured, not assumed):** the intraday
  open→close dispersion does **not** clear the cost barrier at plausible skill.
  Session open→close volatility was **σ_oc ≈ 152 bps** (raw std) / **≈ 114 bps**
  (robust) versus an estimated **220–367 bps round-trip breakeven** (spread +
  impact + fees on the ~40–60 name intraday universe). Modelled net edge was
  **negative at plausible IC: ≈ −6.4 bps @ IC 0.03 and ≈ −3.4 bps @ IC 0.05.** A
  new strategy repo, its permanent CI/ownership/pinning/release/drift overhead, and
  a paired-PR pin migration are **not justified by a negative-net-edge signal.**

- **Trigger to move this ADR from Deferred → Proposed → Accepted:** an explicit
  **alpha GO** from the milestone track — Phase -1 re-run / M1 frozen-policy replay
  must show a **positive, cost-aware net edge** clearing the M1 GO bar
  (placebo-clean OOS IC ≥ 0.03, net Sharpe ≥ 1.0, PSR/DSR ≥ 0.95, PBO < 20%) **and**
  evidence that the 105 lifecycle has diverged enough from 104 to warrant separate
  topology (see §1 alternatives). Absent that GO, 105 experimentation stays
  **outside the production lock** (alternative (c) in §1) and this ADR stays parked.

- The topology in §2 is retained **as a reviewed design** so that, if the trigger
  fires, creation is a fast ratify-and-execute step, not a from-scratch decision.
  Nothing in §2 is active while Status = Deferred.

## 1. Context & alternatives considered

renquant104 (the active daily Panel-LTR cross-sectional strategy, see
[`doc/arch/strategy-104.md`](../strategy-104.md)) is the production topology:
`renquant-base-data` → `renquant-common`/`renquant-pipeline` →
`renquant-model` (factory) → `renquant-artifacts` (registry) →
consumed by `renquant-strategy-104` / `renquant-pipeline` (runtime) /
`renquant-backtesting` / `renquant-orchestrator`, with `renquant-execution`
performing broker actions and `RenQuant` pinning the whole assembly in
`subrepos.lock.json`.

renquant105 is a **proposed intraday (open→close, bounded-turnover) subsystem**
that would run **alongside** 104. Its master design lives in the orchestrator repo
at `doc/design/2026-06-27-renquant105-intraday-system.md` (PR #198) and the
accompanying milestone suite. That design is scoped to **orchestration
responsibilities** (pinning, run-bundle stamping, `--strategy` routing, the
shadow→graduate flow). Per the operating model, **cross-repo architecture must live
ONCE under `RenQuant/doc/arch/` and be referenced, not defined inside a subrepo
PR**; this ADR is that authoritative cross-repo record. The orchestration design
and this ADR are independent (§3): neither blocks on the other being "final."

105 is **not greenfield**: the parked intraday subsystem (Alpaca-IEX fetch,
`*Store` caches, `hourly/minute_features`, the dark websocket consumer, the
reusable fail-closed gate stack) already exists, disabled since 2026-05-04. A 105
activation would be re-activation + an intraday model + tighter gates + a gated buy
path, mostly config/wiring.

**Hard default (operator mandate, inherited verbatim):** live intraday TRADING is
DISABLED. `intraday_buys_enabled=false`; any data/train/shadow work places **zero**
live intraday orders; live intraday would be a deliberate, gated act behind the
validation gate + an armed kill-switch — never the default — and only after an
alpha GO.

### 1.1 Alternatives considered (and the bar a NEW repo must clear)

A new pinned repository is the **highest-overhead** option (permanent CI, CODEOWNERS,
pinning, release cadence, drift surface, paired-PR sequencing). Before committing to
it, the cheaper alternatives below are the **default** and must be ruled out by
explicit criteria.

| # | Alternative | What it is | When it is the right choice |
|---|---|---|---|
| (a) | **Second immutable strategy bundle inside `renquant-strategy-104`** | Add a 105 config bundle alongside 104's, versioned/fingerprinted in the *existing* strategy repo; no new repo. | 105's config lifecycle, release cadence, and ownership stay coupled to 104. **Lowest overhead.** Default if 105 is "another bundle," not a separate product. |
| (b) | **Generic `renquant-strategy` repo keyed by strategy-id** | One strategy repo holding N strategy configs, selected by `strategy_id`; 104 and 105 are entries, not repos. | Multiple strategies are expected and they share tooling/release; consolidates the per-strategy overhead into one repo. Right if a *family* of strategies is anticipated. |
| (c) | **Keep 105 config experimental, OUTSIDE the production lock, until M1 passes** | 105 lives on an experiment branch / worktree / `pending` area; never enters `subrepos.lock.json` `subrepos[]` until an alpha GO. | **Current selection while Status = Deferred.** No production-lock footprint, no CI/ownership/pinning overhead, fully reversible, and Phase -1's negative net edge means there is no measured reason to incur repo overhead yet. |
| (d) | **New `renquant-strategy-105` pinned repo** (the §2 topology) | A dedicated pinned subrepo mirroring 104. | **Only** after (1) an alpha GO clears the M1 bar **and** (2) 105's lifecycle has *sustainably diverged* from 104 (independent release cadence, distinct ownership/CODEOWNERS, config schema that (a)/(b) cannot host without coupling). Until both hold, prefer (a)/(b)/(c). |

**Decision criterion:** choose (d) **only** when an alpha GO has materialized *and*
the sustained-divergence test is met; otherwise prefer (a) or (b), and use (c) for
all pre-GO experimentation. This ADR currently selects **(c)** and parks (d).

## 2. Conditional topology (inactive while Deferred)

*Everything in §2 is a reviewed design that activates ONLY if §0's trigger fires.
While Status = Deferred it is not an authorized change: no repo, no pin order, no
shim.*

Adopt the renquant105 intraday subsystem as a topology change to the canonical
operating model, **conditional on an alpha GO + operator ratification**, with the
following invariants.

### 2.1 New repo: `renquant-strategy-105` (conditional, not created)

A **new pinned subrepo**, mirroring `renquant-strategy-104` — created **only** under
alternative (d) above. It is **not** registered in `subrepos.lock.json` while this
ADR is Deferred; the proposal lives **solely in this ADR** until the repo exists
with a real immutable commit, at which point it graduates into `subrepos[]` with a
real `commit` pin.

**Owns (and ONLY owns):**
- the 105 strategy config skeleton (policy/thresholds, mirrors 104's config bundle);
- the **universe-selection policy/thresholds** — the *rules* for constructing the
  intraday universe (liquidity floor, coverage gate, target name count ~40–60,
  no-survivorship/no-look-ahead constraints). It owns the **policy**, not the
  materialized data (see §2.3);
- the strategy **config-fingerprint** emitted into the run bundle (the same
  `config_fingerprint` contract 104 uses).

**Must NOT own (hard boundary, same as strategy-104 and as CLAUDE.md sets for the
orchestrator):**
- **the materialized point-in-time universe manifest / its as-of lineage** — that is
  a fingerprinted *data* artifact owned by `renquant-base-data` (§2.3), NOT stable
  policy. Do **not** commit daily/per-date universe state to a strategy repo.
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

### 2.3 Universe data ownership — `renquant-base-data` owns the manifest

The **materialized point-in-time universe manifest and its as-of lineage** is a
per-date, continuously changing, fingerprinted **DATA artifact** computed from
market/reference data — it is **not** stable policy/config, so it does **not** live
in a strategy repo. Ownership splits cleanly:

- **`renquant-base-data` OWNS** the materialized universe manifest: for each
  decision date, the frozen liquid/coverage-gated intraday member set
  (~40–60 names) computed only from information available as-of that date
  (no survivorship/look-ahead), plus its **as-of lineage** (source-data
  fingerprints, construction timestamp, immutable manifest id). It is versioned and
  retained with the rest of the data artifacts.
- **`renquant-strategy-105` OWNS only the universe-selection policy/thresholds** —
  the rules (liquidity floor, coverage gate, target count, look-ahead constraints)
  — and **REFERENCES the immutable data manifest by id/fingerprint**. The strategy
  repo never commits daily universe state.
- **Consumers** (`renquant-pipeline` preflight, `renquant-model` training) resolve
  the universe by the immutable manifest id from `renquant-base-data`, and the
  policy fingerprint from strategy-105; preflight fail-closes if either does not
  match the artifact under test.

This keeps operational data out of the stable interface, and keeps reproducibility
and retention with the data layer that already owns lineage.

### 2.4 Artifact contracts (immutable producer/consumer handoff), consistent with 104

The model→pipeline→execution→backtesting handoff for 105 uses the **same
fingerprint-and-pin discipline as 104**, and is governed by **artifact
immutability**: a producer publishes an immutable, fingerprinted artifact; consumers
**reference** it and **never mutate it after publication**. Each capability has
exactly ONE owning repo:

| Capability | Producer (owner) | Consumer | Umbrella role |
|---|---|---|---|
| intraday bar ingestion + `*Store` (incremental, append-only) | `renquant-base-data` | pipeline (canonical loader) | pin base-data; no logic |
| materialized point-in-time universe manifest + as-of lineage | `renquant-base-data` | pipeline preflight / model training | pin base-data |
| intraday features + session-horizon (open→close) forward-return surface | `renquant-base-data` (data) → contract consumed by `renquant-pipeline` | model / decision | pin both |
| intraday label (triple-barrier, open→close) + CPCV / embargo-in-bars, **published as an immutable model bundle** | `renquant-model` (factory) | orchestrator/pipeline reference it | pin model |
| G1–G8 gate stack + decision-ledger wiring | `renquant-pipeline` kernel | — | pin pipeline |
| 105 universe-selection **policy** + config fingerprint (references the base-data manifest) | `renquant-strategy-105` | orchestrator bridge | pin strategy-105 |
| broker-contract checks (intraday-margin/BP fields, rejection + deficit handling, fail-closed on field migration) | `renquant-execution` | pipeline | pin execution |
| pins, run-bundle assembly, `--strategy` routing, shadow→graduate | `renquant-orchestrator` | — | the wiring layer |

**Immutable handoff flow (mirrors 104, corrected for immutability):**

1. **Producer — model factory publishes an immutable manifest.**
   `renquant-model` trains the intraday-label model and **publishes a fingerprinted,
   immutable model bundle to `renquant-artifacts`**. *The published manifest already
   records, at publish time*: the **training-data fingerprint**, the
   **universe-policy fingerprint + the base-data universe manifest id** it trained
   against, the **code commit fingerprints**, the calibrator, and acceptance
   metrics. The bundle is content-addressed by an **immutable `model_bundle_id`**
   (e.g. a digest over the manifest). `promotion_status: shadow` until accepted.
   **No consumer — and specifically NOT the orchestrator — ever writes into or
   re-stamps the published model bundle.**

2. **Consumer — orchestrator references, never mutates.** `renquant-orchestrator`
   assembles a **run bundle** that *references* the immutable `model_bundle_id`, the
   strategy `config_fingerprint`, and the base-data `universe_manifest_id`. The run
   bundle is the orchestrator's own immutable record of which inputs were composed;
   it does **not** modify any upstream artifact.

3. **Validation — pipeline preflight fail-closes on mismatch.**
   `renquant-pipeline` preflight asserts the live feature space + universe match the
   fingerprints recorded **inside the immutable model manifest** (training-data /
   universe-policy / universe-manifest-id), **fail-closed on any mismatch.**

**Producer/consumer schemas + immutable IDs (explicit):**
- `model_bundle` (producer `renquant-model`): `{ model_bundle_id (immutable digest),
  training_data_fingerprint, universe_policy_fingerprint, universe_manifest_id,
  code_commit_fingerprints, calibrator_fingerprint, acceptance_metrics,
  promotion_status }` — write-once.
- `universe_manifest` (producer `renquant-base-data`): `{ universe_manifest_id
  (immutable), as_of_date, members[], source_data_fingerprints, construction_ts }`
  — write-once per date.
- `strategy_config` (producer `renquant-strategy-105`): `{ config_fingerprint,
  universe_policy_fingerprint, references: universe_manifest_id }`.
- `run_bundle` (producer `renquant-orchestrator`, consumer of all of the above):
  `{ run_id, references: { model_bundle_id, config_fingerprint, universe_manifest_id
  }, decision_ledger_ref }` — references only; mutates nothing upstream.

Live promotion requires the full immutable fingerprint set above, same as 104.

### 2.5 Lock / pin migration — phased, contract-before-training

When (and only when) §0's trigger fires, the migration runs in **explicit ordered
phases**. The key invariant from Codex finding #4: **the strategy config contract
must exist BEFORE any model is trained against it** (a model's published manifest
records the universe-policy/config fingerprint it consumed, so that contract must be
merged first). Production *pinning* can be last; the *contract* precedes training.

Ordered phases:

1. **Repo creation (d):** create `renquant-strategy-105`; register it in
   `subrepos.lock.json` `subrepos[]` with a **real immutable commit pin** (NOT a
   `pending` placeholder — the lock only ever records existing pinned repos).
2. **Contract merge:** merge the base-data intraday loader + session-return-surface
   contract **and** the strategy-105 universe-selection policy / config contract,
   each with the contract test the pipeline imports. **These contracts exist before
   any candidate model is trained.**
3. **Candidate training:** `renquant-model` trains the intraday-label model
   **against the now-merged strategy config + base-data universe contract**,
   consuming their fingerprints.
4. **Artifact publication:** the factory **publishes the immutable model bundle** to
   `renquant-artifacts` (`promotion_status: shadow`), recording the training-data /
   universe-policy / universe-manifest-id / code fingerprints (§2.4).
5. **Consumer pinning:** `renquant-pipeline` pins the base-data + model contracts;
   `renquant-strategy-105` + `renquant-orchestrator` pin against the published
   artifact. Each pin advances only after the owning repo's CI is green and the
   umbrella integration check passes; the atomic promote-pin flow
   (`refresh_subrepo_lock.py`, CI-green gate) governs every advance.
6. **Production activation:** flip `--strategy 105` / `intraday_buys_enabled` only
   after the M1 GO bar + shadow acceptance. **Production pinning/activation is the
   LAST step; the config contract (step 2) precedes training (step 3).**

The orchestrator bridge routes by `--strategy` → **no orchestrator code change**;
104 keeps running unchanged.

### 2.6 Rollback

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

### 2.7 Integration test / CI gate the topology change must pass

Before any 105 pin is treated as production:

- the existing umbrella checks stay green: `make subrepo-doctor` (required files,
  remotes, branch, lock commit, **`RENQUANT_REPOS.md` not drifted from the lock**),
  `make subrepo-test`, `make subrepo-smoke`;
- a **cross-repo integration test proves the 105 contract holds end-to-end**: the
  base-data intraday loader + session-return-surface contract + universe manifest;
  the pipeline gate-kernel against the pinned base-data; the model label artifact
  resolvable from `renquant-artifacts` by its immutable `model_bundle_id`;
  strategy-105 policy + config fingerprint and the base-data universe-manifest-id
  referenced in the run bundle and asserted by pipeline preflight (fail-closed on
  mismatch);
- the M1 alpha GO bar (placebo-clean OOS IC ≥ 0.03, net Sharpe ≥ 1.0, PSR/DSR ≥
  0.95, PBO < 20%) is owned by the measured-edge track and is the **trigger** in §0.
  **This ADR ratifies the *topology*; it does not lower or replace those
  quantitative alpha gates, and it stays Deferred until they pass.**

## 3. Relationship to orchestrator PR #198 (independent, no mutual authority)

- This ADR defines the **stable cross-repo topology/interface contracts
  independently.** It does **not** delegate any topology detail to PR #198, and it
  does **not** require #198 to be "final" before it is valid; conversely it does not
  require #198 to treat this ADR as a blocking precondition. The two documents are
  **independent records** with no mutual-authority loop.
- The ADR references PR #198 **only for non-topological experiment policy** — the
  milestone/phase narrative (Phase -1/M0/M1 sequencing, shadow→graduate operational
  flow). Those are operational details that may evolve in the orchestration design
  as a versioned RFC; the **interface contracts in §2 are owned here and do not
  change because #198 changes.**
- PR #198 is scoped to **ORCHESTRATION** (pinning, run-bundle assembly, `--strategy`
  routing, shadow→graduate). Whatever cross-repo wording previously lived there is
  superseded by this ADR's §2, but neither document gates the other's merge.

## 4. Consequences

- **Positive:** deferring avoids standing up permanent CI/ownership/pinning/drift
  overhead for a subsystem whose Phase -1 net edge is negative; the cross-repo
  topology nonetheless has a single reviewed home, ready to ratify fast if an alpha
  GO arrives; 104 is fully insulated (no change while Deferred).
- **Cost / risk if it were activated:** a new pinned subrepo + paired-PR sequencing
  is operational overhead; it is bounded by the strict phased order (§2.5) and the
  integration gate (§2.7), and is reversible via §2.6. This cost is the precise
  reason the repo is deferred until the edge justifies it.
- **Out of scope (decided elsewhere):** the *quantitative alpha verdict* for
  intraday trading is settled by the measured-edge track (Phase -1 result:
  negative net edge → parked; M1 frozen-policy replay would re-decide). This ADR
  authorizes the *plumbing design* to measure it safely; it does **not** authorize
  live intraday alpha capital, and it is **Deferred** precisely because the edge is
  not yet established.

## 5. Notes

- **Umbrella verification targets:** the umbrella `Makefile` exposes
  `make doctor` (and the GitHub CI `check` workflow), not a `make test` target —
  the operating-model text's `make test` reference does not match the umbrella
  Makefile. Umbrella verification for this ADR uses the **actual** umbrella targets
  (`make doctor` + CI `check`). The `make test` / Makefile mismatch is flagged for a
  follow-up doc/Makefile reconciliation; it is out of scope for this ADR.

## 6. References

- Canonical operating model: [`doc/arch/subrepo-operating-model.md`](../subrepo-operating-model.md)
- SOP: [`doc/arch/multirepo-sop.md`](../multirepo-sop.md)
- 104 strategy architecture: [`doc/arch/strategy-104.md`](../strategy-104.md)
- Orchestrator master design (PR #198): `renquant-orchestrator` →
  `doc/design/2026-06-27-renquant105-intraday-system.md` (referenced for
  non-topological experiment policy only)
- Phase -1 measured-edge result (orchestrator PR #199): intraday open→close
  net-edge measurement (negative at plausible IC) — the §0 deferral evidence
- Repo map source of truth: `RenQuant/subrepos.lock.json` (auto-renders
  `RENQUANT_REPOS.md`)
