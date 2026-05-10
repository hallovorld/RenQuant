# Typed Past Migration — multi-week refactor plan

**Status:** foundation shipped 2026-05-10 (M4 track). One Task migrated as proof-of-concept (`DataFreshnessGateTask` → `TypedDataFreshnessGate`). Remaining ≈90 Tasks pending.

**Goal:** make peek-ahead architecturally impossible. Every Task receives a frozen `Past` snapshot pre-sliced to cursor `t` instead of reading freely from `ctx`.

**Pattern reference:** cvxportfolio's `Estimator.values_in_time(t, past)` — see `https://www.cvxportfolio.com/api_documentation/estimator.html`.

## Architecture (foundation)

```
kernel/typed_past/
  past.py                      Past dataclass (frozen) + slice_until factory
  estimator.py                 TypedTask Protocol + TaskResult + adapter
  typed_data_freshness.py      First migrated Task (proof-of-concept)
  MIGRATION.md                 this file
```

Bridge: `TypedTaskAdapter` lets a `TypedTask` drop into a legacy Job's task chain unchanged. Adapter slices a Past from ctx, calls `values_in_time`, propagates explicit ctx_writes. Allows incremental migration without big-bang.

## Migration order — read-only first, emit-order last

The key principle: migrate Tasks that only **read** before Tasks that **emit orders** or **mutate downstream-visible ctx state**. Read-only Tasks are low-risk; once their TypedTask version passes alignment tests, their adapter wires in trivially. Emit-order Tasks own the trade tape, so their migration must be staged carefully (drift here = production divergence).

### Tier 1 — Pure read-only / gate Tasks (start here, ~25 Tasks)

These short-circuit the chain via `False` but never write derived state. Adapter wraps trivially.

| Task | File | Risk | Notes |
|---|---|---|---|
| ✅ DataFreshnessGateTask | task_data_freshness.py | LOW | done — proof-of-concept |
| EarningsFilterTask | task_candidates.py:12 | LOW | reads ctx.earnings_calendar |
| WashSaleFilterTask | task_candidates.py:22 | MED | reads last_sell_dates + last_sell_pls (cost-aware) |
| PostStopCooldownFilterTask | task_post_stop_cooldown.py | LOW | reads last_stop_exit_dates |
| ConfidenceVetoTask | task_gates.py:44 | LOW | reads regime/confidence |
| BullVolOffensiveBlockTask | task_gates.py:85 | LOW | reads ohlcv |
| BEARBranchTask | task_gates.py:115 | LOW | regime branch |
| VelocityCrashTask | task_gates.py:133 | LOW | reads SPY returns |
| EMA50GateTask | task_gates.py:150 | LOW | reads ohlcv |
| DrawdownGateTask | task_gates.py:13 | LOW | reads hwm/drawdown |
| TransitionWindowTask | task_gates.py:23 | LOW | regime transition |
| RealizedVolGateTask | task_risk_gates.py:48 | LOW | reads ohlcv |
| PositionConcentrationGateTask | task_risk_gates.py:119 | LOW | reads holdings |
| ScoreThresholdTask | task_candidates.py:151 | LOW | reads candidates |
| RelativeStrengthTask | task_candidates.py:182 | LOW | reads ohlcv |
| EarningsBlackoutSellTask | task_sell.py:370 | LOW | reads earnings_calendar |
| RecordScoreDistributionTask | task_score_distribution.py | LOW | telemetry-only |
| MonitorIdleStreakTask | task_monitor.py | LOW | reads counters |
| atoms/gates.py (3 Tasks) | atoms/gates.py | TRIVIAL | helpers; convert last |
| atoms/numerical.py (4 Tasks) | atoms/numerical.py | TRIVIAL | guards; convert last |
| atoms/logging_atoms.py (2 Tasks) | atoms/logging_atoms.py | TRIVIAL | telemetry; convert last |
| atoms/ctx_ops.py (3 Tasks) | atoms/ctx_ops.py | DEFER | inherently ctx-coupled, may stay as legacy adapters |

### Tier 2 — Compute / scoring Tasks (~25 Tasks)

These derive new fields (features, scores, blended scores). Past needs to expose model artifacts — extend Past or pass through `TypedTask.__init__`.

| Task | File | Risk | Notes |
|---|---|---|---|
| HurstTask | task_regime.py:16 | LOW | pure ohlcv stat |
| CUSUMTask | task_regime.py:44 | LOW | spy_returns based |
| GMMTask | task_regime.py:76 | MED | depends on loaded gmm artifact |
| BEAROverrideTask | task_regime.py:120 | LOW | regime gate |
| RegimeFinalizeTask | task_regime.py:163 | LOW | combines upstream |
| HWMUpdateTask | task_drawdown.py:12 | MED | mutates hwm — not pure read |
| DrawdownCircuitTask | task_drawdown.py:36 | MED | sets skip_buys |
| BuildFeaturesTask | task_candidates.py:72 | HIGH | feature compute; SimAdapter caches; needs Past extension |
| ScoreBuyTask | task_candidates.py:107 | HIGH | model scoring — feature alignment matters |
| AssembleCandidateTask | task_candidates.py:205 | LOW | output struct |
| BlendScoresTask | task_ranking.py:12 | LOW | combines score columns |
| SortCandidatesTask | task_ranking.py:46 | LOW | ranking |
| ScoreModelTask | task_sell.py:78 | HIGH | sell-side scoring |
| EvaluateExitsTask | task_sell.py:119 | HIGH | exit logic |
| SellGateBTask | task_sell.py:142 | MED | gate |
| PanelConvictionExitTask | task_sell.py:251 | MED | conviction-based exit |
| PrepareHoldingTask | task_sell.py:44 | LOW | struct prep |
| panel_pipeline (~12 Tasks) | panel_pipeline/* | HIGH | model-loading + matrix-build; needs Past extension for artifacts |

### Tier 3 — Emit-order / mutating Tasks (~12 Tasks, last)

These produce orders or significantly mutate ctx. Migrate ONLY after Tier 1+2 are stable for ≥2 weeks of green CI.

| Task | File | Risk | Notes |
|---|---|---|---|
| BuildPairsTask | task_rotation.py:60 | HIGH | rotation pairs |
| ValidatePairsTask | task_rotation.py:633 | HIGH | rotation gate |
| EmitRotationsTask | task_rotation.py:694 | HIGH | writes rotations |
| LimitSellsPerBarTask | task_limit_sells.py | HIGH | caps exits |
| PrepareSelectionTask | task_selection.py:12 | HIGH | candidate prep |
| RunSelectionTask | task_selection.py:75 | HIGH | greedy selection |
| SizeAndEmitTask | task_selection.py:104 | HIGH | order emission |
| TopUpHeldTask | task_topup.py | HIGH | top-up orders |
| TrimHeldTask | task_trim.py | HIGH | trim orders |
| JointActionTask | task_joint_actions.py | DEFERRED | 700-line monolith — see CLAUDE.md §1c table |
| portfolio_qp/* (10 Tasks) | portfolio_qp/* | HIGH | live QP path; cvxportfolio-style — natural fit but high blast radius |
| panel_pipeline (write-side) | panel_pipeline/* | HIGH | calibration + NGBoost outputs |

## Migration recipe (per Task)

1. **Inventory data deps.** Grep the legacy Task body for `ctx.X`. Categorize each access into: (a) past-only data → goes into Past, (b) artifact / config → goes into TypedTask `__init__`, (c) mutation target → comes back via `TaskResult.ctx_writes`.
2. **Write the TypedTask** in `kernel/typed_past/typed_<name>.py`. Method: `values_in_time(self, t, past) -> TaskResult`.
3. **Test §5.13.1**: a paired test that runs both the legacy Task and the typed version on identical inputs through the adapter, and asserts the same continue/raise/ctx_writes outcome.
4. **Wire in.** Replace the legacy Task in the Job's `tasks` list with `TypedTaskAdapter(MyTypedTask())`. Run the relevant alignment test suite (`tests/test_panel_alignment.py`, `tests/test_policy_alignment.py`).
5. **Delete legacy** ONLY after the typed form has run in production (live + sim) for ≥1 week with no regressions.

## Per §5.13.10 — no `if X is not None` defensive code

`Past` fields use `Optional[X]` for genuinely optional values, but the prod path must always populate them (use empty DataFrames / empty MappingProxyType, not None). Any TypedTask that hits an `if ... is not None` short-circuit on a Past field is a code smell — that branch is dead code (NGBoost-σ pattern). Either the field is always present (drop the guard) or it's a config flag (gate the whole TypedTask, not a branch inside it).

## Constraints / risks

- **Big-bang refactor forbidden.** Adapter approach is mandatory. Migrate one Task per PR, with paired alignment tests.
- **Past extension** for model artifacts (`models`, `gmm`, `corr_matrix`) is the next foundational change, needed before Tier 2 can begin. Suggested approach: add a separate frozen `Artifacts` dataclass and pass `(t, past, artifacts)` to TypedTask. Or fold into Past as `past.artifacts: types.MappingProxyType`.
- **Performance.** `Past.slice_until` runs once per bar per Task; for an inference loop with 100+ Tasks this is 100+ `df.loc[:t]` calls per bar. Memoize per (bar, source-id) if profiling shows hotspot. Initial estimate: <50 ms/bar overhead, acceptable.
- **Per §5.13.10 audit at every Tier 2 / 3 step.** Before declaring a Task migrated, grep its production caller in `live/`, `adapters/`, and `pp_inference.py` to confirm the adapter is wired. A TypedTask that no production code calls is dead per §5.13.2.

## Suggested cadence

- Week 1 (this PR): foundation + 1 Tier 1 Task (DataFreshnessGate). Done.
- Week 2-3: rest of Tier 1 (read-only gates, 24 Tasks). One PR per 4-5 Tasks.
- Week 4: Past extension for artifacts (`Artifacts` dataclass).
- Week 5-7: Tier 2 (compute + scoring, ~25 Tasks). Per-PR alignment tests mandatory.
- Week 8+: Tier 3 (emit-order). Each Task gets a multi-day sim-A/B before promotion.
