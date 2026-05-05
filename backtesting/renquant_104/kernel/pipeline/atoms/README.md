# Reusable Task Atom Library

User mandate (2026-05-04): "Task 可以很小，很具体，多复用. Job 可以有
很多 task, 各种排列组合, 各种并发, 各种逻辑关系, 很具体. Pipeline 要
简练, 可以抽象."

This module is the **Task** layer — small, parameterized, single-purpose
atoms designed to be composed by Job classes across the codebase.

## Design rules

1. **Each atom is parameterized** — config in `__init__`, work in `run(ctx)`.
   Atoms read & write *named* ctx fields supplied by the caller. No
   hard-coded ctx field names inside the atom.
2. **Atom body ≤ 30 lines** (much tighter than the 50-line Job-Task target,
   because atoms must be trivially reviewable).
3. **No domain logic** — atoms operate on generic structures (vectors,
   matrices, JSON-like dicts). Domain-specific orchestration belongs in
   the Job that composes atoms.
4. **One verb, one noun**: `LoadParquet`, `WriteJSON`, `IsFiniteGuard`,
   `BuildVector`. If you need two verbs, split.
5. **Composable side-effects**: each atom mutates one field (or one
   counter, or one log line) per run. Compositions live in the Job.

## Module layout

```
atoms/
├── ctx_ops.py        — copy/move/clear/validate ctx fields
├── numerical.py      — finite/range/non-empty/clamp guards
├── vectors.py        — build per-asset vectors from collections
├── persistence.py    — write/load JSON, parquet, SQLite
├── logging_atoms.py  — log_summary, increment_counter
├── gates.py          — skip-if-disabled, skip-if-condition
```

## Usage example (the QP Job, reimagined with atoms)

```python
class JointPortfolioQPJob(Job):
    @property
    def tasks(self):
        return [
            SkipIfConfigDisabledTask(
                "rotation.joint_actions.enabled",
                also_skip_if=("rotation.joint_actions.solver", "!=", "qp"),
            ),
            BuildVectorFromHoldingsTask(
                attr="shares", target="_qp_shares",
                missing_default=0.0,
            ),
            ComputeWeightVectorTask(
                shares="_qp_shares", prices="prices",
                portfolio_value="portfolio_value",
                target="_qp_w_current",
            ),
            BuildVectorFromCandidatesAndHoldingsTask(
                attr="mu", target="_qp_mu", missing_default=0.0,
                fallback_attr="panel_score",
            ),
            BuildVectorFromCandidatesAndHoldingsTask(
                attr="sigma", target="_qp_sigma", missing_default=0.05,
            ),
            ComputeBrownSmithTaxCostTask(
                target="_qp_tax_cost",
            ),
            ComputeWashSaleMaskTask(
                target="_qp_wash_mask",
            ),
            ComputePositionCapsTask(
                target="_qp_w_upper",
            ),
            SolveMarkowitzQPTask(
                w_current="_qp_w_current", mu="_qp_mu", sigma="_qp_sigma",
                # ... refs to other ctx fields
                target="_qp_solution",
            ),
            EmitOrdersFromDeltaWeightsTask(
                solution="_qp_solution",
                source_label="qp",
            ),
            RecordCounterTask("qp_buys",  source="_qp_solution.n_buys"),
            RecordCounterTask("qp_sells", source="_qp_solution.n_sells"),
            LogSummaryTask(
                "JointPortfolioQPJob: n=%d buys=%d sells=%d obj=%.6f iter=%d",
                fields=("_qp_n", "qp_buys", "qp_sells",
                        "_qp_solution.objective", "_qp_solution.n_iter"),
            ),
        ]
```

The Job is now PURELY orchestration. All work is in atoms. Atoms are
reused by Rotation Job, Sell Job, TopUp Job — anything that needs to
build a per-asset vector, validate finiteness, or persist an artifact.

## Test pattern

Each atom file gets a paired test in `tests/acceptance/atoms/test_<file>.py`
that exercises the atom against a stub ctx. Atom tests don't need real
data — just SimpleNamespace ctx fixtures.
