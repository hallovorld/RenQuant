# §8 Step 4g — A/B replay evidence artifact schema

**Date**: 2026-06-03
**Status**: Pre-implementation schema spec for the decision-grade JSON the
§8 Step 4g run will commit under `doc/research/evidence/`.
**Author**: Claude
**References**:
- Parent memo: [`2026-06-02-qp-architecture-review-and-alternatives.md`](2026-06-02-qp-architecture-review-and-alternatives.md) §8 Step 4
- Replay harness: PR #131 (`backtesting/renquant_104/kernel/portfolio_qp/allocator_replay.py`)
- DSR / PBO wiring: PR #132 (`backtesting/renquant_104/kernel/portfolio_qp/replay_significance.py`)

## Purpose

§8 Step 4g produces the **decision-grade artifact** that the user reads
to decide whether to authorize a Hybrid migration, a hard-only QP
simplification, or stay on the current QP. Pinning the JSON shape
in advance lets follow-up PRs (#4d Hybrid, #4e WF loader, #4f
hard-only) wire their outputs into the right keys without re-arguing
the schema each round.

This schema is also the contract Step 5 live shadow runs reproduce —
operational telemetry should fit the same envelope so the offline /
live comparison reads from a single shape.

## Top-level shape

```json
{
  "as_of_date": "<ISO date the replay ran>",
  "cut_range": ["<earliest_cut>", "<latest_cut>"],
  "wf_artifact_root": "<repo-relative path to the WF artifacts the replay consumed>",
  "n_bars": <total bars replayed across cuts>,
  "n_unique_dates": <unique calendar dates>,
  "regime_distribution": {
    "BULL_CALM": <pct>, "BULL_VOLATILE": <pct>,
    "CHOPPY": <pct>, "BEAR": <pct>
  },
  "constraint_snapshot_contract_version": "v1-2026-06-03",
  "allocators": ["current_qp", "hard_only_qp", "hybrid_option_f",
                 "inverse_vol_top_k", "equal_weight_top_k"],
  "per_allocator": { ... },
  "paired_comparisons": { ... },
  "significance": { ... },
  "regime_stratified": { ... },
  "violation_report": { ... },
  "verdict": { ... }
}
```

## `per_allocator`

One block per baseline. Output of `ReplayResult.to_dict()` from the
replay harness (PR #131), with three additional sub-blocks added by
Step 4g:

```json
"per_allocator": {
  "current_qp": {
    "name": "current_qp",
    "bars": 540,
    "sharpe_annual": <float | null>,
    "mean_daily_return": <float>,
    "cumulative_return": <float>,
    "max_drawdown": <float>,
    "mean_turnover": <float>,
    "cap_violations": <int>,
    "violations_per_family": {
      "w_upper_hard": <int>, "w_lower": <int>, "wash_sale": <int>,
      "dw_max": <int>, "cash_budget": <int>, "turnover_max": <int>,
      "sector_cap": <int>, "corr_group_cap": <int>, "gross_max": <int>
    },
    "total_violations": <int>,
    "fallback_to_no_candidates": <int>,
    "per_regime_sharpe": {"BULL_CALM": <float | null>, ...},
    "per_regime_n_bars": {"BULL_CALM": <int>, ...},
    "fallback_rate_when_hybrid": <float | null>,  // populated only for hybrid_option_f; null otherwise
    "qp_solve_rate": <float | null>,              // populated only for hybrid_option_f
    "cost_breakdown_bps": {
      "transaction": <float>,
      "rounding_loss": <float>,                    // future: when share-rounding modeled
      "fee_buffer": <float>                        // future: when broker fee modeled
    }
  },
  "hard_only_qp": { ... },
  "hybrid_option_f": { ... },
  "inverse_vol_top_k": { ... },
  "equal_weight_top_k": { ... }
}
```

## `paired_comparisons`

Allocator-vs-allocator differences on the SAME bar sequence — this is
the §8 "paired daily returns" requirement. Keyed by ordered
`(allocator_a, allocator_b)` tuples; each value is the delta series'
summary:

```json
"paired_comparisons": {
  "current_qp_vs_hard_only_qp": {
    "n_bars": 540,
    "mean_delta_daily_return": <float>,
    "delta_sharpe_annual": <float | null>,
    "win_rate_a_beats_b": <float>,        // pct of bars where a > b
    "max_delta_daily_return": <float>,
    "min_delta_daily_return": <float>,
    "hac_t_stat": <float | null>,         // Newey-West if computable
    "hac_p_value": <float | null>
  },
  ...
}
```

Pairings produced: every (a, b) where `a` is the incumbent
(`current_qp`) and `b` ∈ {hard_only_qp, hybrid_option_f,
inverse_vol_top_k, equal_weight_top_k}.

## `significance`

Output of `verdicts_to_dict(compute_significance_verdicts(results))`
from PR #132. One DSR per allocator; one shared PBO across the
candidate matrix.

```json
"significance": {
  "current_qp": {
    "sharpe_raw_annual": <float>,
    "dsr": <float in [0,1]>,
    "pbo": <shared float in [0,1] | null>,
    "n_returns": <int>,
    "n_trials": <int>,
    "live_promotable_per_clause_7_4": <bool>
  },
  "hard_only_qp": { ... },
  ...
}
```

**§7.4 vs §8 Step 4 gate distinction (codex #134 review correction)**:
- CLAUDE.md §7.4 Tier 3 actually says: `Tier 2 + (DSR > 0.5 OR PBO < 0.5
  OR (n ≥ 30 AND t > 3.0))` — three **disjuncts**, not "both".
- The §8 Step 4 A/B gate is **deliberately stricter** than §7.4
  because the comparison picks among 5 candidates. Multi-candidate
  selection bias is higher than the single-change setup §7.4 is
  written for, so the gate requires `DSR ≥ 0.95 AND
  (PBO is None OR PBO < 0.5)`. DSR ≥ 0.95 is the Bailey-López de Prado
  2014 selection-bias 5% threshold.
- The `live_promotable_per_clause_7_4` field name is therefore
  somewhat misleading — it implements the **stricter §8 Step 4
  adaptation of §7.4**, not §7.4's literal disjunction. The
  implementation field name was chosen for stability; a future
  rename to `live_promotable_per_section_8` is acceptable.

**PBO small-N caveat (codex #134 review)**: with N=5 candidates and
`pbo_n_slices=16`, the CSCV combinatorial count is C(16,8) = 12,870
splits but trial-rank variance is dominated by the 5-allocator
selection. Bailey-Borwein-LdP-Zhu 2015 §3 derives PBO with implicit
larger N (paper examples N ∈ {25, 50, 100}). At small N the PBO
point estimate has high SE. The reporter SHOULD emit a `pbo_se`
bootstrap-SE alongside the point estimate; the verdict reads
`pbo < 0.5 AND (pbo_se is None OR pbo + pbo_se < 0.55)` to avoid
gating on noise. When `pbo_se` is unavailable the verdict falls back
to the point estimate.

## `regime_stratified`

Per-regime breakdown so the §1 PRIME DIRECTIVE (regime-conditional
reporting) is honored: every Sharpe number shows by regime FIRST,
pooled second.

```json
"regime_stratified": {
  "BULL_CALM": {
    "n_bars": <int>,
    "per_allocator": {
      "current_qp": {
        "sharpe_annual": <float | null>,
        "mean_daily_return": <float>,
        "max_drawdown": <float>,
        "mean_turnover": <float>,
        "violations_per_family_total": <int>
      },
      "hard_only_qp": { ... },
      ...
    },
    "best_allocator_by_sharpe": "<name>"
  },
  "BULL_VOLATILE": { ... },
  "CHOPPY": { ... },
  "BEAR": { ... }
}
```

A regime with `n_bars < 30` gets a `"undersampled": true` flag and
the sharpe value is reported but the verdict treats it as not
statistically meaningful (consistent with #128's eligibility model).

## `violation_report`

Step 4 gate: **zero hard-constraint regressions vs `ConstraintSnapshot`
is non-negotiable**. This block surfaces any allocator that breached
any family.

```json
"violation_report": {
  "any_allocator_violated_any_family": <bool>,
  "by_allocator": {
    "current_qp": {
      "total_violations": 0,
      "violations_per_family": { ... all zeros ... }
    },
    "hybrid_option_f": {
      "total_violations": 3,
      "violations_per_family": {"dw_max": 1, "turnover_max": 2, ...},
      "rejected_for_promotion": true
    },
    ...
  }
}
```

Any allocator with `total_violations > 0` is `rejected_for_promotion:
true` regardless of its Sharpe — that's the §8 contract.

## `verdict`

The decision block. The promotion candidate is the allocator that
(a) beats `current_qp` on paired daily returns with `delta_sharpe > 0`
AND (b) passes `live_promotable_per_clause_7_4: true` AND
(c) has zero violations.

```json
"verdict": {
  "promotion_candidate": "<name> | null",
  "rationale": "<short text — points at the key paired_comparisons + significance entries>",
  "fallback_recommendation": "<name>",
  "next_action": "<promote_to_shadow | keep_incumbent | iterate>",
  "non_negotiable_gate_passed": {
    "zero_hard_constraint_regressions": <bool>,
    "pbo_below_0_5": <bool>,
    "dsr_above_0_95": <bool>,
    "win_rate_z_score_above_2": <bool>
  }
}
```

`next_action` is one of (codex #134 review: enum describes the
operational follow-up, not the candidate status):
- `"promote_to_shadow"` — promote the candidate to Step 5 live shadow
  for operational telemetry + implementation parity verification
- `"keep_incumbent"` — current_qp wins on all the gates; no shadow
  run scheduled (the `verdict.promotion_candidate: null` already
  conveys "no winner")
- `"iterate"` — the candidate has potential but failed one gate
  (typically PBO or DSR); refine and re-run

## File-of-record convention

```
doc/research/evidence/
  2026-06-03-qp-ab-replay-verdict.json    ← this canonical artifact
  2026-06-03-qp-ab-replay-per-bar.parquet ← raw daily-returns table (optional)
  2026-06-03-qp-ab-replay-bars-sample.json ← 5-10 sample bars w/ snapshots
                                              (so the verdict is reproducible
                                              without the full WF artifact)
```

The verdict JSON is the artifact future agents + users read. The
per-bar parquet is optional — useful for re-running the significance
pass with a different `pbo_n_slices` or `pbo_max_combinations`
without re-doing the WF replay.

## Reproduction recipe

```bash
.venv/bin/python -m kernel.portfolio_qp.run_ab_replay \
    --wf-artifact-root backtesting/renquant_104/artifacts/walkforward_v2 \
    --start-cut 2024-01-01 \
    --end-cut 2026-03-27 \
    --out doc/research/evidence/2026-06-03-qp-ab-replay-verdict.json \
    --allocators current_qp,hard_only_qp,hybrid_option_f,inverse_vol_top_k,equal_weight_top_k \
    --pbo-n-slices 16
```

(Module `kernel.portfolio_qp.run_ab_replay` is shipped in §8 Step 4g —
once the Hybrid + hard-only + WF-loader PRs land.)

## What's NOT in scope here

This schema is for the **offline A/B verdict**. Step 5 live-shadow
operational telemetry is a separate envelope — same allocator results
shape, but different metadata (live broker, paper / alpaca, real
fill prices, real broker rounding). The two will be siblings under
`doc/research/evidence/` and the verdict block will be missing on
the live shadow until enough days have elapsed.

## Reviewer pre-emption

| Likely concern | Pre-emptive resolution |
|---|---|
| "Why is PBO shared across allocators, not per-allocator?" | CSCV PBO is defined over the candidate set, not a single strategy — Bailey-Borwein-López de Prado-Zhu 2015 §3. One PBO per A/B run, applied to all candidates. |
| "Why deny promotion on a single family violation?" | §8 gate is "zero regressions vs `ConstraintSnapshot`". Any infeasible allocator's Sharpe is unreliable. |
| "Why no Sortino / Calmar?" | They can be added later; the verdict gate uses Sharpe + DSR + PBO + paired delta + violation count as the minimal sufficient set per §7.3 / §7.4. |
| "Why `win_rate > 0.55`?" | (Codex #134 review correction) Fixed 0.55 threshold is unsafe at small N. Bernoulli SE ≈ √(0.25/n) — at n=540 bars SE≈0.022 so 0.55 is 2.4σ (significant), but at n=100 (per-regime slice) SE≈0.05 so 0.55 is 1.0σ (coin flip). The verdict gate is therefore `win_rate_z_score > 2.0` (= `(win_rate − 0.5) / sqrt(0.25 / n_bars) > 2`), parameterised by `n_bars` not by a fixed pct. The `win_rate_above_0_55` field is retained as a diagnostic but does NOT gate.  |
