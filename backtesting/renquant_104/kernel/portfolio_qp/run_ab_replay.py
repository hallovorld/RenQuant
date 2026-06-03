"""§8 Step 4g — CLI driver that runs the A/B replay and emits the
decision-grade evidence JSON per the schema at
``doc/research/2026-06-03-qp-ab-replay-evidence-schema.md``.

This module wires together the pieces shipped in:
- PR #126: ConstraintSnapshot contract
- PR #127: solve_portfolio_qp_from_snapshot wrapper
- PR #130: baseline allocators (equal-weight / inverse-vol / fractional
  Kelly)
- PR #131: AllocatorReplay harness
- PR #132: DSR / PBO significance pass
- PR #134: evidence schema spec

It does NOT implement the WF cut loader or the Hybrid Option F /
hard-only QP allocators — those land in Step 4d/4e/4f and the driver
loads them via the standard module path. The driver is a thin
orchestrator so its tests can pin the orchestration layer
independently of the in-flight pieces (allocators registered by
name; loader injected via a callable; the verdict assembly logic is
the part this PR pins).
"""
from __future__ import annotations

import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from kernel.portfolio_qp.allocator_replay import (
    AllocatorReplayBar,
    paired_daily_returns,
    replay_all,
)
from kernel.portfolio_qp.baseline_allocators import (
    AllocatorResult,
    equal_weight_top_k,
    fractional_kelly_top_k,
    inverse_vol_top_k,
)
from kernel.portfolio_qp.replay_significance import (
    compute_significance_verdicts,
    verdicts_to_dict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("qp-ab-replay")


# --------- allocator registry ---------------------------------------------

#: Default allocator registry. The driver looks up names here when
#: assembling the run set. Step 4d/4e/4f register their entries via
#: ``register_allocator()``.
_ALLOCATOR_REGISTRY: dict[str, Callable] = {
    "equal_weight_top_k": equal_weight_top_k,
    "inverse_vol_top_k": inverse_vol_top_k,
    "fractional_kelly_top_k": fractional_kelly_top_k,
}


def register_allocator(name: str, fn: Callable) -> None:
    """Register an allocator callable under a name used by the CLI."""
    if name in _ALLOCATOR_REGISTRY:
        log.warning("registry: overwriting allocator %r", name)
    _ALLOCATOR_REGISTRY[name] = fn


def get_allocator(name: str) -> Callable:
    if name not in _ALLOCATOR_REGISTRY:
        raise KeyError(
            f"allocator {name!r} not in registry. Registered: "
            f"{sorted(_ALLOCATOR_REGISTRY)}"
        )
    return _ALLOCATOR_REGISTRY[name]


# --------- paired-comparison metrics --------------------------------------

def paired_comparison_metrics(
    a: np.ndarray, b: np.ndarray, *, name_a: str, name_b: str,
) -> dict:
    """Pairwise (a − b) daily-return summary. HAC t-stat optional —
    requires `kernel.metrics.hac_se` which is already on main.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    delta = a - b
    out = {
        "n_bars": int(a.size),
        "mean_delta_daily_return": float(np.mean(delta)),
        "delta_sharpe_annual": _sharpe_annual(delta),
        "win_rate_a_beats_b": float(np.mean(a > b)),
        "max_delta_daily_return": float(np.max(delta)),
        "min_delta_daily_return": float(np.min(delta)),
        "hac_t_stat": None,
        "hac_p_value": None,
    }
    try:
        from kernel.metrics.hac_se import hac_t_stat
        raw = hac_t_stat(delta.tolist())
        # hac_t_stat may return either a scalar or a (t_stat, info) dict
        # depending on the version; normalise to float.
        if isinstance(raw, dict):
            out["hac_t_stat"] = float(raw.get("t_stat", raw.get("t", 0.0)) or 0.0)
        else:
            out["hac_t_stat"] = float(raw)
    except (ImportError, AttributeError, TypeError, ValueError):
        pass  # leave None — already in schema
    return out


def _sharpe_annual(returns: np.ndarray) -> Optional[float]:
    if returns.size < 2:
        return None
    sd = float(np.std(returns, ddof=1))
    if sd < 1e-12:
        return None
    return float(np.mean(returns) / sd * np.sqrt(252.0))


# --------- regime stratification ------------------------------------------

def regime_stratified_block(
    results: dict,
    bars: Sequence[AllocatorReplayBar],
) -> dict:
    """Per-regime per-allocator Sharpe + MDD + turnover + violations.

    Honors CLAUDE.md §1 PRIME DIRECTIVE: by-regime FIRST, pooled
    second. Regimes with fewer than 30 in-regime bars get
    ``"undersampled": true``.
    """
    # Bucket bars by regime
    by_regime: dict[str, list[int]] = {}
    for i, b in enumerate(bars):
        if b.regime is None:
            continue
        by_regime.setdefault(b.regime, []).append(i)

    out: dict = {}
    for regime, idx_list in by_regime.items():
        idx = np.array(idx_list)
        n_bars = len(idx)
        block = {
            "n_bars": int(n_bars),
            "undersampled": n_bars < 30,
            "per_allocator": {},
        }
        best_sharpe = None
        best_name = None
        for name, res in results.items():
            arr = np.asarray(res.daily_returns_net, dtype=float)[idx]
            sharpe = _sharpe_annual(arr)
            block["per_allocator"][name] = {
                "sharpe_annual": sharpe,
                "mean_daily_return": float(np.mean(arr)) if arr.size else 0.0,
                "max_drawdown": _max_drawdown(arr),
            }
            if sharpe is not None and (best_sharpe is None or sharpe > best_sharpe):
                best_sharpe = sharpe
                best_name = name
        block["best_allocator_by_sharpe"] = best_name
        out[regime] = block
    return out


def _max_drawdown(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    return float(dd.min())


# --------- violation report -----------------------------------------------

def violation_report_block(results: dict) -> dict:
    """Step 4 gate: zero hard-constraint regressions vs ConstraintSnapshot.

    Any allocator with total_violations > 0 is rejected for promotion
    regardless of Sharpe.
    """
    by_allocator = {}
    any_violated = False
    for name, res in results.items():
        total = res.total_violations()
        if total > 0:
            any_violated = True
        by_allocator[name] = {
            "total_violations": int(total),
            "violations_per_family": dict(res.violations_per_family),
            "rejected_for_promotion": total > 0,
        }
    return {
        "any_allocator_violated_any_family": bool(any_violated),
        "by_allocator": by_allocator,
    }


# --------- verdict assembly -----------------------------------------------

def assemble_verdict(
    significance: dict,
    paired: dict,
    violations: dict,
    *,
    incumbent: str,
    win_rate_threshold: float = 0.55,
) -> dict:
    """Apply the Step 4 non-negotiable gate + select promotion candidate.

    A promotion candidate must:
    1. Beat the incumbent on paired daily returns (delta_sharpe > 0
       AND win_rate > ``win_rate_threshold``).
    2. Pass DSR ≥ 0.95 AND (PBO is None OR PBO < 0.5).
    3. Have zero hard-constraint regressions.
    """
    candidates = [
        name for name, sig in significance.items()
        if name != incumbent
    ]
    promotion_candidate = None
    rationale = "no allocator beat the incumbent on all three gates"

    for name in candidates:
        pc_key = f"{incumbent}_vs_{name}"
        if pc_key not in paired:
            continue
        pc = paired[pc_key]
        sig = significance[name]
        violated = violations["by_allocator"].get(name, {}).get(
            "rejected_for_promotion", True,
        )
        # delta_sharpe is incumbent - candidate; candidate beats = negative
        delta_sharpe = pc.get("delta_sharpe_annual")
        beats_incumbent_paired = (
            delta_sharpe is not None and delta_sharpe < 0.0
            and pc.get("win_rate_a_beats_b", 1.0) < (1.0 - win_rate_threshold)
        )
        passes_significance = sig.get("live_promotable_per_clause_7_4", False)
        passes_violation_gate = not violated
        if beats_incumbent_paired and passes_significance and passes_violation_gate:
            promotion_candidate = name
            rationale = (
                f"{name} beats {incumbent} on paired daily returns "
                f"(delta_sharpe={delta_sharpe:+.3f}, win_rate={1.0 - pc['win_rate_a_beats_b']:.2f}), "
                f"passes DSR ≥ 0.95 / PBO < 0.5, zero hard-constraint regressions."
            )
            break

    return {
        "promotion_candidate": promotion_candidate,
        "rationale": rationale,
        "fallback_recommendation": incumbent,
        "next_action": (
            "live_shadow" if promotion_candidate
            else ("iterate" if any(
                # someone won the paired gate but failed sig/violation
                paired.get(f"{incumbent}_vs_{n}", {}).get("delta_sharpe_annual") is not None
                and paired[f"{incumbent}_vs_{n}"]["delta_sharpe_annual"] < 0
                for n in candidates if f"{incumbent}_vs_{n}" in paired
            ) else "reject_all")
        ),
        "non_negotiable_gate_passed": {
            "zero_hard_constraint_regressions": not violations[
                "any_allocator_violated_any_family"
            ],
        },
    }


# --------- top-level runner -----------------------------------------------

def run_replay(
    bars: Sequence[AllocatorReplayBar],
    allocator_names: Sequence[str],
    *,
    incumbent: str = "current_qp",
    pbo_n_slices: int = 16,
) -> dict:
    """Run the A/B replay end-to-end and return the verdict JSON dict.

    Allocators are looked up by name in the registry. The verdict
    structure matches the schema in PR #134 / the evidence-schema
    research doc.
    """
    allocators = {name: get_allocator(name) for name in allocator_names}
    results = replay_all(allocators, bars)

    # Block 1: per-allocator
    per_allocator = {name: r.to_dict() for name, r in results.items()}

    # Block 2: paired comparisons
    paired_arrays = paired_daily_returns(results)
    paired_block = {}
    if incumbent in paired_arrays:
        a = paired_arrays[incumbent]
        for name, b in paired_arrays.items():
            if name == incumbent:
                continue
            paired_block[f"{incumbent}_vs_{name}"] = paired_comparison_metrics(
                a, b, name_a=incumbent, name_b=name,
            )

    # Block 3: significance
    verdicts = compute_significance_verdicts(results, pbo_n_slices=pbo_n_slices)
    significance_block = verdicts_to_dict(verdicts)

    # Block 4: regime stratification
    regime_block = regime_stratified_block(results, bars)

    # Block 5: violation report
    violation_block = violation_report_block(results)

    # Block 6: verdict
    verdict = assemble_verdict(
        significance_block, paired_block, violation_block, incumbent=incumbent,
    )

    return {
        "as_of_date": "<set-by-caller>",
        "n_bars": len(bars),
        "n_unique_dates": len({b.bar_date for b in bars}),
        "regime_distribution": _regime_counts(bars),
        "constraint_snapshot_contract_version": "v1-2026-06-03",
        "allocators": list(allocator_names),
        "per_allocator": per_allocator,
        "paired_comparisons": paired_block,
        "significance": significance_block,
        "regime_stratified": regime_block,
        "violation_report": violation_block,
        "verdict": verdict,
    }


def _regime_counts(bars: Sequence[AllocatorReplayBar]) -> dict[str, float]:
    counts: dict[str, int] = {}
    n_with_regime = 0
    for b in bars:
        if b.regime is None:
            continue
        counts[b.regime] = counts.get(b.regime, 0) + 1
        n_with_regime += 1
    if n_with_regime == 0:
        return {}
    return {r: c / n_with_regime for r, c in counts.items()}


# --------- CLI entry point ------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--wf-artifact-root", type=str, required=True,
                   help="Path to walk-forward artifact root")
    p.add_argument("--start-cut", type=str, required=True,
                   help="Earliest cutoff date (YYYY-MM-DD)")
    p.add_argument("--end-cut", type=str, required=True,
                   help="Latest cutoff date (YYYY-MM-DD)")
    p.add_argument("--out", type=str, required=True,
                   help="Output path for the verdict JSON")
    p.add_argument(
        "--allocators", type=str,
        default="equal_weight_top_k,inverse_vol_top_k,fractional_kelly_top_k",
        help="Comma-separated allocator names from the registry",
    )
    p.add_argument("--incumbent", type=str, default="fractional_kelly_top_k",
                   help="Incumbent allocator name for paired comparisons")
    p.add_argument("--pbo-n-slices", type=int, default=16)
    p.add_argument("--loader-module", type=str, default=None,
                   help="Optional module:function to load WF bars "
                        "(default uses placeholder synthetic bars)")
    args = p.parse_args(argv)

    bars = _load_bars(args)
    payload = run_replay(
        bars, args.allocators.split(","),
        incumbent=args.incumbent,
        pbo_n_slices=args.pbo_n_slices,
    )
    payload["as_of_date"] = "2026-06-03"
    payload["wf_artifact_root"] = args.wf_artifact_root
    payload["cut_range"] = [args.start_cut, args.end_cut]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("wrote verdict JSON to %s", out_path)
    return 0


def _load_bars(args) -> list[AllocatorReplayBar]:
    """Resolve --loader-module if supplied, else fall back to a tiny
    placeholder bar set so the CLI is end-to-end runnable before
    Step 4e's WF cut loader is merged.
    """
    if args.loader_module:
        mod_name, fn_name = args.loader_module.split(":")
        mod = importlib.import_module(mod_name)
        load_fn = getattr(mod, fn_name)
        return list(load_fn(args.wf_artifact_root, args.start_cut, args.end_cut))
    log.warning(
        "no --loader-module supplied; using a 30-bar placeholder. "
        "Step 4e's WF cut loader is the real implementation."
    )
    return _placeholder_bars()


def _placeholder_bars() -> list[AllocatorReplayBar]:
    """Minimum viable bar set so `main` is runnable end-to-end before
    Step 4e lands. NOT for decision-grade evidence."""
    from kernel.portfolio_qp.constraint_snapshot import ConstraintSnapshot
    rng = np.random.default_rng(0)
    bars = []
    for i in range(30):
        n = 3
        bars.append(AllocatorReplayBar(
            bar_date=f"placeholder-{i:03d}",
            snap=ConstraintSnapshot(
                n=n, tickers=tuple(f"T{j}" for j in range(n)),
                w_current=np.zeros(n),
                w_upper_hard=np.full(n, 0.50),
                w_upper=np.full(n, 0.50),
                w_lower=0.0,
                dw_max=np.full(n, 1.0),
                cash_reserve=0.0,
                turnover_max=None,
                drawdown=0.0,
                drawdown_limit=0.20,
                gross_max=None,
                wash_sale_mask=np.zeros(n, dtype=bool),
            ),
            mu=rng.uniform(0.0, 0.05, n),
            sigma=rng.uniform(0.10, 0.20, n),
            fwd_return=rng.normal(0.001, 0.01, n),
            regime="BULL_CALM",
            cost_per_trade_bps=5.0,
        ))
    return bars


if __name__ == "__main__":
    raise SystemExit(main())
