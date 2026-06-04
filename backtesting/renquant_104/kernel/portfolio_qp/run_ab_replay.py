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

The driver defaults to the WF cut loader in this package. Custom
loader injection remains available for experiments, but the CLI must
never silently emit decision evidence from synthetic placeholder bars.
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
    hard_only_qp_allocator,
    hybrid_option_f_allocator,
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
    # Step 4d/4f allocators — built but previously unregistered (#204 B3),
    # so they could never be named in --allocators. Registered here so the
    # full 5-baseline A/B (current QP / hard-only QP / Hybrid F / inverse-vol
    # / equal-weight) can actually be compared.
    "hybrid_option_f_allocator": hybrid_option_f_allocator,
    "hard_only_qp_allocator": hard_only_qp_allocator,
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
        "win_rate_a_beats_b_z_score": _win_rate_z_score(float(np.mean(a > b)), int(a.size)),
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


def _win_rate_z_score(win_rate: float, n_bars: int) -> Optional[float]:
    if n_bars <= 0:
        return None
    return float((win_rate - 0.5) / np.sqrt(0.25 / n_bars))


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
    win_rate_z_threshold: float = 2.0,
    constraints_decision_grade: bool = True,
) -> dict:
    """Apply the Step 4 non-negotiable gate + select promotion candidate.

    A promotion candidate must:
    1. Beat the incumbent on paired daily returns (delta_sharpe > 0
       AND candidate win-rate z-score > ``win_rate_z_threshold``).
    2. Pass the stricter §8 DSR/PBO gate.
    3. Have zero hard-constraint regressions.
    4. Be evaluated against decision-grade constraints.
    """
    candidates = [
        name for name, sig in significance.items()
        if name != incumbent
    ]
    promotion_candidate = None
    rationale = "no allocator beat the incumbent on all gates"
    incumbent_violated = violations["by_allocator"].get(incumbent, {}).get(
        "rejected_for_promotion", True,
    )
    gate_bits = {
        "zero_hard_constraint_regressions": not incumbent_violated,
        "pbo_below_0_5": False,
        "pbo_plus_se_below_0_55": False,
        "dsr_above_0_95": False,
        "win_rate_z_score_above_2": False,
        "decision_grade_constraints": bool(constraints_decision_grade),
    }
    if not constraints_decision_grade:
        return {
            "promotion_candidate": None,
            "rationale": (
                "not decision-grade: replay snapshots are missing required "
                "constraint families, so promotion is blocked"
            ),
            "fallback_recommendation": incumbent,
            "next_action": "iterate",
            "non_negotiable_gate_passed": gate_bits,
        }

    any_candidate_won_pair = False

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
        candidate_win_rate = 1.0 - float(pc.get("win_rate_a_beats_b", 1.0))
        candidate_win_rate_z = _win_rate_z_score(
            candidate_win_rate, int(pc.get("n_bars", 0))
        )
        beats_incumbent_paired = (
            delta_sharpe is not None and delta_sharpe < 0.0
            and candidate_win_rate_z is not None
            and candidate_win_rate_z > win_rate_z_threshold
        )
        if beats_incumbent_paired:
            any_candidate_won_pair = True
        passes_significance = sig.get(
            "live_promotable_per_section_8",
            sig.get("live_promotable_per_clause_7_4", False),
        )
        passes_violation_gate = not violated
        pbo = sig.get("pbo")
        pbo_se = sig.get("pbo_se")
        candidate_gate_bits = {
            "zero_hard_constraint_regressions": not violated,
            "pbo_below_0_5": pbo is None or pbo < 0.5,
            "pbo_plus_se_below_0_55": (
                pbo is None or pbo_se is None or (pbo + pbo_se) < 0.55
            ),
            "dsr_above_0_95": sig.get("dsr") is not None and sig["dsr"] >= 0.95,
            "win_rate_z_score_above_2": bool(beats_incumbent_paired),
            "decision_grade_constraints": True,
        }
        if beats_incumbent_paired and passes_significance and passes_violation_gate:
            promotion_candidate = name
            gate_bits = candidate_gate_bits
            rationale = (
                f"{name} beats {incumbent} on paired daily returns "
                f"(delta_sharpe={delta_sharpe:+.3f}, "
                f"win_rate={candidate_win_rate:.2f}, "
                f"win_rate_z={candidate_win_rate_z:.2f}), "
                "passes §8 DSR/PBO, zero hard-constraint regressions."
            )
            break

    return {
        "promotion_candidate": promotion_candidate,
        "rationale": rationale,
        "fallback_recommendation": incumbent,
        "next_action": (
            "promote_to_shadow" if promotion_candidate
            else ("iterate" if any_candidate_won_pair else "keep_incumbent")
        ),
        "non_negotiable_gate_passed": gate_bits,
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
    constraint_fidelity = constraint_fidelity_block(bars)

    # Block 6: verdict
    verdict = assemble_verdict(
        significance_block, paired_block, violation_block, incumbent=incumbent,
        constraints_decision_grade=constraint_fidelity["decision_grade"],
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
        "constraint_fidelity": constraint_fidelity,
        "verdict": verdict,
    }


def constraint_fidelity_block(bars: Sequence[AllocatorReplayBar]) -> dict:
    """Surface whether replay snapshots include the load-bearing hard caps.

    The sim DB loader currently cannot reconstruct per-cut sector maps.
    That output is still useful for smoke tests, but it must not drive a
    promotion decision because sector-cap regressions would be invisible.
    """
    n_bars = len(bars)
    missing_sector = 0
    missing_corr = 0
    for bar in bars:
        snap = bar.snap
        if snap.sector_indicator is None or snap.sector_cap_vec is None:
            missing_sector += 1
        if not snap.corr_group_pairs:
            missing_corr += 1
    missing_critical = []
    if n_bars == 0 or missing_sector:
        missing_critical.append("sector_cap")
    return {
        "decision_grade": not missing_critical,
        "missing_critical_families": missing_critical,
        "bars_missing_sector_cap": missing_sector,
        "bars_missing_corr_group_cap": missing_corr,
        "n_bars_checked": n_bars,
        "note": (
            "Promotion is blocked unless every replay bar carries the "
            "load-bearing hard-constraint families needed for the §8 zero "
            "hard-constraint-regression gate."
        ),
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
                   help="Path to sim_runs.db or a directory containing sim_runs.db")
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
    p.add_argument(
        "--fwd-horizon-days", type=int, default=60,
        help="Forward-return horizon for realised returns in the replay "
             "(must match a populated ticker_forward_returns column: "
             "1, 5, 10, 20, or 60). Default 60 matches the prod label.",
    )
    p.add_argument("--loader-module", type=str, default=None,
                   help="Optional module:function to load WF bars "
                        "(default uses wf_replay_loader.load_replay_bars_from_sim_db)")
    args = p.parse_args(argv)

    bars = _load_bars(args)
    # Fail loud on zero bars (#204 Task 4): the loader returns 0 bars when
    # the sim DB lacks the mu/sigma <-> forward-return overlap on the same
    # (date, ticker), or when --fwd-horizon-days points at an all-NULL
    # column. Previously this crashed deep in np.max() on an empty paired-
    # returns array; now we emit a structured invalid_experiment artifact
    # (mirrors the Kelly-AB no-trade guard, PR #202) and exit non-zero so
    # no consumer mistakes "no data" for "no difference".
    if not bars:
        invalid = {
            "invalid_experiment": True,
            "reason": "no_bars_loaded",
            "detail": (
                "loader returned 0 bars — the sim DB has no (date, ticker) row "
                "with score_distribution.mu+sigma AND "
                "ticker_forward_returns.fwd_<horizon> all non-NULL. Backfill the "
                "forward-return column and co-populate mu/sigma for watchlist "
                "tickers, or pass --fwd-horizon-days to a populated column. See "
                "doc/research/2026-06-04-qp-step4-replay-blocked-no-verdict.md."
            ),
            "wf_artifact_root": args.wf_artifact_root,
            "cut_range": [args.start_cut, args.end_cut],
            "fwd_horizon_days": args.fwd_horizon_days,
            "allocators": args.allocators.split(","),
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(invalid, indent=2, sort_keys=True))
        log.error(
            "no bars loaded for %s..%s @ fwd_%dd — wrote invalid_experiment to "
            "%s (NO verdict produced)",
            args.start_cut, args.end_cut, args.fwd_horizon_days, out_path,
        )
        return 2

    payload = run_replay(
        bars, args.allocators.split(","),
        incumbent=args.incumbent,
        pbo_n_slices=args.pbo_n_slices,
    )
    payload["as_of_date"] = "2026-06-03"
    payload["wf_artifact_root"] = args.wf_artifact_root
    payload["cut_range"] = [args.start_cut, args.end_cut]
    payload["fwd_horizon_days"] = args.fwd_horizon_days

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log.info("wrote verdict JSON to %s", out_path)
    return 0


def _load_bars(args) -> list[AllocatorReplayBar]:
    """Resolve --loader-module if supplied, else use the real WF DB loader.

    The `fwd_horizon_days` arg is recorded in the verdict JSON as the
    horizon actually used, so it MUST plumb through to the custom loader
    when one is supplied. Custom loaders that do not accept the kwarg
    raise loudly here rather than silently emitting verdict evidence
    against a different horizon than the CLI claims.
    """
    if args.loader_module:
        mod_name, fn_name = args.loader_module.split(":")
        mod = importlib.import_module(mod_name)
        load_fn = getattr(mod, fn_name)
        try:
            return list(load_fn(
                args.wf_artifact_root, args.start_cut, args.end_cut,
                fwd_horizon_days=args.fwd_horizon_days,
            ))
        except TypeError as exc:
            raise TypeError(
                f"--loader-module {args.loader_module!r} does not accept "
                f"`fwd_horizon_days` kwarg; the verdict JSON would claim "
                f"horizon={args.fwd_horizon_days} while the loader emitted "
                f"a different one. Add the kwarg to the loader signature, "
                f"or omit --fwd-horizon-days when using a custom loader "
                f"whose horizon is fixed."
            ) from exc

    from kernel.portfolio_qp.wf_replay_loader import (
        load_replay_bars_from_sim_db,
    )

    return load_replay_bars_from_sim_db(
        _default_sim_db_path(args.wf_artifact_root),
        args.start_cut,
        args.end_cut,
        fwd_horizon_days=args.fwd_horizon_days,
    )


def _default_sim_db_path(wf_artifact_root: str) -> Path:
    root = Path(wf_artifact_root)
    if root.is_dir():
        return root / "sim_runs.db"
    return root


if __name__ == "__main__":
    raise SystemExit(main())
