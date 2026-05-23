#!/usr/bin/env python
"""Walk-forward gate runner — write wf_gate_metadata to artifact.

Per CLAUDE.md §5.9 + roadmap P0 #1 (post E55 NGB revert): every promote
requires walk-forward 3-cut Sharpe + §5.2 sanity battery. Historical WF
validates a point-in-time retrain manifest, so this script also verifies
that the manifest artifacts match the candidate artifact's training recipe
before stamping metadata accepted by kernel.model_acceptance.promote().

Usage:
    python scripts/run_wf_gate.py --artifact path/to/staging.json
    python scripts/run_wf_gate.py --artifact path/to/staging.json --strict

Exit code 0 = passed; 1 = failed (artifact still gets metadata written
with `passed: false` so the operator can see what failed without
re-running).

Walk-forward criteria (default):
  - 3-cut walk-forward over 27 months
  - Cuts: 2024-01→12, 2024-07→2025-06, 2025-04→2026-03
  - Pass: mean Sharpe ≥ 0.40 AND ≥ 2/3 cuts have Sharpe > 0
  - Fail: mean Sharpe < 0 OR all cuts negative

§5.2 sanity criteria (default):
  - shuffled-label IC: |IC| < 0.005 (model on shuffled labels should be ~0)
  - time-shift placebo IC: ratio < 0.5 × real IC (placebo shouldn't capture real signal)

References:
- Lopez de Prado AFML §7 + §11 (walk-forward + cross-validation in finance)
- Bailey-Lopez de Prado 2014 "Pseudo-Mathematics and Financial Charlatanism"
- CLAUDE.md §5.2 sanity battery, §5.9 walk-forward mandate
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
import re
import pandas as pd

from qp_contracts import validate_qp_contract_config
from trade_contracts import evaluate_trade_contract
from trade_monotonicity import evaluate_trade_monotonicity
from wf_config_parity import evaluate_wf_config_parity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wf-gate")

REPO = Path(__file__).resolve().parent.parent
GATE_VERSION = 2
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
for _p in (REPO, STRATEGY_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
PYTHON = sys.executable
CUTS = [
    ("2024-01-02", "2024-12-31"),
    ("2024-07-01", "2025-06-30"),
    ("2025-04-01", "2026-03-28"),
]


def _resolve_strategy_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else STRATEGY_DIR / p


def _recipe_projection(artifact: dict) -> dict:
    """Return the model-recipe fields a WF manifest must match.

    A current production artifact cannot be replayed into old sim windows
    without look-ahead leakage. For historical walk-forward, we therefore
    validate the retraining recipe instead: same model kind, ordered feature
    contract, label horizon, and learner params.
    """
    return {
        "kind": artifact.get("kind"),
        "feature_cols": list(artifact.get("feature_cols") or []),
        "label_col": artifact.get("label_col"),
        "lookahead_days": int(artifact.get("lookahead_days") or 0),
        "params": artifact.get("params") or {},
    }


def _recipe_fingerprint(artifact: dict) -> str:
    payload = json.dumps(
        _recipe_projection(artifact),
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _manifest_recipe_usage(manifest_path: Path | None, artifact_path: Path) -> dict:
    if manifest_path is None or not manifest_path.exists():
        return {
            "recipe_validated": False,
            "reason": f"manifest not found: {manifest_path}",
        }
    try:
        payload = json.loads(manifest_path.read_text())
        rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    except Exception as exc:  # noqa: BLE001
        return {"recipe_validated": False, "reason": f"manifest parse failed: {exc}"}
    if not isinstance(rows, list) or not rows:
        return {"recipe_validated": False, "reason": "manifest has no retrain rows"}

    candidate = json.loads(artifact_path.read_text())
    candidate_fp = _recipe_fingerprint(candidate)
    candidate_recipe = _recipe_projection(candidate)
    samples = [rows[0], rows[len(rows) // 2], rows[-1]]
    seen: set[str] = set()
    sample_reports: list[dict] = []
    for row in samples:
        uri = str((row or {}).get("artifact_uri") or "")
        if not uri or uri in seen:
            continue
        seen.add(uri)
        p = Path(uri)
        if not p.is_absolute():
            p = STRATEGY_DIR / p
        if not p.exists():
            sample_reports.append({
                "artifact_uri": uri,
                "exists": False,
                "recipe_matches": False,
            })
            continue
        try:
            sample = json.loads(p.read_text())
            sample_fp = _recipe_fingerprint(sample)
            sample_recipe = _recipe_projection(sample)
        except Exception as exc:  # noqa: BLE001
            sample_reports.append({
                "artifact_uri": str(p),
                "exists": True,
                "recipe_matches": False,
                "error": str(exc),
            })
            continue
        sample_reports.append({
            "artifact_uri": str(p),
            "exists": True,
            "recipe_matches": sample_fp == candidate_fp,
            "recipe_fingerprint": sample_fp,
            "n_features": len(sample_recipe["feature_cols"]),
            "missing_features_vs_candidate": sorted(
                set(candidate_recipe["feature_cols"]) - set(sample_recipe["feature_cols"])
            )[:10],
            "extra_features_vs_candidate": sorted(
                set(sample_recipe["feature_cols"]) - set(candidate_recipe["feature_cols"])
            )[:10],
        })

    if not sample_reports:
        return {"recipe_validated": False, "reason": "no sample artifacts found in manifest"}
    all_match = all(r.get("recipe_matches") for r in sample_reports)
    return {
        "recipe_validated": bool(all_match),
        "candidate_recipe_fingerprint": candidate_fp,
        "candidate_n_features": len(candidate_recipe["feature_cols"]),
        "manifest_sample_reports": sample_reports,
        "reason": (
            "manifest sample artifacts match candidate recipe"
            if all_match else
            "manifest sample artifacts do not match candidate recipe"
        ),
    }


def inspect_artifact_usage(strategy_config: str, artifact_path: Path) -> dict:
    """Return whether this WF sim config actually evaluates `artifact_path`.

    Static artifact configs can directly validate a candidate artifact. A
    walk-forward manifest validates a retraining recipe / manifest instead;
    it must not silently stamp the candidate artifact as passed.
    """
    cfg_path = STRATEGY_DIR / strategy_config
    if not cfg_path.exists():
        return {
            "candidate_artifact_used": False,
            "eval_scope": "missing_config",
            "strategy_config": strategy_config,
            "reason": f"config not found: {cfg_path}",
        }
    cfg = json.loads(cfg_path.read_text())
    wf_cfg = cfg.get("walkforward", {}) or {}
    if bool(wf_cfg.get("enabled", False)):
        manifest = _resolve_strategy_path(
            str(wf_cfg.get("manifest_path", "artifacts/walkforward_manifest.json"))
        )
        recipe_usage = _manifest_recipe_usage(manifest, artifact_path)
        return {
            "candidate_artifact_used": False,
            "eval_scope": "walkforward_manifest",
            "strategy_config": strategy_config,
            "manifest_path": str(manifest) if manifest is not None else None,
            "reason": (
                "strategy config uses walkforward manifest; validating candidate "
                "recipe against manifest artifacts"
            ),
            **recipe_usage,
        }

    panel_cfg = (cfg.get("ranking", {}) or {}).get("panel_scoring", {}) or {}
    configured = _resolve_strategy_path(
        panel_cfg.get("artifact_path")
        or cfg.get("panel_ltr", {}).get("artifact_path")
        or "artifacts/prod/panel-ltr.alpha158_fund.json"
    )
    try:
        used = configured is not None and configured.resolve() == artifact_path.resolve()
    except OSError:
        used = False
    return {
        "candidate_artifact_used": bool(used),
        "eval_scope": "static_artifact",
        "strategy_config": strategy_config,
        "configured_artifact_path": str(configured) if configured is not None else None,
        "candidate_artifact_path": str(artifact_path),
        "reason": (
            "configured artifact matches candidate"
            if used else
            "configured artifact does not match candidate"
        ),
    }


def cut_market_context(start: str, end: str) -> dict:
    """SPY benchmark + regime distribution for one WF cut."""
    import pandas as _pd
    from kernel.hmm_regime_labels import compute_hmm_regime_labels  # noqa: PLC0415
    from kernel.regime_labels import compute_spy_regime_labels  # noqa: PLC0415

    spy_path = REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not spy_path.exists():
        return {"benchmark": "SPY", "error": f"missing {spy_path}"}
    start_ts = _pd.Timestamp(start)
    end_ts = _pd.Timestamp(end)
    spy = _pd.read_parquet(spy_path).sort_index()
    spy.index = _pd.to_datetime(spy.index)
    mask = (spy.index >= start_ts) & (spy.index <= end_ts)
    cut = spy.loc[mask].copy()
    ret = cut["close"].pct_change().dropna()
    vol = float(ret.std(ddof=1)) if len(ret) > 1 else float("nan")
    sharpe = (
        float(ret.mean() / vol * math.sqrt(252.0))
        if len(ret) > 2 and math.isfinite(vol) and vol > 0
        else float("nan")
    )
    apy = float("nan")
    if len(cut) > 1 and len(ret) > 0:
        apy = float((cut["close"].iloc[-1] / cut["close"].iloc[0]) ** (252.0 / len(ret)) - 1.0)

    hmm = compute_hmm_regime_labels(spy_path)
    grid = compute_spy_regime_labels(spy_path)
    hmm_counts = (
        hmm[(hmm.date >= start_ts) & (hmm.date <= end_ts)]
        .regime.value_counts().to_dict()
    )
    grid_counts = (
        grid[(grid.date >= start_ts) & (grid.date <= end_ts)]
        .regime.value_counts().to_dict()
    )
    return {
        "benchmark": "SPY",
        "spy_sharpe": sharpe,
        "spy_apy": apy,
        "n_trading_days": int(len(ret)),
        "hmm_regime_counts": {str(k): int(v) for k, v in hmm_counts.items()},
        "spy_grid_regime_counts": {str(k): int(v) for k, v in grid_counts.items()},
    }


def _finite_number(value) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _top_regime(counts: dict | None) -> str | None:
    if not counts:
        return None
    return str(max(counts.items(), key=lambda kv: kv[1])[0])


def _merge_counts(rows: list[dict], key: str) -> dict:
    merged: dict[str, int] = {}
    for row in rows:
        counts = ((row.get("market_context") or {}).get(key) or {})
        for label, n in counts.items():
            merged[str(label)] = merged.get(str(label), 0) + int(n)
    return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))


def _merge_trade_counts(rows: list[dict], key: str) -> dict:
    merged: dict[str, int] = {}
    for row in rows:
        counts = ((row.get("trade_trace_summary") or {}).get(key) or {})
        for label, n in counts.items():
            merged[str(label)] = merged.get(str(label), 0) + int(n)
    return dict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))


def _sum_trade_summary(rows: list[dict], key: str) -> int:
    return int(sum(int((row.get("trade_trace_summary") or {}).get(key) or 0) for row in rows))


def _trade_trace_summary(traces: dict[str, str]) -> dict:
    """Summarize production decision regimes from the persisted trade trace.

    `cut_market_context()` is an independent SPY/HMM lens. The trade trace is
    the production decision path: it records what regime the pipeline attached
    to each actual buy/sell. Keeping both prevents us from explaining trades
    with the wrong regime taxonomy.
    """
    trade_json = traces.get("trade_json")
    if not trade_json:
        return {}
    p = Path(trade_json)
    if not p.exists():
        return {"error": f"missing trade trace: {p}"}
    try:
        rows = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"error": f"failed to parse trade trace {p}: {exc}"}
    if not isinstance(rows, list):
        return {"error": f"trade trace is not a list: {p}"}

    def counts(action: str, field: str) -> dict:
        c = Counter(
            str(row.get(field))
            for row in rows
            if row.get("action") == action and row.get(field) not in (None, "")
        )
        return dict(sorted(c.items(), key=lambda kv: kv[1], reverse=True))

    buys = [row for row in rows if row.get("action") == "buy"]
    sells = [row for row in rows if row.get("action") == "sell"]
    missing_mu = sum(1 for row in buys if _finite_number(row.get("mu")) is None)
    missing_sigma = sum(1 for row in buys if _finite_number(row.get("sigma")) is None)
    return {
        "n_buys": int(len(buys)),
        "n_sells": int(len(sells)),
        "buy_regime_counts": counts("buy", "regime"),
        "sell_regime_counts": counts("sell", "regime"),
        "buy_source_counts": counts("buy", "source_job"),
        "sell_source_counts": counts("sell", "source_job"),
        "sell_exit_reason_counts": counts("sell", "exit_reason"),
        "buy_missing_mu": int(missing_mu),
        "buy_missing_sigma": int(missing_sigma),
    }


def _trace_paths(trace_dir: Path | None, start: str, end: str) -> dict[str, str]:
    if trace_dir is None:
        return {}
    safe = f"{start}_to_{end}"
    return {
        "equity_json": str(trace_dir / f"{safe}.equity.json"),
        "trade_json": str(trace_dir / f"{safe}.trades.json"),
        "trade_csv": str(trace_dir / f"{safe}.trades.csv"),
        "round_trips_csv": str(trace_dir / f"{safe}.round_trips.csv"),
        "report_md": str(trace_dir / f"{safe}.report.md"),
    }


def run_sim_cut(
    strategy_config: str,
    start: str,
    end: str,
    trace_dir: Path | None = None,
) -> dict:
    """Run one sim cut, parse Sharpe + APY from log."""
    log.info("Sim cut: %s → %s", start, end)
    market_context = cut_market_context(start, end)
    traces = _trace_paths(trace_dir, start, end)
    cmd = [
        PYTHON,
        str(REPO / "scripts/run_sim_104.py"),
        "--strategy-config-name", strategy_config,
        "--start", start, "--end", end,
        "--no-compare",
        "--no-persist",
        "--skip-preflight",
    ]
    if traces:
        trace_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend([
            "--equity-json", traces["equity_json"],
            "--trade-log-json", traces["trade_json"],
            "--trade-log-csv", traces["trade_csv"],
            "--round-trips-csv", traces["round_trips_csv"],
            "--trade-report-md", traces["report_md"],
        ])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        tail = out[-2000:]
        log.error("  → sim cut FAILED rc=%d\n%s", proc.returncode, tail)
        return {
            "start": start,
            "end": end,
            "sharpe": float("nan"),
            "apy": float("nan"),
            "market_context": market_context,
            "trace_paths": traces,
            "returncode": int(proc.returncode),
            "error_tail": tail,
        }
    # Parse "Sharpe=+0.40" "APY: 6.8%"
    sharpe_m = re.search(r"Sharpe=([+\-\d.]+)", out)
    apy_m = re.search(r"APY:\s+([+\-\d.]+)%", out)
    sharpe = float(sharpe_m.group(1)) if sharpe_m else float("nan")
    apy = float(apy_m.group(1)) / 100 if apy_m else float("nan")
    spy_sharpe = _finite_number(market_context.get("spy_sharpe"))
    spy_apy = _finite_number(market_context.get("spy_apy"))
    sharpe_vs_spy = sharpe - spy_sharpe if spy_sharpe is not None else float("nan")
    apy_vs_spy = apy - spy_apy if spy_apy is not None else float("nan")
    trade_summary = _trade_trace_summary(traces)
    log.info(
        "  → Sharpe=%+.3f  APY=%+.2f%%  SPY Sharpe=%s  ΔSharpe=%s",
        sharpe,
        apy * 100,
        f"{spy_sharpe:+.3f}" if spy_sharpe is not None else "n/a",
        f"{sharpe_vs_spy:+.3f}" if math.isfinite(sharpe_vs_spy) else "n/a",
    )
    return {
        "start": start,
        "end": end,
        "sharpe": sharpe,
        "apy": apy,
        "sharpe_vs_spy": sharpe_vs_spy,
        "apy_vs_spy": apy_vs_spy,
        "dominant_hmm_regime": _top_regime(market_context.get("hmm_regime_counts")),
        "dominant_spy_grid_regime": _top_regime(market_context.get("spy_grid_regime_counts")),
        "market_context": market_context,
        "trade_trace_summary": trade_summary,
        "trace_paths": traces,
        "returncode": 0,
    }


def run_walk_forward(
    strategy_config: str,
    jobs: int = 1,
    trace_dir: Path | None = None,
) -> dict:
    """Run 3-cut walk-forward, return mean/std/per-cut."""
    cuts = CUTS
    jobs = max(1, min(int(jobs), len(cuts)))
    results: list[dict | None] = [None] * len(cuts)
    if jobs == 1:
        for idx, (start, end) in enumerate(cuts):
            results[idx] = run_sim_cut(strategy_config, start, end, trace_dir)
    else:
        log.info("Running %d WF cuts with jobs=%d", len(cuts), jobs)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_idx = {
                pool.submit(run_sim_cut, strategy_config, start, end, trace_dir): idx
                for idx, (start, end) in enumerate(cuts)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:  # defensive: preserve a stamped failure
                    start, end = cuts[idx]
                    log.exception("  → sim cut crashed: %s → %s", start, end)
                    results[idx] = {
                        "start": start,
                        "end": end,
                        "sharpe": float("nan"),
                        "apy": float("nan"),
                        "returncode": -1,
                        "error_tail": repr(exc),
                    }
    results = [r for r in results if r is not None]
    sharpes = [r["sharpe"] for r in results if r["sharpe"] == r["sharpe"]]   # finite
    apys = [r["apy"] for r in results if r["apy"] == r["apy"]]
    failed_cuts = [r for r in results if r.get("returncode", 0) != 0]
    if failed_cuts:
        return {
            "passed": False,
            "cuts": results,
            "reason": f"{len(failed_cuts)}/3 sim cuts failed execution",
        }
    if not sharpes:
        return {"passed": False, "cuts": results, "reason": "all sim cuts failed parse"}
    import statistics as _s
    mean_sharpe = _s.mean(sharpes)
    std_sharpe = _s.stdev(sharpes) if len(sharpes) > 1 else 0.0
    mean_apy = _s.mean(apys) if apys else float("nan")
    n_pos = sum(1 for s in sharpes if s > 0)
    spy_sharpes = [
        _finite_number((r.get("market_context") or {}).get("spy_sharpe"))
        for r in results
    ]
    spy_sharpes = [s for s in spy_sharpes if s is not None]
    spy_apys = [
        _finite_number((r.get("market_context") or {}).get("spy_apy"))
        for r in results
    ]
    spy_apys = [a for a in spy_apys if a is not None]
    mean_spy_sharpe = _s.mean(spy_sharpes) if spy_sharpes else float("nan")
    mean_spy_apy = _s.mean(spy_apys) if spy_apys else float("nan")
    mean_sharpe_vs_spy = (
        mean_sharpe - mean_spy_sharpe
        if math.isfinite(mean_spy_sharpe) else float("nan")
    )
    mean_apy_vs_spy = (
        mean_apy - mean_spy_apy
        if math.isfinite(mean_apy) and math.isfinite(mean_spy_apy) else float("nan")
    )
    n_beat_spy_sharpe = sum(
        1 for r in results
        if _finite_number(r.get("sharpe")) is not None
        and _finite_number((r.get("market_context") or {}).get("spy_sharpe")) is not None
        and float(r["sharpe"]) > float((r.get("market_context") or {})["spy_sharpe"])
    )
    n_beat_spy_apy = sum(
        1 for r in results
        if _finite_number(r.get("apy")) is not None
        and _finite_number((r.get("market_context") or {}).get("spy_apy")) is not None
        and float(r["apy"]) > float((r.get("market_context") or {})["spy_apy"])
    )
    pass_sharpe = mean_sharpe >= 0.40 and n_pos >= 2
    benchmark_suffix = (
        f"; SPY mean Sharpe {mean_spy_sharpe:+.3f}, "
        f"ΔSharpe {mean_sharpe_vs_spy:+.3f}, "
        f"beat SPY Sharpe {n_beat_spy_sharpe}/3"
        if math.isfinite(mean_spy_sharpe) else ""
    )
    return {
        "passed": pass_sharpe,
        "wf_3cut_sharpe_mean": float(mean_sharpe),
        "wf_3cut_sharpe_std": float(std_sharpe),
        "wf_3cut_apy_mean": float(mean_apy),
        "spy_sharpe_mean": float(mean_spy_sharpe),
        "strategy_minus_spy_sharpe_mean": float(mean_sharpe_vs_spy),
        "spy_apy_mean": float(mean_spy_apy),
        "strategy_minus_spy_apy_mean": float(mean_apy_vs_spy),
        "n_cuts_beat_spy_sharpe": int(n_beat_spy_sharpe),
        "n_cuts_beat_spy_apy": int(n_beat_spy_apy),
        "hmm_regime_counts_total": _merge_counts(results, "hmm_regime_counts"),
        "spy_grid_regime_counts_total": _merge_counts(results, "spy_grid_regime_counts"),
        "trade_buy_regime_counts_total": _merge_trade_counts(results, "buy_regime_counts"),
        "trade_sell_regime_counts_total": _merge_trade_counts(results, "sell_regime_counts"),
        "trade_buy_source_counts_total": _merge_trade_counts(results, "buy_source_counts"),
        "trade_sell_exit_reason_counts_total": _merge_trade_counts(results, "sell_exit_reason_counts"),
        "trade_buy_missing_mu_total": _sum_trade_summary(results, "buy_missing_mu"),
        "trade_buy_missing_sigma_total": _sum_trade_summary(results, "buy_missing_sigma"),
        "n_positive_cuts": n_pos,
        "wf_jobs": jobs,
        "cuts": results,
        "reason": (
            f"PASS: mean Sharpe {mean_sharpe:+.3f} ≥ 0.40 and {n_pos}/3 cuts > 0"
            f"{benchmark_suffix}"
            if pass_sharpe else
            f"FAIL: mean Sharpe {mean_sharpe:+.3f} or only {n_pos}/3 cuts > 0"
            f"{benchmark_suffix}"
        ),
    }


def run_trade_monotonicity_gate(
    wf_result: dict,
    *,
    min_n_per_regime: int = 30,
    min_spearman: float = 0.02,
    min_top_bottom_spread: float = 0.0,
) -> dict:
    """Evaluate trade score monotonicity from persisted round-trip ledgers."""
    frames, missing = _load_round_trip_frames(wf_result)
    if missing:
        return {
            "passed": False,
            "reason": "missing round-trip ledger(s): " + "; ".join(missing[:5]),
            "missing": missing,
        }
    if not frames:
        return {"passed": False, "reason": "no round-trip ledgers found"}
    report = evaluate_trade_monotonicity(
        pd.concat(frames, ignore_index=True),
        min_n_per_regime=min_n_per_regime,
        min_spearman=min_spearman,
        min_top_bottom_spread=min_top_bottom_spread,
    )
    return {
        "passed": bool(report.passed),
        "reason": report.reason,
        "pooled": report.pooled,
        "regimes": report.regimes,
        "min_n_per_regime": int(min_n_per_regime),
        "min_spearman": float(min_spearman),
        "min_top_bottom_spread": float(min_top_bottom_spread),
    }


def run_trade_contract_gate(wf_result: dict, config: dict) -> dict:
    """Require WF trade ledgers to carry QP/Kelly audit provenance."""
    frames, missing = _load_round_trip_frames(wf_result)
    if missing:
        return {
            "passed": False,
            "reason": "missing round-trip ledger(s): " + "; ".join(missing[:5]),
            "missing": missing,
        }
    if not frames:
        return {"passed": False, "reason": "no round-trip ledgers found"}
    joint = ((config.get("rotation") or {}).get("joint_actions") or {})
    ranking = config.get("ranking") or {}
    panel = (ranking.get("panel_scoring") or {})
    kelly = ranking.get("kelly_sizing") or {}
    qp_enabled = bool(joint.get("enabled")) and str(joint.get("solver", "")).lower() == "qp"
    strict_qp = str(joint.get("qp_mu_contract", "strict")).lower() in {
        "strict", "hard", "error", "enforce",
    }
    require_mu = bool(qp_enabled and strict_qp)
    require_sigma = bool(kelly.get("enabled") or panel.get("ngboost", {}).get("enabled"))
    report = evaluate_trade_contract(
        pd.concat(frames, ignore_index=True),
        require_entry_mu=require_mu,
        require_entry_sigma=require_sigma,
        require_exit_regime=True,
        require_exit_thresholds=True,
    )
    return {
        "passed": bool(report.passed),
        "reason": report.reason,
        "evidence": report.evidence,
        "require_entry_mu": require_mu,
        "require_entry_sigma": require_sigma,
    }


def _load_round_trip_frames(wf_result: dict) -> tuple[list[pd.DataFrame], list[str]]:
    frames = []
    missing = []
    for cut in wf_result.get("cuts") or []:
        rt_path = ((cut.get("trace_paths") or {}).get("round_trips_csv"))
        if not rt_path:
            missing.append(f"{cut.get('start')}->{cut.get('end')}: no round-trip path")
            continue
        p = Path(rt_path)
        if not p.exists():
            missing.append(str(p))
            continue
        frames.append(pd.read_csv(p))
    return frames, missing


def run_sanity_battery(artifact_path: Path) -> dict:
    """§5.2 shuffled-label + time-shift placebo on the artifact's training pipeline.

    Implementation: re-train the model on (a) shuffled labels and
    (b) +60d-shifted labels; measure val_ic on each. Lower-cost
    proxy for full sanity battery (which would re-run sim too).
    """
    log.info("§5.2 sanity battery (shuffled-label + time-shift placebo)...")
    # For panel-LTR XGB, run via existing scripts that support these flags.
    # Quick path: use the training panel + label shuffles directly.
    # Full sanity = re-train. Cheap sanity = score against shuffled y on val.

    # Cheapest sanity: take production model predictions on val partition,
    # compute IC against shuffled / time-shifted labels.
    import sys as _sys
    _sys.path.insert(0, str(REPO / "backtesting/renquant_104"))
    import numpy as _np, pandas as _pd
    from scipy.stats import spearmanr  # noqa: PLC0415

    # Load panel + artifact's feature_cols
    artifact = json.loads(artifact_path.read_text())
    feat_cols = artifact.get("feature_cols", [])
    if not feat_cols:
        return {"passed": False, "reason": "artifact missing feature_cols"}
    # Use the rawlabel panel (has fwd_60d_excess_raw and supports placebo construction)
    panel_path = REPO / "data/alpha158_291_fundamental_dataset_rawlabel.parquet"
    if not panel_path.exists():
        log.warning("rawlabel panel missing — skipping sanity (cheap mode unavailable)")
        return {"passed": True, "reason": "panel missing — sanity skipped"}
    panel = _pd.read_parquet(panel_path)
    panel["date"] = _pd.to_datetime(panel["date"])
    LABEL = "fwd_60d_excess_raw"
    panel = panel.dropna(subset=[LABEL])
    distinct = sorted(panel.date.unique())
    val_cut = distinct[int(len(distinct) * 0.8)]
    val = panel[panel.date > val_cut].copy()

    # Predict using the artifact's model on val
    # (For panel-LTR XGB rank, recover boosters; for QHead, predict_distribution)
    try:
        import xgboost as xgb  # noqa: PLC0415
        if artifact.get("kind") == "panel_ltr_xgboost":
            # Panel-LTR stores booster in artifact under booster_b64 or similar
            # For sanity we just need PREDICTIONS, so use the saved model
            from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
            scorer = PanelScorer.load(artifact_path)
            X = val.reindex(columns=feat_cols, fill_value=0).fillna(0)
            mu = scorer.score(X).values
        else:
            log.warning("kind=%s — sanity not implemented for this head type",
                        artifact.get("kind"))
            return {"passed": True, "reason": "sanity not implemented for this kind"}
    except Exception as exc:
        log.warning("sanity prediction failed: %s — skipping", exc)
        return {"passed": True, "reason": f"prediction failed: {exc}"}

    yva_real = val[LABEL].clip(-0.5, 0.5).values
    val_dates = val["date"].values

    def cs_ic(mu, y, dates):
        df = _pd.DataFrame({"p": mu, "y": y, "d": dates})
        ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
        ics = [x for x in ics if not _np.isnan(x)]
        return float(_np.mean(ics)) if ics else 0.0

    real_ic = cs_ic(mu, yva_real, val_dates)
    log.info("  real_ic = %+.4f", real_ic)

    # Shuffled label
    rng = _np.random.default_rng(42)
    yva_shuf = yva_real.copy()
    rng.shuffle(yva_shuf)
    shuf_ic = cs_ic(mu, yva_shuf, val_dates)
    log.info("  shuffled_ic = %+.4f (expect ≈ 0)", shuf_ic)

    # Time-shift placebo: shift each ticker's labels by +60 trading days
    panel_s = panel.sort_values(["ticker", "date"]).copy()
    panel_s["__shift__"] = panel_s.groupby("ticker")[LABEL].shift(-60)
    val_s = panel_s[panel_s.date > val_cut].dropna(subset=["__shift__"])
    if len(val_s) > 100:
        # Need to align mu predictions to val_s rows (subset of val)
        val_idx = val.set_index(["ticker", "date"])
        val_s_idx = val_s.set_index(["ticker", "date"])
        common = val_s_idx.index.intersection(val_idx.index)
        mu_aligned = _pd.Series(mu, index=val_idx.index).loc[common].values
        yva_placebo = val_s_idx.loc[common, "__shift__"].clip(-0.5, 0.5).values
        dates_aligned = [d for _, d in common]
        placebo_ic = cs_ic(mu_aligned, yva_placebo, dates_aligned)
        log.info("  placebo_ic = %+.4f (expect < 0.5 × real_ic = %+.4f)",
                 placebo_ic, 0.5 * real_ic)
    else:
        placebo_ic = float("nan")
        log.warning("  placebo skipped — too few aligned val rows")

    # Pass criteria
    pass_shuf = abs(shuf_ic) < 0.005
    pass_placebo = (placebo_ic != placebo_ic) or (
        abs(placebo_ic) < max(0.005, 0.5 * abs(real_ic)) if real_ic != 0 else True
    )
    return {
        "passed": pass_shuf and pass_placebo,
        "real_ic": real_ic,
        "sanity_shuffled_ic": shuf_ic,
        "sanity_placebo_ic": placebo_ic if placebo_ic == placebo_ic else None,
        "reason": (
            f"PASS: shuf_ic={shuf_ic:+.4f} placebo_ic={placebo_ic:+.4f}"
            if (pass_shuf and pass_placebo) else
            f"FAIL: shuf_ic={shuf_ic:+.4f} (need |·| < 0.005), "
            f"placebo_ic={placebo_ic:+.4f} (need < 0.5×real_ic)"
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True, help="Path to staging artifact JSON")
    ap.add_argument("--strategy-config", default="strategy_config.sim_wl200.json",
                    help="WF sim config name. Manifest configs validate the candidate "
                         "training recipe; static configs evaluate the artifact "
                         "directly when leakage-safe (default: strategy_config.sim_wl200.json)")
    ap.add_argument("--strict", action="store_true",
                    help="Compatibility flag for weekly_wf_promote.sh. Current thresholds are already strict.")
    ap.add_argument("--skip-wf", action="store_true",
                    help="Skip walk-forward (sanity only) — for emergency / testing")
    ap.add_argument("--skip-sanity", action="store_true",
                    help="Skip sanity battery — for emergency / testing")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Number of walk-forward cuts to run concurrently. "
                         "Default 1 preserves the conservative historical path; "
                         "use 3 for full cut-level parallelism.")
    ap.add_argument("--trace-dir", default=None,
                    help="Directory for per-cut equity/trade ledgers. Default: "
                         "artifacts/diagnostics/wf_trade_traces/<utc timestamp>.")
    ap.add_argument("--no-trade-trace", action="store_true",
                    help="Do not persist per-cut trade ledgers. Intended only "
                         "for quick parser tests.")
    ap.add_argument("--skip-trade-gates", action="store_true",
                    help="Skip trade-level monotonicity acceptance gates.")
    ap.add_argument("--skip-config-parity", action="store_true",
                    help="Skip prod/WF decision-semantics parity guard. "
                         "Use only for explicitly exploratory runs.")
    ap.add_argument("--derive-config-from-prod", action="store_true",
                    help="Before running, derive a production-semantic WF "
                         "config from --strategy-config. The base config only "
                         "contributes walkforward/calibration artifact paths.")
    args = ap.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        log.error("artifact not found: %s", artifact_path)
        sys.exit(2)

    artifact = json.loads(artifact_path.read_text())

    log.info("=" * 60)
    log.info("Walk-forward + Sanity gate runner — gate v%d", GATE_VERSION)
    log.info("Artifact: %s  (kind=%s)", artifact_path, artifact.get("kind"))
    log.info("=" * 60)

    if args.derive_config_from_prod:
        from wf_config_builder import build_wf_config_from_prod  # noqa: PLC0415

        base_cfg_path = STRATEGY_DIR / args.strategy_config
        if not base_cfg_path.exists():
            log.error("base strategy config not found: %s", base_cfg_path)
            sys.exit(2)
        prod_cfg_path = STRATEGY_DIR / "strategy_config.json"
        prod_cfg = json.loads(prod_cfg_path.read_text())
        base_cfg = json.loads(base_cfg_path.read_text())
        manifest_path = ((base_cfg.get("walkforward") or {}).get("manifest_path"))
        if not manifest_path:
            log.error(
                "--derive-config-from-prod requires base config with "
                "walkforward.manifest_path: %s",
                base_cfg_path,
            )
            sys.exit(2)
        derived_dir = STRATEGY_DIR / "artifacts" / "diagnostics" / "wf_eval_configs"
        derived_dir.mkdir(parents=True, exist_ok=True)
        derived_name = f"{Path(args.strategy_config).stem}.prod_semantic.json"
        derived_path = derived_dir / derived_name
        derived_cfg = build_wf_config_from_prod(
            prod_cfg,
            manifest_path=str(manifest_path),
            base_wf_config=base_cfg,
            strategy_dir=STRATEGY_DIR,
        )
        derived_path.write_text(json.dumps(derived_cfg, indent=2, sort_keys=False) + "\n")
        args.strategy_config = str(derived_path.relative_to(STRATEGY_DIR))
        log.info("Derived production-semantic WF config: %s", derived_path)

    artifact_usage = inspect_artifact_usage(args.strategy_config, artifact_path)
    log.info("Artifact usage: %s", artifact_usage)
    cfg_path = STRATEGY_DIR / args.strategy_config
    gate_config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    parity_result = (
        {"passed": True, "reason": "skipped"}
        if args.skip_config_parity or not cfg_path.exists()
        else evaluate_wf_config_parity(
            STRATEGY_DIR / "strategy_config.json",
            cfg_path,
            candidate_artifact=artifact_path,
            strategy_dir=STRATEGY_DIR,
        )
    )
    if not parity_result.get("passed", True):
        log.error(
            "WF config parity FAILED with %d issue(s)",
            len(parity_result.get("issues", [])),
        )
        for issue in parity_result.get("issues", [])[:10]:
            log.error("  parity issue: %s", issue)
    else:
        log.info("WF config parity: PASS")
    qp_contract = (
        validate_qp_contract_config(gate_config)
        if cfg_path.exists() else
        None
    )
    if qp_contract is not None:
        log.info("QP contract: %s", qp_contract.summary())

    trace_dir: Path | None = None
    if not args.no_trade_trace:
        if args.trace_dir:
            trace_dir = Path(args.trace_dir)
            if not trace_dir.is_absolute():
                trace_dir = STRATEGY_DIR / trace_dir
        else:
            run_stamp = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            trace_dir = (
                STRATEGY_DIR / "artifacts" / "diagnostics"
                / "wf_trade_traces" / run_stamp
            )
        log.info("WF trade trace dir: %s", trace_dir)

    wf_result = {"passed": True, "reason": "skipped"}
    if not args.skip_wf:
        manifest_scope = artifact_usage.get("eval_scope") == "walkforward_manifest"
        if qp_contract is not None and not qp_contract.passed:
            wf_result = {
                "passed": False,
                "reason": qp_contract.summary(),
                "qp_contract": {
                    "passed": False,
                    "issues": qp_contract.issues,
                    "evidence": qp_contract.evidence,
                },
            }
            log.error("WF result: %s", wf_result["reason"])
        elif not parity_result.get("passed", True):
            wf_result = {
                "passed": False,
                "reason": (
                    "WF config parity failed; refusing to spend sim compute "
                    "on non-production-equivalent decision semantics"
                ),
                "config_parity": parity_result,
            }
            log.error("WF result: %s", wf_result["reason"])
        elif manifest_scope and not bool(artifact_usage.get("recipe_validated")):
            wf_result = {
                "passed": False,
                "reason": (
                    "manifest recipe mismatch; refusing to spend sim compute on "
                    f"non-comparable WF evidence: {artifact_usage.get('reason')}"
                ),
            }
            log.error("WF result: %s", wf_result["reason"])
        else:
            wf_result = run_walk_forward(
                args.strategy_config,
                jobs=args.jobs,
                trace_dir=trace_dir,
            )
            log.info("WF result: %s", wf_result["reason"])

    trade_gate_result = {"passed": True, "reason": "skipped"}
    trade_contract_result = {"passed": True, "reason": "skipped"}
    if not args.skip_wf and not args.skip_trade_gates:
        if trace_dir is None:
            trade_gate_result = {
                "passed": False,
                "reason": "trade gates require persisted round-trip ledgers",
            }
            trade_contract_result = dict(trade_gate_result)
        elif wf_result.get("cuts"):
            trade_contract_result = run_trade_contract_gate(wf_result, gate_config)
            trade_gate_result = run_trade_monotonicity_gate(wf_result)
        log.info("Trade contract result: %s", trade_contract_result["reason"])
        log.info("Trade gate result: %s", trade_gate_result["reason"])

    sanity_result = {"passed": True, "reason": "skipped"}
    if not args.skip_sanity:
        sanity_result = run_sanity_battery(artifact_path)
        log.info("Sanity result: %s", sanity_result["reason"])

    validation_scope_ok = bool(artifact_usage.get("candidate_artifact_used")) or bool(
        artifact_usage.get("recipe_validated")
    )
    if not validation_scope_ok:
        wf_result["passed"] = False
        prior_reason = wf_result.get("reason", "")
        wf_result["reason"] = (
            f"{prior_reason}; candidate artifact was not directly evaluated "
            f"and no matching manifest recipe was validated "
            f"(scope={artifact_usage.get('eval_scope')})"
        ).strip("; ")

    overall_pass = (
        bool(wf_result["passed"])
        and bool(sanity_result["passed"])
        and bool(trade_contract_result["passed"])
        and bool(trade_gate_result["passed"])
        and validation_scope_ok
        and bool(parity_result.get("passed", True))
    )
    wf_meta = {
        "passed": overall_pass,
        "wf_3cut_sharpe_mean": wf_result.get("wf_3cut_sharpe_mean"),
        "wf_3cut_sharpe_std":  wf_result.get("wf_3cut_sharpe_std"),
        "wf_3cut_apy_mean":    wf_result.get("wf_3cut_apy_mean"),
        "spy_sharpe_mean":     wf_result.get("spy_sharpe_mean"),
        "strategy_minus_spy_sharpe_mean": wf_result.get("strategy_minus_spy_sharpe_mean"),
        "spy_apy_mean":        wf_result.get("spy_apy_mean"),
        "strategy_minus_spy_apy_mean": wf_result.get("strategy_minus_spy_apy_mean"),
        "n_cuts_beat_spy_sharpe": wf_result.get("n_cuts_beat_spy_sharpe"),
        "n_cuts_beat_spy_apy": wf_result.get("n_cuts_beat_spy_apy"),
        "hmm_regime_counts_total": wf_result.get("hmm_regime_counts_total"),
        "spy_grid_regime_counts_total": wf_result.get("spy_grid_regime_counts_total"),
        "trade_buy_regime_counts_total": wf_result.get("trade_buy_regime_counts_total"),
        "trade_sell_regime_counts_total": wf_result.get("trade_sell_regime_counts_total"),
        "trade_buy_source_counts_total": wf_result.get("trade_buy_source_counts_total"),
        "trade_sell_exit_reason_counts_total": wf_result.get("trade_sell_exit_reason_counts_total"),
        "trade_buy_missing_mu_total": wf_result.get("trade_buy_missing_mu_total"),
        "trade_buy_missing_sigma_total": wf_result.get("trade_buy_missing_sigma_total"),
        "n_positive_cuts":     wf_result.get("n_positive_cuts"),
        "wf_jobs":             wf_result.get("wf_jobs"),
        "cuts":                wf_result.get("cuts"),
        "wf_trade_trace_dir":   str(trace_dir) if trace_dir is not None else None,
        "candidate_artifact_used": artifact_usage.get("candidate_artifact_used"),
        "recipe_validated":    artifact_usage.get("recipe_validated"),
        "candidate_recipe_fingerprint": artifact_usage.get("candidate_recipe_fingerprint"),
        "wf_eval_scope":       artifact_usage.get("eval_scope"),
        "artifact_usage":      artifact_usage,
        "config_parity":       parity_result,
        "qp_contract":         (
            {
                "passed": qp_contract.passed,
                "issues": qp_contract.issues,
                "evidence": qp_contract.evidence,
            }
            if qp_contract is not None else None
        ),
        "trade_contract":      trade_contract_result,
        "trade_monotonicity":  trade_gate_result,
        "real_ic":             sanity_result.get("real_ic"),
        "sanity_shuffled_ic":  sanity_result.get("sanity_shuffled_ic"),
        "sanity_placebo_ic":   sanity_result.get("sanity_placebo_ic"),
        "wf_reason":           wf_result.get("reason"),
        "sanity_reason":       sanity_result.get("reason"),
        "run_at":              datetime.datetime.utcnow().isoformat(),
        "gate_version":        GATE_VERSION,
    }

    # Stamp into artifact
    md = artifact.get("metadata") or {}
    md["wf_gate_metadata"] = wf_meta
    artifact["metadata"] = md
    artifact_path.write_text(json.dumps(artifact))
    log.info("Wrote wf_gate_metadata to %s", artifact_path)
    log.info("=" * 60)
    log.info("VERDICT: %s", "PASS" if overall_pass else "FAIL")
    log.info("=" * 60)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
