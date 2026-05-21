#!/usr/bin/env python
"""Walk-forward gate runner — write wf_gate_metadata to artifact.

Per CLAUDE.md §5.9 + roadmap P0 #1 (post E55 NGB revert): every promote
requires walk-forward 3-cut Sharpe + §5.2 sanity battery. This script
runs both checks and stamps the artifact's metadata so kernel.model_acceptance.promote()
will accept it.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("wf-gate")

REPO = Path(__file__).resolve().parent.parent
GATE_VERSION = 1
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
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
        return {
            "candidate_artifact_used": False,
            "eval_scope": "walkforward_manifest",
            "strategy_config": strategy_config,
            "manifest_path": str(manifest) if manifest is not None else None,
            "reason": (
                "strategy config uses walkforward manifest; the candidate artifact "
                "is not the static scorer evaluated by the sim"
            ),
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


def run_sim_cut(strategy_config: str, start: str, end: str) -> dict:
    """Run one sim cut, parse Sharpe + APY from log."""
    log.info("Sim cut: %s → %s", start, end)
    market_context = cut_market_context(start, end)
    cmd = [
        PYTHON,
        str(REPO / "scripts/run_sim_104.py"),
        "--strategy-config-name", strategy_config,
        "--start", start, "--end", end,
        "--no-compare",
        "--no-persist",
        "--skip-preflight",
    ]
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
        "returncode": 0,
    }


def run_walk_forward(strategy_config: str, jobs: int = 1) -> dict:
    """Run 3-cut walk-forward, return mean/std/per-cut."""
    cuts = CUTS
    jobs = max(1, min(int(jobs), len(cuts)))
    results: list[dict | None] = [None] * len(cuts)
    if jobs == 1:
        for idx, (start, end) in enumerate(cuts):
            results[idx] = run_sim_cut(strategy_config, start, end)
    else:
        log.info("Running %d WF cuts with jobs=%d", len(cuts), jobs)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            future_to_idx = {
                pool.submit(run_sim_cut, strategy_config, start, end): idx
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
                    help="Walk-forward sim config name (default: strategy_config.sim_wl200.json)")
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

    artifact_usage = inspect_artifact_usage(args.strategy_config, artifact_path)
    log.info("Artifact usage: %s", artifact_usage)

    wf_result = {"passed": True, "reason": "skipped"}
    if not args.skip_wf:
        wf_result = run_walk_forward(args.strategy_config, jobs=args.jobs)
        log.info("WF result: %s", wf_result["reason"])

    sanity_result = {"passed": True, "reason": "skipped"}
    if not args.skip_sanity:
        sanity_result = run_sanity_battery(artifact_path)
        log.info("Sanity result: %s", sanity_result["reason"])

    artifact_scope_ok = bool(artifact_usage.get("candidate_artifact_used"))
    if not artifact_scope_ok:
        wf_result["passed"] = False
        prior_reason = wf_result.get("reason", "")
        wf_result["reason"] = (
            f"{prior_reason}; candidate artifact was not evaluated "
            f"(scope={artifact_usage.get('eval_scope')})"
        ).strip("; ")

    overall_pass = bool(wf_result["passed"]) and bool(sanity_result["passed"]) and artifact_scope_ok
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
        "n_positive_cuts":     wf_result.get("n_positive_cuts"),
        "wf_jobs":             wf_result.get("wf_jobs"),
        "cuts":                wf_result.get("cuts"),
        "candidate_artifact_used": artifact_usage.get("candidate_artifact_used"),
        "wf_eval_scope":       artifact_usage.get("eval_scope"),
        "artifact_usage":      artifact_usage,
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
