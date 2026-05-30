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
  - Pass: absolute Sharpe floor AND SPY-relative benchmark floor
  - Fail: positive absolute Sharpe that still lags SPY is benchmark-blind

§5.2 sanity criteria (default):
  - shuffled-label IC: |IC| < 0.005 (model on shuffled labels should be ~0)
  - time-shift placebo IC: ratio < 0.5 × aligned real IC (placebo should not
    capture the same signal on the same evaluable rows)

References:
- Lopez de Prado AFML §7 + §11 (walk-forward + cross-validation in finance)
- Bailey-Lopez de Prado 2014 "Pseudo-Mathematics and Financial Charlatanism"
- CLAUDE.md §5.2 sanity battery, §5.9 walk-forward mandate
"""
from __future__ import annotations
import argparse
import ast
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


def _required_validation_skip_reasons(args) -> list[str]:
    """Return skipped gates that make metadata diagnostic-only.

    Emergency skip flags are useful for parser/debug runs, but a skipped WF,
    sanity battery, trade gate, config parity check, or trace cannot be
    promoted as acceptance evidence.
    """
    reasons: list[str] = []
    if bool(getattr(args, "skip_wf", False)):
        reasons.append("walk_forward_skipped")
    if bool(getattr(args, "skip_sanity", False)):
        reasons.append("sanity_skipped")
    if bool(getattr(args, "skip_trade_gates", False)):
        reasons.append("trade_gates_skipped")
    if bool(getattr(args, "allow_pass_open_trade_monotonicity", False)):
        reasons.append("trade_monotonicity_pass_open_allowed")
    if bool(getattr(args, "skip_config_parity", False)):
        reasons.append("config_parity_skipped")
    if bool(getattr(args, "no_trade_trace", False)):
        reasons.append("trade_trace_disabled")
    return reasons


def _compute_overall_pass(
    *,
    wf_result: dict,
    sanity_result: dict,
    trade_contract_result: dict,
    trade_gate_result: dict,
    alpha_economics_result: dict,
    validation_scope_ok: bool,
    parity_result: dict,
    skipped_required_gates: list[str],
) -> bool:
    if skipped_required_gates:
        return False
    return (
        bool(wf_result["passed"])
        and _sanity_result_passed(sanity_result)
        and bool(trade_contract_result["passed"])
        and bool(trade_gate_result["passed"])
        and bool(alpha_economics_result["passed"])
        and validation_scope_ok
        and bool(parity_result.get("passed", True))
    )


def _sanity_result_passed(sanity_result: dict) -> bool:
    """Fail closed unless global and regime-level sanity evidence both pass."""
    if not bool(sanity_result.get("passed")):
        return False
    regime_ic = sanity_result.get("sanity_regime_ic")
    if not isinstance(regime_ic, dict):
        return False
    return bool(regime_ic.get("passed"))


def _placebo_ic_threshold(aligned_real_ic: float) -> float:
    """Maximum acceptable absolute time-shift placebo IC."""
    return max(0.005, 0.5 * abs(float(aligned_real_ic)))


def _placebo_ic_requirement_text(aligned_real_ic: float) -> str:
    threshold = _placebo_ic_threshold(aligned_real_ic)
    return (
        f"threshold={threshold:+.4f} "
        f"(0.5×|aligned_real_ic|, aligned_real_ic={aligned_real_ic:+.4f})"
    )


def _sanity_model_label_col(artifact: dict) -> str:
    """Return the label column the scorer was actually trained to rank."""
    for key in ("label_col", "label"):
        value = artifact.get(key)
        if value:
            return str(value)
    lookahead = artifact.get("lookahead_days")
    try:
        horizon = int(lookahead)
    except (TypeError, ValueError):
        horizon = 60
    if horizon <= 0:
        horizon = 60
    return f"fwd_{horizon}d_excess"


def _resolve_strategy_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_absolute() else STRATEGY_DIR / p


def _resolve_trace_dir_arg(raw: str) -> Path:
    """Resolve --trace-dir without double-prefixing repo-relative paths.

    Most operators pass strategy-relative paths such as
    ``artifacts/diagnostics/...``. Some automation passes repo-relative paths
    such as ``backtesting/renquant_104/artifacts/...``. Treat the latter as
    repo-relative so persisted WF evidence lands where the caller asked.
    """
    p = Path(raw)
    if p.is_absolute():
        return p
    if len(p.parts) >= 2 and p.parts[:2] == ("backtesting", "renquant_104"):
        return REPO / p
    return STRATEGY_DIR / p


_EXECUTION_ONLY_PARAM_KEYS = {
    "nthread",
    "n_jobs",
    "num_threads",
    "total_steps",
    "warmup_steps",
    "verbosity",
    "verbose",
    "silent",
}


def _semantic_params(params: dict) -> dict:
    """Return learner params that define the statistical recipe.

    Thread counts and logging verbosity are execution controls. Treating them
    as recipe fields makes a hardware upgrade invalidate otherwise comparable
    walk-forward manifests, while leaving learning parameters such as eta,
    max_depth, objective, seed, and tree_method in the fingerprint.
    """
    if not isinstance(params, dict):
        return {}
    return {
        k: v for k, v in params.items()
        if str(k) not in _EXECUTION_ONLY_PARAM_KEYS
    }


def _artifact_sidecar_path(path: Path) -> Path | None:
    """Return optional metadata sidecar for non-JSON sequence artifacts."""
    for candidate in (
        path.with_name(path.name + ".metadata.json"),
        path.with_name(path.stem + "_metadata.json"),
        path.with_name(path.stem + "_summary.json"),
    ):
        if candidate.exists():
            return candidate
    return None


def _patchtst_params_from_contract(payload: dict) -> dict:
    contract = payload.get("training_contract") or {}
    hparams = contract.get("hyperparameters") or {}
    keep = (
        "seq_len", "patch_length", "d_model", "n_heads", "n_layers",
        "lr", "weight_decay", "lr_scheduler", "warmup_ratio",
        "nll_loss_weight", "ranking_margin", "distributional_head",
        "film_regime_cond", "cross_stock_attn", "embargo_days",
    )
    return {k: hparams.get(k) for k in keep if k in hparams}


def _load_artifact_payload(path: Path) -> dict:
    """Load JSON artifact or metadata sidecar for a sequence checkpoint."""
    if path.suffix == ".json":
        return json.loads(path.read_text())
    sidecar = _artifact_sidecar_path(path)
    payload = json.loads(sidecar.read_text()) if sidecar else {}
    if path.suffix == ".pt" and (
        "feature_cols" not in payload or "lookahead_days" not in payload
    ):
        try:
            import torch  # noqa: PLC0415
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt = {}
        if isinstance(ckpt, dict):
            payload.setdefault("feature_cols", ckpt.get("feature_cols"))
            payload.setdefault("label_col", ckpt.get("label_col"))
            payload.setdefault("lookahead_days", ckpt.get("lookahead_days"))
            payload.setdefault("training_contract", ckpt.get("training_contract"))
    payload.setdefault("kind", payload.get("arch") or "hf_patchtst")
    payload.setdefault("params", _patchtst_params_from_contract(payload))
    return payload


def _write_artifact_payload(path: Path, payload: dict) -> Path:
    """Persist gate metadata without corrupting binary sequence checkpoints."""
    out_path = path
    if path.suffix != ".json":
        sidecar = _artifact_sidecar_path(path)
        out_path = sidecar or path.with_name(path.name + ".metadata.json")
    out_path.write_text(json.dumps(payload))
    return out_path


def _feature_source_contract_keys(artifact: dict) -> list[str]:
    """Structural projection of the feature-source contract.

    2026-05-27: the recipe fingerprint previously hashed
    ``feature_source_contract`` whole — but its VALUES are human-readable prose
    ("apply all feature_means/stds before scoring …"). Editing that docstring
    (which the subrepo refactor did) changed the recipe fingerprint while the
    actual preprocessing was unchanged, so the weekly gate kept rejecting
    otherwise-comparable candidates with "manifest recipe mismatch". The
    behavioral recipe is fully captured by ``feature_norm_kind`` + the SOURCE
    SPACES the contract declares (its keys, e.g. raw/panel), not the prose. Hash
    only the sorted keys so wording changes never break comparability.
    """
    contract = artifact.get("feature_source_contract")
    if not isinstance(contract, dict):
        return []
    return sorted(str(k) for k in contract.keys())


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
        "feature_norm_kind": list(artifact.get("feature_norm_kind") or []),
        # Structural keys only — NOT the prose values (see helper docstring).
        "feature_source_contract_keys": _feature_source_contract_keys(artifact),
        "label_col": artifact.get("label_col"),
        "lookahead_days": int(artifact.get("lookahead_days") or 0),
        "params": _semantic_params(artifact.get("params") or {}),
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

    candidate = _load_artifact_payload(artifact_path)
    candidate_fp = _recipe_fingerprint(candidate)
    candidate_recipe = _recipe_projection(candidate)
    samples = rows
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
            sample = _load_artifact_payload(p)
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
        "manifest_rows_checked": int(len(rows)),
        "manifest_sample_reports": sample_reports,
        "reason": (
            "manifest artifacts match candidate recipe"
            if all_match else
            "manifest artifacts do not match candidate recipe"
        ),
    }


def _matching_manifest_for_recipe(
    *,
    artifact_path: Path,
    preferred_manifest: Path | None,
    search_dir: Path | None = None,
) -> tuple[Path | None, dict]:
    """Return a WF manifest whose artifacts match the candidate recipe.

    Weekly promotion is only valid when historical point-in-time artifacts use
    the same statistical recipe as the candidate. If the operator-supplied base
    WF config names a manifest, that manifest is the evidence contract: validate
    it and fail closed on mismatch instead of silently substituting a broader
    manifest. Auto-discovery is allowed only when no preferred manifest was
    supplied.
    """
    checked: list[tuple[Path, dict]] = []
    seen: set[Path] = set()

    def add(path: Path | None) -> None:
        if path is None:
            return
        p = path if path.is_absolute() else STRATEGY_DIR / path
        try:
            key = p.resolve()
        except Exception:  # noqa: BLE001
            key = p
        if key in seen:
            return
        seen.add(key)
        checked.append((p, _manifest_recipe_usage(p, artifact_path)))

    add(preferred_manifest)
    if preferred_manifest is not None and checked:
        p, usage = checked[0]
        usage = dict(usage)
        usage["checked_manifest_count"] = 1
        usage["manifest_selection_policy"] = "preferred_manifest_required"
        return p, usage

    root = search_dir or (STRATEGY_DIR / "artifacts" / "sim")
    if root.exists():
        for candidate in sorted(root.glob("walkforward_manifest*.json")):
            add(candidate)

    matches = [
        (p, usage) for p, usage in checked
        if bool(usage.get("recipe_validated"))
    ]
    if matches:
        matches.sort(
            key=lambda item: (
                int(item[1].get("manifest_rows_checked") or 0),
                str(item[0]),
            ),
            reverse=True,
        )
        return matches[0]

    if checked:
        p, usage = checked[0]
        usage = dict(usage)
        usage["checked_manifest_count"] = len(checked)
        return p, usage
    return None, {
        "recipe_validated": False,
        "reason": "no walkforward manifests found",
        "checked_manifest_count": 0,
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


def _value_counts(rows: list[dict], key: str) -> dict[str, int]:
    counts = Counter(
        str(row.get(key))
        for row in rows
        if row.get(key) not in (None, "")
    )
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def _benchmark_by_dominant_regime(rows: list[dict]) -> dict[str, dict]:
    """Summarize WF cuts by dominant benchmark regime before pooled metrics.

    CLAUDE.md's prime directive is regime-conditional evaluation. With only
    three WF cuts this is a coarse cut-level lens, not a promotion statistic,
    but it prevents a pooled Sharpe from hiding "positive but benchmark-losing"
    behavior in the regime where the trades actually occurred.
    """
    import statistics as _s

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        regime = (
            row.get("dominant_spy_grid_regime")
            or row.get("dominant_hmm_regime")
            or "UNKNOWN"
        )
        grouped.setdefault(str(regime), []).append(row)

    out: dict[str, dict] = {}
    for regime, group in grouped.items():
        sharpes = [_finite_number(r.get("sharpe")) for r in group]
        apys = [_finite_number(r.get("apy")) for r in group]
        spy_sharpes = [
            _finite_number((r.get("market_context") or {}).get("spy_sharpe"))
            for r in group
        ]
        spy_apys = [
            _finite_number((r.get("market_context") or {}).get("spy_apy"))
            for r in group
        ]
        sharpe_deltas = [
            _finite_number(r.get("sharpe_vs_spy"))
            if _finite_number(r.get("sharpe_vs_spy")) is not None
            else (
                s - spy_s
                if s is not None and spy_s is not None
                else None
            )
            for r, s, spy_s in zip(group, sharpes, spy_sharpes)
        ]
        apy_deltas = [
            _finite_number(r.get("apy_vs_spy"))
            if _finite_number(r.get("apy_vs_spy")) is not None
            else (
                a - spy_a
                if a is not None and spy_a is not None
                else None
            )
            for r, a, spy_a in zip(group, apys, spy_apys)
        ]

        def mean(vals: list[float | None]) -> float:
            finite = [float(v) for v in vals if v is not None]
            return float(_s.mean(finite)) if finite else float("nan")

        out[regime] = {
            "n_cuts": int(len(group)),
            "mean_sharpe": mean(sharpes),
            "mean_spy_sharpe": mean(spy_sharpes),
            "mean_sharpe_vs_spy": mean(sharpe_deltas),
            "mean_apy": mean(apys),
            "mean_spy_apy": mean(spy_apys),
            "mean_apy_vs_spy": mean(apy_deltas),
            "n_beat_spy_sharpe": int(sum(
                1 for d in sharpe_deltas if d is not None and d > 0
            )),
            "n_beat_spy_apy": int(sum(
                1 for d in apy_deltas if d is not None and d > 0
            )),
        }
    return out


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


def _sim_metrics_from_trace(traces: dict[str, str]) -> dict:
    """Load exact sim metrics from the equity JSON trace.

    ``run_sim_104.py`` prints rounded human-readable metrics. Acceptance must
    use the machine-readable trace when present, and it should prefer the
    annual-net tax reporting path while keeping event-level cash-stress
    metrics for audit.
    """
    p_raw = traces.get("equity_json")
    if not p_raw:
        return {}
    p = Path(p_raw)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"trace_metric_error": f"failed to parse {p}: {exc}"}

    event_sharpe = _finite_number(payload.get("event_level_sharpe"))
    event_apy = _finite_number(payload.get("event_level_apy"))
    if event_sharpe is None:
        event_sharpe = _finite_number(payload.get("sharpe"))
    if event_apy is None:
        event_apy = _finite_number(payload.get("apy"))

    annual_sharpe = _finite_number(payload.get("annual_net_sharpe"))
    annual_apy = _finite_number(payload.get("annual_net_apy"))
    use_annual = annual_sharpe is not None and annual_apy is not None
    return {
        "sharpe": annual_sharpe if use_annual else event_sharpe,
        "apy": annual_apy if use_annual else event_apy,
        "event_level_sharpe": event_sharpe,
        "event_level_apy": event_apy,
        "annual_net_sharpe": annual_sharpe,
        "annual_net_apy": annual_apy,
        "event_level_tax_debited": _finite_number(
            payload.get("event_level_tax_debited")
        ),
        "annual_net_tax_estimate": _finite_number(
            payload.get("annual_net_tax_estimate")
        ),
        "tax_overstatement_vs_annual_net": _finite_number(
            payload.get("tax_overstatement_vs_annual_net")
        ),
        "performance_tax_basis": "annual_net" if use_annual else "event_level",
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
    # Prefer exact machine-readable trace metrics. Fall back to the rounded
    # console summary only for legacy runs without --equity-json.
    trace_metrics = _sim_metrics_from_trace(traces)
    sharpe_m = re.search(r"Sharpe=([+\-\d.]+)", out)
    apy_m = re.search(r"APY:\s+([+\-\d.]+)%", out)
    parsed_sharpe = float(sharpe_m.group(1)) if sharpe_m else float("nan")
    parsed_apy = float(apy_m.group(1)) / 100 if apy_m else float("nan")
    sharpe = (
        float(trace_metrics["sharpe"])
        if _finite_number(trace_metrics.get("sharpe")) is not None
        else parsed_sharpe
    )
    apy = (
        float(trace_metrics["apy"])
        if _finite_number(trace_metrics.get("apy")) is not None
        else parsed_apy
    )
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
        "event_level_sharpe": trace_metrics.get("event_level_sharpe"),
        "event_level_apy": trace_metrics.get("event_level_apy"),
        "annual_net_sharpe": trace_metrics.get("annual_net_sharpe"),
        "annual_net_apy": trace_metrics.get("annual_net_apy"),
        "event_level_tax_debited": trace_metrics.get("event_level_tax_debited"),
        "annual_net_tax_estimate": trace_metrics.get("annual_net_tax_estimate"),
        "tax_overstatement_vs_annual_net": trace_metrics.get(
            "tax_overstatement_vs_annual_net"
        ),
        "performance_tax_basis": trace_metrics.get(
            "performance_tax_basis", "console_parse"
        ),
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
        total_buys = _sum_trade_summary(results, "n_buys")
        total_sells = _sum_trade_summary(results, "n_sells")
        if total_buys == 0 and total_sells == 0:
            import statistics as _s
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
            mean_apy = _s.mean(apys) if apys else 0.0
            mean_spy_apy = _s.mean(spy_apys) if spy_apys else float("nan")
            return {
                "passed": False,
                "wf_3cut_sharpe_mean": float("nan"),
                "wf_3cut_sharpe_std": float("nan"),
                "wf_3cut_apy_mean": float(mean_apy),
                "spy_sharpe_mean": (
                    float(_s.mean(spy_sharpes)) if spy_sharpes else float("nan")
                ),
                "strategy_minus_spy_sharpe_mean": float("nan"),
                "spy_apy_mean": float(mean_spy_apy),
                "strategy_minus_spy_apy_mean": (
                    float(mean_apy - mean_spy_apy)
                    if math.isfinite(mean_spy_apy) else float("nan")
                ),
                "n_cuts_beat_spy_sharpe": 0,
                "n_cuts_beat_spy_apy": 0,
                "benchmark_by_dominant_regime": _benchmark_by_dominant_regime(results),
                "regime_benchmark_failures": [],
                "performance_tax_basis_counts": _value_counts(
                    results, "performance_tax_basis",
                ),
                "hmm_regime_counts_total": _merge_counts(results, "hmm_regime_counts"),
                "spy_grid_regime_counts_total": _merge_counts(results, "spy_grid_regime_counts"),
                "trade_buy_regime_counts_total": {},
                "trade_sell_regime_counts_total": {},
                "trade_buy_source_counts_total": {},
                "trade_sell_exit_reason_counts_total": {},
                "trade_buy_missing_mu_total": 0,
                "trade_buy_missing_sigma_total": 0,
                "n_positive_cuts": 0,
                "wf_jobs": jobs,
                "cuts": results,
                "reason": (
                    "FAIL: zero trades across all WF cuts; decision tree "
                    "admitted no buys, so Sharpe is undefined and SPY "
                    "benchmark cannot be met"
                ),
            }
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
    benchmark_by_regime = _benchmark_by_dominant_regime(results)
    regime_benchmark_failures = [
        regime
        for regime, stats in benchmark_by_regime.items()
        if (
            math.isfinite(float(stats.get("mean_sharpe_vs_spy", float("nan"))))
            and float(stats["mean_sharpe_vs_spy"]) < 0
        )
        or (
            math.isfinite(float(stats.get("mean_apy_vs_spy", float("nan"))))
            and float(stats["mean_apy_vs_spy"]) < 0
        )
    ]
    has_spy_sharpe = len(spy_sharpes) == len(results)
    has_spy_apy = len(spy_apys) == len(results)
    missing_benchmark_metrics: list[str] = []
    if not has_spy_sharpe:
        missing_benchmark_metrics.append("spy_sharpe")
    if not has_spy_apy:
        missing_benchmark_metrics.append("spy_apy")
    absolute_ok = mean_sharpe >= 0.40 and n_pos >= 2
    benchmark_ok = (
        has_spy_sharpe
        and has_spy_apy
        and mean_sharpe_vs_spy >= 0
        and n_beat_spy_sharpe >= 2
        and mean_apy_vs_spy >= 0
        and n_beat_spy_apy >= 2
    )
    regime_ok = not regime_benchmark_failures
    pass_sharpe = bool(absolute_ok and benchmark_ok and regime_ok)
    benchmark_suffix = (
        f"; SPY mean Sharpe {mean_spy_sharpe:+.3f}, "
        f"ΔSharpe {mean_sharpe_vs_spy:+.3f}, "
        f"beat SPY Sharpe {n_beat_spy_sharpe}/3, "
        f"beat SPY APY {n_beat_spy_apy}/3"
        if math.isfinite(mean_spy_sharpe) else ""
    )
    regime_suffix = (
        f"; benchmark-lag regimes={regime_benchmark_failures}"
        if regime_benchmark_failures else ""
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
        "benchmark_data_missing": bool(missing_benchmark_metrics),
        "missing_benchmark_metrics": missing_benchmark_metrics,
        "benchmark_by_dominant_regime": benchmark_by_regime,
        "regime_benchmark_failures": regime_benchmark_failures,
        "performance_tax_basis_counts": _value_counts(results, "performance_tax_basis"),
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
            f"PASS: absolute Sharpe floor met and SPY benchmark met"
            f"{benchmark_suffix}{regime_suffix}"
            if pass_sharpe else
            f"FAIL: absolute_ok={absolute_ok}, benchmark_ok={benchmark_ok}, "
            f"regime_ok={regime_ok}; mean Sharpe {mean_sharpe:+.3f}, "
            f"{n_pos}/3 cuts > 0"
            + (
                f"; benchmark_data_missing={missing_benchmark_metrics}"
                if missing_benchmark_metrics else ""
            )
            + benchmark_suffix
            + regime_suffix
        ),
    }


def run_trade_monotonicity_gate(
    wf_result: dict,
    *,
    score_cols: list[str] | tuple[str, ...] | None = None,
    min_n_per_regime: int = 30,
    min_spearman: float = 0.02,
    min_top_bottom_spread: float = 0.0,
    small_n_inversion_min_n: int = 10,
    allow_pass_open: bool = False,
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
    df = pd.concat(frames, ignore_index=True)
    cols = _normalize_trade_monotonicity_score_cols(score_cols)
    reports: dict[str, dict] = {}
    failed: list[str] = []
    primary_report = None
    for col in cols:
        report = evaluate_trade_monotonicity(
            df,
            score_col=col,
            min_n_per_regime=min_n_per_regime,
            min_spearman=min_spearman,
            min_top_bottom_spread=min_top_bottom_spread,
            small_n_inversion_min_n=small_n_inversion_min_n,
            allow_pass_open=allow_pass_open,
        )
        payload = {
            "passed": bool(report.passed),
            "reason": report.reason,
            "pooled": report.pooled,
            "regimes": report.regimes,
        }
        reports[col] = payload
        if primary_report is None:
            primary_report = report
        if not report.passed:
            failed.append(col)

    assert primary_report is not None
    passed = not failed
    reason = (
        f"score monotonicity passed for {', '.join(cols)}"
        if passed
        else "score monotonicity failed for "
        + ", ".join(f"{c}: {reports[c]['reason']}" for c in failed)
    )
    return {
        "passed": bool(passed),
        "reason": reason,
        "pooled": primary_report.pooled,
        "regimes": primary_report.regimes,
        "score_cols": cols,
        "score_reports": reports,
        "min_n_per_regime": int(min_n_per_regime),
        "min_spearman": float(min_spearman),
        "min_top_bottom_spread": float(min_top_bottom_spread),
        "small_n_inversion_min_n": int(small_n_inversion_min_n),
        "allow_pass_open": bool(allow_pass_open),
    }


def _normalize_trade_monotonicity_score_cols(
    score_cols: list[str] | tuple[str, ...] | None,
) -> list[str]:
    raw = list(score_cols or ["entry_rank_score"])
    out: list[str] = []
    for col in raw:
        c = str(col or "").strip()
        if c and c not in out:
            out.append(c)
    return out or ["entry_rank_score"]


def _trade_monotonicity_score_cols_from_config(config: dict) -> list[str]:
    """Scores whose economic ordering must survive into trade P/L.

    Rank gates decide buy eligibility, while QP consumes μ-like expected
    returns. WF acceptance must therefore validate both surfaces whenever QP
    is the sizing/rebalance layer.
    """
    cols = ["entry_rank_score"]
    joint = ((config.get("rotation") or {}).get("joint_actions") or {})
    qp_enabled = bool(joint.get("enabled")) and str(joint.get("solver", "")).lower() == "qp"
    strict_qp = str(joint.get("qp_mu_contract", "strict")).lower() in {
        "strict", "hard", "error", "enforce",
    }
    if qp_enabled and strict_qp:
        cols.extend(["entry_mu", "entry_expected_return"])
    return _normalize_trade_monotonicity_score_cols(cols)


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
    require_er = bool(qp_enabled and strict_qp)
    require_sigma = bool(kelly.get("enabled") or panel.get("ngboost", {}).get("enabled"))
    report = evaluate_trade_contract(
        pd.concat(frames, ignore_index=True),
        require_entry_mu=require_mu,
        require_entry_sigma=require_sigma,
        require_entry_expected_return=require_er,
        require_entry_horizon=require_er,
        require_exit_regime=True,
        require_exit_thresholds=True,
    )
    return {
        "passed": bool(report.passed),
        "reason": report.reason,
        "evidence": report.evidence,
        "require_entry_mu": require_mu,
        "require_entry_sigma": require_sigma,
        "require_entry_expected_return": require_er,
        "require_entry_horizon": require_er,
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
        try:
            frame = pd.read_csv(p)
        except pd.errors.EmptyDataError:
            continue
        if not frame.empty:
            frame = _recover_round_trip_entry_scores(frame)
            frame["_wf_cut"] = f"{cut.get('start')}_to_{cut.get('end')}"
            frames.append(frame)
    return frames, missing


def _recover_round_trip_entry_scores(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover score columns from entry_score_snapshot for legacy traces."""
    if "entry_score_snapshot" not in frame.columns:
        return frame
    mappings = {
        "rank_score": "entry_rank_score",
        "panel_score": "entry_panel_score",
        "rs_score": "entry_rs_score",
        "mu": "entry_mu",
        "mu_horizon_days": "entry_mu_horizon_days",
        "sigma": "entry_sigma",
        "kelly_target_pct": "entry_kelly_target_pct",
        "expected_return": "entry_expected_return",
        "expected_return_horizon_days": "entry_expected_return_horizon_days",
    }
    out = frame.copy()
    snapshots = out["entry_score_snapshot"].map(_parse_score_snapshot)
    for snap_key, col in mappings.items():
        recovered = snapshots.map(
            lambda snap, key=snap_key: snap.get(key) if isinstance(snap, dict) else None
        )
        if col not in out.columns:
            out[col] = recovered
            continue
        mask = out[col].isna()
        if bool(mask.any()):
            out.loc[mask, col] = recovered[mask]
    return out


def _parse_score_snapshot(value) -> dict | None:
    if isinstance(value, dict):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


def _benchmark_sleeve_entry_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for col in (
        "entry_order_type",
        "order_type",
        "entry_source_job",
        "source_job",
        "entry_source_task",
        "source_task",
        "entry_reason",
        "reason",
    ):
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.lower()
        if col in {"entry_source_job", "source_job"}:
            mask = mask | s.eq("benchmarksleevejob")
        mask = mask | s.str.contains("benchmark_sleeve", regex=False, na=False)
        mask = mask | s.str.contains("benchmarksleevetask", regex=False, na=False)
    return mask


def _load_benchmark_close(benchmark_ticker: str) -> pd.Series | None:
    path = REPO / "data" / "ohlcv" / benchmark_ticker.upper() / "1d.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path).sort_index()
    if "close" not in df.columns:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index).normalize()
    return close.sort_index()


def _close_on_or_before(close: pd.Series, value: object) -> float | None:
    try:
        ts = pd.Timestamp(value).normalize()
    except Exception:
        return None
    idx = close.index.searchsorted(ts, side="right") - 1
    if idx < 0:
        return None
    out = float(close.iloc[idx])
    return out if math.isfinite(out) and out > 0 else None


def run_alpha_economics_gate(
    wf_result: dict,
    *,
    benchmark_ticker: str = "SPY",
    min_positive_cuts: int = 2,
) -> dict:
    """Require benchmark-sleeve runs to prove active alpha economics.

    Full-portfolio WF can include a SPY/core sleeve. That sleeve is useful for
    live beta exposure, but model acceptance must still prove that non-sleeve
    alpha trades add value versus simply putting the same entry capital into
    the benchmark over the same holding window.
    """
    frames, missing = _load_round_trip_frames(wf_result)
    if missing:
        return {
            "passed": False,
            "reason": "missing round-trip ledger(s): " + "; ".join(missing[:5]),
            "missing": missing,
        }
    if not frames:
        return {"passed": False, "reason": "no round-trip ledgers found"}
    df = pd.concat(frames, ignore_index=True)
    status = (
        df["status"].astype(str).str.lower()
        if "status" in df.columns else
        pd.Series(["closed"] * len(df), index=df.index)
    )
    closed = df[status.eq("closed")].copy()
    sleeve_mask = _benchmark_sleeve_entry_mask(closed)
    n_sleeve = int(sleeve_mask.sum())
    if n_sleeve == 0:
        return {
            "passed": True,
            "reason": "no benchmark sleeve closed trades; full WF metrics are alpha-owned",
            "evidence": {"n_benchmark_sleeve_closed": 0},
        }
    alpha = closed[~sleeve_mask].copy()
    if alpha.empty:
        return {
            "passed": False,
            "reason": "benchmark sleeve present but no closed alpha trades",
            "evidence": {"n_benchmark_sleeve_closed": n_sleeve, "n_alpha_closed": 0},
        }
    close = _load_benchmark_close(benchmark_ticker)
    if close is None or close.empty:
        return {
            "passed": False,
            "reason": f"benchmark close data missing for {benchmark_ticker}",
            "evidence": {"benchmark": benchmark_ticker},
        }

    active_by_cut: dict[str, float] = {}
    comparable = 0
    skipped = 0
    for _, row in alpha.iterrows():
        entry_px = _finite_number(row.get("entry_price"))
        exit_px = _finite_number(row.get("exit_price"))
        shares = _finite_number(row.get("shares"))
        net = _finite_number(row.get("net_pnl_after_tax"))
        if entry_px is None or exit_px is None or shares is None or net is None:
            skipped += 1
            continue
        b0 = _close_on_or_before(close, row.get("entry_date"))
        b1 = _close_on_or_before(close, row.get("exit_date"))
        if b0 is None or b1 is None:
            skipped += 1
            continue
        entry_capital = abs(float(shares) * float(entry_px))
        benchmark_pnl = entry_capital * (b1 / b0 - 1.0)
        active = float(net) - benchmark_pnl
        cut = str(row.get("_wf_cut") or "UNKNOWN")
        active_by_cut[cut] = active_by_cut.get(cut, 0.0) + active
        comparable += 1

    if comparable == 0:
        return {
            "passed": False,
            "reason": "benchmark sleeve present but alpha active-return comparison unavailable",
            "evidence": {
                "n_benchmark_sleeve_closed": n_sleeve,
                "n_alpha_closed": int(len(alpha)),
                "skipped_alpha_rows": skipped,
            },
        }
    total_active = float(sum(active_by_cut.values()))
    positive_cuts = int(sum(1 for v in active_by_cut.values() if v > 0))
    passed = bool(total_active > 0.0 and positive_cuts >= int(min_positive_cuts))
    return {
        "passed": passed,
        "reason": (
            "alpha active economics passed"
            if passed else
            "alpha active economics failed versus same-capital benchmark"
        ),
        "evidence": {
            "benchmark": benchmark_ticker,
            "n_benchmark_sleeve_closed": n_sleeve,
            "n_alpha_closed": int(len(alpha)),
            "n_comparable_alpha_closed": int(comparable),
            "skipped_alpha_rows": int(skipped),
            "active_net_after_tax": total_active,
            "active_net_after_tax_by_cut": active_by_cut,
            "positive_active_cuts": positive_cuts,
            "min_positive_cuts": int(min_positive_cuts),
        },
    }


def _effective_artifact_cutoff(artifact: dict) -> pd.Timestamp | None:
    """Return the data cutoff that makes static-artifact sanity OOS-safe."""
    for key in (
        "effective_train_cutoff_date",
        "train_cutoff_date",
        "training_cutoff",
        "train_cutoff",
        "data_end",
        "cutoff_date",
    ):
        value = artifact.get(key)
        if value:
            try:
                return pd.Timestamp(value)
            except Exception:  # noqa: BLE001
                return None
    return None


def _validate_static_sanity_oos_contract(
    artifact: dict,
    eval_start: pd.Timestamp,
) -> dict:
    """Fail closed unless a static artifact's eval window is after its labels."""
    cutoff = _effective_artifact_cutoff(artifact)
    if cutoff is None:
        return {
            "passed": False,
            "reason": (
                "static sanity missing effective training cutoff; trained_date "
                "is wall-clock metadata and cannot prove OOS label separation"
            ),
        }
    lookahead = int(artifact.get("lookahead_days") or 0)
    safe_last_label = cutoff + pd.offsets.BDay(max(0, lookahead))
    if safe_last_label >= pd.Timestamp(eval_start):
        return {
            "passed": False,
            "reason": (
                f"static sanity cutoff + lookahead not before eval_start: "
                f"{cutoff.date()} + {lookahead}BDay = "
                f"{safe_last_label.date()} >= {pd.Timestamp(eval_start).date()}"
            ),
            "cutoff": cutoff.date().isoformat(),
            "lookahead_days": lookahead,
            "safe_last_label_date": safe_last_label.date().isoformat(),
        }
    return {
        "passed": True,
        "cutoff": cutoff.date().isoformat(),
        "lookahead_days": lookahead,
        "safe_last_label_date": safe_last_label.date().isoformat(),
    }


def _validate_static_wf_oos_contract(
    artifact: dict,
    cuts: list[tuple[str, str]],
) -> dict:
    """Fail closed when a static artifact cannot be OOS for WF cuts."""
    cutoff = _effective_artifact_cutoff(artifact)
    if cutoff is None:
        return {
            "passed": False,
            "reason": (
                "static WF missing effective training cutoff; use a "
                "recipe-matched walk-forward manifest instead of a static "
                "full-sample artifact"
            ),
        }
    lookahead = int(artifact.get("lookahead_days") or 0)
    safe_last_label = cutoff + pd.offsets.BDay(max(0, lookahead))
    unsafe = [
        start for start, _end in cuts
        if safe_last_label >= pd.Timestamp(start)
    ]
    if unsafe:
        return {
            "passed": False,
            "reason": (
                "static WF cutoff + lookahead overlaps WF cut(s); use a "
                "recipe-matched walk-forward manifest"
            ),
            "cutoff": cutoff.date().isoformat(),
            "lookahead_days": lookahead,
            "safe_last_label_date": safe_last_label.date().isoformat(),
            "unsafe_cut_starts": unsafe,
        }
    return {
        "passed": True,
        "cutoff": cutoff.date().isoformat(),
        "lookahead_days": lookahead,
        "safe_last_label_date": safe_last_label.date().isoformat(),
    }


def _load_sanity_panel(feat_cols: list[str], label: str) -> tuple[pd.DataFrame, dict]:
    """Load label-safe sanity panel, using training feature panel when needed."""
    raw_path = REPO / "data/alpha158_291_fundamental_dataset_rawlabel.parquet"
    if not raw_path.exists():
        raise FileNotFoundError("panel missing — sanity unavailable")
    raw = pd.read_parquet(raw_path)
    raw["date"] = pd.to_datetime(raw["date"])
    if label not in raw.columns:
        raise KeyError(f"sanity label missing: {label}")
    missing = [c for c in feat_cols if c not in raw.columns]
    if not missing:
        return raw, {
            "sanity_feature_panel": str(raw_path),
            "sanity_label_panel": str(raw_path),
            "feature_panel_merge": False,
        }
    feature_path = REPO / "data/transformer_v4_wl200_clean.parquet"
    if not feature_path.exists():
        raise FileNotFoundError(
            f"sanity feature panel missing: {feature_path}; "
            f"needed columns absent from rawlabel panel: {missing[:10]}"
        )
    feature = pd.read_parquet(feature_path)
    feature["date"] = pd.to_datetime(feature["date"])
    still_missing = [c for c in feat_cols if c not in feature.columns]
    if still_missing:
        raise KeyError(
            "sanity feature columns missing from both rawlabel and "
            f"transformer panels: {still_missing[:20]}"
        )
    merged = feature[["ticker", "date", *feat_cols]].merge(
        raw[["ticker", "date", label]],
        on=["ticker", "date"],
        how="left",
    )
    return merged, {
        "sanity_feature_panel": str(feature_path),
        "sanity_label_panel": str(raw_path),
        "feature_panel_merge": True,
        "feature_cols_supplied_by_feature_panel": missing,
    }


def _manifest_uri_to_path(manifest_path: Path, uri: str) -> Path:
    p = Path(str(uri))
    return p if p.is_absolute() else manifest_path.parent / p


def _manifest_entry_safe_last_label_date(entry) -> pd.Timestamp:
    """Return the last label date a WF manifest entry could have seen.

    Keep this in lockstep with WalkForwardModelLoader. Newer manifests stamp
    effective_train_cutoff_date when the scorer already pre-embargoed rows
    before the selection cutoff; using cutoff_date again double-counts the
    lookahead and makes valid point-in-time folds fail sanity.
    """
    feature_cutoff = (
        getattr(entry, "effective_train_cutoff_date", None)
        or getattr(entry, "cutoff_date")
    )
    return pd.Timestamp(feature_cutoff) + pd.offsets.BDay(
        max(0, int(getattr(entry, "lookahead_days", 0) or 0))
    )


def _score_manifest_sanity(
    val: pd.DataFrame,
    feat_cols: list[str],
    manifest_path: Path,
    candidate_artifact_path: Path,
    candidate_artifact: dict,
    panel_history: pd.DataFrame | None = None,
) -> tuple["pd.Series", dict]:
    """Score validation rows with the same point-in-time manifest contract as WF."""
    import numpy as _np  # noqa: PLC0415
    from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
    from kernel.panel_pipeline.feature_transform import transform_feature_frame  # noqa: PLC0415
    from kernel.walk_forward.loader import WalkForwardModelLoader  # noqa: PLC0415

    recipe_usage = _manifest_recipe_usage(manifest_path, candidate_artifact_path)
    if not recipe_usage.get("recipe_validated"):
        raise ValueError(
            "manifest sanity recipe mismatch: "
            f"{recipe_usage.get('reason')}"
        )

    candidate_lookahead = int(candidate_artifact.get("lookahead_days") or 0)
    loader = WalkForwardModelLoader(manifest_path)
    if not loader.has_walkforward_model():
        raise ValueError(f"manifest sanity has no retrain entries: {manifest_path}")

    date_to_artifact: dict[pd.Timestamp, str] = {}
    safe_dates: list[pd.Timestamp] = []
    skipped_pre_manifest_dates: list[pd.Timestamp] = []
    for raw_d in sorted(pd.to_datetime(val["date"].unique())):
        d = pd.Timestamp(raw_d)
        try:
            entry = loader.entry_as_of(d)
        except ValueError:
            skipped_pre_manifest_dates.append(d)
            continue
        if int(entry.lookahead_days) != candidate_lookahead:
            raise ValueError(
                "manifest sanity lookahead mismatch: "
                f"entry cutoff={entry.cutoff_date.date()} "
                f"lookahead={entry.lookahead_days}, "
                f"candidate={candidate_lookahead}"
            )
        safe_last_label = _manifest_entry_safe_last_label_date(entry)
        if safe_last_label >= d:
            raise ValueError(
                "manifest sanity feature cutoff + lookahead violates eval date: "
                f"{(entry.effective_train_cutoff_date or entry.cutoff_date).date()} "
                f"+ {entry.lookahead_days}BDay = {safe_last_label.date()} "
                f">= {d.date()}"
            )
        date_to_artifact[d] = str(_manifest_uri_to_path(manifest_path, entry.artifact_uri))
        safe_dates.append(d)
    if not safe_dates:
        raise ValueError(
            "manifest sanity has no validation dates covered by manifest "
            f"{manifest_path}; skipped_pre_manifest_dates="
            f"{len(skipped_pre_manifest_dates)}"
        )

    scored = val.copy()
    scored = scored[
        pd.to_datetime(scored["date"]).map(lambda d: pd.Timestamp(d) in date_to_artifact)
    ].copy()
    scored["__sanity_artifact_uri"] = [
        date_to_artifact[pd.Timestamp(d)] for d in pd.to_datetime(scored["date"])
    ]
    mu = pd.Series(_np.nan, index=scored.index, dtype=float)
    n_history_artifacts = 0
    for uri, sub in scored.groupby("__sanity_artifact_uri", sort=False):
        uri_path = Path(uri)
        # Dispatch: .pt is a sequence checkpoint (hf_patchtst); JSON is the
        # GBDT PanelScorer. The hf_patchtst loader registers under
        # renquant_common.scorers, so load_scorer dispatches by manifest.kind.
        if uri_path.suffix == ".pt":
            from types import SimpleNamespace  # noqa: PLC0415
            from renquant_common import load_scorer  # noqa: PLC0415
            scorer = load_scorer(SimpleNamespace(
                kind="hf_patchtst",
                local_artifact_path=str(uri_path),
                uri=f"file://{uri_path}",
            ))
        else:
            scorer = PanelScorer.load(uri_path)
        if getattr(scorer, "requires_history", False) is True:
            if panel_history is None:
                raise ValueError(
                    f"manifest sanity history scorer has no panel_history: {uri}"
                )
            n_history_artifacts += 1
            seq_len = int(getattr(scorer, "seq_len", 64))
            hist_source = panel_history.copy()
            hist_source["date"] = pd.to_datetime(hist_source["date"])
            for raw_d, day_sub in sub.groupby("date", sort=True):
                d = pd.Timestamp(raw_d)
                past = hist_source[hist_source["date"] < d]
                recent_dates = sorted(past["date"].unique())[-seq_len:]
                history = past[past["date"].isin(recent_dates)]
                tickers = [str(t) for t in day_sub["ticker"]]
                pred = scorer.score_with_history(history, tickers)
                mu.loc[day_sub.index] = [
                    float(pred.get(t, _np.nan)) for t in tickers
                ]
        else:
            X = transform_feature_frame(
                sub,
                feat_cols,
                getattr(scorer, "metadata", {}) or {},
                source_space="panel",
            )
            pred = scorer.score(X)
            mu.loc[sub.index] = _np.asarray(getattr(pred, "values", pred), dtype=float)
    if mu.isna().any():
        raise ValueError(
            f"manifest sanity produced {int(mu.isna().sum())} missing predictions"
        )
    return mu, {
        "sanity_eval_scope": "walkforward_manifest",
        "sanity_manifest_path": str(manifest_path),
        "sanity_eval_start": min(safe_dates).date().isoformat() if safe_dates else None,
        "sanity_eval_end": max(safe_dates).date().isoformat() if safe_dates else None,
        "n_oos_dates": int(len(safe_dates)),
        "n_skipped_pre_manifest_dates": int(len(skipped_pre_manifest_dates)),
        "n_manifest_artifacts_used": int(scored["__sanity_artifact_uri"].nunique()),
        "n_history_scorer_artifacts": int(n_history_artifacts),
        "cutoff_contract": (
            "manifest entry effective_train_cutoff_date/cutoff_date "
            "+ lookahead_days < eval date"
        ),
    }


def run_sanity_battery(
    artifact_path: Path,
    artifact_usage: dict | None = None,
) -> dict:
    """§5.2 shuffled-label + time-shift placebo on the artifact's training pipeline.

    Current implementation is the lower-cost existing-model diagnostic:
    score the validation partition once, then measure IC against the real
    label, shuffled labels, and future-shifted labels. It is a production
    acceptance gate, so unavailable sanity evidence fails closed. The shift
    diagnostic can also reflect slow regime/momentum persistence, so we record
    a multi-shift profile instead of treating one shifted IC as self-explaining.
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
    artifact = _load_artifact_payload(artifact_path)
    feat_cols = artifact.get("feature_cols", [])
    if not feat_cols:
        return {"passed": False, "reason": "artifact missing feature_cols"}
    # Validate against the model's own ranking target. Raw return-scale labels
    # are still used by calibrator/economics checks, but mixing them into the
    # rank-LTR placebo gate validates a different objective than the one the
    # scorer optimized.
    LABEL = _sanity_model_label_col(artifact)
    try:
        panel, panel_meta = _load_sanity_panel(feat_cols, LABEL)
    except (FileNotFoundError, KeyError) as exc:
        log.error("sanity panel unavailable — fail closed: %s", exc)
        return {
            "passed": False,
            "reason": str(exc),
            "sanity_method": "existing_model_label_diagnostics",
            "sanity_label_col": LABEL,
        }
    panel = panel.dropna(subset=[LABEL])
    distinct = sorted(panel.date.unique())
    val_cut = distinct[int(len(distinct) * 0.8)]
    val = panel[panel.date > val_cut].copy()
    eval_start = pd.Timestamp(val["date"].min()) if not val.empty else None
    if eval_start is None:
        return {
            "passed": False,
            "reason": "empty validation partition — sanity unavailable",
            "sanity_method": "existing_model_label_diagnostics",
            "sanity_label_col": LABEL,
        }

    # Predict using the artifact's model on val
    # (For panel-LTR XGB rank, recover boosters; for QHead, predict_distribution)
    sanity_meta: dict = {}
    try:
        import xgboost as xgb  # noqa: PLC0415
        manifest_scope = (
            isinstance(artifact_usage, dict)
            and artifact_usage.get("eval_scope") == "walkforward_manifest"
        )
        if manifest_scope:
            manifest_raw = (artifact_usage or {}).get("manifest_path")
            if not manifest_raw:
                return {
                    "passed": False,
                    "reason": "manifest sanity missing manifest_path",
                    "sanity_method": "manifest_point_in_time_label_diagnostics",
                    "sanity_eval_scope": "walkforward_manifest",
                    "sanity_label_col": LABEL,
                }
            mu_s, sanity_meta = _score_manifest_sanity(
                val,
                feat_cols,
                Path(manifest_raw),
                artifact_path,
                artifact,
                panel_history=panel,
            )
            sanity_meta.update(panel_meta)
            val = val.loc[mu_s.index].copy()
            mu = mu_s.loc[val.index].values
        elif artifact.get("kind") == "panel_ltr_xgboost":
            contract = _validate_static_sanity_oos_contract(artifact, eval_start)
            if not contract.get("passed"):
                return {
                    "passed": False,
                    "reason": contract["reason"],
                    "sanity_method": "existing_model_label_diagnostics",
                    "sanity_eval_scope": "static_artifact",
                    "sanity_label_col": LABEL,
                    "cutoff_contract": "artifact cutoff + lookahead_days < eval_start",
                    **contract,
                }
            # Panel-LTR stores booster in artifact under booster_b64 or similar
            # For sanity we just need PREDICTIONS, so use the saved model
            from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
            from kernel.panel_pipeline.feature_transform import transform_feature_frame  # noqa: PLC0415
            scorer = PanelScorer.load(artifact_path)
            X = transform_feature_frame(
                val,
                feat_cols,
                getattr(scorer, "metadata", {}) or {},
                source_space="panel",
            )
            mu = scorer.score(X).values
            sanity_meta = {
                "sanity_eval_scope": "static_artifact",
                "sanity_eval_start": pd.Timestamp(eval_start).date().isoformat(),
                "sanity_eval_end": pd.Timestamp(val["date"].max()).date().isoformat(),
                "n_oos_dates": int(val["date"].nunique()),
                "cutoff_contract": "artifact cutoff + lookahead_days < eval_start",
                **contract,
            }
        elif artifact.get("kind") in ("hf_patchtst", "patchtst_panel"):
            # PatchTST static sanity is non-trivial: the scorer is stateful (per-ticker
            # rolling buffer needs seq_len warmup) and per-day batched, so a direct
            # adaptation of the panel_ltr_xgboost path isn't faithful. The full
            # implementation belongs in a manifest-based sanity that mirrors how the
            # WF sim itself scores PatchTST (load scorer per cutoff, predict on val
            # using the buffer the manifest entry was trained against). Once
            # build_patchtst_wf_manifest output is registered + _score_manifest_sanity
            # gains a kind dispatch, this branch can use it.
            log.warning(
                "kind=%s — PatchTST static sanity is skipped pending "
                "_score_manifest_sanity hf_patchtst dispatch (build_patchtst_wf_manifest "
                "output supplies the manifest; the dispatch is the follow-up)",
                artifact.get("kind"),
            )
            return {
                "passed": False,
                "reason": ("PatchTST static sanity skipped — sequence model needs "
                           "manifest-based sanity with per-cutoff scorer warmup; "
                           "implementation pending"),
                "sanity_method": "patchtst_manifest_required",
                "sanity_label_col": LABEL,
                "sanity_eval_scope": "static_artifact",
            }
        else:
            log.warning("kind=%s — sanity not implemented for this head type",
                        artifact.get("kind"))
            return {
                "passed": False,
                "reason": "sanity not implemented for this kind",
                "sanity_method": "existing_model_label_diagnostics",
                "sanity_label_col": LABEL,
            }
    except Exception as exc:
        log.exception("sanity prediction failed; fail closed")
        return {
            "passed": False,
            "reason": f"prediction failed: {exc}",
            "sanity_label_col": LABEL,
            "sanity_method": (
                "manifest_point_in_time_label_diagnostics"
                if (artifact_usage or {}).get("eval_scope") == "walkforward_manifest"
                else "existing_model_label_diagnostics"
            ),
        }

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

    # Time-shift placebo: shift each ticker's labels forward. Keep the
    # original +60d gate threshold for compatibility, but record a broader
    # shift profile because long-horizon equity labels can remain correlated
    # under slow regime/momentum persistence.
    panel_s = panel.sort_values(["ticker", "date"]).copy()
    val_idx = val.set_index(["ticker", "date"])
    mu_by_idx = _pd.Series(mu, index=val_idx.index)
    placebo_shift_diagnostics = []
    placebo_ic = float("nan")
    placebo_aligned_real_ic = float("nan")
    for shift_days in (5, 10, 20, 40, 60, 80, 120, 180, 252):
        col = f"__shift_{shift_days}__"
        panel_s[col] = panel_s.groupby("ticker")[LABEL].shift(-shift_days)
        val_s = panel_s[panel_s.date > val_cut].dropna(subset=[col])
        if len(val_s) <= 100:
            placebo_shift_diagnostics.append({
                "shift_days": shift_days,
                "ic": None,
                "n_rows": int(len(val_s)),
                "n_dates": 0,
                "skipped": "too_few_rows",
            })
            continue
        val_s_idx = val_s.set_index(["ticker", "date"])
        common = val_s_idx.index.intersection(val_idx.index)
        if len(common) <= 100:
            placebo_shift_diagnostics.append({
                "shift_days": shift_days,
                "ic": None,
                "n_rows": int(len(common)),
                "n_dates": 0,
                "skipped": "too_few_aligned_rows",
            })
            continue
        mu_aligned = mu_by_idx.loc[common].values
        yva_real_aligned = val_s_idx.loc[common, LABEL].clip(-0.5, 0.5).values
        yva_placebo = val_s_idx.loc[common, col].clip(-0.5, 0.5).values
        dates_aligned = [d for _, d in common]
        aligned_real_ic = cs_ic(mu_aligned, yva_real_aligned, dates_aligned)
        ic = cs_ic(mu_aligned, yva_placebo, dates_aligned)
        n_dates = len(set(dates_aligned))
        placebo_shift_diagnostics.append({
            "shift_days": shift_days,
            "ic": ic,
            "aligned_real_ic": aligned_real_ic,
            "full_real_ic": real_ic,
            "n_rows": int(len(common)),
            "n_dates": int(n_dates),
            "abs_ratio_to_aligned_real": (
                abs(ic) / abs(aligned_real_ic) if aligned_real_ic else None
            ),
            "abs_ratio_to_full_real": (
                abs(ic) / abs(real_ic) if real_ic else None
            ),
        })
        if shift_days == 60:
            placebo_ic = ic
            placebo_aligned_real_ic = aligned_real_ic
            log.info(
                "  placebo_ic = %+.4f (expect < %s; full_real_ic=%+.4f)",
                placebo_ic,
                _placebo_ic_requirement_text(placebo_aligned_real_ic),
                real_ic,
            )
    if placebo_ic != placebo_ic:
        log.warning("  placebo skipped — too few aligned val rows; fail closed")

    sanity_regime_ic = {"passed": False, "reason": "not_computed"}
    try:
        from scripts.analyze_manifest_sanity_placebo import (  # noqa: PLC0415
            build_regime_series,
            regime_diagnostics,
            regime_shift_diagnostics,
        )

        mu_series = _pd.Series(mu, index=val.index)
        regimes_df = build_regime_series(val["date"].unique(), strategy_dir=STRATEGY_DIR)
        by_regime = regime_diagnostics(val, mu_series, LABEL, regimes_df)
        by_regime_shift = regime_shift_diagnostics(
            panel,
            val,
            mu_series,
            LABEL,
            regimes_df,
            shifts=(60,),
            min_names=5,
        )
        min_dates = 30
        min_mean_ic = 0.02
        max_placebo_ratio = 0.5
        regimes_out = {}
        failed = []
        eligible_any = False
        for regime, stats in by_regime.items():
            row60 = next(
                (
                    r for r in by_regime_shift.get(regime, [])
                    if r.get("shift_days") == 60
                ),
                {},
            )
            mean_ic = stats.get("mean_ic")
            placebo60 = row60.get("model_placebo_ic")
            aligned_real60 = row60.get("aligned_real_ic")
            n_dates = int(stats.get("n_dates") or 0)
            eligible = n_dates >= min_dates
            passed = False
            if eligible:
                eligible_any = True
                try:
                    mean_ic_f = float(mean_ic)
                except (TypeError, ValueError):
                    mean_ic_f = float("nan")
                placebo_ok = True
                if placebo60 is not None and mean_ic_f == mean_ic_f:
                    placebo_ref = mean_ic_f
                    try:
                        aligned_real60_f = float(aligned_real60)
                        if aligned_real60_f == aligned_real60_f:
                            placebo_ref = aligned_real60_f
                    except (TypeError, ValueError):
                        placebo_ref = mean_ic_f
                    placebo_ok = abs(float(placebo60)) <= max(
                        0.005,
                        max_placebo_ratio * abs(placebo_ref),
                    )
                passed = (
                    mean_ic_f == mean_ic_f
                    and mean_ic_f >= min_mean_ic
                    and placebo_ok
                )
                if not passed:
                    failed.append(regime)
            regimes_out[regime] = {
                **stats,
                "eligible": bool(eligible),
                "passed": bool(passed) if eligible else True,
                "placebo_60_ic": placebo60,
                "placebo_60_aligned_real_ic": aligned_real60,
                "label_autocorr_60_ic": row60.get("label_autocorr_ic"),
            }
        sanity_regime_ic = {
            "passed": bool(eligible_any and not failed),
            "reason": (
                "regime sanity IC passed"
                if eligible_any and not failed else
                "regime sanity IC failed: " + ",".join(sorted(failed))
                if failed else
                "no regime has enough OOS dates for sanity IC validation"
            ),
            "min_n_dates": min_dates,
            "min_mean_ic": min_mean_ic,
            "max_placebo_ratio": max_placebo_ratio,
            "regimes": regimes_out,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("  regime sanity IC unavailable: %s", exc)
        sanity_regime_ic = {
            "passed": False,
            "reason": f"regime sanity IC unavailable: {exc}",
        }

    # Pass criteria
    pass_shuf = abs(shuf_ic) < 0.005
    pass_placebo = (
        (placebo_ic == placebo_ic)
        and (placebo_aligned_real_ic == placebo_aligned_real_ic)
        and (
            abs(placebo_ic) < _placebo_ic_threshold(placebo_aligned_real_ic)
            if placebo_aligned_real_ic != 0 else
            True
        )
    )
    sanity_method = (
        "manifest_point_in_time_label_diagnostics"
        if sanity_meta.get("sanity_eval_scope") == "walkforward_manifest"
        else "existing_model_label_diagnostics"
    )
    pass_regime = bool(sanity_regime_ic.get("passed"))
    pass_all = pass_shuf and pass_placebo and pass_regime
    if pass_all:
        sanity_reason = f"PASS: shuf_ic={shuf_ic:+.4f} placebo_ic={placebo_ic:+.4f}"
    elif pass_shuf and pass_placebo and not pass_regime:
        sanity_reason = (
            "FAIL: regime sanity IC failed: "
            f"{sanity_regime_ic.get('reason', 'unknown')}"
        )
    else:
        sanity_reason = (
            f"FAIL: shuf_ic={shuf_ic:+.4f} (need |·| < 0.005), "
            f"placebo_ic={placebo_ic:+.4f} "
            f"(must be available and < "
            f"{_placebo_ic_requirement_text(placebo_aligned_real_ic)})"
        )
    return {
        "passed": pass_all,
        "real_ic": real_ic,
        "sanity_shuffled_ic": shuf_ic,
        "sanity_placebo_ic": placebo_ic if placebo_ic == placebo_ic else None,
        "sanity_placebo_aligned_real_ic": (
            placebo_aligned_real_ic
            if placebo_aligned_real_ic == placebo_aligned_real_ic
            else None
        ),
        "sanity_label_col": LABEL,
        "sanity_method": sanity_method,
        "placebo_shift_diagnostics": placebo_shift_diagnostics,
        "sanity_regime_ic": sanity_regime_ic,
        "reason": sanity_reason,
        **sanity_meta,
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
    ap.add_argument("--allow-pass-open-trade-monotonicity", action="store_true",
                    help="Diagnostic only: allow insufficient per-regime "
                         "trade evidence to pass-open. Metadata stamped with "
                         "this flag is never promotable.")
    ap.add_argument("--skip-config-parity", action="store_true",
                    help="Skip prod/WF decision-semantics parity guard. "
                         "Use only for explicitly exploratory runs.")
    ap.add_argument("--derive-config-from-prod", action="store_true",
                    help="Before running, derive a production-semantic WF "
                         "config from --strategy-config. The base config only "
                         "contributes walkforward/calibration artifact paths.")
    ap.add_argument("--preserve-experiment-overrides", action="store_true",
                    help="With --derive-config-from-prod, explicitly preserve "
                         "whitelisted semantic experiment knobs from the base "
                         "config. Diagnostic/non-promotable unless production "
                         "config parity also passes.")
    args = ap.parse_args()

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        log.error("artifact not found: %s", artifact_path)
        sys.exit(2)

    artifact = _load_artifact_payload(artifact_path)

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
        preferred_manifest = _resolve_strategy_path(str(manifest_path))
        selected_manifest, selected_usage = _matching_manifest_for_recipe(
            artifact_path=artifact_path,
            preferred_manifest=preferred_manifest,
        )
        if selected_manifest is not None and selected_manifest != preferred_manifest:
            log.warning(
                "Base WF manifest %s is not recipe-compatible with candidate; "
                "using same-recipe manifest %s (%s)",
                preferred_manifest,
                selected_manifest,
                selected_usage.get("reason"),
            )
            manifest_path = str(selected_manifest)
        elif not bool(selected_usage.get("recipe_validated")):
            log.warning(
                "No same-recipe WF manifest found for candidate yet; keeping "
                "base manifest %s so the gate can fail closed (%s)",
                preferred_manifest,
                selected_usage.get("reason"),
            )
        derived_dir = STRATEGY_DIR / "artifacts" / "diagnostics" / "wf_eval_configs"
        derived_dir.mkdir(parents=True, exist_ok=True)
        derived_name = f"{Path(args.strategy_config).stem}.prod_semantic.json"
        derived_path = derived_dir / derived_name
        derived_cfg = build_wf_config_from_prod(
            prod_cfg,
            manifest_path=str(manifest_path),
            base_wf_config=base_cfg,
            strategy_dir=STRATEGY_DIR,
            preserve_experiment_overrides=args.preserve_experiment_overrides,
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
            trace_dir = _resolve_trace_dir_arg(args.trace_dir)
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
        elif artifact_usage.get("eval_scope") == "static_artifact":
            static_contract = _validate_static_wf_oos_contract(artifact, CUTS)
            if not bool(static_contract.get("passed")):
                wf_result = {
                    "passed": False,
                    "reason": static_contract["reason"],
                    "static_wf_oos_contract": static_contract,
                }
                log.error("WF result: %s", wf_result["reason"])
            else:
                wf_result = run_walk_forward(
                    args.strategy_config,
                    jobs=args.jobs,
                    trace_dir=trace_dir,
                )
                log.info("WF result: %s", wf_result["reason"])
        else:
            wf_result = run_walk_forward(
                args.strategy_config,
                jobs=args.jobs,
                trace_dir=trace_dir,
            )
            log.info("WF result: %s", wf_result["reason"])

    trade_gate_result = {"passed": True, "reason": "skipped"}
    trade_contract_result = {"passed": True, "reason": "skipped"}
    alpha_economics_result = {"passed": True, "reason": "skipped"}
    if not args.skip_wf and not args.skip_trade_gates:
        if trace_dir is None:
            trade_gate_result = {
                "passed": False,
                "reason": "trade gates require persisted round-trip ledgers",
            }
            trade_contract_result = dict(trade_gate_result)
            alpha_economics_result = dict(trade_gate_result)
        elif wf_result.get("cuts"):
            trade_contract_result = run_trade_contract_gate(wf_result, gate_config)
            trade_gate_result = run_trade_monotonicity_gate(
                wf_result,
                score_cols=_trade_monotonicity_score_cols_from_config(gate_config),
                allow_pass_open=args.allow_pass_open_trade_monotonicity,
            )
            alpha_economics_result = run_alpha_economics_gate(wf_result)
        log.info("Trade contract result: %s", trade_contract_result["reason"])
        log.info("Trade gate result: %s", trade_gate_result["reason"])
        log.info("Alpha economics result: %s", alpha_economics_result["reason"])

    sanity_result = {"passed": True, "reason": "skipped"}
    if not args.skip_sanity:
        sanity_result = run_sanity_battery(
            artifact_path,
            artifact_usage=artifact_usage,
        )
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

    skipped_required_gates = _required_validation_skip_reasons(args)
    if skipped_required_gates:
        log.warning(
            "Required gate(s) skipped: %s — metadata is diagnostic-only",
            ", ".join(skipped_required_gates),
        )
    overall_pass = _compute_overall_pass(
        wf_result=wf_result,
        sanity_result=sanity_result,
        trade_contract_result=trade_contract_result,
        trade_gate_result=trade_gate_result,
        alpha_economics_result=alpha_economics_result,
        validation_scope_ok=validation_scope_ok,
        parity_result=parity_result,
        skipped_required_gates=skipped_required_gates,
    )
    wf_meta = {
        "passed": overall_pass,
        "diagnostic_only": bool(skipped_required_gates),
        "skipped_required_gates": skipped_required_gates,
        "wf_3cut_sharpe_mean": wf_result.get("wf_3cut_sharpe_mean"),
        "wf_3cut_sharpe_std":  wf_result.get("wf_3cut_sharpe_std"),
        "wf_3cut_apy_mean":    wf_result.get("wf_3cut_apy_mean"),
        "spy_sharpe_mean":     wf_result.get("spy_sharpe_mean"),
        "strategy_minus_spy_sharpe_mean": wf_result.get("strategy_minus_spy_sharpe_mean"),
        "spy_apy_mean":        wf_result.get("spy_apy_mean"),
        "strategy_minus_spy_apy_mean": wf_result.get("strategy_minus_spy_apy_mean"),
        "n_cuts_beat_spy_sharpe": wf_result.get("n_cuts_beat_spy_sharpe"),
        "n_cuts_beat_spy_apy": wf_result.get("n_cuts_beat_spy_apy"),
        "benchmark_by_dominant_regime": wf_result.get("benchmark_by_dominant_regime"),
        "regime_benchmark_failures": wf_result.get("regime_benchmark_failures"),
        "performance_tax_basis_counts": wf_result.get("performance_tax_basis_counts"),
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
        "alpha_economics":     alpha_economics_result,
        "real_ic":             sanity_result.get("real_ic"),
        "sanity_shuffled_ic":  sanity_result.get("sanity_shuffled_ic"),
        "sanity_placebo_ic":   sanity_result.get("sanity_placebo_ic"),
        "sanity_placebo_aligned_real_ic": (
            sanity_result.get("sanity_placebo_aligned_real_ic")
        ),
        "sanity_label_col":    sanity_result.get("sanity_label_col"),
        "sanity_method":       sanity_result.get("sanity_method"),
        "sanity_eval_scope":   sanity_result.get("sanity_eval_scope"),
        "sanity_manifest_path": sanity_result.get("sanity_manifest_path"),
        "sanity_eval_start":   sanity_result.get("sanity_eval_start"),
        "sanity_eval_end":     sanity_result.get("sanity_eval_end"),
        "sanity_n_oos_dates":  sanity_result.get("n_oos_dates"),
        "sanity_cutoff_contract": sanity_result.get("cutoff_contract"),
        "sanity_regime_ic":    sanity_result.get("sanity_regime_ic"),
        "placebo_shift_diagnostics": sanity_result.get("placebo_shift_diagnostics"),
        "wf_reason":           wf_result.get("reason"),
        "sanity_reason":       sanity_result.get("reason"),
        "run_at":              datetime.datetime.utcnow().isoformat(),
        "gate_version":        GATE_VERSION,
    }

    # Stamp into artifact metadata. Non-JSON sequence checkpoints must never
    # be overwritten; gate metadata lands in their JSON sidecar instead.
    md = artifact.get("metadata") or {}
    md["wf_gate_metadata"] = wf_meta
    artifact["metadata"] = md
    written = _write_artifact_payload(artifact_path, artifact)
    log.info("Wrote wf_gate_metadata to %s", written)
    log.info("=" * 60)
    log.info("VERDICT: %s", "PASS" if overall_pass else "FAIL")
    log.info("=" * 60)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
