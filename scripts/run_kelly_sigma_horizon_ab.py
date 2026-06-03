#!/usr/bin/env python3
"""Run or plan the Kelly sigma-horizon A/B required by the 2026-06-03 audit.

Default mode is dry-run: write a reproducible plan and promotion verdict
schema without launching the expensive 27-month OOS sims. Pass ``--execute``
to run the real A/B plus the A/A seed-resplit check.

This script does not mutate production configs. The treatment config is
derived under the diagnostics output directory and sets only:

    ranking.kelly_sizing.sigma_horizon_days = 60
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
DEFAULT_BASE_CONFIG = "strategy_config.sim_baseline.json"
DEFAULT_START = "2024-01-02"
DEFAULT_END = "2026-03-28"
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_AA_SEED_OFFSET = 1000
DEFAULT_SIGMA_HORIZON_DAYS = 60
SUBREPO_IMPORT_ORDER = (
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-model",
    "renquant-pipeline",
    "renquant-execution",
    "renquant-strategy-104",
    "renquant-backtesting",
    "renquant-orchestrator",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    role: str
    config_path: Path
    seeds: tuple[int, ...]

    def as_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "role": self.role,
            "config_path": str(self.config_path),
            "seeds": list(self.seeds),
        }


def parse_seeds(raw: str | None) -> tuple[int, ...]:
    if raw is None or not str(raw).strip():
        return DEFAULT_SEEDS
    text = str(raw).strip()
    if "," not in text:
        n = int(text)
        if n <= 0:
            raise ValueError("seed count must be positive")
        return tuple(range(n))
    out = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not out:
        raise ValueError("seed list must be non-empty")
    return out


def offset_seeds(seeds: tuple[int, ...], offset: int) -> tuple[int, ...]:
    return tuple(int(seed) + int(offset) for seed in seeds)


def resolve_strategy_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else STRATEGY_DIR / path


def bootstrap_subrepo_imports(repo_root: Path = REPO) -> Path:
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from subrepo_paths import resolve_subrepo_root  # noqa: PLC0415

    subrepo_root = resolve_subrepo_root(repo_root).resolve()
    for repo in reversed(SUBREPO_IMPORT_ORDER):
        src = subrepo_root / repo / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    return subrepo_root


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        STRATEGY_DIR
        / "artifacts"
        / "diagnostics"
        / f"kelly_sigma_horizon_ab_{stamp}"
    )


def build_treatment_config(
    *,
    base_config_path: Path,
    treatment_config_path: Path,
    sigma_horizon_days: int,
) -> dict[str, Any]:
    scripts_dir = REPO / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from build_kelly_sigma_horizon_ab_config import (  # noqa: PLC0415
        SIGMA_HORIZON_PATH,
        build_kelly_sigma_horizon_ab_config,
        changed_dotted_paths,
    )

    baseline = json.loads(base_config_path.read_text())
    treatment = build_kelly_sigma_horizon_ab_config(
        baseline,
        sigma_horizon_days=int(sigma_horizon_days),
    )
    treatment_config_path.parent.mkdir(parents=True, exist_ok=True)
    treatment_config_path.write_text(json.dumps(treatment, indent=2) + "\n")
    return {
        "base_config_path": str(base_config_path),
        "treatment_config_path": str(treatment_config_path),
        "changed_paths": changed_dotted_paths(baseline, treatment),
        SIGMA_HORIZON_PATH: int(sigma_horizon_days),
    }


def build_variants(
    *,
    base_config_path: Path,
    treatment_config_path: Path,
    seeds: tuple[int, ...],
    aa_seed_offset: int = DEFAULT_AA_SEED_OFFSET,
) -> list[VariantSpec]:
    return [
        VariantSpec("A_golden", "real_control", base_config_path, seeds),
        VariantSpec("B_sigma_horizon_60", "real_treatment", treatment_config_path, seeds),
        VariantSpec(
            "AA_golden_resplit",
            "aa_resplit",
            base_config_path,
            offset_seeds(seeds, aa_seed_offset),
        ),
    ]


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_placebo_evidence(paths: list[str]) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        payload = json.loads(path.read_text())
        interp = payload.get("interpretation") or {}
        evidence.append({
            "path": str(path),
            "promotion_evidence": bool(interp.get("promotion_evidence")),
            "aligned_real_60_ic": interp.get("aligned_real_60_ic"),
            "placebo_60_ic": interp.get("placebo_60_ic"),
            "label_autocorr_60_ic": interp.get("label_autocorr_60_ic"),
            "primary_warning": interp.get("primary_warning"),
        })
    return {
        "provided": bool(evidence),
        "passed": bool(evidence) and all(row["promotion_evidence"] for row in evidence),
        "items": evidence,
    }


def apply_run_overrides(config: dict[str, Any], *, manifest_path: str = "") -> None:
    if manifest_path:
        wf = config.setdefault("walkforward", {})
        wf["enabled"] = True
        wf["manifest_path"] = manifest_path
        wf.setdefault("fail_on_no_model", True)


def validate_walkforward_manifest(config: dict[str, Any], strategy_dir: Path) -> None:
    wf = config.get("walkforward") or {}
    if not bool(wf.get("enabled", False)):
        return
    raw = wf.get("manifest_path")
    if not raw:
        raise FileNotFoundError("walkforward.enabled=true but manifest_path is missing")
    path = Path(str(raw))
    manifest = path if path.is_absolute() else strategy_dir / path
    if not manifest.exists():
        raise FileNotFoundError(f"walkforward manifest not found: {manifest}")
    preflight_walkforward_manifest(manifest, config=config)


def _resolve_manifest_uri(manifest: Path, raw: str) -> Path:
    path = Path(str(raw))
    return path if path.is_absolute() else manifest.parent / path


def _compact_consistency_error(exc: Exception) -> str:
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    if not lines:
        return exc.__class__.__name__
    keep = [
        line
        for line in lines
        if line.startswith("Config-consistency")
        or line.startswith("Live config fingerprint")
        or line.startswith("Artifact stored fingerprint")
    ]
    return "; ".join(keep[:3] or lines[:1])


def _preflight_artifact_config_consistency(
    *,
    manifest: Path,
    rows: list[Any],
    config: dict[str, Any],
    max_errors: int,
) -> dict[str, Any]:
    panel_cfg = ((config.get("ranking") or {}).get("panel_scoring") or {})
    if not bool(panel_cfg.get("strict_config_consistency", True)):
        return {"artifact_config_checked": 0, "strict_config_consistency": False}

    strategy_dir = Path(config.get("_strategy_dir") or STRATEGY_DIR)
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    from kernel.config_consistency import (  # noqa: PLC0415
        ConfigModelMismatch,
        assert_consistent,
    )

    errors: list[str] = []
    seen: set[Path] = set()
    checked = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        artifact_uri = row.get("artifact_uri")
        if not artifact_uri:
            continue
        artifact_path = _resolve_manifest_uri(manifest, str(artifact_uri))
        if not artifact_path.exists() or artifact_path in seen:
            continue
        seen.add(artifact_path)
        try:
            artifact = json.loads(artifact_path.read_text())
            if not isinstance(artifact, dict):
                raise ValueError("artifact JSON root is not a mapping")
            assert_consistent(
                config,
                artifact,
                artifact_label=artifact_path.name,
                strict=True,
            )
            checked += 1
        except ConfigModelMismatch as exc:
            errors.append(
                f"entry {idx}: {artifact_path}: {_compact_consistency_error(exc)}"
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"entry {idx}: {artifact_path}: {exc}")
        if len(errors) >= max_errors:
            break
    if errors:
        raise ValueError(
            "walkforward artifact config preflight failed for "
            f"{manifest}: " + "; ".join(errors)
        )
    return {
        "artifact_config_checked": checked,
        "strict_config_consistency": True,
    }


def preflight_walkforward_manifest(
    manifest: Path,
    *,
    config: dict[str, Any] | None = None,
    max_errors: int = 5,
) -> dict[str, Any]:
    payload = json.loads(manifest.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError(f"walkforward manifest has no retrain entries: {manifest}")
    errors: list[str] = []
    n_calibrators = 0
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            errors.append(f"entry {idx}: not a mapping")
            continue
        artifact_uri = row.get("artifact_uri")
        calibrator_uri = row.get("calibrator_uri") or row.get("calibration_uri")
        if not artifact_uri:
            errors.append(f"entry {idx}: artifact_uri missing")
        else:
            artifact_path = _resolve_manifest_uri(manifest, str(artifact_uri))
            if not artifact_path.exists():
                errors.append(f"entry {idx}: artifact missing at {artifact_path}")
        if not calibrator_uri:
            errors.append(f"entry {idx}: calibrator_uri missing")
        else:
            n_calibrators += 1
            calibrator_path = _resolve_manifest_uri(manifest, str(calibrator_uri))
            if not calibrator_path.exists():
                errors.append(f"entry {idx}: calibrator missing at {calibrator_path}")
        if len(errors) >= max_errors:
            break
    if errors:
        raise FileNotFoundError(
            "walkforward manifest preflight failed for "
            f"{manifest}: " + "; ".join(errors)
        )
    result = {
        "manifest": str(manifest),
        "entries": len(rows),
        "entries_with_calibrator": n_calibrators,
    }
    if config is not None:
        result.update(_preflight_artifact_config_consistency(
            manifest=manifest,
            rows=rows,
            config=config,
            max_errors=max_errors,
        ))
    return result


def materialize_manifest_for_runner(
    manifest: Path,
    *,
    out_dir: Path,
    strategy_dir: Path = STRATEGY_DIR,
) -> Path:
    """Write a manifest whose local artifact URIs resolve for this runner.

    WalkForwardModelLoader resolves relative URIs against the manifest file's
    directory. Some archived sim manifests store URIs relative to strategy_dir.
    For diagnostics, write a resolved copy under out_dir rather than mutating
    the archived manifest.
    """
    payload = json.loads(manifest.read_text())
    rows = payload.get("retrains", []) if isinstance(payload, dict) else payload
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in ("artifact_uri", "calibrator_uri", "calibration_uri"):
            raw = row.get(key)
            if not raw or "://" in str(raw):
                continue
            path = Path(str(raw))
            if path.is_absolute():
                continue
            manifest_relative = manifest.parent / path
            strategy_relative = strategy_dir / path
            if manifest_relative.exists():
                row[key] = str(manifest_relative.resolve())
            elif strategy_relative.exists():
                row[key] = str(strategy_relative.resolve())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{manifest.stem}.resolved.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    preflight_walkforward_manifest(out)
    return out


def promotion_verdict(
    variant_metrics: dict[str, dict[str, Any]],
    placebo: dict[str, Any],
    *,
    aa_max_abs_sharpe_lift: float = 0.10,
) -> dict[str, Any]:
    control = variant_metrics.get("A_golden") or {}
    treatment = variant_metrics.get("B_sigma_horizon_60") or {}
    aa = variant_metrics.get("AA_golden_resplit") or {}
    reasons: list[str] = []

    control_sharpe = _finite(control.get("sharpe_mean"))
    treatment_sharpe = _finite(treatment.get("sharpe_mean"))
    aa_sharpe = _finite(aa.get("sharpe_mean"))
    treatment_dsr = _finite(treatment.get("dsr"))
    treatment_pbo = _finite(treatment.get("pbo"))
    control_apy = _finite(control.get("apy_mean"))
    treatment_apy = _finite(treatment.get("apy_mean"))

    sharpe_lift = (
        treatment_sharpe - control_sharpe
        if treatment_sharpe is not None and control_sharpe is not None
        else None
    )
    apy_lift = (
        treatment_apy - control_apy
        if treatment_apy is not None and control_apy is not None
        else None
    )
    aa_sharpe_lift = (
        aa_sharpe - control_sharpe
        if aa_sharpe is not None and control_sharpe is not None
        else None
    )

    if sharpe_lift is None or sharpe_lift <= 0:
        reasons.append("treatment Sharpe did not improve over golden")
    if apy_lift is None or apy_lift <= 0:
        reasons.append("treatment APY did not improve over golden")
    if aa_sharpe_lift is None:
        reasons.append("A/A resplit metrics missing")
    elif abs(aa_sharpe_lift) > aa_max_abs_sharpe_lift:
        reasons.append(
            "A/A resplit moved Sharpe by "
            f"{aa_sharpe_lift:+.4f}, above tolerance {aa_max_abs_sharpe_lift:.4f}"
        )
    if not placebo.get("provided"):
        reasons.append("shuffle/time-shift placebo evidence missing")
    elif not placebo.get("passed"):
        reasons.append("placebo evidence did not pass")
    if not (
        (treatment_dsr is not None and treatment_dsr > 0.5)
        or (treatment_pbo is not None and treatment_pbo < 0.5)
    ):
        reasons.append("Tier 3 falsifiability missing: need DSR > 0.5 or PBO < 0.5")

    return {
        "tier3_ready": not reasons,
        "blocked_reasons": reasons,
        "deltas": {
            "apy_lift": apy_lift,
            "sharpe_lift": sharpe_lift,
            "aa_sharpe_lift": aa_sharpe_lift,
        },
        "thresholds": {
            "aa_max_abs_sharpe_lift": float(aa_max_abs_sharpe_lift),
            "tier3_dsr_min": 0.5,
            "tier3_pbo_max": 0.5,
        },
    }


def _json_float(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    try:
        out = float(value)
    except (TypeError, ValueError):
        return str(value)
    return out if math.isfinite(out) else None


def _result_metrics(result: Any) -> dict[str, Any]:
    return {
        "seeds": list(result.seeds),
        "n_seeds": int(result.n_seeds),
        "apy_mean": _json_float(result.apy_mean),
        "apy_std": _json_float(result.apy_std),
        "sharpe_mean": _json_float(result.sharpe_mean),
        "sharpe_std": _json_float(result.sharpe_std),
        "sortino_mean": _json_float(result.sortino_mean),
        "sortino_std": _json_float(result.sortino_std),
        "calmar_mean": _json_float(result.calmar_mean),
        "calmar_std": _json_float(result.calmar_std),
        "max_dd_mean": _json_float(result.max_dd_mean),
        "max_dd_std": _json_float(result.max_dd_std),
        "dsr": _json_float(result.dsr),
        "pbo": _json_float(result.pbo),
        "majority_vote_action_consistency": _json_float(
            result.majority_vote_action_consistency
        ),
    }


def execute_variant(
    variant: VariantSpec,
    *,
    start: str,
    end: str,
    initial_cash: float,
    parallel_seeds: bool,
    manifest_path: str = "",
) -> dict[str, Any]:
    strategy_dir = STRATEGY_DIR
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))
    bootstrap_subrepo_imports(REPO)

    config = json.loads(variant.config_path.read_text())
    config["_strategy_dir"] = str(strategy_dir)
    config["_strategy_config_name"] = str(variant.config_path)
    config["initial_cash"] = float(initial_cash)
    config["backtest_start"] = start
    config["backtest_end"] = end
    config["persistence"] = {"enabled": False}
    config.setdefault("data_freshness", {})["enabled"] = False
    apply_run_overrides(config, manifest_path=manifest_path)
    validate_walkforward_manifest(config, strategy_dir)

    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from sim.runner import run_backtest_multi_seed  # noqa: PLC0415

    benchmark = config.get("benchmark", "SPY")
    spy_df = fetch_ohlcv(benchmark)
    etf_map = config.get("sector_etf_map", {})
    symbols = sorted(set(config.get("watchlist", [])) | set(etf_map.values()))
    ohlcv = {benchmark: spy_df}
    for symbol in symbols:
        try:
            ohlcv[symbol] = fetch_ohlcv(symbol)
        except Exception:
            continue

    result = run_backtest_multi_seed(
        seeds=list(variant.seeds),
        parallel=bool(parallel_seeds),
        config=config,
        strategy_dir=strategy_dir,
        ohlcv=ohlcv,
        spy_df=spy_df,
        sector_etf_map=etf_map,
        initial_cash=float(initial_cash),
        backtest_start=start,
        backtest_end=end,
        snapshot=False,
    )
    return _result_metrics(result)


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    seeds = parse_seeds(args.seeds)
    out_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir()
    base_config_path = resolve_strategy_path(args.base_config).resolve()
    treatment_config_path = (
        Path(args.treatment_config).resolve()
        if args.treatment_config
        else out_dir / f"strategy_config.sim_kelly_sigma_horizon{args.sigma_horizon_days}.json"
    )
    variants = build_variants(
        base_config_path=base_config_path,
        treatment_config_path=treatment_config_path,
        seeds=seeds,
        aa_seed_offset=int(args.aa_seed_offset),
    )
    return {
        "audit": "2026-06-03-kelly-sizing-audit",
        "mode": "execute" if args.execute else "dry_run",
        "start": args.start,
        "end": args.end,
        "initial_cash": float(args.initial_cash),
        "output_dir": str(out_dir),
        "base_config_path": str(base_config_path),
        "treatment_config_path": str(treatment_config_path),
        "sigma_horizon_days": int(args.sigma_horizon_days),
        "run_overrides": {
            "manifest_path": str(args.manifest_path or ""),
        },
        "variants": [variant.as_json() for variant in variants],
        "placebo_json": list(args.placebo_json or []),
        "mandatory_checks": {
            "real_ab": ["A_golden", "B_sigma_horizon_60"],
            "aa_resplit": ["A_golden", "AA_golden_resplit"],
            "placebo": "provide JSON from scripts/analyze_manifest_sanity_placebo.py "
                       "via --placebo-json before promotion",
            "multi_seed_floor": 5,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--treatment-config", default="")
    parser.add_argument("--sigma-horizon-days", type=int, default=DEFAULT_SIGMA_HORIZON_DAYS)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--initial-cash", type=float, default=100_000.0)
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--aa-seed-offset", type=int, default=DEFAULT_AA_SEED_OFFSET)
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--manifest-path",
        default="",
        help="Optional walkforward manifest override applied at run time. "
             "Relative paths resolve under backtesting/renquant_104.",
    )
    parser.add_argument(
        "--no-materialize-manifest",
        action="store_true",
        help="Do not write a resolved manifest copy under output_dir before execute.",
    )
    parser.add_argument("--placebo-json", action="append", default=[])
    parser.add_argument("--aa-max-abs-sharpe-lift", type=float, default=0.10)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--parallel-seeds", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = build_plan(args)
    out_dir = Path(plan["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    treatment_meta = build_treatment_config(
        base_config_path=Path(plan["base_config_path"]),
        treatment_config_path=Path(plan["treatment_config_path"]),
        sigma_horizon_days=int(plan["sigma_horizon_days"]),
    )
    plan["treatment_config"] = treatment_meta
    effective_manifest_path = str(args.manifest_path or "")
    if effective_manifest_path and not args.no_materialize_manifest:
        raw_manifest = resolve_strategy_path(effective_manifest_path).resolve()
        effective_manifest_path = str(materialize_manifest_for_runner(
            raw_manifest,
            out_dir=out_dir,
            strategy_dir=STRATEGY_DIR,
        ))
        plan["run_overrides"]["effective_manifest_path"] = effective_manifest_path
    variant_metrics: dict[str, dict[str, Any]] = {}
    execution_error: dict[str, Any] | None = None

    if args.execute:
        variants = [
            VariantSpec(
                name=row["name"],
                role=row["role"],
                config_path=Path(row["config_path"]),
                seeds=tuple(int(seed) for seed in row["seeds"]),
            )
            for row in plan["variants"]
        ]
        for variant in variants:
            try:
                variant_metrics[variant.name] = execute_variant(
                    variant,
                    start=str(args.start),
                    end=str(args.end),
                    initial_cash=float(args.initial_cash),
                    parallel_seeds=bool(args.parallel_seeds),
                    manifest_path=effective_manifest_path,
                )
            except (FileNotFoundError, ValueError) as exc:
                execution_error = {
                    "variant": variant.name,
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                }
                break

    placebo = load_placebo_evidence(list(args.placebo_json or []))
    verdict = promotion_verdict(
        variant_metrics,
        placebo,
        aa_max_abs_sharpe_lift=float(args.aa_max_abs_sharpe_lift),
    )
    payload = {
        "plan": plan,
        "variant_metrics": variant_metrics,
        "placebo": placebo,
        "promotion_verdict": verdict,
    }
    if execution_error is not None:
        payload["execution_error"] = execution_error
    out_path = out_dir / "kelly_sigma_horizon_ab_plan.json"
    out_path.write_text(json.dumps(payload, indent=2, default=_json_float) + "\n")
    result = {"out": str(out_path), "tier3_ready": verdict["tier3_ready"]}
    if execution_error is not None:
        result["execution_error"] = execution_error
    print(json.dumps(result, indent=2))
    if execution_error is not None:
        return 2
    return 0 if (not args.execute or variant_metrics) else 2


if __name__ == "__main__":
    raise SystemExit(main())
