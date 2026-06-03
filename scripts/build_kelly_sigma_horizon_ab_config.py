#!/usr/bin/env python3
"""Build a Kelly sigma-horizon A/B experiment config.

The production default remains annualized sigma semantics:
``ranking.kelly_sizing.sigma_horizon_days`` absent is equivalent to 252.
This builder derives a treatment config from an existing sim/WF baseline and
sets exactly one semantic knob to run the audit's opt-in 60-day treatment.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
SIGMA_HORIZON_PATH = "ranking.kelly_sizing.sigma_horizon_days"
DEFAULT_BASE_CONFIG = "strategy_config.sim_baseline_hmm.json"


def _resolve_strategy_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else STRATEGY_DIR / path


def _set_path(obj: dict[str, Any], dotted: str, value: Any) -> None:
    cur: Any = obj
    parts = dotted.split(".")
    for part in parts[:-1]:
        nxt = cur.setdefault(part, {})
        if not isinstance(nxt, dict):
            raise TypeError(f"cannot set {dotted}: {part} is not a mapping")
        cur = nxt
    cur[parts[-1]] = value


def _flatten(prefix: str, obj: Any, out: dict[str, Any]) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), value, out)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            _flatten(f"{prefix}.{idx}" if prefix else str(idx), value, out)
    else:
        out[prefix] = obj


def changed_dotted_paths(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    before_flat: dict[str, Any] = {}
    after_flat: dict[str, Any] = {}
    _flatten("", before, before_flat)
    _flatten("", after, after_flat)
    changed: list[str] = []
    for path in sorted(set(before_flat) | set(after_flat)):
        if before_flat.get(path, "<absent>") != after_flat.get(path, "<absent>"):
            changed.append(path)
    return changed


def build_kelly_sigma_horizon_ab_config(
    baseline_config: dict[str, Any],
    *,
    sigma_horizon_days: int = 60,
) -> dict[str, Any]:
    if sigma_horizon_days <= 0:
        raise ValueError("sigma_horizon_days must be positive")
    cfg = copy.deepcopy(baseline_config)
    _set_path(cfg, SIGMA_HORIZON_PATH, int(sigma_horizon_days))
    changed = changed_dotted_paths(baseline_config, cfg)
    if changed != [SIGMA_HORIZON_PATH]:
        raise ValueError(
            "Kelly sigma-horizon A/B derivation changed unexpected path(s): "
            f"{changed}"
        )
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default=DEFAULT_BASE_CONFIG,
        help="Baseline/sim config to derive from. Relative paths resolve under "
             "backtesting/renquant_104.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Treatment config output path. Relative paths resolve under "
             "backtesting/renquant_104.",
    )
    parser.add_argument(
        "--sigma-horizon-days",
        type=int,
        default=60,
        help="Kelly sigma horizon treatment value. Default: 60.",
    )
    args = parser.parse_args()

    base_path = _resolve_strategy_path(args.base_config)
    out_path = _resolve_strategy_path(args.out)
    baseline = json.loads(base_path.read_text())
    treatment = build_kelly_sigma_horizon_ab_config(
        baseline,
        sigma_horizon_days=args.sigma_horizon_days,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(treatment, indent=2, sort_keys=False) + "\n")
    print(json.dumps({
        "base_config": str(base_path),
        "out": str(out_path),
        "changed_paths": changed_dotted_paths(baseline, treatment),
        SIGMA_HORIZON_PATH: args.sigma_horizon_days,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
