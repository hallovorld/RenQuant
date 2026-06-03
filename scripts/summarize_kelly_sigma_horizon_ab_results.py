#!/usr/bin/env python3
"""Summarize post-run Kelly sigma-horizon A/B result rows.

Reads CSV, JSON, or JSONL rows that already contain experiment metrics and
prints per-variant plus per-variant/per-regime mean +/- std summaries. This
script is intentionally read-only: it does not run backtests or mutate configs.

Expected columns include:

    variant, control_type, seed, regime, apy, sharpe, maxdd, cash_pct,
    kelly_target_pct

If DSR/PBO columns are present, their passed-through values are summarized.
If they are absent, the output still contains placeholder fields so promotion
reports can distinguish "not provided" from computed values.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


SUMMARY_METRICS = (
    "apy",
    "sharpe",
    "maxdd",
    "cash_pct",
    "kelly_target_pct",
    "dsr",
    "pbo",
)
METRIC_ALIASES = {
    "maxdd": ("maxdd", "max_dd", "max_drawdown", "max_drawdown_pct"),
    "kelly_target_pct": (
        "kelly_target_pct",
        "kelly_target",
        "kelly_target_mean",
        "kelly_target_pct_mean",
    ),
    "cash_pct": ("cash_pct", "cash", "cash_percent", "cash_pct_mean"),
    "apy": ("apy", "apy_pct"),
    "sharpe": ("sharpe", "sharpe_mean"),
    "dsr": ("dsr", "deflated_sharpe", "deflated_sharpe_ratio"),
    "pbo": ("pbo", "probability_of_backtest_overfitting"),
}
NON_NUMERIC = {"", "-", "--", "na", "n/a", "nan", "none", "null", "inf", "-inf"}


def _read_stdin() -> tuple[str, str]:
    text = sys.stdin.read()
    return text, "jsonl"


def _load_csv(text: str) -> list[dict[str, Any]]:
    return [dict(row) for row in csv.DictReader(text.splitlines())]


def _load_json(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        for key in ("rows", "results", "experiments"):
            value = payload.get(key)
            if isinstance(value, list):
                rows = value
                break
        else:
            rows = [payload]
    else:
        raise ValueError("JSON input must be an object, list, or object with rows/results")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSON rows must be objects")
    return [dict(row) for row in rows]


def _load_jsonl(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL line {lineno} must be an object")
        rows.append(dict(row))
    return rows


def _detect_format(path: str | Path | None, explicit: str | None) -> str:
    if explicit and explicit != "auto":
        return explicit
    if path is None or str(path) == "-":
        return "jsonl"
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"
    if suffix == ".json":
        return "json"
    return "jsonl"


def load_rows(paths: Iterable[str], *, input_format: str = "auto") -> list[dict[str, Any]]:
    """Load result rows from one or more CSV/JSON/JSONL inputs."""
    selected_paths = list(paths) or ["-"]
    rows: list[dict[str, Any]] = []
    for raw_path in selected_paths:
        if raw_path == "-":
            text, default_format = _read_stdin()
            fmt = input_format if input_format != "auto" else default_format
        else:
            path = Path(raw_path)
            text = path.read_text()
            fmt = _detect_format(path, input_format)
        if fmt == "csv":
            rows.extend(_load_csv(text))
        elif fmt == "json":
            rows.extend(_load_json(text))
        elif fmt == "jsonl":
            rows.extend(_load_jsonl(text))
        else:
            raise ValueError(f"unsupported input format: {fmt}")
    return rows


def _get(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row[key]
    lower = {str(k).lower(): v for k, v in row.items()}
    return lower.get(key.lower(), default)


def _metric_value(row: dict[str, Any], metric: str) -> float | None:
    for key in METRIC_ALIASES.get(metric, (metric,)):
        value = _get(row, key)
        number = _finite_float(value)
        if number is not None:
            return number
    return None


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in NON_NUMERIC:
            return None
        if text.endswith("%"):
            text = text[:-1].strip()
            scale = 0.01
        else:
            scale = 1.0
        try:
            out = float(text.replace(",", "")) * scale
        except ValueError:
            return None
    else:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
    return out if math.isfinite(out) else None


def _seed_value(row: dict[str, Any]) -> int | str | None:
    value = _get(row, "seed")
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(str(value).strip())
    except ValueError:
        return str(value)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _json_number(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _fmt_float(value: float | None) -> str:
    return "null" if value is None else f"{value:.6g}"


def _metric_summary(values: list[float], *, column_present: bool) -> dict[str, Any]:
    mean = _mean(values)
    std = _std(values)
    status = "available" if values else ("empty" if column_present else "not_provided")
    return {
        "n": len(values),
        "mean": _json_number(mean),
        "std": _json_number(std),
        "mean_pm_std": f"{_fmt_float(mean)} +/- {_fmt_float(std)}",
        "status": status,
    }


def _has_column(rows: list[dict[str, Any]], metric: str) -> bool:
    aliases = {key.lower() for key in METRIC_ALIASES.get(metric, (metric,))}
    return any(str(key).lower() in aliases for row in rows for key in row)


def _group_summary(
    rows: list[dict[str, Any]],
    *,
    column_presence: dict[str, bool],
) -> dict[str, Any]:
    seeds = sorted(
        {seed for row in rows if (seed := _seed_value(row)) is not None},
        key=lambda value: (str(type(value)), str(value)),
    )
    control_types = sorted(
        {
            str(control_type)
            for row in rows
            if (control_type := _get(row, "control_type")) not in (None, "")
        }
    )
    metrics: dict[str, dict[str, Any]] = {}
    for metric in SUMMARY_METRICS:
        values = [
            value
            for row in rows
            if (value := _metric_value(row, metric)) is not None
        ]
        metrics[metric] = _metric_summary(
            values,
            column_present=column_presence.get(metric, False),
        )
    return {
        "n_rows": len(rows),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "control_types": control_types,
        "metrics": metrics,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build per-variant and per-variant/per-regime summaries."""
    if not rows:
        raise ValueError("no result rows provided")

    column_presence = {metric: _has_column(rows, metric) for metric in SUMMARY_METRICS}
    by_variant_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_regime_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_variant_regime_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        variant = str(_get(row, "variant", "UNKNOWN") or "UNKNOWN")
        regime = str(_get(row, "regime", "ALL") or "ALL")
        by_variant_rows[variant].append(row)
        by_regime_rows[regime].append(row)
        by_variant_regime_rows[(variant, regime)].append(row)

    by_variant = {
        variant: _group_summary(group_rows, column_presence=column_presence)
        for variant, group_rows in sorted(by_variant_rows.items())
    }
    by_regime = {
        regime: _group_summary(group_rows, column_presence=column_presence)
        for regime, group_rows in sorted(by_regime_rows.items())
    }
    by_variant_regime: dict[str, dict[str, Any]] = defaultdict(dict)
    for (variant, regime), group_rows in sorted(by_variant_regime_rows.items()):
        by_variant_regime[variant][regime] = _group_summary(
            group_rows,
            column_presence=column_presence,
        )

    return {
        "n_rows": len(rows),
        "schema": {
            "metrics": list(SUMMARY_METRICS),
            "std": "sample",
            "dsr_pbo": "passed-through only; not computed by this script",
        },
        "by_variant": by_variant,
        "by_regime": by_regime,
        "by_variant_regime": dict(by_variant_regime),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read post-run A/B result rows and emit per-variant plus "
            "per-regime mean +/- std summaries."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="CSV, JSON, or JSONL result files. Use '-' or omit for stdin.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "json", "jsonl"),
        default="auto",
        help="Input format override. Default: infer from extension.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        rows = load_rows(args.inputs, input_format=args.input_format)
        summary = summarize_rows(rows)
    except Exception as exc:
        parser.error(str(exc))

    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
