#!/usr/bin/env python3
"""Read-only Kelly sizing diagnostics for the 2026-06-03 audit.

The tool is intentionally narrow: parse operator logs, CSV/JSON metric
exports, and live-state snapshots, then summarize whether Kelly targets and
cash exposure look too small before promoting a production config change.

Examples:

    python scripts/diagnose_kelly_sizing.py \
        --log 'logs/live_e2e/*.log' \
        --data /tmp/decision_trace.csv \
        --state /tmp/live_state_snapshots.jsonl

    python scripts/diagnose_kelly_sizing.py --format json --data sim_orders.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_SPECS = (
    "logs/live_e2e/*.log",
    "logs/live_e2e/**/*.log",
)
DEFAULT_STATE_SPECS = (
    "backtesting/renquant_104/live_state.alpaca.json",
    "backtesting/renquant_104/live_state.alpaca_shadow.json",
)

KELLY_LOG_RE = re.compile(
    r"ApplyKellySizingTask: .*?"
    r"fractional=(?P<fractional>[-+0-9.eE]+)\s+"
    r"max_conc=(?P<max_conc>[-+0-9.eE]+)\s+"
    r"cands=(?P<cand_nz>\d+)"
    r"(?:/(?P<cand_total>\d+))?\s+non-zero\s+"
    r"\(avg=(?P<cand_avg_pct>[-+0-9.eE]+)%\)\s+"
    r"holdings=(?P<held_nz>\d+)"
    r"(?:/(?P<held_total>\d+))?\s+non-zero\s+"
    r"\(avg=(?P<held_avg_pct>[-+0-9.eE]+)%\)"
    r"(?:\s+zero_reasons\[(?P<zero_reasons>[^\]]*)\])?"
)
DATE_RE = re.compile(r"\b(?P<date>\d{4}-\d{2}-\d{2})\b")

KELLY_KEYS = (
    "kelly_target_pct",
    "entry_kelly_target_pct",
    "exit_kelly_target_pct",
    "kelly_target",
    "kelly_pct",
    "target_w",
)
SIGMA_KEYS = (
    "sigma",
    "c.sigma",
    "candidate_sigma",
)
CASH_PCT_KEYS = (
    "cash_pct",
    "cash_percent",
    "cash%",
    "cash_ratio",
    "cash_weight",
)
CASH_KEYS = ("cash", "available_cash", "cash_actual")
PORTFOLIO_VALUE_KEYS = (
    "portfolio_value",
    "nav",
    "account_value",
    "equity",
)
REGIME_KEYS = ("regime", "market_regime")
DATE_KEYS = ("date", "run_date", "trade_date", "as_of_date", "timestamp")
ROW_CONTAINER_KEYS = (
    "rows",
    "data",
    "records",
    "items",
    "candidate_scores",
    "decision_trace",
    "sim_orders",
    "live_state_snapshots",
    "entry_signals",
)


def _flatten(groups: list[list[str]] | None) -> list[str]:
    if not groups:
        return []
    return [item for group in groups for item in group]


def _resolve_spec(root: Path, spec: str) -> str:
    expanded = Path(spec).expanduser()
    if expanded.is_absolute():
        return str(expanded)
    return str(root / expanded)


def expand_specs(
    root: Path,
    specs: list[str],
    default_specs: tuple[str, ...] = (),
) -> tuple[list[Path], list[str]]:
    """Expand path/glob specs into existing files and human-readable warnings."""
    selected = specs or list(default_specs)
    warnings: list[str] = []
    paths: list[Path] = []

    for spec in selected:
        resolved = _resolve_spec(root, spec)
        if glob.has_magic(resolved):
            matches = [Path(p) for p in glob.glob(resolved, recursive=True)]
        else:
            matches = [Path(resolved)] if Path(resolved).exists() else []
        files = sorted(p for p in matches if p.is_file())
        if not files:
            warnings.append(f"input did not match any files: {spec}")
        paths.extend(files)

    seen: set[Path] = set()
    deduped: list[Path] = []
    for path in paths:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped, warnings


def _to_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
            return None
        text = text.rstrip("%").replace(",", "")
        try:
            out = float(text)
        except ValueError:
            return None
    return out if math.isfinite(out) else None


def _norm_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _get_value(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    normed = {_norm_key(str(k)): v for k, v in row.items()}
    for key in keys:
        if _norm_key(key) in normed:
            return normed[_norm_key(key)]
    return None


def _as_pct(value: float) -> float:
    return value * 100.0 if abs(value) <= 1.0 else value


def _sigma_as_pct(value: float) -> float:
    return value * 100.0 if abs(value) <= 2.0 else value


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[int(pos)]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "min": ordered[0],
        "p10": _percentile(ordered, 10),
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p90": _percentile(ordered, 90),
        "max": ordered[-1],
    }


def _bucket_counts(values: list[float], edges: list[tuple[str, float | None, float | None]]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in edges}
    for value in values:
        for label, lo, hi in edges:
            if (lo is None or value >= lo) and (hi is None or value < hi):
                counts[label] += 1
                break
    return counts


def _extract_date(row: dict[str, Any]) -> str | None:
    value = _get_value(row, DATE_KEYS)
    if value is None:
        return None
    match = DATE_RE.search(str(value))
    return match.group("date") if match else str(value)


def _has_known_metric(row: dict[str, Any]) -> bool:
    for keys in (KELLY_KEYS, SIGMA_KEYS, CASH_PCT_KEYS, CASH_KEYS):
        if _get_value(row, keys) is not None:
            return True
    return False


def _rows_from_json_obj(obj: Any) -> list[dict[str, Any]]:
    if isinstance(obj, list):
        rows: list[dict[str, Any]] = []
        for item in obj:
            rows.extend(_rows_from_json_obj(item))
        return rows

    if not isinstance(obj, dict):
        return []

    rows = []
    for key in ROW_CONTAINER_KEYS:
        child = obj.get(key)
        if isinstance(child, list):
            rows.extend(row for row in child if isinstance(row, dict))
        elif isinstance(child, dict):
            rows.extend(row for row in child.values() if isinstance(row, dict))
    if rows:
        return rows

    if _has_known_metric(obj):
        return [obj]

    nested: list[dict[str, Any]] = []
    for child in obj.values():
        nested.extend(_rows_from_json_obj(child))
    return nested


def read_metric_rows(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as fh:
                return list(csv.DictReader(fh)), warnings
        if suffix in {".jsonl", ".ndjson"}:
            rows: list[dict[str, Any]] = []
            for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    warnings.append(f"{path}:{lineno}: invalid JSONL row: {exc}")
                    continue
                rows.extend(_rows_from_json_obj(obj))
            return rows, warnings
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            return _rows_from_json_obj(obj), warnings
    except (OSError, json.JSONDecodeError) as exc:
        return [], [f"{path}: cannot read metric rows: {exc}"]

    return [], [f"{path}: unsupported input extension {suffix or '<none>'}"]


def _parse_zero_reasons(raw: str | None) -> Counter[str]:
    counts: Counter[str] = Counter()
    if not raw:
        return counts
    for token in raw.split():
        if "=" not in token:
            continue
        key, value = token.rsplit("=", 1)
        try:
            counts[key] += int(value)
        except ValueError:
            continue
    return counts


def parse_kelly_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return [], [f"{path}: cannot read log: {exc}"]

    for lineno, line in enumerate(lines, 1):
        match = KELLY_LOG_RE.search(line)
        if not match:
            continue
        date_match = DATE_RE.search(line)
        record = {
            "source": str(path),
            "line": lineno,
            "date": date_match.group("date") if date_match else None,
            "fractional": _to_float(match.group("fractional")),
            "max_conc": _to_float(match.group("max_conc")),
            "cand_nz": int(match.group("cand_nz")),
            "cand_total": int(match.group("cand_total")) if match.group("cand_total") else None,
            "cand_avg_pct": _to_float(match.group("cand_avg_pct")),
            "held_nz": int(match.group("held_nz")),
            "held_total": int(match.group("held_total")) if match.group("held_total") else None,
            "held_avg_pct": _to_float(match.group("held_avg_pct")),
            "zero_reasons": dict(_parse_zero_reasons(match.group("zero_reasons"))),
        }
        records.append(record)

    if not records:
        warnings.append(f"{path}: no ApplyKellySizingTask summary lines found")
    return records, warnings


def summarize_log_records(records: list[dict[str, Any]], recent_bars: int) -> dict[str, Any]:
    ordered = sorted(records, key=lambda r: (r.get("date") or "", str(r.get("source")), int(r.get("line") or 0)))
    recent = ordered[-recent_bars:] if recent_bars > 0 else ordered
    cand_totals = [(r["cand_nz"], r["cand_total"]) for r in recent if r.get("cand_total")]
    held_totals = [(r["held_nz"], r["held_total"]) for r in recent if r.get("held_total")]
    zero_reasons: Counter[str] = Counter()
    for record in recent:
        zero_reasons.update(record.get("zero_reasons") or {})

    def _ratio_summary(pairs: list[tuple[int, int | None]]) -> dict[str, Any]:
        valid = [(nz, int(total)) for nz, total in pairs if total]
        ratios = [(nz / total) * 100.0 for nz, total in valid if total > 0]
        return {
            "records": len(valid),
            "nonzero_sum": sum(nz for nz, _ in valid),
            "total_sum": sum(total for _, total in valid),
            "ratio_pct": _numeric_summary(ratios),
        }

    return {
        "records": len(records),
        "recent_records": len(recent),
        "recent_bars": recent_bars,
        "files": sorted({str(r["source"]) for r in records}),
        "date_min": min((r["date"] for r in recent if r.get("date")), default=None),
        "date_max": max((r["date"] for r in recent if r.get("date")), default=None),
        "candidates": _ratio_summary(cand_totals),
        "holdings": _ratio_summary(held_totals),
        "candidate_avg_kelly_pct": _numeric_summary(
            [float(r["cand_avg_pct"]) for r in recent if r.get("cand_avg_pct") is not None]
        ),
        "holding_avg_kelly_pct": _numeric_summary(
            [float(r["held_avg_pct"]) for r in recent if r.get("held_avg_pct") is not None]
        ),
        "zero_reasons": dict(sorted(zero_reasons.items())),
    }


def summarize_metric_rows(rows: list[dict[str, Any]], recent_bars: int) -> dict[str, Any]:
    kelly_rows: list[dict[str, Any]] = []
    sigma_values: list[float] = []
    cash_rows: list[dict[str, Any]] = []

    for row in rows:
        kelly = _to_float(_get_value(row, KELLY_KEYS))
        if kelly is not None:
            kelly_rows.append({
                "date": _extract_date(row),
                "regime": str(_get_value(row, REGIME_KEYS) or "(unknown)"),
                "kelly_pct": _as_pct(kelly),
            })

        sigma = _to_float(_get_value(row, SIGMA_KEYS))
        if sigma is not None:
            sigma_values.append(_sigma_as_pct(sigma))

        cash_pct = _to_float(_get_value(row, CASH_PCT_KEYS))
        cash = _to_float(_get_value(row, CASH_KEYS))
        portfolio_value = _to_float(_get_value(row, PORTFOLIO_VALUE_KEYS))
        if cash_pct is None and cash is not None and portfolio_value and portfolio_value > 0:
            cash_pct = (cash / portfolio_value) * 100.0
        elif cash_pct is not None:
            cash_pct = _as_pct(cash_pct)
        if cash_pct is not None:
            cash_rows.append({
                "date": _extract_date(row),
                "regime": str(_get_value(row, REGIME_KEYS) or "(unknown)"),
                "cash_pct": cash_pct,
            })

    kelly_values = [row["kelly_pct"] for row in kelly_rows]
    kelly_by_regime: dict[str, dict[str, Any]] = {}
    for regime in sorted({row["regime"] for row in kelly_rows}):
        vals = [row["kelly_pct"] for row in kelly_rows if row["regime"] == regime]
        kelly_by_regime[regime] = {
            "summary_pct": _numeric_summary(vals),
            "nonzero": sum(1 for value in vals if value > 0),
            "histogram_pct": _bucket_counts(
                vals,
                [
                    ("0", None, 1e-12),
                    ("0..2", 1e-12, 2.0),
                    ("2..5", 2.0, 5.0),
                    ("5..10", 5.0, 10.0),
                    ("10..15", 10.0, 15.0),
                    (">=15", 15.0, None),
                ],
            ),
        }

    sigma_summary = {
        "summary_pct": _numeric_summary(sigma_values),
        "histogram_pct": _bucket_counts(
            sigma_values,
            [
                ("<10", None, 10.0),
                ("10..20", 10.0, 20.0),
                ("20..30", 20.0, 30.0),
                ("30..50", 30.0, 50.0),
                ("50..75", 50.0, 75.0),
                (">=75", 75.0, None),
            ],
        ),
    }

    ordered_cash = sorted(cash_rows, key=lambda r: (r.get("date") or ""))
    recent_cash = ordered_cash[-recent_bars:] if recent_bars > 0 else ordered_cash
    cash_values = [row["cash_pct"] for row in recent_cash]

    return {
        "rows": len(rows),
        "kelly": {
            "summary_pct": _numeric_summary(kelly_values),
            "nonzero": sum(1 for value in kelly_values if value > 0),
            "by_regime": kelly_by_regime,
        },
        "sigma": sigma_summary,
        "cash": {
            "recent_bars": recent_bars,
            "rows": len(cash_rows),
            "recent_rows": len(recent_cash),
            "summary_pct": _numeric_summary(cash_values),
            "latest": recent_cash[-1] if recent_cash else None,
        },
    }


def _read_rows_from_paths(paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for path in paths:
        file_rows, file_warnings = read_metric_rows(path)
        rows.extend(file_rows)
        warnings.extend(file_warnings)
    return rows, warnings


def run_diagnostic(
    *,
    root: Path,
    log_specs: list[str] | None = None,
    data_specs: list[str] | None = None,
    state_specs: list[str] | None = None,
    use_defaults: bool = True,
    recent_bars: int = 30,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    log_defaults = DEFAULT_LOG_SPECS if use_defaults else ()
    state_defaults = DEFAULT_STATE_SPECS if use_defaults else ()

    log_paths, warnings = expand_specs(root, log_specs or [], log_defaults)
    data_paths, data_warnings = expand_specs(root, data_specs or [], ())
    state_paths, state_warnings = expand_specs(root, state_specs or [], state_defaults)
    warnings.extend(data_warnings)
    warnings.extend(state_warnings)

    log_records: list[dict[str, Any]] = []
    for path in log_paths:
        records, file_warnings = parse_kelly_log(path)
        log_records.extend(records)
        warnings.extend(file_warnings)

    data_rows, row_warnings = _read_rows_from_paths(data_paths)
    state_rows, state_row_warnings = _read_rows_from_paths(state_paths)
    warnings.extend(row_warnings)
    warnings.extend(state_row_warnings)
    metric_rows = data_rows + state_rows

    logs = summarize_log_records(log_records, recent_bars)
    metrics = summarize_metric_rows(metric_rows, recent_bars)
    ok = bool(
        logs["records"] > 0
        or metrics["kelly"]["summary_pct"]["n"] > 0
        or metrics["sigma"]["summary_pct"]["n"] > 0
        or metrics["cash"]["summary_pct"]["n"] > 0
    )
    if not ok:
        warnings.append(
            "no usable Kelly/cash diagnostics found; provide --log, --data, "
            "or --state with ApplyKellySizingTask, kelly_target_pct, sigma, "
            "cash_pct, or cash+portfolio_value fields"
        )

    return {
        "ok": ok,
        "root": str(root),
        "inputs": {
            "logs": [str(path) for path in log_paths],
            "data": [str(path) for path in data_paths],
            "state": [str(path) for path in state_paths],
        },
        "warnings": warnings,
        "logs": logs,
        "metrics": metrics,
    }


def _fmt_pct(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2f}%"


def _fmt_summary(summary: dict[str, Any]) -> str:
    if summary.get("n", 0) == 0:
        return "n=0"
    return (
        f"n={summary['n']} median={_fmt_pct(summary.get('median'))} "
        f"p10={_fmt_pct(summary.get('p10'))} p90={_fmt_pct(summary.get('p90'))} "
        f"min={_fmt_pct(summary.get('min'))} max={_fmt_pct(summary.get('max'))}"
    )


def render_text(result: dict[str, Any]) -> str:
    lines = ["Kelly sizing diagnostic (read-only)"]
    if not result["ok"]:
        lines.append("Status: FAIL - no usable diagnostic data found.")
    else:
        lines.append("Status: OK - parsed at least one diagnostic source.")

    if result["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for warning in result["warnings"]:
            lines.append(f"  - {warning}")

    logs = result["logs"]
    lines.append("")
    lines.append("ApplyKellySizingTask logs:")
    lines.append(
        f"  files={len(logs['files'])} records={logs['records']} "
        f"recent_records={logs['recent_records']}"
    )
    if logs.get("date_min") or logs.get("date_max"):
        lines.append(f"  dates={logs.get('date_min') or '?'}..{logs.get('date_max') or '?'}")
    cand = logs["candidates"]
    held = logs["holdings"]
    lines.append(
        f"  cands non-zero/total: sum={cand['nonzero_sum']}/{cand['total_sum']} "
        f"ratio={_fmt_summary(cand['ratio_pct'])}"
    )
    lines.append(
        f"  holdings non-zero/total: sum={held['nonzero_sum']}/{held['total_sum']} "
        f"ratio={_fmt_summary(held['ratio_pct'])}"
    )
    lines.append(
        f"  avg candidate Kelly target: {_fmt_summary(logs['candidate_avg_kelly_pct'])}"
    )
    lines.append(
        f"  avg holding Kelly target: {_fmt_summary(logs['holding_avg_kelly_pct'])}"
    )
    if logs["zero_reasons"]:
        reasons = ", ".join(f"{key}={value}" for key, value in logs["zero_reasons"].items())
        lines.append(f"  zero reasons: {reasons}")

    metrics = result["metrics"]
    lines.append("")
    lines.append("kelly_target_pct rows:")
    kelly = metrics["kelly"]
    lines.append(
        f"  all regimes: {_fmt_summary(kelly['summary_pct'])} "
        f"nonzero={kelly['nonzero']}"
    )
    for regime, row in kelly["by_regime"].items():
        hist = ", ".join(f"{label}={count}" for label, count in row["histogram_pct"].items())
        lines.append(
            f"  {regime}: {_fmt_summary(row['summary_pct'])} "
            f"nonzero={row['nonzero']} hist[{hist}]"
        )

    sigma = metrics["sigma"]
    lines.append("")
    lines.append("sigma rows:")
    sigma_hist = ", ".join(f"{label}={count}" for label, count in sigma["histogram_pct"].items())
    lines.append(f"  annualized sigma: {_fmt_summary(sigma['summary_pct'])} hist[{sigma_hist}]")

    cash = metrics["cash"]
    lines.append("")
    lines.append("cash time series:")
    lines.append(
        f"  rows={cash['rows']} recent_rows={cash['recent_rows']} "
        f"cash_pct={_fmt_summary(cash['summary_pct'])}"
    )
    if cash["latest"]:
        latest = cash["latest"]
        lines.append(
            f"  latest: date={latest.get('date') or '?'} "
            f"regime={latest.get('regime') or '?'} cash={_fmt_pct(latest.get('cash_pct'))}"
        )

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log",
        dest="log_specs",
        action="append",
        nargs="+",
        default=[],
        help="Log path or glob. Repeatable. Defaults to logs/live_e2e/*.log.",
    )
    parser.add_argument(
        "--data",
        dest="data_specs",
        action="append",
        nargs="+",
        default=[],
        help="CSV/JSON/JSONL with kelly_target_pct and optional regime/sigma columns.",
    )
    parser.add_argument(
        "--state",
        dest="state_specs",
        action="append",
        nargs="+",
        default=[],
        help="CSV/JSON/JSONL live-state rows with cash_pct or cash+portfolio_value.",
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Only read files explicitly passed via --log/--data/--state.",
    )
    parser.add_argument(
        "--recent-bars",
        type=int,
        default=30,
        help="Number of most recent parsed log/state rows to summarize for recent views.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = run_diagnostic(
        root=ROOT,
        log_specs=_flatten(args.log_specs),
        data_specs=_flatten(args.data_specs),
        state_specs=_flatten(args.state_specs),
        use_defaults=not args.no_defaults,
        recent_bars=args.recent_bars,
    )
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_text(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
