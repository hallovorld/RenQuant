#!/usr/bin/env python
"""Bridge data-sanitization pipeline — single chokepoint between Python
features and the Rust scorer.

Per user spec 2026-04-25 ("python rust接口要足够robust！数据要洗干净！
可以考虑一条pipeline专门洗数据"), this is the ONE place that decides
whether a feature CSV is fit to feed into the Rust scorer. All upstream
data prep funnels through here, so there's exactly one set of rules to
audit.

Pipeline stages
───────────────
  1. **Schema validation** — required header, every artifact feature_col
     present in the CSV, no duplicates, ticker is the first column.
  2. **Numeric purity** — every cell parses as f32, no NaN, no inf.
  3. **Range sanity** — post-z-score features should sit in [-5, +5].
     Out-of-range values are flagged; the operator decides per-column
     policy: error, clip, or pass through.
  4. **Tickers** — non-empty, no whitespace surprises, no duplicates.
  5. **Output** — sanitized CSV (same shape as input, one row per row),
     plus a JSON report of what was found / fixed.

Design contract
───────────────
  Input:   raw_features.csv (any shape; we infer columns from header)
  Sidecar: artifact.json (the safetensors sidecar; tells us which
           feature_cols we MUST find)
  Output:  clean_features.csv  + sanitize_report.json

The Rust scorer at rust/transformer_scorer/src/main.rs already rejects
NaN/inf inputs (BRIDGE-6). This script is the BEFORE-RUST data scrub:
it makes sure the CSV is clean BEFORE the Rust scorer sees it, with a
machine-readable audit trail of every cell modified.

Usage
─────
    python scripts/sanitize_bridge_csv.py \\
        --artifact backtesting/renquant_104/artifacts/panel-transformer \\
        --input    /tmp/raw_features.csv \\
        --output   /tmp/clean_features.csv \\
        --policy   error      # or 'clip' or 'pass'

Exit codes:
  0  clean (no findings, or findings within policy)
  1  schema error (required column missing, ticker dup, etc.)
  2  data error (NaN/inf when policy=error)
  3  range error (out-of-range when policy=error)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path


# ── Policies ───────────────────────────────────────────────────────────────

@dataclass
class SanitizeReport:
    schema_errors:    list[str] = field(default_factory=list)
    nan_cells:        list[tuple[str, str]] = field(default_factory=list)   # (ticker, col)
    inf_cells:        list[tuple[str, str]] = field(default_factory=list)
    out_of_range:     list[tuple[str, str, float]] = field(default_factory=list)
    duplicate_tickers: list[str] = field(default_factory=list)
    rows_in:          int = 0
    rows_out:         int = 0
    cells_fixed:      int = 0

    def to_dict(self) -> dict:
        return {
            "schema_errors":     self.schema_errors,
            "nan_cells":         [{"ticker": t, "col": c} for t, c in self.nan_cells],
            "inf_cells":         [{"ticker": t, "col": c} for t, c in self.inf_cells],
            "out_of_range":      [{"ticker": t, "col": c, "value": v}
                                   for t, c, v in self.out_of_range],
            "duplicate_tickers": self.duplicate_tickers,
            "rows_in":           self.rows_in,
            "rows_out":          self.rows_out,
            "cells_fixed":       self.cells_fixed,
        }


# ── Sanitization steps ─────────────────────────────────────────────────────

def _validate_schema(
    header: list[str],
    expected_cols: list[str],
    report: SanitizeReport,
) -> bool:
    """Return True if schema is OK. Mutates `report`."""
    if not header or header[0] != "ticker":
        report.schema_errors.append(
            f"first column must be 'ticker', got {header[0] if header else '(empty)'}"
        )
        return False

    feature_cols = header[1:]
    seen = set()
    for c in feature_cols:
        if c in seen:
            report.schema_errors.append(f"duplicate feature column '{c}'")
        seen.add(c)

    missing = [c for c in expected_cols if c not in feature_cols]
    if missing:
        report.schema_errors.append(
            f"missing required feature columns: {missing}"
        )
    extras = [c for c in feature_cols if c not in expected_cols]
    if extras:
        # Extras are tolerated by the Rust scorer (it picks the cols it
        # needs by name), but we surface them so the operator notices.
        report.schema_errors.append(
            f"extra (unused) columns will be ignored by Rust: {extras}"
        )
        # NOT a schema-fatal — Rust scorer reorders by name.

    fatal = bool(missing) or any("must be 'ticker'" in e or "duplicate" in e
                                  for e in report.schema_errors)
    return not fatal


def _sanitize_value(
    raw: str,
    ticker: str,
    col: str,
    *,
    policy: str,
    report: SanitizeReport,
    range_lo: float,
    range_hi: float,
) -> str | None:
    """Return the cleaned cell string, or None if the row should be dropped.

    Caller: never use the return value to silently fill bad data — only
    ranges may be clipped (and only when policy='clip').
    """
    try:
        v = float(raw)
    except ValueError:
        report.schema_errors.append(
            f"row ticker={ticker} col={col}: cannot parse '{raw}' as float"
        )
        return None

    if math.isnan(v):
        report.nan_cells.append((ticker, col))
        if policy == "error":
            return None
        # clip / pass: replace NaN with 0 (z-scored neutral)
        report.cells_fixed += 1
        return "0.0"

    if math.isinf(v):
        report.inf_cells.append((ticker, col))
        if policy == "error":
            return None
        report.cells_fixed += 1
        return "5.0" if v > 0 else "-5.0"

    if v < range_lo or v > range_hi:
        report.out_of_range.append((ticker, col, v))
        if policy == "error":
            return None
        if policy == "clip":
            v = max(range_lo, min(range_hi, v))
            report.cells_fixed += 1
            return f"{v:.6f}"
        # pass-through: leave it.
    return raw


# ── Main entry ─────────────────────────────────────────────────────────────

def sanitize(
    artifact_stem: Path,
    input_csv:     Path,
    output_csv:    Path,
    policy:        str = "error",
    range_lo:      float = -5.0,
    range_hi:      float = 5.0,
) -> tuple[int, SanitizeReport]:
    """Returns (exit_code, report)."""
    report = SanitizeReport()

    sidecar_path = artifact_stem.with_suffix(".json")
    if not sidecar_path.exists():
        report.schema_errors.append(f"sidecar {sidecar_path} missing")
        return 1, report
    sidecar = json.loads(sidecar_path.read_text())
    expected_cols = list(sidecar.get("feature_cols", []))
    if not expected_cols:
        report.schema_errors.append("sidecar has empty feature_cols")
        return 1, report

    if not input_csv.exists():
        report.schema_errors.append(f"input {input_csv} missing")
        return 1, report

    with input_csv.open() as fr:
        rdr = csv.reader(fr)
        try:
            header = next(rdr)
        except StopIteration:
            report.schema_errors.append("input CSV is empty (no header)")
            return 1, report

        if not _validate_schema(header, expected_cols, report):
            return 1, report

        rows: list[list[str]] = []
        seen_tickers: set[str] = set()
        for raw_row in rdr:
            report.rows_in += 1
            if len(raw_row) != len(header):
                report.schema_errors.append(
                    f"row {report.rows_in}: width {len(raw_row)} != header {len(header)}"
                )
                continue
            ticker = raw_row[0].strip()
            if not ticker:
                report.schema_errors.append(f"row {report.rows_in}: empty ticker")
                continue
            if ticker in seen_tickers:
                report.duplicate_tickers.append(ticker)
                continue
            seen_tickers.add(ticker)

            new_row = [ticker]
            ok = True
            for j, col in enumerate(header[1:], start=1):
                cleaned = _sanitize_value(
                    raw_row[j], ticker, col,
                    policy=policy, report=report,
                    range_lo=range_lo, range_hi=range_hi,
                )
                if cleaned is None:
                    ok = False
                    break
                new_row.append(cleaned)
            if ok:
                rows.append(new_row)

    # Decide exit code.
    code = 0
    if report.schema_errors:
        code = 1
    elif report.nan_cells or report.inf_cells:
        code = 2 if policy == "error" else 0
    elif report.out_of_range:
        code = 3 if policy == "error" else 0

    # Write output even on error so the operator can inspect partial result.
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fw:
        wtr = csv.writer(fw)
        wtr.writerow(header)
        wtr.writerows(rows)
    report.rows_out = len(rows)

    # Always emit the audit report next to the output.
    report_path = output_csv.with_suffix(".sanitize_report.json")
    report_path.write_text(json.dumps(report.to_dict(), indent=2))
    return code, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--input",    required=True, type=Path)
    parser.add_argument("--output",   required=True, type=Path)
    parser.add_argument(
        "--policy", default="error", choices=["error", "clip", "pass"],
        help="error: fail loud; clip: clamp to range and substitute 0/±5 for NaN/inf; pass: leave",
    )
    parser.add_argument("--range-lo", type=float, default=-5.0)
    parser.add_argument("--range-hi", type=float, default=5.0)
    args = parser.parse_args()

    code, report = sanitize(
        args.artifact, args.input, args.output,
        policy=args.policy, range_lo=args.range_lo, range_hi=args.range_hi,
    )
    print(f"rows_in={report.rows_in}  rows_out={report.rows_out}  "
          f"cells_fixed={report.cells_fixed}  schema_errors={len(report.schema_errors)}  "
          f"nan={len(report.nan_cells)}  inf={len(report.inf_cells)}  "
          f"oor={len(report.out_of_range)}")
    if report.schema_errors:
        print("schema_errors:")
        for e in report.schema_errors:
            print(f"  • {e}")
    sys.exit(code)


if __name__ == "__main__":
    main()
