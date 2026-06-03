#!/usr/bin/env python3
"""Render Kelly sigma-horizon promotion evidence as a concise Markdown report.

The renderer is intentionally read-only: it loads existing A/B plan and summary
JSON files, plus optional placebo JSON evidence, and never runs backtests or
mutates configs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROMOTION_REGIMES = ("BULL_CALM", "BULL_VOLATILE", "CHOPPY")
CORE_METRICS = ("apy", "sharpe", "maxdd", "cash_pct", "kelly_target_pct")


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _inner_plan(payload: dict[str, Any]) -> dict[str, Any]:
    plan = payload.get("plan")
    return plan if isinstance(plan, dict) else payload


def _promotion_verdict(payload: dict[str, Any]) -> dict[str, Any]:
    verdict = payload.get("promotion_verdict")
    if isinstance(verdict, dict):
        return verdict
    plan = payload.get("plan")
    if isinstance(plan, dict) and isinstance(plan.get("promotion_verdict"), dict):
        return plan["promotion_verdict"]
    return {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt_value(value: Any) -> str:
    if value is None:
        return "not provided"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.6g}"
    return str(value)


def _metric_summary_text(metrics: dict[str, Any], metric: str) -> str:
    row = metrics.get(metric)
    if not isinstance(row, dict):
        return "not provided"
    if row.get("status") == "not_provided":
        return "not provided"
    return str(row.get("mean_pm_std") or _fmt_value(row.get("mean")))


def _variant_seed_count(variants: list[Any]) -> int | None:
    counts = {
        len(variant.get("seeds") or [])
        for variant in variants
        if isinstance(variant, dict) and isinstance(variant.get("seeds"), list)
    }
    return max(counts) if counts else None


def _required_regimes(plan: dict[str, Any], verdict: dict[str, Any]) -> list[str]:
    thresholds = verdict.get("thresholds")
    if isinstance(thresholds, dict) and isinstance(thresholds.get("required_regimes"), list):
        return [str(regime) for regime in thresholds["required_regimes"]]
    mandatory = plan.get("mandatory_checks")
    if isinstance(mandatory, dict) and isinstance(mandatory.get("per_regime"), list):
        return [str(regime) for regime in mandatory["per_regime"]]
    return list(PROMOTION_REGIMES)


def _load_placebo_file(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    interp = payload.get("interpretation") if isinstance(payload.get("interpretation"), dict) else {}
    if "promotion_evidence" in interp:
        passed = bool(interp.get("promotion_evidence"))
        return {
            "path": str(path),
            "passed": passed,
            "promotion_evidence": passed,
            "aligned_real_60_ic": interp.get("aligned_real_60_ic"),
            "placebo_60_ic": interp.get("placebo_60_ic"),
            "label_autocorr_60_ic": interp.get("label_autocorr_60_ic"),
            "primary_warning": interp.get("primary_warning"),
        }
    return {
        "path": str(path),
        "passed": bool(payload.get("passed")),
        "promotion_evidence": payload.get("promotion_evidence"),
        "aligned_real_60_ic": payload.get("aligned_real_60_ic"),
        "placebo_60_ic": payload.get("placebo_60_ic"),
        "label_autocorr_60_ic": payload.get("label_autocorr_60_ic"),
        "primary_warning": payload.get("primary_warning"),
    }


def placebo_status(
    plan_payload: dict[str, Any],
    plan: dict[str, Any],
    placebo_paths: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    placebo = plan_payload.get("placebo")
    if isinstance(placebo, dict) and placebo.get("provided"):
        items = [item for item in _as_list(placebo.get("items")) if isinstance(item, dict)]
        status = "passed" if placebo.get("passed") else "blocked"
        return status, items

    items = [_load_placebo_file(path) for path in placebo_paths]
    if items:
        return ("passed" if all(item.get("passed") for item in items) else "blocked"), items

    planned_paths = _as_list(plan.get("placebo_json"))
    if planned_paths:
        return "listed, not loaded", [{"path": str(path)} for path in planned_paths]

    requirements = plan.get("placebo_requirements")
    if isinstance(requirements, dict) and requirements.get("required"):
        return "required, not provided", []
    return "not provided", []


def _append_variant_lines(lines: list[str], variants: list[Any]) -> None:
    lines.append("## A/B variants")
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        name = _fmt_value(variant.get("name"))
        role = _fmt_value(variant.get("role"))
        seeds = variant.get("seeds") if isinstance(variant.get("seeds"), list) else []
        lines.append(f"- {name}: {role}; seeds={len(seeds)}")
    if not variants:
        lines.append("- not provided")


def _append_verdict_lines(lines: list[str], verdict: dict[str, Any]) -> None:
    lines.append("## Tier 3 verdict")
    if not verdict:
        lines.append("- Tier 3 ready: not provided")
        return
    lines.append(f"- Tier 3 ready: {_fmt_value(verdict.get('tier3_ready'))}")
    reasons = _as_list(verdict.get("blocked_reasons"))
    if reasons:
        lines.append("- Blocked reasons:")
        lines.extend(f"  - {reason}" for reason in reasons)
    else:
        lines.append("- Blocked reasons: none")


def _append_dsr_pbo_lines(lines: list[str], summary: dict[str, Any]) -> None:
    by_variant = summary.get("by_variant") if isinstance(summary.get("by_variant"), dict) else {}
    status_lines: list[str] = []
    for variant, group in sorted(by_variant.items()):
        metrics = group.get("metrics") if isinstance(group, dict) else None
        if not isinstance(metrics, dict):
            continue
        has_dsr = isinstance(metrics.get("dsr"), dict)
        has_pbo = isinstance(metrics.get("pbo"), dict)
        if not has_dsr and not has_pbo:
            continue
        dsr = metrics.get("dsr") if has_dsr else {}
        pbo = metrics.get("pbo") if has_pbo else {}
        if dsr.get("status") == "not_provided" and pbo.get("status") == "not_provided":
            continue
        status_lines.append(
            f"- {variant}: DSR {dsr.get('status', 'not provided')} "
            f"({_metric_summary_text(metrics, 'dsr')}), PBO "
            f"{pbo.get('status', 'not provided')} ({_metric_summary_text(metrics, 'pbo')})"
        )
    if status_lines:
        lines.append("## DSR/PBO status")
        lines.extend(status_lines)


def _append_placebo_lines(lines: list[str], status: str, items: list[dict[str, Any]]) -> None:
    lines.append("## Placebo evidence")
    lines.append(f"- Placebo evidence status: {status}")
    for item in items:
        path = _fmt_value(item.get("path"))
        passed = _fmt_value(item.get("passed", item.get("promotion_evidence")))
        placebo_60 = _fmt_value(item.get("placebo_60_ic"))
        aligned = _fmt_value(item.get("aligned_real_60_ic"))
        warning = item.get("primary_warning")
        suffix = f"; warning={warning}" if warning else ""
        lines.append(
            f"- {path}: passed={passed}; aligned_real_60_ic={aligned}; "
            f"placebo_60_ic={placebo_60}{suffix}"
        )


def _append_regime_sections(lines: list[str], summary: dict[str, Any]) -> None:
    by_variant_regime = summary.get("by_variant_regime")
    if not isinstance(by_variant_regime, dict):
        return
    for regime in PROMOTION_REGIMES:
        rows: list[str] = []
        for variant, regimes in sorted(by_variant_regime.items()):
            if not isinstance(regimes, dict) or not isinstance(regimes.get(regime), dict):
                continue
            group = regimes[regime]
            metrics = group.get("metrics") if isinstance(group.get("metrics"), dict) else {}
            metric_text = "; ".join(
                f"{metric}={_metric_summary_text(metrics, metric)}"
                for metric in CORE_METRICS
            )
            rows.append(
                f"- {variant}: rows={_fmt_value(group.get('n_rows'))}; "
                f"seeds={_fmt_value(group.get('n_seeds'))}; {metric_text}"
            )
        if rows:
            lines.append(f"## {regime} metrics")
            lines.extend(rows)


def render_report(
    *,
    plan_payload: dict[str, Any],
    summary: dict[str, Any],
    placebo_paths: list[str],
) -> str:
    plan = _inner_plan(plan_payload)
    verdict = _promotion_verdict(plan_payload)
    variants = [variant for variant in _as_list(plan.get("variants")) if isinstance(variant, dict)]
    required_regimes = _required_regimes(plan, verdict)
    seed_count = _variant_seed_count(variants)
    placebo_state, placebo_items = placebo_status(plan_payload, plan, placebo_paths)

    lines = [
        "# Kelly Sigma-Horizon Promotion Evidence",
        "",
        f"- Seed count: {_fmt_value(seed_count)}",
        f"- Required regimes: {', '.join(required_regimes) if required_regimes else 'not provided'}",
        "",
    ]
    _append_variant_lines(lines, variants)
    lines.append("")
    _append_verdict_lines(lines, verdict)
    lines.append("")
    _append_dsr_pbo_lines(lines, summary)
    if lines[-1] != "":
        lines.append("")
    _append_placebo_lines(lines, placebo_state, placebo_items)
    lines.append("")
    _append_regime_sections(lines, summary)
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("positional", nargs="*", help="Optional fallback: PLAN_JSON SUMMARY_JSON")
    parser.add_argument("--plan-json", default="", help="Runner plan JSON path")
    parser.add_argument("--summary-json", default="", help="Summarizer JSON path")
    parser.add_argument("--placebo-json", action="append", default=[], help="Optional placebo JSON evidence")
    parser.add_argument("--output", default="", help="Write Markdown report to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    plan_json = args.plan_json or (args.positional[0] if len(args.positional) >= 1 else "")
    summary_json = args.summary_json or (args.positional[1] if len(args.positional) >= 2 else "")
    if not plan_json or not summary_json:
        parser.error("provide --plan-json and --summary-json, or positional PLAN_JSON SUMMARY_JSON")

    try:
        report = render_report(
            plan_payload=load_json(plan_json),
            summary=load_json(summary_json),
            placebo_paths=list(args.placebo_json or []),
        )
    except Exception as exc:
        parser.error(str(exc))

    if args.output:
        Path(args.output).write_text(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
