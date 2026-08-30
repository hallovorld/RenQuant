#!/usr/bin/env python3
"""Compose (and optionally send) the daily_104 BUY-BLOCKED alert — with the WHY.

Until 2026-08-30 the wrapper's fallback alert said only "Full run blocked new
buys; sell-only fallback completed", default priority, on a 6 h cooldown. On
Mon 2026-08-31 the served panel artifact (trained 2026-08-02,
``wf_gate_metadata.passed=false``, ``promotion_basis=freshness_fallback_rfc210``)
turns 29 d > the 28 d RFC #210 serving SLA, the license refuses, P-WF-GATE
hard-fails the 13:55 full run and the book goes sell-only. The operator would
have received a low-priority line that names no cause, no artifact, no age
and no path back to buying.

This helper is READ-ONLY on the artifact and the config. It states:

  * the served artifact's trained date + age, ``wf_gate_metadata.passed``,
    ``promotion_basis`` and the genuine IC the fallback promotion recorded;
  * the license verdict text — the preflight's own ``P-WF-GATE`` line from the
    full-run log when available (that line IS the verdict the run acted on),
    plus the RFC #210 evaluator's reason when ``renquant_pipeline`` is on the
    path (it is, under daily_104.sh's exported PYTHONPATH);
  * that exits continue via the 06:30-13:00 sell-only loop;
  * that buys resume only when a candidate is promoted, with the last exit
    status of the two promotion launchd jobs (``launchctl list``).

``--send`` posts through ``renquant_common.notify.send`` with the urgent
headers; daily_104.sh falls back to curl WITH the same headers when this
process reports the sender is unreachable (exit 3) or the POST failed (exit 4).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

ALERT_TITLE = "RenQuant 104 BUY-BLOCKED (sell-only fallback)"
ALERT_PRIORITY = "urgent"
ALERT_TAGS = "rotating_light,rq104"
DEFAULT_LAUNCHD_LABELS = (
    "com.renquant.weekly-wf-promote",
    "com.renquant.retrain-panel104",
)
EXIT_SENDER_UNAVAILABLE = 3
EXIT_SEND_FAILED = 4

_PREFLIGHT_LINE_RE = re.compile(r"\b(P-[A-Z0-9-]+)\b")


# ── artifact facts (read-only) ─────────────────────────────────────────────

def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def served_artifact_path(config: dict, strategy_dir: Path) -> Path:
    """The path daily_104.sh already resolves for its 'Active model' line."""
    panel = (config.get("ranking") or {}).get("panel_scoring") or {}
    rel = panel.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = Path(rel)
    return p if p.is_absolute() else Path(strategy_dir) / p


def load_artifact(path: Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _fmt_ic(value: object) -> str:
    return f"{value:+.5f}" if isinstance(value, (int, float)) and not isinstance(value, bool) else "n/a"


def served_summary(payload: dict, *, today: dt.date) -> dict:
    """Facts the alert states about the artifact. Every value is READ, never assumed."""
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    wf = meta.get("wf_gate_metadata") if isinstance(meta.get("wf_gate_metadata"), dict) else {}
    trained_raw = payload.get("trained_date") or meta.get("trained_date")
    age_days: int | None = None
    if isinstance(trained_raw, str):
        try:
            age_days = (today - dt.date.fromisoformat(trained_raw.strip())).days
        except ValueError:
            age_days = None
    genuine_ic = meta.get("fallback_genuine_ic")
    if genuine_ic is None:
        genuine_ic = wf.get("sanity_placebo_genuine_ic")
    return {
        "trained_date": trained_raw if isinstance(trained_raw, str) else None,
        "age_days": age_days,
        "wf_gate_passed": wf.get("passed"),
        "promotion_basis": meta.get("promotion_basis", payload.get("promotion_basis")),
        "genuine_ic": genuine_ic,
        "wf_reason": wf.get("wf_reason"),
    }


def license_verdict(payload: dict, config: dict, *, today: dt.date) -> str:
    """RFC #210 evaluator's reason, when renquant_pipeline is importable.

    The 28-day policy is pipeline-owned (kernel.rfc210_license); this helper
    consults it rather than re-encoding it here.
    """
    try:
        from renquant_pipeline.kernel.rfc210_license import (  # noqa: PLC0415
            evaluate_freshness_fallback_license,
        )
    except Exception:  # noqa: BLE001 — umbrella-only runtime
        return "license evaluator unavailable (renquant_pipeline not importable)"
    lic = evaluate_freshness_fallback_license(payload, config=config, today=today)
    return ("SERVED: " if lic.served else "REFUSED: ") + lic.reason


def preflight_verdict_line(log_text: str) -> str | None:
    """The last FAILED (✗) P-* line the preflight printed, minus the log prefix.

    Falls back to the last P-WF-GATE line of any kind so a licensed ✓ line is
    still quoted when no ✗ line was captured.
    """
    failed: str | None = None
    wf_gate: str | None = None
    for line in str(log_text).splitlines():
        m = _PREFLIGHT_LINE_RE.search(line)
        if not m:
            continue
        stripped = line.strip()
        idx = stripped.find(m.group(1))
        prefix = stripped[:idx]
        marker = "✗ " if "✗" in prefix else ("✓ " if "✓" in prefix else "")
        rendered = marker + stripped[idx:]
        if marker == "✗ ":
            failed = rendered
        if m.group(1) == "P-WF-GATE":
            wf_gate = rendered
    return failed or wf_gate


def blocking_gate(preflight_line: str | None) -> str:
    m = _PREFLIGHT_LINE_RE.search(preflight_line or "")
    return m.group(1) if m and (preflight_line or "").startswith("✗") else "a buy-side preflight gate"


# ── promotion-job status (read-only) ───────────────────────────────────────

def parse_launchctl_list(text: str, labels: Iterable[str]) -> dict[str, str]:
    """``launchctl list`` rows are ``PID<TAB>LastExitStatus<TAB>Label``."""
    wanted = {str(label): "not loaded" for label in labels}
    for line in str(text).splitlines():
        parts = line.split("\t") if "\t" in line else line.split()
        if len(parts) < 3:
            continue
        label = parts[-1].strip()
        if label in wanted:
            wanted[label] = parts[1].strip() or "unknown"
    return wanted


def launchd_last_exit(labels: Iterable[str], launchctl_output: str | None = None) -> dict[str, str]:
    labels = tuple(labels)
    if launchctl_output is None:
        try:
            launchctl_output = subprocess.run(
                ["launchctl", "list"], capture_output=True, text=True, timeout=10, check=False,
            ).stdout
        except Exception:  # noqa: BLE001 — never let a status read break the alert
            return {label: "unknown" for label in labels}
    return parse_launchctl_list(launchctl_output, labels)


# ── composition ────────────────────────────────────────────────────────────

def compose_body(
    summary: dict,
    *,
    verdict: str,
    preflight_line: str | None,
    job_status: dict[str, str],
    artifact_path: Path | str | None = None,
    holdings: str = "",
    log_path: str = "",
) -> str:
    age = summary.get("age_days")
    age_s = f"{age}d old" if isinstance(age, int) else "age unknown"
    trained = summary.get("trained_date") or "trained_date missing"
    lines = [
        f"New buys BLOCKED by {blocking_gate(preflight_line)}; the full run was rerun --sell-only.",
        f"Served artifact: trained {trained} ({age_s})"
        + (f" — {artifact_path}" if artifact_path else ""),
        f"wf_gate_metadata.passed={json.dumps(summary.get('wf_gate_passed'))} "
        f"promotion_basis={summary.get('promotion_basis')} "
        f"genuine_ic={_fmt_ic(summary.get('genuine_ic'))}",
        f"License: {verdict}",
    ]
    if preflight_line:
        lines.append(f"Preflight: {preflight_line}")
    if summary.get("wf_reason"):
        lines.append(f"WF stamp: {summary['wf_reason']}")
    lines.append("Exits continue via the 06:30-13:00 sell-only loop; risk controls stay armed.")
    status = ", ".join(f"{label.replace('com.renquant.', '')} last exit status: {code}"
                       for label, code in job_status.items())
    lines.append(
        "Buys resume only when a candidate is promoted "
        f"(weekly-wf-promote / retrain-panel104 — {status})."
    )
    if holdings:
        lines.append(holdings)
    if log_path:
        lines.append(f"Log: {log_path}")
    return "\n".join(lines)


def alert_headers(title: str = ALERT_TITLE) -> dict[str, str]:
    return {"Title": title, "Priority": ALERT_PRIORITY, "Tags": ALERT_TAGS}


def send_alert(title: str, body: str, *, sender=None, suppressed=None) -> int:
    """POST via renquant_common.notify.send. Returns the process exit code.

    0 = posted, or deliberately suppressed by RENQUANT_NO_NOTIFY (handled —
    the wrapper must NOT curl around a suppression); 3 = renquant_common not
    importable; 4 = the POST failed. Diagnostics go to stderr so stdout stays
    the body the wrapper can re-send.
    """
    if sender is None or suppressed is None:
        try:
            from renquant_common.notify import (  # noqa: PLC0415
                notifications_suppressed as _suppressed,
                send as _send,
            )
        except Exception:  # noqa: BLE001
            return EXIT_SENDER_UNAVAILABLE
        sender = sender or _send
        suppressed = suppressed or _suppressed
    if suppressed():
        print(f"[ntfy suppressed] {title}", file=sys.stderr)
        return 0
    ok = sender(title, body, priority=ALERT_PRIORITY, tags=ALERT_TAGS)
    if not ok:
        print(f"[ntfy send failed] {title}", file=sys.stderr)
    return 0 if ok else EXIT_SEND_FAILED


# ── CLI ────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--strategy-config", required=True, type=Path)
    ap.add_argument("--strategy-dir", required=True, type=Path)
    ap.add_argument("--full-run-log", type=Path, default=None,
                    help="captured full-run output; its P-WF-GATE line is quoted")
    ap.add_argument("--preflight-lines", default="",
                    help="already-extracted preflight lines (alternative to --full-run-log)")
    ap.add_argument("--today", default=None, help="YYYY-MM-DD (default: local today)")
    ap.add_argument("--holdings", default="")
    ap.add_argument("--log-path", default="")
    ap.add_argument("--labels", default=",".join(DEFAULT_LAUNCHD_LABELS))
    ap.add_argument("--launchctl-output", default=None,
                    help="stdin-free override for tests; default reads `launchctl list`")
    ap.add_argument("--send", action="store_true",
                    help="POST the alert through renquant_common.notify.send")
    ap.add_argument("--json", action="store_true", help="print title/headers/body as JSON")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    config = load_config(args.strategy_config)
    artifact_path = served_artifact_path(config, args.strategy_dir)
    try:
        payload = load_artifact(artifact_path)
        summary = served_summary(payload, today=today)
        verdict = license_verdict(payload, config, today=today)
    except Exception as exc:  # noqa: BLE001 — the alert must still go out
        payload, summary = {}, served_summary({}, today=today)
        verdict = f"served artifact unreadable at {artifact_path}: {exc}"
    preflight_text = args.preflight_lines
    if args.full_run_log is not None:
        try:
            preflight_text = args.full_run_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    job_status = launchd_last_exit(labels, args.launchctl_output)
    body = compose_body(
        summary,
        verdict=verdict,
        preflight_line=preflight_verdict_line(preflight_text),
        job_status=job_status,
        artifact_path=artifact_path,
        holdings=args.holdings,
        log_path=args.log_path,
    )
    if args.json:
        print(json.dumps({"title": ALERT_TITLE, "headers": alert_headers(), "body": body}, indent=2))
    else:
        print(body)
    if args.send:
        return send_alert(ALERT_TITLE, body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
