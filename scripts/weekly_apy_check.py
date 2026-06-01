#!/usr/bin/env python
"""Read logs/live_104/audit.jsonl, compute rolling live APY, alert on deviation.

Plan D, 2026-04-23. The audit stream is written by daily_104.sh (one JSONL
row per successful daily run). This script:

  1. Reads the last N rows (default 30).
  2. Computes annualized return from (last_equity / first_equity)^(365 / days)-1.
  3. Emits a ntfy alert if the rolling APY is below the `--alert-threshold`
     (default 25%, halfway between T4 golden's 40% and v1 golden's 33%).
  4. Additionally alerts if drawdown has been elevated (> 20%) for more
     than 5 consecutive days — possible sign that the live HWM/equity
     ratio needs attention (see `adapters/runner.py::resolve_hwm`).

Exit codes:
   0 — healthy OR insufficient data
   2 — APY below alert threshold (did not fire ntfy, just stderr)
   3 — persistent drawdown alert

Meant to be invoked weekly via `com.renquant.weekly-apy104.plist` (Sun 12:00 PT)
or on demand: `python scripts/weekly_apy_check.py`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_LOG = REPO_ROOT / "logs" / "live_104" / "audit.jsonl"


def _ntfy(title: str, body: str, topic: str = "renquant") -> None:
    """Best-effort push. Silent on network failure — this is monitoring,
    not the hot path."""
    import os
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        return   # suppressed by tests
    url = f"https://ntfy.sh/{topic}"
    try:
        req = urllib.request.Request(
            url, data=body.encode("utf-8"),
            headers={"Title": title},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5).read()
    except (urllib.error.URLError, OSError):
        pass   # silent fail — we'll still write to stderr below


def _read_recent(path: Path, window_days: int) -> list[dict]:
    if not path.exists():
        return []
    cutoff = datetime.utcnow() - timedelta(days=window_days + 1)
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            try:
                d = datetime.fromisoformat(row["date"])
            except (KeyError, ValueError):
                continue
            if d >= cutoff:
                out.append(row)
    return out


def _compute_rolling_apy(rows: list[dict]) -> tuple[float | None, int]:
    """Return (apy, n_days) from first/last equity in the window.

    apy = (last / first)^(365/days) - 1 where days = date_last - date_first.
    Returns (None, n) when n < 2 or days == 0 or first == 0.
    """
    valid = [r for r in rows if r.get("equity") is not None]
    if len(valid) < 2:
        return None, len(valid)
    first = valid[0]
    last = valid[-1]
    try:
        first_eq = float(first["equity"])
        last_eq  = float(last["equity"])
        d_first = datetime.fromisoformat(first["date"])
        d_last  = datetime.fromisoformat(last["date"])
    except (ValueError, KeyError, TypeError):
        return None, len(valid)
    days = (d_last - d_first).days
    if days <= 0 or first_eq <= 0:
        return None, len(valid)
    apy = (last_eq / first_eq) ** (365.0 / days) - 1.0
    return apy, len(valid)


def _latest_sharpe(db_path: Path) -> tuple:
    """Return (sharpe_21d, sharpe_63d) from the newest portfolio_daily_metrics
    row for live+renquant-104, or None if table/DB missing.

    Added 2026-04-24 — Sharpe=2.0 target tracking. Surface the latest
    rolling Sharpe in the weekly ntfy so deviation is visible at a glance.
    """
    import sqlite3
    if not db_path.exists():
        return None
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute(
            """SELECT sharpe_21d, sharpe_63d
                 FROM portfolio_daily_metrics
                WHERE run_type='live' AND strategy='renquant-104'
                ORDER BY as_of_date DESC LIMIT 1"""
        ).fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    return row


def _drawdown_streak(rows: list[dict], threshold: float) -> int:
    """Return the longest recent-consecutive-day streak with drawdown > threshold."""
    best = cur = 0
    for r in rows:
        dd = r.get("drawdown_pct")
        if dd is not None and dd > threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--window-days", type=int, default=30)
    p.add_argument("--alert-threshold", type=float, default=0.25,
                   help="APY below this value triggers an alert (default 0.25 = 25%)")
    p.add_argument("--drawdown-threshold", type=float, default=0.20,
                   help="Drawdown above this value tracks toward persistent-dd alert")
    p.add_argument("--drawdown-days", type=int, default=5,
                   help="Consecutive days above drawdown_threshold → persistent-dd alert")
    p.add_argument("--audit-log", default=str(AUDIT_LOG),
                   help="Path to audit.jsonl (default: logs/live_104/audit.jsonl)")
    p.add_argument("--topic", default="renquant")
    p.add_argument("--quiet", action="store_true", help="Suppress ntfy push; stderr only.")
    args = p.parse_args()

    audit = Path(args.audit_log)
    rows = _read_recent(audit, args.window_days)

    if not rows:
        print(f"weekly_apy_check: audit log {audit} is empty or missing — no action",
              file=sys.stderr)
        return 0

    apy, n = _compute_rolling_apy(rows)
    dd_streak = _drawdown_streak(rows, args.drawdown_threshold)

    msg_parts = [f"{n} rows"]
    if apy is not None:
        msg_parts.append(f"APY={apy:+.1%}")
    if dd_streak:
        msg_parts.append(f"dd_streak={dd_streak}d")

    # Pull latest Sharpe from portfolio_daily_metrics (target 2.0).
    # Non-fatal if table/DB missing — keeps backward compat with older
    # live states.
    sharpe_info = _latest_sharpe(REPO_ROOT / "data" / "runs.db")
    if sharpe_info:
        s21, s63 = sharpe_info
        if s21 is not None:
            msg_parts.append(f"Sharpe21d={s21:.2f}")
        if s63 is not None:
            msg_parts.append(f"Sharpe63d={s63:.2f}")

    summary = " / ".join(msg_parts)

    exit_code = 0
    alert_title = None
    alert_body  = None

    if apy is not None and apy < args.alert_threshold:
        exit_code = 2
        alert_title = "RenQuant 104 WATCH"
        alert_body  = f"Live rolling {args.window_days}d APY {apy:+.1%} < alert {args.alert_threshold:+.1%} ({summary})"
    elif dd_streak >= args.drawdown_days:
        exit_code = 3
        alert_title = "RenQuant 104 WATCH"
        alert_body  = f"Live drawdown > {args.drawdown_threshold:.0%} for {dd_streak} days — check HWM ({summary})"

    if alert_title:
        print(alert_title, ":", alert_body, file=sys.stderr)
        if not args.quiet:
            _ntfy(alert_title, alert_body, topic=args.topic)
    else:
        print(f"weekly_apy_check: healthy — {summary}")

    return exit_code


def _run_multirepo_weekly_apy(argv: list[str]) -> int:
    orchestrator_src = REPO_ROOT.parent / "renquant-orchestrator" / "src"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{orchestrator_src}{os.pathsep}{env.get('PYTHONPATH', '')}"
    probe = subprocess.run(
        [sys.executable, "-c", "import renquant_orchestrator.weekly_apy_monitor"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode != 0:
        return 127
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "renquant_orchestrator.weekly_apy_monitor",
            "--repo-root",
            str(REPO_ROOT),
            *argv,
        ],
        env=env,
    ).returncode


if __name__ == "__main__":
    if os.environ.get("RQ_WEEKLY_APY_RUNNER", "multirepo") != "legacy":
        rc = _run_multirepo_weekly_apy(sys.argv[1:])
        if rc != 127:
            sys.exit(rc)
        if os.environ.get("RQ_WEEKLY_APY_STRICT") == "1":
            print(
                "ERROR: renquant_orchestrator.weekly_apy_monitor unavailable "
                "and RQ_WEEKLY_APY_STRICT=1",
                file=sys.stderr,
            )
            sys.exit(2)
        print(
            "WARN: renquant_orchestrator.weekly_apy_monitor unavailable; "
            "falling back to umbrella weekly APY check.",
            file=sys.stderr,
        )
    sys.exit(main())
