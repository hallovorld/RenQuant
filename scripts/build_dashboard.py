#!/usr/bin/env python
"""Generate doc/dashboard.md — daily-refresh metrics board.

Reads the broker-tagged runs DB + live state + latest WF / panel artifacts
and renders a markdown summary that GitHub auto-renders. Designed to be
called from the daily cron after `daily_104.sh` settles.

Sections:
  1. Header        — timestamp, portfolio value, daily return, HWM
  2. Live holdings — recent trades + open positions
  3. Daily P/L     — last 21d table
  4. Model health  — last WF mean IC, panel fingerprint, retrain age
  5. Regime        — current regime + confidence
  6. Priorities    — top 5 open roadmap items

Usage::

    python scripts/build_dashboard.py
    python scripts/build_dashboard.py --broker alpaca --out doc/dashboard.md
    python scripts/build_dashboard.py --broker paper

References:
  - User request 2026-05-09: "需要一个 dashboard refresh daily 来显示一些 matrix"
  - CLAUDE.md §4 keep docs current — the dashboard is the live operations doc
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import sqlite3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from subrepo_module_delegate import delegate_to_subrepo_module

REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("dashboard")


# ── Section builders ─────────────────────────────────────────────────────────

def _conn(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.exists():
        log.warning("DB not found: %s", db_path)
        return None
    return sqlite3.connect(str(db_path))


def _fmt_pct(x: float | None, *, sign: bool = True) -> str:
    if x is None or x != x:    # None or NaN
        return "—"
    return f"{x*100:+.2f}%" if sign else f"{x*100:.2f}%"


def _fmt_money(x: float | None) -> str:
    if x is None or x != x:
        return "—"
    return f"${x:,.2f}"


def section_header(broker: str, db: sqlite3.Connection | None,
                   live_state: dict) -> str:
    """Header — current portfolio value, HWM, last refresh time."""
    pv = "—"
    daily = "—"
    as_of = "—"
    if db is not None:
        cur = db.cursor()
        cur.execute(
            "SELECT as_of_date, portfolio_value, daily_return "
            "FROM portfolio_daily_metrics "
            "WHERE run_type='live' "
            "ORDER BY as_of_date DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            as_of = row[0]
            pv = _fmt_money(row[1])
            daily = _fmt_pct(row[2])
    hwm = live_state.get("high_water_mark", None)
    regime = live_state.get("regime", "—")
    conf = live_state.get("regime_confidence", None)

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return (
        f"# RenQuant Dashboard — `{broker}` broker\n\n"
        f"**Refreshed:** {now}  ·  **As-of:** {as_of}  ·  **Strategy:** renquant_104\n\n"
        f"| Portfolio value | Daily return | High-water mark | Regime |\n"
        f"|---|---|---|---|\n"
        f"| {pv} | {daily} | {_fmt_money(hwm)} | "
        f"{regime} ({conf:.2f})| \n"
        if conf is not None else
        f"# RenQuant Dashboard — `{broker}` broker\n\n"
        f"**Refreshed:** {now}  ·  **As-of:** {as_of}  ·  **Strategy:** renquant_104\n\n"
        f"| Portfolio value | Daily return | High-water mark | Regime |\n"
        f"|---|---|---|---|\n"
        f"| {pv} | {daily} | {_fmt_money(hwm)} | {regime} (—)|\n"
    )


def section_recent_trades(db: sqlite3.Connection | None,
                          n_days: int = 7) -> str:
    """Last N days of trades (broker-tagged DB)."""
    if db is None:
        return "## Recent trades\n\n_DB unavailable._\n\n"
    cur = db.cursor()
    cutoff = (datetime.date.today() - datetime.timedelta(days=n_days)).isoformat()
    cur.execute(
        "SELECT run_id, ticker, action, shares, price, exit_reason, pnl_pct "
        "FROM trades WHERE substr(run_id,1,10) >= ? "
        "ORDER BY ROWID DESC LIMIT 30",
        (cutoff,),
    )
    rows = cur.fetchall()
    if not rows:
        return f"## Recent trades (last {n_days}d)\n\n_No trades in window._\n\n"
    out = [f"## Recent trades (last {n_days}d) — {len(rows)} fills\n"]
    out.append("| Date | Ticker | Action | Shares | Price | Reason | P/L% |")
    out.append("|---|---|---|---|---|---|---|")
    for run_id, ticker, action, shares, price, reason, pnl in rows:
        date = run_id[:10] if run_id else "—"
        sh = f"{int(shares)}" if shares and shares == shares else "—"
        px = _fmt_money(price)
        rsn = (reason or "—")[:24]
        pl = _fmt_pct(pnl) if pnl is not None else "—"
        out.append(f"| {date} | {ticker} | {action} | {sh} | {px} | {rsn} | {pl} |")
    return "\n".join(out) + "\n\n"


def section_pnl_sparkline(db: sqlite3.Connection | None) -> str:
    """Last 21 days of portfolio value (numeric table — ASCII charts trip
    GitHub's md renderer).

    Filters out daily_return spikes > ±50% (initial-cash deployment
    transitions polluted as noise — e.g. starting $100k → $10k after
    first sweep produces a -89.82% spurious "loss")."""
    if db is None:
        return ""
    cur = db.cursor()
    cur.execute(
        "SELECT as_of_date, portfolio_value, daily_return "
        "FROM portfolio_daily_metrics WHERE run_type='live' "
        "ORDER BY as_of_date DESC LIMIT 21"
    )
    rows = cur.fetchall()
    if not rows:
        return ""
    # Drop deployment-transition rows
    rows = [(d, pv, ret) for (d, pv, ret) in rows
            if ret is None or abs(ret) < 0.5]
    if not rows:
        return ""
    rows = list(reversed(rows))
    out = ["## Portfolio P/L (last 21d)\n"]
    out.append("| Date | Value | Daily |")
    out.append("|---|---|---|")
    for d, pv, ret in rows:
        out.append(f"| {d} | {_fmt_money(pv)} | {_fmt_pct(ret)} |")
    return "\n".join(out) + "\n\n"


def _resolve_prod_panel_path() -> Path:
    """Resolve panel-LTR prod artifact via strategy_config.golden.json (canonical
    source per §5.13.14 — never hardcode `panel-ltr.json`). Falls back to the
    legacy data/ path if config read fails (e.g. partial install)."""
    try:
        cfg_path = REPO_ROOT / "backtesting/renquant_104/strategy_config.golden.json"
        cfg = json.loads(cfg_path.read_text())
        rel = cfg["ranking"]["panel_scoring"]["artifact_path"]
        # Path is relative to renquant_104/ per convention
        return REPO_ROOT / "backtesting/renquant_104" / rel
    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        # Legacy fallback (pre 2026-05-11 sim/prod isolation) — flagged 🔴 in
        # doc/audits/2026-05-20-deep-code-audit.md P0-5; should never hit.
        return REPO_ROOT / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"


def section_model_health() -> str:
    """Panel fingerprint, retrain age, latest WF mean IC."""
    out = ["## Model health\n"]

    # Panel artifact — resolved via golden config, not hardcoded
    panel_path = _resolve_prod_panel_path()
    if panel_path.exists():
        try:
            mtime = datetime.datetime.fromtimestamp(panel_path.stat().st_mtime)
            age_h = (datetime.datetime.now() - mtime).total_seconds() / 3600
            meta = json.loads(panel_path.read_text())
            fp = (meta.get("model_fingerprint")
                  or meta.get("fingerprint")
                  or meta.get("config_fingerprint")
                  or "—")
            fp = str(fp)[:18]
            out.append(f"- **Panel artifact:** `{panel_path.name}`  ·  fingerprint `{fp}`")
            out.append(f"- **Last retrain:** {mtime:%Y-%m-%d %H:%M}  ({age_h:.1f}h ago)")
        except Exception as e:
            out.append(f"- Panel artifact unreadable: {e}")
    else:
        out.append("- Panel artifact not found")

    # Latest WF JSON
    wf_files = sorted(
        (REPO_ROOT / "data").glob("wf_*.json"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    if wf_files:
        latest = wf_files[0]
        try:
            data = json.loads(latest.read_text())
            ic = data.get("mean_ic", data.get("oos_mean_ic"))
            if ic is None and "results" in data:
                ic = sum(r.get("oos_ic", 0) for r in data["results"]) / max(1, len(data["results"]))
            out.append(f"- **Latest WF:** `{latest.name}`  ·  mean IC = {ic:.4f}" if ic else
                       f"- **Latest WF:** `{latest.name}`  ·  IC unavailable")
        except Exception as e:
            out.append(f"- WF file unreadable: {e}")
    return "\n".join(out) + "\n\n"


def section_regime_gates(live_state: dict) -> str:
    """Show which regime-conditional gates fire for the CURRENT regime.

    2026-05-15: critical operator visibility — when looking at today's
    trades, we want to immediately see "in BULL_CALM, gates X,Y are OFF
    so META-class buys WILL fire" or "in BEAR, gates X,Y,Z are ON so
    deep-drawdown candidates WILL be vetoed".

    Reads `regime_params` overlay + `disabled_in_regimes` config field
    on each gate. Renders a per-gate fire/skip table.
    """
    out = ["## Regime-conditional gate status\n"]
    regime = live_state.get("regime", "—")
    confidence = live_state.get("regime_confidence", None)
    out.append(f"**Current regime:** `{regime}`"
               + (f" (conf={confidence:.2f})" if confidence is not None else "")
               + "\n")

    # Read golden config to check gate status
    cfg_path = REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.json"
    if not cfg_path.exists():
        return "\n".join(out) + "\n_strategy_config.json missing._\n\n"
    try:
        cfg = json.loads(cfg_path.read_text())
    except Exception as e:
        return "\n".join(out) + f"\n_config unreadable: {e}_\n\n"

    rank = cfg.get("ranking", {})
    bqg = rank.get("buy_quality_gates", {})
    ks  = rank.get("kelly_sizing", {})

    # Per-gate: enabled? disabled_in_regimes? fires today?
    def _gate_status(name: str, gate_cfg: dict, applies_in: list[str] | None = None) -> tuple[str, str]:
        if not gate_cfg.get("enabled", False):
            return "OFF", "config.enabled=false"
        disabled = gate_cfg.get("disabled_in_regimes", [])
        if regime in disabled:
            return "SKIP", f"regime in disabled_in_regimes={disabled}"
        if applies_in is not None and regime not in applies_in:
            return "SKIP", f"regime not in applies_in={applies_in}"
        return "**FIRE**", "active in current regime"

    gates_table = [
        ("regime_momentum",       bqg.get("regime_momentum", {}),
         bqg.get("regime_momentum", {}).get("momentum_regimes")),
        ("deep_drawdown_veto",    bqg.get("deep_drawdown_veto", {}), None),
    ]
    out.append("\n| Gate | Status | Why |")
    out.append("|---|---|---|")
    for name, gcfg, applies in gates_table:
        status, reason = _gate_status(name, gcfg, applies)
        out.append(f"| `{name}` | {status} | {reason} |")

    # Kelly path status (use_calibrator_mu / use_realized_vol_fallback are
    # global toggles — no regime conditional yet)
    out.append("")
    out.append(f"**Kelly μ source:** "
               f"{'calibrator.expected_return' if ks.get('use_calibrator_mu') else 'NGBoost μ (likely None → uniform fallback)'}")
    out.append(f"**Kelly σ source:** "
               f"{'realized_vol_60d (clipped)' if ks.get('use_realized_vol_fallback') else 'NGBoost σ (likely None → Kelly returns 0)'}")

    return "\n".join(out) + "\n\n"


def section_priorities() -> str:
    """Top 5 open priorities from roadmap.md (parses ## headings and
    truncates long descriptions)."""
    rm = REPO_ROOT / "doc" / "roadmap.md"
    if not rm.exists():
        return ""
    out = ["## Open priorities (top 5 from `doc/roadmap.md`)\n"]
    in_p0 = False
    items: list[str] = []
    for line in rm.read_text().splitlines():
        if line.startswith("## P0"):
            in_p0 = True
            continue
        if in_p0 and line.startswith("## "):
            break
        if in_p0 and line.startswith("### "):
            title = line[4:].strip()
            items.append(title)
            if len(items) >= 5:
                break
    if not items:
        return ""
    # Strip pre-existing "1. ⭐ Title" → "Title" (numbering THEN glyph)
    import re
    cleaned = [
        re.sub(r"^[★⭐]\s*", "", re.sub(r"^\d+\.\s*", "", t)).strip()
        for t in items
    ]
    for i, t in enumerate(cleaned, 1):
        out.append(f"{i}. {t}")
    return "\n".join(out) + "\n\n"


def section_footer() -> str:
    return (
        "---\n\n"
        "_Auto-generated by `scripts/build_dashboard.py`. To refresh manually: "
        "`python scripts/build_dashboard.py`._\n"
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def build(broker: str, out_path: Path) -> str:
    db_path = REPO_ROOT / "data" / f"runs.{broker}.db"
    state_path = REPO_ROOT / "backtesting" / "renquant_104" / f"live_state.{broker}.json"

    db = _conn(db_path)
    live_state = {}
    if state_path.exists():
        try:
            live_state = json.loads(state_path.read_text())
        except Exception as e:
            log.warning("live_state %s unreadable: %s", state_path, e)

    parts = [
        section_header(broker, db, live_state),
        section_recent_trades(db),
        section_regime_gates(live_state),  # 2026-05-15: gate fire-status visibility
        section_pnl_sparkline(db),
        section_model_health(),
        section_priorities(),
        section_footer(),
    ]
    md = "\n".join(p for p in parts if p)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    log.info("Wrote dashboard → %s (%d bytes)", out_path, len(md))
    return md


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--broker", default="alpaca",
                   choices=["alpaca", "paper"])
    p.add_argument("--out", default="doc/dashboard.md")
    args = p.parse_args()
    build(args.broker, REPO_ROOT / args.out)


if __name__ == "__main__":
    delegated = delegate_to_subrepo_module(
        "renquant_backtesting.reporting.build_dashboard",
        sys.argv[1:],
        repo_root=REPO_ROOT,
        packages=("renquant-backtesting", "renquant-common"),
        runner_env="RQ_BACKTESTING_OPS_RUNNER",
        strict_env="RQ_BACKTESTING_OPS_STRICT",
    )
    if delegated is not None:
        raise SystemExit(delegated)
    main()
