#!/usr/bin/env python
"""Generate operator-facing verdict report for a closed challenger window.

Phase 4b (2026-04-26) of the model-selection systematization plan.

When a shadow window closes (decision_date_min + shadow_period_days ≤ today),
this script:

  1. Loads both the live (`panel-ltr.json`) and challenger artifacts.
  2. Reads `runs.db.challenger_decisions` for the named challenger over
     the window.
  3. Joins disagreements against `ticker_forward_returns` to compute
     "who was right" on the next 5 / 10 / 20 day horizon.
  4. Generates a markdown report at
     `doc/audits/challenger-{end_date}-{name}.md` with:
        - Side-by-side model parameter comparison
        - Decision statistics (agreement, ch_only_buy, live_only_buy)
        - Top-N disagreements by absolute score gap, with forward returns
        - A heuristic recommendation (NOT a hard verdict — operator decides)
  5. Sends an ntfy push with the headline numbers + report path.

Usage::

    # Default — uses today as window-end, challenger from strategy_config
    python scripts/finalize_challenger.py --strategy renquant_104

    # Explicit window
    python scripts/finalize_challenger.py --strategy renquant_104 \
        --challenger-name macro-enabled \
        --start-date 2026-04-12 --end-date 2026-04-26

    # Suppress ntfy (e.g. running in CI)
    RENQUANT_NO_NOTIFY=1 python scripts/finalize_challenger.py --strategy renquant_104

Exit codes:
    0 — report generated, ntfy sent
    2 — no decisions in window (challenger never ran or window mismatch)
    3 — config error (artifact path missing, etc.)
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("finalize-challenger")


DEFAULT_NTFY_TOPIC = "renquant"
DEFAULT_TOP_N_DISAGREEMENTS = 10


# ── Loading helpers ───────────────────────────────────────────────────────────

def _load_artifact_metadata(path: Path) -> dict:
    """Parse the relevant fields from a panel-ltr artifact for the report."""
    if not path.exists():
        return {"_missing": True, "_path": str(path)}
    try:
        d = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        return {"_error": str(exc), "_path": str(path)}
    md = d.get("metadata") or d
    smoke = (md.get("sim_smoke") or {}) if isinstance(md.get("sim_smoke"), dict) else {}
    panel_shape = md.get("panel_shape") or d.get("panel_shape") or {}
    return {
        "trained_date":  md.get("trained_date") or d.get("trained_date"),
        "feature_count": len(d.get("feature_cols") or md.get("feature_cols") or []),
        "panel_rows":    panel_shape.get("rows"),
        "panel_tickers": panel_shape.get("tickers"),
        "panel_dates":   panel_shape.get("dates"),
        "oos_mean_ic":   md.get("oos_mean_ic")  or d.get("oos_mean_ic"),
        "oos_std_ic":    md.get("oos_std_ic")   or d.get("oos_std_ic"),
        "best_iter":     d.get("best_iter")     or md.get("best_iter"),
        "sim_apy":       smoke.get("apy"),
        "sim_sharpe":    smoke.get("sharpe"),
        "sim_calmar":    smoke.get("calmar"),
        "sim_turnover":  smoke.get("turnover_ratio"),
        "sim_max_dd":    smoke.get("max_drawdown"),
        "_path":         str(path),
    }


# ── Verdict computation ───────────────────────────────────────────────────────

def _read_decisions(conn: sqlite3.Connection, name: str,
                    start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT decision_date, ticker, challenger_action, actual_action,
               challenger_score, actual_score, challenger_rank_score
        FROM challenger_decisions
        WHERE challenger_name = ?
          AND decision_date >= ?
          AND decision_date <= ?
        ORDER BY decision_date, ticker
        """,
        conn, params=(name, start.isoformat(), end.isoformat()),
    )


def _attach_forward_returns(conn: sqlite3.Connection, df: pd.DataFrame) -> pd.DataFrame:
    """Left-join fwd_5d/10d/20d from ticker_forward_returns."""
    if df.empty:
        df["fwd_5d"]  = []
        df["fwd_10d"] = []
        df["fwd_20d"] = []
        return df
    fwd = pd.read_sql_query(
        "SELECT as_of_date, ticker, fwd_5d, fwd_10d, fwd_20d FROM ticker_forward_returns",
        conn,
    )
    if fwd.empty:
        df["fwd_5d"]  = None
        df["fwd_10d"] = None
        df["fwd_20d"] = None
        return df
    fwd["as_of_date"] = pd.to_datetime(fwd["as_of_date"]).dt.strftime("%Y-%m-%d")
    df["decision_key"] = pd.to_datetime(df["decision_date"]).dt.strftime("%Y-%m-%d")
    merged = df.merge(
        fwd, how="left",
        left_on=["decision_key", "ticker"],
        right_on=["as_of_date", "ticker"],
    )
    return merged.drop(columns=["decision_key", "as_of_date"], errors="ignore")


def _summary_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "n_decisions": 0,    "agreement_rate": 0.0,
            "challenger_only_buy": 0, "live_only_buy": 0,
            "score_corr": None,
        }
    agree = (df["challenger_action"] == df["actual_action"]).mean()
    ch_only = ((df["challenger_action"] == "BUY") & (df["actual_action"] != "BUY")).sum()
    li_only = ((df["actual_action"]    == "BUY") & (df["challenger_action"] != "BUY")).sum()
    pair = df[["challenger_score", "actual_score"]].dropna()
    corr = (
        float(pair["challenger_score"].corr(pair["actual_score"]))
        if len(pair) >= 3
        and pair["challenger_score"].std() > 0
        and pair["actual_score"].std() > 0
        else None
    )
    return {
        "n_decisions":         int(len(df)),
        "agreement_rate":      float(agree),
        "challenger_only_buy": int(ch_only),
        "live_only_buy":       int(li_only),
        "score_corr":          corr,
    }


def _top_disagreements(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Pick the N rows with largest |challenger_score - actual_score|
    where the actions also disagree. These are the most-informative
    cases for the operator to eyeball."""
    if df.empty:
        return df
    mask = (df["challenger_action"] != df["actual_action"])
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub["_gap"] = (sub["challenger_score"].fillna(0) - sub["actual_score"].fillna(0)).abs()
    sub = sub.sort_values("_gap", ascending=False).drop(columns="_gap")
    return sub.head(n)


# ── Recommendation heuristic ──────────────────────────────────────────────────

def _heuristic_recommendation(stats: dict, dis: pd.DataFrame) -> str:
    """Surface a non-binding suggestion based on agreement_rate + the
    direction of disagreement-correctness. Operator overrides freely."""
    n = stats["n_decisions"]
    if n == 0:
        return "🚫 No decisions logged — challenger never ran or window mismatch. Verify shadow wiring."
    ar = stats["agreement_rate"]
    parts = []
    if ar >= 0.90:
        parts.append("✅ Agreement very high (≥90%); challenger ≈ live in practice.")
    elif ar >= 0.75:
        parts.append("⚠️  Agreement moderate (75-90%); review disagreements before deciding.")
    else:
        parts.append("🚩 Agreement low (<75%); the two models materially differ — high-stakes decision.")

    if not dis.empty:
        # Compute "who was right" on disagreements via fwd_5d
        ch_buys = dis[dis["challenger_action"] == "BUY"]
        if not ch_buys.empty and "fwd_5d" in ch_buys.columns:
            n_pos = (ch_buys["fwd_5d"].fillna(0) > 0).sum()
            n_tot = ch_buys["fwd_5d"].notna().sum()
            if n_tot > 0:
                rate = n_pos / n_tot
                parts.append(
                    f"Of {n_tot} challenger-only BUY decisions with fwd_5d data, "
                    f"{n_pos} ({rate:.0%}) had positive 5d return — "
                    + ("challenger looks insightful" if rate >= 0.6 else
                       "challenger no better than coin-flip" if rate >= 0.45 else
                       "challenger appears miscalibrated")
                    + "."
                )
    parts.append(
        "🤖 NOT a hard verdict — see the detailed report and use your judgement. "
        "When ready: `python scripts/select_best_model.py --promote {name}` to "
        "promote, or `--reject` to archive."
    )
    return " ".join(parts)


# ── Markdown report ───────────────────────────────────────────────────────────

def _fmt(v, fmt=".4f", missing="—"):
    if v is None:
        return missing
    try:
        return f"{v:{fmt}}"
    except (TypeError, ValueError):
        return str(v)


def render_report(*, strategy: str, challenger_name: str,
                  window_start: pd.Timestamp, window_end: pd.Timestamp,
                  live_md: dict, challenger_md: dict,
                  stats: dict, top_dis: pd.DataFrame,
                  recommendation: str) -> str:
    lines: list[str] = []
    lines.append(f"# Challenger Window Verdict — {challenger_name} ({window_start.date()} → {window_end.date()})")
    lines.append("")
    lines.append(f"_Strategy: `{strategy}`. Generated {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z._")
    lines.append("")

    # ── 1. Model parameter comparison ──
    lines.append("## 1. Model parameter comparison")
    lines.append("")
    lines.append("| Field | live (`panel-ltr.json`) | challenger (`{}`) |".format(challenger_md.get("_path", "?").split("/")[-1]))
    lines.append("|---|---|---|")
    keys = [
        ("trained_date",  None,    "trained_date"),
        ("feature_count", None,    "feature_count"),
        ("panel_rows",    None,    "panel_rows"),
        ("panel_tickers", None,    "panel_tickers"),
        ("panel_dates",   None,    "panel_dates"),
        ("oos_mean_ic",   ".4f",   "OOS mean IC"),
        ("oos_std_ic",    ".4f",   "OOS std IC"),
        ("best_iter",     None,    "best_iter"),
        ("sim_apy",       ".2%",   "sim APY"),
        ("sim_sharpe",    ".2f",   "sim Sharpe"),
        ("sim_calmar",    ".2f",   "sim Calmar"),
        ("sim_turnover",  ".2f",   "sim turnover"),
        ("sim_max_dd",    ".2%",   "sim max drawdown"),
    ]
    for key, fmt, label in keys:
        lv = live_md.get(key)
        cv = challenger_md.get(key)
        lines.append(f"| {label} | {_fmt(lv, fmt) if fmt else (lv if lv is not None else '—')} | "
                     f"{_fmt(cv, fmt) if fmt else (cv if cv is not None else '—')} |")
    lines.append("")

    # ── 2. Decision statistics ──
    lines.append("## 2. Decision statistics")
    lines.append("")
    lines.append(f"- **Total decisions logged**: {stats['n_decisions']}")
    lines.append(f"- **Agreement rate**: {stats['agreement_rate']:.1%}")
    lines.append(f"- **Challenger-only BUY**: {stats['challenger_only_buy']} (challenger wanted to buy when live held)")
    lines.append(f"- **Live-only BUY**: {stats['live_only_buy']} (live bought when challenger held)")
    lines.append(f"- **Score correlation (Pearson)**: " +
                 (f"{stats['score_corr']:.3f}" if stats["score_corr"] is not None else "—"))
    lines.append("")

    # ── 3. Top disagreements ──
    lines.append("## 3. Top disagreements (largest score gap)")
    lines.append("")
    if top_dis.empty:
        lines.append("_No disagreements in this window — the two models behaved identically._")
    else:
        lines.append("| date | ticker | live_score | live_action | ch_score | ch_action | fwd_5d | fwd_20d |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in top_dis.iterrows():
            lines.append(
                f"| {pd.Timestamp(r['decision_date']).date()} | {r['ticker']} | "
                f"{_fmt(r.get('actual_score'), '.3f')} | {r.get('actual_action','—')} | "
                f"{_fmt(r.get('challenger_score'), '.3f')} | {r.get('challenger_action','—')} | "
                f"{_fmt(r.get('fwd_5d'), '+.2%')} | {_fmt(r.get('fwd_20d'), '+.2%')} |"
            )
    lines.append("")

    # ── 4. Recommendation ──
    lines.append("## 4. Operator recommendation (heuristic)")
    lines.append("")
    lines.append(recommendation.replace("{name}", challenger_name))
    lines.append("")
    lines.append("---")
    lines.append("_Generated by `scripts/finalize_challenger.py` (Phase 4b). "
                 "See `doc/components/model-selection.md` for the SOP this slots into._")
    return "\n".join(lines)


# ── ntfy ──────────────────────────────────────────────────────────────────────

def notify_ntfy(title: str, body: str, topic: str = DEFAULT_NTFY_TOPIC) -> None:
    if os.environ.get("RENQUANT_NO_NOTIFY") == "1":
        log.info("RENQUANT_NO_NOTIFY=1 — skipping ntfy push")
        return
    url = f"https://ntfy.sh/{topic}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Title": title, "Priority": "high", "Tags": "scales,clipboard"},
        method="POST",
    )
    try:
        urllib.request.urlopen(req, timeout=10)
        log.info("ntfy push sent to %s", topic)
    except Exception as exc:
        log.warning("ntfy push failed: %s", exc)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",        default="renquant_104")
    p.add_argument("--challenger-name", default=None,
                   help="Override the name from strategy_config (e.g. 'macro-enabled')")
    p.add_argument("--start-date",      default=None,
                   help="ISO date; defaults to earliest decision_date for this challenger")
    p.add_argument("--end-date",        default=None,
                   help="ISO date; defaults to today")
    p.add_argument("--top-n",           type=int, default=DEFAULT_TOP_N_DISAGREEMENTS)
    p.add_argument("--ntfy-topic",      default=DEFAULT_NTFY_TOPIC)
    p.add_argument("--no-ntfy",         action="store_true",
                   help="Skip the ntfy push (still writes the report)")
    p.add_argument("--out-dir",         default=None,
                   help="Override the report output directory (default: doc/audits/)")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("strategy config missing: %s", config_path); return 3
    config = json.loads(config_path.read_text())

    acc = config.get("acceptance") or {}
    ch_cfg = acc.get("challenger") or {}
    challenger_name = args.challenger_name or ch_cfg.get("name")
    if not challenger_name:
        log.error("no challenger name (pass --challenger-name or set "
                  "acceptance.challenger.name in strategy_config.json)")
        return 3

    # Resolve DB path the same way kernel.persistence does — relative
    # to repo root, not strategy dir (see persistence.py::_db_path).
    persistence = config.get("persistence") or {}
    raw_db = persistence.get("db_path", "data/runs.db")
    db_path = Path(raw_db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    if not db_path.exists():
        log.error("runs.db missing: %s", db_path); return 3

    end_date = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp.utcnow().normalize()
    if args.start_date:
        start_date = pd.Timestamp(args.start_date)
    else:
        # Auto-pick earliest decision for this challenger
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.execute(
                    "SELECT MIN(decision_date) FROM challenger_decisions "
                    "WHERE challenger_name = ?",
                    (challenger_name,),
                )
                row = cur.fetchone()
        except sqlite3.OperationalError as exc:
            # Table doesn't exist yet (Phase 4a schema not migrated, OR
            # challenger never enabled). Treat as "no decisions" — exit 2.
            log.warning("challenger_decisions table missing or unreadable: %s "
                        "(this is expected if no challenger has run yet)", exc)
            return 2
        if not row or not row[0]:
            log.error("no decisions logged for challenger '%s'", challenger_name)
            return 2
        start_date = pd.Timestamp(row[0])

    log.info("window: %s → %s, challenger=%s", start_date.date(), end_date.date(), challenger_name)

    # ── Load both artifacts ──
    live_md = _load_artifact_metadata(strategy_dir / "artifacts" / "panel-ltr.json")
    ch_path = ch_cfg.get("artifact_path")
    if ch_path:
        challenger_md = _load_artifact_metadata(strategy_dir / ch_path)
    else:
        # Best-effort: try the .bak.json convention
        guess = strategy_dir / "artifacts" / f"panel-ltr.{challenger_name}.bak.json"
        challenger_md = _load_artifact_metadata(guess)

    # ── Read decisions + compute verdict ──
    with sqlite3.connect(str(db_path)) as conn:
        df = _read_decisions(conn, challenger_name, start_date, end_date)
        df = _attach_forward_returns(conn, df)

    if df.empty:
        log.warning("no decisions in window — emitting empty-window report")

    stats = _summary_stats(df)
    top_dis = _top_disagreements(df, args.top_n)
    rec = _heuristic_recommendation(stats, top_dis)
    log.info("verdict: %s", stats)

    # ── Render + write report ──
    out_dir = Path(args.out_dir) if args.out_dir else REPO_ROOT / "doc" / "audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"challenger-{end_date.date()}-{challenger_name}.md"
    md = render_report(
        strategy=args.strategy, challenger_name=challenger_name,
        window_start=start_date, window_end=end_date,
        live_md=live_md, challenger_md=challenger_md,
        stats=stats, top_dis=top_dis, recommendation=rec,
    )
    out_path.write_text(md)
    log.info("report written: %s", out_path)

    # ── ntfy push ──
    if not args.no_ntfy:
        title = f"📊 Challenger window closed — {challenger_name}"
        body = (
            f"Window {start_date.date()} → {end_date.date()}\n"
            f"Decisions: {stats['n_decisions']}\n"
            f"Agreement: {stats['agreement_rate']:.0%}\n"
            f"CH-only BUY: {stats['challenger_only_buy']}  Live-only BUY: {stats['live_only_buy']}\n"
            f"Report: {out_path}"
        )
        notify_ntfy(title, body, args.ntfy_topic)

    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
