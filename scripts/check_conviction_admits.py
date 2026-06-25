#!/usr/bin/env python3
"""Reusable deploy guard: would the conviction gate still ADMIT buys?

A config/pin change that silently zeroes out admissions is the sell-only footgun
(e.g. enabling demean over the wrong cross-section — caught 2026-06-24). This
replays the live conviction_gate config against the most recent recorded
candidate scores and asserts it admits >= --min-admits. It is the generalized,
deterministic, OFFLINE form of the verify that protected the demean go-live —
intended as the standard `promote_pin.py --verify-cmd`, so a deploy that would
stop trading AUTO-REVERTS.

Mirrors ConvictionGateTask: when demean_cross_sectional is on, subtract the FULL
cross-sectional mean of mu (the #147 fix) before the mu_floor.

Exit 0 = admits >= min, 1 = below min (would not buy), 2 = cannot evaluate.
Read-only. Usage: check_conviction_admits.py [--min-admits 1] [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path

REPO = Path("/Users/renhao/git/github/RenQuant")
CFG = REPO / ".subrepo_runtime" / "repos" / "renquant-strategy-104" / "configs" / "strategy_config.json"
DB = REPO / "data" / "runs.alpaca.db"


def conviction_cfg(cfg_path: Path) -> dict:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    return cfg.get("ranking", {}).get("panel_scoring", {}).get("conviction_gate", {}) or {}


def latest_scores(db_path: Path) -> "tuple[str, list[float]]":
    con = sqlite3.connect(str(db_path))
    rids = [r[0] for r in con.execute(
        "select distinct run_id from candidate_scores where run_id like '%-live-%'")]
    # the most recent date, then the run_id with the most scored rows that day
    def date_of(rid: str) -> str:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", rid)
        return m.group(1) if m else ""
    if not rids:
        return "", []
    latest_date = max(date_of(r) for r in rids)
    same_day = [r for r in rids if date_of(r) == latest_date]
    best, best_n = None, -1
    for r in same_day:
        n = con.execute("select count(*) from candidate_scores where run_id=? and expected_return is not null",
                        (r,)).fetchone()[0]
        if n > best_n:
            best, best_n = r, n
    ers = [row[0] for row in con.execute(
        "select expected_return from candidate_scores where run_id=? and expected_return is not null",
        (best,))]
    return f"{best} ({latest_date})", [float(e) for e in ers]


def count_admits(ers: list[float], cfg: dict) -> dict:
    if not cfg.get("enabled") or cfg.get("mu_floor") is None:
        return {"enabled": False, "admits": len(ers), "n": len(ers)}
    mu_floor = float(cfg["mu_floor"])
    demean = bool(cfg.get("demean_cross_sectional", False))
    xs_mean = (sum(ers) / len(ers)) if (demean and ers) else 0.0
    admits = sum(1 for e in ers if (e - xs_mean) >= mu_floor)
    return {"enabled": True, "mu_floor": mu_floor, "demean": demean,
            "xs_mean": round(xs_mean, 4), "admits": admits, "n": len(ers)}


def evaluate(min_admits: int, cfg_path: Path = CFG, db_path: Path = DB) -> dict:
    cfg = conviction_cfg(cfg_path)
    run, ers = latest_scores(db_path)
    if not ers:
        return {"status": "CANNOT_EVALUATE", "detail": "no candidate scores in db"}
    res = count_admits(ers, cfg)
    res["run"] = run
    res["status"] = "OK" if res["admits"] >= min_admits else "WOULD_NOT_BUY"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--min-admits", type=int, default=1)
    ap.add_argument("--config", default=str(CFG))
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    res = evaluate(args.min_admits, Path(args.config), Path(args.db))
    print(json.dumps(res) if args.json else
          f"{res['status']}: admits={res.get('admits')} / n={res.get('n')} "
          f"(demean={res.get('demean')}, mu_floor={res.get('mu_floor')}, run={res.get('run')})")
    return {"OK": 0, "WOULD_NOT_BUY": 1}.get(res["status"], 2)


if __name__ == "__main__":
    raise SystemExit(main())
