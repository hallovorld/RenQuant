#!/usr/bin/env python
"""Operator dashboard — what's happening with my models, in one screen.

Phase 4b UX (2026-04-26). Adjacent to:
  - scripts/select_best_model.py — backend tournament (Phase 3)
  - scripts/finalize_challenger.py — shadow-window verdict (Phase 4b-1)

What this script answers — at a glance, no SQL required:

  1. What's currently in production? (active artifact + key metrics)
  2. What's in flight? (challenger window, retrain in progress)
  3. What's the recent retrain history? (last 10 attempts, gate verdicts)
  4. What's available for promotion? (tournament leaderboard with composite)

Read-only. Never mutates state. Operator runs `select_best_model.py
--promote` or `finalize_challenger.py` to act on what they see here.

Why a separate dashboard vs extending select_best_model.py:
  - select_best_model.py answers "which is best?" in a vacuum.
  - This script answers "what should I do next?" with full context
    (retrain in progress, shadow window status, recent rejections).
  - Different mental model → different command.

Usage:
    python scripts/model_dashboard.py
    python scripts/model_dashboard.py --strategy renquant_104
    python scripts/model_dashboard.py --json    # machine-readable
"""
from __future__ import annotations

import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _md(d: dict | None) -> dict:
    if d is None:
        return {}
    md = d.get("metadata") if isinstance(d.get("metadata"), dict) else d
    return md or {}


# ── Section 1: What's in production ───────────────────────────────────────────

def section_production(strategy_dir: Path) -> dict:
    active = strategy_dir / "artifacts" / "panel-ltr.json"
    prev   = strategy_dir / "artifacts" / "panel-ltr.previous.json"
    if not active.exists():
        return {"status": "MISSING", "detail": f"{active} not found"}
    a = _load(active)
    md = _md(a)
    smoke = md.get("sim_smoke") or {}
    out = {
        "artifact":      str(active.relative_to(REPO_ROOT)),
        "trained_date":  md.get("trained_date") or a.get("trained_date"),
        "backend":       a.get("kind", "?").replace("panel_ltr_", ""),
        "feature_count": len(a.get("feature_cols") or []),
        "oos_mean_ic":   md.get("oos_mean_ic")   or a.get("oos_mean_ic"),
        "oos_std_ic":    md.get("oos_std_ic")    or a.get("oos_std_ic"),
        "sim_apy":       smoke.get("apy"),
        "sim_sharpe":    smoke.get("sharpe"),
        "sim_calmar":    smoke.get("calmar"),
        "rollback_available": prev.exists(),
    }
    if prev.exists():
        p = _load(prev)
        pmd = _md(p)
        out["rollback_target"] = {
            "trained_date":  pmd.get("trained_date") or (p or {}).get("trained_date"),
            "oos_mean_ic":   pmd.get("oos_mean_ic")  or (p or {}).get("oos_mean_ic"),
        }
    return out


# ── Section 2: What's in flight ───────────────────────────────────────────────

def section_in_flight(strategy_dir: Path, config: dict) -> dict:
    out: dict = {"retrain_in_progress": False, "challenger": None}
    # Heuristic: a .staging.json or .pre-train.json hanging around suggests
    # an in-flight retrain (or a dirty crash leaving artifacts behind).
    art = strategy_dir / "artifacts"
    staging = art / "panel-ltr.staging.json"
    pretrain = art / "panel-ltr.pre-train.json"
    if staging.exists() or pretrain.exists():
        out["retrain_in_progress"] = True
        out["evidence"] = [str(p.name) for p in (staging, pretrain) if p.exists()]
        out["warning"] = (
            "Found staging/pre-train artifacts — either a retrain is in flight, "
            "OR a prior retrain crashed mid-flow. Clean up manually if no live retrain."
        )
    # Challenger config
    ch_cfg = (config.get("acceptance") or {}).get("challenger") or {}
    if ch_cfg.get("enabled"):
        out["challenger"] = {
            "name":               ch_cfg.get("name"),
            "shadow_period_days": ch_cfg.get("shadow_period_days"),
            "artifact_path":      ch_cfg.get("artifact_path"),
        }
    return out


# ── Section 3: Recent retrain history ─────────────────────────────────────────

def section_history(db_path: Path, n: int = 10) -> list[dict]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(str(db_path))
        cur = conn.execute(
            """
            SELECT run_id, run_date, oos_mean_ic, train_ic, n_features, n_rows,
                   trigger, elapsed_sec
            FROM training_runs
            ORDER BY run_date DESC
            LIMIT ?
            """,
            (n,),
        )
        rows = [
            {
                "run_id":      r[0],
                "run_date":    r[1],
                "oos_mean_ic": r[2],
                "train_ic":    r[3],
                "n_features":  r[4],
                "n_rows":      r[5],
                "trigger":     r[6],
                "elapsed_sec": r[7],
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return rows
    except Exception as exc:
        return [{"_error": str(exc)}]


# ── Section 4: Tournament leaderboard ─────────────────────────────────────────

def section_tournament(strategy_dir: Path) -> list[dict]:
    """Reuse select_best_model's discovery + scoring logic."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import select_best_model as sbm  # noqa: PLC0415
    except ImportError:
        return [{"_error": "select_best_model not importable"}]

    artifacts_dir = strategy_dir / "artifacts"
    if not artifacts_dir.exists():
        return []
    cands = sbm.discover_candidates(artifacts_dir)
    if not cands:
        return []
    cands = sbm.score_candidates(cands, sbm.parse_weights(None))
    return [
        {
            "rank":          i + 1,
            "name":          c.name,
            "composite":     c.composite,
            "oos_mean_ic":   c.oos_mean_ic,
            "feature_count": c.feature_count,
            "panel_rows":    c.panel_rows,
        }
        for i, c in enumerate(cands)
    ]


# ── Render ────────────────────────────────────────────────────────────────────

def _fmt(v, fmt="", missing="—"):
    if v is None:
        return missing
    try:
        return f"{v:{fmt}}" if fmt else str(v)
    except (ValueError, TypeError):
        return str(v)


def render_text(strategy: str, prod: dict, flight: dict,
                history: list[dict], tournament: list[dict]) -> str:
    lines: list[str] = []
    lines.append(f"\n  ═══ RenQuant Model Dashboard — {strategy} "
                 f"@ {datetime.datetime.utcnow().isoformat(timespec='seconds')}Z ═══\n")

    # Section 1: Production
    lines.append("  ▸ PRODUCTION")
    if prod.get("status") == "MISSING":
        lines.append(f"      ⚠️   {prod.get('detail')}")
    else:
        lines.append(f"      artifact      : {prod.get('artifact')}")
        lines.append(f"      backend       : {prod.get('backend')}")
        lines.append(f"      trained_date  : {prod.get('trained_date')}")
        lines.append(f"      feature_count : {prod.get('feature_count')}")
        lines.append(f"      OOS mean IC   : {_fmt(prod.get('oos_mean_ic'), '.4f')}")
        if prod.get("sim_apy") is not None:
            lines.append(f"      sim APY       : {_fmt(prod.get('sim_apy'), '.2%')}")
            lines.append(f"      sim Sharpe    : {_fmt(prod.get('sim_sharpe'), '.2f')}")
        rb = prod.get("rollback_target")
        if rb:
            lines.append(f"      rollback OK   : prior IC {_fmt(rb.get('oos_mean_ic'), '.4f')} "
                         f"(trained {rb.get('trained_date')})")
        else:
            lines.append("      rollback OK   : no .previous.json")
    lines.append("")

    # Section 2: In flight
    lines.append("  ▸ IN FLIGHT")
    if flight.get("retrain_in_progress"):
        lines.append(f"      retrain       : 🔄 IN PROGRESS (or crashed)")
        lines.append(f"      evidence      : {', '.join(flight.get('evidence', []))}")
        if flight.get("warning"):
            lines.append(f"      ⚠️   {flight['warning']}")
    else:
        lines.append("      retrain       : (none in flight)")
    ch = flight.get("challenger")
    if ch:
        lines.append(f"      challenger    : ✅ ENABLED  name={ch['name']}  "
                     f"shadow_period={ch['shadow_period_days']}d  artifact={ch.get('artifact_path')}")
    else:
        lines.append("      challenger    : (disabled)")
    lines.append("")

    # Section 3: History
    lines.append(f"  ▸ RECENT RETRAINS (last {len(history)})")
    if not history:
        lines.append("      (no training_runs in DB)")
    else:
        lines.append(f"      {'run_date':<22} {'oos_ic':>8}  {'train_ic':>8}  "
                     f"{'rows':>7}  {'feats':>5}  {'trigger':<22}  {'elapsed':>7}")
        for r in history:
            if "_error" in r:
                lines.append(f"      ERROR reading: {r['_error']}")
                continue
            lines.append(
                f"      {(r.get('run_date') or '?')[:22]:<22} "
                f"{_fmt(r.get('oos_mean_ic'), '.4f'):>8}  "
                f"{_fmt(r.get('train_ic'), '.4f'):>8}  "
                f"{_fmt(r.get('n_rows'), ''):>7}  "
                f"{_fmt(r.get('n_features'), ''):>5}  "
                f"{(r.get('trigger') or '?'):<22}  "
                f"{_fmt(r.get('elapsed_sec'), '.0f'):>5}s"
            )
    lines.append("")

    # Section 4: Tournament
    lines.append(f"  ▸ AVAILABLE CANDIDATES (tournament)")
    if not tournament:
        lines.append("      (no .bak.json artifacts)")
    else:
        lines.append(f"      {'rank':>4}  {'name':<22} {'composite':>10}  "
                     f"{'oos_ic':>8}  {'feats':>5}  {'rows':>7}")
        for c in tournament:
            if "_error" in c:
                lines.append(f"      ERROR: {c['_error']}")
                continue
            lines.append(
                f"      {c['rank']:>4}  {c['name']:<22} "
                f"{c['composite']:>+10.3f}  "
                f"{_fmt(c.get('oos_mean_ic'), '.4f'):>8}  "
                f"{_fmt(c.get('feature_count'), ''):>5}  "
                f"{_fmt(c.get('panel_rows'), ''):>7}"
            )
    lines.append("")
    lines.append("  Actions: `select_best_model.py --promote <name>` | "
                 "`finalize_challenger.py` | `train_104.py`")
    lines.append("")
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--json",     action="store_true",
                   help="Machine-readable JSON output (no pretty text)")
    p.add_argument("--history-n", type=int, default=10)
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    config_path  = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        print(f"strategy config not found: {config_path}", file=sys.stderr)
        return 2
    config = json.loads(config_path.read_text())

    # Resolve runs.db path the same way persistence.py does.
    raw_db = (config.get("persistence") or {}).get("db_path", "data/runs.db")
    db_path = Path(raw_db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path

    prod       = section_production(strategy_dir)
    flight     = section_in_flight(strategy_dir, config)
    history    = section_history(db_path, n=args.history_n)
    tournament = section_tournament(strategy_dir)

    if args.json:
        print(json.dumps({
            "strategy":   args.strategy,
            "production": prod,
            "in_flight":  flight,
            "history":    history,
            "tournament": tournament,
        }, indent=2, default=str))
    else:
        print(render_text(args.strategy, prod, flight, history, tournament))
    return 0


if __name__ == "__main__":
    sys.exit(main())
