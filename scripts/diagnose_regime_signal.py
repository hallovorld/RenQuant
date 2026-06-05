#!/usr/bin/env python3
"""Per-regime signal diagnostic — is a weak regime's weakness a model gap
or a regime property?

Reads the decision-trace sim DB (``ticker_forward_returns`` joined to
``pipeline_runs.regime`` by date) and reports, per regime:

  1. Cross-sectional return dispersion (std of fwd at the horizon) and the
     dispersion / |mean| ratio. High dispersion = lots of ranking room;
     low dispersion = "everyone moves together", a ranker can't add value.
  2. The cross-sectional IC of a naive trailing-momentum factor
     (Jegadeesh-Titman 1993) computed by time-lagging the same forward
     column. A momentum IC well above the production model's IC says the
     signal is catchable and the model is under-using a classical factor.

Motivation: 2026-06-05 BULL_CALM diagnostic
(doc/research/2026-06-05-bull-calm-signal-is-catchable-momentum-diagnostic.md).
It found BULL_CALM has the HIGHEST dispersion (11%) yet the lowest model
IC (+0.011), while naive momentum lands +0.039 there — the weakness is a
model/feature gap, not a regime property. This script makes that read
reproducible.

Usage:
    python scripts/diagnose_regime_signal.py --db data/sim_runs.db \
        --horizon 20 --format text
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_VALID_HORIZONS = (1, 5, 10, 20, 60)
_MIN_NAMES_PER_DATE = 5   # need a cross-section to rank


def _fwd_col(horizon: int) -> str:
    if horizon not in _VALID_HORIZONS:
        raise ValueError(
            f"horizon {horizon} not in {_VALID_HORIZONS}; the "
            "ticker_forward_returns table only carries those columns."
        )
    return f"fwd_{horizon}d"


def _load_rows(conn: sqlite3.Connection, horizon: int):
    """Return [(date, ticker, fwd, regime)] for non-null fwd rows joined to
    the per-date regime from pipeline_runs."""
    col = _fwd_col(horizon)
    q = f"""
        SELECT f.as_of_date, f.ticker, f.{col}, pr.regime
          FROM ticker_forward_returns f
          JOIN (SELECT DISTINCT run_date, regime
                  FROM pipeline_runs WHERE run_type='sim') pr
            ON pr.run_date = f.as_of_date
         WHERE f.{col} IS NOT NULL
         ORDER BY f.as_of_date
    """
    return list(conn.execute(q))


def dispersion_by_regime(rows) -> dict:
    """Avg per-date cross-sectional std + |mean| per regime."""
    by_rd: dict[tuple, list[float]] = defaultdict(list)
    for d, _t, fwd, regime in rows:
        by_rd[(regime, d)].append(fwd)
    acc: dict[str, dict] = defaultdict(lambda: {"disp": [], "absmean": []})
    for (regime, _d), fwds in by_rd.items():
        if len(fwds) < _MIN_NAMES_PER_DATE:
            continue
        acc[regime]["disp"].append(statistics.pstdev(fwds))
        acc[regime]["absmean"].append(abs(statistics.mean(fwds)))
    out: dict[str, dict] = {}
    for regime, d in acc.items():
        disp = statistics.mean(d["disp"]) if d["disp"] else 0.0
        am = statistics.mean(d["absmean"]) if d["absmean"] else 0.0
        out[regime or "(null)"] = {
            "n_dates": len(d["disp"]),
            "dispersion_std": disp,
            "abs_mean": am,
            "disp_over_absmean": (disp / am) if am > 0 else 0.0,
        }
    return out


def momentum_ic_by_regime(rows, horizon: int) -> dict:
    """Cross-sectional IC of trailing-`horizon` momentum -> forward return.

    Trailing momentum at date t (per ticker) is approximated by that
    ticker's forward return `horizon` rows earlier in its own date series —
    i.e. the realised return over [t-horizon, t]. Forward is the return at
    t. IC is the per-date cross-sectional Spearman correlation, averaged
    per regime.
    """
    from scipy.stats import spearmanr  # noqa: PLC0415

    tk: dict[str, list[tuple[str, float]]] = defaultdict(list)
    regime_of: dict[str, str] = {}
    for d, t, fwd, regime in rows:
        tk[t].append((d, fwd))
        regime_of[d] = regime
    fwd_map = {(t, d): fwd for t in tk for d, fwd in tk[t]}

    pairs_by_rd: dict[tuple, list[tuple[float, float]]] = defaultdict(list)
    for t, series in tk.items():
        dates = [d for d, _ in series]
        for i in range(horizon, len(dates)):
            d_now, d_prev = dates[i], dates[i - horizon]
            mom = fwd_map.get((t, d_prev))
            fwd = fwd_map.get((t, d_now))
            if mom is None or fwd is None:
                continue
            pairs_by_rd[(regime_of[d_now], d_now)].append((mom, fwd))

    ics_by_regime: dict[str, list[float]] = defaultdict(list)
    for (regime, _d), pairs in pairs_by_rd.items():
        if len(pairs) < _MIN_NAMES_PER_DATE:
            continue
        moms = [p[0] for p in pairs]
        fwds = [p[1] for p in pairs]
        ic, _ = spearmanr(moms, fwds)
        if ic == ic:  # not NaN
            ics_by_regime[regime].append(ic)

    out: dict[str, dict] = {}
    for regime, ics in ics_by_regime.items():
        out[regime or "(null)"] = {
            "n_dates": len(ics),
            "mean_ic": statistics.mean(ics) if ics else 0.0,
            "hit_rate_ic_pos": (sum(1 for x in ics if x > 0) / len(ics)) if ics else 0.0,
        }
    return out


def run_diagnostic(db_path: Path, horizon: int) -> dict:
    if not db_path.exists():
        raise SystemExit(f"diagnose_regime_signal: DB not found at {db_path}")
    conn = sqlite3.connect(str(db_path))
    try:
        rows = _load_rows(conn, horizon)
    finally:
        conn.close()
    return {
        "db": str(db_path),
        "horizon_days": horizon,
        "n_rows": len(rows),
        "dispersion": dispersion_by_regime(rows),
        "momentum_ic": momentum_ic_by_regime(rows, horizon),
    }


def render_text(result: dict) -> str:
    lines = [
        f"=== per-regime signal diagnostic (fwd_{result['horizon_days']}d, "
        f"{result['n_rows']} rows) ===",
        "",
        f"{'regime':<15}{'n_dates':>8}{'disp(std)':>11}{'|mean|':>9}"
        f"{'disp/|mean|':>12}",
        "-" * 55,
    ]
    disp = result["dispersion"]
    for regime in sorted(disp, key=lambda r: -disp[r]["n_dates"]):
        d = disp[regime]
        lines.append(
            f"{regime:<15}{d['n_dates']:>8}{d['dispersion_std']*100:>10.2f}%"
            f"{d['abs_mean']*100:>8.2f}%{d['disp_over_absmean']:>12.2f}"
        )
    lines += ["", f"{'regime':<15}{'n_dates':>8}{'momentum_IC':>13}{'hit>0':>8}", "-" * 44]
    mic = result["momentum_ic"]
    for regime in sorted(mic, key=lambda r: -mic[r]["n_dates"]):
        m = mic[regime]
        lines.append(
            f"{regime:<15}{m['n_dates']:>8}{m['mean_ic']:>+13.4f}"
            f"{m['hit_rate_ic_pos']*100:>7.0f}%"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db", default="data/sim_runs.db",
                   help="Path to the decision-trace sim DB.")
    p.add_argument("--horizon", type=int, default=20, choices=_VALID_HORIZONS,
                   help="Forward-return horizon (must be a populated column).")
    p.add_argument("--format", choices=("text", "json"), default="text")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = REPO_ROOT / db_path
    result = run_diagnostic(db_path, args.horizon)
    if args.format == "json":
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(render_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
