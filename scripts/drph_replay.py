#!/usr/bin/env python3
"""DRPH replay executor — the S2 behavior-identity gate.

Design: renquant-orchestrator
doc/research/2026-06-12-engineering-architecture-deep-plan.md §IV + S2
item 5: "each extraction: behavior-identical PR gated by the replay
harness (one fixed historical day reproduced bit-identically
before/after — the sim infra already supports this)". Substrate:
kernel/drph.py + scripts/drph_capture.py (PR #313).

What it does: runs ONE fixed historical trading day through the full
decision pipeline (sim adapter, seeded, persistence into a throwaway
sqlite db), extracts the canonical decision snapshot, and either freezes
it as a corpus case (capture) or byte-compares it (verify).

Refactor gate protocol:
  1. on main:        drph_replay.py capture --date D --out tests/drph_corpus/sim_D
  2. on the PR head: drph_replay.py verify  --date D --case tests/drph_corpus/sim_D
  PARITY OK ⇒ the extraction is behavior-identical on that day.

Sim-captured cases gate refactors; live-captured cases (drph_capture.py,
e.g. the 2026-06-11 false-BEAR case) are forensic anchors — live and sim
snapshots are structurally different surfaces and are never compared to
each other.

Data drift vs code drift: every capture freezes an OHLCV fingerprint
(per-ticker row-count + sha of the trailing closes up to the replay
date). verify recomputes it first — a fingerprint mismatch means the
data store was restated/backfilled and the case must be re-captured; it
is reported as DATA-DRIFT, not as a parity failure, so a refactor is
never blamed for a data restatement.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))
sys.path.insert(0, str(REPO / "scripts"))

from drph_capture import extract_decisions  # noqa: E402
from kernel.drph import ReplayCase, canonical_json, sha  # noqa: E402

DEFAULT_SEED = 44
FP_TAIL = 250  # trailing closes hashed per ticker — restatement detector
SUBREPO_IMPORT_ORDER = (
    "renquant-common",
    "renquant-base-data",
    "renquant-artifacts",
    "renquant-model",
    "renquant-pipeline",
    "renquant-execution",
    "renquant-strategy-104",
    "renquant-backtesting",
    "renquant-orchestrator",
)


def _bootstrap_subrepo_imports() -> None:
    """Make the replay executable runnable from temp PR worktrees too."""
    from subrepo_paths import resolve_subrepo_root  # noqa: PLC0415

    subrepo_root = resolve_subrepo_root(REPO).resolve()
    for repo in reversed(SUBREPO_IMPORT_ORDER):
        src = subrepo_root / repo / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))


def _load_config() -> dict:
    return json.loads((STRATEGY_DIR / "strategy_config.json").read_text())


def _load_market_data(config: dict):
    from kernel.data import fetch_ohlcv  # noqa: PLC0415

    spy_df = fetch_ohlcv("SPY")
    etf_map = config.get("sector_etf_map", {})
    symbols = sorted(set(config.get("watchlist") or []) | set(etf_map.values()))
    ohlcv: dict = {"SPY": spy_df}
    for sym in symbols:
        try:
            ohlcv[sym] = fetch_ohlcv(sym)
        except Exception:
            pass  # missing ticker data → universe checks handle it downstream
    return ohlcv, spy_df, etf_map


def _ohlcv_fingerprint(ohlcv: dict, date: str) -> dict:
    """Per-ticker (rows≤date, sha of trailing closes≤date): detects
    restatement of the frozen day's input data without false-alarming on
    ordinary forward appends."""
    out = {}
    for sym in sorted(ohlcv):
        df = ohlcv[sym]
        upto = df[df.index <= date]
        closes = upto["close"].tail(FP_TAIL)
        blob = ",".join(f"{v:.6f}" for v in closes.tolist())
        out[sym] = {"rows": int(len(upto)), "sha": sha(blob)}
    return out


def _run_one_day(date: str, seed: int) -> dict:
    """Seeded single-day sim into a throwaway db; returns the canonical
    decision snapshot."""
    _bootstrap_subrepo_imports()
    from sim.runner import run_backtest  # noqa: PLC0415

    config = _load_config()
    tmp = Path(tempfile.mkdtemp(prefix="drph_replay_"))
    sim_db = tmp / "replay.db"
    config["persistence"] = {
        "enabled": True,
        "db_path": str(tmp / "unused_live.db"),
        "sim_db_path": str(sim_db),
    }
    config["_strategy_dir"] = str(STRATEGY_DIR)

    ohlcv, spy_df, etf_map = _load_market_data(config)
    fingerprint = _ohlcv_fingerprint(ohlcv, date)

    # snapshot=False: the gate compares THIS working tree's behavior;
    # artifact isolation is the caller's job (don't retrain mid-gate).
    run_backtest(
        config=config,
        strategy_dir=STRATEGY_DIR,
        ohlcv=ohlcv,
        spy_df=spy_df,
        sector_etf_map=etf_map,
        backtest_start=date,
        backtest_end=date,
        snapshot=False,
        seed=seed,
    )

    conn = sqlite3.connect(sim_db)
    run_ids = [r[0] for r in conn.execute(
        "SELECT run_id FROM pipeline_runs ORDER BY created_at").fetchall()]
    if len(run_ids) != 1:
        raise SystemExit(
            f"expected exactly 1 sim run for {date}, got {len(run_ids)} "
            f"({run_ids}) — not a trading day, or the window leaked")
    decisions = extract_decisions(conn, run_ids[0])
    conn.close()
    return {"decisions": decisions, "ohlcv_fingerprint": fingerprint}


def cmd_capture(args) -> int:
    result = _run_one_day(args.date, args.seed)
    case = ReplayCase(Path(args.out))
    case_id = case.write(
        inputs={
            "ohlcv_fingerprint": result["ohlcv_fingerprint"],
            "capture_meta": {
                "kind": "sim_replay",
                "date": args.date,
                "seed": args.seed,
                "config": "strategy_config.json",
            },
        },
        expected_decisions=result["decisions"],
    )
    book = result["decisions"]["book"]
    print(f"captured sim replay case id={case_id} → {args.out}")
    print(f"  date={args.date} seed={args.seed} regime={book['regime']} "
          f"tickers={len(result['decisions']['tickers'])} "
          f"decisions_sha={sha(canonical_json(result['decisions']))}")
    return 0


def cmd_verify(args) -> int:
    case = ReplayCase(Path(args.case))
    problems = case.check_integrity()
    if problems:
        print("CORPUS INTEGRITY FAILED:")
        for p in problems:
            print(f"  ✗ {p}")
        return 2
    meta = case.read_input("capture_meta")
    if meta.get("kind") != "sim_replay":
        raise SystemExit(
            f"case kind={meta.get('kind')!r} is not a sim_replay case — "
            f"live-captured cases are forensic anchors, not replay gates")
    date, seed = meta["date"], int(meta["seed"])
    result = _run_one_day(date, seed)

    frozen_fp = case.read_input("ohlcv_fingerprint")
    drifted = sorted(
        s for s in set(frozen_fp) | set(result["ohlcv_fingerprint"])
        if frozen_fp.get(s) != result["ohlcv_fingerprint"].get(s))
    if drifted:
        print(f"DATA-DRIFT — {len(drifted)} ticker(s) restated/backfilled "
              f"up to {date} (first 10): {drifted[:10]}")
        print("the input data changed, not (necessarily) the code — "
              "re-capture the case after auditing the restatement")
        return 3

    ok, diffs = case.verify(result["decisions"])
    if ok:
        print(f"PARITY OK — {date} (seed {seed}) reproduces {args.case} "
              f"bit-identically")
        return 0
    print(f"PARITY FAILED — {len(diffs)} diverging path(s) (first 20):")
    for d in diffs:
        print(f"  ✗ {d}")
    return 1


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    cap = sub.add_parser("capture", help="freeze one sim day as a gate case")
    cap.add_argument("--date", required=True, help="YYYY-MM-DD trading day")
    cap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    cap.add_argument("--out", required=True)
    cap.set_defaults(fn=cmd_capture)
    ver = sub.add_parser("verify", help="re-run the frozen day and byte-compare")
    ver.add_argument("--case", required=True)
    ver.set_defaults(fn=cmd_verify)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
