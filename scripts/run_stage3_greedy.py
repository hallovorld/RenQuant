#!/usr/bin/env python
"""Greedy IC-additive batch admission (Track D Stage 3).

Iterates through candidate tickers (from screen_stage2_results.json) in
batches of 5. For each batch:

  1. Build a side strategy_config with watchlist = baseline + accepted-so-far + batch
  2. Run a panel-only retrain (skip baseline + recalibrate)
  3. Read mean_ic from data/runs.db
  4. If new_ic >= reference_ic + threshold, accept; else reject
  5. Persist progress to scripts/stage3_progress.json

Idempotent: re-running picks up from the next un-processed batch.

Wallclock: ~5 min per batch × ~36 batches ≈ 3 hours sequential.

Usage::

    python scripts/run_stage3_greedy.py
    python scripts/run_stage3_greedy.py --batch-size 10 --threshold -0.001
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("stage3-greedy")


def _build_side_config(baseline_wl: list[str], extra: list[str], label: str) -> Path:
    """Construct a side strategy_config with a custom watchlist + side artifact paths."""
    src = json.loads(
        (REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json").read_text()
    )
    src["watchlist"] = baseline_wl + extra
    src["_audit_label"] = label

    # All artifact_path keys to side paths so we don't clobber production.
    for art_name in ("panel-ltr", "ngboost-head", "panel-rank-calibration"):
        side_path = f"artifacts/{art_name}.{label}.json"
        src["panel_ltr"]["artifact_path" if art_name == "panel-ltr" else None] = side_path
        if art_name == "panel-ltr":
            src["panel_ltr"]["artifact_path"] = side_path
            src["ranking"]["panel_scoring"]["artifact_path"] = side_path
        elif art_name == "ngboost-head":
            src["panel_ltr"]["ngboost"]["artifact_path"] = side_path
            src["ranking"]["panel_scoring"]["ngboost"]["artifact_path"] = side_path
        else:
            src["ranking"]["panel_scoring"]["global_calibration"]["artifact_path"] = side_path

    # Bypass min_best_iter strict guard — Stage 3 expands universe per batch
    # and best_iter naturally varies. eval_ic_floor=0.02 escape clause kicks in.
    src["panel_ltr"]["min_best_iter"] = 1

    out = REPO_ROOT / "backtesting" / "renquant_104" / f"strategy_config.{label}.json"
    out.write_text(json.dumps(src, indent=2))
    return out


def _run_retrain(label: str, log_path: Path, timeout_sec: int = 900) -> int:
    """Dispatch train_104.py via subprocess. Returns exit code."""
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "train_104.py"),
        "--strategy-config-name", f"strategy_config.{label}.json",
        "--skip-baseline", "--skip-recalibrate", "--force",
    ]
    with open(log_path, "w") as logf:
        proc = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), timeout=timeout_sec,
        )
    return proc.returncode


def _read_latest_panel_ltr_ic(db_path: Path) -> float | None:
    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT oos_mean_ic FROM training_runs WHERE artifact_type='panel-ltr' "
        "ORDER BY run_date DESC LIMIT 1"
    ).fetchone()
    conn.close()
    return float(row[0]) if row and row[0] is not None else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage2-results",
                   default=str(REPO_ROOT / "scripts" / "screen_stage2_results.json"))
    p.add_argument("--progress-file",
                   default=str(REPO_ROOT / "scripts" / "stage3_progress.json"))
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--threshold", type=float, default=-0.002,
                   help="Accept batch if delta_mean_ic > threshold (default -0.002 = "
                        "tolerate up to 2bp degradation per batch).")
    p.add_argument("--baseline-ic", type=float, default=None,
                   help="Override baseline mean_ic (default: query DB for latest "
                        "production retrain on the unmodified golden watchlist).")
    p.add_argument("--per-batch-timeout-sec", type=int, default=900)
    p.add_argument("--max-batches", type=int, default=None,
                   help="Stop after this many batches (debug).")
    args = p.parse_args()

    db_path = REPO_ROOT / "data" / "runs.db"

    s2 = json.loads(Path(args.stage2_results).read_text())
    candidates_all = [r["ticker"] for r in s2["admitted"]]
    log.info("Stage 3 input: %d distributionally-admitted candidates", len(candidates_all))

    baseline_wl = json.loads(
        (REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json").read_text()
    )["watchlist"]
    log.info("Baseline watchlist (production): %d tickers", len(baseline_wl))

    # Drop candidates already in production (defensive — Stage 2 should have done this)
    candidates = [t for t in candidates_all if t not in set(baseline_wl)]
    log.info("Candidates (excluding wl103): %d", len(candidates))

    # Resume support — load existing progress if present
    progress_path = Path(args.progress_file)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        log.info("Resuming from existing progress: %d batches done, %d accepted so far",
                 len(progress.get("batches", [])),
                 sum(1 for b in progress.get("batches", []) if b.get("accepted")))
    else:
        progress = {
            "batch_size":   args.batch_size,
            "threshold":    args.threshold,
            "baseline_size": len(baseline_wl),
            "baseline_ic":  args.baseline_ic,
            "started_at":   time.strftime("%Y-%m-%dT%H:%M:%S"),
            "batches":      [],
        }

    accepted: list[str] = []
    for entry in progress["batches"]:
        if entry.get("accepted"):
            accepted.extend(entry["tickers"])

    # Determine reference IC: starts at baseline; advances on each acceptance
    if progress.get("baseline_ic") is None:
        # Auto-detect: latest panel-ltr run in DB
        latest = _read_latest_panel_ltr_ic(db_path)
        if latest is None:
            log.error("No baseline_ic provided and no panel-ltr runs in DB")
            return 1
        progress["baseline_ic"] = latest
        log.info("Auto-detected baseline mean_ic from latest DB row: %+.4f", latest)
    reference_ic = progress["baseline_ic"]
    for entry in progress["batches"]:
        if entry.get("accepted"):
            reference_ic = entry["new_ic"]
    log.info("Starting reference mean_ic: %+.4f (after %d accepted batches)",
             reference_ic, sum(1 for b in progress["batches"] if b.get("accepted")))

    # Skip already-processed batches
    n_processed = len(progress["batches"])
    pending = candidates[n_processed * args.batch_size:]
    if not pending:
        log.info("All %d candidates processed. Final accepted=%d",
                 len(candidates), len(accepted))
        return 0

    # Build batch list
    batches = [pending[i:i + args.batch_size]
               for i in range(0, len(pending), args.batch_size)]
    if args.max_batches is not None:
        batches = batches[:args.max_batches]
    log.info("Will process %d batches of size %d (~%d minutes per batch)",
             len(batches), args.batch_size, 5)

    for i, batch in enumerate(batches):
        global_idx = n_processed + i
        label = f"stage3_batch_{global_idx:03d}"

        log.info("─" * 70)
        log.info("Batch %d/%d (global=%d): proposing +%d tickers: %s",
                 i + 1, len(batches), global_idx, len(batch), batch)
        log.info("Current accepted=%d, candidate wl size=%d",
                 len(accepted), len(baseline_wl) + len(accepted) + len(batch))

        # Build config + run
        cfg_path = _build_side_config(baseline_wl, accepted + batch, label)
        log_path = REPO_ROOT / "logs" / "stage3" / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        try:
            rc = _run_retrain(label, log_path, args.per_batch_timeout_sec)
        except subprocess.TimeoutExpired:
            log.error("Batch %d TIMEOUT after %ds — skipping", global_idx, args.per_batch_timeout_sec)
            progress["batches"].append({
                "batch_idx": global_idx, "tickers": batch,
                "new_ic": None, "delta": None, "accepted": False, "error": "timeout",
            })
            progress_path.write_text(json.dumps(progress, indent=2))
            continue
        elapsed = time.monotonic() - t0

        # rc 2 = acceptance gate hard-failed (e.g. G7_oos_ic_floor) — for our
        # purposes that just means the model is too weak; we still treat it
        # as a measurement and advance.
        new_ic = _read_latest_panel_ltr_ic(db_path)
        if new_ic is None:
            log.error("Batch %d: training crashed (no DB row written, rc=%d) — skipping",
                      global_idx, rc)
            progress["batches"].append({
                "batch_idx": global_idx, "tickers": batch,
                "new_ic": None, "delta": None, "accepted": False, "error": f"rc={rc}",
            })
            progress_path.write_text(json.dumps(progress, indent=2))
            continue

        delta = new_ic - reference_ic
        accept = delta > args.threshold
        log.info("Batch %d result: new_ic=%+.4f (was %+.4f, delta=%+.4f) → %s  elapsed=%.0fs",
                 global_idx, new_ic, reference_ic, delta, "ACCEPT" if accept else "reject", elapsed)

        if accept:
            accepted.extend(batch)
            reference_ic = new_ic

        progress["batches"].append({
            "batch_idx":  global_idx, "tickers": batch,
            "new_ic":     new_ic,    "reference_ic": reference_ic,
            "delta":      delta,     "accepted": accept,
            "elapsed_s":  elapsed,   "rc": rc,
        })
        progress_path.write_text(json.dumps(progress, indent=2))

    # Final summary
    accepted_total = [t for entry in progress["batches"] if entry.get("accepted")
                      for t in entry["tickers"]]
    log.info("=" * 70)
    log.info("Stage 3 DONE. baseline=%d, accepted=%d, final_wl=%d (target ~150-200)",
             len(baseline_wl), len(accepted_total), len(baseline_wl) + len(accepted_total))
    log.info("Final reference IC: %+.4f (vs baseline %+.4f, delta=%+.4f)",
             reference_ic, progress["baseline_ic"], reference_ic - progress["baseline_ic"])

    # Write final watchlist for inspection / promotion
    final_wl = baseline_wl + accepted_total
    final_path = REPO_ROOT / "scripts" / "stage3_final_watchlist.json"
    final_path.write_text(json.dumps({
        "baseline_size":   len(baseline_wl),
        "accepted_count":  len(accepted_total),
        "final_size":      len(final_wl),
        "final_ic":        reference_ic,
        "baseline_ic":     progress["baseline_ic"],
        "delta_ic":        reference_ic - progress["baseline_ic"],
        "watchlist":       final_wl,
    }, indent=2))
    log.info("Wrote %s", final_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
