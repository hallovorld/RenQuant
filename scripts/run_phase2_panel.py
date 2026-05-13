#!/usr/bin/env python
"""Phase 2 sim panel runner: K configs × 16 non-overlapping 3mo windows.

Spawns sims subprocess-style with throttled parallelism (default 8 at a
time) so M2 Pro 32GB doesn't swap. Emits equity JSON per (config, window)
into a clean directory structure for downstream eval_paired_returns.py
and eval_regime_stratified.py.

Usage:
    python scripts/run_phase2_panel.py \\
        --configs sim_baseline_ext,sim_vt15_ext,sim_GK094_ext,sim_GK15_ext,sim_GK_conditional_ext \\
        --output-root data/logs/sim_2026-05-12_phase2 \\
        --concurrent 8
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("phase2")

REPO = Path(__file__).resolve().parent.parent

# 16 non-overlapping 3-month windows (2022-04-01 → 2026-04-01 = 48 months)
WINDOWS = [
    ("Q01", "2022-04-01", "2022-07-01"),
    ("Q02", "2022-07-01", "2022-10-01"),
    ("Q03", "2022-10-01", "2023-01-01"),
    ("Q04", "2023-01-01", "2023-04-01"),
    ("Q05", "2023-04-01", "2023-07-01"),
    ("Q06", "2023-07-01", "2023-10-01"),
    ("Q07", "2023-10-01", "2024-01-01"),
    ("Q08", "2024-01-01", "2024-04-01"),
    ("Q09", "2024-04-01", "2024-07-01"),
    ("Q10", "2024-07-01", "2024-10-01"),
    ("Q11", "2024-10-01", "2025-01-01"),
    ("Q12", "2025-01-01", "2025-04-01"),
    ("Q13", "2025-04-01", "2025-07-01"),
    ("Q14", "2025-07-01", "2025-10-01"),
    ("Q15", "2025-10-01", "2026-01-01"),
    ("Q16", "2026-01-01", "2026-03-26"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", required=True,
                   help="Comma-separated config base names (e.g., sim_baseline_ext)")
    p.add_argument("--output-root", required=True)
    p.add_argument("--concurrent", type=int, default=8)
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip (config, window) pairs whose equity JSON already exists")
    args = p.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    # Build full task list: (config, window) pairs
    tasks = []
    for cfg in configs:
        cdir = root / cfg
        (cdir / "equity").mkdir(parents=True, exist_ok=True)
        (cdir / "logs").mkdir(parents=True, exist_ok=True)
        for win, start, end in WINDOWS:
            eq_path = cdir / "equity" / f"{win}.json"
            log_path = cdir / "logs" / f"{win}.log"
            if args.skip_existing and eq_path.exists() and eq_path.stat().st_size > 100:
                log.info(f"SKIP {cfg}/{win} (already done)")
                continue
            tasks.append((cfg, win, start, end, str(eq_path), str(log_path)))
    log.info(f"Phase 2 launcher: {len(tasks)} sim tasks, concurrent={args.concurrent}")

    # Throttled parallel execution
    in_flight: dict[int, dict] = {}
    completed, failed = 0, 0
    task_iter = iter(tasks)
    t_start = time.time()
    try:
        while True:
            while len(in_flight) < args.concurrent:
                try:
                    cfg, win, start, end, eq_path, log_path = next(task_iter)
                except StopIteration:
                    break
                cmd = [
                    sys.executable, str(REPO / "scripts" / "run_sim_104.py"),
                    "--start", start, "--end", end,
                    "--strategy-config-name", f"strategy_config.{cfg}.json",
                    "--equity-json", eq_path,
                    "--no-persist", "--no-compare",
                ]
                fh = open(log_path, "w")
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(REPO))
                in_flight[proc.pid] = {"proc": proc, "fh": fh, "cfg": cfg, "win": win,
                                       "start_time": time.time()}
                log.info(f"  launched {cfg}/{win} PID={proc.pid}")
            if not in_flight:
                break
            # Poll until something exits
            time.sleep(5)
            for pid, info in list(in_flight.items()):
                rc = info["proc"].poll()
                if rc is None:
                    continue
                info["fh"].close()
                dt = time.time() - info["start_time"]
                if rc == 0:
                    completed += 1
                    log.info(f"  ✓ {info['cfg']}/{info['win']} done ({dt:.0f}s)")
                else:
                    failed += 1
                    log.error(f"  ✗ {info['cfg']}/{info['win']} FAILED rc={rc} ({dt:.0f}s)")
                del in_flight[pid]
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt — terminating in-flight sims")
        for info in in_flight.values():
            info["proc"].terminate()
        raise

    elapsed = (time.time() - t_start) / 60
    log.info(f"\n✓ Phase 2 done: {completed} completed, {failed} failed in {elapsed:.1f} min")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
