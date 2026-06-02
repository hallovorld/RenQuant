#!/usr/bin/env python
"""Per-regime walk-forward training driver (Track C, 2026-06-02).

Produces N cuts × K specialists by looping cutoffs × regimes and
delegating per-cutoff training to ``train_per_regime_panel.py``. Emits
one manifest per regime so the WF gate can evaluate each specialist
independently::

    artifacts/sim/walkforward_manifest_v2_{stamp}_per_regime_{regime}.json

Per CLAUDE.md §1 PRIME DIRECTIVE + Track C plan: pooled-mean training
across regimes is the structural reason the production GBDT has no
BULL_CALM ranking signal. Specialists optimize per regime — and this
driver produces the 39×4 evaluation matrix the gate needs.

Per §7.5: no forking. We reuse ``train_per_regime_panel.py``'s
``main()`` (it owns regime filtering) and ``kernel.walk_forward``'s
manifest writer (single source for retrain-entry schema).

Usage::

    python scripts/train_per_regime_walkforward.py \\
        --start-date 2024-01-01 --end-date 2026-03-09 \\
        --cadence-days 21 \\
        --regimes BULL_CALM,BEAR,BULL_VOLATILE,CHOPPY \\
        --artifact-root walkforward_per_regime \\
        --jobs 8
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# §6.5 hardware saturation
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, str(os.cpu_count() or 14))

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
PER_REGIME_DRIVER = REPO / "scripts" / "train_per_regime_panel.py"
CALIBRATOR_SCRIPT = REPO / "scripts" / "fit_calibrator_alpha158_fund.py"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(STRATEGY_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-per-regime-wf")

CANONICAL_REGIMES = ("BULL_CALM", "BEAR", "BULL_VOLATILE", "CHOPPY")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", required=True, help="First retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--end-date",   required=True, help="Last retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--cadence-days", type=int, default=21,
                   help="Days between retrain cutoffs (default 21).")
    p.add_argument("--regimes", default=",".join(CANONICAL_REGIMES),
                   help="Comma-separated regimes to train (default: all 4).")
    p.add_argument("--artifact-root", default="walkforward_per_regime",
                   help="Subdir under strategy_dir/artifacts/ (default walkforward_per_regime).")
    p.add_argument("--manifest-root", default="artifacts/sim",
                   help="Where per-regime manifests are written (relative to strategy dir).")
    p.add_argument("--jobs", type=int, default=1,
                   help="Concurrent (cutoff, regime) retrains.")
    p.add_argument("--label", default=None, help="Override label column.")
    p.add_argument("--fingerprint-config", default=None,
                   help="Strategy config whose model-relevant fields stamp each artifact.")
    p.add_argument("--skip-calibrators", action="store_true",
                   help="Skip the matching causal calibrator fit per cutoff.")
    p.add_argument("--calibrator-method", default="platt", choices=["platt", "isotonic"])
    p.add_argument("--dry-run", action="store_true",
                   help="Print the per-(cutoff, regime) commands and exit.")
    return p.parse_args()


def _parse_regimes(spec: str) -> list[str]:
    regimes = [r.strip().upper() for r in spec.split(",") if r.strip()]
    bad = [r for r in regimes if r not in CANONICAL_REGIMES]
    if bad:
        raise SystemExit(f"--regimes contains unknown values {bad}. Allowed: {CANONICAL_REGIMES}")
    return regimes


def _artifact_path(strategy_dir: Path, artifact_root: str,
                   regime: str, cutoff_iso: str) -> Path:
    return (strategy_dir / "artifacts" / artifact_root /
            regime.lower() / cutoff_iso / "panel-ltr.json")


def _calibrator_path(artifact_path: Path) -> Path:
    return artifact_path.with_name("panel-rank-calibration.json")


def _infer_lookahead_days(label: str | None) -> int:
    import re
    m = re.search(r"fwd_(\d+)d", str(label or "fwd_60d_excess"))
    return int(m.group(1)) if m else 60


def _train_one(cutoff_iso: str, regime: str, args: argparse.Namespace,
               lookahead_days: int) -> tuple[bool, Path, Path | None, str]:
    """Subprocess train_per_regime_panel.py + optional calibrator."""
    art = _artifact_path(STRATEGY_DIR, args.artifact_root, regime, cutoff_iso)
    art.parent.mkdir(parents=True, exist_ok=True)
    side_label = f"specialist_wf_{regime.lower()}_{cutoff_iso}"
    cmd = [
        sys.executable, str(PER_REGIME_DRIVER),
        "--regime-filter", regime,
        "--output-path", str(art),
        "--side-label", side_label,
        "--train-cutoff", cutoff_iso,
    ]
    if args.label:
        cmd.extend(["--label", args.label])
    if args.fingerprint_config:
        cmd.extend(["--fingerprint-config", args.fingerprint_config])
    if args.dry_run:
        print(" ".join(cmd))
        return True, art, None, ""

    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    if proc.returncode != 0:
        return False, art, None, f"exit={proc.returncode} stderr_tail={proc.stderr[-400:]!r}"
    if args.skip_calibrators:
        return True, art, None, ""

    cal = _calibrator_path(art)
    data_end = (pd.Timestamp(cutoff_iso) - pd.offsets.BDay(lookahead_days)).date().isoformat()
    cmd2 = [
        sys.executable, str(CALIBRATOR_SCRIPT),
        "--scorer-artifact", str(art),
        "--out", str(cal),
        "--data-end", data_end,
        "--method", args.calibrator_method,
    ]
    proc2 = subprocess.run(cmd2, cwd=str(REPO), capture_output=True, text=True)
    if proc2.returncode != 0:
        return False, art, cal, f"calibrator exit={proc2.returncode} stderr_tail={proc2.stderr[-400:]!r}"
    return True, art, cal, ""


def _manifest_path(manifest_root: Path, stamp: str, regime: str) -> Path:
    manifest_root.mkdir(parents=True, exist_ok=True)
    return manifest_root / f"walkforward_manifest_v2_{stamp}_per_regime_{regime.lower()}.json"


def _build_manifest_entries(entries_by_cutoff: dict, lookahead_days: int) -> list:
    """Wrap per-cutoff (artifact, calibrator) into RetrainEntry objects."""
    from kernel.walk_forward import RetrainEntry  # noqa: PLC0415
    out = []
    for cutoff_iso in sorted(entries_by_cutoff):
        art, cal = entries_by_cutoff[cutoff_iso]
        if art is None or not Path(art).exists():
            continue
        artifact = json.loads(Path(art).read_text())
        out.append(RetrainEntry(
            cutoff_date=pd.Timestamp(cutoff_iso),
            trained_date=pd.Timestamp(artifact.get("trained_date") or cutoff_iso),
            artifact_uri=str(art),
            lookahead_days=int(lookahead_days),
            calibrator_uri=str(cal) if cal else None,
            effective_train_cutoff_date=(
                pd.Timestamp(artifact["effective_train_cutoff_date"])
                if artifact.get("effective_train_cutoff_date") else None
            ),
        ))
    return out


def _train_regime(regime: str, retrain_dates: list[pd.Timestamp],
                  args: argparse.Namespace, lookahead_days: int) -> dict:
    """Train all cutoffs for one regime; return cutoff_iso → (artifact, calibrator)."""
    jobs = max(1, min(int(args.jobs), len(retrain_dates)))
    entries: dict = {}
    failures: list[tuple[str, str]] = []

    def _do(cutoff: pd.Timestamp):
        ok, art, cal, err = _train_one(cutoff.date().isoformat(), regime, args, lookahead_days)
        return cutoff.date().isoformat(), ok, art, cal, err

    log.info("regime=%s start (%d cutoffs, jobs=%d)", regime, len(retrain_dates), jobs)
    if jobs == 1:
        # 2026-06-02 codex MED fix: `failures` is a list[tuple], not a dict,
        # so the prior `(entries if ok else failures).setdefault(iso, None)`
        # crashed with `AttributeError: 'list' object has no attribute
        # 'setdefault'` on the FIRST sequential cutoff failure. Branch
        # explicitly per outcome — same shape as the parallel branch below.
        for c in retrain_dates:
            iso, ok, art, cal, err = _do(c)
            if ok:
                entries[iso] = (str(art), str(cal) if cal else None)
            else:
                failures.append((iso, err))
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix=f"wf-{regime}") as pool:
            futs = {pool.submit(_do, c): c for c in retrain_dates}
            for fut in as_completed(futs):
                iso, ok, art, cal, err = fut.result()
                if ok:
                    entries[iso] = (str(art), str(cal) if cal else None)
                else:
                    failures.append((iso, err))
    log.info("regime=%s done: %d ok / %d failed", regime, len(entries), len(failures))
    if failures:
        log.warning("regime=%s failures: %s", regime, failures[:5])
    return entries


def main() -> None:
    args = parse_args()
    regimes = _parse_regimes(args.regimes)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    retrain_dates = list(pd.date_range(
        pd.Timestamp(args.start_date),
        pd.Timestamp(args.end_date),
        freq=f"{args.cadence_days}D",
    ))
    lookahead_days = _infer_lookahead_days(args.label)
    manifest_root = STRATEGY_DIR / args.manifest_root

    log.info("Per-regime WF plan: regimes=%s cuts=%d cadence=%dd jobs=%d",
             regimes, len(retrain_dates), args.cadence_days, args.jobs)
    if args.dry_run:
        for regime in regimes:
            print(f"# regime={regime}")
            for c in retrain_dates:
                _train_one(c.date().isoformat(), regime, args, lookahead_days)
            print()
        return

    from kernel.walk_forward import WalkForwardManifest, write_manifest  # noqa: PLC0415
    for regime in regimes:
        entries_by_cutoff = _train_regime(regime, retrain_dates, args, lookahead_days)
        manifest = WalkForwardManifest(
            cadence_days=int(args.cadence_days),
            training_window_years=0.0,
            retrains=_build_manifest_entries(entries_by_cutoff, lookahead_days),
        )
        out = _manifest_path(manifest_root, stamp, regime)
        write_manifest(manifest, str(out))
        log.info("Wrote manifest regime=%s → %s (%d entries)",
                 regime, out, len(manifest.retrains))


if __name__ == "__main__":
    main()
