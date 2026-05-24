#!/usr/bin/env python
"""Walk-forward HF PatchTST training driver for renquant_104.

This is the sequence-model companion to ``train_walkforward_panel.py``. It
does not train inside this file. Each cutoff is a subprocess call to the
canonical HF Trainer script, followed by a causal per-fold calibrator fit and a
standard ``kernel.walk_forward`` manifest entry.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "6")

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "patchtst_hf.py"
CALIBRATOR_SCRIPT = REPO_ROOT / "scripts" / "fit_hf_patchtst_calibrator.py"
DEFAULT_ROOT = "walkforward_patchtst"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STRATEGY_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-walkforward-patchtst")


def infer_label_lookahead_days(label: str | None) -> int:
    import re
    m = re.search(r"fwd_(\d+)d", str(label or "fwd_60d_excess"))
    return int(m.group(1)) if m else 60


def data_end_for_cutoff(cutoff: pd.Timestamp, label: str | None) -> str:
    lookahead = infer_label_lookahead_days(label)
    return (cutoff - pd.offsets.BDay(lookahead)).date().isoformat()


def compute_retrain_dates(start: pd.Timestamp, end: pd.Timestamp,
                          cadence_days: int) -> list[pd.Timestamp]:
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be > 0, got {cadence_days}")
    return list(pd.date_range(start, end, freq=f"{cadence_days}D"))


def artifact_dir(args: argparse.Namespace, cutoff: pd.Timestamp) -> Path:
    root = args.artifact_root or DEFAULT_ROOT
    return STRATEGY_DIR / "artifacts" / root / cutoff.date().isoformat()


def model_path_for(out_dir: Path, seed: int) -> Path:
    return out_dir / f"hf_patchtst_all_seed{seed}_model.pt"


def calibrator_path_for(model_path: Path) -> Path:
    return model_path.with_name("hf_patchtst-calibration.json")


def sidecar_path_for(model_path: Path) -> Path:
    return model_path.with_name(model_path.name + ".metadata.json")


def train_cmd(args: argparse.Namespace, cutoff: pd.Timestamp,
              out_dir: Path) -> list[str]:
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--dataset", args.dataset,
        "--cut", "all",
        "--train-cutoff", cutoff.date().isoformat(),
        "--label", args.label,
        "--epochs", str(args.epochs),
        "--seq-len", str(args.seq_len),
        "--patch-length", str(args.patch_length),
        "--d-model", str(args.d_model),
        "--n-heads", str(args.n_heads),
        "--n-layers", str(args.n_layers),
        "--lr", str(args.lr),
        "--weight-decay", str(args.weight_decay),
        "--device", args.device,
        "--seed", str(args.seed),
        "--save-model",
        "--output-dir", str(out_dir),
    ]
    if args.strategy_config:
        cmd.extend(["--strategy-config", args.strategy_config])
    if args.film_regime_cond:
        cmd.append("--film-regime-cond")
    if args.cross_stock_attn:
        cmd.append("--cross-stock-attn")
    return cmd


def calibrator_cmd(args: argparse.Namespace, cutoff: pd.Timestamp,
                   model_path: Path, cal_path: Path) -> list[str]:
    return [
        sys.executable, str(CALIBRATOR_SCRIPT),
        "--scorer-artifact", str(model_path),
        "--out", str(cal_path),
        "--data-end", data_end_for_cutoff(cutoff, args.label),
        "--batch-size", str(args.calibrator_batch_size),
        "--method", args.calibrator_method,
    ]


def read_contract(model_path: Path) -> dict:
    sidecar = sidecar_path_for(model_path)
    if not sidecar.exists():
        raise FileNotFoundError(f"missing PatchTST metadata sidecar: {sidecar}")
    payload = json.loads(sidecar.read_text())
    contract = payload.get("training_contract") or {}
    if not contract.get("trained_date") or not contract.get("effective_train_cutoff_date"):
        raise ValueError(f"incomplete PatchTST sidecar contract: {sidecar}")
    return contract


def build_entry(cutoff: pd.Timestamp, model_path: Path,
                cal_path: Path | None, label: str | None):
    from kernel.walk_forward import RetrainEntry  # noqa: PLC0415
    contract = read_contract(model_path)
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=pd.Timestamp(contract["trained_date"]),
        artifact_uri=str(model_path),
        lookahead_days=infer_label_lookahead_days(label),
        calibrator_uri=str(cal_path) if cal_path else None,
        effective_train_cutoff_date=pd.Timestamp(
            contract["effective_train_cutoff_date"]
        ),
    )


def run_subprocess(cmd: list[str], label: str) -> tuple[bool, str]:
    log.info("%s start: %s", label, " ".join(cmd))
    t0 = time.monotonic()
    proc = subprocess.run(
        cmd, cwd=str(REPO_ROOT), check=False,
        capture_output=True, text=True,
    )
    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        return False, f"{label} exit={proc.returncode} stderr_tail={proc.stderr[-800:]!r}"
    log.info("%s done in %.1fs", label, elapsed)
    return True, ""


def train_one_cutoff(args: argparse.Namespace, cutoff: pd.Timestamp):
    out_dir = artifact_dir(args, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_path_for(out_dir, int(args.seed))
    if args.reuse_existing and model_path.exists() and sidecar_path_for(model_path).exists():
        log.info("train cutoff=%s reuse existing model: %s",
                 cutoff.date(), model_path)
    else:
        ok, err = run_subprocess(train_cmd(args, cutoff, out_dir),
                                 f"train cutoff={cutoff.date()}")
        if not ok:
            return cutoff, None, err
    cal_path = None
    if not args.skip_calibrators:
        cal_path = calibrator_path_for(model_path)
        if args.reuse_existing and cal_path.exists():
            log.info("calibrate cutoff=%s reuse existing calibrator: %s",
                     cutoff.date(), cal_path)
        else:
            ok, err = run_subprocess(calibrator_cmd(args, cutoff, model_path, cal_path),
                                     f"calibrate cutoff={cutoff.date()}")
            if not ok:
                return cutoff, None, err
    return cutoff, build_entry(cutoff, model_path, cal_path, args.label), ""


def train_cutoffs(args: argparse.Namespace, dates: list[pd.Timestamp]):
    jobs = max(1, min(int(args.jobs), len(dates)))
    entries_by_cutoff: dict[str, object] = {}
    failed: list[tuple[str, str]] = []

    if jobs == 1:
        for cutoff in dates:
            _, entry, err = train_one_cutoff(args, cutoff)
            key = cutoff.date().isoformat()
            if entry is None:
                failed.append((key, err))
            else:
                entries_by_cutoff[key] = entry
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="wf-pt") as pool:
            futures = {pool.submit(train_one_cutoff, args, d): d for d in dates}
            for fut in as_completed(futures):
                cutoff = futures[fut]
                key = cutoff.date().isoformat()
                try:
                    _, entry, err = fut.result()
                except Exception as exc:  # noqa: BLE001
                    failed.append((key, repr(exc)))
                    log.exception("cutoff=%s crashed", key)
                    continue
                if entry is None:
                    failed.append((key, err))
                else:
                    entries_by_cutoff[key] = entry

    entries = [
        entries_by_cutoff[d.date().isoformat()]
        for d in dates
        if d.date().isoformat() in entries_by_cutoff
    ]
    if failed and not args.allow_partial_manifest:
        preview = ", ".join(f"{c}: {e}" for c, e in failed[:5])
        raise RuntimeError(
            "train_walkforward_patchtst: refusing partial manifest "
            f"({len(entries)}/{len(dates)} succeeded). First failures: {preview}"
        )
    return entries, failed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--cadence-days", type=int, default=21)
    p.add_argument("--manifest-output",
                   default=str(STRATEGY_DIR / "artifacts" / "walkforward_patchtst_manifest.json"))
    p.add_argument("--artifact-root", default=None)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--patch-length", type=int, default=4)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    p.add_argument("--strategy-config", default=None)
    p.add_argument("--film-regime-cond", action="store_true")
    p.add_argument("--cross-stock-attn", action="store_true")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--skip-calibrators", action="store_true")
    p.add_argument("--calibrator-batch-size", type=int, default=512)
    p.add_argument("--calibrator-method", default="platt",
                   choices=["platt", "isotonic"])
    p.add_argument("--reuse-existing", action="store_true",
                   help="Reuse existing model sidecar/calibrator artifacts for "
                        "a cutoff instead of rerunning completed subprocesses.")
    p.add_argument("--allow-partial-manifest", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dates = compute_retrain_dates(
        pd.Timestamp(args.start_date), pd.Timestamp(args.end_date),
        int(args.cadence_days),
    )
    if args.dry_run:
        for i, cutoff in enumerate(dates):
            out_dir = artifact_dir(args, cutoff)
            print(f"[{i + 1:02d}/{len(dates)}] cutoff={cutoff.date()} "
                  f"data_end={data_end_for_cutoff(cutoff, args.label)} "
                  f"model={model_path_for(out_dir, int(args.seed))}")
        print(f"Manifest output: {args.manifest_output}")
        return

    from kernel.walk_forward import WalkForwardManifest, write_manifest  # noqa: PLC0415
    entries, failed = train_cutoffs(args, dates)
    manifest = WalkForwardManifest(
        cadence_days=int(args.cadence_days),
        training_window_years=0.0,
        retrains=entries,
    )
    out = write_manifest(manifest, args.manifest_output)
    log.info("Wrote PatchTST manifest with %d/%d retrains -> %s",
             len(entries), len(dates), out)
    if failed:
        log.warning("FAILED cutoffs (%d): %s", len(failed), failed)


if __name__ == "__main__":
    main()
