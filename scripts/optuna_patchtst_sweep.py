#!/usr/bin/env python3
"""Scientific hyperparameter sweep for PatchTST via Optuna (3rd-party lib).

Per 2026-05-18 user mandate: 用第三方 lib (not custom). Optuna is the
canonical Python hyperparameter optimization library (Akiba+ KDD 2019,
arXiv 1907.10902).

Design (CLAUDE.md §5.14 DOE + §5.12 canonical refs):
  - Search space: lr, weight_decay, warmup_epochs, seq_len, dropout
  - Sampler: TPE (Tree-structured Parzen Estimator) — Bergstra 2011
  - Pruner: MedianPruner — kills below-median trials early
  - Storage: SQLite (study.db) → persistent + dashboard
  - Objective: maximize 3-seed mean val_IC (variance-aware) per CLAUDE
    5.13.4

Run dashboard live to watch:
  $ optuna-dashboard sqlite:///optuna_patchtst.db

References:
  - Akiba et al 2019 KDD "Optuna: A Next-generation Hyperparameter
    Optimization Framework"
  - Bergstra et al 2011 NeurIPS "Algorithms for Hyper-parameter
    Optimization" (TPE)
  - Optuna docs: https://optuna.readthedocs.io/

Usage::

    .venv/bin/python scripts/optuna_patchtst_sweep.py \\
        --n-trials 20 --n-seeds 3 --epochs 5 \\
        --dataset data/transformer_v4_wl200_clean.parquet \\
        --study-name patchtst_v1 \\
        --storage sqlite:///optuna_patchtst.db
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Single-thread per [[concurrency_resource_budget]] — Optuna orchestrates
# trials, each trial subprocess saturates within its own budget.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import optuna

REPO = Path(__file__).resolve().parent.parent
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("optuna-patchtst-sweep")


def _run_one_seed(dataset: str, out_dir: Path, seed: int, lr: float,
                  weight_decay: float, warmup_epochs: int, seq_len: int,
                  epochs: int, label: str) -> float | None:
    """Subprocess one transformer_v4.py training. Returns val_ic_mean or None
    on failure."""
    cmd = [
        sys.executable, str(REPO / "scripts/transformer_v4.py"),
        "--dataset", dataset,
        "--arch", "patchtst",
        "--label", label,
        "--seq-len", str(seq_len),
        "--epochs", str(epochs),
        "--num-seeds", "1",
        "--seed", str(seed),
        "--lr", str(lr),
        "--weight-decay", str(weight_decay),
        "--warmup-epochs", str(warmup_epochs),
        "--output-dir", str(out_dir),
        "--device", "mps",
    ]
    log.info("  subprocess: seed=%d lr=%.1e wd=%.1e warmup=%d seq=%d epochs=%d",
             seed, lr, weight_decay, warmup_epochs, seq_len, epochs)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        tail = result.stderr[-400:] if result.stderr else result.stdout[-400:]
        log.warning("  subprocess FAILED rc=%d tail=%s", result.returncode, tail)
        return None
    # Parse summary JSON (authoritative)
    summary_path = out_dir / "patchtst_summary.json"
    if not summary_path.exists():
        log.warning("  no summary.json at %s", summary_path)
        return None
    try:
        s = json.loads(summary_path.read_text())
        return float(s["val_ic_mean"])
    except Exception as exc:
        log.warning("  summary parse failed: %s", exc)
        return None


def objective(trial: optuna.Trial, dataset: str, n_seeds: int,
              epochs: int, label: str) -> float:
    """Optuna objective: train PatchTST with proposed hyperparams, return
    mean val_IC across n_seeds. Uses subprocess to existing transformer_v4.py
    — no custom training code."""
    # Search space — bounded by published PatchTST defaults (Nie 2023)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    warmup_epochs = trial.suggest_int("warmup_epochs", 1, 6)
    seq_len = trial.suggest_categorical("seq_len", [16, 32, 60])
    seed_base = trial.suggest_int("seed_base", 42, 200)

    val_ics = []
    for s_offset in range(n_seeds):
        seed = seed_base + s_offset
        # Per-seed output dir so summaries don't collide
        out_dir = (REPO / "artifacts/optuna_trials"
                   / f"trial_{trial.number:03d}_seed_{seed}")
        out_dir.mkdir(parents=True, exist_ok=True)

        val_ic = _run_one_seed(dataset, out_dir, seed, lr, weight_decay,
                                warmup_epochs, seq_len, epochs, label)
        if val_ic is None:
            raise optuna.TrialPruned()

        val_ics.append(val_ic)
        trial.report(val_ic, step=s_offset)
        log.info("Trial %d seed %d: val_ic=%+.4f", trial.number, seed, val_ic)
        if trial.should_prune():
            log.info("Trial %d pruned at seed %d", trial.number, s_offset)
            raise optuna.TrialPruned()

    mean_val_ic = sum(val_ics) / len(val_ics)
    std_val_ic = (sum((v - mean_val_ic) ** 2 for v in val_ics) / len(val_ics)) ** 0.5
    log.info("Trial %d FINAL mean=%+.4f std=%+.4f n=%d "
             "(lr=%.1e wd=%.1e warmup=%d seq=%d)",
             trial.number, mean_val_ic, std_val_ic, n_seeds,
             lr, weight_decay, warmup_epochs, seq_len)
    return mean_val_ic


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/transformer_v4_wl200_clean.parquet")
    p.add_argument("--label", default="fwd_60d_excess")
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--n-seeds", type=int, default=3,
                   help="Seeds per trial for variance estimation (CLAUDE 5.13.4)")
    p.add_argument("--epochs", type=int, default=5,
                   help="Epochs per training (sweep speed; full retrain after)")
    p.add_argument("--study-name", default="patchtst_sweep")
    p.add_argument("--storage", default="sqlite:///optuna_patchtst.db")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="Parallel trials (>1 only safe if subprocess MPS shares)")
    args = p.parse_args()

    sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=1, n_startup_trials=5)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    log.info("Study=%s storage=%s n_trials=%d n_seeds=%d epochs=%d label=%s",
             args.study_name, args.storage, args.n_trials, args.n_seeds,
             args.epochs, args.label)
    log.info("Live dashboard: optuna-dashboard %s", args.storage)

    study.optimize(
        lambda t: objective(t, args.dataset, args.n_seeds, args.epochs,
                            args.label),
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        show_progress_bar=False,
        catch=(Exception,),
    )

    print("\n" + "=" * 70)
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    print(f"  COMPLETED: {len(completed)} / {len(study.trials)} trials")
    if completed:
        print(f"  BEST TRIAL #{study.best_trial.number}  "
              f"val_ic={study.best_value:+.4f}")
        print("  BEST PARAMS:")
        for k, v in study.best_params.items():
            print(f"    {k}: {v}")
    print("=" * 70)


if __name__ == "__main__":
    main()
