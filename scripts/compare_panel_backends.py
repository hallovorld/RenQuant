#!/usr/bin/env python
"""A/B compare panel-LTR backends (XGBoost vs transformer) on the real panel.

Trains each backend on the exact same panel (same feature frames, same
CV splits, same labels) and reports OOS mean-IC + per-fold IC side by
side. Writes a markdown summary that can be appended to
``doc/experiments/panel-training-runs.md``.

Does NOT touch ``strategy_config.json`` — backend is swapped in-memory.
Existing ``artifacts/panel-ltr.json`` is restored from a backup at the
end, so the sim keeps using whichever backend was shipped when the
comparison started.

Usage::

    python scripts/compare_panel_backends.py --strategy renquant_104
    python scripts/compare_panel_backends.py --strategy renquant_104 \
        --transformer-epochs 30 --device mps

Ship-gate from `doc/components/transformer-104.md §5`:
    transformer mean-IC >= 1.30 x xgboost mean-IC  → candidate to ship
    transformer mean-IC >= 1.10 x xgboost mean-IC  → ensemble

Exit code:
    0  — ran cleanly, any verdict
    2  — transformer run failed (XGBoost baseline still reported)
    3  — XGBoost baseline failed (usually means artifacts missing / stale)
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("compare-panel")


@dataclass
class BackendResult:
    backend:       str
    mean_ic:       float | None
    per_fold_ic:   list[float]
    train_ic:      float | None
    elapsed_sec:   float
    artifact_path: str
    device:        str | None = None
    error:         str | None = None


def _run_one(strategy: str, backend: str, device: str | None,
             transformer_epochs: int | None) -> BackendResult:
    """Train one backend end-to-end. Returns metadata; does NOT affect the
    other backend's artifact path.
    """
    strategy_dir = REPO_ROOT / "backtesting" / strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.pipeline.pp_training_full import (       # noqa: PLC0415
        FullTrainingContext, FullTrainingPipeline,
    )

    config = json.loads((strategy_dir / "strategy_config.json").read_text())
    # Override backend in-memory; don't write to disk.
    panel_cfg = dict(config.get("panel_ltr", {}))
    panel_cfg["backend"] = backend
    # Disable NGBoost head during the A/B. It's a separate Normal(μ, σ) head
    # that doesn't affect the panel CV IC we're comparing, adds ~5-8 min per
    # backend, and would require a follow-up pass to re-fit on whichever
    # backend we ship. Keep the comparison focused on panel IC + speed.
    panel_cfg["ngboost"] = {**panel_cfg.get("ngboost", {}), "enabled": False}

    # Cap CV cost so the A/B finishes in a reasonable wall-clock window.
    # CrossValidateTask uses num_boost_round // 2 per fold; 15-split CPCV
    # multiplies by 15. The default (num_boost_round=150, cv_method=cpcv,
    # cv_n_splits=6 nC2=15) results in 1125 epoch-fits for a transformer
    # — measured ~40+ min CPU single-thread. We use a 5-fold purged K-fold
    # CV here instead (cheaper, close enough for a ranking comparison).
    panel_cfg["cv_method"]     = "purged"
    panel_cfg["cv_n_splits"]   = 5
    # Transformer-side round budget: explicit --transformer-epochs beats
    # config; if not provided, use a modest default. XGBoost keeps its
    # configured num_boost_round so its baseline isn't crippled.
    if backend == "transformer":
        tf = dict(panel_cfg.get("transformer_params", {}))
        if device:
            tf["device"] = device
        if transformer_epochs:
            tf["max_epochs"] = int(transformer_epochs)
        else:
            tf.setdefault("max_epochs", 30)
        panel_cfg["transformer_params"] = tf
        # Override num_boost_round for the CV adapter too (it reads this
        # and halves it per fold; keep total CV work ≈ 5 × 15 = 75 epochs).
        panel_cfg["num_boost_round"] = int(tf.get("max_epochs", 30))
    config["panel_ltr"] = panel_cfg
    # Force the retrain even on non-cadence days.
    config["training"] = {**config.get("training", {}), "cadence": "daily"}
    # Suppress persistence side effects in the A/B path — we'll record the
    # A/B result ourselves in the markdown.
    config["persistence"] = {**config.get("persistence", {}), "enabled": False}

    ctx = FullTrainingContext(
        config=config,
        strategy=strategy,
        strategy_dir=strategy_dir,
        skip_baseline=True,       # don't retrain per-ticker tournament models
        skip_panel=False,
        skip_recalibrate=True,    # don't touch score calibrations
        force_retrain=True,       # bypass cadence gate
    )
    t0 = time.monotonic()
    try:
        FullTrainingPipeline().run(ctx)
    except Exception as exc:
        log.exception("Backend %s failed", backend)
        return BackendResult(
            backend=backend, mean_ic=None, per_fold_ic=[],
            train_ic=None, elapsed_sec=time.monotonic() - t0,
            artifact_path="", error=repr(exc),
        )

    elapsed = time.monotonic() - t0

    # Locate the artifact this backend wrote.
    artifacts_dir = strategy_dir / "artifacts"
    if backend == "transformer":
        sidecar = artifacts_dir / "panel-transformer.json"
    else:
        sidecar = artifacts_dir / "panel-ltr.json"
    if not sidecar.exists():
        return BackendResult(
            backend=backend, mean_ic=None, per_fold_ic=[],
            train_ic=None, elapsed_sec=elapsed,
            artifact_path=str(sidecar), error="artifact not written",
        )
    meta = json.loads(sidecar.read_text())
    return BackendResult(
        backend       = backend,
        mean_ic       = meta.get("oos_mean_ic"),
        per_fold_ic   = list(meta.get("oos_per_fold_ic", [])),
        train_ic      = meta.get("training_train_ic"),
        elapsed_sec   = elapsed,
        artifact_path = str(sidecar),
        device        = meta.get("device")
                        or (panel_cfg.get("transformer_params", {}).get("device")
                            if backend == "transformer" else "cpu"),
    )


def _format_markdown(xgb: BackendResult, tfm: BackendResult) -> str:
    lines = []
    lines.append(f"## Panel backend A/B — {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("| Metric | XGBoost | Transformer |")
    lines.append("|---|---|---|")
    lines.append(f"| OOS mean IC     | {_fmt(xgb.mean_ic)} | {_fmt(tfm.mean_ic)} |")
    lines.append(f"| Train IC        | {_fmt(xgb.train_ic)} | {_fmt(tfm.train_ic)} |")
    lines.append(f"| Per-fold min    | {_fmt(min(xgb.per_fold_ic, default=float('nan')))} "
                 f"| {_fmt(min(tfm.per_fold_ic, default=float('nan')))} |")
    lines.append(f"| Per-fold max    | {_fmt(max(xgb.per_fold_ic, default=float('nan')))} "
                 f"| {_fmt(max(tfm.per_fold_ic, default=float('nan')))} |")
    lines.append(f"| Training time   | {xgb.elapsed_sec:.1f}s | {tfm.elapsed_sec:.1f}s |")
    lines.append(f"| Device          | {xgb.device or '—'} | {tfm.device or '—'} |")
    lines.append(f"| Artifact        | `{xgb.artifact_path}` | `{tfm.artifact_path}` |")
    if xgb.error:
        lines.append(f"| XGBoost error   | `{xgb.error}` |")
    if tfm.error:
        lines.append(f"| Transformer error | `{tfm.error}` |")

    # Verdict
    if xgb.mean_ic and tfm.mean_ic:
        ratio = tfm.mean_ic / xgb.mean_ic if xgb.mean_ic > 0 else float("inf")
        if ratio >= 1.30:
            verdict = f"**SHIP CANDIDATE**  (ratio {ratio:.2f} ≥ 1.30)"
        elif ratio >= 1.10:
            verdict = f"**ENSEMBLE CANDIDATE**  (ratio {ratio:.2f}, in [1.10, 1.30))"
        else:
            verdict = f"shelve  (ratio {ratio:.2f} < 1.10)"
        lines.append(f"\n**Verdict:** {verdict}")
    return "\n".join(lines)


def _fmt(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    return f"{v:+.4f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",            default="renquant_104")
    p.add_argument("--device",              default=None,
                   help="Transformer device override (mps/cuda/cpu). "
                        "Default: config's panel_ltr.transformer_params.device.")
    p.add_argument("--transformer-epochs",  type=int, default=None,
                   help="Override max_epochs for transformer.")
    p.add_argument("--out", default=None,
                   help="Write markdown summary to this path instead of stdout.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    artifacts_dir = strategy_dir / "artifacts"
    xgb_art = artifacts_dir / "panel-ltr.json"
    tfm_art_pt = artifacts_dir / "panel-transformer.pt"
    tfm_art_js = artifacts_dir / "panel-transformer.json"

    # Back up existing artifacts so the sim/runtime isn't left in a
    # half-retrained state if something crashes mid-comparison.
    backup_dir = artifacts_dir / ".compare_backup"
    backup_dir.mkdir(exist_ok=True)
    for p in (xgb_art, tfm_art_pt, tfm_art_js):
        if p.exists():
            shutil.copy2(p, backup_dir / p.name)

    log.info("── Running XGBoost backend ──")
    xgb_res = _run_one(args.strategy, "xgboost", args.device, args.transformer_epochs)
    log.info("XGBoost result: mean_ic=%s  elapsed=%.1fs",
             xgb_res.mean_ic, xgb_res.elapsed_sec)

    log.info("── Running transformer backend ──")
    tfm_res = _run_one(args.strategy, "transformer", args.device,
                       args.transformer_epochs)
    log.info("Transformer result: mean_ic=%s  elapsed=%.1fs",
             tfm_res.mean_ic, tfm_res.elapsed_sec)

    md = _format_markdown(xgb_res, tfm_res)
    print("\n" + md + "\n")
    if args.out:
        Path(args.out).write_text(md + "\n")
        log.info("Wrote markdown summary → %s", args.out)

    # Restore the XGBoost artifact from backup so the sim/runtime keeps
    # using whichever artifact was shipped when the comparison started.
    # The transformer artifact is left in place (it's a new file that
    # didn't exist before the comparison on first run).
    xgb_backup = backup_dir / "panel-ltr.json"
    if xgb_backup.exists():
        shutil.copy2(xgb_backup, xgb_art)
        log.info("Restored XGBoost artifact from backup.")

    if xgb_res.error:
        sys.exit(3)
    if tfm_res.error:
        sys.exit(2)


if __name__ == "__main__":
    main()
