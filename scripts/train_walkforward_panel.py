#!/usr/bin/env python
"""Walk-forward panel-LTR training driver (Track P1, 2026-05-10).

Trains one panel-LTR artifact per `retrain_date` in
[--start-date, --end-date], each on a `[retrain_date - training_window, retrain_date)`
window, and emits a manifest indexed by cutoff_date.

Sim adapters bind to the manifest via
`kernel.walk_forward.WalkForwardModelLoader.model_as_of(today)`. No
look-ahead leakage: every model used at sim bar `t` was trained
strictly before `t`.

Usage::

    # Dry-run: print the 27 retrain dates without training
    python scripts/train_walkforward_panel.py \\
        --start-date 2024-01-01 --end-date 2026-03-26 \\
        --cadence-days 21 --dry-run

    # Real walk-forward training (≈ 3 hours on M2 Pro)
    python scripts/train_walkforward_panel.py \\
        --start-date 2024-01-01 --end-date 2026-03-26 \\
        --cadence-days 21 \\
        --manifest-output artifacts/walkforward_manifest.json

CLAUDE.md §5.10 hardware saturation: sets OMP_NUM_THREADS=10,
MKL_NUM_THREADS=10, OPENBLAS_NUM_THREADS=10 + xgb_params.nthread=10.

CLAUDE.md §5.13.7 data-pipeline change: requires data regen.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# §5.10 hardware saturation — must be set BEFORE numpy / xgboost import.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(STRATEGY_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-walkforward")


# ── Pure helpers (§1c — small, single-responsibility, ≤ 50 lines each) ──

def compute_retrain_dates(
    start: pd.Timestamp, end: pd.Timestamp, cadence_days: int,
) -> list[pd.Timestamp]:
    """Return retrain cutoff dates spanning [start, end] at cadence_days."""
    if cadence_days <= 0:
        raise ValueError(f"cadence_days must be > 0, got {cadence_days}")
    return list(pd.date_range(start, end, freq=f"{cadence_days}D"))


def load_strategy_config(config_path: Path) -> dict:
    """Read strategy_config.json + stamp _strategy_dir."""
    if not config_path.exists():
        raise FileNotFoundError(f"strategy config not found: {config_path}")
    cfg = json.loads(config_path.read_text())
    cfg["_strategy_dir"] = str(config_path.parent)
    cfg["_strategy_config_name"] = config_path.name
    return cfg


def saturate_xgb_threads(cfg: dict) -> dict:
    """Force xgb_params.nthread=10 in panel_ltr config (§5.10)."""
    pl = cfg.setdefault("panel_ltr", {})
    xp = pl.setdefault("xgb_params", {})
    xp["nthread"] = int(xp.get("nthread", 10))
    return cfg


def make_artifact_dir(strategy_dir: Path, cutoff: pd.Timestamp) -> Path:
    """Per-cutoff artifact subdirectory: artifacts/walkforward/<YYYY-MM-DD>/."""
    sub = strategy_dir / "artifacts" / "walkforward" / cutoff.date().isoformat()
    sub.mkdir(parents=True, exist_ok=True)
    return sub


def configure_panel_cutoff(cfg: dict, cutoff: pd.Timestamp,
                           artifact_path: Path) -> dict:
    """Set panel_ltr.train_cutoff + BOTH artifact_path keys for one retrain.

    AUDIT 2026-05-10 §5.13.13/§5.13.14 incident: SaveArtifactTask
    (pp_panel_training.py:2684-2699) reads inference-side
    ``cfg["ranking"]["panel_scoring"]["artifact_path"]`` FIRST and
    falls back to training-side ``cfg["panel_ltr"]["artifact_path"]``
    only when inference-side is unset. Setting only the training-side
    key was a no-op for the writer — every retrain silently overwrote
    the production artifact at the inference-side path.

    Fix: route both keys to the per-cutoff ``walkforward/<date>/`` path.
    Sanity-assert the path contains 'walkforward' so this function
    cannot accidentally be wired to a production-shaped path again.
    """
    p_str = str(artifact_path)
    # Sanity guard per §5.13.3 — refuse paths outside the walkforward/
    # subtree. Pinned by tests/test_walkforward_artifact_isolation.py.
    assert "walkforward" in p_str, (
        f"configure_panel_cutoff: artifact_path {p_str!r} does not "
        f"contain 'walkforward' — refusing to risk overwriting "
        f"production artifact"
    )

    pl = cfg.setdefault("panel_ltr", {})
    pl["train_cutoff"] = cutoff.isoformat()
    pl["artifact_path"] = p_str

    # CRITICAL: also override inference-side path. SaveArtifactTask
    # prefers this key over panel_ltr.artifact_path; without this line,
    # the per-cutoff redirect is a no-op for the writer.
    rk = cfg.setdefault("ranking", {}).setdefault("panel_scoring", {})
    rk["artifact_path"] = p_str
    rk.setdefault("global_calibration", {})["auto_refresh"] = False

    return cfg


def build_retrain_entry(cutoff: pd.Timestamp, trained_dt: datetime,
                         artifact_uri: str):
    """Build a RetrainEntry — wrapper so callers don't have to import it."""
    from kernel.walk_forward import RetrainEntry  # noqa: PLC0415
    return RetrainEntry(
        cutoff_date=cutoff,
        trained_date=pd.Timestamp(trained_dt),
        artifact_uri=artifact_uri,
    )


# ── Per-cutoff training (delegates to PanelTrainingPipeline) ────────────

def train_one_cutoff(base_cfg: dict, cutoff: pd.Timestamp) -> str:
    """Run the panel pipeline once with the given cutoff, return artifact URI."""
    from training_panel.context import PanelTrainingContext  # noqa: PLC0415
    from training_panel.pp_panel_training import PanelTrainingPipeline  # noqa: PLC0415

    cfg = json.loads(json.dumps(base_cfg))  # deep-ish copy via JSON round-trip
    strategy_dir = Path(cfg["_strategy_dir"])
    artifact_dir = make_artifact_dir(strategy_dir, cutoff)
    artifact_path = artifact_dir / "panel-ltr.json"
    configure_panel_cutoff(cfg, cutoff, artifact_path)

    pctx = PanelTrainingContext(
        config=cfg,
        watchlist=list(cfg.get("watchlist", [])),
    )
    log.info("train_one_cutoff: cutoff=%s start", cutoff.date().isoformat())
    t0 = time.monotonic()
    PanelTrainingPipeline().run(pctx)
    log.info("train_one_cutoff: cutoff=%s DONE  %.1fs  artifact=%s",
             cutoff.date().isoformat(), time.monotonic() - t0, artifact_path)
    return str(artifact_path)


# ── CLI driver ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--start-date", required=True,
                   help="First retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--end-date", required=True,
                   help="Last retrain cutoff (YYYY-MM-DD).")
    p.add_argument("--cadence-days", type=int, default=21,
                   help="Days between retrain cutoffs (default: 21).")
    p.add_argument("--config-path",
                   default=str(STRATEGY_DIR / "strategy_config.json"),
                   help="Strategy config to clone for each cutoff.")
    p.add_argument("--manifest-output",
                   default=str(STRATEGY_DIR / "artifacts" / "walkforward_manifest.json"),
                   help="Where to write the manifest JSON.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print retrain dates and exit (no training).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    retrain_dates = compute_retrain_dates(start, end, args.cadence_days)
    log.info("Walk-forward plan: start=%s end=%s cadence=%dd  → %d retrains",
             start.date(), end.date(), args.cadence_days, len(retrain_dates))

    if args.dry_run:
        for i, d in enumerate(retrain_dates):
            print(f"[{i+1:02d}/{len(retrain_dates)}] cutoff={d.date().isoformat()}")
        print(f"Total retrain dates: {len(retrain_dates)}")
        return

    # Lazy imports — only when actually training (avoids requiring
    # heavy deps at dry-run time).
    from kernel.walk_forward import WalkForwardManifest, write_manifest  # noqa: PLC0415

    base_cfg = load_strategy_config(Path(args.config_path))
    saturate_xgb_threads(base_cfg)

    entries = []
    for i, cutoff in enumerate(retrain_dates):
        log.info("── retrain %d/%d  cutoff=%s ──",
                 i + 1, len(retrain_dates), cutoff.date().isoformat())
        try:
            uri = train_one_cutoff(base_cfg, cutoff)
        except Exception as exc:  # noqa: BLE001
            log.error("retrain %d/%d FAILED at cutoff=%s — %s",
                      i + 1, len(retrain_dates), cutoff.date().isoformat(), exc)
            continue
        entries.append(build_retrain_entry(
            cutoff=cutoff,
            trained_dt=datetime.utcnow(),
            artifact_uri=uri,
        ))

    manifest = WalkForwardManifest(
        cadence_days=int(args.cadence_days),
        training_window_years=float(
            base_cfg.get("panel_ltr", {}).get("training_window_years", 3.0)
        ),
        retrains=entries,
    )
    out = write_manifest(manifest, args.manifest_output)
    log.info("Wrote manifest with %d retrains → %s", len(entries), out)


if __name__ == "__main__":
    main()
