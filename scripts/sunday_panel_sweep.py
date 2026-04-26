#!/usr/bin/env python3
"""Sunday panel-LTR sweep — train all 3 backends + produce comparison report.

Per user spec (2026-04-26): on Sunday, train XGBoost first as the production
default (the artifacts that ship to live trading), then ALSO train LightGBM
and Transformer for comparison purposes. Capture per-backend OOS IC and
write a comparison report to ``doc/panel_sunday_sweep_{date}.md``.

Total wall time on M-series Mac: ~75-90 min sequential.

Usage::
    python scripts/sunday_panel_sweep.py --strategy renquant_104

The script:
  1. Reads current ``panel_ltr.backend`` from strategy_config.json.
  2. Runs ``train_104.py --force`` with that backend (XGBoost default).
  3. Backs up the resulting artifacts as ``*.xgboost.bak.json``.
  4. For each of the OTHER backends [lightgbm, transformer]:
     a. Temporarily swap ``backend`` in strategy_config.json.
     b. Run ``train_104.py --force``.
     c. Read the resulting OOS IC from ``panel-rank-calibration.json``
        + ``panel-ltr.json`` metadata.
     d. Save artifacts as ``*.{backend}.bak.json``.
  5. Restore ``backend`` to the original (XGBoost) and restore the
     XGBoost artifacts as the active production set.
  6. Write the comparison report.

Failure handling: if any backend training crashes, the report still
contains the backends that succeeded with a "FAILED" row for the rest.
The XGBoost-as-active invariant is restored regardless via ``finally``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON    = "/Users/renhao/miniconda3/envs/renquant/bin/python"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("sunday_sweep")


# ── Helpers ───────────────────────────────────────────────────────────────────

ARTIFACT_NAMES = (
    "panel-ltr.json",
    "ngboost-head.json",
    "panel-rank-calibration.json",
)


def _config_path(strategy: str) -> Path:
    return REPO_ROOT / "backtesting" / strategy / "strategy_config.json"


def _artifacts_dir(strategy: str) -> Path:
    return REPO_ROOT / "backtesting" / strategy / "artifacts"


def _swap_backend(strategy: str, backend: str) -> None:
    """In-place edit `panel_ltr.backend` in strategy_config.json."""
    cfg_path = _config_path(strategy)
    cfg = json.loads(cfg_path.read_text())
    cfg.setdefault("panel_ltr", {})["backend"] = backend
    cfg_path.write_text(json.dumps(cfg, indent=2))
    log.info("backend → %s in %s", backend, cfg_path.name)


def _backup_artifacts(strategy: str, suffix: str) -> None:
    """Copy current artifacts to ``*.{suffix}.bak.json`` (e.g. xgboost.bak.json)."""
    art_dir = _artifacts_dir(strategy)
    for name in ARTIFACT_NAMES:
        src = art_dir / name
        if not src.exists():
            log.warning("backup skipped — %s missing", name)
            continue
        stem, ext = name.rsplit(".", 1)
        dst = art_dir / f"{stem}.{suffix}.bak.{ext}"
        shutil.copy2(src, dst)
    log.info("backed up artifacts as *.%s.bak.json", suffix)


def _restore_artifacts(strategy: str, suffix: str) -> None:
    """Inverse of _backup_artifacts."""
    art_dir = _artifacts_dir(strategy)
    for name in ARTIFACT_NAMES:
        stem, ext = name.rsplit(".", 1)
        src = art_dir / f"{stem}.{suffix}.bak.{ext}"
        dst = art_dir / name
        if not src.exists():
            log.warning("restore skipped — %s missing", src.name)
            continue
        shutil.copy2(src, dst)
    log.info("restored artifacts from *.%s.bak.json", suffix)


def _read_metrics(strategy: str) -> dict:
    """Pull the OOS IC + scorer metadata from panel-rank-calibration.json + panel-ltr.json."""
    art_dir = _artifacts_dir(strategy)
    out: dict[str, object] = {}
    cal_path = art_dir / "panel-rank-calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        m = cal.get("metadata", {}) or {}
        out["scorer_oos_mean_ic"] = m.get("scorer_oos_mean_ic")
        out["pool_ic"]             = m.get("pool_ic")
        out["n_rows"]              = m.get("n_rows")
        out["n_tickers"]           = m.get("n_tickers")
    ltr_path = art_dir / "panel-ltr.json"
    if ltr_path.exists():
        try:
            ltr = json.loads(ltr_path.read_text())
            m = ltr.get("metadata", {}) or {}
            for k in ("backend", "train_ic", "cpcv_mean_ic", "cpcv_std_ic"):
                if k in m and k not in out:
                    out[k] = m[k]
        except json.JSONDecodeError:
            pass
    return out


def _train_backend(strategy: str, backend: str) -> tuple[bool, dict]:
    """Run train_104 with the given backend; return (ok, metrics)."""
    _swap_backend(strategy, backend)
    log.info("─── training backend=%s ───", backend)
    started = dt.datetime.now()
    cmd = [PYTHON, "scripts/train_104.py", "--strategy", strategy, "--force",
           "--skip-baseline"]
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    elapsed = (dt.datetime.now() - started).total_seconds()
    log.info("backend=%s train exit=%d  elapsed=%.0fs", backend, proc.returncode, elapsed)
    if proc.returncode != 0:
        return False, {"backend": backend, "elapsed_s": elapsed, "status": "FAILED"}
    metrics = _read_metrics(strategy)
    metrics["backend"]  = backend
    metrics["elapsed_s"] = elapsed
    metrics["status"]    = "OK"
    return True, metrics


def _write_report(strategy: str, results: list[dict]) -> Path:
    """Produce the markdown comparison report."""
    today = dt.date.today().isoformat()
    report_path = REPO_ROOT / "doc" / f"panel_sunday_sweep_{today}.md"

    def fmt(v, fmt_spec: str = ".4f") -> str:
        if v is None or v == "":
            return "—"
        if isinstance(v, (int, float)):
            try:
                return format(v, fmt_spec)
            except Exception:
                return str(v)
        return str(v)

    rows = ["| backend | scorer_oos_mean_ic | pool_ic | train_ic | n_rows | elapsed | status |",
            "|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        rows.append(
            f"| {r.get('backend','—')} "
            f"| {fmt(r.get('scorer_oos_mean_ic'))} "
            f"| {fmt(r.get('pool_ic'))} "
            f"| {fmt(r.get('train_ic'))} "
            f"| {r.get('n_rows','—')} "
            f"| {fmt(r.get('elapsed_s'), '.0f')}s "
            f"| {r.get('status','—')} |"
        )

    valid_results = [r for r in results
                     if r.get("status") == "OK"
                     and isinstance(r.get("scorer_oos_mean_ic"), (int, float))]
    if valid_results:
        winner = max(valid_results, key=lambda r: r["scorer_oos_mean_ic"])
        winner_line = (
            f"\n## Winner — {winner['backend']} "
            f"(OOS IC = {winner['scorer_oos_mean_ic']:.4f})\n"
        )
    else:
        winner_line = "\n## Winner — none (all backends failed)\n"

    body = (
        f"# Sunday Panel-LTR Sweep — {today}\n\n"
        f"Strategy: `{strategy}`. All 3 backends trained on the same panel "
        f"with identical CV split (CPCV, 15 folds), same NGBoost head, same "
        f"calibrator. XGBoost remains the active production backend; LightGBM "
        f"and Transformer trained for comparison only.\n\n"
        f"## Results\n\n"
        + "\n".join(rows)
        + "\n"
        + winner_line
        + f"\n## Active artifacts\n\n"
        f"`backtesting/{strategy}/artifacts/panel-ltr.json` and friends are "
        f"restored to the XGBoost run after this sweep. Per-backend backups "
        f"live as `*.{{backend}}.bak.json` in the same directory.\n"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(body)
    log.info("report → %s", report_path)
    return report_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy", default="renquant_104")
    ap.add_argument(
        "--backends", nargs="+",
        default=["xgboost", "lightgbm", "transformer"],
        help="ordered list of backends to train. First one becomes production.",
    )
    args = ap.parse_args()

    strategy = args.strategy
    backends_to_run = list(args.backends)
    if not backends_to_run:
        log.error("--backends list is empty")
        return 2

    cfg_path = _config_path(strategy)
    original_cfg = json.loads(cfg_path.read_text())
    original_backend = (original_cfg.get("panel_ltr", {})
                                    .get("backend", "xgboost"))
    production_backend = backends_to_run[0]

    results: list[dict] = []
    try:
        for backend in backends_to_run:
            ok, metrics = _train_backend(strategy, backend)
            results.append(metrics)
            # Always backup, OK or not (so partial artifacts can be inspected)
            _backup_artifacts(strategy, backend)
            if not ok:
                log.warning("backend=%s training failed — see log", backend)
    finally:
        # Restore the production backend to be active.
        log.info("restoring production backend=%s + its artifacts",
                 production_backend)
        _swap_backend(strategy, production_backend)
        _restore_artifacts(strategy, production_backend)
        # If original config had a different backend (e.g. someone manually
        # swapped to lightgbm before sweep), respect that and warn.
        if original_backend != production_backend:
            log.warning(
                "original config had backend=%s; this sweep overrode to %s. "
                "Re-edit strategy_config.json if you want to revert.",
                original_backend, production_backend,
            )

    # Even if some backends failed, write the report with what we have.
    report_path = _write_report(strategy, results)

    # Push notification with summary
    valid = [r for r in results if r.get("status") == "OK"]
    n_ok, n_total = len(valid), len(results)
    log.info("Sunday sweep DONE — %d/%d backends OK; report=%s",
             n_ok, n_total, report_path)
    return 0 if n_ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
