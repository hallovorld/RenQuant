#!/usr/bin/env python
"""Backend tournament — pick the best panel-LTR artifact among rivals.

Phase 3 of the model-selection systematization plan (2026-04-26).

Problem this solves:
    The artifacts dir accumulates `panel-ltr.{xgboost,lightgbm,transformer,
    macro-enabled,...}.bak.json` files from prior retrains. Each has stored
    OOS IC and (optionally) sim_smoke metrics. Today, picking the right
    one to ship is a manual judgment call. This script formalizes it.

How it works:
    1. Discovers every `panel-ltr.*.bak.json` and `panel-ltr.json`
       (current production) under the strategy's artifacts dir.
    2. Loads each artifact's metadata: oos_mean_ic, oos_std_ic, panel_shape,
       sim_smoke (apy/sharpe/calmar/turnover_ratio/max_drawdown).
    3. Computes a composite score per candidate:
            composite = w_ic   × normalized_ic
                      + w_sh   × normalized_sharpe
                      + w_cal  × normalized_calmar
       Default weights: 0.5 / 0.3 / 0.2. Set to 1/0/0 if no sim_smoke
       metrics on any candidate (IC-only mode).
    4. Prints a leaderboard sorted by composite descending.
    5. Optional `--promote <name>`: atomically copy the winner over
       `panel-ltr.json` (using kernel.model_acceptance.promote), preserving
       prior at `panel-ltr.previous.json`.

Importantly: this script does NOT retrain. It compares already-trained
artifacts. For retrain-time gating, see ModelAcceptanceGate (Phase 1+2).

Composite design:
    - Normalize each metric to z-scores across candidates (so weights
      are dimensionally consistent: a 0.01 IC delta has equivalent
      weight to a 0.5 sharpe delta).
    - Candidates with missing metrics get z=0 (median-ish), so they
      don't get unfair credit OR penalty for absence.
    - The composite is symmetric — flipping signs (negative IC) hurts
      the candidate.

Caveats (printed at run-time):
    - Cross-candidate OOS IC is only fair when all artifacts were
      trained on the same panel (rows × dates). If panel_shape differs,
      the script warns; the operator must verify train sets match.
    - Composite weights are an opinion. Override with --weights
      "ic=0.7,sharpe=0.2,calmar=0.1" if you disagree.

Usage::

    python scripts/select_best_model.py --strategy renquant_104
    python scripts/select_best_model.py --strategy renquant_104 \
        --weights "ic=0.5,sharpe=0.3,calmar=0.2"
    python scripts/select_best_model.py --strategy renquant_104 \
        --promote xgboost   # promotes panel-ltr.xgboost.bak.json
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("select-best-model")


# ── Candidate discovery + metric extraction ───────────────────────────────────

@dataclass
class Candidate:
    name:           str
    path:           Path
    oos_mean_ic:    float | None
    oos_std_ic:     float | None
    panel_rows:     int | None
    panel_tickers:  int | None
    panel_dates:    int | None
    sim_apy:        float | None
    sim_sharpe:     float | None
    sim_calmar:     float | None
    sim_turnover:   float | None
    sim_max_dd:     float | None
    feature_count:  int | None
    trained_date:   str | None
    composite:      float = 0.0      # filled in by score_candidates


def _extract_metrics(artifact: dict, name: str, path: Path) -> Candidate:
    """Parse known fields out of an artifact JSON. Tolerant of flat /
    nested metadata layouts."""
    md = artifact.get("metadata") if isinstance(artifact.get("metadata"), dict) else artifact
    smoke = (md.get("sim_smoke") or {}) if isinstance(md.get("sim_smoke"), dict) else {}
    panel_shape = md.get("panel_shape") or artifact.get("panel_shape") or {}
    feat = artifact.get("feature_cols") or md.get("feature_cols") or []

    return Candidate(
        name           = name,
        path           = path,
        oos_mean_ic    = _f(md.get("oos_mean_ic")    or artifact.get("oos_mean_ic")),
        oos_std_ic     = _f(md.get("oos_std_ic")     or artifact.get("oos_std_ic")),
        panel_rows     = _i(panel_shape.get("rows")),
        panel_tickers  = _i(panel_shape.get("tickers")),
        panel_dates    = _i(panel_shape.get("dates")),
        sim_apy        = _f(smoke.get("apy")),
        sim_sharpe     = _f(smoke.get("sharpe")),
        sim_calmar     = _f(smoke.get("calmar")),
        sim_turnover   = _f(smoke.get("turnover_ratio")),
        sim_max_dd     = _f(smoke.get("max_drawdown")),
        feature_count  = len(feat) if feat else None,
        trained_date   = md.get("trained_date") or artifact.get("trained_date"),
    )


def _f(x: Any) -> float | None:
    try:
        return float(x) if x is not None else None
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> int | None:
    try:
        return int(x) if x is not None else None
    except (TypeError, ValueError):
        return None


# Match panel-ltr.{name}.bak.json AND panel-ltr.json (the live model).
_BAK_RE = re.compile(r"^panel-ltr\.(?P<name>[^.]+)\.bak\.json$")


def discover_candidates(artifacts_dir: Path) -> list[Candidate]:
    out: list[Candidate] = []
    for p in sorted(artifacts_dir.glob("panel-ltr*.json")):
        # Skip the current-staging / previous / archive / golden suffixes
        if any(s in p.name for s in (".staging.", ".previous.", ".pre-train.")):
            continue
        if p.name == "panel-ltr.json":
            name = "current"
        elif (m := _BAK_RE.match(p.name)):
            name = m.group("name")
        else:
            # Other naming variants like panel-ltr.golden-daily.json,
            # panel-ltr.PRE-MINUTE.json, panel-ltr.hourly.json — include as-is
            name = p.stem.replace("panel-ltr.", "")
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("skip %s: %s", p.name, exc)
            continue
        out.append(_extract_metrics(data, name, p))
    return out


# ── Composite score ───────────────────────────────────────────────────────────

def _zscore(values: list[float | None]) -> list[float]:
    """Z-score a list of values, mapping None → 0 (neutral). All-None or
    zero-std lists → all zeros."""
    nums = [v for v in values if v is not None]
    if len(nums) < 2:
        return [0.0 for _ in values]
    import statistics
    mean = statistics.mean(nums)
    std  = statistics.pstdev(nums)
    if std == 0:
        return [0.0 for _ in values]
    return [((v - mean) / std) if v is not None else 0.0 for v in values]


def score_candidates(cands: list[Candidate], weights: dict) -> list[Candidate]:
    """Compute composite score in-place; return cands sorted descending."""
    if not cands:
        return cands
    z_ic     = _zscore([c.oos_mean_ic for c in cands])
    z_sharpe = _zscore([c.sim_sharpe  for c in cands])
    z_calmar = _zscore([c.sim_calmar  for c in cands])

    w_ic  = float(weights.get("ic",     0.5))
    w_sh  = float(weights.get("sharpe", 0.3))
    w_cal = float(weights.get("calmar", 0.2))

    for c, zi, zs, zc in zip(cands, z_ic, z_sharpe, z_calmar):
        c.composite = (w_ic * zi) + (w_sh * zs) + (w_cal * zc)
    return sorted(cands, key=lambda x: x.composite, reverse=True)


# ── Output formatting ─────────────────────────────────────────────────────────

def _fmt(v: float | None, fmt: str = ".4f", missing: str = "—") -> str:
    if v is None:
        return missing
    try:
        return f"{v:{fmt}}"
    except (TypeError, ValueError):
        return missing


def print_leaderboard(cands: list[Candidate], weights: dict) -> None:
    print()
    print(f"  Composite weights: ic={weights.get('ic',0.5):.2f} "
          f"sharpe={weights.get('sharpe',0.3):.2f} "
          f"calmar={weights.get('calmar',0.2):.2f}")
    print()
    header = (f"  {'rank':>4}  {'name':<22} {'composite':>10}  "
              f"{'oos_ic':>8}  {'sharpe':>8}  {'calmar':>8}  "
              f"{'apy':>8}  {'turn':>8}  {'feat':>5}  {'rows':>7}")
    print(header)
    print("  " + "─" * (len(header) - 2))
    for i, c in enumerate(cands, 1):
        print(f"  {i:>4}  {c.name:<22} {c.composite:>+10.3f}  "
              f"{_fmt(c.oos_mean_ic):>8}  "
              f"{_fmt(c.sim_sharpe, '.2f'):>8}  "
              f"{_fmt(c.sim_calmar, '.2f'):>8}  "
              f"{_fmt(c.sim_apy, '.2%'):>8}  "
              f"{_fmt(c.sim_turnover, '.2f'):>8}  "
              f"{c.feature_count or '—':>5}  "
              f"{c.panel_rows or '—':>7}")
    print()


def print_warnings(cands: list[Candidate]) -> None:
    """Surface caveats so the operator doesn't trust a misleading rank."""
    rows = {c.panel_rows for c in cands if c.panel_rows is not None}
    if len(rows) > 1:
        log.warning("⚠️  Candidates have DIFFERENT panel_rows (%s) — OOS ICs "
                    "are NOT directly comparable. Verify all artifacts were "
                    "trained on overlapping panels before trusting the ranking.",
                    sorted(rows))
    no_smoke = [c.name for c in cands
                if c.sim_apy is None and c.sim_sharpe is None and c.sim_calmar is None]
    if no_smoke and len(no_smoke) < len(cands):
        log.warning("⚠️  Candidates without sim_smoke metrics (treated as "
                    "neutral z=0): %s. Consider running sim_smoke on these "
                    "before promoting based on composite.", no_smoke)
    if all(c.sim_apy is None for c in cands):
        log.info("ℹ️  No candidate has sim_smoke metrics — falling back to "
                 "IC-only ranking. Composite ≈ z(oos_mean_ic) × w_ic.")


# ── Promote winner ────────────────────────────────────────────────────────────

def promote_winner(strategy_dir: Path, candidate_name: str,
                   cands: list[Candidate]) -> int:
    target = next((c for c in cands if c.name == candidate_name), None)
    if target is None:
        log.error("Candidate '%s' not found. Available: %s",
                  candidate_name, [c.name for c in cands])
        return 1
    if target.name == "current":
        log.info("'current' is already the production model — no-op.")
        return 0

    sys.path.insert(0, str(strategy_dir))
    from kernel.model_acceptance import promote  # noqa: PLC0415

    active = strategy_dir / "artifacts" / "panel-ltr.json"
    # Stage the winner under the standard staging path so promote()'s
    # mv(staging→active) + mv(active→.previous) atomic-ish swap fires.
    staging = active.with_suffix(".staging.json")
    shutil.copy2(target.path, staging)
    promote(staging, active)
    log.info("✅ Promoted %s → %s (prior preserved at .previous.json)",
             target.path.name, active.name)
    return 0


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_weights(spec: str | None) -> dict:
    if not spec:
        return {"ic": 0.5, "sharpe": 0.3, "calmar": 0.2}
    out: dict[str, float] = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        k, _, v = kv.partition("=")
        out[k.strip()] = float(v.strip())
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="renquant_104")
    p.add_argument("--weights", default=None,
                   help="Composite weights, e.g. 'ic=0.5,sharpe=0.3,calmar=0.2'")
    p.add_argument("--promote", default=None,
                   help="If given, promote the named candidate to "
                        "panel-ltr.json after ranking. Use 'current' to no-op.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    artifacts_dir = strategy_dir / "artifacts"
    if not artifacts_dir.exists():
        log.error("artifacts dir not found: %s", artifacts_dir)
        return 2

    weights = parse_weights(args.weights)
    cands = discover_candidates(artifacts_dir)
    if not cands:
        log.error("no panel-ltr*.json candidates found in %s", artifacts_dir)
        return 2

    cands = score_candidates(cands, weights)
    print_warnings(cands)
    print_leaderboard(cands, weights)

    if args.promote:
        return promote_winner(strategy_dir, args.promote, cands)
    return 0


if __name__ == "__main__":
    sys.exit(main())
