#!/usr/bin/env python3
"""Regenerate the durable OOS pick table — Track A evidence-base prerequisite.

Per renquant-orchestrator `doc/design/2026-06-28-renquant105-direction-decision.md`
§4 and `doc/design/2026-07-01-104-105-design-review-amendments.md` A7: the A1
audit that anchors the renquant105 direction decision (genuine per-name IC has a
CI spanning 0; per-regime BULL_CALM genuine IC ~= -0.003) lived only as deleted
`/tmp` scratch from unmerged scripts — not a committed, re-runnable artifact.
This script is Track A's first concrete step: a committed generator that
re-scores the prod GBDT walk-forward manifest, read-only, through the SAME
point-in-time manifest contract the walk-forward gate itself uses
(`scripts.run_wf_gate._score_manifest_sanity` / `WalkForwardModelLoader`), and
persists the per-(date, name) result as a durable parquet.

This does NOT reimplement the scoring/manifest-dispatch logic — it reuses the
exact functions `scripts/analyze_manifest_sanity_placebo.py` already uses for
its own IC/placebo diagnostics on this same manifest, so the regenerated table
is provably the same computation the original (deleted) A1 audit ran, not a
fresh reinterpretation.

Output schema (one row per (date, name), `data/exp/oos_pick_table_recipe_v2.parquet`):
    date              -- pick date (Timestamp)
    name              -- ticker
    score             -- the point-in-time model's raw score (mu)
    decile_rank       -- cross-sectional decile of `score` WITHIN THE DATE,
                         0 (worst) .. 9 (best); decile 9 is the model's
                         top-decile long-side candidates.
    fwd_60d_excess    -- the RAW (unclipped) realized forward-60-trading-day
                         excess return actually observed for that (date, name)
    regime            -- the live regime label (BEAR/BULL_CALM/CHOPPY/
                         BULL_VOLATILE) at the pick date, from the same
                         regime-classifier task chain the live book uses

This is an EXPERIMENT path (`data/exp/`), never a canonical prod path — it does
not feed any live trading decision. It exists so Track A's conditional-pick-
quality test (direction-decision doc §4) has a durable table to run against
instead of depending on ephemeral scratch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
for _p in (REPO, REPO / "scripts", STRATEGY_DIR):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from scripts.run_wf_gate import (  # noqa: E402
    _load_artifact_payload,
    _load_sanity_panel,
    _manifest_uri_to_path,
    _sanity_model_label_col,
    _score_manifest_sanity,
)
from scripts.analyze_manifest_sanity_placebo import (  # noqa: E402
    build_regime_series,
    summarize_ic,
)

DEFAULT_MANIFEST = (
    STRATEGY_DIR / "artifacts" / "sim" / "walkforward_manifest_gbdt_prod_recipe_v2.json"
)
DEFAULT_OUTPUT = REPO / "data" / "exp" / "oos_pick_table_recipe_v2.parquet"
N_DECILES = 10


def _reference_artifact_path(manifest_path: Path) -> Path:
    """The manifest's LAST retrain entry, resolved via the shared bounded URI
    resolver (`_manifest_uri_to_path` — same contract the WF loader and sim
    adapter use, not a hand-rolled path join).

    Self-consistent by construction: every entry in one manifest shares the
    same training recipe, so the last entry is guaranteed to match the
    recipe every OTHER entry in this manifest was built with. Used ONLY to
    resolve `feature_cols`/label/lookahead metadata and for
    `_score_manifest_sanity`'s internal recipe-validation cross-check —
    per-date SCORING dispatches through the manifest's own entries via
    `WalkForwardModelLoader`, not through this reference artifact.
    """
    manifest = json.loads(manifest_path.read_text())
    retrains = manifest.get("retrains") or []
    if not retrains:
        raise ValueError(f"manifest has no retrain entries: {manifest_path}")
    last = retrains[-1]
    return _manifest_uri_to_path(
        manifest_path,
        last["artifact_uri"],
        expected_digest=last.get("artifact_sha256"),
    )


def _decile_rank(scores: pd.Series) -> pd.Series:
    """Cross-sectional decile of `scores` within one date, 0 (worst)..9
    (best/top) — decile 9 is the model's top-decile long-side candidates.

    Ties are broken by `rank(method="first")` before `qcut` so `qcut` bins
    strictly-ordered integer ranks (never raw floats), which is what makes
    `duplicates="drop"` a pure safety net rather than something that
    silently reshuffles bin membership. Falls back to fewer than
    `N_DECILES` buckets on a date with too few distinct names for 10 clean
    bins (documented, not a crash) — `qcut` needs at least as many distinct
    values as bins.
    """
    n_unique = int(scores.nunique())
    if n_unique < 2:
        return pd.Series(0, index=scores.index, dtype=int)
    n_bins = min(N_DECILES, n_unique)
    ranks = pd.qcut(
        scores.rank(method="first"), n_bins, labels=False, duplicates="drop"
    )
    return ranks.astype(int)


def build_oos_pick_table(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    min_names: int = 5,
) -> tuple[pd.DataFrame, dict]:
    """Re-score the prod GBDT walk-forward manifest read-only and return the
    per-(date, name) pick table + a metadata dict (manifest/artifact
    provenance, val_cut, row/date/name counts, upstream panel/score
    metadata) for the caller to persist alongside the table."""
    reference_artifact_path = _reference_artifact_path(manifest_path)
    artifact = _load_artifact_payload(reference_artifact_path)
    label = _sanity_model_label_col(artifact)
    feat_cols = list(artifact.get("feature_cols") or [])
    if not feat_cols:
        raise ValueError(f"reference artifact missing feature_cols: {reference_artifact_path}")

    panel, panel_meta = _load_sanity_panel(feat_cols, label)
    panel = panel.dropna(subset=[label]).copy()
    panel["date"] = pd.to_datetime(panel["date"])
    distinct = sorted(panel["date"].unique())
    # Same 80th-percentile-date val split analyze_manifest_sanity_placebo.py
    # uses for its own IC/placebo diagnostics on this manifest — not a fresh
    # choice, the same computation the (deleted) A1 audit ran.
    val_cut = pd.Timestamp(distinct[int(len(distinct) * 0.8)])
    val = panel[panel["date"] > val_cut].copy()
    if val.empty:
        raise ValueError("empty validation partition")

    mu, score_meta = _score_manifest_sanity(
        val,
        feat_cols,
        manifest_path,
        reference_artifact_path,
        artifact,
        panel_history=panel,
    )
    val = val.loc[mu.index].copy()
    mu = mu.loc[val.index]

    regimes = build_regime_series(val["date"].unique(), strategy_dir=STRATEGY_DIR)
    regimes = regimes[["date", "regime"]].copy()
    regimes["date"] = pd.to_datetime(regimes["date"])

    table = pd.DataFrame({
        "date": pd.to_datetime(val["date"]).values,
        "name": val["ticker"].astype(str).values,
        "score": mu.astype(float).values,
        # RAW (unclipped) realized value — Track A's spec (direction-decision
        # doc §4) computes the candidate-success label directly off this, and
        # clipping (as analyze_manifest does for its own IC-robustness math)
        # would silently distort the economic net-of-cost evaluation.
        "fwd_60d_excess": val[label].astype(float).values,
    })
    table["decile_rank"] = table.groupby("date")["score"].transform(_decile_rank)
    table = table.merge(regimes, on="date", how="left")
    table = table[["date", "name", "score", "decile_rank", "fwd_60d_excess", "regime"]]
    table = table.sort_values(["date", "name"]).reset_index(drop=True)

    meta = {
        "manifest": str(manifest_path),
        "reference_artifact": str(reference_artifact_path),
        "label": label,
        "val_cut": val_cut.date().isoformat(),
        "n_rows": int(len(table)),
        "n_dates": int(table["date"].nunique()),
        "n_names": int(table["name"].nunique()),
        "panel_meta": panel_meta,
        "score_meta": score_meta,
    }
    return table, meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--min-names", type=int, default=5,
                     help="minimum names per date for a cross-sectional IC to count")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.output).resolve()

    table, meta = build_oos_pick_table(manifest_path=manifest_path, min_names=args.min_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)

    print(f"wrote {out_path}")
    print(f"  manifest={meta['manifest']}")
    print(f"  reference_artifact={meta['reference_artifact']}")
    print(f"  label={meta['label']} val_cut={meta['val_cut']}")
    print(f"  n_rows={meta['n_rows']} n_dates={meta['n_dates']} n_names={meta['n_names']}")

    overall = summarize_ic(
        table["score"], table["fwd_60d_excess"], table["date"], min_names=args.min_names
    )
    print(f"  overall genuine IC: mean={overall['mean_ic']} n_dates={overall['n_dates']} "
          f"n_rows={overall['n_rows']}")
    for regime_val, sub in table.groupby("regime", dropna=False):
        r = summarize_ic(
            sub["score"], sub["fwd_60d_excess"], sub["date"], min_names=args.min_names
        )
        print(f"  regime={regime_val}: mean_ic={r['mean_ic']} n_dates={r['n_dates']} "
              f"n_rows={r['n_rows']}")


if __name__ == "__main__":
    main()
