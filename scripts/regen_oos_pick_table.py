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
import hashlib
import json
import subprocess
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head_commit(repo: Path) -> str | None:
    """Best-effort HEAD sha at generation time — not a promise the working
    tree was clean, just provenance of which generator revision ran."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(repo),
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _relpath(path: Path, root: Path) -> str:
    """Portable-as-possible reference for the manifest: relative to REPO when
    the path genuinely lives under this checkout; falls back to the
    ``backtesting/...`` suffix (stable across any clone/worktree of this
    repo, since that layout is repo-internal) when it doesn't — e.g. a
    worktree whose local input files are symlinked to a sibling checkout's
    absolute path, which must never leak that machine's home directory into
    a committed artifact. Only falls back to a genuine absolute path if
    neither resolves, so this stays informational, never load-bearing (the
    sha256 alongside it is the actual durable identity)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        pass
    marker = "backtesting/"
    text = str(resolved)
    idx = text.find(marker)
    if idx != -1:
        return text[idx:]
    return str(path)


def build_manifest(
    meta: dict,
    *,
    manifest_path: Path,
    reference_artifact_path: Path,
) -> dict:
    """A REPRODUCIBILITY RECIPE for regenerating the pick table byte-for-byte —
    explicitly NOT "the durable artifact" itself, since the parquet payload
    (the actual data PAYLOAD) is deliberately NOT committed to git (Codex
    review on #430: `data/exp/oos_pick_table_recipe_v2.parquet` matches
    renquant-orchestrator's `agent_workflows.PROD_PATH_RULES` protected-path
    regex `(^|/)data/.*\\.parquet$` — a mechanically-enforced merge-review
    check, not a style preference). No DVC/LFS/object-storage backend is
    configured anywhere in this repo (checked: no `.gitattributes`, no dvc
    config), so there is nowhere durable for the payload itself to live yet —
    this manifest only tells a future reader/script exactly how to REGENERATE
    it on demand; it does not contain or stand in for the data.

    Codex review round 2 (#430): a git COMMIT hash for the generator is
    self-referential and unreliable — the manifest is written by a run of
    THIS script, so any commit-hash it stamps necessarily predates (or is
    unrelated to) whatever commit the manifest itself eventually lands in;
    a later change to this script (e.g. this very fix) leaves a stale,
    wrong `generator_commit` sitting in the previously-committed manifest
    with no mechanical way to detect the mismatch. Fixed by stamping a
    CONTENT hash of the generator script's own bytes (``generator_sha256``)
    instead — a pure function of what actually ran, valid regardless of git
    history or commit timing. ``generator_commit`` (best-effort ``git
    rev-parse HEAD`` at generation time) is kept as informational context
    only; ``generator_sha256`` is the verifiable provenance anchor a
    consumer should actually check (see ``tests/test_regen_oos_pick_table.py``
    for a test that a fresh checkout's on-disk script hash still matches)."""
    generator_path = Path(__file__).resolve()
    return {
        "schema": {
            "columns": ["date", "name", "score", "decile_rank", "fwd_60d_excess", "regime"],
            "description": (
                "one row per (date, name); score = point-in-time model raw score "
                "(mu); decile_rank = cross-sectional decile within date, "
                "0(worst)-9(best/top); fwd_60d_excess = RAW unclipped realized "
                "forward-60-trading-day excess return; regime = live regime label "
                "at pick date"
            ),
        },
        "recipe": {
            "generator": "scripts/regen_oos_pick_table.py",
            "generator_sha256": _sha256_file(generator_path),
            "generator_commit": _git_head_commit(REPO),
            "generator_commit_note": (
                "best-effort `git rev-parse HEAD` at generation time — "
                "informational only, NOT the provenance anchor (a later commit "
                "to this script would leave this field stale in an "
                "already-committed manifest with no way to detect it; "
                "generator_sha256 above is a content hash of the generator's "
                "own bytes and is always self-consistent regardless of git "
                "history/commit timing)"
            ),
            "manifest_input": _relpath(manifest_path, REPO),
            "manifest_input_sha256": _sha256_file(manifest_path),
            "reference_artifact": _relpath(reference_artifact_path, REPO),
            "reference_artifact_sha256": _sha256_file(reference_artifact_path),
            "label": meta["label"],
            "val_cut": meta["val_cut"],
        },
        "counts": {
            "n_rows": meta["n_rows"],
            "n_dates": meta["n_dates"],
            "n_names": meta["n_names"],
        },
        "object_uri": (
            "NOT PERSISTED - no DVC/LFS/object-storage backend is configured for "
            "this repo yet. This manifest is a REPRODUCIBILITY RECIPE, not the "
            "durable artifact itself: the parquet payload (the actual data) is "
            "regeneratable on demand via `python3 scripts/regen_oos_pick_table.py` "
            "against the SAME pinned manifest_input/reference_artifact "
            "(hash-verified against the fields above, generator hash-verified "
            "against generator_sha256), and is deliberately NOT committed to git "
            "(matches PROD_PATH_RULES in renquant-orchestrator's "
            "agent_workflows.py: `data/.*\\.parquet$` is a protected "
            "production-data path)."
        ),
        "note": (
            "wall-clock generation timestamp intentionally omitted for "
            "determinism/reproducibility framing; generator_sha256 + the input "
            "hashes are the provenance record, not a run timestamp."
        ),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--manifest-output", default=None,
                     help="path for the durable regeneration manifest JSON; "
                          "default is <output stem>.manifest.json next to --output")
    ap.add_argument("--min-names", type=int, default=5,
                     help="minimum names per date for a cross-sectional IC to count")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    out_path = Path(args.output).resolve()
    manifest_out_path = (
        Path(args.manifest_output).resolve() if args.manifest_output
        else out_path.with_suffix("").with_suffix(".manifest.json")
    )

    table, meta = build_oos_pick_table(manifest_path=manifest_path, min_names=args.min_names)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_path, index=False)

    regen_manifest = build_manifest(
        meta,
        manifest_path=manifest_path,
        reference_artifact_path=Path(meta["reference_artifact"]),
    )
    manifest_out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_out_path.write_text(json.dumps(regen_manifest, indent=2, sort_keys=True) + "\n")

    print(f"wrote {out_path} (NOT committed to git — see {manifest_out_path})")
    print(f"wrote durable regeneration manifest {manifest_out_path}")
    print(f"  manifest={meta['manifest']}")
    print(f"  reference_artifact={meta['reference_artifact']}")
    print(f"  label={meta['label']} val_cut={meta['val_cut']}")
    print(f"  n_rows={meta['n_rows']} n_dates={meta['n_dates']} n_names={meta['n_names']}")

    overall = summarize_ic(
        table["score"], table["fwd_60d_excess"], table["date"], min_names=args.min_names
    )
    # NOT "genuine (leak-controlled)" IC — this is the naive per-date Spearman
    # IC only, with no placebo/persistence-injection adjustment. See #431 for
    # the leak-controlled reproduction and why the two must not be conflated.
    print(f"  overall naive IC (no leak-control adjustment): mean={overall['mean_ic']} "
          f"n_dates={overall['n_dates']} n_rows={overall['n_rows']}")
    for regime_val, sub in table.groupby("regime", dropna=False):
        r = summarize_ic(
            sub["score"], sub["fwd_60d_excess"], sub["date"], min_names=args.min_names
        )
        print(f"  regime={regime_val}: naive mean_ic={r['mean_ic']} n_dates={r['n_dates']} "
              f"n_rows={r['n_rows']}")


if __name__ == "__main__":
    main()
