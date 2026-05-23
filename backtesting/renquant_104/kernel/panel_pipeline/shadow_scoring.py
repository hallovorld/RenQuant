"""Shadow model scoring — record alt-model decisions via MLflow tracking.

Per 2026-05-18 user request: inference accepts shadow models that run
the full pipeline but DON'T submit orders; all data recorded for review.

Per 2026-05-18 second update: use 3rd-party library (MLflow) instead of
custom SQLite — battle-tested experiment tracking, built-in UI for
comparison, standard schema.

MLflow tracking layout:
  experiment_name = ranking.panel_scoring.shadow_experiment
                    (default: "renquant_104_shadow")
  per inference: ONE MLflow Run per (date, shadow_model)
    tags:    as_of_date / shadow_name / shadow_kind / primary_kind
    metrics: mean_primary_score / mean_shadow_score / mean_diff
             corr_primary_shadow / rank_agreement_top5 / top5_overlap
    artifact: comparison.csv (per-ticker primary vs shadow scores)

Query later (UI or programmatic):
  $ mlflow ui --backend-store-uri file:./mlruns
  → http://127.0.0.1:5000 → experiment → compare runs

  # Or programmatic:
  import mlflow
  exp = mlflow.get_experiment_by_name("renquant_104_shadow")
  runs = mlflow.search_runs(exp.experiment_id, filter_string="tags.shadow_name='patchtst_v1'")
  print(runs[["start_time", "metrics.mean_diff", "metrics.corr_primary_shadow"]])

Why MLflow over custom SQLite:
  - Standard schema, well-documented
  - Built-in comparison UI
  - Run-level filtering/aggregation
  - Artifact storage (per-bar comparison tables)
  - 3rd-party maintained, battle-tested in production
  - No new dependency (mlflow 3.12.0 already installed)

Tests in tests/test_shadow_scoring.py.
"""
from __future__ import annotations
import datetime
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Job, Task

log = logging.getLogger("kernel.panel_pipeline.shadow_scoring")

# OMP fix per [[concurrency_resource_budget]]: ensure single-thread BEFORE
# any torch model construction in shadow scorers (PatchTST etc).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Default MLflow tracking URI: file-based local store at repo/mlruns
_DEFAULT_TRACKING_URI = "file:" + str(Path(__file__).resolve().parents[4] / "mlruns")
_DEFAULT_EXPERIMENT = "renquant_104_shadow"
_SCORER_CACHE: dict[tuple[str, str], object] = {}


def _ensure_mlflow_setup(tracking_uri: Optional[str] = None,
                         experiment_name: Optional[str] = None) -> str:
    """Set MLflow tracking URI + experiment. Returns experiment_id."""
    import mlflow  # noqa: PLC0415
    mlflow.set_tracking_uri(tracking_uri or _DEFAULT_TRACKING_URI)
    name = experiment_name or _DEFAULT_EXPERIMENT
    exp = mlflow.get_experiment_by_name(name)
    if exp is None:
        exp_id = mlflow.create_experiment(name)
    else:
        exp_id = exp.experiment_id
    return exp_id


def _log_shadow_run(experiment_id: str, as_of_date, shadow_name: str,
                    shadow_kind: str, primary_kind: str,
                    primary_scores: dict[str, float],
                    shadow_scores: dict[str, float],
                    primary_ranks: dict[str, int],
                    shadow_ranks: dict[str, int]) -> None:
    """Persist one shadow run's comparison to MLflow."""
    import mlflow  # noqa: PLC0415

    # Aggregate metrics
    tickers = sorted(set(primary_scores) & set(shadow_scores))
    if not tickers:
        return
    ps = np.array([primary_scores[t] for t in tickers])
    ss = np.array([shadow_scores[t] for t in tickers])
    diffs = ss - ps
    # Rank agreement: how many of top-5 primary are in top-5 shadow
    n_top = min(5, len(tickers))
    top_primary = sorted(primary_ranks.items(), key=lambda x: x[1])[:n_top]
    top_shadow = sorted(shadow_ranks.items(), key=lambda x: x[1])[:n_top]
    top_primary_set = {t for t, _ in top_primary}
    top_shadow_set = {t for t, _ in top_shadow}
    overlap = len(top_primary_set & top_shadow_set)
    # Pearson correlation
    if np.std(ps) > 1e-9 and np.std(ss) > 1e-9:
        corr = float(np.corrcoef(ps, ss)[0, 1])
    else:
        corr = float("nan")

    run_name = f"{shadow_name}_{as_of_date}"
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.set_tags({
            "as_of_date": str(as_of_date),
            "shadow_name": shadow_name,
            "shadow_kind": shadow_kind,
            "primary_kind": primary_kind,
            "n_candidates": str(len(tickers)),
        })
        mlflow.log_metrics({
            "mean_primary_score": float(np.mean(ps)),
            "mean_shadow_score": float(np.mean(ss)),
            "mean_diff": float(np.mean(diffs)),
            "std_diff": float(np.std(diffs)),
            "corr_primary_shadow": corr,
            f"top{n_top}_overlap": float(overlap),
            f"top{n_top}_overlap_pct": float(overlap / n_top) if n_top else 0.0,
        })
        # Per-ticker comparison table as artifact
        comparison = pd.DataFrame({
            "ticker": tickers,
            "primary_score": ps,
            "shadow_score": ss,
            "diff": diffs,
            "primary_rank": [primary_ranks.get(t, -1) for t in tickers],
            "shadow_rank": [shadow_ranks.get(t, -1) for t in tickers],
            "rank_diff": [shadow_ranks.get(t, 0) - primary_ranks.get(t, 0)
                          for t in tickers],
        })
        # MLflow log_table writes to artifacts/<artifact_file>
        mlflow.log_table(comparison, "comparison.json")


class ApplyShadowScoringTask(Task):
    """Run each configured shadow model on the SAME candidates as primary,
    record scores via MLflow tracking. Read-only — no order submission.

    Reads:
      - ctx.candidates (with .panel_score set by main)
      - ctx.config["ranking"]["panel_scoring"]["shadow_models"]
      - ctx.config["ranking"]["panel_scoring"]["shadow_tracking_uri"]
        (default: file:<repo>/mlruns)
      - ctx.config["ranking"]["panel_scoring"]["shadow_experiment"]
        (default: "renquant_104_shadow")

    Writes:
      - MLflow run per shadow model per inference bar

    Soft-fail: shadow errors logged, primary pipeline unaffected.
    """

    name = "ApplyShadowScoringTask"

    def run(self, ctx: InferenceContext) -> bool | None:
        panel_cfg = ctx.config.get("ranking", {}).get("panel_scoring", {})
        if panel_cfg.get("shadow_enabled", True) is False:
            return None
        shadow_models = panel_cfg.get("shadow_models", []) or []
        if not shadow_models:
            return None

        # Primary scores (must be set by main ApplyScoresTask)
        cands = list(ctx.candidates) if ctx.candidates else []
        if not cands:
            log.info("ApplyShadowScoringTask: 0 candidates — skip")
            return None
        primary_scores = {c.ticker: float(c.panel_score)
                          for c in cands if c.panel_score is not None}
        if not primary_scores:
            return None
        sorted_primary = sorted(primary_scores.items(), key=lambda x: -x[1])
        primary_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_primary)}
        primary_kind = panel_cfg.get("kind", "xgb")

        shadow_log_mlflow = bool(panel_cfg.get("shadow_log_mlflow", True))
        exp_id = None
        if shadow_log_mlflow:
            try:
                exp_id = _ensure_mlflow_setup(
                    panel_cfg.get("shadow_tracking_uri"),
                    panel_cfg.get("shadow_experiment"))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: MLflow setup failed: %s — skip",
                             exc)
                return None

        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415
        repo = Path(__file__).resolve().parents[4]

        for sm in shadow_models:
            name = sm.get("name", "unnamed_shadow")
            kind = sm.get("kind")
            artifact_path = sm.get("artifact_path")
            if not kind or not artifact_path:
                log.warning("ApplyShadowScoringTask: shadow %s missing "
                             "kind/artifact_path", name)
                continue
            p = Path(artifact_path)
            if not p.is_absolute():
                p = repo / p
            try:
                handler = registry.get(kind)
            except ValueError as exc:
                log.warning("ApplyShadowScoringTask: %s", exc)
                continue

            # Inject shadow's feature_cols + seq_len + regime_router into config copy
            shadow_panel_cfg = dict(panel_cfg)
            if "feature_cols" in sm:
                shadow_panel_cfg["feature_cols"] = sm["feature_cols"]
            if "seq_len" in sm:
                shadow_panel_cfg["seq_len"] = sm["seq_len"]
            if "regime_router" in sm:  # composite scorer sub-config
                shadow_panel_cfg["regime_router"] = sm["regime_router"]
            shadow_cfg = dict(ctx.config)
            shadow_cfg.setdefault("ranking", {})["panel_scoring"] = shadow_panel_cfg

            cache_key = (kind, str(p))
            scorer = _SCORER_CACHE.get(cache_key)
            if scorer is None:
                try:
                    scorer = handler.scorer_loader(p, shadow_cfg)
                except Exception as exc:
                    log.warning("ApplyShadowScoringTask: shadow %s (%s) load failed: %s",
                                 name, kind, exc)
                    continue
                _SCORER_CACHE[cache_key] = scorer

            target_tickers = list(primary_scores.keys())
            try:
                if getattr(scorer, "requires_history", False):
                    panel_history = getattr(ctx, "_panel_history", None)
                    if panel_history is None:
                        panel_parquet = (repo / "data"
                                          / "alpha158_291_fundamental_dataset.parquet")
                        full = pd.read_parquet(panel_parquet)
                        full["date"] = pd.to_datetime(full["date"])
                        today_ts = pd.Timestamp(getattr(ctx, "today",
                                                          datetime.date.today()))
                        past = full[full["date"] < today_ts]
                        dates = sorted(past["date"].unique())[-scorer.seq_len:]
                        panel_history = past[past["date"].isin(dates)]
                    # If scorer accepts current_regime (RegimeRouterScorer), pass it
                    import inspect as _inspect  # noqa: PLC0415
                    sig = _inspect.signature(scorer.score_with_history)
                    if "current_regime" in sig.parameters:
                        series = scorer.score_with_history(
                            panel_history, target_tickers,
                            current_regime=getattr(ctx, "regime", "BULL_CALM"))
                    else:
                        series = scorer.score_with_history(
                            panel_history, target_tickers)
                else:
                    X = getattr(ctx, "_panel_matrix", None)
                    if X is None or X.empty:
                        log.warning("ApplyShadowScoringTask: shadow %s needs "
                                     "matrix but ctx._panel_matrix empty", name)
                        continue
                    fc = scorer.feature_cols
                    missing = [c for c in fc if c not in X.columns]
                    if missing:
                        log.warning("ApplyShadowScoringTask: shadow %s missing "
                                     "cols: %s", name, missing[:5])
                        continue
                    series = scorer.score(X[fc].fillna(0))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: shadow %s score failed: %s",
                             name, exc)
                continue

            shadow_dict = series.to_dict()
            sorted_shadow = sorted(shadow_dict.items(), key=lambda x: -x[1])
            shadow_ranks = {t: i + 1 for i, (t, _) in enumerate(sorted_shadow)}

            # 2026-05-19 (user mandate "want to know what shadow will do in
            # ntfy"): stash a compact summary on ctx so live.runner can
            # surface it. Single-line-of-ntfy friendly: shadow top-3 picks,
            # top-10 overlap with primary, Spearman rank correlation.
            try:
                import numpy as _np  # noqa: PLC0415
                top10_primary = set(t for t, _ in sorted_primary[:10])
                top10_shadow = set(t for t, _ in sorted_shadow[:10])
                overlap = len(top10_primary & top10_shadow)
                common = sorted(set(primary_scores) & set(shadow_dict))
                if len(common) >= 5:
                    pr = _np.array([primary_ranks[t] for t in common])
                    sr = _np.array([shadow_ranks[t] for t in common])
                    from scipy.stats import spearmanr as _sp  # noqa: PLC0415
                    rho, _ = _sp(pr, sr)
                    rho = float(rho) if _np.isfinite(rho) else float("nan")
                else:
                    rho = float("nan")
                top3 = [t for t, _ in sorted_shadow[:3]]
                summary = {
                    "name": name, "kind": kind,
                    "top3": top3,
                    "top10_overlap": overlap,
                    "n_candidates": len(shadow_dict),
                    "spearman_vs_primary": rho,
                }
                if not hasattr(ctx, "_shadow_summary"):
                    ctx._shadow_summary = []  # noqa: SLF001
                ctx._shadow_summary.append(summary)  # noqa: SLF001
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: ctx summary failed for %s: %s",
                             name, exc)

            if not shadow_log_mlflow or exp_id is None:
                log.info("ApplyShadowScoringTask: shadow %s (%s) scored %d "
                         "candidates (MLflow disabled)",
                         name, kind, len(shadow_dict))
                continue

            try:
                _log_shadow_run(
                    exp_id, getattr(ctx, "today", datetime.date.today()),
                    name, kind, primary_kind,
                    primary_scores, shadow_dict,
                    primary_ranks, shadow_ranks,
                )
                log.info("ApplyShadowScoringTask: shadow %s (%s) logged %d "
                         "candidates via MLflow", name, kind, len(shadow_dict))
            except Exception as exc:
                log.warning("ApplyShadowScoringTask: MLflow log failed for %s: %s",
                             name, exc)

        return None


__all__ = ["ApplyShadowScoringTask"]
