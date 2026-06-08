#!/usr/bin/env python
"""Per-regime specialist panel-LTR training driver (Track C, 2026-06-02).

Wraps ``scripts/train_production_model.py`` with a ``--regime-filter`` flag
that restricts training rows to a single detector regime BEFORE the
``rank:pairwise`` group construction. Per-date intra-regime groups keep the
ranking loss intra-cohort.

Per CLAUDE.md §1 PRIME DIRECTIVE: pooled-mean training averages BEAR
dispersion-gradient rows with BULL_CALM calm-gradient rows, so the
"production" GBDT cannot express the BULL_CALM-specific signal. A per-regime
specialist optimizes for its regime's return distribution.

Per §7.5 single source of truth: this driver does NOT fork
``train_production_model``'s booster/normalization code. It reuses the
underlying functions (``load_and_slice_panel``, ``build_normalization``,
``train_xgb``, ``build_artifact``, ``stamp_fingerprint``,
``attach_inference_smoke``) and inserts a regime-filter pass between the
panel load and the training step.

Usage:
    python scripts/train_per_regime_panel.py \\
        --regime-filter BULL_CALM \\
        --output-path backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.bull_calm.json \\
        --side-label specialist_bull_calm_20260602

    # Baseline (no filter, equivalent to train_production_model.py):
    python scripts/train_per_regime_panel.py --regime-filter ALL --output-path ...
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path

# §6.5 hardware saturation — match train_production_model.py defaults.
_THREAD_COUNT = str(os.cpu_count() or 14)
for _k in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, _THREAD_COUNT)

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(STRATEGY_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-per-regime")

CANONICAL_REGIMES = ("BEAR", "BULL_CALM", "BULL_VOLATILE", "CHOPPY")
REGIME_CHOICES = ("ALL", *CANONICAL_REGIMES)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a per-regime specialist panel-LTR artifact.",
    )
    p.add_argument(
        "--regime-filter", choices=REGIME_CHOICES, default="ALL",
        help="Filter training rows to a single detector regime. "
             "ALL = no filter (baseline-equivalent).",
    )
    p.add_argument(
        "--output-path", required=True,
        help="Artifact output path. Required (refusing to default-overwrite).",
    )
    p.add_argument(
        "--side-label", default=None,
        help="Required when --regime-filter != ALL. Stamped into training_notes.",
    )
    p.add_argument(
        "--label", default=None,
        help="Override label column (default fwd_60d_excess from train_production_model).",
    )
    p.add_argument(
        "--watchlist-file", default=None,
        help="Restrict tickers to the named watchlist JSON.",
    )
    p.add_argument(
        "--include-features", default=None,
        help="Comma-list of opt-in addendum feature names (Track B/C: "
             "mom_carry_12_1,beta_dm,rvar_total,idio_vol_market). Threaded into "
             "load_and_slice_panel so a per-regime specialist can use them.",
    )
    p.add_argument(
        "--fingerprint-config", default=None,
        help="Strategy config whose model-relevant fields stamp into the artifact.",
    )
    p.add_argument(
        "--train-cutoff", default=None,
        help="Optional ISO cutoff date — rows date < (cutoff - lookahead) only.",
    )
    p.add_argument(
        "--cv-n-splits", type=int, default=3,
        help="Purged walk-forward folds stamped into the artifact.",
    )
    p.add_argument(
        "--cv-embargo-days", type=int, default=60,
        help="Trading-day embargo between each CV train window and validation fold.",
    )
    p.add_argument(
        "--skip-cv", action="store_true",
        help="Emergency only: skip OOS contract evaluation (artifact will not satisfy strict contract).",
    )
    p.add_argument(
        "--min-group-warn", type=int, default=10,
        help="Warn when intra-regime intra-date groups have fewer than this many rows.",
    )
    return p.parse_args()


def _require_side_label(args: argparse.Namespace) -> None:
    if args.regime_filter != "ALL" and not args.side_label:
        raise SystemExit(
            "--regime-filter != ALL requires --side-label for training_notes provenance."
        )


def _filter_rows_by_regime(
    train: pd.DataFrame,
    regime_filter: str,
    config: dict,
    strategy_dir: Path,
    min_group_warn: int,
) -> pd.DataFrame:
    """Restrict training rows to a single detector regime, intra-date groups intact.

    Reuses the SAME regime-replay helper that the sentiment-gate path uses
    (``_build_training_regime_map`` in pp_panel_training) — single source
    of truth for the detector contract.
    """
    if regime_filter == "ALL":
        return train

    from training_panel.pp_panel_training import _build_training_regime_map  # noqa: PLC0415

    # Synthetic context for the regime replay function.
    spy_path = REPO / "data" / "ohlcv" / "SPY" / "1d.parquet"
    if not spy_path.exists():
        raise FileNotFoundError(f"SPY OHLCV missing for regime replay: {spy_path}")
    spy_df = pd.read_parquet(spy_path)
    from types import SimpleNamespace  # noqa: PLC0415
    cfg = dict(config)
    cfg.setdefault("_strategy_dir", str(strategy_dir))
    ctx = SimpleNamespace(config=cfg, spy_df=spy_df, strategy_dir=strategy_dir)

    log.info("Replaying regime detector across %d unique dates…",
             train["date"].nunique())
    regime_map = _build_training_regime_map(ctx, train["date"].unique())

    norm = pd.to_datetime(train["date"]).dt.normalize()
    regime_series = norm.map(regime_map)
    n_unlabeled = int(regime_series.isna().sum())
    if n_unlabeled:
        log.warning("Regime replay: %d rows had no regime label "
                    "(warmup window) — dropped from specialist training.",
                    n_unlabeled)

    keep = (regime_series == regime_filter).fillna(False)
    filtered = train.loc[keep].copy()
    if filtered.empty:
        raise SystemExit(
            f"--regime-filter {regime_filter} produced 0 training rows "
            f"(panel dates spanning {train['date'].min()} → {train['date'].max()}). "
            "Check the SPY history and detector config."
        )

    # Per-date group size sanity (rank:pairwise needs ≥ a few per date).
    group_sizes = filtered.groupby("date").size()
    log.info("Regime %s: %d rows over %d dates "
             "(per-date mean=%.1f median=%.0f p10=%.0f p90=%.0f)",
             regime_filter, len(filtered), filtered["date"].nunique(),
             float(group_sizes.mean()), float(group_sizes.median()),
             float(group_sizes.quantile(0.10)), float(group_sizes.quantile(0.90)))
    n_small = int((group_sizes < int(min_group_warn)).sum())
    if n_small:
        log.warning("Regime %s: %d/%d dates have <%d candidates per group "
                    "(rank:pairwise loss is weak on tiny groups).",
                    regime_filter, n_small, filtered["date"].nunique(),
                    int(min_group_warn))
    return filtered


def main() -> None:
    args = parse_args()
    _require_side_label(args)

    # Lazy imports — only the heavy training module is touched when we actually train.
    from scripts.train_production_model import (  # noqa: PLC0415
        attach_inference_smoke,
        apply_sentiment_training_gate,
        build_artifact,
        build_fingerprint_config,
        build_normalization,
        build_sentiment_training_regime_map,
        evaluate_walk_forward_cv,
        load_and_slice_panel,
        stamp_fingerprint,
        train_xgb,
    )

    cutoff_date = pd.Timestamp(args.train_cutoff) if args.train_cutoff else None
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    include_features = None
    if args.include_features:
        include_features = [c.strip() for c in args.include_features.split(",") if c.strip()]
    train, feat_cols, label_used = load_and_slice_panel(
        cutoff_date,
        watchlist_file=args.watchlist_file,
        label_override=args.label,
        include_features=include_features,
    )

    # Build the fingerprint config FIRST so the regime replay has the
    # same config the rest of the production path uses.
    fingerprint_cfg = build_fingerprint_config(
        fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file,
        label_used=label_used,
        feat_cols=feat_cols,
    )

    train = _filter_rows_by_regime(
        train, args.regime_filter, fingerprint_cfg, STRATEGY_DIR,
        args.min_group_warn,
    )

    # Sentiment-gate replay is regime-aware too; reuse the existing helper.
    regime_map = build_sentiment_training_regime_map(
        train["date"].unique(), fingerprint_cfg,
    )
    train, sentiment_contract = apply_sentiment_training_gate(
        train, feat_cols, fingerprint_cfg, regime_map,
    )

    mu, sd, norm_kind, raw_clip_low, raw_clip_high = build_normalization(train, feat_cols)
    cv_result = None
    if not args.skip_cv:
        cv_result = evaluate_walk_forward_cv(
            train, feat_cols,
            label=label_used,
            n_splits=args.cv_n_splits,
            embargo_days=args.cv_embargo_days,
        )
        log.info("CV: mean_ic=%+.4f std=%+.4f folds=%s",
                 cv_result["oos_mean_ic"], cv_result["oos_std_ic"],
                 [round(x, 4) for x in cv_result["oos_per_fold_ic"]])

    booster, train_ic = train_xgb(
        train, feat_cols,
        label=label_used,
        feature_means=mu, feature_stds=sd, feature_norm_kind=norm_kind,
    )
    side_label = args.side_label or (
        f"specialist_{args.regime_filter.lower()}" if args.regime_filter != "ALL"
        else None
    )
    artifact = build_artifact(
        booster, feat_cols, mu, sd, train,
        cutoff_date, side_label,
        feature_norm_kind=norm_kind,
        feature_raw_clip_low=raw_clip_low,
        feature_raw_clip_high=raw_clip_high,
        label_used=label_used,
        train_ic=train_ic,
        cv_result=cv_result,
        train_run_id=str(uuid.uuid4())[:8],
        sentiment_contract_metadata=sentiment_contract,
    )
    # Stamp specialist provenance so RegimeEnsemblePanelScorer can audit at load.
    artifact["regime_filter"] = args.regime_filter
    artifact.setdefault("training_notes", "")
    artifact["training_notes"] += f" [regime_filter={args.regime_filter}]"

    fp = stamp_fingerprint(
        artifact,
        fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file,
        label_used=label_used,
        feat_cols=feat_cols,
    )
    log.info("Fingerprint: %s (regime_filter=%s)", fp, args.regime_filter)
    attach_inference_smoke(artifact, booster, feat_cols)

    out_path.write_text(json.dumps(artifact))
    log.info("Saved specialist artifact: %s (size=%.1f MB, regime=%s)",
             out_path, out_path.stat().st_size / 1e6, args.regime_filter)


if __name__ == "__main__":
    main()
