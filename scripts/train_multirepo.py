#!/usr/bin/env python
"""Multi-repo GBDT trainer — train the production panel-LTR model through the pins.

The MODEL-SIDE training (booster, walk-forward CV, artifact assembly) runs out of
the pinned ``renquant-model`` engine (``renquant_model_gbdt.legacy_panel_trainer``,
a byte-identical reconcile of the legacy trainer). The DATA-SIDE (panel load,
normalization from on-disk stats/fund files, config fingerprint, sentiment gate,
inference-smoke) is reused from the umbrella's ``scripts/train_production_model``
— those read files and are not "the model"; the umbrella stays the baseline.

Result is byte-identical to ``python scripts/train_production_model.py`` for the
same args, excluding the two fields that script randomizes by design
(``train_run_id`` = uuid4, ``trained_date`` = utcnow).

Usage (mirrors train_production_model.py):
    python scripts/train_multirepo.py                       # full-panel prod artifact
    python scripts/train_multirepo.py --train-cutoff 2024-06-01 --side-label wf
    python scripts/train_multirepo.py --output-path /tmp/panel-ltr.json --skip-cv
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
SIBLINGS = REPO.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"

# Pinned subrepo source roots that the model-side import needs.
_PIN_SRCS = [
    "renquant-common", "renquant-base-data", "renquant-artifacts", "renquant-model",
]

log = logging.getLogger("train-multirepo")


def _bootstrap() -> None:
    """Put the engine pin (+ its deps) and the strategy dir on sys.path."""
    for name in _PIN_SRCS:
        src = SIBLINGS / name / "src"
        if src.is_dir() and str(src) not in sys.path:
            sys.path.insert(0, str(src))
    # Strategy dir for the umbrella data-side functions + kernel.feature_transform.
    if str(STRATEGY_DIR) not in sys.path:
        sys.path.insert(0, str(STRATEGY_DIR))
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--train-cutoff", type=str, default=None)
    p.add_argument("--output-path", type=str, default=None)
    p.add_argument("--side-label", type=str, default=None)
    p.add_argument("--label", type=str, default=None)
    p.add_argument("--watchlist-file", type=str, default=None)
    p.add_argument("--fingerprint-config", type=str, default=None)
    p.add_argument("--cutoff-embargo-days", type=int, default=None)
    p.add_argument("--cv-n-splits", type=int, default=3)
    p.add_argument("--cv-embargo-days", type=int, default=60)
    p.add_argument("--skip-cv", action="store_true")
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    _bootstrap()
    args = parse_args()

    # ── Data-side: reuse the umbrella's production loaders/contract builders ──
    from scripts.train_production_model import (  # noqa: PLC0415
        LABEL, N_ROUNDS, PARAMS,
        apply_sentiment_training_gate, build_fingerprint_config,
        build_normalization, build_sentiment_training_regime_map,
        infer_label_lookahead_days, stamp_fingerprint,
    )
    # ── Model-side: the pinned engine (byte-identical legacy reconcile) ──
    from renquant_model_gbdt.legacy_panel_trainer import (  # noqa: PLC0415
        build_model_artifact, evaluate_walk_forward_cv, train_xgb,
    )
    # attach_inference_smoke lives in the umbrella trainer (contract-side).
    from scripts.train_production_model import (  # noqa: PLC0415
        attach_inference_smoke, load_and_slice_panel,
    )
    sys.stderr.write(
        "[train-multirepo] model-side=renquant_model_gbdt.legacy_panel_trainer (pin); "
        "data-side=umbrella scripts.train_production_model\n"
    )

    cutoff_date = pd.Timestamp(args.train_cutoff) if args.train_cutoff else None
    if cutoff_date is not None and not args.side_label:
        raise SystemExit("--side-label is required when --train-cutoff is set")
    out_path = (
        Path(args.output_path) if args.output_path
        else REPO / "data" / "panel-ltr-prod-alpha158-fund-fwd60d.json"
    )
    # §5.13.13: a cutoff run is a walk-forward/side artifact — refuse to write it
    # to a path that could clobber the production artifact (mirrors the legacy guard).
    if cutoff_date is not None and "walkforward" not in str(out_path).lower():
        raise SystemExit(
            f"--train-cutoff set but --output-path {str(out_path)!r} does not contain "
            "'walkforward'. §5.13.13: refusing to risk overwriting production artifact."
        )

    train, feat_cols, label_used = load_and_slice_panel(
        cutoff_date, watchlist_file=args.watchlist_file, label_override=args.label,
        cutoff_embargo_days=args.cutoff_embargo_days,
    )
    fingerprint_cfg = build_fingerprint_config(
        fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file, label_used=label_used, feat_cols=feat_cols,
    )
    regime_map = build_sentiment_training_regime_map(train["date"].unique(), fingerprint_cfg)
    train, sentiment_contract = apply_sentiment_training_gate(
        train, feat_cols, fingerprint_cfg, regime_map,
    )
    mu, sd, norm_kind, raw_clip_low, raw_clip_high = build_normalization(train, feat_cols)

    cv_result = None
    if not args.skip_cv:
        cv_result = evaluate_walk_forward_cv(  # ENGINE
            train, feat_cols, normalization_builder=build_normalization,
            label=label_used, params=PARAMS, num_boost_round=N_ROUNDS,
            n_splits=args.cv_n_splits, embargo_days=args.cv_embargo_days,
        )
        log.info("OOS contract CV: mean_ic=%+.4f std=%+.4f folds=%s",
                 cv_result["oos_mean_ic"], cv_result["oos_std_ic"],
                 [round(x, 4) for x in cv_result["oos_per_fold_ic"]])
    else:
        log.warning("--skip-cv set: artifact will not satisfy strict contract")

    booster, train_ic = train_xgb(  # ENGINE
        train, feat_cols, label=label_used, params=PARAMS, num_boost_round=N_ROUNDS,
        feature_means=mu, feature_stds=sd, feature_norm_kind=norm_kind,
    )
    lookahead_days = infer_label_lookahead_days(label_used)
    base_notes = (
        "alpha158 + SEC fund (5) + PEAD (3, E47 promoted 2026-05-08) on R1K "
        "291 tickers, fwd_60d label. PEAD real_signal lift +0.022 over "
        "alpha158+5fund baseline (paired §5.2 sanity passed)."
    )
    notes = base_notes + (f" [side_label={args.side_label}]" if args.side_label else "")
    artifact = build_model_artifact(  # ENGINE
        booster, feat_cols, mu, sd, train, params=PARAMS, num_boost_round=N_ROUNDS,
        feature_norm_kind=norm_kind, feature_raw_clip_low=raw_clip_low,
        feature_raw_clip_high=raw_clip_high, label_used=label_used,
        lookahead_days=lookahead_days, train_ic=train_ic, cv_result=cv_result,
        train_run_id=str(uuid.uuid4())[:8], training_notes=notes,
    )
    # ── Data/contract-side fields the legacy build_artifact layers on ──
    if cutoff_date is not None:
        artifact["cutoff_date"] = cutoff_date.isoformat()
        artifact["cutoff_embargo_days"] = int(
            lookahead_days if args.cutoff_embargo_days is None else args.cutoff_embargo_days
        )
        artifact["effective_train_cutoff_date"] = (
            cutoff_date - pd.offsets.BDay(artifact["cutoff_embargo_days"])
        ).isoformat()
    if args.side_label is not None:
        artifact["side_label"] = args.side_label
    if sentiment_contract:
        artifact.update(sentiment_contract)

    fp = stamp_fingerprint(
        artifact, fingerprint_config_path=args.fingerprint_config,
        watchlist_file=args.watchlist_file, label_used=label_used, feat_cols=feat_cols,
    )
    log.info("Fingerprint: %s", fp)
    attach_inference_smoke(artifact, booster, feat_cols)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    log.info("Saved artifact: %s  (size=%.1f MB)", out_path, out_path.stat().st_size / 1e6)
    log.info("Feature cols (n=%d): %s ... %s", len(feat_cols), feat_cols[:3], feat_cols[-3:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
