#!/usr/bin/env python
"""Fit panel-rank-calibration directly on the pre-built alpha158+fund panel.

The standard scripts/fit_panel_calibrator.py rebuilds features via
build_inference_matrix → only knows the production 30-feat panel,
not alpha158. So scoring with the new 163-feature alpha158_fund
artifact through that pipeline produced garbage (pool_ic=-0.013,
prob head collapsed to 3 unique y values).

This script bypasses the rebuild: load the pre-built panel parquet,
predict with the panel-LTR XGB artifact directly, then call
fit_global_calibrator with the predictions + actual returns.

Output: backtesting/renquant_104/artifacts/panel-rank-calibration.json
"""
from __future__ import annotations
import argparse, hashlib, json, logging, re, sys
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
from kernel.panel_pipeline.feature_transform import transform_feature_frame  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fit-calib-direct")


def _resolve_repo_path(raw_path: str | None, default: Path) -> Path:
    if not raw_path:
        return default
    p = Path(raw_path)
    return p if p.is_absolute() else REPO / p


def _artifact_fingerprint(path: Path, payload: dict) -> str:
    """Return scorer-file identity, never a shared strategy config identity."""
    return (
        payload.get("artifact_fingerprint")
        or payload.get("artifact_sha256")
        or payload.get("model_fingerprint")
        or payload.get("fingerprint")
        or "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    )


def _infer_raw_er_label(label_col: str) -> str:
    """Return the raw-return label that matches a normalized LTR label.

    The panel-LTR scorer is allowed to train on a cross-sectionally
    normalized label, but the expected-return calibration head feeds Kelly/QP
    as μ, so it must be on return units. The canonical raw-label builder uses
    the same stem plus ``_raw``.
    """
    if label_col.endswith("_raw"):
        return label_col
    m = re.fullmatch(r"(fwd_\d+d_excess)", label_col)
    if m:
        return f"{m.group(1)}_raw"
    return f"{label_col}_raw"


def _infer_label_lookahead_days(label_col: str) -> int:
    """Infer the trading-day forward horizon from a label column name."""
    m = re.search(r"fwd_(\d+)d", str(label_col or ""))
    return int(m.group(1)) if m else 60


def _label_scale_diagnostics(frame: pd.DataFrame, label_col: str) -> dict[str, float | int | bool]:
    if label_col not in frame.columns:
        raise KeyError(f"label column not present: {label_col}")
    s = pd.to_numeric(frame[label_col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        raise ValueError(f"{label_col}: no finite labels")
    if "date" in frame.columns:
        by_date = frame.assign(date=pd.to_datetime(frame["date"]))
        per_date_std = (
            by_date.dropna(subset=[label_col])
            .groupby("date")[label_col]
            .std()
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
    else:
        per_date_std = pd.Series(dtype=float)
    per_date_std_median = float(per_date_std.median()) if not per_date_std.empty else float("nan")
    per_date_std_mean = float(per_date_std.mean()) if not per_date_std.empty else float("nan")
    global_std = float(s.std())
    abs_gt_20 = float((s.abs() > 0.20).mean())
    # Production normalized labels have per-date std≈1 and ~80% of rows outside
    # ±20%. Raw 60d excess returns sit around std≈0.18 with only ~14% outside.
    looks_standardized = (
        global_std > 0.50
        and 0.75 <= per_date_std_median <= 1.25
        and abs_gt_20 > 0.50
    )
    return {
        "n": int(len(s)),
        "mean": float(s.mean()),
        "std": global_std,
        "min": float(s.min()),
        "max": float(s.max()),
        "abs_gt_20pct_fraction": abs_gt_20,
        "per_date_std_mean": per_date_std_mean,
        "per_date_std_median": per_date_std_median,
        "looks_cross_sectional_standardized": bool(looks_standardized),
    }


def _load_expected_return_labels(
    *,
    scoring_panel: pd.DataFrame,
    panel_path: Path,
    raw_label_panel_path: Path,
    model_label_col: str,
    er_label_col: str | None,
    allow_normalized_er_label: bool,
) -> tuple[pd.DataFrame, str, dict[str, float | int | bool], str]:
    """Attach the raw expected-return label and enforce its unit contract."""
    chosen = er_label_col or _infer_raw_er_label(model_label_col)
    source = str(panel_path)

    if chosen not in scoring_panel.columns:
        if not raw_label_panel_path.exists():
            if allow_normalized_er_label and model_label_col in scoring_panel.columns:
                chosen = model_label_col
            else:
                raise FileNotFoundError(
                    f"Expected-return label {chosen!r} is not in {panel_path} and "
                    f"raw-label panel is missing: {raw_label_panel_path}. "
                    "Run scripts/build_raw_fwd60d_label.py or pass --raw-label-panel."
                )
        else:
            log.info("Loading raw expected-return labels from %s", raw_label_panel_path)
            raw_labels = pd.read_parquet(raw_label_panel_path, columns=["ticker", "date", chosen])
            raw_labels["date"] = pd.to_datetime(raw_labels["date"])
            before = len(scoring_panel)
            scoring_panel = scoring_panel.merge(
                raw_labels,
                on=["ticker", "date"],
                how="left",
                validate="many_to_one",
            )
            source = str(raw_label_panel_path)
            log.info(
                "Merged ER label %s from raw-label panel: rows=%d finite=%d",
                chosen,
                before,
                int(scoring_panel[chosen].notna().sum()),
            )

    if chosen not in scoring_panel.columns:
        raise KeyError(
            f"Expected-return label {chosen!r} is unavailable after raw-label merge. "
            "Pass --er-label-col or rebuild the raw-label panel."
        )

    diag = _label_scale_diagnostics(scoring_panel, chosen)
    log.info(
        "ER label %s diagnostics: n=%d mean=%+.4f std=%.4f abs(|r|>20%%)=%.1f%% "
        "per-date-std-median=%.4f source=%s",
        chosen,
        diag["n"],
        diag["mean"],
        diag["std"],
        100 * float(diag["abs_gt_20pct_fraction"]),
        diag["per_date_std_median"],
        source,
    )
    if diag["looks_cross_sectional_standardized"] and not allow_normalized_er_label:
        raise ValueError(
            f"EXPECTED-RETURN-LABEL CONTRACT FAIL: {chosen!r} looks like a "
            "cross-sectionally standardized rank label (global/std and per-date "
            "std near 1, most rows outside ±20%). Kelly/QP μ must be on raw "
            "return units. Use fwd_60d_excess_raw from "
            "data/alpha158_291_fundamental_dataset_rawlabel.parquet, or pass "
            "--allow-normalized-er-label only for isolated research diagnostics."
        )

    return scoring_panel, chosen, diag, source


def main():
    # 2026-05-11 audit G2: prod artifacts moved to artifacts/prod/.
    # Defaults updated; CLI args added so sim + ablation paths can override.
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scorer-artifact", default=None,
        help="Path to panel-LTR XGB JSON. Defaults to artifacts/prod/panel-ltr.alpha158_fund.json. "
             "Relative paths resolve against repo root. Use a sim-only scorer "
             "(trained with cutoff < sim_start) to get a leak-free sim calibrator.",
    )
    p.add_argument(
        "--out", default=None,
        help="Output calibrator path. Defaults to artifacts/prod/panel-rank-calibration.json.",
    )
    p.add_argument(
        "--panel", default=None,
        help="Panel parquet. Defaults to data/alpha158_291_fundamental_dataset.parquet.",
    )
    p.add_argument(
        "--raw-label-panel", default=None,
        help="Panel parquet carrying raw forward-return labels. Defaults to "
             "data/alpha158_291_fundamental_dataset_rawlabel.parquet.",
    )
    p.add_argument(
        "--er-label-col", default=None,
        help="Raw expected-return label to use for Kelly/QP μ calibration. "
             "Default infers '<model label>_raw' (e.g. fwd_60d_excess_raw).",
    )
    p.add_argument(
        "--allow-normalized-er-label", action="store_true",
        help="Research-only escape hatch. By default the script hard-fails if "
             "the expected-return label looks cross-sectionally standardized.",
    )
    p.add_argument(
        "--data-start", default=None,
        help="ISO date. Drop scoring dates < this. Used with --data-end for "
             "true OOS calibration (scorer trained ≤T → score (T, T+window)).",
    )
    p.add_argument(
        "--data-end", default=None,
        help="ISO date. Drop scoring dates >= this. Must be ≤ "
             "(sim_start - lookahead_days - safety_buffer) for leak-free sim.",
    )
    p.add_argument(
        "--method", default="platt", choices=["platt", "isotonic"],
        help="Calibration method. Default platt (sigmoid, strictly monotone, "
             "no flat regions). Use isotonic ONLY for sim re-evaluation of "
             "legacy artifacts; do NOT promote isotonic to prod (2026-05-18 "
             "incident: 57%% flat region tied 79%% of candidates).",
    )
    args = p.parse_args()
    # Make args visible to fit logic
    global cli_args
    cli_args = args

    panel_path = _resolve_repo_path(
        args.panel,
        REPO / "data" / "alpha158_291_fundamental_dataset.parquet",
    )
    raw_label_panel_path = _resolve_repo_path(
        args.raw_label_panel,
        REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet",
    )
    art_path = _resolve_repo_path(
        args.scorer_artifact,
        REPO / "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json",
    )
    out_path = _resolve_repo_path(
        args.out,
        REPO / "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json",
    )
    LABEL_60D  = "fwd_60d_excess"

    log.info("Loading panel + panel-LTR artifact...")
    art = json.loads(art_path.read_text())
    feat_cols = art["feature_cols"]
    # 2026-05-23 contract: every calibrator must bind to the exact scorer
    # distribution it was fitted against. A config fingerprint can be shared
    # by many WF folds; use scorer artifact/file identity instead.
    fingerprint = _artifact_fingerprint(art_path, art)
    # Round 3 audit (G10): label column from the artifact, not hardcoded.
    # A short-horizon scorer (fwd_5d / fwd_20d) used with the previous
    # hardcoded `fwd_60d_excess` produced a silent label/horizon mismatch.
    label_col = art.get("label_col", LABEL_60D)
    lookahead_days = _infer_label_lookahead_days(label_col)
    log.info("Artifact fingerprint=%s  features=%d  label_col=%s",
             fingerprint, len(feat_cols), label_col)

    booster = xgb.Booster()
    booster.load_model(bytearray(art["booster_raw_json"].encode("utf-8")))

    panel = pd.read_parquet(panel_path)
    panel["date"] = pd.to_datetime(panel["date"])
    log.info("Panel: rows=%d tickers=%d dates %s..%s",
             len(panel), panel["ticker"].nunique(),
             panel["date"].min().date(), panel["date"].max().date())

    # 2026-05-11: optional date window filter for OOS sim calibration.
    if args.data_start:
        start = pd.Timestamp(args.data_start)
        before = len(panel)
        panel = panel[panel["date"] >= start]
        log.info("--data-start=%s: filtered %d → %d rows", args.data_start, before, len(panel))
    if args.data_end:
        end = pd.Timestamp(args.data_end)
        before = len(panel)
        panel = panel[panel["date"] < end]
        log.info("--data-end=%s: filtered %d → %d rows", args.data_end, before, len(panel))

    panel, er_label_col, er_label_diag, er_label_source = _load_expected_return_labels(
        scoring_panel=panel,
        panel_path=panel_path,
        raw_label_panel_path=raw_label_panel_path,
        model_label_col=label_col,
        er_label_col=args.er_label_col,
        allow_normalized_er_label=args.allow_normalized_er_label,
    )

    # Score the entire panel in the same feature space the scorer used at
    # training. The prebuilt panel already has normalized alpha columns but
    # raw fundamental columns; transform_feature_frame(source_space="panel")
    # applies only the columns declared raw-in-panel by the artifact.
    log.info("Scoring %d rows...", len(panel))
    X = transform_feature_frame(panel, feat_cols, art, source_space="panel")
    panel["panel_score"] = booster.predict(xgb.DMatrix(X.values.astype(np.float64)))

    # Sanity check IC vs the model's ranking label.
    from scipy.stats import spearmanr
    valid = panel.dropna(subset=[label_col])
    ics = []
    for _, g in valid.groupby("date"):
        if len(g) < 5: continue
        ic, _ = spearmanr(g["panel_score"], g[label_col])
        if not np.isnan(ic): ics.append(ic)
    log.info("In-sample fwd_60d cross-sectional IC: mean=%+.4f median=%+.4f n_dates=%d",
             np.mean(ics), np.median(ics), len(ics))

    valid_er = panel.dropna(subset=[er_label_col])
    er_ics = []
    for _, g in valid_er.groupby("date"):
        if len(g) < 5:
            continue
        ic, _ = spearmanr(g["panel_score"], g[er_label_col])
        if not np.isnan(ic):
            er_ics.append(ic)
    log.info(
        "In-sample raw-ER cross-sectional IC: mean=%+.4f median=%+.4f n_dates=%d",
        float(np.mean(er_ics)) if er_ics else float("nan"),
        float(np.median(er_ics)) if er_ics else float("nan"),
        len(er_ics),
    )

    # Build the dicts the calibrator wants:
    #   panel_scores   = {ticker: series indexed by date → score}
    #   future_returns = {ticker: series indexed by date → fwd_excess_return}
    log.info("Building per-ticker score + return series for calibrator pool...")
    panel_scores = {}
    future_returns = {}
    for tkr, g in panel.groupby("ticker"):
        gs = g.sort_values("date").set_index("date")
        panel_scores[tkr] = gs["panel_score"]
        # Expected-return calibration feeds Kelly/QP as μ, so it must be raw
        # return units, not the cross-sectional z-scored training label.
        if er_label_col in gs.columns:
            future_returns[tkr] = gs[er_label_col].dropna()

    log.info("Pool: %d tickers with both score + raw 60d-fwd returns",
             len(set(panel_scores) & set(future_returns)))

    # Fit calibrator. Use lookahead_days=60 to MATCH the label horizon,
    # threshold_mode=crosssectional so the base rate is ~50% regardless
    # of the bull-skew on 60-day windows (per global_calibrator.py docs).
    #
    # 2026-05-18 method=platt (was isotonic): isotonic creates wide FLAT
    # regions where the model's negative scores don't reliably predict
    # negative returns. Today's incident: ~57% of x-range [-0.59, +0.07]
    # collapsed to single y=0.478. 79% of today's candidates landed in
    # this region → tied → MCD rebuy via panel_score tie-break.
    # Platt (sigmoid) is strictly monotone, no flat regions. Slightly less
    # adaptive than isotonic but no degenerate ranking failures.
    # See doc/research/2026-05-18-mcd-rebuy-incident.md.
    method = (cli_args.method if hasattr(cli_args, "method") and cli_args.method
              else "platt")
    from training_panel.global_calibrator import fit_global_calibrator
    log.info(
        "Fitting calibrator (method=%s, lookahead=%dd, threshold_mode=crosssectional)",
        method,
        lookahead_days,
    )
    calib = fit_global_calibrator(
        panel_scores, future_returns,
        lookahead_days=lookahead_days,
        threshold=0.0,                # ignored when threshold_mode='crosssectional'
        threshold_mode="crosssectional",
        method=method,
        min_rows=1000,
    )

    # 2026-05-18 ACCEPTANCE GATE: refuse to save if curve has degenerate
    # flat region. Even after switching isotonic → platt, a future
    # regression / data shift could re-introduce flat regions. Catching
    # at fit-time prevents bad artifacts from ever landing in prod.
    # 2026-05-18 user audit: DRY — single source of truth in
    # backtesting/renquant_104/kernel/calibrator_quality.py
    import sys as _sys
    _sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
    from kernel.calibrator_quality import flat_region_stats  # noqa: PLC0415
    flat_frac = flat_region_stats(calib.prob_x, calib.prob_y)["fraction"]
    er_flat = flat_region_stats(calib.er_x, calib.er_y)
    MAX_FLAT_FRAC = 0.30  # ≤ 30% of x-domain may be flat
    if flat_frac > MAX_FLAT_FRAC:
        log.error("ACCEPTANCE-GATE FAIL: calibrator probability curve has "
                  "flat region spanning %.1f%% of x-domain (max allowed %.0f%%). "
                  "Refusing to save artifact. Try a different method or "
                  "investigate model signal quality on negative tail. See "
                  "doc/research/2026-05-18-mcd-rebuy-incident.md.",
                  flat_frac * 100, MAX_FLAT_FRAC * 100)
        sys.exit(2)
    if er_flat["fraction"] > MAX_FLAT_FRAC:
        log.error("ACCEPTANCE-GATE FAIL: calibrator expected_return curve has "
                  "flat region spanning %.1f%% of x-domain (max allowed %.0f%%). "
                  "Refusing to save artifact because Kelly/QP consumes this "
                  "curve as μ; a plateau ties candidate target weights. Use "
                  "the smooth bounded ER head or investigate signal quality.",
                  er_flat["fraction"] * 100, MAX_FLAT_FRAC * 100)
        sys.exit(2)
    log.info("ACCEPTANCE-GATE PASS: probability flat %.1f%%, ER flat %.1f%% ≤ %.0f%% (method=%s)",
             flat_frac * 100, er_flat["fraction"] * 100, MAX_FLAT_FRAC * 100, method)

    # Save through GlobalPanelCalibration.save so the G12 acceptance gate
    # (probability.y in [0,1], expected_return.y within ±20%) cannot be
    # bypassed by this production script.
    log.info("Saving artifact to %s", out_path)
    metadata = dict(calib.metadata)
    # Stamp the source artifact path so we can detect drift later
    metadata["scorer_artifact"] = str(art_path)
    metadata["scorer_artifact_fingerprint"] = fingerprint
    metadata["scorer_oos_mean_ic"] = float(np.mean(ics))
    metadata["scorer_oos_mean_ic_vs_er_label"] = float(np.mean(er_ics)) if er_ics else None
    metadata["model_label_col"] = label_col
    metadata["expected_return_label_col"] = er_label_col
    metadata["expected_return_label_source"] = er_label_source
    metadata["expected_return_label_contract"] = "raw_return_units_required"
    metadata["expected_return_label_diagnostics"] = er_label_diag
    metadata["method"] = method
    # 2026-05-11: record OOS window for future audits.
    if args.data_start:
        metadata["data_window_start"] = args.data_start
    if args.data_end:
        metadata["data_window_end"] = args.data_end
    metadata["lookahead_days_used"] = lookahead_days

    calib.save(out_path, metadata=metadata)
    log.info("Saved: n_unique_prob_y=%d  pool_ic=%+.4f  per_date_ic=%+.4f  base_rate=%.4f",
             metadata["n_unique_prob_y"],
             metadata["pool_ic"],
             metadata["per_date_ic_mean"],
             metadata.get("prob_base_rate", float("nan")))


if __name__ == "__main__":
    main()
