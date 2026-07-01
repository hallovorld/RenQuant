#!/usr/bin/env python
"""Phase D2 — Proper NGBoost training (Duan 2020 large-data config).

Prior NGBoost runs in this codebase were misconfigured ("1h+ didn't
finish" → replaced with XGB-quantile). Per Duan 2020 §4 paragraph on
Year MSD (515,345 samples — similar to our 568k panel):

    "For the Year MSD dataset, being extremely large relative to the
     rest, was fit using a learning rate η of 0.1 ... For the Year MSD
     dataset we use a mini-batch size of 10%, for all other datasets we
     use 100%."

Recommended large-data config (from paper §4):
  - Distribution: Normal (loc, log-scale parameterization)
  - Base learner: DecisionTreeRegressor max_depth=3
  - Score: LogScore (NLL) — default, fastest
  - n_estimators: M chosen by val NLL via early stop
  - learning_rate: 0.1 (large data) vs 0.01 (small)
  - minibatch_frac: 0.1 (large data) vs 1.0 (small)
  - col_sample: 1.0 default

Param/sample math at our scale:
  568,563 train rows × 0.1 minibatch = 56,856 rows per iteration
  Per iter: 2 trees (loc, log-scale) × DecisionTreeRegressor depth=3
  Cost ≈ O(N · log(N) · n_features · p) per tree ≈ feasible

Compare to a caller-supplied same-panel XGB baseline. The old E51
baseline (+0.0294 ± 0.0029) is historical context only and is no longer
accepted as an implicit quality gate.
Hypothesis: NGBoost with proper config + natural gradient should
match or beat XGB-quantile on val_mu_ic, with strictly proper LogScore
(NLL) optimization.

References:
- Duan, Avati, Ding, Thai, Basu, Ng, Schuler 2020. "NGBoost: Natural
  Gradient Boosting for Probabilistic Prediction" ICML 2020.
- ngboost source: github.com/stanfordmlgroup/ngboost

σ-head _rawlabel admission (2026-07): this is the REAL, currently-live
training entrypoint for the σ-head (--panel-path defaults to the raw-label
corpus renquant-orchestrator's RefreshSigmaHeadRawLabelTask keeps in
lockstep with the fund panel; see renquant-orchestrator PR #218). main()
refuses to read the corpus / fit / write an artifact unless it is admissible
— see assert_rawlabel_admissible() below. --allow-unadmitted-rawlabel is a
research-only escape hatch; never set it against the production corpus.
"""
from __future__ import annotations
import argparse
import json, time, sys, logging, hashlib, os
from pathlib import Path
from datetime import datetime
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from ngboost import NGBRegressor
from ngboost.distns import Normal
from ngboost.scores import LogScore
from sklearn.tree import DecisionTreeRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ngb-proper")

REPO = Path(__file__).resolve().parent.parent
LABEL = "fwd_60d_excess_raw"
HORIZON = 60

# ── σ-head _rawlabel admission (consumer-side enforcement) ─────────────────
# This mirrors, by file-contract (not by cross-repo import — RenQuant does
# not depend on renquant-orchestrator), the receipt/provenance schema written
# by renquant-orchestrator's RefreshSigmaHeadRawLabelTask
# (retrain_alpha158_fund.assert_rawlabel_admissible / _write_rawlabel_provenance):
#   <rawlabel>.INVALID.json     — written whenever the lockstep refresh fails,
#                                  produces an empty/rejected build, or the
#                                  upstream fund panel is missing.
#   <rawlabel>.provenance.json  — written only on a fully-validated swap;
#                                  carries horizon + source-panel sha256
#                                  digest/frontier + row/ticker/finite stats.
# Before this guard, a swallowed refresh failure left the corpus silently
# stale (or an INVALID receipt could exist) while this script trained on it
# anyway — the fail-open gap this whole lockstep-refresh mechanism exists to
# close. See renquant-orchestrator PR #218.
RAWLABEL_INVALID_SUFFIX = ".INVALID.json"
RAWLABEL_PROVENANCE_SUFFIX = ".provenance.json"


class RawlabelAdmissionError(RuntimeError):
    """The σ-head ``_rawlabel`` corpus failed downstream admission and MUST
    NOT be consumed by NGBoost training (missing corpus, an active
    invalidation receipt, no provenance stamp, or a provenance digest/horizon
    that no longer matches what is live on disk)."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def assert_rawlabel_admissible(
    rawlabel_path: Path,
    *,
    expected_horizon: int,
    source_panel_path: Path,
) -> dict:
    """Refuse (raise :class:`RawlabelAdmissionError`) unless the σ-head
    ``_rawlabel`` corpus at ``rawlabel_path`` is admissible for training:

    - the corpus file must exist;
    - no active ``<rawlabel>.INVALID.json`` invalidation receipt may sit
      beside it;
    - a ``<rawlabel>.provenance.json`` stamp must exist — an un-provenanced
      corpus (e.g. hand-copied, or predating the lockstep refresh task) was
      never validated and must not be trusted (fail CLOSED, not open);
    - the provenance ``horizon`` must equal ``expected_horizon``;
    - the provenance ``source_panel_sha256`` must equal the sha256 of the
      CURRENT ``source_panel_path`` on disk — this proves the raw-label
      corpus was built from the fund panel that is live right now, catching
      drift even when no invalidation receipt was written (e.g. the panel
      moved forward after the corpus was stamped, without a failure being
      recorded).

    Returns the parsed provenance dict on success.
    """
    if not rawlabel_path.exists():
        raise RawlabelAdmissionError(f"σ-head _rawlabel corpus is missing: {rawlabel_path}")

    receipt = rawlabel_path.with_name(rawlabel_path.name + RAWLABEL_INVALID_SUFFIX)
    if receipt.exists():
        try:
            reason = json.loads(receipt.read_text()).get("reason", "unknown")
        except (OSError, json.JSONDecodeError):
            reason = "unreadable receipt"
        raise RawlabelAdmissionError(
            f"σ-head _rawlabel corpus is INVALIDATED ({receipt.name}: {reason}); "
            "refusing to train until a fresh, validated refresh clears the receipt."
        )

    provenance_path = rawlabel_path.with_name(rawlabel_path.name + RAWLABEL_PROVENANCE_SUFFIX)
    if not provenance_path.exists():
        raise RawlabelAdmissionError(
            f"σ-head _rawlabel corpus has no provenance stamp ({provenance_path.name}); "
            "an un-provenanced corpus was never validated by the lockstep refresh task "
            "and must not be trusted (fail closed)."
        )
    try:
        provenance = json.loads(provenance_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RawlabelAdmissionError(
            f"σ-head _rawlabel provenance is unreadable: {provenance_path}: {exc}"
        ) from exc

    prov_horizon = provenance.get("horizon")
    if prov_horizon != expected_horizon:
        raise RawlabelAdmissionError(
            f"σ-head _rawlabel provenance horizon={prov_horizon!r} != expected "
            f"{expected_horizon}; refusing to train on a horizon-mismatched corpus."
        )

    if not source_panel_path.exists():
        raise RawlabelAdmissionError(
            "current source fund panel is missing, cannot verify the provenance "
            f"digest: {source_panel_path}"
        )
    current_digest = _sha256_file(source_panel_path)
    prov_digest = provenance.get("source_panel_sha256")
    if prov_digest != current_digest:
        raise RawlabelAdmissionError(
            f"σ-head _rawlabel provenance source_panel_sha256={prov_digest!r} does not "
            f"match the CURRENT source panel digest {current_digest!r} ({source_panel_path}); "
            "the raw-label corpus was built from a DIFFERENT panel than the one live now — "
            "it is stale even though no invalidation receipt is present. Refusing to train."
        )
    return provenance


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    return float(raw)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the experimental NGBoost head on the 104 panel. "
            "Refuses artifact save unless a current XGB baseline is supplied."
        )
    )
    parser.add_argument(
        "--panel-path",
        type=Path,
        default=REPO / "data" / "alpha158_291_fundamental_dataset_rawlabel.parquet",
        help="Raw-label panel parquet used for NGBoost residual training.",
    )
    parser.add_argument(
        "--panel-artifact",
        type=Path,
        default=REPO / "backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json",
        help="Panel-LTR artifact whose feature_cols define the NGBoost input contract.",
    )
    parser.add_argument(
        "--source-panel-path",
        type=Path,
        default=REPO / "data" / "alpha158_291_fundamental_dataset.parquet",
        help=(
            "Fund panel the --panel-path raw-label corpus was derived from. Used "
            "only to verify the corpus's provenance digest still matches what is "
            "live on disk (see assert_rawlabel_admissible)."
        ),
    )
    parser.add_argument(
        "--allow-unadmitted-rawlabel",
        action="store_true",
        help=(
            "Bypass the σ-head _rawlabel admission check (missing corpus, active "
            "invalidation receipt, missing/mismatched provenance). RESEARCH ONLY — "
            "never set this against the production raw-label corpus."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO / "backtesting/renquant_104/artifacts/sim/ngboost-head.json",
        help="Destination for the best-seed NGBoost artifact.",
    )
    parser.add_argument(
        "--seeds",
        default=os.environ.get("NGB_SEEDS", "42,7,123,2024,31415"),
        help="Comma-separated random seeds. Default is 5 seeds; single seed is exploratory only.",
    )
    parser.add_argument(
        "--missing-feature-policy",
        choices=("error", "zero"),
        default="error",
        help=(
            "How to handle feature_cols present in the artifact but absent from the panel. "
            "'error' is production-safe; 'zero' is only for controlled exploratory runs."
        ),
    )
    parser.add_argument(
        "--xgb-baseline-mean",
        type=float,
        default=_env_float("XGB_BASELINE_MEAN"),
        help="Current same-panel XGB baseline mean IC. Env fallback: XGB_BASELINE_MEAN.",
    )
    parser.add_argument(
        "--xgb-baseline-std",
        type=float,
        default=_env_float("XGB_BASELINE_STD"),
        help="Current same-panel XGB baseline IC std. Env fallback: XGB_BASELINE_STD.",
    )
    parser.add_argument(
        "--allow-save-without-baseline",
        action="store_true",
        help="Allow artifact save without a current XGB baseline gate. Research only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs, split, and feature matrix contract without fitting NGBoost.",
    )
    return parser.parse_args(argv)


def cs_ic(mu, y, dates):
    df = pd.DataFrame({"p": mu, "y": y, "d": dates})
    ics = [spearmanr(g["p"], g["y"])[0] for _, g in df.groupby("d") if len(g) >= 5]
    ics = [x for x in ics if not np.isnan(x)]
    return float(np.mean(ics)) if ics else float("nan")


def _seed_list(seed_csv: str) -> list[int]:
    seeds = [int(s) for s in seed_csv.split(",") if s.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def _apply_missing_feature_policy(
    panel: pd.DataFrame,
    feat_cols: list[str],
    *,
    policy: str,
) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in feat_cols if c not in panel.columns]
    if not missing:
        return panel, []
    sample = ", ".join(missing[:10])
    suffix = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
    if policy == "error":
        raise ValueError(
            f"Panel is missing {len(missing)} feature column(s) required by "
            f"the panel artifact: {sample}{suffix}. Regenerate the raw-label "
            "panel from the same feature pipeline, or rerun with "
            "--missing-feature-policy zero for an explicitly exploratory run."
        )
    panel = panel.copy()
    for col in missing:
        panel[col] = 0.0
    return panel, missing


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    panel_path = args.panel_path
    art_panel = args.panel_artifact

    # ── Admission gate — MUST run before any read of panel_path. A missing
    # corpus, an active invalidation receipt, a missing provenance stamp, or a
    # provenance digest/horizon mismatch refuses training here: no panel read,
    # no NGBoost fit, no artifact write. See assert_rawlabel_admissible above.
    if not args.allow_unadmitted_rawlabel:
        try:
            provenance = assert_rawlabel_admissible(
                panel_path,
                expected_horizon=HORIZON,
                source_panel_path=args.source_panel_path,
            )
        except RawlabelAdmissionError as exc:
            log.error("σ-head _rawlabel ADMISSION REFUSED: %s", exc)
            return 3
        log.info(
            "σ-head _rawlabel admitted: rows=%s tickers=%s finite_fraction=%s "
            "frontier=%s source_panel_sha256=%s",
            provenance.get("n_rows"), provenance.get("n_tickers"),
            provenance.get("finite_fraction"), provenance.get("source_panel_frontier"),
            provenance.get("source_panel_sha256"),
        )
    else:
        log.warning(
            "σ-head _rawlabel admission check BYPASSED (--allow-unadmitted-rawlabel); "
            "research only — never against the production corpus."
        )

    panel_meta = json.loads(art_panel.read_text())
    feat_cols = list(panel_meta["feature_cols"])

    panel = pd.read_parquet(panel_path)
    try:
        panel, missing_cols = _apply_missing_feature_policy(
            panel, feat_cols, policy=args.missing_feature_policy,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    if missing_cols:
        log.warning(
            "Missing-feature policy '%s' filled %d column(s) with 0.0: %s",
            args.missing_feature_policy,
            len(missing_cols),
            ", ".join(missing_cols[:10]) + (" ..." if len(missing_cols) > 10 else ""),
        )
    panel["date"] = pd.to_datetime(panel["date"])
    panel = panel.dropna(subset=[LABEL])
    distinct_dates = sorted(panel["date"].unique())
    val_cut_idx = int(len(distinct_dates) * 0.8)
    val_cut = distinct_dates[val_cut_idx]
    # Apply purge — drop training rows whose forward window overlaps val
    train_cut_idx = max(0, val_cut_idx - HORIZON)
    train_cut = distinct_dates[train_cut_idx]
    train = panel[panel["date"] <= train_cut].copy()
    val   = panel[panel["date"] >  val_cut].copy()
    log.info("Train PURGED: %d rows (≤ %s) | Val: %d rows (> %s)",
             len(train), train_cut.date(), len(val), val_cut.date())

    Xtr = train[feat_cols].fillna(0).values.astype(np.float64)   # ngboost wants float64
    Xva = val[feat_cols].fillna(0).values.astype(np.float64)
    ytr = train[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    yva = val[LABEL].clip(-0.5, 0.5).values.astype(np.float64)
    val_dates = val["date"].values
    log.info(
        "Feature contract: %d columns, missing_filled=%d, Xtr=%s, Xva=%s",
        len(feat_cols), len(missing_cols), Xtr.shape, Xva.shape,
    )
    if args.dry_run:
        log.info("DRY RUN OK — no NGBoost fit or artifact write performed.")
        return 0

    # Paper-recommended large-data config (§4 Year MSD)
    # n_estimators=500 with early_stopping_rounds for safety
    base_learner = DecisionTreeRegressor(
        criterion="friedman_mse",
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_depth=3,
        splitter="best",
    )
    log.info("Config: Normal dist, LogScore, max_depth=3, lr=0.1, minibatch_frac=0.1, n_est=500")

    # 5-seed A/A per CLAUDE.md §5.2 — single-seed +0.0356 was promising;
    # need σ characterization to claim significance vs a current XGB baseline.
    # 2026-05-17: keep all 5 models in memory + pick the best by val_IC and
    # save its ensemble to the sim artifact path. Quality gate prevents the
    # silent-degrade incident (today's Sunday sweep saved val_IC=-0.0165
    # straight to prod with no gate).
    #
    XGB_BASELINE_MEAN = args.xgb_baseline_mean
    XGB_BASELINE_STD = args.xgb_baseline_std
    if (XGB_BASELINE_MEAN is None) ^ (XGB_BASELINE_STD is None):
        log.error("Supply both --xgb-baseline-mean and --xgb-baseline-std, or neither.")
        return 2
    if XGB_BASELINE_MEAN is None:
        log.warning(
            "No current XGB baseline supplied; t-stat and quality gate are disabled. "
            "Artifact save will be refused unless --allow-save-without-baseline is set."
        )
    else:
        log.info("XGB baseline (same-panel required): mean=%.4f std=%.4f",
                 XGB_BASELINE_MEAN, XGB_BASELINE_STD)

    # Params dict — used in artifact metadata so downstream tools know
    # exactly how the model was fitted. Mirrors NGBRegressor() kwargs below.
    params = dict(
        Dist="Normal", Score="LogScore",
        n_estimators=500, learning_rate=0.1, minibatch_frac=0.1,
        col_sample=1.0, natural_gradient=True,
        early_stopping_rounds=20, validation_fraction=0.1,
        base_max_depth=3,
    )
    val_ics = []
    sigma_calibs = []
    mu_xs_stds = []
    fit_times = []
    best_iters = []
    models = []
    # 2026-05-17: default flipped from single seed to 5-seed. Single-seed
    # runs are exploratory only; production claims need the full variance.
    try:
        SEED_LIST = _seed_list(args.seeds)
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    log.info("Running %d seed(s): %s", len(SEED_LIST), SEED_LIST)
    for SEED in SEED_LIST:
        log.info("Fitting NGBoost (seed=%d)...", SEED)
        t0 = time.time()
        model = NGBRegressor(
            Dist=Normal,
            Score=LogScore,
            Base=DecisionTreeRegressor(
                criterion="friedman_mse",
                max_depth=3,
                splitter="best",
            ),
            natural_gradient=True,
            n_estimators=500,
            learning_rate=0.1,
            minibatch_frac=0.1,
            col_sample=1.0,
            verbose=False,        # quiet for 5-seed loop
            random_state=SEED,
            validation_fraction=0.1,
            early_stopping_rounds=20,
        )
        model.fit(Xtr, ytr, X_val=Xva, Y_val=yva)
        ft = time.time() - t0

        dist = model.pred_dist(Xva)
        mu_va = dist.loc
        sigma_va = dist.scale
        v_ic = cs_ic(mu_va, yva, val_dates)
        sc = float(spearmanr(sigma_va, np.abs(yva - mu_va))[0])
        ms = float(pd.DataFrame({"mu": mu_va, "d": val_dates}).groupby("d")["mu"].std().mean())
        bi = model.best_val_loss_itr or model.n_estimators
        log.info("  seed=%-5d val_ic=%+.4f σ-calib=%+.3f μ_xs_std=%.5f best_iter=%d (%.1fs)",
                 SEED, v_ic, sc, ms, bi, ft)
        val_ics.append(v_ic); sigma_calibs.append(sc); mu_xs_stds.append(ms)
        fit_times.append(ft); best_iters.append(bi); models.append((SEED, model))

    log.info("=" * 60)
    log.info("NGBoost-proper %d-seed result (Duan 2020 §4 large-data config)",
             len(SEED_LIST))
    log.info("=" * 60)
    if len(val_ics) > 1:
        log.info("  val μ-IC mean=%+.4f std=%.4f range=[%+.4f, %+.4f]",
                 np.mean(val_ics), np.std(val_ics, ddof=1), min(val_ics), max(val_ics))
        log.info("  σ̂ calib mean=%+.3f", np.mean(sigma_calibs))
    else:
        log.info("  val μ-IC = %+.4f (single seed)", val_ics[0])
        log.info("  σ̂ calib = %+.3f", sigma_calibs[0])
    log.info("  μ̂ x-sec std mean=%.5f", np.mean(mu_xs_stds))
    log.info("  fit time mean=%.1fs total=%.0fs", np.mean(fit_times), sum(fit_times))
    log.info("")
    n_seeds = len(val_ics)
    if XGB_BASELINE_MEAN is not None:
        log.info("Compare baseline XGB-quantile: mean=+%.4f std=%.4f",
                 XGB_BASELINE_MEAN, XGB_BASELINE_STD)
        delta = np.mean(val_ics) - XGB_BASELINE_MEAN
        if n_seeds > 1:
            se = np.sqrt(XGB_BASELINE_STD**2/5 + np.std(val_ics, ddof=1)**2/n_seeds)
        else:
            # Single-seed: just use XGB baseline std as the noise floor (rough)
            se = XGB_BASELINE_STD
        t = delta / se if se > 0 else float("inf")
        log.info("Δ(NGB-proper - XGB) = %+.4f  t-stat = %+.2f", delta, t)
        if abs(t) > 2.0 and delta > 0:
            log.info("SIGNIFICANT BEAT — NGBoost-proper > XGB-quantile at 95%%")
        elif delta > 0:
            log.info("Trend positive but not 2σ significant on n=%d", n_seeds)
        else:
            log.info("NGBoost-proper does NOT beat XGB-quantile")
    else:
        log.info("No baseline comparison computed.")
    if n_seeds == 1:
        log.info("[reference: 5/15 full 5-seed validation = +0.0360 ± 0.0036, "
                 "t=+2.76 vs XGB baseline; logged in CLAUDE.md status]")

    # ── Save best-by-val_IC artifact to sim path (NOT prod) ──────────────
    log.info("")
    log.info("=" * 60)
    log.info("Saving best-seed artifact")
    log.info("=" * 60)
    best_idx = int(np.argmax(val_ics))
    best_seed, best_model = models[best_idx]
    best_val_ic = val_ics[best_idx]
    best_sigma_calib = sigma_calibs[best_idx]
    best_mu_xs_std = mu_xs_stds[best_idx]
    best_iter = best_iters[best_idx]
    log.info("Best seed = %d  val_ic=%+.4f  σ-calib=%+.3f  best_iter=%d",
             best_seed, best_val_ic, best_sigma_calib, best_iter)

    # Quality gate — refuse save if even the BEST seed doesn't beat XGB baseline.
    # This is the safety mechanism missing from Sunday sweep (today's 11:20 incident).
    if XGB_BASELINE_MEAN is None and not args.allow_save_without_baseline:
        log.warning(
            "QUALITY GATE UNAVAILABLE — no current same-panel XGB baseline supplied. "
            "Refusing to save artifact. Pass --xgb-baseline-mean/--xgb-baseline-std "
            "or explicitly add --allow-save-without-baseline for research."
        )
        return 1
    if XGB_BASELINE_MEAN is not None and best_val_ic < XGB_BASELINE_MEAN:
        log.warning(
            "QUALITY GATE FAILED — best val_IC=%+.4f < XGB baseline %+.4f. "
            "Refusing to save artifact (would silently degrade prod). "
            "Best-seed model NOT saved.",
            best_val_ic, XGB_BASELINE_MEAN,
        )
        return 1

    # Pickle the best model + meta in the same schema as
    # train_ngboost_alpha158_fund.py so downstream consumers (NGBoostFitTask
    # at inference time) read it identically.
    import base64, pickle
    blob = base64.b64encode(pickle.dumps(best_model)).decode("ascii")
    medians = np.nanmedian(Xtr, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)

    pred_tr = best_model.pred_dist(Xtr)
    mu_tr, sd_tr = pred_tr.loc, pred_tr.scale
    pred_va = best_model.pred_dist(Xva)
    mu_va, sd_va = pred_va.loc, pred_va.scale

    fp_fields = {
        "feature_cols": feat_cols,
        "params": params,
        "label_col": LABEL,
        "panel_artifact_fingerprint": panel_meta.get("config_fingerprint",
                                                     "unknown"),
        "seed": best_seed,
        "all_seeds_val_ic": val_ics,
    }
    fp = hashlib.sha256(json.dumps(fp_fields, sort_keys=True, default=str)
                        .encode()).hexdigest()[:16]

    artifact = {
        "version": 1,
        "kind":    "ngboost_head",
        "trained_date": str(datetime.utcnow().date()),
        "feature_cols": feat_cols,
        "params": {**params, "seed": best_seed, "best_iter": int(best_iter)},
        "regressor_pickle_b64": blob,
        "feature_medians": medians.tolist(),
        "train_run_id": f"proper_5seed_{datetime.utcnow().strftime('%Y%m%dT%H%M')}",
        "training_notes": (
            f"NGBoost-proper 5-seed training (Duan 2020 §4 large-data config). "
            f"Selected best-by-val_IC seed={best_seed} from {len(SEED_LIST)} seeds. "
            f"All-seed val_IC: {[round(v,4) for v in val_ics]}. "
            f"Best val_IC={best_val_ic:+.4f}, σ-calib={best_sigma_calib:+.3f}, "
            f"μ_xs_std={best_mu_xs_std:.5f}. "
            + (
                f"XGB-quantile baseline mean=+{XGB_BASELINE_MEAN:.4f}±{XGB_BASELINE_STD:.4f}. "
                "Quality gate: val_IC > XGB baseline (passed). "
                if XGB_BASELINE_MEAN is not None
                else "Quality gate: bypassed by --allow-save-without-baseline. "
            )
            +
            f"Panel fingerprint={fp_fields['panel_artifact_fingerprint']}."
        ),
        "train_mu_mean":    float(mu_tr.mean()),
        "train_sigma_mean": float(sd_tr.mean()),
        "train_mu_ic":      cs_ic(mu_tr, ytr, train["date"].values),
        "val_mu_ic":        best_val_ic,
        "val_sigma_calib":  best_sigma_calib,
        "val_mu_xs_std":    best_mu_xs_std,
        "best_iter":        int(best_iter),
        "n_rows":           int(len(panel)),
        "n_rows_train":     int(len(train)),
        "n_rows_val":       int(len(val)),
        "all_seeds_val_ic": val_ics,
        "all_seeds_sigma_calib": sigma_calibs,
        "config_fingerprint":        f"sha256:{fp}",
        "config_fingerprint_fields": fp_fields,
    }
    out_path = args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact))
    log.info("✓ Saved → %s  (size=%.1f MB)", out_path,
             out_path.stat().st_size / 1e6)
    log.info("Fingerprint: sha256:%s", fp)
    log.info("")
    log.info("PROMOTION (manual, after rollback rehearsal per CLAUDE.md §5.5):")
    log.info("  cp -v backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json \\")
    log.info("        backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json.bak_$(date +%%Y%%m%%d)")
    log.info("  cp -v %s \\", out_path)
    log.info("        backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json")
    log.info("σ wire activation (real-$ change) still gated on user authorization.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
