"""Model inference for all four artifact types + score calibration.

Self-contained: only numpy, json, math.  No common/ imports.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.models")

# Stable abstain reasons (ScoreResult.abstain_reason) and the blocked_by
# reasons they map to (``er_abstain_<reason>``, telemetry / decision trace).
# A per-ticker model that has NO opinion must not score 0.0 — the isotonic
# ER map turns 0.0 into a positive expected return (x12 at the 60d horizon).
#
# * ``unseen_state``  — Q-learning: the resolved Q-row was never updated in
#   training (all-zero) or the state index is outside the table.
# * ``nan_features``  — any per-ticker model type (classification, xgboost,
#   qlearning): a required feature column is missing / None / NaN, so there
#   is no input to score. Pre-2026-08-30 ``predict_classification`` and the
#   xgboost branch of ``score_artifact`` returned ``raw = 0.0`` here — the
#   same trap pipeline#303 closed for qlearning.
ABSTAIN_UNSEEN_STATE = "unseen_state"
ABSTAIN_NAN_FEATURES = "nan_features"
REASON_ER_ABSTAIN_PREFIX = "er_abstain_"
REASON_ER_ABSTAIN_UNSEEN_STATE = REASON_ER_ABSTAIN_PREFIX + ABSTAIN_UNSEEN_STATE
REASON_ER_ABSTAIN_NAN_FEATURES = REASON_ER_ABSTAIN_PREFIX + ABSTAIN_NAN_FEATURES
ABSTAIN_REASONS = (ABSTAIN_UNSEEN_STATE, ABSTAIN_NAN_FEATURES)


def abstain_block_reason(abstain_reason: str) -> str:
    """``ScoreResult.abstain_reason`` → the candidate ``blocked_by`` reason."""
    return REASON_ER_ABSTAIN_PREFIX + str(abstain_reason)


def is_abstain_block_reason(blocked_by: object) -> bool:
    return isinstance(blocked_by, str) and blocked_by.startswith(REASON_ER_ABSTAIN_PREFIX)


def abstain_breakdown(blocked_by_values) -> dict[str, int]:
    """Count abstain ``blocked_by`` reasons: ``{er_abstain_<reason>: n}``.

    Every known reason is present (0 when absent) so the Phase 2b summary
    line and the run counters have a stable shape run to run.
    """
    out = {abstain_block_reason(r): 0 for r in ABSTAIN_REASONS}
    for b in blocked_by_values:
        if is_abstain_block_reason(b):
            out[b] = out.get(b, 0) + 1
    return out


# ── Calibration ───────────────────────────────────────────────────────────────

def calibrate_score(raw_score: float, calibration: dict | None) -> float:
    """Map a raw model score to a calibrated rank_score ∈ [0, 1].

    Audit fix M-1 (Round 6, 2026-04-25): pre-fix, NaN/inf raw_score
    leaked through every method:
      - identity:          returned NaN directly
      - isotonic:          np.interp(NaN, xs, ys) = NaN → clip(NaN) = NaN
      - platt:             coef*NaN + intercept = NaN → exp(-NaN) = NaN
        → 1/(1+NaN) = NaN → clip(NaN) = NaN
    Result: NaN rank_score propagated into tier-threshold gates (NaN <
    0.10 = False → silent skip, OK), into ranking `sorted(..., key=...)`
    (raises if mixed with floats), into ranking blend weight ops (NaN
    spreading), etc.

    Post-fix: explicit non-finite check at the top, fall back to
    base_rate (default 0.0). All downstream code can rely on rank_score
    ∈ [0, 1] always being finite.
    """
    if raw_score is None or not math.isfinite(float(raw_score)):
        # Non-finite input: return base_rate if calibration carries one,
        # else 0.0 (treated as "low conviction" by tier gates).
        if calibration:
            return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))
        return 0.0
    if not calibration:
        return float(raw_score)
    method = calibration.get("method", "identity")
    if method == "identity":
        return float(raw_score)
    if method == "constant_probability":
        return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))
    if method == "isotonic":
        x_thresh = calibration.get("x_thresholds") or []
        y_thresh = calibration.get("y_thresholds") or []
        if not x_thresh or not y_thresh:
            return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))
        return float(np.clip(np.interp(raw_score, x_thresh, y_thresh), 0.0, 1.0))
    if method == "platt":
        coef      = calibration.get("platt_coef")
        intercept = calibration.get("platt_intercept")
        if coef is None or intercept is None:
            return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))
        scale_std = calibration.get("platt_scale_std")
        scale_mean = calibration.get("platt_scale_mean")
        if (scale_std is None
                or not math.isfinite(float(scale_std))
                or float(scale_std) <= 0
                or scale_mean is None
                or not math.isfinite(float(scale_mean))):
            return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))
        scaled = (raw_score - float(scale_mean)) / float(scale_std)
        log_odds  = coef * scaled + intercept
        return float(np.clip(1.0 / (1.0 + math.exp(-log_odds)), 0.0, 1.0))
    return float(np.clip(calibration.get("base_rate", 0.0), 0.0, 1.0))


def expected_return_from_calibration(
    raw_score: float | None,
    calibration: dict | None,
    *,
    horizon_days: int | None = None,
) -> float | None:
    """E[stock_return - SPY_return] in fraction units over `horizon_days`.

    Uses the er_* fields of the calibration block (written by
    training.scoring.fit_expected_return_calibration).  Returns 0.0 when no
    expected-return calibration is available — rotation gracefully degrades
    to "no swap" rather than mis-ranking on stale probability scores.

    Returns ``None`` when ``raw_score`` is ``None`` / non-finite: the model
    ABSTAINED (no state to score), so there is no expected return. The
    calibrator is never evaluated on an absent score — pre-2026-08-30 an
    unseen Q-state scored ``raw=0.0`` and the isotonic ``er_y(0.0)`` mapped
    that to a POSITIVE ER (×12 at the 60d horizon), which bought 3 of 11
    live names in 2026-08-18..28 on a model that had never seen the state.
    """
    if raw_score is None or not math.isfinite(float(raw_score)):
        return None
    if not calibration:
        return 0.0
    er_method   = calibration.get("er_method", "none")
    er_lookahead = int(calibration.get("er_lookahead", 5))

    if er_method == "isotonic":
        xs = calibration.get("er_x_thresholds") or []
        ys = calibration.get("er_y_thresholds") or []
        if not xs or not ys:
            base = 0.0
        else:
            base = float(np.interp(raw_score, xs, ys))
    elif er_method == "linear":
        coef      = calibration.get("er_coef")
        intercept = calibration.get("er_intercept", 0.0) or 0.0
        if coef is None:
            base = 0.0
        else:
            base = float(coef * raw_score + intercept)
    elif er_method == "constant":
        base = float(calibration.get("er_constant", 0.0) or 0.0)
    else:
        base = 0.0

    if horizon_days is None or horizon_days == er_lookahead or er_lookahead <= 0:
        return base
    return base * (horizon_days / er_lookahead)


# ── Model type inference ──────────────────────────────────────────────────────

def _traverse_tree(tree: list, row: list) -> float:
    idx = 0
    while True:
        feat, split_val, left_off, right_off = tree[idx]
        if feat == -1:
            return split_val
        idx += int(left_off) if row[int(feat)] <= split_val else int(right_off)


def missing_features(artifact: dict, feature_row: pd.Series) -> list[str]:
    """Required ``feature_columns`` that are absent / None / NaN in the row.

    Non-empty means the per-ticker model has no input to score and must
    ABSTAIN (``ABSTAIN_NAN_FEATURES``) — for every model type. A value that
    cannot be coerced to float counts as missing too.
    """
    missing: list[str] = []
    for col in artifact.get("feature_columns", []) or []:
        val = feature_row.get(col) if feature_row is not None else None
        if val is None:
            missing.append(col)
            continue
        try:
            if math.isnan(float(val)):
                missing.append(col)
        except (TypeError, ValueError):
            missing.append(col)
    return missing


def predict_classification(artifact: dict, feature_row: pd.Series) -> float | None:
    """Mean vote of the random-forest trees, or ``None`` to ABSTAIN.

    ``None`` (2026-08-30, follow-up to pipeline#303) on any missing / NaN
    feature: the forest cannot be traversed, so there is no vote. Pre-fix
    this returned ``0.0`` "neutral" — but the isotonic ER calibrator maps
    0.0 to a positive expected return on live artifacts (the qlearning trap
    of #303, same shape). A REAL vote that happens to land exactly on the
    calibrator's neutral value is still a vote and is returned unchanged:
    only absent input abstains, never a value the trees produced.
    """
    feat_cols = artifact["feature_columns"]
    if missing_features(artifact, feature_row):
        return None
    feat_vals = [float(feature_row.get(c)) for c in feat_cols]
    trees = artifact["trees"]
    return sum(_traverse_tree(t, feat_vals) for t in trees) / len(trees)


def predict_qlearning(
    artifact: dict, feature_row: pd.Series, holdings: int = 0,
) -> float | None:
    """Q(buy) − Q(sell) for the row's discretised state, or ``None`` to ABSTAIN.

    ``None`` (2026-08-30, orch forensic on pinned afb73626) means the model
    has no opinion and MUST NOT be scored:

    * a NaN / missing feature — no state can be resolved. Pre-fix (Issue 38,
      2026-05-04) this returned ``0.0`` "neutral"; but 0.0 is not neutral
      once the isotonic ER calibrator maps it (``er_y(0.0) > 0`` on live
      artifacts) and the rotation horizon multiplies it ×12.
    * an UNSEEN state — the resolved Q-row is all-zero, i.e. the tabular
      learner never updated it in training (99.5% of the 300k rows of a live
      artifact are such rows). ``Q(buy) − Q(sell) = 0.0`` there is an
      initialisation constant, not a forecast. Live 2026-08-28: 19/87
      candidates scored ``raw == 0.0``, 14 of them with ER > 0.
    * a state index outside the table (artifact / feature mismatch).

    The tabular Q-learner has no visit counts in its artifact, so the
    all-zero row IS the unseen-state signal; a visited row cannot be
    all-zero after a single update with a non-zero reward.
    """
    feat_cols  = artifact["feature_columns"]
    bin_edges  = {col: np.array(edges) for col, edges in artifact["bin_edges"].items()}
    n_bins     = int(artifact.get("n_bins", 5))
    q_table    = np.array(artifact["q_table"])

    # 2026-05-04 audit Issue 38: NaN feature value silently routed to the
    # LAST bin via `np.digitize(NaN, edges)` (returns `len(edges)+1`, then
    # `np.clip(... - 1, 0, n_bins-1)` pinned to the top bin) — a missing
    # feature looked like an extreme-value signal. 2026-08-30: the Issue-38
    # answer (return 0.0) is replaced by ABSTAIN — see the docstring.
    for col in feat_cols:
        val = feature_row.get(col)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return None

    state = 0
    for col in feat_cols:
        val     = float(feature_row.get(col, 0))
        bin_idx = int(np.clip(np.digitize(val, bin_edges[col]) - 1, 0, n_bins - 1))
        state   = state * n_bins + bin_idx
    holding_bucket = 2 if holdings > 0 else (0 if holdings < 0 else 1)
    state = state * 3 + holding_bucket
    if state < 0 or state >= len(q_table):
        return None
    q_vals = np.asarray(q_table[state], dtype=float)
    if q_vals.size == 0 or not np.any(q_vals != 0.0):
        return None                        # unseen state: never updated
    return float(q_vals[0] - q_vals[1])   # Q(buy) - Q(sell)


def predict_manual(artifact: dict, feature_row: pd.Series) -> float:
    score = 0
    for rule in artifact.get("score_rules", []):
        val = feature_row.get(rule["col"])
        if val is None or (isinstance(val, float) and math.isnan(val)):
            continue
        if "buy_below"  in rule and rule["buy_below"]  is not None and val < rule["buy_below"]:
            score += 1
        if "buy_above"  in rule and rule["buy_above"]  is not None and val > rule["buy_above"]:
            score += 1
        if "sell_above" in rule and rule["sell_above"] is not None and val > rule["sell_above"]:
            score -= 1
        if "sell_below" in rule and rule["sell_below"] is not None and val < rule["sell_below"]:
            score -= 1
    return float(score)


def predict_xgboost(artifact: dict, feat_vals: list[float]) -> float:
    """Pure-Python XGBoost inference (binary:logistic). Returns P ∈ [0, 1].

    Audit fix M-4 (Round 6, 2026-04-25): pre-fix, the loop used
    `val <= sc[node]` for ALL values including NaN — but `NaN <= x`
    is False in Python, so NaN inputs always went to the RIGHT child
    deterministically. XGBoost's actual semantics: each split has a
    `default_left` flag persisted in the JSON model that says which
    direction to take on missing values (auto-learned during training).
    Pre-fix inference therefore diverged from training behavior on any
    NaN input — explains some of the train/inference parity issues we
    saw in earlier audits.

    Post-fix: when val is NaN (or feature index out of range), route to
    the side indicated by `default_left[node]`. Falls back to old
    behaviour for trees missing the `default_left` field (older
    artifacts).
    """
    trees = artifact["learner"]["gradient_booster"]["model"]["trees"]
    total = 0.0
    for tree in trees:
        lc   = tree["left_children"]
        rc   = tree["right_children"]
        sc   = tree["split_conditions"]
        si   = tree["split_indices"]
        bw   = tree["base_weights"]
        dl   = tree.get("default_left")  # may be None on legacy artifacts
        node = 0
        while lc[node] != -1:
            fi  = si[node]
            if fi >= len(feat_vals):
                # Missing feature — use default_left if available.
                go_left = bool(dl[node]) if dl is not None else False
                node = lc[node] if go_left else rc[node]
                continue
            val = feat_vals[fi]
            if val is None or (isinstance(val, float) and math.isnan(val)):
                go_left = bool(dl[node]) if dl is not None else False
                node = lc[node] if go_left else rc[node]
            else:
                node = lc[node] if val <= sc[node] else rc[node]
        total += bw[node]
    return 1.0 / (1.0 + math.exp(-total))


# ── Artifact loading ──────────────────────────────────────────────────────────

def load_artifact(model_dir: Path, ticker: str) -> dict | None:
    """Load policy-metadata and model weights for *ticker*.

    Returns a unified dict with keys: policy_type, feature_columns, buy_threshold,
    sell_threshold, score_calibration, and type-specific weight keys.
    Returns None if any required file is missing.
    """
    meta_path = model_dir / f"{ticker}-policy-metadata.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    ptype        = meta["policy_type"]
    feat_cols    = meta.get("feature_columns", [])
    buy_thresh   = meta.get("buy_threshold",  0.1)
    sell_thresh  = meta.get("sell_threshold", -0.1)
    calibration  = meta.get("score_calibration")

    artifact: dict = {
        "policy_type":     ptype,
        "feature_columns": feat_cols,
        "buy_threshold":   buy_thresh,
        "sell_threshold":  sell_thresh,
        "score_calibration": calibration,
        "_metadata":       meta,
    }

    if ptype == "classification":
        p = model_dir / f"{ticker}-rf-trees.json"
        if not p.exists():
            return None
        with open(p) as f:
            artifact["trees"] = json.load(f)

    elif ptype == "manual":
        p = model_dir / f"{ticker}-manual-rules.json"
        if not p.exists():
            return None
        with open(p) as f:
            d = json.load(f)
        artifact["score_rules"]   = d["score_rules"]
        artifact["buy_threshold"] = d["buy_threshold"]
        artifact["sell_threshold"] = d["sell_threshold"]

    elif ptype == "qlearning":
        qp = model_dir / f"{ticker}-qtable.json"
        ep = model_dir / f"{ticker}-bin-edges.json"
        if not qp.exists() or not ep.exists():
            return None
        with open(qp) as f:
            artifact["q_table"] = json.load(f)
        with open(ep) as f:
            artifact["bin_edges"] = json.load(f)
        artifact["n_bins"] = meta.get("n_bins", 5)

    elif ptype == "xgboost":
        arts     = meta.get("artifacts", {})
        buy_path  = model_dir / arts.get("buy_model",  f"{ticker}-xgb-buy.json")
        sell_path = model_dir / arts.get("sell_model", f"{ticker}-xgb-sell.json")
        if not buy_path.exists() or not sell_path.exists():
            return None
        with open(buy_path) as f:
            artifact["xgb_buy"] = json.load(f)
        with open(sell_path) as f:
            artifact["xgb_sell"] = json.load(f)
        bt = meta.get("buy_threshold", 0.1)
        artifact["buy_threshold"]  = bt
        artifact["sell_threshold"] = -bt
    else:
        return None

    return artifact


# ── Score + action ────────────────────────────────────────────────────────────

@dataclass
class ScoreResult:
    raw_score:       float | None
    rank_score:      float | None
    signal:          str   # "buy" | "hold" | "sell" | "abstain"
    expected_return: float | None = 0.0   # E[R - SPY] over `er_lookahead` days
    # Non-None iff the model ABSTAINED (signal == "abstain"): raw_score,
    # rank_score and expected_return are all None — there is no number to
    # gate, rank, calibrate or count a strike on.
    abstain_reason:  str | None = None

    @property
    def abstained(self) -> bool:
        return self.abstain_reason is not None


def abstain_result(reason: str = ABSTAIN_UNSEEN_STATE) -> ScoreResult:
    """The one shape every "model has no opinion" outcome takes."""
    return ScoreResult(
        raw_score=None, rank_score=None, signal="abstain",
        expected_return=None, abstain_reason=reason,
    )


def score_artifact(
    artifact: dict,
    feature_row: pd.Series,
    holdings: int = 0,
    *,
    horizon_days: int | None = None,
) -> ScoreResult:
    """Compute raw score, calibrated rank_score, expected_return, and signal.

    `horizon_days` overrides the calibration's native lookahead for the
    expected_return field — pass the rotation target horizon (e.g. 20) to
    get E[R - SPY] over that period rather than the 5-day calibration window.
    """
    ptype = artifact["policy_type"]

    # 2026-08-30 (follow-up to pipeline#303): a missing / NaN required
    # feature is the absence of an input, for every learned model type.
    # ABSTAIN before dispatch — never the pre-fix ``raw = 0.0`` that the
    # isotonic ER map turned into a positive expected return. ``manual``
    # is rule-based and skips NaN rules by contract (predict_manual); it is
    # not a learned model and is left alone.
    if ptype in ("classification", "qlearning", "xgboost"):
        if missing_features(artifact, feature_row):
            return abstain_result(ABSTAIN_NAN_FEATURES)

    if ptype == "classification":
        raw = predict_classification(artifact, feature_row)
        if raw is None:                    # defensive: same contract as above
            return abstain_result(ABSTAIN_NAN_FEATURES)

    elif ptype == "manual":
        raw = predict_manual(artifact, feature_row)

    elif ptype == "qlearning":
        raw = predict_qlearning(artifact, feature_row, holdings=holdings)
        if raw is None:
            # Unseen / unresolvable Q-state: ABSTAIN. No calibration, no
            # ER, no signal — never the pre-fix 0.0 → positive-ER path.
            return abstain_result(ABSTAIN_UNSEEN_STATE)

    elif ptype == "xgboost":
        feat_cols = artifact["feature_columns"]
        feat_vals = [float(feature_row.get(c)) for c in feat_cols]
        p_buy  = predict_xgboost(artifact["xgb_buy"],  feat_vals)
        p_sell = predict_xgboost(artifact["xgb_sell"], feat_vals)
        raw    = float(p_buy - p_sell)
    else:
        raw = 0.0

    calibration = artifact.get("score_calibration")
    rank = calibrate_score(raw, calibration)
    er   = expected_return_from_calibration(
        raw, calibration, horizon_days=horizon_days,
    )

    if raw > artifact.get("buy_threshold", 0.1):
        signal = "buy"
    elif raw < artifact.get("sell_threshold", -0.1):
        signal = "sell"
    else:
        signal = "hold"

    return ScoreResult(
        raw_score=raw, rank_score=rank, signal=signal, expected_return=er,
    )


# ── Horizon-extrapolation visibility ──────────────────────────────────────────

def horizon_extrapolation_report(
    models: dict, target_horizon_days: int | None,
) -> dict | None:
    """Summarise how far ``expected_return`` is extrapolated past its fit.

    ``score_artifact`` scales the calibrator's native ``er_lookahead``-day ER
    linearly to ``rotation.target_horizon_days`` (``base × horizon/lookahead``).
    Live 2026-08: ``er_lookahead=5`` vs ``target_horizon_days=60`` → every ER
    is the 5-day fit ×12. Whether that extrapolation is sound is a separate
    design decision (NOT changed here); this makes the factor visible.

    Returns ``None`` when nothing is extrapolated beyond 2× (or there are no
    calibrated models / no horizon), else a dict with the artifact count,
    the lookahead(s) seen and the largest factor. Pure — the caller logs.
    """
    if not models or not target_horizon_days or int(target_horizon_days) <= 0:
        return None
    horizon = int(target_horizon_days)
    n_models = 0
    n_flagged = 0
    lookaheads: set[int] = set()
    max_factor = 0.0
    for art in models.values():
        cal = (art or {}).get("score_calibration") if isinstance(art, dict) else None
        if not cal:
            continue
        try:
            la = int(cal.get("er_lookahead", 5))
        except (TypeError, ValueError):
            continue
        if la <= 0:
            continue
        n_models += 1
        if la < horizon / 2.0:
            n_flagged += 1
            lookaheads.add(la)
            max_factor = max(max_factor, horizon / la)
    if n_flagged == 0:
        return None
    return {
        "target_horizon_days": horizon,
        "n_calibrated_models": n_models,
        "n_flagged": n_flagged,
        "er_lookaheads": sorted(lookaheads),
        "max_factor": max_factor,
    }


def warn_horizon_extrapolation(models: dict, config: dict | None) -> dict | None:
    """Log ONE WARNING per run when ``er_lookahead < target_horizon_days/2``."""
    horizon = int(((config or {}).get("rotation", {}) or {})
                  .get("target_horizon_days", 20))
    rep = horizon_extrapolation_report(models, horizon)
    if rep is not None:
        log.warning(
            "ER_HORIZON_EXTRAPOLATION: %d/%d calibrated per-ticker models fit "
            "expected_return over er_lookahead=%s days but rotation."
            "target_horizon_days=%d — expected_return is the short-horizon fit "
            "scaled linearly by up to x%.1f (er_lookahead < target/2). The "
            "extrapolation itself is unchanged; this line exists so the "
            "mismatch is visible in every run log.",
            rep["n_flagged"], rep["n_calibrated_models"],
            ",".join(str(x) for x in rep["er_lookaheads"]),
            rep["target_horizon_days"], rep["max_factor"],
        )
    return rep
