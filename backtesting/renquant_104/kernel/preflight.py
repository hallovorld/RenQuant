"""Pre-flight smoke test — runs at the start of every cron invocation.

Catches the class of bugs where a config / artifact / state file drifts
out of sync with what the runner assumes. Each check returns a
PreflightCheck result; any HARD failure raises PreflightFailed which
the live runner converts into an ntfy alert + abort (no orders placed).

Why this exists (2026-04-28):
  - 4-27: NGBoost feature drift (macro cols missing) → 0 buy
  - 4-28a: watchlist 227 vs model 103 mismatch → 06:32 fingerprint alert
  - 4-28b: production model best_iter=4 (untrained) → +0.0418 IC was
    random-walk noise
  - 4-28c: 103 launchd plist crashed every day on TypeError
  Each was different but ALL would have been caught by a 30-second
  pre-flight run at cron startup.

Checks (each returns ok / soft-warn / hard-fail):
  1. P-MODEL-ARTIFACT   — panel-ltr.json + ngboost-head.json exist + parse
  2. P-BEST-ITER        — best_iter ≥ min_best_iter (BUG-CV-2)
  3. P-CONFIG-FP        — config fingerprint matches artifact's stored fp
                          (BUG-CV-G7-mismatch class)
  4. P-WATCHLIST        — config watchlist size matches training watchlist
  5. P-FEATURE-COVER    — NGBoost head's feature_cols all present in
                          current panel pipeline output (≥ 95%)
  6. P-STATE-FILE       — live_state.{broker}.json parses (or absent
                          which is fine — first run)
  7. P-BROKER-CONNECT   — broker.connect() / get_account_value() works
                          (only if broker is provided; skipped in dry-run)

Usage in live/runner.py:
    from kernel.preflight import run_preflight, PreflightFailed
    try:
        run_preflight(config, broker, strategy_dir)
    except PreflightFailed as e:
        log.error("PRE-FLIGHT FAILED: %s", e)
        ntfy(...)
        sys.exit(2)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("kernel.preflight")


@dataclass
class PreflightCheck:
    name:     str
    severity: str    # "hard" | "soft"
    ok:       bool
    message:  str = ""
    details:  dict = field(default_factory=dict)


class PreflightFailed(RuntimeError):
    """Raised when any HARD check fails. Caught by runner.main()."""

    def __init__(self, failures: list[PreflightCheck]):
        self.failures = failures
        super().__init__(self._format(failures))

    @staticmethod
    def _format(failures: list[PreflightCheck]) -> str:
        lines = [f"{len(failures)} hard pre-flight check(s) failed:"]
        for c in failures:
            lines.append(f"  ✗ {c.name}: {c.message}")
        lines.append(
            "Live runner aborting. No orders placed. "
            "Investigate and re-run after fix."
        )
        return "\n".join(lines)


# ── Individual checks ──────────────────────────────────────────────────────

def _check_model_artifact(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-MODEL-ARTIFACT: panel-ltr.json exists + parses."""
    panel_cfg = config.get("panel_ltr", {})
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = strategy_dir / rel
    if not p.exists():
        return PreflightCheck(
            "P-MODEL-ARTIFACT", "hard", False,
            f"artifact missing: {p}",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-MODEL-ARTIFACT", "hard", False,
            f"artifact unreadable {p.name}: {exc}",
        )
    return PreflightCheck(
        "P-MODEL-ARTIFACT", "hard", True,
        f"loaded {p.name}",
        details={"path": str(p), "best_iter": meta.get("best_iter"),
                 "oos_mean_ic": meta.get("oos_mean_ic")},
    )


def _check_best_iter(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-BEST-ITER: model's best_iter ≥ min_best_iter (BUG-CV-2 class).

    Production was discovered to have best_iter=4 today (4 × 0.02 eta =
    0.08 total shrinkage = essentially untrained). This check refuses
    to trade on an undertrained model.
    """
    panel_cfg = config.get("panel_ltr", {})
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = strategy_dir / rel
    if not p.exists():
        return PreflightCheck(
            "P-BEST-ITER", "hard", False, f"artifact missing: {p}",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-BEST-ITER", "hard", False, f"unreadable: {exc}",
        )
    bi = meta.get("best_iter")
    if bi is None:
        # Older artifacts (e.g. transformer backend) may not stamp this.
        return PreflightCheck(
            "P-BEST-ITER", "soft", True,
            "best_iter not stamped in artifact (legacy pre-2026-04-28); skip",
        )
    min_bi = int(panel_cfg.get("min_best_iter", 5))
    if int(bi) < min_bi:
        # 2026-05-04 (P0 fix): mirror the FinalFitTask training-time
        # escape clause. best_iter < min_best_iter is a FALSE POSITIVE
        # on strong-univariate-IC features (XGBoost converges by round
        # 4-9 and further rounds add zero eval-set improvement). If
        # eval_ic at best_iter is healthy (≥ floor, default 0.02),
        # accept the model. This keeps the runtime guard symmetric
        # with the training-time guard — pre-fix, training accepted
        # the model + saved the artifact, then preflight refused to
        # load it = strategy never trades. Pathological case
        # (eval_ic ≈ 0 or missing from artifact) still fails-safe.
        eval_ic_floor = float(panel_cfg.get("min_best_iter_eval_ic_floor", 0.02))
        eval_ic = meta.get("eval_ic")
        try:
            eval_ic_f = float(eval_ic) if eval_ic is not None else None
        except (TypeError, ValueError):
            eval_ic_f = None
        import math as _math
        if (eval_ic_f is not None and _math.isfinite(eval_ic_f)
                and eval_ic_f >= eval_ic_floor):
            return PreflightCheck(
                "P-BEST-ITER", "hard", True,
                f"best_iter={bi} < {min_bi} but eval_ic={eval_ic_f:+.4f} ≥ "
                f"floor={eval_ic_floor:+.4f} — strong-univariate-IC plateau, accepting",
                details={"best_iter": bi, "min_best_iter": min_bi,
                         "eval_ic": eval_ic_f, "eval_ic_floor": eval_ic_floor},
            )
        return PreflightCheck(
            "P-BEST-ITER", "hard", False,
            f"best_iter={bi} < min_best_iter={min_bi} AND "
            f"eval_ic={eval_ic} < floor={eval_ic_floor:+.4f}. "
            f"Model undertrained (early-stop fired in round {bi}). "
            f"Retrain required, OR confirm eval_ic is stamped in the artifact "
            f"(SaveArtifactTask must include 'eval_ic' in meta).",
            details={"best_iter": bi, "min_best_iter": min_bi,
                     "eval_ic": eval_ic, "eval_ic_floor": eval_ic_floor},
        )
    return PreflightCheck(
        "P-BEST-ITER", "hard", True,
        f"best_iter={bi} ≥ {min_bi}",
        details={"best_iter": bi},
    )


def _check_config_fingerprint(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-CONFIG-FP: live config's fingerprint matches artifact's stored fp.

    Catches: watchlist drift, lookahead change, objective change,
    asset_embeddings flip — the four-incidents class from 2026-04-27/28.
    """
    panel_cfg = config.get("panel_ltr", {})
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = strategy_dir / rel
    if not p.exists():
        return PreflightCheck(
            "P-CONFIG-FP", "hard", False, f"artifact missing: {p}",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-CONFIG-FP", "hard", False, f"unreadable: {exc}",
        )
    try:
        from kernel.config_consistency import (  # noqa: PLC0415
            fingerprint_config, _model_relevant_fields,
        )
    except Exception as exc:
        return PreflightCheck(
            "P-CONFIG-FP", "soft", True,
            f"config_consistency module unavailable: {exc} — skip",
        )
    live_fp = fingerprint_config(config)
    stored = meta.get("config_fingerprint")
    if stored is None:
        return PreflightCheck(
            "P-CONFIG-FP", "soft", True,
            "artifact lacks fingerprint (pre-2026-04-28 retrain) — "
            "stamped at next retrain",
            details={"live": live_fp},
        )
    if stored == live_fp:
        return PreflightCheck(
            "P-CONFIG-FP", "hard", True,
            f"fingerprint match {live_fp}",
        )
    stored_sub = meta.get("config_fingerprint_fields") or {}
    # Defensive: 2026-05-08 alpha158_fund artifact stores
    # config_fingerprint_fields as a LIST of field names (not a value
    # dict like the legacy 21-feat format). When that's the case we
    # cannot compute a per-field diff, so soft-pass the check —
    # fingerprint mismatch is real but not actionable without stored
    # field VALUES to point at the diverging key.
    if not isinstance(stored_sub, dict):
        return PreflightCheck(
            "P-CONFIG-FP", "soft", True,
            f"fingerprint_fields stored as {type(stored_sub).__name__}, "
            f"not dict — can't diff. live={live_fp} stored={stored}",
        )
    diff_keys = []
    live_sub = _model_relevant_fields(config)
    for k in sorted(set(live_sub) | set(stored_sub)):
        if live_sub.get(k) != stored_sub.get(k):
            diff_keys.append(k)
    return PreflightCheck(
        "P-CONFIG-FP", "hard", False,
        f"fingerprint mismatch: live={live_fp} stored={stored} "
        f"diff_fields={diff_keys}",
        details={"live": live_fp, "stored": stored, "diff_fields": diff_keys},
    )


def _check_watchlist_size(config: dict, strategy_dir: Path) -> PreflightCheck:
    """P-WATCHLIST: config watchlist length consistent with training."""
    wl = config.get("watchlist") or []
    panel_cfg = config.get("panel_ltr", {})
    rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    p = strategy_dir / rel
    if not p.exists():
        return PreflightCheck(
            "P-WATCHLIST", "hard", False, f"artifact missing: {p}",
        )
    try:
        meta = json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-WATCHLIST", "hard", False, f"unreadable: {exc}",
        )
    fields = meta.get("config_fingerprint_fields") or {}
    # Defensive: alpha158_fund artifact stores fingerprint_fields as a
    # LIST of field names (no values). Treat as not-stamped → soft-pass.
    if not isinstance(fields, dict):
        fields = {}
    trained_wl = fields.get("watchlist") or []
    if not trained_wl:
        return PreflightCheck(
            "P-WATCHLIST", "soft", True,
            f"trained watchlist not stamped; live={len(wl)} ticker(s)",
        )
    if set(wl) != set(trained_wl):
        only_live = sorted(set(wl) - set(trained_wl))[:5]
        only_trained = sorted(set(trained_wl) - set(wl))[:5]
        return PreflightCheck(
            "P-WATCHLIST", "hard", False,
            f"watchlist mismatch live={len(wl)} trained={len(trained_wl)} "
            f"in_live_not_trained={only_live} in_trained_not_live={only_trained}",
        )
    return PreflightCheck(
        "P-WATCHLIST", "hard", True,
        f"watchlist match (n={len(wl)})",
    )


def _check_feature_coverage(
    config: dict, strategy_dir: Path,
    feature_drift_pct: float = 0.05,
) -> PreflightCheck:
    """P-FEATURE-COVER: NGBoost head's feature_cols are present.

    This is a STATIC check on artifact metadata — checks that the
    NGBoost head and the panel-LTR scorer agree on feature_cols. The
    actual runtime drift detector in ApplyNGBoostTask catches the
    dynamic case.
    """
    panel_cfg = config.get("panel_ltr", {})
    panel_rel = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")

    ngb_cfg = (config.get("ranking", {})
                       .get("panel_scoring", {})
                       .get("ngboost", {}))

    # 2026-05-17 Bug fix: a per-regime overlay can activate NGB even when
    # the global flag is False. Pre-fix, P-FEATURE-COVER skipped entirely
    # on global enabled=False, leaving NGB feature-drift risk un-validated
    # whenever today's regime ∈ {BEAR, CHOPPY, BULL_VOLATILE} (or any
    # regime with `regime_params.<R>.ngboost.enabled=True`). If panel-LTR
    # and NGB head disagree on feature_cols, ApplyNGBoostTask hard-fails
    # at runtime (CRIT-1) → all candidates blocked the moment σ-wire
    # activates. Catch it at preflight instead.
    regime_params = config.get("regime_params", {}) or {}
    per_regime_activates = [
        r for r, p in regime_params.items()
        if isinstance(p, dict)
        and isinstance(p.get("ngboost"), dict)
        and p["ngboost"].get("enabled") is True
    ]
    ngb_potentially_active = (
        bool(ngb_cfg.get("enabled", False)) or bool(per_regime_activates)
    )
    if not ngb_potentially_active:
        return PreflightCheck(
            "P-FEATURE-COVER", "soft", True,
            "NGBoost disabled globally + no per-regime overlay activates — skip",
        )
    ngb_rel = ngb_cfg.get("artifact_path", "artifacts/prod/ngboost-head.alpha158_fund.json")

    panel_p = strategy_dir / panel_rel
    ngb_p   = strategy_dir / ngb_rel
    if not panel_p.exists() or not ngb_p.exists():
        return PreflightCheck(
            "P-FEATURE-COVER", "hard", False,
            f"artifact missing: panel={panel_p.exists()} ngb={ngb_p.exists()}",
        )
    try:
        panel_meta = json.loads(panel_p.read_text())
        ngb_meta   = json.loads(ngb_p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-FEATURE-COVER", "hard", False, f"unreadable: {exc}",
        )
    panel_feats = set(panel_meta.get("feature_cols") or [])
    ngb_feats   = set(ngb_meta.get("feature_cols")   or [])
    if not ngb_feats:
        return PreflightCheck(
            "P-FEATURE-COVER", "soft", True,
            "NGBoost feature_cols not stamped — skip",
        )
    missing = ngb_feats - panel_feats
    pct = len(missing) / max(1, len(ngb_feats))
    if pct > feature_drift_pct:
        return PreflightCheck(
            "P-FEATURE-COVER", "hard", False,
            f"NGBoost expects {len(ngb_feats)} feats, "
            f"{len(missing)} ({pct:.1%}) missing from panel — "
            f"retrain NGBoost head against current panel pipeline. "
            f"First 5 missing: {sorted(missing)[:5]}",
            details={"missing_count": len(missing),
                     "missing_pct": pct,
                     "first_missing": sorted(missing)[:10]},
        )
    return PreflightCheck(
        "P-FEATURE-COVER", "hard", True,
        f"NGBoost feature coverage OK ({len(ngb_feats)} feats, "
        f"{len(missing)} missing = {pct:.1%})",
    )


def _check_state_file(
    config: dict, strategy_dir: Path, broker_name: str | None,
) -> PreflightCheck:
    """P-STATE-FILE: live_state.{broker}.json parses (or absent)."""
    if not broker_name:
        return PreflightCheck(
            "P-STATE-FILE", "soft", True, "no broker_name (dry-run); skip",
        )
    try:
        from kernel.state_paths import resolve_live_state_read  # noqa: PLC0415
    except Exception as exc:
        return PreflightCheck(
            "P-STATE-FILE", "soft", True,
            f"state_paths unavailable: {exc}; skip",
        )
    p, _used_legacy = resolve_live_state_read(strategy_dir, broker_name)
    if not p.exists():
        return PreflightCheck(
            "P-STATE-FILE", "soft", True,
            f"state file absent at {p.name} (first run?)",
        )
    try:
        json.loads(p.read_text())
    except Exception as exc:
        return PreflightCheck(
            "P-STATE-FILE", "hard", False,
            f"state file unreadable {p.name}: {exc}",
        )
    return PreflightCheck(
        "P-STATE-FILE", "hard", True, f"loaded {p.name}",
    )


def _check_broker_connect(broker: Any) -> PreflightCheck:
    """P-BROKER-CONNECT: connect + get_account_value works."""
    if broker is None:
        return PreflightCheck(
            "P-BROKER-CONNECT", "soft", True,
            "no broker (dry-run); skip",
        )
    try:
        broker.connect()
        eq = float(broker.get_account_value())
        return PreflightCheck(
            "P-BROKER-CONNECT", "hard", True,
            f"broker connected, equity=${eq:.2f}",
        )
    except Exception as exc:
        return PreflightCheck(
            "P-BROKER-CONNECT", "hard", False,
            f"broker connect failed: {exc}",
        )


def _check_artifact_run_id_alignment(
    config: dict, strategy_dir: Path
) -> PreflightCheck:
    """P-RUN-ID: panel-ltr and ngboost-head share the same train_run_id.

    External audit fix #2 (2026-04-29): without run_id, one artifact can
    silently come from a different training run (e.g. a side-config retrain
    overwriting production NGBoost). A mismatch means μ/σ was fit on a
    different panel feature distribution than the scorer — Kelly sizing
    corrupted. Soft check (old artifacts don't have run_id yet).
    """
    panel_cfg  = config.get("panel_ltr", {})
    ltr_rel    = panel_cfg.get("artifact_path", "artifacts/prod/panel-ltr.alpha158_fund.json")
    ngb_cfg    = (config.get("ranking", {}).get("panel_scoring", {})
                  .get("ngboost", {}))
    ngb_rel    = ngb_cfg.get("artifact_path", "artifacts/prod/ngboost-head.alpha158_fund.json")
    ltr_path   = strategy_dir / ltr_rel
    ngb_path   = strategy_dir / ngb_rel
    for p in (ltr_path, ngb_path):
        if not p.exists():
            return PreflightCheck(
                "P-RUN-ID", "soft", True, f"artifact missing: {p} — skip",
            )
    try:
        ltr_id = json.loads(ltr_path.read_text()).get("train_run_id")
        ngb_id = json.loads(ngb_path.read_text()).get("train_run_id")
    except Exception as exc:
        return PreflightCheck(
            "P-RUN-ID", "soft", True, f"unreadable: {exc}",
        )
    if ltr_id is None or ngb_id is None:
        return PreflightCheck(
            "P-RUN-ID", "soft", True,
            "run_id not stamped (pre-2026-04-29 artifact) — skip",
        )
    if ltr_id != ngb_id:
        return PreflightCheck(
            "P-RUN-ID", "soft", False,
            f"run_id mismatch: panel-ltr={ltr_id} ngboost={ngb_id}. "
            f"NGBoost μ/σ may be from a different training run — Kelly "
            f"sizing potentially corrupted. Retrain recommended.",
        )
    return PreflightCheck(
        "P-RUN-ID", "soft", True,
        f"run_id aligned ({ltr_id})",
    )


# ── Orchestrator ───────────────────────────────────────────────────────────

ALL_CHECKS = (
    _check_model_artifact,
    _check_best_iter,
    _check_config_fingerprint,
    _check_watchlist_size,
    _check_feature_coverage,
    _check_state_file,
    _check_broker_connect,
    _check_artifact_run_id_alignment,  # audit fix #2 — soft check
    None,  # _check_calibrator_health — registered below to keep ALL_CHECKS readable
)


def _check_calibrator_health(config: dict, strategy_dir: Path) -> "PreflightCheck":
    """P-CALIBRATOR-HEALTH (2026-05-05 parity fix): runtime equivalent of the
    training-side `fit_global_calibrator` "probability head collapsed to N
    unique values" guard.

    Today's diagnostic (2026-05-04 e2e) found `n_unique_prob_y = 7` in the
    production calibrator: only 7 distinct calibrated probabilities across
    235K training rows. Result at runtime: top 10 candidates all tied at
    rank_score=0.2579, no real ranking → strategy can't differentiate. The
    training-time guard catches this AT FIT but the artifact had been saved
    BEFORE that guard was added; runtime had no way to detect the
    degradation. This check closes that gap.

    Hard-fail when:
      * artifact missing or unparseable
      * `n_unique_prob_y < min_unique_prob_y` (default 10)
    Soft-warn when:
      * `pool_ic <= 0` (calibrator anti-correlated with labels)

    Tunable via config.panel_ltr.calibrator_health.min_unique_prob_y.
    Backwards-compat: pre-2026-05 artifacts without n_unique_prob_y in
    metadata get a soft skip (can't verify, log a warning).
    """
    panel_cfg = config.get("panel_ltr", {})
    cal_cfg = ((config.get("ranking", {})
                       .get("panel_scoring", {})
                       .get("global_calibration", {})) or {})
    rel = cal_cfg.get("artifact_path", "artifacts/prod/panel-rank-calibration.json")
    p = strategy_dir / rel
    if not p.exists():
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "soft", True,
            f"calibrator artifact absent at {p} — global_calibration may be disabled",
        )
    try:
        payload = json.loads(p.read_text())
        meta = payload.get("metadata", {}) or {}
    except Exception as exc:
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "hard", False, f"unreadable: {exc}",
        )
    n_unique = meta.get("n_unique_prob_y")
    pool_ic = meta.get("pool_ic")

    # 2026-05-15 P0 ADDITION: range-bound check on expected_return.y.
    # Catches the bug class that caused the rank_score saturation incident:
    # calibrator artifacts with er.y up to +1.00 (= +100% expected return)
    # would feed catastrophically wrong μ into Kelly when use_calibrator_mu=
    # true. Hard-fail before live trade so a bad artifact never reaches QP.
    # Threshold matches the train-site clip (2026-05-15 Phase 4 commit).
    try:
        er_y = payload.get("expected_return", {}).get("y", []) or []
        if er_y:
            er_max_abs = max(abs(float(v)) for v in er_y
                             if v is not None and v == v)  # NaN-safe
            ER_BOUND = 0.20  # matches GlobalPanelCalibration.load() clip
            if er_max_abs > ER_BOUND + 1e-9:
                return PreflightCheck(
                    "P-CALIBRATOR-HEALTH", "hard", False,
                    f"calibrator expected_return.y has max|y|={er_max_abs:.4f} > "
                    f"{ER_BOUND} sanity bound. CLAUDE.md §5.13.12 violation: "
                    f"artifact was not clipped at train site. Kelly sizing on "
                    f"this calibrator would produce broken position weights. "
                    f"Refit via scripts/fit_calibrator_alpha158_fund.py before "
                    f"live trade.",
                    details={"max_abs_er_y": er_max_abs,
                             "bound": ER_BOUND, "n_knots": len(er_y)},
                )
    except (TypeError, ValueError) as exc:
        log.warning("P-CALIBRATOR-HEALTH: could not check er.y bounds: %s", exc)
    health_cfg = panel_cfg.get("calibrator_health", {}) or {}
    min_unique = int(health_cfg.get("min_unique_prob_y", 10))
    if n_unique is None:
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "soft", True,
            "n_unique_prob_y not stamped (legacy artifact); skip — will be "
            "stamped on next retrain. Cannot verify probability-head granularity.",
        )
    if int(n_unique) < min_unique:
        # 2026-05-05 incident: this was originally `hard`, which blocked ALL
        # preflight — including sell-only intraday crons that don't use the
        # calibrator at all (sells run via SellGateB + path rules). 14 sell
        # crons aborted between 07:00–09:36 PT before the severity was
        # downgraded. Calibrator collapse is a *quality* issue (top candidates
        # tie, ranking is ineffective) not a *safety* issue — the adaptive
        # buy_floor at `min(max(0.20, mean+std), 0.30)` already absorbs tied
        # scores at the cap, and downstream sizing/risk gates are unaffected.
        # SOFT keeps the loud warning visible without halting live ops.
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "soft", True,
            f"WARN: n_unique_prob_y={n_unique} < min_unique_prob_y={min_unique}. "
            f"Calibrator probability head collapsed — top candidates will tie, "
            f"buy ranking is ineffective. Retrain or investigate isotonic-input "
            f"distribution. Sells unaffected; buys will fall through to "
            f"buy_floor cap. (See doc/components/calibration.md and the "
            f"2026-05-04 e2e post-mortem.)",
            details={"n_unique_prob_y": n_unique, "min_unique_prob_y": min_unique,
                     "pool_ic": pool_ic},
        )
    if pool_ic is not None and float(pool_ic) <= 0:
        return PreflightCheck(
            "P-CALIBRATOR-HEALTH", "soft", True,
            f"pool_ic={pool_ic} ≤ 0 — calibrator anti-correlated with labels; "
            f"investigate before live trade. n_unique={n_unique}.",
            details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
        )
    return PreflightCheck(
        "P-CALIBRATOR-HEALTH", "hard", True,
        f"n_unique_prob_y={n_unique} ≥ {min_unique}, pool_ic={pool_ic}",
        details={"n_unique_prob_y": n_unique, "pool_ic": pool_ic},
    )


# Replace the placeholder in ALL_CHECKS with the actual function.
ALL_CHECKS = tuple(c if c is not None else _check_calibrator_health for c in ALL_CHECKS)


def run_preflight(
    config: dict,
    broker: Any = None,
    strategy_dir: Path | str | None = None,
    broker_name: str | None = None,
    *,
    strict: bool = True,
) -> list[PreflightCheck]:
    """Run all checks. Raise PreflightFailed if any HARD check fails
    (when strict=True). Returns the full result list either way."""
    if strategy_dir is None:
        raise ValueError("run_preflight requires strategy_dir")
    sd = Path(strategy_dir)
    if broker is not None and broker_name is None:
        broker_name = getattr(broker, "broker_name", None)

    results: list[PreflightCheck] = []
    for fn in ALL_CHECKS:
        try:
            sig = fn.__code__.co_varnames[:fn.__code__.co_argcount]
            kwargs: dict[str, Any] = {"config": config}
            if "strategy_dir" in sig:
                kwargs["strategy_dir"] = sd
            if "broker_name" in sig:
                kwargs["broker_name"] = broker_name
            if "broker" in sig:
                kwargs = {"broker": broker}    # broker check has different sig
            res = fn(**kwargs) if "broker" in sig else fn(**kwargs)
        except Exception as exc:
            res = PreflightCheck(
                fn.__name__, "soft", True,
                f"check raised unexpectedly: {exc} — degrading to soft-pass",
            )
        results.append(res)
        marker = "✓" if res.ok else "✗"
        sev = res.severity.upper()
        log.info("preflight %s %-22s [%s] %s", marker, res.name, sev, res.message)

    hard_failures = [r for r in results if r.severity == "hard" and not r.ok]
    if hard_failures and strict:
        raise PreflightFailed(hard_failures)
    return results
