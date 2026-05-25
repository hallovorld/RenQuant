"""Model acceptance gates — promote new model only if all hard gates pass.

User spec 2026-04-26: "我们有没有机制进行模型accpetance verification，如果不
通过的话，继续用原来的模型跑E2E？"

Without these gates, retraining is dangerous: a broken or quality-degraded
panel-ltr.json silently swaps in and corrupts every subsequent live run.
This module formalizes the staging → gate → promote OR rollback workflow.

Workflow:
    1. FullTrainingPipeline writes the new artifact to a STAGING path
       (e.g., artifacts/panel-ltr.staging.json).
    2. ``ModelAcceptanceGate.evaluate(staging_path, active_path)`` runs
       all hard gates. Each gate compares staging vs active artifact
       metadata, OR runs an isolated sanity check on staging alone.
    3. If all hard gates pass → ``promote()``:
         - mv  active   → previous (rollback target)
         - mv  staging  → active   (atomic from operator's view)
    4. If any hard gate fails → ``reject()``:
         - mv  staging  → archives/_acceptance_log/{ts}_REJECTED.json
         - active stays unchanged → live runner sees the prior model
         - ntfy alert with reason

Soft gates emit warnings but don't block promotion.

Architecture choice — why NOT atomic via os.rename trickery:
The active artifact is a JSON file read at process start. Live runner
loads it once per pipeline; we don't need within-second atomicity. The
mv-mv pair is "atomic enough" — at worst, a concurrent reader sees
either the prior or the new file fully, never a partial. Use locks if
multiple operators retrain concurrently (out of scope for solo use).
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("kernel.model_acceptance")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class GateResult:
    """One gate's verdict."""
    name:       str
    severity:   str          # "hard" or "soft"
    passed:     bool
    metric:     float | None
    threshold:  float | None
    detail:     str

    def __str__(self) -> str:
        m = f"{self.metric:.4f}" if isinstance(self.metric, (int, float)) else "n/a"
        t = f"{self.threshold:.4f}" if isinstance(self.threshold, (int, float)) else "n/a"
        status = "PASS" if self.passed else "FAIL"
        return f"[{self.severity:<4}] {self.name:<28} {status}  metric={m}  threshold={t}  {self.detail}"


@dataclass
class AcceptanceVerdict:
    """Aggregated result over all gates."""
    all_hard_passed: bool
    results:         list[GateResult]
    timestamp:       datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    def hard_failures(self) -> list[GateResult]:
        return [r for r in self.results if r.severity == "hard" and not r.passed]

    def soft_warnings(self) -> list[GateResult]:
        return [r for r in self.results if r.severity == "soft" and not r.passed]

    def summary(self) -> str:
        lines = [f"AcceptanceVerdict @ {self.timestamp.isoformat(timespec='seconds')}Z"]
        for r in self.results:
            lines.append(f"  {r}")
        n_hard_fail = len(self.hard_failures())
        n_soft_fail = len(self.soft_warnings())
        verdict = "ACCEPT" if self.all_hard_passed else "REJECT"
        lines.append(f"  ─── VERDICT: {verdict} (hard fails={n_hard_fail}, soft warns={n_soft_fail})")
        return "\n".join(lines)


# ── Gate definitions ──────────────────────────────────────────────────────────

@dataclass
class AcceptanceGate:
    name:     str
    severity: str          # "hard" | "soft"
    check:    Callable[[dict, dict | None], GateResult]


def _safe_get_metadata(artifact: dict) -> dict:
    """Extract `metadata` block; some artifacts have it nested, some flat."""
    md = artifact.get("metadata")
    if isinstance(md, dict):
        merged = {k: v for k, v in artifact.items() if k != "metadata"}
        merged.update(md)
        return merged
    return artifact   # flat


def _gate_g1_schema_compatibility(staging: dict, active: dict | None) -> GateResult:
    """G1: feature_cols length matches active OR diff is in expected-add list.

    Without this, a transformer trained with 33 extra macro features
    would write a sidecar that the production XGBoost consumer can't
    read.
    """
    new_cols = staging.get("feature_cols")
    if not isinstance(new_cols, list):
        return GateResult("G1_schema", "hard", False, None, None,
                          "staging artifact missing feature_cols")
    new_n = len(new_cols)
    if active is None:
        # No prior — accept any non-empty list
        return GateResult("G1_schema", "hard", new_n > 0, float(new_n), 1.0,
                          f"no prior artifact; accept {new_n} cols")
    prior_cols = active.get("feature_cols") or []
    prior_n = len(prior_cols)
    new_set = set(new_cols)
    prior_set = set(prior_cols)
    # Three accepted cases: identical, identical content (different order),
    # or new is a strict superset (e.g. + macro features).
    if new_set == prior_set:
        return GateResult("G1_schema", "hard", True, float(new_n), float(prior_n),
                          f"feature_cols match ({new_n} cols)")
    if prior_set < new_set:
        added = sorted(new_set - prior_set)
        return GateResult("G1_schema", "hard", True, float(new_n), float(prior_n),
                          f"superset OK (+{len(added)} new cols: {added[:5]})")
    return GateResult(
        "G1_schema", "hard", False, float(new_n), float(prior_n),
        f"feature_cols changed in unexpected way: prior={prior_n} new={new_n} "
        f"(missing from new: {sorted(prior_set - new_set)[:5]}, "
        f"unexpected in new: {sorted(new_set - prior_set)[:5]})",
    )


def _gate_g2_calibrator_non_collapse(staging: dict, active: dict | None) -> GateResult:
    """G2: calibrator probability head must have ≥5 unique y values.

    Already enforced at fit-time (Bug #4 fix in global_calibrator.py),
    but defense-in-depth: re-check here so a hand-written or imported
    artifact can't bypass.
    """
    md = _safe_get_metadata(staging)
    n_unique = md.get("n_unique_prob_y")
    if n_unique is None:
        # Calibrator artifact looks different — try the calibration
        # sidecar instead. Fail open: not all backend artifacts have
        # this metric (e.g., panel-ltr.json before calibrator was wired).
        return GateResult("G2_calibrator_unique", "hard", True, None, 5.0,
                          "no n_unique_prob_y on this artifact (skip)")
    return GateResult(
        "G2_calibrator_unique", "hard", n_unique >= 5,
        float(n_unique), 5.0,
        f"calibrator unique y={n_unique}",
    )


def _gate_g3_pool_ic_positive(staging: dict, active: dict | None) -> GateResult:
    """G3: pool IC must be > 0 (not collapsed to base rate).

    Negative pool IC means the calibrator inverted the signal.
    """
    md = _safe_get_metadata(staging)
    pool_ic = md.get("pool_ic")
    if pool_ic is None:
        return GateResult("G3_pool_ic_positive", "hard", True, None, 0.0,
                          "no pool_ic on this artifact (skip)")
    return GateResult(
        "G3_pool_ic_positive", "hard", pool_ic > 0,
        float(pool_ic), 0.0,
        f"pool_ic={pool_ic:+.6f}",
    )


def _gate_g4_oos_ic_vs_prior(staging: dict, active: dict | None,
                              max_degradation: float = 0.05) -> GateResult:
    """G4: new oos_mean_ic ≥ prior × (1 - max_degradation).

    Default 5% (Phase 1 tightening 2026-04-26): the old 30% default
    would have ACCEPTED the macro-enabled XGBoost (0.0393) vs the prior
    non-macro (0.0482) at -18.5% degradation. 5% rejects anything worse
    than 0.0482 × 0.95 = 0.0458, forcing operator review on regressions.

    Configurable per `acceptance.g4_max_degradation` in strategy_config.json.
    """
    md_new = _safe_get_metadata(staging)
    new_ic = md_new.get("oos_mean_ic")
    if new_ic is None:
        return GateResult("G4_oos_ic_vs_prior", "hard", False, None, None,
                          "staging artifact missing oos_mean_ic")
    if active is None:
        # No prior to compare — pass if positive, defer absolute floor to G7
        return GateResult("G4_oos_ic_vs_prior", "hard", new_ic > 0,
                          float(new_ic), 0.0,
                          f"no prior — accept on new>0 (got {new_ic:+.4f})")
    md_prior = _safe_get_metadata(active)
    prior_ic = md_prior.get("oos_mean_ic")
    if prior_ic is None:
        # Prior has no IC — degenerate, accept new
        return GateResult("G4_oos_ic_vs_prior", "hard", True,
                          float(new_ic), None,
                          f"no prior IC reference; accept new={new_ic:+.4f}")
    if prior_ic <= 0:
        # Audit fix #3 (2026-04-26): pre-fix only required new > prior,
        # so a prior of -0.01 made any new IC > -0.01 pass — including
        # near-zero noise (e.g. new=0.001). That's degenerate. Now we
        # additionally require new > 0 (strict positive — a broken prior
        # doesn't license shipping a near-zero new model).
        passed = (new_ic > prior_ic) and (new_ic > 0.0)
        return GateResult("G4_oos_ic_vs_prior", "hard", passed,
                          float(new_ic), 0.0,
                          f"prior was non-positive ({prior_ic:+.4f}); "
                          f"new must beat prior AND be strictly positive")
    threshold = prior_ic * (1.0 - max_degradation)
    return GateResult(
        "G4_oos_ic_vs_prior", "hard", new_ic >= threshold,
        float(new_ic), float(threshold),
        f"new={new_ic:+.4f} prior={prior_ic:+.4f} (max_degradation={max_degradation:.0%})",
    )


def _gate_g5_score_range_coverage(staging: dict, active: dict | None) -> GateResult:
    """G5: new model's score output range covers ≥80% of prior's range.

    Catches the "calibrator collapsed to constant" failure mode where
    every input maps to base_rate. Requires a sample of scores —
    typically a smoke-test set passed in via metadata.score_sample_range.
    """
    md_new = _safe_get_metadata(staging)
    rng = md_new.get("score_sample_range")    # [low, high] from smoke test
    if rng is None or not isinstance(rng, (list, tuple)) or len(rng) != 2:
        # No smoke-test data — soft-pass (it's a soft-ish gate but we
        # still want it hard if data is present)
        return GateResult("G5_score_range", "hard", True, None, None,
                          "no score_sample_range; skip")
    new_lo, new_hi = float(rng[0]), float(rng[1])
    new_span = new_hi - new_lo
    if active is None or _safe_get_metadata(active).get("score_sample_range") is None:
        return GateResult("G5_score_range", "hard", new_span > 0.001,
                          new_span, 0.001,
                          f"new span={new_span:.4f} (no prior to compare)")
    prior_lo, prior_hi = _safe_get_metadata(active)["score_sample_range"]
    prior_span = float(prior_hi) - float(prior_lo)
    if prior_span <= 0:
        return GateResult("G5_score_range", "hard", new_span > 0.001,
                          new_span, 0.001,
                          f"prior span degenerate; new={new_span:.4f}")
    coverage = new_span / prior_span
    return GateResult(
        "G5_score_range", "hard", coverage >= 0.80,
        float(coverage), 0.80,
        f"new_span={new_span:.4f} prior_span={prior_span:.4f} (coverage={coverage:.0%})",
    )


def _gate_g6_inference_smoke(staging: dict, active: dict | None) -> GateResult:
    """G6: a stored smoke-test sample of inference outputs must be all
    finite and non-NaN.

    This catches model artifacts where the model loads but produces
    NaN scores under any input (often after a serialization bug).
    """
    md = _safe_get_metadata(staging)
    smoke = md.get("inference_smoke_test")  # {"n": 32, "all_finite": true, "n_unique": 31}
    if smoke is None:
        # No smoke test recorded — soft-pass; we'll add it to the
        # FullTrainingPipeline output in a later commit.
        return GateResult("G6_inference_smoke", "hard", True, None, None,
                          "no inference_smoke_test (skip)")
    all_finite = bool(smoke.get("all_finite", False))
    return GateResult(
        "G6_inference_smoke", "hard", all_finite, None, None,
        f"smoke_test all_finite={all_finite} (n={smoke.get('n')})",
    )


def _gate_g7_oos_ic_absolute_floor(staging: dict, active: dict | None,
                                    floor: float = 0.02) -> GateResult:
    """G7: OOS IC above absolute noise floor.

    Phase 1 (2026-04-26): hardened from soft → hard. A model with +0.005
    IC is technically positive but not useful — soft-warn alone doesn't
    prevent it from shipping. Configurable via `acceptance.g7_floor`;
    set `acceptance.g7_severity = "soft"` to revert to warn-only.

    Skips (passes-open) when staging artifact lacks oos_mean_ic
    (e.g., a non-panel sidecar artifact wired through the same gate).
    """
    md = _safe_get_metadata(staging)
    new_ic = md.get("oos_mean_ic")
    if new_ic is None:
        return GateResult("G7_oos_ic_floor", "hard", True, None, floor,
                          "no oos_mean_ic (skip)")
    return GateResult(
        "G7_oos_ic_floor", "hard", new_ic >= floor,
        float(new_ic), floor,
        f"oos_mean_ic={new_ic:+.4f} vs floor={floor:+.4f}",
    )


def _gate_g9_sim_apy_vs_prior(staging: dict, active: dict | None,
                               max_pp_drop: float = 1.0) -> GateResult:
    """G9 (Phase 2): sim APY ≥ prior APY − max_pp_drop percentage points.

    Catches the "looks good in OOS IC, falls apart in actual sim" case
    that pure-IC gates can't see. Reads from `metadata.sim_smoke.apy`
    populated by `kernel.sim_smoke.add_smoke_metrics_to_artifact`.

    Skip-pass when sim_smoke metrics absent on staging — this gate is
    only meaningful when both staging AND prior have run smoke tests.
    """
    md_new = _safe_get_metadata(staging)
    smoke_new = md_new.get("sim_smoke") or {}
    new_apy = smoke_new.get("apy")
    if new_apy is None:
        return GateResult("G9_sim_apy", "hard", True, None, None,
                          "no sim_smoke.apy on staging (skip)")
    if active is None:
        return GateResult("G9_sim_apy", "hard", new_apy > -0.10,
                          float(new_apy), -0.10,
                          f"no prior — accept if apy > -10% (got {new_apy:+.2%})")
    md_prior = _safe_get_metadata(active)
    smoke_prior = md_prior.get("sim_smoke") or {}
    prior_apy = smoke_prior.get("apy")
    if prior_apy is None:
        return GateResult("G9_sim_apy", "hard", True, None, None,
                          "no sim_smoke.apy on prior (skip)")
    threshold = float(prior_apy) - (max_pp_drop / 100.0)
    return GateResult(
        "G9_sim_apy", "hard", new_apy >= threshold,
        float(new_apy), float(threshold),
        f"new={new_apy:+.2%} prior={prior_apy:+.2%} (max_pp_drop={max_pp_drop:.1f}pp)",
    )


def _gate_g10_sim_sharpe_vs_prior(staging: dict, active: dict | None,
                                   max_drop: float = 0.1) -> GateResult:
    """G10 (Phase 2): sim Sharpe ≥ prior Sharpe − max_drop.

    A model with the same APY but much lower Sharpe is risk-degraded.
    Skip-pass when smoke metrics absent.
    """
    md_new = _safe_get_metadata(staging)
    smoke_new = md_new.get("sim_smoke") or {}
    new_sharpe = smoke_new.get("sharpe")
    if new_sharpe is None:
        return GateResult("G10_sim_sharpe", "hard", True, None, None,
                          "no sim_smoke.sharpe on staging (skip)")
    if active is None:
        return GateResult("G10_sim_sharpe", "hard", new_sharpe > 0.0,
                          float(new_sharpe), 0.0,
                          f"no prior — accept on sharpe > 0 (got {new_sharpe:+.2f})")
    md_prior = _safe_get_metadata(active)
    smoke_prior = md_prior.get("sim_smoke") or {}
    prior_sharpe = smoke_prior.get("sharpe")
    if prior_sharpe is None:
        return GateResult("G10_sim_sharpe", "hard", True, None, None,
                          "no sim_smoke.sharpe on prior (skip)")
    threshold = float(prior_sharpe) - max_drop
    return GateResult(
        "G10_sim_sharpe", "hard", new_sharpe >= threshold,
        float(new_sharpe), float(threshold),
        f"new={new_sharpe:+.2f} prior={prior_sharpe:+.2f} (max_drop={max_drop:.2f})",
    )


def _gate_g11_turnover_ratio(staging: dict, active: dict | None,
                              max_multiplier: float = 1.5) -> GateResult:
    """G11 (Phase 2): turnover_ratio ≤ prior × max_multiplier.

    A model that triples turnover for the same returns is paying more
    in slippage/tax — net real performance is worse even if gross
    APY/Sharpe look the same. Configurable via `g11_max_multiplier`.

    Skip-pass when smoke metrics absent.
    """
    md_new = _safe_get_metadata(staging)
    smoke_new = md_new.get("sim_smoke") or {}
    new_to = smoke_new.get("turnover_ratio")
    if new_to is None:
        return GateResult("G11_turnover", "soft", True, None, None,
                          "no sim_smoke.turnover_ratio on staging (skip)")
    if active is None:
        return GateResult("G11_turnover", "soft", True, None, None,
                          "no prior — accept any turnover")
    md_prior = _safe_get_metadata(active)
    smoke_prior = md_prior.get("sim_smoke") or {}
    prior_to = smoke_prior.get("turnover_ratio")
    if prior_to is None or prior_to <= 0:
        return GateResult("G11_turnover", "soft", True, None, None,
                          "no prior turnover (skip)")
    threshold = float(prior_to) * max_multiplier
    return GateResult(
        "G11_turnover", "soft", new_to <= threshold,
        float(new_to), float(threshold),
        f"new={new_to:.2f}x prior={prior_to:.2f}x (max_multiplier={max_multiplier:.2f})",
    )


def _gate_g8_per_ticker_variance(staging: dict, active: dict | None,
                                  min_std: float = 0.001) -> GateResult:
    """G8 (soft): per-bar score std > 0.001.

    Calibrator collapse pattern: every ticker on a given bar gets the
    same score → variance is zero → ranking is broken. Soft-warn.
    """
    md = _safe_get_metadata(staging)
    smoke = md.get("inference_smoke_test")
    if smoke is None:
        return GateResult("G8_per_ticker_variance", "soft", True, None, min_std,
                          "no smoke_test (skip)")
    score_std = smoke.get("score_std")
    if score_std is None:
        return GateResult("G8_per_ticker_variance", "soft", True, None, min_std,
                          "smoke_test missing score_std (skip)")
    return GateResult(
        "G8_per_ticker_variance", "soft", score_std >= min_std,
        float(score_std), min_std,
        f"score_std={score_std:.6f}",
    )


# ── Default gate list ─────────────────────────────────────────────────────────
#
# IMPORTANT (audit fix #7, 2026-04-26): DEFAULT_GATES is a fallback-only
# reference. PRODUCTION CALL PATH goes through `build_gates_from_config(cfg)`
# which honors per-gate severity overrides from strategy_config.json. If you
# read DEFAULT_GATES directly (bypassing config), you get the hardcoded
# severities here, NOT what the operator configured. New CLIs should call
# `ModelAcceptanceGate(config=acc_cfg)` instead of `ModelAcceptanceGate()`.

DEFAULT_GATES: list[AcceptanceGate] = [
    AcceptanceGate("G1_schema",            "hard", _gate_g1_schema_compatibility),
    AcceptanceGate("G2_calibrator_unique", "hard", _gate_g2_calibrator_non_collapse),
    AcceptanceGate("G3_pool_ic_positive",  "hard", _gate_g3_pool_ic_positive),
    AcceptanceGate("G4_oos_ic_vs_prior",   "hard", _gate_g4_oos_ic_vs_prior),
    AcceptanceGate("G5_score_range",       "hard", _gate_g5_score_range_coverage),
    AcceptanceGate("G6_inference_smoke",   "hard", _gate_g6_inference_smoke),
    AcceptanceGate("G7_oos_ic_floor",      "hard", _gate_g7_oos_ic_absolute_floor),
    AcceptanceGate("G8_per_ticker_variance","soft", _gate_g8_per_ticker_variance),
    AcceptanceGate("G9_sim_apy",           "hard", _gate_g9_sim_apy_vs_prior),
    AcceptanceGate("G10_sim_sharpe",       "hard", _gate_g10_sim_sharpe_vs_prior),
    AcceptanceGate("G11_turnover",         "soft", _gate_g11_turnover_ratio),
]


def build_gates_from_config(config: dict) -> list[AcceptanceGate]:
    """Build gate list with thresholds + severities sourced from config.

    Config keys (all optional — defaults match DEFAULT_GATES):
        g4_max_degradation:  float (0.05 default)
        g4_severity:         "hard"|"soft" ("hard" default)
        g7_floor:            float (0.02 default)
        g7_severity:         "hard"|"soft" ("hard" default)
        g8_min_std:          float (0.001 default)
        g8_severity:         "hard"|"soft" ("soft" default)

    Phase 1 (2026-04-26): added so operators can tune per environment
    without forking the gate code (e.g., loosen G4 for an exploratory
    panel rebuild known to drift under noise).
    """
    g4_md    = float(config.get("g4_max_degradation",  0.05))
    g4_sev   = str(  config.get("g4_severity",          "hard"))
    g7_fl    = float(config.get("g7_floor",             0.02))
    g7_sev   = str(  config.get("g7_severity",          "hard"))
    g8_min   = float(config.get("g8_min_std",           0.001))
    g8_sev   = str(  config.get("g8_severity",          "soft"))
    g9_pp    = float(config.get("g9_max_pp_drop",       1.0))
    g9_sev   = str(  config.get("g9_severity",          "hard"))
    g10_drop = float(config.get("g10_max_sharpe_drop",  0.1))
    g10_sev  = str(  config.get("g10_severity",         "hard"))
    g11_mult = float(config.get("g11_max_multiplier",   1.5))
    g11_sev  = str(  config.get("g11_severity",         "soft"))

    def _g4(s, a, _md=g4_md): return _gate_g4_oos_ic_vs_prior(s, a, max_degradation=_md)
    def _g7(s, a, _fl=g7_fl, _sev=g7_sev):
        r = _gate_g7_oos_ic_absolute_floor(s, a, floor=_fl)
        return GateResult(r.name, _sev, r.passed, r.metric, r.threshold, r.detail)
    def _g8(s, a, _min=g8_min, _sev=g8_sev):
        r = _gate_g8_per_ticker_variance(s, a, min_std=_min)
        return GateResult(r.name, _sev, r.passed, r.metric, r.threshold, r.detail)
    def _g9(s, a, _pp=g9_pp, _sev=g9_sev):
        r = _gate_g9_sim_apy_vs_prior(s, a, max_pp_drop=_pp)
        return GateResult(r.name, _sev, r.passed, r.metric, r.threshold, r.detail)
    def _g10(s, a, _drop=g10_drop, _sev=g10_sev):
        r = _gate_g10_sim_sharpe_vs_prior(s, a, max_drop=_drop)
        return GateResult(r.name, _sev, r.passed, r.metric, r.threshold, r.detail)
    def _g11(s, a, _mult=g11_mult, _sev=g11_sev):
        r = _gate_g11_turnover_ratio(s, a, max_multiplier=_mult)
        return GateResult(r.name, _sev, r.passed, r.metric, r.threshold, r.detail)

    return [
        AcceptanceGate("G1_schema",            "hard", _gate_g1_schema_compatibility),
        AcceptanceGate("G2_calibrator_unique", "hard", _gate_g2_calibrator_non_collapse),
        AcceptanceGate("G3_pool_ic_positive",  "hard", _gate_g3_pool_ic_positive),
        AcceptanceGate("G4_oos_ic_vs_prior",   g4_sev, _g4),
        AcceptanceGate("G5_score_range",       "hard", _gate_g5_score_range_coverage),
        AcceptanceGate("G6_inference_smoke",   "hard", _gate_g6_inference_smoke),
        AcceptanceGate("G7_oos_ic_floor",      g7_sev, _g7),
        AcceptanceGate("G8_per_ticker_variance",g8_sev, _g8),
        AcceptanceGate("G9_sim_apy",           g9_sev, _g9),
        AcceptanceGate("G10_sim_sharpe",       g10_sev,_g10),
        AcceptanceGate("G11_turnover",         g11_sev,_g11),
    ]


# ── Main evaluator ────────────────────────────────────────────────────────────

class ModelAcceptanceGate:
    """Run all gates against staging vs active artifact; return verdict."""

    def __init__(self, gates: list[AcceptanceGate] | None = None,
                 config: dict | None = None):
        if gates is not None:
            self.gates = gates
        elif config is not None:
            self.gates = build_gates_from_config(config)
        else:
            self.gates = list(DEFAULT_GATES)

    @staticmethod
    def _load_artifact(path: Path) -> dict | None:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("ModelAcceptanceGate: cannot load %s: %s", path, exc)
            return None

    def evaluate(self, staging_path: Path,
                 active_path: Path | None = None) -> AcceptanceVerdict:
        staging = self._load_artifact(staging_path)
        if staging is None:
            return AcceptanceVerdict(
                all_hard_passed=False,
                results=[GateResult("staging_load", "hard", False, None, None,
                                    f"failed to load {staging_path}")],
            )
        active = self._load_artifact(active_path) if active_path else None

        results: list[GateResult] = []
        for gate in self.gates:
            try:
                r = gate.check(staging, active)
            except Exception as exc:
                # Defensive — a buggy gate shouldn't crash the verdict.
                # Treat unexpected exception as hard failure.
                r = GateResult(gate.name, gate.severity, False, None, None,
                               f"gate raised {type(exc).__name__}: {exc}")
            results.append(r)

        all_hard_passed = all(r.passed for r in results if r.severity == "hard")

        # Audit fix #4+#11 (2026-04-26): when a HARD gate skip-passes
        # because metadata is missing (e.g. G5 score_sample_range, G6
        # inference_smoke_test, G9/G10/G11 sim_smoke), emit an explicit
        # warning so the operator knows which hard checks are silent.
        # Pre-fix, an operator could believe G5/G6 were protecting against
        # calibrator collapse when they were actually skip-passing every
        # model.
        for r in results:
            if r.severity == "hard" and r.passed and r.metric is None and r.threshold is None and "skip" in (r.detail or "").lower():
                log.warning("ModelAcceptanceGate: HARD gate %s skipped — %s", r.name, r.detail)

        return AcceptanceVerdict(all_hard_passed=all_hard_passed, results=results)


# ── Atomic-swap promote / reject ──────────────────────────────────────────────

def _check_wf_gate(staging_data: dict, staging_path: Path) -> None:
    """2026-05-09 P0 #1 — walk-forward gate enforcement.

    Per E55/roadmap rewrite: every promote requires walk-forward 3-cut
    Sharpe + §5.2 sanity battery (shuffled-label + time-shift placebo)
    BEFORE the artifact is allowed into the active path.

    Today's revelation (NGB on/off A/B): single-cut sim Sharpe was
    misleading us into promoting models that lost 3.78 APY pp on
    walk-forward. Past Sharpe claims (1.06, 1.10, 2.01) were on
    contaminated code AND single-window cuts. This gate forces the
    discipline CLAUDE.md §5.9 already required.

    Required artifact metadata schema (written by
    `scripts/run_wf_gate.py`):
      wf_gate_metadata:
        passed:               bool       # overall verdict
        wf_3cut_sharpe_mean:  float
        wf_3cut_sharpe_std:   float
        wf_3cut_apy_mean:     float
        sanity_shuffled_ic:   float       # must be ≈ 0
        sanity_placebo_ic:    float       # must be < 0.5 × same-row aligned real IC
        sanity_placebo_aligned_real_ic: float
        candidate_artifact_used: bool     # True for leakage-safe static eval
        recipe_validated:      bool       # True for matching manifest eval
        run_at:               str (ISO-8601)
        gate_version:         int

    Failure modes:
      - missing wf_gate_metadata        → refuse promote
      - passed != True                  → refuse promote
      - neither static artifact nor matching manifest recipe was evaluated
                                           → refuse promote
      - run_at older than 14 days       → refuse promote (must re-run)

    Emergency override: set env `RQ_ALLOW_NO_WF=1` (logged loudly,
    rate-limited per CLAUDE.md §5.5 rollback rehearsal).
    """
    import os as _os                                    # noqa: PLC0415
    import datetime as _dt                              # noqa: PLC0415
    if _os.environ.get("RQ_ALLOW_NO_WF") == "1":
        log.warning(
            "PROMOTE OVERRIDE: RQ_ALLOW_NO_WF=1 set — bypassing walk-forward "
            "gate for %s. This is an emergency-only override; per CLAUDE.md "
            "§5.5 you MUST rehearse rollback within 24h. Reason should be "
            "documented in commit message.",
            staging_path.name,
        )
        return
    md = staging_data.get("metadata", {}) or {}
    wf = md.get("wf_gate_metadata")
    if not isinstance(wf, dict):
        # Also check top-level (some artifacts store metadata at root)
        wf = staging_data.get("wf_gate_metadata")
    if not isinstance(wf, dict):
        raise ValueError(
            f"promote: refused — staging artifact missing wf_gate_metadata "
            f"({staging_path}). Run `python scripts/run_wf_gate.py "
            f"--artifact {staging_path}` first. Override with "
            f"RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    if not bool(wf.get("passed")):
        raise ValueError(
            f"promote: refused — wf_gate_metadata.passed=False on "
            f"{staging_path.name}. Detail: "
            f"sharpe_mean={wf.get('wf_3cut_sharpe_mean')} "
            f"shuffled_ic={wf.get('sanity_shuffled_ic')} "
            f"placebo_ic={wf.get('sanity_placebo_ic')} "
            f"aligned_real_ic={wf.get('sanity_placebo_aligned_real_ic')}. "
            f"Override with RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    trade_monotonicity = (
        wf.get("trade_monotonicity")
        if isinstance(wf.get("trade_monotonicity"), dict)
        else None
    )
    if not trade_monotonicity or trade_monotonicity.get("passed") is not True:
        raise ValueError(
            f"promote: refused — wf_gate_metadata is missing passing "
            f"trade_monotonicity on {staging_path.name}. Detail: "
            f"{trade_monotonicity.get('reason') if isinstance(trade_monotonicity, dict) else 'absent'}. "
            f"Re-run `scripts/run_wf_gate.py` with persisted trade trace so "
            f"entry-score monotonicity participates in the promotion verdict. "
            f"Override with RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    if bool(trade_monotonicity.get("allow_pass_open")):
        raise ValueError(
            f"promote: refused — trade_monotonicity was allowed to pass-open "
            f"on {staging_path.name}; this is diagnostic-only evidence. "
            f"Re-run without --allow-pass-open-trade-monotonicity. Override "
            f"with RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    regimes_raw = trade_monotonicity.get("regimes")
    if isinstance(regimes_raw, dict):
        regimes = [
            stats for stats in regimes_raw.values()
            if isinstance(stats, dict)
        ]
    elif isinstance(regimes_raw, list):
        regimes = [stats for stats in regimes_raw if isinstance(stats, dict)]
    else:
        regimes = []
    eligible = [stats for stats in regimes if bool(stats.get("eligible", False))]
    failed = [stats for stats in eligible if stats.get("passed") is not True]
    if not eligible or failed:
        raise ValueError(
            f"promote: refused — trade_monotonicity lacks passing eligible "
            f"regime evidence on {staging_path.name}. "
            f"eligible={len(eligible)} failed={len(failed)} reason="
            f"{trade_monotonicity.get('reason')}. Override with "
            f"RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    alpha_economics = (
        wf.get("alpha_economics")
        if isinstance(wf.get("alpha_economics"), dict)
        else None
    )
    if not alpha_economics or alpha_economics.get("passed") is not True:
        raise ValueError(
            f"promote: refused — wf_gate_metadata is missing passing "
            f"alpha_economics on {staging_path.name}. Detail: "
            f"{alpha_economics.get('reason') if isinstance(alpha_economics, dict) else 'absent'}. "
            f"Re-run `scripts/run_wf_gate.py` with persisted trade trace so "
            f"benchmark-sleeve runs prove active alpha economics. Override "
            f"with RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    sanity = wf.get("sanity_regime_ic") if isinstance(wf.get("sanity_regime_ic"), dict) else None
    if not sanity or sanity.get("passed") is not True:
        raise ValueError(
            f"promote: refused — wf_gate_metadata is missing passing "
            f"sanity_regime_ic on {staging_path.name}. Detail: "
            f"{sanity.get('reason') if isinstance(sanity, dict) else 'absent'}. "
            f"Re-run `scripts/run_wf_gate.py` so regime-layered placebo/IC "
            f"evidence participates in the promotion verdict. Override with "
            f"RQ_ALLOW_NO_WF=1 (emergency only)."
        )
    if wf.get("candidate_artifact_used") is False and wf.get("recipe_validated") is not True:
        raise ValueError(
            f"promote: refused — wf_gate_metadata says the WF sim did not "
            f"evaluate the candidate artifact ({staging_path.name}) and did "
            f"not validate a matching manifest recipe; "
            f"scope={wf.get('wf_eval_scope')!r}. Re-run the gate with a "
            f"leakage-safe static artifact config or a manifest whose "
            f"artifacts match the candidate recipe."
        )
    # Staleness check: WF results older than 14 days are not credible
    # for current model state (panel data + bug-fix lineage may differ).
    run_at = wf.get("run_at")
    if run_at:
        try:
            ran = _dt.datetime.fromisoformat(str(run_at).replace("Z", "+00:00"))
            if ran.tzinfo is not None:
                ran = ran.replace(tzinfo=None)
            age_days = (_dt.datetime.utcnow() - ran).total_seconds() / 86400
            if age_days > 14:
                raise ValueError(
                    f"promote: refused — wf_gate_metadata stale "
                    f"({age_days:.1f} days old, threshold 14d). Re-run "
                    f"`scripts/run_wf_gate.py`. Override with RQ_ALLOW_NO_WF=1."
                )
        except (ValueError, TypeError) as exc:
            if "stale" in str(exc):
                raise
            log.warning("wf_gate_metadata.run_at unparseable: %r — skipping age check", run_at)


def promote(staging_path: Path, active_path: Path) -> None:
    """Atomically swap staging into active, archiving prior to .previous.

    Audit fix #2 (2026-04-26): validate staging JSON BEFORE swapping.
    select_best_model.py --promote can copy a corrupted .bak.json into
    staging; without validation, promote() blindly moves it into active
    and the live runner crashes at first load. We re-parse the file
    and require the basic schema (kind + feature_cols) before swapping.

    Audit fix #12 (2026-04-26): use a temp-file + os.rename idiom so
    the active_path is never missing during the swap. Pre-fix, two
    shutil.move calls left a window where active didn't exist; a
    concurrent live-runner load would fail. Now: move staging → temp
    next to active first, then atomically rename active → previous,
    then rename temp → active. On the same filesystem, os.rename is
    atomic (POSIX); the active path is always EITHER the prior OR the
    new file, never absent.

    Walk-forward gate (2026-05-09 P0 #1): refuses promote without
    wf_gate_metadata.passed=True; see _check_wf_gate.
    """
    staging_path = Path(staging_path)
    active_path  = Path(active_path)
    if not staging_path.exists():
        raise FileNotFoundError(f"staging artifact missing: {staging_path}")

    # ── Audit fix #2: validate staging JSON before any move ──
    try:
        data = json.loads(staging_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"promote: staging artifact is not valid JSON ({staging_path}): {exc}"
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(f"promote: staging artifact is not a JSON object: {staging_path}")
    if "kind" not in data and "feature_cols" not in data:
        # A panel artifact must have at least one of these — protects
        # against the file existing but being totally wrong shape.
        raise ValueError(
            f"promote: staging artifact missing both 'kind' and 'feature_cols' "
            f"({staging_path}); refusing to swap into active"
        )

    # 2026-05-09 P0 #1 — walk-forward gate enforcement (after E55 NGB revert)
    _check_wf_gate(data, staging_path)

    previous_path = active_path.with_suffix(".previous.json")
    # ── Audit fix #12: stage the new file next to active first so
    # the rename swap can be atomic. Same-filesystem rename is POSIX-
    # atomic; live runner reading active_path always sees prior or new,
    # never empty.
    temp_active = active_path.with_suffix(".incoming.json")
    shutil.copy2(str(staging_path), str(temp_active))
    if active_path.exists():
        # Step 1: rotate the current active to .previous (rollback target)
        os.replace(str(active_path), str(previous_path))
    # Step 2: rename incoming → active (atomic on same filesystem)
    os.replace(str(temp_active), str(active_path))
    # Step 3: drop the staging copy now that the active rotation is done
    try:
        staging_path.unlink()
    except FileNotFoundError:
        pass
    log.info("PROMOTE: %s → %s (prior preserved at %s)",
             staging_path.name, active_path.name, previous_path.name)


def reject(staging_path: Path, archive_dir: Path,
           verdict: AcceptanceVerdict) -> None:
    """Archive staging artifact + verdict log; active is left untouched."""
    staging_path = Path(staging_path)
    archive_dir  = Path(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    if not staging_path.exists():
        log.warning("reject: staging missing %s — nothing to archive", staging_path)
        return
    ts = verdict.timestamp.strftime("%Y-%m-%dT%H%M%S")
    archive_path = archive_dir / f"{ts}_REJECTED_{staging_path.name}"
    log_path = archive_dir / f"{ts}_REJECTED_verdict.txt"
    shutil.move(str(staging_path), str(archive_path))
    log_path.write_text(verdict.summary())
    log.warning("REJECT: %s → %s | reasons:\n%s",
                staging_path.name, archive_path.name, verdict.summary())


def rollback(active_path: Path) -> None:
    """Operator-triggered rollback: swap active ← previous."""
    active_path   = Path(active_path)
    previous_path = active_path.with_suffix(".previous.json")
    if not previous_path.exists():
        raise FileNotFoundError(f"no rollback target at {previous_path}")
    # Archive current active before overwriting
    archive = active_path.parent / f"_acceptance_log/auto-rollback-{datetime.datetime.utcnow().strftime('%Y-%m-%dT%H%M%S')}_{active_path.name}"
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(active_path), str(archive))
    shutil.move(str(previous_path), str(active_path))
    log.warning("ROLLBACK: restored %s from .previous.json (current archived to %s)",
                active_path.name, archive.name)


__all__ = [
    "AcceptanceGate", "GateResult", "AcceptanceVerdict",
    "ModelAcceptanceGate", "DEFAULT_GATES",
    "promote", "reject", "rollback",
]
