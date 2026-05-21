#!/usr/bin/env python
"""Static contract checks for QP/Kelly configs.

The QP optimizer consumes a mean-return vector. A calibrated probability,
raw panel score, or model rank is not a valid optimizer mean unless it is
explicitly converted to expected-return-like units. These checks are kept
script-local so acceptance and sim entrypoints can fail before spending
compute on a non-comparable run.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STRICT_QP_CONTRACT_MODES = {"strict", "hard", "error", "enforce"}


@dataclass(frozen=True)
class QPContractReport:
    passed: bool
    qp_enabled: bool
    issues: list[str]
    evidence: dict[str, Any]

    def summary(self) -> str:
        if not self.qp_enabled:
            return "QP disabled"
        if self.passed:
            return "QP contract OK"
        return "QP contract failed: " + "; ".join(self.issues)


def validate_qp_contract_config(config: dict[str, Any]) -> QPContractReport:
    """Validate the static config has a legal route to QP μ and σ.

    Runtime still validates the actual candidate/holding objects. This
    catches stale side configs where the correct prod flags never reached
    the sim path.
    """
    rotation = config.get("rotation", {}) or {}
    joint = rotation.get("joint_actions", {}) or {}
    qp_enabled = bool(joint.get("enabled", False)) and (
        str(joint.get("solver", "greedy")).lower() == "qp"
    )
    if not qp_enabled:
        return QPContractReport(True, False, [], {"solver": joint.get("solver")})

    ranking = config.get("ranking", {}) or {}
    panel = ranking.get("panel_scoring", {}) or {}
    kelly = ranking.get("kelly_sizing", {}) or {}
    alpha_to_mu = ranking.get("alpha_to_mu", {}) or {}
    ngboost = panel.get("ngboost", {}) or {}
    global_cal = panel.get("global_calibration", {}) or {}

    raw_mode = joint.get("qp_mu_contract")
    mode = str(raw_mode).lower() if raw_mode is not None else ""
    forced_source = str(ranking.get("qp_mu_source", "mu")).lower()
    alpha_enabled = bool(alpha_to_mu.get("enabled", False))
    cal_mu_enabled = bool(kelly.get("use_calibrator_mu", False)) and bool(
        global_cal.get("enabled", False)
    )
    ngb_enabled = bool(ngboost.get("enabled", False))
    vol_fallback = bool(kelly.get("use_realized_vol_fallback", False))
    kelly_enabled = bool(kelly.get("enabled", False))

    issues: list[str] = []
    if mode not in STRICT_QP_CONTRACT_MODES:
        issues.append(
            "rotation.joint_actions.qp_mu_contract must be strict for sim/WF"
        )
    if forced_source not in {"", "none", "mu"} and not alpha_enabled:
        issues.append(
            "ranking.qp_mu_source forces raw score input but alpha_to_mu is disabled"
        )
    if not (alpha_enabled or cal_mu_enabled or ngb_enabled):
        issues.append(
            "no legal QP μ source: enable alpha_to_mu, calibrator μ, or NGBoost μ"
        )
    if kelly_enabled and not (ngb_enabled or vol_fallback):
        issues.append(
            "Kelly enabled without NGBoost σ or realized-vol σ fallback"
        )

    evidence = {
        "qp_mu_contract": mode or None,
        "qp_mu_source": forced_source,
        "alpha_to_mu_enabled": alpha_enabled,
        "calibrator_mu_enabled": cal_mu_enabled,
        "ngboost_mu_enabled": ngb_enabled,
        "kelly_enabled": kelly_enabled,
        "realized_vol_fallback_enabled": vol_fallback,
    }
    return QPContractReport(not issues, True, issues, evidence)
