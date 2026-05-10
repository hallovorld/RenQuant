"""Promotion-tier assets: walk-forward gate + final promote decision.

This is the load-bearing pair. ``promote_decision`` literally cannot
materialise without ``wf_gate_pass`` upstream — the Dagster topology kills
the ``RQ_ALLOW_NO_WF=1`` bypass class by construction.

See CLAUDE.md §5.13.15: a gate that exists in code but is bypassed in the
launchd promote shell is theatrical. Encoding the dependency in the
asset graph makes "promote without gate" un-runnable, not just discouraged.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dagster import LegacyFreshnessPolicy, asset

from dagster_renquant._paths import (
    PROMOTE_DECISION_JSON,
    WF_GATE_PASS_JSON,
)
from dagster_renquant.assets.training import calibrator, panel_model

WEEKLY_FRESHNESS = LegacyFreshnessPolicy(maximum_lag_minutes=7 * 24 * 60)


@asset(
    deps=[panel_model],
    legacy_freshness_policy=WEEKLY_FRESHNESS,
    description=(
        "Walk-forward gate sentinel. Written by scripts/run_wf_gate.py "
        "on success. Absence == gate not yet run for this panel_model."
    ),
    group_name="promote",
)
def wf_gate_pass() -> dict:
    """Validate that a fresh wf_gate_pass sentinel exists.

    The sentinel is intentionally a separate file from the staging
    artifact's ``wf_gate_metadata`` so that the Dagster graph has a single
    read-target invariant. The weekly launchd plist
    (``com.renquant.weekly-wf-promote.plist``) is responsible for writing
    it; this asset only verifies its presence.
    """
    if not WF_GATE_PASS_JSON.is_file():
        raise FileNotFoundError(
            f"wf_gate_pass sentinel missing: {WF_GATE_PASS_JSON}. "
            "Run scripts/weekly_wf_promote.sh (without RQ_ALLOW_NO_WF=1) "
            "to materialize it."
        )
    payload = json.loads(WF_GATE_PASS_JSON.read_text())
    if not payload.get("passed", False):
        raise RuntimeError(
            f"wf_gate_pass sentinel exists but reports failure: {payload}"
        )
    return payload


@asset(
    # CRITICAL: deps=[wf_gate_pass, calibrator]. Removing wf_gate_pass from
    # this list is what the test_dagster_assets.py regression guard pins.
    deps=[wf_gate_pass, calibrator],
    description=(
        "Final promote decision. Cannot materialize without a fresh "
        "wf_gate_pass upstream — kills the RQ_ALLOW_NO_WF=1 bypass class."
    ),
    group_name="promote",
)
def promote_decision() -> dict:
    """Write a tiny decision record stamping the upstream gate."""
    record = {
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
        "wf_gate_pass_path": str(WF_GATE_PASS_JSON),
        "decision": "promote",
    }
    PROMOTE_DECISION_JSON.write_text(json.dumps(record, indent=2))
    return record
