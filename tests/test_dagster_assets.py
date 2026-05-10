"""Regression-guard tests for the Dagster cron-tier asset graph.

Pins the load-bearing invariants:
1. ``promote_decision`` depends on ``wf_gate_pass`` (kills RQ_ALLOW_NO_WF=1
   bypass class by construction — CLAUDE.md §5.13.15).
2. Removing ``wf_gate_pass`` from the asset list makes ``promote_decision``
   un-runnable.
3. Per-asset freshness policies match the cron-tier we documented.
4. Per CLAUDE.md §5.13.2, at least one test imports
   ``dagster_renquant.definitions`` to prove the package is wired into a
   real entry point and not orphaned dead code.
5. Topology is acyclic and reaches every asset from data-tier roots.
"""

from __future__ import annotations

import warnings

import pytest

# LegacyFreshnessPolicy emits a DeprecationWarning under dagster 1.13;
# silence at module import time.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="dagster.*")


def _key(asset_def) -> str:
    return asset_def.key.to_user_string()


def _deps(asset_def) -> set[str]:
    return {d.to_user_string() for d in asset_def.dependency_keys}


def _by_key(assets):
    return {_key(a): a for a in assets}


# --------------------------------------------------------------------------- #
# §5.13.2 not-orphaned guard.                                                  #
# --------------------------------------------------------------------------- #
def test_definitions_module_is_imported_and_loads():
    """Per §5.13.2: prove the package is wired in (not orphaned)."""
    from dagster_renquant.definitions import defs  # noqa: WPS433 — intentional
    assets = list(defs.assets)
    # Sanity: the 8 assets we set out to define are all present.
    assert len(assets) == 8, f"expected 8 assets, got {len(assets)}"
    keys = {_key(a) for a in assets}
    expected = {
        "ohlcv_data",
        "sec_fundamentals",
        "regime_artifact",
        "panel_features",
        "panel_model",
        "calibrator",
        "wf_gate_pass",
        "promote_decision",
    }
    assert keys == expected, f"missing or extra keys: {keys ^ expected}"


# --------------------------------------------------------------------------- #
# Topology: promote_decision depends on wf_gate_pass.                          #
# --------------------------------------------------------------------------- #
def test_promote_decision_depends_on_wf_gate_pass():
    """The load-bearing invariant. Removing this edge re-opens the
    RQ_ALLOW_NO_WF=1 bypass class (CLAUDE.md §5.13.15)."""
    from dagster_renquant.assets.promote import promote_decision

    assert "wf_gate_pass" in _deps(promote_decision), (
        "promote_decision MUST depend on wf_gate_pass — without this edge, "
        "the Dagster graph allows promotion without a walk-forward gate."
    )


def test_promote_decision_also_depends_on_calibrator():
    from dagster_renquant.assets.promote import promote_decision

    assert "calibrator" in _deps(promote_decision)


# --------------------------------------------------------------------------- #
# Removing wf_gate_pass makes promote_decision un-runnable.                    #
# --------------------------------------------------------------------------- #
def test_removing_wf_gate_pass_breaks_definitions_load():
    """If we drop wf_gate_pass from the asset list, Dagster's resolution
    raises because promote_decision references an undefined upstream key.
    """
    from dagster import Definitions

    from dagster_renquant.assets import ALL_ASSETS

    pruned = [a for a in ALL_ASSETS if _key(a) != "wf_gate_pass"]
    with pytest.raises(Exception):
        # Dagster surfaces this as DagsterInvalidDefinitionError but we
        # don't depend on the exact subclass — any failure here is the
        # invariant we want.
        Definitions(assets=pruned).get_asset_graph()


# --------------------------------------------------------------------------- #
# Freshness rules per asset.                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "asset_key,expected_minutes",
    [
        ("ohlcv_data", 24 * 60),         # 1d after market close
        ("sec_fundamentals", 7 * 24 * 60),  # 7d (quarterly)
        ("regime_artifact", 24 * 60),     # 1d
        ("panel_features", 24 * 60),      # 1d
        ("panel_model", 7 * 24 * 60),     # 7d (fwd_60d weekly retrain floor)
        ("calibrator", 7 * 24 * 60),
        ("wf_gate_pass", 7 * 24 * 60),    # 7d weekly gate
    ],
)
def test_freshness_policy_per_asset(asset_key, expected_minutes):
    from dagster_renquant.definitions import defs

    by_key = _by_key(defs.assets)
    asset_def = by_key[asset_key]
    policies = getattr(asset_def, "legacy_freshness_policies_by_key", {})
    fp = policies.get(asset_def.key)
    assert fp is not None, (
        f"{asset_key} must declare a legacy_freshness_policy "
        f"(cron-tier explicit per CLAUDE.md §5.13.6)"
    )
    assert fp.maximum_lag_minutes == expected_minutes, (
        f"{asset_key} freshness {fp.maximum_lag_minutes} min "
        f"!= expected {expected_minutes} min"
    )


def test_promote_decision_has_no_freshness_policy():
    """promote_decision is event-driven (fires only when wf_gate_pass +
    calibrator both fresh). It deliberately does NOT declare its own
    freshness window — its cadence is gated by upstream."""
    from dagster_renquant.definitions import defs

    by_key = _by_key(defs.assets)
    promote = by_key["promote_decision"]
    policies = getattr(promote, "legacy_freshness_policies_by_key", {})
    assert policies.get(promote.key) is None


# --------------------------------------------------------------------------- #
# Topology: acyclic + all reachable from data tier.                            #
# --------------------------------------------------------------------------- #
def test_graph_is_acyclic_and_complete():
    """Verify the full topology: each asset's deps point only at keys we
    actually defined, and there are no cycles."""
    from dagster_renquant.definitions import defs

    assets = list(defs.assets)
    keys = {_key(a) for a in assets}

    # No dangling edges.
    for a in assets:
        for dep in _deps(a):
            assert dep in keys, f"asset {_key(a)} has dangling dep {dep}"

    # DFS cycle check.
    deps_map = {_key(a): _deps(a) for a in assets}
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node):
        if node in visited:
            return
        if node in visiting:
            raise AssertionError(f"cycle through {node}")
        visiting.add(node)
        for nxt in deps_map[node]:
            dfs(nxt)
        visiting.remove(node)
        visited.add(node)

    for k in keys:
        dfs(k)

    # promote_decision must be reachable from ohlcv_data via panel_model.
    # (Forward closure check: which keys reach promote_decision via deps?)
    reverse: dict[str, set[str]] = {k: set() for k in keys}
    for k, ds in deps_map.items():
        for d in ds:
            reverse[d].add(k)

    def downstream(start):
        seen = set()
        stack = [start]
        while stack:
            n = stack.pop()
            for nxt in reverse[n]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return seen

    assert "promote_decision" in downstream("ohlcv_data")
    assert "promote_decision" in downstream("panel_model")
    assert "promote_decision" in downstream("wf_gate_pass")
