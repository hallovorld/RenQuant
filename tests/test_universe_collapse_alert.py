"""Regression tests for the universe-collapse OUTAGE alert (2026-07-11).

Incident 2026-07-08/09: per-ticker admission metadata regressed to a
2026-04 vintage (live-tree ``git checkout HEAD -- models/``), the 60d
freshness gate correctly fail-closed on 133/145 models, and TWO full
sessions ran with zero buy capability while ntfy reported a normal
"no trade (no_candidates)" — a silent availability outage rendered as a
market decision.

These tests pin the observability contract added in response:
  1. collapse verdict (`_universe_health`): floor fraction of the
     watchlist + the zero-loaded special case;
  2. rejection-reason cause bucketing (per-cause staleness counts);
  3. config-keyed floor with a safe, un-disableable default;
  4. ntfy TITLE carries a ``UNIVERSE-OUTAGE`` marker distinct from a
     plain DECISION (full runs only, never sell-only);
  5. run bundle records ``universe_collapse`` + ``universe_health``.

Observability only — no trading behaviour is asserted or changed.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
sys.path.insert(0, str(STRATEGY_DIR))

from live.runner import (  # noqa: E402
    _UNIVERSE_COLLAPSE_FLOOR_DEFAULT,
    _notify_decision,
    _universe_floor_frac,
    _universe_health,
    _universe_rejection_cause,
)


# ── 1+2. Collapse verdict + cause bucketing ──────────────────────────────────

def _incident_rejections(n_stale: int = 133, n_no_artifact: int = 9) -> dict:
    """The 2026-07-08 shape: axis staleness on live_train_end + ETF no-dirs."""
    rej = {f"T{i:03d}": "stale_76d_limit_60:live_train_end" for i in range(n_stale)}
    for i in range(n_no_artifact):
        rej[f"E{i:02d}"] = "no_artifact"
    return rej


def test_incident_shape_collapses_with_per_cause_counts():
    """4/145 loaded (the held names) + 133 stale → collapsed, causes counted."""
    health = _universe_health(
        n_loaded=4, watchlist_size=145,
        rejections=_incident_rejections(), floor_frac=0.5,
    )
    assert health["collapsed"] is True
    assert health["loaded"] == 4
    assert health["watchlist"] == 145
    assert health["causes"]["stale:live_train_end"] == 133
    assert health["causes"]["no_artifact"] == 9
    # Largest cause first — the actionable diagnostic leads.
    assert next(iter(health["causes"])) == "stale:live_train_end"


def test_healthy_universe_does_not_collapse():
    """Post-recovery steady state (125/145 ≈ 86%) is healthy."""
    health = _universe_health(
        n_loaded=125, watchlist_size=145, rejections={}, floor_frac=0.5,
    )
    assert health["collapsed"] is False


def test_zero_loaded_collapses_even_with_zero_floor():
    """Zero eligible models on a non-empty watchlist is ALWAYS an outage —
    the floor knob cannot opt out of the total-collapse case."""
    health = _universe_health(
        n_loaded=0, watchlist_size=145, rejections={}, floor_frac=0.0,
    )
    assert health["collapsed"] is True


def test_empty_watchlist_never_collapses():
    health = _universe_health(
        n_loaded=0, watchlist_size=0, rejections={}, floor_frac=0.5,
    )
    assert health["collapsed"] is False


def test_boundary_at_floor_is_not_collapsed():
    """Exactly at the floor (>=) passes; strictly below fails."""
    at = _universe_health(50, 100, {}, 0.5)
    below = _universe_health(49, 100, {}, 0.5)
    assert at["collapsed"] is False
    assert below["collapsed"] is True


def test_cause_bucketing_covers_known_reason_formats():
    assert _universe_rejection_cause("stale_76d_limit_60:live_train_end") == (
        "stale:live_train_end"
    )
    # Legacy trained_date gate emits no field suffix.
    assert _universe_rejection_cause("stale_75d_limit_60") == "stale"
    assert _universe_rejection_cause("no_artifact") == "no_artifact"
    assert _universe_rejection_cause("load_error_ValueError") == "load_error"
    assert _universe_rejection_cause("sharpe_0.400_below_0.5") == "below_floor:sharpe"
    assert _universe_rejection_cause("sharpe_missing") == "sharpe_missing"
    assert _universe_rejection_cause("trained_date_missing") == "trained_date_missing"
    assert _universe_rejection_cause(
        "data_cutoff_unparseable:live_train_end"
    ) == "data_cutoff_unparseable:live_train_end"
    assert _universe_rejection_cause("auto_drop_5d_filter_streak") == "auto_drop"
    # Variable ages/values aggregate into one bucket.
    a = _universe_rejection_cause("stale_76d_limit_60:live_train_end")
    b = _universe_rejection_cause("stale_79d_limit_60:live_train_end")
    assert a == b


# ── 3. Config-keyed floor, safe default ──────────────────────────────────────

def test_floor_default_and_override():
    assert _universe_floor_frac({}) == _UNIVERSE_COLLAPSE_FLOOR_DEFAULT
    assert _universe_floor_frac(None) == _UNIVERSE_COLLAPSE_FLOOR_DEFAULT
    assert _universe_floor_frac({"universe_collapse_floor_frac": 0.2}) == 0.2


def test_floor_bad_values_fall_back_to_default():
    """A config typo must never silently disable the outage page."""
    for bad in ("nope", -0.1, 1.5, None, float("nan")):
        assert _universe_floor_frac(
            {"universe_collapse_floor_frac": bad}
        ) == _UNIVERSE_COLLAPSE_FLOOR_DEFAULT


# ── 4. ntfy title marker ─────────────────────────────────────────────────────

def _quiet_ctx(universe_health=None):
    """A zero-trade cycle ctx, minimal attrs for _notify_decision."""
    return SimpleNamespace(
        orders_placed=[], orders_pending=[], orders_skipped=[],
        exits_placed=[], exits_pending=[], exits_failed=[],
        counters={}, ranked=[], holdings={},
        regime="BULL_CALM", confidence=0.61, portfolio_value=10627.0,
        universe_health=universe_health,
    )


def _capture_ntfy(monkeypatch):
    sent: list[dict] = []

    def fake_post(url, *, title, body, priority, taxonomy="INFO",
                  key=None, cooldown_seconds=0, force=False):
        sent.append({
            "title": title, "body": body, "priority": priority,
            "taxonomy": taxonomy, "key": key,
        })
        return True

    import live.runner as runner_mod
    monkeypatch.setattr(runner_mod, "_post_ntfy_with_retries", fake_post)
    return sent


def test_full_run_collapse_pages_as_outage(monkeypatch):
    """The exact 07-08 rendering, fixed: title carries the OUTAGE marker,
    body leads with loaded/watchlist + per-cause counts, priority raised."""
    sent = _capture_ntfy(monkeypatch)
    health = _universe_health(4, 145, _incident_rejections(), 0.5)
    _notify_decision("RENQUANT-104", "full", _quiet_ctx(health))
    assert len(sent) == 1
    assert sent[0]["title"] == "RENQUANT-104 [full] UNIVERSE-OUTAGE DECISION"
    assert sent[0]["body"].startswith("UNIVERSE-OUTAGE: 4/145")
    assert "stale:live_train_end=133" in sent[0]["body"]
    assert "no trade (no_candidates)" in sent[0]["body"]
    assert sent[0]["priority"] == "high"


def test_healthy_full_run_keeps_plain_decision_title(monkeypatch):
    sent = _capture_ntfy(monkeypatch)
    health = _universe_health(125, 145, {}, 0.5)
    _notify_decision("RENQUANT-104", "full", _quiet_ctx(health))
    assert sent[0]["title"] == "RENQUANT-104 [full] DECISION"
    assert "UNIVERSE-OUTAGE" not in sent[0]["body"]
    assert sent[0]["priority"] == "default"


def test_sell_only_run_never_carries_outage_marker(monkeypatch):
    """Sell-only cycles run no buy scan — collapse marker not meaningful."""
    sent = _capture_ntfy(monkeypatch)
    health = _universe_health(4, 145, _incident_rejections(), 0.5)
    _notify_decision("RENQUANT-104", "sell-only", _quiet_ctx(health))
    assert "UNIVERSE-OUTAGE" not in sent[0]["title"]
    assert "UNIVERSE-OUTAGE" not in sent[0]["body"]


def test_ctx_without_universe_health_is_unchanged(monkeypatch):
    """Callers that never stamp universe_health (sim/legacy) are untouched."""
    sent = _capture_ntfy(monkeypatch)
    ctx = _quiet_ctx(None)
    del ctx.universe_health
    _notify_decision("RENQUANT-104", "full", ctx)
    assert sent[0]["title"] == "RENQUANT-104 [full] DECISION"


def test_outage_and_healthy_dedup_keys_differ(monkeypatch):
    """An outage no-trade must never be cooldown-suppressed by a prior
    healthy no-trade that produced an otherwise identical alert key."""
    sent = _capture_ntfy(monkeypatch)
    _notify_decision(
        "RENQUANT-104", "full",
        _quiet_ctx(_universe_health(4, 145, _incident_rejections(), 0.5)),
    )
    _notify_decision(
        "RENQUANT-104", "full",
        _quiet_ctx(_universe_health(125, 145, {}, 0.5)),
    )
    assert sent[0]["key"] != sent[1]["key"]


# ── 5. Run-bundle persistence ────────────────────────────────────────────────

def test_run_bundle_records_universe_collapse():
    from kernel.artifact_contract import build_run_bundle

    health = _universe_health(4, 145, _incident_rejections(), 0.5)
    ctx = SimpleNamespace(
        ohlcv={}, buy_blocked=False, skip_buys=False, bear_only=False,
        regime="BULL_CALM", confidence=0.61,
        universe_health=health,
    )
    bundle = build_run_bundle(
        {"watchlist": ["AAA"]}, STRATEGY_DIR,
        run_id="r-outage", run_type="live", ctx=ctx,
    )
    assert bundle["universe_collapse"] is True
    assert bundle["universe_health"]["loaded"] == 4
    assert bundle["universe_health"]["causes"]["stale:live_train_end"] == 133


def test_run_bundle_without_universe_health_has_no_collapse_key():
    from kernel.artifact_contract import build_run_bundle

    ctx = SimpleNamespace(
        ohlcv={}, buy_blocked=False, skip_buys=False, bear_only=False,
        regime="BULL_CALM", confidence=0.61,
    )
    bundle = build_run_bundle(
        {"watchlist": ["AAA"]}, STRATEGY_DIR,
        run_id="r-plain", run_type="sim", ctx=ctx,
    )
    assert "universe_collapse" not in bundle
    assert "universe_health" not in bundle
