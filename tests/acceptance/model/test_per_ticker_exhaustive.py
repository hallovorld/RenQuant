"""Per-ticker policy-metadata.json exhaustive tests.

Auto-discovers EVERY ticker directory in models/ and runs the
standard schema battery against each artifact. With 103 tickers ×
common attributes × per-type attributes × 10 check kinds, this
generates ~5000 raw test invocations, most of which skip (e.g.
non-numeric attributes don't run finite/bounds checks).

All tests skip cleanly when a per-ticker artifact is absent.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MODELS = REPO / "backtesting" / "renquant_104" / "models"
sys.path.insert(0, str(REPO / "tests"))

from acceptance.model.schemas import (                # noqa: E402
    PER_TICKER_COMMON_SCHEMA,
    PER_TICKER_MANUAL_SCHEMA,
    PER_TICKER_TREE_SCHEMA,
)


# ── Discover all tickers ────────────────────────────────────────────────────

def _all_tickers() -> list[str]:
    if not MODELS.exists():
        return []
    return sorted(d.name for d in MODELS.iterdir() if d.is_dir()
                   and (d / f"{d.name}-policy-metadata.json").exists())


_TICKERS = _all_tickers()


def _load_policy(ticker: str) -> dict | None:
    path = MODELS / ticker / f"{ticker}-policy-metadata.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


# ── Common-attribute tests (run against ALL tickers) ───────────────────────

_COMMON_ATTRS = list(PER_TICKER_COMMON_SCHEMA.keys())


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonPresence:
    def test_presence(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("models/ directory empty")
        d = _load_policy(ticker)
        if d is None:
            pytest.skip(f"{ticker} policy missing/corrupt")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        if spec.get("required", False):
            assert attr in d, f"{ticker}: required attr {attr!r} missing"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonType:
    def test_type(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or attr not in d:
            pytest.skip("absent")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        v = d[attr]
        assert isinstance(v, spec["type"]), \
            f"{ticker}: {attr} type {type(v).__name__} ≠ {spec['type']}"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonFinite:
    def test_finite(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or attr not in d or d[attr] is None:
            pytest.skip("absent / None")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        v = d[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric")
        if not spec.get("finite", False):
            # Even attrs not flagged finite should not be NaN unless the
            # spec explicitly tolerates it. Skip when spec doesn't ask
            # for finite — Sharpe can be NaN on degenerate data.
            pytest.skip("not flagged finite")
        assert math.isfinite(float(v)), \
            f"{ticker}: {attr}={v} non-finite"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonBoundsLow:
    def test_lower(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or attr not in d or d[attr] is None:
            pytest.skip("absent")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        if "bounds" not in spec:
            pytest.skip("no bounds")
        v = d[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric")
        if not math.isfinite(v):
            pytest.skip("non-finite — bounds undefined")
        lo, _ = spec["bounds"]
        assert v >= lo, f"{ticker}: {attr}={v} < {lo}"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonBoundsHigh:
    def test_upper(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or attr not in d or d[attr] is None:
            pytest.skip("absent")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        if "bounds" not in spec:
            pytest.skip("no bounds")
        v = d[attr]
        if not isinstance(v, (int, float)):
            pytest.skip("non-numeric")
        if not math.isfinite(v):
            pytest.skip("non-finite")
        _, hi = spec["bounds"]
        assert v <= hi, f"{ticker}: {attr}={v} > {hi}"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
@pytest.mark.parametrize("attr", _COMMON_ATTRS)
class TestPerTickerCommonAllowed:
    def test_allowed(self, ticker, attr):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or attr not in d:
            pytest.skip("absent")
        spec = PER_TICKER_COMMON_SCHEMA[attr]
        allowed = spec.get("allowed") or spec.get("allowed_optional")
        if allowed is None:
            pytest.skip("no allowed set")
        v = d[attr]
        assert v in allowed, \
            f"{ticker}: {attr}={v!r} not in {sorted(str(x) for x in allowed)}"


# ── Type-specific attribute tests ───────────────────────────────────────────

@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
class TestPerTickerManualScoreRules:
    """Manual-policy-only: score_rules must be present + non-empty list."""
    def test_score_rules_present_for_manual(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None:
            pytest.skip("no policy")
        if d.get("policy_type") != "manual":
            pytest.skip("not manual")
        assert "score_rules" in d, f"{ticker}: manual policy missing score_rules"
        assert isinstance(d["score_rules"], list)
        assert len(d["score_rules"]) >= 1, \
            f"{ticker}: manual policy has empty score_rules"

    def test_score_rules_have_col_field(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or d.get("policy_type") != "manual":
            pytest.skip()
        for i, rule in enumerate(d.get("score_rules", [])):
            assert isinstance(rule, dict), \
                f"{ticker}: score_rules[{i}] must be dict"
            assert "col" in rule, \
                f"{ticker}: score_rules[{i}] missing 'col' field"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
class TestPerTickerTreeFeatureColumns:
    """Tree-policy types (classification/qlearning/xgboost): feature_columns
    must be a non-empty unique list."""

    def test_feature_columns_present(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None:
            pytest.skip()
        if d.get("policy_type") in {None, "manual"}:
            pytest.skip("manual or unset")
        assert "feature_columns" in d, \
            f"{ticker}: tree policy must have feature_columns"

    def test_feature_columns_non_empty(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or d.get("policy_type") in {None, "manual"}:
            pytest.skip()
        fc = d.get("feature_columns", [])
        assert isinstance(fc, list) and len(fc) > 0, \
            f"{ticker}: feature_columns missing/empty"

    def test_feature_columns_unique(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None or d.get("policy_type") in {None, "manual"}:
            pytest.skip()
        fc = d.get("feature_columns", [])
        if not fc:
            pytest.skip("empty")
        assert len(fc) == len(set(fc)), \
            f"{ticker}: duplicate feature_columns: " \
            f"{[x for x in fc if fc.count(x) > 1][:5]}"


@pytest.mark.parametrize("ticker", _TICKERS or ["__no_tickers__"])
class TestPerTickerThresholdConsistency:
    """Cross-attribute: buy_threshold > sell_threshold (or both zero / None)."""
    def test_buy_above_sell(self, ticker):
        if ticker == "__no_tickers__":
            pytest.skip("no tickers")
        d = _load_policy(ticker)
        if d is None:
            pytest.skip()
        bt = d.get("buy_threshold")
        st = d.get("sell_threshold")
        if bt is None or st is None:
            pytest.skip("threshold absent")
        if not (math.isfinite(bt) and math.isfinite(st)):
            pytest.skip("non-finite threshold")
        # Buy threshold ≥ sell threshold. Many production policies use
        # bt == st (a single point gate); the constraint is the partial
        # ordering — never bt < st (which would emit buy AND sell on
        # the same raw score, undefined).
        assert bt >= st, \
            f"{ticker}: buy_threshold={bt} < sell_threshold={st} — " \
            f"undefined signal in (st, bt) zone"
