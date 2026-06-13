"""Property/invariant tests for normalize_fill_record (multi-broker schema map).

Eng plan S2 item 6 (test-ladder rebalance). The STATE-EXT-SELL attribution
suite (test_state_ext_sell_fill_attribution.py) exercises this normalizer only
INDIRECTLY, through lookup_ext_sell_fills. But normalize_fill_record is the
codex #76 fix that lets fill attribution work across the two in-repo brokers
with DIFFERENT keys (umbrella: action/avg_price/qty; execution subrepo:
filled_avg_price/filled_qty and no side field). A regression here silently
mis-projects a fill — wrong side fails the fail-closed guard, or a buy gets
read as a sell — so the projection deserves direct invariants.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): inputs
are swept over a deterministic seeded grid of mixed-schema fill dicts.

Invariants pinned:
- output schema is ALWAYS exactly {order_id, side, price, qty, filled_at}.
- side ∈ {"sell","buy",""}; "" IFF no truthy side/action key — the fail-closed
  signal the caller relies on (absent direction != "buy").
- price/qty are None or strictly positive; non-positive/unparseable candidates
  are skipped.
- first-valid-wins precedence across the per-field key lists (avg_price >
  fill_price > filled_avg_price; qty > filled_qty > fill_qty;
  order_id > id) — the property that makes cross-broker reads deterministic.
- cross-schema agreement: the umbrella and execution-subrepo encodings of the
  same economic fill normalize to the same price/qty.
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from adapters.runner_ext_sell import normalize_fill_record  # noqa: E402

SEED = 0xF111
N = 4000
_PRICE_KEYS = ("avg_price", "fill_price", "filled_avg_price")
_QTY_KEYS = ("qty", "filled_qty", "fill_qty")
_ID_KEYS = ("order_id", "id")
_SIDE_KEYS = ("side", "action")
_OUT_KEYS = {"order_id", "side", "price", "qty", "filled_at"}


def _rand_value(rng):
    return rng.choice([
        None, 0, -1, -3.5, 0.0, "", "x",
        rng.uniform(0.01, 5000), float(rng.randint(1, 9999)),
        str(rng.uniform(1, 100)),  # numeric-as-string, parseable
    ])


def _rand_fill(rng):
    f = {}
    for keys in (_PRICE_KEYS, _QTY_KEYS, _ID_KEYS):
        for k in keys:
            if rng.random() < 0.45:
                f[k] = _rand_value(rng)
    for k in _SIDE_KEYS:
        if rng.random() < 0.4:
            f[k] = rng.choice(["BUY", "SELL", "sell", "buy", "sell_short",
                               "", None, "hold", "xyz"])
    if rng.random() < 0.7:
        f["filled_at"] = rng.choice(["2026-01-01", "2026-06-13T09:30:00Z",
                                     "", None, 12345])
    return f


def _first_positive(f, keys):
    for k in keys:
        v = f.get(k)
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > 0:
            return fv
    return None


class TestSchemaAndTypes:

    def test_output_schema_is_fixed(self):
        rng = random.Random(SEED)
        for _ in range(N):
            out = normalize_fill_record(_rand_fill(rng))
            assert set(out.keys()) == _OUT_KEYS, out

    def test_side_domain_and_failclosed(self):
        rng = random.Random(SEED + 1)
        for _ in range(N):
            f = _rand_fill(rng)
            out = normalize_fill_record(f)
            assert out["side"] in ("sell", "buy", ""), out["side"]
            has_truthy_side = any(f.get(k) for k in _SIDE_KEYS)
            if not has_truthy_side:
                assert out["side"] == "", (f, out)

    def test_price_qty_none_or_positive(self):
        rng = random.Random(SEED + 2)
        for _ in range(N):
            out = normalize_fill_record(_rand_fill(rng))
            for field in ("price", "qty"):
                v = out[field]
                assert v is None or (isinstance(v, float) and v > 0), (field, v)

    def test_filled_at_is_str(self):
        rng = random.Random(SEED + 3)
        for _ in range(N):
            out = normalize_fill_record(_rand_fill(rng))
            assert isinstance(out["filled_at"], str)


class TestPrecedence:

    def test_price_first_valid_wins(self):
        rng = random.Random(SEED + 4)
        for _ in range(N):
            f = _rand_fill(rng)
            assert normalize_fill_record(f)["price"] == _first_positive(f, _PRICE_KEYS)

    def test_qty_first_valid_wins(self):
        rng = random.Random(SEED + 5)
        for _ in range(N):
            f = _rand_fill(rng)
            assert normalize_fill_record(f)["qty"] == _first_positive(f, _QTY_KEYS)

    def test_order_id_precedence_and_stringified(self):
        rng = random.Random(SEED + 6)
        for _ in range(N):
            f = _rand_fill(rng)
            out = normalize_fill_record(f)
            expected = None
            for k in _ID_KEYS:
                if f.get(k):
                    expected = str(f[k])
                    break
            assert out["order_id"] == expected, (f, out)

    def test_explicit_precedence_table(self):
        # avg_price beats filled_avg_price; first positive qty key wins.
        out = normalize_fill_record(
            {"avg_price": 10.0, "filled_avg_price": 99.0,
             "qty": 5, "filled_qty": 7, "order_id": "A", "id": "B"})
        assert out["price"] == 10.0
        assert out["qty"] == 5.0
        assert out["order_id"] == "A"
        # when the higher-precedence key is non-positive, fall through.
        out2 = normalize_fill_record(
            {"avg_price": 0, "filled_avg_price": 99.0,
             "qty": -1, "filled_qty": 7, "id": "B"})
        assert out2["price"] == 99.0
        assert out2["qty"] == 7.0
        assert out2["order_id"] == "B"


class TestSideDetection:

    def test_sell_and_buy_substring_match(self):
        for raw, want in [("SELL", "sell"), ("sell_short", "sell"),
                          ("BUY", "buy"), ("buy_to_cover", "buy"),
                          ("hold", ""), ("", ""), ("xyz", "")]:
            assert normalize_fill_record({"action": raw})["side"] == want, raw
            assert normalize_fill_record({"side": raw})["side"] == want, raw

    def test_side_key_precedence(self):
        # "side" is consulted before "action"; first TRUTHY one wins.
        assert normalize_fill_record({"side": "SELL", "action": "BUY"})["side"] == "sell"
        assert normalize_fill_record({"side": "", "action": "BUY"})["side"] == "buy"


class TestCrossSchemaAgreement:

    def test_umbrella_and_execution_encodings_agree_on_economics(self):
        """The same economic fill, written in the umbrella schema vs the
        execution-subrepo schema, normalizes to identical price/qty/order_id.
        Only `side` differs (execution rows carry no direction)."""
        rng = random.Random(SEED + 7)
        for _ in range(N):
            price = float(rng.randint(1, 5000))
            qty = float(rng.randint(1, 1000))
            oid = f"ord{rng.randint(0, 1_000_000)}"
            at = "2026-06-13T09:30:00Z"
            umbrella = {"action": "SELL", "avg_price": price, "qty": qty,
                        "order_id": oid, "filled_at": at}
            execution = {"filled_avg_price": price, "filled_qty": qty,
                         "id": oid, "status": "filled", "filled_at": at}
            nu = normalize_fill_record(umbrella)
            ne = normalize_fill_record(execution)
            assert (nu["price"], nu["qty"], nu["order_id"], nu["filled_at"]) == \
                   (ne["price"], ne["qty"], ne["order_id"], ne["filled_at"])
            assert nu["side"] == "sell" and ne["side"] == ""  # fail-closed
