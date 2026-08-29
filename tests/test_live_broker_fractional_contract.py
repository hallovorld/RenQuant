"""S-FRAC capability gate, leg (a): the REAL live broker exposes the contract.

``adapters/commit_contract.py::fractional_capability_gate`` requires — with
``execution.fractional_shares.enabled`` ON — that the broker object expose a
callable ``is_fractionable`` AND a callable no-submit classifier
(``classify_broker_result`` or ``is_no_submit_status``). The stage-0 tests in
``test_s_frac_stage0_commit_contract.py`` prove that contract against a
``FakeBroker``; this file carries the burden on the class the live runner
actually instantiates (``live/runner.py`` → ``live.alpaca_broker.AlpacaBroker``).

Everything here runs offline: no credentials, no network, no ``connect()``.
The broker is built with ``AlpacaBroker.__new__`` (the pattern
``test_live_alpaca_bounded_timeout.py`` already uses) and the asset lookup is
a stub recorded per call.

Semantics mirrored from renquant-execution (pin 91c7bf88,
``src/renquant_execution/alpaca_broker.py`` ``_lookup_fractionable`` /
``is_fractionable`` and ``src/renquant_execution/broker.py``
``NO_SUBMIT_STATUSES`` / ``is_no_submit_status``):
  * confirmed True/False verdicts are cached (one lookup per symbol);
  * a lookup failure answers False and is NOT cached (the next call retries);
  * ``is_no_submit_status`` is case/whitespace-insensitive membership in the
    owner vocabulary; unknown / empty / None → False.

Inertness pin: with the flag OFF (every config in the repo today) the gate
never reads a single broker attribute (commit_contract.py: the probes sit
under ``if enabled:``), so the live broker gaining these methods changes
nothing on the flag-off path.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY = REPO_ROOT / "backtesting" / "renquant_104"
for _p in (str(REPO_ROOT), str(_STRATEGY), str(REPO_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from adapters.commit_contract import fractional_capability_gate  # noqa: E402
from live import broker as live_broker  # noqa: E402
from live.alpaca_broker import AlpacaBroker, _FractionableLookupError  # noqa: E402
from live.broker import BaseBroker  # noqa: E402

FLAG_ON = {"execution": {"fractional_shares": {"enabled": True}}}
FLAG_OFF_CONFIGS = (
    None,
    {},
    {"execution": {}},
    {"execution": {"fractional_shares": {}}},
    {"execution": {"fractional_shares": {"enabled": False}}},
)


# ── Stubs ───────────────────────────────────────────────────────────────────

class _AssetClient:
    """Stand-in for alpaca-py TradingClient exposing only ``get_asset``.

    ``verdicts`` maps symbol -> bool (fractionable) or an Exception instance
    to raise. Every call is recorded so caching can be asserted exactly.
    """

    def __init__(self, verdicts: dict):
        self.verdicts = dict(verdicts)
        self.calls: list[str] = []

    def get_asset(self, symbol):
        self.calls.append(symbol)
        v = self.verdicts[symbol]
        if isinstance(v, Exception):
            raise v
        return SimpleNamespace(fractionable=v)


class _ExplodingClient:
    """Any attribute access proves the gate touched the broker's client."""

    def __getattr__(self, name):  # pragma: no cover — reaching here IS the failure
        raise AssertionError(f"broker client attribute {name!r} was accessed")


class _ArmedStops:
    def is_armed(self):
        return True


class _UnarmedStops:
    def is_armed(self):
        return False


def _broker(client=None) -> AlpacaBroker:
    """Offline AlpacaBroker: no __init__, no credentials, no connect()."""
    b = AlpacaBroker.__new__(AlpacaBroker)
    b._trading_client = client
    return b


# ── Leg (a): structural contract on the REAL class ──────────────────────────

class TestLiveBrokerExposesLegA:
    def test_class_exposes_exactly_what_the_gate_probes(self):
        # The gate's literal probes (commit_contract.py, under ``if enabled:``).
        assert callable(getattr(AlpacaBroker, "is_fractionable", None))
        assert callable(getattr(AlpacaBroker, "is_no_submit_status", None))

    def test_constructed_instance_exposes_the_contract_without_connecting(self):
        b = AlpacaBroker()  # env-derived empty credentials; no network
        assert b._trading_client is None
        assert callable(b.is_fractionable)
        assert callable(b.is_no_submit_status)
        assert b._fractionable_cache == {}

    def test_real_gate_passes_leg_a_for_the_live_broker(self):
        gate = fractional_capability_gate(FLAG_ON, _broker(), _ArmedStops())
        assert gate["enabled"] is True
        assert gate["ok"] is True
        assert gate["missing"] == []

    def test_real_gate_reports_only_the_stop_layer_when_stops_unarmed(self):
        """Leg (a) is satisfied by the live broker on its own; the only thing
        the gate still wants is leg (b), the software-stop layer."""
        gate = fractional_capability_gate(FLAG_ON, _broker(), _UnarmedStops())
        assert gate["ok"] is False
        assert gate["missing"] == ["software_stop_layer"]

    def test_base_broker_alone_does_not_satisfy_leg_a(self):
        """The base class carries the vocabulary classifier but deliberately
        NOT an ``is_fractionable`` default — a base default would let every
        broker pass the structural probe and turn the gate fail-open."""
        assert callable(getattr(BaseBroker, "is_no_submit_status", None))
        assert getattr(BaseBroker, "is_fractionable", None) is None

        class Bare(BaseBroker):
            def connect(self): ...
            def disconnect(self): ...
            def get_position(self, symbol): return 0.0
            def get_account_value(self): return 0.0
            def place_order(self, symbol, action, quantity): return {}

        gate = fractional_capability_gate(FLAG_ON, Bare(), _ArmedStops())
        assert gate["missing"] == ["broker_fractional_contract"]


# ── Inertness: the flag-off path never touches the broker ───────────────────

class TestFlagOffIsInert:
    @pytest.mark.parametrize("cfg", FLAG_OFF_CONFIGS)
    def test_gate_never_reads_the_broker_when_flag_off(self, cfg):
        b = _broker(_ExplodingClient())
        gate = fractional_capability_gate(cfg, b, None)
        assert gate == {
            "contract": "fractional-v2-stage0",
            "enabled": False,
            "ok": True,
            "missing": [],
        }
        assert "_fractionable_cache" not in b.__dict__  # lookup never ran

    def test_gate_flag_on_is_a_structural_probe_not_a_lookup(self):
        """Even with the flag ON the gate only checks callability — it never
        performs an asset lookup, so no network call hides in preflight."""
        b = _broker(_ExplodingClient())
        gate = fractional_capability_gate(FLAG_ON, b, _ArmedStops())
        assert gate["ok"] is True
        assert "_fractionable_cache" not in b.__dict__


# ── is_fractionable semantics (mirrors renquant-execution) ──────────────────

class TestIsFractionable:
    def test_true_verdict_is_cached(self):
        client = _AssetClient({"AAPL": True})
        b = _broker(client)
        assert b.is_fractionable("AAPL") is True
        assert b.is_fractionable("AAPL") is True
        assert client.calls == ["AAPL"]
        assert b._fractionable_cache == {"AAPL": True}

    def test_false_verdict_is_cached(self):
        client = _AssetClient({"BRK.A": False})
        b = _broker(client)
        assert b.is_fractionable("BRK.A") is False
        assert b.is_fractionable("BRK.A") is False
        assert client.calls == ["BRK.A"]
        assert b._fractionable_cache == {"BRK.A": False}

    def test_missing_fractionable_field_is_false_and_cached(self):
        """An asset payload without the field is a CONFIRMED "no" (owner
        semantics: ``bool(getattr(asset, "fractionable", False))``)."""
        class _Client:
            calls = 0
            def get_asset(self, symbol):
                self.calls += 1
                return SimpleNamespace()
        client = _Client()
        b = _broker(client)
        assert b.is_fractionable("XYZ") is False
        assert b.is_fractionable("XYZ") is False
        assert client.calls == 1
        assert b._fractionable_cache == {"XYZ": False}

    def test_lookup_failure_is_false_and_not_cached_so_it_retries(self):
        client = _AssetClient({"MSFT": RuntimeError("HTTP 503")})
        b = _broker(client)
        assert b.is_fractionable("MSFT") is False
        assert b._fractionable_cache == {}
        assert b.is_fractionable("MSFT") is False
        assert client.calls == ["MSFT", "MSFT"]  # second call re-queried
        # Transient failure clears → the confirmed verdict is now cached.
        client.verdicts["MSFT"] = True
        assert b.is_fractionable("MSFT") is True
        assert b.is_fractionable("MSFT") is True
        assert client.calls == ["MSFT", "MSFT", "MSFT"]
        assert b._fractionable_cache == {"MSFT": True}

    def test_lookup_failure_logs_loudly(self, caplog):
        b = _broker(_AssetClient({"MSFT": RuntimeError("HTTP 503")}))
        with caplog.at_level("WARNING", logger="live.alpaca_broker"):
            assert b.is_fractionable("MSFT") is False
        assert any("is_fractionable(MSFT): lookup failed" in r.getMessage()
                   for r in caplog.records)

    def test_not_connected_is_false_and_not_cached(self):
        b = _broker(None)
        assert b.is_fractionable("AAPL") is False
        assert b._fractionable_cache == {}
        with pytest.raises(_FractionableLookupError):
            b._lookup_fractionable("AAPL")

    def test_disconnect_does_not_poison_the_cache(self):
        client = _AssetClient({"AAPL": True})
        b = _broker(client)
        b._data_client = None
        assert b.is_fractionable("AAPL") is True
        b.disconnect()
        # Cached verdict still served; not-connected only affects NEW lookups.
        assert b.is_fractionable("AAPL") is True
        assert b.is_fractionable("NVDA") is False  # not connected → no lookup
        assert client.calls == ["AAPL"]

    def test_cache_key_is_case_insensitive(self):
        client = _AssetClient({"aapl": True})
        b = _broker(client)
        assert b.is_fractionable("aapl") is True
        assert b.is_fractionable("AAPL") is True
        assert client.calls == ["aapl"]
        assert b._fractionable_cache == {"AAPL": True}

    def test_lookup_raises_typed_error_with_cause(self):
        boom = ValueError("bad symbol")
        b = _broker(_AssetClient({"???": boom}))
        with pytest.raises(_FractionableLookupError) as ei:
            b._lookup_fractionable("???")
        assert ei.value.__cause__ is boom


# ── is_no_submit_status vocabulary ──────────────────────────────────────────

NO_SUBMIT_EXAMPLES = (
    "rejected_non_fractionable",
    "rejected_fractionable_lookup_failed",
    "rejected_precision_exceeds_9dp",
    "rejected_below_min_notional",
    "rejected_invalid_fractional_order",
    "skipped_non_fractionable_dust",
)
NOT_NO_SUBMIT_EXAMPLES = (
    "filled", "partially_filled", "accepted", "new", "pending_new",
    "rejected", "canceled", "submitted", "", None, 0, "unknown_status",
)


class TestIsNoSubmitStatus:
    @pytest.mark.parametrize("status", NO_SUBMIT_EXAMPLES)
    def test_vocabulary_members_classify_true(self, status):
        assert AlpacaBroker.is_no_submit_status(status) is True
        assert _broker().is_no_submit_status(status) is True
        assert BaseBroker.is_no_submit_status(status) is True

    @pytest.mark.parametrize("status", NOT_NO_SUBMIT_EXAMPLES)
    def test_non_members_classify_false(self, status):
        assert AlpacaBroker.is_no_submit_status(status) is False
        assert _broker().is_no_submit_status(status) is False

    def test_normalisation_is_case_and_whitespace_insensitive(self):
        assert AlpacaBroker.is_no_submit_status("  REJECTED_NON_FRACTIONABLE\n") is True

    def test_every_vocabulary_entry_round_trips(self):
        for status in live_broker.NO_SUBMIT_STATUSES:
            assert AlpacaBroker.is_no_submit_status(status) is True

    def test_vocabulary_source_is_declared(self):
        assert live_broker.NO_SUBMIT_VOCABULARY_SOURCE in (
            "renquant_execution", "local_fallback",
        )
        if live_broker.NO_SUBMIT_VOCABULARY_SOURCE == "local_fallback":
            assert live_broker.NO_SUBMIT_STATUSES is live_broker._FALLBACK_NO_SUBMIT_STATUSES

    def test_local_fallback_is_verbatim_owner_vocabulary(self):
        """Drift tripwire: the umbrella's fallback frozenset must equal the
        pinned renquant-execution vocabulary. Resolved via the sibling
        checkout in subrepos.lock.json (same pattern as _order_math_owner)."""
        from _order_math_owner import _inject_sibling_src_paths  # noqa: PLC0415
        _inject_sibling_src_paths()
        try:
            from renquant_execution import broker as owner  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            pytest.skip("renquant_execution.broker unavailable (no sibling checkout)")
        assert live_broker._FALLBACK_NO_SUBMIT_STATUSES == owner.NO_SUBMIT_STATUSES
        for status in list(owner.NO_SUBMIT_STATUSES) + list(NOT_NO_SUBMIT_EXAMPLES):
            assert (
                live_broker.is_no_submit_status(status)
                is owner.is_no_submit_status(status)
            ), status
