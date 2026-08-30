"""Buys are sized on SETTLED cash; ``execution.buying_power_mode`` is honoured.

Incident (2026-08-27 / 08-28, live alpaca book ~$10.8k): ``AlpacaBroker
.get_cash()`` was hard-wired to ``account.non_marginable_buying_power``
(settled cash + UNSETTLED sell proceeds). On a margin account that figure lets
a buy clear against proceeds that have not settled and the broker fronts the
difference as a margin debit: HPE ($1,034) was bought with settled cash of
$33, then WELL ($1,904) + NET with settled cash of −$1,140 — the book ended
1.11x on margin. The strategy config had declared
``execution.buying_power_mode`` for sim/live parity; the sim read it, the
live adapter never did.

Contract pinned here (fix/size-on-settled-cash):
  * default (key absent) = ``settled_cash`` = ``account.cash`` floored at 0;
  * ``non_marginable_buying_power`` / ``buying_power`` honoured ONLY when
    explicitly configured;
  * negative settled cash → $0 budget, reason ``no_settled_cash``, no buys;
  * the runner logs ``cash=… nmbp=… mode=…`` on every read;
  * same-bar (T+1-unsettled) sell proceeds are NOT credited to the buy budget
    under settled-cash sizing;
  * the vocabulary equals the sim's (``adapters/sim_order_helpers.py``,
    ``adapters/lean_account.py``) so one config key drives both paths.

No network, no credentials: the alpaca trading client is a SimpleNamespace.
Balance figures below are SYNTHETIC stand-ins shaped like the incident, not
account records.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from live.broker import (  # noqa: E402
    BUYING_POWER_MODE_ALIASES,
    BUYING_POWER_MODE_MARGIN,
    BUYING_POWER_MODE_NMBP,
    BUYING_POWER_MODE_SETTLED,
    DEFAULT_BUYING_POWER_MODE,
    NO_BUYING_POWER_REASON,
    NO_SETTLED_CASH_REASON,
    UNREADABLE_CASH_REASON,
    normalize_buying_power_mode,
    resolve_sizing_cash,
)
from live.alpaca_broker import AlpacaBroker  # noqa: E402
from live.broker_readonly import ReadOnlyBrokerWrapper  # noqa: E402
from live.paper_broker import PaperBroker  # noqa: E402
from adapters.runner_execmath import (  # noqa: E402
    resolve_buy_sizing_cash,
    unsettled_proceeds_spendable,
)

RUNNER_SOURCE = (STRATEGY_DIR / "adapters" / "runner.py").read_text()
ALPACA_SOURCE = (REPO / "live" / "alpaca_broker.py").read_text()

# Shaped like the 08-28 close: settled cash negative (on margin), unsettled
# proceeds make nmbp positive, margin buying power larger still. SYNTHETIC.
ON_MARGIN = dict(cash="-1139.70", non_marginable_buying_power="1904.00",
                 buying_power="21600.00", equity="10800.00")
# Shaped like 08-27: $33 settled, ~$1k of unsettled proceeds. SYNTHETIC.
THIN_SETTLED = dict(cash="33.00", non_marginable_buying_power="1034.00",
                    buying_power="20000.00", equity="10800.00")


class _Client:
    """Stand-in for alpaca-py TradingClient: counts account reads."""

    def __init__(self, **fields):
        self.account = SimpleNamespace(**fields)
        self.reads = 0

    def get_account(self):
        self.reads += 1
        return self.account


def _alpaca(fields: dict, **ctor) -> tuple[AlpacaBroker, _Client]:
    b = AlpacaBroker(api_key="k", secret_key="s", paper=True, **ctor)
    client = _Client(**fields)
    b._trading_client = client
    return b, client


# ── Vocabulary ───────────────────────────────────────────────────────────────

class TestNormalizeBuyingPowerMode:
    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_absent_is_settled_cash(self, raw):
        assert normalize_buying_power_mode(raw) == BUYING_POWER_MODE_SETTLED
        assert DEFAULT_BUYING_POWER_MODE == BUYING_POWER_MODE_SETTLED

    @pytest.mark.parametrize("raw", ["cash", "settled", "settled_cash", " Cash "])
    def test_settled_aliases(self, raw):
        assert normalize_buying_power_mode(raw) == BUYING_POWER_MODE_SETTLED

    @pytest.mark.parametrize("raw", [
        "non_marginable_buying_power", "cash_plus_unsettled", "unsettled",
    ])
    def test_nmbp_aliases(self, raw):
        assert normalize_buying_power_mode(raw) == BUYING_POWER_MODE_NMBP

    @pytest.mark.parametrize("raw", ["buying_power", "margin"])
    def test_margin_aliases(self, raw):
        assert normalize_buying_power_mode(raw) == BUYING_POWER_MODE_MARGIN

    @pytest.mark.parametrize("raw", ["2x", "nmbp", "account_cash", 0, 1.5])
    def test_unrecognised_raises(self, raw):
        with pytest.raises(ValueError, match="buying_power_mode"):
            normalize_buying_power_mode(raw)


class TestVocabularyMatchesSim:
    """One key, one vocabulary: every alias the sim accepts, live accepts to
    the same canonical name. (Live additionally accepts ``buying_power`` —
    the sim has no margin model and must keep rejecting it.)"""

    def test_sim_order_helpers_aliases_are_a_subset(self):
        from adapters.sim_order_helpers import _BUYING_POWER_ALIASES as sim
        for raw, canonical in sim.items():
            assert BUYING_POWER_MODE_ALIASES[raw] == canonical, raw

    def test_lean_account_aliases_are_a_subset(self):
        from adapters.lean_account import _BUYING_POWER_ALIASES as lean
        for raw, canonical in lean.items():
            assert BUYING_POWER_MODE_ALIASES[raw] == canonical, raw

    def test_sim_rejects_the_margin_mode_live_accepts(self):
        from adapters.sim_order_helpers import _normalize_buying_power_mode as sim_norm
        with pytest.raises(ValueError):
            sim_norm(BUYING_POWER_MODE_MARGIN)

    def test_every_repo_config_declares_a_value_both_paths_accept(self):
        from adapters.sim_order_helpers import _normalize_buying_power_mode as sim_norm
        configs = sorted(STRATEGY_DIR.glob("strategy_config*.json"))
        assert configs, "no strategy configs found"
        for cfg_path in configs:
            cfg = json.loads(cfg_path.read_text())
            raw = (cfg.get("execution") or {}).get("buying_power_mode")
            if raw is None:
                continue
            assert normalize_buying_power_mode(raw) == sim_norm(raw), cfg_path.name


# ── Pure resolver ────────────────────────────────────────────────────────────

class TestResolveSizingCash:
    def test_settled_negative_floors_to_zero_with_no_settled_cash(self):
        out = resolve_sizing_cash("settled_cash", **{
            "settled_cash": -1139.70,
            "non_marginable_buying_power": 1904.0,
            "buying_power": 21600.0,
        })
        assert out["sizing_cash"] == 0.0
        assert out["sizing_reason"] == NO_SETTLED_CASH_REASON
        assert out["sizing_source"] == "account.cash"
        assert out["settled_cash"] == pytest.approx(-1139.70)
        assert out["non_marginable_buying_power"] == pytest.approx(1904.0)

    def test_settled_thin_is_not_topped_up_by_unsettled_proceeds(self):
        out = resolve_sizing_cash("settled_cash", settled_cash=33.0,
                                  non_marginable_buying_power=1034.0)
        assert out["sizing_cash"] == 33.0
        assert out["sizing_reason"] is None

    def test_nmbp_mode_uses_nmbp(self):
        out = resolve_sizing_cash("non_marginable_buying_power",
                                  settled_cash=-1139.70,
                                  non_marginable_buying_power=1904.0)
        assert out["sizing_cash"] == 1904.0
        assert out["sizing_source"] == "account.non_marginable_buying_power"

    def test_nmbp_unavailable_degrades_to_settled_never_larger(self):
        out = resolve_sizing_cash("non_marginable_buying_power",
                                  settled_cash=500.0,
                                  non_marginable_buying_power=None,
                                  buying_power=9000.0)
        assert out["sizing_cash"] == 500.0
        assert "unavailable" in out["sizing_source"]

    def test_nmbp_nonpositive_is_no_buying_power(self):
        out = resolve_sizing_cash("non_marginable_buying_power",
                                  settled_cash=-50.0,
                                  non_marginable_buying_power=0.0)
        assert out["sizing_cash"] == 0.0
        assert out["sizing_reason"] == NO_BUYING_POWER_REASON

    def test_margin_mode_chain(self):
        assert resolve_sizing_cash("buying_power", settled_cash=1.0,
                                   non_marginable_buying_power=2.0,
                                   buying_power=4.0)["sizing_cash"] == 4.0
        assert resolve_sizing_cash("buying_power", settled_cash=1.0,
                                   non_marginable_buying_power=2.0,
                                   buying_power=None)["sizing_cash"] == 2.0
        assert resolve_sizing_cash("buying_power", settled_cash=1.0,
                                   non_marginable_buying_power=None,
                                   buying_power=None)["sizing_cash"] == 1.0

    @pytest.mark.parametrize("bad", [None, "nan", "inf", "n/a", object()])
    def test_unparseable_settled_is_unreadable_not_zero_cash(self, bad):
        out = resolve_sizing_cash("settled_cash", settled_cash=bad)
        assert out["sizing_cash"] == 0.0
        assert out["sizing_reason"] == UNREADABLE_CASH_REASON
        assert out["settled_cash"] is None

    def test_mode_is_normalised(self):
        assert resolve_sizing_cash("cash", settled_cash=7.0)["mode"] == BUYING_POWER_MODE_SETTLED
        with pytest.raises(ValueError):
            resolve_sizing_cash("2x", settled_cash=7.0)


# ── AlpacaBroker (the module live/runner.py imports) ─────────────────────────

class TestAlpacaBrokerGetCash:
    def test_default_mode_is_settled_cash(self):
        b, _ = _alpaca(THIN_SETTLED)
        assert b.buying_power_mode == BUYING_POWER_MODE_SETTLED

    def test_on_margin_get_cash_is_zero_not_nmbp(self):
        """The 08-28 shape: pre-fix this returned 1904.00 and WELL was bought."""
        b, _ = _alpaca(ON_MARGIN)
        assert b.get_cash() == 0.0
        snap = b.get_buying_power_snapshot()
        assert snap["sizing_reason"] == NO_SETTLED_CASH_REASON
        assert snap["settled_cash"] == pytest.approx(-1139.70)
        assert snap["non_marginable_buying_power"] == pytest.approx(1904.0)

    def test_thin_settled_get_cash_is_settled_not_nmbp(self):
        """The 08-27 shape: pre-fix this returned 1034.00 and HPE was bought."""
        b, _ = _alpaca(THIN_SETTLED)
        assert b.get_cash() == 33.0

    def test_explicit_nmbp_mode_is_honoured(self):
        b, _ = _alpaca(THIN_SETTLED, buying_power_mode="non_marginable_buying_power")
        assert b.buying_power_mode == BUYING_POWER_MODE_NMBP
        assert b.get_cash() == 1034.0
        b2, _ = _alpaca(ON_MARGIN)
        assert b2.get_buying_power_snapshot("non_marginable_buying_power")["sizing_cash"] == 1904.0

    def test_margin_mode_is_honoured_and_warned(self, caplog):
        b, _ = _alpaca(THIN_SETTLED)
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            snap = b.get_buying_power_snapshot("buying_power")
        assert snap["sizing_cash"] == 20000.0
        assert any("MARGIN" in r.getMessage() for r in caplog.records)

    def test_nmbp_field_missing_degrades_to_settled_with_warning(self, caplog):
        b, _ = _alpaca(dict(cash="250.00", equity="1000"))
        with caplog.at_level(logging.WARNING, logger="live.alpaca_broker"):
            snap = b.get_buying_power_snapshot("non_marginable_buying_power")
        assert snap["sizing_cash"] == 250.0
        assert snap["non_marginable_buying_power"] is None
        assert any("unavailable" in r.getMessage() for r in caplog.records)

    def test_one_account_read_per_snapshot(self):
        b, client = _alpaca(THIN_SETTLED)
        b.get_buying_power_snapshot()
        assert client.reads == 1
        b.get_cash()
        assert client.reads == 2

    def test_unrecognised_ctor_mode_fails_at_construction(self):
        with pytest.raises(ValueError):
            AlpacaBroker(api_key="k", secret_key="s", paper=True, buying_power_mode="2x")

    def test_old_unconditional_nmbp_return_is_gone(self):
        """Verify the OLD text is gone, not just that new text exists."""
        assert "return float(nmbp)" not in ALPACA_SOURCE
        assert "falling back to settled cash (pending settlement NOT counted)" not in ALPACA_SOURCE


class TestReadOnlyWrapperForwards:
    def test_snapshot_forwarded_with_mode(self, monkeypatch):
        monkeypatch.delenv("RENQUANT_READONLY_TAG", raising=False)
        real, _ = _alpaca(ON_MARGIN)
        w = ReadOnlyBrokerWrapper(real)
        assert w.get_buying_power_snapshot()["sizing_cash"] == 0.0
        assert w.get_buying_power_snapshot("non_marginable_buying_power")["sizing_cash"] == 1904.0
        assert w.get_cash() == 0.0


class TestBaseBrokerDefault:
    """A broker with no unsettled/margin figures: get_cash() IS settled cash."""

    def test_paper_broker_snapshot(self):
        b = PaperBroker(initial_cash=5000.0)
        snap = b.get_buying_power_snapshot()
        assert snap["mode"] == BUYING_POWER_MODE_SETTLED
        assert snap["sizing_cash"] == 5000.0
        assert snap["non_marginable_buying_power"] is None
        # nmbp mode on a broker that cannot report it: conservative.
        assert b.get_buying_power_snapshot("non_marginable_buying_power")["sizing_cash"] == 5000.0


# ── Adapter-side helper (what runner.make_context calls) ─────────────────────

class _LegacyBroker:
    """A test double with only get_cash(): treated as settled cash."""

    def __init__(self, cash):
        self._cash = cash

    def get_cash(self):
        return self._cash


class TestResolveBuySizingCash:
    def test_key_absent_sizes_on_settled_cash(self, caplog):
        b, _ = _alpaca(THIN_SETTLED)
        with caplog.at_level(logging.INFO, logger="adapters.runner"):
            snap = resolve_buy_sizing_cash(b, {"execution": {}})
        assert snap["mode"] == BUYING_POWER_MODE_SETTLED
        assert snap["sizing_cash"] == 33.0
        assert snap["configured_mode"] is None
        line = next(r.getMessage() for r in caplog.records
                    if r.getMessage().startswith("runner: buy-sizing"))
        assert "cash=33.00" in line and "nmbp=1034.00" in line
        assert "mode=settled_cash" in line

    def test_no_execution_section_at_all(self):
        b, _ = _alpaca(THIN_SETTLED)
        assert resolve_buy_sizing_cash(b, {})["sizing_cash"] == 33.0
        assert resolve_buy_sizing_cash(b, None)["sizing_cash"] == 33.0

    def test_explicit_nmbp_is_honoured(self):
        b, _ = _alpaca(ON_MARGIN)
        cfg = {"execution": {"buying_power_mode": "non_marginable_buying_power"}}
        snap = resolve_buy_sizing_cash(b, cfg)
        assert snap["mode"] == BUYING_POWER_MODE_NMBP
        assert snap["sizing_cash"] == 1904.0
        assert snap["configured_mode"] == "non_marginable_buying_power"

    def test_cash_alias_is_settled(self):
        b, _ = _alpaca(ON_MARGIN)
        snap = resolve_buy_sizing_cash(b, {"execution": {"buying_power_mode": "cash"}})
        assert snap["sizing_cash"] == 0.0
        assert snap["sizing_reason"] == NO_SETTLED_CASH_REASON

    def test_negative_cash_is_zero_budget_and_warns(self, caplog):
        b, _ = _alpaca(ON_MARGIN)
        with caplog.at_level(logging.WARNING, logger="adapters.runner"):
            snap = resolve_buy_sizing_cash(b, {"execution": {}})
        assert snap["sizing_cash"] == 0.0
        assert snap["sizing_reason"] == NO_SETTLED_CASH_REASON
        assert any("BUY budget is $0" in r.getMessage()
                   and NO_SETTLED_CASH_REASON in r.getMessage()
                   for r in caplog.records)

    def test_unrecognised_mode_raises_so_runner_fail_safes(self):
        b, _ = _alpaca(THIN_SETTLED)
        with pytest.raises(ValueError):
            resolve_buy_sizing_cash(b, {"execution": {"buying_power_mode": "2x"}})

    def test_legacy_broker_get_cash_is_settled_cash(self):
        snap = resolve_buy_sizing_cash(_LegacyBroker(120.0), {"execution": {}})
        assert snap["sizing_cash"] == 120.0
        assert snap["non_marginable_buying_power"] is None
        assert snap["broker_api"].startswith("get_cash")
        # nmbp mode on such a broker cannot exceed what get_cash() said.
        cfg = {"execution": {"buying_power_mode": "non_marginable_buying_power"}}
        assert resolve_buy_sizing_cash(_LegacyBroker(120.0), cfg)["sizing_cash"] == 120.0
        neg = resolve_buy_sizing_cash(_LegacyBroker(-5.0), {"execution": {}})
        assert neg["sizing_cash"] == 0.0 and neg["sizing_reason"] == NO_SETTLED_CASH_REASON


class TestUnsettledProceedsSpendable:
    @pytest.mark.parametrize("mode", [None, "", "settled_cash", "cash", "garbage", "unavailable"])
    def test_conservative_default(self, mode):
        assert unsettled_proceeds_spendable(mode) is False

    @pytest.mark.parametrize("mode", ["non_marginable_buying_power", "buying_power"])
    def test_modes_that_count_unsettled_by_definition(self, mode):
        assert unsettled_proceeds_spendable(mode) is True


# ── Runner wiring (source pins; the end-to-end context needs strategy deps) ──

class TestRunnerWiring:
    def test_make_context_resolves_via_the_helper_and_records_on_ctx(self):
        assert "buy_sizing = resolve_buy_sizing_cash(broker, self._config)" in RUNNER_SOURCE
        assert 'cash = float(buy_sizing["sizing_cash"])' in RUNNER_SOURCE
        assert "ctx.buy_sizing_cash = dict(buy_sizing)" in RUNNER_SOURCE

    def test_old_hardwired_call_is_gone(self):
        assert "cash = broker.get_cash()" not in RUNNER_SOURCE

    def test_fail_safe_zero_survives(self):
        i = RUNNER_SOURCE.index("buy_sizing = resolve_buy_sizing_cash(")
        block = RUNNER_SOURCE[i:i + 1500]
        assert "cash = 0.0" in block and "log.error" in block
        assert '"sizing_reason": "cash_read_failed"' in block

    def test_same_bar_sell_credit_is_gated_before_it_is_added(self):
        gate = RUNNER_SOURCE.index("not unsettled_proceeds_spendable(buy_sizing_mode)")
        add = RUNNER_SOURCE.index("buy_cash_remaining += sell_credit")
        assert gate < add
        assert "sell_credit = 0.0" in RUNNER_SOURCE[gate:add]

    def test_zero_budget_skips_carry_the_named_reason(self):
        assert '"skip_reason": buy_budget_reason' in RUNNER_SOURCE
        assert "if buy_cash_remaining <= 0.0 and buy_budget_reason:" in RUNNER_SOURCE

    def test_no_trade_rollup_surfaces_the_named_reason(self):
        src = (REPO / "live" / "runner.py").read_text()
        assert 'buy_sizing = getattr(ctx, "buy_sizing_cash", None)' in src
        assert 'return str(buy_sizing["sizing_reason"])' in src
