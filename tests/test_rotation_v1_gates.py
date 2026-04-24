"""Rotation V1 gates — raw_advantage depth + persistence_bars.

User hypothesis (2026-04-24): current rotations subtract ~-2.5 APY
because the net-adv threshold alone can clear on marginal, transient
signal-vs-noise edges. V1 adds two flag-gated filters:

  rotation.min_raw_advantage_pct — pre-tax, pre-cost edge floor
  rotation.persistence_bars      — same pair must appear in prior N bars

Both default off (0.0 and 0). When on, they compose with the existing
net-adv threshold: a pair must clear ALL gates to fire.
"""
from __future__ import annotations

import datetime
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


@dataclass
class _Cand:
    ticker: str
    rank_score: float
    expected_return: float


def _base_meta(entry_date="2025-01-01", entry_price=100.0, current_price=100.0):
    # Default: no unrealized gain → no tax drag. Tests that care about
    # tax should set current_price explicitly.
    return {
        "entry_date":    datetime.date.fromisoformat(entry_date),
        "entry_price":   entry_price,
        "current_price": current_price,
    }


def _default_cfg(**overrides):
    cfg = {
        "enabled":                      True,
        "min_expected_advantage_pct":   0.01,   # loose enough that ER gate alone fires
        "target_horizon_days":          20,
        "transaction_cost_pct":         0.0,
        "min_rotation_hold_days":       0,       # tests don't care about hold
        "lt_protection_days":           0,
        "max_rotations_per_bar":        2,
    }
    cfg.update(overrides)
    return cfg


def _held(ticker, score=0.30, er=-0.02, entry_price=100.0, current_price=100.0):
    """Build the 3 dicts the kernel primitive wants for one held position."""
    return (
        {ticker: score},
        {ticker: er},
        {ticker: _base_meta(entry_price=entry_price, current_price=current_price)},
    )


class TestMinRawAdvantage:
    def test_raw_adv_below_floor_blocks(self):
        """raw_adv = cand.er - held.er. If floor=0.05 and raw=0.03, reject."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.01)
        cand = _Cand("AMD", rank_score=0.40, expected_return=0.02)  # raw=0.03

        cfg = _default_cfg(min_raw_advantage_pct=0.05)
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert pairs == []

    def test_raw_adv_above_floor_permits(self):
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)  # raw=0.10

        cfg = _default_cfg(min_raw_advantage_pct=0.05)
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert len(pairs) == 1
        assert pairs[0].sell_ticker == "NVDA"
        assert pairs[0].buy_ticker == "AMD"

    def test_default_zero_gate_is_off(self):
        """min_raw_advantage_pct=0 → never rejects, matches pre-V1 behavior."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.00)
        cand = _Cand("AMD", rank_score=0.40, expected_return=0.02)  # raw=0.02

        cfg = _default_cfg()  # no min_raw_advantage_pct
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert len(pairs) == 1


class TestPersistenceBars:
    def test_no_history_blocks(self):
        """persistence_bars=3 with 0 history → can't fire (fail closed)."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg(persistence_bars=3)
        cfg["_prior_proposals"] = []   # empty history
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert pairs == []

    def test_partial_history_blocks(self):
        """persistence_bars=3 with 2 bars of history → still blocks."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg(persistence_bars=3)
        cfg["_prior_proposals"] = [{("NVDA", "AMD")}, {("NVDA", "AMD")}]
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert pairs == []

    def test_full_history_with_consistent_pair_fires(self):
        """persistence_bars=3 with 3 bars of (NVDA, AMD) → fires."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg(persistence_bars=3)
        cfg["_prior_proposals"] = [
            {("NVDA", "AMD")},
            {("NVDA", "AMD")},
            {("NVDA", "AMD")},
        ]
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert len(pairs) == 1

    def test_history_with_different_pair_blocks(self):
        """persistence_bars=3, but prior bars proposed a DIFFERENT pair."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg(persistence_bars=3)
        cfg["_prior_proposals"] = [
            {("TSLA", "MSFT")},
            {("TSLA", "MSFT")},
            {("NVDA", "AMD")},   # only the latest matches
        ]
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert pairs == []

    def test_default_zero_disables_gate(self):
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg()   # persistence_bars default = 0
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert len(pairs) == 1


class TestBothGatesCompose:
    def test_all_must_pass(self):
        """raw_adv gate OK but persistence empty → blocked."""
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)

        cfg = _default_cfg(min_raw_advantage_pct=0.05, persistence_bars=2)
        cfg["_prior_proposals"] = []   # persistence blocks
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert pairs == []

    def test_both_gates_pass_fires(self):
        from kernel.rotation import find_rotation_pairs

        hs, he, hm = _held("NVDA", er=-0.05)
        cand = _Cand("AMD", rank_score=0.50, expected_return=0.05)   # raw=0.10

        cfg = _default_cfg(min_raw_advantage_pct=0.05, persistence_bars=2)
        cfg["_prior_proposals"] = [{("NVDA", "AMD")}, {("NVDA", "AMD")}]
        pairs = find_rotation_pairs(
            held_scores=hs, held_er=he, held_meta=hm,
            candidates=[cand], today=datetime.date(2025, 6, 1),
            rotation_cfg=cfg, tax_cfg={},
        )
        assert len(pairs) == 1
