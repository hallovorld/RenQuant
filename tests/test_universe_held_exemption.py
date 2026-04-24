"""Plan V regression — HELD tickers are exempt from universe_floor.

**The bug (2026-04-23 live):** AMZN held at cost $249. AMZN's per-ticker
model sharpe slipped to 0.668, below `universe_floor.threshold=1.0`.
`FilterUniverseFloorTask` dropped AMZN → `ctx.models["AMZN"]` missing
→ `ScoreModelTask` saw `tc.model = None` → `model_action = "hold"`
forever → the `model_sell_streak` exit path was **unreachable** for
AMZN. Only non-model exits (stop_loss / trailing / max_hold) could
ever trigger. A flat low-vol held position became structurally
un-sellable via signals.

**The design intent:** universe_floor gates OFFENSIVE new buys (don't
open new positions with weak models). Held positions need their models
*precisely so* exit signals can fire.

This suite pins:
  1. `FilterUniverseFloorTask` admits currently-held tickers even when
     their floor metric is below threshold.
  2. It reads held tickers from `live_state.json::position_hwm` (no
     broker coupling).
  3. Non-held tickers below threshold are still filtered (existing
     behaviour preserved).
  4. Defensives + held are both admitted (union, not XOR).
  5. Missing live_state.json → empty held → no change in behaviour.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

from kernel.pipeline.job_universe import (  # noqa: E402
    FilterUniverseFloorTask,
    UniverseContext,
    _load_held_tickers,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _art(sharpe: float | None = None) -> dict:
    meta: dict = {}
    if sharpe is not None:
        meta["sharpe"] = sharpe
    return {"_metadata": meta}


def _ctx(*, tmp_path: Path, loaded: dict, held: set[str] | None = None,
         defensives: list[str] | None = None, threshold: float = 1.0) -> UniverseContext:
    """Build a UniverseContext with an on-disk live_state.json stub."""
    state_file = tmp_path / "live_state.json"
    state_file.write_text(json.dumps({
        "position_hwm": {t: 100.0 for t in (held or set())},
    }))
    return UniverseContext(
        config={
            "ranking": {"universe_floor": {
                "type":      "sharpe",
                "threshold": threshold,
            }},
            "defensive_tickers": defensives or [],
        },
        strategy_dir=tmp_path,
        loaded_models=dict(loaded),
    )


# ── _load_held_tickers ───────────────────────────────────────────────────────

class TestLoadHeldTickers:
    def test_empty_when_file_missing(self, tmp_path):
        assert _load_held_tickers(tmp_path) == set()

    def test_reads_position_hwm_keys(self, tmp_path):
        (tmp_path / "live_state.json").write_text(json.dumps({
            "position_hwm": {"AMZN": 250.0, "CAT": 840.0, "XLU": 46.0},
        }))
        assert _load_held_tickers(tmp_path) == {"AMZN", "CAT", "XLU"}

    def test_handles_missing_key(self, tmp_path):
        (tmp_path / "live_state.json").write_text(json.dumps({
            "regime": "BULL_CALM",
        }))
        assert _load_held_tickers(tmp_path) == set()

    def test_handles_corrupt_json(self, tmp_path):
        (tmp_path / "live_state.json").write_text("not json")
        assert _load_held_tickers(tmp_path) == set()


# ── FilterUniverseFloorTask — held exemption ────────────────────────────────

class TestHeldExempt:
    def test_held_below_floor_still_admitted(self, tmp_path):
        """The AMZN 2026-04-23 scenario: held + sharpe below floor."""
        ctx = _ctx(
            tmp_path   = tmp_path,
            loaded     = {"AMZN": _art(sharpe=0.668)},
            held       = {"AMZN"},
            threshold  = 1.0,
        )
        FilterUniverseFloorTask().run(ctx)
        assert "AMZN" in ctx.loaded_models, (
            "Held AMZN must stay loaded even with sharpe < floor — "
            "otherwise model_sell_streak exit path is dead"
        )

    def test_non_held_below_floor_still_filtered(self, tmp_path):
        """Confirm the existing behaviour is preserved for non-held."""
        ctx = _ctx(
            tmp_path  = tmp_path,
            loaded    = {
                "AMZN": _art(sharpe=0.668),   # held → admit
                "NVO":  _art(sharpe=-0.403),  # not held → filter
            },
            held      = {"AMZN"},
            threshold = 1.0,
        )
        FilterUniverseFloorTask().run(ctx)
        assert "AMZN" in ctx.loaded_models
        assert "NVO" not in ctx.loaded_models
        rejection_tickers = {t for t, _ in ctx.rejections}
        assert "NVO" in rejection_tickers and "AMZN" not in rejection_tickers

    def test_defensive_and_held_both_admitted(self, tmp_path):
        """Union, not XOR: defensives AND held positions all pass."""
        ctx = _ctx(
            tmp_path   = tmp_path,
            loaded     = {
                "AMZN": _art(sharpe=0.668),    # held
                "XLU":  _art(sharpe=0.250),    # defensive
                "NVO":  _art(sharpe=-0.403),   # neither → filter
            },
            held       = {"AMZN"},
            defensives = ["XLU"],
            threshold  = 1.0,
        )
        FilterUniverseFloorTask().run(ctx)
        assert "AMZN" in ctx.loaded_models
        assert "XLU"  in ctx.loaded_models
        assert "NVO"  not in ctx.loaded_models

    def test_held_above_floor_no_special_path(self, tmp_path):
        """Normal case: held + above threshold → admitted (was already OK)."""
        ctx = _ctx(
            tmp_path  = tmp_path,
            loaded    = {"CAT": _art(sharpe=2.04)},
            held      = {"CAT"},
            threshold = 1.0,
        )
        FilterUniverseFloorTask().run(ctx)
        assert "CAT" in ctx.loaded_models
        assert ctx.rejections == []

    def test_empty_live_state_same_as_no_held_exemption(self, tmp_path):
        """If live_state.json is missing, behaviour reverts to
        defensives-only exemption (existing pre-Plan-V behaviour)."""
        # No live_state.json here
        ctx = UniverseContext(
            config={
                "ranking": {"universe_floor": {"type": "sharpe", "threshold": 1.0}},
                "defensive_tickers": [],
            },
            strategy_dir=tmp_path,
            loaded_models={"AMZN": _art(sharpe=0.668)},
        )
        FilterUniverseFloorTask().run(ctx)
        # AMZN should be filtered (no held-exemption safety net)
        assert "AMZN" not in ctx.loaded_models


# ── 2026-04-23 replay ───────────────────────────────────────────────────────

class TestAMZNIncident20260423:
    """End-to-end: the actual holdings/models combo from 2026-04-23.

    Holdings: AMZN, CAT, GOOG, PLTR, XLU
    Universe: all 5 above plus a few noise-filtered non-held tickers.
    """

    def test_all_five_holdings_admitted_despite_sharpe_variance(self, tmp_path):
        ctx = _ctx(
            tmp_path   = tmp_path,
            loaded     = {
                # held + sub-floor
                "AMZN": _art(sharpe=0.668),
                # held + above-floor
                "CAT":  _art(sharpe=2.043),
                "GOOG": _art(sharpe=1.521),
                "PLTR": _art(sharpe=2.043),
                # held + defensive
                "XLU":  _art(sharpe=0.250),
                # non-held, sub-floor
                "NVO":  _art(sharpe=-0.403),
                "LMT":  _art(sharpe=0.755),
            },
            held       = {"AMZN", "CAT", "GOOG", "PLTR", "XLU"},
            defensives = ["XLU", "GLD", "TLT", "XLV"],
            threshold  = 1.0,
        )
        FilterUniverseFloorTask().run(ctx)
        # All 5 holdings admitted
        for t in ("AMZN", "CAT", "GOOG", "PLTR", "XLU"):
            assert t in ctx.loaded_models, f"{t} must stay loaded (held)"
        # Non-held sub-floor tickers filtered
        for t in ("NVO", "LMT"):
            assert t not in ctx.loaded_models, f"{t} should be filtered"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
