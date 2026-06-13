"""Property/invariant tests for resolve_hwm — drawdown-gate HWM resolution.

Eng plan S2 item 6 (test-ladder rebalance). test_runner_hwm_guard.py pins the
snap/ratchet behavior with worked EXAMPLES but has NO coverage of the RU-1
audit fix (2026-04-25): a non-finite account_value or stored_hwm used to make
resolve_hwm return NaN, which then made DrawdownCircuitTask's
`(NaN - equity) / NaN` evaluate `NaN >= halt_pct` → False, SILENTLY DISABLING
the drawdown gate in live trading for the rest of the run. The isfinite guards
that fixed it are unpinned. This module pins them.

The load-bearing safety invariant is simple and absolute: **the resolved HWM
is always finite**, for every possible (stored_hwm, account_value) — including
NaN / ±inf. A finite HWM keeps the live drawdown circuit armed; that is the
whole point of the guard.

No `hypothesis` dependency (hermetic requirements.lock.txt lacks it): inputs
are swept over a deterministic seeded grid that deliberately includes the
non-finite corners.

Invariants pinned:
- resolved HWM is ALWAYS finite and the return is always (float, bool).
- non-finite account_value ⇒ preserve a finite stored HWM unchanged (no snap);
  if stored is also non-finite, fall back to 0.0 — never propagate NaN.
- non-finite stored_hwm with finite account_value ⇒ reset to account_value,
  snapped=True (re-arm against good data).
- stale-seed snap: finite inputs, account_value>0, stored > ratio*account ⇒
  (account_value, True).
- ratchet (the non-snap finite path) never widens the HWM DOWN: resolved ==
  max(stored, account_value) ≥ stored, snapped=False — the "HWM only moves up"
  property, with snapping the only documented exception.
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

from adapters.runner import resolve_hwm  # noqa: E402

# the module default; mirror it here so the stale-snap reference matches.
_STALE_RATIO = 1.5
SEED = 0x4D11
N = 5000

_NONFINITE = [float("nan"), float("inf"), float("-inf")]


def _values(rng):
    return rng.choice(_NONFINITE + [
        0.0, -rng.uniform(0, 1e6),
        rng.uniform(0, 10), rng.uniform(10, 5e5), rng.uniform(5e5, 5e6),
    ])


class TestAlwaysFinite:

    def test_resolved_hwm_is_always_finite(self):
        """RU-1 core: no input — NaN, ±inf, negative, zero — may produce a
        non-finite resolved HWM, because that silently disables the live
        drawdown circuit."""
        rng = random.Random(SEED)
        for _ in range(N):
            stored, acct = _values(rng), _values(rng)
            hwm, snapped = resolve_hwm(stored, acct)
            assert isinstance(hwm, float) and math.isfinite(hwm), (stored, acct, hwm)
            assert isinstance(snapped, bool), (stored, acct, snapped)

    def test_finite_under_custom_stale_ratio(self):
        rng = random.Random(SEED + 1)
        for _ in range(N):
            stored, acct = _values(rng), _values(rng)
            ratio = rng.choice([0.0, 0.5, 1.0, 1.5, 3.0, rng.uniform(0.1, 5)])
            hwm, _ = resolve_hwm(stored, acct, ratio)
            assert math.isfinite(hwm), (stored, acct, ratio, hwm)


class TestNonFiniteFailSafe:

    def test_nonfinite_account_preserves_finite_stored(self):
        rng = random.Random(SEED + 2)
        for _ in range(1000):
            stored = rng.uniform(1, 5e6)
            for acct in _NONFINITE:
                hwm, snapped = resolve_hwm(stored, acct)
                assert hwm == float(stored) and snapped is False, (stored, acct)

    def test_nonfinite_account_and_stored_falls_back_to_zero(self):
        for stored in _NONFINITE:
            for acct in _NONFINITE:
                hwm, snapped = resolve_hwm(stored, acct)
                assert hwm == 0.0 and snapped is False, (stored, acct)

    def test_nonfinite_stored_with_good_account_resets_and_snaps(self):
        rng = random.Random(SEED + 3)
        for _ in range(1000):
            acct = rng.uniform(1, 5e6)
            for stored in _NONFINITE:
                hwm, snapped = resolve_hwm(stored, acct)
                assert hwm == float(acct) and snapped is True, (stored, acct)


class TestSnapAndRatchet:

    def test_stale_seed_snaps_to_account(self):
        rng = random.Random(SEED + 4)
        for _ in range(N):
            acct = rng.uniform(1, 1e6)
            stored = acct * rng.uniform(_STALE_RATIO + 1e-6, 50)  # strictly stale
            hwm, snapped = resolve_hwm(stored, acct)
            assert snapped is True and hwm == float(acct), (stored, acct, hwm)

    def test_ratchet_never_widens_down(self):
        """The non-snap finite path: resolved == max(stored, account) and is
        therefore >= stored. The HWM only moves up here — the drawdown
        denominator never silently shrinks."""
        rng = random.Random(SEED + 5)
        for _ in range(N):
            acct = rng.uniform(0, 1e6)
            # NOT stale: stored <= ratio*account (and >= 0 so it's a real HWM)
            stored = rng.uniform(0, _STALE_RATIO * acct) if acct > 0 else rng.uniform(0, 1e6)
            hwm, snapped = resolve_hwm(stored, acct)
            assert snapped is False
            assert hwm == max(float(stored), float(acct))
            assert hwm >= float(stored) - 1e-9, (stored, acct, hwm)

    def test_zero_and_negative_account_fall_through_to_ratchet(self):
        # account_value <= 0 skips the stale branch (guarded by >0); stored is
        # preserved via max(). No snap, still finite.
        for acct in (0.0, -1.0, -1e6):
            hwm, snapped = resolve_hwm(stored_hwm=50_000.0, account_value=acct)
            assert snapped is False and hwm == 50_000.0, acct

    def test_boundary_at_exactly_stale_ratio(self):
        # stored == ratio * account is NOT strictly greater → ratchet, no snap.
        acct = 100_000.0
        hwm, snapped = resolve_hwm(stored_hwm=_STALE_RATIO * acct, account_value=acct)
        assert snapped is False and hwm == _STALE_RATIO * acct
        # one cent above the threshold DOES snap.
        hwm2, snapped2 = resolve_hwm(stored_hwm=_STALE_RATIO * acct + 0.01,
                                     account_value=acct)
        assert snapped2 is True and hwm2 == acct
