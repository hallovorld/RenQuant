"""Kelly-optimal position sizing — continuous-returns formulation.

  For a log-utility trader facing a continuous return r ~ 𝒩(μ, σ²),
  the Kelly-optimal fraction of wealth to bet is:

                    f*  =  μ / σ²

  (Thorp 1962, Kelly 1956 generalised — see Rotando-Thorp for
   continuous-time derivation.)

  We already predict both moments:

      μ_i  ← NGBoost head      (expected excess return over lookahead)
      σ_i  ← NGBoost head      (predicted stdev)

  So sizing falls out of the bet-size theorem directly. No
  heuristics, no hand-picked ladders. Pure μ/σ² scaled by three
  risk knobs:

      fractional         safety multiplier (classical: 0.25 =
                         "quarter Kelly", widely used in live trading
                         to absorb μ estimation error + log-utility
                         variance drag).
      min_edge           μ floor — if expected return is below this,
                         the bet is noise, size = 0.
      max_concentration  hard single-ticker ceiling. Even the best
                         signal capped here (risk management). Default
                         0.35 — 100% would require IC >> 0.033.

  Finally capped by the regime's `max_position_pct` so a strong Kelly
  bet can't violate the regime's aggression policy.

Beauty invariants:

  1. **One formula, one place**. All three decision layers
     (SizeAndEmit for new buys, TopUpHeld for add-to-existing,
     Rotation for swap) read the SAME `kelly_target_pct` field. No
     drift.

  2. **Input-shape agnostic**. Takes a Candidate or a Holding —
     both have mu / sigma / panel_score.

  3. **Graceful degradation**. If NGBoost didn't run (sigma=None),
     returns 0 — sizing falls back to whatever the caller does with
     a zero target (e.g. SizeAndEmit skips).

  4. **Symmetric**. The same number drives "how much to buy" and
     "how much to trim to". Currently we only top-up, but trim is
     a one-line addition once we have partial-sell plumbing.
"""
from __future__ import annotations

import logging

log = logging.getLogger("kernel.kelly")


def kelly_target_pct(
    mu:                float | None,
    sigma:             float | None,
    *,
    max_pct:           float,
    max_concentration: float = 0.35,
    fractional:        float = 0.25,
    min_edge:          float = 0.0,
) -> float:
    """Return the Kelly-optimal target position weight in [0, 1].

    f* = μ / σ²
    target = min(max_concentration, max_pct, fractional * f*)

    Returns 0 on missing inputs, non-positive σ, or μ ≤ min_edge —
    all signal "don't bet".
    """
    if mu is None or sigma is None or sigma <= 0 or mu <= min_edge:
        return 0.0
    f_kelly = float(mu) / (float(sigma) ** 2)
    f_frac  = fractional * f_kelly
    # Double cap: regime's `max_pct` AND global `max_concentration`.
    return max(0.0, min(float(max_pct), float(max_concentration), f_frac))


__all__ = ["kelly_target_pct"]
