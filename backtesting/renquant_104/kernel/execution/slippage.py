"""Slippage model: half-spread + linear participation impact.

Single source of truth per CLAUDE.md §5.13.5 — sim and LEAN must agree
on the per-order fill adjustment. Two components, both in basis points:

1. **Half-spread** — buyer pays ask, seller hits bid. Each side eats
   one half-spread vs the mid. For liquid S&P names the round-trip
   spread is ~4 bps, so 2 bps per side is a defensible default.
2. **Participation impact** — linear in the fraction of average daily
   volume (ADV) consumed. Almgren-Chriss 2000 ("Optimal execution of
   portfolio transactions", J. Risk 3.2) shows linear-in-rate impact
   is a reasonable first approximation for retail order sizes. We
   keep the coefficient ``impact_bps_per_pct_adv`` configurable and
   default to 0 (off) since retail share counts at our portfolio
   scale rarely exceed 0.1% ADV.

References:
- Almgren & Chriss (2000), "Optimal execution of portfolio
  transactions", Journal of Risk 3(2), 5-39.
- Kissell (2014), "The Science of Algorithmic Trading", Ch. 4.

Per CLAUDE.md §5.13.11 / §5.13.12: every numeric path is finite-guarded
and the bps factor is hard-clipped to 50 bps each side to prevent a
config typo (e.g. "2" misread as "200") from blowing up the equity
curve.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# §5.13.12: clip half-spread + impact to a sane absolute ceiling so a
# fat-finger config (e.g. ``200`` bps instead of ``2``) can never deliver
# > 50 bps of one-sided cost on a fill.
_MAX_SLIP_BPS_PER_SIDE: float = 50.0


@dataclass(frozen=True)
class SlippageConfig:
    """Immutable slippage parameters.

    Defaults are Alpaca-equity-retail-grade: 2 bps half-spread, no
    participation impact (Alpaca's smart-order-router handles small
    orders by atomic-routing them across venues).
    """

    half_spread_bps: float = 2.0
    impact_bps_per_pct_adv: float = 0.0


# ── Helpers (each ≤ 50 lines per CLAUDE.md §1c) ────────────────────────────


def _clip_bps(raw_bps: float) -> float:
    """Clamp a bps figure to ``[0, _MAX_SLIP_BPS_PER_SIDE]``.

    NaN / inf collapse to 0 (defensive — would otherwise propagate).
    """
    if not math.isfinite(raw_bps):
        return 0.0
    if raw_bps < 0:
        return 0.0
    if raw_bps > _MAX_SLIP_BPS_PER_SIDE:
        return _MAX_SLIP_BPS_PER_SIDE
    return raw_bps


def _impact_bps(shares: float,
                adv_shares: "float | None",
                cfg: SlippageConfig) -> float:
    """Linear impact in bps: ``(shares/adv) * impact_bps_per_pct_adv``.

    ADV is ``None`` or 0 → impact is 0 (defensive per spec: a missing
    ADV reading must NOT block the fill — return zero impact instead).
    """
    if adv_shares is None or not math.isfinite(adv_shares) or adv_shares <= 0:
        return 0.0
    if not math.isfinite(shares) or shares <= 0:
        return 0.0
    if not math.isfinite(cfg.impact_bps_per_pct_adv):
        return 0.0
    frac_adv = shares / adv_shares
    return frac_adv * cfg.impact_bps_per_pct_adv


def _side_sign(side: str) -> int:
    """+1 for buy (pay more), -1 for sell (receive less). Raises on typo."""
    s = side.lower().strip()
    if s == "buy":
        return 1
    if s == "sell":
        return -1
    raise ValueError(f"slip_fill_price: side must be 'buy' or 'sell', got {side!r}")


# ── Public API ─────────────────────────────────────────────────────────────


def slip_fill_price(market_price: float,
                    side: str,
                    shares: float,
                    adv_shares: "float | None",
                    cfg: SlippageConfig) -> float:
    """Return the fill price after applying slippage.

    Args:
        market_price: bar-close price feed.
        side: 'buy' or 'sell'.
        shares: order size (positive).
        adv_shares: 20-day or 60-day ADV in shares; ``None`` → no impact.
        cfg: :class:`SlippageConfig`.

    Returns:
        Adjusted fill price (still positive — clipped at 0.01 floor for
        defensive divide-by-zero protection in downstream qty math).

    Behavior:
        buy  → market * (1 + (half_spread + impact) * 1e-4)
        sell → market * (1 - (half_spread + impact) * 1e-4)

    Per §5.13.11: non-finite market_price returns market_price unchanged
    (caller should already reject NaN via SAB-3 guard; defense in depth).
    """
    if not math.isfinite(market_price) or market_price <= 0:
        return market_price
    sign = _side_sign(side)
    half = _clip_bps(cfg.half_spread_bps)
    impact = _clip_bps(_impact_bps(shares, adv_shares, cfg))
    total_bps = half + impact
    factor = 1.0 + sign * total_bps * 1.0e-4
    fill = market_price * factor
    if not math.isfinite(fill) or fill <= 0:
        # Pathological multiplier collapsed price; fall back to market.
        return market_price
    return fill
