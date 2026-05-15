"""Regime-conditional config knob resolver (PRIME DIRECTIVE 2026-05-14).

Single source of truth for the resolution order any task uses when reading
a knob that might be specialized per regime:

    regime_params.<regime>.<flat_overlay_key>   (highest precedence)
       > <top_section>.<knob>                    (global section default)
       > <knob>                                   (top-level key, if no top_section)
       > default                                  (fallback supplied by the caller)

The overlay key is FLAT — `regime_params` is a flat dict. The convention is:
  * Top-level scalar (e.g. ``max_position_pct``):
      overlay at ``regime_params.<regime>.max_position_pct``
  * Nested key (e.g. ``long_short.enabled``):
      overlay at ``regime_params.<regime>.long_short_enabled``
      (i.e. ``<top_section>_<knob>``, joined with ``_``)

This prevents key collisions across nested sections (otherwise `enabled` is
ambiguous: shorts, meta-label, vol-target, etc.).

Reference: CLAUDE.md PRIME DIRECTIVE; doc/roadmap.md P1.
"""
from __future__ import annotations

from typing import Any


def resolve_regime_knob(
    ctx,
    top_section: str | None,
    knob: str,
    default: Any,
    *,
    regime: str | None = None,
    overlay_key: str | None = None,
) -> Any:
    """Resolve a config knob with regime-overlay precedence.

    Args:
      ctx:         InferenceContext (any object exposing `.config` and `.regime`).
      top_section: Name of the top-level config block holding the global default
                   (e.g. ``"long_short"``). Pass None if the knob lives at
                   ``config[<knob>]`` directly (e.g. ``"max_position_pct"``).
      knob:        Name of the knob to resolve.
      default:     Returned only when no overlay, top-section, or top-level
                   value is found.
      regime:      Optional explicit regime override (mostly for tests).
                   Defaults to ``getattr(ctx, "regime", None)``.
      overlay_key: Optional explicit overlay key under regime_params.<regime>.
                   Defaults to ``<top_section>_<knob>`` if top_section is given,
                   else ``<knob>``.

    Resolution (first hit wins):
      1. ``ctx.config["regime_params"][regime][overlay_key]``
      2. ``ctx.config[top_section][knob]`` (if top_section provided)
      3. ``ctx.config[knob]`` (if no top_section)
      4. ``default``

    No type coercion — caller casts the result.
    """
    config = getattr(ctx, "config", None) or {}
    reg = regime if regime is not None else getattr(ctx, "regime", None)

    # Compute the overlay key — disambiguated by top_section prefix
    if overlay_key is None:
        overlay_key = f"{top_section}_{knob}" if top_section else knob

    # 1. Per-regime overlay
    if reg is not None:
        rp = config.get("regime_params", {}) or {}
        overlay = rp.get(reg, {}) or {}
        if overlay_key in overlay:
            return overlay[overlay_key]

    # 2/3. Global default
    if top_section is not None:
        section = config.get(top_section, {}) or {}
        if knob in section:
            return section[knob]
    else:
        if knob in config:
            return config[knob]

    # 4. Fallback
    return default
