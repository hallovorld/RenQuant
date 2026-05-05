"""Short-side acceptance gates (G7-short, G12, G13).

Stage S1a of the long/short rollout (see doc/research/short-side-design.md).
These gates extend ``model_acceptance.py``'s existing G1–G11 framework with
short-specific criteria. They consume *optional* artifact fields populated
by the offline short-side analysis script (S1b, run after long-side ship).

Invariants enforced
-------------------

* **G7_short** — bottom-decile (the cohort we'd short) must, on average,
  underperform when ranked low. The short OOS mean IC therefore must be
  ≥ 0 in absolute value (we read it as positive ``short_oos_mean_ic`` —
  meaning correlation between predicted-low rank and realised-low return,
  which is what the long-side IC means in mirror image). Floor default
  +0.02 — same as long-side G7.
* **G12_long_short_parity** — |long_top_IC − short_bot_IC| ≤
  parity_ratio × max(long, short). When parity_ratio = 0.5 the model is
  rejected if one side is more than 2× stronger than the other (sign that
  the model is essentially long-only and shorting adds noise without
  proportional alpha). Default ratio 0.5.
* **G13_short_crowdedness** — short P&L attribution should not be
  ≥ 50% concentrated in high-short-interest tickers (>20% of float).
  Crowded shorts are squeeze-prone; concentration there means the
  P&L is dominated by squeeze-risk premium, not fundamental decay.

Default behaviour: each gate **passes-open (SKIP)** when its required
artifact field is missing — long-only artifacts pre-S1b have no
``short_*`` fields, so they're unaffected. Once short-side analysis
populates the fields, gates fire automatically.

Severity: hard. The point of these gates is *refusing to ship* a model
that hasn't proven itself on the short side. Override via
``acceptance.short_side.{g7_short,g12,g13}.severity = "soft"`` if a
specific deployment wants warn-only — but default is hard.
"""
from __future__ import annotations

import logging
from typing import Any

from kernel.model_acceptance import (
    AcceptanceGate, GateResult, _safe_get_metadata,
)

log = logging.getLogger("kernel.model_acceptance_short")


# ── Individual gate functions ───────────────────────────────────────────────


def gate_g7_short_oos_ic_floor(
    staging: dict,
    active: dict | None,
    floor: float = 0.02,
    severity: str = "hard",
) -> GateResult:
    """Short-side OOS IC floor.

    Reads ``short_oos_mean_ic`` (Spearman rank-IC of the bottom-decile
    cohort vs forward returns, signed so positive = profitable short).
    Reject if short_oos_mean_ic < floor.

    Pass-open (SKIP) when short_oos_mean_ic absent: long-only artifacts.
    """
    md = _safe_get_metadata(staging)
    val = md.get("short_oos_mean_ic")
    if val is None:
        return GateResult(
            "G7_short_oos_ic_floor", severity, True, None, floor,
            "no short_oos_mean_ic on artifact (skip — long-only)",
        )
    try:
        v = float(val)
    except (TypeError, ValueError):
        return GateResult(
            "G7_short_oos_ic_floor", severity, False, None, floor,
            f"short_oos_mean_ic not numeric: {val!r}",
        )
    return GateResult(
        "G7_short_oos_ic_floor", severity, v >= floor, v, floor,
        f"short_oos_mean_ic={v:+.4f} vs floor={floor:+.4f}",
    )


def gate_g12_long_short_parity(
    staging: dict,
    active: dict | None,
    parity_ratio: float = 0.5,
    severity: str = "hard",
) -> GateResult:
    """Long/short IC parity.

    Reads both ``oos_mean_ic`` (long-side / overall) and
    ``short_oos_mean_ic``. Computes asymmetry = |long − short| /
    max(|long|, |short|). Reject if asymmetry > parity_ratio.

    parity_ratio = 0.5 means: the weaker side must be at least half the
    stronger side. asymmetry > 0.5 = model is essentially one-sided.

    Pass-open when either value is missing.
    """
    md = _safe_get_metadata(staging)
    long_ic = md.get("oos_mean_ic")
    short_ic = md.get("short_oos_mean_ic")
    if long_ic is None or short_ic is None:
        return GateResult(
            "G12_long_short_parity", severity, True, None, parity_ratio,
            "long or short IC missing (skip — long-only)",
        )
    try:
        long_v = float(long_ic)
        short_v = float(short_ic)
    except (TypeError, ValueError):
        return GateResult(
            "G12_long_short_parity", severity, False, None, parity_ratio,
            f"non-numeric ICs: long={long_ic!r} short={short_ic!r}",
        )
    denom = max(abs(long_v), abs(short_v))
    if denom <= 0:
        return GateResult(
            "G12_long_short_parity", severity, False, 0.0, parity_ratio,
            "both ICs are zero — model has no signal on either side",
        )
    asymmetry = abs(long_v - short_v) / denom
    return GateResult(
        "G12_long_short_parity", severity, asymmetry <= parity_ratio,
        asymmetry, parity_ratio,
        f"asymmetry={asymmetry:.3f} (long={long_v:+.4f}, short={short_v:+.4f})",
    )


def gate_g13_short_crowdedness(
    staging: dict,
    active: dict | None,
    max_crowded_pct: float = 0.50,
    crowded_si_threshold: float = 0.20,
    severity: str = "hard",
) -> GateResult:
    """Short P&L not over-concentrated in high-SI tickers.

    Reads ``short_pnl_attribution_high_si`` — the *fraction* (in [0, 1])
    of total short-side P&L that originated from positions in tickers
    whose short interest at entry exceeded ``crowded_si_threshold``
    (default 20% of float).

    Reject when this fraction exceeds ``max_crowded_pct`` (default 0.50).
    The short side's alpha must come predominantly from un-crowded names;
    otherwise the strategy is implicitly betting on crowded shorts not
    squeezing — that's gambling on tail behaviour, not factor edge.

    Pass-open when attribution missing.
    """
    md = _safe_get_metadata(staging)
    val = md.get("short_pnl_attribution_high_si")
    if val is None:
        return GateResult(
            "G13_short_crowdedness", severity, True, None, max_crowded_pct,
            f"no short_pnl_attribution_high_si on artifact "
            f"(skip — long-only or no SI data; threshold={crowded_si_threshold:.0%})",
        )
    try:
        v = float(val)
    except (TypeError, ValueError):
        return GateResult(
            "G13_short_crowdedness", severity, False, None, max_crowded_pct,
            f"short_pnl_attribution_high_si not numeric: {val!r}",
        )
    if not (0.0 <= v <= 1.0):
        return GateResult(
            "G13_short_crowdedness", severity, False, v, max_crowded_pct,
            f"short_pnl_attribution_high_si out of [0,1]: {v}",
        )
    return GateResult(
        "G13_short_crowdedness", severity, v <= max_crowded_pct,
        v, max_crowded_pct,
        f"high_si_pnl_share={v:.1%} vs cap={max_crowded_pct:.0%} "
        f"(crowded threshold SI≥{crowded_si_threshold:.0%})",
    )


# ── Builder integrating into the existing acceptance pipeline ───────────────


def build_short_gates_from_config(config: dict) -> list[AcceptanceGate]:
    """Construct the three short-side gates honoring config knobs.

    Schema::

      acceptance:
        short_side:
          enabled: false        # default false — gates skip-open until
                                # explicitly turned on
          g7_short:
            floor: 0.02
            severity: hard
          g12:
            parity_ratio: 0.5
            severity: hard
          g13:
            max_crowded_pct: 0.50
            crowded_si_threshold: 0.20
            severity: hard

    When ``enabled = false`` (default) the gates are still INSTANTIATED
    and run, but pass-open on missing fields — so a long-only artifact
    is unaffected. When enabled, the artifact MUST carry the relevant
    fields or the gates fail (no skip).
    """
    cfg = (config or {}).get("acceptance", {}).get("short_side", {})
    enabled = bool(cfg.get("enabled", False))

    g7 = cfg.get("g7_short", {})
    g12 = cfg.get("g12", {})
    g13 = cfg.get("g13", {})

    gates: list[AcceptanceGate] = [
        AcceptanceGate(
            name="G7_short_oos_ic_floor",
            severity=g7.get("severity", "hard"),
            check=lambda s, a, _floor=float(g7.get("floor", 0.02)),
                          _sev=g7.get("severity", "hard"),
                          _enabled=enabled:
                _short_gate_with_required_field(
                    gate_g7_short_oos_ic_floor,
                    s, a, "short_oos_mean_ic", _enabled,
                    floor=_floor, severity=_sev,
                ),
        ),
        AcceptanceGate(
            name="G12_long_short_parity",
            severity=g12.get("severity", "hard"),
            check=lambda s, a, _ratio=float(g12.get("parity_ratio", 0.5)),
                          _sev=g12.get("severity", "hard"),
                          _enabled=enabled:
                _short_gate_with_required_field(
                    gate_g12_long_short_parity,
                    s, a, "short_oos_mean_ic", _enabled,
                    parity_ratio=_ratio, severity=_sev,
                ),
        ),
        AcceptanceGate(
            name="G13_short_crowdedness",
            severity=g13.get("severity", "hard"),
            check=lambda s, a,
                          _max=float(g13.get("max_crowded_pct", 0.50)),
                          _thresh=float(g13.get("crowded_si_threshold", 0.20)),
                          _sev=g13.get("severity", "hard"),
                          _enabled=enabled:
                _short_gate_with_required_field(
                    gate_g13_short_crowdedness,
                    s, a, "short_pnl_attribution_high_si", _enabled,
                    max_crowded_pct=_max, crowded_si_threshold=_thresh,
                    severity=_sev,
                ),
        ),
    ]
    return gates


def _short_gate_with_required_field(
    gate_fn,
    staging: dict,
    active: dict | None,
    required_field: str,
    enabled: bool,
    **kwargs: Any,
) -> GateResult:
    """Wrap a gate so missing field is FAIL when shorts enabled, SKIP otherwise.

    The gate functions themselves pass-open on missing fields (so long-only
    artifacts breeze through). When ``acceptance.short_side.enabled = true``,
    we promote that pass-open into a hard fail — once shorts are turned on,
    the offline analysis MUST populate these fields.
    """
    md = _safe_get_metadata(staging)
    has_field = md.get(required_field) is not None
    if enabled and not has_field:
        # Required field missing despite shorts being enabled — fail loud.
        return GateResult(
            gate_fn.__name__.replace("gate_", "").upper(),
            kwargs.get("severity", "hard"),
            False, None, None,
            f"short-side enabled but {required_field!r} missing on artifact — "
            f"run scripts/analyze_short_side_ic.py to populate.",
        )
    # Otherwise delegate to the gate (pass-open on missing if disabled).
    return gate_fn(staging, active, **kwargs)
