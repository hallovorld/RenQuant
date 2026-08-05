# 2026-08-04 — fleet callsigns + terse ntfy (operator directive: 简练,人话)

Operator-chosen scheme (功能缩写): R=reversal, C=classifier, S=slow momentum,
f=fast momentum (lowercase deliberate). Mapping in `live/runner.py
LANE_CALLSIGNS`:

| tag | callsign |
|---|---|
| alpaca_shadow_blend | RC |
| alpaca_shadow_blend_mom | RSs |
| alpaca_shadow_blend_mom_fast | Rf |
| alpaca_shadow_blend_rb_mom | RCS |
| alpaca_shadow_blend_rb_fast | RCf |

Prod (composition RS) keeps its existing live title format. Contracts held:
legacy `alpaca_shadow` prefix stays byte-identical `[READONLY]`; every shadow
prefix still STARTS with `[READONLY]` (the is_shadow classification — a
shadow message must never classify as live); unknown future tags fall back to
the full upper tag, never a bare `[READONLY]`.

Terse body: the `SHADOW/HYPOTHETICAL (no live orders)` boilerplate sentence is
REMOVED — the title carries the shadow identity twice ([READONLY][…] +
SHADOW-* tag). Before/after:

- before: `[READONLY][ALPACA_SHADOW_BLEND_MOM]RENQUANT-104 [full] SHADOW-ACTION | SHADOW/HYPOTHETICAL (no live orders) | BUY GOOG x1 …`
- after: `[READONLY][RSs]RENQUANT-104 [full] SHADOW-ACTION | BUY GOOG x1 …`

Tests: readonly-tag + trade-ntfy suites updated to the new pins (84 passed);
the 2 remaining failures (`TestSourceLevel` anchored on the RETIRED "Step 4:
Shadow e2e" heading) fail identically on clean main — pre-existing test rot
from the Step-4 retirement, noted for a separate cleanup.
