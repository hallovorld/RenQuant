# Self-audit — bugs in this session's shipped work (2026-04-24 PT)

User prompted an audit. Below are bugs I found in my own shipped code
today, classified by severity, with fixes. Running this openly so the
second AI's audit doesn't have to re-discover them.

## HIGH-severity — silent no-ops

### Bug 1: `find_thesis_symmetric_pairs` shipped but NOT wired

**Commit introducing:** `709032d`

**Symptom:** user sets `rotation.scoring_mode="thesis_symmetric"` expecting V4 4-point logic. Actual behavior: the string "thesis_symmetric" didn't match any branch in `BuildPairsTask`, so execution silently fell through to ER mode. The V4 kernel function + 10 tests existed but were reachable only from unit tests.

**Fix:** `f3a0000` (this session, post-audit) — added `if rotation_mode == "thesis_symmetric":` branch that calls `find_thesis_symmetric_pairs` with DB lookup + own-momentum.

### Bug 2: `own_momentum` param wired into kernel but no caller populates it

**Commit introducing:** `9463e4c` (own-momentum gate)

**Symptom:** even if someone set `rotation.thesis_symmetric.own_momentum_enabled=true`, the `own_momentum` dict passed to the kernel was always None/empty, so the Moskowitz gate was a no-op regardless of flag state.

**Fix:** same commit as Bug 1 — the new `if rotation_mode == "thesis_symmetric":` branch builds `own_mom` dict from OHLCV close series (63d return) and passes to the kernel.

## HIGH-severity — false claim in roadmap

### Bug 3: baseline IC reported 0.0326 was wrong; actual is 0.0391

**Commit introducing:** `322f5aa` ("minute panel CPCV IC +0.0355 prelim")

**Symptom:** I wrote that minute-enabled panel IC of 0.0355 was "+0.003 lift over baseline 0.0326". Actual current golden `panel-ltr.json` OOS IC is **0.0391**. So minute-enabled panel is **0.0355 = -0.0036 WORSE**, not better.

**Root cause:** I pulled the 0.0326 number from an old doc snippet (`renquant_104_design.md` §5d Round 1-5 log) which was a prior panel's IC. The actual current golden is trained_date 2026-04-23 and has IC 0.0391.

**Correction:** added to roadmap; the 10-min A/B conclusion is **minute features HURT IC**, pending NGBoost + sim phases which might recover some. The "transformer retry unlock" argument still holds on row-count grounds (744k > 200k gate), but it's not backed by an IC lift on the current run.

## MEDIUM-severity — test fixture brittleness

### Bug 4: V2 test used held with entry_price != current_price

**Commit introducing:** `0000b91` (V2 scoring mode)

**Symptom:** `test_rotation_v2_scoring.py::test_default_er_mode_unchanged` initially failed because my `_held()` helper defaulted `current_price=110` (10% unrealized), causing tax_drag to eat the edge. Caught in pre-ship test run.

**Fix:** changed default `current_price=entry_price` in the test helper so tax_drag is zero unless tests explicitly set it.

## MEDIUM-severity — operational

### Bug 5: panel_exit V2 "OR mode" still fired 0 exits on current panel

**Commit introducing:** `b022ad6`

**Symptom:** even with `trigger_mode="or"` (fires when panel<0.20 OR μ≤0.0), the 27-mo OOS A/B showed 0 panel_exits. The thresholds are too tight for our holdings' score distribution. Gate shipped correctly but is practically non-firing under current panel.

**Not a code bug** — parameter-tuning issue. The AND→OR change is sound; thresholds need raising (suggest panel_sell_floor=0.30, mu_sell_ceiling=-0.01) to see any fires.

### Bug 6: rotation V1/V2/V3 also 0 rotations in A/B

**Commit introducing:** `9eb188b`, `0000b91`, `f674b3f` (all three rotation versions)

**Symptom:** all rotation variants produce 0 rotations in A/B. Earlier A/B (Route A in roadmap) had produced 3 rotations at threshold=0.005. Between sessions the panel was retrained; current panel's ER distribution is too tight for the existing threshold to fire.

**Not a code bug** — documented in roadmap as "candidate-supply bottleneck". The rotation machinery works; it's starved of candidates to rotate TO.

## LOW-severity — minor

### Bug 7: `rotation.bear_only` check order vs V3 regime gate order

**Status:** ctx.bear_only check runs BEFORE V3 enabled_regimes check. If `bear_only=True` AND `enabled_regimes=["BEAR"]`, V3 never reaches. Arguably correct behavior (bear_only is an explicit suppression) but worth documenting. Not fixing — bear_only takes precedence by design.

## What I'm NOT claiming

- No guarantee I caught every bug this session. Welcome the second AI's audit.
- No guarantee the fixes are themselves bug-free. Tests passing is necessary but not sufficient.
- No guarantee the rotation-V4 wiring is complete — DB lookup depends on `ctx._db` being set on the adapter (done for SimAdapter in this commit; NOT done for LeanAdapter or RunnerAdapter).

## Recommendations going forward

1. Every new Task/Job/kernel fn needs a **wire-up test** that proves the code path from config flag → kernel fn → output. Unit tests alone don't catch dead branches.
2. Every numeric claim in docs/commit messages should be traceable to a specific file + line + timestamp. The IC misquote came from referencing a stale summary doc.
3. A/B results should be reported with clean variance attribution (same panel, same cache, same sim length). "Baseline drift" between A/Bs is a warning that the comparison isn't apples-to-apples.
