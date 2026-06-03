# 2026-06-03 — HON single_day_loss exit: post-trade analysis

**Trigger**: operator asked "why HON sold?" after the intraday sell-only
pass liquidated the HON position on 2026-06-03.

**Verdict**: a correct, in-policy `single_day_loss` (SDL) exit on a real
−4.85% intraday gap-down that cleared the σ-adaptive threshold by 25 bps.
Two secondary observations worth a decision (manual-position policy) and
a fix (reconciler attribution mislabel).

---

## 1 · What happened

| Field | Value |
|---|---|
| Symbol | HON (Honeywell) |
| Exit time | 2026-06-03 12:24:23 PDT (intraday sell-only pass) |
| Order | `d98d2cbc-d806-4b9f-8a42-5d0c3a1c8c13`, SELL 2 shares |
| Fill price | $223.8282 |
| Exit type | `single_day_loss` |
| Regime at decision | BULL_CALM (conf 0.63, hurst 0.74 → MOM) |
| Portfolio equity | $11,089 (4 holdings before sell) |
| Entry | "Manual" entry, dated 2026-04-30 |

Source: `logs/intraday_104/2026-06-03.log` (PENDING-EXIT ntfy line +
`live.alpaca_broker` order line).

## 2 · Why the gate fired (quantified)

`single_day_loss` (`kernel/exits.py::check_single_day_loss`) measures the
drop vs the **previous close** and compares it to a threshold. In
BULL_CALM the gate is **fully σ-adaptive** — golden config sets
`max_single_day_loss_pct = 0.0` and `sdl_n_sigma = 3.0`, so:

```
threshold = max(abs_pct, sdl_n_sigma × daily_realized_vol)
          = max(0, 3 × daily_vol)
```

Computed from the local 1d parquet (`data/ohlcv/HON/1d.parquet`):

| Quantity | Value |
|---|---|
| prev_close (2026-06-02) | $235.23 |
| intraday sell fill (2026-06-03) | $223.83 |
| **intraday drop** | **−4.85%** |
| HON trailing-60d daily vol | 1.53% (≈24.3% annualized) |
| **SDL threshold = 3 × 1.53%** | **4.60%** |
| drop ≥ threshold? | **4.85% ≥ 4.60% → YES** |

The position cleared the bar by 25 bps. This is a genuine signal, not a
noise trip: a 4.85% single-day move on a 24%-vol industrial is a real
gap-down, exactly the case the σ-adaptive SDL is designed to catch
(per the 2026-05-04 SDL redesign motivation in `check_single_day_loss`'s
docstring — replace the absolute 6% threshold that over-tripped high-vol
names with a per-ticker σ-scaled one).

After the exit, the runner stamped a **30-day wash-sale clock** on HON
(no re-entry until ~2026-07-03), per the standard post-sell flow.

## 3 · Observation A (decision needed) — SDL fires on MANUAL positions

HON was a **"Manual" entry** (operator-opened on 2026-04-30), yet the
system's risk gate liquidated it. This is current behavior by design:
`SellOnlyPipeline` applies exits to every held position regardless of
entry source — it reads the broker's positions, not just runner-opened
ones.

**The policy question**: should the σ-adaptive SDL (and the other exit
gates — trailing stop, stop loss, max-hold, model-sell) act on positions
the operator opened by hand?

- **Argument for current behavior**: a position is a position; risk
  management shouldn't care who opened it. A −4.85% gap-down is a
  −4.85% gap-down. Leaving manual positions un-gated creates a
  silent risk hole.
- **Argument against**: a manual entry may encode a thesis the model
  doesn't see (event-driven, longer horizon, hedge leg). Auto-liquidating
  it on a single-day move overrides operator intent and crystallizes a
  loss + a 30-day wash-sale lock the operator didn't ask for.

**Options if a change is wanted** (none implemented — this doc only
raises the question):

1. **Status quo** — all positions gated uniformly. (Current.)
2. **Manual-exempt flag** — tag manual entries (`entry_signal="Manual"`
   already exists in state) and skip non-stop-loss exits for them, while
   keeping hard stop-loss as a floor.
3. **Per-regime / per-source config** — a `manual_position_exit_policy`
   knob under `regime_params` that selects which gates apply.

Recommend **not** changing without a decision: silently exempting manual
positions would be a §7.7-class risk hole (a gate that looks active but
skips a class of positions). If we change, it must be an explicit,
documented config flag with a paired test.

## 4 · Observation B (fix candidate) — reconciler mislabels its own fill

At 12:36 (12 min after the runner placed + filled its own SDL sell), the
state-reconciliation pass logged:

```
STATE-EXT-SELL: HON disappeared from broker without runner sell —
stamping wash-sale clock today (2026-06-03) ... attribution:
source=external_or_manual order_id=d98d2cbc-... price=223.8282
```

The `order_id` it cites (`d98d2cbc`) is **the runner's own SDL order**
from 12:24 — not an external/manual sell. The reconciler saw the position
gone and attributed the disappearance to `external_or_manual` without
first checking whether the gone position matches an order the runner
itself placed earlier in the session.

**Impact**: cosmetic in this case (the wash-sale clock gets stamped either
way), but the attribution is wrong — `source=external_or_manual` for an
order the runner placed pollutes the decision-trace audit surface and
could mislead future analysis of "how many positions left via external
vs runner action."

**Fix shape** (not in this PR — flagged for a follow-up): before tagging
a disappeared position as `external_or_manual`, the reconciler should
match the broker's fill `order_id` against the set of order_ids the runner
submitted this session (the runner already logs every order_id it places).
If it matches, attribute `source=runner_<exit_type>` instead. Pairs with a
regression test that places a runner sell, lets it fill, and asserts the
reconciler attributes it to the runner — not external.

## 5 · Bottom line

- HON's exit was **correct and in-policy**. No bug in the SDL gate.
- The −4.85% drop vs a 4.60% σ-threshold is the kind of marginal trip the
  σ-adaptive design intends to catch on a high-vol name.
- Two open items: (A) a **policy decision** on whether exits should act on
  manual positions, and (B) a **reconciler attribution fix** so the runner
  stops labeling its own fills as external.

Neither (A) nor (B) is actioned here — this doc records the analysis and
surfaces the decisions. Code changes, if any, ship as separate PRs with
paired tests per §7.1.

---

Agent-Origin: Claude
