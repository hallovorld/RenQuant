# Sell-Side Quality Floor — Design (2026-04-26)

**Authors**: RenQuant team (round-7 audit)
**Status**: Shipped (default OFF; opt-in per CLAUDE.md feature-flag protocol)
**Files**: `kernel/pipeline/task_sell.py::SellGateBTask`,
          `kernel/pipeline/task_limit_sells.py::LimitSellsPerBarTask`
**Tests**: `tests/test_sell_gate_b.py` (27), `tests/test_limit_sells_per_bar.py` (21)

## Motivation (the bug we're fixing)

E2E sim on Sun 2026-04-26 emitted `model_sell` for **3 of 6 holdings** in
a single bar (GOOG, AMZN, BA). User reaction: *"这他妈的合理吗？把我有
的股票全卖了？"* — "Is this fucking reasonable? Sell ALL my stocks?"

Two distinct failure modes:

1. **Asymmetric protection**: the BUY path has a quality floor (Gate A
   distribution / Gate B edge-Sharpe / Gate C no-trade band) — see
   `doc/components/buy-logic-design.md`. The SELL path had only
   per-ticker model + path-dependent rules. A single-day model spike
   could exit a position the panel + NGBoost μ/σ still liked.
2. **Portfolio-level invisibility**: per-ticker rules can't see how many
   sells are firing simultaneously. Concentrated same-bar liquidations
   destroy the diversification benefit and incur outsized execution
   cost, but no individual `TickerSellJob` could see the others.

This document specifies the two sell-side guards added to address both.

---

## Gate B (sell-side mirror): `SellGateBTask`

### Hypothesis

The buy-side Gate B uses **edge-Sharpe** (Lo 2002) `μ/σ ≥ +τ_B` as the
signal-strength floor before entering a position. By symmetry, an
**exit decision driven by the per-ticker model** should also pass an
edge-Sharpe test — but on the OTHER side: `μ/σ ≤ -τ_S`.

If the panel + NGBoost head still says "no negative edge" (μ/σ
above `-τ_S`), the model_sell signal is suspect and should be
suppressed. Path-dependent rules (stop_loss, trailing_stop, single_day_loss,
max_hold) are EXEMPT — they are the risk-management layer and must
always fire, just like the buy side never lets Gate B block them.

### Algorithm (formal)

For each holding `i` with current `exit_signal`:

```
if not enabled:                           return  # no-op
if exit_signal is None:                   return  # nothing to gate
if exit_signal.exit_type != "model_sell": return  # path rules exempt
if μ_i is None or σ_i is None:            return  # warmup → fail-safe
if σ_i ≤ 0 or NaN(μ_i, σ_i):              return  # defensive

edge_sharpe = μ_i / σ_i
if edge_sharpe > -threshold:
    exit_signal := None                   # block model_sell
    # streak NOT touched — fires immediately when μ/σ drops below floor
```

### Defaults

```yaml
ranking:
  panel_scoring:
    sell_gate_b:
      enabled:   false      # opt-in
      threshold: 0.10       # symmetric to buy default 0.20, but more
                            # conservative on sell side (easier to KEEP)
```

### Why the streak is not reset

Once a model has been signaling `sell` for ≥`consecutive_required` bars
(default 3), the streak represents accumulated evidence. Gate B is a
veto on the **execution decision**, not on the **evidence**. As soon
as `μ/σ` drops below the floor, the existing streak fires immediately
on the next bar without needing to re-accumulate. This is the symmetric
behavior to the buy side, where Gate B blocks an entry but the
candidate's score is preserved for the next bar's re-evaluation.

### Pipeline placement

```
TickerSellJob.tasks = [
    PrepareHoldingTask,         # validate holding + price
    ScoreModelTask,             # per-ticker model → tc.model_action
    EvaluateExitsTask,          # 5 priority rules → tc.exit_signal
    SellGateBTask,              # ← THIS — μ/σ guard on model_sell only
    PanelConvictionExitTask,    # tiebreaker; runs only if exit_signal None
]
```

When SellGateB clears `exit_signal`, `PanelConvictionExitTask` still
gets to consider firing (via its own μ ≤ ceiling rule). This preserves
the original tiebreaker semantics: panel conviction can still pull the
trigger if BOTH `rank_score < 0.20` AND `μ ≤ 0`.

### References

- **Lo, A.W. (2002)**. "The Statistics of Sharpe Ratios", *Financial
  Analysts Journal* 58(4): 36-52. The edge-Sharpe `μ/σ` criterion that
  the buy-side gate uses; this is its symmetric sell-side analog.
- **Grinold & Kahn (1999)**. *Active Portfolio Management* (2nd ed.),
  McGraw-Hill. Ch. 5: information ratio = α/ω as the action threshold
  in active management.
- **Kahneman & Tversky (1979)**. "Prospect Theory: An Analysis of
  Decision under Risk", *Econometrica* 47(2): 263-291. Loss aversion
  / disposition-effect motivation: sell-side gate balances asymmetric
  pain of forced exit vs. holding cost.

---

## Per-bar sell cap: `LimitSellsPerBarTask`

### Hypothesis

Multiple per-ticker model sells firing in one bar produce two
portfolio-level harms that no per-ticker task can see:

1. **Execution cost** scales super-linearly in concentrated unwinds
   (Almgren-Chriss 2000): `E[cost] ∝ X × (1 + γ·X/V)` where X is order
   size and V is bar volume.
2. **Diversification destruction**: Markowitz (1952) — the variance
   benefit of holding N positions disappears in one bar; the rebuild
   cost (re-entering through Gate A/B/C) is spread over weeks.

A portfolio manager with a discretionary brake would never approve
3-of-6 sells in one bar absent regime-shift evidence. This task is
that brake.

### Algorithm

```
max_n = config["risk"]["max_sells_per_bar"]   # 0 = uncapped (default)
if max_n ≤ 0 or len(ctx.exits) == 0:
    return

# Partition: risk exits exempt; model_sells go through cap.
risk_kept   = [(t, s) for t, s in ctx.exits if s.exit_type ∈ RISK_TYPES]
model_sells = [(t, s, μ(t)) for t, s in ctx.exits
                            if s.exit_type == "model_sell"]
# Other types (preserve, fail-open) → risk_kept

if len(model_sells) ≤ max_n:
    return   # under cap

# Sort by μ ascending (most-bearish first); μ=None → +inf (drop first)
model_sells.sort(key = lambda x: x[2])
ctx.exits = risk_kept + model_sells[:max_n]
ctx.exits_throttled.extend(model_sells[max_n:])    # diagnostic
ctx.counters["model_sell_throttled"] += dropped_count
```

### Risk types (always exempt)

```python
RISK_EXIT_TYPES = {
    "stop_loss", "trailing_stop", "single_day_loss", "max_hold",
    "panel_conviction", "rotation", "kelly_trim", "joint_sell",
    # legacy aliases:
    "sdl", "trailing_stop_loss", "gap_down", "max_hold_days",
}
```

The "fail-open" rule is intentional: any unrecognized exit_type passes
through (preserved). Better to risk one extra sell than to suppress a
risk-management signal we don't recognize.

### Defaults

```yaml
risk:
  max_sells_per_bar: 0        # 0 = uncapped (preserves existing behaviour)
                              # Recommended live: 2 (max 2 model_sells per
                              # bar; risk exits unaffected)
```

### Pipeline placement

```python
# pp_inference.py — BOTH InferencePipeline AND SellOnlyPipeline:
sell_tctxs = [...]
run_parallel(sell_tctxs, TickerSellJob())
for tc in sell_tctxs:
    if tc.exit_signal and tc.exit_signal.should_exit:
        ctx.exits.append((tc.ticker, tc.exit_signal))
LimitSellsPerBarTask().run(ctx)   # ← portfolio-level cap
# (then PanelRankVetoJob runs in InferencePipeline only)
```

### References

- **Almgren, R. & Chriss, N. (2000)**. "Optimal Execution of Portfolio
  Transactions", *J. Risk* 3(2): 5-39. Temporary market impact grows
  with execution rate; concentrated same-bar liquidations incur
  super-linear cost penalty.
- **Bertsimas, D. & Lo, A.W. (1998)**. "Optimal Control of Execution
  Costs", *J. Financial Markets* 1: 1-50. Formal cost-of-haste model
  for unwinding multiple positions.
- **Markowitz, H. (1952)**. "Portfolio Selection", *J. Finance* 7(1):
  77-91. Diversification rationale; mass-exit destroys variance benefit
  accumulated through prior position-building.

---

## Interaction with existing controls

| Existing task                | Order                          | Interaction |
|------------------------------|--------------------------------|-------------|
| `EvaluateExitsTask`          | per-ticker, before SellGateB   | Produces the model_sell candidate; SellGateB then guards it. |
| `SellGateBTask` (new)        | per-ticker, after EvaluateExits| **Symmetric** to buy gate B. Path rules exempt. |
| `PanelConvictionExitTask`    | per-ticker, last in chain      | Runs ONLY if exit_signal is None — so SellGateB clearing it gives PCT a chance to evaluate independently. |
| `LimitSellsPerBarTask` (new) | portfolio-level, after agg     | Final brake. Risk exits exempt. |
| `PanelRankVetoJob`           | portfolio-level, after limit   | Existing `model_sell` veto on `rank_score`. Works on whatever LimitSells passes through. |

### Order matters

Layer-by-layer protection from per-ticker model spike → portfolio-level
brake → cross-sectional veto:

1. Per-ticker rules (5 priority chain) → produce candidate signal
2. **Per-ticker μ/σ guard** (SellGateB) → block weak edge-Sharpe sells
3. **Portfolio-level cap** (LimitSells) → keep only top-N most bearish
4. Cross-sectional rank veto (PanelRankVeto) → block sells where panel
   rank disagrees with bearish view

Each layer adds defense without bypassing the others. Path-dependent
risk exits skip all four — they are the always-on safety layer.

---

## A/B testing protocol

Per CLAUDE.md §2a, ship at +2 pp APY OR theory-aligned positive margin
with mechanism-clean change. These two changes are mechanism-clean
(default OFF; only flag-on changes behavior; no panel retrain needed)
so any positive APY margin will trigger a golden update.

### Recommended next session A/B

```bash
# A: golden v4.1 (current production)
python scripts/validate_buy_logic.py --strategy renquant_104 --baseline

# B: with sell-side floor enabled
# (edit strategy_config.json: sell_gate_b.enabled=true,
#  risk.max_sells_per_bar=2)
python scripts/validate_buy_logic.py --strategy renquant_104
```

Hypothesis: B reduces same-bar mass exits → smoother equity curve →
higher Sharpe (>= +0.05) and similar APY (-1 to +1 pp range).

If hypothesis confirmed:
- Promote `sell_gate_b.enabled=true` + `max_sells_per_bar=2` to golden.
- Update `doc/ops/golden-config.md` → v4.2 entry.
- Sync `strategy_config.golden.json` to live values in same commit.

If APY drops > 2 pp, this is per CLAUDE.md §2b "unexpected A/B result"
— audit before accepting:
1. Print per-bar log of `LimitSellsPerBarTask` to verify it's only
   firing on intended bars (not single-sell scenarios).
2. Verify `SellGateBTask` is reading `μ`, `σ` from the holding (not
   from candidate which doesn't apply on sell side).
3. Check whether the dropped model_sells were positions that should
   have exited (e.g. did we bag-hold through a real downturn?).
