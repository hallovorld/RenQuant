# Golden Config — v4.1 (Kelly half + A-gate + CUSUM wall-time) + round-7 acceptance block

**Current golden.** Promoted 2026-04-24 (v4.1 from +37.85 → +39.82 APY), builds on v4 (`eb8fab5`, Kelly half + A-gate). Round-7 additions on 2026-04-26 (acceptance gates Phase 1+2, challenger infra) added to the golden snapshot but do NOT change measured APY — they are infrastructure for future safe retrains.

**Frozen snapshot:** `backtesting/renquant_104/strategy_config.golden.json`
**Live config file:** `backtesting/renquant_104/strategy_config.json`

**Pre-commit drift check** (`scripts/check_config_drift.py`) refuses commits that leave `strategy_config.json` and `strategy_config.golden.json` out of sync. To intentionally promote a config change to golden: edit BOTH files in the same commit.

---

## Round-7 (2026-04-26) — model-selection block added

Single new top-level block (`acceptance`) gates retraining via the 11-gate `ModelAcceptanceGate`. SOP: [`../components/model-selection.md`](../components/model-selection.md). Snapshot from `strategy_config.golden.json`:

```json
"acceptance": {
  "enabled": true,
  "g4_max_degradation": 0.05,
  "g4_severity": "hard",
  "g7_floor": 0.02,
  "g7_severity": "hard",
  "g8_min_std": 0.001,
  "g8_severity": "soft",
  "g9_max_pp_drop": 1.0,
  "g9_severity": "hard",
  "g10_max_sharpe_drop": 0.1,
  "g10_severity": "hard",
  "g11_max_multiplier": 1.5,
  "g11_severity": "soft",
  "run_sim_smoke": false,
  "challenger": {
    "enabled": false,
    "artifact_path": null,
    "name": null,
    "shadow_period_days": 0
  }
}
```

`run_sim_smoke=false` keeps Phase 2 sim-based gates (G9/G10/G11) in skip-pass mode (operator opts in by populating sim metrics via `kernel.sim_smoke.add_smoke_metrics_to_artifact`). `challenger.enabled=false` keeps Phase 4a infra dormant (live wiring is deferred to Phase 4b). Bypass per-run with `--skip-acceptance` or set `acceptance.enabled: false`.

---

## v4.1 — CUSUM cooldown wall-time (2026-04-24)

**Sweep (27-mo OOS, `allow_fetch=False`):** +37.85 → **+39.82% APY** (+1.97 pts). Below the default +2 pt promotion floor, but **promoted under CLAUDE.md §2a**: live/sim parity fix with matched theoretical prediction (roadmap predicted "~2 pt drift closure"; result +1.97) under rigorously-controlled variables (same panel, same everything, one flag flipped).

**Change:**
```json
"regime": {
  "cusum_cooldown_mode": "wall_time",     // was: "bar_count" default
  "cusum_cooldown_days":  3.0
}
```

**Mechanism:** `TransitionWindowTask` no longer hard-blocks buys on regime-switch; instead `SizeAndEmitTask` scales `max_position_pct × cooldown_progress` (0 immediately after switch → 1.0 after 3 calendar days). Prevents intraday runners from ticking the bar-count cooldown 10× per day, keeping sim and live wall-clock-consistent.

**Buys 115 → 117, streak 42d → 37d, win rate 82 → 83%.**

---

## Measured performance (sweep, `allow_fetch=False` handicap)

27-mo OOS sim, 4-config sweep:

| Config | APY | Δ GOLDEN | Win | Buys | Streak |
|---|---:|---:|---:|---:|---:|
| v3 (hourly panel, no Kelly) | +25.91% | — | 81% | 144 | 25d |
| A + Kelly(quarter) | +36.23% | +10.32 | 87% | 117 | 43d |
| **A + Kelly(half)** ⭐ | **+37.82%** | **+11.91** | 85% | 115 | 43d |
| A + Kelly(tight cap) | +36.23% | +10.32 | 87% | 117 | 43d |

**Absolute APY caveat:** sweep uses `allow_fetch=False` (fundamentals / earnings / insider fetch disabled for reproducibility) so the absolute APY is lower than the live-fetch-enabled number. All 4 configs share the same handicap — the **relative ranking is the valid signal**. Expected live APY under v4 is ≈ +44.2% × (37.82/25.91) = **~+65%** — to be confirmed on next Tue/Thu/Sun retrain.

**All numbers are after-tax.** Sim computes `total_return = final_portfolio_value / initial_cash - 1`, with tax debited on every sell (`compute_trade_tax`).

---

## What's new in v4

User request: *"如果 calibrate score 足够好可以买进已经 hold 的股票！这只股票好的话甚至可以全仓"* + *"A/B 当门槛，C 决定 size"*.

Shipped:
- `kernel/kelly.py` — continuous-Kelly sizing `f* = μ/σ²`
- `ApplyKellySizingTask` in PanelScoringJob — writes `kelly_target_pct` on both candidates AND holdings
- `TopUpHeldTask` after `SelectionJob` — emits BUY orders to bring existing holdings up to Kelly target when `kelly_target - current_pct > top_up_threshold`
- A-gate: `tiered_thresholds = [0.27, 0.45, 0.60]` anchored to calibrator `base_rate = 0.273` (pre-fix tier-1 = 0.10 was admitting "below random")

## Active config — Kelly + A-gate block

```json
"tiered_thresholds": [
  {"min_model_score": 0.27},   ← ≥ base_rate (positive edge required)
  {"min_model_score": 0.45},   ← + ~1σ rank_score spread
  {"min_model_score": 0.60}    ← + ~2σ
],
"ranking": {
  "kelly_sizing": {
    "enabled":           true,
    "fractional":        0.50,    ← half Kelly (sweep winner over 0.25)
    "max_concentration": 0.35,    ← single-ticker ceiling
    "min_edge":          0.0,
    "top_up_threshold":  0.05,    ← Δ(kelly_target, current_pct) → top-up
    "base_rate":         0.273
  }
}
```

## Kelly formula

`kernel/kelly.py::kelly_target_pct`:
```
f* = μ / σ²                       # classical continuous Kelly
target = min(max_pct, max_concentration, fractional × f*)
```
- `μ` = NGBoost predicted excess return
- `σ` = NGBoost predicted std
- `fractional = 0.50` halves full Kelly for estimation-error absorption
- `max_concentration = 0.35` hard cap per ticker
- `max_pct` = `regime_params.max_position_pct` × `confidence`

## Trade-off: streak 25d → 43d

Kelly is disciplined — skips bets with low μ/σ². Max consecutive no-trade 25d → **43d** over the 27-mo window. Average trade gap still 4.9d (115 buys / 570 days). `monitoring.max_no_trade_days=15` alert will fire more often under Kelly — that's by design, proves the system is deliberately idle not stuck.

## Why half-Kelly (0.50), not quarter (0.25)

- **Empirical:** half wins +1.59 APY pts over quarter in sweep.
- **Theoretical:** quarter-Kelly is industry safety default for *unreliable* μ estimates. Our μ comes from NGBoost trained on 52k rows with OOS-IC validation (+0.033). Estimation error is bounded, so less safety margin is appropriate.
- **Backstop:** `max_concentration=0.35` still caps any single name even when f*=1.0.

## Why A-gate (tier 1 = 0.27), not v3's 0.10

v3's tier 1 = 0.10 admits candidates with `P(outperform) < base_rate = 0.273` — i.e. *below chance*. Selection loop still worked because it picks in descending rank order, but the floor was too permissive: in quiet markets SelectionJob would take "the least bad" candidate. Re-anchoring to `≥ base_rate` means *only positive-edge candidates* can ever be selected.

**Empirical validation** (from AA decision-factor DB, commit `429298d`, 55k rows):
- `rank_score ≥ 0.10` (v3) → 71.1% P(fwd>0), +3.67% mean fwd_10d
- `rank_score ≥ 0.27` (v4) → **78.9% P(fwd>0), +5.28% mean fwd_10d** — real 8-pt step

## What's still v3 under the hood

- Hourly-enhanced 47k × 31 feature panel (Plan G)
- NGBoost head (Stage 2 μ,σ)
- Global calibrator (pooled, not regime-conditional)
- Universe floor sharpe 1.0 + held exemption (V fix)
- CUSUM cooldown only on regime SWITCH (B² fix)
- ntfy on every decision cycle

## Rollback

**Restore v3 (hourly without Kelly):**

```python
import json
for p in ('backtesting/renquant_104/strategy_config.json',
          'backtesting/renquant_104/strategy_config.golden.json'):
    c = json.load(open(p))
    c['tiered_thresholds'] = [{'min_model_score': t} for t in (0.10, 0.30, 0.50)]
    c['ranking']['kelly_sizing']['enabled'] = False
    json.dump(c, open(p, 'w'), indent=2)
```

**Restore v1 (tight xgb_params, pre-T4):**

```json
"panel_ltr": {
    "num_boost_round": 150,
    "xgb_params": {
      "eta": 0.02,
      "max_depth": 2,
      "min_child_weight": 100,
      "subsample": 0.4,
      "colsample_bytree": 0.4,
      "lambda": 10.0,
      "alpha": 5.0
    }
}
```
Then retrain panel: `python scripts/train_104.py --skip-baseline --skip-recalibrate --force`

---

## History

### v3 — hourly panel features (+4.18 APY pts)

Shipped 2026-04-23 as part of Plan G (`e65b081`). Enabling `panel_ltr.hourly.enabled` and retraining with 6 intraday-derived factor columns (morning_drift, afternoon_drift, vwap_premium, vol_ratio, intraday_realized_vol, overnight_gap) lifted after-tax APY **40.02% → +44.20%** on the same 27-month OOS sim, and lifted win rate 79% → **82%**. OOS panel IC actually dropped (+0.0411 → +0.0326) — the live improvement came from feature-interaction lifts in specific regimes that averaged IC doesn't capture. Full writeup: `doc/experiments/panel-training-runs.md` (2026-04-23 late PT).

| Metric | v2 | v3 (hourly-enhanced) |
|---|---|---|
| Final value | $211,639 | **$225,573** |
| APY | +40.1% | **+44.20%** |
| Win rate | 77% | **82%** |
| Avg P&L / trade | +11.6% | **+13.6%** |
| Longest no-trade streak | 22d | 26d |
| Rotations | 19 | 19 |

### v2 — T4 xgb_params revert (+7 APY pts)

Shipped 2026-04-23 as the T4 revert (`ee4faab`). Reverted `panel_ltr.num_boost_round`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `lambda`, `alpha` to the pre-regression pre-2026-04-22 values. This alone moved APY 33.1% → **40.1%** without touching any other config — confirming the 2026-04-22 regression was primarily panel-capacity driven, not systemic.

| Key delta vs v1 |
|---|
| `num_boost_round`: 150 → 300 |
| `max_depth`: 2 → 3 |
| `min_child_weight`: 100 → 60 |
| `subsample`: 0.4 → 0.5 |
| `colsample_bytree`: 0.4 → 0.5 |
| `lambda`: 10.0 → 5.0 |
| `alpha`: 5.0 → 2.0 |

### v1 — drawdown skip_buys reset (2.4% → 33.1%)

Shipped 2026-04-23 as the first stage recovery (`d3ef68f`, on top of `e586018`).

| Metric | Value |
|---|---|
| Final value | $188,693 |
| APY | **+33.1%** |
| Win rate | 72% |
| Avg P&L per trade | +9.6% |
| Longest no-trade streak | 22d (from 153d pre-fix) |
| Rotations | 1 |

Recovery progression:

| Stage | Commit | Fix | APY |
|---|---|---|---|
| Regression (2026-04-22) | `5fdba09` + later | buggy state | 2.4% |
| R1 | `2df4e21` | universe_floor sharpe preference + floor 1.0 | 10.1% |
| R2 | `33c0e9b` | ConfidenceVeto disabled | 10.9% |
| R3 (v1) | `e586018` | **drawdown skip_buys resets on recovery** | **33.1%** |
| R4 (v2) | `ee4faab` | panel xgb_params revert (T4) | 40.1% |
| G (v3) | `e65b081` | hourly panel features | 44.20% |
| Kelly (v4) | `eb8fab5` | A-gate + half-Kelly sizing | **+65% live expected** |

R3's drawdown-reset fix was the dominant APY lever (2.4% → 33.1%); v2-v4 each stacked on top via capacity + features + sizing.

---

## Panel-LTR cross-validated IC (CPCV 15-split)

| Metric | v1 | v2 | v3 | v4 |
|---|---:|---:|---:|---:|
| OOS mean IC | +0.0397 | +0.0411 | +0.0326 | +0.0326 (same panel, Kelly sizing only) |
| OOS std IC | 0.0267 | 0.0243 | — | — |
| OOS q05 IC | +0.0100 | +0.0123 | — | — |
| Train/OOS ratio | 4.3× | 6.4× | — | — |

---

## Sustainability signals

**Win rate up, not down.** v3 82% vs v1 72%. A config trading more from overfit noise would *dilute* win rate. Instead v3 takes 13 more round-trips AND wins MORE often.

**Avg P&L per trade up.** +13.6% (v3) vs +9.6% (v1). Larger moves caught, not more paper-cuts.

**CPCV OOS IC improved** on v2 (+0.0411 vs v1 +0.0397). Panel ranks forward-returns better across the 15 OOS windows.

One yellow flag on v2/v3: **Train/OOS ratio widened** (4.3× → 6.4×). Mitigated by the win-rate + OOS-IC evidence. Watch: if live trading shows a 30-day window with APY < 30% sustained, consider rolling back to v1's tight `xgb_params`.

---

## Related commits

- `2df4e21` — universe_floor: prefer tournament sharpe + floor 1.0
- `33c0e9b` — regime: disable ConfidenceVeto (GMM posterior cap)
- `e586018` — drawdown: reset `skip_buys` on recovery (THE fix)
- `d3ef68f` — v1 golden snapshot
- `ee4faab` — T4 xgb_params revert (v2)
- `e65b081` — Plan G hourly panel features (v3)
- `eb8fab5` — v4 Kelly + A-gate promote

## Regression tests guarding this state

- `tests/test_universe_alignment.py::TestSharpeEvaluatorPrefersTournament`
- `tests/test_pipeline.py::TestDrawdownCircuitTaskResets`
- `tests/test_kelly_sizing.py` (Kelly math)
- `tests/test_top_up_held.py` (TopUpHeldTask)
- `tests/test_trim_held.py` (TrimHeldTask — Kelly rebalance sells)
- `tests/test_kelly_rotation_gate.py` (BC rotation Kelly delta)

Any future touch of `_eval_sharpe`, `DrawdownCircuitTask`, Kelly sizing, or the tier threshold math that breaks these tests would re-introduce the regression that made these fixes necessary.
