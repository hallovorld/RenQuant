# 2026-05-17 Dense Panel Outcome — Detector Is The Blocker


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

Status: PAUSED — handoff for resumption.

## TL;DR

The A1-v2 + B-track dense panel ran on 8 × 6-week windows specifically
chosen in BEAR/CHOPPY-dense zones. **The overlays only fired in 1 of 8
windows** (W4: 2022-08-15..10-01). The dense panel did NOT measure the
overlay theses — it measured the regime detector.

**Verdict: the regime detector still mis-labels catastrophic-loss
periods as BULL_CALM, even after the 5/15 `3925c0d` MA50 fix.** Until
the detector is fixed, every `regime_params.{BEAR,CHOPPY}.*` knob is
decorative.

## The 8-window evidence

Source: `data/logs/sim_baseline_2026-05-16_dense/logs/W*.log` regime tags.

| window | period                       | baseline APY | BEAR | CHOPPY | BULL_CALM | overlay fired |
|--------|------------------------------|--------------|------|--------|-----------|---------------|
| W1     | 2022-04-01..05-15            | −45.5%       | 8    | 0      | 22        | no            |
| W2     | 2022-05-15..07-01            | 0%           | 7    | 0      | 29        | no            |
| W3     | 2022-07-01..08-15 (rally)    | +67.8%       | 0    | 0      | 31        | no            |
| W4     | 2022-08-15..10-01            | −53.1%       | 12   | 0      | 22        | **SDL +2.5pp** |
| W5     | 2023-02-15..04-01 (SVB)      | −36.6%       | 0    | 0      | 32        | no            |
| W6     | 2023-10-15..12-01            | +36.6%       | 0    | 0      | 34        | no            |
| W7     | 2024-07-15..08-31 (Aug vol)  | −3.4%        | 0    | 0      | 35        | no            |
| W8     | 2025-01-15..03-01 (DeepSeek) | −22.8%       | 0    | 0      | 31        | no            |

Key facts:
- **CHOPPY = 0 bars in every window** — the CHOPPY label is dead under
  the current detector.
- **SVB crisis (−36.6% in 6 weeks) → 100% BULL_CALM.**
- **DeepSeek + tariff vol (−22.8%) → 100% BULL_CALM.**
- **Start of 2022 bear (−45.5%) → 28% BEAR + 72% BULL_CALM.**
- Only W4 (Sept 2022 rate-fear leg) has enough BEAR labels (12/79 bars)
  to even fire the overlay, and even there the QP CVaR knob never
  triggered (probably because no rebalance happened on those 12 bars).

## What this means for A1 + B-track

The rigorous analyzer (auto-run when btrack W2 completes — currently
mid-run at pause time) will report pooled-mean Δ for sdl_n2_BC and
cvar025_BC. Those numbers are effectively **n=1 active observation**
diluted with 7 NOOPs. Pool ≈ +0.31pp / +0.00pp respectively — within
noise of a single window.

**Do not interpret the analyzer verdict as a thesis test.** The thesis
("σ-aware SDL helps in BEAR/CHOPPY", "per-regime CVaR helps in
BEAR/CHOPPY") was never measured on enough firing bars to resolve.

The wiring works (W4 SDL +2.5pp confirms the kernel patch +
regime_params plumbing). The thesis can't be tested until the detector
fires.

## NGB trainer status at pause

PID 97229 alive at 11:54. Started 10:12:22 with single fit, full-batch
+ natural gradient on 568,563 rows × 169 features, n_estimators=400,
lr=0.01. No per-iteration logging (verbose=False). No artifact yet at
`backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json`.

On completion:
- Verify artifact size + sha256 fingerprint vs current prod
  (`backtesting/renquant_104/artifacts/prod/ngboost-head.alpha158_fund.json`
   md5=b90a23d49bbb5f15a2cbe915fd11a2e6, 2.9 MB, mtime 2026-05-16 21:30
   — that prod copy is the stale May-12 file copied during the no-op
   "promotion" earlier this session).
- If new artifact's val_IC is in the +0.030-+0.040 range (consistent
  with the 5-seed CONFIRMED result), backup prod and copy.
- **NGB σ wire activation** (`use_ngboost_sigma=true`) is a real-$
  Kelly behavior change — **DO NOT FLIP without user authorization**
  per CLAUDE.md §5.5.

## Detector audit — DONE 2026-05-17 13:00

Script: `scripts/audit_regime_detector.py` — replays the exact routing
from `task_regime.py::{BEAROverrideTask, RegimeFinalizeTask}` on five
known-objective windows using local SPY parquet. Output:
`data/logs/reeval_results/regime_detector_audit_2026-05-17.txt`.

Defaults (golden config): `BEAR_VOL_THR=0.35`, `BEAR_RET_THR=-0.08`,
`VOL_WINDOW=20`, `HURST=(0.52, 0.65)`.

| window                | period                    | SPY %  | BEAR | CHOPPY | BULL_CALM | vol20 max | ret20 min | hurst range |
|-----------------------|---------------------------|--------|------|--------|-----------|-----------|-----------|-------------|
| 2022_Q2_BEAR_START    | 2022-03-01..05-15         | −6.6%  | 34   | 0      | 19        | 32.4%     | −11.5%    | 0.65..0.80  |
| 2022_DEEP_RATEFEAR    | 2022-07-15..10-01         | −7.3%  | 24   | 0      | 31        | 25.2%     | −12.3%    | 0.67..0.77  |
| 2023_Q1_SVB           | 2023-01-15..04-01         | +2.9%  | 6    | 0      | 47        | **19.2%** | **−5.7%** | 0.58..0.74  |
| 2024_AUG_VOL_SPIKE    | 2024-06-15..08-31         | +3.0%  | 0    | 0      | 53        | **22.3%** | **−7.6%** | 0.70..0.81  |
| 2025_DEEPSEEK_TARIFF  | 2024-12-01..2025-03-01    | −1.6%  | 0    | 0      | 60        | **17.7%** | **−4.3%** | 0.66..0.79  |

### Two distinct failure modes

**Failure 1 — BEAR thresholds are GFC-calibrated, miss brief crises.**
`hard_bear` requires `vol_20d > 35%` (annualized) OR `ret_20d < −8%`.
SVB, DeepSeek, and the August-2024 vol spike never crossed either
threshold. The Hurst-MOMENTUM fallback also needs SPY to break both
MA50 AND MA200 — brief crises don't.
- SVB: max vol 19.2%, max drawdown −5.7%. Detector: 11% BEAR.
- Aug 2024: max vol 22.3%, max drawdown −7.6%. Detector: 0% BEAR.
- DeepSeek+tariff: max vol 17.7%, max drawdown −4.3%. Detector: 0% BEAR.

35% annualized vol = 2.2% daily, GFC-level. The real-world frequency
of windows above this is ~1 in 5 years. A detector calibrated this
way is correct on 2008 and 2020 and blind to everything else.

**Failure 2 — CHOPPY is dead.** Zero CHOPPY labels across 5 windows /
274 trading days. CHOPPY needs `Hurst < 0.52` (REVERSION); SPY almost
never anti-persistent because long-term trend dominates 63-bar rolling
window. Every `regime_params.CHOPPY.*` knob in the strategy config is
decorative.

### Fix candidates (ranked by leverage)

**A. Add short-horizon BEAR trigger.** Add `bear_vol_threshold_5d`
(e.g., 0.25) and `bear_return_threshold_5d` (e.g., −0.04). Catches
1-week crises before they become 1-month bears. SVB 5-day drop was
−4.5% (would fire); DeepSeek 5-day drawdown around tariff news ~−3.5%
(borderline); Aug-2024 5-day −6.4% (would fire). **Highest leverage,
lowest false-positive risk (5-day windows are noisy but symmetric).**

**B. Lower 20-day vol threshold from 0.35 to 0.20-0.25.** Risk: false
BEAR labels in routine 1.5%-daily bull-rally vol. Calibration would
need a known-objective backtest (2022-Q1, 2023-Q4, 2024 BULL_CALM
periods) to verify FP rate.

**C. Resurrect CHOPPY via vol-clustering, not Hurst.** Define CHOPPY =
`(realized_vol_5d > rolling_vol_60d_p75) AND (|spy_drift_20d| < 2%)`.
Hurst is the wrong tool for cross-sectional choppiness on SPY.

**D. Use MA50-only with 3-of-5 persistence in BEAR-from-Hurst gate.**
Drop MA200, replace with "SPY < MA50 on ≥3 of last 5 days." Catches
SVB (broke MA50 on 3/9; below for 7 of next 10 days). Risk: regresses
the 2026-05-14 Q11 BULL_STRONG fix (8% false-BEAR days from MA50 alone).

### Decision point

User input needed: which combination to land first?
- A+C is the safest combo (orthogonal axes, low FP risk on routine bull regimes)
- A+B is the highest-impact combo (BEAR signal density jumps everywhere)
- A alone is the smallest-blast-radius pilot

After fix lands, regression test: re-run `scripts/audit_regime_detector.py`
+ check Q11 (2024 BULL_STRONG) doesn't regress to >5% false-BEAR days,
then re-run dense panel on the same 8 windows and re-evaluate
A1/B-track verdicts.

## ~~Resumption — next experiment, in priority order~~

~~### P0: Detector audit (blocks everything regime-conditional)~~

Hypothesis: the Hurst/MA50 logic in `3925c0d` requires SPY persistence
below MA50 to label BEAR. SVB (1-week), DeepSeek (1-2 weeks), and the
start of 2022 bear are too short for the persistence filter, so they
land in BULL_CALM. CHOPPY label likely has stricter conditions that
never fire under the current calibration.

Audit checklist:
1. Read `kernel/regime.py` (standalone) AND
   `kernel/pipeline/task_regime.py` (production task path — both must
   stay in sync per CLAUDE.md PRIME DIRECTIVE).
2. Run the detector standalone on SPY 2023-02-15..04-01 (SVB) and
   2025-01-15..03-01 (DeepSeek) bar-by-bar — print:
     - Hurst exponent per day
     - MA20 vs MA50 vs MA200
     - SPY return MTD / 60-day
     - Detected regime label
3. Find the conditional branch that routes SVB → BULL_CALM instead of
   BEAR/CHOPPY. Likely culprits:
     - Hurst > 0.65 forces "trending" branch which goes BULL_CALM
       regardless of direction (this was the 5/14 bug, supposedly
       fixed by `3925c0d`)
     - MA50 persistence requirement too long
     - CHOPPY label requires vol > X AND |Hurst − 0.5| < Y — calibration
       may be wrong
4. Test fix on KNOWN-OBJECTIVE windows: 2022-Q2 (DEEP_BEAR), 2023-Q1
   (SVB BEAR_BRIEF), 2023-Q4 BULL_CALM, 2024-Q3 CHOPPY_BRIEF, 2025-Q1
   DEEPSEEK_BEAR_BRIEF.
5. Target: ≥40% non-BULL_CALM labels in any of these 5 known-bear
   windows.

### P1: Re-run dense panel after detector fix

The 3 panel configs (`sim_baseline_2026-05-16`,
`sim_overlay_sdl_n2_BC`, `sim_btrack_cvar025_BC`) and the runner
(`scripts/run_dense_panel.sh` for parallel /
`scripts/rerun_dense_w1w2.sh` for sequential) are ready to re-launch.
Should take ~90min total sequential.

### P2: NGB σ wire decision

If new artifact lands and beats sim XGB-quantile baseline → ask user
about `use_ngboost_sigma=true`. NOT in scope of autonomous-action
memory (real-$ Kelly behavior change).

## Files touched this session (post-pause)

Commits 91a28df and 39ff891 already on origin/main:
- `scripts/build_regime_overlay_configs.py`
- `scripts/build_btrack_cvar_per_regime.py`
- `scripts/run_dense_panel.sh`
- `scripts/rerun_dense_w1w2.sh` (NEW — sequential W1/W2 re-run after
  parallel OOM SIGTERM)
- `scripts/monitor_panel_health.sh` (added is_overlay_batch guard)
- `scripts/validate_sim_config_active.py` (added per-regime QP paths)
- `backtesting/renquant_104/kernel/portfolio_qp/tasks.py`
  (`_qp_cfg` per-regime override layer)
- `tests/test_qp_cfg_per_regime_override.py` (8 tests, all green)
- `backtesting/renquant_104/strategy_config.sim_btrack_cvar025_BC.json`
- `doc/research/2026-05-16-experiment-master-plan.md`

Outputs at pause:
- `data/logs/sim_baseline_2026-05-16_dense/equity/W{1..8}.json` (8/8 ✓)
- `data/logs/sim_overlay_sdl_n2_BC_dense/equity/W{1..8}.json` (8/8 ✓)
- `data/logs/sim_btrack_cvar025_BC_dense/equity/W{1..7}.json` (7/8 —
  W2 in progress at pause)
- `data/logs/reeval_results/dense_2026-05-16_rigorous.md` (will be
  written by `rerun_dense_w1w2.sh` when btrack W2 finishes)
- `backtesting/renquant_104/artifacts/ngboost-head.alpha158_fund.json`
  (will be written when NGB trainer PID 97229 finishes)

## Active background processes at pause

| PID    | what                                        |
|--------|---------------------------------------------|
| 97187  | `scripts/rerun_dense_w1w2.sh` (waiting on btrack W2) |
| 97229  | `train_ngboost_alpha158_fund.py` (single fit, ~2h elapsed, no log) |
