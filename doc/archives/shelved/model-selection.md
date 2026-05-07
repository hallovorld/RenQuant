# Model Selection — Systematic 4-Tier SOP

**Status**: Phase 1+2+3+4a shipped (2026-04-26). Phase 4c (auto-reject after shadow) deferred pending empirical thresholds.

This document is the **authoritative SOP** for deciding which model artifact ships to production. Before this doc existed, decisions were ad-hoc — agent recommendations, manual eyeballing of OOS IC, no formal cross-backend tournament. Round-7 audit caught that the macro-enabled XGBoost (OOS IC −18% vs prod) would have AUTO-PROMOTED under the old 30% G4 default.

When to read this doc:
- About to retrain → see [Tier 1-2: Acceptance Gates](#tier-1--2-acceptance-gates).
- Multiple `.bak.json` artifacts on disk, picking a winner → see [Tier 3: Backend Tournament](#tier-3-backend-tournament).
- New backend / radically different artifact → see [Tier 4: Shadow / Challenger](#tier-4-shadow--challenger).
- Disagree with a default threshold → see [Configuration](#configuration).

---

## The Problem

A retrained model auto-replaces `panel-ltr.json` once training succeeds. If the new model is worse — calibrator collapsed, OOS IC negative, schema changed unexpectedly — live trading uses it from the next bar. **There is no rollback once it ships unless an operator manually swaps `.previous.json` back.**

The systematic answer: a **layered defense** at four tiers, with progressively tighter / more expensive checks.

---

## The 4 Tiers

| Tier | Layer | Catches | Cost | Today's status |
|------|-------|---------|------|---------------|
| 1 | Catastrophic Block (G1-G3, G5-G6) | Schema mismatch, NaN scores, calibrator collapse | ~0.1s | ✅ Phase 1 |
| 2 | Regression Block (G4, G7-G8) | OOS-IC degradation, sub-floor IC, zero-variance scores | ~0.1s | ✅ Phase 1 |
| 3 | Promotion Floor (G9-G11) | Sim APY/Sharpe regression, turnover bloat | 30-60s sim | ✅ Phase 2 (gates ship; sim runner stub) |
| 3.5 | Backend Tournament | Pick winner among multiple `.bak.json` candidates | manual | ✅ Phase 3 (`select_best_model.py`) |
| 4a | Shadow Logging | Record what challenger would have done, no trade | n/a (platform) | ✅ Phase 4a |
| 4c | Auto-Reject after Shadow | Promote/reject post-window via G12 | tbd | ⏳ Deferred |

---

## Tier 1 + 2: Acceptance Gates

When `scripts/train_104.py` finishes, **before** the new artifact replaces production, `kernel.model_acceptance.ModelAcceptanceGate` runs 11 gates. Hard fail → keep prior, archive staging to `_acceptance_log/`. Soft fail → warn but promote.

Implementation: `backtesting/renquant_104/kernel/model_acceptance.py:332`. Wired in `scripts/train_104.py:111`.

### Gate inventory

| Gate | Severity (default) | Catches | Default threshold | Tunable key |
|------|---|---|---|---|
| G1 schema | hard | `feature_cols` shrunk or unexpected diff vs prior | identical or strict superset | — |
| G2 calibrator unique | hard | Calibrator probability head collapsed to constant | `n_unique_prob_y ≥ 5` | — |
| G3 pool_ic positive | hard | Calibrator inverted the signal | `pool_ic > 0` | — |
| G4 OOS IC vs prior | **hard** | Catastrophic IC regression | `≥ prior × (1 - 0.05)` | `g4_max_degradation` |
| G5 score range | hard | Output collapsed to base-rate | `coverage ≥ 80%` of prior span | — |
| G6 inference smoke | hard | NaN-on-any-input post-serialization bug | `all_finite` | — |
| G7 OOS IC absolute floor | **hard** | Tiny-but-positive IC (signal vs noise) | `≥ 0.02` | `g7_floor`, `g7_severity` |
| G8 per-ticker variance | soft | Same score for every ticker on a bar | `score_std ≥ 0.001` | `g8_min_std`, `g8_severity` |
| G9 sim APY vs prior | hard | Real-money APY regression | `≥ prior - 1.0pp` | `g9_max_pp_drop`, `g9_severity` |
| G10 sim Sharpe vs prior | hard | Risk-adjusted return regression | `≥ prior - 0.1` | `g10_max_sharpe_drop`, `g10_severity` |
| G11 turnover ratio | soft | Slippage/tax bloat | `≤ prior × 1.5` | `g11_max_multiplier`, `g11_severity` |

### Threshold rationale

**G4 = 5% (was 30%)** — the macro-vs-prod case (-18.5%) MUST fail. 5% allows for normal CPCV-fold noise but blocks anything materially worse. Loosen if doing exploratory rebuilds known to drift.

**G7 = 0.02** — IC below this is noise-floor territory; not worth shipping even if vs-prior unchanged. Set per `doc/components/calibration.md` saturation analysis.

**G9 = 1pp APY drop** — within typical 27-mo OOS sim noise (we've seen ±0.5pp from rerunning the same model). 1pp = 1σ-ish "is this just noise or a real regression?" boundary.

**G10 = 0.1 Sharpe drop** — same logic, dimensional translation.

**G11 = 1.5x multiplier** — generous; turnover doubling on the same APY/Sharpe is a real signal of a worse model (slippage-adjusted P&L will be lower).

### Bypass paths

```bash
# operator override for one run (DANGEROUS — only for known-broken-but-recoverable)
python scripts/train_104.py --skip-acceptance

# OR disable gates entirely via config
"acceptance": {"enabled": false}
```

---

## Tier 3: Backend Tournament

When you have multiple `.bak.json` artifacts (xgboost vs lightgbm vs transformer vs macro-enabled, etc.), pick a winner with:

```bash
python scripts/select_best_model.py --strategy renquant_104
python scripts/select_best_model.py --strategy renquant_104 --weights "ic=0.7,sharpe=0.2,calmar=0.1"
python scripts/select_best_model.py --strategy renquant_104 --promote xgboost
```

**Composite score** = `z(oos_mean_ic)·w_ic + z(sim_sharpe)·w_sh + z(sim_calmar)·w_cal`.

Default weights: `ic=0.5, sharpe=0.3, calmar=0.2`. Override per situation:
- New backend, no sim run yet → `ic=1, sharpe=0, calmar=0` (IC-only)
- Reducing turnover-driven slippage cost → bump calmar weight
- Risk-conscious operator → bump sharpe weight

**Caveats** the script surfaces at runtime:
- Different `panel_rows` across candidates → IC not directly comparable. Verify train sets overlap before trusting the rank.
- Missing `sim_smoke` metrics → that candidate's `z=0` (neutral). Run sim_smoke first if it's a top candidate.

Implementation: `scripts/select_best_model.py`. Tests: `tests/test_select_best_model.py`.

### When to invoke

- After a multi-backend retrain weekend.
- After a cross-asset feature experiment (e.g., adding macro factors).
- Before any major promotion when 2+ candidates exist.

---

## Tier 4: Shadow / Challenger

A new backend or major architecture change deserves a **shadow period** — challenger runs alongside live production, scoring the same universe, but does NOT trade. After N sessions, operator compares decisions to validate.

### Phase 4a (today)

Schema + API + config block shipped. Live wiring is Phase 4b.

```json
"acceptance": {
  "challenger": {
    "enabled": false,
    "artifact_path": null,
    "name": null,
    "shadow_period_days": 0
  }
}
```

To enable (when Phase 4b lands):

```json
"challenger": {
  "enabled": true,
  "artifact_path": "artifacts/panel-ltr.macro-enabled.bak.json",
  "name": "macro-enabled",
  "shadow_period_days": 14
}
```

### DB schema

`runs.db.challenger_decisions`:
- `decision_id` PK
- `run_id`, `decision_date`, `ticker`
- `challenger_name`, `challenger_score`, `challenger_rank_score`, `challenger_action`
- `actual_score`, `actual_action`
- `created_at`

Indexes: `idx_challenger_run`, `idx_challenger_window`.

### Operator verdict (after shadow_period_days)

```python
from kernel.challenger import compare_window
import sqlite3
conn = sqlite3.connect("data/runs.db")
verdict = compare_window(
    conn,
    challenger_name="macro-enabled",
    start_date=pd.Timestamp("2026-04-12"),
    end_date=pd.Timestamp("2026-04-26"),
)
# verdict = {n_decisions, agreement_rate, challenger_only_buy,
#            live_only_buy, score_corr, score_rank_corr}
```

Operator reads, decides, runs `select_best_model.py --promote` if convinced.

### Phase 4c: Auto-Reject (deferred)

When we have empirical priors on what `agreement_rate` / `score_corr` to expect, we add a G12 gate that reads `metadata.challenger_window_verdict` (written by `scripts/finalize_challenger.py`) and auto-promotes/rejects.

**Why deferred**: defining "good enough" agreement rate without data is a guess. Phase 4a collects the data; revisit when 2+ shadow windows are complete.

---

## Configuration

All gate thresholds + severities live under `acceptance` in `strategy_config.json` (and `strategy_config.golden.json`). Default values:

```json
{
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
}
```

To downgrade a hard gate: set `g7_severity: "soft"`. To loosen G4 for an exploratory window: `g4_max_degradation: 0.15`.

---

## Decision Matrix

| Scenario | Use which tier(s)? |
|----------|---|
| Daily retrain (small panel update) | T1+T2 (Phase 1+2 gates auto) |
| Hyperparameter sweep | T1+T2; if winner ≥ +2pt APY, ship per CLAUDE.md §2a |
| Adding new feature (e.g., macro) | T1+T2 + Tier 3 (compare against `.bak`) |
| New backend (LightGBM, Transformer, …) | All four tiers; require shadow period |
| Recovering from a bad ship | `kernel.model_acceptance.rollback()` → restore `.previous.json` |
| Architecture overhaul (e.g., adding NGBoost head) | All four tiers; require Phase 4 shadow |

---

## What this SOP does NOT do

- **Does not run the actual sim** for G9-G11 metrics. `kernel/sim_smoke.py` ships a stub `run_smoke_test()`; operator wires their own SimAdapter run and calls `add_smoke_metrics_to_artifact()` to populate metadata. Phase 5 (future) integrates this with `select_best_model.py`.
- **Does not auto-promote a challenger** post-shadow. Operator decides; Phase 4c may add G12 once we have priors.
- **Does not enforce composite-score winner promotion**. `select_best_model.py --promote` is operator-triggered; the tournament gives a recommendation, not a mandate.

---

## File index

| File | Purpose |
|------|---------|
| `kernel/model_acceptance.py` | Gate definitions G1-G11, `ModelAcceptanceGate`, `promote() / reject() / rollback()` |
| `kernel/sim_smoke.py` | `compute_metrics_from_equity_curve`, `add_smoke_metrics_to_artifact`, `run_smoke_test` (stub) |
| `kernel/challenger.py` | `ChallengerConfig`, `ChallengerEvaluator`, `log_decision`, `compare_window` |
| `kernel/persistence.py` | `challenger_decisions` table schema |
| `scripts/train_104.py` | Wires gates into FullTrainingPipeline output |
| `scripts/select_best_model.py` | Backend tournament + `--promote` |
| `tests/test_model_acceptance.py` | 36 gate tests (Phase 1) |
| `tests/test_acceptance_phase2.py` | 25 G9/G10/G11 + sim_smoke tests |
| `tests/test_select_best_model.py` | 14 tournament tests |
| `tests/test_acceptance_phase4.py` | 13 challenger tests |

**Total**: 88 model-selection tests pinning the SOP semantics.

---

## Change log

- **2026-04-26 Phase 4a**: shipped challenger / shadow infrastructure (platform only)
- **2026-04-26 Phase 3**: shipped `select_best_model.py` backend tournament
- **2026-04-26 Phase 2**: added G9 / G10 / G11 sim-based gates + `sim_smoke` helper
- **2026-04-26 Phase 1**: tightened G4 (30%→5%) + hardened G7 (soft→hard); config-driven thresholds
- **2026-04-26 round-7 audit**: original ModelAcceptanceGate framework with 8 gates
