# RenQuant 104 Training Trust & High-Level Design Review


> **📅 Historical snapshot — content below reflects state at the date in filename/header.**
> Verify against current code per CLAUDE.md §1 "code is the source of truth" before acting on
> present-tense claims. For current state see `doc/roadmap.md` § "📍 Current state" +
> `CLAUDE.md` § "🗂 Current state".

Date: 2026-05-15

Author: Codex

Purpose: give Claude Code a sharp, high-level review of whether the RenQuant 104 model training process and model outputs are trustworthy, and what design changes should be made before trusting the system with autonomous live capital.

Related files:

- `doc/audits/renquant_104_deep_audit_2026-05-15.md`
- `doc/audits/renquant_104_claude_claim_check_2026-05-15.md`
- `logs/ngb_proper/2026-05-15.log`

## Executive Verdict

RenQuant 104 is research-promising, but not production-trustworthy yet.

The model may contain real signal. The panel-LTR direction is reasonable, and the NGBoost rerun has encouraging early evidence. But the full chain from training to live execution is not clean enough to trust autonomously:

```text
data -> features -> labels -> model -> calibration -> sizing -> QP allocator -> broker execution
```

Several links in that chain are still weak. A good alpha model can still lose money if calibration, expected-return units, artifact lineage, data freshness, sizing, or allocation logic are wrong.

Current rating:

```text
Alpha research value:        promising
Training process trust:      low / medium
Backtest trust:              low / medium
Live autonomous trust:       low
Paper / shadow-mode trust:   acceptable and useful
```

Do not treat the current 104 stack as "fixed" just because several narrow regression tests pass.

## Is The Training Process Trustable?

Not yet.

The main problem is not XGBoost or NGBoost. The main problem is model governance: artifact lineage, promotion discipline, calibration contracts, path consistency, experiment reproducibility, and production-readiness gates.

Observed concerns:

1. Experiment bookkeeping is weak.

Claude claimed commit `7e1f80c`, but that commit does not exist in this checkout. The actual local commits are:

```text
e65877a test: ship 3 audit-mandated regression tests + p0activated sim configs
555f5b1 fix(qp): hoist vol-target + DD-Kelly out of dormant Kelly path
```

The tests are real and pass, but a wrong commit reference is a warning sign for experiment traceability.

2. Training/inference artifact paths are inconsistent.

The system still has flat artifact paths, `artifacts/prod/`, `artifacts/sim/`, symlinks, and config-specific variants. This creates a real risk that training validates one artifact while live inference loads another.

3. Calibrator/scorer pairing is not strict enough.

The model scorer, global calibrator, NGBoost head, and QP expected-return layer must be treated as one contract. At present, they can drift.

4. Runtime calibration clipping is a red flag.

The p0activated sim logs show:

```text
expected_return.y has max|y|=0.9540 > 0.20 sanity bound ... clipping to +/-0.20
Until the calibrator is retrained with clipped targets, Kelly sizing on this signal is suspect.
```

Runtime clipping is not a clean production fix. The calibrator should be trained on the intended target transform and validated before promotion.

5. `sample_end` / date semantics are suspect.

A future-dated `sample_end` and inconsistent data freshness semantics make OOS boundaries and live freshness harder to trust.

6. Long-running experiments are not controlled well enough.

`scripts/train_ngboost_proper.py --help` starts full training because the script has no argparse/help path. During verification, two `--help` processes were consuming near-full CPU. That is not a model-quality issue, but it is a serious experiment-control smell.

7. The p0activated A/B validation is incomplete.

The p0activated run was genuinely running, but at the time of review only partial windows had landed. Early windows were mixed:

```text
Q01 -4.3%
Q02 -3.8%
Q03 -1.5%
Q04 -0.3%
Q05 +2.9%
Q06 -5.3%
```

This is not enough to declare the new flags validated.

## Is The Model Trustable?

The model is plausible. It is not yet trusted.

Evidence in favor:

1. Panel-LTR is the right class of model for the stated goal.

Cross-sectional ranking is more appropriate than isolated per-ticker binary classification for a watchlist allocation problem.

2. NGBoost has encouraging early results.

From `logs/ngb_proper/2026-05-15.log`:

```text
seed=42   val_ic=+0.0372  sigma-calib=+0.275
seed=7    val_ic=+0.0365  sigma-calib=+0.269
seed=123  val_ic=+0.0293  sigma-calib=+0.273
```

This supports the hypothesis that NGBoost sigma contains information and is not pure noise. However, 3 seeds are not the final proof. Seed 123 is essentially equal to the XGB baseline, so the 5-seed test still matters.

3. The system has meaningful test coverage in parts.

The new tests prove that vol-target and drawdown-Kelly exposure scaling can hit QP upper bounds independently of NGBoost.

Evidence against trusting it yet:

1. Good IC does not guarantee good live portfolio behavior.

The decisive object is not the model alone. It is the full trading system after calibration, sizing, risk constraints, QP allocation, taxes, turnover, and execution.

2. QP and sizing behavior still show instability.

p0activated logs include repeated QP infeasibility, Kelly zeroing, and insufficient-cash warnings after top-up orders.

3. Feature-health warnings still appear.

Some one-ticker defensive cases may naturally have low cross-sectional diversity, but repeated feature-collapse warnings should not be hand-waved away. A panel ranker using degraded runtime features can look statistically valid in training and still behave badly in production.

4. Live-score saturation and calibrator drift remain central risks.

The live system previously showed saturated rank scores. Unless the promotion gate proves score distributions are inside calibration support, the model output is not reliable for sizing.

## Highest-Level Design Issue

RenQuant 104 currently has too many components that are allowed to make final trading decisions.

The architecture should become:

```text
Models propose expected return / rank / uncertainty.
Risk gates veto impossible or disallowed trades.
One allocator owns final target weights.
Execution translates final weights into orders.
```

Right now, the boundaries are blurry:

- panel score can become rank score;
- calibration can become expected return;
- NGBoost sigma can influence sizing;
- Kelly can zero candidates;
- QP can solve weights;
- post-QP top-up can still emit orders;
- sell gates can fire independently;
- config flags can silently change which layer owns the trade.

This is too much hidden coupling for a live quant system.

## Recommended Target Architecture

### 1. Artifact Registry

Stop treating loose JSON files and symlinks as production artifacts.

Every promoted model bundle should have a manifest:

```json
{
  "model_id": "panel_ltr_2026_05_15_xxx",
  "git_commit": "...",
  "training_command": "...",
  "strategy_config_hash": "...",
  "data_fingerprint": "...",
  "feature_schema_hash": "...",
  "label_schema_hash": "...",
  "scorer_artifact": "...",
  "calibrator_artifact": "...",
  "ngboost_artifact": "...",
  "qp_config_hash": "...",
  "validation_report": "...",
  "promotion_status": "PROMOTED"
}
```

Live runner should load only `PROMOTED` bundles.

### 2. Promotion Gate

No artifact should be eligible for live trading unless it passes a fixed promotion suite.

Minimum required checks:

- 5-seed IC and t-test versus current baseline
- 16-window A/B simulation
- pre-2024 / post-2024 split
- regime-level performance table
- turnover and tax drag table
- max drawdown and tail-loss table
- calibration support / saturation report
- feature coverage and feature-collapse report
- scorer/calibrator/ngboost fingerprint match
- QP feasibility rate
- no-trade streak report
- live dry-run replay

Promotion should fail hard if any of these are missing.

### 3. Calibration As A First-Class Model Component

The calibrator is not a helper. It is part of the model.

Rules:

- scorer and calibrator must be trained and promoted together;
- expected-return target clipping must happen during training, not at runtime;
- live scores outside calibrator support should hard-fail or degrade to safe mode;
- calibration output units must be explicit: probability, expected return, z-score, or rank percentile;
- QP must not consume a rank score as if it were expected return.

### 4. One Allocator Rule

If QP is enabled, QP must be the only component that emits final buy/sell target weights.

No post-QP top-up should fire. No second allocator should mutate positions after QP. Other layers may propose candidates, veto trades, or set constraints, but QP should own final weights.

Suggested ownership:

```text
Candidate jobs: propose names
Scoring jobs: estimate alpha / uncertainty
Risk gates: apply hard constraints
QP: choose target weights
Execution: convert target-weight delta to broker orders
```

### 5. Expected-Return Unit Contract

The system needs a typed contract for `mu`.

Do not let these be interchangeable:

```text
raw panel score
rank percentile
calibrated probability
expected excess return
NGBoost mu
Kelly edge
QP return vector
```

Each should have a named field. QP should consume only an explicitly validated expected-return field.

### 6. Shadow-First Deployment

Before live capital, run shadow mode beside the current production path.

For each trading day, persist:

- feature coverage;
- candidate list;
- raw scores;
- calibrated scores;
- expected returns;
- sigma;
- Kelly target;
- QP target weights;
- final order deltas;
- actual broker holdings;
- rejected orders and reasons.

Require a clean shadow period before promotion.

### 7. Reproducible Experiment Runner

Training scripts should be non-ambiguous command-line tools.

Minimum:

- argparse with real `--help`;
- required output directory;
- run id;
- seed list;
- config path;
- data fingerprint;
- automatic log file;
- no training on import;
- no training when help is requested.

## Priority Fixes

### P0 - Before Any Live Trust

1. Build artifact bundle + promotion manifest.
2. Enforce scorer/calibrator/ngboost fingerprint matching.
3. Make QP the only allocator when enabled; disable post-QP top-up.
4. Fix data freshness semantics after market close.
5. Fix production covariance artifact path loading.
6. Add live preflight hard fail for score saturation and out-of-support calibration.
7. Remove runtime calibration clipping by retraining calibrator correctly.

### P1 - Before Increasing Capital

1. Complete 16-window p0activated A/B table.
2. Complete NGBoost 5-seed test and report t-stat.
3. Add QP feasibility-rate report by regime and window.
4. Add feature-health summary to every sim and live run.
5. Add explicit expected-return unit contract.
6. Add timeout enforcement for parallel ticker jobs.
7. Add Q12 per-bar weight/decision diff tooling.

### P2 - System Hardening

1. Replace loose artifact symlinks with registry entries.
2. Add automatic stale-fundamental warnings and coverage thresholds.
3. Add run-level reproducibility metadata.
4. Add shadow-mode report generation.
5. Make all experiment scripts deterministic and CLI-safe.

## What Claude Code Should Not Do

Do not declare 104 fixed because:

- three exposure-scaling tests pass;
- one NGBoost seed beats baseline;
- a sim batch is running;
- logs look active;
- the model has positive IC.

Those are useful signals, not production proof.

Do not turn on new flags in live trading until the promotion gate proves the full chain works.

Do not let QP, Kelly, top-up, and sell gates all independently mutate the portfolio without a single final allocation owner.

## Final Recommendation

Keep RenQuant 104 in research / paper / shadow mode until the system has:

1. promoted artifact bundles;
2. strict scorer-calibrator pairing;
3. explicit expected-return units;
4. one allocator owner;
5. completed multi-window A/B validation;
6. live preflight checks for calibration and feature health;
7. reproducible experiment metadata.

The model may be good. The system around the model is not yet trustworthy enough.

