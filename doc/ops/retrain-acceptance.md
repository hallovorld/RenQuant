# Retrain & Acceptance Gate

**Status:** authoritative as of 2026-05-15. If code disagrees, code wins; fix this doc.

## TL;DR

Every retrain (cadence OR anomaly-triggered) runs the full `FullTrainingPipeline`
followed by **11 hard acceptance gates**. If any hard gate fails, the new model
is rejected and the prior model is preserved — **zero downtime**.

```
SNAPSHOT prior → RUN training → STAGE new → RESTORE prior → EVALUATE gates
                                                              │
                                                              ├─ ALL PASS → promote(new)
                                                              └─ ANY FAIL → reject(new) + ntfy + exit 2
```

## Triggers

Three paths invoke `scripts/train_104.py --force`:

| Trigger | Script | Cadence | --trigger tag |
|---|---|---|---|
| Daily cadence | `scripts/launchd/com.renquant.retrain-panel104.plist` → `scripts/retrain_panel.sh` | Weekday nights | `cadence` |
| SPY anomaly | `scripts/conditional_retrain_104.sh` (cron) | When SPY \|Δ\| > 2% | `anomaly_spy_2pct` |
| VIX anomaly | same as above | When VIX \|Δ\| > 5% | `anomaly_vix_5pct` |

All three end up in `train_104.py main()` with `args.force=True` and `args.trigger=<tag>`.
The trigger tag is logged but does NOT alter training flow — both cadence and
anomaly retrains run the SAME pipeline and SAME gates.

## What gets retrained

`FullTrainingPipeline` (entry: `scripts/train_104.py:148`) sequences three jobs:

```
FullTrainingPipeline
├─ BaselineTournamentJob       — per-ticker baseline XGBoost (142 wl200 tickers, parallel)
├─ PanelTrainingJob            — alpha158 panel-LTR (172-feature XGB rank:pairwise)
│   └─ PanelTrainingPipeline
│       ├─ PanelDataJob
│       ├─ PanelFeatureJob
│       ├─ PanelAssemblyJob
│       ├─ PanelModelJob       — writes artifacts/prod/panel-ltr.alpha158_fund.json (172 features)
│       ├─ PanelNGBoostJob     — writes artifacts/prod/ngboost-head.alpha158_fund.json (promoted 2026-05-17, val_IC +0.0352, σ-wire dormant)
│       └─ RefreshPanelCalibratorJob — Platt scaling (switched from isotonic 2026-05-18); pool_IC +0.094
└─ RecalibrationJob            — Platt-scaling calibrator pool_ic + H2a/H2b hard gates (2026-05-17 commit `637594e`)
```

Walltime: ~60-70 min on M4 Pro 14c (was ~90-120 min on M2 Pro 10c). BaselineTournamentJob remains the bottleneck.

**Daily vs Weekly distinction (2026-05-17 walk-forward gate enforcement, commit `96af42b`)**:
- Daily ops (Mon-Fri 14:00 PT via `daily_104.sh`): smoke/export/live/shadow only; no model promote path.
- Weekly walk-forward (Sat 04:00 PT via `weekly_wf_promote.sh`): retrains into unique scorer/calibrator staging paths, runs the strict 3-cut WF gate + §5.2 sanity battery, then swaps active scorer+calibrator only when `wf_gate_metadata.passed=True`.
- Removed `RQ_ALLOW_NO_WF=1` setdefault from `train_104.py`. Emergency shell-env override still works.

### Not retrained by this path (separate crons)

| Model | Cron | Cadence |
|---|---|---|
| Meta-label classifier | `monthly_meta_label_retrain.sh` | Monthly |
| HMM regime detector | `scripts/train_spy_hmm.py` | No cron yet (manual) |
| SPY GMM regime detector | `train_spy_gmm.py` | Monthly |
| Side-strategy linear scorer | `retrain_alpha158_linear.sh` | Daily |
| Watchlist screen | `screen-watchlist.plist` | Weekly |

## Acceptance gate flow (`train_104.py:107-209`)

### Step 1 — Snapshot prior model (line 121-137)

```python
panel_cfg = config.get("panel_ltr", {})
artifact_rel = panel_cfg.get("artifact_path", "artifacts/panel-ltr.json")
active_path = strategy_dir / artifact_rel          # e.g. backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.json
pre_train_snapshot = active_path.with_suffix(".pre-train.json")
shutil.copy2(str(active_path), str(pre_train_snapshot))
```

**Why:** the pipeline overwrites the active path in step 2. We need the prior
artifact preserved at a distinct path so gate evaluation has two artifacts
to compare (NEW vs PRIOR).

### Step 2 — Run training (line 139-148)

```python
ctx = FullTrainingContext(config=config, strategy=args.strategy, ...)
FullTrainingPipeline().run(ctx)
```

After this, `active_path` contains the NEW (just-trained) model.
`pre_train_snapshot` contains the PRIOR.

### Step 3 — Stage new + restore prior (line 154-165)

```python
staging_path = active_path.with_suffix(".staging.json")
shutil.move(str(active_path), str(staging_path))       # new → staging
shutil.copy2(str(pre_train_snapshot), str(active_path))  # prior → active (restored)
```

**This is the zero-downtime trick.** After step 3:
- `active_path` = PRIOR model (live runner reads this; sees no change)
- `staging_path` = NEW model (waiting for gate evaluation)
- `pre_train_snapshot` = PRIOR (backup, kept for archive)

If the next step crashes for any reason, live trading continues with the prior model.

### Step 4 — Evaluate gates (line 167-169)

```python
verdict = ModelAcceptanceGate(config=acceptance_cfg).evaluate(staging_path, active_path)
```

The gate reads BOTH artifacts and compares. Hard gates and soft gates run.

### Step 5 — Promote or reject (line 173-208)

```python
if verdict.all_hard_passed:
    promote(staging_path, active_path)    # new replaces prior at active path
else:
    reject(staging_path, archive_dir, verdict)  # new → _acceptance_log/<timestamp>/
    # ntfy alert with failure summary
    sys.exit(2)                            # operator script sees non-zero
```

## The 11 hard gates (kernel/model_acceptance.py)

| Gate | What it checks | Why it matters |
|---|---|---|
| **G1** | `feature_cols` length matches active OR diff is in expected-add list | A transformer trained with extra macros would write an artifact the XGBoost consumer can't read. |
| **G2** | Calibrator probability head has ≥5 unique y values | Catches "calibrator collapsed to single output" (the BUG #4 failure mode). |
| **G3** | Pool IC > 0 (didn't flip sign) | Negative pool IC means the calibrator inverted the signal — model would short the winners. |
| **G4** | new oos_mean_ic ≥ prior × (1 − `max_degradation`), default 5% | Don't promote a regression. Phase 1 tightened from 30% → 5%. |
| **G5** | NEW score output range covers ≥80% of PRIOR's range | Catches "everything maps to base_rate" (flat-output failure). |
| **G6** | Smoke-test sample of inference outputs all finite/non-NaN | NaN in production scores = positions sized to NaN dollars = crash. |
| **G7** | OOS IC ≥ absolute noise floor (Phase 1 hardened soft→hard) | A model with +0.005 IC is technically positive but useless. |
| **G8-G11** | Other artifact-integrity checks (file structure, schema, training metadata) | Defensive sanity checks. |

Soft gates (not blocking but logged): regime-stratified IC, breadth stability, etc.

## Walk-forward gate (G8 or runs separately?)

The 75-minute walk-forward sim gate is the heaviest check. It's NOT run on
every cadence retrain — daily cron sets `RQ_ALLOW_NO_WF=1` to bypass it
(`train_104.py:184`, with a loud WARNING log so operator can audit).

Walk-forward gate has its own weekly cron: `com.renquant.weekly-wf-promote.plist`.
Run order:
1. Mon-Thu nights: daily retrain, gates G1-G11, WF bypass
2. Sunday: full walk-forward sim across 27-month OOS, comparison vs prior
3. If WF says reject, archive the most recent week's model + rollback (manual)

## Ntfy alerts surface acceptance state

| Event | Title | Priority |
|---|---|---|
| Retrain start | `RenQuant 104 retrain fired` | default |
| Retrain succeeded + gates passed | `RenQuant 104 retrain OK` | default |
| Retrain failed before gates (argparse / pipeline crash) | `RenQuant 104 retrain ERROR` | high |
| Retrain succeeded but gates REJECTED | `RenQuant 104 RETRAIN REJECTED` + summary of failed gates + archive path | high |

## Override paths (use sparingly)

```bash
# Bypass acceptance entirely (DANGEROUS — only for known-broken-but-recoverable cases)
python scripts/train_104.py --strategy renquant_104 --force --skip-acceptance

# Disable globally via config
# strategy_config.json: {"acceptance": {"enabled": false}}

# Tune per-gate thresholds (e.g. relax G4 max_degradation from 5% to 10%)
# strategy_config.json: {"acceptance": {"max_degradation": 0.10}}
```

## Manual rollback if a promoted model goes bad live

```bash
# Restore the pre-train snapshot from before the bad retrain
cd backtesting/renquant_104/artifacts
cp panel-ltr.alpha158_fund.pre-train.json panel-ltr.alpha158_fund.json
# (Live runner will pick up the change on next bar)

# Audit which retrain caused the regression
ls -t _acceptance_log/
# Each subdirectory is a rejected retrain attempt with verdict.json + artifact snapshot
```

## Files referenced

| Path | Purpose |
|---|---|
| `scripts/train_104.py` | Entry point — argparse, snapshotting, gate driver |
| `scripts/conditional_retrain_104.sh` | Anomaly-triggered retrain wrapper |
| `scripts/check_retrain_triggers.py` | SPY/VIX daily change check |
| `kernel/model_acceptance.py` | `ModelAcceptanceGate`, gate definitions G1-G11 |
| `backtesting/renquant_104/artifacts/_acceptance_log/` | Rejected-retrain archive |
| `backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.pre-train.json` | Prior-model snapshot (rewritten on every retrain) |
| `backtesting/renquant_104/artifacts/panel-ltr.alpha158_fund.staging.json` | New-model staging path (transient, lives only during gate eval) |

## Related ops docs

- [`schedule.md`](schedule.md) — daily/weekly/monthly cron cadence overview
- [`golden-config.md`](golden-config.md) — what production currently uses

## Incident log

- **2026-05-15 13:10** — VIX +5.68% triggered conditional retrain but
  `train_104.py` argparse rejected `--trigger` arg → retrain failed
  before reaching the pipeline. Fixed in commit `ff64f85` (added
  `--trigger` to argparse, logging-only). Next anomaly retrain
  works correctly.
