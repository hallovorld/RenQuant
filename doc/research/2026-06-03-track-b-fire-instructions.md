# Track B — Fire Instructions

**Date**: 2026-06-03
**Status**: Firing surface — user-fire when ready. NO compute consumed
by this memo.
**Parent**: [`2026-06-02-track-b-feature-audit.md`](2026-06-02-track-b-feature-audit.md)
**Plan**: [`2026-06-02-bull-calm-signal-recovery-plan.md`](2026-06-02-bull-calm-signal-recovery-plan.md)

## What is being fired

A walk-forward retrain of the panel-LTR model with the 4 BULL_CALM
recovery features:

| Feature | Source | Expected sign in BULL_CALM |
|---|---|---|
| `mom_carry_12_1` | Kelly-Gu-Xiu RFS 2020 Table 9 (MOM12_m) | positive |
| `beta_dm` | Frazzini-Pedersen JFE 2014 (BAB) | negative |
| `rvar_total` | Ang-Hodrick-Xing-Zhang JF 2006 | negative |
| `idio_vol_market` | Ang et al. 2006 (2-factor variant) | negative |

Code already lives in:
- `renquant-base-data#16` (feature builders, merged)
- `renquant-model#29` (panel-data wiring, merged)
- `RenQuant#120` (`--include-features` flag, merged)
- Audit memo: `doc/research/2026-06-02-track-b-feature-audit.md` (merged)

## Pre-fire gate (per §7.2 + audit memo R2)

Per the §7.2.1 R2 rule: **no IC / Sharpe number from this 176-feature
variant may be quoted in any commit, doc, or status report without a
companion placebo verdict block**.

Required artifacts BEFORE quoting any IC:
1. `aa_mean` from ≥ 3 seeds (`tests/seeds/{42, 43, 44}` is the canonical set)
2. `shuffle_ic` — same recipe, per-date label-shuffled (must be within ±2σ of 0)
3. `timeshift_ic` at shift = 120 days (2× the fwd_60d horizon — audit
   memo's contract; default in `wf_sanity_paired.py` is +60 which is 1×
   — set explicitly)
4. `real_signal = aa_mean − shuffle_ic` (per E40-corrected metric, not
   `aa_mean − 0`)

These ride alongside the per-regime IC slice (§1.3 PRIME DIRECTIVE — by
regime first, pooled second).

## Firing path A — Quick triad on existing panel (~30 min)

Use this to confirm Track B columns persist through the panel build
and the triad metrics are non-zero before spending hours on a full WF
retrain. **No retrain happens — uses the latest panel parquet.**

```bash
# 0. Sync (§3.2).
git fetch origin && git checkout main && git pull --ff-only origin main

# 1. Confirm panel parquet contains the 4 Track B columns.
.venv/bin/python -c "
import pandas as pd
p = pd.read_parquet('data/alpha158_291_fund_regime_dataset.parquet')
for f in ['mom_carry_12_1', 'beta_dm', 'rvar_total', 'idio_vol_market']:
    assert f in p.columns, f'missing {f}'
print('OK — all 4 Track B columns present')
print('rows=', len(p), 'features=', len(p.columns))
"

# 2. Triad on the existing panel.
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \\
.venv/bin/python scripts/wf_sanity_paired.py 2>&1 | tee \\
    logs/track_b_triad_$(date +%Y%m%d-%H%M%S).log
```

The script writes `data/sanity_paired_baseline_vs_regime.json`. Inspect:

```bash
.venv/bin/python -c "
import json
v = json.load(open('data/sanity_paired_baseline_vs_regime.json'))
b, c = v['baseline'], v['regime']
print(f\"baseline real_signal = {b['real_signal']:+.4f}\")
print(f\"candidate real_signal = {c['real_signal']:+.4f}\")
print(f\"delta = {c['real_signal']-b['real_signal']:+.4f}\")
print(f\"shuffle gate (must be within ±2σ of 0):\")
print(f\"  baseline shuffle_ic = {b['shuffle_ic']:+.4f}\")
print(f\"  candidate shuffle_ic = {c['shuffle_ic']:+.4f}\")
"
```

**Caveat**: `wf_sanity_paired.py` defaults to `shift_days=60` (1× horizon).
Audit memo's R2 contract requires 2× (120). Patch in the run, or treat
this as a smoke check only.

## Firing path B — Full WF retrain (~3-5 hours)

Use this once path A confirms the panel + signal are real. Produces
per-cutoff `panel-ltr.json` artifacts under
`backtesting/renquant_104/artifacts/walkforward_addendum_*/`.

```bash
# 0. Sync (§3.2).
git fetch origin && git checkout main && git pull --ff-only origin main

# 1. Full WF retrain with Track B features. train_walkforward_panel.py
#    requires --start-date / --end-date; cadence defaults to 21 days
#    per scripts/train_walkforward_panel.py:308. The Track B variant
#    lands per-cutoff panel-ltr.json under
#    backtesting/renquant_104/artifacts/walkforward_track_b/<cut>/.
OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 \\
nohup .venv/bin/python scripts/train_walkforward_panel.py \\
    --start-date 2024-01-01 --end-date 2026-03-26 \\
    --include-features mom_carry_12_1,beta_dm,rvar_total,idio_vol_market \\
    --artifact-root walkforward_track_b \\
    --manifest-output artifacts/walkforward_manifest_track_b.json \\
    > logs/track_b_wf_$(date +%Y%m%d-%H%M%S).log 2>&1 &
echo $! > /tmp/track_b_wf.pid

# 2. Tail progress.
tail -f logs/track_b_wf_*.log

# 3. Assemble the verdict JSON. There is NO existing WF-manifest
#    analyzer that emits the schema below — scripts/analyze_panels_rigorous.py
#    consumes sim equity panels (not WF model manifests) and writes a
#    Markdown report, so it cannot produce artifacts/track_b_verdict.json.
#
#    Until a dedicated WF-manifest analyzer lands (tracked separately as
#    "Track B verdict-assembler" follow-up), assemble the verdict JSON
#    by hand from:
#      - artifacts/walkforward_manifest_track_b.json (per-cutoff IC)
#      - the path-A triad output (placebo_block — caveat: 60d shift,
#        not the 120d the R2 contract requires)
#      - a separate per-regime IC slice from
#        scripts/analyze_regime_stratified.py (verify input args
#        before firing).
#
#    DO NOT quote any IC / Sharpe number publicly until the
#    placebo_block is filled with the R2-compliant 120d-shift run.
```

## Sanity verdict JSON contract

`artifacts/track_b_verdict.json` must populate (per §7.2.1 R2):

```json
{
  "as_of_date": "YYYY-MM-DD",
  "feature_recipe": "alpha158+5fund+regime+track_b_4",
  "n_features": 176,
  "per_regime": {
    "BULL_CALM":     {"mean_ic": <float>, "n_bars": <int>, "undersampled": <bool>},
    "BULL_VOLATILE": {"mean_ic": <float>, "n_bars": <int>, "undersampled": <bool>},
    "BEAR":          {"mean_ic": <float>, "n_bars": <int>, "undersampled": <bool>},
    "CHOPPY":        {"mean_ic": <float>, "n_bars": <int>, "undersampled": <bool>}
  },
  "placebo_block": {
    "shuffle_ic":   <float>,
    "shuffle_gate_passed": <bool>,
    "timeshift_ic": <float>,
    "timeshift_shift_days": 120,
    "timeshift_gate_passed": <bool>,
    "aa_seeds":     [<float>, <float>, <float>],
    "aa_std":       <float>
  },
  "verdict": {
    "promotion_tier": "1_reject" | "2_screen" | "3_live_promotable",
    "promotion_blocker": <string | null>
  }
}
```

The `placebo_block` is the §7.2.1 R2 compliance proof. The
`per_regime` block is the §1.3 PRIME DIRECTIVE compliance proof.

## Pre-fire checklist

- [ ] `data/alpha158_291_fund_regime_dataset.parquet` exists and has
      all 4 Track B columns (path A step 1).
- [ ] No other heavy compute job is running (`top -l 1 | grep CPU` shows
      idle).
- [ ] `logs/` has space (path B logs grow to ~50 MB).
- [ ] `.venv` activated; `OMP_NUM_THREADS=14` set (CLAUDE.md §6.5).

## Promotion gate (per §7.4)

Tier 3 (LIVE-PROMOTABLE) requires Tier 2 + (DSR > 0.5 OR PBO < 0.5 OR
n ≥ 30 with t > 3.0). Track B targets ≥ +0.020 mean BULL_CALM IC
lift per the recovery plan; the §7.4 multi-comparison guard then
decides whether to promote, screen further, or reject.

## What this PR does NOT do

- It does NOT fire any retrain. The user fires per the recovery plan's
  governance.
- It does NOT add the audit memo (already merged in PR #120's commit).
- It does NOT modify any code paths — only the runner script
  `scripts/run_track_b_battery.sh` (separate convenience wrapper) and
  this fire-instruction memo.
