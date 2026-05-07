# Post-Stage 3 experiment queue

When Stage 3 (`run_stage3_greedy.py`) finishes — runs ~10 PT 2026-05-03 ish —
the following sequence captures the maximum value from today's setup work
without operator intervention.

## Step 1 — B2 hold-out sim on Stage 3 winner (auto-dispatch ready)

```bash
bash scripts/run_b2_on_stage3_winner.sh
```

Reads `scripts/stage3_progress.json`, finds the highest-mean_ic accepted batch,
runs B2 hold-out sim (train end 2024-12-31, sim 2025-01-02 → 2026-04-30).

Outputs APY / Sharpe / Sortino / Calmar / MaxDD on the proposed
expanded watchlist.

**Decision rule:**
- If APY ≥ baseline (current production fwd_5d) + 1pt AND Sharpe doesn't drop
  more than 0.05 → proceed to Step 2 (golden update proposal)
- Otherwise → keep wl103 production, log Stage 3 lift as IC-only finding

Wallclock: ~30-40 min CPU.

## Step 2 — Golden config update proposal (if Step 1 passes)

Generate a side config `strategy_config.wl_stage3.json` with:
- `watchlist` = Stage 3 final accepted set (from `stage3_final_watchlist.json`)
- All other golden config entries unchanged
- Audit label `_audit_label = wl_stage3_2026_05_03`

```bash
python3 -c "
import json
final = json.load(open('scripts/stage3_final_watchlist.json'))
golden = json.load(open('backtesting/renquant_104/strategy_config.golden.json'))
golden['watchlist'] = final['watchlist']
golden['_audit_label'] = 'wl_stage3_2026_05_03'
# Use side artifact paths — DO NOT clobber prod artifacts
for k in ['panel-ltr', 'ngboost-head', 'panel-rank-calibration']:
    p = f'artifacts/{k}.wl_stage3.json'
    if k == 'panel-ltr':
        golden['panel_ltr']['artifact_path'] = p
        golden['ranking']['panel_scoring']['artifact_path'] = p
    elif k == 'ngboost-head':
        golden['panel_ltr']['ngboost']['artifact_path'] = p
        golden['ranking']['panel_scoring']['ngboost']['artifact_path'] = p
    else:
        golden['ranking']['panel_scoring']['global_calibration']['artifact_path'] = p
json.dump(golden, open('backtesting/renquant_104/strategy_config.wl_stage3.json', 'w'), indent=2)
print('OK')
"
```

Train the production-style model on this config:

```bash
python scripts/train_104.py \
    --strategy-config-name strategy_config.wl_stage3.json \
    --skip-baseline --skip-recalibrate --force
```

Wallclock: ~25-35 min. Verify CPCV mean_ic matches Stage 3 final reference IC.

## Step 3 — §5.2 sanity sequence on the wl_stage3 model

**MANDATORY before promoting** (Track F lesson):

```bash
# Shuffled-label sanity (expect mean_ic ≈ 0)
python3 -c "import json; c=json.load(open('backtesting/renquant_104/strategy_config.wl_stage3.json')); c['_audit_label']='wl_stage3_shuffled'; c['panel_ltr']['label_shuffle_seed']=42; ...; json.dump(c, open('backtesting/renquant_104/strategy_config.wl_stage3_shuffled.json','w'), indent=2)"

python scripts/train_104.py --strategy-config-name strategy_config.wl_stage3_shuffled.json --skip-baseline --skip-recalibrate --force

# Time-shift placebo +60d (expect mean_ic ≈ baseline placebo value, ~+0.029)
python3 -c "... label_shift_days=60 ..."
python scripts/train_104.py --strategy-config-name strategy_config.wl_stage3_placebo.json --skip-baseline --skip-recalibrate --force
```

Both must produce mean_ic ≪ wl_stage3 real mean_ic. If placebo ≈ real, the
Stage 3 lift is regime-persistence over-fit (same failure mode as Track F);
DO NOT promote.

## Step 4 — Promote to golden (if all sanity passes)

Production-touching change. Per CLAUDE.md §5.5 needs operator review.
DO NOT auto-execute.

Operator action:
1. Cherry-pick the wl_stage3 config + all artifacts into golden:
   ```bash
   cp backtesting/renquant_104/strategy_config.wl_stage3.json \
      backtesting/renquant_104/strategy_config.golden.json
   cp backtesting/renquant_104/artifacts/panel-ltr.wl_stage3.json \
      backtesting/renquant_104/artifacts/panel-ltr.json
   # etc
   ```
2. Commit + push to main.
3. Sunday retrain (next 2026-05-04 10:00 PT) auto-uses new wl.

## Open questions / followups

- **Stage 3 ticker order**: ran alphabetically. If batches early in alphabet
  happen to be more compatible with wl103, alphabetical bias might inflate
  early-batch IC. Mitigation: re-run with random-shuffled order if Step 1
  APY doesn't pan out.

- **Greedy vs optimal**: greedy admits whatever doesn't hurt IC by >2bp,
  but doesn't search for optimal subset. After Stage 3, could iterate by
  removing 1 ticker at a time and seeing if removal hurts (leave-one-out).
  Lower priority.

- **Sector balance**: Stage 3 didn't enforce sector quotas. If accepted set
  skews tech-heavy, diversification benefit may be muted. Check sector
  histogram in `stage3_final_watchlist.json` after Step 1.
