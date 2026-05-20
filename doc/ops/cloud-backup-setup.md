# Cloud backup setup — GitHub private repo

Backs up critical local-only state to a GitHub private repo on a 4-hour cadence
via launchd. Free, single-developer setup. Sizes below are typical 2026-05-20;
re-verify with `du -sh <path>` before declaring storage exhausted.

## What gets backed up

| File | Typical size | Recoverable from elsewhere? |
|---|---|---|
| `data/runs.db` | ~600K-1M | NO — panel training history (oos_mean_ic, train_ic, all training_runs rows) |
| `data/runs.alpaca.db` | ~50-100M (grows ~50K/day with wl200 142-ticker scoring) | NO — Alpaca live broker state, scoring history |
| `backtesting/renquant_104/live_state.alpaca.json` | 4-8K | NO — current positions / regime cooldowns / sell streaks |
| `backtesting/renquant_104/live_state.alpaca_shadow.json` | 4-8K | NO — shadow positions for HF PatchTST shadow path (2026-05-19) |
| `backtesting/renquant_104/live_state.paper.json` | 4K | NO — paper trading state |
| `data/insider_trades/*.parquet` | ~500K (wl200 142 tickers) | YES but slow (SEC fetch ~24h with rate limits) |
| `data/news_sentiment/*.parquet` | growing daily (2026-05-18 shipped) | YES via daily refresh cron but expensive |
| `data/iv_snapshots/*.parquet` | growing daily (accumulation phase) | NO — historical IV snapshots not refetchable from Alpaca Free Options |
| `scripts/stage3_progress.json` | small | NO — Track D Stage 3 batch admission progress |

**NOT backed up** (intentional):
- Training artifacts (`artifacts/patchtst_*`, `artifacts/dlinear_*`, `artifacts/hf_*` — multi-GB; regeneratable via `eval_*_5cut_5seed.py` drivers)
- Production model artifacts (`backtesting/renquant_104/artifacts/prod/*.json` — already in main RenQuant repo)
- Source code, configs, docs (in main RenQuant repo)
- `.tmp_dagster_home_*/` temp dirs (Dagster runtime)
- `mlruns/` (MLflow shadow-scoring logs — re-buildable from training history)

**NOT backed up here** (already in main RenQuant repo):
- Production artifacts (`backtesting/renquant_104/artifacts/*.json`)
- Source code, configs, docs

## One-time setup

### Step 1 — Create private repo on GitHub

Visit https://github.com/new

- Name: `renquant-state-backup` (recommended)
- Visibility: **Private**
- Do NOT initialize with README — keep empty

Copy the repo URL. Either form works:
- SSH: `git@github.com:USER/renquant-state-backup.git` (recommended if SSH keys set)
- HTTPS: `https://github.com/USER/renquant-state-backup.git` (requires PAT for push)

### Step 2 — First backup (clones repo + pushes initial snapshot)

```bash
BACKUP_REMOTE=git@github.com:USER/renquant-state-backup.git \
    bash scripts/backup_to_github.sh
```

Verify on GitHub web that the repo now contains `data/runs.db` etc.

### Step 3 — Wire to launchd for periodic backups

```bash
cp scripts/com.renquant.backup.plist ~/Library/LaunchAgents/
mkdir -p logs/backup
launchctl load ~/Library/LaunchAgents/com.renquant.backup.plist
```

Verify loaded:
```bash
launchctl list | grep com.renquant.backup
```

Schedule: 07:00, 11:00, 15:00, 19:00, 23:00 PT (every 4 hours during waking hours).
Skip 02:00-06:00 PT (no market activity).

### Step 4 — Verify next scheduled backup

```bash
tail -f logs/backup/launchd_stdout.log
```

After the next scheduled hour you should see "Backup pushed at YYYY-MM-DDTHH:MM:SSZ"
or "No changes since last backup; skipping commit." (the latter means nothing
changed, which is fine).

## Operations

### Manual backup (any time)

```bash
bash scripts/backup_to_github.sh
```

### Restore on a new machine

```bash
git clone git@github.com:USER/renquant-state-backup.git ~/.renquant-state-backup

# Restore DBs
cp ~/.renquant-state-backup/data/runs.db /path/to/RenQuant/data/runs.db
cp ~/.renquant-state-backup/data/runs.alpaca.db /path/to/RenQuant/data/runs.alpaca.db
cp ~/.renquant-state-backup/live_state.*.json /path/to/RenQuant/backtesting/renquant_104/

# Restore insider cache (optional but saves SEC re-fetch)
rsync -aq ~/.renquant-state-backup/data/insider_trades/ /path/to/RenQuant/data/insider_trades/
```

### Disable temporarily

```bash
launchctl unload ~/Library/LaunchAgents/com.renquant.backup.plist
```

Re-enable with `launchctl load`.

## Failure handling

The script sends ntfy push to topic `renquant` on failure (same channel as
production retrain alerts). Common failure modes:

- **First-run remote not set**: script aborts with clear instruction.
- **GitHub auth expired**: SSH key revoked or PAT expired → push fails.
  Fix by re-adding SSH key / refreshing PAT.
- **Large file > 100MB**: `data/runs.alpaca.db` grows ~50K/day at wl200 scale
  (142 tickers); will cross 100MB sometime in 2026. Pre-emptively configure
  Git LFS now or plan a 100MB cutover.

## Costs

GitHub private repos: **$0** for the current total (~50-100MB) under their
1GB+ allowance. No Git LFS yet. Plan LFS migration when `runs.alpaca.db`
approaches 100MB single-file limit.
