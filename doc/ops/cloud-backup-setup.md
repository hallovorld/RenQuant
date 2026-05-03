# Cloud backup setup — GitHub private repo

Backs up critical local-only state to a GitHub private repo on a 4-hour cadence
via launchd. Free, ~46MB total, single-developer setup.

## What gets backed up

| File | Size | Recoverable from elsewhere? |
|---|---|---|
| `data/runs.db` | ~568K | NO — panel training history (oos_mean_ic, train_ic, all 30+ runs this week) |
| `data/runs.alpaca.db` | ~45M | NO — Alpaca live broker state, scoring history |
| `backtesting/renquant_104/live_state.alpaca.json` | 4K | NO — current positions / regime cooldowns / sell streaks |
| `backtesting/renquant_104/live_state.paper.json` | 4K | NO — paper trading state |
| `data/insider_trades/*.parquet` | ~328K | YES but slow (SEC fetch ~24h with rate limits) |
| `scripts/stage3_progress.json` | small | NO — Track D Stage 3 batch admission progress |

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
- **Large file > 100MB**: shouldn't happen (largest is 45MB), but if
  data/runs.alpaca.db ever grows past 100MB, switch to Git LFS.

## Costs

GitHub private repos: **$0** for any size under reasonable limits (we're at
46MB on a 1GB+ allowance). No Git LFS needed (single file < 100MB).
