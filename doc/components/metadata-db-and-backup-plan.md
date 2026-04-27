# Model Metadata DB + Cloud Backup — Plan

**Status**: Plan / not yet implemented (2026-04-26).
**Trigger**: User spec "每个模型的metadata应该进数据库，模型artifact应该有云备份".

This doc plans **two complementary features**:

1. **Metadata DB** — every trained artifact's metadata (OOS IC, gates, sim_smoke, lineage) lives in `runs.db`, queryable via SQL, the artifact JSON is no longer the source-of-truth for "what models exist".
2. **Cloud backup** — every promoted artifact is mirrored to S3 / B2 / GitHub LFS so a local disk loss doesn't lose 6 months of training. Includes a restore procedure.

---

## Feature 1: Metadata DB

### Today's state

- `runs.db.training_runs` exists (`backtesting/renquant_104/kernel/persistence.py:183`)
- It captures: `run_id, run_date, strategy, artifact_type, oos_mean_ic, train_ic, n_rows, feature_cols, artifact_path, commit_sha, elapsed_sec, trigger, n_tickers, n_dates, n_features, device, deterministic, training_window_years, notes`
- Written by `record_training_run()` at the end of FullTrainingPipeline

**What's missing** (vs what we need for systematic model selection):
- Phase 1+2 gate verdicts: which gates passed/failed, severities, thresholds
- Phase 2 sim_smoke metrics: APY / Sharpe / Calmar / turnover / max_dd
- Phase 3 composite tournament rank
- Phase 4 challenger linkage: was this model EVER a challenger? what was the verdict?
- Lineage: which `panel-ltr.previous.json` did this model REPLACE?
- Promotion timeline: promoted_at, demoted_at (if rollback fired)
- Hash / checksum: for cloud-backup integrity verification

### Proposed schema (additive to `training_runs`)

```sql
-- Add columns to training_runs (idempotent migration via _COLUMN_MIGRATIONS)
ALTER TABLE training_runs ADD COLUMN sim_apy           REAL;
ALTER TABLE training_runs ADD COLUMN sim_sharpe        REAL;
ALTER TABLE training_runs ADD COLUMN sim_calmar        REAL;
ALTER TABLE training_runs ADD COLUMN sim_max_drawdown  REAL;
ALTER TABLE training_runs ADD COLUMN sim_turnover      REAL;
ALTER TABLE training_runs ADD COLUMN promoted_at       TIMESTAMP;   -- NULL if rejected
ALTER TABLE training_runs ADD COLUMN demoted_at        TIMESTAMP;   -- when superseded
ALTER TABLE training_runs ADD COLUMN replaced_run_id   TEXT;        -- FK → training_runs.run_id
ALTER TABLE training_runs ADD COLUMN sha256_hex        TEXT;        -- artifact bytes
ALTER TABLE training_runs ADD COLUMN size_bytes        INTEGER;
ALTER TABLE training_runs ADD COLUMN cloud_backup_url  TEXT;        -- s3://... once uploaded

-- New table: per-gate verdicts (1 row per gate per training_run)
CREATE TABLE IF NOT EXISTS training_run_gates (
    run_id      TEXT NOT NULL,
    gate_name   TEXT NOT NULL,            -- 'G1_schema', 'G4_oos_ic_vs_prior', ...
    severity    TEXT NOT NULL,            -- 'hard' | 'soft'
    passed      INTEGER NOT NULL,         -- 0|1
    metric      REAL,
    threshold   REAL,
    detail      TEXT,
    PRIMARY KEY (run_id, gate_name),
    FOREIGN KEY (run_id) REFERENCES training_runs(run_id)
);
CREATE INDEX IF NOT EXISTS idx_trg_run ON training_run_gates(run_id);

-- New table: tournament rankings (snapshot when select_best_model.py runs)
CREATE TABLE IF NOT EXISTS tournament_rankings (
    snapshot_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date   TIMESTAMP NOT NULL,
    candidate_name  TEXT NOT NULL,
    composite_score REAL,
    rank            INTEGER,
    weights_json    TEXT,                 -- {"ic":0.5, "sharpe":0.3, ...}
    promoted        INTEGER DEFAULT 0     -- 1 if --promote chose this snapshot's winner
);
```

### What gets written when

| Event | Update |
|---|---|
| Training pipeline succeeds | INSERT training_runs row (`sha256`, `size_bytes`, `oos_*` fields populated) |
| `ModelAcceptanceGate.evaluate` runs | INSERT 11 rows into `training_run_gates` |
| Promote (gate passed) | UPDATE `training_runs.promoted_at`; UPDATE prior model's `demoted_at`; SET `replaced_run_id` |
| `select_best_model.py` runs | INSERT N rows into `tournament_rankings` (one per candidate) |
| `--promote winner` chosen | UPDATE the chosen row's `promoted=1`; same UPDATE chain as gate-promote |
| Cloud backup completes | UPDATE `training_runs.cloud_backup_url` |

### Operator queries this enables

```sql
-- "What's currently in production, what was its IC, when did it ship?"
SELECT run_id, oos_mean_ic, sim_sharpe, promoted_at FROM training_runs
WHERE demoted_at IS NULL AND promoted_at IS NOT NULL ORDER BY promoted_at DESC LIMIT 1;

-- "Show me the last 10 retrain attempts and which gates failed"
SELECT t.run_id, t.run_date, t.oos_mean_ic, g.gate_name, g.metric, g.threshold
FROM training_runs t
LEFT JOIN training_run_gates g ON t.run_id=g.run_id AND g.passed=0
ORDER BY t.run_date DESC LIMIT 10;

-- "Average OOS IC by month for the production lineage"
SELECT strftime('%Y-%m', promoted_at) AS month, AVG(oos_mean_ic), COUNT(*)
FROM training_runs WHERE promoted_at IS NOT NULL GROUP BY 1 ORDER BY 1;
```

### Implementation phases

1. **Phase A (1.5h)**: schema migration + write-path in `record_training_run()` and `model_acceptance.promote()`. Backfill is empty (only NEW retrains populate).
2. **Phase B (1h)**: backfill from existing `panel-ltr*.json` artifacts on disk → INSERT rows for the 9 historical artifacts so queries work today, not just on next retrain.
3. **Phase C (30min)**: `scripts/model_history.py` — pretty-print queries above. Replaces "manual SQL" UX.

---

## Feature 2: Cloud Backup

### Today's state

- All artifacts live on local disk under `backtesting/renquant_104/artifacts/`
- Total: 31 files, ~23 MB. Largest: 519k (lgbm.bak.json)
- Disk loss = retrain everything from scratch (multi-hour, requires cached panel data which is also local)
- `data/runs.db` (~10 MB) holds all the training history — also local

### Risk surface

| Risk | Today's exposure | Backup mitigates? |
|---|---|---|
| MacBook SSD failure | Total loss | ✅ |
| `rm -rf` accident | Total loss | ✅ |
| Local git history loss (hardware) | Recoverable from GitHub | ⚠️  partial — artifacts gitignored |
| GitHub repo loss / corruption | Local copy intact | ❌ orthogonal — push more often |
| Machine theft | Total loss | ✅ |

### Proposed: tiered backup

**Tier 1 — Hot (every promote): `s3://renquant-models/` or `b2://`**
- Triggered by `kernel.model_acceptance.promote()` after successful gate pass
- Object key: `{strategy}/{run_id}.json` (immutable; new run_id per training)
- Latest pointer: `{strategy}/CURRENT.json` overwritten atomically
- Lifecycle policy: keep all (artifacts are tiny — 23 MB total today, ~50 MB/year growth)
- Cost: B2 = ~$0.005/GB/mo → < $0.01/year. Negligible

**Tier 2 — Warm (daily): runs.db backup**
- `daily_104.sh` post-train: `sqlite3 data/runs.db ".backup data/runs.daily.db"` then upload
- Object key: `{strategy}/db/runs-{date}.db`
- Lifecycle: keep 90 days, then 1/month for 12 months, then quarterly forever (≤ 1 GB/decade)

**Tier 3 — Cold (weekly): full artifacts/ + data/ tar**
- Sunday cron: `tar czf renquant-{date}.tar.gz backtesting/renquant_104/artifacts data/runs.db`
- Object key: `{strategy}/snapshots/{date}.tar.gz`
- Lifecycle: keep 4 weekly + 12 monthly + 5 yearly

### Provider choice

| Provider | Pros | Cons | Recommended? |
|---|---|---|---|
| AWS S3 | Standard, mature, fast | Most expensive, AWS account hassle | If you already use AWS |
| Backblaze B2 | Cheapest ($0.005/GB), S3-compatible API | Slightly slower egress | **✅ Default** for solo use |
| Cloudflare R2 | Free egress, S3-compatible | 10 GB free tier covers years | ✅ Alternative |
| GitHub LFS | Native, free up to 1 GB | Slow for big files; gitops awkward | ❌ |
| iCloud / Dropbox | Zero setup | Folder-based, no API, no integrity checks | ❌ |

**Recommendation**: Backblaze B2. ~$0.50/year for everything we'll ever back up. S3-compatible API means standard `boto3` works.

### Restore procedure (drill required quarterly)

```bash
# Tier 1 — restore latest production
b2 download-file-by-name renquant-models renquant_104/CURRENT.json \
  backtesting/renquant_104/artifacts/panel-ltr.json

# Tier 2 — restore yesterday's runs.db
b2 download-file-by-name renquant-models \
  renquant_104/db/runs-$(date -v-1d +%Y-%m-%d).db data/runs.db

# Tier 3 — restore everything from latest weekly snapshot
b2 download-file-by-name renquant-models \
  renquant_104/snapshots/$(b2 ls --json renquant-models renquant_104/snapshots/ \
    | jq -r 'sort_by(.uploadTimestamp) | reverse | .[0].fileName') \
  /tmp/renquant-restore.tar.gz
tar xzf /tmp/renquant-restore.tar.gz -C /
```

### Implementation phases

1. **Phase A (1h)**: `kernel/cloud_backup.py` — `upload_artifact(path, run_id)` using `boto3` with B2 endpoint config. ENV: `RENQUANT_B2_ACCESS_KEY` / `RENQUANT_B2_SECRET_KEY` / `RENQUANT_B2_BUCKET`.
2. **Phase B (30min)**: hook into `promote()` via callback (idempotent — re-uploads same hash skip).
3. **Phase C (1h)**: `scripts/backup_db.sh` + `scripts/backup_weekly.sh` for tiers 2-3, add to launchd plist.
4. **Phase D (15min)**: `scripts/restore_from_backup.sh` for the tested restore procedure.
5. **Phase E (15min)**: quarterly drill — operator runs restore on a clean machine, verifies fidelity.

---

## Combined timeline

If shipping both features:

| Day | Work |
|---|---|
| Day 1 (3h) | Metadata DB Phase A+B+C + tests |
| Day 2 (2h) | Cloud backup Phase A (B2 client + tests with mocked uploads) |
| Day 3 (1.5h) | Cloud backup Phase B (promote() integration) + Phase C (cron) |
| Day 4 (1h) | Phase D (restore script) + Phase E (run drill) |

**Total**: ~7.5 hours of focused work. Can be parallelized: metadata DB and cloud backup are orthogonal.

---

## Decisions still needed (from operator)

1. **Provider**: B2 (recommended) or another?
2. **Bucket name**: `renquant-models` or other?
3. **Retention**: agree to "keep tier 1 forever" (cost ≈ $1/year over a decade)?
4. **What triggers cloud backup**: every successful train (even rejected by gates), or only promoted? Recommend: ONLY promoted.
5. **Encryption**: B2 has free server-side encryption; client-side encryption requires extra key management. Recommend: server-side only (simple, your account is the auth boundary anyway).
6. **GitHub LFS as fallback?**: when no B2 connectivity, allow `git lfs` push of artifacts? Simpler but adds complexity. Recommend: skip.

---

## What this doc does NOT prescribe

- A specific UI for browsing the metadata DB. SQL is sufficient for solo use; a Datasette or Grafana panel can be Phase D.
- Multi-region replication of B2 (overkill for a personal trading workstation).
- Versioned schemas for the artifact JSON itself (the artifact format is owned by the `kind` field; schema migration is per-backend, not centralized).
