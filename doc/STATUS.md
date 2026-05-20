# RenQuant — Status

**Last updated:** 2026-05-20

This file used to maintain a parallel status snapshot. Per `feedback_journal_lessons_in_session`, status is now load-bearing in:
- **`CLAUDE.md` § "🗂 Current state"** — active strategy, live mode mandate, NAV invariant, roadmap pointers
- **`doc/roadmap.md` § "📍 Current state"** — production model fingerprint, watchlist, calibrator, cron schedule
- **`doc/dashboard.md`** — auto-generated per-build artifact + KPIs
- **`doc/research/2026-05-19-patchtst-improvement-plan.md`** — current PatchTST capability roadmap (Pillar A/B/C × Tier 1/2/3)
- **`doc/experiments/2026-05-19-hf-trainer-refactor-journal.md`** — lessons from recent HF Trainer refactor

Read those before assuming this file is current. Historical snapshots are in `doc/archives/sessions/`.

---

## What changed since the last STATUS.md (2026-05-09 EOD)

Use `git log --oneline --since=2026-05-09` for authoritative history. Highlights:

- **Calibrator P0 refit** (2026-05-15): isotonic→Platt scaling; `expected_return.y` clipped to `[-0.20, +0.20]` with load-time guard; G12 preflight; pool_IC preserved at +0.094
- **NGBoost head re-confirmed** (2026-05-15): 5-seed Duan §4 val_IC=+0.0351 ± 0.0036, σ-calib=+0.271; promoted to prod 2026-05-17 (md5 `30b0460a`); σ-wire stays OFF per A/B
- **News-sentiment features shipped** (2026-05-18): panel went 169→172 (`sentiment_pos_share`, `mean_sentiment`, `n_articles`); regime-conditional gate live
- **wl200 (142 ticker quality-first) promoted** (2026-05-18): replaced wl103; breadth lift √(142/103) ≈ 1.17×
- **HIFO lot-selection default** (2026-05-17): replaced FIFO; pure accounting change, `feedback_no_tax_driven_logic`-safe
- **Detector 5-day BEAR + vol-cluster CHOPPY** (2026-05-17): catches SVB / DeepSeek / Aug-2024 crises the 20-day rule missed
- **DDV (deep_drawdown_veto) disabled globally** (2026-05-17): per HXZ 2020 "Replicating Anomalies"; META incident motivation
- **min_share_floor for high-price stocks** (2026-05-17): unblocked EQIX-class ($1059/share)
- **Walk-forward gate enforcement** (2026-05-17): removed `RQ_ALLOW_NO_WF=1` setdefault; daily retrain stages only, weekly `weekly_wf_promote.sh` does the promote
- **Anti-churn `min_reentry_days=5`** (2026-05-18, MCD incident): compounds on top of §1091 wash-sale
- **HF PatchTST shadow** (2026-05-19): full second-pipeline-run with HF PatchTST primary, hard-isolated; shadow scorer at `kernel/panel_pipeline/hf_patchtst_scorer.py`
- **HF Trainer refactor** (2026-05-19): `scripts/patchtst_hf.py` swapped hand-rolled train loop to `transformers.Trainer`; multi-task head (rank + Student-t dist); `load_best_model_at_end=True`; per-regime IC callback; cosine LR + warmup; Margin Ranking loss
- **FiLM regime conditioning** (2026-05-19): `--film-regime-cond` flag; FiLMLayer at γ, β = MLP(regime); identity-at-init = strict superset of baseline
- **DLinear baseline shipped** (2026-05-19): §5.12 must-have for transformer overhead validation
- **5-cut × 5-seed eval drivers + arch comparator + σ-calibration verifier shipped** (2026-05-19)

Single source of truth for code: `git log` + the code itself.
