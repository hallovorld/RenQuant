# RenQuant — Roadmap

**Single source of truth for what's next.** Ordered by priority. Last reset **2026-05-08 EOD** (post alpha158+fund promote + Track-B sweep + cron pipeline P0 fix).

---

## End goal

Long-running quant trading workstation that does the boring infra well so the user can iterate on signal research:

- **Live alpha** — production XGB panel-LTR producing trades that net of cost/tax beat SPY by ≥ 5 pts annualized.
- **Walk-forward defensible** — 7-cut OOS with mean IC > 0 and ≥5/7 positive; no single-cut promotions.
- **Self-maintaining** — daily retrain cron keeps every active model fresh; rollback rehearsed for each promotion.
- **Reproducible** — every config + artifact + run lineage tracked; no file-system clutter.

---

## Right now (2026-05-08 EOD)

| | |
|---|---|
| Production model | **alpha158+5fund XGB rank:pairwise** (`panel-ltr.alpha158_fund.json`, 163 features, fingerprint `bf6455c2315bca06`) — promoted commit `ca350c0` |
| Validation | 7-cut WF mean IC +0.067, sanity-adjusted real signal +0.038 (E40); portfolio sim Sharpe 1.06 LO / 1.04 LS (E41) |
| Calibrator | Refit on new model 2026-05-09 — `n_unique_prob_y=85`, `pool_ic=+0.103`, preflight P-CALIBRATOR-HEALTH HARD PASS |
| NGBoost head | DISABLED (old 21-feat head, single-thread NGBoost retrain ran 1h41m without finishing). **Replaced by 3-quantile XGBoost head** (this commit), multi-threaded, ~30s |
| Live cron | daily104 + open104 + preclose104 + intraday104 ENABLED; Sunday retrain ENABLED. **daily104 wired to new alpha158+fund Pipeline 2026-05-08** (commit `3d0efee`, Task/Job per §1b + caching, 18/18 tests). Cached re-run = 0.2s; full retrain = 122s |
| Live broker | Alpaca live account, equity $10,582; today's BUY LMT x1 + ON x10 ACCEPTED via E2E pipeline |
| Portfolio QP | cvxpy + CLARABEL primary; Davis-Norman no-trade band active (today skipped MPWR @ $1576 because Δw too small for 1 share) |

---

## P0 — Track 2 (alpha exploration) by ROI

Today (2026-05-08) cleared the production-readiness backlog (A2 calibrator + A3 cron). Remaining ROI-positive Track 2 items, ordered:

### B2 — Per-sector model (cheap, novel) ⏵ NEXT

Train one XGB rank head per sector (11 sectors, 22 tickers/sector avg). Sector is a STATIC label (not slow-moving feature), so avoids the regime-as-feature artifact that killed E44 (real signal −0.013 after sanity battery). Each sector model sees only ~50k rows (still plenty for XGB d=5).

**Implementation**: extend `wf_panel_args.py` to bucket-per-sector + train independently + aggregate cross-sectional IC across sectors.
**Pass gate**: cross-sector mean IC > production +0.067 by ≥0.005, AND each sector contributes net-positive (no sector dragging mean down).
**Sanity**: §5.2 paired battery on per-sector setup vs global — must not introduce stock-type artifact growth.

### B3 — Analyst consensus revisions (medium, theory-backed)

EPS revisions are classic sell-side alpha (Womack 1996, Stickel 1991). SimFin API has analyst data; not yet pulled. **Implementation**: fetch analyst consensus per ticker per quarter, derive `eps_revision_4w_pct` and `target_price_change_4w_pct` features, append to alpha158+5fund panel as 2 extra features.
**Pass gate**: paired WF Δmean IC > +0.005 + sanity battery (real signal lift, not artifact growth).

### B4 — Long-horizon PEAD revisit (cheap, prior was fwd_5d failure)

E23 closed PEAD at fwd_5d (-1.3 σ artifact). Resume condition was fwd_20d/60d horizon. Production fwd_60d label may capture the 30-60d post-earnings drift Bernard-Thomas (1989) documents. **Implementation**: paired WF on alpha158+5fund + 3 PEAD features (`days_since_earnings`, `surprise_decay`, `surprise_quintile_rank`).
**Pass gate**: same as B3.

### B5 — Cross-sectional dispersion features ⚠️ likely fails E44 sanity

Per-date features (vol-of-vol, dispersion of returns) broadcast to all tickers. **Risk**: same anti-pattern as E44 regime-as-feature (broadcast features grow stock-type artifact more than they add cross-sectional alpha). Run sanity FIRST.

### B6 — Sector-rotation overlay (D-track follow-up)

D2 multi-horizon NO-GO and D1 vol-target neutral. Resume only if NGBoost-σ + Kelly path also fails on portfolio sim.

---

## P1 — Architecture exploration

Skip until P0 (B2/B3/B4) clears. Ordered by literature support:

### C1 — Stacked ensemble (Linear + XGB + meta-ranker)

Linear (158 alpha158 features) + XGB (5 fund + interactions) + meta-ranker learns blend. Reference: Wolpert 1992 stacked generalization, Ribeiro 2020 AdaStack.

### C2 — PatchTST on top-of-rank residuals

Use Transformer ONLY on top-30 residuals (not standalone — at 700k rows we'd violate CLAUDE.md param/sample > 1/100 rule). Reference: Nie et al. 2023 ICLR PatchTST.

### C3 — LightGBM retest on alpha158+fund

E27-era rejection was on different panel. Quick test (~1 min). Resume condition: only after B2/B3/B4 exhausted.

---

## P2 — Infrastructure

### A1 — Quantile-XGBoost head (in progress 2026-05-08 evening)

Replaces NGBoost (single-thread, 1h+ on 516k×163). Method: 3 XGBoost-quantile regressors (q=0.16/0.50/0.84), σ̂ = (q_0.84−q_0.16)/2. Multi-threaded, ~30s. Reference: Koenker & Bassett 1978, Lim et al. 2021 TFT §3. Output artifact at `ngboost-head.alpha158_fund.json` with `kind="quantile_head"` — needs corresponding loader in `ApplyNGBoostTask` (separate Task).

Once landed, unlocks `ranking.panel_scoring.ngboost.enabled=true` → σ-aware QP + Kelly sizing in production.

### A4 — Acceptance gates for new daily retrain output

Currently `daily_retrain_alpha158_fund.sh` retrains and overwrites the production artifact unconditionally. Wire `kernel/model_acceptance.py` so a bad retrain (IC drop > 5pt vs prior, or sanity placebo lift > 1pt) auto-rolls back to the previous artifact. Existing `acceptance` block in golden config has the gates — need to plumb them into the new pipeline.

### A5 — Sunday retrain (`retrain_panel.sh`) — also wire to new pipeline

Currently calls `sunday_panel_sweep.py` which does an XGB hyperparameter sweep on the OLD 21-feat panel. Either (a) update sweep to use 163-feat panel and write to new artifact, or (b) replace with weekly version of `daily_retrain_alpha158_fund` that does extra sanity (full §5.2 battery) and CV.

---

## Closed today (2026-05-08, do not re-open)

| Track | Outcome | Why |
|---|---|---|
| E39 fund_ext (replication) | NO-GO | XGB IC −0.020 vs base, multicollinearity with existing 5 fund features |
| **E44 fund_regime (broadcast GMM probs)** | **NO-GO** | Real signal **−0.013** after sanity (raw +5bp was masked by +15bp shuffled-label artifact growth) |
| **E45 R2K (1640 tickers)** | **NO-GO + audited** | XGB IC drops 75% (+0.067 → +0.018). Audit: top-300 liquid + alpha158-only didn't recover. Structural — Cakici 2023 ML-alpha-larger-on-small-caps doesn't apply at this signal scale |
| **E42 multi-horizon ensemble** | **NO-GO** | Raw IC +0.007 lift but portfolio Sharpe −0.07 (1.06 → 0.99). IC↔Sharpe disconnect |
| **D1 vol-target v2 sweep** (4 configs) | NEUTRAL | Best (v2d 20% vol target) Sharpe 1.02 vs base 1.06; trades 4bp Sharpe for −7pt MaxDD. Not promote-worthy |

---

## P1 — engineering hygiene

4. **Migrate side configs out of file system** (Task #38). DB schema + `migrate_experiment_configs_to_db.py` exist; `holdout_backtest.py --experiment-label` flag works. Remaining: live runner + retrain cron read-from-DB; delete stale `strategy_config.*.json` files once everything reads DB.
5. **Calibrator runtime SOFT-WARN** (CLAUDE.md status §1). Production `panel-rank-calibration.json` has `n_unique_prob_y=7 < 10` floor; refit after panel-LTR pin or boost `min_best_iter` so XGB plateaus higher.
6. **Test suite hygiene**. Currently ~14k passing; trim slow tests, surface ones that exercise sim-level integration (V8 leverage bug shipped because no sim test pinned cvxportfolio backend constraints).

---

## P2 — research

These are deferred until P0 ships. Don't start without explicit user request.

7. **Microstructure / hourly bar (Track C)**. Alpaca hourly cache empty (0/178 tickers). Data fetch + feature build ~3-5 days.
8. **Regime ensemble (T2-3)**. Wait for >150k panel rows or watchlist expansion past 150 tickers.
9. **Walk-forward retraining** (sliding train window). Per-cut retrain on the cut-specific train period; tells us robustness vs a single fixed model. ~10× compute of a static walk-forward.

---

## Closed (per CLAUDE.md status — don't re-open)

Each of these has a shelved design doc in `doc/archives/shelved/` for git-log provenance.

| Track | Verdict |
|---|---|
| Macro overlay (v1–v4) | All variants net negative IC at panel size 103. Revisit at 200+ tickers. |
| Asset embeddings (T2-2) | +0.0001 IC delta = no lift. |
| LightGBM panel | -60% IC vs XGBoost; REJECTED. |
| Boyd rotation (T2-4) | -2.5 APY pts; default OFF. |
| Insider feature (Track A) | -0.0008 contribution at 44% coverage. |
| PEAD enrichment (Track B) | 17-22σ negative on fwd_5d (label too short for drift). |
| Watchlist 183 (Track D) | Sharpe collapse: wl183 +0.55 vs wl103 +0.68. |
| Triple-barrier label (Track F) | +98bp lift was placebo (time-shift +60d also +). |
| Walk-forward XGB (E27) | Mean alpha vs SPY −15.62% ± 10.21% across 3 cuts. |
| QP refactor — adopt cvxportfolio | DONE 2026-05-06: cvxpy + CLARABEL primary, SinglePeriodOpt opt-in backend. |
| Doc reorganization | DONE 2026-05-07: 103 docs → 63; archived/shelved 40 + deleted REORG_PLAN. |

---

## How to use this doc

- Pick the topmost unblocked item.
- Open a small branch, ship the smallest reversible step, commit, test vs golden.
- If APY +2 pts (or CLAUDE.md §2a exception applies) → promote golden in the same commit → mark item done here.
- Otherwise update this doc with what you learned and pick the next item.

Working rhythm: ship don't ponder; one task in flight; commit each meaningful chunk.

---

## Renquant_105 (future, planned)

The 30-min level model. Designed but not started.

- Pre-requisite: renquant_104 stable for ≥ 3 months on live (build a baseline track record first).
- Pre-requisite: minute-bar panel > 1M rows (103 tickers × 2 yrs × ~16 bars/day).
- Independent strategy dir at `backtesting/renquant_105/`; does NOT reuse 104's `strategy_config.json`.

105 design work resumes after 104 is stably live.
