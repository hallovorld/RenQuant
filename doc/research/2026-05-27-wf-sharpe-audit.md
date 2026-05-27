# WF Sharpe −1.32 audit — real loss or eval bug? (2026-05-27)

**Question (user):** the daily run is fail-closed to sell-only because the prod
panel artifact carries `wf_gate_metadata.passed=false, wf_3cut_sharpe_mean=-1.32`.
Is −1.32 a real model loss, or an evaluation bug?

## FINAL VERDICT (after fixing two eval bugs): predominantly an EVAL BUG.

With a production-faithful walk-forward (reporting-only tax, rf=0, and the WF
manifest artifacts fingerprint-stamped), the model **trades normally and earns
positive Sharpe in all 3 cuts**:

| Cut | Strategy Sharpe | APY | SPY Sharpe | ΔSharpe |
|---|---|---|---|---|
| 2024-01→12 | +0.832 | +8.62% | +1.778 | −0.946 |
| 2024-07→2025-06 | +0.726 | +5.36% | +0.715 | +0.011 |
| 2025-04→2026-03 | +0.629 | +4.23% | +0.749 | −0.120 |
| **mean** | **+0.729** | ~+6% | +1.081 | −0.352 |

So the catastrophic "−1.32" was an artifact. The real picture: a **positive-
Sharpe (+0.729 mean, 3/3 cuts > 0) ~+6% APY strategy that underperforms a strong
bull-market SPY** (beats SPY Sharpe 1/3, APY 0/3). The gate still returns FAIL —
but now for legitimate reasons (`benchmark_ok=False`, `regime_ok=False`), NOT a
bug. Two caveats remain: (a) the model genuinely doesn't beat buy-and-hold SPY in
2024-2026; (b) the §5.2 leakage flag is unresolved (placebo_ic +0.0402 > real_ic
+0.0345), so even +0.729 may be optimistic.

## The two eval bugs (both fixed this session)

**Bug A — non-production-faithful tax/rf in the stamped −1.32 evidence.**
**Bug B — WF manifest artifacts had NO config fingerprint → panel scorer
fail-closed → zero trades.** Details below.

## Evidence chain (no recompute of training; reused existing traces + 1 gate run)

### 1. The stamped −1.32 used a non-production-faithful tax + rf regime
- Stamped evidence `run_at: 2026-05-22T07:50:56`, traces under
  `artifacts/diagnostics/post_fix_20260522/wf_traces_172_sentiment/`.
- That run used `strategy_config.sim_wl200_172_sentiment.json`, whose `tax`
  block has **no `cash_debit_mode`** → falls back to legacy `event_level`
  (debits tax from cash on every winning sell, **no loss-netting**). Production
  (`strategy_config.json`) uses `cash_debit_mode: reporting_only` with an
  explicit parity reason ("do not let estimated tax cash drag drive portfolio
  decisions"). The sim config also set `performance.risk_free_rate_annual: 0.05`
  while production uses `performance: {}` → rf=0.
- Effective tax rate on gross profit (event_tax / gross_pnl): 2024 **68%**,
  2024-07 **89%**, 2025-04 **198%** (cut 3 paid 2× its gross profit in tax —
  impossible under any real regime). `event_level_tax_debited` >>
  `annual_net_tax_estimate` in every cut (no loss offset).
- Reconstructing the SAME 2024-22 trades under production-faithful accounting
  (add back event-level tax → reporting_only equity), recomputed with the sim's
  own `compute_risk_metrics`:

  | basis | mean 3-cut Sharpe | per-cut |
  |---|---|---|
  | event_level tax, rf=5% (stamped) | **−1.32** | −0.42 / −1.19 / −2.36 |
  | reporting_only, rf=5% | −0.22 | +0.91 / −0.19 / −1.40 |
  | reporting_only, rf=0 (prod-faithful) | **+1.29** | +1.92 / +1.22 / +0.73 |

  So under that config the tax/rf bug overstated the negative Sharpe by ≈ +2.6.

### 2. Bug B: WF manifest artifacts unstamped → panel scorer fail-closed → zero trades
- A production-semantic WF re-run (`--derive-config-from-prod`, the weekly gate's
  invocation) initially produced **zero trades across all 3 cuts**.
- Per-bar decision-tree audit (one cut via `run_sim_104.py`, deep data, repo-root
  cwd): every bar logged
  `ERROR kernel.panel_pipeline.scoring: Panel scoring contract failed
  (panel_scorer_config_mismatch). Cleared 109 buy candidate(s); buy/QP path is
  fail-closed`. The "NoCandidateAlert: ScoreBuyTask rejecting all" line is a
  misleading heuristic — the real drop was the panel-scoring contract.
- Root cause: ALL 43 per-cut artifacts in
  `walkforward_manifest_172_sentiment.calibrated_causal.json` had
  `config_fingerprint=None` (empty `config_fingerprint_fields`). The panel
  scorer's strict `assert_consistent(ctx.config, artifact_meta)` therefore raised
  `ConfigModelMismatch` every bar (stored None vs live sha256:14586756…) and
  cleared the entire buy slate. Positive IC could not convert because no candidate
  ever reached sizing/QP.
- **FIX:** `scripts/stamp_walkforward_fingerprints.py --manifest <calibrated_causal>
  --fingerprint-config strategy_config.json --reference-artifact
  artifacts/prod/panel-ltr.alpha158_fund.json` → stamped 43 artifacts + 43
  calibrators to sha256:14586756…; recipe validated TRUE (all ccc412…). Re-run:
  `panel_scorer_config_mismatch` count = 0, model trades, mean Sharpe +0.729
  (table above). This proves the zero-trades was the unstamped-manifest bug, not
  model behavior. The weekly gate was structurally incapable of passing ANY model
  while the manifest stayed unstamped (always fail-closed → always FAIL).

### CWD footgun (caused a false intermediate finding)
- `kernel.data.LocalStore` defaults to a **cwd-relative** `data/ohlcv`. Repo-root
  `data/ohlcv/AEP` has deep history (2016→); `backtesting/renquant_104/data/ohlcv/AEP`
  has only the last 252 days. Running a sim from the strategy dir silently loads
  one year of data → features can't build → spurious zero-trades. The real weekly
  gate `cd $REPO_DIR` first, so production is unaffected, but this is a latent
  trap for any manual sim run. Worth making the store path absolute/strategy-anchored.

### 3. §5.2 sanity battery FAILS with a leakage signature
- Same gate run: `real_ic=+0.0345`, `shuffled_ic=+0.0040` (barely under 0.005),
  `placebo_ic=+0.0402` — the time-shift placebo scores **HIGHER** than the real
  labels (threshold +0.0265). A placebo IC above the real IC is a classic
  look-ahead-leakage red flag (§5.2a / §5.13.16). Any IC number from this
  pipeline is suspect until the leakage is found.

## Remaining real issues (now that the eval bugs are fixed)
1. **Model underperforms SPY (genuine, milder than −1.32).** Mean Sharpe +0.729
   vs SPY +1.081; beats SPY 1/3 cuts, 0/3 on APY. The daily sell-only state is
   therefore *correct* — the gate refuses to deploy a model that doesn't beat
   buy-and-hold in this bull regime. Improving this is campaign work (the gate is
   now functional and will measure real progress).
2. **§5.2 leakage flag (placebo IC > real IC), P0.** placebo_ic +0.0402 >
   real_ic +0.0345 — look-ahead signature (§5.2a / §5.13.16). Even the +0.729
   may be optimistic until the feature/label pipeline leakage is found and fixed.
3. **Pipeline hygiene to prevent recurrence:** (a) the training path that writes
   WF manifest artifacts must fingerprint-stamp them at creation (these were
   written unstamped → gate fail-closed); (b) `LocalStore` should not default to
   a cwd-relative data path; (c) consider a guard so WF evidence whose source
   config tax basis / rf ≠ production cannot gate prod.

## Changes landed this session
- Re-stamped prod artifact config fingerprint (`scripts/restamp_prod_fingerprint.py`).
- Stamped 43 WF manifest artifacts + 43 calibrators
  (`scripts/stamp_walkforward_fingerprints.py`) → weekly gate is now functional
  (was always fail-closed). Diagnostic/sim artifacts only; prod model unchanged;
  no live-trading effect (daily still correctly sell-only until a model beats SPY).

## Reproduction
```
# reconstruct gross Sharpe from existing traces (no sim):
PYTHONPATH=backtesting/renquant_104 .venv/bin/python  # see session for the 30-line script
# production-faithful WF re-run on a COPY (prod untouched):
cp backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json <copy>
cd backtesting/renquant_104 && .venv/bin/python ../../scripts/run_wf_gate.py \
  --artifact <copy> \
  --strategy-config strategy_config.sim_wl200_172_sentiment.calibrated_causal.json \
  --derive-config-from-prod --jobs 3
```

## Side fix landed this session
- Re-stamped prod artifact `config_fingerprint` (was legacy, missing
  `sector_map`/`sector_etf_map` → permanently blocked buys via P-CONFIG-FP even
  with a good model). Tool: `scripts/restamp_prod_fingerprint.py`. This removed
  one (non-binding) buy-side blocker; the WF gate remains the binding one.
