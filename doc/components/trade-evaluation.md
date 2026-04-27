# Trade-Evaluation DB + RL Off-Policy Evaluation — Design (2026-04-26)

**Author:** RenQuant team (round-7 session)
**Status:** 🔴 Design only; implementation queued in roadmap.
**Driver:** User spec 2026-04-26 — *"我想要一个db，来存储我的trade，这样
7天，14天，28天后可以re evaluate我的trade的合理性，用这个数据来校验我的
model，用点强化学习的概念理解我的需求"*

## 1. Problem statement

Today's "did the trade work?" answer happens informally and only on extreme
outcomes. We lack a **systematic, time-delayed, model-attributable** trade-
evaluation pipeline. The user wants:

1. **Storage**: every trade decision archived with full decision-time
   context.
2. **Time-delayed evaluation**: at fixed horizons after the trade
   (T+7d, T+14d, T+28d), automatically compute realized-vs-expected
   outcome.
3. **Model validation feedback loop**: aggregate per-trade regret
   over time → which model/regime/feature combinations consistently
   succeed vs fail.
4. **RL framing**: treat each trade as an `(s, a, r)` tuple; the
   "policy" is the current model; we want **off-policy evaluation**
   (OPE) so we can estimate counterfactual policy performance without
   re-running history.

This isn't backtesting (which is in-sample noise per the P0 honest-
backtest item). It's **forward-causal evaluation**: did the actual
trade earn the expected return on out-of-sample forward windows?

## 2. RL formulation

Map the trading workflow to standard RL terminology:

| RL concept | RenQuant equivalent |
|---|---|
| **State `s_t`** | (regime, confidence, holdings, market features at decision time) — already captured in `live_state_snapshots` + `ticker_daily_state` |
| **Action `a_t`** | The trade tuple `(ticker, side, shares, price)` — captured in `trades` |
| **Policy π(a\|s)** | Current `InferencePipeline` configuration — versioned by `commit_sha` in `pipeline_runs` |
| **Reward `r_t`** | Realized P&L over horizon h, optionally tax-adjusted, optionally compared to SPY benchmark — **NOT YET STORED PER-TRADE** |
| **Value V_π(s)** | Expected forward return for the policy from state s — what we want to estimate |
| **Behavior policy μ** | The actual policy that placed the trade (= our live π) |
| **Target policy π'** | A counterfactual policy we want to evaluate (e.g. "what if Gate B threshold was 0.20?") |
| **Importance weight ρ** | π'(a\|s) / μ(a\|s) — for off-policy evaluation |
| **Episode** | A position lifecycle: entry → exit (or current as of T+28d) |

The **off-policy evaluation** subfield of RL is exactly this: estimate
the value of one policy from data collected by a different (or older)
policy.

## 3. Academic references

### Off-policy evaluation (core RL methodology)

1. **Sutton & Barto (2018)**. *Reinforcement Learning: An Introduction*
   (2nd ed.), MIT Press. Chapters 5.5-5.9 + 7.4 (off-policy methods).
   The canonical reference.
2. **Precup, Sutton & Singh (2000)**. *"Eligibility Traces for Off-
   Policy Policy Evaluation"*, ICML 2000. — The foundational
   importance-sampling estimator.
3. **Jiang & Li (2016)**. *"Doubly Robust Off-policy Value Evaluation
   for Reinforcement Learning"*, ICML 2016. — Combines importance
   sampling with a value-function estimate; lower variance.
4. **Thomas & Brunskill (2016)**. *"Data-Efficient Off-Policy Policy
   Evaluation for Reinforcement Learning"*, ICML 2016. — More variance-
   reduction techniques.
5. **Doroudi, Thomas & Brunskill (2017)**. *"Importance Sampling for
   Fair Policy Selection"*, IJCAI 2017. — How to compare multiple
   policies fairly off-policy.

### Causal inference + counterfactual analysis

6. **Pearl (2009)**. *Causality: Models, Reasoning, and Inference*
   (2nd ed.), Cambridge UP. — Why "what if we'd done X instead?" is
   a causal question, not a statistical one.
7. **Athey & Imbens (2019)**. *"Machine Learning Methods That
   Economists Should Know About"*, Annu. Rev. Econ. — Practical
   doubly-robust estimators in finance contexts.

### Finance-specific

8. **López de Prado (2018)**. *Advances in Financial Machine
   Learning*, Wiley.
   - **Ch. 11** (Cross-validation in finance) — purged + embargoed CV
     for time-series; we already use CPCV for panel training.
   - **Ch. 14** (Backtesting on Synthetic Data) — combinatorial
     methods to estimate "haircut" Sharpe vs reported Sharpe.
   - **Ch. 16** (Machine learning asset allocation) — formal RL
     framing of portfolio construction.
9. **Bailey, Borwein, López de Prado & Zhu (2014)**. *"Pseudo-
   Mathematics and Financial Charlatanism: The Effects of Backtest
   Overfitting on Out-of-Sample Performance"*, Notices of the AMS
   61(5). — DSR (deflated Sharpe ratio) for multiple-testing
   correction; relevant when we re-evaluate many trades.
10. **Cont (2001)**. *"Empirical properties of asset returns: stylized
    facts and statistical issues"*, Quant. Finance 1(2). — Why
    forward returns have fat tails + how to reason about them.

### Reward shaping in finance

11. **Moody & Saffell (2001)**. *"Learning to Trade via Direct
    Reinforcement"*, IEEE Trans. Neural Networks 12(4). — Differential
    Sharpe ratio as a reward signal; avoids the
    P&L-monotone-but-Sharpe-degrading trap.
12. **Deng, Bao, Kong, Ren & Dai (2017)**. *"Deep Direct
    Reinforcement Learning for Financial Signal Representation and
    Trading"*, IEEE Trans. Neural Networks. — Modern application of
    Moody-Saffell with deep nets.

### Policy gradient + bandit framing (alternative lens)

13. **Sutton, McAllester, Singh & Mansour (1999)**. *"Policy Gradient
    Methods for Reinforcement Learning with Function Approximation"*,
    NIPS. — The starting point if we want to do gradient-based
    policy improvement using OPE estimates.
14. **Li, Chu, Langford & Schapire (2010)**. *"A Contextual-Bandit
    Approach to Personalized News Article Recommendation"*, WWW. —
    Trades-as-bandits framing; good intro for non-sequential decisions.

## 4. Schema additions

Three new tables in `runs.db`:

### 4.1 `trade_outcomes` — realized return at multiple horizons

```sql
CREATE TABLE IF NOT EXISTS trade_outcomes (
    run_id            TEXT NOT NULL,        -- FK trades.run_id
    ticker            TEXT NOT NULL,
    action            TEXT NOT NULL,        -- 'buy'|'sell'|'trim'|'rotation'
    decision_date     DATE NOT NULL,        -- when the trade was placed
    decision_price    REAL NOT NULL,
    decision_regime   TEXT,
    decision_confidence REAL,
    -- Forward outcomes at standard horizons (NULL until backfilled)
    fwd_1d_pct        REAL,
    fwd_5d_pct        REAL,
    fwd_7d_pct        REAL,                 -- per user spec
    fwd_14d_pct       REAL,                 -- per user spec
    fwd_28d_pct       REAL,                 -- per user spec
    -- Benchmark-relative (ticker minus SPY at same horizon)
    fwd_1d_excess     REAL,
    fwd_5d_excess     REAL,
    fwd_7d_excess     REAL,
    fwd_14d_excess    REAL,
    fwd_28d_excess    REAL,
    -- Decision-time predictions (for regret/calibration)
    expected_pnl_pct  REAL,                 -- from kelly / panel / mu
    expected_horizon_days INTEGER,          -- when the model thought it would realize
    -- Realized regret (per horizon, after backfill)
    regret_7d         REAL,                 -- expected - realized at 7d
    regret_14d        REAL,
    regret_28d        REAL,
    backfill_status   TEXT DEFAULT 'pending', -- 'pending'|'partial'|'complete'
    last_backfill_at  TIMESTAMP,
    PRIMARY KEY (run_id, ticker, action, decision_date),
    FOREIGN KEY (run_id) REFERENCES trades(run_id)
);
CREATE INDEX idx_to_decision_date ON trade_outcomes(decision_date);
CREATE INDEX idx_to_backfill_status ON trade_outcomes(backfill_status);
```

### 4.2 `policy_versions` — version each "policy" config so OPE can compare

```sql
CREATE TABLE IF NOT EXISTS policy_versions (
    policy_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    version_label     TEXT NOT NULL,        -- e.g. 'v4.1', 'v5-sell-gate-b'
    activated_at      TIMESTAMP NOT NULL,
    deactivated_at    TIMESTAMP,            -- NULL = currently active
    config_snapshot   TEXT NOT NULL,        -- JSON of strategy_config.json
    commit_sha        TEXT NOT NULL,
    notes             TEXT
);
CREATE INDEX idx_pv_activated ON policy_versions(activated_at);
```

Each `pipeline_run.run_id` can be joined to `policy_versions.policy_id`
via `pipeline_runs.commit_sha → policy_versions.commit_sha`.

### 4.3 `policy_evaluations` — periodic rollups for ops/UI

```sql
CREATE TABLE IF NOT EXISTS policy_evaluations (
    eval_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    policy_id         INTEGER NOT NULL,     -- FK policy_versions
    eval_date         DATE NOT NULL,        -- when evaluation was computed
    horizon_days      INTEGER NOT NULL,     -- 7, 14, 28
    n_trades_eval     INTEGER NOT NULL,
    -- Realized metrics
    avg_pnl_pct       REAL,
    sharpe            REAL,
    win_rate          REAL,                 -- fraction with pnl > 0
    -- Calibration metrics
    avg_regret        REAL,                 -- mean(expected - realized)
    regret_std        REAL,
    -- OPE estimators (for compare against alternative policies)
    ope_is_value      REAL,                 -- importance-sampling estimator
    ope_doubly_robust REAL,                 -- DR estimator
    notes             TEXT,
    UNIQUE(policy_id, eval_date, horizon_days),
    FOREIGN KEY (policy_id) REFERENCES policy_versions(policy_id)
);
```

## 5. Workflow

### 5.1 At trade-placement time (T)

`adapters/runner.py::commit()` — already records to `trades` table.
**Extension**: also INSERT a `trade_outcomes` row with:
- All decision-time fields populated
- All `fwd_*` fields NULL
- `backfill_status = 'pending'`

### 5.2 Nightly backfill (T+1 to T+28d)

New script `scripts/backfill_trade_outcomes.py`:
1. Find all `trade_outcomes` with `backfill_status != 'complete'` AND
   `decision_date <= today - 1d`.
2. For each, compute `fwd_Nd_pct` for whichever horizons have enough
   forward data (T+1d / T+5d / T+7d / T+14d / T+28d).
3. Compute `fwd_Nd_excess` against SPY.
4. Compute `regret_Nd = expected_pnl_pct - fwd_Nd_pct` (sign-corrected
   for buy vs sell).
5. Set `backfill_status = 'partial'` until T+28d backfilled, then `'complete'`.

Hook: append to `daily_104.sh` step 2b (right after
`backfill_forward_returns.py`).

### 5.3 Weekly rollup (Sun)

New script `scripts/rollup_policy_evaluations.py`:
1. For each active `policy_id`, for each horizon ∈ {7, 14, 28}:
   - Pull all completed `trade_outcomes` for that policy + horizon
   - Compute `avg_pnl_pct`, `sharpe`, `win_rate`, `avg_regret`, etc.
   - Compute OPE estimators if there's a candidate counterfactual
     policy in `policy_versions` (e.g., the previous golden config)
2. INSERT row into `policy_evaluations`.
3. Fire ntfy if any horizon shows degradation > 1σ vs the rolling
   8-week average.

Hook: add to Sun retrain pipeline (after `retrain_panel.sh`).

### 5.4 Operator dashboard (manual review)

`scripts/show_policy_evaluations.py` — pretty-print:

```
Policy v4.1 (since 2026-04-23, 47 trades evaluated):
  T+7d   avg=+1.2%  sharpe=1.45  win=58%  regret=+0.3%
  T+14d  avg=+2.4%  sharpe=1.52  win=61%  regret=-0.1%
  T+28d  avg=+4.1%  sharpe=1.61  win=64%  regret=+0.5%

Policy v5-candidate (sell-gate-b enabled, 12 trades evaluated):
  T+7d   avg=+0.9%  sharpe=1.31  win=55%  regret=+0.4%
  ...
```

## 6. Off-policy evaluation (the RL part)

The simple version: importance sampling estimator. For each trade
`(s_t, a_t)` placed by the live policy μ, compute the probability the
candidate policy π' would have placed the same trade:

```
ρ_t = π'(a_t | s_t) / μ(a_t | s_t)
V_OPE(π') = (1/N) Σ ρ_t · r_t
```

Where `r_t` is the realized horizon-h return.

**Concrete in our system:**

| Counterfactual question | How to compute π'(a|s) |
|---|---|
| "What if Gate B threshold = 0.20 (vs 0.10)?" | Re-evaluate `_gate_b_edge_sharpe` for each historical candidate; π' = 1 if it would have been kept, 0 if rejected. ρ_t ∈ {0, 1, ∞} (deterministic policies). |
| "What if max_sells_per_bar = 3 (vs 2)?" | Re-run `LimitSellsPerBarTask` logic on stored `ctx.exits`; same {0, 1} weighting. |
| "What if we switched to transformer backend?" | Need to re-score with the transformer artifact; expensive but tractable. |

**Doubly robust extension (Jiang & Li 2016):**
Combines IS with a learned value function `V̂(s)` so that bias from
either source is corrected by the other. Cuts variance ~10× vs raw IS
on typical financial data.

**Practical: start with raw IS at horizons 7/14/28. Add DR in phase 3.**

## 7. Implementation phases

### Phase 1 — Schema + write-path (P1, ~3 hours)

- [ ] Add `_SCHEMA_SQL` for the 3 new tables in `kernel/persistence.py`.
- [ ] Add `record_trade_outcome(conn, run_id, ticker, action, ...)` writer.
- [ ] Hook into `adapters/runner.py::commit()` after each trade.
- [ ] Backfill from existing `trades` table for the last 30 days.

### Phase 2 — Backfill + benchmarks (P1, ~2 hours)

- [ ] `scripts/backfill_trade_outcomes.py` — nightly idempotent.
- [ ] Hook into `daily_104.sh` step 2b.
- [ ] Tests: unit test on synthetic 7-day window.

### Phase 3 — Weekly rollup + ntfy (P1, ~3 hours)

- [ ] `scripts/rollup_policy_evaluations.py`.
- [ ] Hook into `retrain_panel.sh`.
- [ ] ntfy alert on > 1σ degradation.

### Phase 4 — OPE estimators (P2, ~5 hours)

- [ ] `kernel/ope.py` with `importance_sampling()`, `doubly_robust()`.
- [ ] CLI: `scripts/eval_counterfactual_policy.py --threshold-gate-b 0.20`.
- [ ] Tests against known synthetic distributions.

### Phase 5 — Dashboard (P2, ~2 hours)

- [ ] `scripts/show_policy_evaluations.py` (pretty-print).
- [ ] Optional: simple HTML dashboard in `docs/` for at-a-glance view.

### Phase 6 — Closed-loop policy improvement (P3, ~weeks)

The endgame the user is gesturing toward: **use the OPE results to
automatically nominate config changes**. E.g., if v5-candidate-A
estimates +0.5σ Sharpe vs current golden across 50+ historical trades,
auto-promote (or at least auto-generate the A/B sim job to confirm).

This is **off-policy improvement** — much harder than evaluation.
Requires:
- Continuous-action policy gradient (or discrete action space + Q-learning)
- Confidence bounds on OPE estimates (Bottou et al. 2013 — "Counterfactual
  Reasoning and Learning Systems")
- Safe-RL constraints to not promote a worse policy in volatile periods

**Defer until Phase 1-5 produce 6+ months of trade data.** Premature
without it.

## 8. Cross-references

- **Existing**: `ticker_forward_returns` table (used for panel labels);
  `candidate_scores` (decision-time features); `trades` (action record).
- **Related design docs**: `doc/roadmap.md` §B1-B4 (honest
  backtest framework — the same problem from the SIM side); `doc/
  database.md` (current schema).
- **Adjacent open work**: roadmap §144 (streak → db migration —
  also part of "make db the canonical store" theme); roadmap §B1
  (walk-forward sim runner — provides the BEHAVIOR-policy dataset
  for OPE).

## 9. What this DOES and DOESN'T do

**Does:**
- Forward-causal evaluation of placed trades.
- Time-stratified outcome attribution (7d / 14d / 28d).
- Counterfactual policy comparison (without re-running history).
- Audit trail for "did this trade actually work?"

**Doesn't:**
- Replace honest backtesting (P0 in roadmap) — that's separate.
- Provide statistical significance for individual trades (sample size
  too small until Phase 6).
- Account for execution slippage (use `live/logs/` JSON for that).
- Tax-aware evaluation (mark for Phase 4 — needs holding-period logic).
