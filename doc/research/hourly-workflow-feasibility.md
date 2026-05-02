# 把 renquant_104 转换成"每小时运行一次"全流水线 — 可行性深入分析

> **状态**: 设计讨论 / 决策前文档。
> **作者要求 (2026-05-01)**: "把 104 的 full workflow 转换成每小时运行一次，对整个系统要进行多大的修改？对模型有什么要求？对数据有什么要求？做全面深入的分析和讨论。"
> **结论 TL;DR**: 工程上可行 (~3-4 周 + 1-2 个月 stabilization)，但有 5 个真正硬约束在等着 — 标的池有限、alpha 半衰期、滑点 + 税务、计算预算、规制。下文按"系统改动 / 模型要求 / 数据要求 / 风险 / 迁移"五个维度展开。

---

## 0. 先把术语钉死

"每小时运行一次"在我们这套系统里至少有四种解释，必须先选一个：

| Mode | 意思 | 范围 |
|---|---|---|
| **A. 推理小时化** | 每小时跑一次完整 InferencePipeline (regime → drawdown → buy gates → sell → buy candidates → ranking → rotation → selection)，在交易时间内 6-7 次 / 日 | 触及 plist + adapter，不动模型 |
| **B. 特征小时化** | 模型仍在日线训练，但 inference 用小时聚合特征 (Plan G 已部分有) | 部分 |
| **C. 标签小时化** | 模型重训以 fwd_1h / fwd_4h 为标签，从根本上预测短期收益 | 模型架构变更 |
| **D. 全流水线小时化** | 数据、特征、标签、训练、推理全部以小时为基本时间单位 | 接近重写 |

**用户所说的 "full workflow" 最可能对应 C 或 D**。本文按 **D (最完整)** 推演，并指出哪些步骤可以分阶段（先 A → B → C → D）渐进。

---

## 1. 当前系统在"什么频率上"做什么

| 组件 | 当前频率 | 关键代码 | 备注 |
|---|---|---|---|
| OHLCV ingestion | 日线 + 小时（部分缓存） | `data/ohlcv/{T}/1d.parquet`, `data/intraday/{T}/1h.parquet` | 50 票 × 2 年小时数据已缓存（CLAUDE.md） |
| Per-ticker tournament 模型 | 每周 (Tue/Thu/Sun) | `kernel/pipeline/pp_training.py` | 训练数据是日线 |
| Panel-LTR XGBoost | 每周 (Sun 主) | `training_panel/pp_panel_training.py` | 27 features，标签 fwd_5d / fwd_20d |
| NGBoost μ/σ head | 每周 | 同上 | 蒙日线训练 |
| Conformal Gate B 拟合 | 手动 (按需) | `scripts/fit_conformal_gate_b.py` | 已 fitted (BULL_CALM/CHOPPY) |
| **完整 Inference**（买卖+排序+轮换） | **每日开盘前** | launchd `daily_104.sh` plist | 1 次 / 日 |
| **Sell-only Inference** | **每 12 分钟**（07:00-12:24 PT 工作日） | launchd intraday plist | ~35 次 / 日 |
| 日志持久化 (runs.db) | 每 inference | `kernel/persistence.py` | ticker_forward_returns 日线粒度 |
| Acceptance gates | 训练后 | `kernel/model_acceptance.py` | 作用于训练产物 |

**关键现状：完整 Inference 一日一跑；sell-only 12-min 跑（但只跑 SellOnlyPipeline，跳过买入候选生成）**。 用户要求的"每小时运行一次"= 把完整 Inference 提到至少 1 次/小时（理想 6-7 次/工作日）。

---

## 2. 系统改动 — 工程量评估

按系统层从下往上：

### 2.1 数据层 (4-7 天工作量)

#### 当前
- 日线 OHLCV 缓存全（235 票 → 现在因 universe fetch 拉到 1006 票）
- 小时数据：~50 票 × 2 年 (locally cached via `data/intraday/{T}/1h.parquet`)
- 10-min 数据：744k 行 × 50 票（CLAUDE.md）

#### 需要补
- **小时 OHLCV 缓存覆盖整个 wl103/wl178/wl500**：需要 ~150-500 票 × 2-5 年小时数据。yfinance 的 1h 接口能取，但每票 ~2 秒 → 200 票 × 5 sec = ~17 分钟（多线程 8 worker 并行）
- **小时 forward-return 计算**：当前 `scripts/backfill_forward_returns.py` 写的是 fwd_1d/5d/10d/20d。需要扩展到 fwd_1h/4h/8h/1d。schema 改动：`ticker_forward_returns` 加 4 列（ALTER TABLE）
- **fundamentals + macro 不变频率**：财报、insider、macro 都是日级以上，broadcast 到每小时即可。`broadcast` 函数 trivial

#### 风险点
- yfinance 的 1h 接口**只允许最近 730 天**。要更长历史得用 polygon.io / Alpaca data API ($$$) 或 Tiingo (cheaper)
- 财报盘前盘后发布，小时粒度上"事件穿越"问题更严重 — 当前 EarningsBlackoutSellTask 只看日历日，需要扩展到小时戳

### 2.2 特征层 (5-10 天工作量)

#### 已经有（Plan G + 10-min 已落）
- `morning_drift`, `afternoon_drift`, `vwap_premium`, `vol_ratio`, `intraday_realized_vol`, `overnight_gap` (hourly aggregates)
- `m_morning_drift`, `m_morning_30min_drift`, ..., `m_reversal_ratio` (10-min aggregates)
- 这些设计时是作为**日线模型的辅助特征**，不是给小时模型直接用的

#### 需要重新设计的特征
日线特征（mom_12_1, beta_60d, resid_mom, realized_vol, drawdown_peak, amihud_illiq）每个都需要"小时版本":
- mom_12_1 → mom_5h_1h（5 个小时之前 vs 1 小时之前的 momentum，避免 overlap）
- beta_60d → beta_252h（252 个小时窗口，约 1.5 个月）
- realized_vol → realized_vol_252h
- amihud_illiq → 小时粒度的 illiquidity（dollar volume / price-move ratio）

注意：仅仅"把日线公式换成小时数据再算一遍"是错的——因为小时数据有显著的**自相关 / 噪声 / overnight gap** 而日线没有。每个特征需要单独验证（比如小时 RSI 应该用什么周期，daily 用 14 天，hourly 用 14 小时还是 100 小时？）

参考: **Bouchaud-Bonart-Donier-Gould 2018 *Trades, Quotes and Prices*** — 高频价格的 microstructure 特征与日级完全不同，模型公式需要重设计。

#### 与现有架构的冲突
- 当前 `RAW_FACTOR_COLS_FOR_NORM` 列表混合了日和小时聚合特征，但都按日线 cross-section 标准化。小时模型应该按**小时 cross-section** 标准化（每个小时戳跨标的的 z-score）。
- `cross_sectional_zscore` 函数当前 group by `date`，需要改成 group by `(date, hour)` 或更细
- BUG 1+2+3 那批 sector-aware fix 在小时模型上需要重做验证 — 小时频率下 sector heterogeneity 可能更甚（intraday flow 行业差异巨大）

### 2.3 模型层 (1-2 周工作量)

#### 标签
- 日线: fwd_5d, fwd_20d（信息半衰期约 1-3 周）
- 小时模型应预测什么: fwd_1h（噪声大），fwd_4h（更稳但 EOD effect），fwd_8h（基本是 next-day open）
- **建议起点: fwd_4h 作为主标签，fwd_8h 作为辅助校验**。Cross-sectional Gaussianize 仍然有效但 group by 应改 (date, hour)

#### 模型架构
- **XGBoost rank:pairwise 仍然适用**（不是必须换 NN）— 但需要：
  - 小时 panel 行数会膨胀 ~6.5× (每天 6.5 小时 × 252 trading days vs 252)
  - 训练时间膨胀 ~6.5×（M2 Pro 上从 5 min → 30 min/CV-fold）
  - CV 必须按时间分割，不能跨小时混
  - PurgedKFold embargo 调整：日线 embargo=20 天，小时模型 embargo 应该是 ~5 天（5 × 6.5 = 32.5 小时，覆盖 fwd_4h 标签泄漏窗口）

#### 特殊问题: 隔夜效应 + 周末效应
- `fwd_4h` 标签从 14:30 出发往后 4 小时跨过收盘 (16:00) → 实际是 `next_day_open + 30min`，这是 **隔夜 gap + 开盘 30min**，与盘中 4h 完全不同分布
- 必须在标签构造时 **MASK OUT** 跨日窗口，或者干脆切两套模型（盘中 + 跨日）
- 周五 14:30 + 4h 跨周末 → 同样要 mask

参考: **Hasbrouck 2007 *Empirical Market Microstructure*** Ch.7 — 跨夜与盘中 return distribution 显著不同，混合训练会让模型学到平均值 noise

### 2.4 推理层 (3-5 天工作量)

#### 核心改动: cron cadence
现在 launchd plist 大约长这样：
- `com.renquant.daily-104.plist`: 每日 06:30 PT
- `com.renquant.intraday-104.plist`: 每 12 min, 07:00-12:24 PT

新增/改：
- `com.renquant.hourly-104.plist`: 每小时整点 06:30, 07:30, 08:30, ..., 12:30 PT（盘中 7 次 + 盘后回归 1 次）
- 现有 `daily-104` 仍保留作为 EOD ledger / persistence anchor

#### Pipeline tasks 改动
当前 `InferencePipeline` 设计假设一天一跑。多次/日运行需要：
- **持仓状态机要改**：当前 `live_state.json` 的 streaks / HWM 是日级累积的；小时跑时不应该每个小时都把 streak +1（一天 7 个小时 → streak 跳到 7）。需要新字段 `bar_id` 或 `hourly_streak` 与 `daily_streak` 分开
- **rotation cooldown 改**：`rotation.persistence_bars` 等参数原本以"日"为单位，需要变成"bar"（保持 bar-level）OR 写两套常数
- **acceptance gates 改**：`max_no_trade_streak_days` 等参数变成 `bars`

#### Sell-only path
SellOnlyPipeline 已经在 12-min 跑，把它升级到完整 InferencePipeline 是可行的，但要注意:
- 12-min 时持仓退出快，可能产生**频繁尾巴动作**（卖一买一卖一）
- 每小时跑买入候选时，候选池在小时之间几乎不变（features 慢变量），candidates 大概率重复 → 我们需要**去重 / cooldown 机制**避免买同一只票每小时一次

### 2.5 风险/Gate 层 (3-5 天工作量)

| Gate | 当前阈值 | 小时模式下要求 |
|---|---|---|
| `tiered_thresholds` (A-gate) | 日截面 percentile | 改成小时截面 percentile |
| Conformal Gate B (M3) | 已 fit (BULL_CALM/CHOPPY) | **必须按小时重新 fit** — 历史 base FDR 在小时粒度上完全不同 |
| `panel_exit` | 日级 rank_score | 同样小时 rank_score |
| `single_day_loss` | 日级 (close-to-close) | 改成 `bar_loss` (1h close-to-close) ? 还是保留日级守卫? — **建议保留日级**作为长期 risk anchor + 加新的 `intraday_loss` 守卫 |
| `EarningsBlackoutSell` | 日级 ±buffer | 改 hourly buffer：财报前 24-48h 全否决，财报后 24-48h 全否决 |

### 2.6 持久化层 (2-3 天工作量)

#### `runs.db` schema 改动
- `pipeline_runs.run_date` (TEXT YYYY-MM-DD) → `run_timestamp` (TEXT ISO 8601)
- `candidate_scores` 加 `bar_id` 列
- `ticker_forward_returns.as_of_date` → `as_of_timestamp` + 新列 `fwd_1h / fwd_4h / fwd_8h`
- `live_state_snapshots` 频率从每日 → 每小时 → DB 增长 ~6.5×

存储量：当前 `runs.alpaca.db` ~45MB，预测 ~300 MB 一年。可控。

#### 影响下游
- `compare_panel_experiments.py` 要支持小时粒度
- `holdout_backtest.py` 需要支持小时窗口和 sample_end with hour
- `fit_conformal_gate_b.py` 同上

### 2.7 broker / live 层 (2-3 天工作量)

- Alpaca 支持 minute 粒度数据 + 几乎无延迟下单。完美适配小时跑
- 但 **wash sale + 短期资本利得**: 美国税法 30 天 wash sale 规则在小时模式下变成无处不在 — 我们当前 `wash_sale_days` 是日级，每小时跑会大量触发
- 短期资本利得 (held < 1 year) 比长期重 ~2× tax — 小时模式下持有期极短 → 几乎全部短期 → **税后回报会被严重稀释**

参考: **Constantinides 1983 (Econometrica) "Capital Market Equilibrium with Personal Tax"** — 高频策略的 after-tax return 比 pre-tax 严重打折，需要在 sizing/holding 阶段显式模拟税务

### 2.8 总工程量估算

| 阶段 | 工作量 | 是否串行 |
|---|---|---|
| Data layer (小时缓存 + DB schema) | 4-7 天 | 串行（其他依赖） |
| Feature layer (小时特征 + 重设计) | 5-10 天 | 部分并行 |
| Model layer (标签 + CV + 训练) | 1-2 周 | 依赖 feature |
| Inference layer (pipeline + cron) | 3-5 天 | 部分并行 |
| Risk/Gate layer | 3-5 天 | 依赖 inference |
| Persistence layer | 2-3 天 | 并行 |
| Broker/live layer | 2-3 天 | 串行（最后） |
| Tests + 验证 + B2 hold-out | 1-2 周 | 串行 |
| **总计** | **6-10 周** | 可压缩到 4-6 周 if 重度并行 |

---

## 3. 模型要求

### 3.1 必须验证的不变性
- **Predictability persists at hourly horizon**: cross-sectional rank-pairwise loss 在小时数据上仍然能产生 OOS Spearman IC > 0.02 (类似日线下限)。**未验证之前不能上线**。
- 推荐先做的实验：用现有 wl103 + 已 cached 50 票 × 2y 小时数据 → 训一个小时模型 → 看 CPCV mean_ic。如果 ≤ 0.02，说明小时频率没有可榨取的 cross-sectional alpha，整个项目应该停。

### 3.2 标签设计要求
- fwd_4h 主标签（盘中），fwd_8h 跨日辅助
- **隔夜窗口必须 mask**（否则训练学到的是 overnight gap noise，对盘中决策没用）
- Gaussianize per-(date, hour) 而非 per-date

### 3.3 特征要求
- 小时特征 27+ 个（日级直接借助有用，但 momentum / vol / illiquidity 全部小时化）
- **加入新的 microstructure 特征** — order flow imbalance (Cont 2014), Kyle's lambda (price impact per dollar volume), bid-ask spread proxy (Roll 1984)
- 数据要求扩到 bid/ask quotes — yfinance 不提供，需要 polygon 或 IEX

### 3.4 模型选择
- **XGBoost rank:pairwise 仍是基线**（CLAUDE.md 路径，已验证）
- panel_ltr.training_resolution='hourly' 已经在 BuildHourlyResolutionPanelTask 里有钩子（CLAUDE.md 提到 Stage C-2）— 半成品，需要完成
- 如果小时 OOS IC < 0.02，可能需要 NN backend (Phase C/D 设计) — 因为小时数据更适合 LSTM/Transformer

### 3.5 校准要求
- Conformal Gate B 必须按小时**全部重 fit**
- Per-regime calibration 在小时粒度上数据更稀疏 — 可能需要把 BULL_VOLATILE 和 BEAR 合并(已经有这个问题在日线了)
- Acceptance gates 中 `min_panel_rows >= 75k` 这种阈值在小时模式下要 ×6.5 → ~500k

---

## 4. 数据要求

### 4.1 必有
- ✅ 小时 OHLCV，~5 年历史，所有 watchlist 票（部分已 cached）
- ✅ Forward returns 在小时粒度（**当前缺**，需扩展 backfill_forward_returns.py + DB schema）
- ✅ Earnings calendar 含**精确盘前/盘后时间戳**（当前只有日期；yfinance .info.earningsTimestamp 提供 UTC 戳）

### 4.2 强烈建议
- 🟡 Bid/ask quotes (5 年小时级别) — 需要付费数据源
- 🟡 Order book imbalance / book pressure（深度数据）— 同样付费
- 🟡 Macro factors 日内更新（VIX 实时，HYG 实时） — Bloomberg / Polygon

### 4.3 数据成本估算
- yfinance: 免费，但 1h 历史只 730 天。盘前/盘后数据一般。
- Polygon.io basic plan: $99/月，全市场 1h + 5min OHLCV + 5 年历史
- Alpaca data API: 免费 tier 提供 1m 数据但限速；付费 $99/月扩量

**建议: 先用 yfinance 730 天历史快速验证概念可行性。如果 OOS IC > 0.02，再投资 Polygon 拉 5 年完整历史。** 不要在概念验证前花这个钱。

---

## 5. 风险 / 隐性硬约束（5 个真正难处）

### 5.1 Alpha 半衰期问题
- 日线 alpha 的半衰期 ~1-3 周（Jegadeesh-Titman 1993; Asness-Moskowitz-Pedersen 2013）
- 小时 alpha 的半衰期估算 ~1-3 天 (Boudoukh-Israel-Richardson 2009 的 momentum decay 文献)
- **意味着**：模型必须重训得很快 — 周一训出来到周三就过期
- 当前训练管道 30 分钟一次 retrain，小时 cadence 下推荐每天重训一次
- 计算预算：每天 1 次 × 30 min × 5 工作日 = 2.5 hours/week — M2 Pro 抗得住

### 5.2 滑点和交易成本
- Alpaca 零佣金，但 spread + 价格冲击是真实的
- 小时 turnover 可能日级 2-5×（更频繁调仓）
- 0.05% spread × 5 turnover = 25 bps drag/day → ~63% drag/年 — **会吃掉所有 alpha**
- **意味着**：模型必须产生 OOS IC > 0.04 (vs 当前 0.04) 加上明显更高的 turnover 控制

参考: **Almgren-Chriss 2000 "Optimal Execution"** — 频繁调仓下 trading impact 是支配性成本。

### 5.3 税务后劣势
- 几乎所有 trades 触发 short-term gain (持有 < 1y)
- 边际税率 35-37% (federal) + state — 净税后 alpha = pre-tax × 0.6
- 当前长期持有部分（30 days approaching 365）是 alpha 的主要部分
- **意味着**：小时模式 vs 日模式 if 一样的 pre-tax IC，**after-tax IC 比日模式低 30-40%**

memory `feedback_after_tax_principle.md` 直接说：所有性能数字必须是 after-tax 的。**小时 mode 的 sales pitch 比日 mode 难得多**。

### 5.4 计算预算
- 每小时跑完整 InferencePipeline (含 178 票候选扫描 + panel scoring + ranking) 当前约 ~3-5 秒
- 7 次/日 × 工作日 — 计算无问题
- 但每天重训整个 panel ~30 min — 在每天 06:30 PT 之前必须完成 — **这是新的硬时间窗口**

### 5.5 监管 / pattern day trader rule
- US: 5 天内 >= 4 day-trades 就要 maintain $25k account minimum
- 当前 ~$10k 账户在小时 mode 下**几乎肯定触发**
- 需要：要么扩账户到 $25k+，要么严格限制 day-trades （sell + same-day buy of same security = day trade）

参考: FINRA Rule 4210(f)(8)(B)(ii) — pattern day trader definition

---

## 6. 验证策略 (B2 hold-out + Live A/B)

### 6.1 阶段验证 (gate)
不能直接上线。逐阶段 gate：

```
Phase 0: feasibility study
  → train 小时 panel-LTR on cached 50-ticker × 2yr
  → CPCV mean_ic on hourly fwd_4h
  → STOP if < 0.020

Phase 1: B2 hold-out
  → train_end = 2024-12-31, sim 2025-01-02 → today on hourly
  → Compare apy_holdout (hourly) vs apy_holdout (daily, current production)
  → STOP if hourly APY <= daily APY × 0.9 (after-tax)

Phase 2: Paper trading 2 weeks
  → live runner on Alpaca paper account, hourly cadence
  → Verify executions match sim, no infrastructure crashes
  → STOP if any infrastructure failure

Phase 3: Real $1k 1 month
  → smallest meaningful real-money account
  → Compare to production daily-104 in same window

Phase 4: Promotion (only if Phase 0-3 all clear)
  → upgrade golden config + promote hourly
  → daily-104 archived as fallback
```

### 6.2 KPI 比较表

| Metric | Daily (current production) | Hourly target |
|---|---|---|
| OOS CPCV mean_ic | +0.0418 (wl103) | ≥ +0.030 (after lower-quality penalty) |
| Pre-tax APY | ~+65% (live, ~6 months data) | ≥ +90% (must compensate higher turnover) |
| **After-tax APY** | ~+40% (long-term + short-term mix) | ≥ +50% (compensate short-term tax penalty) |
| Sharpe | not yet formally tracked | ≥ Daily's Sharpe |
| Max DD | ~6% | ≤ 8% |
| Turnover (annual) | ~3-5× | < 15× (else costs eat alpha) |
| Avg holding period | ~14 days | ≥ 3 days (else wash sale 噩梦) |

---

## 7. 迁移计划 (按 6-10 周排)

```
Week 1-2  Phase 0 feasibility — minimum viable hourly panel-LTR
          ✓ Extend ticker_forward_returns DB schema (add fwd_1h/4h/8h cols)
          ✓ Backfill hourly forward returns for cached 50 tickers
          ✓ Adapt cross_sectional_zscore to (date, hour) groups
          ✓ Adapt PanelTrainingPipeline for hourly resolution (BuildHourlyResolutionPanelTask)
          ✓ Train tiny hourly panel + CPCV
          ✓ DECISION GATE: CPCV mean_ic ≥ 0.020 ? else STOP

Week 3-4  Phase 1 — full data pipeline + label correctness
          ✓ Pull hourly OHLCV for full wl103 (or wl178 if v2 sector-aware passes)
          ✓ Hourly forward returns end-to-end
          ✓ Fix overnight / weekend label masking
          ✓ Hourly EarningsBlackoutSell with timestamps
          ✓ Run B2 hold-out hourly

Week 5-6  Phase 2 — inference + risk + persistence
          ✓ launchd hourly plist
          ✓ DB schema migration (run_date → run_timestamp; add bar_id)
          ✓ Streak / cooldown counters: bar_id-aware
          ✓ Conformal Gate B re-fit at hourly granularity
          ✓ Pattern day trader rule guard

Week 7-8  Phase 3 — paper trading + tax simulation
          ✓ Paper trade live for 2 weeks
          ✓ After-tax APY sim with explicit short-term gain modeling
          ✓ Slippage measurement (real fill vs target)

Week 9-10  Phase 4 — promotion or rollback decision
          ✓ Side-by-side metrics
          ✓ If pass: promote golden + archive daily fallback
          ✓ If fail: write to failed-experiments-log, keep daily, next idea
```

---

## 8. 开放问题（要决策）

1. **A/B/C/D 的哪一种是真正想要的？** 如果只是 A (推理小时化但模型不变)，工作量缩到 1 周。建议**先做 A 验证执行有意义，再逐步推 D**。
2. **预算多少给数据？** Polygon $99/月 是 break-even — 如果 alpha-cost ratio > 1.2 就值得。
3. **是否扩账户到 $25k+ 解 PDT 锁？** 否则小时模式无法在美零售账户下完整跑。
4. **当前 wl178 sector-aware 还在火 — 应该等它落地再开 hourly 吗？** 我倾向是 — 因为 hourly 也会受 sector heterogeneity 影响，单跑两次实验是浪费。
5. **fwd_4h 还是 fwd_8h 主标签？** 需要小型实验验证（Phase 0）。
6. **NN backend 要不要在小时模式上一起做？** Phase D MIGA 在小时数据上的论文支持比日模式更强（多数论文用 30-min CSI300）— 可以考虑把 NN + hourly 一起作为单个大 phase。

---

## 9. 我的推荐立场

**技术上可行 + 经济上不一定值得**。

理由：
- ✅ 工程量 6-10 周，明确可分解（每个 phase 有 gate）
- ✅ 数据已部分准备（10-min bars + Plan G hourly features 已落）
- ✅ 模型架构 (XGBoost rank:pairwise) 已验证可在小时数据上跑（Stage C-2 钩子已设计）
- ❌ **税后 alpha 风险**: 短期资本利得 35-37% 边际税率会吃掉小时模式的额外 alpha
- ❌ **滑点 + turnover 风险**: 高频调仓会让真实 alpha 比 paper sim 低 30-50%
- ❌ **PDT 规则**: 当前账户 size 在小时模式下不可行
- ⚠️ **机会成本**: 同样 6-10 周可以做 wl500 + Phase C/D NN backend，我倾向那个 ROI 更高

**建议路径**: 先做 Phase 0 (1-2 周 minimum viable feasibility test) — 用 cached 数据训一个小时 panel，看 CPCV IC。如果 ≥ 0.030 才继续，否则归档。绝大部分预期是 IC 不够支撑高频，但试验本身便宜。

写到这里，剩下等你的方向：A/B/C/D 哪一个？feasibility test 立即开 (1-2 周) 还是等 sector-aware 落地后？
