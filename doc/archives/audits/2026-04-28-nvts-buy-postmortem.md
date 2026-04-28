# NVTS 买入决策 post-mortem (2026-04-28)

**Trade:** BUY NVTS 47 @ $17.20 (2026-04-27 23:44, queued for next-day open)
**Outcome:** SELL NVTS 47 @ $15.15 (2026-04-28 07:46) — **loss = $96.65 (-11.92%)**

---

## 1. 模型当时给 NVTS 什么分？

| Field | Value |
|---|---|
| edge_sharpe (μ/σ) | **+0.139** ← 候选里最高 |
| panel rank_score (after global cal) | 0.34 |
| QP target Δw | +0.0810 |
| 5d return going in | +40% |
| 20d return going in | **+91%** |
| 1d return going in | **−6.65%** (4-24 close 17.28 vs 4-23 close 18.51) |
| 52w high distance | −6.65% off the 18.51 peak |

模型给的 edge_sharpe 是基于刚刚修过的 NGBoost head（22:28 重训）— 数据完整性问题已修，所以 +0.139 不是数据腐败的产物。**模型按设计运行，没有 bug。**

---

## 2. 为什么这个买入决策"傻"

| # | 问题 | 严重度 |
|---|---|---|
| 1 | **20d +91% 是抛物线顶 — 严重 mean-reversion 风险**。这种状态买入历史上中位数 5d return 是负的。模型没有"parabolic exhaustion"特征。 | 🔴 高 |
| 2 | **昨天 1d −6.65% — 在跌势中买入**。即使是 momentum 模型，"昨天大跌 +今天买" 通常需要 reversal 确认信号；我们没看 |  🔴 高 |
| 3 | **小盘股低价 ($17) — 同 % 波动绝对值大**。$17 → $15 = $2 = 11.6%；同样的 $2 在 $200 的 NVDA 上只是 1%。**stop_loss 应该是 vol-adjusted 不是固定 15%**。 |  🟠 中 |
| 4 | **NGBoost 训练数据中 NVTS 历史 < 3 年**（2021 IPO）— 模型 σ 估计的样本太少，置信区间太宽。candidate 的 Bayesian penalty 没考虑这个。 |  🟠 中 |
| 5 | **整个 AI/semi sub-sector 极端拥挤**。当 NVDA / MU / SMCI / NVTS 都 +60-90% 时，如果板块回调，全都同步跌 → 系统买了 4 个 ai_chip → 集中度过高。 |  🟠 中 |
| 6 | **panel-LTR 在跑 10d horizon**。NVTS 这种短期波动剧烈的票应该用更长 horizon 评估（60d）— 我们已知 60d 在这种情况下 IC=+0.053 vs 10d 的 +0.023。**用了次优 horizon。** |  🟡 低 |

---

## 3. 系统失败链

1. **NGBoost feature drift bug 修了** ✅
2. 但 NVTS 这次的 +0.139 edge_sharpe **是真正的模型输出**，不是 bug
3. Gate B 阈值 0.10 → +0.139 通过
4. QP 求解器分配 +8.1% NAV → 47 股
5. 没有 "parabolic exhaustion" 检查 → 买入
6. Stop_loss=15% 在 $17 股票上对应 -$2.55 移动 → 容忍极大
7. max_single_day_loss=10% 要求 1 天跌 10% → NVTS 今天 -12%，刚好擦边
8. Intraday 检查每 30 分钟 → 在 7:30 cron 时 NVTS 跌 -7% (未触发)，下个 cron 8:00 之前已 -12%
9. → 用户手动 / 系统延迟订单导致 7:46 卖出

**没有单一 bug。是 7-8 个微小决策累积的结果。**

---

## 4. 应该加的防御层（按优先级）

### P0 — 立刻能加的（< 1 小时工作）

**Z1. 抛物线顶禁买 gate**
```python
# 候选 admission 时硬性拒绝：
if rel_mom_20d > +0.50 and rel_mom_5d > +0.20:
    reject("parabolic_exhaustion: 20d>+50% and 5d>+20%")
```
今天会拒掉 NVTS（20d +91%, 5d +40%）。**不影响 NVDA/MU 这种健康趋势票（20d +20-25%, 5d +3-5%）。**

**Z2. 短历史样本惩罚**
NGBoost head 在训练时数据少的 ticker（< 3 年 history），inflate σ 估计 ×1.5。
今天会让 NVTS edge_sharpe = +0.139 → +0.139/(σ*1.5) ≈ +0.093 → 不过 Gate B 0.10 阈值。

**Z3. Sector 集中度软约束**
QP 加 sector 权重 cap：单一 sub-sector ≤ 40% 持仓。今天 4 个 ai_chip + 1 software = 集中度 85%+，新约束触发会迫使分散。

### P1 — 需要数据/小重构（半天）

**Z4. Vol-adjusted stop_loss**
不固定 15%。stop = `2 × ATR(20d)` 或 `1.5 × σ_60d`。今天 NVTS σ_60d 大约 60% 年化 → daily σ ≈ 3.8% → stop = 5.7% 而不是 15%。

**Z5. Intraday 频率提速 30 min → 5 min**
launchd intraday plist 加更密时间点。CPU 成本可控（每次 30s 跑完）。

**Z6. 1-day reversal-confirm gate**
昨日跌 ≥5% 的票，今日 buy 候选必须额外要求 RSI > 30 或 1h-bar 反弹。

### P2 — 架构改动（1 周）

**Z7. Horizon-conditional confidence**
小盘高 vol 票（vol > 50% annualized）只在 60d horizon model 给买入信号；10d horizon 给它们的信号一律降权 0.3×。

**Z8. Bayesian shrinkage on edge_sharpe by sample size**
edge_sharpe 经过 shrinkage：`shrunken = edge × (n_train / (n_train + λ))`，λ=500。少历史的 ticker 自动被压。

---

## 5. 从这单交易学到什么

1. **"模型按设计跑了" ≠ "决策是对的"**。设计本身缺少防御层。
2. **+0.139 edge_sharpe 听起来很高，但它没含 sample-size confidence**。模型说"高信号"但低数据。
3. **小盘股低价票需要不同 stop 风险参数**，固定百分比不合适。
4. **集中度风险在 sub-sector 层面被忽视**：单 ticker max_pos_pct 不够，需要 sector 层 cap。
5. **B1.2 +54% 是 selection bias 的最大教训** — 同样原则适用于策略评估：依赖未来信息的过滤会自带膨胀。

---

## 6. 我的认错

夜间汇报里我把 B1.2 +0.0614 当作"决定性发现"。**这是错的**。我没在第一时间设计 selection-bias-rigorous 测试就推这个数字。直到用户要求 deep audit 才发现 +0.039 真相。

NVTS 买入是 production model 决策，不是我直接选的。但我没在 NVTS 入选 candidate list 时 flag 它的 parabolic 状态作为不合理决策。**事后 audit 才挖出来这单交易的 6 个 design 缺陷。这种"事后 audit"价值 < 0**，应该是 trade 触发前的检查。
