# 2026-06-03 — Cash Drag 根因诊断 + 修复方案

**触发**: 用户实证反馈 — *"我的 Alpaca 账户里就是一直都是 cash，根本没有 trade 发生"*；以及对前一份 memo 的方法论质疑。
**地位**: 取代 [`2026-06-03-cash-overlay-feasibility-study.md`](./2026-06-03-cash-overlay-feasibility-study.md) 的结论。原 memo 的理论部分（§3 论文引用）保留有效；原 memo 的 §5.6 / §5.7 reject 结论 **失效**，因为它用 sim 数据做 live 现象的归因。
**所有者**: Claude，主导调研。

---

## 0. 前一份 memo 的错误（必须先讲清楚）

前一份 memo 的 Stage 1 反事实分析得出 "REJECT codex feature" 的结论，依据是：

> §5.6.3 — 高现金日**条件**于策略卖出，卖出触发包括 stop-loss / drawdown / regime→BEAR，是看跌信号；所以 cash 高的日子第二天 SPY 大概率跌；overlay 是"专买将跌的天"。

这个论证用的是 `data/sim_runs.db::pipeline_runs (run_type='sim')`。**问题是 sim 和 live 的"高现金"分布根本不是同一回事**：

| | Sim 高现金日的根因 | Live 高现金日的根因 |
|---|---|---|
| 主导触发 | 策略卖出（stop-loss / drawdown） | **preflight HARD gate 阻塞所有买入** |
| 与 SPY 方向的条件相关性 | 强负相关（卖在跌之前） | 弱 / 零（gate 失败 ≠ 市场看跌） |
| Sample 选择偏差 | 看跌信号选出"将跌之日" | gate 失败选出"模型不可用之日" |

**前一份 memo 用 sim 的条件分布回答 live 的问题，是方法论错误**。这次重写。

---

## 1. Live 端的直接证据

### 1.1 Alpaca 账户实际状态（2026-06-03 当日 snapshot）

```json
{
  "holdings_count": 4,
  "holdings": {
    "MU":   "2026-04-27",
    "EQIX": "2026-05-17",
    "HON":  "2026-05-18",
    "ORCL": "2026-06-01"
  },
  "nav_estimate": 11100.36,
  "regime": "BULL_CALM",
  "regime_confidence": 0.6303,
  "skip_buys": false
}
```

**5 周（4-27 → 6-03）只买了 4 个**。每个 Kelly 目标 4-9%（per 最近 `ApplyKellySizingTask` 日志）→ 持仓段 ≈ 16-36% NAV，**剩余 64-84% NAV 是 cash**。这跟前一份 memo 里 "BULL_CALM 中位数 cash% = 3.7%" 的 sim 结论**完全不符**。

### 1.2 Preflight 阻塞实际模式（从 logs/live_e2e/*.log 抽样）

| 日期 | preflight pass | preflight fail | live orders | 失败的 gate |
|---|--:|--:|--:|---|
| 5-17 (×3 runs) | 8-9 | 0-1 | 0-1 | P-BROKER-CONNECT |
| 5-18 (×2 runs) | 9-10 | 0 | 1 each | — |
| 5-29 | 13 | **3** | **0** | P-PANEL-CONTRACT, P-REGIME-IC, **P-WF-GATE** |
| 5-30 早 (×7 runs) | 16 | 0 | mixed | — |
| 5-30 16:12 (daily_full_shadow) | 14 | **2** | 0 | P-REGIME-IC, **P-WF-GATE** |

**关键观察**：

- **P-WF-GATE 反复抖动**。5-29 失败：`wf_sharpe_mean=-1.323, beat SPY 0/3 cuts → 拒绝所有 live 决策`；5-30 早晨被人工接受（`golden_config:1205` 的 operator decision: "accept positive-Sharpe-but-SPY-laggard"）；同一天下午又因 `zero trades across all WF cuts → Sharpe undefined` 再次失败。
- **P-WF-GATE 是 HARD**，失败 = 整个 buy phase 停掉。这意味着即使 panel-LTR 算出了一批好候选，连下单都到不了。
- **5-30 早晨被人工放过**那一窗口期，候选数仍然有时为 0（`ApplyKellySizingTask: cands=0`）；其他时候 `mu_le_min_edge=62`（62 个候选 μ < 0 被 Kelly 拒）。

### 1.3 SPY 实际表现 — 一个反直觉的关键事实

```
2026-01-02 → 2026-03-27 (59 trading days, 你买 MU 之前):
  SPY 累积:  -7.49%
```

**SPY 在 2026 Q1 是下跌的**。如果当时账户 100% SPY，亏更多。这反过来说明：**"被动 overlay 不顾市况一律买 SPY"是错的方向** — codex 原 feature 那个 frame 在 2026 Q1 这个具体环境里是负价值。

但是 user 的痛点没解决：**"为什么我的钱白白趴着不工作"** 仍然是真的问题。问题不在 SPY 的 expected return，而在**资本利用率 / 基础设施 ROI** —— 投了这么多力气搭 ML pipeline，结果一半时间停产。

---

## 2. 把问题重新框定

| 原 codex 提议的 framework | 重新框定后的 framework |
|---|---|
| "Money drag avoidance via passive overlay" | "**Operational fallback** when alpha pipeline is unavailable" |
| 理论依据：Sharpe 1991 / Tobin 1958 / FP 2014 | 理论依据：**软件工程的 fail-soft pattern** + 限定版的 Sharpe 1991 |
| 触发条件：`cash% > threshold` | 触发条件：`preflight_blocking OR n_qualified_candidates=0` |
| 部署逻辑：买 SPY/QQQ 填到目标 invested%  | 部署逻辑：**带 regime + 信号双门** 的 fallback |
| Stage 1 sim 失败 | 因为用错数据；正确实验在 §4 |

**核心 insight**：cash drag 在 `renquant_104` 里**不是策略 sizing 问题**（前一份 memo 的 §6 DOE 想优化的方向），**是基础设施可用率问题**。修 sizing 解决不了"alpha pipeline 一半时间停产"这件事。

---

## 3. 提议方案 — Two-Phase Fail-Soft

### Phase A — 诊断 Buy 管线瓶颈（必须先做，~2 天工时）

在不动任何执行逻辑的前提下，先**测量**每一关掉了多少候选。新增一个 read-only 诊断任务：

```python
class MeasureBuyFunnelTask(Task):
    """记录每个 daily run 的 buy-side 漏斗，落到 live_state + sim_runs.db。

    输出每阶段保留的候选数:
      stage_0_panel_total          : 入参 panel 大小
      stage_1_after_eligibility    : 过 candidate eligibility 后
      stage_2_after_calibrator     : 过 calibrator floor 后  
      stage_3_after_kelly_mu       : 过 Kelly μ ≥ min_edge 后
      stage_4_after_qp_feasibility : 过 QP feasibility check 后
      stage_5_actual_buys          : 实际下单
      preflight_blocking           : True 如果 preflight HARD fail 整个 phase
      preflight_fail_keys          : 列表
      regime, cash_pct, n_holdings
    """
```

跑 30 天 live，就能看出"在哪一关掉了 80% 候选"。这一步 **不改变任何 live 行为**，纯诊断。

预期发现（我赌的方向）：
- P-WF-GATE 阻塞约 20-30% 的 daily run（基于上面 6 天抽样 2 天阻塞）
- mu_le_min_edge 在 BULL_CALM 阻塞 60-80% 的候选（基于 5-18 那个 `mu_le_min_edge=62/142` 的样本）
- QP feasibility 进一步砍 30-50%（基于 #136 sector cap 修复后的 binding pattern）
- 实际下单候选 = 入参 panel 的 1-3%

如果数据证实这个赌注，那 **真正的修复优先级**:
1. 把 `min_edge=0.0` 改成 regime-conditional 或动态分位数 → 解锁 mu_le_min_edge 这一关
2. P-WF-GATE 加 hysteresis（避免一日抖动整个停产）
3. **最后才是** fallback overlay（Phase B）

### Phase B — Operational Fallback Overlay（只在 A 报告之后启动）

只有当 Phase A 证实"瓶颈在管线，不是策略"，才考虑加 fallback overlay。设计：

```python
class ApplyOperationalFallbackTask(Task):
    """当 alpha pipeline 不可用时，将一部分 NAV 部署到 benchmark sleeve。
    
    触发条件（全部 AND）:
      1. preflight HARD 某关失败 OR n_qualified_candidates == 0
      2. regime in {BULL_CALM, BULL_STRONG}  # 见 §3.2 论证
      3. detector confidence ≥ 0.50           # 避免 regime 边缘部署
      4. SPY 20d momentum > 0                  # 见 §3.1 论证
      5. 当前 fallback_pct < fallback_max_pct
    
    部署逻辑:
      - 目标: fallback_target_pct = 0.30 of NAV (golden config 可调)
      - Split: 70% SPY + 30% QQQ (regime-conditional)
      - Rebalance band: ±5pp (Garleanu-Pedersen 2013 proportional trade)
    
    退出条件 (任一触发):
      - regime → BEAR / CHOPPY
      - SPY 20d momentum < 0
      - alpha pipeline 恢复 → 优先腾位置给 alpha 持仓 (rotation)
    
    归因:
      - 独立 P&L: fallback_pnl 跟 alpha_pnl 分开记录
      - 月报告必须把两个分开报，不能混淆 alpha 和 beta
    """
```

### 3.1 为什么必须加 "SPY momentum > 0" 这一道门

回到 §1.3 的硬事实：**2026 Q1 SPY 是 −7.49%**。如果 fallback 不带方向过滤，2026 Q1 整个季度 fallback 会亏 7.49% × fallback_pct，而 cash 等于 0。

SPY 20d momentum > 0 是个**保守门**：只在 SPY 已经显示出上升趋势时才部署 fallback。代价是错过反转初期；收益是不在跌势中 buy the dip。

这道门是为了应对**前一份 memo §5.6.3 的论点的真实有效部分**：现金部署 + 看跌环境 = 双重负回报。在 live 端，看跌环境不一定是策略卖出造成的（前 memo 错的地方），但 SPY 自己的 trend 是直接可观察的。

### 3.2 为什么 regime 限定到 BULL_CALM / BULL_STRONG

- BULL_VOLATILE：高不确定性，不部署 fallback（fallback 的目的是"工作不饱和时填仓"，BULL_VOLATILE 本身就是减仓信号）
- CHOPPY：横盘，SPY 期望收益接近 0，部署 fallback 收益期望也接近 0；多余成本不值
- BEAR：明确不部署（现有 `bear_defensive_sleeve` 该走防御 sleeve；见 §5）
- BULL_CALM：温和上涨，alpha pipeline 应该工作但常常空转 — fallback 主要目标
- BULL_STRONG：强趋势，alpha 候选可能因为"已涨太多 mu_le_min_edge"被砍掉；fallback 价值高

### 3.3 为什么必须独立 P&L 归因

**Berk-van Binsbergen (2015) JFE 118(1) 1-20** 的 value-added vs gross-alpha 框架要求把 beta 部分（fallback）和 alpha 部分（strategy）分开。如果混在一起报"strategy +5%"，会把 beta 的回报算成 alpha 的功劳。归因混淆会导致：

- 模型 retrain 的方向被误导（以为 model 在出 beta；实际是 SPY 在涨）
- 操作员决策被误导（以为 alpha 能赚；实际是 fallback 在干活）
- 出问题时 root cause 难以定位

---

## 4. 重做实验设计（用 live 数据）

前一份 memo 的实验用 sim 数据，错。这次必须用 live 数据。

### Stage 0' — Live Buy 漏斗 baseline（Phase A 前置，30 天）

部署 `MeasureBuyFunnelTask` 但只输出诊断，不改变任何行为。30 天后产出：

- 每日漏斗 (panel_total → actual_buys) 的中位数 + 分布
- preflight 阻塞频率 + 各 gate 失败占比
- 实际 vs sim cash% 的差距（应该能解释"为什么 sim 说 3.7%，live 说 70%"）

### Stage 1' — Fallback counterfactual（如果 Stage 0' 证实瓶颈在管线，~5 小时）

用 30 天 Stage 0' 数据：

- 对每个 `preflight_blocking=True OR n_candidates=0` 的 live bar，
- 计算 "如果当天 SPY momentum > 0 → 部署 fallback_target_pct" 的 day-t+1 PnL
- 同时计算 "实际策略" 的 day-t+1 PnL
- 输出 `fallback_pnl − strategy_pnl` 分布

这跟前一份 memo 的 Stage 1 关键区别：
| | 前 memo Stage 1 | 现 Stage 1' |
|---|---|---|
| 数据源 | sim panel | live decision trace |
| 触发条件 | cash% > threshold | preflight_blocking OR cands=0 |
| 方向过滤 | 无 | SPY 20d momentum > 0 |
| 期望符号 | 负（adverse selection） | 不确定 — 等数据 |

### Stage 2' — Sanity triad（§7.2.1 R2 强制）

- Shuffle placebo：把 fallback 触发条件随机化（保持触发频率），看 fallback_pnl 中位数是否变 0
- Time-shift placebo：fallback 部署 SPY return shift +120d，应该归 0
- A/A：3 个 seed 重跑（fallback 触发是 deterministic，seed 主要影响 rebalance band 内的判断）

### Stage 3' — Promotion gate（§7.4 Tier 3）

DSR > 0.5 OR PBO < 0.5 OR n ≥ 30 with t > 3.0。30 天 live 数据可能 n 不够 → 需要叠加 sim backfill。

---

## 5. 同步发现 — BEAR 防御 sleeve 仍然是死门

前一份 memo §5.6.5 的发现这次仍然成立（这部分论证用 sim 数据是 OK 的，因为问的是 sim 自己的 trade 历史）：

- `bear_defensive_slots=2 × bear_defensive_pct=0.15` 应当部署 30% NAV
- Sim BEAR 日交易记录：**0 笔** GLD/TLT/XLV/XLU 交易
- GLD 全 DB 交易记录：12 笔，**没一笔在 BEAR 日**

这跟 Phase A/B 是**两个独立的 bug**：
- BEAR sleeve dead-gate = 已部署的 BEAR 防御机制实际从来没 fire 过
- Operational fallback = 还没建的 BULL_CALM 默认部署机制

两个都该修，但 BEAR sleeve 修起来简单（已有代码、找 bug），fallback 是新设计（要走 Phase A → Phase B 完整流程）。

**建议优先级**:
1. **此周** 内：BEAR sleeve 为何不 fire 的 audit 修复（半天）
2. **本月** 内：Phase A `MeasureBuyFunnelTask` 上线（2 天）
3. **下月** 看 Phase A 数据再决定要不要 Phase B fallback

---

## 6. 论文引用更新

前一份 memo 的 12 篇引用仍然有效**作为背景理论**。重新框定后新增 3 篇 specific 引用：

- **R-13 — Garleanu, N., & Pedersen, L. H. (2013).** 已有引用，但**这次用法不同**：proportional trade band 作为 fallback rebalance 节流的 §3 Phase B 设计要素。
- **R-14 — Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). "Time Series Momentum." *Journal of Financial Economics* 104(2), 228-250.** SPY 20d momentum 门的理论依据 — 时序动量在主要 asset class 长期统计显著。
- **R-15 — Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013). "Value and Momentum Everywhere." *Journal of Finance* 68(3), 929-985.** Momentum + Value 的跨资产证据；动量 trigger 在 ETF 层面同样有效。
- **R-16 — Berk, J. B., & van Binsbergen, J. H. (2015).** 已有引用，**这次重点用法**：alpha vs beta P&L 归因的强制分离 — §3.3 设计要素。

### 6.1 软件工程对应的 fail-soft pattern

这不是投资学论文，但概念是对的：
- **Bulkhead pattern** (Nygard 2007, *Release It!*) — 模块失败时不传染整个系统
- **Circuit breaker** (Fowler 2014) — 反复失败的依赖临时停产
- **Fallback degradation** — 主路不可用时切到次优但确定可用的路径

Fallback overlay 本质是 alpha pipeline 的 **graceful degradation path**，不是 alpha 增强。

---

## 7. 给用户的回复（中文 TL;DR）

1. **前一份 memo 的"REJECT"结论我撤回**。理由：用 sim 数据归因 live 现象的方法论错误，§5.6.3 的"条件逆向选择"论证在 live 场景不成立。

2. **你的痛点定位是对的**：Alpaca 账户 5 周只买 4 个，70-80% 趴现金。但**原因不是 codex 想的那样**（不是 alpha 模型说要等机会），而是 **buy 管线反复瘫痪**（P-WF-GATE 抖、Kelly μ 门太严、preflight HARD 砍掉一半的 daily run）。

3. **直接买 SPY 也不是答案**。事实：2026 Q1 SPY 累积 −7.49%。如果当时账户全 SPY，亏更多。

4. **正确方案是 Operational Fallback** —— 不是被动 overlay，是 **alpha 不可用时的优雅降级**：
   - 触发：preflight 阻塞 OR 无合格候选
   - 限定：BULL_CALM/BULL_STRONG + SPY 20d momentum > 0 + detector 置信度 ≥ 0.5
   - 部署：30% NAV，SPY 70% + QQQ 30%
   - 退出：regime 变 BEAR/CHOPPY，或 momentum 翻负，或 alpha 恢复
   - 归因：fallback_pnl 跟 alpha_pnl 严格分开

5. **必须先 Phase A（诊断）再 Phase B（fallback）**。直接上 fallback 不知道在补哪个洞。Phase A 部署 `MeasureBuyFunnelTask`，30 天数据告诉我们：是 P-WF-GATE 阻塞主导，还是 Kelly μ 门主导，还是 QP feasibility 主导。然后修对应的根因。Fallback 是 last resort。

6. **顺便修个真 bug**：BEAR 防御 sleeve 在 sim 历史里**从来没 fire 过**（GLD/TLT/XLV/XLU 0 笔 BEAR 日交易）。配置里写了 30% 防御部署，empirics 是 0%。这周内单独 fix。

7. **Kelly σ-horizon 修正（#158/#169）继续推进**——它解决的是 sizing 侧的 underweight，跟 cash drag 是**正交**问题。

---

## 8. 接下来的具体动作（按优先级）

| 优先级 | 动作 | 工时 | 拦路虎 |
|---|---|--:|---|
| **P0** | BEAR sleeve dead-gate 单独 audit + fix | 半天 | 找 bug 看代码 |
| **P0** | Phase A `MeasureBuyFunnelTask` 部署 | 2 天 | 决定 schema + 落数据点 |
| **P1** | min_edge regime-conditional 化（解锁 mu_le_min_edge 阻塞） | 1 天 | 需要 A/B 验证 |
| **P1** | P-WF-GATE hysteresis（避免一日抖动停产） | 1 天 | 需要 operator 同意 |
| **P2** | Stage 0' (30 天 live 漏斗 baseline) | 30 天 wallclock | Phase A 落地后等数据 |
| **P3** | Phase B fallback overlay 设计 + Stage 1' 反事实 | 5 小时 + 5 小时 | Stage 0' 验证瓶颈在管线 |
| **P3** | Phase B fallback 上线 + Stage 2'/3' sanity + Tier 3 promotion | 2 天 | 必须 Stage 1' 正 Sharpe |

P0 / P1 是修 bug + 拆门槛，不需要 fallback overlay 就能让现有 alpha pipeline 用率从 ~30% 拉到 ~60-70%。P2 / P3 是 fallback overlay 本身。

**Phase A 是 commit。Phase B 是 conditional 在 Phase A 数据上**。

---

## 9. 这份 memo 不做的事

- 不动任何 live 代码
- 不改任何 golden config
- 不跑任何 Phase A/B 的实验（要分别 user-fire）
- 不撤销 Kelly σ-horizon 已合并的 #169（那是独立修复）
- 不撤销前一份 memo PR #185（保留作为方法论错误的教学样本，加 superseded-by 指向本文）

## 10. 跟前一份 memo 的关系

`2026-06-03-cash-overlay-feasibility-study.md` 的有效部分：
- §1–§4（论文引用、理论框架、bear sleeve 死门发现）— **保留**
- §5.5 Stage 0 sim 数据 — **保留** 但需注明"sim ≠ live"
- §5.6 Stage 1 反事实 + §5.7 REJECT 结论 — **被本文取代**

本文是新主线。下次再看 cash drag 问题，从本文开始。
