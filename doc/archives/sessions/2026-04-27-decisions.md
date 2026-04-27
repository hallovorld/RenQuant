# 2026-04-27 — 关键决策与发现汇总

本文档汇总 2026-04-27 全天的关键实验结论、架构决策和 Roadmap 调整。
下次开启新会话时直接读此文件可跳过所有背景解释。

---

## 项目当前状态（截至 2026-04-27）

| 维度 | 数值 |
|------|------|
| **生产模型** | XGBoost rank:pairwise，28 个特征，无 macro |
| **OOS IC (CPCV)** | +0.0482（CPCV 轻微虚高，诚实估计 0.032–0.040） |
| **实盘账户** | Alpaca ~$10k；持仓 PLTR / CAT / AMZN / GOOG / XLU |
| **真实回撤** | ~0.12% |
| **Watchlist** | 99 ticker；面板 103 ticker |
| **面板规模** | ~75k rows（99 ticker × 753 dates） |

---

## 已确认关闭的方向：Macro

### 结论：当前 breadth（99 ticker）下 macro 信噪比为负

**实验数据（单调递减）：**

| 配置 | OOS IC | Δ vs no-macro |
|------|--------|---------------|
| no-macro（生产） | **+0.0482** | — |
| + 11 ETF macro v1 broadcast | +0.0393 | −18% |
| + 11 ETF macro v1 + LightGBM | +0.0224 | −53% |
| + 30 ETF macro | +0.0370 | −23% |
| + 30 ETF + 22 FRED macro | +0.0344 | −29% |

**根本原因（v1 broadcast）：** 同一天所有 ticker macro 值相同 → within-date variance = 0 → cross-sectional rank loss 接收零梯度。macro 列实质上是噪声。

**Macro v2 per-ticker β 修复了 3 个 bug：**
- Bug1：broadcast gate 逻辑错误
- Bug2：训练与推理路径不对称（已由 Bug #25 修复，commit `e436fd9`）
- Bug3：β 单位错配

**v2 修复后结论：** OOS IC +0.0371（−23% vs PROD）——仍低于基线。架构上的问题（per-ticker β 对 cross-sectional rank 的信息增量在 99-ticker panel 下不足以补偿引入噪声的代价）无法通过 bug 修复解决。

**操作决策：**
- macro 代码路径全部关闭（gate = false）
- 代码已写入、测试已补充，不再需要重新实验
- 等 watchlist 扩展到 200+ ticker 后重新评估（Grinold-Kahn：`IR ∝ IC × √breadth`，200 ticker ≈ +42% IR ceiling）
- **下次会话不需要再解释 macro 为什么关闭**

---

## 已确认有效的方向：T2-2 Asset Embeddings

### 结论：embedding 携带独立 alpha 信号，与传统特征正交

**实验状态：** 16-D InfoNCE embedding 已训练完成。
**Artifact 路径：** `backtesting/renquant_104/artifacts/asset-embeddings.json`
**设计文档：** `doc/components/asset-embeddings-design.md`
**代码模块：** `training_panel/asset_embeddings.py`（Phase A 已完成）

**关键实验结果：**

| 指标 | 数值 |
|------|------|
| Raw IC（16 维平均） | 0.0147（9/16 维超 0.01） |
| Residual IC（去除动量 / 波动率后） | **0.0356**（16/16 维超 0.01） |
| OLS A/B 线性 IC 提升 | **+26%** |

Residual IC 显著高于 Raw IC → embedding 与传统特征正交，携带**独立 alpha 信号**，而非只是动量/波动率的重新包装。

**下一步（执行阶段）：**
1. 在 `strategy_config.json` 启用：`panel_ltr.asset_embeddings.enabled: true`
2. 重跑 `PanelModelJob`（`scripts/train_104.py --skip-baseline --skip-tournament --force`）
3. 预期 OOS IC 提升至 **0.043–0.047**（+5%～+14%）

---

## 已确认的架构问题：Sim / Backtest 污染（不影响当前模型评估）

**问题：** 静态 artifact 模型训练窗口覆盖整个 sim 期。
`live_train_end = 2026-04-17`，sim 跑 `2024-01-01 → 2026-03-26`
→ 整个 28 个月"OOS"窗口都在训练集内 → **sim 所有 APY/Sharpe 数字为纯 in-sample**。

**直接表现：** 2026 年 IC = 0.43（真实 IC 的约 10 倍）。这是回测基础设施问题，**不是模型质量问题**。

**当前阶段决策：**
- **CPCV OOS IC 是当前工作基准，可以信任**（轻微虚高不影响相对比较）
- 模型 A/B 比较用 CPCV IC 做决策是合理的
- 真正的 OOS 验证等 2026-04-22 之后 live 数据积累后再处理（Roadmap P0 B1/B2/B3）
- **下次会话不需要每次解释 sim backtest 泄漏——那是另一个独立问题，当前阶段不处理**

---

## 已修复的操作 bug（本日之前已完成）

### DrawdownCircuit HWM 错误
- **Bug：** 使用 `initial_cash=100k` 作为 HWM，而 Alpaca 账户只有 $10k → 算出 89.9% 假回撤 → 触发锁定
- **修复：** `resolve_hwm()` 已修复（commit `ab1006d`）
- **状态：** ✅ 已修复

### Plan O：防御型 ETF 在 BULL_VOLATILE 被错误买入
- **Bug：** XLU 等防御型标的在 BULL_VOLATILE regime 被买入
- **修复：** 非 BEAR 时跳过防御型标的（commit `52bf718`）
- **状态：** ✅ 已修复，+10 regression tests

---

## Roadmap 优先级（2026-04-27 确认）

| 优先级 | 项目 | 状态 | 说明 |
|--------|------|------|------|
| **1（当前）** | T2-2 Asset Embeddings 接入 → 重训 | 🟡 代码 ready，执行阶段 | embedding artifact 已在，只需启用 + 重训 |
| **2** | Watchlist 扩展 99→200 | 🔴 未开始 | Grinold-Kahn breadth 效益，+42% IR ceiling |
| **3** | T2-3 Regime Ensemble | 🔴 等待数据 | 等面板行数 > 150k；当前 GMM 不稳定 + 样本量不足 |
| **4** | 真正 OOS Backtest 基础设施修复（B1-B3） | 🔴 等待数据 | 等 live 运行数据积累后再处理 |
| **关闭** | Macro v3（更多 ETF / FRED） | ❌ 关闭 | 实验证明单调递减 |
| **关闭** | T2-4 Boyd Rotation | ❌ 低优先级 | 当前 IC 不够高时 rotation 每次 −2.5 APY |
| **关闭** | T2-1 LightGBM 替换 | ❌ 已拒绝 | 在当前面板 −60% IC（REJECTED 2026-04-27） |

---

## 关于评估指标的共识（避免下次重复解释）

- **CPCV OOS IC** 是当前工作基准，可以信任
- 模型 A/B 比较用 CPCV IC 做决策是合理的；轻微虚高对相对比较无影响
- **不需要每次讨论 sim backtest 泄漏问题**——那是 Roadmap P0 B1-B3 的独立任务，当前阶段优先级低于模型质量提升
- 真正的 OOS 验证 = live 数据积累（2026-04-22+ 之后的交易记录）

---

## 交叉引用

| 文档 | 内容 |
|------|------|
| `doc/components/asset-embeddings-design.md` | T2-2 完整设计 + Phase A-C 实现计划 |
| `doc/components/macro-factor-frame-redesign.md` | Macro v2 per-ticker β 设计 + 实验结论 |
| `doc/components/t2-4-and-macro-v2-deep-audit-2026-04-27.md` | 21 个 bug 详细审计（含 macro v2 3 个关键 bug） |
| `doc/components/lgbm-deep-audit-2026-04-27.md` | LightGBM 12 个 bug（T2-1 拒绝的根本原因） |
| `doc/components/full-training-deep-audit-2026-04-27.md` | 训练流水线全面审计 |
| `doc/research/macro-data-expansion-plan-2026-04-27.md` | Macro Tier 2 扩展计划（已关闭） |
| `doc/experiments/ab-journal.md` | 所有 A/B 实验记录 |
| `doc/roadmap.md` | 当前 Roadmap（权威来源） |
