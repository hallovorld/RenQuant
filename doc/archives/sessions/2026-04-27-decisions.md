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

---

## 🌙 2026-04-27 晚间补充 — embeddings/macro 决定性 A/B + 生产事故修复

> 上面（白天部分）写的"OOS IC 0.0482"是 **stale**。当天晚间用修过 bug 的 code + rank:pairwise 重训复现，**真正的 baseline OOS IC = +0.0418**（15-fold CPCV）。

### 1. 今天盘中 0 buy 的根因：NGBoost feature drift（生产事故）

**现象：** 2026-04-27 daily_104 跑出 41/103 in-universe → 10 candidates → **0 selected**，唯一成交是 `qp_sell TSM 3 shares`。

**直接症状：** 10 candidates 全部被 `quality_floor:gate_b` 砍（edge_sharpe < 0.10 阈值）。Top 4：JNJ +0.078 / APP +0.041 / LMT +0.036 / GLD +0.031。

**根因（数字证据）：**
- 部署的 `ngboost-head.json` 是某次 macro v3 实验的产物，含 **184 feature_cols**
- 当前 panel pipeline 只产出 **28 cols**（macro v3 关闭后不再有 vxx_*/hyg_*/dgs10_*/cpiaucsl_*/... 等 156 列）
- ApplyNGBoostTask 默认行为：缺失列**静默零填充**（"z-scored neutral"）→ **84.8% 输入是 0**
- σ 预测彻底是噪声 → `edge_sharpe = μ/σ` 全部被压扁到 < 0.10

**修复（commit `68b1c03` + `63e8aee`，已 push）：**
1. 用当前 code 重训 panel-LTR + NGBoost head：
   - `panel-ltr.json`：27 features，OOS IC = **+0.0400**（vs baseline +0.0418，within noise）
   - `ngboost-head.json`：27/27 feature_cols 完美对齐（0 drift）
   - `panel-rank-calibration.json`：22:41 同步刷新
2. config 漂移回退：
   - `panel_ltr.xgb_params.objective`: `rank:ndcg` → `rank:pairwise`（chat 改了 ndcg 但配套 hypers eta=0.05/num_boost=800/early_stop=100 从未落下，残废 ndcg 跑出 OOS IC=0.005）
   - `panel_ltr.asset_embeddings.enabled`: `true` → `false`（见下面 A/B 决议）
   - `strategy_config.golden.json` 同步 `strategy_config.json`
3. **NGBoost feature drift detector（防再犯）：** ApplyNGBoostTask 现在硬阈值检查，缺失列 > `ngboost.max_feature_drift_pct`（默认 5%）就 SKIP NGBoost 而非静默零填充，并 log.error 引导重训命令。

**端到端验证（V3 smoke 22:51）：**

| | PRE-FIX (14:05 daily) | POST-FIX (22:51 smoke) |
|---|---|---|
| candidates | 10 | 11 |
| Gate B rejected | **10 / 10** | **8 / 11** |
| selected | **0** | **3** |
| top edge_sharpe | JNJ +0.078 | **SMCI +0.1928** |
| #2 | APP +0.041 | FTNT +0.1180 |
| #3 | LMT +0.036 | NET +0.1068 |
| ApplyNGBoostTask warning | "missing cols [...140 cols]" | clean (0 missing) |

3 个票首次过 Gate B（SMCI / FTNT / NET），σ 分布回到合理区间。**修复端到端可观测有效。**

### 2. T2-2 Asset Embeddings — DEFINITIVE NEGATIVE

之前 dispatch agent 给的 GO 判断（基于 OLS A/B + per-feature IC 单变量正交，预期 OOS IC 0.043–0.047）在 XGB rank 树模型下不成立。**重训 + 修 2 个 bug 后实测：**

| Bug 修过的 | 内容 |
|---|---|
| 1 | `pd.Timestamp.utcnow()` 是 tz-aware；OHLCV parquet index 是 tz-naive → `<=` 比较 TypeError。strip tz |
| 2 | `LocalStore` 默认 `backtesting/renquant_104/data/ohlcv/` 只有 56/104 ticker → 加 fallback 到 repo root `data/ohlcv/`（114 ticker） |

修后重训 embeddings：104 ticker 全覆盖，loss=0.21，无 collapse。

**Paired CPCV A/B（15 folds，所有臂 rank:pairwise）：**

| Arm | Setup | OOS IC | Δ vs A | paired t | 决策 |
|---|---|---|---|---|---|
| A | 27 features，0 emb | **+0.0418** | — | — | baseline |
| B | + 16D embeddings (104 ticker 全覆盖) | +0.0341 | **−18.5%** | −1.45 | **NO-GO** |

每个 emb_i 单独的 IC 在树模型 splits 里看着健康（emb_5 IC=+0.0364，emb_6 +0.0258），但**作为 16 维新特征整体加入后，模型 OOS 反而退化** —— 树模型的 noise-variable detection 不如线性模型，容易在 16 维上过拟合 CV training fold。

`asset_embeddings.enabled=false` 已落到 production config。`asset-embeddings.json` 留在 artifacts 里，将来 watchlist 扩大或换 backend 后可重新评估，**当前不接入**。

### 3. Macro-as-panel-row v4 — DEFINITIVE NEGATIVE（再加一票）

之前 macro 的 v1 (broadcast 零梯度) / v2 (per-ticker β −23%) / v3 (Tier 1+Tier 2 单调递减) 都被否决，理由总结为"broadcast 列加成对 panel-LTR 不友好"。今晚做了 v4：**macro 不再作为 broadcast feature，而是直接作为 panel rows**（把 GLD/TLT/XLU/XLV/XLE/XLF/XLI/XLK/XLY 共 8 个 ETF 加进 watchlist，让它们跟 stocks 一样被 cross-sectional ranker 排）。

| Arm | OOS IC | Δ vs A | t | 决策 |
|---|---|---|---|---|
| C (watchlist + 8 macro panel rows + emb) | +0.0298 | **−28.8%** | −1.98 | **NO-GO** |

`best_iter` 只有 4（A=19，B=24），跟早些时候 rank:ndcg 崩盘的特征一样 —— 说明 ETF 行的 forward-return 分布跟 stock 行差太多，rank:pairwise 早停直接放弃学习。**Macro 与 stock 在 panel ranker 里的混合是结构性问题，不是哪个 bug 能修的。**

四代 macro 实验全部否决：v1（broadcast 零梯度）→ v2（per-ticker β −23%）→ v3（30 ETF + 22 FRED 单调递减）→ v4（panel rows −29%）。**等 watchlist 扩到 200+ 再重评**。

### 4. 还没修但已诊断的次级问题

| # | 问题 | 路径 | 状态 |
|---|---|---|---|
| E1 | 57/103 票被 universe_floor 卡 + NVDA/AMD per-ticker 模型说 "hold"（NVDA 选中 QLearning sharpe=1.46，AMD 选中 Manual rules sharpe=1.05）→ Panel-LTR 看不到这些票 | 写了 `scripts/ab_bypass_ticker_gate.py`，sim 验证 `bypass_ticker_gate=true` 是否提升 APY | **待跑 sim 决策** |
| E2 | Watchlist 99→200 breadth 扩展（+42% IR ceiling 是当前最有希望的 lever） | 调研中 | 待启动 |
| E3 | NVDA / LITE / COHR 缺真正的 XGB 模型；AMD 的 XGB 是 Apr 13 老的；这些 ticker 的 per-ticker tournament 选了非 XGB 模型 | 如果 E1 通过则不需要修；否则要重训 | 待 E1 决定 |

### 5. 备份层级（"保护好模型和 golden conf"）

5 层防护：

1. **read-only 文件权限** (`chmod 444`) on bak + checkpoint
2. **Local immutable checkpoint dir** `backtesting/renquant_104/artifacts/checkpoint_2026-04-27_22h28/` + SHA256 manifest + RESTORE.md
3. **Local pre-fix bak files** `panel-ltr.pre-fix-2026-04-27.bak.json` + `ngboost-head.pre-fix-2026-04-27.bak.json`（可回滚到出 bug 状态做 forensics）
4. **Git local commit** `68b1c03` + `63e8aee`
5. **Git remote** github.com/hallovorld/RenQuant push 完成

### 6. 下一会话直接读这一段

**生产模型（截至 2026-04-27 22:28）：**
- Panel-LTR：rank:pairwise，27 features，无 emb，无 macro，OOS IC = +0.0400
- NGBoost head：27 features，跟 panel 完美对齐
- Asset embeddings 关闭（实测 −18.5%）
- Macro 关闭（v1-v4 全 NEGATIVE）
- Drift detector 已上线（5% 阈值硬失败）

**优先顺序：**
1. ~~V3 inference smoke 验证 edge_sharpe 真的恢复~~ ✅ 22:51 PASS（3 candidates 过 Gate B）
2. E1 `bypass_ticker_gate=true` sim A/B → NVDA/AMD 解决方案
3. E2 watchlist 99→200（最有希望的 lever）
4. T2-3 Regime Ensemble（等面板 > 150k rows）
