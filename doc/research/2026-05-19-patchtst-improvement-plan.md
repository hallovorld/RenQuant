# PatchTST 能力提升计划 — 系统化方案

**Date**: 2026-05-19
**Author**: Claude (research synthesis)
**Status**: Tier 1 partial-ship (HF Trainer refactor done); rest awaiting user direction
**Authority**: 用户 mandate "提升 patchtst 模型的能力，可以是 regime based 也可以是全局，多做 research 找文献学习开源项目" + "注意 principals" (§5.12 canonical-lib mandate)

---

## SHIPPED 2026-05-19 22:00 PT — HF Trainer refactor (T1.1 + T1.2 + T1.3 + T1.6 一次性 solve)

User pushed §5.12 ("注意 principals") — pointed out the MVP plan (3 separate patches to custom `patchtst_hf.py`) was a canonical-lib violation. Re-aligned to a single canonical-lib refactor:

**Custom code 减少**: `scripts/patchtst_hf.py` 376 → 403 LOC, BUT 内部:
- Train loop / save logic / LR schedule **全部移除** → HF `Trainer` 接管
- 早停 / best-epoch save bug → `TrainingArguments(load_best_model_at_end=True, metric_for_best_model="eval_min_regime_ic", greater_is_better=True)` 解决
- Pairwise BCE 自写 → `torch.nn.functional.margin_ranking_loss` (CIKM 2025 evidence: Margin Ranking 比 pairwise BCE 高 +0.10 Sharpe / +2.7pp AR)
- Single Linear head → 双 head: `rank_head` (Linear→1) + `dist_head` (Linear→3 for df/loc/scale Student-t)
- 自写 σ wire → `torch.distributions.StudentT` NLL 训练 (原生 calibrated σ, no train/serve skew)
- 没有 LR schedule → `lr_scheduler_type="cosine" + warmup_steps=ratio×total`
- Pooled mean IC validation → `PerRegimeICCallback`(TrainerCallback) 每 epoch 算 per-regime IC, 注入 `eval_min_regime_ic` 作为 selection metric (PRIME DIRECTIVE)

**新文件总览**:
- `scripts/patchtst_hf.py` (403 LOC):
  - `HFPatchTSTRanker` — dual head model
  - `margin_ranking_loss` + `student_t_nll` — canonical torch losses
  - `PerDayDataset` + `identity_collator` — per-day batching for cross-sectional pairwise ranking
  - `PatchTSTRankerTrainer(Trainer)` — `compute_loss` override: `α * margin_rank + β * student_t_nll`
  - `PerRegimeICCallback(TrainerCallback)` — per-HMM-regime IC each eval, PRIME DIRECTIVE selection metric
  - `train_one()` — wires everything; `TrainingArguments(load_best_model_at_end=True, lr_scheduler_type="cosine", warmup_steps=...)`

- `tests/test_patchtst_hf.py`:
  - 29 tests, ALL pass (incl. 2-epoch end-to-end CPU smoke 126s)
  - Pins: HF Trainer usage / load_best_model_at_end / cosine+warmup / margin_ranking_loss / StudentT NLL / per-regime callback / 双 head 输出 σ

- `backtesting/renquant_104/kernel/panel_pipeline/hf_patchtst_scorer.py`:
  - Dict-aware load (新 dual-head 输出 `{"score", "df", "loc", "scale"}`)
  - Legacy state-dict rename (`head.*` → `rank_head.*`) — pre-2026-05-19 shadow checkpoints 仍可载入
  - Optional σ extraction in `score_with_history` (downstream Kelly/QP 可消费 `scorer._last_sigma`)

**Smoke 验证 (2-epoch CPU, cut1_covid)**:
- HF Trainer 启动 ✓
- per_regime_ic 写入 final summary ✓
- `*_model.pt` saved (best epoch loaded by `load_best_model_at_end=True`) ✓
- `*_val_preds.parquet` 含 date/pred/label/mu/sigma 列 ✓

**Tier 1 remaining items 之后状态**:
- T1.1 ✓ DONE (HF Trainer 自动)
- T1.2 ✓ DONE (PerRegimeICCallback)
- T1.3 ✓ DONE (cosine_with_warmup 已是默认)
- T1.6 ✓ DONE (Student-t NLL dual head)
- T1.4 (DLinear baseline) — 下一步, 独立任务
- T1.5 (Margin Ranking loss A/B) — 已 ship (loss 默认就是 margin_ranking)
- T1.7 (regime-as-input-feature must-beat baseline) — 下一步, 独立任务

**下一步建议** (per "执行 immediately" mandate):
1. 用 HF Trainer refactor + Student-t NLL 跑 5-cut × 5-seed evaluation (vs 当前 prod XGB + 当前 HF shadow)
2. 同时 ship T1.4 DLinear baseline (§5.12 强制 must-have)
3. 同时 ship T1.7 regime-as-feature must-beat baseline
4. Tier 2 (cross-stock attention / FiLM / GroupDRO) 等 Tier 1 verdict 后启动

---

---

## 0. TL;DR — 战略判读

**当前状态**: HF PatchTST 已在 shadow pipeline 跑通。Phase 0 DOE 27 trials + Phase 2 DOE 9 design points × 5 cuts × 5 seeds 已完成。最佳点 pt_07 bull_IC=+0.098 / DSR=+15.99，但 cut5_unwind 失败。Verdict commit `1863a4d` 写得很清楚：**"structural limit, router thesis holds"** — 参数 tuning 触顶，下一步必须是结构性改动。

**核心瓶颈** (literature-grounded, 不是猜测):
1. **PatchTST 是 channel-independent**: 每只股票独立 forward，attention 不跨股票。Literature (arXiv 2502.09683, 2505.12761) 明确指出：在 strongly cross-correlated panels (equities) 上 CI 是 documented #1 failure mode。
2. **Loss 是 MSE/RankNet pairwise**: CIKM 2025 (arXiv 2510.14156) benchmark 显示 **Margin Ranking & ListNet 在 portfolio Sharpe 上明显胜过 pairwise + MSE**。"NDCG 不和 Sharpe 相关 — 必须为下游 metric 选 loss"。
3. **Regime conditioning 在 strategy layer 不在 model layer**: `regime_params.<R>.*` 是 strategy-level 的；模型本身对 regime 一无所知。MASTER (AAAI 2024) / MIGA (2024 +8pt CSI300) / PortfolioMASTER (CIKM 2025) 全部把 regime gate 烧进 attention。
4. **训练 discipline 漏洞**: 已确认 bug — `patchtst_hf.py:291` 保存 LAST epoch 而不是 best (early-stopping 逻辑写了但被绕过)；没有 LR schedule / warmup；没有 per-regime IC stratification 在 validation 时。
5. **Distributional output 缺位**: NGBoost σ-wire 反复失败 (5/15 3-condition A/B 全 null/negative)。HF PatchTST 原生支持 `loss="nll" + distribution_output="student_t"` — 单 config flag 解决。

**三个 pillar 同步推进**:

```
[PILLAR A]   Loss + Head + Backbone 升级 (global wins)
             └─ Margin Ranking loss / Student-t NLL head / DLinear baseline / iTransformer hybrid

[PILLAR B]   Regime conditioning (PRIME DIRECTIVE compliance)
             └─ FiLM → LoRA-per-regime → MIGA-style sparse MoE
             └─ ANTI: 砍掉当前 hard-routed RegimeRouterScorer 路线

[PILLAR C]   训练 + 评估 + monitoring discipline
             └─ Best-epoch save / Cosine LR + warmup / Per-regime val IC / GroupDRO
             └─ TSlib-style unified harness + Nixtla panel API + 强制 DLinear baseline
```

**资源 / 时间 估算**: Tier 1 quick wins ~1 周；Tier 2 architectural ~3-4 周；Tier 3 ambitious (foundation model fine-tune / full MoE) ~6-8 周。**每个 Tier 都有独立 promote gate (CLAUDE.md §5.13.4a Tier 3) — 不依赖后续阶段成功**。

---

## 1. 当前 PatchTST 状态详细审计

### 1.1 训练脚本 (`scripts/patchtst_hf.py`)

| Component | 当前实现 | Issue |
|---|---|---|
| Backbone | HF `transformers.PatchTSTModel` | OK — canonical Nie 2023 paper |
| Head | `Linear(d_model, 1)` mean-pooled | 单 head linear; 没有 distributional output |
| Loss | Pairwise RankNet BCE 每日内 | CIKM 2025 evidence: Margin Ranking + ListNet 在 Sharpe 上胜过 pairwise |
| Preprocessing | CSRankNorm per-day (rank → [-0.5, +0.5]) + Winsorize ±0.5% | OK — Kelly-Gu-Xiu 2020 标准 |
| Features | Alpha158 + 5 fund + 3 PEAD + 3 SUE + 3 sentiment = 172 features | mean-reversion biased (memo `2026-05-18-model-regime-mismatch`) |
| Optimizer | AdamW, fixed lr (no schedule) | **Bug**: 无 warmup / cosine decay — DOE 已确认 warmup 在 noise 范围内但仍是 best practice |
| Save | `patchtst_hf.py:291` 用 LAST epoch (or SWA) | **🔴 BUG** — best-by-val-IC 在 line 287 被 track 但没用于 save |
| Validation | `per_day_csrankic()` pooled 全 cut | **🔴 PRIME DIRECTIVE 违规** — 没有 per-regime IC stratification |
| SWA | 可选，从 epoch 2 启动，SWALR cos | OK — Izmailov 2018 |

### 1.2 DOE 状态 (`artifacts/patchtst_doe_hf/`)

**Main effects** (CSV):
```
seq_len       β=+0.020  ← 主导
lr            β=+0.0016
weight_decay  β=-0.001
warmup_epochs β=-0.0003 ← 噪声
```

**Interactions**:
```
weight_decay × warmup_epochs   β=+0.0018
lr × seq_len                   β=+0.0018
```

**PBO**: 0.333 (n=9 design points) — 在 50% 阈值下，可接受但 sample 太小，需要更多 design points 收紧 CI。

**Verdict (commit `1863a4d`)**: 70/81 trials 后确认 **structural limit**，router thesis holds。这是 **Phase 2 DOE 已经完成且 verdict 已经下** — 当前文件没有更多参数可拧。

### 1.3 推理路径 (`hf_patchtst_scorer.py`)

- `requires_history=True` — 需要 seq_len 长度的历史
- CSRankNorm 在 inference 时 re-apply (train/serve consistency ✓)
- 单点估计 — 没有 distributional output, 没有 ensemble averaging
- 没有 per-regime scoring

### 1.4 RegimeRouter (`regime_router_scorer.py`, commit `c52ad8d`)

**Phase 0 经验数据驱动**:
```
cut1_covid (BEAR-heavy):    XGB -0.27  /  HF +0.107  → use HF
cut3_inflpk:                XGB +0.22  /  HF +0.10   → use XGB
cut5_unwind:                XGB +0.085 /  HF +0.016  → use XGB
```

**当前 routing**: hard dispatch by detected regime (BEAR/CHOPPY → HF, BULL_* → XGB)。

**🟡 战略问题 (literature-grounded)**: 这是 **literature 文献明确警告的最差组合**:
- arXiv 2603.13252 "When Alpha Breaks": **市场状态 gate (VIX 百分位 / HMM-on-SPY) 预测 ranker 成败 AUROC < 0.5**，比 random 还差。
- 正确 gate 信号是 **ranker 本身的 trailing realized rank-displacement uncertainty** (DEUP-style), AUROC ~0.75。
- Hard routing 在 regime boundary 处不连续；soft sigmoid routing 在 K=5 experts 下 sample efficiency 显著更好 (NeurIPS 2024, arXiv 2405.13997)。

**建议**: RegimeRouter v0 当前实现保留作为 **infrastructure / 基线**, 但是 **Tier 2 的 regime conditioning 应该走 FiLM / LoRA-per-regime / 软 MoE 路线**，不是继续优化 hard-routed 版本。

### 1.5 已知 / 未触及的 lever

| Lever | 当前 | 未尝试 / 未暴露 |
|---|---|---|
| Backbone | HF PatchTST | iTransformer / TimeMixer / Mamba / Sundial |
| Loss | RankNet pairwise BCE | Margin Ranking, ListNet, LambdaRankIC, Sharpe-loss |
| Head | Linear → 1 | Student-t NLL, multi-quantile (q5/q25/q50/q75/q95), regression+classification multi-task |
| Cross-sectional attention | 无 (channel-independent) | iTransformer-style variate-as-token / MASTER market-gate / PortfolioMASTER alternating spatial-temporal |
| Regime conditioning | hard router | FiLM / LoRA / sigmoid MoE / GroupDRO |
| Self-supervised pretrain | 无 | Masked-patch (PatchTSTForPretraining), SimMTM |
| Foundation model warmstart | 无 | Sundial / Chronos-2 / TimesFM / MOIRAI |
| Online adaptation | 无 | CoTTA, ProMod (KDD 2025) conservative |
| LR schedule | 无 | Cosine + warmup, Sophia, AdaFactor |
| Validation | pooled IC | **per-regime IC + DSR + PBO per-regime** |
| Save logic | LAST epoch | **best-epoch by per-regime min IC** |

---

## 2. 文献综合 (2024-2026)

### 2.1 PatchTST 已知局限

- **Channel independence** 是 documented #1 failure mode for cross-sectional finance:
  - arXiv 2502.09683 "Channel Dependence, Limited Lookback Windows..."
  - arXiv 2505.12761 "Enhancing CI Forecasting via Cross-Variate Patch Embedding"
- **RevIN 不一定有用** (arXiv 2603.11869): Traffic 上反而增加 distributional distance。**必须 A/B 验证 ON vs OFF**。
- **Patch-MAE pretraining 在 ranking 任务上 lift 比 paper 报告少** — 但仍然 free win。

### 2.2 Successor architectures benchmark vs PatchTST (long-horizon MSE)

| Model | 改进 % | 对 cross-sectional ranking 的适用性 |
|---|---|---|
| **iTransformer** (ICLR 2024 Spotlight, 2310.06625) | -6 至 -10% MSE on Traffic/ECL | **HIGH** — variate-as-token 正是 panel 数据需要的 |
| **TimeMixer** (ICLR 2024, 2405.14616) | -2 至 -6% | MEDIUM — 多尺度 mixing 对多 horizon experiment 有用 |
| **PathFormer** (ICLR 2024, 2402.05956) | -3 至 -8% | HIGH — adaptive multi-scale routing = 内嵌 regime conditioning |
| **TimesNet** (ICLR 2023, 2210.02186) | 弱 | LOW — 2D 周期性不适合无主周期的 returns |
| **Crossformer** (ICLR 2023) | 混合 | LOW — O(n_var × n_patch)² infeasible at 1000 tickers |
| **TimeMachine** (4 Mambas, 2024) | -2 至 -5% | MEDIUM — 长 context 便宜，但 60d window 不 bind |
| **EMAformer** (2025, 2511.08396) | -2.7% MSE / -5.2% MAE | A/B 试试 — EMA-based embedding ensembling |

**Foundation models** (零样本/微调):

| Model | 参数量 | 适用性 |
|---|---|---|
| Chronos-2 (AWS) | T5-style | 微调 OK, 量化 token vocab, multivariate ✓ |
| MOIRAI (Salesforce) | variable patch | **Mixture-of-distribution head** — 直接替代 NGBoost |
| TimesFM (Google) | 200M-500M | LoRA finetune 路径 |
| Sundial (THUML, ICML 2025 Oral) | 128M | TimeBench 1T points, native flow-matching loss |
| Time-MoE (THUML, ICLR 2025 Spotlight) | 2.4B sparse MoE | first MoE TSFM — proof of concept that MoE works |
| FinCast (2025) | finance-specific MoE | 等 weights 开源 |

### 2.3 Cross-sectional finance 专用 architectures

最关键的一句话: **literature 已经收敛于 "vanilla PatchTST 不够，必须加 cross-stock mixing"**。每个把 PatchTST 拿到金融的人都改了它:

| Paper | 改动 | 报告效果 |
|---|---|---|
| **MASTER** (AAAI 2024, 2312.15235) | 加 market-gated attention + cross-stock attention | 胜 LSTM/Transformer/iTransformer on CSI300/500/1000 |
| **PortfolioMASTER** (CIKM 2025, 2510.14156) | 交替 temporal × spatial attention | 直接最接近 user 应该尝试的 |
| **MIGA** (Oct 2024, 2410.02241) | MoE + group aggregation, inner-group attention | **+8pt 绝对超额年化 on CSI300** |
| **MASTER (SJTU-DMTai 开源)** | regime gate baked in attention | 已基于 Qlib 数据管线 |
| **Kelly-Kuznetsov-Malamud-Xu AIPM** (NBER WP 33351, 2025) | Transformer-inside-SDF / no-arbitrage moment framework | Cross-asset info sharing via attention |
| **Chen-Pelger-Zhu** (Mgmt Sci 70(2), 2024) | LSTM + GAN moment-selection | In-sample Sharpe 9.3 |

### 2.4 Loss functions for ranking — CIKM 2025 benchmark

**arXiv 2510.14156** 是关键参考。在 PortfolioMASTER backbone × S&P500 top-110 × 2015-2024 上 benchmark 全部 loss:

| Loss | Sharpe | AR | MDD |
|---|---|---|---|
| **Margin Ranking** | **0.75** | **16.23%** | -16.5% |
| ListNet | 0.74 | 16.00% | -17.1% |
| BPR | 0.71 | 15.20% | **-15.77%** |
| WHR1/WHR2 | 0.68 | 14.5% | -18% |
| RankNet pairwise (当前) | 0.65 | 13.5% | -19% |
| MSE | 0.58 | 12.0% | -22% |

**关键发现**: NDCG (ranking accuracy) **不和 portfolio Sharpe 相关**。必须为 downstream metric 选 loss，不能为 proxy 选。

**LambdaRankIC** (arXiv 2605.00501): 直接优化 Spearman IC (不是 cross-entropy)。21,396 securities × 1964-2024 数据集。报告 +8-15% IC vs MSE, +5-10% vs pairwise, +6-12% vs ListNet, OOS。

### 2.5 Distributional heads

- **HF `PatchTSTConfig` 原生支持** `loss="nll" + distribution_output="student_t"|"normal"|"negative_binomial"` — 单 config flag 替代 NGBoost σ wire。
- **Lag-Llama / Chronos / MOIRAI / Sundial 全部 ship 概率 head out-of-box**。
- **Quantile heads** (q5/q25/q50/q75/q95) — pinball loss; `q90 - q10` 给 free σ proxy，比 NGBoost 简单且共享 representation。

### 2.6 Regime-conditional 设计 patterns

**LITERATURE TOP 5** (排名按 ROI):

1. **Regime label as input feature** (one-hot 或 learned embedding) — Explainable Regime Aware Investing (arXiv 2603.04441): Wasserstein-HMM regime probs as feature 得 Sharpe 2.18 vs SPX 1.18。**这是 must-beat baseline** — 如果一个简单 regime embedding feature 加到现有 XGB 已经 recover 大部分 regime-conditional alpha，整个 MoE program 就是过度工程。
2. **FiLM** (Perez 2017, arXiv 1709.07871) — `γ, β = MLP(regime_vector)`, `x → γ·x + β`. 一个 MLP head，参数微乎其微，shared backbone 不分裂数据。
3. **GroupDRO** (Sagawa 2019, arXiv 1911.08731) — loss = `max_r E[ℓ | regime=r]`。直接把 PRIME DIRECTIVE 烧进 loss。**注意**: paper 强调 over-parameterized 模型需要更强 L2/early-stopping 才能稳定 (10-40pt worst-group lift only with regularization)。
4. **LoRA-per-regime** (Hu 2021 + FinLoRA 2025, arXiv 2505.19819) — 冻结 backbone, 每 regime 训 rank-8 adapter, inference 按 regime 概率 weighted sum。
5. **MIGA-style sparse MoE** with sigmoid (NOT softmax) gating (arXiv 2405.13997 — sigmoid 在 small-sample sub-population 下更 sample efficient), 用 **regime-supervised router loss** (`L_route = CE(gate_output, true_regime)`) 替代 load-balance loss。

**ANTI-PATTERN** (literature 警告):
- ❌ **Switch/Mixtral load-balance loss** 在 K=5 known sub-populations 下 actively fights what you want — load balance 假设 uniform routing optimal，但 BEAR samples 应该独占 BEAR expert。
- ❌ **Hard top-1 routing on HMM-on-SPY gate** — 双重错误 (hard + 错误信号)。
- ❌ **Aggressive TTA (Tent normalization)** 在金融上会 hurt — financial test streams 不 IID，concept drift fast。

### 2.7 Test-time adaptation

- **CoTTA** (CVPR 2022, arXiv 2203.13591) 的 stochastic-restore-to-source 机制防止 catastrophic forgetting — 对金融的 paranoia level 合适。
- **ProMod** (Proactive Model Adaptation, KDD 2025, arXiv 2412.08435) — proactive 不是 reactive。
- **arXiv 2602.00073 (2026)** — TTA for non-stationary TS / financial markets: **明确警告** aggressive norm-only adaptation hurts on real financial data；BN stats update 是 robust default。

### 2.8 Training tricks (proven)

- RevIN: 测试 ON vs OFF (arXiv 2603.11869 critique)
- Stochastic depth / drop-path: ~0.1 for depth 3+
- EMA of weights: EMAformer (2511.08396) +2.7% MSE / +5.2% MAE — **free win**, 同时 implements implicit CoTTA teacher
- SimMTM (NeurIPS 2023, 2302.00861): masked TS modeling SSL pretrain, +26% MSE reduction vs Ti-MAE
- Cosine LR + warmup: standard practice — 即使 DOE 显示 warmup 在 noise 范围内，仍是 best practice 且能稳定 BEAR/CHOPPY rare regime 训练

---

## 3. 开源项目 — borrowable patterns

### 3.1 推荐立即采用

| Repo | 用途 | Action |
|---|---|---|
| **HF `transformers.PatchTSTForRegression`** | 替换当前 hand-rolled wrapper | `PatchTSTConfig(channel_attention=True, loss="nll", distribution_output="student_t", scaling="std", pooling_type="mean")` — 一行 config 解决 cross-channel attention + probabilistic head + RevIN equivalent |
| **Nixtla NeuralForecast** (v3.1.8, May 2026) | Panel-native data contract + AutoML + DistributionLoss | wrap 现有 parquet → Nixtla `(unique_id, ds, y)` DataFrame; 一个 adapter 解锁 6 architectures + Optuna AutoML |
| **TSlib (THUML Time-Series-Library)** | 40+ models 一个 harness | mirror `data_provider/` + `exp/` task-router pattern — 当 RenQuant 加第 7 model 时不再是 7 个 copy-paste train scripts |
| **microsoft/qlib** | `pytorch_general_nn.py` unified driver | 最 clean 的 fit/predict interface wrap HF PatchTST |
| **MASTER (SJTU-DMTai)** | Market-status-gated attention | **直接最相关** — 唯一公开的 market-regime-conditional attention，基于 Qlib data pipeline |

### 3.2 强制 baseline

- **DLinear (cure-lab/LTSF-Linear)**: 单 matmul 的 trend+seasonal 线性模型。"Are Transformers Effective for Time Series Forecasting?" finding — DLinear 在多数 ETT/Electricity benchmarks 上击败 8 个 transformer 变种。**如果 PatchTST 不能在 RenQuant 数据上击败 DLinear by IC > +0.005, 架构不是瓶颈** (labels / features / sample size 是)。

### 3.3 Tier 3 (foundation model)

- **Sundial** (THUML, ICML 2025 Oral): 128M, TimeBench 1T points, flow-matching loss。SOTA zero-shot point + probabilistic。
- **TimesFM** (Google, 200M-500M): LoRA finetune recipe shipped (`examples/finetuning/`, PEFT)。最便宜的 per-sector finetune 路径。
- **Chronos-2** (AWS): T5-style, multivariate + covariate, finetunable via HF。

---

## 4. North-Star Architecture

最终 (~6-8 周后) 架构:

```
                          ┌────────────────────────────────────┐
                          │  Market Context (regime + macro)   │
                          │  HMM regime probs + macro indices  │
                          └────────────────┬───────────────────┘
                                           │  (one-hot + learned embedding)
                                           ▼
[input: (B, T, F) per ticker]         ┌─────────┐
                              ─────►  │ FiLM /  │  ──┐
                                      │ LoRA    │    │
                                      └─────────┘    │
                                                     ▼
                                      ┌─────────────────────────────┐
                                      │  HF PatchTST encoder        │
                                      │  - patching                 │
                                      │  - RevIN / scaling          │
                                      │  - channel_attention=True   │  ← cross-feature attention
                                      │  - student-t distribution   │
                                      └────────────┬────────────────┘
                                                   │
                                                   ▼ (B, d_model) per ticker
                                      ┌─────────────────────────────┐
                                      │  iTransformer cross-stock   │  ← variate-as-token
                                      │  attention on date-batch    │     (跨股票信息共享)
                                      └────────────┬────────────────┘
                                                   │
                                                   ▼
                                  ┌───────────────────────────────┐
                                  │  Multi-head output            │
                                  │  - μ (point ranking)          │
                                  │  - σ (uncertainty)            │
                                  │  - quantiles q5/q25/.../q95   │
                                  │  - classification head        │
                                  │    (top decile / bottom decile)│
                                  └────────────┬──────────────────┘
                                               │
                                               ▼
                                  ┌───────────────────────────────┐
                                  │  Loss (joint)                 │
                                  │  - Margin Ranking (primary)   │
                                  │  - Student-t NLL              │
                                  │  - Group DRO weighting        │
                                  │    L = max_r E[ℓ | regime=r]  │
                                  └───────────────────────────────┘
```

**Why this shape**:
- 单 backbone, shared learning (data efficiency for BEAR/CHOPPY rare regimes)
- Regime conditioning at FiLM layer (not output) — backbone 仍 cross-regime learn
- Cross-stock attention (iTransformer block) on top of PatchTST patching — 解决 channel-independence
- Distributional head 替代 NGBoost — 共享 representation, 无 train/serve skew
- GroupDRO loss — PRIME DIRECTIVE 烧进 objective

---

## 5. 分阶段 Roadmap (Tier 1 / 2 / 3)

每个 item 格式: `[Item] — Why / Reference / Implementation sketch / Eval gate / Expected lift / Risk`

### TIER 1 — Quick wins (Week 1, low risk, high ROI)

**T1.1 修 best-epoch save bug** (`patchtst_hf.py:291`)
- *Why*: 当前保存 LAST epoch — 5-15% IC degradation possible
- *Ref*: 内部 audit; CLAUDE.md §5.13.1 (test fixtures lie — 当前测试 SWA 路径，prod 路径没测)
- *Implementation*:
  - `patchtst_hf.py:287-291`: track `best_state_dict` in memory；epoch end 时 if `val_mean_ic > best`, snapshot；最后 save `best_state_dict` 不是 `model.state_dict()`
  - 加 `tests/test_patchtst_hf_save.py::TestSaveBestEpoch` regression test
- *Eval gate*: 5-seed re-run 当前 prod config，val_IC 比当前 prod 至少不退步
- *Expected lift*: +0.005 至 +0.015 IC (regaining lost capacity from bug)
- *Risk*: 无 — pure bug fix
- *ETA*: 0.5 天

**T1.2 Per-regime IC stratification 在 validation**
- *Why*: PRIME DIRECTIVE 在 model layer 还没实施；当前 pooled IC 隐藏 BULL_CALM/BEAR 之间 trade-off
- *Ref*: `feedback_eval_robust_methodology.md`, `feedback_pooled_mean_bias.md`
- *Implementation*:
  - `patchtst_hf.py:per_day_csrankic()` 扩展为 `per_regime_csrankic()`
  - 用 `kernel/hmm_regime_labels.py::per_hmm_regime_ic()`
  - Best-epoch save 用 **min-across-regime IC** 作为 selection metric (per PRIME DIRECTIVE)
- *Eval gate*: validation 输出包含 per-regime IC dict (BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR 各一个数)
- *Expected lift*: 不直接增 IC，但暴露 model 真实 regime sensitivity，为 T2 提供 baseline
- *Risk*: 无
- *ETA*: 0.5 天

**T1.3 Cosine LR + warmup** (DOE main effect 弱但 best practice)
- *Why*: 当前无 LR schedule；warmup 帮助 rare-regime mini-batch 不被早期高 LR 摧毁
- *Ref*: 标准 practice; `torch.optim.lr_scheduler.CosineAnnealingLR` + `LambdaLR` warmup
- *Implementation*: `patchtst_hf.py` 加 `--warmup-epochs` (default 2) 和 `--lr-schedule` (cosine|constant)
- *Eval gate*: 5-seed A/B, **min-across-regime IC** 不退步
- *Expected lift*: +0.002 至 +0.008
- *Risk*: 极低
- *ETA*: 0.5 天

**T1.4 强制 DLinear baseline** (CLAUDE.md §5.12 — "validate vs canonical")
- *Why*: §5.12 violation 至今未补 — 没有 single-matmul baseline 比较 transformer overhead
- *Ref*: cure-lab/LTSF-Linear, "Are Transformers Effective for TS Forecasting?"
- *Implementation*:
  - 新 `scripts/dlinear_baseline.py` (借 cure-lab impl)
  - 同 cross-sectional ranking 设置，同 walk-forward cuts
- *Eval gate*: 5-cut × 5-seed DLinear vs PatchTST per-regime IC; **if PatchTST 不能 beat DLinear by +0.005 per-regime min IC**, **STOP — 问题不在架构**
- *Expected output*: probably PatchTST wins 但要确认 lift 大小 — 这给后续 Tier 2 architectural changes 提供 reality check
- *Risk*: 暴露 "transformer 不必要" — 但这是 epistemic win
- *ETA*: 1 天

**T1.5 Margin Ranking loss A/B** (CIKM 2025 winner)
- *Why*: arXiv 2510.14156 在 PortfolioMASTER × S&P500 上明确 Margin Ranking > pairwise > MSE on Sharpe
- *Ref*: arXiv 2510.14156, `torch.nn.MarginRankingLoss(margin=0.1)`
- *Implementation*:
  - `patchtst_hf.py` 加 `--loss {pairwise, margin_ranking, listnet}`
  - Margin Ranking: 同 pair 构造 (高 label vs 低 label), 但 hinge loss with margin
  - ListNet: top-1 probability cross-entropy
- *Eval gate*: 3-way A/B (pairwise / margin / listnet), 5-cut × 5-seed, **min-regime IC + portfolio Sharpe in shadow sim**
- *Expected lift*: per CIKM benchmark, Sharpe +0.10 / AR +2-3pp
- *Risk*: 低 — 完全 backward-compatible config flag
- *ETA*: 1-2 天

**T1.6 Student-t NLL head** (替代 NGBoost σ wire)
- *Why*: NGBoost σ-wire 5/15 3-condition A/B 全 null/negative；HF 原生支持
- *Ref*: HF `PatchTSTConfig(loss="nll", distribution_output="student_t")`; MOIRAI 同 pattern
- *Implementation*:
  - 改 `PatchTSTForRegression` config: `loss="nll", distribution_output="student_t"`
  - Sample μ + σ in inference; 灌到 Kelly μ/σ pipeline
- *Eval gate*:
  - 5-seed σ-calibration test (per Duan 2020 §4)
  - σ-calib coefficient >0.20 (与 NGB 5/15 confirmed 持平)
  - Per-regime min IC 不退步
- *Expected lift*: σ wire activation in golden (NGB 至今 OFF — μ predictions 不消费)
- *Risk*: 中 — 改 loss 影响 backbone 行为，需要小心 re-tune learning rate
- *ETA*: 2-3 天

**T1.7 Regime label as input feature (must-beat baseline)**
- *Why*: arXiv 2603.04441 — 简单 regime embedding feature 加到现有 model 得 Sharpe 2.18 vs SPX 1.18. **Tier 2 architectural changes 必须 beat this**
- *Implementation*:
  - 在 panel 数据 build 阶段 (`training_panel/`) 加 `regime_*` features:
    - one-hot {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR}
    - regime probability (HMM posterior)
  - 同 features 给 XGB primary + HF PatchTST shadow
- *Eval gate*: 5-cut min-regime IC, 必须 ≥ current prod
- *Expected lift*: literature suggests +0.01 至 +0.03 IC (regime-aware features alone)
- *Risk*: 低 — 数据增强
- *ETA*: 1-2 天

**Tier 1 总 ETA**: ~1 周。每 item 独立 promote-able per CLAUDE.md §5.13.4a Tier 3 gate。

---

### TIER 2 — Architectural lift (Week 2-4, medium risk, high ROI)

**T2.1 Cross-stock attention layer (iTransformer-style)**
- *Why*: Channel-independence 是 #1 documented failure mode for cross-sectional finance (arXiv 2502.09683, 2505.12761)。MASTER / PortfolioMASTER 全部加这一层。
- *Ref*: thuml/iTransformer 源码; PortfolioMASTER (arXiv 2510.14156)
- *Implementation*:
  - 新 `kernel/models/cross_stock_attention.py`: 在 PatchTST encoder 输出 (B, d_model) per ticker 之后, 同一日期内 reshape 为 (1, N_tickers, d_model), 跑 iTransformer-style attention block (`MultiheadAttention(d_model, 4)` + FFN), 输出 (1, N_tickers, d_model)
  - 同 training loop 中需要 group-by-date batching (现有 pairwise loss 已经是 per-day batching，兼容)
  - 1-2 个 cross-stock attention layer, dropout 0.1
- *Eval gate*: 5-cut × 5-seed, min-regime IC + DSR + PBO per CLAUDE.md §5.13.4a Tier 3
- *Expected lift*: literature 报告 +5-10% over vanilla PatchTST on multivariate benchmarks; finance 上 MASTER/MIGA 报告更大
- *Risk*: 中 — 新 layer, 需要 sanity test + per-regime IC monitor
- *ETA*: 1 周

**T2.2 FiLM regime conditioning**
- *Why*: PRIME DIRECTIVE 在 model layer 实施的最 lightweight 方式 (Perez 2017)。Shared backbone + 1 MLP head, 不分裂数据。
- *Ref*: arXiv 1709.07871; Temporal FiLM (NeurIPS 2019, 1909.06628)
- *Implementation*:
  - 新 `kernel/models/film_conditioning.py`:
    - Input: PatchTST intermediate activation `x: (B, T, d_model)` + regime_vector `r: (B, K)` where K=4 regimes
    - `γ = MLP(r) → (B, d_model)`, `β = MLP(r) → (B, d_model)`
    - Output: `γ.unsqueeze(1) * x + β.unsqueeze(1)`
  - Insert 1-2 FiLM layers in PatchTST encoder (after RevIN, before/between attention blocks)
  - Regime vector: HMM posterior (soft prob, not hard one-hot — 容忍 detector noise)
- *Eval gate*: A/B vs T1.7 (regime as input feature). FiLM 必须 beat T1.7 by ≥ +0.005 min-regime IC for 复杂度 justified.
- *Expected lift*: literature 上 FiLM 通常 +5-10% relative on regime-shifted benchmarks
- *Risk*: 中 — regime detector quality 现在 bind (CLAUDE.md PRIME DIRECTIVE 警告) → 必须先确认 detector accuracy on objective regimes
- *ETA*: 1 周

**T2.3 GroupDRO loss**
- *Why*: PRIME DIRECTIVE objective-function 实施。**直接替代 pooled-mean training**, 与 evaluation 对齐。
- *Ref*: Sagawa 2019 (arXiv 1911.08731); `feedback_prime_directive_in_objective_funcs`
- *Implementation*:
  - 训练 loop 中, batch 是 per-day (已经是 per-day)；regime label 来自 HMM
  - 每 epoch end 时:
    - 计算 per-regime mean loss `L_r`
    - 更新 regime weight `q_r ← q_r * exp(η * L_r)`, normalize
  - Loss = `Σ q_r * L_r` instead of `mean(L)`
  - **重要**: Sagawa 警告 — needs stronger L2/early-stopping; 加 `--gdro-l2-coef`, `--gdro-eta` flags
- *Eval gate*: min-regime IC must improve vs pooled-mean; pooled IC 可能 slightly degrade (acceptable trade-off per PRIME DIRECTIVE)
- *Expected lift*: 10-40pt worst-regime improvement per Sagawa, but 需要正确 regularization
- *Risk*: 中-高 — 训练 dynamics 改变, 需要 careful hyperparameter tune
- *ETA*: 1 周

**T2.4 Masked-patch SSL pretrain → finetune**
- *Why*: PatchTST 原 paper 报告 SSL pretrain 在 6/8 datasets 击败 from-scratch; HF `PatchTSTForPretraining` ships
- *Ref*: arXiv 2211.14730 (Section 5.2); SimMTM (NeurIPS 2023, 2302.00861)
- *Implementation*:
  - Phase A: pretrain `PatchTSTForPretraining` on **整个 Russell 1000 + cross-sectional history** (mask ratio 0.4, MSE on masked patches), 不需要 label
  - Phase B: swap head to `PatchTSTForRegression` (or whatever T2.1-2.3 stack final 是), finetune on ranking objective
- *Eval gate*: A/B finetune-from-pretrain vs from-scratch, min-regime IC
- *Expected lift*: arXiv reports up to +26% MSE reduction over from-scratch (但 ranking 任务上 lift 可能更小)
- *Risk*: 低 — pretrain 失败 fallback 到 from-scratch
- *ETA*: 1 周 (pretrain 自动 overnight, finetune 1-2 天)

**Tier 2 总 ETA**: ~3-4 周, **可并行**。

```
Week 2          Week 3          Week 4
  │               │               │
  T2.1 cross-stock attention (1w) ┤
                  │
                  T2.2 FiLM (1w)  ┤
                                  │
                  T2.3 GroupDRO (1w) ──────┤
                                           │
                                  T2.4 SSL pretrain (1w, BG) ──┤
                                                                │
                                          ┌─ Tier 2 integration A/B ─┐
                                          │  全部组合 5-cut × 5-seed   │
                                          │  per-regime promote gate   │
                                          └────────────────────────────┘
```

---

### TIER 3 — Ambitious (Week 5-8+, high risk, high upside)

**T3.1 Foundation model fine-tune (Sundial / TimesFM / Chronos-2)**
- *Why*: 给 capacity upper bound — 如果 foundation model finetune 不能 beat T2 stack, 说明 ranker bottleneck 不在 model size
- *Ref*: Sundial (arXiv 2502.00816), TimesFM finetuning recipe, Chronos-2 HF
- *Implementation*:
  - 选 1 (推荐 TimesFM 因为 PEFT/LoRA recipe shipped):
    - Freeze backbone, train rank-16 LoRA on output projection + selected attention layers
    - Per-ticker representation → cross-stock attention (借 T2.1) → ranking head
  - 训练: 5-cut × 3-seed (compute-heavy)
- *Eval gate*: vs full T2 stack (T2.1 + T2.2 + T2.3); min-regime IC + DSR + PBO
- *Expected outcome*: 50/50 — foundation models 可能 beat 也可能 plateau (TS foundation models 的 pretrain corpus 几乎无金融数据)
- *Risk*: 高 compute cost, 中 outcome uncertainty
- *ETA*: 3-4 周

**T3.2 Sparse MoE (MIGA-style)**
- *Why*: 如果 FiLM/LoRA/GroupDRO 不能 saturate regime-conditional capacity, 升级到 K experts
- *Ref*: MIGA (arXiv 2410.02241), MoEC cluster-dropout (2207.09094); sigmoid gating (2405.13997)
- *Implementation*:
  - K=5 PatchTST encoders (or shared backbone + K LoRA adapters)
  - Sigmoid gating network on regime probability + ranker uncertainty signal (DEUP-style, arXiv 2603.13252 — gate on ranker's trailing realized rank-displacement uncertainty)
  - Loss: prediction loss + `L_route = CE(gate, true_regime) * 0.1` (regime-supervised, NOT load-balance)
  - MoEC cluster-dropout for rare-regime experts (BEAR/CHOPPY)
- *Eval gate*: vs T2 stack; per-expert IC + per-regime aggregated IC + DSR + PBO
- *Expected lift*: MIGA +8pt CSI300 absolute AR — applied to RenQuant possibly +5-15pt
- *Risk*: 高 — 训练 complexity, 数据 fragmentation 风险 (尤其 BEAR/CHOPPY)
- *ETA*: 4-6 周

**T3.3 Multi-quantile head + portfolio-level Sharpe loss**
- *Why*: End-to-end 优化 downstream metric (Sharpe), 不是 proxy (IC)
- *Ref*: GQFormer (multi-quantile transformer); Sharpe-loss papers (arXiv 2603.19288, 2509.24144)
- *Implementation*:
  - Multi-quantile head (q5, q25, q50, q75, q95) with pinball loss
  - Differentiable Sharpe: portfolio weight = `softmax(predicted_returns / temperature)`; backtest 30-day rolling Sharpe as loss
  - Joint training: pinball loss + Sharpe loss (weighted)
  - **重要**: per PRIME DIRECTIVE — Sharpe loss 必须是 `min_regime Sharpe`, NOT pooled
- *Eval gate*: 5-cut × 5-seed; per-regime Sharpe + DSR + PBO
- *Risk*: 高 — Sharpe loss 训练 dynamics tricky; over-fit 风险
- *ETA*: 2-3 周

**T3.4 Conservative online adaptation (CoTTA-style)**
- *Why*: 模型 deploy 后 regime 漂移；inference-time adapt 而不是 retrain
- *Ref*: CoTTA (arXiv 2203.13591), ProMod (arXiv 2412.08435); 警告 paper arXiv 2602.00073
- *Implementation*:
  - Only adapt RevIN affine params (not weights)
  - Use BN-stats rolling update on last 20-trading-day window
  - Stochastic-restore-to-source: 每 N 步 with prob p restore 原 weights (prevent catastrophic forgetting)
- *Eval gate*: shadow mode (parallel to non-adapted version), 至少 4 周 live data; **detector-error stress test** (intentionally flip regime label, measure degradation slope)
- *Expected lift*: 在 regime-shift period 上 +5-10% per literature; 但金融上 paper 警告小心
- *Risk*: 高 — 一切 online adaptation 在金融上都 fragile
- *ETA*: 2-3 周

**Tier 3 总 ETA**: ~6-8 周。**不 commit 全部** — Tier 2 verdict 后 user 决定 Tier 3 选 1-2 个。

---

## 6. Anti-recommendations (literature 警告 — 不要做)

| Anti-pattern | Why | 推荐替代 |
|---|---|---|
| 优化当前 hard-routed RegimeRouterScorer | arXiv 2603.13252: market-stress gate AUROC < 0.5 (worse than random); hard routing boundary discontinuous | **冻结当前 router 作为 baseline**, 走 FiLM / LoRA / soft MoE 路线 |
| Switch/Mixtral load-balance loss for K=5 regime MoE | Load balance 假设 uniform routing 是 optimal — 但 K=5 known sub-populations 下，你 WANT expert collapse (BEAR samples 独占 BEAR expert) | **regime-supervised router loss** `L_route = CE(gate, true_regime)` |
| Aggressive Tent-style TTA (full normalization adapt) | arXiv 2602.00073 明确警告 hurts on real financial data | CoTTA conservative (only RevIN affine + stochastic restore) |
| 用 NDCG / MAP 作为 model selection | CIKM 2025 (arXiv 2510.14156): NDCG **不** 和 portfolio Sharpe 相关 | Min-regime IC + portfolio shadow Sharpe |
| 加 RevIN 不验证 | arXiv 2603.11869: RevIN 在 Traffic 上 actually 增加 distributional distance | A/B test ON vs OFF on finance data |
| TimesNet for ranking | 2D 周期性是错 inductive bias — returns 无主周期 | 跳过 |
| Crossformer for full 1000-ticker panel | O(n_var × n_patch)² infeasible | iTransformer (linear in n_var) |
| Lag-Llama 作为 backbone | Stagnant repo (Jun 2024 last commit) | TimesFM / Chronos-2 / Sundial |

---

## 7. 评估方法 (PRIME DIRECTIVE compliant)

每个 Tier 1/2/3 item 必须通过:

### 7.1 Sanity Triad (CLAUDE.md §5.2, **mandatory**)
1. **A/A test**: 5-seed 同 config, σ should be < 0.005 IC; 大 σ = unstable training
2. **Shuffled-label** (`--label-shuffle-seed N`): IC ≈ 0; 否则 leakage
3. **Time-shift placebo** (`--label-shift-days N`): IC ≈ 0; 否则 lookahead

### 7.2 Per-regime IC matrix (PRIME DIRECTIVE)
- Validation 输出 dict {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR} → IC
- Selection metric: **min(IC across regimes)** 不是 mean
- Promote gate: min-regime IC ≥ baseline min-regime IC across ALL regimes

### 7.3 Walk-forward + 5-seed
- 5 cuts from `kernel/walk_forward_splits.build_default_cuts()` — covid / inflation peak / unwind 等
- 5 seeds per cut
- Per-point summary: mean ± std of per-regime IC

### 7.4 Multi-comparison correction (CLAUDE.md §5.14.4)
- **DSR** (Bailey-López de Prado 2014): `n_trials = N_design_points * N_seeds * N_cuts`
- **PBO via CSCV** (Bailey-Borwein-López de Prado-Zhu 2015)
- Promote 必须: DSR > 0.5 OR PBO < 0.5 OR (n ≥ 30 AND t > 3.0)

### 7.5 Detector-error stress test (regime-conditional items only)
- Train with true regime label
- Evaluate with 10% / 20% noisy regime labels
- Measure IC degradation slope
- 如果 5% noise → 20% IC drop = 模型对 detector 太敏感, deploy risk

### 7.6 Shadow mode 强制 (Tier 2/3)
- 任何 architectural change 必须 ≥ 4 周 shadow run 在 live data 上
- Per-day per-regime IC 监控
- Detector-trust metric (DEUP-style trailing rank-displacement uncertainty)

### 7.7 Held-out regime test (Tier 2/3)
- Train on {BULL_CALM, BULL_VOLATILE, CHOPPY}
- Test on BEAR (or vice versa)
- 测试 architecture 是否 generalize new regime vs memorize seen ones

---

## 8. 开放问题 — 需要用户决定

1. **Tier 1 全部并行 or 顺序?**
   - 推荐: T1.1, T1.2, T1.3, T1.4 同 1 周并行 (低 coupling);
   - T1.5, T1.6, T1.7 在 T1.1-1.4 落地后 第 2 周并行 (依赖 best-epoch save 修好)

2. **当前 RegimeRouterScorer 如何处置?**
   - Option A: 冻结为 baseline, 不再优化 (推荐)
   - Option B: 删除, 完全走 soft conditioning 路线
   - Option C: 继续作为 production primary 直到 T2.2 (FiLM) 击败它

3. **Tier 2 完成后 Tier 3 选哪 1-2 个?**
   - 这个等 Tier 2 verdict — 不预先 commit

4. **Compute budget**:
   - Tier 1: ~5-10 GPU-hours (M4 Pro MPS 已足够)
   - Tier 2: ~50-100 GPU-hours (overnight runs)
   - Tier 3: ~200-500 GPU-hours (foundation model finetune 需要)
   - 是否需要 cloud GPU (A100) 加速 Tier 3?

5. **是否需要 universe expansion** (wl-183 / R1K) 同时进行?
   - 用户 2026-05-14 verdict: "structural change required" — universe 是 structural lever 之一
   - 但 model 改 + universe 改同时做会 confound A/B
   - 推荐: model 先 (Tier 1+2 完成后), universe 后

6. **新 model promote 后老 XGB 如何处理?**
   - Option A: XGB 进 shadow 作为 fallback
   - Option B: RegimeRouter (新 PatchTST primary + XGB shadow per regime)
   - Option C: 完全 retire XGB
   - 推荐: A — XGB 保留 as 1-line config rollback

---

## 9. References & 代码 pointers

### 9.1 Canonical papers (literature read 列表)

**PatchTST 家族**:
- [PatchTST (Nie et al., ICLR 2023)](https://arxiv.org/abs/2211.14730) · [official code](https://github.com/yuqinie98/PatchTST)
- [Channel Dependence bias (2025)](https://arxiv.org/html/2502.09683)
- [RevIN critique (2026)](https://arxiv.org/html/2603.11869)

**Successor architectures**:
- [iTransformer (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.06625)
- [TimeMixer (Wang et al., ICLR 2024)](https://arxiv.org/abs/2405.14616)
- [PathFormer (ICLR 2024)](https://arxiv.org/abs/2402.05956)
- [EMAformer (2025)](https://arxiv.org/abs/2511.08396)

**Cross-sectional finance**:
- [MASTER (AAAI 2024)](https://arxiv.org/abs/2312.15235) · [code](https://github.com/SJTU-DMTai/MASTER)
- [MIGA (2024)](https://arxiv.org/abs/2410.02241)
- [PortfolioMASTER + loss eval (CIKM 2025)](https://arxiv.org/abs/2510.14156)
- [Kelly-Kuznetsov-Malamud-Xu AIPM (NBER WP 33351, 2025)](https://www.nber.org/papers/w33351)
- [Chen-Pelger-Zhu "Deep Learning in Asset Pricing" (Mgmt Sci 2024)](https://pubsonline.informs.org/doi/10.1287/mnsc.2023.4695)

**Loss functions**:
- [LambdaRankIC (2026)](https://arxiv.org/pdf/2605.00501)

**Regime conditioning**:
- [FiLM (Perez 2017)](https://arxiv.org/abs/1709.07871)
- [Temporal FiLM (NeurIPS 2019)](https://arxiv.org/abs/1909.06628)
- [GroupDRO (Sagawa 2019)](https://arxiv.org/abs/1911.08731)
- [Sigmoid vs Softmax Gating for MoE (NeurIPS 2024)](https://arxiv.org/abs/2405.13997)
- [MoEC Mixture of Expert Clusters](https://arxiv.org/abs/2207.09094)
- [Explainable Regime Aware Investing](https://arxiv.org/abs/2603.04441)
- [When Alpha Breaks — Two-Level Uncertainty (2026)](https://arxiv.org/abs/2603.13252)
- [FinLoRA Benchmark (2025)](https://arxiv.org/abs/2505.19819)

**SSL pretrain**:
- [SimMTM (NeurIPS 2023)](https://arxiv.org/abs/2302.00861)

**TTA / online adaptation**:
- [CoTTA (CVPR 2022)](https://arxiv.org/abs/2203.13591)
- [Proactive Adaptation (KDD 2025)](https://arxiv.org/abs/2412.08435)
- [TTA for Financial Markets (2026)](https://arxiv.org/abs/2602.00073)

**Foundation models**:
- [Sundial (ICML 2025 Oral)](https://arxiv.org/abs/2502.00816)
- [Time-MoE (ICLR 2025 Spotlight)](https://arxiv.org/abs/2409.16040)
- [Chronos](https://github.com/amazon-science/chronos-forecasting)
- [TimesFM](https://github.com/google-research/timesfm)
- [MOIRAI (uni2ts)](https://github.com/SalesforceAIResearch/uni2ts)

### 9.2 借用 open-source repos

| Repo | Stars | 用途 |
|---|---|---|
| `huggingface/transformers` (PatchTST classes) | ~131k | Primary backbone — use config flags |
| `Nixtla/neuralforecast` | ~4.1k | Panel data contract + DistributionLoss + AutoML |
| `thuml/Time-Series-Library` | ~12.3k | 40+ models harness pattern |
| `microsoft/qlib` | ~43.2k | `pytorch_general_nn.py` unified driver |
| `SJTU-DMTai/MASTER` | — | Market-gated attention 直接借 |
| `thuml/iTransformer` | ~2.1k | Cross-variate attention 借鉴 |
| `cure-lab/LTSF-Linear` | ~2.5k | DLinear baseline (强制) |
| `yuqinie98/PatchTST` | ~2.6k | Original — masked-patch SSL pretrain recipe |

### 9.3 RenQuant repo 代码 anchors

- `scripts/patchtst_hf.py:287-291` — best-epoch save bug
- `scripts/patchtst_hf.py:104-122` — CSRankNorm + Winsorize 已实施
- `scripts/patchtst_hf.py:77-89` — RankNet pairwise BCE loss
- `scripts/patchtst_doe_hf.py:65-71` — DOE 当前 knob 范围
- `backtesting/renquant_104/kernel/panel_pipeline/hf_patchtst_scorer.py` — inference path
- `backtesting/renquant_104/kernel/panel_pipeline/regime_router_scorer.py` — 当前 hard router (将冻结)
- `backtesting/renquant_104/kernel/panel_pipeline/model_registry.py` — extensible registry
- `kernel/hmm_regime_labels.py` — regime label provider
- `kernel/regime.py` — production regime detector (PRIME DIRECTIVE P0)
- `artifacts/patchtst_doe_hf/main_effects.csv` — DOE main effects
- `doc/research/2026-05-18-overnight-patchtst-plan.md` — Phase 0-4 cron job state machine
- `doc/research/2026-05-18-model-regime-mismatch.md` — strategic finding mean-reversion-vs-momentum

---

## 10. 下一步 (等 user sign-off)

**首先**: 确认本文档方向（Pillar A/B/C + Tier 1/2/3 框架）。

**确认后** (Day 1):
1. 创建 7 个 Tier 1 任务 in TaskCreate / roadmap
2. 立即开工 T1.1 (best-epoch save bug fix) — 0.5 天 ship
3. T1.2 (per-regime IC validation) 同步 — 0.5 天 ship
4. T1.4 (DLinear baseline) 并行 BG — 1 天

**Week 1 end**: Tier 1 全部 ship + verdict; Tier 2 design 文档 (per item 实现 sketch + sanity test plan)

**Week 2-4**: Tier 2 三个并行 track (cross-stock attention / FiLM / GroupDRO + SSL pretrain BG)

**Week 4 end**: Tier 2 integration A/B + per-regime promote gate decision

**Week 5+**: 基于 Tier 2 verdict 选 Tier 3 1-2 items

---

**EOF — proposal ready for review.**
