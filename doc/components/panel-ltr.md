# Panel-LTR + NGBoost — 训练方法与术语入门

这份文档解释 renquant_104 里 **Panel-LTR** 和 **NGBoost 头** 的训练方法、设计意图,以及所有相关缩写/术语。

适用对象:不熟悉这两个模块的使用者/维护者。
前置要求:对 OHLCV 数据和"横截面 vs 时间序列"有基本概念即可。

> **2026-05-20 update**: production is **alpha158+fund XGBoost** (`kind: xgb`,
> `artifacts/prod/panel-ltr.alpha158_fund.json`, **172 features** = 158 alpha158
> + 5 SEC fund + 3 PEAD + 3 SUE + 3 sentiment). NGBoost head **trained + promoted
> to prod 2026-05-17** (val_IC +0.0352, σ-calib +0.274) but the σ-wire to Kelly
> stays OFF per 3-condition A/B all NULL/negative. μ from calibrator (Platt,
> switched from isotonic 2026-05-18); σ from `realized_vol_60d` fallback.
> **HF PatchTST shadow** wired 2026-05-19 (commits `cf6311c`, `4e156e2`) — see
> `scripts/patchtst_hf.py` + `doc/research/2026-05-19-patchtst-improvement-plan.md`.
> The legacy 27-feat XGB and alpha158_linear paths are dormant.

相关代码:
- `backtesting/renquant_104/training_panel/pp_panel_training.py` — 5 阶段 XGB 训练 pipeline
- `backtesting/renquant_104/training_panel/ltr_model.py` — XGBoost LTR 模型封装
- `backtesting/renquant_104/training_panel/linear_ltr.py` — sklearn LinearRegression wrapper (`PanelLinearScorer`)
- `backtesting/renquant_104/training_panel/ngboost_head.py` — NGBoost Normal(μ, σ) 头 (legacy XGB path only)
- `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py` — 推理侧任务链 (dispatches on `metadata.kind`)
- `backtesting/renquant_104/kernel/panel_pipeline/alpha158_features.py` — Qlib alpha158 feature builder
- `scripts/build_alpha158_qlib.py` — alpha158 dataset builder (output: `data/alpha158_qlib_dataset.parquet`)
- `scripts/train_panel_linear.py` — alpha158_linear training entry point
- `scripts/retrain_alpha158_linear.sh` — daily retrain wrapper (not yet scheduled)

---

## 1. 核心定位:这是一个横截面排序问题

**问题:** 每天从 wl200 (142 只股票, quality-first) 中挑 ~8 只买入。

这**不是**一个时间序列回归(预测某只股票未来 60 天涨多少)问题,而是一个**横截面排序**问题(今天这 142 只股票的相对强弱次序)。不同的目标函数和训练方式。

### 为什么这个分别很重要

- **时间序列回归** 的损失函数关心 **绝对值**:"预测 AAPL 明天涨 1.2%,实际涨 1.0% → 误差 0.2pp"。这种损失对市场整体走势(bull/bear)很敏感,模型很容易学成 "β tracker"——跟着 SPY 涨跌。
- **横截面排序** 的损失函数关心 **相对次序**:"在今天的 38 只股票里,AAPL 应该排在 NVDA 前面。是不是真的排前面了?"。绝对涨跌不重要,只有相对位置有意义。这是真正的 alpha 生成视角。

---

## 2. 103 vs 104:从 38 个独立模型到 1 个面板模型

### renquant_103 的老做法

每只股票独立训练一个模型(Classification / QLearning / Manual / XGBoost 四选一),按 OOS Sharpe 选赢家。

**问题:**

| 问题 | 具体表现 |
|---|---|
| 数据饥饿 | 每个模型只有 500–1000 行训练数据。XGBoost 默认参数直接过拟合。 |
| 校准噪音 | 每只股票的 score_calibration 只用自己的 500 行 OOS 拟合,方差极大。 |
| 无跨股票学习 | NVDA 和 AMD 的宏观敏感度几乎一样,但两个模型互相看不见。 |
| 新股冷启动 | 上市不久的股票没法训练。 |

### renquant_104 的新做法

**把 38 只股票的历史数据堆到一个大面板(panel)里,训练一个统一模型。**

```
Panel (long format):
  (AAPL, 2024-01-03, features...) → label
  (AMZN, 2024-01-03, features...) → label
  ...
  (TSLA, 2024-01-03, features...) → label
  (AAPL, 2024-01-04, features...) → label
  ...
```

训练数据量:500 行 × 38 只 = **~80,000 行** (过去 ~10 年)。

---

## 3. 训练流程(5 个阶段)

代码结构对应 `PanelTrainingPipeline`:

```
PanelDataJob        — fetch OHLCV + sector momentum + fundamentals
  └── FetchOHLCVTask
  └── SectorMomentumTask
  └── LoadFundamentalsTask

PanelFeatureJob     — 每只股票并行处理
  └── TickerPanelFeatureJob     (build indicators)
  └── TickerPanelNeutralizeJob  (partial feature neutralization)
  └── TickerPanelFactorJob      (compute raw factors)

PanelAssemblyJob    — 组装 panel
  └── FactorZScoreTask
  └── LabelsTask
  └── BuildPanelTask

PanelModelJob       — 训练 LTR 模型
  └── CrossValidateTask
  └── FinalFitTask
  └── SaveArtifactTask

PanelNGBoostJob     — 训练不确定性头(可选)
  └── NGBoostFitTask
  └── NGBoostSaveTask
```

### 阶段 1 — 原始数据与宏观状态

- **OHLCV** (Open/High/Low/Close/Volume):每只股票每天 5 个价格字段。
- **技术指标**:RSI、MACD、ADX、CCI、OBV slope、Williams %R、BBP — 从 OHLCV 派生的传统量化指标。
- **SPY 宏观状态**:用 3 层检测器合成一个 regime 标签。

#### 宏观状态的 3 层检测

**第 1 层 — Hurst 指数(63 天滚动)**

用来判断"市场当前是趋势性还是均值回归性"。

- **H > 0.55**:趋势性,今天涨明天大概率继续涨
- **H < 0.45**:均值回归,涨多了会跌回来
- **H ≈ 0.50**:随机游走,没规律

Hurst = Harold Edwin Hurst 的姓,他 1951 年在研究尼罗河水位时发明了这个指数。

**第 2 层 — CUSUM(累积和)**

CUSUM = Cumulative Sum。检测时间序列里的**突变点**(structural break)。比如 SPY 从 bull market 突然转 bear,CUSUM 会很快触发一个 changepoint。用途:在 regime 切换的 3 个 bar 内禁止新买入,防止模型在 regime 转换期出错。

**第 3 层 — GMM(高斯混合模型)**

GMM = Gaussian Mixture Model。把 SPY 的 4 个特征(20 天实现波动率、ADX、20 天累计收益、收益自相关)聚成 3 个 cluster,cluster 标签对应 regime:

| Regime | 特点 | 买入策略 |
|---|---|---|
| BULL_CALM | 低波动、趋势向上 | 动量买入,最大仓位 15% |
| BULL_VOLATILE | 高波动、趋势向上 | 抄底买入(高波动下跌日),最大仓位 20% |
| CHOPPY | 横盘震荡 | 相对强度买入(跑赢行业 ETF 的股票),最大仓位 15% |
| BEAR | 下跌 | 禁新多头,只买防御性资产(GLD/XLU) |

GMM 给出每个 regime 的概率,最高概率的 regime 是标签;概率本身也用来缩放仓位(confidence scaling)。

### 阶段 2 — 特征中性化(partial feature neutralization)

这步是为了防止模型学成"β tracker"。

**问题:** 技术指标里的动量类特征(`rel_mom_20d`、`rel_mom_60d`、`trend`、`trend_long`)本质上都是"过去 N 天的累计收益"。如果直接喂给模型,它会优先学"过去涨得多 → 未来排名高"——但这其实就是 momentum factor,和行业 β 严重相关,不是 alpha。

**对策:** 对这 4 个动量特征,用同行业 ETF 的动量做回归,扣掉:

```python
neutralized_rel_mom_20d = rel_mom_20d − β_sector × sector_etf_mom_20d
```

回归窗口用 252 天。**均值回归类指标(RSI/BBP/Williams %R)不做中性化**,它们本身就是"距离均衡的位置",中性化会破坏信号。

这招叫 **Numerai-style partial neutralization**,来自加密币对冲基金 Numerai 的公开文档。

### 阶段 3 — 横截面因子(cross-sectional factors)

除了技术指标,我们另外算 4 个**跨股票的 z-score**:

| 因子 | 计算 | 经济含义 |
|---|---|---|
| `size_z` | log(close × shares_out) 的横截面 z-score | 市值因子(大盘 vs 小盘) |
| `mom_12_1_z` | 过去 252d 收益 − 过去 21d 收益,再 z-score | 经典 Fama-French **12-1 动量**(跳过最近 1 个月是为了避开短期反转) |
| `beta_60d_z` | 对 SPY 的 60 天滚动 β,再 z-score | 系统性风险暴露 |
| `resid_mom_z` | `mom_12_1 − β_60d × SPY_mom_12_1`,再 z-score | 扣掉市场贡献后的"纯个股动量" |

**为什么 z-score?**

`z = (x − cross_sectional_mean) / cross_sectional_std`

每天对 38 只股票计算一次。这样 AAPL 的 `size_z = +1.8` 的含义是"今天它的市值比平均值高 1.8 个标准差",跨日可比。XGBoost 做 rank 目标时,这种标准化特征比原始数值更稳定。

### 阶段 4 — 标签(labels):β 中性化 + Gaussianization

这是整个 pipeline 里最关键的一步。label 质量决定模型能否学到信号。

#### Step 4.1 — 计算原始 5 天远期收益

```
fwd_return_i,t = close_i,t+5 / close_i,t − 1
```

#### Step 4.2 — β 中性化(去掉市场和行业暴露)

```
residual_return_i,t = fwd_return_i,t
                      − β_spy_i,t × spy_fwd_return_t
                      − β_sector_i,t × sector_fwd_return_t
```

β 用滚动 OLS 回归(rolling OLS)拟合,窗口默认 60 天(Run 2 改成 252 天减少噪音)。**关键:回归只用"严格先于 t"的数据**,防止未来信息泄漏——这叫 **purge** (清洗)。

- **β_spy** = 这只股票对 SPY 的敏感度
- **β_sector** = 这只股票对同行业 ETF 的敏感度(同行业 ETF 通过 `sector_etf_map` 配置,如 NVDA → XLK)
- **OLS** = Ordinary Least Squares 最小二乘法
- **residual_return** = idiosyncratic 部分,即"这只股票今天涨跌里,不能被市场/行业解释的那一块"

NGBoost 训练直接用这个 residual_return(有实值尺度,才能算 σ)。

#### Step 4.3 — 横截面 Gaussianize(只给 Panel-LTR 用)

把每一天 38 只股票的 residual_return 做:

```
rank → (rank + 0.5) / N → norm.ppf(u) → N(0, 1)
```

- `norm.ppf` 是标准正态分布的 **inverse CDF**(逆累积分布函数),把均匀分布 [0, 1] 映射回标准正态分布。
- 结果:每天 38 个 label 是均值 0、标准差 1 的正态样本,排名最高的是最大的正值,最低的是最大的负值。

**为什么 Gaussianize?**

1. **跨日尺度归一化** — 大涨大跌的日子 residual_return 的绝对值会很大,Gaussianize 之后每一天都是 N(0, 1),可比。
2. **XGBoost rank:pairwise 损失函数最适合连续、对称分布的 label** — 原始 residual_return 有肥尾和偏度,打折 LTR 的效果。

### 阶段 5 — 训练 Panel-LTR

```python
import xgboost as xgb

dtrain = xgb.DMatrix(X, label=y)
dtrain.set_group(group_sizes)        # 每天 N 只股票是一组
dtrain.set_weight(sample_weights)    # 防过拟合:按 concurrency 倒数加权

model = xgb.train(
    params={
        "objective":        "rank:pairwise",
        "eta":              0.05,
        "max_depth":        6,
        "min_child_weight": 20,
        "subsample":        0.8,
        "colsample_bytree": 0.7,
        "lambda":           1.0,
        "alpha":            0.5,
    },
    dtrain=dtrain,
    num_boost_round=400,
)
```

关键点:

#### `rank:pairwise`

每一天的 38 只股票会生成 C(38, 2) = 703 对。损失函数是:对每一对 (i, j),如果真实标签 label_i > label_j,模型预测 pred_i - pred_j 应该 > 0。用 logistic loss 做连续松弛。

#### `set_group(group_sizes)`

告诉 XGBoost "这 N 行是同一个 query,做 pairwise 时只在组内枚举"。对我们来说 group_size = 每天的股票数(不一定是 38,因为新股可能还没上市)。

#### `sample_weight`

AFML (Advances in Financial Machine Learning,Marcos López de Prado 的书) 里的 concurrency weighting:5 天远期 label 意味着相邻两天的 label 窗口重叠 80%,XGBoost 会把它们当成 5 个独立样本来训练,过度放大信息量。对策:给每行一个权重 = 1 / (当天 label 窗口的并发度)。

### 阶段 6 — NGBoost 头(可选,Stage 2)

Panel-LTR 只输出一个排序分数,不输出"预期收益"或"置信度"。**NGBoost** 解决这个空缺。

**NGBoost** = **N**atural **G**radient **Boost**ing。斯坦福 ML Group 2020 年的工作 (Duan et al., arXiv:1910.03225)。

**和普通 XGBoost 的区别:**

| | XGBoost | NGBoost |
|---|---|---|
| 输出 | 一个数 (μ) | 一个概率分布的所有参数 (μ, σ) |
| 梯度 | 欧氏梯度 | **自然梯度**(manifold-aware) |
| 用途 | 回归/分类 | 分布式回归(distributional regression) |

**训练数据:** 和 Panel-LTR 共用同一批 24 个特征,但 label 是**阶段 4 的 residual_return 原始值**(不做 Gaussianize,要保留实际尺度)。

**推理输出:**

```python
dist = ngb_head.predict_distribution(X)
# dist.mu:     该股票未来 5 天残差收益的期望值
# dist.sigma:  预测的不确定性(标准差)
```

**应用 1 — σ-aware 排序分数:**

```
combined_score = μ − λσ
```

λ 默认 1.0。µ 高的股票吸引力强,但 σ 也高(不确定)就打折。这直接覆盖掉 Panel-LTR 的 rank_score。

**应用 2 — σ-sizing 仓位缩放:**

```python
universe_median_sigma = median(σ_i for i in today_candidates)
sigma_mult = clip(universe_median_sigma / σ_i, floor=0.3, ceiling=1.0)
max_position_pct *= sigma_mult
```

σ 大于中位数 → mult < 1 → 仓位缩小;σ 小于中位数 → mult = 1(不放大,防止过度集中)。

---

## 4. 评估方法:IC + Purged K-fold CV

### IC = Information Coefficient

**每天 38 只股票的 (模型打分, 真实 residual_return) Spearman 相关**,再在所有交易日求平均。

```python
for date in oos_dates:
    cross_section = panel[panel.date == date]
    ic_date = spearmanr(cross_section.model_score, cross_section.label)
mean_ic = mean(ic_date for all date)
```

参照标准:
- IC = 0 → 纯随机
- IC = 0.02 → 弱信号(勉强能用)
- IC = 0.08 → 可上线(renquant_104 的门槛)
- IC = 0.15 → 金融 ML 竞赛水平

**Spearman** 只看**排序**,不看绝对值。因为 Panel-LTR 输出的是 rank score,不是绝对收益预测,Spearman 和训练目标是对齐的。

### Purged K-fold CV with Embargo

普通 K-fold 有两个金融特有问题:

1. **Label 重叠** — 5 天远期 label 会泄漏到后续 5 个 bar。训练集里的样本 t 的 label 覆盖 [t+1, t+5],如果测试集包含 t+3,就污染了。
2. **短期相关性** — 时间序列有强短期自相关,测试集的相邻 bar 很容易从训练集"偷看"到模式。

对策(López de Prado AFML 第 7 章):

#### Purge (清洗)

训练集里所有 label 窗口和测试集窗口有任何重叠的样本,**丢掉**。

```
测试集:[t_test_start, t_test_end]
移除训练集:所有满足 label_end > t_test_start 且 label_start < t_test_end 的样本
```

#### Embargo (禁运期)

测试集结束后,空出 N 个 bar 不进任何 fold 的训练集。防止"刚看完测试期马上训练"的短期自相关泄漏。

```
测试集结束:t_test_end
禁运期:[t_test_end + 1, t_test_end + embargo_days]
```

N 默认等于 lookahead_days = 5。

---

## 5. 全部缩写/术语对照表

### 统计 & ML

| 缩写 | 全称 | 含义 |
|---|---|---|
| IC | Information Coefficient | 横截面 Spearman 相关系数(跨日平均) |
| CV | Cross-Validation | 交叉验证 |
| OOS / IS | Out-of-Sample / In-Sample | 样本外 / 样本内 |
| CDF | Cumulative Distribution Function | 累积分布函数 |
| PPF / inverse CDF | Point Probability Function | 分位数函数(CDF 的反函数) |
| OLS | Ordinary Least Squares | 最小二乘法回归 |
| LTR | Learning to Rank | 学习排序 |
| MLE | Maximum Likelihood Estimation | 极大似然估计 |
| GMM | Gaussian Mixture Model | 高斯混合模型 |
| CUSUM | Cumulative Sum | 累积和突变点检测 |
| MSE | Mean Squared Error | 均方误差 |

### 符号

| 符号 | 含义 |
|---|---|
| μ (mu) | 均值(期望值) |
| σ (sigma) | 标准差 |
| β (beta) | 线性回归斜率,通常指股票对市场的敏感度 |
| α (alpha) | 截距,或"超额收益"/市场无法解释的部分 |
| λ (lambda) | 超参数,通常表示正则化强度或惩罚权重 |
| ρ (rho) | 相关系数 |

### 金融

| 术语 | 含义 |
|---|---|
| OHLCV | Open/High/Low/Close/Volume — 日频 5 字段 |
| alpha | 非市场因素带来的超额收益(idiosyncratic returns) |
| β (market beta) | 股票对市场(SPY)的敏感度 |
| idiosyncratic | 个股特有(去掉市场/行业暴露后的部分) |
| momentum | 动量,"涨过的继续涨" |
| mean reversion | 均值回归,"涨过头了会跌回去" |
| value | 价值因子(低估值股票倾向跑赢) |
| quality | 质量因子(高 ROE/低负债的公司) |
| ROE | Return on Equity 股权回报率 = 净利 / 股东权益 |
| gross profitability | Novy-Marx 提出的质量因子 = 毛利 / 总资产 |
| book-to-price | 账面价值 / 市值(low 对应成长,high 对应价值) |
| Fama-French | 经典多因子资产定价模型(市场 β、size、value ± momentum、profitability、investment) |
| 12-1 momentum | 过去 12 个月收益减去最近 1 个月收益(Jegadeesh-Titman 1993) |
| rank_score | 104 里 CandidateResult 上的排序分数,经过校准/Panel-LTR 覆盖后 ∈ [0, 1] 或是 μ−λσ |
| rs_score | Relative Strength vs sector ETF(20 天股票收益 − 20 天行业 ETF 收益) |

### 104 专有术语

| 术语 | 含义 |
|---|---|
| Panel | 长格式数据 = 横截面 + 时间序列堆叠 |
| panel_score | Panel-LTR 模型对某 (ticker, date) 的输出分数 |
| regime | 当前市场状态 ∈ {BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR} |
| confidence | GMM 输出的 regime 概率,用于缩放仓位 |
| wash sale | 美国税法:卖出亏损股票 30 天内再买入,亏损不能抵税。系统里有 wash_sale_days 保护期 |
| LT / ST | Long-Term / Short-Term Capital Gains — 持仓满 365 天后税率从 37% 降到 20%,系统里的 lt_protection_days 保护期避免在临近 LT 门槛时被轮换出局 |
| tiered_thresholds | 分层阈值 — 第 1 个仓位要求 rank_score ≥ 0.10,第 2 个 ≥ 0.30,第 3 个 ≥ 0.50,越往后越挑剔 |
| rotation | 轮换 — 当持仓 A 的 ER 低于某个新候选 B 的 ER 一定幅度,卖 A 买 B |
| sector guard | 行业集中度保护 — 单个行业最多持仓 3 只 |
| correlation guard | 相关性保护 — 新买入的股票和已持仓的任何一只 120 天收益相关 > 0.70 就拒绝 |
| HWM | High-Water Mark — 账户历史最高净值,drawdown 以此为基准 |
| drawdown circuit breaker | 熔断器 — 账户跌破 HWM 15% 时禁止新买入 |
| trailing stop | 追踪止损 — BULL_CALM regime 专属,在持仓涨过 20% 之后,从高点回落 18% 就平仓 |

### XGBoost / 模型调参

| 参数 | 含义 |
|---|---|
| `objective: rank:pairwise` | LTR 损失函数,逐对比较 |
| `eta` (aka `learning_rate`) | 梯度缩放,每轮树的权重 |
| `max_depth` | 单棵树最大深度。深 → 强表达能力 → 更易过拟合 |
| `min_child_weight` | 叶子节点最少样本权重。大 → 正则化强 |
| `subsample` | 每棵树用的样本比例(< 1.0 做 bagging 正则) |
| `colsample_bytree` | 每棵树用的特征比例 |
| `lambda` | L2 正则化强度 |
| `alpha` | L1 正则化强度 |
| `num_boost_round` | 树的总数量 |

---

## 6. 延伸阅读

- Poh, D. et al. (2020). *Building Cross-Sectional Systematic Strategies By Learning to Rank*. arXiv:2012.07149
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. **第 4 章**:concurrency weighting,**第 7 章**:Purged K-fold CV
- Duan, T. et al. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction*. arXiv:1910.03225
- Novy-Marx, R. (2013). *The Other Side of Value: The Gross Profitability Premium*. Journal of Financial Economics
- Numerai — [Feature Neutralization 文档](https://docs.numer.ai/numerai-tournament/models/feature-neutralization)

---

## 7. 当前状态(随时间更新)

- 配置文件:`backtesting/renquant_104/strategy_config.json` → `panel_ltr` + `ranking.panel_scoring` 两块
- 训练脚本:`python scripts/train_104.py --force`
- 运行记录:`doc/experiments/panel-training-runs.md`(每次重训追加一条)
- 调度:Tue/Thu after-close 1:55 PM PT + Sunday 10:00 AM PT(launchd plists 在 `~/Library/LaunchAgents/com.renquant.*104.plist`)
