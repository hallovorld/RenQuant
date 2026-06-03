# 2026-06-03 — Kelly Sizing Audit: 是不是太保守了？

**Trigger**: user observation, 2026-06-03 — "我们下单都过于保守，导致现金比例
一直过高". Audit-only memo. **No code change** in this PR; recommends a
targeted A/B in §7 that respects §7.2 sanity + §7.4 promotion gating.

## TL;DR

是的，**有一个具体的、可量化的、单源 bug**：当前 Kelly sizing 的 σ 和
μ 不在同一个时间尺度上。具体来说：

- **μ** 来自 calibrator (`c.expected_return`)，是 **60 天 horizon** 的期望
  excess return (`panel_ltr.lookahead_days=60` /
  `qp_mu_horizon_days=60`，2026-05-24 Codex horizon-contract 修过)。
- **σ** 在 NGBoost 当前 disabled 的状态下（E55 失败后 2026-05-25 Codex
  contract repair 锁住），fallback 到
  `_realized_vol_annualized` —— **年化** 的 60 天 realized stdev
  (`std × sqrt(252)`)。
- Kelly 公式 `f* = μ / σ²` 要求 μ 和 σ 同周期。当前实现把 60 天的 μ
  除以年化的 σ²，导致**分母被放大 ~4.2×** (`252/60`)，所以
  `kelly_target_pct` 系统性地偏小约 **4×**。

**这就是 cash% 高的主要可解释项**。其他 7 个嫌疑层都不动，单独修
σ-horizon 这一项预计就能把 BULL_CALM 的实际 invested% 从今天的"长期
不到 50%" 抬到接近 regime 的 `max_position_pct × N_positions` 上限。

Recommendation：**不要直接翻 production**（违反 §7.4）。先做 A/B：

1. 加一个 `ranking.kelly_sizing.sigma_horizon_days` 配置项，默认沿用
   当前年化口径以保持 byte-equivalent，但允许显式设 60 让 σ 在喂
   `kelly_target_pct` 之前先 `σ × sqrt(60/252)`。
2. 27-month OOS sim：golden vs sigma_horizon=60。
3. 三件套：A/A、shuffle-placebo、time-shift-placebo + DSR + PBO
   (§7.2 / §7.3 / §9 mandate)。
4. 通过 §7.4 Tier 3 才能翻到 live。

---

## 1. The Kelly chain end-to-end

`kernel.kelly.kelly_target_pct` (`backtesting/renquant_104/kernel/kelly.py`):

```python
f_kelly = mu / sigma**2            # raw Kelly
f_frac  = fractional * f_kelly     # safety-multiplier (half/quarter Kelly)
target  = max(0, min(max_pct,      # regime max_position_pct × conf_mult
                     max_conc,     # global hard ceiling
                     f_frac))
```

调用点：`ApplyKellySizingTask`
(`kernel/panel_pipeline/job_panel_scoring.py:2811`)，紧跟在
`ApplyRealizedVolFallbackTask`
(`job_panel_scoring.py:2731`) 后面 —— 这步在 NGBoost 关掉时把
`c.sigma` 用 `_realized_vol_annualized` 填满。

Golden config 现状 (`strategy_config.golden.json:904`)：

| Knob | Value | 来源 / 含义 |
|---|---|---|
| `enabled` | `true` | 2026-05-15 calibrator P0 Phase 3 激活 |
| `fractional` | `0.5` | half-Kelly（标准保守，吸收 μ 估计误差） |
| `max_concentration` | `0.35` | 全局硬顶 |
| `min_edge` | `0.0` | μ 噪声地板（不绑） |
| `use_calibrator_mu` | `true` | μ ← `c.expected_return` (bounded ±0.20) |
| `use_realized_vol_fallback` | `true` | σ ← `_realized_vol_annualized` (clip 0.05–1.50) |
| `realized_vol_window_days` | `60` | trailing-60 daily ret stdev |
| `top_up_threshold` | `0.05` | TopUpHeld 不开仓，除非 kelly target − 当前 weight ≥ 5pp |

下游 consumer：
- 新买：`kernel/pipeline/task_selection.py:240` —— `max_pct = kelly *
  conv * sig_m`，被注入到 QP 上界。
- 加仓：`kernel/pipeline/task_topup.py:92, 238` —— TopUpHeld 用
  `hs.kelly_target_pct` 作 target，差 ≥ `top_up_threshold` 才动手。
- 换仓：`kernel/pipeline/task_rotation.py:113, 521` —— Kelly-delta
  rotation gate，候选 Kelly 要比持仓 Kelly 高至少 `kelly_rot_advantage`。
- Trim：`exit_types.py` 的 `kelly_trim` (PORTFOLIO_RISK exit)。

---

## 2. 主嫌：μ horizon vs σ horizon 错配

### 2.1 当前实现

`_realized_vol_annualized` (`job_panel_scoring.py:2785–2806`)：

```python
std = daily_returns.std()
return std * math.sqrt(252.0)        # ← 年化
```

返回的 σ 是 **年化** 的 (e.g., NVDA 年化 σ ≈ 0.50, AAPL ≈ 0.30, JNJ ≈ 0.18)。

μ 路径 (`job_panel_scoring.py:1730, 1973–2025`)：

```python
qp_mu_horizon = _qp_mu_horizon_days(ctx, cal)        # ← 60d (config)
c.mu = _calibrator_expected_return_at_horizon(
    cal, c.raw_score, horizon_days=qp_mu_horizon, ...
)                                                      # ← 60d-scale
```

calibrator P0 train/load-site clip 把 `expected_return.y` 钳在 ±0.20 ——
即 **±20% over 60 days**。这是绝对的 horizon return，**不是年化**。

### 2.2 单位为什么必须匹配

Kelly 公式 `f* = μ / σ²` 来自 log-utility 最大化
(Thorp 1962 / Rotando-Thorp continuous-time)。导出时的隐含假设是：μ
和 σ 都是 **同一个 bet period 内** 的 mean 和 stdev。如果 bet period
是 1 年，就用年化 μ 和年化 σ；如果是 60 天，就用 60 天 μ 和 60 天 σ。

如果你混用 — **公式不再是 Kelly**，只是一个名字叫 Kelly 的数。具体偏差：

设 σ_year = σ_60d × sqrt(252/60) (在 iid 假设下)。等价地：

```
σ_60d ² = σ_year ² × (60/252) = σ_year ² × 0.238
```

也就是说 60 天的方差只有年化方差的 **23.8%**。把年化 σ² 当 60 天 σ²
用的 Kelly 是：

```
f_buggy = μ_60d / σ_year²
f_true  = μ_60d / σ_60d²  =  μ_60d / (σ_year² × 0.238)
        =  f_buggy × 4.2
```

**当前实现把 Kelly 系统性地缩水到真正同周期 Kelly 的 ~24%**。

### 2.3 数字例子

典型 BULL_CALM 候选：μ_60d = 0.02 (panel-LTR top-decile 期望 2% / 60d
excess)，σ_year = 0.35。

| 算法 | 计算 | Kelly | × fractional 0.5 | 比 max_pct=0.15 |
|---|---|---|---|---|
| 当前 (buggy) | 0.02 / 0.35² = 0.163 | 16.3% | **8.2%** | 不绑，输出 8.2% |
| 同周期 (true) | σ_60d = 0.171, 0.02/0.0292 = 0.685 | 68.5% | **34.3%** | 绑在 15%，输出 15% |

强信号 (μ_60d = 0.05)：

| 算法 | f_frac | 输出 (max_pct=0.15) |
|---|---|---|
| 当前 | 0.5 × (0.05 / 0.35²) = 0.204 | 15% (绑) |
| 同周期 | 0.5 × (0.05 / 0.0292) = 0.856 | 15% (绑) |

弱信号 (μ_60d = 0.005)：

| 算法 | f_frac | 输出 (max_pct=0.15) |
|---|---|---|
| 当前 | 0.5 × (0.005 / 0.35²) = 0.020 | **2.0%** |
| 同周期 | 0.5 × (0.005 / 0.0292) = 0.086 | **8.6%** |

→ 弱中信号被压得非常厉害；强信号区被 `max_position_pct` 截掉，看不到区别。
这跟用户主观感受一致：**少数强信号能开到 cap，多数中等信号被缩得很小，
合在一起就是"总仓位低、cash 高"**。

---

## 3. NGBoost 现状

当前 4 个 named regime (`BULL_CALM`, `BULL_VOLATILE`, `CHOPPY`, `BEAR`) 全部
`regime_params.<R>.ngboost.enabled = false`
(`golden.json:262, 288, 313`)，原因 (2026-05-25 Codex contract repair)：

> 全局 NGBoost 已经在 E55 失败后关了 (NGB-on 输给 NGB-off)。这个 stale
> regime overlay 还激活了一个 train_run_id 错配的旧 NGBoost head，先关
> 着，等同 run NGBoost head 通过 strict WF/regime evidence 再开。

含义：**今天的 Kelly σ 100% 走 realized-vol fallback**。所以 σ-horizon
bug 影响 100% 的 Kelly 决策；如果将来 NGBoost 回来，需要分别确认
NGBoost head 是同周期还是年化输出 σ。

---

## 4. 其他可能压低仓位的层（每个都不动，但要点名）

### 4.1 fractional = 0.5 (half-Kelly)

行业惯例，吸收 μ 估计误差 + log-utility variance drag。**不建议动这个**：
half-Kelly 已经是"激进端"的标准实践 (1/4 才是经典 Thorp)。比 σ-horizon
fix 更激进的改动应该等 §7 A/B 给完结果再讨论。

### 4.2 `min_edge = 0.0`

不绑（任何正 μ 都过）。**没问题**。

### 4.3 `max_concentration = 0.35`

只在强信号 + σ 很低（年化 < 5%）时绑。**没问题**，是风控硬顶。

### 4.4 每 regime `max_position_pct`

| Regime | max_position_pct | 含义 |
|---|---|---|
| BULL_CALM | 0.15 | "accumulation" 的保守 cap |
| BULL_VOLATILE | 0.20 | 高波动开仓 |
| CHOPPY | 0.15 | 震荡限仓 |
| BEAR | 0.00 | 全空 (PRIME DIRECTIVE) |

§1 PRIME DIRECTIVE 的核心 (per-regime conviction)，**不建议改 cap**，
但可以观察 σ-horizon fix 之后绑没绑、绑的频率。

### 4.5 每 regime `cash_reserve_pct`

| Regime | cash_reserve_pct |
|---|---|
| BULL_CALM | **0.0** |
| BULL_VOLATILE | **0.2** |
| CHOPPY | **0.3** |
| BEAR | **1.0** |

**BULL_CALM 现金下限是 0**——所以"现金高"在 BULL_CALM 不是 reserve
导致的，是 Kelly target 加起来不够 100%（除去 turnover ramp 已经到位
的部分）。**BULL_VOLATILE / CHOPPY 的现金下限本身就高**，这两个 regime
里"cash% 高"有相当部分是 by design，不要全算在 Kelly 头上。

### 4.6 `qp_turnover_max`

BULL_CALM = 0.15 (2026-05-23 Codex churn fix)。每天 QP L1 turnover
≤ 15% NAV。即使 Kelly 一次性说"目标 invested 80%"，也得多天 ramp。这
是 BULL_CALM 里 cash% 高的次因：**新仓位是 ramp-in 的，瞬时看 cash%
偏高是正常的**。如果 σ-horizon 修完之后 Kelly target 普遍升 4×，
turnover_max 会变成新的 binding constraint —— 这一点要在 A/B sim 里
独立看一下，必要时引入 §7.4 Tier 3 决定是否调。

### 4.7 `top_up_threshold = 0.05`

TopUpHeld 不动手除非 Kelly target − 当前 weight ≥ 5pp。当 Kelly target
本身就只有 2-8% 的时候（见 §2.3），这个阈值容易把 incremental top-up 都
吃掉。σ-horizon 修完之后 Kelly target 上去了，5pp 阈值的阻力会自然下降。

### 4.8 `min_model_score` (buy gate)

BULL_CALM 0.10 / BULL_VOLATILE 0.15 / CHOPPY 0.15 / BEAR 0.0 — panel-LTR
rank_score 的开仓底。如果今天 panel-LTR 输出大量 score < 0.10 的候选，
即使 Kelly 算得对也开不了仓。**这是另一条独立 hypothesis**，不在本审
计范围；如有需要单独发一份 buy-gate 命中率 audit。

### 4.9 calibrator μ-clip 在 ±0.20

`global_calibrator.py:255–276` 在 load site 把 expected_return.y 钳在
[-0.20, +0.20]。**60 天 20% 是非常高的 horizon excess**，正常 panel-LTR
输出极少触顶——所以这个 clip 几乎不绑。**不动**。

---

## 5. Cash% 的"会计学" 分解

观察等式（在 §3 NGBoost-off 的事实下）：

```
1.0  =  cash_observed
     +  Σ_i  weight_i
     +  pending_settle_pct
     +  in_flight_orders_pct  (今天 QP 已经下单但还没成交的部分)
```

每个 regime 的"理论上的 cash% 下限"分解：

| Regime | 下限 cash | 主要驱动 |
|---|---|---|
| BULL_CALM | ~0% (reserve 0)；实际取决于 N × `max_pos × kelly_realized` | σ-horizon 后看 |
| BULL_VOLATILE | ≥ 20% | reserve 硬下限 + Kelly 削顶 |
| CHOPPY | ≥ 30%, ≤ 4 positions | reserve + position-count cap |
| BEAR | 100% | by design (`max_position_pct=0`) |

在 BULL_CALM ——也是用户最可能感受到"现金过高"的 regime——
σ-horizon bug 是几乎可以独立解释的主因。

---

## 6. 现有未关的相关 finding

从 `doc/AUDIT_2026-05-12_dead_paths.md` 和 `doc/AUDIT_2026-05-09.md`:

- **B5 vol-target / B6 DD-Kelly dead-path**：2026-05-15 P0 cleanup 已经从
  `ApplyKellySizingTask` 的 local-variable 路径里删掉了。Live exposure
  scaling 现在走 `kernel/portfolio_qp/tasks.py::ApplyExposureScalingTask`
  写 `ctx._vol_target_scale` / `ctx._dd_kelly_scale` 进 QP 上界。这条
  路径已经迁好，**不在本 audit 重新打开**。
- **calibrator P0 (2026-05-15)**：之前 `mu_none=33` 让 Kelly 全 0，QP
  退化成 uniform 10.17% sizing。已经修完，`use_calibrator_mu=true`
  + train-site/load-site clip 已经生效。本 audit 假设 μ side 是干净的；
  σ side 是新的发现。

---

## 7. Recommendation

**不要直接翻 production**（§7.4：no live flip without Tier 3 evidence）。
按以下序列：

### 7.1 加配置项（PR1，本 audit 不含 — 后续 follow-up PR）

```json
"ranking": {
  "kelly_sizing": {
    "...": "...",
    "sigma_horizon_days": 252,
    "_sigma_horizon_days_reason": "Default 252 keeps byte-equivalent
        behavior with the legacy `_realized_vol_annualized` σ output.
        Set to `panel_ltr.lookahead_days` (60) to match the Kelly
        period to the μ horizon, restoring f* = μ/σ² same-period
        consistency."
  }
}
```

在 `ApplyKellySizingTask.run` 里，在调 `kelly_target_pct` 之前：

```python
sigma_horizon_days = float(kelly_cfg.get("sigma_horizon_days", 252.0))
if sigma_horizon_days != 252.0:
    sg_f = sg_f * math.sqrt(sigma_horizon_days / 252.0)
```

byte-equivalent 默认 (252 ≡ 当前)，opt-in 才生效。

### 7.2 27-month OOS sim A/B

- A = golden (sigma_horizon_days 缺省 / 252)
- B = sigma_horizon_days = 60 (match μ horizon)

§7.2 mandatory triad：A/A re-split + shuffle-placebo + time-shift-placebo。
§7.3：≥ 5 seeds，报 `mean ± std`。§9：DSR + PBO。

### 7.3 Per-regime 评估 (§1 PRIME DIRECTIVE)

BULL_CALM / BULL_VOLATILE / CHOPPY 三个 regime 各自独立看 APY、Sharpe、
MaxDD、cash%、Kelly distribution。**不能** 只看 pooled mean —— §1.3 红线。

预期 (mechanism prediction)：
- BULL_CALM：Kelly target 中位数从 ~3% 升到 ~12%，cash% 从今天的 50%+
  降到 < 20%（reserve 是 0）。APY 升、MaxDD 升（因为仓位高了）、
  per-regime Sharpe 应该升（如果 calibrator μ 是真信号；§7.2 placebo
  必须先证否假信号）。
- BULL_VOLATILE：Kelly 升 4× 但被 `max_position_pct=0.20` 和
  `cash_reserve_pct=0.20` 上下围住，effect 较小。
- CHOPPY：类似 BULL_VOLATILE，effect 更小。

### 7.4 §7.4 promotion

只有 Tier 3 (Tier 2 + DSR > 0.5 或 PBO < 0.5) 才翻 golden。
中间任何一步 Sharpe 没升、placebo 也升 → 回到 §7.12 audit (theory
predicted X, result ¬X → 第一个怀疑是 implementation bug)，先不要否
定 hypothesis。

### 7.5 不在本次 audit / A/B scope 的事

- 不动 `fractional` (0.5 已经是激进端)。
- 不动 `max_concentration` (0.35 风控硬顶)。
- 不动 任何 `regime_params.<R>.max_position_pct` 或
  `cash_reserve_pct` — §1.5 promotion 是 per-regime conditional。
- 不动 `qp_turnover_max` —— σ-horizon 修完之后这条可能变成新的
  binding constraint，再单独审。
- 不开 NGBoost —— 等 §3 提到的 same-run NGBoost head 通过 WF/regime
  evidence。

---

## 8. Open data items（本 audit 没拉的数）

- [ ] 最近 30 天 `ApplyKellySizingTask` log 里 `cands non-zero / total`、
      `holdings non-zero / total`、`avg kelly%` 的实际分布
      （`grep "ApplyKellySizingTask" logs/live_e2e/`）。
- [ ] `sim_orders` / `decision_trace` 表里 `kelly_target_pct` 的
      per-regime 直方图。
- [ ] `live_state.alpaca.json` / `live_state.alpaca_shadow.json` 里
      最近 30 个 bar 的 cash% 时间序列。
- [ ] candidate-side `c.sigma` 实际分布（年化口径下中位数应该 ≈ 0.30，
      尾部 ≈ 0.50）。

这些数据点不挡 §7.1 follow-up PR——加配置项是 σ-horizon hypothesis 的
最小落地，sim A/B 出结果之前不动 golden。但落 follow-up PR 之前应该
先把这 4 条 grep 一下，验证"实际 Kelly target 中位数确实只有 2-5%"
这个假设是真的。

---

## 9. Audit verdict

- **是的，Kelly 当前过于保守。** 主因是 σ-horizon 错配，把 60d μ 喂给
  年化 σ²，让 Kelly target 系统性偏小 ~4× (`252/60 = 4.2`)。
- 这是单源 bug：`_realized_vol_annualized` 与 `qp_mu_horizon_days = 60`
  之间没有 horizon rescale。
- 修复方案是**加配置项 + opt-in rescale + A/B sim + Tier 3 gate**，
  不是直接翻 production。
- 其他 7 个可能压低仓位的层（fractional / min_edge / max_concentration /
  per-regime cap / cash_reserve / turnover / top_up_threshold）**本 audit
  不动**；σ-horizon fix 是阻塞最小、机理最干净的第一步。
- BEAR 100% cash 是 §1 PRIME DIRECTIVE，不属于"过于保守" 的范畴。

---

Agent-Origin: Claude

🤖 Generated with [Claude Code](https://claude.com/claude-code)
