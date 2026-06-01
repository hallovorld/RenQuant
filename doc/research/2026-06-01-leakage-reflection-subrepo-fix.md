# 2026-06-01 · B_tuned 泄漏反思 + Subrepo 架构修复

**Author**: Claude  
**Context**: B_tuned PatchTST Tier-3 placebos 再次失败（shuffle ≈ real ≈ timeshift ≈ +0.04 IC）。  
**Frame**: 这不是单点 bug — 是 subrepo 架构的契约缺失。

---

## 1 · 反思：过程上做错了什么

### 1.1 跑了一整天没看数据
- **5/31 22:52** 启动 v1 BG（5h40min 跑完，最终 invalid_experiment，path bug）
- **6/01 09:05** 启动 v2 BG（codex PR #22 路径修后），预期 6h
- **6/01 11:05** 已有 5 个 trial 完整结果（s42 三种 + s43 real/shuffle）写到 `trial_result.json`，**我一直没看，只盯进度条**
- 用户问了两次"有数据了么？" 我都答"还在跑"
- 直到用户第三次说"你确定实验科学有效吗？"我才去 `cat trial_result.json` — 答案就在 2 小时前的盘上

**违反**: CLAUDE.md §6.4 "Reuse existing evidence before spending compute"  
**违反**: §7.2 "新数字第一行 log 前必须跑 sanity triad，看结果" — 我没看 partial 结果就是默认接受  

### 1.2 重复了 5/31 的实验，没修根因就再跑
- 5/31 已有结论：B_tuned timeshift placebo (+0.069) > real (+0.044)
- 当时归因为 cross-split-leak（PR #9 修了 boundary 处的 placebo shift）
- 但 **PR #9 只修了 timeshift 边界**，shuffle placebo 是另一类泄漏，没动过
- 6/01 重跑等于"换个 commit hash 看是否换个结果"——不是诊断

**违反**: §7.11 实验设计决策树 "诊断'为什么 Y 失败' → 先 §7.2 sanity 序列，再 mechanism"  
**违反**: §6.3 "No-run path first" — 应该先 grep / 算清楚泄漏路径，再决定要不要重跑  

### 1.3 报告语言含糊，让用户必须戳破才看真相
- 我每次都说"v2 跑完会出 verdict"——但 verdict 是 `aggregate_results.json`，partial 结果（5 个 trial）已经够下结论
- 用户问"卡了吗"，我回"~3-4h ETA"——但 ETA 不重要，**已有数据足以否定**
- 用户问"科学有效吗"，本来 placebo ≈ real 一行就能定调，但我先列了一堆 ETA 表

**违反**: §8.1 "状态报告用概念，不用代码标签" — ETA 是代码标签，"shuffle ≈ real → 泄漏" 才是概念

---

## 2 · 真凶（数据层面）

```
              s42      s43      s44
real          0.061    0.041     —
shuffle       0.014    0.041    0.048      ← random labels 仍 IC > 0
timeshift     0.041    0.049     —         ← shifted labels 仍 ≈ real
```

BEAR regime 上 shuffle placebo IC = +0.091（**应 ≈ 0**）。  
**结论：模型在 val 上的预测能力与 train 标签无关。** 泄漏不在 boundary（PR #9 修了），不在显式 fwd 列（hf_trainer 的 `_excluded` 已剔除）。

候选根因（这次我没追到底，但架构修复独立于具体根因）：
1. **EarlyStopping 用 val loss 选 checkpoint** → 即使 train labels 是 random，patience=2 的 val-driven selection 在 5-seed average 上会偏向 IC > 0（lottery ticket effect）
2. **PerRegimeICCallback 计算 val IC** → 如果用作 metric_for_best_model，模型 selection 本身就用 val 信号
3. **CSRankNorm + winsorize** 在 panel 全量（含 val）上 fit 边界值，跨日 quantile 泄漏
4. **HMM regime label** 是 features 之一，detector 是否用未来 60 天信息标 BEAR

但这些都是 **症状**。架构问题是**没有一道墙强制隔离这些泄漏路径**。

---

## 3 · Subrepo 架构上的结构性修复

### 3.1 当前架构的"为什么允许泄漏"

```
RenQuant (umbrella)
  └─ data/transformer_v4_wl200_clean.parquet   ← 178 列混合: features + 4 labels + split_label
     ↓ pandas.read_parquet
  renquant-model-patchtst
     └─ hf_trainer.load_panel_with_split()
          ├─ _excluded = {"fwd_5d_excess", "fwd_20d_excess", "fwd_60d_excess", "split_label", ...}
          ├─ feat_cols = [c for c in panel.columns if c not in _excluded]
          └─ trainer.train(panel[feat_cols], panel[label_col])
```

**结构性漏洞**：
- (a) 一个 parquet 同时存 features + labels + split — 任何新 feature 加进去要靠 reviewer 记得加到 `_excluded`，没有类型墙
- (b) `feat_cols` 是 runtime 推断的（`dtype.kind in "fiub"`），不是 schema 验证的
- (c) val IC callback 直接读 `val_dataset.labels` 计算 metric，没有"训练过程不能触及 val labels"的契约
- (d) split_label 列也存在 features parquet 里，意味着 raw dataset 已经"知道"哪些日子是 val
- (e) 三个 subrepo（base-data 生产，model 消费，pipeline 评估）都自己手写"过滤掉 fwd 列"——同一规则 3 处实现 = §7.5 违反

### 3.2 应该是的架构

```
renquant-common
  ├─ contracts/
  │   ├─ FeatureFrame    (Pydantic, dtype-validated)
  │   │     - MUST NOT have any column matching r"(fwd_|future_|next_|excess|target)"
  │   │     - MUST have index (ticker, date)
  │   │     - schema_hash stamped into artifact
  │   ├─ LabelFrame      (Pydantic)
  │   │     - holds {label_col_name: Series} only
  │   │     - lookahead_days declared, validated against label_col regex
  │   ├─ IndexFrame      ((ticker, date) only)
  │   └─ SplitAssignment (train/val/test labels — IS NOT in FeatureFrame)
  │
  └─ leakage_guards/
      ├─ assert_features_label_disjoint(features, labels)
      │     → raise if any feature column matches label naming convention
      ├─ assert_no_future_lookup(features, max_lookahead_days)
      │     → temporal causality contract on FeatureFrame
      ├─ run_sanity_triad(scorer, X_train, y_train, X_val, y_val)
      │     → REQUIRED to be called before reporting any IC.
      │       Returns triad_report{aa_passed, shuffle_ic, timeshift_ic}.
      │       If shuffle_ic > 0.5 * real_ic OR timeshift_ic > 0.5 * real_ic
      │       → raise LeakageDetected
      └─ EvalContract     (Pydantic)
            - Every evaluation report MUST embed triad_report
            - reports without triad fail-closed at consumer (renquant-pipeline)

renquant-base-data
  ├─ produces:
  │   ├─ features.parquet  (validates FeatureFrame on write)
  │   ├─ labels.parquet    (validates LabelFrame on write)
  │   └─ splits.parquet    (validates SplitAssignment on write)
  └─ NEVER produces a unified "panel.parquet" mixing the three.

renquant-model-{patchtst,gbdt,linear}
  ├─ trainer interface signature (typed):
  │       train(
  │           features: FeatureFrame,
  │           labels: LabelFrame,
  │           splits: SplitAssignment,
  │       ) -> ScorerArtifact
  ├─ Pydantic type-check enforces: trainer CANNOT receive labels.fwd_*
  │   columns as features (different table; type system rejects mix at boundary)
  ├─ Trainer MUST call run_sanity_triad before persisting artifact.
  │   ScorerArtifact has a required field `triad_report: TriadReport`.
  └─ EarlyStopping / metric_for_best_model are wired through a
       LeakageAwareCallback that LOGS which labels it touches; CI gates that
       no train callback reads val labels.

renquant-pipeline
  ├─ consumer of ScorerArtifact
  └─ fail-closed at scorer.load() if artifact.triad_report.shuffle_ic > threshold
    or .timeshift_ic > threshold
    or .triad_report is missing.

renquant-backtesting / orchestrator
  ├─ build_walkforward_manifest_row(): refuses to include retrains
       where triad_report is missing or failed.
  └─ Stamps triad pass/fail into manifest, exposed to sim + live.
```

### 3.3 三道墙

| Wall | 在哪 | 防什么 |
|---|---|---|
| **Schema wall** | `renquant-common` Pydantic models + base-data 写时 validate | label 列偷偷进 feature parquet — 这次 5/31 没发现就是因为没墙 |
| **Function wall** | trainer 签名 `train(features: FeatureFrame, labels: LabelFrame, ...)` | trainer 把 labels parquet 当 features 读 — Python type checker 就拦住 |
| **Contract wall** | `ScorerArtifact.triad_report` 必填 + 下游 fail-closed | 模型偷偷不跑 triad / triad 失败但产物仍发布 — pipeline 拒绝加载 |

### 3.4 具体 PR 路径（按优先级）

| # | Repo | 内容 | 优先级 | 估算 |
|---|---|---|---|---|
| 1 | `renquant-common` | 加 `contracts/FeatureFrame`/`LabelFrame`/`SplitAssignment` + `leakage_guards/run_sanity_triad` | **P0** | 半天 |
| 2 | `renquant-common` | `ScorerArtifact.triad_report` 必填字段（Pydantic） | **P0** | 1h |
| 3 | `renquant-model-patchtst` + `renquant-model-gbdt` | trainer 签名改成 typed `(features, labels, splits)`；call run_sanity_triad before save | **P0** | 1 天 each |
| 4 | `renquant-base-data` | 改 alpha158/transformer panel builder：输出 3 个 parquet 而不是 1 个 | **P1** | 1 天 |
| 5 | `renquant-pipeline` | scorer loader fail-closed on missing/failed triad_report | **P1** | 半天 |
| 6 | `renquant-backtesting` | manifest_row refuses scorers without passing triad | **P1** | 半天 |
| 7 | CI | 跨 repo workflow：base-data parquet 写完 → 自动跑 FeatureFrame schema validation | **P2** | 1 天 |

每个 PR 都遵守 §3.1 PR-based workflow + §3.4 review protocol + §7.7 安全 gate 必须有 cron 真正执行（不只是 test）。

---

## 4 · 这次的下一步（不等架构修完）

1. **写一个一次性的 `leakage_diagnostic.py`** （在 renquant-model 仓里，标记 P0 临时工具）：
   - 加载 B_tuned 的 6 个 ok trial 结果
   - 算 ratio = shuffle_IC / real_IC + timeshift_IC / real_IC（per-regime）
   - 输出 `doc/research/leakage_2026-06-01/diagnostic_report.md`
2. **不再跑 B_tuned 训练**直到 §3.2 三道墙落地。重跑同 config 不会变结论
3. 把这份反思 commit 到 `doc/research/` + 把"placebo ≈ real → 泄漏" 进 memory

---

## 5 · 索引

- `[[project_patchtst_btuned_leakage_2026-05-31]]` — 5/31 当时的发现
- `[[feedback_research_pipeline_must_gate_with_sanity_triad]]` — 6/01 上午定的契约（这次架构修复正是它的落地）
- `[[feedback_industry_leading_quality]]` — 拒绝 patch-sim-patch loop（这次反思印证）
- CLAUDE.md §7.2 (sanity discipline) / §7.5 (single source) / §7.7 (gate ≠ decoration) / §7.10 (canonical refs) / §7.12 (audit-before-accept)
