# 2026-06-01 · B_tuned 泄漏反思 + Subrepo 架构修复（v2 — addressed codex review）

**Author**: Claude  
**Context**: B_tuned PatchTST Tier-3 placebos 再次失败（shuffle ≈ real ≈ timeshift ≈ +0.04 IC）。  
**Frame**: 不是单点 bug — subrepo 架构的契约缺失。  
**Revision history**: v1 → v2 after codex strict review (HIGH×2 + MED×4)。详见 §6 changelog。

---

## 1 · 反思：过程上做错了什么

### 1.1 跑了一整天没看数据
- **5/31 22:52** 启动 v1 BG（5h40min 跑完，最终 invalid_experiment，path bug）
- **6/01 09:05** 启动 v2 BG（codex PR #22 路径修后），预期 6h
- **6/01 11:05** 已有 5 个 trial 完整结果（s42 三种 + s43 real/shuffle）写到 `trial_result.json`，**我一直没看，只盯进度条**
- 用户问了两次"有数据了么？"我都答"还在跑"
- 直到用户第三次说"你确定实验科学有效吗？"我才去 `cat trial_result.json` — 答案就在 2 小时前的盘上

**违反**: CLAUDE.md §6.4 "Reuse existing evidence before spending compute"  
**违反**: §7.2 "新数字第一行 log 前必须跑 sanity triad，看结果" — 没看 partial 结果就是默认接受  

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

### 2.1 这是哪一类泄漏（codex review #1）

⚠️ **重要区分**（v1 没说清楚，codex 指出）：

| 类型 | 怎么测 | 成本 | 这次 B_tuned 中招的是哪个 |
|---|---|---|---|
| **Scorer-level sanity** | 固定已训练 scorer，喂 shuffled/timeshifted val labels，看 IC | 秒级 | ❌ 这个测了也没用 |
| **Trainer-level placebo** | **重新训练**于 shuffled/timeshifted labels，再 eval | 同训练成本（~25min × 3 placebos = 75min） | ✅ 这才是今天 5/31+6/01 都看到的 placebo ≈ real |

**Scorer-level 只能拦 "label 计算错误"** —— 例如算 IC 时算反方向。  
**Trainer-level 才能拦 "训练过程偷看 val label"** —— 也就是今天这次。

Architecture 必须显式区分两者，不是"一个 triad 一刀切"。

---

## 3 · Subrepo 架构上的结构性修复（v2 — codex 反馈已 addressed）

### 3.1 当前架构的"为什么允许泄漏"

```
RenQuant (umbrella)
  └─ data/transformer_v4_wl200_clean.parquet   ← 178 列混合: features + 4 labels + split_label
     ↓ pandas.read_parquet（裸读）
  renquant-model-patchtst
     └─ hf_trainer.load_panel_with_split()
          ├─ _excluded = {"fwd_5d_excess", ..., "split_label", ...}  ← blacklist regex
          ├─ feat_cols = [c for c in panel.columns if c not in _excluded and dtype.kind in "fiub"]
          └─ trainer.train(panel[feat_cols], panel[label_col])  ← 裸 DataFrame
```

**结构性漏洞**：
- (a) 一个 parquet 同时存 features + labels + split — 加 feature 时靠 reviewer 记得加 `_excluded`
- (b) `feat_cols` 是 runtime 推断的（`dtype.kind in "fiub"`），不是 schema 验证的
- (c) val IC callback / EarlyStopping 在训练中**读 val labels** —— 没契约禁止
- (d) `split_label` 列存在 features parquet 里 —— dataset 已"知道"哪些日子是 val
- (e) 同一 `_excluded` 过滤规则 base-data / model / pipeline 3 处实现 = §7.5 违反

### 3.2 三道墙（v2 修订 — codex finding 1/2/3 已 address）

#### 墙 1 · Schema wall（**positive manifest 而非 regex blacklist**）

**v1 错的地方**（codex finding 3）: 用 regex `r"(fwd_|future_|next_|excess|target)"` 当主要 wall。会漏 `y` `label` `ret_5d` `return_20d` 这类别名，也会误伤含 `excess`/`target` 的合法 feature 名。

**v2 正确做法**: **declared manifest**，不是黑名单。

```python
# renquant-common/src/renquant_common/contracts/data.py

class FeatureManifest(pydantic.BaseModel):
    """Positive declaration of every feature column.

    Adding a column to base-data without registering it here = write fails.
    """
    feature_cols: list[str]                # closed list
    feature_dtypes: dict[str, str]         # name → dtype (float32/int8/...)
    schema_hash: str                       # SHA256 of (sorted feature_cols, dtypes, lookahead)
    lookahead_metadata: dict[str, int]     # for each feature, max forward bars touched
                                           # (e.g., a rolling std with window 60 uses 60 PAST bars,
                                           # lookahead = 0; an "all-history quantile" has lookahead = INF)

class LabelManifest(pydantic.BaseModel):
    label_cols: list[str]                  # e.g., ["fwd_5d_excess", "fwd_60d_excess"]
    label_lookahead_days: dict[str, int]   # required per label

class SplitManifest(pydantic.BaseModel):
    split_dates: dict[str, pd.Timestamp]   # train_end, val_start, val_end
    embargo_days: int

class DatasetManifest(pydantic.BaseModel):
    features: FeatureManifest
    labels: LabelManifest
    splits: SplitManifest

    @model_validator(mode="after")
    def features_labels_disjoint(self):
        overlap = set(self.features.feature_cols) & set(self.labels.label_cols)
        if overlap:
            raise ValueError(f"feature ∩ label = {overlap}; declared sets must be disjoint")
        return self
```

Regex blacklist 仍存在 — 但只是 **defense in depth** layer，不是主要 wall。

#### 墙 2 · Function wall（**runtime validators, not just type annotations**）

**v1 错的地方**（codex finding 2）: 说"typed `train(features: FeatureFrame, ...)` 阻止 callers" — Python 类型注解 **不阻止运行时调用**。bypass door 永远存在如果 `pd.read_parquet` + 自选列还可用。

**v2 正确做法**: **runtime constructor** + **删除/标记 unsafe_ 旧入口**。

```python
# renquant-common/src/renquant_common/contracts/data.py

class FeatureFrame:
    """DataFrame wrapper validated against a FeatureManifest.

    Construction MUST go through .from_parquet() — never bare DataFrame init.
    """
    def __init__(self, df: pd.DataFrame, manifest: FeatureManifest):
        self._validate(df, manifest)
        self._df = df
        self._manifest = manifest

    @classmethod
    def from_parquet(cls, path: Path, manifest_path: Path) -> "FeatureFrame":
        df = pd.read_parquet(path)
        manifest = FeatureManifest.parse_file(manifest_path)
        cls._validate(df, manifest)  # schema_hash check, dtypes match, columns ⊆ declared
        return cls(df, manifest)

    @staticmethod
    def _validate(df, manifest):
        actual_cols = set(df.columns) - {"ticker", "date"}
        declared = set(manifest.feature_cols)
        extra = actual_cols - declared
        if extra:
            raise ValueError(f"parquet has columns not in manifest: {extra}")
        # ... dtype checks, schema_hash recompute, etc.
```

```python
# renquant-model-patchtst/src/renquant_model_patchtst/hf_trainer.py

# DELETED in same PR wave (P0):
#   def load_panel_with_split(dataset_path: Path, ...) -> tuple[pd.DataFrame, list[str]]

# REPLACED with typed entrypoint:
def train(
    features: FeatureFrame,
    labels: LabelFrame,
    splits: SplitAssignment,
    args: TrainArgs,
) -> ScorerArtifact:
    ...

# Old DataFrame path:
#   - Either deleted in same P0 wave (preferred), OR
#   - Renamed `_unsafe_train_from_panel(...)` + module-level DeprecationWarning + CI
#     check that orchestrator/CLI does NOT call `_unsafe_*` entrypoints.
```

**`pd.read_parquet` 在 trainer 模块禁止** — 通过 ruff/grep CI 规则（lint rule）。

#### 墙 3 · Contract wall（**两类 triad + async validation state**）

**v1 错的地方**（codex finding 1 + 5）: 说 "trainer 必须跑 triad before save" — 但 (a) 把 scorer-level 和 trainer-level triad 混为一谈，(b) 同步 triad 让每个 PatchTST 训练多花 75min，cripples iteration。

**v2 正确做法**:

```python
class TriadReport(pydantic.BaseModel):
    """Two distinct triad tiers, both required for a complete report."""

    # Tier 1 — Scorer-level (cheap, must always run, blocks save)
    # Perturbed labels evaluated AGAINST FIXED SCORER. Catches label-calc bugs.
    scorer_sanity: ScorerSanityReport  # required
    #   - shuffled_val_ic (≈ 0 expected)
    #   - timeshifted_val_ic (≈ 0 expected)
    #   - aa_split_real_ic_replicate

    # Tier 2 — Trainer-level placebo (expensive, async-allowed, blocks promotion)
    # Re-train under shuffled/timeshifted labels. Catches train-time leakage
    # (callbacks reading val labels, val data leaking through preprocessing, etc.)
    trainer_placebo: TrainerPlaceboReport | None = None  # may be pending
    #   - shuffle_placebo_real_ic
    #   - shuffle_placebo_min_regime_ic
    #   - timeshift_placebo_real_ic
    #   - timeshift_placebo_min_regime_ic
    #   - PASSED iff abs(placebo_ic) < 0.01 AND abs(placebo_ic) < 0.3 * real_ic

    # Status binding
    artifact_fingerprint: str   # sha256 of (model bytes, feature schema, label hash, code sha, triad config)
    triad_status: Literal["passed", "pending", "failed"]
    triad_completed_at: datetime | None
```

**`ScorerArtifact.triad_report: TriadReport`** — Pydantic 必填字段（schema_hash 失配 = `scorer.load()` raise）。

**Save flow**:
1. Training finish → run **Tier 1** (scorer sanity, seconds) → must pass to save
2. Save artifact with `triad_status="pending"`, `trainer_placebo=None`
3. **Async** post-save: trigger trainer-level placebo runner (own subprocess or queue)
4. Placebo runner writes updated `triad_status` to artifact's sidecar metadata
5. **Downstream gate** (pipeline / backtesting / orchestrator):

```python
# renquant-pipeline/src/renquant_pipeline/kernel/panel_pipeline/panel_scorer.py
def load(uri: str) -> Scorer:
    artifact = ScorerArtifact.parse_file(uri + ".metadata.json")
    if artifact.triad_report.triad_status != "passed":
        raise ArtifactNotValidated(
            f"refusing to load scorer with triad_status={artifact.triad_report.triad_status}"
        )
    # ... bind to scorer
```

**`pending` ≡ `failed` for promotion purposes** — orchestrator / WF manifest / sim all refuse `pending`.

#### 墙 0 · 保留 PR #9 cross-split-leak guard（codex finding 6）

**v1 没说清**: 似乎 PR #9 boundary guard 在新架构下"过时了"。  
**v2 显式承认**: 不过时。PR #9 是 **lower-level invariant** —— split embargo 在任何 timeshift placebo 实现里都是 hard invariant。新 3 道墙是 **defense in depth**，不替代 PR #9。

---

## 4 · 修订后的 PR 路径（v2 — MVP first per codex finding 4）

### 4.1 MVP（**P0 — 5 个 PR ≤ 2 天**）

**目标**: 立刻能拦今天这次 leak class（artifact 没跑 trainer-level placebo 就不许进 production）。**不**等 split-parquet 重构。

| # | Repo | 内容 | 估算 |
|---|---|---|---|
| 1 | `renquant-common` | `contracts/triad.py`: `ScorerSanityReport` / `TrainerPlaceboReport` / `TriadReport` Pydantic models + `triad_status` Literal | 1h |
| 2 | `renquant-common` | `leakage_guards/scorer_sanity.py`: 实现 scorer-level triad（喂 shuffled/timeshifted val labels 对已训 scorer）—— 秒级运行 | 半天 |
| 3 | `renquant-common` | `leakage_guards/trainer_placebo_runner.py`: 实现 trainer-level placebo subprocess runner（async, queue-able） | 半天 |
| 4 | `renquant-model-patchtst` + `-gbdt` | 训练 finish 时强制跑 Tier 1（scorer sanity）→ save；Tier 1 fail → raise，不 save。Tier 2（trainer placebo）触发 async runner，artifact 初始 `triad_status="pending"` | 半天 |
| 5 | `renquant-pipeline` + `renquant-backtesting` + `renquant-orchestrator` | `Scorer.load()` + manifest_row + sim_driver fail-closed if `triad_status != "passed"` | 半天 |

**MVP 完成后**：今天这次（shuffle placebo ≈ real）会在第 4 步 trainer placebo runner 完成时把 artifact 标 `triad_status="failed"`，pipeline 拒绝加载，sim 拒绝出单。**架构上 unvalidated scorer 不可能进 production。**

### 4.2 完整 architecture（**P1 — 跟 MVP 解耦的下一波**）

| # | Repo | 内容 | 估算 |
|---|---|---|---|
| 6 | `renquant-common` | `contracts/data.py`: `FeatureManifest` / `LabelManifest` / `SplitManifest` / `DatasetManifest` Pydantic models + `FeatureFrame.from_parquet()` runtime validator | 半天 |
| 7 | `renquant-base-data` | Refactor alpha158/transformer panel builder: 输出 3 个 parquet + DatasetManifest.json 而不是 1 个混合 parquet | 1 天 |
| 8 | `renquant-model-patchtst` + `-gbdt` | typed trainer entrypoint `train(features: FeatureFrame, labels: LabelFrame, ...)`; 删除 `load_panel_with_split` 或标 `_unsafe_*`; ruff rule 禁止 trainer 模块 `pd.read_parquet` | 1 天 each |
| 9 | CI | 跨 repo workflow: base-data parquet 写完 → 自动跑 DatasetManifest validation + features ∩ labels = ∅ 检查 | 1 天 |

### 4.3 不在本架构内的事

- PR #9 cross-split-leak boundary guard 保留（defense in depth）
- 任何 "researcher exploratory" 训练 — 允许 `triad_status="pending"`，但 artifact 在 `artifacts/research/` 路径，**不进 prod/sim**
- Promotion (Tier 3) 要求 `triad_status="passed"` AND 现有 DSR/PBO/n≥30 gates

---

## 5 · 这次的下一步（不等架构修完）

1. **写一个一次性的 `leakage_diagnostic.py`** （在 renquant-model 仓里，标记 P0 临时工具）：
   - 加载 B_tuned 6 个 ok trial 结果
   - 算 ratio = shuffle_IC / real_IC + timeshift_IC / real_IC（per-regime）
   - 输出 `doc/research/leakage_2026-06-01/diagnostic_report.md`
2. **不再跑 B_tuned 训练**直到 §4.1 MVP 落地（specifically 第 4 + 5 PR）。重跑同 config 不会变结论
3. 把这份反思 + memory 写进 commit；让下次会话第一时间读

---

## 6 · v1 → v2 changelog（codex review 落实）

| Codex finding | 级别 | v2 修订 |
|---|---|---|
| 1: triad 分两层 (scorer-level 秒级 vs trainer-level 75min) | HIGH | §2.1 显式区分；§3.2 墙 3 拆 `ScorerSanityReport` + `TrainerPlaceboReport` |
| 2: Function wall 需要 runtime validator + 删除旧 entry | HIGH | §3.2 墙 2 改成 `FeatureFrame.from_parquet()` 构造器 + 删除 `load_panel_with_split` + ruff CI ban `pd.read_parquet` in trainers |
| 3: Schema 是 blacklist regex，应是 declared manifest | MED | §3.2 墙 1 改成 `FeatureManifest` 显式声明 + `model_validator` disjointness check；regex 降级到 defense in depth |
| 4: 7-PR plan 太大，要 MVP first | MED | §4.1 MVP（5 个 PR ≤ 2 天）和 §4.2 完整重构解耦 |
| 5: Async validation state — `pending|passed|failed` | MED | §3.2 墙 3 引入 `triad_status` Literal + Tier 2 async + `pending ≡ failed` for promotion |
| 6: PR #9 boundary guard 不过时 | MED | §3.2 墙 0 显式保留 |

Codex 问题答案（也在 §3-4 reflected）：
- Q1 (3-wall 分解对吗): 方向对，但 Function wall 必须 runtime 强制，不能只靠类型注解
- Q2 (跨 disk 边界保持 typed): `FeatureFrame.from_parquet(path, manifest)`，trainer 禁止 raw `pd.read_parquet`
- Q3 (旧 DataFrame entry): 同 P0 wave 删除/标 `_unsafe_*`
- Q4 (75min triad 成本): save as `pending` + async + 下游 reject pending
- Q5 (PR #9 是否过时): 不过时，保留
- Q6 (smaller MVP): §4.1 — 5 个 PR，只要 artifact 状态 + 下游 reject 就够拦今天的 incident
- Q7 (memory rule 为啥没拦住): rule 是 advisory 不是 executable —— gap 是 missing machine-checked artifact gate。MVP 第 5 个 PR 就是填这个 gap。

---

## 7 · 索引

- `[[project_patchtst_btuned_leakage_2026-05-31]]` — 5/31 当时的发现
- `[[feedback_research_pipeline_must_gate_with_sanity_triad]]` — 6/01 上午定的契约（codex 指出："advisory 不是 executable"，本架构第 5 个 PR 是把它变成 executable）
- `[[feedback_industry_leading_quality]]` — 拒绝 patch-sim-patch loop（这次反思印证）
- `[[feedback_leakage_three_walls]]` — 本架构的 3 道墙摘要（同样 v2 修订）
- CLAUDE.md §7.2 (sanity discipline) / §7.5 (single source) / §7.7 (gate ≠ decoration) / §7.10 (canonical refs) / §7.12 (audit-before-accept)
