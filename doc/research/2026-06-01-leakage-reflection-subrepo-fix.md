# 2026-06-01 · B_tuned 泄漏反思 + Multirepo 架构修复（v3 — multirepo architect frame）

**Author**: Claude  
**Reviewer**: Codex（v1 → v2 review HIGH×2 + MED×4，已 address）  
**Context**: B_tuned PatchTST Tier-3 placebos 二次失败（shuffle ≈ real ≈ timeshift ≈ +0.04 IC）。  
**Frame**: 不是单点 bug。不是"加道墙"。是 **13 仓库 multirepo 架构里没有 contract ownership + cross-repo CI gate + 版本协调协议**。

**v3 关键差异 vs v2**: v2 是"代码该长什么样"。v3 是"13 个 repo 该如何分工 + 谁拥有契约 + 跨仓 CI 怎么拓扑 + 版本怎么协调升 + 灾难时怎么回滚"。

---

## 1 · 反思：过程上做错了什么（unchanged from v2，省略 detail）

参见 §6 changelog 末尾。核心 4 条 violations：
- §6.4 (reuse evidence before compute) — 跑 2h 没看盘上 partial result
- §7.11 (no-run path first) — 5/31 没修根因就 6/01 重跑
- §8.1 (concept not code-label) — 报 ETA 不报"shuffle ≈ real → 泄漏"
- §7.2 (sanity discipline) — 实验既已显示 placebos pass，还在等 verdict

---

## 2 · 数据 + 真凶类别（unchanged from v2）

```
              s42      s43      s44
real          0.061    0.041     —
shuffle       0.014    0.041    0.048
timeshift     0.041    0.049     —
BEAR shuffle  0.048    0.091    0.084     ← 应 ≈ 0
```

### 2.1 这是哪一类泄漏（codex finding 1 已 address）

| Triad tier | 测什么 | 成本 | 这次 B_tuned 中招的是哪个 |
|---|---|---|---|
| **Scorer-level sanity** | 固定已训 scorer，喂 shuffled/timeshifted val labels | 秒级 | ❌ 测了也没用 |
| **Trainer-level placebo** | **重训**于 shuffled/timeshifted labels，再 eval | 75min/trial × 3 | ✅ 今天就是这类 |

---

## 3 · 13-仓库 Multirepo 架构 — 当前 vs 应该

### 3.1 当前依赖图 + 漏洞清单

```
                            ┌─────────────────────────────┐
                            │  RenQuant (umbrella)        │
                            │  - scripts/                  │
                            │  - backtesting/renquant_104/ │
                            │  - data/*.parquet            │ ← BAD: 178-列混合 parquet
                            └────────────┬─────────────────┘
                                         │ pd.read_parquet (bare)
                                         ↓
   ┌──────────────────────┬────────────────────────┬─────────────────────────┐
   │ renquant-base-data   │ renquant-model-{patchtst,gbdt,linear} │ renquant-strategy-104
   │ (data builders)      │ (trainers)                            │ (config, configs/)
   └──────────┬───────────┴───────────┬────────────┴─────────────────────────┘
              │                       │
              ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓
                       renquant-common
                       (contracts, shared primitives)
                       │
                       └─→ FeatureFrame? NO
                       └─→ TriadReport?  NO
                       └─→ DatasetManifest? NO
                       
   ┌─ renquant-pipeline ─┐   ┌─ renquant-backtesting ─┐   ┌─ renquant-orchestrator ─┐
   │ (decision pipeline) │   │ (LEAN + sim + wf_gate) │   │ (cron + retrain wiring) │
   │ - scorer.load()     │   │ - manifest_row(...)    │   │ - build_*_wf_manifest   │
   │ - NO triad gate     │   │ - NO triad gate        │   │ - NO triad gate         │
   └─────────────────────┘   └────────────────────────┘   └─────────────────────────┘
                                                            │
                                  ┌─ renquant-execution ────┴────────┐
                                  │ (Alpaca / IBKR live broker)      │
                                  │ - reads manifests w/o gate check │
                                  └──────────────────────────────────┘
```

**Multirepo 漏洞清单**（不是代码 bug，是架构层）：

| # | 漏洞 | 当前症状 |
|---|---|---|
| **M1** | **`renquant-common` 没有 data contract ownership** | FeatureFrame/LabelFrame/DatasetManifest 不存在；每个 model repo 自己定义 `_excluded` 列表（§7.5 violation × 3 处） |
| **M2** | **没有 cross-repo CI gate**: base-data 写 parquet → model 不验证 schema | 加 feature 不需要 update 任何 model repo 就能写，下游靠 dtype heuristic 推断 |
| **M3** | **artifact 契约没有"validated"状态字段** | scorer artifact 仅是 model.pt + metadata.json；triad_status 不是 typed field |
| **M4** | **pipeline / backtesting / orchestrator / execution 4 个下游消费 artifact，但都没 fail-closed gate** | unvalidated artifact 可以一路从训练流到 live broker |
| **M5** | **版本协调没标准化** | 今天 widening sweep 是 ad-hoc 4 个 PR (`<0.7 → <0.8`)；没有"common 加新契约 → 所有 sibling 必须 bump pin"的 enforced workflow |
| **M6** | **没有 ownership matrix 明文化** | 谁决定 FeatureFrame schema 加列？谁批 ScorerArtifact schema 改？现在没人，结果谁都能加 |
| **M7** | **Agent automation labels 没覆盖 contract PR** | 改 contract 的 PR 不会自动触发跨仓影响审查 |

### 3.2 应该是的依赖图

```
                              ┌──────────────────────────┐
                              │    renquant-common       │   ← Single owner of:
                              │  - contracts/data.py     │     - FeatureManifest
                              │  - contracts/triad.py    │     - LabelManifest
                              │  - contracts/scorer.py   │     - DatasetManifest
                              │  - leakage_guards/*.py   │     - ScorerArtifact (incl. triad_status)
                              │                          │     - TriadReport (Scorer + TrainerPlacebo)
                              │  Semver: MAJOR for       │   ← changes are coordinated across all consumers
                              │  contract breaking,      │
                              │  MINOR for additive      │
                              └────────────┬─────────────┘
                                           │ EVERY consumer pins to a version range
                       ┌───────────────────┼───────────────────┬─────────────────┐
                       ↓                   ↓                   ↓                 ↓
              ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
              │ renquant-      │  │ renquant-      │  │ renquant-      │  │ renquant-      │
              │ base-data      │  │ model-*        │  │ pipeline       │  │ strategy-104   │
              │                │  │                │  │                │  │                │
              │ writes to      │  │ writes         │  │ READS          │  │ reads cfg.json │
              │ disk:          │  │ ScorerArtifact │  │ ScorerArtifact │  │                │
              │ - features.pq  │  │ with required  │  │                │  │                │
              │ - labels.pq    │  │ triad_status   │  │ fail-closed    │  │                │
              │ - splits.pq    │  │ ∈ {pending,    │  │ if triad_status│  │                │
              │ - manifest.json│  │   passed,      │  │ ≠ "passed"     │  │                │
              │                │  │   failed}      │  │                │  │                │
              │ all validate   │  │                │  │ at scorer.load │  │                │
              │ against        │  │ Tier 1 blocks  │  │                │  │                │
              │ DatasetManifest│  │ save sync.     │  │                │  │                │
              │                │  │ Tier 2 async   │  │                │  │                │
              │                │  │ via runner.    │  │                │  │                │
              └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘
                       │                   │                   │                    │
                       └───────────────────┼───────────────────┘                    │
                                           │                                        │
                       ┌───────────────────┼────────────────────────────────────────┘
                       ↓                   ↓
              ┌────────────────┐  ┌────────────────┐
              │ renquant-      │  │ renquant-      │
              │ backtesting    │  │ orchestrator   │
              │                │  │                │
              │ manifest_row() │  │ build_*_wf_    │
              │ refuses        │  │ manifest()     │
              │ scorers w/     │  │ refuses scorer │
              │ triad ≠ passed │  │ w/o triad gate │
              └────────┬───────┘  └────────┬───────┘
                       │                   │
                       └────────┬──────────┘
                                ↓
                       ┌────────────────┐
                       │ renquant-      │
                       │ execution      │   ← live broker side
                       │                │     LAST line of defense:
                       │ refuses orders │     refuses orders from
                       │ from scorers   │     scorers without triad
                       │ w/o triad      │     even if upstream failed
                       └────────────────┘
```

**5 道闸门**（每一道独立 fail-closed）：

| Gate | 在哪 | 拦什么 | 防什么 |
|---|---|---|---|
| G0 | `renquant-base-data` 写 features.parquet 时 | DatasetManifest validate (feature_cols ∩ label_cols = ∅) | label 列偷进 feature parquet |
| G1 | `renquant-model-*` trainer save 时 | Tier 1 scorer sanity → fail = no save | label-calc bug |
| G2 | `renquant-model-*` Tier 2 async runner | shuffle/timeshift placebo retrain → stamps triad_status | train-time leakage (今天这次) |
| G3 | `renquant-pipeline` scorer.load() | refuse if triad_status ≠ "passed" | unvalidated artifact 进 sim |
| G4 | `renquant-backtesting` + `renquant-orchestrator` manifest_row | refuse scorer without passing triad in walk-forward manifest | unvalidated artifact 进 prod cron |
| G5 | `renquant-execution` live broker | refuse orders sourced from manifest w/ failed/missing triad | last-line defense if upstream gates bypassed |

任何 ONE gate fail → unvalidated artifact 进不去 live broker。**今天 B_tuned 这次能 5/31 + 6/01 两次跑到 placebo 失败还没 fail-closed，是因为 G1-G5 全部不存在。**

---

## 4 · Multirepo Architect 决策 — 7 个明文协议

### 4.1 Contract ownership matrix（M6 fix）

| Contract | Owner repo | Consumer repos | Bump 谁批 |
|---|---|---|---|
| `FeatureManifest`, `LabelManifest`, `SplitManifest`, `DatasetManifest` | renquant-common | base-data (writer), model-* (validator), pipeline/backtesting (eval consumer) | Architect (人) |
| `ScorerArtifact`, `TriadReport`, `ScorerSanityReport`, `TrainerPlaceboReport` | renquant-common | model-* (writer), pipeline/backtesting/orchestrator/execution (reader, fail-closed) | Architect (人) |
| `FeatureFrame.from_parquet()`, `LabelFrame.from_parquet()` validators | renquant-common | all trainer entrypoints | Architect (人) |
| Per-strategy threshold values (DSR > 0.5, n ≥ 30, etc.) | renquant-strategy-104 | backtesting promotion gate | Strategy lead (人) |
| Trainer-level placebo runner CLI | renquant-common | invoked by model-* trainers post-save | Architect (人) |

**Forbidden patterns**:
- ❌ model-patchtst 自定义 `TriadReport` schema → must use common's
- ❌ base-data 加 feature 而不 bump common 的 `FeatureManifest` schema version → CI block
- ❌ pipeline 加自己的 `triad_status` 枚举 → must import common's Literal

### 4.2 Cross-repo CI 拓扑（M2 fix）

```yaml
# renquant-common 加新 contract → 触发 fan-out check
.github/workflows/contract-bump-check.yml:
  on:
    push:
      paths: ['src/renquant_common/contracts/**', 'src/renquant_common/leakage_guards/**']
  jobs:
    fan_out:
      strategy:
        matrix:
          consumer: [renquant-base-data, renquant-model-patchtst, renquant-model-gbdt,
                     renquant-pipeline, renquant-backtesting, renquant-orchestrator,
                     renquant-execution]
      steps:
        - checkout this PR's common branch
        - git clone consumer at main
        - pip install -e ../renquant-common && pip install -e .  # consumer
        - pytest tests/  # consumer test suite must still pass
      # If any consumer fails → block this PR
```

```yaml
# base-data 写 parquet 时跑 schema check
renquant-base-data/.github/workflows/dataset-schema-check.yml:
  on: pull_request
  steps:
    - regenerate one small sample dataset (1 ticker, 100 days)
    - load via FeatureFrame.from_parquet(sample, manifest)  # raises if validation fails
    - assert no column in features.parquet matches label naming convention
```

```yaml
# model-* save artifact → CI runs Tier 1 sanity inline
renquant-model-patchtst/.github/workflows/triad-tier1.yml:
  on: pull_request
  steps:
    - train on tiny synthetic dataset (60s)
    - assert ScorerArtifact.triad_report.scorer_sanity is populated
    - assert triad_status in {"pending", "passed"}
    - if triad_status == "failed" → block PR
```

```yaml
# pipeline/backtesting/orchestrator → load test refuses unvalidated artifacts
renquant-{pipeline,backtesting,orchestrator}/.github/workflows/artifact-gate.yml:
  steps:
    - synthesize artifact with triad_status="pending"
    - assert Scorer.load() raises ArtifactNotValidated
```

### 4.3 版本协调协议（M5 fix）

**Trigger**: `renquant-common` PR 加新 contract 字段（additive）或改 contract（breaking）。

**Procedure** (newly enforced via `contract-bump-check.yml`):

| 改动类型 | semver | 协调步骤 |
|---|---|---|
| **Additive**（加字段，旧消费者 backwards compatible） | MINOR (0.7.0 → 0.8.0) | 1. 开 common PR；fan-out check 自动跑所有 consumer 测试。2. 跑通 → merge。3. consumer 任意时机 pin bump 升级到新最低版本（rolling）。 |
| **Breaking**（删字段，改 enum 值） | MAJOR (0.x.y → 1.0.0) | 1. **必须先**开 N+1 个 PR：common 的 PR 标 `agent:contract:breaking`；同时开 N 个 consumer PR pin bump + 必要代码改动。2. 所有 consumer PR CI 全绿 = condition for common merge。3. **Stacked merge** by orchestrator script (`scripts/coordinated_merge_contract_bump.sh`)：按顺序 (common → base-data → model-* → pipeline → backtesting → orchestrator → execution) merge，每一步 verify 下一步 CI 仍绿。4. 任何一步 fail → revert stack。 |
| **Hotfix**（紧急回滚 contract change） | PATCH (0.7.0 → 0.7.1) | 1. Architect 直接 commit common 0.7.1（revert 0.7.0 的 contract change）。2. fan-out check 自动跑。3. 消费者 pin 范围 `>=0.6,<0.8` 自动覆盖（这就是为什么 pin 总是放宽到 next major）。 |

**Why 4.1 + 4.2 + 4.3 加起来叫"协议"而不是"流程"**: 因为 CI 拦死它。Architect 文档建议 "记得 pin bump" 没用 — `contract-bump-check.yml` workflow 直接 fail the PR if any consumer broken。

### 4.4 迁移序列（M1+M3+M4 fix — MVP 优先 per codex finding 4）

```
[Day 0 现在]
    │
    ↓ MVP wave (5 PRs，目标 ≤ 2 天)
    │
    ↓ ① renquant-common: contracts/triad.py + leakage_guards/* (Tier 1 + Tier 2 runner)
    │   → adds: ScorerArtifact.triad_report field (additive, MINOR bump)
    │   → trigger: contract-bump-check fan-out
    │   
    ↓ ② renquant-common: artifact validation helpers + ArtifactNotValidated exception
    │
    ↓ ③ renquant-model-patchtst + renquant-model-gbdt: wire Tier 1 (block save) + Tier 2 (async)
    │
    ↓ ④ renquant-pipeline: scorer.load() fail-closed on missing/pending/failed
    │
    ↓ ⑤ renquant-backtesting + orchestrator: manifest_row refuses unvalidated
    │   (renquant-execution gate ⑥ deferred — backtesting/orchestrator already cuts off most paths)
    │
[Day 2 — MVP complete]
    │   ✅ unvalidated artifact 进不去 prod cron / live broker
    │   ✅ 今天的 incident class fully blocked
    │
    ↓ Full architecture wave (4 PRs，目标 ≤ 1 周)
    │
    ↓ ⑥ renquant-common: contracts/data.py (FeatureManifest etc.) + FeatureFrame.from_parquet
    │   → MAJOR bump if existing FeatureFrame removed; MINOR if additive
    │
    ↓ ⑦ renquant-base-data: refactor builders to write 3 parquet + manifest.json
    │
    ↓ ⑧ renquant-model-*: typed train(features: FeatureFrame, ...) entrypoint
    │   → DELETE load_panel_with_split in same PR (per codex finding 2)
    │   → ruff rule: pd.read_parquet forbidden in trainer modules
    │
    ↓ ⑨ renquant-execution: G5 last-line broker-side gate
    │
[Day 7 — full architecture deployed]
```

### 4.5 灾难回滚 + escape hatch（M3 corollary）

**Scenario**: Tier 2 trainer placebo runner broke / takes too long → 所有 fresh artifacts stuck in `pending` → live trading stops 因为 G4/G5 拒绝加载。

**Escape hatches** (architect 必须设计上 — 否则 G3-G5 是 production single point of failure):

1. **`agent:emergency:bypass-triad` PR label** + 单 contract（owner = Architect 人）：在 RenQuant umbrella `strategy_config.golden.json` 加 `"emergency_bypass_triad_until": "2026-06-15T00:00:00Z"`。downstream gates **不是** unconditional fail-closed — 而是 "if bypass date set AND not expired AND artifact has `triad_status=pending` (not failed) → load with warning"。
2. **Bypass 必须有 expiry** — 不能 forever-on（防止"临时"变永久）
3. **Bypass 触发 alert** — slack + memory note，每次绕过都记账
4. **`failed` 绝对不可 bypass** — escape hatch 只覆盖 `pending`（runner stuck）；`failed`（placebo > real）任何时候 hard-stop

代码骨架:
```python
def _allow_load(artifact: ScorerArtifact, cfg: dict) -> bool:
    if artifact.triad_report.triad_status == "passed":
        return True
    if artifact.triad_report.triad_status == "failed":
        return False  # hard stop, no bypass
    # pending
    bypass_until_str = cfg.get("emergency_bypass_triad_until")
    if not bypass_until_str:
        return False
    bypass_until = pd.Timestamp(bypass_until_str)
    if pd.Timestamp.now() > bypass_until:
        return False
    log.warning("triad bypass active — loading PENDING artifact (expires %s)",
                bypass_until_str)
    record_bypass_event(artifact.artifact_fingerprint, bypass_until_str)
    return True
```

### 4.6 Agent automation 与 contract PR 的交互（M7 fix）

新 label scheme（**在 RenQuant + 所有 13 仓加 label**，via existing repo-config script）：

| Label | 在哪 | 触发什么 |
|---|---|---|
| `agent:contract:additive` | renquant-common PR adding fields | fan-out check workflow + auto codex review request |
| `agent:contract:breaking` | renquant-common PR removing/changing fields | fan-out check + REQUIRE N consumer PRs linked + REQUIRE architect (human) sign-off via specific GitHub UI button — agents cannot self-merge |
| `agent:emergency:bypass-triad` | umbrella PR setting emergency_bypass_triad_until | requires architect (人) review + auto-expires reminder cron |
| `agent:multirepo-review` | any PR touching ≥ 2 repo's contract surface | auto request both Claude + codex review |

### 4.7 Test surface 跨仓（CI gate 总成）

```
                   ┌─────────────────────────────────────┐
                   │ renquant-common                     │
                   │ tests/                              │
                   │  - test_contracts.py (Pydantic round-trip)
                   │  - test_leakage_guards.py (synth-data)
                   └────────────────┬────────────────────┘
                                    │ contract-bump-check.yml triggers ↓
        ┌───────────────┬───────────┼────────────┬──────────────┐
        ↓               ↓           ↓            ↓              ↓
   ┌────────────┐ ┌────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ base-data  │ │ model-*    │ │ pipeline │ │ backtest │ │ orchest. │
   │ tests/     │ │ tests/     │ │ tests/   │ │ tests/   │ │ tests/   │
   │ - dataset- │ │ - triad-   │ │ - artif- │ │ - manif. │ │ - manif. │
   │   schema-  │ │   tier1.py │ │   gate.  │ │   gate.  │ │   gate.  │
   │   check.py │ │ - triad-   │ │   py     │ │   py     │ │   py     │
   │            │ │   tier2-   │ │          │ │          │ │          │
   │            │ │   async.py │ │          │ │          │ │          │
   └────────────┘ └────────────┘ └──────────┘ └──────────┘ └──────────┘

EACH consumer test imports common's contracts, runs a synthesized
artifact path (write → load → verify gate behavior), no real model.
Test wall clock ≤ 30s per repo.

E2E test in umbrella (RenQuant tests/test_multirepo_triad_e2e.py):
spawn synthetic feature/label parquet via base-data → tiny xgboost
train via model-gbdt → assert artifact has triad_report → load via
pipeline → assert backtesting manifest_row records it → assert
execution refuses if triad_status pending.
```

---

## 5 · v1 → v2 → v3 changelog

| Codex review finding | v2 | v3 multirepo lens |
|---|---|---|
| 1: triad 分层 | ✅ Scorer vs Trainer split | + §3.2 G1 vs G2 是 5 道 cross-repo gate 中的两道，明文 owner=common |
| 2: Function wall runtime | ✅ from_parquet + delete unsafe | + §4.1 ownership: from_parquet 在 common owner；§4.2 ruff rule 在每个 model repo CI |
| 3: Schema manifest 不是 regex | ✅ FeatureManifest Pydantic | + §3.2 G0 写时 validate；§4.7 base-data tests/dataset-schema-check.py |
| 4: MVP first | ✅ 5+4 PR split | + §4.4 Day-by-day 序列；MVP 5 PR 跨 4 个 repo 明文（common × 2 + model × 1 + pipeline × 1 + backtesting × 1） |
| 5: async + pending state | ✅ Literal[passed,pending,failed] | + §4.5 escape hatch + emergency bypass 协议 + alerting；pending 不 forever-stuck |
| 6: PR #9 保留 | ✅ defense-in-depth | + §3.2 G2 内部使用 boundary guard 作为 Tier 2 runner 的 sanity invariant |
| User pushback "我要 multirepo architect" | — | + §3.1 漏洞清单 M1-M7（架构层不是代码层）；§4.1 ownership matrix；§4.2 cross-repo CI 拓扑；§4.3 版本协调协议（additive/breaking/hotfix 流程）；§4.5 灾难回滚；§4.6 agent label scheme；§4.7 跨仓 test 总成 |

---

## 6 · 索引

- `[[project_patchtst_btuned_leakage_2026-05-31]]`
- `[[feedback_research_pipeline_must_gate_with_sanity_triad]]` — advisory 不是 executable，本架构 §3.2 G3 是 executable 化
- `[[feedback_industry_leading_quality]]` — 拒绝 patch-sim-patch loop
- `[[feedback_leakage_three_walls]]` — v2 摘要（v3 显著扩展为 5 gates + multirepo）
- `[[feedback_pr_based_workflow]]` — §4.6 label scheme 接续这条 multirepo 协议
- `[[feedback_multirepo_code_placement]]` — §4.1 ownership matrix 是这条的硬化
- `[[project_multirepo_sop_2026-05-28]]` — §4.3 版本协调协议 supersedes
- CLAUDE.md §3.1 (PR workflow), §3.5 (multirepo code placement), §7.5 (single source), §7.7 (gate ≠ decoration)
