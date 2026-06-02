# 实验有效性审计 + 检讨 — 2026-06-02

**触发事件**: 2026-06-01 20:19 PT, B_tuned Tier-3 placebo 重跑后 verdict =
`invalid_experiment`. `timeshift_placebo IC (+0.067) > real_ic (+0.044)`
— 教科书级泄漏特征. 2026-05-31 修复的 cross-split-leak guard
(renquant-model commit
[`7245d84`](https://github.com/hallovorld/renquant-model/commit/7245d84))
不是真正泄漏源.

**结论**: 这不是 harness bug. 是模型本身泄漏. 最近所有以 B_tuned 为
基线的"改进"实验, **无法被信任**.

---

## 0. Canonical evidence (codex #80 #3)

Verdict files live in the renquant-model checkout. They are NOT
git-tracked (the `artifacts/patchtst_research/` tree is in the repo's
`.gitignore` — that's by design; experiment artifacts are workstation-
local). Filesystem paths + sha256 below are reproducible.

| Field | Value |
|---|---|
| Experiment dir | `~/git/github/renquant-model/artifacts/patchtst_research/tier3_doe_postfix_20260601-174726/20260602T004728Z_doe_250a6ec1_edd7c5b38ff5/` |
| `invalid_experiment.json` sha256 | `a59dd13276015022977c036f69c29dee8a0f5ef23d82c7ba39540f4b44b02a9e` |
| `placebo_gate.json` sha256 | `90febb52ac8c28dbdeb8e767805fb1a23b180e3b38534af5b2c716944b4bee7f` |
| renquant-model HEAD at run time | `1128911` (`fix: resolve patchtst runtime paths explicitly`) |
| Cross-split-leak fix that did NOT close the leak | `7245d84` (`fix(hf_trainer): timeshift placebo MUST NOT cross split boundary (Tier-3 root cause)`) — landed 2026-05-31 11:33 PT |
| Run command | `python -m renquant_model_patchtst.research --phase doe --configs B_tuned --cuts cut1_covid,cut2_fed --seeds 42,43 --epochs 4 --device mps --out-dir artifacts/patchtst_research/tier3_doe_postfix_20260601-174726` |
| Run launch log | `~/git/github/renquant-model/logs/btuned_tier3_postfix_20260601-174726.log` |

Key fields read from `invalid_experiment.json`:

```json
{
  "verdict": "invalid_experiment",
  "placebo_gate": {
    "shuffle_placebo":   {"hard_gate": true,  "passed": false, "real_ic_mean": 0.0437, "placebo_ic_mean": 0.0307, "threshold": 0.0109},
    "timeshift_placebo": {"hard_gate": true,  "passed": false, "real_ic_mean": 0.0437, "placebo_ic_mean": 0.0668, "threshold": 0.0218},
    "aa_split":          {"hard_gate": false, "passed": true,  "reason": "alternate cut/seed evidence present"}
  },
  "regime_contract": {"passed": true, "detector_version": "v2026-05-31"}
}
```

### Task ID disambiguation

The "Task #N" references throughout this document refer to the **Claude
TaskList** (the runtime task state surfaced by the Claude CLI's
`TaskCreate` / `TaskUpdate` tools — not git/repo state) — they are NOT
GitHub PR or issue numbers. The TaskList isn't persistently checked
into the repo; its contents are stamped into recent commit-message
trailers and into [`MEMORY.md`](../../../.claude/projects/-Users-renhao-git-github-RenQuant/memory/MEMORY.md)
pointer lines. Distinct namespaces in this doc:

- `Task #17` / `Task #30` / `Task #49` / `Task #53` — Claude TaskList IDs.
- `PR #43` / `PR #48` / `PR #57` / `PR #80` etc. — GitHub PR numbers in
  `hallovorld/RenQuant`.
- `pipeline #10` / `backtesting #22` / `renquant-common #5` — GitHub PRs
  in the named subrepo.

When a future agent needs to look up a Claude TaskList ID and finds
nothing in the repo, the source is the active Claude session's
TaskList tool surface, not the git tree.

---

## 1. 哪些实验现在被宣布无效

下表按严重程度排序. 用法: 这些数字都不能当作 "已验证的事实"; 在
泄漏闭环之前, 都需要打 `PRE-LEAK-AUDIT` 标记 OR 删除 OR 重跑.

| 严重度 | 实验 / 主张 | 来自哪里 | 为什么无效 |
|---|---|---|---|
| 🚨 P0 | **Phase 2D B_tuned 基线 `+0.1476` pooled IC** | `[[project_patchtst_btuned_leakage_2026-05-31]]` | Phase 2D 跑时未启用 placebo; 现 6-01 confirm 是泄漏 inflate 出来的. |
| 🚨 P0 | **Task #17 "Improve PatchTST: tuned config + cross-stock attn + FiLM"** | TaskList #17 | 所有 FiLM / cross-stock attn 改进相对 B_tuned 测的; 基线泄漏意味着 delta 没意义. |
| 🚨 P0 | **`[[project_patchtst_signal_validated_2026-05-29]]` "placebo-clean val IC ~+0.066"** | 5-29 memory | 那时用的不是 codex Tier-3 harness; 不同 placebo 实现 — 现在用更严格的版本 IC 跌到 +0.044 且 timeshift placebo 跑赢. 5-29 的 "通过" 应改成 "看似通过但被更严格的 6-01 harness 否决". |
| 🚨 P0 | **Task #49 + #53 "B_tuned full-panel Tier-3 re-run"** (我标记成 completed) | TaskList #49, #53 | BG 跑完 ≠ task 完成. verdict 是 invalid_experiment. 我把 "process ran" 误当成 "the experiment passed". 是 §6.4 ("Reuse existing evidence before spending compute") 的反向违规 — 我以为重跑就完事了. |
| ⚠️ P1 | **Task #32 W1 DLinear/NLinear "falsification test for Transformer line"** | TaskList #32 | 想用 Linear 基线证明 Transformer 必要性. 但 Transformer (PatchTST) 测出来的 IC 本身是泄漏的, 所以 "Transformer 更好" 的对比没意义. Linear 基线是干净的 (我相信), 但比较的另一边脏. |
| ⚠️ P1 | **Task #33 + #34 PatchTSMixer 基线** | TaskList #33, #34 | 同样基于 hf_trainer.load_panel_with_split, 同样的 seq_len=24 sequence boundary 假设. 极有可能同源泄漏. |
| ⚠️ P1 | **`config_fingerprint: sha256:14586756d4f67691` 在 prod 跑的 panel-ltr 数字** | `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` | 这是 XGBoost prod 模型 (不是 PatchTST). 但 promotion_status=`gated_buys` 的 reason 已经说: `WF gate FAIL sharpe -1.323 vs SPY +1.081; §5.2 placebo FAIL`. 这一条本来就被诚实标记成不可信; 不是新发现, 但应当在审计文档中显示出来. |
| ⚠️ P1 | **Task #18 / #19 / #20 / #22 各类 "Bug C/D/E/G"** | TaskList | 都是基础设施修复 (per-fold 校准, 字节绑定, sidecar stamp, 分数压缩诊断). 修复本身合理. 但其中**任何引用 IC/Sharpe 数字作为证据**的部分都需要重测. |
| ✅ | 其它任务 (#24 kernel.* 提升, #27 regime_labels 提升, #28 detector 修复, #44 #45 common, #48 #51 #52 lift, #54 leakage architecture, 等) | TaskList | 纯架构 / lift / repo organization, 没声明任何 IC 数字 — 不受影响. |
| ✅ | 今天 PR #43 leakage architecture v12 + 23 falsifiers | merged | 防御设计本身正确; 唯一遗憾是没在 land 之后**立即**用它去 gate 已有的 B_tuned baseline. |
| ✅ | PR #48 / #57 / #65 / #69 / #74 daily observability + shadow stamper + calibrator refit | merged | 运营修复, 不涉及任何 IC 主张. 全部有效. |

## 2. 检讨 — 我具体错在哪里

### 2.1 把 "BG 跑完" 误当成 "实验通过"

Task #49 ("B_tuned full-panel Tier-3 re-run on post-PR-#9 main") 我标
`completed`. 但实际 verdict 是 `invalid_experiment`. 我看见 BG exit 0 +
有 trial outputs 就以为可以勾选了. **这是 §6.4 + §7.13 的复合违规**:
exit code 不等于 placebo 通过. 应该读 `placebo_gate.json` 拿到 verdict
才能 close task.

修复: TaskUpdate(status=completed) 必须先读 verdict file 并复述出关键数字
到 commit message 或 status report 里. "task ran" 不是 "task done".

### 2.2 建了 leakage 防御架构 (PR #43) 但没用它 gate 已有 baseline

PR #43 ship 了 G3 gate (artifact 必须带通过的 triad_report 才能 load).
但今天看下来, 实际的 PatchTST seed_44 sidecar 既没有 triad_report
也没有 G3-pass 证据 — 它能正常进 daily 是因为 PR #43 是 **新建** 路径,
**没有 retrofit 到现有 artifacts**. 等于建了一道门, 但没装在房子上.

**§7.7 ("safety gate ≠ enforced safety gate")** 的标准例子. 修复:
G3 enforcement 必须**早于**任何新训练 PR 落地. 否则新训练继续 inherit
旧 baseline 的不可信状态.

### 2.3 在已知 baseline 可能泄漏的情况下继续跑 "改进" 实验

`[[project_patchtst_btuned_leakage_2026-05-31]]` 5-31 就标记了
"baseline contaminated, Task #17 BLOCKED". 但 6-01 上半天我还在继续
跑 BG 验 fix-后状态 — 这没问题. 错的是: 我应该 **完整 audit 6 个泄漏
hypotheses** 才重跑, 而不是仅 fix #1 (cross-split-leak) 就 BG it.
**§6.3 ("no-run path first")**: 重跑前应当 audit 全部 6 个 hypothesis,
而不是 "fix 一个 + 跑跑看".

### 2.4 引用 IC 数字时没附带 placebo 状态

CLAUDE.md §7.3 (multi-measurement) 要求 mean±std. 但我多次在 status
report 引用 `oos_mean_ic=0.045` / `pool_ic=0.115` 这类单点 IC, 没附带
placebo verdict. **§7.2 ("sanity discipline")** 第 1 条: 每个新数字都
要带 §7.2 mandatory triad 至少一个 sanity check. 我没遵守.

### 2.5 5-29 "placebo-clean +0.066" 这条结论应该早就被怀疑

Memory `[[project_patchtst_signal_validated_2026-05-29]]` 写了
"placebo-clean val IC ~+0.066 (~3-4× XGB +0.017)". 5-31 codex
research harness 跑出 +0.044 + timeshift placebo +0.069 (现在 confirm
+0.067). 5-29 vs 5-31 两个不同的 placebo 实现给出截然相反的判断 — 我
应当**立刻**追问: 哪一个是对的, 而不是把 5-31 当 "新的更严格的版本" 就
继续往前走. 现在 6-01 verdict 同 5-31 ⇒ codex harness 是对的, 5-29
是错的, 但**两周之间没人追问过**. 这是 **§6.4 reuse evidence + §7.10
canonical references** 的复合疏忽.

## 3. 流程修改 (从今天开始执行)

| 修改 | 触发条件 | 强制动作 |
|---|---|---|
| **R1** | 任何 BG 实验任务 close | 必须先 `cat verdict.json` (或等价) 把 verdict + key numbers 复述到 TaskUpdate 的 description 字段; verdict ≠ `pass` 则强制保持 `in_progress`. |
| **R2** | 任何 PR 引用 IC / Sharpe / APY 数字 | PR body 必须含一个 "placebo verdict" 段: shuffle + timeshift + a/a, 三个里至少一个 with concrete numbers + threshold. 没附就请 codex 标 HIGH blocker. |
| **R3** | G3 gate (PR #43) 在 land 后 30 天内 | 必须 retrofit 所有已有的 prod + shadow artifacts. 没 triad_report 的 artifact 在 LoadScorerTask 应该 fail-closed (现在是 advisory). |
| **R4** | 任何 "leak hypothesis" 列表 | 必须把全部条目逐个 audit 完, 出**单条 audit log** 给每个 hypothesis (ruled-in / ruled-out / inconclusive + evidence), 再决定下一步实验. 不能"修 1 个跑跑看". |
| **R5** | Memory 之间互相矛盾时 (例如 5-29 vs 5-31 placebo verdict) | 必须在写第二条 memory 时显式 invalidate 第一条 OR 解释为什么两者不冲突. 不能两条并存. |

## 4. 修复路径 (Task #30 / #17)

按当前 6 个 hypothesis 排序, **逐个 audit 必须出 log**:

1. ❌ Cross-split leak in placebo harness — 已 ruled out (5-31 fix 落地, 6-01 重跑仍 fail)
2. **⭐ Sequence boundary (seq_len=24 PatchTST window 跨 train/val)** — 最物理的 timeshift_placebo > real 解释. 下一步重点.
3. PerRegimeICCallback regime injection — 中等优先, 看 callback 是否 leak regime label 进训练.
4. CSRankNorm NaN-fill 跨日 — 已审计代码看似 safe 但要 grep `fillna` 全部上下文.
5. Winsor bounds — 已审计 fit_mask=train, 复查无遗漏.
6. Dataloader / seed policy — 看 split_label 在 iteration 时是否真被 honored.

每个 hypothesis 给一个**专门的 audit memo** (短文档, 不是 code change)
说明 ruled-in / ruled-out + evidence. 然后**才**开新训练.

## 5. 哪些 PR / 任务现在受影响

| 状态 | 项 | 影响 |
|---|---|---|
| 撤回 task completion | Task #49, #53 | 应该是 `in_progress` 或 `deleted` — 它们的 "completed" 是误标. |
| 改 task description | Task #17, #32, #33, #34 | 加 "BLOCKED on Task #30 leak closure". 注意 #17 现在虽然标 completed, 但所有 IC 数字已无效. |
| 加 PRE-LEAK-AUDIT 标 | `doc/roadmap.md` § 1 "PatchTST sequence model" | 该节直接引用 B_tuned IC 数字, 加 warning banner 限制在 PatchTST 段内. |
| 不需要 banner — 文件 scoping 检查后无 B_tuned/PatchTST 主张 | `doc/experiments/panel-training-runs.md` (XGBoost Panel-LTR 训练日志, 与 PatchTST 是不同 pipeline), `doc/experiments/ab-journal.md` | codex review on PR #80 [(comment)](https://github.com/hallovorld/RenQuant/pull/80#issuecomment-4598656286) 正确指出: 给 XGBoost 日志加 "B_tuned 同一训练 pipeline" 的 banner 是错误归因, 会制造假 provenance chain. 这两个文件保持原样, 在本表显式记录其 scoping 已被审计过. |
| 新增 task | (待开) "G3 retrofit on existing PatchTST artifacts" | 让 PR #43 的 G3 gate 真的 enforce 在现有 artifact 上. |

## 6. 公开声明 (写给未来读这份文档的 agent + 自己)

**今天之前所有声称 "PatchTST 改进了 X" 的结论, 都应当在重读时打上引号
直到 Task #30 (leak source identification) 关闭.** XGBoost prod 线
不受影响 — 它独立标 `gated_buys` 已经是诚实标签. 受影响的纯粹是
PatchTST + 任何把 PatchTST 作为对比基线的实验.

这是个慢错误 — 5-29 / 5-31 两次都有信号说哪里不对劲, 我两次都没正面
处理. 这次 codex Tier-3 harness 把它定钉了. 接下来 Task #30 必须按
hypothesis-by-hypothesis audit 推进, 而不是再来一次 "fix 一个 + BG it".

---

**Files affected by this audit**: `doc/roadmap.md`, `doc/experiments/`,
`backtesting/renquant_104/artifacts/patchtst_shadow/*`, `[[memory]]`.

**Author**: Claude (admitting to systemic process violations).

**Reviewers requested**: Codex (please be strict — this is exactly the
class of self-deception the §7.2 sanity discipline is designed to catch).
