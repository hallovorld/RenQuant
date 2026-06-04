# 2026-06-03 — BEAR Defensive Sleeve Dead-Gate 审计（纯代码）

**触发**: cash-drag 根因调查的同步发现 —— sim 历史里 BEAR 日 0 笔 GLD/TLT/XLV/XLU 交易，配置却写着 30% 防御部署。
**方法**: **纯代码 + 文件系统审计**（不依赖 sim 数据时效）。用户在前一轮明确选了"路 1：代码审计优先"。
**Multi-repo**: 审计对象在 **renquant-pipeline**（`pp_inference._buy_universe`、`job_universe.LoadArtifactsTask`），config 在 **renquant-strategy-104**（`watchlist` / `defensive_tickers`）。umbrella 是 byte-equivalent 镜像。
**所有者**: Claude。

---

## 0. 先撤回前一轮的 stale-data 误判

前一轮（`2026-06-03-cash-drag-root-cause-and-fix.md` 的 §1.x）我用 `data/sim_runs.db` 下了"BEAR sleeve 死、4 个 bug"的结论。**那批 sim 数据全部生成于 2026-05-11（commit `8e55357`），3 周前**，而 `pipeline_runs.bear_only` 列是之后通过 schema migration 加的 → 旧行全是 NULL。

→ 撤回 **"Bug 4: bear_only 持久化漏写"** —— 当前 `sim.py:1547` 的 caller 已经传 `bear_only=bool(getattr(ctx,"bear_only",False))`，NULL 只是列后加的副作用。

违反了 CLAUDE.md §7.2（先 audit 数据 pipeline 再下结论）+ §7.12（unexpected result → audit before accepting）。本 memo 改用纯代码审计，结论不依赖任何 sim 数据时效。

---

## 1. 确定根因 —— `_buy_universe` 的 `t in ctx.models` 条件链

### 1.1 BEAR sleeve 的注入入口

`renquant_pipeline/kernel/pipeline/pp_inference.py::_buy_universe`（line 154-163）：

```python
if ctx.bear_only:
    defensives = set(ctx.config.get("defensive_tickers", []))
    return [t for t in defensives
            if t in ctx.models          # ← 关键过滤条件
            and t not in held
            and t != sleeve_ticker
            and t not in pending_at_broker and t in ctx.ohlcv]
```

defensive ticker 必须满足 **`t in ctx.models`** 才会被注入 BEAR buy universe。

### 1.2 `ctx.models` 怎么来 —— 只装 watchlist 里有 artifact 的

`job_universe.py::LoadArtifactsTask`（line 65）：

```python
for ticker in uctx.config.get("watchlist", []):   # ← 只 iterate watchlist
    art = load_artifact(models_dir / ticker, ticker)
    if art is None:
        uctx.rejections.append((ticker, "no_artifact"))
        continue
    uctx.loaded_models[ticker] = art               # = ctx.models
```

`ctx.models` 只包含 **(a) 在 watchlist 里 且 (b) 有 model artifact** 的 ticker。

### 1.3 配置实况（纯文件验证）

`strategy_config.golden.json`：

| defensive_ticker | model artifact 存在 | 在 watchlist |
|---|---|---|
| GLD | ✓ `models/GLD/GLD-qtable.json` | ✓ |
| **TLT** | ✓ `models/TLT/TLT-policy-metadata.json` | **❌** |
| **XLV** | ✓ `models/XLV/XLV-bin-edges.json` | **❌** |
| XLU | ✓ `models/XLU/XLU-bin-edges.json` | ✓ |

（`watchlist` size = 142；`defensive_tickers = [GLD, TLT, XLV, XLU]`。）

---

## 2. 确定 Bug —— TLT / XLV 永久不可买入

**Bug A（确定，零数据依赖）**:
`defensive_tickers` 列了 TLT、XLV，但它们**不在 watchlist**。链条：

```
TLT/XLV 不在 watchlist
  → LoadArtifactsTask 不 iterate 它们（即使 artifact 存在）
  → 不进 loaded_models / ctx.models
  → _buy_universe BEAR 分支 `t in ctx.models` == False
  → 永久从 BEAR buy universe 过滤掉
  → defensive sleeve 4 个 ticker 里 2 个结构性不可能被买入
```

`job_universe.py:210-213` 的注释自相矛盾：

> "`config.defensive_tickers` — they exist specifically to be available
> when the regime demands them (BEAR / bear_only branch). Filtering them
> out here would make BEAR buys structurally impossible."

但 TLT/XLV 在**更上游**的 `LoadArtifactsTask` watchlist-gate 就被挡了，根本到不了 `FilterUniverseFloorTask` 那个"豁免 defensives"。豁免是 dead code（豁免一个永远不在 `loaded_models` 里的东西）。这是 §7.7 教科书级 "configured but never enforced"。

---

## 3. 待验证（需 fresh sim，旧数据不可信）

**GLD / XLU 在 watchlist + 有 model**，所以理论上能进 BEAR buy universe —— 但能否真 fire 取决于：

1. **`ctx.bear_only` 是否被正确 set**：`BEARBranchTask` 要求 `confidence ≥ 0.60` 且非 transition。旧数据 77% BEAR 日 confidence≥0.60，但 `bear_only` 字段 NULL 不可信。
2. **GLD/XLU 是否被当 alpha 候选污染**：它们在 watchlist + model → 非-BEAR 分支（line 164）也把它们当普通 alpha 候选。旧数据 GLD 在 BULL_CALM 进候选 611 次、`blocked_by=kelly_zero:mu_none`、611 次只 selected 11 次 —— 说明 GLD 被当 alpha 候选（没 μ̂ → Kelly=0 → 基本不选）。这是设计混淆：同一个 ticker 既当防御 sleeve 又当 alpha 候选。

**这部分只能 fresh sim 回答** —— 不在本纯代码审计的确定结论范围内。

---

## 4. 设计层面的根本问题

BEAR defensive sleeve **复用了 alpha 的 model-gated universe** (`ctx.models`)，但防御 ETF 的本质跟 alpha 候选冲突：

| | alpha 候选 | 防御 ETF |
|---|---|---|
| 需要 panel-LTR model | 是（用 μ̂ 排序 + Kelly sizing） | 否（固定 % 部署，不需要预测） |
| 在 watchlist | 是 | 不一定（TLT/XLV 就不在） |
| sizing | Kelly `μ/σ²` | `bear_defensive_pct` 固定 % |

强行让防御 ETF 走 alpha 的 universe → 两头不讨好：
- 没 model 的（TLT/XLV）被 watchlist+`t in ctx.models` 挡死
- 有 model 的（GLD/XLU）被当 alpha 候选，`mu_none` 让 Kelly 给 0，固定-% sizing 路径（`bear_defensive_pct`）跟 Kelly sizing 打架

---

## 5. 分层 fix 方案（multi-repo 布局）

### Fix 1 — config drift（P0，低风险，确定）
**Repo**: renquant-strategy-104（canonical config owner）+ umbrella mirror。
**改动**: 解决 `defensive_tickers` 与 `watchlist` 的不一致。两个方向二选一：
- **(a) 收窄 defensive_tickers** 到 watchlist 内的 `[GLD, XLU]` —— 如果防御只用这俩。最小改动，立即消除 TLT/XLV 的死引用。
- **(b) 把 TLT/XLV 加进 watchlist** —— 但会让它们进入**所有 regime 的 alpha 候选**，污染 alpha 选择（§4 的混淆）。不推荐，除非配套 Fix 2。

推荐 **(a)** 作为即时止血，把"是否要 TLT/XLV 防御"作为 Fix 2 的设计输入。

### Fix 2 — defensive 独立 universe 路径（P1，需 A/B，设计）
**Repo**: renquant-pipeline（canonical）+ umbrella mirror。
**改动**: 给 defensive ETF 一条**独立于 alpha model 的注入路径**：
- `LoadArtifactsTask` 额外加载 `defensive_tickers`（即使不在 watchlist），标记 `role=defensive`
- `_buy_universe` BEAR 分支不要求 `t in ctx.models`，只要求 `t in ctx.ohlcv`（防御 ETF 只需要价格，不需要 model）
- sizing 走 `bear_defensive_pct` 固定 %，**绕过 Kelly**（防御 ETF 没 μ̂ 是正常的，不该被 `mu_none` 砍）
- 非-BEAR 时 defensive ETF **不进** alpha 候选（消除 §4 混淆）

这是改 buy 控制流，按 §7.13 必须走 A/B + 27-mo WF 验证。

### Fix 3 — fresh sim 验证（P0，并行）
跑一段最新代码的 BEAR 窗口 sim，生成带 `bear_only` 的新数据，回答 §3 的待验证问题（GLD/XLU 在当前代码下 BEAR 到底 fire 不 fire）。这决定 Fix 2 的紧迫性。

---

## 6. 这次审计的确定产出 vs 待验证

| 结论 | 状态 |
|---|---|
| `_buy_universe` BEAR 分支要求 `t in ctx.models` | ✅ 确定（代码 line 161） |
| `ctx.models` 只装 watchlist + 有 artifact 的 ticker | ✅ 确定（代码 line 65） |
| TLT/XLV 在 defensive_tickers 但不在 watchlist → 永久不可买 | ✅ 确定（纯文件） |
| `FilterUniverseFloorTask` 豁免 defensives 是 dead code | ✅ 确定（豁免不在 loaded_models 的东西） |
| GLD/XLU 在当前代码下 BEAR 能否 fire | ⚠️ 待 fresh sim |
| 前一轮"bear_only 持久化漏写" | ❌ 撤回（stale 数据误判） |

---

## 7. 不做的事

- 不动 live 代码 / golden config（本 memo 纯诊断）
- 不基于 stale sim 数据下"prod 100% 死"的断言
- Fix 1/2/3 各自需要单独 PR + user-fire；Fix 2 必须 A/B
- 不撤销 cash-drag memo（#195），本 memo 是它 BEAR-sleeve 部分的纯代码深挖 + stale-data 校正

## 8. 跟前序 memo 的关系

- `2026-06-03-cash-drag-root-cause-and-fix.md`（#195，已 merge）的 BEAR-sleeve 死门发现 → 本 memo **确认核心、校正方法**（纯代码取代 stale sim 数据）
- 撤回该 memo 隐含的"4 bug"里的 Bug 4（bear_only 持久化）
- 确认并精确化 Bug 1（defensive 没进候选）→ 根因是 `t in ctx.models` + watchlist drift
