# 2026-06-03 — BEAR Defensive Sleeve Fix 2 实现设计

**地位**: 实现蓝图（设计先于动手，§7.11 + §10）。**无代码改动**，待 codex/用户 review 设计方向后实现。
**前序**: [`2026-06-03-bear-sleeve-deadgate-audit.md`](./2026-06-03-bear-sleeve-deadgate-audit.md)（#205，根因诊断）。
**Multi-repo**: canonical 实现在 **renquant-pipeline**；paired mirror 到 umbrella；config 在 **renquant-strategy-104** + golden mirror。
**所有者**: Claude。

---

## 1. 为什么是新 task，不是改 `_buy_universe`

#205 audit 确认 BEAR sleeve 死门是三层叠加：

| 层 | 现状 |
|---|---|
| Sizing | ✅ 已就绪 — `sizing.py:151` `override_pct` 让 defensive 用 `bear_defensive_pct` 固定 % 绕过 Kelly |
| Universe | ❌ `_buy_universe` 要求 `t in ctx.models`；`LoadArtifactsTask` 只 iterate watchlist |
| Model | ❌ defensive 用 renquant_103 legacy per-ticker model；**TLT 缺 xgb-buy/sell、XLV 缺 xgb-buy/sell** — 残缺 |

**关键洞察**：强行让防御 ETF 走 alpha 的 `model → score_candidates → Kelly` 管线是设计错误。防御资产的本质是"BEAR 时无条件配置避险，不需要 alpha 预测"。改 `_buy_universe` 去掉 `t in ctx.models` 还是会卡在 model 残缺 + ranking 对无-score ticker 的处理。

**正确做法**：一个 additive 新 task `ApplyBearDefensiveSleeveTask`，在 `bear_only` 时**完全绕过 model/ranking/Kelly**，直接按固定 % emit defensive buy orders。§5.2 single-responsibility，default-off opt-in，零 alpha 污染。

---

## 2. 核心设计

### 2.1 新 task（renquant-pipeline `kernel/pipeline/task_selection.py` 或独立文件）

```python
class ApplyBearDefensiveSleeveTask(Task):
    """BEAR-only: 无条件按固定 % 配置避险 ETF，绕过 model/ranking/Kelly.

    Single responsibility: emit defensive buy orders when bear_only.
    Additive — 不碰 alpha 候选路径。Default-off (opt-in config flag).
    防御资产不需要 alpha 预测，所以不依赖 panel-LTR / legacy per-ticker
    model（这正是 TLT/XLV model 残缺仍能 fire 的原因）。
    """

    def should_skip(self, ctx) -> bool:
        # 只在 bear_only 且 sleeve 显式启用时跑
        if not getattr(ctx, "bear_only", False):
            return True
        cfg = ctx.config.get("bear_defensive_sleeve", {}) or {}
        return not bool(cfg.get("enabled", False))   # default-off

    def run(self, ctx) -> bool | None:
        defensive_set = list(ctx.config.get("defensive_tickers", []))
        pct_total = float(ctx.config.get("bear_defensive_pct", 0.15))
        slots     = int(ctx.config.get("bear_defensive_slots", 2))
        if slots <= 0 or pct_total <= 0:
            return None
        pct_per_slot = pct_total / slots

        # 已持有的 defensive 不重复买；已有 order 的不重复。
        # Sizing uses the fixed per-slot cap, not pct_total / remaining
        # targets: if one of two 15% slots is already held, the one new order
        # is still capped at 15%, not the full 30% sleeve budget.
        held = set(ctx.holdings.keys())
        already_ordered = {o["ticker"] for o in ctx.orders}
        already_def_held = sum(1 for t in held if t in set(defensive_set))
        open_slots = max(slots - already_def_held, 0)
        if open_slots <= 0:
            return None

        # 只需价格 (ohlcv)，不需要 model。优先级 = defensive_tickers 顺序
        eligible = [t for t in defensive_set
                    if t in ctx.ohlcv and t not in held
                    and t not in already_ordered]
        targets = eligible[:open_slots]
        if not targets:
            return None

        reserve  = float(ctx.config.get("regime_params", {})
                          .get(ctx.regime, {}).get("cash_reserve_pct", 0.0))
        remaining = float(getattr(ctx, "cash", 0.0) or 0.0) \
                    - reserve * float(ctx.portfolio_value or 0.0)

        import math
        for t in targets:
            price = ctx.prices.get(t)
            if price is None or not math.isfinite(price) or price <= 0:
                continue
            invest = min(pct_per_slot * float(ctx.portfolio_value or 0.0), remaining)
            shares = int(invest / price)
            if shares <= 0:
                continue
            ctx.orders.append(stamp_order_attribution({
                "ticker": t, "shares": shares, "price": price,
                "invest": shares * price,
                "target_pct": (shares * price) / float(ctx.portfolio_value or 1.0),
                "regime": ctx.regime, "confidence": ctx.confidence,
                "conviction": None, "sigma_mult": None,
                "rank_score": None, "rs_score": None, "panel_score": None,
                "sigma": None, "mu": None, "kelly_target_pct": None,
                "detail": "bear_defensive_sleeve",
                "order_source": "BEAR_DEFENSIVE_SLEEVE",
            }))
            remaining -= shares * price
        return None
```

### 2.2 注册（renquant-pipeline `kernel/pipeline/job_selection.py`）

```python
class SelectionJob(Job):
    def tasks(self):
        return [
            PrepareSelectionTask(),
            RunSelectionTask(),
            SizeAndEmitTask(),
            ApplyBearDefensiveSleeveTask(),   # ← 新增，在 alpha sizing 之后
        ]
```

放在 `SizeAndEmitTask` 之后：alpha sizing 先跑（bear_only 时 alpha 候选基本空），然后 sleeve 用剩余 cash 补 defensive。`should_skip` 保证非-BEAR / 未启用时零开销。

### 2.3 config schema（renquant-strategy-104 + golden mirror）

```jsonc
"bear_defensive_sleeve": {
    "enabled": false   // opt-in, default OFF (保守, 不直接翻 golden)
},
// 复用已有: bear_defensive_pct (0.15), bear_defensive_slots (2),
//          defensive_tickers ([GLD,TLT,XLV,XLU])
```

注意 #205 的 watchlist drift（TLT/XLV 不在 watchlist）**对这个 task 无影响** —— sleeve 不走 `LoadArtifactsTask`/`ctx.models`，只看 `ctx.ohlcv`（价格对所有 ticker 都有）。这正是新 task 设计的好处。

---

## 3. 退出路径（unwind）

BEAR → BULL 转换时，defensive 持仓要卖掉腾位置给 alpha。**现有 sell 路径已覆盖**：`task_sell.py` 对所有 holdings（含 defensive）跑 sell gates。但需验证：

- defensive ETF 没 panel model → sell 决策不能依赖 model score
- 退出信号应该是 **regime 转出 BEAR**（GLD/TLT 不再需要），不是 alpha sell

**开放问题 O-1**：defensive 的退出是否需要一条 regime-driven 卖出（regime != BEAR → 卖 defensive sleeve）？还是靠 rotation（alpha 候选回来时轮换掉 defensive）？待实现时确认 `task_sell` / `task_rotation` 对 defensive 的现有处理。

---

## 4. multi-repo 布局（§3.5）

| 改动 | repo (canonical) | mirror |
|---|---|---|
| `ApplyBearDefensiveSleeveTask` 实现 | renquant-pipeline `kernel/pipeline/task_selection.py` | umbrella |
| `SelectionJob` 注册 | renquant-pipeline `kernel/pipeline/job_selection.py` | umbrella |
| 单元测试 | renquant-pipeline `tests/` | umbrella |
| config flag `bear_defensive_sleeve.enabled` | renquant-strategy-104 configs | umbrella golden |

Paired PR，§3.5 byte-equivalent mirror，MD5 等价测试覆盖。

---

## 5. A/B 验证计划（§7.13 强制 + §9 DOE）

production buy 路径改动，**不直接翻 golden**。default-off，A/B 验证后才 Tier 3 promote。

### 5.1 实验
- **baseline**: golden（sleeve off → BEAR 100% cash，#205 现状）
- **treatment**: `bear_defensive_sleeve.enabled=true`（BEAR 配 30% defensive）
- 27-mo WF sim，覆盖 2025-03~05 的 BEAR 窗口（sim DB 有 52 个 BEAR 日）

### 5.2 关键 metric（§1 PRIME DIRECTIVE — 按 regime first）
- **BEAR 期间**: Δ return / Δ Sharpe / Δ MaxDD（sleeve vs 100% cash）
- 净效果 = 避险资产回报 − 交易成本 − 在 BEAR→BULL 转换点的 unwind 损失
- pooled 27-mo: 确认 sleeve 不伤其它 regime（应该不会，additive + bear_only-gated）

### 5.3 sanity（§7.2.1 R2）
sleeve 是确定性配置（非 model 预测），placebo 三件套不完全适用，但需：
- **A/A**: 同 config 重跑，BEAR 期间 PnL 可复现
- **退出审计**: 确认 BEAR→BULL 转换点 defensive 被正确 unwind（O-1）

### 5.4 promotion（§7.4）
Tier 3 才翻 golden `enabled=true`。BEAR 期间 ΔSharpe ≥ 0 且 ΔMaxDD ≤ 0（避险应降回撤）是核心判据。

---

## 6. 测试计划（renquant-pipeline `tests/`）

| 测试 | 断言 |
|---|---|
| `test_sleeve_fires_on_bear_only` | bear_only=True + enabled → emit defensive orders |
| `test_sleeve_skips_when_not_bear` | bear_only=False → should_skip True，零 order |
| `test_sleeve_skips_when_disabled` | enabled=false → should_skip True |
| `test_sleeve_no_model_required` | TLT/XLV 无完整 model 也能 fire（只需 ohlcv）|
| `test_sleeve_uses_fixed_slot_cap` | 每个新 target ≤ `bear_defensive_pct / bear_defensive_slots` |
| `test_sleeve_excludes_held_defensives` | 已持有 GLD → 不重复买，slots 递减 |
| `test_sleeve_does_not_overallocate_when_partially_held` | 已持有 1/2 defensive slot 时，新 order 仍只吃 1 个 slot cap，总 sleeve 不超过配置 |
| `test_sleeve_respects_remaining_cash` | invest 累计 ≤ cash − reserve |
| `test_sleeve_respects_slots` | open_slots = bear_defensive_slots − 已持有 defensive |
| `test_sleeve_order_attribution` | order_source == "BEAR_DEFENSIVE_SLEEVE"，可审计 |

通过真实 `SimAdapter` 跑一个 BEAR bar（§7.1 test-through-real-adapters）。

---

## 7. 开放问题 / 风险

| # | 问题 | 处理 |
|---|---|---|
| O-1 | defensive 退出路径（regime-driven sell vs rotation） | 实现时确认 task_sell/rotation 现有处理 |
| O-2 | defensive 优先级（GLD>TLT>XLV>XLU?）当 slots < len(defensives) | 默认 defensive_tickers 顺序；A/B 可调 |
| O-3 | 等权 vs 按避险特性加权（TLT 久期 vs GLD 黄金） | v1 等权；加权留 follow-up |
| O-4 | BEAR 误判（detector 短暂 BEAR）触发 sleeve 买入又立即卖 | bear_only 已有 confidence≥0.60 + 非-transition gate (BEARBranchTask) 兜底 |
| R-1 | sleeve 在 BEAR 买入后市场继续跌（defensive 也跌） | A/B 的 ΔMaxDD 判据捕获；GLD/TLT 通常负 β 对冲 |

---

## 8. 不做的事

- 不改 `_buy_universe` / `LoadArtifactsTask`（新 task 绕过它们，#205 的 universe 死门对 sleeve 无影响）
- 不动 watchlist（TLT/XLV 不进 watchlist 不影响 sleeve）
- 不给 TLT/XLV 补 legacy model（sleeve 不需要 model）
- 不直接翻 golden `enabled`（default-off，A/B Tier 3 后才翻）
- 不在本 doc 写任何代码（设计蓝图，实现是下一步）

## 9. 下一步

1. codex/用户 review 本设计（尤其 O-1 退出路径 + §2.2 注册位置）
2. review 通过 → 实现 paired PR（pipeline canonical + umbrella mirror）+ 测试
3. config flag PR（strategy-104 + golden，default-off）
4. sim A/B（§5）→ Tier 3 → 翻 golden
