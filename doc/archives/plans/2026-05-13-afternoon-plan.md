# 2026-05-13 下午并发实验 + Phase 2A 计划

## 当前状态（15:40 PT）

**运行中**: 7 wl174_retrained sims (Q01-Q07, pre-2024 aux), ~30min each (174 tickers slow), 完成 ~16:10 PT

**已就绪 (不冲突)**:
- 5 个 walkforward manifests trained: wl174_retrained, fwd_5d, fwd_20d
- 4 个 side configs queued: wl174_retrained, horizon_fwd5d, horizon_fwd20d, sector_cap3, corr05
- Phase 2A code: 4/8 organic components done

## 6 个 Pending 实验 — 预注册表

| # | 实验 | 假设 H1 | 文献依据 | 预期 effect | Tier 3 触发自动 promote? |
|---|---|---|---|---|---|
| 1 | **wl174_retrained** (跑中) | √breadth IR 上升 30% | Grinold-Kahn 1999 §5 | +3-5pt APY | ✅ Yes (universe flip) |
| 2 | horizon_fwd5d | 短horizon = 噪声 → REJECT | LdP AFML §17 | -1 to -5pt | ❌ No (期望失败) |
| 3 | horizon_fwd20d | 中horizon ≈ baseline | LdP AFML §17 | ±2pt | ❌ No |
| 4 | sector_cap3 | 紧 sector cap → vol ↓ Sharpe ↑ | Markowitz 1952 | Sharpe +0.05-0.10 | ✅ Yes |
| 5 | corr05 | 紧 pair corr → 集中度 ↓ | Bouchaud-Potters 2003 | Sharpe +0.05 | ✅ Yes |
| 6 | long_short_v1 | L-S +29.2%/yr 实现 +5-7%/yr alpha | Grinold-Kahn × √2 | +5-7pt | ❌ No (需 Phase 2D-2I + 用户 OK) |

## 并发执行时间表

```
时间    后台 Sim (用 8 cores)             前台 Code (Phase 2A)
─────  ──────────────────────────────  ──────────────────────────────
15:40  wl174_retrained Q01-Q07 跑中    [等 sim 释放 RAM]
16:10  Q01-Q07 完成 → 启 Q08-Q16       short candidate selection 写代码
16:50  wl174 完整 16win 完成 → analyze sector-neutral hard constraint
17:00  启 horizon_fwd5d 16 sims        trailing-stop short-aware
17:50  horizon_fwd5d done → analyze    Phase 2A 代码全部测试
18:00  启 horizon_fwd20d 16 sims       Phase 2C: long_short smoke sim
19:00  horizon_fwd20d done → analyze   分析 long_short smoke
19:10  启 sector_cap3 16 sims          
20:00  sector_cap3 done → analyze      
20:10  启 corr05 16 sims                 
21:00  全部 5 实验 16win panels done   综合 5-实验 Tier 表
21:30  Tier 3 auto-promote (若有)      
```

**总今日 compute**: ~5.5h (按队列序列)，每批 8 并发 sim。

## 科学严谨保证

每个实验都遵守：

1. **Pre-registration**: H0/H1 + 预期 effect size + Tier 3 触发规则（本文档即是）
2. **同框架**: 16 non-overlapping 3mo windows, paired daily Δ, statsmodels HAC, arch bootstrap
3. **K_trials counter**: 当前 K = 6 + 6 新 = 12, DSR penalty 适用
4. **Tier 3 标准 (canonical)**:
   - t_pool > 3.0 (Harvey-Liu-Zhu 2016)
   - DSR > 0.5 (Bailey-LdP 2014)
   - PBO < 0.5 (Bailey-Borwein-LdP-Zhu 2015)
   - p_NW < 0.01
   - Bonferroni-Holm adjusted at α=0.01

## 自动 promote 协议（per feedback_auto_promote_to_prod.md）

若任何实验通过 Tier 3:
1. 跑 analyzer，确认 DSR/PBO 数字
2. 备份当前 golden.json
3. 改 golden.json (flip flag/universe)
4. 加 pinning test
5. pytest full suite
6. 1 window sanity sim
7. Commit with `[promote]` tag
8. 写记忆 + 通知用户

## 不自动 promote 的红线

- long_short (需 Phase 2D-2I + 用户 OK)
- horizon_fwd5d (期望失败的 sanity)
- broker 切换（不在范围内）
- 风险约束放松（如把 stop_loss 调宽）

## 不做的事

- 不再尝试 LGBM swap（已 shelved，prior 测过 ≈ XGB）
- 不今天做 PatchTST（多周工程）
- 不今天做 Options-IV / FinBERT（多周数据 pipeline）
- 不今天进 live broker（任何 prod flip 都按 9-step protocol）

## 风险预案

- 若某 sim 批次崩溃 → 跳过，记录失败，继续下一个
- 若 prod cron 14:06 PT 受影响（不会，因为新 code 都 OFF by default）→ 立即停 sim 处理
- 若 RAM 不够 → 8-concurrent throttle 严格遵守
