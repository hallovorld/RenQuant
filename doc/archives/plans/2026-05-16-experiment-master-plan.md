# 2026-05-16 实验总计划 — 严谨性 / 并发 / 早杀监控

## 0. 出发点（已知事实）

- 5/16 rigorous: 6 个旋钮全 NULL，批次 PBO=93% → 当前旋钮空间到达局部最优
- PRIME DIRECTIVE: regime-stratified 是 PRIMARY signal，pooled mean 是 SECONDARY
- 5/15 EVENING calibrator refit → 5/14 baseline 已永久污染，所有 A/B 必须配同日 fresh baseline 或 proxy
- 5/16 fix 已 ship: `preflight_panel.sh`（静态+smoke）+ `preflight_analyzer.sh`（artifact freshness）+ `run_sim_104.py` 单点 gate

## 1. 严谨性协议（每个实验必须遵守）

| 节点 | 检查 | 工具 | 失败行动 |
|---|---|---|---|
| **配置写入时** | 静态路径验证（kernel 实际读取） | `validate_sim_config_active.py` | sys.exit(2) — 不准 dump |
| **配置写入后** | 1-month smoke 验证（equity 非 bit-identical） | `preflight_panel.sh` | sys.exit(2) — 不准发面板 |
| **面板启动时** | side config preflight gate | `run_sim_104.py` 内置 | sys.exit(2) — 整个 sim 拒跑 |
| **每个 window 完成后** | bit-identical-to-baseline 早杀 | `monitor_panel_health.sh`（新） | 立即 kill panel，省 13/16 窗口 |
| **每个面板完成后** | analyzer baseline freshness | `preflight_analyzer.sh` | exit 1 — 不准分析 |
| **统计验收时** | bootstrap CI + Newey-West HAC + DSR + PBO via CSCV | `analyze_panels_rigorous.py` | DSR<0 ∨ PBO>50% → NULL verdict |
| **per-regime 报告** | 必须 n≥3 per regime 才报 verdict | analyzer | n<3 标 INSUFFICIENT |

**禁用列表**：pooled-mean 单数字 verdict / 单 seed claim / `--skip-preflight` 在非 baseline 场合 / 用 5/14 baseline 做 A/B / `tiered_thresholds` 类 Kelly 假设（kernel 无 tier）

## 2. 并发设计（M2 Pro 10 核 / 32 GB）

| 实验 | -P 并行 | RAM/worker | 总并行 | 30s stagger | 预期 wall |
|---|---|---|---|---|---|
| A 道 (A1+A2+fresh-baseline) | 2 each | ~1.8 GB | 6 workers | yes | ~3h |
| B 道 (after A 验收) | 2 each | ~1.8 GB | 4 workers | yes | ~3h |
| A3 (NGB 训练) | -1 (独立 CPU saturating) | ~6 GB | 1 process | n/a | ~3h |

**A 道 + A3 可同时跑**（A3 CPU 重，A1/A2 IO 重，互补）。

**SPY-fetch race 规避**：30s 启动 stagger（5/16 在 6-panel 并行时已验证可行）。

**前置 fix**（已 ship）：xargs -P 子壳变量传递（`export -f` + 内部局部变量）。

## 3. 早杀监控（新 ship — 这次的核心防御）

`monitor_panel_health.sh` 每 5 分钟检查一遍：

1. **进程死亡** — `kill -0 $pid` 失败 → ntfy "PANEL X DIED" 红色
2. **30 分钟无新 equity JSON** — 卡死或 SPY-fetch race → ntfy 黄色
3. **No-op 早杀** — 当 3 个 window 完成后，逐 window 与 baseline 比 equity；如果 bit-identical → `pkill -P <pid>` + ntfy 红色（旋钮无效，浪费 80min 中止）
4. **内存压缩 > 5 GB** — vm_stat compressor → ntfy 黄色，提示考虑 -P 降并发
5. **load average > 12** — top → ntfy 黄色
6. **artifact 在 baseline 后 refit** — preflight_analyzer.sh 周期性 trigger → ntfy 红色

监控自身用 launchctl-friendly nohup 长跑；写状态到 `data/logs/monitor/<batch>_state.json`，关电脑后再开机也能恢复。

## 4. 三个赛道详细规划

### A 道 — 立即可做（5/16 之后唯一带方向性信号的两条线）

```
[t=0]            [t=+30min]          [t=+3h]                [t=+3.5h]
build configs ─→ preflight (静+smoke) ─→ 3 panels 并行 ────→ rigorous analyzer
                                          ├ fresh-baseline   (auto via runner)
                                          ├ A1 sdl_n2 overlay
                                          └ A2 cvar025 overlay
                                          ↑
                                          monitor 全程并发
```

**假设**：sdl n_sigma=2.0（A1）和 cvar λ=0.25（A2）在 BEAR/CHOPPY 上的条件性胜出是真信号；只在 BEAR/CHOPPY 启用，其他 regime 留 golden 默认值，能消除 BULL 拖累 → pooled+stratified 都正。

**配置写法**（关键 — 与 5/16 全 regime 写法不同）：
```python
# A1 sdl_n2 overlay
for r in ("BEAR", "CHOPPY"):
    cfg["regime_params"][r]["sdl_n_sigma"] = 2.0
# BULL_CALM / BULL_VOLATILE / BULL_STRONG 不动 → 保留 golden 默认
```

**验收门槛**（Tier-3 才能 promote）：
- per-regime n（BEAR + CHOPPY）≥ 3 windows each
- bootstrap 95% CI 下限 > 0
- DSR > 0.5
- PBO < 0.5（按 2 个实验算）
- pooled Δ ≥ 0 OR pooled Δ < 0 但 BEAR+CHOPPY 各自 CI 下限 > 0 且 BULL 各自 CI 上限 < 0（典型条件性胜出）

### B 道 — A 道之后（前置 detector fix）

| 编号 | 前置 | 实验 | 设计要点 |
|---|---|---|---|
| **B1** | 无 | regime detector RESPONSE 软切（renormalize 而非 hard switch） | per-regime mix weight 0~1 而不是 0/1 hard switch；Kaminski-Lo 2014 |
| **B2** | B1 | BEAR-hybrid 实施（slow→DEFENSIVE, hard→shorts-only） | 用 B1 软切后的 weights 当 mix 系数 |
| **B3** | 无 | Phase 2D long-short cover-stop + 税法 §1233 | paper Alpaca 已接，1 周观察 |

### C 道 — 结构突破（user 已点名"single-knob 扫尽"的必经之路）

| 编号 | 改动 | 预期收益 |
|---|---|---|
| C1 | wl183 universe 扩展 | IC peak 历史 +0.045 |
| C2 | R1K full Russell 1000 | 主要降 wl-bias |
| C3 | Kelly-Gu-Xiu RFS 2020 cross-sectional features | +0.01~+0.03 IC |
| C4 | Qlib TransformerModel / PatchTST | +0.005~+0.015 IC |

### A3 — NGBoost 投产（独立赛道，可与 A 道并行）

```
[t=0]              [t=+3h]              [t=+3.5h]            [t=+8h]
retrain NGB ────→ σ-aware Kelly wire ─→ smoke 验证 ────────→ 5-seed val IC + σ-calib
                                                              ↓
                                                       决策：投产 / 回滚
```

**通过门槛**：5-seed val IC > +0.030（5/15 已测 +0.0351）且 σ-calib > +0.20 且 t vs XGB-quantile > +2.0

## 5. 失败模式与应对

| 失败模式 | 早期信号 | 应对 |
|---|---|---|
| 配置写错路径（5/15 重演） | preflight smoke 失败 | sys.exit(2)，build 阶段就停 |
| baseline 污染 | preflight_analyzer 红 | proxy baseline fallback |
| no-op 旋钮（kelly_t1 重演） | monitor 早杀（3 window 后） | 立即 kill + ntfy |
| SPY-fetch race | monitor 检测 .parquet.tmp 多文件 | 已用 30s stagger 规避 |
| 内存压缩 | monitor 检测 compressor>5GB | 降 -P 重启 |
| process zombie | monitor kill -0 失败 | ntfy + auto-cleanup |
| 假阳性 verdict（n=1 BEAR） | analyzer 报 INSUFFICIENT | 拒签 verdict |

## 6. 关电脑期间安排

所有脚本设计为**可断点续跑**：
- `scripts/run_regime_overlay_experiments.sh` — 每个 window 单独 equity JSON；如果已存在非空则 skip
- `scripts/monitor_panel_health.sh` — 状态写 `data/logs/monitor/<batch>_state.json`
- `scripts/run_a3_ngboost_retrain.sh` — 每阶段检查产物，已生成则 skip

**笔记本休眠**：进程会停。开机后重跑 launcher 脚本即从断点继续。

**远程触发**（可选）：用户可以从手机 ntfy.sh/renquant 收到红警 → 通过 RemoteTrigger 触发 stop / restart。

## 7. 交付清单（本次会话产出）

| 文件 | 用途 | 状态 |
|---|---|---|
| `doc/research/2026-05-16-experiment-master-plan.md` | 本文档 | ✓ |
| `scripts/build_regime_overlay_configs.py` | A1+A2 配置生成 + preflight | 待写 |
| `scripts/run_regime_overlay_experiments.sh` | A 道并行 runner + 自动分析 | 待写 |
| `scripts/monitor_panel_health.sh` | 早杀 + 健康监控 | 待写 |
| `scripts/run_a3_ngboost_retrain.sh` | A3 编排 | 待写 |

## 8. 命令速查（用户开机后执行）

```bash
cd /Users/renhao/git/github/RenQuant && source .venv/bin/activate

# Step 1: build A1+A2 (含 preflight)
python scripts/build_regime_overlay_configs.py

# Step 2: 启动 A 道并行 (后台)
nohup ./scripts/run_regime_overlay_experiments.sh \
  > logs/reeval_queue/overlay_$(date +%Y%m%d).log 2>&1 &
echo $! > /tmp/overlay_runner.pid

# Step 3: 启动监控 (后台)
nohup ./scripts/monitor_panel_health.sh overlay_2026-05-16 \
  > logs/reeval_queue/monitor_overlay_$(date +%Y%m%d).log 2>&1 &
echo $! > /tmp/overlay_monitor.pid

# Step 4 (独立 / 可并行): A3 NGBoost
nohup ./scripts/run_a3_ngboost_retrain.sh \
  > logs/a3_ngboost_$(date +%Y%m%d).log 2>&1 &
```
