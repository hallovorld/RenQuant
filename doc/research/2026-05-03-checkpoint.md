# 2026-05-03 实验进度断点 (关机前快照)

## 状态截图

### 已落地的训练结果（持久化在 `data/runs.db`）

| wl | OOS IC | features | 备注 |
|---|---|---|---|
| 108 | +0.0345 | 27 | Stage 3 batch 0 (重训) |
| 113 | +0.0359 | 27 | Stage 3 batch 0 |
| 123 | +0.0369 | 27 | Stage 3 batch 1 |
| 133 | +0.0408 | 27 | Stage 3 batch 2 |
| 143 | +0.0379 / +0.0385 / **+0.0437** | 27 | Stage 3 batch 3-5 (5=peak) |
| 153 | +0.0344 → +0.0434 (4 retrains) | 27 | Stage 3 batch 6-11 |
| 173 | +0.0429 | 27 | Stage 3 batch 12（progress.json 误标 ic=None，DB 真值落地） |
| 173 | +0.0429 | 27 | （重复行同上）|
| **183** | **+0.0450** ← **NEW PEAK** | 27 | wl-sweep |
| 188 | +0.0407 | **29 (含新因子)** | wl-sweep（wl=188 起包含 idio_vol_z + mom_1m_reversal_z）|
| 193 | +0.0370 | 27 | wl-sweep |
| 203 | +0.0378 | 27 | wl-sweep |
| 223 | +0.0351 | 27 | wl-sweep |
| 243 | +0.0380 | 27 | wl-sweep |
| 263 | +0.0370 | 27 | wl-sweep |
| 281 | +0.0344 | 27 | top-down 上限 |

**生产 baseline (CPCV mean_ic) = +0.034**。当前最优是 **wl=183 +0.0450**（+11 bp 提升）。

### 关机前正在跑的（会被中断）

- **wl=178 训练** (PID 32035, etime ~4 分钟, ETA 还需 ~45 分钟才能落地)
  - 重启后需要手动重启：`bash /tmp/dispatch_wl178.sh` 已经 stale，改用：
    ```
    cd /Users/renhao/git/github/RenQuant
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate renquant
    export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10
    python scripts/train_104.py --strategy-config-name strategy_config.wl_sweep_178.json --skip-baseline --skip-recalibrate --force > /tmp/wl_sweep_178.log 2>&1 &
    ```

### 已完成 + 已持久化（survives reboot）

1. **新因子代码已落地**：
   - `backtesting/renquant_104/training_panel/factors.py` — 加了 `compute_idio_vol`(Ang 2006 IVOL) + `compute_short_term_reversal`(Jegadeesh 1990)
   - `backtesting/renquant_104/training_panel/pp_panel_training.py` — 通过 raw_factor → z-scoring → factor_frames 全链路接入
   - `tests/test_factors_idio_vol_reversal.py` — 7/7 passing

2. **4 路特征消融配置已写好**（4 个 side config 在 `backtesting/renquant_104/`）：
   - `strategy_config.ablation_A_drop8.json` (19 features = 27 - 8 weak technicals - 2 new)
   - `strategy_config.ablation_B_add2.json` (29 features = 27 + idio_vol_z + mom_1m_reversal_z)
   - `strategy_config.ablation_C_ultra.json` (21 features = 27 - 8 + 2)
   - `strategy_config.ablation_D_control.json` (27 features = baseline reproducer，drop 新因子)
   - 全部 wl=183（当前 sweep peak）

3. **触发脚本已写好**：
   - `scripts/run_feature_ablation_4way.sh` — 训练 4 路 (有 ALLOW_PARALLEL guard 避免与扫描冲突)
   - `scripts/run_ablation_followups.sh` — 选 winner → §5.2 三件套 → B2 sim

## 重启后的恢复流程

### 步骤 1：补完 wl=178（找 173→183 间是否还有更高峰）

```bash
cd /Users/renhao/git/github/RenQuant
source ~/miniconda3/etc/profile.d/conda.sh && conda activate renquant
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=10 OPENBLAS_NUM_THREADS=10 VECLIB_MAXIMUM_THREADS=10 NUMEXPR_NUM_THREADS=10
python scripts/train_104.py --strategy-config-name strategy_config.wl_sweep_178.json --skip-baseline --skip-recalibrate --force > /tmp/wl_sweep_178.log 2>&1 &
```

ETA ~50 分钟，落地后会写到 `data/runs.db`，可用以下命令查：
```bash
sqlite3 data/runs.db "SELECT n_tickers, oos_mean_ic FROM training_runs WHERE artifact_path LIKE '%wl_sweep_178%' AND artifact_type='panel-ltr' ORDER BY run_date DESC LIMIT 1;"
```

### 步骤 2：触发 4 路特征消融（wl=178 完成后）

```bash
cd /Users/renhao/git/github/RenQuant
bash scripts/run_feature_ablation_4way.sh
```

ETA ~3 小时 20 分（4 × 50min）。脚本会按顺序训练 A/B/C/D 然后打印汇总。
配置已经预生成，脚本只是触发训练。

### 步骤 3：选 winner + §5.2 三件套 + B2 sim

```bash
cd /Users/renhao/git/github/RenQuant
bash scripts/run_ablation_followups.sh
```

脚本会：
1. 自动选 OOS IC 最高的 arm
2. 与 control (D) 比较，lift ≥ 1 bp 才往下走（run-to-run σ ≈ 0.6 bp）
3. 跑 §5.2 三件套（A/A + shuffled-label + 时移 placebo），三项都过才进 B2
4. 跑 B2 hold-out sim 拿 APY/Sharpe

ETA ~1 小时。

### 步骤 4：人工 ship/no-ship 决断

输入：消融 winner + lift bp + sanity 三项 PASS/FAIL + B2 APY/Sharpe。
判定标准：
- IC lift ≥ +1 bp（vs D control）
- §5.2 三项 PASS
- B2 APY ≥ baseline（CLAUDE.md §2a 例外条款适用：mechanism-clean 改动 + theory-aligned 即使 < +2 pt 也可以 promote）

## 已知问题

**🔴 progress.json 与 DB 不一致**：Stage 3 batch 12 在 progress.json 标 `accepted=False, ic=None`（timeout-vs-DB pattern，老 bug），DB 实际落地 wl=173 +0.0429。批次本就该被拒（−8 bp vs best），但理由记录错误。低优先级修复。

**🟡 progress.json 多个 batch 的 `delta_ic`、`base_ic` 全 None**：是 dataclass 序列化问题，不是结果问题。低优先级修复。

## 关键 in-flight 任务清单

- #4 in_progress: Watchlist 99 → 200 expansion（被本次 wl=183 peak 推进）
- #30 in_progress: 扩股池实验 — 逐批加股训重比较
- #33 in_progress: WL 大小扫描曲线 173→263（差 wl=178 一点完整）
- #37 in_progress: 4 路特征消融配置已建（已 build，待 dispatch）
- #38 pending: §5.2 sanity triple on ablation winner

## 修复后续

- progress.json fix-up 脚本（把 batch 12 的 ic_None 改为 DB 真值）— 优先级低
- ntfy 区分 sanity vs production retrain（任务 #29）— 优先级低
- 把所有 strategy_config.{ablation_*,wl_sweep_*,stage3_*,topdown_*}.json 加进 .gitignore 或一次性整理

## Git 状态要点（本次未 commit）

修改：
- `backtesting/renquant_104/training_panel/factors.py`
- `backtesting/renquant_104/training_panel/pp_panel_training.py`

新建：
- `tests/test_factors_idio_vol_reversal.py`
- `scripts/run_feature_ablation_4way.sh`
- `scripts/run_ablation_followups.sh`
- `backtesting/renquant_104/strategy_config.ablation_*.json` (4 文件)
- `doc/research/2026-05-03-checkpoint.md` ← 本文件

待用户决定是否 commit。建议消融落地后再做一次性 commit（一并附 winner OOS IC + B2 APY 到 commit message）。
