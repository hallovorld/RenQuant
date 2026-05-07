# Cloud GPU training + local inference — design plan

> **状态**: 设计文档，未实施。
> **触发条件**: 当 Phase C / Phase D NN backend (graph attention / MIGA MoE) 开始落地，或本地训练 wallclock 超过 4h/单跑。
> **核心设计**: 训练在云 GPU 跑（贵但快），artifacts 推到云存储，本地 M2 Pro 拉下来做生产推理（便宜可靠）。

---

## 1. 为什么这套架构

**现实**：
- XGBoost rank:pairwise (现 production) 是 CPU-bound。GPU 加速有限 (~1.5×). 当前 ~30 min 单跑可接受。
- **NN backends (Phase C TGC, Phase D MIGA MoE)** 是真正吃 GPU 的：M2 Pro MPS 跑得通但**慢 5-10×**，且 PyTorch MPS 有已知 gaps (CLAUDE.md §2026-04-25 长期 architectural items 已记录)。
- **Inference 路径** 在 M2 Pro 上 ~3-5s 每 bar，完全不需要 GPU。

**得失权衡**:
- **本地训练**: 免费，但 wallclock 长 / MPS 不稳 / 笔记本占用风扰开发
- **云训练**: 付费 $/h，但 wallclock 短 / 真 CUDA 没 MPS 坑 / 笔记本同时可干别的
- **云推理**: 不必要 — inference 轻量，且 broker (Alpaca) 在本地，加云链路引入失败模式

**结论**: **train-cloud / infer-local hybrid** 是最优。

---

## 2. Cloud GPU provider 对比

| Provider | Hardware | $/h | Storage | 体验 | 适用场景 |
|---|---|---:|---|---|---|
| **Lambda Labs** | A10 / A100 / H100 | $0.50 / $1.10 / $2.50 | 持久 | SSH + Jupyter, 持久卷 | 主推荐 — 简单可靠 |
| **RunPod** | A6000 / A100 / H100 | $0.39 / $0.79 / $1.99 | 持久 + 网络卷 | 有 API + UI, 持久卷 | 备选 — 价格更优 |
| **vast.ai** | 各种二手卡 | $0.10-0.40 (spot) | volatile | spot 风格，会被抢占 | 仅适合可中断批训练 |
| **Modal Labs** | T4/A10/A100/H100 | $0.59-3.10 (per-second) | ephemeral + S3 mount | serverless, Python decorator API | 适合按需跑 (我们的场景) |
| **AWS p3/p4** | V100/A100 | $3-30/h on-demand | EBS | 复杂，但企业级 | 不推荐小团队 |
| **GCP A2** | A100 | $3.67/h | persistent disk | 类 AWS | 不推荐 |

**首选: Lambda Labs A10 (~$0.25-0.50/h on-demand)**
- 24GB VRAM 够 178-ticker × 750-date NN panel 训练
- $$/月预算: 假设 NN 重训每周 1 次 × 4h = ~$8-15/月 — 真便宜
- 持久存储 ($0.20/GB/月) 放数据 + checkpoints

**次选: RunPod A6000** (48GB VRAM, $0.39/h) — 多 VRAM 留给 transformer 实验

**用 Modal 或 vast.ai 之前要确认**：
- Modal 适合"零运维 serverless 单跑" 但 storage / network 限制多。我们大数据 panels 上传慢
- vast.ai spot 会被抢占，长跑训练崩

---

## 3. 数据 / Artifact 流向

```
   ┌────────────────────────────────┐
   │   LOCAL (M2 Pro)               │
   │                                │
   │   git repo                     │
   │   data/ohlcv/ (1 GB)           │
   │   data/intraday/ (83 MB)       │
   │   data/runs.alpaca.db          │
   │   live_state.json              │
   │   live runner / Alpaca broker  │
   │                                │
   │   artifacts/panel-ltr.json     │ ← latest production artifact
   └────────────────────────────────┘
          │ ↑                 ↓ ↑
          │ │ git push        │ │ rclone / b2 sync
          │ ↓                 │ │
   ┌────────────────────────────────┐
   │   CLOUD (Lambda Labs A10)       │
   │                                │
   │   git clone (latest code)       │
   │   data/ rsync from B2 (cached) │
   │   training artifacts → B2      │
   │   logs → B2                    │
   └────────────────────────────────┘
                  │
                  ↓
   ┌────────────────────────────────┐
   │   B2 BUCKET (renquant-train)   │
   │                                │
   │   data/snapshots/{date}/...    │
   │   artifacts/{run_id}/          │
   │     panel-ltr.json             │
   │     ngboost-head.json          │
   │     transformer-state.pt       │
   │   logs/{run_id}/train.log      │
   └────────────────────────────────┘
```

**关键点：**
- 代码在 GitHub。云 box 启动后 `git clone` 拉最新
- 数据 (OHLCV + fundamentals) 是公开市场数据 → 推到 B2 + 云从 B2 拉（rclone）
- **绝不上传到云的**: `.env` (Alpaca keys), `live_state.json` (持仓状态), `data/runs.alpaca.db` (含真实 trade history) — 这些只在本地
- 训练完成的 artifacts 推到 B2 → 本地 rclone pull → 本地 model_acceptance 做 promotion gate

---

## 4. 工作流 (一次完整 cloud-train cycle)

```bash
# 1. Local: trigger cloud training
$ ./scripts/cloud_train_dispatch.sh --backend graph_attention --epochs 10

# What this does internally:
#   a) Spin up Lambda A10 instance via API (lambdalabs.com/api/v1/instance-operations/launch)
#   b) Wait for SSH ready
#   c) ssh ubuntu@<instance>: git clone repo + b2 sync data
#   d) ssh: python scripts/train_104.py --backend graph_attention --epochs 10 \
#       2>&1 | tee /tmp/train.log
#   e) On completion: b2 sync artifacts/ → bucket
#   f) Terminate instance
#   g) Local: rclone pull artifacts/run_<id>/ → local artifacts/
#   h) Local: python scripts/promote_artifact.py run_<id>  (acceptance gates)

# Result: latest production-grade artifact on local disk, ready for daily_104.sh
```

**单跑成本 (NN A10 × 4h)**: ~$1-2 + $0.05 storage/month

---

## 5. 实施阶段

### Phase G.0 — 设计 + 选 provider (1 天)
- [ ] 选定 provider (推荐 Lambda Labs A10)
- [ ] 创建 B2 bucket (`renquant-train`)
- [ ] 注册 1Password 凭据存放: B2 access key, Lambda API key

### Phase G.1 — Storage layer (2-3 天)
- [ ] `scripts/b2_sync_data.py` — push local data/ohlcv → B2
- [ ] `scripts/b2_pull_artifacts.py` — pull cloud artifacts → local
- [ ] Test: 完整数据上下行成功

### Phase G.2 — Cloud orchestration (3-5 天)
- [ ] `scripts/cloud_train_dispatch.sh` — provision + SSH + run + tear down
- [ ] Lambda Cloud API integration (instance launch / status / terminate)
- [ ] Cleanup hooks (失败时自动 terminate 防 $$$)
- [ ] 测试: 简单 hello-GPU 跑通

### Phase G.3 — Train integration (3-5 天)
- [ ] `train_104.py --backend graph_attention` 现有钩子完善
- [ ] 训练脚本输出 artifacts 写到本地 + 上传 B2
- [ ] 训练日志结构化 (per-epoch IC / loss) → B2
- [ ] 测试: 端到端从本地 dispatch 到本地拿到 artifact 全程

### Phase G.4 — Acceptance + promotion (2-3 天)
- [ ] `promote_artifact.py` 跑本地 acceptance gates 后将 artifact 移到 production path
- [ ] 自动 ntfy 通知训练完成 + 是否通过 gates
- [ ] B2 上保留 promoted + rejected artifacts 历史

### Phase G.5 — Cost monitoring (1 天)
- [ ] 月度账单 / Lambda billing dashboard 自动 ntfy 周报
- [ ] 失败 instance 报警（防止 zombie instance 烧钱）

**总工作量**: ~2 周（一个工程师持续推 2 周）

---

## 6. 安全 / Ops 考虑

1. **凭据隔离**:
   - Alpaca live keys → ONLY 本地 + 1Password
   - B2 read/write 凭据 → 1Password；环境变量注入云 instance
   - GitHub 凭据 → 云 instance via `--ssh-key` upload, 不写到磁盘

2. **数据安全**:
   - OHLCV 公开数据，B2 不需加密。但 restic encryption 保护 metadata
   - **绝不要**把 `data/runs.alpaca.db` 推到云 — 含真实交易记录
   - **绝不要**把 `.env` 推到云

3. **失败模式**:
   - Cloud instance 启动后训练崩溃 → 自动 terminate，本地报警
   - B2 上传失败 → 本地保留 artifacts 副本，可手动重传
   - 云 box 被 spot 抢占（vast.ai） → 不用 spot

4. **回滚**:
   - 云训练出 bad artifact → acceptance gates 拒绝 → production 不变，rejected artifact 归档到 `_acceptance_log/`
   - **生产模型永远是本地的，云只是"训练加速器"** — 云爆炸不影响生产运行

---

## 7. 何时启动这个项目

**触发条件 (any of)**:
- Phase C 或 Phase D NN backend 单跑超过 4h on M2 Pro
- 需要并行跑 5+ 架构实验，本地 CPU 抢占严重
- 需要训练大 transformer (tickers > 500, dates > 1500)

**当前状态**:
- 现 production 只 XGBoost + 小 NGBoost，本地 30 min 完成 → 不需要
- Sector-aware 实验阶段 → 本地够用
- 一旦 Phase C 落地 → 立即上 cloud GPU

**预算估算 (年度)**:
- 周训：4h × $0.50 × 52 = ~$104/year
- 月训 (大 NN ablation): 8h × $0.50 × 12 = ~$48/year
- B2 storage: 5GB × $0.005 × 12 = ~$0.30/year
- **总: ~$150-200/year, 完全可承受**

---

## 8. Open questions

1. **Provider 锁定**: 推荐 Lambda Labs，但要不要也评估 RunPod (便宜 22%) 或 Modal (按秒计费)?
2. **Spot vs on-demand**: spot 便宜 50%, 但被抢占。我们要不要做 checkpoint resume 机制?
3. **统一存储**: B2 同时给 backup plan (#4) 和 train plan (这里) 用同一个 bucket，还是分两个?
4. **Multi-cloud**: 万一 Lambda 容量满，要不要有 RunPod fallback?
5. **CI/CD**: GitHub Actions 触发云训练 vs 本地命令触发？
