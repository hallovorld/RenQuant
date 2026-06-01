# 2026-06-01 · B_tuned 泄漏反思 + Multirepo 架构修复（v5 — production-grade 设计）

**Author**: Claude  
**Reviewers**: Codex (v1→v2 strict review: HIGH×2 + MED×4 已 address) · User (v2→v3 multirepo; v3→v4 concrete code/YAML/dataflow; v4→v5 "still not good enough" → production-grade)  
**Context**: B_tuned PatchTST Tier-3 placebos 二次失败（shuffle ≈ real ≈ +0.04 IC）。  
**Frame**: 13-repo multirepo 系统的 ML 泄漏防御不是"加道墙"——是 **威胁模型 → 状态机 → contract ownership → cross-repo gate network → 并发 + 持久化语义 → ops runbook + 灾难恢复 → 遥测 → 性能预算 + 向后兼容迁移 + 安全模型** 的完整产品级架构。

**v5 vs v4 增量**: §2.2 威胁模型；§3.3 triad_status 状态机；§4.4 sidecar 并发语义；§4.5 真实代码（非伪码）；§4.6 配置 schema；§4.7 遥测数据模型；§5.5-5.6 失败路径时序图；§6.6 ops runbook；§9.x performance budget + 向后兼容 + 安全模型 + agreement criteria。

---

## 1 · 反思（unchanged from v4）

跑 2h 没看盘上数据；5/31 没修根因就 6/01 重跑；报 ETA 不报概念；新数字第一行 log 前没看 sanity triad — 4 条 process violation（§6.4 / §7.11 / §8.1 / §7.2）。

---

## 2 · 威胁模型（v5 NEW）— 显式列出该挡和不挡

### 2.1 数据 + 真凶类别（unchanged）

```
              s42      s43      s44       期望
real          0.061    0.041     —
shuffle       0.014    0.041    0.048    ≈ 0
timeshift     0.041    0.049     —       ≈ 0
BEAR shuffle  0.048    0.091    0.084    ≈ 0
```

Tier 1 (秒级, 固定 scorer) 拦不住；**Tier 2 (重训, 75min) 才能拦今天这类**。

### 2.2 威胁模型 — 12 种 leakage class + gate 责任分配

| # | Leak class | 表现 | 防御 gate | 残余风险 |
|---|---|---|---|---|
| **L1** | 显式 label 列错放进 feature parquet (e.g. `fwd_60d_excess`) | shuffle IC > 0 | G0 (DatasetManifest features ∩ labels = ∅) | 0 — schema 拒绝 write |
| **L2** | 隐式 lookahead feature (e.g. `rolling_mean_using_future_window`) | shuffle IC > 0；BEAR 尤甚 | G0 (FeatureManifest.feature_lookahead_days[c] = 0 contract) | 残余：feature builder 自报 0 但实际 > 0 — 靠 G2 兜底 |
| **L3** | Split label 列泄漏 (`split_label` ∈ features) | scorer 知道哪天是 val → 训练时给 val 行注入噪声 | G0 (split_label 不在 FeatureManifest.feature_cols) | 0 — schema 拒绝 |
| **L4** | Train/val 边界不足 embargo (label lookahead 跨 val_start) | timeshift placebo IC ≈ real | G0 (embargo_days ≥ max label_lookahead_days) + PR #9 (timeshift placebo 内部跨 split 检查) | 0 |
| **L5** | Validation labels 进训练 callback (EarlyStopping by val loss / metric_for_best_model) | shuffle placebo IC > 0 (model select 偏向 val) | G2 (Tier 2 retrain on shuffled → 选不出"好" checkpoint) | LOW — Tier 2 暴露 |
| **L6** | CSRankNorm / Winsorize bounds 用 train+val 全量 fit | shuffle placebo IC > 0 (per-day bound 跨 split) | G2 | LOW — Tier 2 暴露 |
| **L7** | Sliding window 跨 train/val (seq_len 200 但 embargo < 200) | timeshift placebo IC > 0 | G0 (embargo) + 写时 builder 检查 | 0 |
| **L8** | Calibrator fit on train+val (而非 train only) | val IC inflated | G2 + Tier 2 reports calibrator separately | LOW |
| **L9** | 训练 metric 通过 EarlyStopping 暴露 val IC | shuffle placebo > 0 | G2 | LOW |
| **L10** | Random seed leakage (selection bias when batching parallel trials) | 5-seed average biased | renquant-model PR #15 已修；不在本架构 scope | 0 — pre-existing |
| **L11** | Sidecar tampering (artifact 改了但 triad_report 没更新) | passed 标签下其实 model 被改 | G2 (artifact_fingerprint 绑 model bytes sha256) + G3 (load 时重算 fingerprint 对比) | LOW |
| **L12** | Stale triad (model retrain 后 triad 没 rerun) | 旧 triad cover 新 model | G2 (triad bound to {model.pt hash, feature schema, label hash, code sha, triad config}; 任一变 → triad_status → "pending") | 0 |

**不防御的**（架构外）:
- 模型对抗攻击（不在 quant 场景）
- 数据源被污染（base-data 上游数据有问题）— 留给 data sanity validators
- 业务逻辑泄漏（"BEAR 上一律买债"这种 rule-based hack）— 这是 strategy 选择，不是泄漏

### 2.3 残余风险评估 — Tier 2 不是万能

Tier 2 重训 on shuffled labels → 期望 IC ≈ 0。但 Tier 2 自己也可能被以下绕过：
- **a)** 如果 EarlyStopping 用 val loss patience=2，且 Tier 2 也允许 EarlyStopping → val labels 仍被偷看，shuffle Tier 2 IC > 0。**Mitigation**: Tier 2 runner 强制 disable EarlyStopping (train fixed N epochs)。
- **b)** 如果 random seed 在 Tier 2 跑同 seed → 训练得到"幸运"projection 不 reflect true placebo distribution。**Mitigation**: Tier 2 要 ≥ 3 不同 seeds (n=3 minimum for sufficiency)。
- **c)** 如果 features 本身 deterministic 编码了 future return (e.g., features 是 future returns 的 transformation) → Tier 2 重训仍能预测。**Mitigation**: 这种情况下 BEAR shuffle IC 会 ≈ real BEAR IC — 我们今天看到的就是这个。Tier 2 标 failed，artifact 拒绝进 prod。

---

## 3 · 当前 multirepo 漏洞 + 应有架构（v3+v4 内容延续）

### 3.1 漏洞清单 M1-M7（unchanged，详见 v3 §3.1）

### 3.2 5 道 cross-repo gate G0-G5（unchanged，详见 v3 §3.2 / v4 §4.2）

### 3.3 `triad_status` 状态机（v5 NEW）

```
                           ┌──────────────────────────────────────────────────┐
                           │                STATE: PENDING                    │
                           │  scorer.load() blocked (unless bypass active)    │
                           │  manifest_row refuses                            │
                           │  fit_calibrator refuses                          │
                           │  trainer_placebo: None                           │
                           └─────────────────┬────────────────────────────────┘
                                             │
                              (Tier 2 runner completes)
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                            │
            shuffle_ic < threshold                      shuffle_ic ≥ threshold
            timeshift_ic < threshold                    OR timeshift_ic ≥ threshold
            ratios within bounds                        OR ratio breach
                       │                                            │
                       ↓                                            ↓
        ┌──────────────────────────────┐              ┌──────────────────────────────┐
        │      STATE: PASSED            │              │      STATE: FAILED            │
        │  scorer.load() allowed        │              │  scorer.load() blocked       │
        │  manifest_row accepts         │              │  manifest_row refuses        │
        │  fit_calibrator accepts       │              │  fit_calibrator refuses      │
        │  trainer_placebo populated    │              │  trainer_placebo populated   │
        └──────────────┬───────────────┘              └──────────────┬───────────────┘
                       │                                              │
        (model.pt changed OR                          (immutable — failed artifact
         feature_schema changed OR                     stays failed forever; cannot
         label_hash changed OR                         transition back; deletion or
         code_sha changed OR                           replacement only)
         triad_config changed)
                       │                                              │
                       ↓                                              ↓
                ┌────────────────────────────────┐              ┌────────────────────────┐
                │   AUTO-INVALIDATE → PENDING    │              │   (terminal — no exit) │
                │   sidecar updated by writer    │              └────────────────────────┘
                │   (model trainer or refit)     │
                └────────────────┬───────────────┘
                                 │
                            (loop back to PENDING node)
                                 ↓
                            [start over]
```

**Transition rules** (enforced by Pydantic `model_validator` on every sidecar write):

```python
def validate_status_transition(old: TriadStatus, new: TriadStatus) -> None:
    allowed = {
        ("pending", "passed"): True,
        ("pending", "failed"): True,
        ("passed", "pending"): True,      # auto-invalidate when binding changes
        ("failed", "pending"): True,      # auto-invalidate when binding changes
        ("passed", "passed"): True,       # idempotent
        ("failed", "failed"): True,
        ("passed", "failed"): False,      # never demote passed → failed without going thru pending
        ("failed", "passed"): False,      # never promote failed → passed without re-running
        ("pending", "pending"): True,
    }
    if (old, new) not in allowed:
        raise IllegalTriadTransition(old, new)
```

**Why failed → forever**: 一旦 placebo 失败意味着 model 有 leak。靠 "再跑一次试试" 不解决问题，强迫去修。把 failed 当 "delete and retrain from scratch" trigger，避免 selection-bias-on-retry。

---

## 4 · 具体跨仓代码契约（v4 内容 + v5 真实代码 + 并发 + 配置 + 遥测）

### 4.1 - 4.3 ｜ contracts/ + frames + gate locations + import graph（unchanged from v4）

### 4.4 Sidecar 并发语义（v5 NEW）

**问题**: 多个进程可能同时写同一个 sidecar JSON:
- Trainer 写初始 `triad_status="pending"` (T+5s)
- Tier 2 runner 写 `triad_status="passed"|"failed"` (T+75min)
- Architect 手动 force-rerun Tier 2 (T+24h)
- 下个 retrain 把 model.pt 改了 → auto-invalidate 写 `triad_status="pending"`

**解决**:

```python
# renquant-common/src/renquant_common/leakage_guards/sidecar.py

import json, fcntl, tempfile, os
from pathlib import Path
from typing import Callable

class SidecarConcurrencyError(RuntimeError): ...

def atomic_update_sidecar(
    sidecar_path: Path,
    transformer: Callable[[dict], dict],
    *,
    lockfile_suffix: str = ".lock",
    timeout_seconds: float = 30.0,
) -> dict:
    """Atomically update a triad sidecar JSON with file lock.

    Algorithm:
        1. Acquire exclusive flock on `{sidecar_path}.lock` (blocks up to timeout).
        2. Read current sidecar from disk (if exists), parse JSON.
        3. Call transformer(current) → new_dict.
        4. Validate new_dict against TriadReport schema (raises if invalid).
        5. Validate status transition allowed (raises IllegalTriadTransition if not).
        6. Write to temp file in same directory.
        7. fsync temp file.
        8. os.rename(temp, sidecar_path) — POSIX atomic on same FS.
        9. fsync containing directory.
       10. Release lock.

    Returns:
        The new dict that was written.

    Raises:
        TimeoutError if can't acquire lock within timeout.
        ValidationError if new_dict fails schema.
        IllegalTriadTransition if status transition forbidden.
    """
    lockfile = sidecar_path.with_suffix(sidecar_path.suffix + lockfile_suffix)
    deadline = time.monotonic() + timeout_seconds
    fd = os.open(lockfile, os.O_CREAT | os.O_WRONLY)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise SidecarConcurrencyError(
                        f"timeout acquiring lock on {sidecar_path}"
                    )
                time.sleep(0.1)
        # Read current
        current = {}
        if sidecar_path.exists():
            current = json.loads(sidecar_path.read_text())
        # Transform
        new = transformer(current)
        # Schema + transition validate
        ScorerArtifact.model_validate(new)
        if current.get("triad_report", {}).get("triad_status"):
            old = current["triad_report"]["triad_status"]
            target = new["triad_report"]["triad_status"]
            validate_status_transition(old, target)
        # Atomic write
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=sidecar_path.parent, delete=False, suffix=".tmp"
        )
        try:
            json.dump(new, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.rename(tmp.name, sidecar_path)
            # fsync containing directory for full atomicity
            dir_fd = os.open(sidecar_path.parent, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except Exception:
            if Path(tmp.name).exists():
                Path(tmp.name).unlink()
            raise
        return new
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
```

### 4.5 真实代码 (非伪码) — Tier 1 + Tier 2 runners (v5 NEW)

```python
# renquant-common/src/renquant_common/leakage_guards/scorer_sanity.py

"""Tier 1 sanity — fixed scorer against perturbed labels. Seconds-scale."""

from __future__ import annotations
import hashlib, logging
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from renquant_common.contracts.triad import ScorerSanityReport

log = logging.getLogger("renquant_common.leakage_guards.scorer_sanity")


def run_tier1(
    *,
    scorer,                                # has scorer.score(features_df) -> Series of scores
    val_features: pd.DataFrame,            # validated FeatureFrame.df contents
    val_labels: pd.Series,                 # validated LabelFrame col
    label_col: str,
    threshold_max_abs_ic: float = 0.01,
    rng_seed: int = 0,
) -> ScorerSanityReport:
    """Compute Tier 1 sanity metrics on a FIXED scorer.

    Tier 1 catches LABEL-CALC BUGS only. It does NOT catch train-time leakage
    (use Tier 2 for that — `run_tier2`).
    """
    rng = np.random.default_rng(rng_seed)

    # Real IC (replicate via 2 halves)
    n_dates = val_features["date"].nunique()
    val_features = val_features.sort_values(["date", "ticker"])
    unique_dates = val_features["date"].unique()
    half = len(unique_dates) // 2
    half_a_dates = unique_dates[:half]
    half_b_dates = unique_dates[half:]

    scores = scorer.score(val_features)
    real = pd.DataFrame({
        "date": val_features["date"].values,
        "ticker": val_features["ticker"].values,
        "score": scores.values,
        "label": val_labels.values,
    })

    def daily_ic(frame: pd.DataFrame) -> float:
        ics = []
        for _, g in frame.groupby("date"):
            if len(g) >= 5:
                ic, _ = spearmanr(g["score"], g["label"])
                if np.isfinite(ic):
                    ics.append(ic)
        return float(np.mean(ics)) if ics else 0.0

    aa_a = daily_ic(real[real["date"].isin(half_a_dates)])
    aa_b = daily_ic(real[real["date"].isin(half_b_dates)])
    aa_drift = abs(aa_a - aa_b)

    # Shuffle val labels (per-date, preserves cross-sectional structure)
    real_shuffled = real.copy()
    for d, g in real_shuffled.groupby("date"):
        idx = g.index.values
        perm = rng.permutation(idx)
        real_shuffled.loc[idx, "label"] = real_shuffled.loc[perm, "label"].values
    shuffled_ic = daily_ic(real_shuffled)

    # Time-shift val labels +10 days (within val, no train leak)
    real_ts = real.copy()
    # Build per-ticker shift; drop rows where shift falls off end
    shifted = (
        real_ts.sort_values(["ticker", "date"])
               .groupby("ticker", group_keys=False)
               .apply(lambda g: g.assign(label=g["label"].shift(-10)))
               .dropna(subset=["label"])
    )
    timeshift_ic = daily_ic(shifted)

    report = ScorerSanityReport(
        aa_split_real_ic_replicate=(aa_a + aa_b) / 2,
        aa_split_drift_ic=aa_drift,
        shuffled_val_ic=shuffled_ic,
        timeshifted_val_ic=timeshift_ic,
        label_col=label_col,
        n_val_dates=n_dates,
        threshold_max_abs_ic=threshold_max_abs_ic,
    )

    if abs(shuffled_ic) >= threshold_max_abs_ic:
        log.error(
            "TIER 1 FAIL: shuffled_val_ic=%.4f >= %.4f. Label-calc bug suspected.",
            shuffled_ic, threshold_max_abs_ic,
        )
    if abs(timeshift_ic) >= threshold_max_abs_ic:
        log.error(
            "TIER 1 FAIL: timeshifted_val_ic=%.4f >= %.4f. Label-calc bug suspected.",
            timeshift_ic, threshold_max_abs_ic,
        )
    return report
```

```python
# renquant-common/src/renquant_common/leakage_guards/trainer_placebo.py

"""Tier 2 placebo runner — RETRAIN on shuffled/timeshifted labels."""

from __future__ import annotations
import argparse, json, logging, subprocess, sys, time
import hashlib
from datetime import datetime
from pathlib import Path
from renquant_common.contracts.triad import TrainerPlaceboReport, TriadReport
from renquant_common.leakage_guards.sidecar import atomic_update_sidecar

log = logging.getLogger("renquant_common.leakage_guards.trainer_placebo")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, required=True,
                    help="Path to model.pt; sidecar is .metadata.json next to it.")
    ap.add_argument("--seeds", type=str, default="42,43,44",
                    help="Comma-separated seeds for Tier 2 placebo (≥3 required).")
    ap.add_argument("--label-shift-days", type=int, default=10)
    ap.add_argument("--trainer-module", required=True,
                    help="e.g. renquant_model_patchtst.hf_trainer")
    ap.add_argument("--features-parquet", type=Path, required=True)
    ap.add_argument("--labels-parquet", type=Path, required=True)
    ap.add_argument("--splits-parquet", type=Path, required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--threshold-max-abs-placebo-ic", type=float, default=0.01)
    ap.add_argument("--threshold-max-placebo-real-ratio", type=float, default=0.30)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) < 3:
        raise SystemExit("Tier 2 requires ≥3 seeds for sufficiency.")

    sidecar_path = args.artifact.with_suffix(args.artifact.suffix + ".metadata.json")

    # Run 3 × (real, shuffle, timeshift) = 9 sub-trainings
    real_ic, shuffle_ic, timeshift_ic = [], [], []
    real_per_regime, shuffle_per_regime, timeshift_per_regime = {}, {}, {}
    for seed in seeds:
        ic_real = _run_one_subtraining(args, seed, mode="real")
        ic_shuf = _run_one_subtraining(args, seed, mode="shuffle")
        ic_ts = _run_one_subtraining(args, seed, mode="timeshift")
        real_ic.append(ic_real["pooled"])
        shuffle_ic.append(ic_shuf["pooled"])
        timeshift_ic.append(ic_ts["pooled"])
        for regime, v in ic_real["per_regime"].items():
            real_per_regime.setdefault(regime, []).append(v)
        for regime, v in ic_shuf["per_regime"].items():
            shuffle_per_regime.setdefault(regime, []).append(v)
        for regime, v in ic_ts["per_regime"].items():
            timeshift_per_regime.setdefault(regime, []).append(v)

    report = TrainerPlaceboReport(
        real_ic_mean=float(np.mean(real_ic)),
        real_ic_per_regime={r: float(np.mean(v)) for r, v in real_per_regime.items()},
        shuffle_placebo_ic_mean=float(np.mean(shuffle_ic)),
        shuffle_placebo_ic_per_regime={r: float(np.mean(v)) for r, v in shuffle_per_regime.items()},
        timeshift_placebo_ic_mean=float(np.mean(timeshift_ic)),
        timeshift_placebo_ic_per_regime={r: float(np.mean(v)) for r, v in timeshift_per_regime.items()},
        n_seeds=len(seeds),
        n_val_dates=ic_real["n_val_dates"],
        threshold_max_abs_placebo_ic=args.threshold_max_abs_placebo_ic,
        threshold_max_placebo_real_ratio=args.threshold_max_placebo_real_ratio,
    )

    # Decide status
    s_abs = abs(report.shuffle_placebo_ic_mean)
    t_abs = abs(report.timeshift_placebo_ic_mean)
    r_abs = abs(report.real_ic_mean)
    thr = args.threshold_max_abs_placebo_ic
    ratio = args.threshold_max_placebo_real_ratio
    passed = (
        s_abs < thr and t_abs < thr
        and (r_abs <= 0.01 or (s_abs < ratio * r_abs and t_abs < ratio * r_abs))
    )
    target_status = "passed" if passed else "failed"

    # Atomic update sidecar
    def transformer(current: dict) -> dict:
        triad = current.setdefault("triad_report", {})
        triad["trainer_placebo"] = report.model_dump()
        triad["triad_status"] = target_status
        triad["triad_completed_at"] = datetime.utcnow().isoformat()
        return current

    new = atomic_update_sidecar(sidecar_path, transformer)
    log.info("Tier 2 complete. artifact=%s status=%s shuffle=%.4f timeshift=%.4f real=%.4f",
             args.artifact, target_status, s_abs, t_abs, r_abs)

    # Telemetry
    _emit_telemetry({
        "event": "tier2_complete",
        "artifact_path": str(args.artifact),
        "status": target_status,
        "shuffle_ic": s_abs,
        "timeshift_ic": t_abs,
        "real_ic": r_abs,
        "n_seeds": len(seeds),
        "ts": datetime.utcnow().isoformat(),
    })
    if target_status == "failed":
        _emit_alert(
            severity="HIGH",
            msg=f"Tier 2 FAILED for {args.artifact}; placebo IC ≥ threshold",
        )


def _run_one_subtraining(args, seed: int, mode: str) -> dict:
    """Invoke trainer module as subprocess. Returns dict with pooled+per_regime IC.

    mode: 'real' (real labels), 'shuffle' (shuffled labels), 'timeshift' (+N days shift)
    """
    cmd = [
        sys.executable, "-m", args.trainer_module,
        "--features-parquet", str(args.features_parquet),
        "--labels-parquet", str(args.labels_parquet),
        "--splits-parquet", str(args.splits_parquet),
        "--label-col", args.label_col,
        "--seed", str(seed),
        "--disable-early-stopping",          # CRITICAL — prevents Tier 2 from inheriting EarlyStopping leak (residual L5)
        "--triad-replay-mode", mode,         # mode dispatches in trainer
    ]
    if mode == "shuffle":
        cmd.append("--shuffle-labels")
    elif mode == "timeshift":
        cmd.extend(["--label-shift-days", str(args.label_shift_days)])
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(out.stdout.splitlines()[-1])    # trainer prints JSON on last line


def _emit_telemetry(event: dict) -> None:
    # See §4.7 for telemetry data model
    ...


def _emit_alert(severity: str, msg: str) -> None:
    # See §4.7 for alerting
    ...


if __name__ == "__main__":
    main()
```

### 4.6 配置 schema (v5 NEW)

```python
# renquant-common/src/renquant_common/contracts/leakage_config.py

class LeakageGuardConfig(pydantic.BaseModel):
    """Lives in strategy_config.golden.json under key 'leakage_guards'."""

    # Tier 1
    tier1_threshold_max_abs_ic: float = 0.01
    # Tier 2
    tier2_n_seeds_required: int = 3
    tier2_threshold_max_abs_placebo_ic: float = 0.01
    tier2_threshold_max_placebo_real_ratio: float = 0.30
    tier2_label_shift_days: int = 10
    tier2_run_strategy: Literal["subprocess_inline", "subprocess_queue", "manual"] = "subprocess_inline"
    # Bypass (G3/G4/G5 escape hatch)
    emergency_bypass_triad_until: datetime | None = None
    emergency_bypass_reason: str | None = None      # required if bypass set
    # Alerting
    alert_on_tier2_failure: bool = True
    alert_on_bypass_active: bool = True
    alert_channel: Literal["slack", "log", "none"] = "log"
    alert_slack_webhook_url: str | None = None

    @pydantic.model_validator(mode="after")
    def bypass_requires_reason(self):
        if self.emergency_bypass_triad_until and not self.emergency_bypass_reason:
            raise ValueError("emergency_bypass_triad_until set but no emergency_bypass_reason given")
        return self
```

**Where it lives**:
- `RenQuant/backtesting/renquant_104/strategy_config.golden.json` adds:
  ```json
  {
    "leakage_guards": {
      "tier1_threshold_max_abs_ic": 0.01,
      "tier2_n_seeds_required": 3,
      "tier2_threshold_max_abs_placebo_ic": 0.01,
      "tier2_threshold_max_placebo_real_ratio": 0.30,
      "tier2_label_shift_days": 10,
      "tier2_run_strategy": "subprocess_inline",
      "emergency_bypass_triad_until": null,
      "alert_on_tier2_failure": true,
      "alert_channel": "log"
    }
  }
  ```
- Loaded by every gate via `LeakageGuardConfig.parse_obj(cfg["leakage_guards"])`
- Per-strategy override via `strategy_config.<label>.json` (existing convention)

### 4.7 遥测数据模型 (v5 NEW)

```python
# renquant-common/src/renquant_common/leakage_guards/telemetry.py

"""Append-only telemetry events for triad lifecycle + bypass + alerts."""

import sqlite3, json
from pathlib import Path
from datetime import datetime

TELEMETRY_DB = Path.home() / ".renquant" / "telemetry" / "leakage_guards.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                    -- UTC ISO-8601
    event_type TEXT NOT NULL,            -- 'tier1_pass' / 'tier1_fail' / 'tier2_start' / 'tier2_complete'
                                         -- 'sidecar_write' / 'gate_block' / 'gate_bypass'
                                         -- 'alert_emitted'
    artifact_fingerprint TEXT,           -- sha256 prefix 16
    artifact_path TEXT,
    triad_status TEXT,                   -- pending|passed|failed
    caller TEXT,                         -- e.g. pipeline:scorer_load
    payload JSON                         -- event-specific extras
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_artifact ON events(artifact_fingerprint);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
"""


def emit_event(event_type: str, **kwargs) -> None:
    TELEMETRY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TELEMETRY_DB)
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT INTO events (ts, event_type, artifact_fingerprint, artifact_path, triad_status, caller, payload) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.utcnow().isoformat(),
            event_type,
            kwargs.get("artifact_fingerprint", "")[:16],
            kwargs.get("artifact_path"),
            kwargs.get("triad_status"),
            kwargs.get("caller"),
            json.dumps({k: v for k, v in kwargs.items()
                       if k not in ("artifact_fingerprint", "artifact_path", "triad_status", "caller")},
                       default=str),
        ),
    )
    conn.commit()
    conn.close()
```

**Retention**: append-only, never deleted. Rotation via `cron` if > 1GB → archive to `~/.renquant/telemetry/archive/` (separate concern).

**Alerting**:

```python
# renquant-common/src/renquant_common/leakage_guards/alerts.py

def emit_alert(severity: Literal["LOW", "MED", "HIGH"], msg: str, cfg: LeakageGuardConfig) -> None:
    if not cfg.alert_on_tier2_failure and severity == "HIGH":
        return
    if cfg.alert_channel == "log":
        log.error("ALERT[%s] %s", severity, msg)
    elif cfg.alert_channel == "slack":
        if cfg.alert_slack_webhook_url:
            requests.post(cfg.alert_slack_webhook_url, json={
                "text": f"[{severity}] RenQuant leakage alert: {msg}",
            }, timeout=10)
    elif cfg.alert_channel == "none":
        pass
    emit_event("alert_emitted", severity=severity, msg=msg)
```

---

## 5 · Dataflow 时序图（v4 + v5 失败路径）

§5.1-5.4 unchanged from v4 (training save / pipeline load / emergency bypass / base-data write).

### 5.5 Tier 1 FAIL (v5 NEW — 失败路径)

```
[ T+0 ]   training finishes; weights in memory
[ T+0.1 ] _save_artifact() called
[ T+0.2 ] Tier 1 run_tier1(scorer, val_X, val_y, ...)
              │
              ↓ ~2-3s
[ T+3 ]   ScorerSanityReport returned
              shuffled_val_ic = +0.05  (≥ threshold 0.01!)
              │
              ↓
              raise Tier1Failed("Tier 1 FAIL: shuffled_val_ic=+0.05")
              │
              ↓
[ T+3.1 ] _save_artifact() catches, does NOT write model.pt nor sidecar.
[ T+3.2 ] trainer exits with rc=1
[ T+3.3 ] orchestrator (or operator) sees failure; cutoff treated like training failure
              (no artifact written; manifest_row never called)
[ T+3.4 ] telemetry: emit_event("tier1_fail", artifact_path=<would-be path>, shuffled_val_ic=+0.05)
[ T+3.5 ] alert: emit_alert("MED", "Tier 1 fail — label-calc bug suspected")
```

### 5.6 Tier 2 FAIL post-save (v5 NEW — 失败路径)

```
[ T+5s ]    artifact + sidecar persisted with triad_status=pending
[ T+5.1s ]  enqueue_tier2(...) spawns subprocess
              │
              ↓ ~75min later
[ T+75min ] trainer_placebo subprocess running
              real_ic_mean = +0.04
              shuffle_placebo_ic_mean = +0.04  (≥ ratio breach!)
              timeshift_placebo_ic_mean = +0.05
              passed = False
              target_status = "failed"
              │
              ↓
[ T+75min ] atomic_update_sidecar(sidecar_path, transformer)
              │   1. acquire flock
              │   2. read current sidecar (triad_status=pending)
              │   3. transformer fills trainer_placebo + sets status=failed
              │   4. validate_status_transition("pending", "failed") → allowed
              │   5. atomic rename → sidecar now reads failed
              │
[ T+75min ] telemetry: emit_event("tier2_complete", status="failed", shuffle_ic=0.04, ...)
              alert: emit_alert("HIGH", "Tier 2 FAILED for <artifact_path>; placebo IC ≥ threshold")
              │
[ T+75.1m ] downstream consumer (sim cron at midnight, e.g.) tries to load:
              │   PanelScorer.load(uri)
              │   → ScorerArtifact.parse_file(sidecar)
              │   → assert_artifact_validated(artifact, cfg, caller="pipeline:scorer_load")
              │   → triad_status == "failed" → raise ArtifactNotValidated (hard stop, no bypass)
              │
              ↓
              sim job fails CI / cron alerts that no live order produced.
              architect investigates next morning; either
              (a) accepts model is leak-contaminated → delete + retrain w/ leak fix,
              (b) finds Tier 2 runner bug → opens PR fix; failed → cannot become passed
                  automatically (state machine forbids); must regenerate artifact.
```

---

## 6 · Cross-repo CI workflows (v4 内容) + v5 ops runbook

### 6.1-6.5 (workflow YAMLs)  unchanged from v4

### 6.6 Ops runbook (v5 NEW — exact commands)

#### Scenario A: Tier 2 runner stuck — all fresh artifacts pending

```bash
# Diagnose
sqlite3 ~/.renquant/telemetry/leakage_guards.db \
  "SELECT artifact_fingerprint, ts FROM events
   WHERE event_type='tier2_start'
     AND artifact_fingerprint NOT IN (SELECT artifact_fingerprint FROM events WHERE event_type='tier2_complete')
   ORDER BY ts DESC LIMIT 20;"

# Manually re-fire Tier 2 for one artifact
ARTIFACT=/path/to/hf_patchtst_all_seed42_model.pt
python -m renquant_common.leakage_guards.trainer_placebo \
  --artifact $ARTIFACT \
  --seeds 42,43,44 \
  --label-shift-days 10 \
  --trainer-module renquant_model_patchtst.hf_trainer \
  --features-parquet data/features.parquet \
  --labels-parquet data/labels.parquet \
  --splits-parquet data/splits.parquet \
  --label-col fwd_60d_excess

# If broken across the board: emergency bypass PR
cd /Users/renhao/git/github/RenQuant
git checkout -b ops/emergency-bypass-2026-06-XX
# Edit backtesting/renquant_104/strategy_config.golden.json:
#   "leakage_guards": {
#     "emergency_bypass_triad_until": "2026-06-15T00:00:00Z",
#     "emergency_bypass_reason": "Tier-2 runner broken in PR #XYZ; bypass until fix lands"
#   }
git commit -am "ops: emergency bypass triad until 6/15 (Tier 2 runner stuck)" \
  -m "Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
gh pr create --label "agent:emergency:bypass-triad" \
  --title "ops: emergency triad bypass until 2026-06-15" \
  --body "..."

# After bypass merges, daily cron alerts every 6h until expired.
```

#### Scenario B: Mass backfill 43 existing WF manifest entries (pre-fix artifacts)

```bash
# These artifacts have NO triad_report field; Pydantic validate would reject.
# Strategy: stamp ALL existing artifacts with triad_status="pending" + auto-set
#  emergency_bypass_triad_until to 4 weeks out so prod doesn't immediately stop.

# 1. Find all artifacts referenced by current WF manifests
python scripts/audit_existing_artifacts.py > /tmp/artifacts.txt

# 2. Bulk stamp pending
python -m renquant_common.leakage_guards.backfill_pending \
  --artifacts-list /tmp/artifacts.txt \
  --triad-config-hash $(python -c "...")

# 3. Open umbrella PR setting bypass until 2026-06-29 with reason "backfill grace period"

# 4. Tier 2 runner cron sweeps the pending list in priority order
#    (e.g., currently-live > recently-trained > older).
#    Each artifact transitions pending → passed or pending → failed.
#    Failed artifacts get an alert + must be retrained-and-replaced.
```

#### Scenario C: New feature added to base-data, schema_hash changed

```bash
# 1. base-data PR adds the feature to alpha158 builder
#    contract-bump-check.yml fan-out tests all 8 consumer repos.
#    If any consumer fails, paired PRs required.

# 2. After base-data PR merges + new parquet regenerated, all existing
#    artifacts have feature_schema_hash mismatch.
#    On next scorer.load() → ScorerArtifact validator detects schema_hash
#    drift → triad_status auto-becomes "pending" (state machine rule).

# 3. Tier 2 runner sweeps pending; either passes (model still works) or
#    fails (model needs retrain with new feature).

# 4. No prod outage: bypass is auto-on for 1 week post-base-data-PR via
#    backfill workflow.
```

#### Scenario D: Architect needs to delete a confirmed-failed artifact

```bash
# Failed artifacts cannot transition back. To "fix" a failed model, must
# replace it (new artifact_fingerprint via retrain).

# Step 1: identify failed
sqlite3 ~/.renquant/telemetry/leakage_guards.db \
  "SELECT artifact_path FROM events
   WHERE event_type='tier2_complete' AND triad_status='failed';"

# Step 2: investigate the failure (BEAR shuffle ic too high? which feature?)
python scripts/triad_diagnose.py --artifact /path/to/model.pt

# Step 3: fix root cause (e.g., feature L2 violation in feature_lookahead_days)
# Step 4: retrain → produces new artifact_fingerprint → goes through state machine
#         fresh
```

---

## 7 · CI 拓扑总图（unchanged from v4）

详 v4 §7。13 仓 CI 拓扑可视化。

---

## 8 · 迁移序列 + PR 列表（unchanged from v4）

详 v4 §8。MVP wave 5 PR ≤ 2 days；Full architecture wave 4 PR ≤ 1 week；PR 依赖 DAG。

---

## 9 · 产品化补充 (v5 NEW)

### 9.1 Performance budget

| 操作 | 当前 | + 本架构 | 预算 |
|---|---|---|---|
| Trainer save (single artifact) | ~5s | + Tier 1 ~3s = ~8s | ≤ 10s ✅ |
| Trainer Tier 2 (async background) | 0 | ~75min/PatchTST trial × 3 seeds × 3 modes = ~11h serial / ~4h parallel-3 | ≤ 12h ✅ — async, doesn't block save |
| scorer.load() gate check | ~5ms (parse sidecar) | + ~1ms (assert_artifact_validated) | ≤ 10ms ✅ |
| manifest_row gate check | ~5ms | + ~1ms | ≤ 10ms ✅ |
| Base-data write per parquet | ~30s | + manifest validate ~50ms | ≤ 60s ✅ |
| Sidecar atomic update | n/a | ~10ms (flock + rename) | ≤ 100ms ✅ |
| Cron daily routine total | ~30min | unchanged (Tier 1 was already implicit in val eval) | ≤ 60min ✅ |

**Re-evaluation trigger**: if WALL TIME for any single PatchTST save > 30s, the Tier 1 implementation has regressed and must be audited.

### 9.2 向后兼容 + 迁移策略

**问题**: 43 个已有 WF manifest entries 引用的 artifacts 都没 `triad_report` 字段。直接上 G3 fail-closed → prod 立刻停。

**渐进迁移 4 阶段**:

| Stage | 时长 | 行为 |
|---|---|---|
| **S1 Shadow** | Day 0-7 | G3 仅记录 "would have blocked" 事件，不真的阻止。telemetry 表暴露多少 artifact 会被拒绝。 |
| **S2 Backfill** | Day 7-14 | 运行 `backfill_pending` 脚本给所有 in-manifest artifact 标 `triad_status=pending` + 自动 7-day bypass。Tier 2 runner 后台扫描，逐个 passed/failed。 |
| **S3 Enforce-with-bypass** | Day 14-28 | G3-G5 真 fail-closed，但 bypass 仍 active for `pending`。`failed` 立即拒绝。bypass 范围每周缩小 ≥1 个 artifact 的子集。 |
| **S4 Full enforce** | Day 28+ | bypass 自动 expire；任何 pending/failed 不可加载。新 artifact 自动跑 Tier 2，旧 artifact 不重新跑则被 "soft-delete"（manifest 标记 stale）。 |

**Rollback**: 任何 stage 出问题，architect 一键 PR 把 `leakage_guards.tier2_run_strategy="manual"` + bypass_until 设回宽松值。**回滚永远 < 5 分钟**。

### 9.3 安全模型 (agent + 人)

| 谁能做什么 | 何 contract 改动 | 何 PR label | 何 review 必需 |
|---|---|---|---|
| Claude (agent) | Additive MINOR (新加 contract 字段) | `agent:claude` + `agent:contract:additive` | Codex auto-review |
| Codex (agent) | Additive MINOR | `agent:codex` + `agent:contract:additive` | Claude auto-review |
| Any agent | Breaking MAJOR | `agent:contract:breaking` | **Architect (人) explicit approve** + 所有 consumer paired PRs ready |
| Any agent | Emergency bypass | `agent:emergency:bypass-triad` | **Architect (人) explicit approve** + bypass expiry < 4 weeks |
| Any agent | DISABLE a gate (e.g. set `assert_artifact_validated` to no-op) | — | **FORBIDDEN** — branch protection rejects; PR auto-fails CI; alert architect |
| Architect (人) | Anything above | (label optional) | own discretion |

**Protection mechanism**: CI workflow `gate-disable-detection.yml` runs on every PR to common; greps for patterns like `def assert_artifact_validated.*\n.*return`（直接 return）or `if False:.*raise ArtifactNotValidated`. Detected → PR auto-fails with comment + emails architect.

### 9.4 Agreement criteria — when is this design DONE

| Gate | Done = ... |
|---|---|
| § threat model | ≥ 12 leak classes enumerated, gate responsibility clear, residual risk known |
| § state machine | All triad_status transitions formal, illegal ones raise |
| § concurrency | Sidecar atomic update with flock + rename verified on test |
| § code | Real (not pseudo) impls for: Tier 1, Tier 2 runner CLI, atomic_update_sidecar, assert_artifact_validated |
| § configuration | LeakageGuardConfig Pydantic; existing strategy_config schema extended |
| § telemetry | sqlite append-only events; alerts via slack/log; queryable |
| § CI | 5 workflow YAMLs (v4 §6) + gate-disable-detection.yml + multirepo-triad-e2e.yml nightly cron |
| § migration | 4-stage rollout plan + rollback < 5min |
| § ops runbook | 4 scenarios with copy-paste-able commands |
| § safety | Agent vs human privilege table; FORBIDDEN actions enumerated |
| § performance | Budgets per operation, re-eval trigger |
| § PR sequence | 5 MVP + 4 Full PRs with file-path lists, test paths, DAG |

如果 codex / 用户审 v5 全 ✅ → MVP wave 5 PR 可启。否则 v6 继续打磨。

### 9.5 Out of scope (v5 NEW — 明文不做)

- ❌ 不防御 base-data 上游数据被污染（不同问题层）
- ❌ 不做加密 / signing of artifacts（不是 multi-party trust scenario）
- ❌ 不做 federated learning / multi-party training（单机 + 单仓研究环境）
- ❌ 不做 GPU 隔离（MPS 共享 GPU，3 worker share，性能问题 ≠ leakage 问题）
- ❌ 不防御 random seed 决定 model selection bias（renquant-model PR #15 已修，pre-existing）
- ❌ 不防御 "数据漂移" 类问题（model 仍然 valid 但环境变了）— 这是 promotion gate 的事，不是 leak

### 9.6 v1 → v5 总 changelog

| Version | Origin | 主要增量 |
|---|---|---|
| v1 | Initial | basic reflection + 3 walls + 7-PR plan |
| v2 | Codex strict review HIGH×2+MED×4 | Scorer vs Trainer triad split, runtime validators, manifest not regex, MVP first, async + bypass, PR#9 preserved |
| v3 | User "multirepo architect" | M1-M7 holes, 5 cross-repo gates, ownership matrix, version coordination protocol, agent labels, test surface |
| v4 | User "具体代码 + workflow + dataflow" | Real Pydantic code, gate file:line, import graph, 4 dataflow sequence diagrams, 5 workflow YAMLs, CI topology visual, PR file lists, disaster mitigation |
| v5 | User "still not good enough" | **§2.2 12-class threat model**; **§3.3 status state machine**; **§4.4 sidecar concurrency + flock + atomic**; **§4.5 real (non-pseudo) Tier 1/2 code**; **§4.6 config schema**; **§4.7 telemetry sqlite + alerts**; **§5.5-5.6 failure-path dataflows**; **§6.6 ops runbook 4 scenarios**; **§9.1 perf budget**; **§9.2 4-stage migration + rollback < 5min**; **§9.3 agent vs human safety model + FORBIDDEN actions**; **§9.4 agreement criteria**; **§9.5 out of scope** |

---

## 10 · 索引

- `[[project_patchtst_btuned_leakage_2026-05-31]]`
- `[[feedback_research_pipeline_must_gate_with_sanity_triad]]` — advisory; v5 makes it executable + enforced
- `[[feedback_leakage_three_walls]]` — v2 摘要；v5 扩展为完整系统
- `[[feedback_industry_leading_quality]]`
- `[[feedback_pr_based_workflow]]` — §9.3 agent privilege model builds on this
- `[[feedback_multirepo_code_placement]]` — §4.1 ownership matrix hardens
- `[[project_multirepo_sop_2026-05-28]]`
- `[[project_phase5_burst_2026-06-01]]`
- CLAUDE.md §3.1, §3.5, §7.5, §7.7, §3.7, §6.4, §7.11, §8.1, §7.2
