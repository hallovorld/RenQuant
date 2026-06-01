# 2026-06-01 · B_tuned 泄漏反思 + Multirepo 架构修复（v4 — 含具体代码、workflow、dataflow）

**Author**: Claude  
**Reviewers**: Codex (v1→v2: HIGH×2 + MED×4 已 address) · User (v2→v3: "我要 multirepo architect"; v3→v4: "具体代码 + workflow + dataflow 要更详细")  
**Context**: B_tuned PatchTST Tier-3 placebos 二次失败。  
**Frame**: 13-repo multirepo 架构的 contract ownership + cross-repo CI gates + 版本协调协议 + 具体代码契约 + dataflow 拓扑。

---

## 1 · 反思（unchanged from v3）— §6 末尾索引

跑 2h 没看盘上数据；5/31 没修根因就 6/01 重跑；报 ETA 不报概念；新数字第一行 log 前没看 sanity triad — 4 条 process violation。详 v3 历次 commit。

---

## 2 · 数据（unchanged）

```
              s42      s43      s44
real          0.061    0.041     —
shuffle       0.014    0.041    0.048
timeshift     0.041    0.049     —
BEAR shuffle  0.048    0.091    0.084     ← 应 ≈ 0
```

**两类 triad 必须分开**:
- Tier 1 (scorer-level, 秒级): 固定 scorer 喂 shuffled labels → 只能拦 label-calc 错误
- Tier 2 (trainer-level, 75min): **重训** on shuffled labels → 拦今天的 train-time leakage

---

## 3 · 当前 multirepo 漏洞 + 应有架构（v3 内容 — §3.1 漏洞清单 M1-M7 + §3.2 5 道闸门 G0-G5）

参见 v3 commit 13cf3f0 §3。v4 在此基础上加 **具体代码 + workflow + dataflow** 的实现细节。

---

## 4 · 具体跨仓代码契约（v4 NEW）

### 4.1 `renquant-common` 公开 API surface（contract owner）

**新加文件树**:

```
renquant-common/src/renquant_common/
  ├─ contracts/
  │   ├─ __init__.py
  │   ├─ data.py          # FeatureManifest, LabelManifest, SplitManifest, DatasetManifest
  │   ├─ frames.py        # FeatureFrame, LabelFrame, SplitAssignment  (runtime validators)
  │   ├─ scorer.py        # EXTEND existing: add triad_report field
  │   └─ triad.py         # TriadReport, ScorerSanityReport, TrainerPlaceboReport, TriadStatus
  ├─ leakage_guards/
  │   ├─ __init__.py
  │   ├─ scorer_sanity.py    # Tier 1: cheap, sync, blocks save
  │   ├─ trainer_placebo.py  # Tier 2: expensive, async runner
  │   └─ gate.py             # SHARED helper for all 5 consumer-side gates (§7.5)
  └─ tests/
      ├─ test_contracts_data.py
      ├─ test_contracts_triad.py
      ├─ test_leakage_guards_scorer_sanity.py
      └─ test_leakage_guards_gate.py
```

**核心契约 (Pydantic models)**:

```python
# renquant-common/src/renquant_common/contracts/triad.py

from __future__ import annotations
from datetime import datetime
from typing import Literal
import pydantic

TriadStatus = Literal["pending", "passed", "failed"]


class ScorerSanityReport(pydantic.BaseModel):
    """Tier 1: fixed scorer vs perturbed labels. Cheap (seconds)."""

    aa_split_real_ic_replicate: float          # split val into 2, expect similar IC
    aa_split_drift_ic: float                   # |diff between halves|; expect small
    shuffled_val_ic: float                     # |IC| on val w/ shuffled labels; expect ≈ 0
    timeshifted_val_ic: float                  # |IC| on val w/ +10d shifted labels; expect ≈ 0
    label_col: str
    n_val_dates: int
    threshold_max_abs_ic: float = 0.01         # gate threshold


class TrainerPlaceboReport(pydantic.BaseModel):
    """Tier 2: retrain on shuffled/timeshifted labels. Expensive (~75min/PatchTST trial)."""

    real_ic_mean: float                        # baseline from real trial
    real_ic_per_regime: dict[str, float]
    shuffle_placebo_ic_mean: float             # mean across seeds
    shuffle_placebo_ic_per_regime: dict[str, float]
    timeshift_placebo_ic_mean: float
    timeshift_placebo_ic_per_regime: dict[str, float]
    n_seeds: int                               # ≥ 3 for sufficiency
    n_val_dates: int
    threshold_max_abs_placebo_ic: float = 0.01
    threshold_max_placebo_real_ratio: float = 0.30


class TriadReport(pydantic.BaseModel):
    """Required field on every ScorerArtifact."""

    triad_status: TriadStatus
    scorer_sanity: ScorerSanityReport          # always populated (Tier 1 sync)
    trainer_placebo: TrainerPlaceboReport | None = None   # None iff status == pending

    # Binding: triad is BOUND to specific artifact + dataset + code rev
    artifact_fingerprint: str                  # sha256(model.pt bytes)
    feature_schema_hash: str                   # from FeatureManifest.schema_hash
    label_hash: str                            # sha256 of label col bytes used for training
    code_sha: str                              # git rev of model trainer at save time
    triad_config_hash: str                     # sha256 of (n_seeds, thresholds, label_shift_days)

    triad_completed_at: datetime | None        # None if pending
    triad_started_at: datetime                 # always set at save (== save time)

    @pydantic.model_validator(mode="after")
    def status_consistent(self):
        if self.triad_status == "pending":
            assert self.trainer_placebo is None, "pending must have trainer_placebo=None"
        elif self.triad_status == "passed":
            assert self.trainer_placebo is not None, "passed must have trainer_placebo populated"
            p = self.trainer_placebo
            t = p.threshold_max_abs_placebo_ic
            r = p.threshold_max_placebo_real_ratio
            assert abs(p.shuffle_placebo_ic_mean) < t
            assert abs(p.timeshift_placebo_ic_mean) < t
            if abs(p.real_ic_mean) > 0.01:
                assert abs(p.shuffle_placebo_ic_mean) < r * abs(p.real_ic_mean)
                assert abs(p.timeshift_placebo_ic_mean) < r * abs(p.real_ic_mean)
        elif self.triad_status == "failed":
            # at least one threshold violated
            assert self.trainer_placebo is not None
        return self
```

```python
# renquant-common/src/renquant_common/contracts/scorer.py
# EXTEND existing ScorerArtifact (already exists for some fields)

class ScorerArtifact(pydantic.BaseModel):
    # ... existing fields (model_uri, feature_cols, seq_len, etc.) ...

    triad_report: TriadReport                  # NEW REQUIRED FIELD
    # ↑ Pydantic raises ValidationError on .parse_file() if absent.
    # Old artifacts (pre-PR) must be re-stamped or treated as unloadable.
```

```python
# renquant-common/src/renquant_common/contracts/data.py

class FeatureManifest(pydantic.BaseModel):
    """Closed declaration. Adding a column without bumping this = base-data write fails."""

    feature_cols: list[str]
    feature_dtypes: dict[str, str]
    feature_lookahead_days: dict[str, int]     # for each feature: max forward bars touched (0 = pure backward)
    schema_hash: str                           # sha256(sorted(feature_cols), dtypes, lookahead)
    schema_version: int                        # semver MINOR; bump on additive change

    @pydantic.model_validator(mode="after")
    def lookahead_zero(self):
        violators = {k: v for k, v in self.feature_lookahead_days.items() if v > 0}
        if violators:
            raise ValueError(f"features with positive lookahead = leakage candidates: {violators}")
        return self


class LabelManifest(pydantic.BaseModel):
    label_cols: list[str]
    label_lookahead_days: dict[str, int]       # required positive (it's a label after all)
    schema_hash: str


class SplitManifest(pydantic.BaseModel):
    train_end: datetime
    val_start: datetime
    val_end: datetime
    embargo_days: int                          # ≥ max label_lookahead_days
    schema_hash: str


class DatasetManifest(pydantic.BaseModel):
    features: FeatureManifest
    labels: LabelManifest
    splits: SplitManifest
    base_data_version: str
    written_at: datetime

    @pydantic.model_validator(mode="after")
    def features_labels_disjoint(self):
        overlap = set(self.features.feature_cols) & set(self.labels.label_cols)
        if overlap:
            raise ValueError(f"features ∩ labels = {overlap}; must be disjoint")
        return self

    @pydantic.model_validator(mode="after")
    def embargo_covers_lookahead(self):
        max_label_lookahead = max(self.labels.label_lookahead_days.values())
        if self.splits.embargo_days < max_label_lookahead:
            raise ValueError(
                f"embargo_days={self.splits.embargo_days} < max label lookahead "
                f"{max_label_lookahead}; cross-split leak possible"
            )
        return self
```

```python
# renquant-common/src/renquant_common/contracts/frames.py

class FeatureFrame:
    """Construction MUST go through .from_parquet(). Bare __init__ raises if df+manifest inconsistent."""

    def __init__(self, df: pd.DataFrame, manifest: FeatureManifest):
        self._validate(df, manifest)
        self._df = df
        self._manifest = manifest

    @classmethod
    def from_parquet(cls, features_path: Path, manifest_path: Path) -> FeatureFrame:
        df = pd.read_parquet(features_path)
        manifest = FeatureManifest.parse_file(manifest_path)
        return cls(df, manifest)  # __init__ runs _validate

    @staticmethod
    def _validate(df: pd.DataFrame, manifest: FeatureManifest) -> None:
        index_cols = {"ticker", "date"}
        actual = set(df.columns) - index_cols
        declared = set(manifest.feature_cols)
        if actual != declared:
            extra = actual - declared
            missing = declared - actual
            raise ValueError(f"FeatureFrame schema mismatch — extra={extra}, missing={missing}")
        for col, expected_dtype in manifest.feature_dtypes.items():
            if str(df[col].dtype) != expected_dtype:
                raise ValueError(f"{col}: dtype={df[col].dtype}, expected {expected_dtype}")
        # Recompute schema_hash and check
        import hashlib
        h = hashlib.sha256()
        for c in sorted(manifest.feature_cols):
            h.update(c.encode())
            h.update(manifest.feature_dtypes[c].encode())
            h.update(str(manifest.feature_lookahead_days[c]).encode())
        if h.hexdigest() != manifest.schema_hash:
            raise ValueError("schema_hash mismatch — manifest tampered with")

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def manifest(self) -> FeatureManifest:
        return self._manifest
```

**Shared gate helper (§7.5 single source for G3/G4/G5)**:

```python
# renquant-common/src/renquant_common/leakage_guards/gate.py

class ArtifactNotValidated(RuntimeError):
    """Raised by consumer gates (G3/G4/G5) refusing unvalidated scorer artifacts."""

def assert_artifact_validated(
    artifact: ScorerArtifact,
    *,
    cfg: dict | None = None,
    caller: str,                                  # "pipeline:scorer_load" / "backtesting:manifest_row" / etc.
) -> None:
    """SINGLE implementation used by all 4 consumer-side gates.

    Behavior:
        triad_status == "passed":  allow
        triad_status == "failed":  hard-stop, no bypass
        triad_status == "pending": allow IFF cfg["emergency_bypass_triad_until"] is in the future
                                   AND the artifact's triad_started_at is within the bypass window.
                                   Logs a warning + records bypass event.
    """
    s = artifact.triad_report.triad_status
    if s == "passed":
        return
    if s == "failed":
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer with triad_status='failed' "
            f"(fingerprint={artifact.triad_report.artifact_fingerprint[:16]}); "
            f"no bypass allowed for failed."
        )
    # pending
    cfg = cfg or {}
    bypass_until_str = cfg.get("emergency_bypass_triad_until")
    if not bypass_until_str:
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer with triad_status='pending' "
            f"(no emergency_bypass_triad_until set in cfg)."
        )
    bypass_until = pd.Timestamp(bypass_until_str)
    if pd.Timestamp.now() > bypass_until:
        raise ArtifactNotValidated(
            f"{caller}: emergency_bypass_triad_until={bypass_until_str} EXPIRED; "
            f"refusing pending artifact."
        )
    import logging
    log = logging.getLogger("renquant_common.leakage_guards.gate")
    log.warning(
        "TRIAD BYPASS ACTIVE: %s loading pending artifact %s (bypass expires %s)",
        caller, artifact.triad_report.artifact_fingerprint[:16], bypass_until_str,
    )
    _record_bypass_event(artifact.triad_report.artifact_fingerprint, caller, bypass_until_str)
```

### 4.2 5 道 cross-repo gate — 具体调用点

| Gate | 文件 + 行 | 调用契约 |
|---|---|---|
| **G0** base-data write-time | `renquant-base-data/src/renquant_base_data/builders/alpha158.py` (将由 #7 PR 加) | 在 `_write_dataset()` 末尾，pre-existing parquet.write 之前调用 `DatasetManifest(...).model_validate(...)` |
| **G1** trainer Tier-1 sync | `renquant-model-patchtst/src/renquant_model_patchtst/hf_trainer.py::_save_artifact` (在现有 save 之前插入) | `from renquant_common.leakage_guards.scorer_sanity import run_tier1`; if `result.failed`: raise; else stamp into ScorerArtifact |
| **G2** trainer Tier-2 async | `renquant-model-patchtst/src/renquant_model_patchtst/post_save_hook.py` (NEW) | spawn `python -m renquant_common.leakage_guards.trainer_placebo --artifact <uri>` as subprocess (or queue submit); runner writes `triad_status` back to artifact sidecar JSON |
| **G3** pipeline scorer.load | `renquant-pipeline/src/renquant_pipeline/kernel/panel_pipeline/panel_scorer.py::PanelScorer.load` | wrap existing load with `assert_artifact_validated(artifact, cfg=load_cfg, caller="pipeline:scorer_load")` |
| **G4a** backtesting manifest_row | `renquant-orchestrator/src/renquant_orchestrator/build_patchtst_wf_manifest.py::manifest_row` (and gbdt sibling) | call `assert_artifact_validated(...)` before appending to manifest; failed/pending → skip cutoff |
| **G4b** umbrella `scripts/fit_walkforward_calibrators.py::_fit_one` | same: refuse to fit calibrator on unvalidated scorer | |
| **G5** live execution | `renquant-execution/src/renquant_execution/broker_adapter.py::submit_order` (将由 #5/#9 PR 加) | resolve scorer artifact behind the order; `assert_artifact_validated(...)`; refused → log "refused_order_unvalidated_scorer" telemetry, do not submit |

### 4.3 完整 import 关系图（concrete cross-repo statements）

```python
# ============== renquant-base-data ==============
# alpha158 builder writes 3 parquet + manifest.json
from renquant_common.contracts.data import (
    DatasetManifest, FeatureManifest, LabelManifest, SplitManifest
)
# NO import of model-* (data layer doesn't know about models)


# ============== renquant-model-patchtst ==============
# hf_trainer.py reads typed frames, writes ScorerArtifact w/ triad
from renquant_common.contracts.frames import FeatureFrame, LabelFrame, SplitAssignment
from renquant_common.contracts.scorer import ScorerArtifact
from renquant_common.contracts.triad import (
    TriadReport, TriadStatus, ScorerSanityReport, TrainerPlaceboReport
)
from renquant_common.leakage_guards.scorer_sanity import run_tier1
from renquant_common.leakage_guards.trainer_placebo import enqueue_tier2

# NO import of renquant-pipeline / renquant-backtesting (model layer doesn't know about consumers)


# ============== renquant-pipeline ==============
# panel_scorer.py uses gate helper at load()
from renquant_common.contracts.scorer import ScorerArtifact
from renquant_common.leakage_guards.gate import assert_artifact_validated, ArtifactNotValidated


# ============== renquant-backtesting ==============
# wf_gate/runner.py + sim_driver same import pattern
from renquant_common.contracts.scorer import ScorerArtifact
from renquant_common.leakage_guards.gate import assert_artifact_validated


# ============== renquant-orchestrator ==============
# build_patchtst_wf_manifest.py
from renquant_common.contracts.scorer import ScorerArtifact
from renquant_common.leakage_guards.gate import assert_artifact_validated


# ============== renquant-execution ==============
from renquant_common.contracts.scorer import ScorerArtifact
from renquant_common.leakage_guards.gate import assert_artifact_validated


# ============== RenQuant (umbrella) ==============
# All scripts/ are thin wrappers. Same imports as above when they touch artifacts.
# Notably: scripts/fit_walkforward_calibrators.py uses gate before fitting cal.
```

**Key invariant**: `renquant-common` is the only repo every other one imports from. No A→B→C chains across repos. **Acyclic by construction**.

---

## 5 · Dataflow 时序图（v4 NEW）

### 5.1 Training save (Tier 1 sync + Tier 2 async)

```
[ Tick t=0 ]   model.train() finishes; weights live in memory
              │
              ↓
[ t+0s ]   renquant-model-patchtst/hf_trainer.py::_save_artifact()
              │
              ├──→ ScorerArtifact draft = ScorerArtifact(
              │       model_uri=<path>.pt,
              │       triad_report=None,         ← will populate below
              │       ... other fields ...
              │   )
              │
              ├──→ Tier 1 sync block (G1):
              │   from renquant_common.leakage_guards.scorer_sanity import run_tier1
              │   tier1_report: ScorerSanityReport = run_tier1(
              │       scorer=in_memory_scorer,
              │       val_features=val_X, val_labels=val_y,
              │       label_col=label_col,
              │       threshold_max_abs_ic=0.01,
              │   )
              │   ↓ 1-3 seconds
              │   tier1_failed = (
              │       abs(tier1_report.shuffled_val_ic) >= 0.01
              │       OR abs(tier1_report.timeshifted_val_ic) >= 0.01
              │   )
              │   if tier1_failed:
              │       raise Tier1Failed(...)   ← model file NOT saved; trainer exits 1
              │
[ t+3s ]   draft.triad_report = TriadReport(
              │       triad_status="pending",
              │       scorer_sanity=tier1_report,
              │       trainer_placebo=None,
              │       artifact_fingerprint=sha256(model.pt bytes),
              │       feature_schema_hash=manifest.schema_hash,
              │       label_hash=sha256(label_col bytes),
              │       code_sha=git_rev_parse_head,
              │       triad_config_hash=sha256(triad_config_payload),
              │       triad_started_at=now(),
              │       triad_completed_at=None,
              │   )
              │
              ├──→ Persist:
              │       torch.save(<model.pt>)
              │       <model.pt>.metadata.json ← write draft.model_dump_json()
              │
[ t+5s ]   Save complete; trainer exits 0
              │
              ├──→ Tier 2 enqueue (G2, async):
              │   enqueue_tier2(
              │       artifact_path=<model.pt>,
              │       seeds=[42, 43, 44],          ← ≥3 seeds for sufficiency
              │       label_shift_days=10,
              │       run_strategy="subprocess" | "queue",
              │   )
              │
              ↓ background process
              │
[ t+~75min ] tier2 subprocess (renquant_common.leakage_guards.trainer_placebo):
              │   for seed in (42, 43, 44):
              │       train_replay(features, shuffled_labels, seed) → ic_shuffle
              │       train_replay(features, timeshifted_labels, seed) → ic_timeshift
              │       train_replay(features, real_labels, seed) → ic_real
              │   placebo_report = TrainerPlaceboReport(
              │       real_ic_mean=mean(ic_real),
              │       shuffle_placebo_ic_mean=mean(ic_shuffle),
              │       timeshift_placebo_ic_mean=mean(ic_timeshift),
              │       ...
              │   )
              │
              ├──→ Decide triad_status:
              │   passed = (
              │       abs(placebo_report.shuffle_placebo_ic_mean) < 0.01
              │       AND abs(placebo_report.timeshift_placebo_ic_mean) < 0.01
              │       AND (
              │           abs(real_ic_mean) <= 0.01
              │           OR abs(shuffle_ic) < 0.30 * abs(real_ic)
              │           AND abs(timeshift_ic) < 0.30 * abs(real_ic)
              │       )
              │   )
              │
[ t+75min ] Update sidecar JSON in-place:
              │   sidecar = json.load(<model.pt>.metadata.json)
              │   sidecar["triad_report"]["triad_status"] = "passed" if passed else "failed"
              │   sidecar["triad_report"]["trainer_placebo"] = placebo_report.model_dump()
              │   sidecar["triad_report"]["triad_completed_at"] = now()
              │   json.dump(sidecar, <model.pt>.metadata.json)  ← atomic via temp+rename
              │
              ↓
[ t+75min ] Sidecar atomically updated. Downstream consumers see "passed" or "failed".
```

### 5.2 Pipeline scorer.load() (G3 fail-closed)

```
[ Time T ]   downstream caller asks scorer for predictions
              │
              ↓
              PanelScorer.load(uri="...hf_patchtst_seed42_model.pt")
              │
              ├──→ artifact = ScorerArtifact.parse_file(uri + ".metadata.json")
              │   (Pydantic raises ValidationError if triad_report missing entirely
              │    — old pre-fix artifacts behave as "unloadable")
              │
              ├──→ from renquant_common.leakage_guards.gate import assert_artifact_validated
              │   try:
              │       assert_artifact_validated(
              │           artifact,
              │           cfg=self._load_cfg,
              │           caller="pipeline:scorer_load",
              │       )
              │   except ArtifactNotValidated as e:
              │       log.error(e)
              │       raise  ← scorer.load() propagates; caller decides skip or stop
              │
              ├──→ Load model weights, build PanelScorer instance
              │
              ↓
              Return PanelScorer
```

### 5.3 Emergency bypass (escape hatch)

```
[ Scenario ]   Tier-2 runner has a bug; all fresh artifacts stuck on triad_status=pending
               Live trading day approaches; team needs to deploy a known-good earlier artifact
                or accept pending on a manually-vetted one.
              │
              ↓
              Architect (human) creates RenQuant umbrella PR:
              ─────────────────────────────────────────
              # strategy_config.golden.json
              {
                ...
                "emergency_bypass_triad_until": "2026-06-15T00:00:00Z",
                "emergency_bypass_reason": "Tier-2 runner broken; bug fix in PR #XYZ; bypass until 6/15",
                ...
              }
              ─────────────────────────────────────────
              PR labels: agent:emergency:bypass-triad (auto-requests architect human sign-off)
              PR cannot self-merge — requires GitHub branch protection override OR
                  human review approval — automation gates block.
              │
              ↓
              After merge:
              │
              ├──→ Cron jobs that call scorer.load() now see cfg["emergency_bypass_triad_until"]
              ├──→ assert_artifact_validated() logic:
              │       triad_status == "passed":    allow (unchanged)
              │       triad_status == "failed":    BLOCK (bypass NEVER applies to failed)
              │       triad_status == "pending":   allow IFF now() < 2026-06-15
              │                                    → log "TRIAD BYPASS ACTIVE: ..."
              │                                    → call _record_bypass_event(...)
              │                                       (writes to telemetry + slack alert)
              │
              ↓
[ T+1day ]    cron: scripts/alert_active_triad_bypass.py (runs hourly)
              │   reads cfg; if bypass active → slack ping every 6h until expired
              │
[ T+expired ] assert_artifact_validated() refuses again; pending artifacts blocked
              │
              ↓
              Architect either (a) opens new PR to extend (with reason), or
                                (b) lets it expire and resolves Tier-2 runner.
              Auto-expiry prevents "temporary" from becoming forever.
```

### 5.4 Base-data write-time validation (G0)

```
[ Trigger ]  base-data/scripts/build_alpha158_panel.py runs
              │
              ├──→ Builds raw panel: features_df (DataFrame), labels_df (DataFrame)
              │
              ├──→ Constructs DatasetManifest:
              │       feat_manifest = FeatureManifest(
              │           feature_cols=list(features_df.columns - {ticker, date}),
              │           feature_dtypes={c: str(features_df[c].dtype) for c in ...},
              │           feature_lookahead_days={c: 0 for c in ...},  ← features must all be 0
              │           schema_hash=...,
              │           schema_version=N,
              │       )
              │       lab_manifest = LabelManifest(label_cols=["fwd_60d_excess", ...], ...)
              │       split_manifest = SplitManifest(train_end=..., embargo_days=60, ...)
              │       ds_manifest = DatasetManifest(features=feat_manifest, labels=lab_manifest,
              │                                     splits=split_manifest, ...)
              │
              ├──→ Pydantic model_validator runs:
              │       - features ∩ labels = ∅  → raise if violated (e.g., fwd_60d in feature_cols)
              │       - embargo_days ≥ max label_lookahead_days
              │       - feature_lookahead_days all 0
              │       - schema_hash matches recomputation
              │
              ├──→ Persist atomically:
              │       features.parquet
              │       labels.parquet
              │       splits.parquet
              │       manifest.json  ← ds_manifest.model_dump_json()
              │
[ Downstream ] model-* loads via FeatureFrame.from_parquet(features.parquet, manifest.json):
              │   validates that on-disk parquet still matches manifest at load time
              │   (catches manual tampering with parquet between write and read)
```

---

## 6 · Cross-repo CI workflows (v4 NEW — exact YAML)

### 6.1 contract-bump-check.yml (runs in renquant-common on every PR touching contracts)

```yaml
# renquant-common/.github/workflows/contract-bump-check.yml
name: contract-bump-check
on:
  pull_request:
    paths:
      - 'src/renquant_common/contracts/**'
      - 'src/renquant_common/leakage_guards/**'

jobs:
  fan_out:
    runs-on: ubuntu-latest
    strategy:
      fail-fast: false
      matrix:
        consumer:
          - hallovorld/renquant-base-data
          - hallovorld/renquant-model-patchtst
          - hallovorld/renquant-model-gbdt
          - hallovorld/renquant-model-linear
          - hallovorld/renquant-pipeline
          - hallovorld/renquant-backtesting
          - hallovorld/renquant-orchestrator
          - hallovorld/renquant-execution
    steps:
      - name: checkout this PR's common
        uses: actions/checkout@v4
        with: { path: renquant-common }
      - name: checkout consumer at main
        uses: actions/checkout@v4
        with:
          repository: ${{ matrix.consumer }}
          path: consumer
      - name: install common from PR + consumer from main
        run: |
          pip install -e renquant-common
          pip install -e consumer
      - name: run consumer test suite
        run: cd consumer && python -m pytest tests/ -q --tb=line
      - name: report status
        if: failure()
        run: |
          echo "::error::Consumer ${{ matrix.consumer }} broken by this contract change."
          echo "::error::Either (a) keep change additive, or (b) open paired consumer PR."
```

### 6.2 dataset-schema-check.yml (runs in renquant-base-data on every PR)

```yaml
# renquant-base-data/.github/workflows/dataset-schema-check.yml
name: dataset-schema-check
on: pull_request

jobs:
  schema_validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: install
        run: pip install -e . pytest
      - name: generate synthetic dataset
        run: |
          python -c "
          from renquant_base_data.builders.alpha158 import build_small_sample
          build_small_sample(out_dir='/tmp/test_ds', n_tickers=2, n_days=100)
          "
      - name: validate manifest
        run: |
          python -c "
          from pathlib import Path
          from renquant_common.contracts.data import DatasetManifest
          ds = DatasetManifest.parse_file('/tmp/test_ds/manifest.json')
          # model_validators raise on disjoint/embargo/lookahead violations
          print('manifest valid; features:', len(ds.features.feature_cols))
          "
      - name: validate FeatureFrame load
        run: |
          python -c "
          from renquant_common.contracts.frames import FeatureFrame
          ff = FeatureFrame.from_parquet(
              '/tmp/test_ds/features.parquet',
              '/tmp/test_ds/manifest.json'
          )
          print('FeatureFrame loaded:', ff.df.shape)
          "
```

### 6.3 triad-tier1.yml (runs in renquant-model-* on every PR)

```yaml
# renquant-model-patchtst/.github/workflows/triad-tier1.yml
name: triad-tier1
on: pull_request

jobs:
  tier1_synth:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: install
        run: pip install -e . pytest
      - name: synth tiny train (60s wall clock)
        run: |
          python -c "
          from renquant_model_patchtst.hf_trainer import train_typed
          import tests.fixtures.synth as fx
          features = fx.synth_feature_frame(n_tickers=3, n_days=200, n_features=5)
          labels = fx.synth_label_frame_with_known_ic(features, ic=0.05)
          splits = fx.synth_split(features, val_frac=0.2)
          artifact = train_typed(features, labels, splits, args=fx.tiny_args())
          # ScorerArtifact must have populated triad_report.scorer_sanity
          assert artifact.triad_report.scorer_sanity is not None
          assert artifact.triad_report.triad_status in ('pending',)  # Tier 2 not yet run
          # ScorerSanityReport must show |shuffle_ic| < threshold
          assert abs(artifact.triad_report.scorer_sanity.shuffled_val_ic) < 0.01
          print('Tier 1 sanity passed on synthetic data.')
          "
```

### 6.4 artifact-gate.yml (runs in renquant-pipeline/backtesting/orchestrator/execution)

```yaml
# renquant-pipeline/.github/workflows/artifact-gate.yml
name: artifact-gate
on: pull_request

jobs:
  gate_behavior:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e . pytest
      - name: synthesize artifact at each triad_status
        run: |
          python -c "
          import json, tempfile
          from pathlib import Path
          from renquant_common.contracts.scorer import ScorerArtifact
          from renquant_common.contracts.triad import (
              TriadReport, ScorerSanityReport, TrainerPlaceboReport
          )
          from renquant_common.leakage_guards.gate import (
              assert_artifact_validated, ArtifactNotValidated
          )
          # Test 1: passed → allow
          art_passed = ... # synth with triad_status='passed'
          assert_artifact_validated(art_passed, caller='test')
          # Test 2: failed → block (no bypass)
          art_failed = ... # synth with triad_status='failed'
          try:
              assert_artifact_validated(art_failed, cfg={'emergency_bypass_triad_until': '2099-01-01'}, caller='test')
              raise AssertionError('failed must hard-stop even with bypass set')
          except ArtifactNotValidated:
              pass
          # Test 3: pending without bypass → block
          art_pending = ... # synth with triad_status='pending'
          try:
              assert_artifact_validated(art_pending, caller='test')
              raise AssertionError('pending without bypass must block')
          except ArtifactNotValidated:
              pass
          # Test 4: pending with active bypass → allow
          assert_artifact_validated(
              art_pending,
              cfg={'emergency_bypass_triad_until': '2099-01-01T00:00:00Z'},
              caller='test',
          )
          # Test 5: pending with expired bypass → block
          try:
              assert_artifact_validated(
                  art_pending,
                  cfg={'emergency_bypass_triad_until': '2020-01-01T00:00:00Z'},
                  caller='test',
              )
              raise AssertionError('expired bypass must block')
          except ArtifactNotValidated:
              pass
          print('Gate behavior tests passed.')
          "
```

### 6.5 E2E integration test (RenQuant umbrella)

```yaml
# RenQuant/.github/workflows/multirepo-triad-e2e.yml
name: multirepo-triad-e2e
on:
  workflow_dispatch: {}  # manual + nightly
  schedule:
    - cron: '0 5 * * *'  # 5am UTC daily

jobs:
  e2e:
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4
        with: { repository: hallovorld/RenQuant }
      - name: pull all renquant siblings
        run: |
          mkdir -p ~/work/rq
          for r in renquant-common renquant-base-data renquant-model-patchtst \
                   renquant-pipeline renquant-backtesting renquant-orchestrator; do
            git clone --depth 1 https://github.com/hallovorld/$r.git ~/work/rq/$r
          done
      - name: install all in editable
        run: |
          for r in ~/work/rq/*; do (cd $r && pip install -e .); done
          pip install -e .
      - name: e2e — base-data → model → pipeline → backtesting
        run: python -m pytest tests/test_multirepo_triad_e2e.py -v
```

```python
# RenQuant/tests/test_multirepo_triad_e2e.py
def test_e2e_unvalidated_artifact_blocked():
    # 1. Use base-data builders to write synth features/labels/splits + manifest
    # 2. Use model-patchtst trainer to produce ScorerArtifact (triad_status=pending after Tier1)
    # 3. Try to load via pipeline.PanelScorer.load() — should raise ArtifactNotValidated
    # 4. Try to assemble manifest_row via orchestrator — should raise
    # 5. Try to fit calibrator via fit_walkforward_calibrators — should raise
```

---

## 7 · Multirepo CI 拓扑总图

```
                                   ┌─────────────────────────────────────────────┐
                                   │  renquant-common CI:                         │
                                   │  ✓ unit tests                                │
                                   │  ✓ contract-bump-check.yml (fan-out matrix)  │
                                   │                                              │
                                   │  ← any contract PR triggers consumer tests   │
                                   │    on 8 sibling repos. PR red if any sib     │
                                   │    fails.                                    │
                                   └────────────────────┬─────────────────────────┘
                                                        │
        ┌───────────────────┬───────────────────┬───────┴────────────┬─────────────────────┬───────────────────────┐
        ↓                   ↓                   ↓                    ↓                     ↓                       ↓
   ┌────────────┐    ┌────────────┐     ┌────────────┐      ┌────────────┐         ┌────────────┐          ┌────────────┐
   │ base-data  │    │ model-*    │     │ pipeline   │      │ backtest   │         │ orchest.   │          │ execution  │
   │ ─────────  │    │ ─────────  │     │ ─────────  │      │ ─────────  │         │ ─────────  │          │ ─────────  │
   │ unit       │    │ unit       │     │ unit       │      │ unit       │         │ unit       │          │ unit       │
   │ dataset-   │    │ triad-     │     │ artifact-  │      │ artifact-  │         │ artifact-  │          │ artifact-  │
   │ schema-    │    │ tier1.yml  │     │ gate.yml   │      │ gate.yml   │         │ gate.yml   │          │ gate.yml   │
   │ check.yml  │    │ triad-     │     │            │      │            │         │            │          │            │
   │            │    │ tier2-     │     │ ALSO runs  │      │ ALSO runs  │         │ ALSO runs  │          │ ALSO runs  │
   │            │    │ async.yml  │     │ contract-  │      │ contract-  │         │ contract-  │          │ contract-  │
   │            │    │            │     │ bump-check │      │ bump-check │         │ bump-check │          │ bump-check │
   │            │    │            │     │ on PRs     │      │ on PRs     │         │ on PRs     │          │ on PRs     │
   └────────────┘    └────────────┘     └────────────┘      └────────────┘         └────────────┘          └────────────┘
                                                        │
                                  ┌─────────────────────┴─────────────────────┐
                                  │  RenQuant (umbrella):                      │
                                  │  ✓ unit tests                              │
                                  │  ✓ multirepo-triad-e2e.yml (nightly cron)  │
                                  │                                            │
                                  │  ← clones all sibs, runs full e2e          │
                                  │    base-data → model → pipeline →          │
                                  │    backtesting → execution refused         │
                                  └────────────────────────────────────────────┘
```

---

## 8 · 迁移序列 + 具体 PR 列表（v4 精确化）

### 8.1 MVP wave (5 PRs ≤ 2 days, stops today's incident class)

| # | Repo | Branch | Files touched (concrete) | Tests added | Triggers |
|---|---|---|---|---|---|
| ① | renquant-common | `feat/triad-contracts-and-guards` | `src/renquant_common/contracts/triad.py` (new), `src/renquant_common/contracts/scorer.py` (extend with triad_report field), `src/renquant_common/leakage_guards/scorer_sanity.py` (new), `src/renquant_common/leakage_guards/trainer_placebo.py` (new — async runner CLI), `src/renquant_common/leakage_guards/gate.py` (new — shared helper), pyproject version 0.7.x→0.8.0 (additive); `tests/test_contracts_triad.py`, `tests/test_leakage_guards_*.py` | + 4 test files | contract-bump-check.yml fan-out on all 8 consumers (must stay green) |
| ② | renquant-pipeline | `feat/scorer-load-triad-gate` | `src/renquant_pipeline/kernel/panel_pipeline/panel_scorer.py` — wrap `PanelScorer.load()` with `assert_artifact_validated`; pin renquant-common>=0.8,<0.9; `tests/test_scorer_load_gate.py` (new) | + 1 test file | artifact-gate.yml |
| ③ | renquant-model-patchtst | `feat/wire-triad-tier1-tier2` | `src/renquant_model_patchtst/hf_trainer.py` — insert Tier-1 block in `_save_artifact`; new `src/renquant_model_patchtst/post_save_hook.py`; pin renquant-common>=0.8,<0.9; `tests/patchtst/test_triad_tier1_wiring.py`, `tests/patchtst/test_post_save_hook_enqueues_tier2.py` | + 2 test files | triad-tier1.yml + contract-bump-check.yml |
| ④ | renquant-model-gbdt | `feat/wire-triad-tier1-tier2` | mirror of ③ for GBDT trainer | + 2 test files | same |
| ⑤ | renquant-backtesting + renquant-orchestrator | `feat/manifest-row-triad-gate` (paired) | `src/renquant_backtesting/wf_gate/runner.py`, `src/renquant_backtesting/wf_gate/sim_driver.py`, `src/renquant_orchestrator/build_patchtst_wf_manifest.py`, `src/renquant_orchestrator/build_gbdt_wf_manifest.py` — wrap manifest_row with `assert_artifact_validated`; pin renquant-common>=0.8,<0.9; tests stamping pending/failed/passed artifacts and asserting manifest_row behavior | + 4 test files | artifact-gate.yml |

**MVP done** ⇒ G1 + G2 + G3 + G4 live. unvalidated artifact 进不去 prod cron / sim / live broker.

### 8.2 Full architecture wave (4 PRs ≤ 1 week, schema-split refactor)

| # | Repo | Files | Risk |
|---|---|---|---|
| ⑥ | renquant-common `feat/data-contracts-and-frames` | `contracts/data.py`, `contracts/frames.py`; bump 0.8 → 0.9 additive | LOW |
| ⑦ | renquant-base-data `feat/split-parquet-output` | alpha158 builder writes 3 parquets + manifest; bump renquant-base-data minor; pin common>=0.9 | MED — old single-parquet path deleted; need to regen all data |
| ⑧ | renquant-model-* `feat/typed-train-entry` | introduce `train(features: FeatureFrame, ...)`; DELETE `load_panel_with_split`; add ruff rule banning `pd.read_parquet` in trainer; pin common>=0.9, base-data>=new | HIGH — touches main trainer entrypoint |
| ⑨ | renquant-execution `feat/broker-side-triad-gate` | G5: refuse orders sourced from manifests with unvalidated scorers | MED |

### 8.3 PR dependencies (DAG)

```
①(common)
   ├──┬──→ ②(pipeline)        [pipe gate G3]
   │  ├──→ ③(model-patchtst)  [tier1+2 G1+G2]
   │  ├──→ ④(model-gbdt)      [tier1+2 G1+G2]
   │  └──→ ⑤(backtest+orch)   [manifest G4]
   │
   │ ────────[MVP done — incident class blocked]────────
   │
   ↓
   ⑥(common)──┬──→ ⑦(base-data)     [split parquet]
              ├──→ ⑧(model-typed-entry) [delete unsafe path]
              └──→ ⑨(execution G5)
```

---

## 9 · Disaster scenarios + 应对

| Scenario | Detection | Mitigation | Auto-recovery |
|---|---|---|---|
| Tier-2 runner subprocess crashes on `s44` only | nightly multirepo-triad-e2e.yml fails | gate `assert_artifact_validated` refuses pending → no live order | architect bumps bypass_until + reruns Tier-2 manually; pending → passed once placebo completes |
| Multiple sibling repos break on contract additive bump | contract-bump-check.yml fails fan-out matrix on common PR | refuse to merge common PR until consumer PRs paired | escalate to architect human; reject contract change or coordinate consumer migrations |
| Live broker keeps refusing all orders due to G5 | execution side telemetry "refused_order_unvalidated_scorer" spikes | check `strategy_config.golden.json`; verify bypass not expired | one-off bypass extension PR by architect; sustain until Tier-2 fixed |
| Old pre-fix artifacts on disk (no `triad_report` field) | `ScorerArtifact.parse_file` Pydantic ValidationError on load | wrap with `try/except ValidationError → log + skip`; OR run one-off backfill stamping `triad_status="pending"` + 1-month bypass | backfill script + grace period; eventually all old artifacts re-validated or deleted |
| Schema_hash drift (parquet modified out-of-band) | FeatureFrame.from_parquet raises on load | refuse to train | regen via base-data builder |

---

## 10 · v1 → v4 changelog

| Source | Version | Sections added |
|---|---|---|
| v1 (initial) | — | basic reflection + 3 walls + 7-PR plan |
| Codex review v1→v2 | HIGH×2 + MED×4 | §2.1 scorer vs trainer triad split; §3.2 runtime validators; §3.2 manifest not regex; §4.1 MVP first; §4.5 async + bypass; §3.2 PR#9 defense-in-depth |
| User v2→v3 ("multirepo architect") | M1-M7 multirepo holes | §3.1 holes + §3.2 5 gates + §4.1 ownership matrix + §4.2 cross-repo CI + §4.3 version protocol + §4.4 migration sequence + §4.5 escape hatch + §4.6 agent labels + §4.7 test surface |
| User v3→v4 ("具体代码 + workflow + dataflow") | concrete code + YAML + sequence diagrams | §4.1 actual Pydantic + class code; §4.2 file:line gate locations; §4.3 import-graph; §5.1-5.4 dataflow sequence diagrams; §6 actual workflow YAMLs (5 files); §7 CI topology; §8 PR-level file lists; §9 disaster scenarios |

---

## 11 · 索引

- `[[project_patchtst_btuned_leakage_2026-05-31]]`
- `[[feedback_research_pipeline_must_gate_with_sanity_triad]]` — was advisory; this PR is the executable version
- `[[feedback_leakage_three_walls]]` — v2 摘要（updated 不止 3 walls，扩展为 5 gates + multirepo coordination）
- `[[feedback_industry_leading_quality]]` — patch-sim-patch loop rejected
- `[[feedback_pr_based_workflow]]` — §4.6 agent labels build on this
- `[[feedback_multirepo_code_placement]]` — §4.1 ownership matrix hardens this
- `[[project_multirepo_sop_2026-05-28]]` — §4.3 version coordination protocol supersedes
- `[[project_phase5_burst_2026-06-01]]` — today's PR burst that exposed the ad-hoc nature of cross-repo pin sweeps
- CLAUDE.md §3.1, §3.5, §7.5, §7.7, §3.7 (agent labels)
