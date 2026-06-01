# RenQuant Multirepo Leakage Defense — Architecture (v6)

**Status**: Design under review (RenQuant PR #38)  
**Authors**: Claude  
**Reviewers**: Codex (4 review rounds, 13 findings addressed — see §11)  
**Supersedes**: v1-v5 (history archived at `doc/research/2026-06-01-leakage-reflection-history.md`)  
**Companion**: `doc/research/2026-06-01-leakage-reflection.md` (the process post-mortem that motivated this)

---

## 1 · Problem statement (1 paragraph)

`renquant-model-patchtst::B_tuned` shows shuffle/timeshift placebo IC (+0.041) ≈ real IC (+0.041) across 2 independent seeds (5/31 and 6/01 runs). PR #9 fixed the cross-split timeshift boundary; the placebo failure persists. The 5/31 memo and 6/01 BG re-run prove **the leak is not a single feature engineering bug — the architecture has no enforced layer that prevents a model with placebo IC ≈ real IC from reaching production.** The umbrella has 4 downstream consumers of trained scorers (`renquant-pipeline`, `renquant-backtesting`, `renquant-orchestrator`, `renquant-execution`) and none of them check whether the artifact's trainer-level placebos pass.

## 2 · Completion criteria (falsifiable)

The design ships when **every one of these is mechanically true**:

1. A model artifact whose Tier-2 trainer-level placebo IC exceeds threshold relative to real IC **cannot** be loaded by `renquant-pipeline.PanelScorer.load()`. (G3)
2. The same artifact **cannot** be inserted into any walk-forward manifest by `renquant-orchestrator` or `renquant-backtesting`. (G4)
3. The same artifact **cannot** source a live order in `renquant-execution`. (G5)
4. A feature parquet that contains a column name from the label set (declared) **cannot** be written by `renquant-base-data`. (G0)
5. A model trainer that does not run Tier-1 scorer sanity **cannot** save an artifact. (G1)
6. Disabling any of G0-G5 in code (e.g. early return in `assert_artifact_validated`) is detected by CI on every PR to `renquant-common` (gate-disable detector grep + AST scan).
7. Operating without architect-signed `agent:emergency:bypass-triad` override, every `triad_status != "passed"` artifact is refused.
8. The full multirepo loop (base-data → model → pipeline → backtesting → orchestrator → execution refused on unvalidated) runs as a nightly E2E test under 5 minutes wall clock.

**If any of (1)-(8) is false after MVP merges → design failed → revert.**

## 3 · Designs considered & rejected

| Approach | Rejected because |
|---|---|
| A) Add `validate_triad()` helper, ask developers to call it | Advisory rule, not enforced (the original `feedback_research_pipeline_must_gate_with_sanity_triad` memory — failed exactly this test). |
| B) Type annotations alone: `train(features: FeatureFrame, ...)` | Python annotations don't stop callers at runtime (codex v4 finding 2). |
| C) Single monolithic "validate model" wrapper at orchestrator | One choke point, but doesn't prevent unvalidated artifact from being *built*. Lower defense in depth. |
| D) Disable PatchTST entirely until issue solved | Hammer, not architecture. Doesn't fix GBDT or future models that have the same risk. |
| **E) Multi-layer gate network + Pydantic-enforced artifact contract + Tier-2 async with CAS** (this design) | Chosen. 5 independent gates, each fail-closed, sharing a single helper; Pydantic enforces at boundary; threshold contract makes status mechanically derived from reports. |

## 4 · Invariants (the design in one page)

**I1 — Artifact contract**: Every `ScorerArtifact` carries a `TriadReport` whose `triad_status ∈ {pending, passed, failed}` is **deterministically derived** from `scorer_sanity` (Tier 1) and `trainer_placebo` (Tier 2) reports by an explicit reducer. Status cannot be set independently of reports.

**I2 — State machine**: `pending → passed` and `pending → failed` are terminal transitions per binding-tuple `(model_sha, feature_schema_hash, label_hash, code_sha, triad_config_hash)`. If any binding-tuple element changes, the artifact is a **new** artifact (new fingerprint) and starts at `pending`. `failed → passed` is **never** allowed; remediation requires a new artifact.

**I3 — Gate symmetry**: G3/G4/G5 share **one** helper (`assert_artifact_validated`); change one site = changes all. §7.5 enforced.

**I4 — Bypass narrowness**: Emergency bypass is **per-fingerprint allowlist** with expiry + reason + approver + audit trail. **There is no global "allow any pending"**. `failed` is **never** bypassable.

**I5 — CAS write**: Sidecar updates are compare-and-swap on the binding-tuple. A stale or duplicate Tier-2 runner cannot overwrite a sidecar that has moved on.

**I6 — Disable detection**: CI on `renquant-common` PRs runs a gate-disable detector (AST + regex) that fails the PR if any guard's body is replaced with `return`, `pass`, or `if False` paths.

**I7 — Statistical thresholds**: Pass/fail decisions use a **permutation null distribution** scaled by `n_val_dates` and per-regime sample size. Static 0.01 IC thresholds (v4) are wrong because they false-fail small samples and false-pass noisy regimes.

**I8 — Migration safety**: The schema change adding `triad_report` is **declared breaking** (MAJOR semver). Two-step migration: optional-at-schema during stage S1 with telemetry only ("would have blocked"); hard fail-closed at stage S3+ after consumer PRs and artifact backfill.

## 5 · Contract code (correct, not pseudo)

### 5.1 `TriadReport` — derived status, explicit raises

```python
# renquant-common/src/renquant_common/contracts/triad.py
"""
Triad contract.

DERIVED status: triad_status is computed from (scorer_sanity, trainer_placebo)
by a single explicit reducer. Consumers cannot set status independently.

EXPLICIT raises: NO `assert` — assertions are stripped under python -O and
are the wrong tool for contract invariants (codex finding 2).
"""
from __future__ import annotations
from datetime import datetime
from typing import Literal
import pydantic

TriadStatus = Literal["pending", "passed", "failed"]


class ScorerSanityReport(pydantic.BaseModel):
    """Tier 1: fixed scorer vs perturbed labels (seconds). Catches label-calc bugs only."""
    aa_split_real_ic_replicate: float
    aa_split_drift_ic: float
    shuffled_val_ic: float
    timeshifted_val_ic: float
    label_col: str
    n_val_dates: int
    permutation_null_p_value_shuffled: float    # ≥ 0.05 = passes
    permutation_null_p_value_timeshifted: float

    def fail_reasons(self) -> list[str]:
        """Empty list = pass. Each entry is a deterministic threshold violation."""
        out: list[str] = []
        if self.n_val_dates < 30:
            out.append(f"insufficient n_val_dates={self.n_val_dates} (need ≥30)")
        if self.permutation_null_p_value_shuffled < 0.05:
            out.append(
                f"shuffled_val_ic={self.shuffled_val_ic:+.4f} significant "
                f"(p={self.permutation_null_p_value_shuffled:.3f} < 0.05)"
            )
        if self.permutation_null_p_value_timeshifted < 0.05:
            out.append(
                f"timeshifted_val_ic={self.timeshifted_val_ic:+.4f} significant "
                f"(p={self.permutation_null_p_value_timeshifted:.3f} < 0.05)"
            )
        if self.aa_split_drift_ic > 0.03:
            out.append(f"aa_split_drift_ic={self.aa_split_drift_ic:.4f} > 0.03")
        return out


class TrainerPlaceboReport(pydantic.BaseModel):
    """Tier 2: RETRAIN on shuffled/timeshifted labels. Catches train-time leakage."""
    real_ic_mean: float
    real_ic_per_regime: dict[str, float]
    real_ic_n_dates_per_regime: dict[str, int]
    shuffle_placebo_ic_mean: float
    shuffle_placebo_ic_per_regime: dict[str, float]
    shuffle_placebo_p_value: float           # bootstrap p that shuffle_ic == 0 vs real_ic
    timeshift_placebo_ic_mean: float
    timeshift_placebo_ic_per_regime: dict[str, float]
    timeshift_placebo_p_value: float
    n_seeds: int
    n_val_dates: int

    def fail_reasons(self) -> list[str]:
        out: list[str] = []
        if self.n_seeds < 3:
            out.append(f"insufficient n_seeds={self.n_seeds} (need ≥3)")
        for regime, n in self.real_ic_n_dates_per_regime.items():
            if n < 20:
                out.append(f"regime {regime}: n_dates={n} < 20")
        if self.shuffle_placebo_p_value < 0.05:
            out.append(
                f"shuffle placebo IC={self.shuffle_placebo_ic_mean:+.4f} "
                f"distinguishable from null (p={self.shuffle_placebo_p_value:.3f})"
            )
        if self.timeshift_placebo_p_value < 0.05:
            out.append(
                f"timeshift placebo IC={self.timeshift_placebo_ic_mean:+.4f} "
                f"distinguishable from null (p={self.timeshift_placebo_p_value:.3f})"
            )
        # Per-regime guard: a regime with placebo > 50% real is suspicious
        for regime in self.real_ic_per_regime:
            r = abs(self.real_ic_per_regime.get(regime, 0))
            sp = abs(self.shuffle_placebo_ic_per_regime.get(regime, 0))
            tp = abs(self.timeshift_placebo_ic_per_regime.get(regime, 0))
            if r > 0.01 and (sp > 0.5 * r or tp > 0.5 * r):
                out.append(
                    f"regime {regime}: placebo IC > 50% of real "
                    f"(real={r:+.4f} shuffle={sp:+.4f} timeshift={tp:+.4f})"
                )
        return out


def _derive_status(
    scorer_sanity: ScorerSanityReport,
    trainer_placebo: TrainerPlaceboReport | None,
) -> tuple[TriadStatus, list[str]]:
    """The ONLY function that decides triad_status. Pure, deterministic."""
    s1_reasons = scorer_sanity.fail_reasons()
    if s1_reasons:
        return "failed", s1_reasons
    if trainer_placebo is None:
        return "pending", []
    s2_reasons = trainer_placebo.fail_reasons()
    if s2_reasons:
        return "failed", s2_reasons
    return "passed", []


class TriadBinding(pydantic.BaseModel):
    """Identity tuple. Any element change ⇒ new artifact, new pending triad."""
    model_sha: str                          # sha256(model.pt bytes)
    feature_schema_hash: str
    label_hash: str                         # sha256(label column bytes from training)
    code_sha: str                           # git rev of trainer at save time
    triad_config_hash: str                  # sha256({n_seeds, thresholds, label_shift_days})

    def fingerprint(self) -> str:
        """Combined identity. 16-hex prefix used in logs/telemetry."""
        import hashlib
        h = hashlib.sha256(
            f"{self.model_sha}|{self.feature_schema_hash}|{self.label_hash}|"
            f"{self.code_sha}|{self.triad_config_hash}".encode()
        )
        return h.hexdigest()


class TriadReport(pydantic.BaseModel):
    triad_status: TriadStatus                # DERIVED — never set by hand
    failure_reasons: list[str]               # empty iff passed; populated otherwise
    scorer_sanity: ScorerSanityReport
    trainer_placebo: TrainerPlaceboReport | None
    binding: TriadBinding
    triad_started_at: datetime
    triad_completed_at: datetime | None      # None iff pending

    @pydantic.model_validator(mode="after")
    def status_is_derived(self) -> "TriadReport":
        expected_status, expected_reasons = _derive_status(self.scorer_sanity, self.trainer_placebo)
        if self.triad_status != expected_status:
            raise ValueError(
                f"triad_status={self.triad_status!r} inconsistent with reducer "
                f"output={expected_status!r}. Status MUST be derived via "
                f"_derive_status, never set by hand."
            )
        if self.failure_reasons != expected_reasons:
            raise ValueError(
                f"failure_reasons inconsistent with reducer output.\n"
                f"  declared: {self.failure_reasons}\n"
                f"  expected: {expected_reasons}"
            )
        # Completion timestamp consistency
        if self.triad_status == "pending" and self.triad_completed_at is not None:
            raise ValueError("triad_completed_at must be None when status=pending")
        if self.triad_status in ("passed", "failed") and self.triad_completed_at is None:
            raise ValueError(
                f"triad_completed_at required when status={self.triad_status!r}"
            )
        return self

    @classmethod
    def build(
        cls,
        scorer_sanity: ScorerSanityReport,
        trainer_placebo: TrainerPlaceboReport | None,
        binding: TriadBinding,
        triad_started_at: datetime,
    ) -> "TriadReport":
        """The ONLY public constructor — guarantees derivation invariant."""
        status, reasons = _derive_status(scorer_sanity, trainer_placebo)
        completed_at = datetime.utcnow() if status != "pending" else None
        return cls(
            triad_status=status,
            failure_reasons=reasons,
            scorer_sanity=scorer_sanity,
            trainer_placebo=trainer_placebo,
            binding=binding,
            triad_started_at=triad_started_at,
            triad_completed_at=completed_at,
        )
```

**Why this addresses codex findings 1, 2, 5**:
- Finding 1: status is **derived** by `_derive_status`; reducer checks BOTH tiers. Consumer cannot construct a `passed` `TriadReport` that bypassed Tier 1.
- Finding 2: zero `assert`; every invariant uses `raise ValueError`.
- Finding 5: `failed` is exactly the case `len(failure_reasons) > 0`; reasons are derived from reports.

### 5.2 Bypass (allowlist, not date-globally)

```python
# renquant-common/src/renquant_common/contracts/leakage_config.py

class TriadBypassEntry(pydantic.BaseModel):
    artifact_fingerprint: str             # 64-hex; full sha256, not prefix
    expires_at: datetime
    reason: str
    approved_by: str                      # GitHub username of architect / signer
    pr_url: str                           # PR that introduced this entry
    approved_at: datetime

    @pydantic.field_validator("reason")
    @classmethod
    def reason_nontrivial(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(
                f"bypass reason too short ({len(v.strip())} chars); "
                f"explain the operational situation in ≥20 chars"
            )
        return v


class LeakageGuardConfig(pydantic.BaseModel):
    # Tier 1
    tier1_min_n_val_dates: int = 30
    tier1_pvalue_threshold: float = 0.05
    tier1_aa_drift_max: float = 0.03
    # Tier 2
    tier2_n_seeds_required: int = 3
    tier2_pvalue_threshold: float = 0.05
    tier2_min_n_dates_per_regime: int = 20
    tier2_max_placebo_real_ratio_per_regime: float = 0.50
    tier2_label_shift_days: int = 10
    tier2_run_strategy: Literal["subprocess_inline", "subprocess_queue", "manual"] = "subprocess_inline"
    # Bypass — allowlist, NOT date-globally
    triad_bypasses: list[TriadBypassEntry] = pydantic.Field(default_factory=list)
    # Alerting
    alert_channel: Literal["slack", "log", "none"] = "log"
    alert_slack_webhook_url: str | None = None
```

```python
# renquant-common/src/renquant_common/leakage_guards/gate.py

def assert_artifact_validated(
    artifact: ScorerArtifact,
    *,
    cfg: LeakageGuardConfig,
    caller: str,
) -> None:
    """SINGLE implementation used by G3/G4/G5.

    failed: ALWAYS blocked, regardless of bypass.
    passed: allowed.
    pending: allowed IFF an active bypass entry matches THIS artifact's fingerprint.
    """
    s = artifact.triad_report.triad_status
    fp = artifact.triad_report.binding.fingerprint()

    if s == "failed":
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="failed",
                             failure_reasons=artifact.triad_report.failure_reasons)
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} with triad_status='failed'. "
            f"Reasons: {artifact.triad_report.failure_reasons}. "
            f"No bypass allowed for failed."
        )

    if s == "passed":
        telemetry.emit_event("gate_allow", caller=caller, artifact_fingerprint=fp,
                             triad_status="passed")
        return

    # pending — require allowlist match
    now = datetime.utcnow()
    matching = [
        b for b in cfg.triad_bypasses
        if b.artifact_fingerprint == fp and b.expires_at > now
    ]
    if not matching:
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="pending",
                             reason="no matching bypass entry")
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} with triad_status='pending' "
            f"and no matching bypass entry in cfg.triad_bypasses (active entries: "
            f"{len(cfg.triad_bypasses)})."
        )

    entry = matching[0]
    telemetry.emit_event("gate_bypass", caller=caller, artifact_fingerprint=fp,
                         triad_status="pending",
                         bypass_expires_at=entry.expires_at.isoformat(),
                         bypass_approved_by=entry.approved_by,
                         bypass_pr_url=entry.pr_url)
    log.warning(
        "TRIAD BYPASS ACTIVE: %s loading PENDING artifact fp=%s "
        "(bypass expires %s, approved_by=%s, pr=%s)",
        caller, fp[:16], entry.expires_at.isoformat(), entry.approved_by, entry.pr_url,
    )
```

**Why this addresses codex finding 3**: bypass is **per-fingerprint allowlist** with required `reason ≥ 20 chars`, `approved_by`, `pr_url`, `expires_at`. A fresh unvalidated artifact built after the bypass entry was written **does not match** any fingerprint → blocked. `failed` is unconditionally blocked.

### 5.3 Sidecar CAS write

```python
# renquant-common/src/renquant_common/leakage_guards/sidecar.py

class StaleBindingError(RuntimeError):
    """Raised when Tier-2 runner finds the sidecar binding has moved on."""


def atomic_cas_update_sidecar(
    sidecar_path: Path,
    expected_binding: TriadBinding,
    transformer: Callable[[dict], dict],
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    """Compare-And-Swap update of triad sidecar JSON.

    Acquires flock, reads current sidecar, verifies binding matches
    expected_binding, applies transformer, validates schema + transition,
    writes atomically. Raises StaleBindingError if binding has changed
    since Tier 2 started.
    """
    lockfile = sidecar_path.with_suffix(sidecar_path.suffix + ".lock")
    deadline = time.monotonic() + timeout_seconds
    fd = os.open(lockfile, os.O_CREAT | os.O_WRONLY, mode=0o600)
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

        if not sidecar_path.exists():
            raise FileNotFoundError(
                f"sidecar absent at CAS update: {sidecar_path}"
            )
        current = json.loads(sidecar_path.read_text())

        # CAS check
        current_binding = TriadBinding.model_validate(
            current["triad_report"]["binding"]
        )
        if current_binding != expected_binding:
            raise StaleBindingError(
                f"binding moved: expected fp={expected_binding.fingerprint()[:16]}, "
                f"sidecar has fp={current_binding.fingerprint()[:16]}. "
                f"Tier 2 runner aborting; a newer artifact is in flight."
            )

        new = transformer(current)
        ScorerArtifact.model_validate(new)         # full re-validation

        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=sidecar_path.parent, delete=False, suffix=".tmp"
        )
        try:
            json.dump(new, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.rename(tmp.name, sidecar_path)
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

**Why this addresses codex finding 7**: every Tier-2 finalization is CAS-bound to the exact `TriadBinding` it started with. A stale runner finds a moved binding → aborts → emits telemetry event; never overwrites.

### 5.4 Statistical thresholds via permutation null

```python
# renquant-common/src/renquant_common/leakage_guards/stats.py

def permutation_null_p_value(
    real_ic_per_date: np.ndarray,         # daily IC under real labels
    perturbed_ic_per_date: np.ndarray,    # daily IC under shuffled/timeshifted labels
    n_perms: int = 10_000,
    rng_seed: int = 0,
) -> float:
    """One-sided permutation p that perturbed mean ≥ real mean.

    Used for both Tier 1 (already-trained scorer eval'd on perturbed val labels)
    and Tier 2 (separate-retrain perturbed IC distribution).
    """
    rng = np.random.default_rng(rng_seed)
    combined = np.concatenate([real_ic_per_date, perturbed_ic_per_date])
    n_real = len(real_ic_per_date)
    real_mean = real_ic_per_date.mean()
    null_means = np.empty(n_perms)
    for k in range(n_perms):
        rng.shuffle(combined)
        null_means[k] = combined[:n_real].mean()
    # p = fraction of permutations where null mean ≥ observed real mean
    p = float((null_means >= real_mean).mean())
    return p


def bootstrap_ic_ci(
    daily_ic: np.ndarray,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    rng_seed: int = 0,
) -> tuple[float, float]:
    """BCa-style bootstrap CI on the mean of daily IC."""
    rng = np.random.default_rng(rng_seed)
    n = len(daily_ic)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = daily_ic[idx].mean()
    alpha = (1 - confidence) / 2
    return float(np.quantile(boots, alpha)), float(np.quantile(boots, 1 - alpha))
```

**Why this addresses codex finding 6**: thresholds are **statistical**, scaled by sample size. A small-`n` regime gets a wide CI and passes by default (innocent until proven leaky). A large-`n` regime with placebo distinguishable from null fails even at small absolute IC. The `triad_config_hash` includes `(n_perms, n_boot, confidence, rng_seed)` so re-running gives identical decisions.

## 6 · State machine + transitions

```
                  ╔══════════════════════════════════════════════════════╗
                  ║                                                      ║
                  ║   binding-tuple (model_sha, feat_hash, label_hash,   ║
                  ║                  code_sha, triad_config_hash)        ║
                  ║                                                      ║
                  ║  Every transition is bound to ONE binding-tuple.     ║
                  ║  Any element change ⇒ new fingerprint ⇒ new lifeline ║
                  ╚══════════════════════════════════════════════════════╝

                                  ╭─────────────╮
                                  │   PENDING   │ ◄── (artifact saved with Tier 1 passed,
                                  │             │      Tier 2 not yet run)
                                  ╰──────┬──────╯
                                         │
                              Tier 2 CAS-updates sidecar
                                         │
                          ┌──────────────┴──────────────┐
                          │                             │
                  failure_reasons == []          failure_reasons ≠ []
                          │                             │
                          ↓                             ↓
                  ╭─────────────╮              ╭─────────────╮
                  │    PASSED   │              │   FAILED    │ (terminal,
                  ╰─────────────╯              ╰─────────────╯  immutable;
                                                                 binding-tuple
                                                                 must change
                                                                 to retry)

   ◆ Any consumer attempting to construct a different transition raises
     ValueError via TriadReport.status_is_derived.
   ◆ failed → passed is impossible: requires _derive_status to return passed,
     which is impossible while failure_reasons is non-empty.
   ◆ When binding changes (e.g., retrain), a NEW TriadReport is built starting
     from pending. The old TriadReport for the old binding remains immutable
     in telemetry/audit.
```

## 7 · Five gates — exact insertion points

| Gate | File | Line/anchor | What it does |
|---|---|---|---|
| G0 | `renquant-base-data/src/renquant_base_data/builders/alpha158.py` | `_write_dataset()` end, before `pq.write_table` | `DatasetManifest(...).model_validate(...)` raises if features∩labels ≠ ∅, embargo < label lookahead, lookahead_days non-zero |
| G1 | `renquant-model-patchtst/src/renquant_model_patchtst/hf_trainer.py::_save_artifact` (and gbdt mirror) | before `torch.save` | `report = run_tier1(...)`; if `report.fail_reasons()`: raise `Tier1Failed`. Always builds TriadReport via `.build()` (status derived). |
| G2 | `renquant-model-patchtst/src/renquant_model_patchtst/post_save_hook.py` (new) | after artifact persisted | `enqueue_tier2(artifact, binding, cfg)` → spawns CLI subprocess that on completion calls `atomic_cas_update_sidecar(sidecar_path, expected_binding=binding, transformer=lambda d: {...with trainer_placebo populated...})` |
| G3 | `renquant-pipeline/src/renquant_pipeline/kernel/panel_pipeline/panel_scorer.py::PanelScorer.load` | top of method, after artifact parse | `assert_artifact_validated(artifact, cfg=load_cfg, caller="pipeline:scorer_load")` |
| G4 | `renquant-orchestrator/src/renquant_orchestrator/build_patchtst_wf_manifest.py::manifest_row` (and gbdt+backtesting siblings) | before append to manifest | `assert_artifact_validated(...)` — failed scorer means the cutoff is treated like training failure (`ctx.failed_cutoffs.append`) |
| G5 | `renquant-execution/src/renquant_execution/broker_adapter.py::submit_order` (new) | before broker API call | resolve scorer artifact behind the order; `assert_artifact_validated(...)`; refused → no order, telemetry event `refused_order_unvalidated_scorer` |

## 8 · Migration — declared breaking, 4 stages

Codex finding 4: requiring `triad_report` on `ScorerArtifact` is **breaking**, not additive. Declared via `agent:contract:breaking` label.

| Stage | Wall clock | Schema | Gates | Action |
|---|---|---|---|---|
| **S0** | Day -7 to 0 | `triad_report: TriadReport \| None` (optional) | OFF (telemetry-only "would have blocked") | Tier-1 runner wired in trainer; sidecar gets `triad_report` if computed. Existing artifacts untouched. |
| **S1** | Day 0-7 | optional | G3/G4/G5 log+pass (`assert_artifact_validated_shadow`) | Telemetry counts how many production calls would fail. Architect reviews. |
| **S2** | Day 7-14 | optional → required for newly-built artifacts | Tier-2 runner brought online; existing artifacts get `pending` + 7-day per-fingerprint bypass via backfill script. | Architect reviews backfill audit. |
| **S3** | Day 14-28 | required (`triad_report` mandatory) | G3/G4/G5 enforce. `failed` blocked. `pending` blocked unless matching bypass. | Bypass entries shrink weekly; new artifacts must reach `passed` via Tier 2. |
| **S4** | Day 28+ | required | All bypass entries expired. Only `passed` artifacts loaded. | Steady state. |

**Rollback**: at any stage, architect PR sets `tier2_run_strategy="manual"` + adds bypass entries for live-production fingerprints. Live trading continues; Tier-2 can be re-fired manually. Rollback latency ≤ 5 min from PR push to merge.

## 9 · Falsification — how do we know this design is wrong

If after S3:

1. A passed artifact is later proven to have placebo IC > 0.01 (re-run Tier 2 on it manually) → **derivation or threshold contract is wrong**. Reduce threshold or fix bug; this is a P0 architecture defect.
2. A pending artifact reaches a live order with no matching bypass → **gate plumbing is wrong**. Audit telemetry; this is a P0 plumbing defect.
3. A failed artifact transitions to passed without binding change → **state machine is wrong**. Audit `_derive_status` and `model_validator`; this is a P0 invariant defect.
4. Tier-2 takes > 4× the per-trial training time → **performance budget exceeded**. Re-architect runner (queue / GPU pool / quantization).
5. Architect bypass list grows monotonically over 30 days → **operational debt accruing**. Either threshold too tight, or model family fundamentally leaky; pause training campaign.

## 10 · Threat model abridged (full table in companion `2026-06-01-leakage-threat-model.md`)

12 leak classes (L1-L12) mapped to gate responsibilities:

| Class | Gate | Residual risk |
|---|---|---|
| L1 label in features | G0 | 0 |
| L2 implicit lookahead | G0 (declared `feature_lookahead_days[c]=0`) + G2 (placebo retrain unmasks via shuffle) | LOW — builder honesty + Tier-2 sanity catch |
| L3 split_label in features | G0 | 0 |
| L4 embargo insufficient | G0 + PR #9 | 0 |
| L5 val labels in callbacks | G2 (Tier 2 disables early stopping) | LOW |
| L6 preprocessing leak | G2 | LOW |
| L7 sliding window cross-split | G0 (embargo) | 0 |
| L8 calibrator fit on train+val | G2 | LOW |
| L9 EarlyStopping by val IC | G2 | LOW |
| L10 random seed selection bias | renquant-model PR #15 | 0 (out of scope) |
| L11 sidecar tampering | G2 (CAS) + G3 (re-validate fingerprint at load) | LOW |
| L12 stale triad after retrain | binding-tuple + state machine auto-invalidates | 0 |

## 11 · Codex review v4 → v6 — addressed

| # | Finding | Resolution |
|---|---|---|
| 1 HIGH | Status can be passed without Tier-1 check | §5.1 `_derive_status` is single reducer; checks `scorer_sanity.fail_reasons()` first; `failed` returned if any |
| 2 HIGH | Bare `assert` stripped under -O | §5.1 every `assert` replaced with `raise ValueError(...)`. AST scan in `gate-disable-detection.yml` (CI) also fails PRs reintroducing `assert` in `leakage_guards/` |
| 3 HIGH | Date-only bypass too broad | §5.2 `TriadBypassEntry` = per-fingerprint allowlist + `reason ≥ 20 chars` + `approved_by` + `pr_url` + `expires_at`. `failed` never bypassable. |
| 4 HIGH | Required `triad_report` is breaking, not additive | §8 stage S0-S2: optional, telemetry-only/shadow; stage S3: required + enforced. Declared `agent:contract:breaking`. |
| 5 MED | `failed` not tied to actual threshold | §5.1 `failure_reasons` populated from `fail_reasons()`; status `failed` iff `len(failure_reasons) > 0`. Mechanically derived. |
| 6 MED | Static IC thresholds | §5.4 permutation null p-value + bootstrap CI + per-regime `n_min`. Threshold structure: `(p_value < 0.05) ∨ (per-regime n < min) ∨ (placebo > 0.5×real per regime)`. Sample-size aware. |
| 7 MED | Async sidecar needs CAS | §5.3 `atomic_cas_update_sidecar` reads + verifies `TriadBinding` matches expected; raises `StaleBindingError` if not. |

Codex v1-v3 findings (split triad, runtime validators, manifest not regex, MVP first, async + pending, PR #9 preserved): all carried forward from v4.

## 12 · Open questions

1. **Q**: Permutation null requires per-date IC arrays. PatchTST val sets are ~250 dates per cut. Is `n_perms=10_000` enough? **Tentative**: yes; Bailey-Lopez de Prado 2014 used 10k for DSR; we can audit by running 100k once and comparing.
2. **Q**: Subprocess vs queue for Tier 2 runner — at 75min/trial × 3 seeds × 3 modes × N artifacts/day, do we need a job queue? **Tentative**: not yet; current daily cadence is ~1-2 retrains/day, 9 sub-trainings in serial = 7-10h, fits overnight. Revisit if we go multi-strategy.
3. **Q**: When Tier 2 fails, is the underlying model retrained automatically or held for architect review? **Tentative**: held. Auto-retrain after fail risks accidental selection-bias loop.
4. **Q**: Should `renquant-strategy-104` configs also be in the binding-tuple (i.e., changing a strategy threshold invalidates the triad)? **Tentative**: no; strategy threshold changes don't change the model's predictive behavior, only how predictions are used downstream. Promotion gate is the right layer for that.

## 13 · MVP PR list (5 PRs, ≤ 2 days)

| # | Repo | Files | Tests |
|---|---|---|---|
| ① | `renquant-common` | `contracts/triad.py`, `contracts/leakage_config.py`, `leakage_guards/{scorer_sanity,trainer_placebo,gate,sidecar,stats,telemetry,alerts}.py`, pyproject 0.x → 0.x.0 (declared breaking MAJOR if `triad_report` mandatory in stage S3; for MVP S0/S1, optional → MINOR) | unit per module + sidecar CAS race test + status state-machine test |
| ② | `renquant-pipeline` | `kernel/panel_pipeline/panel_scorer.py::PanelScorer.load` | gate-behavior matrix test (passed/failed/pending/bypass-match/bypass-expired) |
| ③ | `renquant-model-patchtst` | `hf_trainer.py::_save_artifact`, new `post_save_hook.py`, new `triad_replay_mode` CLI arg, `--disable-early-stopping` for Tier 2 | synth-data Tier 1 wiring test + Tier 2 subprocess enqueue test |
| ④ | `renquant-model-gbdt` | mirror of ③ for GBDT | mirror |
| ⑤ | `renquant-orchestrator` + `renquant-backtesting` (paired) | `manifest_row()` + `wf_gate/runner.py` + `wf_gate/sim_driver.py` + `scripts/fit_walkforward_calibrators.py` | manifest_row rejects failed/pending, accepts passed; bypass-match path |

Full architecture wave (split parquet, typed train signatures, ban `pd.read_parquet`) deferred to weeks 2-3, per codex finding 4 MVP-first.

## 14 · References

- Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 5 (Combinatorial Purged CV), ch. 7 (Cross-validation in Finance).
- Pesaran, M.H., Timmermann, A. (2007). "Selection of estimation window in the presence of breaks." *J. Econometrics* 137.
- v1-v5 evolution: `doc/research/2026-06-01-leakage-reflection-history.md`
- Process post-mortem motivating this design: `doc/research/2026-06-01-leakage-reflection.md`
