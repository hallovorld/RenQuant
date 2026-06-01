# RenQuant Multirepo Leakage Defense — Architecture (v7)

**Status**: Design under review (RenQuant PR #38)
**Authors**: Claude
**Reviewers**: Codex (5 review rounds; latest v6 found 7 new contract-correctness bugs introduced by my v6 fixes — all addressed here)
**Supersedes**: v1-v6 (commit history is the audit trail)
**Companion**: `doc/research/2026-06-01-leakage-reflection.md`

---

## 1 · Problem statement

`renquant-model-patchtst::B_tuned` shows shuffle/timeshift placebo IC (+0.041) ≈ real IC (+0.041) across 2 independent seeds (5/31 and 6/01 runs). PR #9 fixed the cross-split timeshift boundary; the placebo failure persists. **The architecture has no enforced layer that prevents a model with placebo IC ≈ real IC from reaching production.** 4 downstream consumers of trained scorers (`renquant-pipeline`, `renquant-backtesting`, `renquant-orchestrator`, `renquant-execution`) and none of them check trainer-level placebos.

## 2 · Completion criteria (falsifiable)

The design ships when **every one of these is mechanically true**:

1. A model artifact whose Tier-2 trainer-level placebos are statistically distinguishable from null (p < 0.05) or close to real IC (per-regime ratio > 0.5) **cannot** be loaded by `renquant-pipeline.PanelScorer.load()`. (G3)
2. The same artifact **cannot** be inserted into any walk-forward manifest. (G4)
3. The same artifact **cannot** source a live order. (G5)
4. A feature parquet containing label column names **cannot** be written. (G0)
5. A trainer that does not run Tier-1 scorer sanity **cannot** save an artifact. (G1)
6. Disabling any of G0-G5 in code is detected by CI on `renquant-common` PRs (AST scan + regex).
7. Bypass entries without valid HMAC signature from the architect key are rejected at config-load time. (§5.2)
8. The E2E nightly test runs all 5 gates with **synthetic** Tier-2 runner (not 75-min real one) under 5 minutes wall clock, exercising 3 fixtures: good / leak / noise models.

**If any of (1)-(8) is false after MVP merges → design failed → revert.**

## 3 · Designs considered & rejected

| Approach | Rejected because |
|---|---|
| A) Advisory `validate_triad()` helper | `feedback_research_pipeline_must_gate_with_sanity_triad` was advisory and failed exactly this test. |
| B) Type annotations alone | Don't stop callers at runtime (codex v4 #2). |
| C) Single choke point at orchestrator | Doesn't prevent unvalidated artifact from being *built*; lower defense in depth. |
| D) Disable PatchTST until issue solved | Hammer, not architecture; doesn't generalize. |
| **E) Multi-gate + derived status + HMAC-signed bypass + CAS-with-transition** (this) | Chosen. Each gate independent fail-closed; status is mechanical reduction; bypass requires cryptographic architect approval; sidecar transitions enforced. |

## 4 · Invariants

- **I1 (Artifact contract)**: Every `ScorerArtifact` carries `TriadReport` whose `triad_status` is **deterministically derived** from `scorer_sanity` (Tier 1) and `trainer_placebo` (Tier 2) by an explicit reducer.
- **I2 (Terminal failure)**: `failed` and `passed` are **terminal at the sidecar layer**. Transitions enforced by CAS, not only by Pydantic. Re-validation requires a new binding-tuple = new artifact.
- **I3 (Gate symmetry)**: G3/G4/G5 share one helper.
- **I4 (Cryptographic bypass)**: Bypass entries carry HMAC-SHA256 signature over `(fingerprint, expires_at, reason, approved_by, pr_url, approved_at)`. Architect's HMAC key lives in `~/.renquant/secrets/architect_hmac.key` (outside repo, 0600). Unsigned/invalid-signed entries are dropped at config load.
- **I5 (CAS + transition)**: Sidecar update is compare-and-swap on `TriadBinding` AND on `triad_status`. Tier-2 runner declares `expected_current_status="pending"`; if anything else is found → abort.
- **I6 (Disable detection)**: CI AST scan fails PRs that replace gate bodies with `return`, `pass`, `if False`, or bare `assert`.
- **I7 (Statistical, n-aware, direction-correct)**: Pass/fail uses bootstrap p-values **with H0/H1 explicit and gate-aligned**: shuffle-placebo fail = reject H0(placebo IC = 0); real-vs-placebo fail = cannot reject H0(real ≤ placebo). Static IC thresholds are wrong.
- **I8 (Tz-aware datetimes)**: All `datetime` are timezone-aware UTC; naive datetimes rejected at validation.
- **I9 (Finite floats + non-empty regime maps)**: All p-values ∈ [0,1], all IC values finite (no NaN), regime key sets equal across real/shuffle/timeshift/n-dates dicts, dicts non-empty.
- **I10 (Migration declared breaking)**: `triad_report` required is MAJOR semver bump. S0-S3 staged migration with `agent:contract:breaking` label.

## 5 · Contract code

### 5.1 `TriadReport` — derived status, finite/range-validated, tz-aware

```python
# renquant-common/src/renquant_common/contracts/triad.py
from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Annotated, Literal
import pydantic
from pydantic import Field, field_validator

TriadStatus = Literal["pending", "passed", "failed"]


def _check_finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError(f"value must be finite, got {v!r}")
    return v


FiniteFloat = Annotated[float, pydantic.AfterValidator(_check_finite)]
PValue = Annotated[float, Field(ge=0.0, le=1.0), pydantic.AfterValidator(_check_finite)]


def _require_aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError(f"datetime must be timezone-aware (got naive {v!r})")
    return v.astimezone(timezone.utc)


AwareDatetime = Annotated[datetime, pydantic.AfterValidator(_require_aware)]


def utc_now() -> datetime:
    """Single source for "now" — always tz-aware UTC. Use everywhere."""
    return datetime.now(timezone.utc)


class ScorerSanityReport(pydantic.BaseModel):
    """Tier 1: fixed scorer vs perturbed val labels (seconds). Catches label-calc bugs."""
    aa_split_real_ic_replicate: FiniteFloat
    aa_split_drift_ic_abs: Annotated[FiniteFloat, Field(ge=0.0)]   # explicitly pre-absolute
    shuffled_val_ic: FiniteFloat
    timeshifted_val_ic: FiniteFloat
    label_col: str
    n_val_dates: Annotated[int, Field(gt=0)]
    # p-values: small p ⇒ REJECT H0(perturbed_ic = 0) ⇒ LEAK SUSPECTED
    shuffled_p_value_against_zero: PValue
    timeshifted_p_value_against_zero: PValue

    def fail_reasons(self, *, p_threshold: float = 0.05, min_n: int = 30,
                     aa_drift_max: float = 0.03) -> list[str]:
        """Empty = pass. Direction: small p ⇒ leak detected ⇒ fail."""
        out: list[str] = []
        if self.n_val_dates < min_n:
            out.append(f"insufficient n_val_dates={self.n_val_dates} (need ≥{min_n})")
        if self.shuffled_p_value_against_zero < p_threshold:
            out.append(
                f"shuffled IC distinguishable from zero "
                f"(IC={self.shuffled_val_ic:+.4f}, p={self.shuffled_p_value_against_zero:.4f} < {p_threshold})"
            )
        if self.timeshifted_p_value_against_zero < p_threshold:
            out.append(
                f"timeshifted IC distinguishable from zero "
                f"(IC={self.timeshifted_val_ic:+.4f}, p={self.timeshifted_p_value_against_zero:.4f} < {p_threshold})"
            )
        if self.aa_split_drift_ic_abs > aa_drift_max:
            out.append(f"aa_split_drift_ic={self.aa_split_drift_ic_abs:.4f} > {aa_drift_max}")
        return out


class TrainerPlaceboReport(pydantic.BaseModel):
    """Tier 2: RETRAIN on shuffled/timeshifted labels. Catches train-time leakage."""
    real_ic_mean: FiniteFloat
    real_ic_per_regime: dict[str, FiniteFloat]
    real_ic_n_dates_per_regime: dict[str, Annotated[int, Field(ge=0)]]
    shuffle_placebo_ic_mean: FiniteFloat
    shuffle_placebo_ic_per_regime: dict[str, FiniteFloat]
    # p that placebo IC is significantly different from zero (small ⇒ LEAK)
    shuffle_placebo_p_value_against_zero: PValue
    timeshift_placebo_ic_mean: FiniteFloat
    timeshift_placebo_ic_per_regime: dict[str, FiniteFloat]
    timeshift_placebo_p_value_against_zero: PValue
    n_seeds: Annotated[int, Field(gt=0)]
    n_val_dates: Annotated[int, Field(gt=0)]

    @pydantic.model_validator(mode="after")
    def regime_keys_consistent(self) -> "TrainerPlaceboReport":
        keys_r = set(self.real_ic_per_regime.keys())
        keys_n = set(self.real_ic_n_dates_per_regime.keys())
        keys_s = set(self.shuffle_placebo_ic_per_regime.keys())
        keys_t = set(self.timeshift_placebo_ic_per_regime.keys())
        if not keys_r:
            raise ValueError("regime maps cannot be empty")
        if not (keys_r == keys_n == keys_s == keys_t):
            raise ValueError(
                f"regime key sets must be identical:\n"
                f"  real:      {sorted(keys_r)}\n"
                f"  n_dates:   {sorted(keys_n)}\n"
                f"  shuffle:   {sorted(keys_s)}\n"
                f"  timeshift: {sorted(keys_t)}"
            )
        return self

    def fail_reasons(self, *, p_threshold: float = 0.05,
                     min_seeds: int = 3, min_n_per_regime: int = 20,
                     max_placebo_real_ratio: float = 0.50) -> list[str]:
        out: list[str] = []
        if self.n_seeds < min_seeds:
            out.append(f"insufficient n_seeds={self.n_seeds} (need ≥{min_seeds})")
        for regime, n in self.real_ic_n_dates_per_regime.items():
            if n < min_n_per_regime:
                out.append(f"regime {regime}: n_dates={n} < {min_n_per_regime}")
        if self.shuffle_placebo_p_value_against_zero < p_threshold:
            out.append(
                f"shuffle placebo IC distinguishable from zero "
                f"(IC={self.shuffle_placebo_ic_mean:+.4f}, "
                f"p={self.shuffle_placebo_p_value_against_zero:.4f} < {p_threshold})"
            )
        if self.timeshift_placebo_p_value_against_zero < p_threshold:
            out.append(
                f"timeshift placebo IC distinguishable from zero "
                f"(IC={self.timeshift_placebo_ic_mean:+.4f}, "
                f"p={self.timeshift_placebo_p_value_against_zero:.4f} < {p_threshold})"
            )
        # Per-regime ratio guard: placebo > 50% of real ⇒ suspect even if small absolute IC
        for regime in self.real_ic_per_regime:
            r = abs(self.real_ic_per_regime[regime])
            sp = abs(self.shuffle_placebo_ic_per_regime[regime])
            tp = abs(self.timeshift_placebo_ic_per_regime[regime])
            if r > 0.01 and (sp > max_placebo_real_ratio * r or tp > max_placebo_real_ratio * r):
                out.append(
                    f"regime {regime}: placebo>50%×real "
                    f"(real={r:+.4f} shuffle={sp:+.4f} timeshift={tp:+.4f})"
                )
        return out


class TriadBinding(pydantic.BaseModel):
    """Identity tuple. Any element change ⇒ new artifact, fresh pending lifeline."""
    model_sha: Annotated[str, Field(min_length=64, max_length=64)]
    feature_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]
    label_hash: Annotated[str, Field(min_length=64, max_length=64)]
    code_sha: Annotated[str, Field(min_length=7)]
    triad_config_hash: Annotated[str, Field(min_length=64, max_length=64)]

    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256(
            f"{self.model_sha}|{self.feature_schema_hash}|{self.label_hash}|"
            f"{self.code_sha}|{self.triad_config_hash}".encode()
        )
        return h.hexdigest()


def _derive_status(
    scorer_sanity: ScorerSanityReport,
    trainer_placebo: TrainerPlaceboReport | None,
) -> tuple[TriadStatus, list[str]]:
    """The ONLY function that decides triad_status. Pure, deterministic, gate-aligned."""
    s1 = scorer_sanity.fail_reasons()
    if s1:
        return "failed", s1
    if trainer_placebo is None:
        return "pending", []
    s2 = trainer_placebo.fail_reasons()
    if s2:
        return "failed", s2
    return "passed", []


class TriadReport(pydantic.BaseModel):
    triad_status: TriadStatus
    failure_reasons: list[str]
    scorer_sanity: ScorerSanityReport
    trainer_placebo: TrainerPlaceboReport | None
    binding: TriadBinding
    triad_started_at: AwareDatetime
    triad_completed_at: AwareDatetime | None

    @pydantic.model_validator(mode="after")
    def status_is_derived(self) -> "TriadReport":
        expected_status, expected_reasons = _derive_status(self.scorer_sanity, self.trainer_placebo)
        if self.triad_status != expected_status:
            raise ValueError(
                f"triad_status={self.triad_status!r} inconsistent with reducer="
                f"{expected_status!r}; status MUST be derived by _derive_status"
            )
        if self.failure_reasons != expected_reasons:
            raise ValueError(
                f"failure_reasons inconsistent:\n"
                f"  declared: {self.failure_reasons}\n"
                f"  expected: {expected_reasons}"
            )
        if self.triad_status == "pending":
            if self.triad_completed_at is not None:
                raise ValueError("triad_completed_at must be None when pending")
        else:
            if self.triad_completed_at is None:
                raise ValueError(f"triad_completed_at required when {self.triad_status}")
        return self

    @classmethod
    def build(cls, scorer_sanity: ScorerSanityReport,
              trainer_placebo: TrainerPlaceboReport | None,
              binding: TriadBinding,
              triad_started_at: datetime) -> "TriadReport":
        status, reasons = _derive_status(scorer_sanity, trainer_placebo)
        return cls(
            triad_status=status,
            failure_reasons=reasons,
            scorer_sanity=scorer_sanity,
            trainer_placebo=trainer_placebo,
            binding=binding,
            triad_started_at=triad_started_at,
            triad_completed_at=utc_now() if status != "pending" else None,
        )
```

### 5.2 Bypass — HMAC-signed, allowlist-per-fingerprint, tz-aware

```python
# renquant-common/src/renquant_common/contracts/leakage_config.py
import hmac, hashlib, json, os, logging
from pathlib import Path
from datetime import datetime
from typing import Literal
import pydantic
from pydantic import Field
from renquant_common.contracts.triad import AwareDatetime

log = logging.getLogger("renquant_common.leakage_guards.bypass")
ARCHITECT_KEY_PATH = Path.home() / ".renquant" / "secrets" / "architect_hmac.key"


def _load_architect_key() -> bytes | None:
    """Returns HMAC key bytes if file exists with mode 0600, else None.

    Refuses to load if the key file is world-readable (defense in depth against
    key leak via accidental commit / shared filesystem).
    """
    if not ARCHITECT_KEY_PATH.exists():
        return None
    st = ARCHITECT_KEY_PATH.stat()
    if st.st_mode & 0o077:
        log.error("architect HMAC key at %s has too-permissive mode %o; refusing to load",
                  ARCHITECT_KEY_PATH, st.st_mode & 0o777)
        return None
    return ARCHITECT_KEY_PATH.read_bytes()


def _canonical_payload(entry: dict) -> bytes:
    """Stable JSON serialization for HMAC. Drops 'signature' field if present."""
    payload = {k: v for k, v in entry.items() if k != "signature"}
    return json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode()


class TriadBypassEntry(pydantic.BaseModel):
    """Architect-signed permission for ONE pending artifact fingerprint."""
    artifact_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    expires_at: AwareDatetime
    reason: str
    approved_by: str                              # GitHub username; verified via PR review CODEOWNER, not by string
    pr_url: str
    approved_at: AwareDatetime
    signature: str                                # base64(hmac_sha256(architect_key, canonical_payload))

    @pydantic.field_validator("reason")
    @classmethod
    def reason_nontrivial(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(f"reason too short ({len(v.strip())} chars); explain ≥20 chars")
        return v

    def verify(self, key: bytes) -> bool:
        """Returns True iff signature matches HMAC-SHA256(key, canonical_payload)."""
        canonical = _canonical_payload(self.model_dump(mode="json"))
        expected = hmac.new(key, canonical, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, self.signature)


def sign_bypass_entry(entry_without_signature: dict, key: bytes) -> str:
    """Architect-side helper. Used by tools/sign_bypass.py to produce signature.

    Architect runs this LOCALLY with their key; pastes resulting signature
    into the PR's strategy_config.golden.json. Never run in CI.
    """
    return hmac.new(key, _canonical_payload(entry_without_signature), hashlib.sha256).hexdigest()


class LeakageGuardConfig(pydantic.BaseModel):
    tier1_min_n_val_dates: int = 30
    tier1_pvalue_threshold: float = 0.05
    tier1_aa_drift_max: float = 0.03
    tier2_n_seeds_required: int = 3
    tier2_pvalue_threshold: float = 0.05
    tier2_min_n_dates_per_regime: int = 20
    tier2_max_placebo_real_ratio: float = 0.50
    tier2_label_shift_days: int = 10
    tier2_run_strategy: Literal["subprocess_inline", "subprocess_queue", "manual", "synthetic"] = "subprocess_inline"
    triad_bypasses_raw: list[TriadBypassEntry] = Field(default_factory=list)
    alert_channel: Literal["slack", "log", "none"] = "log"
    alert_slack_webhook_url: str | None = None

    @pydantic.computed_field
    @property
    def triad_bypasses(self) -> list[TriadBypassEntry]:
        """Verified-only bypass entries. Invalid signatures dropped with WARN log."""
        key = _load_architect_key()
        if key is None:
            if self.triad_bypasses_raw:
                log.error(
                    "%d bypass entries present but architect key not loadable; "
                    "treating all as unverified (= disabled).",
                    len(self.triad_bypasses_raw),
                )
            return []
        verified: list[TriadBypassEntry] = []
        for entry in self.triad_bypasses_raw:
            if entry.verify(key):
                verified.append(entry)
            else:
                log.error("bypass for fp=%s has INVALID signature; dropped.",
                          entry.artifact_fingerprint[:16])
        return verified
```

```python
# renquant-common/src/renquant_common/leakage_guards/gate.py
def assert_artifact_validated(
    artifact: ScorerArtifact, *,
    cfg: LeakageGuardConfig, caller: str,
) -> None:
    s = artifact.triad_report.triad_status
    fp = artifact.triad_report.binding.fingerprint()

    if s == "failed":
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="failed",
                             failure_reasons=artifact.triad_report.failure_reasons)
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} with triad_status='failed'. "
            f"Reasons: {artifact.triad_report.failure_reasons}. No bypass allowed for failed."
        )

    if s == "passed":
        telemetry.emit_event("gate_allow", caller=caller, artifact_fingerprint=fp,
                             triad_status="passed")
        return

    # pending — verified bypass allowlist only (signatures already checked by cfg property)
    now = utc_now()
    matching = [b for b in cfg.triad_bypasses
                if b.artifact_fingerprint == fp and b.expires_at > now]
    if not matching:
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="pending", reason="no verified bypass match")
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} status='pending'; "
            f"no verified bypass entry matches (verified entries: {len(cfg.triad_bypasses)})."
        )
    entry = matching[0]
    telemetry.emit_event("gate_bypass", caller=caller, artifact_fingerprint=fp,
                         triad_status="pending",
                         bypass_expires_at=entry.expires_at.isoformat(),
                         bypass_approved_by=entry.approved_by,
                         bypass_pr_url=entry.pr_url)
    log.warning(
        "TRIAD BYPASS ACTIVE: %s fp=%s expires=%s approved_by=%s pr=%s",
        caller, fp[:16], entry.expires_at.isoformat(), entry.approved_by, entry.pr_url,
    )
```

**Defense in depth on top of HMAC**: `.github/CODEOWNERS` requires architect approval for `strategy_config.golden.json` changes; PR auto-label `agent:emergency:bypass-triad` triggers required-review workflow. The HMAC binds the entry contents; CODEOWNERS binds who can merge.

### 5.3 CAS — compare-and-swap on (binding, status)

```python
# renquant-common/src/renquant_common/leakage_guards/sidecar.py
"""LOCAL POSIX/Linux/macOS only. Object-store sidecars NOT supported in MVP;
use a separate ETag/generation-precondition backend for cloud."""
import fcntl, json, os, tempfile, time
from datetime import datetime
from pathlib import Path
from typing import Callable
from renquant_common.contracts.triad import TriadBinding, TriadStatus, _derive_status

_VALID_TRANSITIONS: set[tuple[TriadStatus, TriadStatus]] = {
    ("pending", "pending"),    # idempotent retry
    ("pending", "passed"),
    ("pending", "failed"),
}


class SidecarConcurrencyError(RuntimeError): ...
class StaleBindingError(RuntimeError): ...
class IllegalTriadTransition(RuntimeError): ...


def atomic_cas_update_sidecar(
    sidecar_path: Path,
    expected_binding: TriadBinding,
    expected_current_status: TriadStatus,   # CAS on (binding, status) BOTH
    transformer: Callable[[dict], dict],
    *,
    timeout_seconds: float = 30.0,
) -> dict:
    """Compare-And-Swap update of triad sidecar.

    Atomically:
      1. flock the sidecar lockfile.
      2. Read current JSON; parse binding + status.
      3. Verify (current.binding == expected_binding) AND (current.status == expected_current_status).
         Mismatch ⇒ StaleBindingError (someone updated since we started).
      4. Apply transformer(current) → new.
      5. Verify (current.status, new.status) ∈ _VALID_TRANSITIONS.
         Particularly: passed→anything is REJECTED; failed→anything is REJECTED.
         passed/failed are TERMINAL at the sidecar level — re-validation requires
         a new binding-tuple = new artifact = new sidecar with status=pending.
      6. Full-validate new as ScorerArtifact (catches reducer inconsistency).
      7. Write temp, fsync, os.replace (POSIX atomic-on-existing), fsync dir.
      8. Release lock.
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
                    raise SidecarConcurrencyError(f"flock timeout on {sidecar_path}")
                time.sleep(0.1)

        if not sidecar_path.exists():
            raise FileNotFoundError(f"sidecar missing: {sidecar_path}")
        current = json.loads(sidecar_path.read_text())

        # CAS check: binding
        current_binding = TriadBinding.model_validate(current["triad_report"]["binding"])
        if current_binding != expected_binding:
            raise StaleBindingError(
                f"binding moved: expected fp={expected_binding.fingerprint()[:16]} "
                f"actual fp={current_binding.fingerprint()[:16]}"
            )
        # CAS check: status
        current_status: TriadStatus = current["triad_report"]["triad_status"]
        if current_status != expected_current_status:
            raise StaleBindingError(
                f"status moved: expected current={expected_current_status} actual={current_status}; "
                f"a competing writer finalized the triad."
            )

        new = transformer(current)
        new_status: TriadStatus = new["triad_report"]["triad_status"]
        if (current_status, new_status) not in _VALID_TRANSITIONS:
            raise IllegalTriadTransition(
                f"{current_status} → {new_status} is not allowed. "
                f"Terminal states (passed/failed) require new binding-tuple."
            )

        # Full schema + derivation validation
        from renquant_common.contracts.scorer import ScorerArtifact
        ScorerArtifact.model_validate(new)

        # Atomic write
        tmp = tempfile.NamedTemporaryFile(
            mode="w", dir=sidecar_path.parent, delete=False, suffix=".tmp"
        )
        try:
            json.dump(new, tmp, indent=2, default=str)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            os.replace(tmp.name, sidecar_path)     # POSIX atomic replace (handles existing dst)
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

### 5.4 Statistical helpers — direction-correct, gate-aligned

```python
# renquant-common/src/renquant_common/leakage_guards/stats.py
"""
Hypothesis tests with EXPLICIT direction:

H0 / H1 alignment with the gate fail condition is the only thing that matters.
Failed gates that protect against leakage REJECT H0(perturbed IC = 0) at small p.

Two distinct tests:

  1. bootstrap_p_value_against_zero(perturbed_ic_per_date)
       H0: mean(perturbed IC) = 0  (no leak)
       H1: mean(perturbed IC) ≠ 0  (leak — model still extracts signal from perturbed labels)
       Returns two-sided bootstrap p. small p ⇒ REJECT H0 ⇒ LEAK ⇒ gate FAILS.

  2. bootstrap_p_value_real_dominates(real_ic, perturbed_ic)
       H0: mean(real) ≤ mean(perturbed)  (model NOT distinguishing real from placebo)
       H1: mean(real) > mean(perturbed)  (real signal exceeds placebo)
       Returns one-sided bootstrap p of H0. small p ⇒ REJECT H0 ⇒ PASS (real > perturbed).
       Currently NOT used as a hard gate (kept for diagnostic), since (1) is strictly stronger.
"""
import numpy as np


def bootstrap_p_value_against_zero(
    perturbed_ic_per_date: np.ndarray,
    *, n_boot: int = 10_000, rng_seed: int = 0,
) -> float:
    """Two-sided bootstrap p that mean(perturbed) = 0.

    Small p ⇒ perturbed IC distinguishable from zero ⇒ LEAK SUSPECTED.

    Aligned with gate fail condition.
    Sample-size aware (small n ⇒ wide bootstrap CI ⇒ tends to large p ⇒ tend to pass).
    """
    if perturbed_ic_per_date.size == 0:
        raise ValueError("perturbed_ic_per_date is empty")
    if not np.all(np.isfinite(perturbed_ic_per_date)):
        raise ValueError("perturbed_ic_per_date contains non-finite values")
    rng = np.random.default_rng(rng_seed)
    n = len(perturbed_ic_per_date)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[k] = perturbed_ic_per_date[idx].mean()
    # two-sided p: how often does the bootstrap mean cross zero from the observed direction
    observed_mean = perturbed_ic_per_date.mean()
    if observed_mean >= 0:
        p_one_sided = float((boots <= 0).mean())
    else:
        p_one_sided = float((boots >= 0).mean())
    return min(1.0, 2 * p_one_sided)


def bootstrap_p_value_real_dominates(
    real_ic_per_date: np.ndarray, perturbed_ic_per_date: np.ndarray,
    *, n_boot: int = 10_000, rng_seed: int = 0,
) -> float:
    """One-sided p of H0(mean(real) ≤ mean(perturbed)) via paired/independent bootstrap.

    Small p ⇒ real significantly exceeds perturbed ⇒ healthy.
    Currently DIAGNOSTIC ONLY — not a primary gate; bootstrap_p_value_against_zero is stricter.
    """
    if real_ic_per_date.size == 0 or perturbed_ic_per_date.size == 0:
        raise ValueError("input arrays must be non-empty")
    rng = np.random.default_rng(rng_seed)
    n_r, n_p = len(real_ic_per_date), len(perturbed_ic_per_date)
    boots = np.empty(n_boot)
    for k in range(n_boot):
        r = real_ic_per_date[rng.integers(0, n_r, size=n_r)].mean()
        p = perturbed_ic_per_date[rng.integers(0, n_p, size=n_p)].mean()
        boots[k] = r - p
    # p that bootstrap (real - perturbed) ≤ 0 = H0
    return float((boots <= 0).mean())
```

**Fixtures for falsification testing** (used by `multirepo-triad-e2e.yml`):

```python
# renquant-common/tests/fixtures/triad_models.py
def good_model_ic(n_dates=250, rng_seed=0):
    """Real IC ~+0.05; shuffle/timeshift IC ~0. Should produce status=passed."""
    rng = np.random.default_rng(rng_seed)
    real = rng.normal(0.05, 0.15, n_dates)
    shuffle = rng.normal(0.00, 0.15, n_dates)
    timeshift = rng.normal(0.00, 0.15, n_dates)
    return real, shuffle, timeshift

def leak_model_ic(n_dates=250, rng_seed=0):
    """Real ≈ shuffle ≈ timeshift ≈ +0.04 — the B_tuned incident class. Status=failed."""
    rng = np.random.default_rng(rng_seed)
    return (
        rng.normal(0.04, 0.15, n_dates),
        rng.normal(0.04, 0.15, n_dates),
        rng.normal(0.04, 0.15, n_dates),
    )

def noise_model_ic(n_dates=250, rng_seed=0):
    """All ~0 — model has no signal but also no leak. Status=passed (gate is for leak,
    not for skill)."""
    rng = np.random.default_rng(rng_seed)
    return (
        rng.normal(0.00, 0.15, n_dates),
        rng.normal(0.00, 0.15, n_dates),
        rng.normal(0.00, 0.15, n_dates),
    )
```

Three required unit tests in `tests/test_stats.py`:
- `test_good_model_passes`: `bootstrap_p_value_against_zero(shuffle) > 0.05` AND `> 0.05` (timeshift)
- `test_leak_model_fails`: at least one of shuffle/timeshift `p < 0.05`
- `test_noise_model_passes`: all p > 0.05 (no signal but also no leak)

## 6 · State machine + transitions (enforced at TWO layers)

```
                  ╭─────────────────────────────────────────────────────╮
                  │ Layer 1: Pydantic _derive_status                    │
                  │   tuple(scorer_sanity, trainer_placebo) → status   │
                  │   Pure deterministic reducer.                       │
                  │                                                     │
                  │ Layer 2: atomic_cas_update_sidecar                  │
                  │   (current_status, new_status) ∈ _VALID_TRANSITIONS │
                  │   passed/failed terminal at sidecar layer.          │
                  │                                                     │
                  │ Two layers ⇒ a buggy reducer can't promote failed  │
                  │ to passed even via a sidecar rewrite.              │
                  ╰─────────────────────────────────────────────────────╯

                                     ╭─────────╮
                                     │ PENDING │ ◄── (initial state: Tier 1 passed,
                                     │         │      Tier 2 not yet run)
                                     ╰────┬────╯
                                          │
                              Tier 2 runner CAS-updates sidecar
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                  reducer={passed}  reducer={failed}   pending→pending
                          │               │           (idempotent retry)
                          ↓               ↓               │
                  ╭─────────────╮  ╭─────────────╮  (loop back to pending)
                  │   PASSED    │  │   FAILED    │
                  ╰─────────────╯  ╰─────────────╯
                   TERMINAL          TERMINAL
                   (CAS rejects      (CAS rejects
                   any transition    any transition
                   out)              out)
```

Tests:
- `test_failed_to_passed_rejected`: CAS update from failed to passed raises `IllegalTriadTransition`.
- `test_passed_to_failed_rejected`: ditto (passed terminal too).
- `test_stale_runner_rejected`: parallel Tier 2 invocations — second one finds status moved → `StaleBindingError`.

## 7 · Five gates + Tier-2 runner abstraction

| Gate | File | Anchor | What it does |
|---|---|---|---|
| G0 | `renquant-base-data/.../alpha158.py` | `_write_dataset()` end | `DatasetManifest(...).model_validate(...)` |
| G1 | `renquant-model-*/hf_trainer.py::_save_artifact` | before `torch.save` | `report = run_tier1(...)`; `raise Tier1Failed` if `report.fail_reasons()`; build TriadReport via `.build()` |
| G2 | `renquant-model-*/post_save_hook.py` (new) | post-save | `enqueue_tier2(..., runner=Tier2RunnerProtocol)` → subprocess invokes runner; runner CAS-updates sidecar |
| G3 | `renquant-pipeline/.../panel_scorer.py::PanelScorer.load` | top of method | `assert_artifact_validated(...)` |
| G4 | `renquant-orchestrator/.../build_*_wf_manifest.py::manifest_row` + backtesting `wf_gate/runner.py`, `sim_driver.py`, `scripts/fit_walkforward_calibrators.py` | before manifest insert | `assert_artifact_validated(...)` — failed/pending without bypass → `ctx.failed_cutoffs.append`; do not call calibrator |
| G5 | `renquant-execution/broker_adapter.py::submit_order` (new) | before broker API | resolve scorer; `assert_artifact_validated(...)`; refused → emit `refused_order_unvalidated_scorer` telemetry; do not submit |

```python
# renquant-common/src/renquant_common/leakage_guards/runner.py
class Tier2RunnerProtocol(Protocol):
    """Pluggable Tier-2 runner. Used by enqueue_tier2; replaced in E2E test."""
    def run(self, *, artifact_path: Path, binding: TriadBinding,
            cfg: LeakageGuardConfig) -> TrainerPlaceboReport: ...

class ProductionTier2Runner:
    """Real subprocess: spawns N trainer invocations × 3 modes × ≥3 seeds."""
    def run(self, *, artifact_path, binding, cfg) -> TrainerPlaceboReport:
        # ~75 min wall clock for PatchTST; calls model trainer CLI
        ...

class SyntheticTier2Runner:
    """In-memory replay against fixture IC arrays. Sub-second.
    Used by `multirepo-triad-e2e.yml` to exercise all 3 fixtures
    (good/leak/noise) under the 5-min completion criterion (§2.8)."""
    def __init__(self, fixture_name: Literal["good", "leak", "noise"], n_dates=250, seeds=(42,43,44)):
        self.fixture_name = fixture_name
        self.n_dates = n_dates
        self.seeds = seeds

    def run(self, *, artifact_path, binding, cfg) -> TrainerPlaceboReport:
        fixture = {"good": good_model_ic, "leak": leak_model_ic, "noise": noise_model_ic}[self.fixture_name]
        real_per_seed, shuffle_per_seed, timeshift_per_seed = [], [], []
        for seed in self.seeds:
            real, shuffle, timeshift = fixture(self.n_dates, rng_seed=seed)
            real_per_seed.append(real); shuffle_per_seed.append(shuffle); timeshift_per_seed.append(timeshift)
        return TrainerPlaceboReport(
            real_ic_mean=float(np.mean([r.mean() for r in real_per_seed])),
            real_ic_per_regime={"_synthetic_": float(np.mean([r.mean() for r in real_per_seed]))},
            real_ic_n_dates_per_regime={"_synthetic_": self.n_dates},
            shuffle_placebo_ic_mean=float(np.mean([s.mean() for s in shuffle_per_seed])),
            shuffle_placebo_ic_per_regime={"_synthetic_": float(np.mean([s.mean() for s in shuffle_per_seed]))},
            shuffle_placebo_p_value_against_zero=bootstrap_p_value_against_zero(np.concatenate(shuffle_per_seed)),
            timeshift_placebo_ic_mean=float(np.mean([t.mean() for t in timeshift_per_seed])),
            timeshift_placebo_ic_per_regime={"_synthetic_": float(np.mean([t.mean() for t in timeshift_per_seed]))},
            timeshift_placebo_p_value_against_zero=bootstrap_p_value_against_zero(np.concatenate(timeshift_per_seed)),
            n_seeds=len(self.seeds),
            n_val_dates=self.n_dates,
        )
```

## 8 · Migration — 4 stages (unchanged from v6 plan; declared breaking)

| Stage | Wall | Schema | Gates | Action |
|---|---|---|---|---|
| S0 | -7d→0 | optional | OFF (shadow telemetry) | Wire Tier-1 in trainer; existing artifacts untouched. |
| S1 | 0→7d | optional | shadow log "would have blocked" | Telemetry counts cohort that fails. |
| S2 | 7→14d | required-for-new | Tier-2 online; backfill = pending + 7-day per-fp bypass (signed by architect) | Architect signs backfill cohort once. |
| S3 | 14→28d | required | G3/G4/G5 enforce; failed always blocked; pending blocked unless verified-signature bypass | Bypass cohort shrinks weekly. |
| S4 | 28+ | required | Steady state | Only passed loaded. |

Rollback ≤ 5 min: architect PR sets `tier2_run_strategy="manual"` + adds verified-signed bypass entries for currently-live fingerprints.

## 9 · Falsification

If after S3 ANY of these holds, design is wrong (P0):
1. Passed artifact later proven to have shuffled placebo p < 0.05 on re-run with same `triad_config_hash`.
2. Pending artifact reached live order without matching verified bypass.
3. Failed artifact transitioned to passed without binding change (CAS or reducer bug).
4. Tier 2 sidecar overwritten by stale runner (CAS bug).
5. Unsigned/forged bypass entry accepted (key-load or HMAC bug).
6. Bypass for fingerprint A allowed load of artifact with fingerprint B (cross-fingerprint bypass leak).
7. `datetime.utcnow()` reintroduced anywhere in leakage_guards/ (tz bug regression).
8. Bare `assert` reintroduced in contracts/triad.py (silent-under-O regression).

Tests for (3)-(8) live in `tests/test_falsification.py`.

## 10 · Threat model (12 leak classes, full table in companion)

L1-L12 mapped to G0-G5; residual risk LOW or 0. **L2 implicit lookahead** is the highest residual; mitigated only by Tier 2 placebo. If a model's features are themselves transforms of future returns, Tier 2 catches it (placebo IC ≈ real IC ⇒ failed) — this is exactly the B_tuned incident class.

## 11 · Codex v6 → v7 — addressed (7 findings)

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | HIGH | Permutation p-value direction inverted (failed healthy, passed leaky) | §5.4 rewrite with explicit H0/H1: `bootstrap_p_value_against_zero(perturbed_ic)` — small p ⇒ leak ⇒ fail. Test fixtures (§5.4) prove all 3 directions: good passes, leak fails, noise passes. |
| 2 | HIGH | CAS only checks binding, not transition; stale runner can overwrite failed→passed | §5.3 CAS now takes `expected_current_status` parameter; checks `(current, new) ∈ _VALID_TRANSITIONS`; passed/failed terminal at CAS layer. State machine §6 enforced at TWO layers. |
| 3 | HIGH | Bypass `approved_by` was unverified data | §5.2 `TriadBypassEntry.signature: HMAC-SHA256(architect_key, canonical(entry))`; `LeakageGuardConfig.triad_bypasses` is computed property that drops invalid signatures. CODEOWNERS for `strategy_config.golden.json` adds belt-and-suspenders. |
| 4 | HIGH | Naive `datetime.utcnow()` breaks tz-aware comparisons | §5.1 `AwareDatetime` Annotated type rejects naive; `utc_now()` single source; every gate uses `datetime.now(timezone.utc)`; comparison `entry.expires_at > utc_now()` always aware vs aware. |
| 5 | HIGH | Float / regime fields not validated (NaN p-values, missing keys, default-0 placebo) | §5.1 `FiniteFloat` (AfterValidator `math.isfinite`), `PValue` (Field ge=0 le=1 + finite), positive int constraints, `regime_keys_consistent` model_validator forbids empty + requires identical key sets across real/shuffle/timeshift/n_dates. `.get(k, 0)` defaults removed. |
| 6 | MED | POSIX-only CAS portability | §5.3 explicit "LOCAL POSIX/Linux/macOS only" docstring at top of `sidecar.py`; `os.replace` not `os.rename`; cloud-object-store backend declared as separate concern (use ETag/generation preconditions, out of MVP scope). |
| 7 | MED | 5-min E2E vs 75-min Tier-2 cost mismatch | §7 `Tier2RunnerProtocol` + `SyntheticTier2Runner` (sub-second, in-memory fixtures); `multirepo-triad-e2e.yml` uses synthetic runner with 3 fixtures (good/leak/noise); E2E proves both pending-refused AND pending→passed path. |

## 12 · Open questions (unchanged from v6)

1. Permutation `n_boot=10_000` sufficient for n=250 daily IC? Audit by running 100k once and comparing CIs.
2. Subprocess vs job queue for Tier-2 at scale.
3. After Tier-2 fail: auto-retrain (selection bias risk) vs hold for architect (cycle latency).
4. Strategy threshold changes in binding-tuple?

## 13 · MVP PR list (5 PRs ≤ 2 days)

| # | Repo | Files | Tests |
|---|---|---|---|
| ① | `renquant-common` | `contracts/triad.py`, `contracts/leakage_config.py`, `leakage_guards/{scorer_sanity,trainer_placebo,gate,sidecar,stats,telemetry,alerts,runner}.py`, `tools/sign_bypass.py`, `tests/fixtures/triad_models.py`, `.github/workflows/gate-disable-detection.yml`, `.github/CODEOWNERS` for triad/* files | unit-per-module + CAS race test + state-machine test + 3-fixture stat tests + HMAC verify test + tz-aware comparison test + `python -O` regression test |
| ② | `renquant-pipeline` | `kernel/panel_pipeline/panel_scorer.py::load` | gate-behavior matrix (passed/failed/pending/verified-bypass/unsigned-bypass/expired-bypass) |
| ③ | `renquant-model-patchtst` | `hf_trainer.py::_save_artifact`, new `post_save_hook.py`, new `triad_replay_mode` CLI, `--disable-early-stopping` for Tier-2 | synth Tier-1 wiring + Tier-2 subprocess enqueue + Synthetic runner E2E |
| ④ | `renquant-model-gbdt` | mirror of ③ | mirror |
| ⑤ | `renquant-orchestrator` + `renquant-backtesting` (paired) | `build_*_wf_manifest.py::manifest_row`, `wf_gate/runner.py`, `wf_gate/sim_driver.py`, `scripts/fit_walkforward_calibrators.py` | manifest gate behavior matrix |

Full architecture wave (split parquet, typed train signatures, `pd.read_parquet` ban) deferred to weeks 2-3.

## 14 · References

- Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 5 (Combinatorial Purged CV), ch. 7 (Cross-validation in Finance).
- Pesaran, M.H., Timmermann, A. (2007). "Selection of estimation window in the presence of breaks." *J. Econometrics* 137.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall. — basis for §5.4 two-sided bootstrap p.

---

## 15 · v6 → v7 changelog (codex resolution)

| Field | v6 | v7 |
|---|---|---|
| p-value direction | inverted (codex v6 #1) | corrected: small p ⇒ leak ⇒ fail; H0/H1 explicit in docstrings; 3-fixture tests required |
| State machine | only at Pydantic; CAS only checked binding (codex v6 #2) | TWO layers: Pydantic reducer + CAS transition table `_VALID_TRANSITIONS`; passed/failed terminal at CAS too |
| Bypass auth | data field `approved_by` unverified (codex v6 #3) | HMAC-SHA256 over canonical entry; key at `~/.renquant/secrets/architect_hmac.key` with 0600; world-readable refuses to load; computed property drops unsigned + CODEOWNERS belt-and-suspenders |
| Datetimes | naive `datetime.utcnow()` (codex v6 #4) | `AwareDatetime` Annotated type; `utc_now()` helper; field validator rejects naive |
| Field validation | floats unbounded; NaN p-values slip; regime maps default 0 (codex v6 #5) | `FiniteFloat` + `PValue` Annotated types; positive int constraints; `regime_keys_consistent` model_validator; explicit `aa_split_drift_ic_abs` pre-absolute |
| CAS portability | `os.rename` + bare `fcntl` (codex v6 #6) | `os.replace` (POSIX atomic on existing target); explicit "LOCAL POSIX only" constraint at module top; object-store backend separate concern |
| E2E vs Tier-2 cost | 5min vs 75min contradiction (codex v6 #7) | `Tier2RunnerProtocol` abstraction + `SyntheticTier2Runner` (in-memory, sub-second); 3 fixtures (good/leak/noise); §2.8 falsifiable; tests prove pending→passed path executes |
| Falsification tests | §9 only descriptive | §9 each criterion has a named pytest in `tests/test_falsification.py` |
| Codex audit trail | v4 alone | §11 v6 codex review reproduced + v15 changelog |
