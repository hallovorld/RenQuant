# RenQuant Multirepo Leakage Defense — Architecture (v8)

**Status**: Design under review (RenQuant PR #38)
**Authors**: Claude
**Reviewers**: Codex (6 review rounds; v7 found 3 HIGH + 2 MED in policy binding / artifact integrity / signature primitive — all addressed)
**Supersedes**: v1-v7 (commit history is the audit trail)
**Companion**: `doc/research/2026-06-01-leakage-reflection.md`

---

## 1 · Problem statement

`renquant-model-patchtst::B_tuned` shows shuffle/timeshift placebo IC (+0.041) ≈ real IC (+0.041) across 2 seeds (5/31 and 6/01). PR #9 fixed cross-split timeshift boundary; placebo failure persists. **No enforced layer prevents a model with placebo IC ≈ real IC from reaching production.** 4 downstream consumers (`renquant-pipeline`, `renquant-backtesting`, `renquant-orchestrator`, `renquant-execution`) don't check trainer-level placebos.

## 2 · Completion criteria (falsifiable)

Design ships when **every one is mechanically true**:

1. An artifact with shuffle/timeshift placebo distinguishable from null (p < threshold) **cannot** load via `PanelScorer.load()`. (G3)
2. Same artifact **cannot** insert into any WF manifest. (G4)
3. Same artifact **cannot** source a live order. (G5)
4. A feature parquet with label-named columns **cannot** be written. (G0)
5. A trainer that skips Tier-1 **cannot** save. (G1)
6. Disabling any G0-G5 in code detected by CI AST scan.
7. Bypass entries without valid Ed25519 signature by an enrolled architect pubkey rejected at config load.
8. Bypass entry signed for gate X is **rejected by gate Y** (per-gate scope binding).
9. Artifact whose recomputed `sha256(model.pt bytes)` ≠ `binding.model_sha` rejected at `Scorer.load()`. (artifact integrity)
10. Reducer policy used at validation time is the policy whose `sha256(canonical)` = `binding.triad_config_hash`. Mismatch ⇒ reject.
11. Consumer with `minimum_acceptable_policy` stricter than artifact's policy rejects the artifact, regardless of `triad_status`. (policy-strength gate)
12. E2E nightly runs 3 fixtures (good/leak/noise) × synthetic Tier-2 under 5 min wall clock.

**Any false ⇒ design failed ⇒ revert.**

## 3 · Designs considered & rejected (unchanged from v7)

A advisory / B annotations alone / C single chokepoint / D disable model — all rejected.
E (this design): multi-gate + policy-bound derived status + Ed25519-signed scoped bypass + artifact-byte verification + CAS-with-transition.

## 4 · Invariants

- **I1 (Artifact contract)**: `ScorerArtifact` carries `TriadReport` with `triad_status` deterministically derived from `policy + scorer_sanity + trainer_placebo`.
- **I2 (Terminal failure)**: passed/failed terminal at BOTH Pydantic (reducer) AND CAS layer (`_VALID_TRANSITIONS`).
- **I3 (Gate symmetry)**: G3/G4/G5 share one helper.
- **I4 (Artifact-byte integrity)**: `ScorerArtifact.load(path)` recomputes `sha256(model.pt bytes)`, rejects on mismatch with `binding.model_sha`. Sidecars cannot vouch for arbitrary bytes.
- **I5 (Policy-hash binding)**: `binding.triad_config_hash = sha256(canonical(TriadPolicy))`. Reducer takes policy explicitly. Consumer rejects artifacts whose policy hash isn't in `consumer.acceptable_policy_hashes` OR whose policy is `weaker_than(minimum_acceptable_policy)`.
- **I6 (Asymmetric signed bypass)**: `TriadBypassEntry.signature` is Ed25519 over the canonical bytes. Verify keys live in repo at `keys/architect_pubkeys.json`; signing key lives offline. Compromise of a verifier cannot mint bypasses. `key_id` field supports rotation.
- **I7 (Per-gate bypass scope)**: bypass `allowed_gates` is part of signed payload; gate refuses if `caller` not in that set. A backfill bypass cannot authorize a live order.
- **I8 (Canonical serialization)**: signing and verification both use `canonical_payload()` defined once. UTC datetimes serialized as `YYYY-MM-DDTHH:MM:SS.ffffffZ`. No `default=str`. Golden vector test pinned in repo.
- **I9 (CAS + transition)**: sidecar update CAS-bound to `(binding, current_status)`; transitions table; only `pending → {pending, passed, failed}` permitted.
- **I10 (Statistical, n-aware, direction-correct)**: bootstrap p-values with explicit H0/H1, gate-aligned (small p ⇒ leak ⇒ fail).
- **I11 (Tz-aware datetimes)**: `AwareDatetime`; naive rejected.
- **I12 (Finite floats + non-empty regime maps)**: `FiniteFloat`, `PValue`, `regime_keys_consistent`.
- **I13 (Disable detection)**: CI AST scan + regex on `leakage_guards/`.
- **I14 (Migration breaking)**: declared MAJOR semver; 4-stage S0-S3.

## 5 · Contract code

### 5.1 `TriadPolicy` (NEW v8) — frozen, hash-bound to artifact

```python
# renquant-common/src/renquant_common/contracts/triad.py (excerpt)
import math
from datetime import datetime, timezone
from typing import Annotated, Literal
import pydantic
from pydantic import Field

# --- primitive types (v7 carried forward) -----------------------------------

def _check_finite(v: float) -> float:
    if not math.isfinite(v):
        raise ValueError(f"value must be finite, got {v!r}")
    return v

FiniteFloat = Annotated[float, pydantic.AfterValidator(_check_finite)]
PValue = Annotated[float, Field(ge=0.0, le=1.0), pydantic.AfterValidator(_check_finite)]

def _require_aware(v: datetime) -> datetime:
    if v.tzinfo is None:
        raise ValueError(f"datetime must be tz-aware (got naive {v!r})")
    return v.astimezone(timezone.utc)

AwareDatetime = Annotated[datetime, pydantic.AfterValidator(_require_aware)]

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


# --- v8 NEW: TriadPolicy -----------------------------------------------------

class TriadPolicy(pydantic.BaseModel):
    """The set of thresholds that determine pass/fail. Frozen.

    `policy_hash()` is the canonical SHA-256 that goes into
    TriadBinding.triad_config_hash. Reducer takes a TriadPolicy explicitly;
    no hidden defaults.
    """
    model_config = pydantic.ConfigDict(frozen=True)

    # Tier 1 (scorer sanity)
    tier1_p_threshold: PValue = 0.05
    tier1_min_n_val_dates: Annotated[int, Field(gt=0)] = 30
    tier1_aa_drift_max: Annotated[FiniteFloat, Field(ge=0.0)] = 0.03

    # Tier 2 (trainer placebo)
    tier2_p_threshold: PValue = 0.05
    tier2_min_n_seeds: Annotated[int, Field(gt=0)] = 3
    tier2_min_n_dates_per_regime: Annotated[int, Field(gt=0)] = 20
    tier2_max_placebo_real_ratio: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.50

    # Bootstrap / permutation parameters (must hash too)
    bootstrap_n_iters: Annotated[int, Field(gt=0)] = 10_000
    bootstrap_rng_seed: int = 0

    def policy_hash(self) -> str:
        """Canonical SHA-256. The ONLY way to compute binding.triad_config_hash."""
        import hashlib, json
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def strength_score(self) -> tuple:
        """Used by `weaker_than`. Larger tuple ⇒ stricter policy.

        Comparison: tighter p (smaller value), more seeds, more dates, lower
        placebo ratio. We negate the small-is-tight ones for tuple ordering.
        """
        return (
            -self.tier1_p_threshold,
            -self.tier2_p_threshold,
            self.tier1_min_n_val_dates,
            self.tier2_min_n_seeds,
            self.tier2_min_n_dates_per_regime,
            -self.tier2_max_placebo_real_ratio,
        )


def weaker_than(a: TriadPolicy, minimum: TriadPolicy) -> bool:
    """Return True iff `a` is strictly weaker than `minimum` on ANY axis."""
    a_s = a.strength_score()
    m_s = minimum.strength_score()
    return any(av < mv for av, mv in zip(a_s, m_s))


# --- ScorerSanityReport / TrainerPlaceboReport — now take policy ------------

class ScorerSanityReport(pydantic.BaseModel):
    aa_split_real_ic_replicate: FiniteFloat
    aa_split_drift_ic_abs: Annotated[FiniteFloat, Field(ge=0.0)]
    shuffled_val_ic: FiniteFloat
    timeshifted_val_ic: FiniteFloat
    label_col: str
    n_val_dates: Annotated[int, Field(gt=0)]
    shuffled_p_value_against_zero: PValue
    timeshifted_p_value_against_zero: PValue

    def fail_reasons(self, policy: TriadPolicy) -> list[str]:
        """Now POLICY-EXPLICIT; no hidden defaults."""
        out: list[str] = []
        if self.n_val_dates < policy.tier1_min_n_val_dates:
            out.append(f"n_val_dates={self.n_val_dates} < {policy.tier1_min_n_val_dates}")
        if self.shuffled_p_value_against_zero < policy.tier1_p_threshold:
            out.append(
                f"shuffled IC≠0 (IC={self.shuffled_val_ic:+.4f}, "
                f"p={self.shuffled_p_value_against_zero:.4f} < {policy.tier1_p_threshold})"
            )
        if self.timeshifted_p_value_against_zero < policy.tier1_p_threshold:
            out.append(
                f"timeshifted IC≠0 (IC={self.timeshifted_val_ic:+.4f}, "
                f"p={self.timeshifted_p_value_against_zero:.4f} < {policy.tier1_p_threshold})"
            )
        if self.aa_split_drift_ic_abs > policy.tier1_aa_drift_max:
            out.append(f"aa_split_drift={self.aa_split_drift_ic_abs:.4f} > {policy.tier1_aa_drift_max}")
        return out


class TrainerPlaceboReport(pydantic.BaseModel):
    real_ic_mean: FiniteFloat
    real_ic_per_regime: dict[str, FiniteFloat]
    real_ic_n_dates_per_regime: dict[str, Annotated[int, Field(ge=0)]]
    shuffle_placebo_ic_mean: FiniteFloat
    shuffle_placebo_ic_per_regime: dict[str, FiniteFloat]
    shuffle_placebo_p_value_against_zero: PValue
    timeshift_placebo_ic_mean: FiniteFloat
    timeshift_placebo_ic_per_regime: dict[str, FiniteFloat]
    timeshift_placebo_p_value_against_zero: PValue
    n_seeds: Annotated[int, Field(gt=0)]
    n_val_dates: Annotated[int, Field(gt=0)]

    @pydantic.model_validator(mode="after")
    def regime_keys_consistent(self) -> "TrainerPlaceboReport":
        keys = (set(self.real_ic_per_regime), set(self.real_ic_n_dates_per_regime),
                set(self.shuffle_placebo_ic_per_regime), set(self.timeshift_placebo_ic_per_regime))
        if not keys[0]:
            raise ValueError("regime maps cannot be empty")
        if not all(k == keys[0] for k in keys):
            raise ValueError(f"regime key sets must be identical: {[sorted(k) for k in keys]}")
        return self

    def fail_reasons(self, policy: TriadPolicy) -> list[str]:
        out: list[str] = []
        if self.n_seeds < policy.tier2_min_n_seeds:
            out.append(f"n_seeds={self.n_seeds} < {policy.tier2_min_n_seeds}")
        for regime, n in self.real_ic_n_dates_per_regime.items():
            if n < policy.tier2_min_n_dates_per_regime:
                out.append(f"regime {regime}: n_dates={n} < {policy.tier2_min_n_dates_per_regime}")
        if self.shuffle_placebo_p_value_against_zero < policy.tier2_p_threshold:
            out.append(f"shuffle placebo IC≠0 (p={self.shuffle_placebo_p_value_against_zero:.4f})")
        if self.timeshift_placebo_p_value_against_zero < policy.tier2_p_threshold:
            out.append(f"timeshift placebo IC≠0 (p={self.timeshift_placebo_p_value_against_zero:.4f})")
        for regime in self.real_ic_per_regime:
            r = abs(self.real_ic_per_regime[regime])
            sp = abs(self.shuffle_placebo_ic_per_regime[regime])
            tp = abs(self.timeshift_placebo_ic_per_regime[regime])
            if r > 0.01 and (sp > policy.tier2_max_placebo_real_ratio * r
                             or tp > policy.tier2_max_placebo_real_ratio * r):
                out.append(f"regime {regime}: placebo>{policy.tier2_max_placebo_real_ratio:.0%}×real "
                           f"(real={r:+.4f} sh={sp:+.4f} ts={tp:+.4f})")
        return out


# --- binding + TriadReport ---------------------------------------------------

class TriadBinding(pydantic.BaseModel):
    """Identity tuple. Any element change ⇒ new artifact, fresh pending."""
    model_config = pydantic.ConfigDict(frozen=True)
    model_sha: Annotated[str, Field(min_length=64, max_length=64)]      # sha256(model.pt bytes)
    feature_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]
    label_hash: Annotated[str, Field(min_length=64, max_length=64)]
    code_sha: Annotated[str, Field(min_length=7, max_length=40)]        # trainer git rev
    triad_config_hash: Annotated[str, Field(min_length=64, max_length=64)]  # = policy.policy_hash()

    def fingerprint(self) -> str:
        import hashlib
        h = hashlib.sha256(
            f"{self.model_sha}|{self.feature_schema_hash}|{self.label_hash}|"
            f"{self.code_sha}|{self.triad_config_hash}".encode()
        )
        return h.hexdigest()


def _derive_status(
    policy: TriadPolicy,                          # v8: explicit policy, no hidden defaults
    scorer_sanity: ScorerSanityReport,
    trainer_placebo: TrainerPlaceboReport | None,
) -> tuple[Literal["pending", "passed", "failed"], list[str]]:
    s1 = scorer_sanity.fail_reasons(policy)
    if s1:
        return "failed", s1
    if trainer_placebo is None:
        return "pending", []
    s2 = trainer_placebo.fail_reasons(policy)
    if s2:
        return "failed", s2
    return "passed", []


class TriadReport(pydantic.BaseModel):
    triad_status: Literal["pending", "passed", "failed"]
    failure_reasons: list[str]
    policy: TriadPolicy                            # v8: stored ON the report
    scorer_sanity: ScorerSanityReport
    trainer_placebo: TrainerPlaceboReport | None
    binding: TriadBinding
    triad_started_at: AwareDatetime
    triad_completed_at: AwareDatetime | None

    @pydantic.model_validator(mode="after")
    def status_and_policy_consistent(self) -> "TriadReport":
        # 1. policy.policy_hash() == binding.triad_config_hash
        if self.policy.policy_hash() != self.binding.triad_config_hash:
            raise ValueError(
                f"policy hash mismatch: policy={self.policy.policy_hash()[:16]} "
                f"vs binding.triad_config_hash={self.binding.triad_config_hash[:16]}"
            )
        # 2. status is reducer output of THIS policy
        expected_status, expected_reasons = _derive_status(
            self.policy, self.scorer_sanity, self.trainer_placebo
        )
        if self.triad_status != expected_status:
            raise ValueError(
                f"triad_status={self.triad_status!r} != reducer={expected_status!r}"
            )
        if self.failure_reasons != expected_reasons:
            raise ValueError(
                f"failure_reasons inconsistent: declared={self.failure_reasons} "
                f"expected={expected_reasons}"
            )
        # 3. completed_at consistency
        if self.triad_status == "pending":
            if self.triad_completed_at is not None:
                raise ValueError("pending must have triad_completed_at=None")
        else:
            if self.triad_completed_at is None:
                raise ValueError(f"{self.triad_status} requires triad_completed_at")
        return self

    @classmethod
    def build(cls, policy: TriadPolicy, scorer_sanity: ScorerSanityReport,
              trainer_placebo: TrainerPlaceboReport | None,
              binding: TriadBinding, triad_started_at: datetime) -> "TriadReport":
        # CRITICAL: binding MUST carry policy.policy_hash() — caller responsibility,
        # checked in status_and_policy_consistent above.
        status, reasons = _derive_status(policy, scorer_sanity, trainer_placebo)
        return cls(
            triad_status=status, failure_reasons=reasons,
            policy=policy, scorer_sanity=scorer_sanity, trainer_placebo=trainer_placebo,
            binding=binding, triad_started_at=triad_started_at,
            triad_completed_at=utc_now() if status != "pending" else None,
        )
```

**Why this addresses codex v7 #1**: `triad_config_hash` is now functionally bound — `_derive_status(policy, ...)` requires policy explicitly, the Pydantic validator verifies `policy.policy_hash() == binding.triad_config_hash`. A consumer cannot accept an artifact whose reducer ran with policy A while binding claims policy B's hash.

### 5.2 `ScorerArtifact.load()` — artifact-byte verification (v8 NEW)

```python
# renquant-common/src/renquant_common/contracts/scorer.py
import hashlib
from pathlib import Path
import pydantic
from renquant_common.contracts.triad import TriadReport


class ArtifactBytesMismatch(RuntimeError):
    """Recomputed sha256(model.pt) does not match binding.model_sha."""


class ScorerArtifact(pydantic.BaseModel):
    model_uri: str                            # path or s3:// to model.pt
    feature_cols: list[str]                   # closed list
    seq_len: int | None = None                # for sequence models
    triad_report: TriadReport                 # required (after S2 migration)
    # ... other existing fields ...

    @classmethod
    def load(cls, sidecar_path: Path) -> "ScorerArtifact":
        """The ONLY entry point that consumers use. Verifies artifact bytes."""
        artifact = cls.model_validate_json(sidecar_path.read_text())
        model_path = sidecar_path.parent / sidecar_path.stem.replace(".pt.metadata", ".pt")
        if not model_path.exists():
            raise FileNotFoundError(f"model file missing for sidecar: {model_path}")
        # Recompute bytes hash
        h = hashlib.sha256()
        with model_path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        actual = h.hexdigest()
        if actual != artifact.triad_report.binding.model_sha:
            raise ArtifactBytesMismatch(
                f"model.pt bytes hash mismatch:\n"
                f"  expected (binding.model_sha): {artifact.triad_report.binding.model_sha}\n"
                f"  actual   (recomputed):        {actual}\n"
                f"  sidecar:                      {sidecar_path}\n"
                f"  model:                        {model_path}\n"
                f"A copied sidecar cannot vouch for a different model file."
            )
        return artifact
```

**Why this addresses codex v7 #2**: every consumer goes through `ScorerArtifact.load()`. A passing sidecar copied next to a different `.pt` raises `ArtifactBytesMismatch` before any gate even runs. `model_sha` becomes a meaningful integrity check, not metadata.

### 5.3 Ed25519-signed, scope-bound bypass (v8 NEW — replaces HMAC)

```python
# renquant-common/src/renquant_common/contracts/leakage_config.py
import json, logging, os
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal
import pydantic
from pydantic import Field
import nacl.signing, nacl.exceptions, nacl.encoding
from renquant_common.contracts.triad import AwareDatetime, utc_now

log = logging.getLogger("renquant_common.leakage_guards.bypass")

GateCaller = Literal[
    "pipeline:scorer_load",
    "backtesting:manifest_row",
    "backtesting:sim_driver",
    "backtesting:fit_calibrator",
    "orchestrator:manifest_row",
    "execution:submit_order",
]

# Public verify keys live in repo (committed); private signing key is offline-only.
ARCHITECT_PUBKEYS_PATH = Path(__file__).parent.parent / "keys" / "architect_pubkeys.json"


def _load_architect_pubkeys() -> dict[str, nacl.signing.VerifyKey]:
    """Read renquant-common/keys/architect_pubkeys.json:

    {
      "v1": "base64-encoded-32-byte-ed25519-pubkey",
      "v2": "...",
      "...": "..."
    }

    `key_id` in bypass entries selects which verify key. Old keys can be
    removed by deleting their entry. Compromised pubkey ⇒ delete key entry
    ⇒ all bypasses signed by that key reject immediately.
    """
    if not ARCHITECT_PUBKEYS_PATH.exists():
        return {}
    raw = json.loads(ARCHITECT_PUBKEYS_PATH.read_text())
    return {kid: nacl.signing.VerifyKey(b64, encoder=nacl.encoding.Base64Encoder)
            for kid, b64 in raw.items()}


def canonical_payload(unsigned_entry: dict) -> bytes:
    """SINGLE canonical-serialization function used by both signing AND verification.

    Rules (PINNED by golden-vector test):
      - JSON, sort_keys=True
      - separators=(",", ":")  (no spaces)
      - datetimes formatted as 'YYYY-MM-DDTHH:MM:SS.ffffffZ' explicitly
        (NOT default=str, NOT pydantic JSON-mode iso strings)
      - lists in declared order (allowed_gates etc.)
      - 'signature' field MUST be absent
      - 'key_id' MUST be present
      - no NaN/Inf values
    """
    payload = {k: v for k, v in unsigned_entry.items() if k != "signature"}

    def encode(obj):
        if isinstance(obj, datetime):
            if obj.tzinfo is None:
                raise ValueError(f"refusing to canonicalize naive datetime {obj!r}")
            obj_utc = obj.astimezone(timezone.utc)
            # Fixed format with microseconds, trailing Z
            return obj_utc.strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"
        if isinstance(obj, list):
            return [encode(x) for x in obj]
        if isinstance(obj, dict):
            return {k: encode(v) for k, v in obj.items()}
        if isinstance(obj, float):
            if not math.isfinite(obj):
                raise ValueError(f"non-finite float in payload: {obj}")
        return obj

    return json.dumps(encode(payload), sort_keys=True, separators=(",", ":")).encode()


class TriadBypassEntryUnsigned(pydantic.BaseModel):
    """Payload that gets signed. Same shape as TriadBypassEntry minus signature."""
    model_config = pydantic.ConfigDict(frozen=True)

    artifact_fingerprint: Annotated[str, Field(min_length=64, max_length=64)]
    allowed_gates: Annotated[list[GateCaller], Field(min_length=1)]   # v8 NEW: scope binding
    expires_at: AwareDatetime
    reason: str
    approved_by: str                              # GitHub handle (informational)
    pr_url: str
    approved_at: AwareDatetime
    key_id: str                                   # which architect pubkey signed

    @pydantic.field_validator("reason")
    @classmethod
    def reason_nontrivial(cls, v: str) -> str:
        if len(v.strip()) < 20:
            raise ValueError(f"reason too short ({len(v.strip())} chars); need ≥20")
        return v


class TriadBypassEntry(TriadBypassEntryUnsigned):
    signature: str                                # base64(Ed25519 sig of canonical_payload(unsigned))

    def verify(self, pubkeys: dict[str, nacl.signing.VerifyKey]) -> bool:
        """True iff Ed25519 signature is valid for an enrolled key_id."""
        vk = pubkeys.get(self.key_id)
        if vk is None:
            log.error("bypass key_id=%r not enrolled; rejecting", self.key_id)
            return False
        # Re-canonicalize the unsigned portion
        unsigned_dump = TriadBypassEntryUnsigned.model_validate(
            self.model_dump(exclude={"signature"})
        ).model_dump()
        canonical = canonical_payload(unsigned_dump)
        try:
            sig_bytes = nacl.encoding.Base64Encoder.decode(self.signature.encode())
            vk.verify(canonical, sig_bytes)
            return True
        except nacl.exceptions.BadSignatureError:
            log.error("bypass for fp=%s key_id=%s has BAD SIGNATURE; rejecting",
                      self.artifact_fingerprint[:16], self.key_id)
            return False


def sign_bypass_entry_offline(unsigned: TriadBypassEntryUnsigned,
                              signing_key: nacl.signing.SigningKey) -> str:
    """Architect-side helper. NEVER runs in CI. NEVER imports prod env."""
    canonical = canonical_payload(unsigned.model_dump())
    sig = signing_key.sign(canonical).signature
    return nacl.encoding.Base64Encoder.encode(sig).decode()


class LeakageGuardConfig(pydantic.BaseModel):
    minimum_acceptable_policy: TriadPolicy        # v8 NEW: consumer rejects weaker
    triad_bypasses_raw: list[TriadBypassEntry] = Field(default_factory=list)
    alert_channel: Literal["slack", "log", "none"] = "log"
    alert_slack_webhook_url: str | None = None

    @pydantic.computed_field
    @property
    def triad_bypasses(self) -> list[TriadBypassEntry]:
        """Verified entries only. Invalid signatures / unknown key_ids dropped with WARN."""
        pubkeys = _load_architect_pubkeys()
        if not pubkeys:
            if self.triad_bypasses_raw:
                log.error("no architect pubkeys enrolled; all %d bypasses treated as invalid",
                          len(self.triad_bypasses_raw))
            return []
        return [b for b in self.triad_bypasses_raw if b.verify(pubkeys)]
```

```python
# renquant-common/src/renquant_common/leakage_guards/gate.py
def assert_artifact_validated(
    artifact: ScorerArtifact, *,
    cfg: LeakageGuardConfig,
    caller: GateCaller,
) -> None:
    fp = artifact.triad_report.binding.fingerprint()
    s = artifact.triad_report.triad_status

    # v8: policy strength gate — refuse artifacts whose policy is weaker than minimum acceptable
    if weaker_than(artifact.triad_report.policy, cfg.minimum_acceptable_policy):
        telemetry.emit_event("gate_block_policy_weak", caller=caller, artifact_fingerprint=fp,
                             artifact_policy_hash=artifact.triad_report.policy.policy_hash(),
                             minimum_policy_hash=cfg.minimum_acceptable_policy.policy_hash())
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} whose triad policy is weaker than minimum.\n"
            f"  artifact policy hash: {artifact.triad_report.policy.policy_hash()[:16]}\n"
            f"  required minimum:     {cfg.minimum_acceptable_policy.policy_hash()[:16]}"
        )

    if s == "failed":
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="failed",
                             failure_reasons=artifact.triad_report.failure_reasons)
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} status=failed. "
            f"Reasons: {artifact.triad_report.failure_reasons}. No bypass for failed."
        )

    if s == "passed":
        telemetry.emit_event("gate_allow", caller=caller, artifact_fingerprint=fp,
                             triad_status="passed")
        return

    # pending — verified+scoped bypass match
    now = utc_now()
    matching = [
        b for b in cfg.triad_bypasses
        if b.artifact_fingerprint == fp
        and b.expires_at > now
        and caller in b.allowed_gates                # v8: per-gate scope check
    ]
    if not matching:
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="pending",
                             reason="no verified+scoped bypass match",
                             total_verified_bypasses=len(cfg.triad_bypasses))
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} status=pending; "
            f"no verified bypass entry matches both fingerprint AND allowed_gates=...{caller}... "
            f"(verified entries: {len(cfg.triad_bypasses)})"
        )
    entry = matching[0]
    telemetry.emit_event("gate_bypass", caller=caller, artifact_fingerprint=fp,
                         triad_status="pending",
                         bypass_expires_at=entry.expires_at.isoformat(),
                         bypass_approved_by=entry.approved_by,
                         bypass_pr_url=entry.pr_url,
                         bypass_key_id=entry.key_id)
    log.warning(
        "TRIAD BYPASS: %s fp=%s expires=%s key_id=%s approved_by=%s pr=%s allowed_gates=%s",
        caller, fp[:16], entry.expires_at.isoformat(), entry.key_id,
        entry.approved_by, entry.pr_url, entry.allowed_gates,
    )
```

**Why this addresses codex v7 #3, #4, #5**:
- **#3** Ed25519 asymmetric: verify keys public (committed `architect_pubkeys.json`), signing key offline. Compromise of any verifier cannot mint bypasses. `key_id` supports rotation; remove key entry to revoke.
- **#4** `allowed_gates` is in the signed payload. A backfill bypass with `allowed_gates=["pipeline:scorer_load"]` is REJECTED by `execution:submit_order` even with valid signature.
- **#5** Canonical serialization is a single function `canonical_payload()`; sign and verify both call it. UTC datetimes use fixed `strftime` (not `default=str`, not Pydantic JSON mode). Golden-vector test in `tests/test_canonical_vector.py` (next subsection).

### 5.4 Canonical serialization — golden vector test (v8 NEW)

```python
# renquant-common/tests/test_canonical_vector.py
"""GOLDEN VECTOR. Any change to canonical_payload() that breaks this test
is a breaking semver change and invalidates all existing signed bypasses."""

from datetime import datetime, timezone
from renquant_common.contracts.leakage_config import canonical_payload

def test_golden_canonical_payload():
    unsigned = {
        "artifact_fingerprint": "a" * 64,
        "allowed_gates": ["pipeline:scorer_load", "backtesting:manifest_row"],
        "expires_at": datetime(2026, 6, 15, 12, 0, 0, 123456, tzinfo=timezone.utc),
        "reason": "B_tuned backfill grace period 7 days",
        "approved_by": "architect-user",
        "pr_url": "https://github.com/hallovorld/RenQuant/pull/999",
        "approved_at": datetime(2026, 6, 1, 20, 30, 0, 0, tzinfo=timezone.utc),
        "key_id": "v1",
    }
    expected = (
        b'{"allowed_gates":["pipeline:scorer_load","backtesting:manifest_row"],'
        b'"approved_at":"2026-06-01T20:30:00.000000Z",'
        b'"approved_by":"architect-user",'
        b'"artifact_fingerprint":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",'
        b'"expires_at":"2026-06-15T12:00:00.123456Z",'
        b'"key_id":"v1",'
        b'"pr_url":"https://github.com/hallovorld/RenQuant/pull/999",'
        b'"reason":"B_tuned backfill grace period 7 days"}'
    )
    assert canonical_payload(unsigned) == expected
```

### 5.5 Statistical helpers (unchanged from v7 — direction-correct)

`bootstrap_p_value_against_zero(perturbed_ic)` — H0: mean = 0; small p ⇒ leak. `bootstrap_p_value_real_dominates` diagnostic-only. 3 fixtures (good/leak/noise). All carried forward from v7 §5.4.

### 5.6 CAS — unchanged from v7

`atomic_cas_update_sidecar(sidecar_path, expected_binding, expected_current_status, transformer)` with `_VALID_TRANSITIONS = {(pending,pending), (pending,passed), (pending,failed)}`. Uses `os.replace`. POSIX-only.

## 6 · State machine (unchanged from v7) — two-layer enforcement

Pydantic reducer + CAS transitions. `passed`/`failed` terminal at BOTH layers.

## 7 · Five gates + Tier-2 runner abstraction (unchanged from v7)

G0-G5; `Tier2RunnerProtocol` with `Production` (real, 75min) and `Synthetic` (in-memory, <1s) implementations. E2E uses synthetic with 3 fixtures.

## 8 · Migration — 4 stages (S0-S3) — declared MAJOR breaking

Plus v8 addition: `minimum_acceptable_policy` upgrade is a separate change-management process — bumping it stricter requires re-running Tier-2 for all currently-passed artifacts under the new policy, since their `triad_config_hash` will no longer match.

## 9 · Falsification

Falsifiers from v7 carry forward (1-8). New v8 falsifiers:

9. Artifact whose `binding.model_sha` doesn't match recomputed `sha256(model.pt)` was loaded by any consumer.
10. Artifact whose `policy.policy_hash() != binding.triad_config_hash` was accepted.
11. Bypass signed for `allowed_gates=["pipeline:scorer_load"]` authorized `execution:submit_order`.
12. Bypass with `key_id` not in `architect_pubkeys.json` was accepted.
13. Signing or verification function reads `triad_bypass_entry` and produces different canonical bytes than the golden-vector test.
14. Consumer with `minimum_acceptable_policy` stricter accepted artifact whose policy is weaker on any axis.

## 10 · Threat model

12 leak classes (L1-L12) carried forward. v8 adds attack-surface table:

| Attack | Mitigation in v8 |
|---|---|
| Copy sidecar+artifact pair to point at different model.pt | §5.2 `ScorerArtifact.load()` recomputes sha256(model.pt), raises `ArtifactBytesMismatch` |
| Tamper with `triad_config_hash` to claim stricter policy | Pydantic validator: `policy.policy_hash() == binding.triad_config_hash` |
| Run reducer with weakened policy (lower threshold, fewer seeds) | `policy_hash` derived from policy; mismatch detected |
| Mint a fake bypass on a compromised verifier | Asymmetric Ed25519 — verifier has only public key |
| Reuse a backfill bypass for live trading | `allowed_gates` in signed payload; gate checks `caller in allowed_gates` |
| Pre-validation signing / mid-spec drift | Single `canonical_payload()` function + golden vector test |
| Replay bypass after expiry | `expires_at` in signed payload; gate compares to `utc_now()` |
| Race two Tier-2 runners | CAS on `(binding, current_status)`; second runner gets `StaleBindingError` |
| Force `failed → passed` by sidecar rewrite | `_VALID_TRANSITIONS` table at CAS layer rejects |

## 11 · Codex v7 → v8 — addressed

| # | Sev | Finding | Resolution |
|---|---|---|---|
| 1 | HIGH | `triad_config_hash` decorative; reducer used hardcoded defaults | §5.1 `TriadPolicy` Pydantic + `policy.policy_hash()` + `_derive_status(policy, ...)` + Pydantic validator `policy.policy_hash() == binding.triad_config_hash` |
| 2 | HIGH | `binding.model_sha` trusted from metadata; no byte verification | §5.2 `ScorerArtifact.load()` recomputes `sha256(model.pt)` and raises `ArtifactBytesMismatch` if not equal to `binding.model_sha` |
| 3 | HIGH | HMAC means verifier == signer ⇒ blast-radius / no rotation | §5.3 Ed25519 asymmetric. Public keys in repo `keys/architect_pubkeys.json` keyed by `key_id`. Signing key offline. Rotation = remove key entry. |
| 4 | MED | Bypass scope too broad (backfill bypass also auths G5) | §5.3 `allowed_gates: list[GateCaller]` in SIGNED payload; gate refuses if `caller` not in allowed set |
| 5 | MED | Canonicalization underspecified (datetime format, ordering, default=str) | §5.3 single `canonical_payload()` for both sign + verify; explicit `strftime` for tz-aware UTC; §5.4 golden vector test pinned in repo; `TriadBypassEntryUnsigned` validated before signing |

Plus v8 self-added (preempting next round):
- **policy strength gate**: consumer can require `minimum_acceptable_policy`; an artifact with weaker thresholds is rejected. Prevents an "old artifact with lax thresholds" attack.

## 12 · Open questions

1. `nacl.signing` (libsodium-based) is the recommended Ed25519 lib; pure-stdlib Ed25519 (`cryptography` package) also acceptable — does CI/uv lock have either pinned?
2. Key rotation cadence (architect): 90 days? Stamp on each retrain?
3. After Tier-2 fail: auto-retrain (selection bias) vs hold (latency)?
4. Strategy threshold changes in binding-tuple? — answer: NO, they're consumed downstream, not in model training.

## 13 · MVP PR list (5 PRs ≤ 2 days)

| # | Repo | Files | Tests |
|---|---|---|---|
| ① | `renquant-common` | `contracts/triad.py` (incl. `TriadPolicy`), `contracts/scorer.py` (incl. `load()` byte-verify), `contracts/leakage_config.py` (Ed25519 + scope), `leakage_guards/*`, `keys/architect_pubkeys.json` (committed), `tools/sign_bypass.py` (offline architect-only), `tests/test_canonical_vector.py` (golden vector), `tests/test_falsification.py` (all 14 falsifiers), `tests/fixtures/triad_models.py` (3 fixtures), `.github/workflows/gate-disable-detection.yml`, `.github/CODEOWNERS` for architect-only files | unit + race + state-machine + 3-fixture + HMAC→Ed25519 round-trip + golden vector + tz-aware compare + `python -O` regression + scope-rejection + byte-mismatch reject + policy-hash mismatch reject |
| ② | `renquant-pipeline` | `kernel/panel_pipeline/panel_scorer.py::load` route through `ScorerArtifact.load()` | gate-behavior matrix incl. scope-bypass / byte-mismatch / policy-weaker |
| ③ | `renquant-model-patchtst` | `hf_trainer.py::_save_artifact` (Tier 1 + policy), new `post_save_hook.py` (Tier 2 enqueue) | synth Tier-1 + Tier-2 enqueue + synthetic runner E2E |
| ④ | `renquant-model-gbdt` | mirror of ③ | mirror |
| ⑤ | `renquant-orchestrator` + `renquant-backtesting` (paired) | `manifest_row` + `wf_gate/runner.py` + `sim_driver.py` + `scripts/fit_walkforward_calibrators.py` | manifest gate behavior matrix incl. scope binding |

Full architecture wave (split parquet, typed train sig, ban `pd.read_parquet`) deferred to weeks 2-3.

## 14 · References

- Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 5, 7.
- Pesaran, M.H., Timmermann, A. (2007). "Selection of estimation window." *J. Econometrics* 137.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Bernstein, D.J., Duif, N., Lange, T., Schwabe, P., Yang, B-Y. (2012). "High-speed high-security signatures." *J. Cryptographic Engineering* 2:77.  (Ed25519)
- NIST SP 800-186 — Recommendations for Discrete Logarithm-based Cryptography (Ed25519 inclusion 2023).

---

## 15 · v7 → v8 changelog (codex v7 resolution + self-preempted next-round)

| Element | v7 | v8 |
|---|---|---|
| Policy binding | `triad_config_hash` field on binding; reducer used hardcoded defaults | `TriadPolicy` Pydantic; `binding.triad_config_hash = policy.policy_hash()`; Pydantic validator enforces consistency; `_derive_status(policy, ...)` |
| Artifact bytes | trusted from `binding.model_sha` metadata | `ScorerArtifact.load()` recomputes `sha256(model.pt)`, raises `ArtifactBytesMismatch` |
| Bypass crypto | HMAC-SHA256 (verifier = signer = blast radius) | Ed25519 asymmetric; pubkeys committed `keys/architect_pubkeys.json`; signing offline; `key_id` rotation |
| Bypass scope | global per fingerprint | `allowed_gates: list[GateCaller]` in signed payload; gate checks `caller in allowed_gates` |
| Canonicalization | `model_dump(mode="json")` + `default=str` underspecified | single `canonical_payload()` with explicit `strftime("%Y-%m-%dT%H:%M:%S.%fZ")`; `TriadBypassEntryUnsigned` validated before signing; golden vector test |
| Policy strength gate (v8 self-added) | none | `LeakageGuardConfig.minimum_acceptable_policy`; consumer refuses artifacts whose `weaker_than(artifact.policy, minimum)` |
| Falsifiers | 8 (v7 §9) | 14 (added byte-hash, policy-hash, scope binding, key_id enrollment, golden vector, policy-strength) |
| Attack-surface table | not enumerated | §10 explicit attack → mitigation table |
