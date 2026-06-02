# RenQuant Multirepo Leakage Defense — Architecture (v12)

**Status**: Design under review (RenQuant PR #43; supersedes #38 which merged at v8)
**Authors**: Claude
**Reviewers**: Codex (10 review rounds). v11 found 1 HIGH + 3 MED + 1 LOW; v12 addresses all five in the canonical sections:
  - **cache-poisoning** (codex v11 #1) — `load_artifact()` always wins via atomic `os.replace()`; pre-existing `sha256-<x>.bin` is overwritten by the freshly verified temp file. `_enforce_cache_dir()` refuses symlink / non-owner / mode > 0o700 cache dirs; settled entries get `chmod 0o400`. New falsifier #21.
  - **duplicate `VerifiedArtifact` class** (codex v11 #2) — §10A.2 collapsed to a single definition placed before `load_artifact()`. CI grep + AST scan reject duplicates. New falsifier #22.
  - **`extra="forbid"` on versioned-hash models** (codex v11 #3) — `TargetSpecManifestV1`, `TriadPolicyV1/V2`, `FeatureSchemaManifest`, `TargetSpecManifestV2` all set `ConfigDict(frozen=True, extra="forbid")`. Unknown fields raise — semantic adds force a schema-version bump. New falsifier #23.
  - **MVP rows pointing at deprecated API** (codex v11 #4) — §13 row ① now lists `load_artifact() -> VerifiedArtifact` + `parse_policy` + `parse_target_spec` + `FeatureSchemaManifest` and "all 20 falsifiers"; row ② routes panel_scorer through `load_artifact()`, AST scan rejects `artifact.model_uri` reopens.
  - **v8-era threat-model wording** (codex v11 #5) — §10 rewritten to v11 attack-surface table; rows for cache poisoning + target-code drift + feature-schema bypass added.
**Falsifier count**: **23** (v7: 1-8, v8: 9-14, v10: 15-20, v12: 21-23).
**Supersedes**: v1-v11 (commit history is the audit trail)
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
11. Consumer with `minimum_acceptable_policy` stricter than artifact's policy rejects the artifact, regardless of `triad_status`. Direction-correct per-field: larger `p_threshold` ⇒ stricter, larger `min_n` ⇒ stricter, smaller `max_placebo_real_ratio` ⇒ stricter, smaller `aa_drift_max` ⇒ stricter, larger `bootstrap_n_iters` ⇒ stricter.
12. E2E nightly runs 3 fixtures (good/leak/noise) × synthetic Tier-2 under 5 min wall clock.
13. **Runtime binding (v10)**: `assert_artifact_validated()` takes a runtime binding from the call site, split by call context — `RuntimeBindingInference(feature_schema_hash, target_spec_hash)` for G3/G5 (pre-outcome — no realized labels), `RuntimeBindingTraining(feature_schema_hash, target_spec_hash, label_bytes_hash)` for G4. Gate dispatches on input type and rejects if any field ≠ artifact `binding.<field>`. A passed model inserted into a manifest using different features or different target definition is rejected.
14. **Bypass environment scope** (v9): `allowed_environments: list[Env]` is part of the signed payload; staging bypass cannot authorize prod.
15. **Artifact store + verified bytes (v10)**: `load_artifact(sidecar_path, store=ArtifactStore) -> VerifiedArtifact`. The verified bytes are hashed-while-streaming and returned ON the handle. Consumer (`PanelScorer.load()` / `torch.load`) MUST use `verified.open_bytes()` or `verified.spooled_path()` — never re-open `model_uri`. Eliminates the v9 TOCTOU between hash check and file load.

**Any false ⇒ design failed ⇒ revert.**

## 3 · Designs considered & rejected (unchanged from v7)

A advisory / B annotations alone / C single chokepoint / D disable model — all rejected.
E (this design): multi-gate + policy-bound derived status + Ed25519-signed scoped bypass + artifact-byte verification + CAS-with-transition.

## 4 · Invariants

- **I1 (Artifact contract)**: `ScorerArtifact` carries `TriadReport` with `triad_status` deterministically derived from `policy + scorer_sanity + trainer_placebo`.
- **I2 (Terminal failure)**: passed/failed terminal at BOTH Pydantic (reducer) AND CAS layer (`_VALID_TRANSITIONS`).
- **I3 (Gate symmetry)**: G3/G4/G5 share one helper.
- **I4 (Artifact-byte integrity, v10)**: `load_artifact(sidecar_path, store) -> VerifiedArtifact` hashes the bytes streamed from `model_uri` once and returns a handle holding either the verified bytes (small artifacts) or a verified spooled temp-file path / mmap (large artifacts >1 GB). Consumer constructs the scorer via `verified.open_bytes()` / `verified.spooled_path()` — **never** re-opens `model_uri` itself. Sidecar TOCTOU is eliminated.
- **I5 (Policy-hash binding, v10)**: `binding.triad_config_hash = canonical_sha256(parse_policy(payload))`. The reducer takes a versioned policy explicitly via `parse_policy(payload) -> TriadPolicyV{N}`; consumers reject artifacts whose `policy_schema_version` isn't in their registry, AND whose policy is `weaker_than(minimum_acceptable_policy)`. Old artifacts continue to parse against the version they were stamped with; future schema bumps cannot silently rehash old payloads.
- **I6 (Asymmetric signed bypass)**: `TriadBypassEntry.signature` is Ed25519 over the canonical bytes. Verify keys live in repo at `keys/architect_pubkeys.json`; signing key lives offline. Compromise of a verifier cannot mint bypasses. `key_id` field supports rotation.
- **I7 (Per-gate bypass scope)**: bypass `allowed_gates` is part of signed payload; gate refuses if `caller` not in that set. A backfill bypass cannot authorize a live order.
- **I8 (Canonical serialization)**: signing and verification both use `canonical_payload()` defined once. UTC datetimes serialized as `YYYY-MM-DDTHH:MM:SS.ffffffZ`. No `default=str`. Golden vector test pinned in repo.
- **I9 (CAS + transition)**: sidecar update CAS-bound to `(binding, current_status)`; transitions table; only `pending → {pending, passed, failed}` permitted.
- **I10 (Statistical, n-aware, direction-correct)**: bootstrap p-values with explicit H0/H1, gate-aligned (small p ⇒ leak ⇒ fail).
- **I11 (Tz-aware datetimes)**: `AwareDatetime`; naive rejected.
- **I12 (Finite floats + non-empty regime maps)**: `FiniteFloat`, `PValue`, `regime_keys_consistent`.
- **I13 (Disable detection)**: CI AST scan + regex on `leakage_guards/`.
- **I14 (Migration breaking)**: declared MAJOR semver; 4-stage S0-S3.
- **I15 (Versioned TargetSpec, v10)**: `target_spec_hash = canonical_sha256(TargetSpecManifest)` where `TargetSpecManifest` carries `target_spec_schema_version`, `label_col`, `lookahead_days`, `return_type`, `benchmark_id`, `calendar_id`, `universe_filter_id`, `target_transform_code_version`, `embargo_policy_id`, `manifest_id`. Golden vector test pinned. Same six surface-level fields under a different target-transform-code-version produce a different hash; consumers reject unknown `target_spec_schema_version`.
- **I16 (FeatureSchemaManifest closed-keys, v10)**: `feature_schema_hash = canonical_sha256(FeatureSchemaManifest)`. Validator enforces (a) `set(feature_cols) == set(feature_dtypes.keys()) == set(feature_lookahead_days.keys())` — closed-key equality, (b) `feature_cols` has no duplicates AND is sorted (canonical form), (c) every `feature_lookahead_days[c]` is exactly `0` (positive AND negative both rejected), (d) no overlap with declared `LabelManifest.label_cols`. Same columns produced by a different `feature_transform_code_version` hash differently.
- **I17 (Memory-safe verified handle, v10)**: `load_artifact()` materializes verified bytes via a strategy chosen by `max_in_memory_bytes` config: ≤ threshold uses in-memory `BytesIO`; > threshold uses a spooled tempfile (`tempfile.NamedTemporaryFile(delete=False)`) with `os.chmod(0o600)` and the SHA verified before rename into a content-addressed cache dir. Consumer reads via `verified.spooled_path()` instead of `open_bytes()`. No path that the scorer touches is ever the user-supplied `model_uri`.

## 5 · Contract code

> **🛑 v10 reader notice — read §5 with §10A applied**
>
> The code blocks in §5.1 / §5.2 / §5.3 below are **v9 history snapshots
> kept for diff continuity**. v10 supersedes them with the contract in
> §10A:
>
> - §5.1 single-class `TriadPolicy` → §10A.3 `TriadPolicyV1 | TriadPolicyV2`
>   + `parse_policy(payload)` dispatcher. Use the versioned form.
> - §5.2 `RuntimeDataBinding(feature_schema_hash, label_hash)` →
>   §10A.1 `RuntimeBindingInference` + `RuntimeBindingTraining`. G3/G5
>   take the inference form; G4 takes the training form.
> - §5.2 `ScorerArtifact.load()` returning metadata-only →
>   §10A.2 `load_artifact() -> VerifiedArtifact` returning hashed bytes
>   (or a spooled-tempfile path for large artifacts).
> - §5.3 `FeatureManifest` / `feature_schema_hash` plain string →
>   §10A.4 `FeatureSchemaManifest` with closed-key validation + golden
>   vector test.
>
> Implementation owners: copy contracts from §10A, not the historical
> blocks below. The next doc rev (v11+) will collapse §5 + §10A into a
> single canonical §5 once the v10 contract has been independently
> reviewed in code form.

### 5.1 `TriadPolicy` — frozen, hash-bound to artifact, version-stamped

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


# --- TriadPolicy: frozen, hash-bound, version-stamped ------------------------

class TriadPolicy(pydantic.BaseModel):
    """Frozen policy bound to artifact via `policy_hash()`.

    DIRECTION SEMANTICS (v9 fix to v8 bug):
      The gate FAILS when `placebo_p_value < tier_p_threshold`.
      So LARGER threshold ⇒ MORE artifacts fail ⇒ STRICTER.
      v8's `strength_score()` had `-tier_p_threshold`, treating 0.01 as
      stricter than 0.05 — backwards. Fixed in `weaker_than()` below
      with per-field explicit direction; `strength_score()` removed.

    POLICY VERSIONING (v9 NEW per codex v8 #4):
      `policy_schema_version` bumps on adding/removing fields (this
      class's structural change). `stats_algorithm_version` bumps on
      changing the math (e.g., bootstrap → block bootstrap, p
      computation direction). Both are in `policy_hash()` so old
      artifacts keep their hash AS-WRITTEN; intentional migration is
      explicit.
    """
    model_config = pydantic.ConfigDict(frozen=True)

    # Version stamps (v9 NEW)
    policy_schema_version: Annotated[int, Field(gt=0)] = 1
    stats_algorithm_version: Annotated[int, Field(gt=0)] = 1

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
        """Canonical SHA-256. The ONLY way to compute binding.triad_config_hash.

        Old artifacts whose hash was computed before a new field was added
        are NOT silently rehashable: their stored hash matches their stored
        policy_schema_version=N record; consumers refuse mismatched-version
        policies via `weaker_than` (any newer-version policy is "different",
        and the per-field strictness comparison enforces direction).
        """
        import hashlib, json
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def weaker_than(a: TriadPolicy, minimum: TriadPolicy) -> bool:
    """True iff `a` is strictly weaker (LESS strict, lets MORE artifacts pass)
    than `minimum` on ANY axis. Per-field correct direction (v9 fix).

    DIRECTION TABLE (gate fails when placebo_p < threshold):
       p_threshold:                LARGER ⇒ stricter (blocks more marginal placebos)
       min_n_val_dates:            LARGER ⇒ stricter (more evidence required)
       min_n_seeds:                LARGER ⇒ stricter
       min_n_dates_per_regime:     LARGER ⇒ stricter
       aa_drift_max:               SMALLER ⇒ stricter (tighter drift tolerance)
       max_placebo_real_ratio:     SMALLER ⇒ stricter (tighter ratio tolerance)
       bootstrap_n_iters:          LARGER ⇒ stricter (smaller MC error in p)

    Also requires identical version stamps; differing algorithm versions
    cannot be ordered along axes, so they fail the comparison.
    """
    if a.policy_schema_version != minimum.policy_schema_version:
        return True   # different schema versions: artifact policy is "different", treat as weaker
    if a.stats_algorithm_version != minimum.stats_algorithm_version:
        return True
    return (
        a.tier1_p_threshold < minimum.tier1_p_threshold
        or a.tier2_p_threshold < minimum.tier2_p_threshold
        or a.tier1_min_n_val_dates < minimum.tier1_min_n_val_dates
        or a.tier2_min_n_seeds < minimum.tier2_min_n_seeds
        or a.tier2_min_n_dates_per_regime < minimum.tier2_min_n_dates_per_regime
        or a.tier1_aa_drift_max > minimum.tier1_aa_drift_max
        or a.tier2_max_placebo_real_ratio > minimum.tier2_max_placebo_real_ratio
        or a.bootstrap_n_iters < minimum.bootstrap_n_iters
    )


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
    policy: TriadPolicy,                          # explicit policy, no hidden defaults
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
    policy: TriadPolicy                            # stored ON the report; hash-bound to binding
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

### 5.2 `ScorerArtifact.load()` — artifact-byte verification through ArtifactStore (v9)

```python
# renquant-common/src/renquant_common/contracts/scorer.py
import hashlib
from pathlib import Path
from typing import Iterator, Protocol
import pydantic
from urllib.parse import urlparse
from renquant_common.contracts.triad import TriadReport


class ArtifactBytesMismatch(RuntimeError):
    """Recomputed sha256(model bytes) does not match binding.model_sha."""


class UnsupportedArtifactScheme(RuntimeError):
    """ArtifactStore does not implement the URI scheme."""


class ArtifactStore(Protocol):
    """Resolves `model_uri` and provides an authoritative content hash.

    Implementations must verify object identity at fetch time (e.g., S3 versionId
    in the URI; ETag preconditions on read; local filesystem stat).
    """
    def supports(self, uri: str) -> bool: ...
    def stream_bytes(self, uri: str) -> Iterator[bytes]: ...
    def authoritative_sha256(self, uri: str) -> str: ...
    """Either object-store-attested (e.g., S3 GetObjectAttributes returns
    Checksum) OR streamed-and-hashed locally. MUST be the same answer either
    way; consumers cannot trust a sha computed elsewhere without revalidation."""


class LocalFileArtifactStore:
    """For `file://` and bare local paths."""
    def supports(self, uri: str) -> bool:
        p = urlparse(uri)
        return p.scheme in ("", "file")

    def _to_path(self, uri: str) -> Path:
        p = urlparse(uri)
        return Path(p.path) if p.scheme in ("", "file") else Path(uri)

    def stream_bytes(self, uri: str) -> Iterator[bytes]:
        path = self._to_path(uri)
        if not path.exists():
            raise FileNotFoundError(f"local artifact missing: {path}")
        with path.open("rb") as f:
            yield from iter(lambda: f.read(65536), b"")

    def authoritative_sha256(self, uri: str) -> str:
        h = hashlib.sha256()
        for chunk in self.stream_bytes(uri):
            h.update(chunk)
        return h.hexdigest()


class S3VersionedArtifactStore:
    """`s3://bucket/key?versionId=...` REQUIRES versionId for integrity binding.

    Object overwrite without versionId ⇒ ambiguous; rejected.
    """
    def __init__(self, s3_client):
        self._s3 = s3_client

    def supports(self, uri: str) -> bool:
        return urlparse(uri).scheme == "s3"

    def _parse(self, uri: str) -> tuple[str, str, str]:
        p = urlparse(uri)
        bucket = p.netloc
        key = p.path.lstrip("/")
        from urllib.parse import parse_qs
        qs = parse_qs(p.query)
        if "versionId" not in qs:
            raise ValueError(
                f"S3 artifact URI MUST include versionId for integrity binding: {uri!r}"
            )
        return bucket, key, qs["versionId"][0]

    def stream_bytes(self, uri: str) -> Iterator[bytes]:
        bucket, key, vid = self._parse(uri)
        obj = self._s3.get_object(Bucket=bucket, Key=key, VersionId=vid)
        body = obj["Body"]
        try:
            for chunk in body.iter_chunks(chunk_size=65536):
                yield chunk
        finally:
            body.close()

    def authoritative_sha256(self, uri: str) -> str:
        bucket, key, vid = self._parse(uri)
        # Prefer ChecksumSHA256 attribute when bucket has checksums enabled
        attrs = self._s3.get_object_attributes(
            Bucket=bucket, Key=key, VersionId=vid,
            ObjectAttributes=["Checksum"],
        )
        cks = attrs.get("Checksum", {}).get("ChecksumSHA256")
        if cks:
            import base64
            return base64.b64decode(cks).hex()
        # Fallback: stream and hash
        h = hashlib.sha256()
        for chunk in self.stream_bytes(uri):
            h.update(chunk)
        return h.hexdigest()


def make_default_store() -> ArtifactStore:
    """Composite store. MVP: local-only. S3 added when execution lift lands."""
    return LocalFileArtifactStore()


class ScorerArtifact(pydantic.BaseModel):
    model_uri: str                            # local path / file:// / s3://bucket/key?versionId=...
    feature_cols: list[str]
    seq_len: int | None = None
    triad_report: TriadReport
    # ... other existing fields ...

    @classmethod
    def load(cls, sidecar_path: Path, *, store: ArtifactStore | None = None) -> "ScorerArtifact":
        """The ONLY entry point that consumers use.

        Uses `model_uri` (NOT derived from sidecar filename — v9 fix to v8 #3).
        Resolves through the supplied `store` so local+remote paths share the
        same integrity invariant.
        """
        store = store or make_default_store()
        artifact = cls.model_validate_json(sidecar_path.read_text())
        if not store.supports(artifact.model_uri):
            raise UnsupportedArtifactScheme(
                f"no store supports {artifact.model_uri!r}; "
                f"available: {type(store).__name__}"
            )
        actual = store.authoritative_sha256(artifact.model_uri)
        if actual != artifact.triad_report.binding.model_sha:
            raise ArtifactBytesMismatch(
                f"model bytes hash mismatch:\n"
                f"  expected (binding.model_sha): {artifact.triad_report.binding.model_sha}\n"
                f"  actual   (store-authoritative): {actual}\n"
                f"  sidecar:                       {sidecar_path}\n"
                f"  model_uri:                     {artifact.model_uri}\n"
            )
        return artifact
```

```python
# renquant-common/src/renquant_common/contracts/runtime_binding.py  (v9 NEW)

class RuntimeDataBinding(pydantic.BaseModel):
    """Computed at the GATE CALL SITE from the data/manifest the consumer is
    actually about to use. Passed into assert_artifact_validated().

    The gate verifies:
       runtime.feature_schema_hash == artifact.binding.feature_schema_hash
       runtime.label_hash          == artifact.binding.label_hash

    Without this, a passed-and-byte-verified model can still be inserted into
    a manifest using a different feature definition or label window.

    Computed at:
      G3 (pipeline scorer.load): hash the inference-time feature manifest +
                                  the label currently being scored against
      G4 (manifest_row):          hash the WF training feature manifest +
                                  the lookahead window referenced in the row
      G5 (broker submit):         hash the manifest_row's binding (just pass
                                  through from G4's stamping)
    """
    feature_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]
    label_hash: Annotated[str, Field(min_length=64, max_length=64)]
```

**Why this addresses codex v8 #2, #3**:
- #2: `RuntimeDataBinding` parameter on `assert_artifact_validated` forces every gate to bind runtime data hashes to artifact's binding. A passed model in the wrong manifest fails the gate.
- #3: `model_uri` is the source of truth via `ArtifactStore`. Local-only MVP uses `LocalFileArtifactStore`; S3 backend requires `versionId` for integrity binding (no ambiguous "bucket/key" without version).

### 5.3 Ed25519-signed, scope-bound bypass (v8 NEW — replaces HMAC)

```python
# renquant-common/src/renquant_common/contracts/leakage_config.py
import json, logging, math, os, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal
import pydantic
from pydantic import Field
import nacl.signing, nacl.exceptions, nacl.encoding
from renquant_common.contracts.triad import AwareDatetime, TriadPolicy, utc_now

log = logging.getLogger("renquant_common.leakage_guards.bypass")

GateCaller = Literal[
    "pipeline:scorer_load",
    "backtesting:manifest_row",
    "backtesting:sim_driver",
    "backtesting:fit_calibrator",
    "orchestrator:manifest_row",
    "execution:submit_order",
]

Environment = Literal["dev", "shadow_sim", "wf_backfill", "prod_cron", "prod_live"]
"""Closed taxonomy. v9 NEW per codex v8 #5: bypass scope must include environment.

   dev:           operator iteration, no live impact
   shadow_sim:    backtesting/sim runs that DO NOT affect any live decision
   wf_backfill:   filling triad_status on existing artifacts during S2 migration
   prod_cron:     daily_104 cron path producing manifest rows
   prod_live:     execution submits orders against this manifest
"""

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
    allowed_gates: Annotated[list[GateCaller], Field(min_length=1)]
    allowed_environments: Annotated[list[Environment], Field(min_length=1)]  # v9 NEW per codex v8 #5
    expires_at: AwareDatetime
    reason: str
    approved_by: str                              # GitHub handle (informational; verified separately by CODEOWNERS)
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
from renquant_common.contracts.runtime_binding import RuntimeDataBinding

def assert_artifact_validated(
    artifact: ScorerArtifact, *,
    cfg: LeakageGuardConfig,
    caller: GateCaller,
    runtime_binding: RuntimeDataBinding,   # v9 NEW per codex v8 #2
    environment: Environment,              # v9 NEW per codex v8 #5
) -> None:
    fp = artifact.triad_report.binding.fingerprint()
    s = artifact.triad_report.triad_status
    b = artifact.triad_report.binding

    # v9: runtime data binding match. Even byte-verified passed artifact rejected
    # if features/labels currently in use disagree with what was trained on.
    if b.feature_schema_hash != runtime_binding.feature_schema_hash:
        telemetry.emit_event("gate_block_feature_schema_mismatch",
                             caller=caller, artifact_fingerprint=fp,
                             artifact_hash=b.feature_schema_hash,
                             runtime_hash=runtime_binding.feature_schema_hash)
        raise ArtifactNotValidated(
            f"{caller}: feature_schema_hash mismatch:\n"
            f"  artifact (train-time): {b.feature_schema_hash[:16]}\n"
            f"  runtime (call-site):   {runtime_binding.feature_schema_hash[:16]}"
        )
    if b.label_hash != runtime_binding.label_hash:
        telemetry.emit_event("gate_block_label_mismatch",
                             caller=caller, artifact_fingerprint=fp,
                             artifact_hash=b.label_hash,
                             runtime_hash=runtime_binding.label_hash)
        raise ArtifactNotValidated(
            f"{caller}: label_hash mismatch:\n"
            f"  artifact (train-time): {b.label_hash[:16]}\n"
            f"  runtime (call-site):   {runtime_binding.label_hash[:16]}"
        )

    # Policy strength gate (v8) — per-field direction-correct (v9 fix)
    if weaker_than(artifact.triad_report.policy, cfg.minimum_acceptable_policy):
        telemetry.emit_event("gate_block_policy_weak", caller=caller, artifact_fingerprint=fp,
                             artifact_policy_hash=artifact.triad_report.policy.policy_hash(),
                             minimum_policy_hash=cfg.minimum_acceptable_policy.policy_hash())
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} whose triad policy is weaker than minimum."
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
                             triad_status="passed", environment=environment)
        return

    # pending — verified bypass must match fingerprint AND allowed_gates AND allowed_environments
    now = utc_now()
    matching = [
        b for b in cfg.triad_bypasses
        if b.artifact_fingerprint == fp
        and b.expires_at > now
        and caller in b.allowed_gates
        and environment in b.allowed_environments    # v9 NEW per codex v8 #5
    ]
    if not matching:
        telemetry.emit_event("gate_block", caller=caller, artifact_fingerprint=fp,
                             triad_status="pending", environment=environment,
                             reason="no verified+scoped bypass match",
                             total_verified_bypasses=len(cfg.triad_bypasses))
        raise ArtifactNotValidated(
            f"{caller}: refusing scorer fp={fp[:16]} status=pending env={environment}; "
            f"no verified bypass entry matches (fingerprint ∧ gate∈allowed_gates ∧ env∈allowed_environments)."
        )
    entry = matching[0]
    telemetry.emit_event("gate_bypass", caller=caller, artifact_fingerprint=fp,
                         triad_status="pending", environment=environment,
                         bypass_expires_at=entry.expires_at.isoformat(),
                         bypass_approved_by=entry.approved_by,
                         bypass_pr_url=entry.pr_url,
                         bypass_key_id=entry.key_id,
                         allowed_gates=list(entry.allowed_gates),
                         allowed_environments=list(entry.allowed_environments))
    log.warning(
        "TRIAD BYPASS: %s fp=%s env=%s expires=%s key_id=%s approved_by=%s pr=%s gates=%s envs=%s",
        caller, fp[:16], environment, entry.expires_at.isoformat(), entry.key_id,
        entry.approved_by, entry.pr_url, entry.allowed_gates, entry.allowed_environments,
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
        "allowed_environments": ["wf_backfill", "shadow_sim"],
        "expires_at": datetime(2026, 6, 15, 12, 0, 0, 123456, tzinfo=timezone.utc),
        "reason": "B_tuned backfill grace period 7 days",
        "approved_by": "architect-user",
        "pr_url": "https://github.com/hallovorld/RenQuant/pull/999",
        "approved_at": datetime(2026, 6, 1, 20, 30, 0, 0, tzinfo=timezone.utc),
        "key_id": "v1",
    }
    expected = (
        b'{"allowed_environments":["wf_backfill","shadow_sim"],'
        b'"allowed_gates":["pipeline:scorer_load","backtesting:manifest_row"],'
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

v10 falsifiers (codex v10 #5 — was missing explicit regression coverage):
15. **G3 callability** — `assert_artifact_validated(..., runtime_binding=RuntimeBindingInference(...))` is callable at PanelScorer.load time without ANY realized label hash; the gate accepts/rejects based ONLY on `feature_schema_hash` + `target_spec_hash`. (Pinned by `tests/test_runtime_binding_inference_callable.py`.)
16. **TOCTOU reopen prevented** — any code path that opens `artifact.model_uri` after `load_artifact()` returns must be caught by a CI grep + AST scan; the only access patterns are `verified.open_bytes()` and `verified.spooled_path()`. (Pinned by `tests/test_no_artifact_uri_reopen.py` + `.github/workflows/gate-disable-detection.yml` extension.)
17. **G0 missing-metadata rejected** — `FeatureSchemaManifest` rejects missing/extra entries in `feature_dtypes` or `feature_lookahead_days`; rejects duplicates in `feature_cols`; rejects negative lookahead. (Pinned by `tests/test_dataset_manifest_g0.py` — 7 cases.)
18. **Policy v1 payloads parse v1 only** — a `policy_schema_version=1` payload deserialized via `parse_policy()` MUST instantiate `TriadPolicyV1`; trying to validate it as `TriadPolicyV2` raises. Old artifacts never silently rehash. (Pinned by `tests/test_policy_registry_dispatch.py`.)
19. **TargetSpec versioning binds transform code** — `TargetSpecManifestV1` with the same six surface fields but different `target_transform_code_version` produces a different `target_spec_hash`; consumer rejects mismatch. (Pinned by `tests/test_target_spec_versioning.py`.)
20. **Memory-safe spill** — a 1-GB synthetic artifact loaded via `load_artifact()` with default `in_memory_threshold_bytes=256 MiB` materializes via spooled tempfile path; the process peak RSS during load stays below `2× threshold`. (Pinned by `tests/test_load_artifact_memory_budget.py`.)

v11 falsifiers (codex v11 #1+#2+#3 — cache poisoning, single-class, extra="forbid"):

21. **Pre-existing cache entry cannot bypass freshly verified bytes** — write a `~/.renquant/artifact_cache/sha256-<expected>.bin` containing *wrong* bytes, then call `load_artifact()` with a sidecar whose `binding.model_sha == <expected>` and `model_uri` pointing at the legitimate bytes. The freshly verified temp file must atomically overwrite the cache entry via `os.replace()`; `verified.spooled_path()` must read the legitimate bytes. (Pinned by `tests/test_load_artifact_cache_poison_reject.py`.) Companion: cache_dir as a symlink, as world-readable (mode 0o755), or owned by another UID all raise `UnsafeCacheDir`.
22. **VerifiedArtifact single-class invariant** — only ONE `class VerifiedArtifact` definition exists in `renquant_common.contracts.scorer`; CI grep + AST scan rejects any duplicate. (Pinned by `tests/test_verified_artifact_single_class.py` + workflow assertion.)
23. **`extra="forbid"` on versioned-hash models rejects unknown fields** — `TriadPolicyV1.model_validate({**v1_fields, "rogue_field": 1})` raises; same for `TargetSpecManifestV1` and `FeatureSchemaManifest`. Unknown semantics MUST force a schema-version bump, not be silently dropped before hashing. (Pinned by `tests/test_versioned_models_extra_forbid.py` — 3 cases.)

## 10 · Threat model

12 leak classes (L1-L12) carried forward. **v11 attack-surface table**
(codex v11 #5 — old v8 table referenced deprecated `ScorerArtifact.load()`):

| Attack | Mitigation in v11 |
|---|---|
| Copy sidecar+artifact pair to point at different model.pt | §10A.2 `load_artifact()` streams `model_uri` once, hashes-while-streaming, raises `ArtifactBytesMismatch` if `sha256 ≠ binding.model_sha` |
| Swap `model.pt` between hash check and `torch.load` | §10A.2 `VerifiedArtifact` carries the verified bytes (small) OR a content-addressed spooled cache path (large). Consumer reads via `verified.open_bytes()` / `verified.spooled_path()`; AST scan rejects any code path that reopens `artifact.model_uri`. Falsifier #16. |
| Pre-seed `~/.renquant/artifact_cache/sha256-<x>.bin` with malicious bytes | §10A.2 cache writes always go through `os.replace()` — freshly verified temp file wins atomically. `_enforce_cache_dir` refuses symlink / world-readable / non-owner cache dirs. Falsifier #21. |
| Run reducer with weakened policy (lower threshold, fewer seeds) | `policy_hash` derived from policy; binding mismatch raises |
| Submit a future schema as a v1 payload to skip new gates | §10A.3 `parse_policy()` rejects unknown `policy_schema_version` with `UnknownPolicyVersion`. `extra="forbid"` on V1 rejects unknown payload keys (codex v11 #3). Falsifier #18. |
| Mint a fake bypass on a compromised verifier | Asymmetric Ed25519 — verifier has only public key |
| Reuse a backfill bypass for live trading | `allowed_gates` in signed payload; gate checks `caller in allowed_gates` |
| Pre-validation signing / mid-spec drift | Single `canonical_payload()` function + golden vector test |
| Replay bypass after expiry | `expires_at` in signed payload; gate compares to `utc_now()` |
| Race two Tier-2 runners | CAS on `(binding, current_status)`; second runner gets `StaleBindingError` |
| Force `failed → passed` by sidecar rewrite | `_VALID_TRANSITIONS` table at CAS layer rejects |
| Ship a target-code change without bumping target_spec_hash | §10A.1 `target_transform_code_version` in canonical hash payload; same surface fields under a new transform produce a new hash. Falsifier #19. |
| Submit a feature parquet with extra columns / negative lookahead / label-name overlap | §10A.4 `FeatureSchemaManifest.closed_keys_and_sorted` + `assert_disjoint_from_labels`; G0 writer-time block. Falsifier #17. |

## 10A · v10 contract supplements (codex v9 review)

### 10A.1 — Inference-time vs training-time binding (codex v9 #1)

**Problem**: v9's `RuntimeDataBinding` required `feature_schema_hash` AND
`label_hash`. G3 (`PanelScorer.load` at inference) cannot compute
`label_hash` — labels are FUTURE returns not yet realized. The gate is
either un-callable or implementers stub the hash, defeating the binding.

**v10 fix**: split into two binding shapes.

```python
# renquant-common/src/renquant_common/contracts/runtime_binding.py (v10)

class TargetSpecManifestV1(pydantic.BaseModel):
    """V10-hardened (codex v10 #4): target spec is a versioned Pydantic
    manifest, NOT a bare model_dump. Same-surface fields under a different
    target_transform_code_version OR embargo_policy hash differently.

    Includes everything that affects label semantics:
      - target_spec_schema_version: opens room for V2 (e.g., when we add
        macro-stratification or sector-relative excess returns)
      - target_transform_code_version: bump when label-construction code
        changes even if the field set stays the same
      - embargo_policy_id: bump when the embargo days OR rule changes
      - benchmark_id / calendar_id / universe_filter_id: opaque IDs
        bound to specific data manifests (NOT raw strings)
      - manifest_id: UUID for this concrete target build
    """
    # v11 (codex v11 #3): extra="forbid" — versioned/canonical-hash models
    # must REJECT unknown fields, not silently drop them before hashing.
    # New semantics force a schema-version bump, not a stealth payload.
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    target_spec_schema_version: Literal[1] = 1
    label_col: str
    lookahead_days: Annotated[int, Field(gt=0)]
    return_type: Literal["excess", "log", "simple", "rank", "raw"]
    benchmark_id: str                    # opaque ID resolved via target_manifest.json
    calendar_id: str
    universe_filter_id: str
    target_transform_code_version: Annotated[int, Field(gt=0)]
    embargo_policy_id: str
    manifest_id: str

    def target_spec_hash(self) -> str:
        import hashlib
        canonical = canonical_payload_target_manifest(self.model_dump())  # same canonical_payload pattern
        return hashlib.sha256(canonical).hexdigest()


class TargetSpecManifestV2(pydantic.BaseModel):
    """Placeholder for next schema. Add fields explicitly; don't default-rehash V1."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    target_spec_schema_version: Literal[2] = 2
    # ... V1 fields + new fields ...
    def target_spec_hash(self) -> str: ...


TargetSpecManifest = TargetSpecManifestV1 | TargetSpecManifestV2


def parse_target_spec(payload: dict) -> TargetSpecManifest:
    """Dispatch by target_spec_schema_version. Unknown → UnknownTargetSpecVersion.

    Same contract as parse_policy (§10A.3): old artifacts continue to
    parse against their stamped version even after V2 ships. Golden vector
    test in tests/test_target_spec_canonical.py pins the V1 canonical bytes.
    """
    version = payload.get("target_spec_schema_version", 1)
    if version == 1:
        return TargetSpecManifestV1.model_validate(payload)
    if version == 2:
        return TargetSpecManifestV2.model_validate(payload)
    raise UnknownTargetSpecVersion(version)


# Legacy export alias for back-compat (will be removed once §5 v9 blocks deleted):
TargetSpec = TargetSpecManifestV1


class RuntimeBindingInference(pydantic.BaseModel):
    """For G3 (PanelScorer.load) and G5 (broker submit_order). Pre-outcome
    knowable. NO label_bytes_hash."""
    feature_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]
    target_spec_hash: Annotated[str, Field(min_length=64, max_length=64)]


class RuntimeBindingTraining(pydantic.BaseModel):
    """For G4 (manifest_row, fit_walkforward_calibrators). Training-time
    consumers HAVE realized labels for their completed cohort."""
    feature_schema_hash: Annotated[str, Field(min_length=64, max_length=64)]
    target_spec_hash: Annotated[str, Field(min_length=64, max_length=64)]
    label_bytes_hash: Annotated[str, Field(min_length=64, max_length=64)]


RuntimeBinding = RuntimeBindingInference | RuntimeBindingTraining
```

`TriadBinding` (§5.1) gains `target_spec_hash` + `label_bytes_hash`
separately. `assert_artifact_validated` dispatches on input type:

```python
def assert_artifact_validated(artifact, *, cfg, caller, runtime_binding, environment):
    b = artifact.triad_report.binding
    if b.feature_schema_hash != runtime_binding.feature_schema_hash:
        raise ArtifactNotValidated(...)
    if b.target_spec_hash != runtime_binding.target_spec_hash:
        raise ArtifactNotValidated(...)
    if isinstance(runtime_binding, RuntimeBindingTraining):
        if b.label_bytes_hash != runtime_binding.label_bytes_hash:
            raise ArtifactNotValidated(...)
    ...
```

### 10A.2 — VerifiedArtifact eliminates TOCTOU (codex v9 #2, v10 #3, v11 #1+#2)

**Problem (v9)**: v9's `ScorerArtifact.load()` hashed `model_uri` then
returned only metadata. Consumer's `torch.load(uri)` re-opens the path
→ file can be swapped between hash check and model construction.

**v10 problem (codex v10 #3)**: joining all chunks into one bytes + wrapping
in `BytesIO` held ~2× RAM for multi-GB models.

**v11 problem (codex v11 #1)**: when a freshly verified spooled tempfile
collided with a pre-existing `sha256-<x>.bin` in the cache dir, the v10
implementation deleted the fresh temp file and returned the EXISTING cache
path WITHOUT rehashing it. That reopens the verified-vs-loaded split the
design is supposed to close: corruption, tampering, or a symlinked entry in
the cache dir would all bypass the freshly verified hash.

**v11 fix**: single source of truth for the returned bytes — the freshly
verified temp file ALWAYS wins via `os.replace()` (atomic, replaces existing).
The cache dir is permissions-locked at `0o700` and pre-validated against
symlink traversal on every call. There is exactly ONE `VerifiedArtifact`
class definition (codex v11 #2 — v10 left a stale dataclass before the
final one).

```python
import errno, hashlib, io, os, stat, tempfile
from pathlib import Path

_VERIFICATION_TOKEN = object()    # module-private; cannot be imported


def _enforce_cache_dir(cache_dir: Path) -> None:
    """Refuse to use a cache dir that is a symlink, world-writable, or
    permissive (mode > 0o700). Created with mode 0o700 if absent.
    Falsifier #21 pins this."""
    try:
        cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    except FileExistsError:
        pass
    st = cache_dir.lstat()
    if stat.S_ISLNK(st.st_mode):
        raise UnsafeCacheDir(f"{cache_dir} is a symlink")
    if not stat.S_ISDIR(st.st_mode):
        raise UnsafeCacheDir(f"{cache_dir} is not a directory")
    if (st.st_mode & 0o077) != 0:
        # Group/other bits set → not owner-private
        raise UnsafeCacheDir(
            f"{cache_dir} mode {oct(st.st_mode & 0o777)} > 0o700; refuse"
        )
    if st.st_uid != os.geteuid():
        raise UnsafeCacheDir(f"{cache_dir} not owned by euid")


class VerifiedArtifact:
    """SINGLE definition (v11 — codex v11 #2 collapsed v10 duplicates).

    Holds EITHER the verified bytes (small artifacts) OR a path to a
    content-addressed verified spooled file (large artifacts). Consumers
    MUST use one of the accessors — never re-open `artifact.model_uri`.

    Constructed only via `load_artifact()`; the module-private token
    prevents external synthesis. Pinned by falsifier #16 (TOCTOU reopen
    prevention).
    """
    __slots__ = ("_artifact", "_model_bytes", "_spooled_path")

    def __init__(self, *, _artifact, _model_bytes, _spooled_path, _token):
        if _token is not _VERIFICATION_TOKEN:
            raise RuntimeError("Use load_artifact() — private token required.")
        if (_model_bytes is None) == (_spooled_path is None):
            raise RuntimeError(
                "exactly one of _model_bytes / _spooled_path required"
            )
        self._artifact = _artifact
        self._model_bytes = _model_bytes
        self._spooled_path = _spooled_path

    @property
    def artifact(self): return self._artifact

    def open_bytes(self) -> io.BufferedReader | io.BytesIO:
        """Fresh file-like over the verified content. Small artifacts
        get io.BytesIO; large artifacts get an open read-only file
        descriptor on the spooled cache path."""
        if self._model_bytes is not None:
            return io.BytesIO(self._model_bytes)
        # Reopen via the cached path, NOT model_uri.
        return open(self._spooled_path, "rb")

    def spooled_path(self) -> Path | None:
        """For consumers that need a real path (e.g. mmap, torch.load with
        explicit path arg). None for small artifacts kept in-memory."""
        return self._spooled_path


def load_artifact(
    sidecar_path: Path,
    *,
    store: ArtifactStore | None = None,
    max_bytes: int = 16 * 1024**3,                  # 16 GiB hard cap
    in_memory_threshold_bytes: int = 256 * 1024**2, # ≤ 256 MiB → in-memory bytes; > → spooled tempfile
    cache_dir: Path | None = None,                  # content-addressed cache (default: ~/.renquant/artifact_cache)
) -> VerifiedArtifact:
    """SINGLE entry point. Streams URI once, hashes-while-streaming, and
    chooses a memory-safe materialization strategy:

      - ≤ in_memory_threshold_bytes (default 256 MiB): bytes kept in-memory;
        `verified.open_bytes() -> BytesIO`. No file I/O after first stream.
      - > in_memory_threshold_bytes: streamed into a content-addressed
        spooled tempfile in `cache_dir`. The temp file's sha256 is verified
        BEFORE the temp file is atomically `os.replace()`d into the final
        cache path. The freshly verified bytes ALWAYS win; any pre-existing
        `sha256-<x>.bin` is overwritten. (codex v11 #1 — pre-existing cache
        entry must NEVER be trusted blind.) Consumer reads via
        `verified.spooled_path()` — a fresh path NOT equal to model_uri,
        so a subsequent `torch.load(path)` only reads the verified bytes.

    Both strategies guarantee the consumer never reads bytes that were not
    just hashed: eliminates v9 TOCTOU (codex v9 #2), v10 RAM doubling
    (codex v10 #3), and v11 cache-poisoning (codex v11 #1).
    """
    store = store or make_default_store()
    cache_dir = cache_dir or (Path.home() / ".renquant" / "artifact_cache")
    _enforce_cache_dir(cache_dir)
    artifact = ScorerArtifact.model_validate_json(sidecar_path.read_text())
    if not store.supports(artifact.model_uri):
        raise UnsupportedArtifactScheme(artifact.model_uri)

    expected_sha = artifact.triad_report.binding.model_sha
    h = hashlib.sha256()
    in_mem_buf: list[bytes] = []
    tmp_fp = None
    spilled = False
    total = 0
    try:
        for chunk in store.stream_bytes(artifact.model_uri):
            total += len(chunk)
            if total > max_bytes:
                raise ArtifactTooLarge(
                    f"{artifact.model_uri} > {max_bytes:,} bytes"
                )
            h.update(chunk)
            if not spilled and total <= in_memory_threshold_bytes:
                in_mem_buf.append(chunk)
            else:
                if not spilled:
                    # Spill what we have so far into the verified tempfile.
                    tmp_fp = tempfile.NamedTemporaryFile(
                        dir=cache_dir, prefix=".verify-", suffix=".tmp",
                        delete=False, mode="wb",
                    )
                    os.chmod(tmp_fp.name, 0o600)
                    for prior in in_mem_buf:
                        tmp_fp.write(prior)
                    in_mem_buf.clear()
                    spilled = True
                tmp_fp.write(chunk)
        actual_sha = h.hexdigest()
        if actual_sha != expected_sha:
            raise ArtifactBytesMismatch(
                f"expected {expected_sha}, got {actual_sha} "
                f"for {artifact.model_uri}"
            )
        if spilled:
            tmp_fp.flush()
            os.fsync(tmp_fp.fileno())
            tmp_fp.close()
            final_path = cache_dir / f"sha256-{actual_sha}.bin"
            # codex v11 #1 fix — refuse pre-existing entry if it is a
            # symlink, then ATOMICALLY replace it with the freshly verified
            # temp file. os.replace() is atomic on POSIX + Windows and
            # overwrites any existing regular file. The freshly verified
            # bytes are the ONLY source of truth that leaves this function.
            if final_path.exists() and final_path.is_symlink():
                raise UnsafeCacheDir(
                    f"{final_path} is a symlink; refuse cache reuse"
                )
            os.replace(tmp_fp.name, final_path)
            try:
                os.chmod(final_path, 0o400)   # read-only after settle
            except OSError as e:
                if e.errno != errno.EPERM:
                    raise
            return VerifiedArtifact(
                _artifact=artifact,
                _model_bytes=None,
                _spooled_path=final_path,
                _token=_VERIFICATION_TOKEN,
            )
        return VerifiedArtifact(
            _artifact=artifact,
            _model_bytes=b"".join(in_mem_buf),
            _spooled_path=None,
            _token=_VERIFICATION_TOKEN,
        )
    except Exception:
        if tmp_fp is not None:
            tmp_fp.close()
            try:
                Path(tmp_fp.name).unlink(missing_ok=True)
            except Exception:
                pass
        raise
```

Consumer usage (PanelScorer):
```python
verified = load_artifact(sidecar_path, store=...)
assert_artifact_validated(verified.artifact, cfg=..., caller=..., runtime_binding=..., environment=...)
# CRITICAL: use verified.open_bytes() or verified.spooled_path(), NOT torch.load(uri)
state_dict = torch.load(verified.open_bytes(), map_location="cpu")
# Large artifact alternative:
#   state_dict = torch.load(verified.spooled_path(), map_location="cpu")
```

### 10A.3 — Versioned policy registry (codex v9 #4)

**Problem**: v9 `TriadPolicy` was a single class with `policy_schema_version`
field. Adding a defaulted field later still affects parsing of old v1
artifacts before the version check fires.

**v10 fix**: explicit per-version classes + dispatch:

```python
# renquant-common/src/renquant_common/contracts/triad_policy.py

class TriadPolicyV1(pydantic.BaseModel):
    """v1 fields frozen. Future changes go in TriadPolicyV2.

    v11 (codex v11 #3): extra="forbid" — unknown fields raise, forcing the
    callsite to bump policy_schema_version instead of silently dropping
    payload entries before hashing.
    """
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    policy_schema_version: Literal[1] = 1
    stats_algorithm_version: Annotated[int, Field(gt=0)] = 1
    tier1_p_threshold: PValue = 0.05
    tier1_min_n_val_dates: Annotated[int, Field(gt=0)] = 30
    tier1_aa_drift_max: Annotated[FiniteFloat, Field(ge=0.0)] = 0.03
    tier2_p_threshold: PValue = 0.05
    tier2_min_n_seeds: Annotated[int, Field(gt=0)] = 3
    tier2_min_n_dates_per_regime: Annotated[int, Field(gt=0)] = 20
    tier2_max_placebo_real_ratio: Annotated[FiniteFloat, Field(ge=0.0, le=1.0)] = 0.50
    bootstrap_n_iters: Annotated[int, Field(gt=0)] = 10_000
    bootstrap_rng_seed: int = 0

    def policy_hash(self) -> str: ...   # canonical sha256


class TriadPolicyV2(pydantic.BaseModel):
    """Placeholder for the next schema version. Add fields explicitly."""
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    policy_schema_version: Literal[2] = 2
    # ... v1 fields + new fields here ...
    def policy_hash(self) -> str: ...


TriadPolicy = TriadPolicyV1 | TriadPolicyV2     # discriminated union


def parse_policy(payload: dict) -> TriadPolicy:
    """Dispatch by policy_schema_version. UnknownPolicyVersion raised on
    versions not in the registry — consumers must update before reading
    artifacts stamped with a newer policy."""
    version = payload.get("policy_schema_version", 1)
    if version == 1:
        return TriadPolicyV1.model_validate(payload)
    if version == 2:
        return TriadPolicyV2.model_validate(payload)
    raise UnknownPolicyVersion(
        f"policy_schema_version={version} not in registry. "
        f"Update renquant-common to a version that knows about it."
    )
```

Old artifacts stamped with `policy_schema_version=1` continue to parse via
`TriadPolicyV1` even after `TriadPolicyV2` ships — no silent rehash.

### 10A.4 — FeatureSchemaManifest (codex v9 #5)

**Problem**: `feature_schema_hash` was just a 64-hex string with no
spec. Same column names + dtypes can be produced by different code /
calendars / data vintages and still pass the gate.

**v10 fix**: explicit Pydantic manifest + golden vector test.

```python
# renquant-common/src/renquant_common/contracts/feature_manifest.py

class FeatureSchemaManifest(pydantic.BaseModel):
    """Positive declaration of feature surface + provenance.

    Hashing this gives feature_schema_hash. Same columns produced by
    different transform_code_version OR calendar OR universe will have
    different hashes — even if dtypes match.

    v10 (codex v10 #2): validator hardened to reject ALL bypass routes:
      - missing key in feature_dtypes OR feature_lookahead_days
      - duplicate entries in feature_cols
      - feature_cols not sorted (canonical form is sorted ascending)
      - negative lookahead values (only exact 0 allowed)
      - overlap with declared LabelManifest.label_cols (passed in at construction)

    v11 (codex v11 #3): extra="forbid" — unknown payload keys raise.
    Forces every new feature-surface field to be an explicit schema bump.
    """
    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    feature_cols: list[str]                       # sorted ASCENDING + no duplicates (validator-enforced)
    feature_dtypes: dict[str, str]                # MUST cover every feature_cols entry
    feature_lookahead_days: dict[str, Annotated[int, Field(ge=0, le=0)]]  # exactly 0; ge=0 le=0 = {0}
    feature_transform_code_version: Annotated[int, Field(gt=0)]
    universe_filter: str                          # "alpha158_5b_active" / "renquant_104_watchlist"
    calendar_name: str                            # "NYSE" / "trading" / "calendar"
    data_vintage_date: AwareDatetime              # when source data was snapshot
    manifest_id: str                              # opaque UUID for this build

    @pydantic.model_validator(mode="after")
    def closed_keys_and_sorted(self) -> "FeatureSchemaManifest":
        # 1. No duplicates in feature_cols
        if len(self.feature_cols) != len(set(self.feature_cols)):
            duplicates = sorted({c for c in self.feature_cols
                                 if self.feature_cols.count(c) > 1})
            raise ValueError(f"feature_cols contains duplicates: {duplicates}")
        # 2. feature_cols must be sorted (canonical form for stable hashing)
        if list(self.feature_cols) != sorted(self.feature_cols):
            raise ValueError(f"feature_cols must be sorted ascending; got {self.feature_cols[:5]}...")
        # 3. Closed-key equality across the three dicts
        cols = set(self.feature_cols)
        dtypes_keys = set(self.feature_dtypes.keys())
        lookahead_keys = set(self.feature_lookahead_days.keys())
        if dtypes_keys != cols:
            extra = dtypes_keys - cols; missing = cols - dtypes_keys
            raise ValueError(f"feature_dtypes mismatch — extra={extra}, missing={missing}")
        if lookahead_keys != cols:
            extra = lookahead_keys - cols; missing = cols - lookahead_keys
            raise ValueError(f"feature_lookahead_days mismatch — extra={extra}, missing={missing}")
        # (4. ge=0 le=0 on values already enforced by Annotated; both negative
        #     and positive lookahead rejected at field validation time.)
        return self

    def assert_disjoint_from_labels(self, label_cols: list[str]) -> None:
        """Called at base-data write time AND at G3/G4 gate time. Refuses
        when a feature column name matches a declared label column.

        Codex v10 #2: G0 MVP row previously compared against
        `labels.feature_cols` which was a typo — it must compare against
        `labels.label_cols` (the declared label surface).
        """
        overlap = set(self.feature_cols) & set(label_cols)
        if overlap:
            raise ValueError(
                f"FeatureSchemaManifest contains label columns: {sorted(overlap)}. "
                "Labels MUST live in a separate parquet (G0 invariant)."
            )

    def schema_hash(self) -> str:
        """Canonical sha256. Single function, used by sign + verify + golden vector."""
        import hashlib
        canonical = canonical_payload_feature_manifest(self.model_dump())
        return hashlib.sha256(canonical).hexdigest()
```

Golden vector test pinned (mirror of bypass canonical_payload test, §5.4):

```python
# renquant-common/tests/test_feature_manifest_canonical.py
def test_golden_feature_manifest_canonical():
    m = {
        "feature_cols": ["alpha158_1", "alpha158_2", "fundamental_3"],
        "feature_dtypes": {"alpha158_1": "float32", "alpha158_2": "float32",
                           "fundamental_3": "int32"},
        "feature_lookahead_days": {"alpha158_1": 0, "alpha158_2": 0, "fundamental_3": 0},
        "feature_transform_code_version": 7,
        "universe_filter": "alpha158_5b_active",
        "calendar_name": "NYSE",
        "data_vintage_date": datetime(2026, 5, 31, 0, 0, 0, 0, tzinfo=timezone.utc),
        "manifest_id": "deadbeef-cafe-...",
    }
    assert canonical_payload_feature_manifest(m) == _GOLDEN_BYTES
```

### 10A.5 — G0 added to MVP (codex v9 #3)

**Problem**: v9 MVP PR list (§13) had no `renquant-base-data` PR; G0
writer-time validation was deferred to weeks 2-3. But completion
criterion #4 says "feature parquet with label-named columns cannot be
written" — without G0 in MVP, criterion #4 cannot be true at MVP-done.

**v10 fix**: MVP grows to **6 PRs**. New PR ⑥:

| #  | Repo | Files | Tests |
|----|---|---|---|
| ⑥ NEW | `renquant-base-data` | `src/renquant_base_data/builders/alpha158.py` — wire `DatasetManifest.model_validate(...)` at end of `_write_dataset()`. Refuses to write features.parquet if `set(features.columns) ∩ set(labels.label_cols) ≠ ∅` (v10 fix to codex v10 #2: was incorrectly `labels.feature_cols`) OR if any `features.feature_lookahead_days[c] != 0` OR if `embargo_days < max(labels.label_lookahead_days)` OR if `FeatureSchemaManifest.closed_keys_and_sorted` invariant fails. | `tests/test_dataset_manifest_g0.py` — 7 cases: valid manifest writes; label-name overlap rejects; positive lookahead rejects; negative lookahead rejects (v10 NEW); insufficient embargo rejects; duplicate feature_cols rejects (v10 NEW); missing dtypes/lookahead key rejects (v10 NEW). |

This makes criterion #4 mechanically true at MVP-done.


## 11 · Codex v11 → v12 — addressed

| # | Sev | v11 finding | v12 resolution |
|---|---|---|---|
| 1 | HIGH | Pre-existing `~/.renquant/artifact_cache/sha256-<x>.bin` could bypass freshly verified bytes — v11 deleted the temp file and returned the existing cache path without rehashing it | §10A.2 rewritten: freshly verified temp file ALWAYS wins via atomic `os.replace()`. Cache dir is pre-validated by `_enforce_cache_dir()` — refuses symlink / world-readable / non-owner-private (mode > 0o700). Pre-existing entries that happen to be symlinks raise `UnsafeCacheDir`. After settle, cache entries are `chmod 0o400`. New falsifier #21. |
| 2 | MED | §10A.2 contained TWO `VerifiedArtifact` class definitions — stale in-memory-only one at line 1036, replacement at line 1156. Copy-paste trap. | §10A.2 collapsed to **one** definition placed BEFORE `load_artifact()`. CI grep + AST scan reject duplicates. New falsifier #22. |
| 3 | MED | `TargetSpecManifestV1`, `TriadPolicyV1`, `FeatureSchemaManifest` used `ConfigDict(frozen=True)` — Pydantic default is to IGNORE extra fields, so a payload could carry semantically meaningful fields that silently dropped before hashing | All three now use `ConfigDict(frozen=True, extra="forbid")`. `TriadPolicyV2`/`TargetSpecManifestV2` placeholders also forbid extras. Unknown fields raise — new semantics MUST force a schema-version bump. New falsifier #23 (3 cases). |
| 4 | MED | §13 row ① said "all 17 falsifiers" and row ② routed through deprecated `ScorerArtifact.load()` (v9 API) | Row ① now lists `load_artifact() -> VerifiedArtifact` + `parse_policy` + `parse_target_spec` + `FeatureSchemaManifest` and "**all 23 falsifiers**". Row ② routes through `load_artifact()` per v11 contract, with AST scan rejecting `artifact.model_uri` reopens. |
| 5 | LOW | §10 threat model still said "Mitigation in v8" and referenced deprecated `ScorerArtifact.load()` | §10 rewritten to **v11 attack-surface table**. Each row now references §10A code paths + the named falsifier that pins the mitigation. Added rows for cache poisoning, target-code drift, feature-schema bypass. |

## 12 · Open questions

1. `nacl.signing` (libsodium-based) is the recommended Ed25519 lib; pure-stdlib Ed25519 (`cryptography` package) also acceptable — does CI/uv lock have either pinned?
2. Key rotation cadence (architect): 90 days? Stamp on each retrain?
3. After Tier-2 fail: auto-retrain (selection bias) vs hold (latency)?
4. Strategy threshold changes in binding-tuple? — answer: NO, they're consumed downstream, not in model training.

## 13 · MVP PR list (6 PRs ≤ 2 days — v10 added ⑥ per codex v9 #3)

| # | Repo | Files | Tests |
|---|---|---|---|
| ① | `renquant-common` | `contracts/triad.py` (incl. `parse_policy` → `TriadPolicyV1 \| TriadPolicyV2` per §10A.3), `contracts/scorer.py` (incl. `load_artifact() -> VerifiedArtifact` per §10A.2 — v11 single class, `os.replace()` cache-write, `_enforce_cache_dir`), `contracts/target_manifest.py` (`TargetSpecManifestV1` per §10A.1, `parse_target_spec`), `contracts/feature_manifest.py` (`FeatureSchemaManifest` per §10A.4, `closed_keys_and_sorted` validator), `contracts/leakage_config.py` (Ed25519 + scope), `leakage_guards/*`, `keys/architect_pubkeys.json` (committed), `tools/sign_bypass.py` (offline architect-only), `tests/test_canonical_vector.py` (golden vector incl. v11 unknown-field rejection per codex v11 #3), `tests/test_falsification.py` (**all 23 falsifiers** — see §9), `tests/fixtures/triad_models.py` (3 fixtures), `.github/workflows/gate-disable-detection.yml`, `.github/CODEOWNERS` for architect-only files | unit + race + state-machine + 3-fixture + HMAC→Ed25519 round-trip + golden vector + tz-aware compare + `python -O` regression + scope-rejection + byte-mismatch reject + policy-hash mismatch reject + **cache-poisoning reject** (codex v11 #1) + **VerifiedArtifact single-class invariant** (codex v11 #2) + **extra="forbid" reject** (codex v11 #3) |
| ② | `renquant-pipeline` | `kernel/panel_pipeline/panel_scorer.py::load` route through `load_artifact() -> VerifiedArtifact` (codex v11 #4 — `ScorerArtifact.load()` is v9 and deprecated). Consumer MUST call `verified.open_bytes()` or `verified.spooled_path()`; the AST scan in (gate-disable-detection.yml extension) rejects any re-open of `artifact.model_uri`. | gate-behavior matrix incl. scope-bypass / byte-mismatch / policy-weaker / TOCTOU-reopen-reject / cache-poisoning-reject |
| ③ | `renquant-model` (PatchTST surface) — codex v10 #6: target the merged repo (the old `renquant-model-patchtst` standalone is deprecated `MIGRATED_TO_renquant-model.md`) | `src/renquant_model_patchtst/hf_trainer.py::_save_artifact` (Tier 1 + policy), new `src/renquant_model_patchtst/post_save_hook.py` (Tier 2 enqueue) | synth Tier-1 + Tier-2 enqueue + synthetic runner E2E |
| ④ | `renquant-model` (GBDT surface) | mirror of ③ via `src/renquant_model_gbdt/` | mirror |
| ⑤ | `renquant-orchestrator` + `renquant-backtesting` (paired) | `manifest_row` + `wf_gate/runner.py` + `sim_driver.py` + `scripts/fit_walkforward_calibrators.py` | manifest gate behavior matrix incl. scope binding |
| ⑥ v10 NEW | `renquant-base-data` | `src/renquant_base_data/builders/alpha158.py` — wire `DatasetManifest.model_validate(...)` at end of `_write_dataset()`. Refuses parquet write on label-name overlap (against `labels.label_cols`, codex v10 #2 fix), positive OR negative `feature_lookahead_days`, missing/extra dtypes-or-lookahead keys, duplicate `feature_cols`, insufficient embargo (codex v9 #3 — G0 in MVP). | `tests/test_dataset_manifest_g0.py` — **7 cases** (v11): valid manifest writes; label-name overlap rejects; positive lookahead rejects; negative lookahead rejects; insufficient embargo rejects; duplicate feature_cols rejects; missing dtypes/lookahead key rejects. |

Full architecture wave (split parquet, typed train sig, ban `pd.read_parquet`) deferred to weeks 2-3.

## 14 · References

- Bailey, D.H., Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio." *J. Portfolio Management* 40(5).
- López de Prado, M. (2018). *Advances in Financial Machine Learning*, ch. 5, 7.
- Pesaran, M.H., Timmermann, A. (2007). "Selection of estimation window." *J. Econometrics* 137.
- Efron, B., Tibshirani, R. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Bernstein, D.J., Duif, N., Lange, T., Schwabe, P., Yang, B-Y. (2012). "High-speed high-security signatures." *J. Cryptographic Engineering* 2:77.  (Ed25519)
- NIST SP 800-186 — Recommendations for Discrete Logarithm-based Cryptography (Ed25519 inclusion 2023).

---

## 15 · v9 → v10 changelog (codex v9 resolution)

| Element | v9 | v10 |
|---|---|---|
| Inference-time binding | required `label_hash` always — impossible at G3 (future label) | `RuntimeBindingInference` (feature + target_spec only) for G3/G5; `RuntimeBindingTraining` (also label_bytes_hash) for G4. `TriadBinding` carries both. |
| Artifact byte handle | `load()` returned metadata only; consumer re-opened URI for `torch.load` (TOCTOU) | `load_artifact()` returns `VerifiedArtifact` with hashed bytes; consumer uses `verified.open_bytes()` |
| MVP scope | 5 PRs; criterion #4 (G0) deferred to weeks 2-3 | 6 PRs; ⑥ adds `renquant-base-data` writer-time `DatasetManifest.model_validate()` |
| Policy registry | single class `TriadPolicy` + version field | `TriadPolicyV1` + `TriadPolicyV2` per-version classes; `parse_policy(payload)` dispatcher |
| `feature_schema_hash` spec | 64-hex string with no scope | `FeatureSchemaManifest` Pydantic incl. `feature_transform_code_version`, `universe_filter`, `calendar_name`, `data_vintage_date`, `manifest_id`; golden vector test |
| Doc metadata | `PR #38`, `14 falsifiers` | `PR #43`, 17 falsifiers (§9 carryover) |

## 16 · v10 → v11 changelog (codex v10 resolution)

| Element | v10 | v11 |
|---|---|---|
| §2 / §4 / §5 vs §10A | §10A added v10 contract; canonical sections still defined v9 — split-brain | §2 criteria 13 + 15 rewritten to v10 contract; §4 invariants I4 + I5 rewritten; §4 adds I15-I17 (versioned `TargetSpec`, `FeatureSchemaManifest` closed-keys, memory-safe verified handle); §5 prepended with v10 reader notice redirecting to §10A; §5.1-§5.3 retained as v9 history snapshots. |
| `FeatureSchemaManifest` G0 validation | only checked provided `feature_lookahead_days > 0`; missing entries, duplicates, negative lookahead all passed | `closed_keys_and_sorted` model_validator enforces (a) closed-key equality across `feature_cols`/`feature_dtypes`/`feature_lookahead_days`, (b) `feature_cols` unique + sorted, (c) `Field(ge=0, le=0)` rejects positive AND negative lookahead, (d) `assert_disjoint_from_labels(label_cols)` rejects overlap. MVP G0 row corrected to `labels.label_cols`. |
| `load_artifact()` memory profile | joined all chunks into one `bytes` + wrapped in `BytesIO` — multi-GB models hold ~2× RAM | `in_memory_threshold_bytes=256 MiB`; over threshold → spooled tempfile (`0o600`) renamed to content-addressed cache dir after SHA verified. `VerifiedArtifact` carries EITHER `_model_bytes` OR `_spooled_path`; consumer reads via `open_bytes()` (small) or `spooled_path()` (large). |
| `TargetSpec` | bare 6 fields (label / lookahead / return type / benchmark / calendar / universe) — transform-code drift gave the same hash | `TargetSpecManifestV1` versioned + `target_transform_code_version` + `embargo_policy_id` + ID-based benchmark/calendar/universe; `parse_target_spec` dispatcher; golden vector test. |
| §9 falsifiers | 14 (v7's 1-8 + v8's 9-14); v9/v10 regressions not pinned | **20** — added 15: G3 callable without realized labels; 16: TOCTOU reopen prevention; 17: G0 missing-metadata reject; 18: policy v1 payloads parse v1 only; 19: target-spec versioning binds transform code; 20: memory-safe spill RSS budget. Each pinned to a named pytest. |
| MVP repo names | ③/④ named `renquant-model-patchtst` / `renquant-model-gbdt` standalone (deprecated) | ③ targets `renquant-model` via `src/renquant_model_patchtst/`; ④ via `src/renquant_model_gbdt/`. Standalone repos marked `MIGRATED_TO_renquant-model.md`. |
| Doc metadata | `v10`, 17 falsifiers | `v11`, 20 falsifiers, header reviewer summary captures v10 → v11 deltas. |

## 17 · v11 → v12 changelog (codex v11 resolution)

| Element | v11 | v12 |
|---|---|---|
| Cache write (large artifacts) | `os.rename(tmp, final)` IF `final` absent, else `tmp.unlink()` and return pre-existing path | `os.replace(tmp, final)` ALWAYS — atomic, freshly-verified bytes overwrite any prior entry. Settled cache entries get `chmod 0o400`. |
| Cache-dir security | not validated | `_enforce_cache_dir()` refuses symlink / non-owner / mode > 0o700; creates with mode 0o700 if absent. Raises `UnsafeCacheDir` on violation. |
| `VerifiedArtifact` class | two definitions side-by-side at lines 1036 + 1156 | one definition placed BEFORE `load_artifact()`. CI grep + AST scan reject duplicates. |
| Versioned-hash Pydantic models | `ConfigDict(frozen=True)` — unknown payload fields silently dropped before hashing | `ConfigDict(frozen=True, extra="forbid")` — unknown fields raise. Applies to `TargetSpecManifestV1/V2`, `TriadPolicyV1/V2`, `FeatureSchemaManifest`. |
| §13 MVP rows | row ① "all 17 falsifiers", row ② "route through `ScorerArtifact.load()`" | row ① "all 20 falsifiers" + lists `load_artifact()` / `parse_policy` / `parse_target_spec` / `FeatureSchemaManifest`; row ② routes through `load_artifact() -> VerifiedArtifact`; AST scan rejects `artifact.model_uri` reopens. |
| §10 threat-model header | "Mitigation in v8" + `ScorerArtifact.load()` rows | "Mitigation in v11" + §10A code-path references + falsifier IDs. New attack rows: cache poisoning (#21), target-code drift (#19), feature-schema bypass (#17). |
| Falsifiers | 20 | **23** — added 21: cache-poisoning reject + symlink/perm reject; 22: VerifiedArtifact single-class invariant; 23: extra="forbid" reject. |
| Doc metadata | `v11`, 20 falsifiers | `v12`, 23 falsifiers, header reviewer summary captures v11 → v12 deltas. |

### Previous changelogs (compressed)



| Element | v8 | v9 |
|---|---|---|
| Policy strictness direction | `strength_score()` negated p-thresholds; treated 0.01 as stricter than 0.05 (backwards: gate fails p<threshold, so larger threshold = stricter); missing `aa_drift_max` + `bootstrap_n_iters` | `strength_score` REMOVED; `weaker_than(a, minimum)` is direct per-field comparison with explicit DIRECTION TABLE; all 8 axes including aa_drift_max + bootstrap_n_iters checked |
| Runtime data binding | `feature_schema_hash` + `label_hash` decorative at gate site | `RuntimeDataBinding(feature_schema_hash, label_hash)` parameter; gate refuses if not equal to artifact binding |
| Artifact storage abstraction | `ScorerArtifact.load()` derived local path from sidecar filename, ignoring `model_uri` | `ArtifactStore` Protocol + `LocalFileArtifactStore` + `S3VersionedArtifactStore` (requires versionId); `load(sidecar_path, store=...)` resolves via `model_uri` |
| Policy versioning | `policy_hash()` over unversioned fields; future schema bump silently rehashes old | `policy_schema_version` + `stats_algorithm_version` in policy; both in hash; `weaker_than` rejects on mismatch |
| Bypass environment scope | per-gate only | + `allowed_environments: list[Environment]` in SIGNED payload; gate takes `environment` parameter |
| Imports | snippet missing `math`, `timezone` | added |
| Falsifiers | 14 (v8 §9) | 17 (added: runtime binding mismatch, env-scope rejection, policy-version mismatch) |
| Completion criteria | 12 (v8 §2) | 15 (added: runtime binding, environment, store abstraction) |
