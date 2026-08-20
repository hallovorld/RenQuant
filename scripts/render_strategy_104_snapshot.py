#!/usr/bin/env python3
"""Render renquant_104's production snapshot from the PINNED strategy config +
artifact metadata — never hand-maintained.

doc/arch/strategy-104.md previously carried a hand-written "Production
snapshot" table that drifted from reality (it stayed dated 2026-06-08 through
at least one later prod/shadow switch, and a design review round was wasted
trusting it — see doc/design/2026-07-01-104-105-design-review-amendments.md
amendment A6, and the unified 107 master plan M9).

The first generated version of this snapshot (PR #429) read
``backtesting/renquant_104/strategy_config.json`` — the umbrella WORKING-COPY
config. That file is not what the daily run consumes: production config comes
from the PINNED subrepo checkout at
``.subrepo_runtime/repos/renquant-strategy-104/configs/`` (pin-aligned to the
``renquant-strategy-104`` commit recorded in ``subrepos.lock.json``). The
umbrella working copy went stale across the 2026-06-23 XGB re-promotion, so
the "generated" snapshot faithfully rendered a stale source — the same rot
class one level down. This version therefore reads:

* active config:   <repo>/.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json
* shadow config:   <repo>/.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json
* pin identities:  <repo>/subrepos.lock.json
* artifact + calibrator metadata: files referenced by those configs, resolved
  relative to <repo>/backtesting/renquant_104 (the historical artifact base).

Output is deterministic (byte-for-byte reproducible given the same inputs;
the ``Source fingerprint:`` line is itself a hash over the per-file source
hashes below it, so it changes iff the actual sources change — no wall-clock
field to exclude from anything), so ``--check`` can assert "regenerating now
produces exactly the committed file" as the staleness gate, not a day-count
heuristic. A pinned-runtime/lock mismatch (PIN DRIFT) makes both the default
render and ``--check`` refuse outright unless ``--allow-pin-drift`` is passed
explicitly for a diagnostic (never-committed) render.

Missing files or fields NEVER crash the render — they are stated explicitly
as ``unknown (field absent)`` / ``unknown (file missing: ...)`` so the
snapshot is honest about what its sources do not stamp.

Modes
-----
* default:            render and write ``--output``.
* ``--check``:        exit 1 if a fresh render differs from ``--output``.
* ``--verify-pinned-declaration``: semantic CI check — verify the committed
  snapshot's machine block against a strategy-104 ``configs/`` directory
  checked out at the ``subrepos.lock.json`` pin (used by
  ``.github/workflows/strategy-104-snapshot-fresh.yml`` on lock-pin bumps,
  where the hosted runner has no ``.subrepo_runtime`` and no live artifacts).
* ``--selftest``:     build a synthetic fixture tree in a temp dir and prove
  render / check / verify behave, without touching any real source.
* ``--allow-pin-drift``: DIAGNOSTIC ONLY, combine with default or ``--check``.
  Without it, both refuse outright when the pinned runtime checkout's HEAD
  disagrees with ``subrepos.lock.json`` — a drifted-but-unflagged snapshot
  would legitimize an unpinned runtime as pin-aligned. Never commit output
  produced with this flag as the canonical snapshot.

Not every historical "Production snapshot" row has a clean current-state
source (e.g. a specific walk-forward run's mean IC, or a regime-detector
commit hash) — those stay as hand-written narrative in strategy-104.md
itself. This script only renders what is honestly derivable from the pinned
config + artifact metadata right now.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]

# Locations relative to --repo-root (the tree that holds the SOURCES; on the
# live machine this is the live umbrella tree, in CI it is the checkout).
PINNED_CONFIGS_REL = Path(".subrepo_runtime") / "repos" / "renquant-strategy-104" / "configs"
PINNED_GIT_DIR_REL = Path(".subrepo_runtime") / "repos" / "renquant-strategy-104" / ".git"
STRATEGY_DIR_REL = Path("backtesting") / "renquant_104"
LOCK_FILE_REL = Path("subrepos.lock.json")
ACTIVE_CONFIG_NAME = "strategy_config.json"
SHADOW_CONFIG_NAME = "strategy_config.shadow.json"
PINNED_SUBREPO_NAME = "renquant-strategy-104"

DEFAULT_OUTPUT = REPO_ROOT / "doc" / "arch" / "strategy-104-snapshot.md"

UNKNOWN_ABSENT = "unknown (field absent)"

SOURCE_FINGERPRINT_PREFIX = "Source fingerprint:"
PIN_DRIFT_MARKER = "PIN DRIFT"
MACHINE_BLOCK_BEGIN = "<!-- snapshot-machine-block"
MACHINE_BLOCK_END = "-->"

# Same field-priority convention as model_freshness_monitor.py
# (renquant-orchestrator) / shadow_scoring.py's _DATA_CUTOFF_FIELDS: most
# binding first. Kept here as a small literal list rather than a
# cross-repo import — this script has no dependency on the orchestrator repo.
_DATA_CUTOFF_FIELDS = (
    "label_observation_cutoff",
    "effective_selection_cutoff_date",
    "effective_train_cutoff_date",
    "data_cutoff_date",
    "live_train_end",
    "cutoff_date",
)

_BINARY_ARTIFACT_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt", ".safetensors", ".onnx"}

# Codex review (PR #429): the generated snapshot must WHITELIST fields rather
# than serialize arbitrary artifact metadata — an artifact's metadata JSON
# could, in principle, ever carry an operational/internal key (a credential,
# a local debug path, free-form notes) alongside the legitimate provenance
# fields, and a generic "copy every key" renderer would leak it into a
# committed, widely-read doc. `_extract_allowed` below is the ONLY place
# artifact metadata values are read — never `for k, v in meta.items(): ...`.
# This constant exists so that guarantee is visible and auditable at a
# glance, and so extending the rendered fields later is a deliberate
# one-line addition here, not an accidental widening.
_ALLOWED_METADATA_FIELDS = (
    "trained_date",
    *_DATA_CUTOFF_FIELDS,
    "lookahead_days",
    "config_fingerprint",
    "kind",
    "version",
    "label_col",
    "train_run_id",
    "oos_mean_ic",
    "promotion_status",
    "feature_count",
)

# Calibrator artifacts (global_panel_calibration JSONs) use their own field
# set; same whitelist discipline. Top-level fields and metadata.* fields.
_ALLOWED_CALIBRATOR_FIELDS = ("kind", "version", "trained_date")
_ALLOWED_CALIBRATOR_METADATA_FIELDS = (
    "calibration_method",
    "method",
    "pool_ic",
    "lookahead_days",
    "scorer_model_content_fingerprint",
    "scorer_artifact_fingerprint",
    "data_window_start",
    "data_window_end",
)

# wf_gate_metadata sub-fields rendered (booleans / timestamps / dates only).
_ALLOWED_WF_GATE_FIELDS = ("passed", "run_at", "sanity_eval_end")


# --------------------------------------------------------------------------
# Small tolerant helpers
# --------------------------------------------------------------------------

def _relativize_for_display(raw: str, *, repo_root: Path) -> str:
    """Defense-in-depth (Codex review, PR #429): the rendered snapshot must
    never contain an absolute local filesystem path — today's
    strategy_config.json only ever stores repo-relative artifact paths, but
    this guards against a future hand-edited config regressing that. An
    absolute path found to be inside ``repo_root`` is rewritten relative to
    it; one found to be outside is redacted to just its basename with a
    marker, rather than ever echoing a full local path into a committed doc.
    """
    p = Path(raw)
    if not p.is_absolute():
        return raw
    try:
        return str(p.relative_to(repo_root))
    except ValueError:
        return f"<redacted-external-path>/{p.name}"


_CANONICAL_RESOLVER: Any = None


def _canonical_resolver(repo_root: Path):
    """renquant-pipeline's ONE artifact-resolution authority
    (``kernel.artifact_resolver``: the pure absolute→strategy_dir→repo_root
    contract the daily loader AND the pre-deploy CI gate (#525) both use),
    imported from the pinned runtime checkout under ``repo_root``. Memoized.

    codex #524 CR: the snapshot must not carry a SECOND, divergent resolver.
    The old strategy_dir-only join rendered the PatchTST shadow ref (which lives
    under the umbrella root, not ``strategy_dir``) as ``unknown`` while the real
    loader resolved and scored it — a green verifier over a null digest that is
    not evidence the pin restores a traceable scorer. Returns None only when the
    pinned pipeline is absent (a declaration-only context, e.g. a hosted CI
    runner with no ``.subrepo_runtime`` and no live artifacts); the caller then
    records provenance as context-unverified rather than mis-resolving it."""
    global _CANONICAL_RESOLVER
    if _CANONICAL_RESOLVER is not None:
        return _CANONICAL_RESOLVER or None
    pin_src = repo_root / ".subrepo_runtime" / "repos" / "renquant-pipeline" / "src"
    if pin_src.is_dir() and str(pin_src) not in sys.path:
        sys.path.insert(0, str(pin_src))
    try:
        from renquant_pipeline.kernel.artifact_resolver import (  # noqa: PLC0415
            locate_artifact,
        )
        _CANONICAL_RESOLVER = (locate_artifact,)
    except ImportError:
        _CANONICAL_RESOLVER = ()  # sentinel: tried, unavailable in this context
    return _CANONICAL_RESOLVER or None


def _resolve_artifact_path(raw: str, *, strategy_dir: Path, repo_root: Path) -> Path:
    """Resolve an artifact ref through the canonical pipeline resolver — the SAME
    absolute→strategy_dir→repo_root order the daily run loads with — so a
    repo-root-relative ref (e.g. the shadow ``artifacts/patchtst_shadow/…`` under
    the umbrella root, not ``strategy_dir``) resolves to the file that is actually
    scored, not a phantom ``strategy_dir/artifacts/…`` that silently renders
    provenance ``unknown`` (codex #524 CR). Falls back to the strategy_dir join
    ONLY when the pinned pipeline is unavailable (declaration-only CI) — a context
    with no live artifact to resolve, so the fallback is display-only and matches
    ``locate_artifact``'s own no-file return (the strategy_dir candidate)."""
    p = Path(raw)
    if p.is_absolute():
        return p
    resolver = _canonical_resolver(repo_root)
    if resolver is not None:
        (locate_artifact,) = resolver
        return Path(locate_artifact(raw, strategy_dir=strategy_dir, repo_root=repo_root))
    return (strategy_dir / p).resolve()


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def _metadata_file_for(artifact_path: Path) -> Path:
    """JSON artifacts (XGB/GBDT) carry metadata inline; binary checkpoints
    (.pt et al, e.g. hf_patchtst) carry it in a "<path>.metadata.json"
    sidecar (the same convention model_freshness_monitor.py's
    _freshness_path_for reads)."""
    if artifact_path.suffix in _BINARY_ARTIFACT_SUFFIXES:
        return artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")
    return artifact_path


def _sha256_file(path: Path) -> Optional[str]:
    try:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None
    return f"sha256:{digest[:16]}"


def _fmt(value: Any) -> str:
    if value is None:
        return UNKNOWN_ABSENT
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:+.4f}" if abs(value) < 1 else f"{value:.4f}"
    return str(value)


def _binding_cutoff(meta: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    for field in _DATA_CUTOFF_FIELDS:
        assert field in _ALLOWED_METADATA_FIELDS  # structural whitelist guard
        val = meta.get(field)
        if val:
            return field, str(val)[:10]
    return None, None


def _extract_allowed(meta: dict[str, Any]) -> dict[str, Any]:
    """The ONLY point in this module that reads scorer-artifact metadata
    field values — every other function receives its data from HERE, never
    from ``meta`` directly. Returns exactly the ``_ALLOWED_METADATA_FIELDS``
    keys present in ``meta`` plus two explicitly derived values
    (``n_features`` from the LENGTH of feature_cols, and the three
    whitelisted wf_gate_metadata sub-fields); anything else in the source
    metadata (whatever it might be — a credential, a local debug path,
    free-form notes) is structurally unreachable past this point."""
    out = {field: meta[field] for field in _ALLOWED_METADATA_FIELDS if field in meta}
    feature_cols = meta.get("feature_cols")
    if isinstance(feature_cols, list) and "feature_count" not in out:
        out["feature_count"] = len(feature_cols)
    nested = meta.get("metadata")
    wf_gate = nested.get("wf_gate_metadata") if isinstance(nested, dict) else None
    if isinstance(wf_gate, dict):
        out["wf_gate"] = {
            field: wf_gate[field] for field in _ALLOWED_WF_GATE_FIELDS if field in wf_gate
        }
    return out


def _extract_allowed_calibrator(doc: dict[str, Any]) -> dict[str, Any]:
    """Whitelist boundary for calibrator artifacts — same discipline as
    ``_extract_allowed``."""
    out = {field: doc[field] for field in _ALLOWED_CALIBRATOR_FIELDS if field in doc}
    nested = doc.get("metadata")
    if isinstance(nested, dict):
        for field in _ALLOWED_CALIBRATOR_METADATA_FIELDS:
            if field in nested and field not in out:
                out[field] = nested[field]
    if "method" not in out and "calibration_method" in out:
        out["method"] = out["calibration_method"]
    return out


def _read_pinned_checkout_head(git_dir: Path) -> Optional[str]:
    """Resolve the pinned runtime checkout's HEAD commit via PLAIN FILE
    READS only — this script must never run a git command against the live
    tree (hard operational rule; see doc/arch/subrepo-operating-model.md)."""
    try:
        if git_dir.is_file():  # worktree/submodule indirection: "gitdir: <path>"
            text = git_dir.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                indirect = Path(text.split(":", 1)[1].strip())
                if not indirect.is_absolute():
                    indirect = (git_dir.parent / indirect).resolve()
                git_dir = indirect
            else:
                return None
        head_file = git_dir / "HEAD"
        if not head_file.exists():
            return None
        head = head_file.read_text(encoding="utf-8").strip()
        if not head.startswith("ref:"):
            return head or None
        ref = head.split(None, 1)[1].strip()
        ref_file = git_dir / ref
        if ref_file.exists():
            return ref_file.read_text(encoding="utf-8").strip() or None
        packed = git_dir / "packed-refs"
        if packed.exists():
            for line in packed.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith(("#", "^")):
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
    except OSError:
        return None
    return None


# --------------------------------------------------------------------------
# Collection
# --------------------------------------------------------------------------

def _describe_model(
    *, role: str, kind: Optional[str], artifact_rel: Optional[str],
    strategy_dir: Path, repo_root: Path, name: Optional[str] = None,
    sources: Optional[dict[str, Optional[str]]] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": role, "name": name, "kind": kind,
        "artifact_path": (
            _relativize_for_display(artifact_rel, repo_root=repo_root)
            if artifact_rel else artifact_rel
        ),
        "trained_date": None, "binding_cutoff_field": None, "binding_cutoff": None,
        "label_observation_cutoff": None,
        "lookahead_days": None, "config_fingerprint": None,
        "label_col": None, "train_run_id": None, "oos_mean_ic": None,
        "promotion_status": None, "feature_count": None,
        "wf_gate": None,
        "metadata_source": None, "artifact_file_fingerprint": None,
        "metadata_missing": False,
    }
    if not artifact_rel:
        return row
    resolved = _resolve_artifact_path(artifact_rel, strategy_dir=strategy_dir, repo_root=repo_root)
    metadata_file = _metadata_file_for(resolved)
    if sources is not None:
        sources[_relativize_for_display(str(metadata_file), repo_root=repo_root)] = (
            _sha256_file(metadata_file)
        )
    row["artifact_file_fingerprint"] = _sha256_file(metadata_file)
    raw_meta = _load_json(metadata_file)
    if not raw_meta:
        row["metadata_missing"] = True
        return row
    meta = _extract_allowed(raw_meta)  # whitelist boundary — see docstring
    row["trained_date"] = meta.get("trained_date")
    row["binding_cutoff_field"], row["binding_cutoff"] = _binding_cutoff(meta)
    row["label_observation_cutoff"] = meta.get("label_observation_cutoff")
    row["lookahead_days"] = meta.get("lookahead_days")
    row["config_fingerprint"] = meta.get("config_fingerprint")
    row["label_col"] = meta.get("label_col")
    row["train_run_id"] = meta.get("train_run_id")
    row["oos_mean_ic"] = meta.get("oos_mean_ic")
    row["promotion_status"] = meta.get("promotion_status")
    row["feature_count"] = meta.get("feature_count")
    row["wf_gate"] = meta.get("wf_gate")
    row["metadata_source"] = (
        "sidecar" if resolved.suffix in _BINARY_ARTIFACT_SUFFIXES else "inline"
    )
    return row


def _describe_calibrator(
    *, artifact_rel: Optional[str], strategy_dir: Path, repo_root: Path,
    sources: Optional[dict[str, Optional[str]]] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "artifact_path": (
            _relativize_for_display(artifact_rel, repo_root=repo_root)
            if artifact_rel else artifact_rel
        ),
        "kind": None, "trained_date": None, "method": None, "pool_ic": None,
        "lookahead_days": None, "scorer_model_content_fingerprint": None,
        "data_window_start": None, "data_window_end": None,
        "artifact_file_fingerprint": None, "metadata_missing": False,
    }
    if not artifact_rel:
        return row
    resolved = _resolve_artifact_path(artifact_rel, strategy_dir=strategy_dir, repo_root=repo_root)
    if sources is not None:
        sources[_relativize_for_display(str(resolved), repo_root=repo_root)] = (
            _sha256_file(resolved)
        )
    row["artifact_file_fingerprint"] = _sha256_file(resolved)
    doc = _load_json(resolved)
    if not doc:
        row["metadata_missing"] = True
        return row
    allowed = _extract_allowed_calibrator(doc)  # whitelist boundary
    for field in ("kind", "trained_date", "method", "pool_ic", "lookahead_days",
                  "scorer_model_content_fingerprint", "data_window_start",
                  "data_window_end"):
        row[field] = allowed.get(field)
    return row


def _policy_knobs(config: dict[str, Any]) -> dict[str, Any]:
    """Explicit, named scalar knobs only — never free-form ``_reason`` prose
    (which can carry local paths / narrative that belongs in dated docs)."""
    def get(*path: str) -> Any:
        node: Any = config
        for key in path:
            if not isinstance(node, dict):
                return None
            node = node.get(key)
        return node

    regimes: dict[str, dict[str, Any]] = {}
    regime_params = config.get("regime_params")
    if isinstance(regime_params, dict):
        for regime in sorted(regime_params):
            params = regime_params[regime]
            if not isinstance(params, dict):
                continue
            row = {
                "max_position_pct": params.get("max_position_pct"),
                "qp_turnover_max": params.get("qp_turnover_max"),
                "cash_reserve_pct": params.get("cash_reserve_pct"),
                "stop_loss_pct": params.get("stop_loss_pct"),
            }
            if all(value is None for value in row.values()):
                continue  # regime declares none of the rendered caps — skip the noise
            regimes[regime] = row

    return {
        "conviction_gate_enabled": get("ranking", "panel_scoring", "conviction_gate", "enabled"),
        "mu_floor": get("ranking", "panel_scoring", "conviction_gate", "mu_floor"),
        "demean_cross_sectional": get(
            "ranking", "panel_scoring", "conviction_gate", "demean_cross_sectional"),
        "signal_gate_prefer_calibrated_mu": get(
            "ranking", "panel_scoring", "signal_gate_prefer_calibrated_mu"),
        "buy_floor": get("ranking", "panel_scoring", "buy_floor"),
        "buy_floor_min": get("ranking", "panel_scoring", "buy_floor_min"),
        "panel_buy_top_n": get("rotation", "panel_buy_top_n"),
        "rotation_min_expected_advantage_pct": get("rotation", "min_expected_advantage_pct"),
        "rotation_target_horizon_days": get("rotation", "target_horizon_days"),
        "kelly_enabled": get("ranking", "kelly_sizing", "enabled"),
        "kelly_fractional": get("ranking", "kelly_sizing", "fractional"),
        "kelly_max_concentration": get("ranking", "kelly_sizing", "max_concentration"),
        "kelly_min_edge": get("ranking", "kelly_sizing", "min_edge"),
        "kelly_use_calibrator_mu": get("ranking", "kelly_sizing", "use_calibrator_mu"),
        "max_concurrent_positions": config.get("max_concurrent_positions"),
        "max_position_pct": get("position_sizing", "max_position_pct"),
        "max_positions_per_sector": config.get("max_positions_per_sector"),
        "model_staleness_days": config.get("model_staleness_days"),
        "qp_risk_aversion": get("rotation", "joint_actions", "qp_risk_aversion"),
        "qp_turnover_max": get("rotation", "joint_actions", "qp_turnover_max"),
        "qp_no_trade_band_cap": get("rotation", "joint_actions", "qp_no_trade_band_cap"),
        "qp_mu_horizon_days": get("rotation", "joint_actions", "qp_mu_horizon_days"),
        "qp_admission_min_rank_score": get(
            "rotation", "joint_actions", "qp_admission_gate", "min_rank_score"),
        "wf_gate_benchmark_required": get("wf_gate", "benchmark_required"),
        "wf_gate_regime_required": get("wf_gate", "regime_required"),
        "wf_gate_sanity_regime_ic_required": get("wf_gate", "sanity_regime_ic_required"),
        "regimes": regimes,
    }


def _lock_pins(lock: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not lock or not isinstance(lock.get("subrepos"), list):
        return []
    pins = []
    for entry in lock["subrepos"]:
        if not isinstance(entry, dict):
            continue
        pins.append({
            "name": entry.get("name"),
            "branch": entry.get("branch"),
            "commit": entry.get("commit"),
            "status": entry.get("status"),
        })
    return sorted(pins, key=lambda p: str(p["name"]))


def collect_snapshot(
    repo_root: Path,
    *,
    configs_dir: Optional[Path] = None,
    lock_path: Optional[Path] = None,
    strategy_dir: Optional[Path] = None,
    pinned_git_dir: Optional[Path] = None,
    source_pin: Optional[str] = None,
) -> dict[str, Any]:
    """Pure collection: sources under ``repo_root`` -> snapshot dict. Only
    reads files; never writes, never invokes git. Every path can be
    overridden for tests / CI fixtures."""
    repo_root = repo_root.resolve()
    configs_dir = (configs_dir or repo_root / PINNED_CONFIGS_REL).resolve()
    lock_path = (lock_path or repo_root / LOCK_FILE_REL).resolve()
    strategy_dir = (strategy_dir or repo_root / STRATEGY_DIR_REL).resolve()
    if pinned_git_dir is None:
        pinned_git_dir = repo_root / PINNED_GIT_DIR_REL

    sources: dict[str, Optional[str]] = {}

    active_config_path = configs_dir / ACTIVE_CONFIG_NAME
    shadow_config_path = configs_dir / SHADOW_CONFIG_NAME
    active_config = _load_json(active_config_path)
    shadow_config = _load_json(shadow_config_path)
    lock = _load_json(lock_path)
    for path in (active_config_path, shadow_config_path, lock_path):
        sources[_relativize_for_display(str(path), repo_root=repo_root)] = _sha256_file(path)

    warnings: list[str] = []
    if active_config is None:
        warnings.append(
            f"pinned active config unreadable or missing: "
            f"{_relativize_for_display(str(active_config_path), repo_root=repo_root)}"
        )
    if shadow_config is None:
        warnings.append(
            f"pinned shadow config unreadable or missing: "
            f"{_relativize_for_display(str(shadow_config_path), repo_root=repo_root)}"
        )
    if lock is None:
        warnings.append(
            f"lock file unreadable or missing: "
            f"{_relativize_for_display(str(lock_path), repo_root=repo_root)}"
        )

    # Pin coupling: the runtime checkout the configs came from vs the lock.
    # ``source_pin`` (from --source-pin) renders the snapshot as a PRE-DEPLOY
    # interface declaration from a CANDIDATE lock assembly (a strategy-104
    # checkout at the lock pin), so the snapshot never depends on a
    # post-deploy/pin-aligned live runtime (release-contract requirement).
    runtime_head = source_pin or _read_pinned_checkout_head(pinned_git_dir)
    lock_pin = None
    if lock and isinstance(lock.get("subrepos"), list):
        for entry in lock["subrepos"]:
            if isinstance(entry, dict) and entry.get("name") == PINNED_SUBREPO_NAME:
                lock_pin = entry.get("commit")
                break
    if runtime_head and lock_pin and runtime_head != lock_pin:
        warnings.append(
            f"{PIN_DRIFT_MARKER}: pinned runtime checkout HEAD {runtime_head} != "
            f"subrepos.lock.json {PINNED_SUBREPO_NAME} pin {lock_pin}"
        )

    panel_scoring = ((active_config or {}).get("ranking") or {}).get("panel_scoring") or {}
    active = _describe_model(
        role="active", kind=panel_scoring.get("kind"),
        artifact_rel=panel_scoring.get("artifact_path"),
        strategy_dir=strategy_dir, repo_root=repo_root, sources=sources,
    )
    # The pooled calibrator is a per-promote RE-FIT live artifact (uncommitted,
    # mutable). codex #524 CR: a candidate-lock snapshot cannot fold mutable live
    # artifact state into its reproducible Source fingerprint — else the same
    # candidate pin renders a different fingerprint whenever the calibrator is
    # re-fit. So the calibrator is NOT passed `sources` (excluded from the
    # candidate-interface fingerprint); its digest is recorded on the row as a
    # time-stamped runtime OBSERVATION (rendered under a distinct section).
    active_calibrator = _describe_calibrator(
        artifact_rel=(panel_scoring.get("global_calibration") or {}).get("artifact_path"),
        strategy_dir=strategy_dir, repo_root=repo_root,
    )
    in_run_shadows = [
        _describe_model(
            role="shadow", kind=sm.get("kind"), artifact_rel=sm.get("artifact_path"),
            strategy_dir=strategy_dir, repo_root=repo_root, name=sm.get("name"),
            sources=sources,
        )
        for sm in (panel_scoring.get("shadow_models") or [])
        if isinstance(sm, dict)
    ]

    shadow_scoring = ((shadow_config or {}).get("ranking") or {}).get("panel_scoring") or {}
    shadow_e2e = _describe_model(
        role="shadow-e2e", kind=shadow_scoring.get("kind"),
        artifact_rel=shadow_scoring.get("artifact_path"),
        strategy_dir=strategy_dir, repo_root=repo_root, sources=sources,
    )
    shadow_e2e_calibrator = _describe_calibrator(  # runtime observation, not in fingerprint (see active_calibrator)
        artifact_rel=(shadow_scoring.get("global_calibration") or {}).get("artifact_path"),
        strategy_dir=strategy_dir, repo_root=repo_root,
    )

    # codex #524 CR: a CONFIGURED scorer whose artifact does not resolve to a
    # real file has no verifiable identity — the pin does not provably restore a
    # traceable scorer, and a green verifier over that null digest is not
    # evidence. Surface it as a warning so the pre-deploy snapshot is never
    # silently green over a required-scorer-rendered-``unknown``. Gated on the
    # canonical resolver being available (i.e. a live/artifact-bearing context);
    # a declaration-only CI render with no ``.subrepo_runtime`` has no artifacts
    # to resolve and is not expected to. Calibrators are runtime observations
    # (excluded above) and are intentionally exempt.
    if _canonical_resolver(repo_root) is not None:
        for _m in (active, shadow_e2e, *in_run_shadows):
            if _m.get("kind") and _m.get("artifact_path") and _m.get("metadata_missing"):
                warnings.append(
                    f"SCORER PROVENANCE UNRESOLVED: {_m.get('role')} scorer "
                    f"{_m.get('name') or _m.get('kind')} artifact "
                    f"{_m.get('artifact_path')!r} did not resolve to a metadata-"
                    "bearing file under the canonical resolver — no digest; the "
                    "candidate pin does not provably restore a traceable scorer"
                )

    # Known rot vector: the umbrella WORKING-COPY config that the previous
    # snapshot version treated as canonical. Flag when it disagrees with the
    # pinned config so the divergence is visible in the committed doc.
    umbrella_config_path = strategy_dir / ACTIVE_CONFIG_NAME
    umbrella_config = _load_json(umbrella_config_path)
    if umbrella_config is not None and active_config is not None:
        umbrella_kind = ((umbrella_config.get("ranking") or {}).get("panel_scoring") or {}).get("kind")
        if umbrella_kind != panel_scoring.get("kind"):
            warnings.append(
                "UMBRELLA WORKING-COPY DRIFT: "
                f"{_relativize_for_display(str(umbrella_config_path), repo_root=repo_root)} "
                f"declares kind={umbrella_kind!r} but the pinned config declares "
                f"kind={panel_scoring.get('kind')!r} — the pinned config is what "
                "the daily run consumes; the working copy is stale"
            )

    watchlist = (active_config or {}).get("watchlist") or []

    return {
        "configs_dir": _relativize_for_display(str(configs_dir), repo_root=repo_root),
        "runtime_checkout_commit": runtime_head,
        "lock_strategy_104_pin": lock_pin,
        "active": active,
        "active_calibrator": active_calibrator,
        "in_run_shadows": in_run_shadows,
        "shadow_e2e": shadow_e2e,
        "shadow_e2e_calibrator": shadow_e2e_calibrator,
        "policy": _policy_knobs(active_config or {}),
        "pins": _lock_pins(lock),
        "watchlist_size": len(watchlist),
        "warnings": warnings,
        "sources": dict(sorted(sources.items())),
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _model_rows(row: dict[str, Any]) -> list[tuple[str, str]]:
    cutoff = (
        f"{row['binding_cutoff_field']}={row['binding_cutoff']}"
        if row.get("binding_cutoff_field") else UNKNOWN_ABSENT
    )
    wf_gate = row.get("wf_gate") or {}
    wf = UNKNOWN_ABSENT
    if wf_gate:
        wf = (
            f"passed={_fmt(wf_gate.get('passed'))}; run_at={_fmt(wf_gate.get('run_at'))}; "
            f"sanity_eval_end={_fmt(wf_gate.get('sanity_eval_end'))}"
        )
    rows = [
        ("Scorer kind", f"`{row['kind']}`" if row.get("kind") else UNKNOWN_ABSENT),
        ("Artifact", f"`{row['artifact_path']}`" if row.get("artifact_path") else UNKNOWN_ABSENT),
        ("Artifact metadata file fingerprint", _fmt(row.get("artifact_file_fingerprint"))
         if row.get("artifact_file_fingerprint")
         else "unknown (file missing)"),
        ("trained_date", _fmt(row.get("trained_date"))),
        ("Binding data cutoff", cutoff),
        ("label_observation_cutoff", _fmt(row.get("label_observation_cutoff"))),
        ("lookahead_days", _fmt(row.get("lookahead_days"))),
        ("label_col", _fmt(row.get("label_col"))),
        ("Feature count", _fmt(row.get("feature_count"))),
        ("train_run_id", _fmt(row.get("train_run_id"))),
        ("oos_mean_ic (stamped)", _fmt(row.get("oos_mean_ic"))),
        ("promotion_status", _fmt(row.get("promotion_status"))),
        ("config_fingerprint", _fmt(row.get("config_fingerprint"))),
        ("WF gate (stamped)", wf),
    ]
    if row.get("name"):
        rows.insert(0, ("Name", f"`{row['name']}`"))
    if row.get("metadata_missing"):
        rows.append(("Metadata", "unknown (metadata file missing or unreadable)"))
    return rows


def _calibrator_rows(row: dict[str, Any]) -> list[tuple[str, str]]:
    window = UNKNOWN_ABSENT
    if row.get("data_window_start") or row.get("data_window_end"):
        window = f"{_fmt(row.get('data_window_start'))} → {_fmt(row.get('data_window_end'))}"
    rows = [
        ("Artifact", f"`{row['artifact_path']}`" if row.get("artifact_path") else UNKNOWN_ABSENT),
        ("Artifact file fingerprint", _fmt(row.get("artifact_file_fingerprint"))
         if row.get("artifact_file_fingerprint") else "unknown (file missing)"),
        ("kind", _fmt(row.get("kind"))),
        ("trained_date", _fmt(row.get("trained_date"))),
        ("Method", _fmt(row.get("method"))),
        ("pool_ic (stamped)", _fmt(row.get("pool_ic"))),
        ("lookahead_days", _fmt(row.get("lookahead_days"))),
        ("Bound scorer content fingerprint", _fmt(row.get("scorer_model_content_fingerprint"))),
        ("Fit data window", window),
    ]
    if row.get("metadata_missing"):
        rows.append(("Metadata", "unknown (file missing or unreadable)"))
    return rows


def _table(rows: list[tuple[str, str]]) -> list[str]:
    lines = ["| | |", "|---|---|"]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    return lines


def _machine_block(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "strategy_104_pin": snapshot.get("runtime_checkout_commit"),
        "lock_strategy_104_pin": snapshot.get("lock_strategy_104_pin"),
        "active_kind": (snapshot.get("active") or {}).get("kind"),
        "active_artifact": (snapshot.get("active") or {}).get("artifact_path"),
        "in_run_shadow_kinds": [
            row.get("kind") for row in snapshot.get("in_run_shadows") or []
        ],
        "shadow_e2e_kind": (snapshot.get("shadow_e2e") or {}).get("kind"),
        "sources_sha256": snapshot.get("sources") or {},
    }


def _source_fingerprint(sources: dict[str, Optional[str]]) -> str:
    """Deterministic aggregate over the per-file source hashes — changes iff
    the actual pinned/artifact content changes, unlike a wall-clock
    timestamp (which churned the committed doc on every regeneration with
    zero semantic change, per Codex review round 3)."""
    joined = "\n".join(f"{path}:{digest or ''}" for path, digest in sorted(sources.items()))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def render_markdown(snapshot: dict[str, Any], *, generated_at: Optional[str] = None) -> str:
    # `generated_at` is accepted for backward-compatible call sites (tests,
    # --selftest) but no longer rendered: a wall-clock value churned the
    # committed doc on every regeneration with zero semantic change (Codex
    # review round 3). The doc instead carries a deterministic fingerprint
    # over the actual source hashes below.
    del generated_at
    policy = snapshot.get("policy") or {}
    lines = [
        "# renquant_104 — generated production snapshot",
        "",
        "GENERATED FILE — do not hand-edit. Regenerate with:",
        "`python3 scripts/render_strategy_104_snapshot.py` (or `make snapshot`);",
        "verify with `make snapshot-check`.",
        "",
        "Rendered from the PINNED strategy-104 config (the "
        "`.subrepo_runtime/repos/renquant-strategy-104/configs/` checkout that the",
        "daily run actually consumes — NOT the umbrella working copy, which is a",
        "known rot vector) plus each referenced artifact's own stamped metadata —",
        "see amendment A6, doc/design/2026-07-01-104-105-design-review-amendments.md",
        "and the unified 107 master plan M9. This states ONLY what the pinned",
        "sources say AS OF the last regeneration — a current fact, never a",
        "historical/promotion claim (\"active since <date>\", \"promoted on <date>\");",
        "that narrative, with its own dating and provenance, belongs in",
        "doc/arch/strategy-104.md instead. Fields the sources do not stamp are",
        "rendered as explicit unknowns, never invented.",
        "",
        f"{SOURCE_FINGERPRINT_PREFIX} "
        f"{_source_fingerprint(snapshot.get('sources') or {})} "
        "(sha256 over the sorted per-file source hashes below — deterministic;"
        " changes iff pinned/artifact CONTENT changes, never on a bare"
        " regeneration. EXCLUDES the pooled calibrators: they are re-fit per"
        " promote (mutable live state) and are recorded below as runtime"
        " observations, NOT folded into this candidate-interface fingerprint)",
        "",
        "## Provenance",
        "",
    ]
    lines.extend(_table([
        ("Pinned config root", f"`{_fmt(snapshot.get('configs_dir'))}`"),
        ("strategy-104 runtime checkout commit", _fmt(snapshot.get("runtime_checkout_commit"))),
        ("subrepos.lock.json strategy-104 pin", _fmt(snapshot.get("lock_strategy_104_pin"))),
    ]))
    warnings = snapshot.get("warnings") or []
    if warnings:
        lines.extend(["", "### Source warnings", ""])
        lines.extend(f"- **{warning}**" for warning in warnings)
    lines.extend(["", "## Active scorer", ""])
    lines.extend(_table(_model_rows(snapshot["active"])))
    lines.extend(["", "## Active calibrator", "",
                  "> Runtime observation, not a locked identity: the pooled"
                  " calibrator is re-fit per promote, so the digest below"
                  " reflects the live artifact as of this regeneration and is"
                  " EXCLUDED from the candidate Source fingerprint above.", ""])
    lines.extend(_table(_calibrator_rows(snapshot["active_calibrator"])))
    lines.extend(["", "## In-run shadow scorers (readonly, same run)", ""])
    if snapshot.get("in_run_shadows"):
        for row in snapshot["in_run_shadows"]:
            lines.extend(_table(_model_rows(row)))
            lines.append("")
        lines.pop()
    else:
        lines.append("(none configured)")
    lines.extend(["", "## Shadow e2e config (strategy_config.shadow.json)", ""])
    lines.extend(_table(_model_rows(snapshot["shadow_e2e"])))
    lines.extend(["", "### Shadow e2e calibrator", "",
                  "> Runtime observation (re-fit per promote), excluded from the"
                  " candidate Source fingerprint — same as the active calibrator.",
                  ""])
    lines.extend(_table(_calibrator_rows(snapshot["shadow_e2e_calibrator"])))
    lines.extend(["", "## Key policy knobs (active pinned config)", ""])
    lines.extend(_table([
        ("Watchlist size", f"{snapshot.get('watchlist_size')} tickers"),
        ("Conviction gate μ floor",
         f"enabled={_fmt(policy.get('conviction_gate_enabled'))}; "
         f"mu_floor={_fmt(policy.get('mu_floor'))}; "
         f"demean_cross_sectional={_fmt(policy.get('demean_cross_sectional'))}"),
        ("signal_gate_prefer_calibrated_mu",
         _fmt(policy.get("signal_gate_prefer_calibrated_mu"))),
        ("Buy floor",
         f"mode={_fmt(policy.get('buy_floor'))}; min={_fmt(policy.get('buy_floor_min'))}"),
        ("panel_buy_top_n", _fmt(policy.get("panel_buy_top_n"))),
        ("Rotation",
         f"min_expected_advantage_pct={_fmt(policy.get('rotation_min_expected_advantage_pct'))}; "
         f"target_horizon_days={_fmt(policy.get('rotation_target_horizon_days'))}"),
        ("Kelly sizing",
         f"enabled={_fmt(policy.get('kelly_enabled'))}; "
         f"fractional={_fmt(policy.get('kelly_fractional'))}; "
         f"max_concentration={_fmt(policy.get('kelly_max_concentration'))}; "
         f"min_edge={_fmt(policy.get('kelly_min_edge'))}; "
         f"use_calibrator_mu={_fmt(policy.get('kelly_use_calibrator_mu'))}"),
        ("Position caps",
         f"max_concurrent_positions={_fmt(policy.get('max_concurrent_positions'))}; "
         f"max_position_pct={_fmt(policy.get('max_position_pct'))}; "
         f"max_positions_per_sector={_fmt(policy.get('max_positions_per_sector'))}"),
        ("model_staleness_days", _fmt(policy.get("model_staleness_days"))),
        ("QP",
         f"risk_aversion={_fmt(policy.get('qp_risk_aversion'))}; "
         f"turnover_max={_fmt(policy.get('qp_turnover_max'))}; "
         f"no_trade_band_cap={_fmt(policy.get('qp_no_trade_band_cap'))}; "
         f"mu_horizon_days={_fmt(policy.get('qp_mu_horizon_days'))}; "
         f"admission_min_rank_score={_fmt(policy.get('qp_admission_min_rank_score'))}"),
        ("WF gate relaxations (lock-declared)",
         f"benchmark_required={_fmt(policy.get('wf_gate_benchmark_required'))}; "
         f"regime_required={_fmt(policy.get('wf_gate_regime_required'))}; "
         f"sanity_regime_ic_required={_fmt(policy.get('wf_gate_sanity_regime_ic_required'))}"),
    ]))
    regimes = policy.get("regimes") or {}
    if regimes:
        lines.extend(["", "### Per-regime caps", "",
                      "| Regime | max_position_pct | qp_turnover_max | cash_reserve_pct | stop_loss_pct |",
                      "|---|---|---|---|---|"])
        for regime, params in regimes.items():  # already sorted at collection
            lines.append(
                f"| {regime} | {_fmt(params.get('max_position_pct'))} "
                f"| {_fmt(params.get('qp_turnover_max'))} "
                f"| {_fmt(params.get('cash_reserve_pct'))} "
                f"| {_fmt(params.get('stop_loss_pct'))} |"
            )
    lines.extend(["", "## Subrepo pins (subrepos.lock.json)", ""])
    pins = snapshot.get("pins") or []
    if pins:
        lines.extend(["| Subrepo | Branch | Commit | Status |", "|---|---|---|---|"])
        for pin in pins:
            commit = pin.get("commit")
            commit_short = str(commit)[:12] if commit else UNKNOWN_ABSENT
            lines.append(
                f"| {_fmt(pin.get('name'))} | {_fmt(pin.get('branch'))} "
                f"| `{commit_short}` | {_fmt(pin.get('status'))} |"
            )
    else:
        lines.append("(lock file missing or empty)")
    lines.extend(["", "## Source fingerprints", ""])
    for path, digest in (snapshot.get("sources") or {}).items():
        lines.append(f"- `{path}` — {digest if digest else 'unknown (file missing)'}")
    lines.extend([
        "",
        MACHINE_BLOCK_BEGIN,
        json.dumps(_machine_block(snapshot), indent=1, sort_keys=True),
        MACHINE_BLOCK_END,
        "",
    ])
    return "\n".join(lines)




def parse_machine_block(text: str) -> Optional[dict[str, Any]]:
    match = re.search(
        re.escape(MACHINE_BLOCK_BEGIN) + r"\n(.*?)\n" + re.escape(MACHINE_BLOCK_END),
        text, re.DOTALL,
    )
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# --------------------------------------------------------------------------
# CI semantic verification (no live tree required)
# --------------------------------------------------------------------------

#: One rendered row of the "## Subrepo pins (subrepos.lock.json)" table, in the
#: exact shape ``render``/``_fmt`` emits: ``| name | branch | `sha12` | status |``.
_PIN_ROW_RE = re.compile(
    r"^\|\s*(?P<name>[^|]+?)\s*\|\s*(?P<branch>[^|]+?)\s*\|\s*`(?P<commit>[^`]*)`\s*\|"
)


def parse_rendered_pin_rows(text: str) -> dict[str, str]:
    """The `{subrepo: short_commit}` a reader of the committed snapshot sees.

    Parsed from the RENDERED table rather than the machine block, because the
    rendered table is the thing an operator actually reads — and it is the
    thing that went stale (see `verify_pinned_declaration`).
    """
    rows: dict[str, str] = {}
    in_section = False
    for line in text.splitlines():
        if line.startswith("## Subrepo pins"):
            in_section = True
            continue
        if in_section:
            if line.startswith("## "):
                break
            m = _PIN_ROW_RE.match(line)
            if m and m.group("name") not in {"Subrepo", "---"}:
                rows[m.group("name")] = m.group("commit")
    return rows


def verify_pinned_declaration(
    *, snapshot_path: Path, configs_dir: Path, lock_path: Path,
) -> list[str]:
    """Compare the committed snapshot's machine block against a strategy-104
    ``configs/`` directory (checked out at the ``subrepos.lock.json`` pin by
    CI) + the lock itself. Catches the A6 rot class at the moment it is
    introduced: a lock-pin bump that changes the production declaration
    without a regenerated snapshot. Artifact metadata is intentionally NOT
    compared here — hosted CI has no live artifact tree; `make
    snapshot-check` on the operator machine covers that byte-exactly."""
    problems: list[str] = []
    text = snapshot_path.read_text(encoding="utf-8") if snapshot_path.exists() else None
    if text is None:
        return [f"snapshot missing: {snapshot_path}"]
    block = parse_machine_block(text)
    if block is None:
        return [f"snapshot has no parseable machine block: {snapshot_path}"]

    lock = _load_json(lock_path)
    lock_pin = None
    if lock and isinstance(lock.get("subrepos"), list):
        for entry in lock["subrepos"]:
            if isinstance(entry, dict) and entry.get("name") == PINNED_SUBREPO_NAME:
                lock_pin = entry.get("commit")
                break
    if lock_pin is None:
        problems.append(f"lock file has no {PINNED_SUBREPO_NAME} pin: {lock_path}")
    elif block.get("strategy_104_pin") != lock_pin:
        problems.append(
            f"snapshot was generated from strategy-104 pin "
            f"{block.get('strategy_104_pin')} but subrepos.lock.json pins {lock_pin} "
            "— regenerate the snapshot (make snapshot) as part of this pin bump"
        )

    # EVERY rendered pin row, not just strategy-104's [codex on RenQuant#596].
    # The check above reads one field of the machine block, so a bump to any
    # OTHER subrepo regenerated nothing and failed nothing: RenQuant#596 shipped
    # a snapshot whose table still said renquant-pipeline f9f488d59759 and
    # renquant-backtesting 8c2c44564957 while the lock in the same commit
    # pinned 3d9c7fb17c75 and e5f9bae3b1e2, and this verifier passed it green.
    # A declaration that renders the lock must be checked against the whole
    # lock, or it certifies only the row someone happened to think of.
    rendered = parse_rendered_pin_rows(text)
    expected = {
        str(p["name"]): str(p["commit"])[:12]
        for p in _lock_pins(lock)
        if p.get("name") and p.get("commit")
    }
    if expected and not rendered:
        problems.append(
            "snapshot renders no subrepo-pin rows while the lock declares "
            f"{len(expected)} — regenerate the snapshot (make snapshot)"
        )
    for name, want in sorted(expected.items()):
        got = rendered.get(name)
        if got is None:
            problems.append(
                f"snapshot's subrepo-pin table is missing {name}, which the lock "
                "pins — regenerate the snapshot (make snapshot)"
            )
        elif got != want:
            problems.append(
                f"snapshot declares {name} at {got} but subrepos.lock.json pins "
                f"{want} — regenerate the snapshot (make snapshot) as part of "
                "this pin bump"
            )
    for name in sorted(set(rendered) - set(expected)):
        problems.append(
            f"snapshot declares a subrepo pin for {name}, which the lock does not "
            "contain — regenerate the snapshot (make snapshot)"
        )

    config = _load_json(configs_dir / ACTIVE_CONFIG_NAME)
    if config is None:
        problems.append(f"cannot read pinned config: {configs_dir / ACTIVE_CONFIG_NAME}")
        return problems
    panel_scoring = (config.get("ranking") or {}).get("panel_scoring") or {}
    if block.get("active_kind") != panel_scoring.get("kind"):
        problems.append(
            f"snapshot active kind {block.get('active_kind')!r} != pinned config "
            f"kind {panel_scoring.get('kind')!r}"
        )
    if block.get("active_artifact") != panel_scoring.get("artifact_path"):
        problems.append(
            f"snapshot active artifact {block.get('active_artifact')!r} != pinned "
            f"config artifact {panel_scoring.get('artifact_path')!r}"
        )
    declared_shadow_kinds = [
        sm.get("kind") for sm in (panel_scoring.get("shadow_models") or [])
        if isinstance(sm, dict)
    ]
    if list(block.get("in_run_shadow_kinds") or []) != declared_shadow_kinds:
        problems.append(
            f"snapshot in-run shadow kinds {block.get('in_run_shadow_kinds')!r} != "
            f"pinned config shadow kinds {declared_shadow_kinds!r}"
        )
    shadow_config = _load_json(configs_dir / SHADOW_CONFIG_NAME)
    if shadow_config is not None:
        shadow_kind = ((shadow_config.get("ranking") or {}).get("panel_scoring") or {}).get("kind")
        if block.get("shadow_e2e_kind") != shadow_kind:
            problems.append(
                f"snapshot shadow-e2e kind {block.get('shadow_e2e_kind')!r} != pinned "
                f"shadow config kind {shadow_kind!r}"
            )
    return problems


# --------------------------------------------------------------------------
# Selftest fixture
# --------------------------------------------------------------------------

def _write_selftest_fixture(root: Path) -> None:
    configs = root / PINNED_CONFIGS_REL
    configs.mkdir(parents=True)
    git_dir = root / PINNED_GIT_DIR_REL
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("f" * 40 + "\n", encoding="utf-8")
    strategy_dir = root / STRATEGY_DIR_REL
    (strategy_dir / "artifacts" / "prod").mkdir(parents=True)
    (strategy_dir / "artifacts" / "shadow").mkdir(parents=True)

    (strategy_dir / "artifacts" / "prod" / "scorer.json").write_text(json.dumps({
        "kind": "panel_ltr_xgboost", "trained_date": "2026-05-18",
        "lookahead_days": 60, "label_col": "fwd_60d_excess",
        "config_fingerprint": "sha256:f8fb2259b2bf1537",
        "train_run_id": "selftest_run", "oos_mean_ic": 0.0447,
        "promotion_status": "gated_buys",
        "feature_cols": ["f1", "f2", "f3"],
        "metadata": {"wf_gate_metadata": {
            "passed": True, "run_at": "2026-05-30T16:36:38", "sanity_eval_end": "2026-02-10",
        }},
        "api_key": "sk-selftest-must-never-render",
    }), encoding="utf-8")
    (strategy_dir / "artifacts" / "prod" / "calib.json").write_text(json.dumps({
        "kind": "global_panel_calibration", "trained_date": "2026-07-01", "version": 1,
        "metadata": {"method": "platt", "pool_ic": 0.0993, "lookahead_days": 60,
                     "scorer_model_content_fingerprint": "sha256:9c4bbd74b51adc17"},
    }), encoding="utf-8")
    ckpt = strategy_dir / "artifacts" / "shadow" / "model.pt"
    ckpt.write_bytes(b"not a real checkpoint")
    ckpt.with_suffix(".pt.metadata.json").write_text(json.dumps({
        "trained_date": "2026-05-22", "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60,
        "config_fingerprint": "sha256:f8fb2259b2bf1537", "feature_count": 172,
    }), encoding="utf-8")

    (configs / ACTIVE_CONFIG_NAME).write_text(json.dumps({
        "watchlist": ["AAPL", "MSFT", "NVDA"],
        "max_concurrent_positions": 8,
        "model_staleness_days": 60,
        "position_sizing": {"max_position_pct": 0.15},
        "regime_params": {"BULL_CALM": {"max_position_pct": 0.12,
                                         "qp_turnover_max": 0.15,
                                         "cash_reserve_pct": 0,
                                         "stop_loss_pct": 0.15}},
        "rotation": {"panel_buy_top_n": 3, "min_expected_advantage_pct": 0.06,
                     "target_horizon_days": 60,
                     "joint_actions": {"qp_risk_aversion": 3, "qp_turnover_max": 0.2,
                                        "qp_no_trade_band_cap": 0.05,
                                        "qp_mu_horizon_days": 60,
                                        "qp_admission_gate": {"min_rank_score": 0.55}}},
        "wf_gate": {"benchmark_required": False},
        "ranking": {
            "panel_scoring": {
                "kind": "xgb",
                "artifact_path": "artifacts/prod/scorer.json",
                "conviction_gate": {"enabled": True, "mu_floor": 0.03,
                                     "demean_cross_sectional": False},
                "buy_floor": "adaptive_mean_std", "buy_floor_min": 0.2,
                "global_calibration": {"enabled": True,
                                        "artifact_path": "artifacts/prod/calib.json"},
                "shadow_models": [{"name": "prev_primary", "kind": "hf_patchtst",
                                    "artifact_path": "artifacts/shadow/model.pt"}],
            },
            "kelly_sizing": {"enabled": True, "fractional": 0.3,
                              "max_concentration": 0.12, "min_edge": 0,
                              "use_calibrator_mu": True},
        },
    }), encoding="utf-8")
    (configs / SHADOW_CONFIG_NAME).write_text(json.dumps({
        "ranking": {"panel_scoring": {
            "kind": "hf_patchtst",
            "artifact_path": "artifacts/shadow/model.pt",
            "global_calibration": {"artifact_path": "artifacts/prod/calib.json"},
        }},
    }), encoding="utf-8")
    (root / LOCK_FILE_REL).write_text(json.dumps({
        "schema_version": 1,
        "subrepos": [
            {"name": PINNED_SUBREPO_NAME, "branch": "main", "commit": "f" * 40,
             "status": "bootstrapped"},
            {"name": "renquant-common", "branch": "main", "commit": "a" * 40,
             "status": "bootstrapped"},
        ],
    }), encoding="utf-8")


def run_selftest() -> int:
    failures: list[str] = []

    def check(cond: bool, label: str) -> None:
        print(f"  {'PASS' if cond else 'FAIL'}: {label}")
        if not cond:
            failures.append(label)

    with tempfile.TemporaryDirectory(prefix="snapshot-selftest-") as tmp:
        root = Path(tmp)
        _write_selftest_fixture(root)
        snapshot = collect_snapshot(root)
        rendered = render_markdown(snapshot, generated_at="SELFTEST")
        rendered_again = render_markdown(collect_snapshot(root), generated_at="SELFTEST")
        check(rendered == rendered_again, "render is deterministic given same inputs")
        check("`xgb`" in rendered, "active kind rendered")
        check("| trained_date | 2026-05-18 |" in rendered, "active trained_date rendered")
        check("| label_observation_cutoff | unknown (field absent) |" in rendered,
              "absent field rendered as explicit unknown")
        check("effective_selection_cutoff_date=2026-02-10" in rendered,
              "shadow binding cutoff rendered")
        check("sk-selftest-must-never-render" not in rendered,
              "non-whitelisted metadata never rendered")
        check("global_panel_calibration" in rendered, "calibrator identity rendered")
        check("| mu_floor" not in rendered and "mu_floor=+0.0300" in rendered,
              "mu_floor knob rendered")
        check(f"| {PINNED_SUBREPO_NAME} | main | `ffffffffffff` |" in rendered,
              "pin table rendered")

        out = root / "snapshot.md"
        rc_write = main(["--repo-root", str(root), "--output", str(out)])
        rc_check = main(["--repo-root", str(root), "--output", str(out), "--check"])
        check(rc_write == 0 and rc_check == 0, "--check passes right after regeneration")

        problems = verify_pinned_declaration(
            snapshot_path=out, configs_dir=root / PINNED_CONFIGS_REL,
            lock_path=root / LOCK_FILE_REL,
        )
        check(problems == [], f"verify-pinned-declaration passes on fresh snapshot {problems}")

        # Mutate the pinned declaration -> both guards must trip.
        config_path = root / PINNED_CONFIGS_REL / ACTIVE_CONFIG_NAME
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["ranking"]["panel_scoring"]["kind"] = "hf_patchtst"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        rc_stale = main(["--repo-root", str(root), "--output", str(out), "--check"])
        check(rc_stale == 1, "--check fails after the pinned config changes")
        problems = verify_pinned_declaration(
            snapshot_path=out, configs_dir=root / PINNED_CONFIGS_REL,
            lock_path=root / LOCK_FILE_REL,
        )
        check(any("active kind" in p for p in problems),
              "verify-pinned-declaration catches an active-kind flip")

        # Missing artifact metadata must degrade to explicit unknowns.
        (root / STRATEGY_DIR_REL / "artifacts" / "prod" / "scorer.json").unlink()
        degraded = render_markdown(collect_snapshot(root), generated_at="SELFTEST")
        check("unknown (metadata file missing or unreadable)" in degraded,
              "missing artifact metadata degrades to explicit unknown, no crash")

    if failures:
        print(f"SELFTEST FAILED ({len(failures)} failing checks)")
        return 1
    print("SELFTEST OK")
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--repo-root", type=Path, default=REPO_ROOT,
        help="tree holding the SOURCES (.subrepo_runtime configs, lock, artifacts);"
             " defaults to this script's repo",
    )
    ap.add_argument(
        "--configs-dir", type=Path, default=None,
        help="override the pinned strategy-104 configs dir "
             "(default <repo-root>/.subrepo_runtime/repos/renquant-strategy-104/configs)",
    )
    ap.add_argument("--lock-file", type=Path, default=None,
                    help="override <repo-root>/subrepos.lock.json")
    ap.add_argument("--source-pin", default=None,
                    help="explicit strategy-104 source commit for the snapshot "
                         "(render from a CANDIDATE lock assembly pre-deploy; "
                         "overrides reading the live runtime checkout HEAD)")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--check", action="store_true",
        help="do not write; exit 1 if a fresh render differs from --output "
             "(byte-exact; the source fingerprint line is itself deterministic "
             "and part of the comparison)",
    )
    ap.add_argument(
        "--verify-pinned-declaration", action="store_true",
        help="semantic CI check: verify --output's machine block against "
             "--configs-dir (a strategy-104 checkout at the lock pin) + --lock-file",
    )
    ap.add_argument("--selftest", action="store_true",
                    help="run the built-in fixture selftest and exit")
    ap.add_argument(
        "--allow-pin-drift", action="store_true",
        help="DIAGNOSTIC MODE ONLY: proceed even if the pinned runtime checkout's "
             "HEAD disagrees with subrepos.lock.json. Without this flag, both "
             "generation and --check refuse to treat a drifted runtime as a valid "
             "source for the canonical snapshot (a drifted-but-unflagged snapshot "
             "would legitimize an unpinned runtime while claiming pin alignment).",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        return run_selftest()
    if args.verify_pinned_declaration:
        configs_dir = args.configs_dir or (
            args.repo_root.resolve() / PINNED_CONFIGS_REL
        )
        lock_path = args.lock_file or (args.repo_root.resolve() / LOCK_FILE_REL)
        problems = verify_pinned_declaration(
            snapshot_path=args.output, configs_dir=configs_dir, lock_path=lock_path,
        )
        for problem in problems:
            sys.stderr.write(f"VERIFY FAIL: {problem}\n")
        if not problems:
            print("verify-pinned-declaration: OK")
        return 1 if problems else 0

    snapshot = collect_snapshot(
        args.repo_root, configs_dir=args.configs_dir, lock_path=args.lock_file,
        source_pin=args.source_pin,
    )
    pin_drift = [w for w in (snapshot.get("warnings") or []) if PIN_DRIFT_MARKER in w]
    if pin_drift and not args.allow_pin_drift:
        for w in pin_drift:
            sys.stderr.write(f"REFUSED: {w}\n")
        sys.stderr.write(
            "The pinned runtime checkout does not match subrepos.lock.json — "
            "generating or checking the CANONICAL snapshot against a drifted "
            "runtime would legitimize an unpinned state as pin-aligned. Sync the "
            "runtime to its pin first, or pass --allow-pin-drift for an explicit "
            "diagnostic render (never commit that output as the canonical snapshot).\n"
        )
        return 1
    rendered = render_markdown(snapshot)
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if existing is None or existing != rendered:
            sys.stderr.write(
                f"{args.output} is STALE relative to the pinned sources under "
                f"{args.repo_root} — regenerate with: "
                "python3 scripts/render_strategy_104_snapshot.py (make snapshot)\n"
            )
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
