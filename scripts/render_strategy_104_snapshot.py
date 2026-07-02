#!/usr/bin/env python3
"""Render renquant_104's production snapshot from the PINNED config + artifact
metadata — never hand-maintained.

doc/arch/strategy-104.md previously carried a hand-written "Production
snapshot" table that drifted from reality (it stayed dated 2026-06-08 through
at least one later prod/shadow switch, and a related design review round was
wasted trusting it — see doc/design/2026-07-01-104-105-design-review-amendments.md
amendment A6). This script reads backtesting/renquant_104/strategy_config.json
(the PINNED prod config weekly_wf_promote.sh treats as canonical) plus each
referenced artifact's own stamped metadata (trained_date, binding data cutoff,
config_fingerprint) and renders a snapshot table. Output is deterministic
(byte-for-byte reproducible given the same inputs) — no wall-clock timestamp
is embedded in the diffed content, so scripts/check_strategy_104_snapshot_fresh.py
can assert "regenerating now produces exactly the committed file" as the
staleness gate, not a day-count heuristic.

Not every historical "Production snapshot" row has a clean current-state
source (e.g. a specific walk-forward run's mean IC, or a regime-detector
commit hash) — those stay as hand-written narrative in strategy-104.md
itself. This script only renders what is honestly derivable from the pinned
config + artifact metadata right now.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
DEFAULT_CONFIG = STRATEGY_DIR / "strategy_config.json"
DEFAULT_OUTPUT = REPO_ROOT / "doc" / "arch" / "strategy-104-snapshot.md"

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
# committed, widely-read doc. `_describe_model` below only ever reads the
# fields named here, by explicit dict-literal construction — never
# `for k, v in meta.items(): ...`. This constant exists so that guarantee is
# visible and auditable at a glance, not just an emergent property of one
# function's code, and so extending the rendered fields later is a deliberate
# one-line addition here, not an accidental widening.
_ALLOWED_METADATA_FIELDS = (
    "trained_date",
    *_DATA_CUTOFF_FIELDS,
    "lookahead_days",
    "config_fingerprint",
)


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


def _resolve_artifact_path(raw: str, *, strategy_dir: Path) -> Path:
    """Artifact paths in strategy_config.json are relative to STRATEGY_DIR
    (e.g. "../../artifacts/..." or "artifacts/prod/...")."""
    p = Path(raw)
    if p.is_absolute():
        return p
    return (strategy_dir / p).resolve()


def _load_artifact_metadata(artifact_path: Path) -> dict[str, Any]:
    """JSON artifacts (XGB/GBDT) carry metadata inline; binary checkpoints
    (.pt et al, e.g. hf_patchtst) carry it in a "<path>.metadata.json"
    sidecar (the same convention model_freshness_monitor.py's
    _freshness_path_for reads)."""
    if artifact_path.suffix in _BINARY_ARTIFACT_SUFFIXES:
        sidecar = artifact_path.with_suffix(artifact_path.suffix + ".metadata.json")
        if not sidecar.exists():
            return {}
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
    if not artifact_path.exists():
        return {}
    try:
        return json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _binding_cutoff(meta: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    for field in _DATA_CUTOFF_FIELDS:
        assert field in _ALLOWED_METADATA_FIELDS  # structural whitelist guard
        val = meta.get(field)
        if val:
            return field, str(val)[:10]
    return None, None


def _extract_allowed(meta: dict[str, Any]) -> dict[str, Any]:
    """The ONLY point in this module that reads artifact metadata field
    values — every other function receives its data from HERE, never from
    ``meta`` directly. Returns exactly the ``_ALLOWED_METADATA_FIELDS`` keys
    present in ``meta``; anything else in the source metadata (whatever it
    might be — a credential, a local debug path, free-form notes) is
    structurally unreachable past this point."""
    return {field: meta[field] for field in _ALLOWED_METADATA_FIELDS if field in meta}


def _describe_model(
    *, role: str, kind: Optional[str], artifact_rel: Optional[str],
    strategy_dir: Path, repo_root: Path = REPO_ROOT, name: Optional[str] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "role": role, "name": name, "kind": kind,
        "artifact_path": (
            _relativize_for_display(artifact_rel, repo_root=repo_root)
            if artifact_rel else artifact_rel
        ),
        "trained_date": None, "binding_cutoff_field": None, "binding_cutoff": None,
        "lookahead_days": None, "config_fingerprint": None,
        "metadata_source": None,
    }
    if not artifact_rel:
        return row
    resolved = _resolve_artifact_path(artifact_rel, strategy_dir=strategy_dir)
    raw_meta = _load_artifact_metadata(resolved)
    if not raw_meta:
        return row
    meta = _extract_allowed(raw_meta)  # whitelist boundary — see docstring
    row["trained_date"] = meta.get("trained_date")
    row["binding_cutoff_field"], row["binding_cutoff"] = _binding_cutoff(meta)
    row["lookahead_days"] = meta.get("lookahead_days")
    row["config_fingerprint"] = meta.get("config_fingerprint")
    row["metadata_source"] = (
        "sidecar" if resolved.suffix in _BINARY_ARTIFACT_SUFFIXES else "inline"
    )
    return row


def collect_snapshot(
    config_path: Path, *, strategy_dir: Optional[Path] = None, repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Pure function: config -> snapshot dict. No I/O beyond reading config +
    referenced artifact metadata files. ``repo_root`` is exposed (default the
    real repo root) so tests can prove the absolute-path redaction in
    ``_relativize_for_display`` without touching the real filesystem."""
    strategy_dir = strategy_dir or config_path.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    panel_scoring = (config.get("ranking") or {}).get("panel_scoring") or {}

    primary = _describe_model(
        role="primary", kind=panel_scoring.get("kind"),
        artifact_rel=panel_scoring.get("artifact_path"), strategy_dir=strategy_dir,
        repo_root=repo_root,
    )
    shadows = [
        _describe_model(
            role="shadow", kind=sm.get("kind"), artifact_rel=sm.get("artifact_path"),
            strategy_dir=strategy_dir, name=sm.get("name"), repo_root=repo_root,
        )
        for sm in (panel_scoring.get("shadow_models") or [])
    ]
    watchlist = config.get("watchlist") or []

    return {
        "config_path": str(config_path.relative_to(repo_root))
        if config_path.is_relative_to(repo_root) else str(config_path),
        "primary": primary,
        "shadows": shadows,
        "watchlist_size": len(watchlist),
    }


def _fmt_model_row(row: dict[str, Any]) -> str:
    bits = [f"`kind=\"{row['kind']}\"`" if row["kind"] else "kind=?"]
    if row["name"]:
        bits.append(f"name=`{row['name']}`")
    if row["artifact_path"]:
        bits.append(f"artifact=`{row['artifact_path']}`")
    if row["trained_date"]:
        bits.append(f"trained_date={row['trained_date']}")
    if row["binding_cutoff_field"]:
        bits.append(f"{row['binding_cutoff_field']}={row['binding_cutoff']}")
    else:
        bits.append("binding data cutoff=unknown")
    if row["lookahead_days"]:
        bits.append(f"lookahead_days={row['lookahead_days']}")
    if row["config_fingerprint"]:
        bits.append(f"fingerprint={row['config_fingerprint']}")
    return "; ".join(bits)


def render_markdown(snapshot: dict[str, Any]) -> str:
    lines = [
        "# renquant_104 — generated production snapshot",
        "",
        "GENERATED FILE — do not hand-edit. Regenerate with:",
        "`python3 scripts/render_strategy_104_snapshot.py`",
        "",
        "Rendered directly from the PINNED config + each referenced artifact's own",
        "stamped metadata (never hand-maintained prose) — see amendment A6,",
        "doc/design/2026-07-01-104-105-design-review-amendments.md. This states ONLY",
        "what the pinned config says AS OF the last regeneration (CI-enforced fresh —",
        "see the workflow below) — a current fact, never a historical/promotion claim",
        "(\"active since <date>\", \"promoted on <date>\"); that narrative, with its own",
        "dating and provenance, belongs in doc/arch/strategy-104.md instead. Fields",
        "with no clean current-state source (a specific WF run's mean IC, a",
        "regime-detector commit hash, etc.) are NOT rendered here either, for the",
        "same reason — they stay as dated narrative in doc/arch/strategy-104.md.",
        "",
        f"Source config: `{snapshot['config_path']}`",
        "",
        "| | |",
        "|---|---|",
        f"| Active model | {_fmt_model_row(snapshot['primary'])} |",
    ]
    if snapshot["shadows"]:
        for shadow in snapshot["shadows"]:
            lines.append(f"| Shadow model | {_fmt_model_row(shadow)} |")
    else:
        lines.append("| Shadow model | (none configured) |")
    lines.append(f"| Watchlist size | {snapshot['watchlist_size']} tickers |")
    lines.append("")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strategy-config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument(
        "--check", action="store_true",
        help="do not write; exit 1 if regenerated content differs from --output",
    )
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    snapshot = collect_snapshot(args.strategy_config.resolve())
    rendered = render_markdown(snapshot)
    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else None
        if existing != rendered:
            sys.stderr.write(
                f"{args.output} is STALE relative to {args.strategy_config} — "
                "regenerate with: python3 scripts/render_strategy_104_snapshot.py\n"
            )
            return 1
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
