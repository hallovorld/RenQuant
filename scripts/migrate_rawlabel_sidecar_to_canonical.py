#!/usr/bin/env python3
"""One-time supervised migration of the served RAWLABEL sidecar to the canonical
single-writer contract (AC-D of the sidecar single-writer amendment).

CONTEXT (base-data ``doc/design/2026-07-18-sidecar-single-writer-amendment.md``):
the served ``alpha158_291_fundamental_dataset_rawlabel.parquet`` had TWO active
weekly writers with contradictory recipes — the base-data builder
(``renquant_base_data.rawlabel_sidecar``) and the orchestrator σ-head refresh
(``retrain_alpha158_fund``). That writer war deadlocked the weekly PatchTST
corpus refresh (07-11 / 07-18: ``staged corpus dropped columns … sentiment``).
The amendment resolves it to ONE file, ONE writer: the base-data builder is the
SOLE producer; the canonical contract CARRIES the three sentiment columns
(**179 columns**) and, for THIS artifact, drops the bar-frontier extension rows.
orchestrator#553 (MERGED) retires the σ-head writer and makes it a
fail-closed CONSUMER of the canonical file.

WHAT THIS SCRIPT DOES (AC-D — migration integrity, inherits the base RFC's
AC-2 verbatim): a supervised, hash-verified, one-time regeneration of the live
served sidecar to the canonical contract via the Stage-1 canonical builder.

  * BEFORE snapshot: builder revision + contract fingerprint, input fingerprints
    (fund panel sha256, OHLCV dir signature), served-file sha256 + schema
    digest, row count, primary-key / date coverage, and a per-retained-column
    checksum.
  * regenerate to a same-directory CANDIDATE via the canonical builder
    (``extend_to_bar_frontier=False`` — the served-file recipe).
  * AFTER snapshot of the candidate.
  * assert the diff is ONLY the intended contract change: the candidate's
    columns equal ``RAWLABEL_SIDECAR_COLUMNS`` in order, ZERO bar-frontier
    extension rows, no fabricated rows (candidate keys ⊆ served keys), the only
    dropped rows are the served file's extension rows, and every RETAINED column
    (present in both files) is checksum-identical over the shared (ticker, date)
    rows.
  * atomic swap: back up the served file to a timestamped ``.bak`` (digest
    recorded), then ``os.replace`` the candidate into place; verify the served
    file now hashes to the candidate digest.
  * ``--rollback`` restores the EXACT backed-up bytes — verified by the recorded
    sha256, never merely by ``.bak`` filename.

SAFETY: this script NEVER runs automatically and NEVER defaults to a mutating
mode — exactly one of ``--dry-run`` / ``--execute`` / ``--rollback`` /
``--preflight`` is required. Only ``--execute`` mutates the served file, and it
writes a CONTAINMENT record (CLAUDE.md §5) in the same action batch. ``--dry-run``
does everything except the swap. The actual destructive migration is an
ask-first, operator-gated landing action (live-tree mutation preflight) — this
script is the runbook artifact, not the trigger.

DEPLOYMENT ORDERING (the amendment §2 hazard — see the runbook
``doc/ops/2026-07-18-rawlabel-sidecar-canonical-migration-runbook.md``):
this migration MUST run and produce the canonical file BEFORE any pin-bump
deploys orchestrator#553's σ-head cessation, else the newly-deployed consume
path fails closed with no canonical file present. ``--preflight`` is the gate
that REFUSES the #553 pin-bump while the canonical file is absent or
non-canonical.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

log = logging.getLogger("migrate_rawlabel_sidecar_to_canonical")

# ── canonical contract invariants (base-data#48/#49 §2.2) ───────────────────
#: The three event-driven sentiment columns the canonical contract now CARRIES.
SENTIMENT_COLS = ("sentiment_pos_share", "mean_sentiment", "n_articles_log")
#: The raw forward excess-return label (always the LAST column).
RAW_LABEL_COL = "fwd_60d_excess_raw"
#: (ticker, date) primary key.
KEY_COLS = ("ticker", "date")
#: The canonical column count: 178-col fund-panel schema (sentiment INCLUDED)
#: + the raw label. A builder that yields any other count is the wrong (stale
#: 176-col, sentiment-free) revision and is REFUSED by the contract preflight.
EXPECTED_CANON_NCOLS = 179

DEFAULT_SERVED_PATH = Path(
    "/Users/renhao/git/github/RenQuant/data/"
    "alpha158_291_fundamental_dataset_rawlabel.parquet"
)
DEFAULT_FUND_PANEL_PATH = Path(
    "/Users/renhao/git/github/RenQuant/data/alpha158_291_fundamental_dataset.parquet"
)
DEFAULT_OHLCV_DIR = Path("/Users/renhao/git/github/RenQuant/data/ohlcv")


class MigrationIntegrityError(RuntimeError):
    """AC-D integrity assertion failed — the candidate differs from the served
    file by more than the intended contract change (fail closed; no swap)."""


class MigrationPreflightError(RuntimeError):
    """A precondition for a safe migration is not met (stale builder, missing
    canonical file, deployment-ordering violation)."""


# ─────────────────────────── digest primitives ─────────────────────────────


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of a file's bytes (works on multi-hundred-MB parquets)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def schema_digest(columns) -> str:
    """sha256 over the ORDERED column-name list — schema identity that is
    sensitive to column ADDITION, REMOVAL, and REORDERING."""
    return hashlib.sha256("\n".join(str(c) for c in columns).encode()).hexdigest()


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def ohlcv_dir_signature(ohlcv_dir: str | Path) -> str:
    """Cheap, deterministic fingerprint of the per-ticker OHLCV cache dir:
    sha256 over sorted ``<ticker>/1d.parquet`` (size, mtime_ns) tuples. Records
    the label-input provenance without hashing hundreds of MB of bars."""
    ohlcv_dir = Path(ohlcv_dir)
    if not ohlcv_dir.exists():
        return "absent"
    entries = []
    for sub in sorted(p for p in ohlcv_dir.iterdir() if p.is_dir()):
        bar = sub / "1d.parquet"
        if bar.exists():
            st = bar.stat()
            entries.append(f"{sub.name}:{st.st_size}:{st.st_mtime_ns}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


# ─────────────────────────── frame primitives ──────────────────────────────


def _keyed(df):
    """Return ``df`` re-indexed by the normalized (ticker, date) primary key
    (ticker upper-cased string, date as datetime64), for row alignment."""
    import pandas as pd  # noqa: PLC0415

    keyed = df.copy()
    keyed["__ticker"] = keyed["ticker"].astype("string").str.upper()
    keyed["__date"] = pd.to_datetime(keyed["date"])
    return keyed.set_index(["__ticker", "__date"])


def count_extension_rows(df) -> int:
    """Count bar-frontier EXTENSION rows — key-only rows whose every non-key
    column is NaN (§2.3). Legitimately-unlabeled tail rows (features present,
    only the forward label NaN) and no-OHLCV rows are NOT extension rows."""
    non_key = [c for c in df.columns if c not in KEY_COLS]
    if not non_key:
        return 0
    return int(df[non_key].isna().all(axis=1).sum())


def _key_set(df) -> set:
    keyed = _keyed(df)
    return set(keyed.index)


def frame_snapshot(path: str | Path, df) -> dict:
    """Integrity snapshot of a sidecar parquet (file bytes + logical content).

    Records everything AC-D compares BEFORE and AFTER: file sha256, ordered
    columns + schema digest, row/ticker counts, PK/date coverage, duplicate-key
    flag, and the bar-frontier extension-row count.
    """
    import pandas as pd  # noqa: PLC0415

    columns = [str(c) for c in df.columns]
    keyed_dupes = int(_keyed(df).index.duplicated().sum())
    dates = pd.to_datetime(df["date"]) if "date" in df.columns else pd.Series([], dtype="datetime64[ns]")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "n_rows": int(len(df)),
        "n_cols": len(columns),
        "columns": columns,
        "schema_digest": schema_digest(columns),
        "n_tickers": int(df["ticker"].nunique()) if "ticker" in df.columns else 0,
        "n_unique_keys": int(len(_key_set(df))),
        "duplicate_key_rows": keyed_dupes,
        "date_min": dates.min().date().isoformat() if len(dates) else None,
        "date_max": dates.max().date().isoformat() if len(dates) else None,
        "n_extension_rows": count_extension_rows(df),
    }


def _column_checksum(series) -> str:
    """Order-deterministic content checksum of a single (already key-sorted)
    column, NaN-stable via pandas' hashing."""
    import pandas as pd  # noqa: PLC0415

    hashed = pd.util.hash_pandas_object(series, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def retained_column_checksums(before_df, after_df, retained_cols) -> dict:
    """Per-retained-column content checksums over the SHARED (ticker, date) rows.

    Returns ``{"common_key_count", "before": {col: digest}, "after": {...},
    "mismatched": [cols]}``. A retained column whose shared-row content changed
    appears in ``mismatched`` — the AC-D "only intended diff" assertion fails on
    a non-empty list.
    """
    b = _keyed(before_df)
    a = _keyed(after_df)
    common = b.index.intersection(a.index)
    b_common = b.loc[common].sort_index()
    a_common = a.loc[common].sort_index()
    before_ck: dict = {}
    after_ck: dict = {}
    mismatched: list = []
    for col in retained_cols:
        bc = _column_checksum(b_common[col])
        ac = _column_checksum(a_common[col])
        before_ck[col] = bc
        after_ck[col] = ac
        if bc != ac:
            mismatched.append(col)
    return {
        "common_key_count": int(len(common)),
        "before": before_ck,
        "after": after_ck,
        "mismatched": mismatched,
    }


# ─────────────────────────── AC-D core assertion ───────────────────────────


def assert_only_intended_diff(
    before_df, after_df, before_snap: dict, after_snap: dict, canon_columns
) -> dict:
    """AC-D: prove the candidate differs from the served file by ONLY the
    intended contract change. Raises ``MigrationIntegrityError`` otherwise.

    Intended change:
      1. columns == ``canon_columns`` in order (schema → canonical);
      2. ZERO bar-frontier extension rows.
    Integrity guarantees (nothing else changed):
      3. no fabricated rows — candidate keys ⊆ served keys;
      4. the only dropped rows are the served file's extension rows;
      5. every RETAINED column is checksum-identical over the shared rows.
    """
    canon = [str(c) for c in canon_columns]
    problems: list = []

    # (1) canonical schema, in order.
    if after_snap["columns"] != canon:
        problems.append(
            "candidate columns are not the canonical contract in order "
            f"(n={after_snap['n_cols']} vs {len(canon)}; "
            f"digest {after_snap['schema_digest'][:12]} vs "
            f"{schema_digest(canon)[:12]})"
        )
    # (2) zero extension rows.
    if after_snap["n_extension_rows"] != 0:
        problems.append(
            f"candidate carries {after_snap['n_extension_rows']} bar-frontier "
            "extension row(s) — the canonical served recipe must have zero (§2.3)"
        )
    # duplicate-key guard on both sides (alignment integrity).
    if before_snap["duplicate_key_rows"]:
        problems.append(
            f"served file has {before_snap['duplicate_key_rows']} duplicate "
            "(ticker, date) row(s) — cannot prove a clean diff"
        )
    if after_snap["duplicate_key_rows"]:
        problems.append(
            f"candidate has {after_snap['duplicate_key_rows']} duplicate "
            "(ticker, date) row(s)"
        )

    before_keys = _key_set(before_df)
    after_keys = _key_set(after_df)
    # (3) no fabricated rows.
    fabricated = after_keys - before_keys
    if fabricated:
        problems.append(
            f"candidate introduces {len(fabricated)} (ticker, date) key(s) absent "
            "from the served file — a migration must not fabricate rows"
        )
    # (4) the only dropped rows are served extension rows.
    dropped_keys = before_keys - after_keys
    if dropped_keys:
        ext_keys = _extension_key_set(before_df)
        non_ext_dropped = dropped_keys - ext_keys
        if non_ext_dropped:
            problems.append(
                f"candidate drops {len(non_ext_dropped)} NON-extension row(s) from "
                "the served file — only bar-frontier extension rows may be dropped"
            )

    # (5) retained-column checksum equality over shared rows. "Retained" =
    # columns present in BOTH the served file and the candidate.
    retained_cols = [
        c for c in after_snap["columns"] if c in before_snap["columns"]
    ]
    checks = retained_column_checksums(before_df, after_df, retained_cols)
    if checks["mismatched"]:
        problems.append(
            "retained column(s) changed content over the shared rows "
            f"(fail): {checks['mismatched'][:8]}"
            f"{'…' if len(checks['mismatched']) > 8 else ''}"
        )

    if problems:
        raise MigrationIntegrityError(
            "AC-D integrity assertion FAILED — the candidate is NOT a clean "
            "canonical migration of the served file:\n  - " + "\n  - ".join(problems)
        )

    return {
        "intended_schema_change": before_snap["schema_digest"] != after_snap["schema_digest"],
        "added_columns": [c for c in canon if c not in before_snap["columns"]],
        "removed_columns": [c for c in before_snap["columns"] if c not in canon],
        "reordered": (
            before_snap["schema_digest"] != after_snap["schema_digest"]
            and sorted(before_snap["columns"]) == sorted(after_snap["columns"])
        ),
        "extension_rows_dropped": len(before_keys) - len(after_keys),
        "retained_column_count": len(retained_cols),
        "retained_columns_checksum_equal": True,
        "common_key_count": checks["common_key_count"],
    }


def _extension_key_set(df) -> set:
    non_key = [c for c in df.columns if c not in KEY_COLS]
    if not non_key:
        return set()
    keyed = _keyed(df)
    mask = keyed[non_key].isna().all(axis=1)
    return set(keyed.index[mask.to_numpy()])


# ─────────────────────────── builder resolution ────────────────────────────


def builder_contract_preflight(canon_columns) -> None:
    """Refuse the STALE (176-col, sentiment-free) builder. The canonical
    contract MUST be 179 cols, carry the three sentiment columns, and end with
    the raw label. Running the migration against the pre-amendment builder would
    write a 176-col file that orchestrator#553's consumer then REFUSES — the
    exact failure the migration exists to remove."""
    canon = [str(c) for c in canon_columns]
    problems = []
    if len(canon) != EXPECTED_CANON_NCOLS:
        problems.append(
            f"canonical contract has {len(canon)} columns, expected "
            f"{EXPECTED_CANON_NCOLS} — this looks like the stale/wrong builder "
            "revision (base-data pin predates #49?)"
        )
    missing_sent = [c for c in SENTIMENT_COLS if c not in canon]
    if missing_sent:
        problems.append(
            f"canonical contract is missing sentiment column(s) {missing_sent} — "
            "the amendment REQUIRES sentiment (§2.2); refusing the pre-amendment "
            "sentiment-free builder"
        )
    if not canon or canon[-1] != RAW_LABEL_COL:
        problems.append(
            f"canonical contract does not end with {RAW_LABEL_COL!r} "
            f"(last col = {canon[-1] if canon else None!r})"
        )
    if problems:
        raise MigrationPreflightError(
            "builder-contract preflight FAILED:\n  - " + "\n  - ".join(problems)
        )


def default_canonical_builder() -> tuple[Callable, tuple, str]:
    """Resolve the canonical Stage-1 builder from ``renquant_base_data`` and
    verify its contract. Returns ``(build_fn, canon_columns, builder_revision)``.

    ``build_fn(fund_panel, ohlcv_dir, output_path)`` builds the canonical
    (extension-free) sidecar. Raises ``MigrationPreflightError`` if the builder
    is absent (pin not synced) or is the stale sentiment-free revision.
    """
    try:
        from renquant_base_data import rawlabel_sidecar as rls  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - exercised only off-pin
        raise MigrationPreflightError(
            "canonical builder unresolvable (renquant_base_data.rawlabel_sidecar): "
            f"{exc}; the base-data pin predates the single-writer amendment "
            "(#49) or the subrepo PYTHONPATH is not set — refusing to migrate "
            "with no canonical builder."
        ) from exc
    canon_columns = tuple(rls.RAWLABEL_SIDECAR_COLUMNS)
    builder_contract_preflight(canon_columns)

    def build_fn(fund_panel, ohlcv_dir, output_path) -> dict:
        return rls.build_rawlabel_sidecar(
            fund_panel,
            ohlcv_dir,
            output_path,
            extend_to_bar_frontier=False,  # §2.3 — canonical served recipe.
        )

    revision = _resolve_builder_revision(rls, canon_columns)
    return build_fn, canon_columns, revision


def _resolve_builder_revision(module, canon_columns) -> str:
    """Best-effort builder-revision string for the BEFORE snapshot: the
    base-data package version + git rev of the builder's source tree (if
    resolvable) + the deterministic contract fingerprint."""
    parts = []
    try:
        import renquant_base_data  # noqa: PLC0415

        parts.append(f"pkg={getattr(renquant_base_data, '__version__', 'unknown')}")
    except Exception:  # pragma: no cover - defensive
        pass
    try:
        src = Path(module.__file__).resolve().parent
        import subprocess  # noqa: PLC0415

        rev = subprocess.run(
            ["git", "-C", str(src), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if rev.returncode == 0:
            parts.append(f"git={rev.stdout.strip()}")
    except Exception:  # pragma: no cover - defensive
        pass
    parts.append(f"contract={schema_digest(canon_columns)[:12]}")
    parts.append(f"ncols={len(canon_columns)}")
    return " ".join(parts)


# ─────────────────────────── ordering preflight (§2 hazard) ─────────────────


def ordering_preflight(served_path: str | Path, canon_columns) -> dict:
    """Deployment-ordering gate (amendment §2): the σ-head cessation
    (orchestrator#553) must NOT be pin-deployed until the canonical file is
    present AND canonical. REFUSES (raises ``MigrationPreflightError``) when the
    served file is absent or does not match the canonical contract — because the
    deployed #553 consumer fails closed with no canonical file.

    Returns a status dict when the gate PASSES.
    """
    import pandas as pd  # noqa: PLC0415

    served_path = Path(served_path)
    if not served_path.exists():
        raise MigrationPreflightError(
            f"deployment-ordering gate FAILED: canonical sidecar ABSENT at "
            f"{served_path}. Do NOT pin-bump orchestrator#553 (σ-head cessation) "
            "yet — its deployed consumer fails closed with no canonical file. "
            "Run this migration (--execute) first (runbook step a)."
        )
    df = pd.read_parquet(served_path)
    snap = frame_snapshot(served_path, df)
    canon = [str(c) for c in canon_columns]
    problems = []
    if snap["columns"] != canon:
        problems.append(
            f"served file is not the canonical contract in order "
            f"(n={snap['n_cols']} vs {len(canon)})"
        )
    if snap["n_extension_rows"] != 0:
        problems.append(
            f"served file carries {snap['n_extension_rows']} extension row(s) "
            "(canonical recipe requires zero)"
        )
    if problems:
        raise MigrationPreflightError(
            "deployment-ordering gate FAILED: served file present but NOT "
            "canonical — do NOT pin-bump #553:\n  - " + "\n  - ".join(problems)
        )
    log.info(
        "ordering preflight PASS: canonical sidecar present (%d rows, %d cols, "
        "sha256=%s) — #553 pin-bump may proceed",
        snap["n_rows"],
        snap["n_cols"],
        snap["sha256"][:12],
    )
    return {"ok": True, "snapshot": snap}


# ─────────────────────────── containment record ────────────────────────────


def build_containment_record(
    *,
    served_path: Path,
    before_snap: dict,
    after_snap: dict,
    backup_path: Path,
    backup_sha256: str,
    builder_revision: str,
    task_ref: str,
    owner: str,
    restore_condition: str,
) -> dict:
    """CLAUDE.md §5 containment record for the live-surface mutation: what
    changed, the LITERAL revert steps, owner, tracked task, and restore
    condition — written in the SAME action batch as the swap."""
    return {
        "kind": "rawlabel-sidecar-canonical-migration",
        "timestamp": _utc_now_iso(),
        "surface": str(served_path),
        "what_changed": (
            "The served rawlabel sidecar was regenerated to the canonical "
            "single-writer contract (179 cols, sentiment-carrying, zero "
            "bar-frontier extension rows) via the Stage-1 canonical builder."
        ),
        "before_sha256": before_snap["sha256"],
        "after_sha256": after_snap["sha256"],
        "before_n_cols": before_snap["n_cols"],
        "after_n_cols": after_snap["n_cols"],
        "builder_revision": builder_revision,
        "backup_path": str(backup_path),
        "backup_sha256": backup_sha256,
        "revert_steps": [
            "scripts/migrate_rawlabel_sidecar_to_canonical.py --rollback "
            f"--served-path {served_path} --backup-path {backup_path} "
            f"--expected-backup-sha256 {backup_sha256}",
            "(rollback verifies the backup bytes hash to expected-backup-sha256 "
            "BEFORE restoring, then re-verifies the restored file — hash-verified, "
            "not filename-trusted)",
        ],
        "owner": owner,
        "task_ref": task_ref,
        "restore_condition": restore_condition,
        "reviewed_surface_note": (
            "This is a one-time reviewed migration, not an emergency containment; "
            "the weekly writer (refresh_transformer_corpus.py RebuildRawLabelSidecar"
            "Task, calling the same canonical builder) remains the ongoing surface "
            "and requires no manifest change."
        ),
    }


def write_json(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


# ─────────────────────────── migration driver ──────────────────────────────


def run_migration(
    *,
    mode: str,
    served_path: str | Path,
    fund_panel_path: str | Path,
    ohlcv_dir: str | Path,
    build_fn: Callable | None = None,
    canon_columns=None,
    builder_revision: str | None = None,
    report_out: str | Path | None = None,
    containment_out: str | Path | None = None,
    backup_path: str | Path | None = None,
    expected_backup_sha256: str | None = None,
    task_ref: str = "UNSET",
    owner: str = "UNSET",
    restore_condition: str = "UNSET",
    run_id: str | None = None,
) -> dict:
    """Execute one migration mode. Pure aside from filesystem writes; ``build_fn``
    and ``canon_columns`` are injectable (default: the canonical base-data
    builder) so the logic is testable without an on-pin 179-col builder."""
    import pandas as pd  # noqa: PLC0415

    served_path = Path(served_path)
    run_id = run_id or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if mode == "rollback":
        return _rollback(served_path, backup_path, expected_backup_sha256)

    # dry-run / execute / preflight all need the canonical contract resolved.
    if build_fn is None or canon_columns is None:
        build_fn, canon_columns, revision = default_canonical_builder()
        builder_revision = builder_revision or revision
    else:
        builder_contract_preflight(canon_columns)
        builder_revision = builder_revision or _resolve_builder_revision_fallback(canon_columns)

    if mode == "preflight":
        status = ordering_preflight(served_path, canon_columns)
        if report_out:
            write_json(Path(report_out), {"mode": "preflight", **status})
        return {"mode": "preflight", **status}

    if mode not in ("dry-run", "execute"):
        raise ValueError(f"unknown mode: {mode!r}")

    if not served_path.exists():
        raise MigrationPreflightError(
            f"served sidecar not found at {served_path} — nothing to migrate "
            "(a fresh install builds via the weekly writer, not this one-time "
            "migration)"
        )

    # ── BEFORE snapshot ──────────────────────────────────────────────────────
    before_df = pd.read_parquet(served_path)
    before_snap = frame_snapshot(served_path, before_df)
    before_snap["builder_revision"] = builder_revision
    before_snap["fund_panel_sha256"] = sha256_file(fund_panel_path)
    before_snap["ohlcv_dir_signature"] = ohlcv_dir_signature(ohlcv_dir)

    # ── regenerate to a same-directory candidate ─────────────────────────────
    candidate_path = served_path.with_name(served_path.name + f".candidate-{run_id}")
    if candidate_path.exists():
        candidate_path.unlink()
    build_report = build_fn(fund_panel_path, ohlcv_dir, candidate_path)
    log.info("canonical rebuild report: %s", build_report)

    after_df = pd.read_parquet(candidate_path)
    after_snap = frame_snapshot(candidate_path, after_df)

    # ── AC-D: assert ONLY the intended contract change ───────────────────────
    try:
        diff = assert_only_intended_diff(
            before_df, after_df, before_snap, after_snap, canon_columns
        )
    except MigrationIntegrityError:
        # fail closed: never leave a rejected candidate behind, never swap.
        if candidate_path.exists():
            candidate_path.unlink()
        raise

    result: dict = {
        "mode": mode,
        "run_id": run_id,
        "served_path": str(served_path),
        "before": before_snap,
        "after": after_snap,
        "build_report": build_report,
        "diff": diff,
        "swapped": False,
    }

    if mode == "dry-run":
        # everything except the swap; discard the candidate.
        if candidate_path.exists():
            candidate_path.unlink()
        result["note"] = "dry-run: integrity verified; served file UNTOUCHED"
        log.info(
            "DRY-RUN OK: canonical rebuild is a clean migration "
            "(before sha256=%s -> after sha256=%s); served file untouched",
            before_snap["sha256"][:12],
            after_snap["sha256"][:12],
        )
        if report_out:
            write_json(Path(report_out), result)
        return result

    # ── mode == execute: atomic, backed-up, hash-verified swap ───────────────
    backup_path = Path(
        backup_path
        or served_path.with_name(
            served_path.name + f".pre-canonical-migration-{run_id}.bak"
        )
    )
    shutil.copy2(str(served_path), str(backup_path))
    backup_sha256 = sha256_file(backup_path)
    if backup_sha256 != before_snap["sha256"]:
        # backup is not a faithful copy of the served file — abort BEFORE swap.
        raise MigrationIntegrityError(
            f"backup digest {backup_sha256} != served digest "
            f"{before_snap['sha256']} — refusing to swap; served file untouched"
        )

    os.replace(str(candidate_path), str(served_path))
    post_sha256 = sha256_file(served_path)
    if post_sha256 != after_snap["sha256"]:  # pragma: no cover - fs anomaly
        raise MigrationIntegrityError(
            f"post-swap served digest {post_sha256} != candidate digest "
            f"{after_snap['sha256']} — swap did not land the verified bytes"
        )

    containment = build_containment_record(
        served_path=served_path,
        before_snap=before_snap,
        after_snap=after_snap,
        backup_path=backup_path,
        backup_sha256=backup_sha256,
        builder_revision=builder_revision or "unknown",
        task_ref=task_ref,
        owner=owner,
        restore_condition=restore_condition,
    )
    result.update(
        swapped=True,
        backup_path=str(backup_path),
        backup_sha256=backup_sha256,
        post_swap_sha256=post_sha256,
        containment=containment,
    )
    if containment_out:
        write_json(Path(containment_out), containment)
    if report_out:
        write_json(Path(report_out), result)
    log.info(
        "EXECUTE OK: served sidecar migrated to canonical (sha256 %s -> %s); "
        "backup=%s (sha256=%s); containment recorded",
        before_snap["sha256"][:12],
        post_sha256[:12],
        backup_path,
        backup_sha256[:12],
    )
    return result


def _resolve_builder_revision_fallback(canon_columns) -> str:
    return f"injected contract={schema_digest(canon_columns)[:12]} ncols={len(canon_columns)}"


def _rollback(
    served_path: Path, backup_path, expected_backup_sha256
) -> dict:
    """Restore the EXACT backed-up bytes — verified by the recorded sha256, not
    the ``.bak`` filename."""
    if backup_path is None or expected_backup_sha256 is None:
        raise MigrationPreflightError(
            "--rollback requires --backup-path AND --expected-backup-sha256 "
            "(hash-verified restore; a filename alone is not trusted)"
        )
    backup_path = Path(backup_path)
    if not backup_path.exists():
        raise MigrationPreflightError(f"backup not found: {backup_path}")
    actual = sha256_file(backup_path)
    if actual != expected_backup_sha256:
        raise MigrationIntegrityError(
            f"backup sha256 {actual} != expected {expected_backup_sha256} — the "
            "backup is not the exact file recorded at migration time; REFUSING "
            "to restore (served file untouched)"
        )
    os.replace(str(backup_path), str(served_path))
    restored = sha256_file(served_path)
    if restored != expected_backup_sha256:  # pragma: no cover - fs anomaly
        raise MigrationIntegrityError(
            f"restored served digest {restored} != expected "
            f"{expected_backup_sha256} — rollback did not land the backed-up bytes"
        )
    log.info(
        "ROLLBACK OK: served sidecar restored to backed-up bytes (sha256=%s)",
        restored[:12],
    )
    return {
        "mode": "rollback",
        "served_path": str(served_path),
        "restored_sha256": restored,
        "verified": True,
    }


# ─────────────────────────── CLI ───────────────────────────────────────────


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Do everything (BEFORE/AFTER snapshot + AC-D integrity assertion) "
        "EXCEPT the swap. The served file is never touched.",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="Perform the atomic, backed-up, hash-verified swap AND write a "
        "containment record. The only mutating mode; ask-first operator landing.",
    )
    mode.add_argument(
        "--rollback",
        action="store_true",
        help="Restore the served file from a backup whose sha256 matches "
        "--expected-backup-sha256 (hash-verified, not filename-trusted).",
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Deployment-ordering gate: REFUSE if the canonical file is absent "
        "or non-canonical (run before the orchestrator#553 pin-bump).",
    )
    p.add_argument("--served-path", type=Path, default=DEFAULT_SERVED_PATH)
    p.add_argument("--fund-panel", type=Path, default=DEFAULT_FUND_PANEL_PATH)
    p.add_argument("--ohlcv-dir", type=Path, default=DEFAULT_OHLCV_DIR)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument("--containment-out", type=Path, default=None)
    p.add_argument("--backup-path", type=Path, default=None)
    p.add_argument("--expected-backup-sha256", default=None)
    # containment metadata (required for --execute).
    p.add_argument("--task-ref", default="UNSET")
    p.add_argument("--owner", default="UNSET")
    p.add_argument(
        "--restore-condition",
        default="UNSET",
        help="Explicit restore/expiry condition for the containment record.",
    )
    return p.parse_args(argv)


def main(argv: "list | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args(argv)
    if args.dry_run:
        mode = "dry-run"
    elif args.execute:
        mode = "execute"
    elif args.rollback:
        mode = "rollback"
    else:
        mode = "preflight"

    if mode == "execute":
        missing = [
            name
            for name, val in (
                ("--task-ref", args.task_ref),
                ("--owner", args.owner),
                ("--restore-condition", args.restore_condition),
            )
            if val == "UNSET"
        ]
        if missing:
            print(
                "--execute requires containment metadata: "
                + ", ".join(missing)
                + " (CLAUDE.md §5 — a live-surface mutation with no record is an "
                "incident, not a fix)",
                file=sys.stderr,
            )
            return 2

    try:
        result = run_migration(
            mode=mode,
            served_path=args.served_path,
            fund_panel_path=args.fund_panel,
            ohlcv_dir=args.ohlcv_dir,
            report_out=args.report_out,
            containment_out=args.containment_out,
            backup_path=args.backup_path,
            expected_backup_sha256=args.expected_backup_sha256,
            task_ref=args.task_ref,
            owner=args.owner,
            restore_condition=args.restore_condition,
        )
    except (MigrationIntegrityError, MigrationPreflightError) as exc:
        print(f"MIGRATION REFUSED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
