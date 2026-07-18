#!/usr/bin/env python3
"""AC-C dry-run harness — prove the weekly Saturday deadlock clears against the
MIGRATED canonical sidecar, WITHOUT promoting and WITHOUT touching the live file.

AC-C (amendment ``2026-07-18-sidecar-single-writer-amendment.md`` §3): a full
dry-run of the Saturday chain (refresh → guard → non-promoting retrain
preparation) must pass end-to-end against the unified 179-col contract, and the
former ``staged corpus dropped columns … sentiment`` rejection must no longer
fire.

This harness runs that chain against a SANDBOX COPY of the migration candidate
(``--candidate``) and asserts:

  1. the REAL refresh guard (``RebuildTransformerCorpusTask._sanity_reasons`` in
     ``scripts/refresh_transformer_corpus.py`` — not a re-implementation)
     returns NO rejection reasons when comparing the served candidate (prior)
     against a freshly-built staged sidecar; and
  2. specifically the ``dropped columns (recipe/schema drift)`` reason — the
     07-11 / 07-18 failure signature — is absent; and
  3. a NON-PROMOTING retrain preparation clears the former failure boundary: the
     candidate is admissible for the PatchTST / σ-head consume path (179-col
     canonical schema, sentiment present, zero bar-frontier extension rows, a
     non-empty finite raw-label subset to fit on).

It NEVER swaps or promotes anything and REFUSES to run against the live served
sidecar path. Both the staged build and all reads happen inside a caller-owned
sandbox directory. ``build_fn`` / ``canon_columns`` are injectable so the harness
is testable without an on-pin 179-col builder.

The static dry-run holds inputs fixed, so the guard's date-ADVANCE requirement
is not exercised here (a fresh build from a fixed panel does not advance the
frontier); the real Saturday run advances naturally as OHLCV gains bars. The
harness therefore drives the guard with ``require_date_advance=False`` — the
schema/recipe drift check (the deadlock) is fully exercised; the orthogonal
weekly date-advance is not the failure this closes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Callable

log = logging.getLogger("ac_c_sidecar_dryrun_harness")

SENTIMENT_COLS = ("sentiment_pos_share", "mean_sentiment", "n_articles_log")
RAW_LABEL_COL = "fwd_60d_excess_raw"
KEY_COLS = ("ticker", "date")
DROPPED_COLUMNS_SIGNATURE = "dropped columns"

LIVE_SERVED_PATH = Path(
    "/Users/renhao/git/github/RenQuant/data/"
    "alpha158_291_fundamental_dataset_rawlabel.parquet"
)


class AcCHarnessError(RuntimeError):
    """The AC-C dry-run failed — the deadlock did not clear, or the candidate is
    inadmissible for the consume path."""


def _guard_module():
    """Import the REAL refresh guard (lazily; its base-data imports are all
    function-local so importing the module is side-effect free)."""
    import scripts.refresh_transformer_corpus as mod  # noqa: PLC0415

    return mod


def guard_schema_reasons(
    prior_path: str | Path,
    staged_path: str | Path,
    *,
    require_date_advance: bool = False,
) -> list:
    """Run the REAL ``_sanity_reasons`` guard comparing a prior (served) sidecar
    against a freshly-staged sidecar; return its list of rejection reasons
    (empty == guard passes)."""
    import types  # noqa: PLC0415

    mod = _guard_module()
    prior_rows, prior_date = mod._default_corpus_stats(Path(prior_path))
    prior_schema = mod._default_corpus_schema(Path(prior_path))
    staged_rows, staged_date = mod._default_corpus_stats(Path(staged_path))
    staged_schema = mod._default_corpus_schema(Path(staged_path))
    ctx = types.SimpleNamespace(
        require_date_advance=require_date_advance,
        validate_schema=True,
        min_row_ratio=getattr(mod, "DEFAULT_MIN_ROW_RATIO", 0.5),
        min_ticker_coverage_ratio=getattr(mod, "DEFAULT_MIN_TICKER_COVERAGE_RATIO", 0.9),
    )
    return mod.RebuildTransformerCorpusTask._sanity_reasons(
        ctx, prior_rows, prior_date, prior_schema, staged_rows, staged_date, staged_schema
    )


def dropped_columns_fired(reasons) -> bool:
    """True iff the former 07-11 / 07-18 failure signature is present."""
    return any(DROPPED_COLUMNS_SIGNATURE in str(r) for r in reasons)


def retrain_prep_admissible(candidate_path: str | Path, canon_columns) -> dict:
    """NON-PROMOTING retrain preparation: assert the migrated candidate is
    admissible for the PatchTST / σ-head consume path through the FORMER failure
    boundary — canonical 179-col schema in order, sentiment present, ZERO
    bar-frontier extension rows, and a non-empty finite raw-label subset to fit
    on. Raises ``AcCHarnessError`` on any failure. Promotes nothing."""
    import pandas as pd  # noqa: PLC0415

    canon = [str(c) for c in canon_columns]
    df = pd.read_parquet(candidate_path)
    cols = [str(c) for c in df.columns]
    problems = []
    if cols != canon:
        problems.append(
            f"schema is not the canonical contract in order (n={len(cols)} vs {len(canon)})"
        )
    missing_sent = [c for c in SENTIMENT_COLS if c not in cols]
    if missing_sent:
        problems.append(f"sentiment column(s) missing: {missing_sent}")
    non_key = [c for c in cols if c not in KEY_COLS]
    n_ext = int(df[non_key].isna().all(axis=1).sum()) if non_key else 0
    if n_ext:
        problems.append(f"{n_ext} bar-frontier extension row(s) present (must be zero)")
    labeled = df[RAW_LABEL_COL].notna().sum() if RAW_LABEL_COL in cols else 0
    if labeled <= 0:
        problems.append("no finite raw-label rows to fit on")
    if problems:
        raise AcCHarnessError(
            "retrain-prep admission FAILED for the candidate:\n  - "
            + "\n  - ".join(problems)
        )
    return {
        "n_rows": int(len(df)),
        "n_cols": len(cols),
        "n_labeled_rows": int(labeled),
        "n_extension_rows": n_ext,
        "sentiment_present": True,
    }


def default_canonical_builder():
    """Resolve the canonical builder from the migration script (single source of
    truth for the builder-contract preflight)."""
    import scripts.migrate_rawlabel_sidecar_to_canonical as migrate  # noqa: PLC0415

    return migrate.default_canonical_builder()


def run_ac_c_dryrun(
    candidate_path: str | Path,
    fund_panel_path: str | Path,
    ohlcv_dir: str | Path,
    *,
    build_fn: Callable | None = None,
    canon_columns=None,
    sandbox_dir: str | Path | None = None,
    require_date_advance: bool = False,
) -> dict:
    """Run the AC-C Saturday-chain dry-run against a sandbox copy of the migrated
    candidate. Returns a report dict; raises ``AcCHarnessError`` if the deadlock
    did not clear or the candidate is inadmissible."""
    candidate_path = Path(candidate_path).resolve()
    if candidate_path == LIVE_SERVED_PATH.resolve():
        raise AcCHarnessError(
            "refusing to run the AC-C dry-run against the LIVE served sidecar "
            f"({LIVE_SERVED_PATH}); pass a sandbox copy via --candidate"
        )
    if not candidate_path.exists():
        raise AcCHarnessError(f"candidate not found: {candidate_path}")

    if build_fn is None or canon_columns is None:
        build_fn, canon_columns, _rev = default_canonical_builder()

    owns_sandbox = sandbox_dir is None
    sandbox_dir = Path(sandbox_dir or tempfile.mkdtemp(prefix="ac_c_dryrun_"))
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    staged_path = sandbox_dir / "staged_rawlabel_sidecar.parquet"
    try:
        # (1) refresh: build the staged sidecar via the canonical builder.
        build_report = build_fn(fund_panel_path, ohlcv_dir, staged_path)
        # (2) guard: the REAL _sanity_reasons over prior(candidate) vs staged.
        reasons = guard_schema_reasons(
            candidate_path, staged_path, require_date_advance=require_date_advance
        )
        dropped_fired = dropped_columns_fired(reasons)
        # (3) non-promoting retrain preparation admission.
        prep = retrain_prep_admissible(candidate_path, canon_columns)
    finally:
        if owns_sandbox:
            import shutil  # noqa: PLC0415

            shutil.rmtree(sandbox_dir, ignore_errors=True)

    problems = []
    if dropped_fired:
        problems.append(
            "the 'dropped columns' rejection STILL fires — deadlock not cleared: "
            f"{[r for r in reasons if DROPPED_COLUMNS_SIGNATURE in str(r)]}"
        )
    if reasons:
        problems.append(f"guard did not pass; reasons: {reasons}")
    if problems:
        raise AcCHarnessError(
            "AC-C dry-run FAILED:\n  - " + "\n  - ".join(problems)
        )

    log.info(
        "AC-C dry-run PASS: guard clean (no reasons), 'dropped columns' absent, "
        "candidate admissible (%d labeled rows, %d cols)",
        prep["n_labeled_rows"],
        prep["n_cols"],
    )
    return {
        "ok": True,
        "candidate": str(candidate_path),
        "guard_reasons": list(reasons),
        "dropped_columns_fired": dropped_fired,
        "retrain_prep": prep,
        "staged_build_report": build_report,
    }


def parse_args(argv: "list | None" = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--candidate",
        type=Path,
        required=True,
        help="SANDBOX COPY of the migrated canonical sidecar (never the live file).",
    )
    p.add_argument("--fund-panel", type=Path, required=True)
    p.add_argument("--ohlcv-dir", type=Path, required=True)
    p.add_argument("--sandbox-dir", type=Path, default=None)
    p.add_argument("--report-out", type=Path, default=None)
    p.add_argument(
        "--require-date-advance",
        action="store_true",
        help="Also require the staged build to advance the date frontier (off by "
        "default for a static dry-run; the real Saturday run advances naturally).",
    )
    return p.parse_args(argv)


def main(argv: "list | None" = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    args = parse_args(argv)
    try:
        report = run_ac_c_dryrun(
            args.candidate,
            args.fund_panel,
            args.ohlcv_dir,
            sandbox_dir=args.sandbox_dir,
            require_date_advance=args.require_date_advance,
        )
    except AcCHarnessError as exc:
        print(f"AC-C DRY-RUN FAILED: {exc}", file=sys.stderr)
        return 1
    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
