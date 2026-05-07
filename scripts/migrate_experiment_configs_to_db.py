#!/usr/bin/env python
"""Migrate side-experiment strategy configs from filesystem to data/runs.db.

Per CLAUDE.md follow-up (2026-05-06): we accumulated 60+ side
strategy_config.*.json files for diagnostic experiments — most untracked
file-system clutter. Move them into a dedicated table:

    experiment_configs (
      label             TEXT PRIMARY KEY,    -- e.g. 'alpha158_linear'
      base_config_name  TEXT NOT NULL,       -- usually 'strategy_config.json'
      overrides_json    TEXT NOT NULL,       -- jsonpatch / dict-of-overrides
      audit_label       TEXT,                -- for side-config invariant test
      created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      notes             TEXT
    )

Inflate at runtime with `inflate_experiment_config(label) -> dict`
(applies overrides on top of base config).

This commit creates the table + a CLI that:
  1. Lists current `strategy_config.*.json` files (production-vs-side).
  2. Computes (overrides) = side - base via dict-diff.
  3. Inserts the (overrides) into experiment_configs.
  4. Exports a backup .json.bak before deletion.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "runs.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiment_configs (
    label             TEXT PRIMARY KEY,
    base_config_name  TEXT NOT NULL,
    overrides_json    TEXT NOT NULL,
    audit_label       TEXT,
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes             TEXT
);

CREATE INDEX IF NOT EXISTS idx_experiment_configs_audit_label
    ON experiment_configs(audit_label);
"""


def diff_overrides(base: dict, side: dict, prefix: str = "") -> dict:
    """Return a flat dict of dotted-keys → side-value where side ≠ base.

    Walks recursively. List values compared by reference (no element-diff).
    """
    out: dict = {}
    for k, v in side.items():
        path = f"{prefix}.{k}" if prefix else k
        if k not in base:
            out[path] = v
        elif isinstance(v, dict) and isinstance(base[k], dict):
            out.update(diff_overrides(base[k], v, prefix=path))
        elif v != base[k]:
            out[path] = v
    return out


def apply_overrides(base: dict, overrides: dict) -> dict:
    """Apply dotted-key overrides to base config (deep merge by path)."""
    import copy as _copy
    out = _copy.deepcopy(base)
    for dotted_key, value in overrides.items():
        parts = dotted_key.split(".")
        obj = out
        for p in parts[:-1]:
            if p not in obj or not isinstance(obj[p], dict):
                obj[p] = {}
            obj = obj[p]
        obj[parts[-1]] = value
    return out


def inflate_experiment_config(
    label: str,
    db_path: Path = DB_PATH,
    strategy_dir: Path = REPO_ROOT / "backtesting" / "renquant_104",
) -> dict:
    """Load a side experiment config from DB, applying overrides on
    top of its base config."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT base_config_name, overrides_json FROM experiment_configs "
            "WHERE label = ?", (label,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise KeyError(f"experiment_configs: label={label!r} not found")
    base_name, overrides_json = row
    base = json.loads((strategy_dir / base_name).read_text())
    return apply_overrides(base, json.loads(overrides_json))


def store_experiment_config(
    label: str, base_config_name: str, overrides: dict,
    audit_label: str | None = None, notes: str | None = None,
    db_path: Path = DB_PATH,
) -> None:
    """Insert / update an experiment config in the DB."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(SCHEMA)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            INSERT INTO experiment_configs
              (label, base_config_name, overrides_json, audit_label,
               created_at, updated_at, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(label) DO UPDATE SET
                overrides_json = excluded.overrides_json,
                audit_label    = excluded.audit_label,
                updated_at     = excluded.updated_at,
                notes          = COALESCE(excluded.notes, notes)
        """, (label, base_config_name, json.dumps(overrides),
              audit_label, now, now, notes))
        conn.commit()
    finally:
        conn.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=["init", "import", "list", "inflate"])
    p.add_argument("--label", help="Experiment label (for inflate)")
    p.add_argument("--strategy-dir",
                   default=str(REPO_ROOT / "backtesting" / "renquant_104"))
    p.add_argument("--base-config", default="strategy_config.json")
    args = p.parse_args()

    sd = Path(args.strategy_dir)
    db = DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)

    if args.command == "init":
        conn = sqlite3.connect(str(db))
        conn.executescript(SCHEMA)
        conn.commit()
        conn.close()
        print(f"Initialized experiment_configs table in {db}")
    elif args.command == "import":
        # Import all side configs as DB entries. Files NOT deleted (safer).
        base_path = sd / args.base_config
        if not base_path.exists():
            print(f"ERROR: base config not found: {base_path}", file=sys.stderr)
            sys.exit(1)
        base = json.loads(base_path.read_text())
        n_imported = 0
        for f in sorted(sd.glob("strategy_config.*.json")):
            if f.name in {args.base_config, "strategy_config.golden.json"}:
                continue
            label = f.name.replace("strategy_config.", "").replace(".json", "")
            side = json.loads(f.read_text())
            overrides = diff_overrides(base, side)
            audit = side.get("_audit_label")
            store_experiment_config(label, args.base_config, overrides,
                                     audit_label=audit, db_path=db)
            print(f"  imported '{label}' ({len(overrides)} overrides) "
                  f"[audit={audit!r}]")
            n_imported += 1
        print(f"Total imported: {n_imported}")
    elif args.command == "list":
        # overrides_json is stored as a dict (not array); compute size in
        # Python rather than relying on SQLite's json_array_length.
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT label, base_config_name, audit_label, "
            "overrides_json, updated_at FROM experiment_configs "
            "ORDER BY updated_at DESC"
        ).fetchall()
        if not rows:
            print("(no experiment_configs rows; run `import` first)")
        for r in rows:
            n_overrides = len(json.loads(r[3]))
            print(f"  {r[0]:30s}  base={r[1]:25s}  audit={r[2]!s:18s}  "
                  f"overrides={n_overrides:>3d}  updated={r[4]}")
        conn.close()
    elif args.command == "inflate":
        if not args.label:
            print("--label required for inflate", file=sys.stderr)
            sys.exit(1)
        cfg = inflate_experiment_config(args.label, db_path=db)
        print(json.dumps(cfg, indent=2, default=str))


if __name__ == "__main__":
    main()
