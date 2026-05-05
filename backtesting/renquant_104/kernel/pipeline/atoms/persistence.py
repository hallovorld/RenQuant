"""Persistence atoms — write/read JSON, parquet, atomic file ops.

Replaces ad-hoc `path.write_text(json.dumps(...))` and `pd.read_parquet`
patterns with declarative atoms. Atomic writes (tmp + rename) by default
to prevent half-written files on Ctrl-C.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from ..pipeline import Task
from .ctx_ops import _get_path

log = logging.getLogger("kernel.pipeline.atoms.persistence")


class WriteJSONArtifactTask(Task):
    """Write `ctx.<source_field>` to `path` as JSON. Atomic via tmp+rename.

    `path` may contain `{strategy_dir}` which resolves to ctx.config['_strategy_dir'].
    """

    def __init__(
        self,
        source_field: str,
        path_template: str,
        indent: int = 2,
        on_error: str = "warn",   # "warn" | "raise" | "skip"
    ):
        self.source_field = source_field
        self.path_template = path_template
        self.indent = indent
        self.on_error = on_error

    @property
    def name(self) -> str:
        return f"WriteJSON({self.source_field}→{self.path_template})"

    def run(self, ctx) -> bool | None:
        try:
            payload = _get_path(ctx, self.source_field)
            sd = (ctx.config or {}).get("_strategy_dir", "")
            path = Path(self.path_template.format(strategy_dir=sd))
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, indent=self.indent, default=str))
            tmp.replace(path)
            log.info("WriteJSONArtifactTask: %s → %s",
                     self.source_field, path)
        except Exception as exc:
            if self.on_error == "raise":
                raise
            if self.on_error == "warn":
                log.warning("WriteJSONArtifactTask: %s — skipping write", exc)


class LoadParquetTask(Task):
    """Read a parquet file into `ctx.<target_field>` (as a pandas DataFrame).

    Returns False if file missing (so a chain can short-circuit cleanly).
    """

    def __init__(
        self,
        path_template: str,
        target_field: str,
        on_missing: str = "skip",   # "skip" | "raise"
    ):
        self.path_template = path_template
        self.target_field = target_field
        self.on_missing = on_missing

    @property
    def name(self) -> str:
        return f"LoadParquet({self.path_template}→{self.target_field})"

    def run(self, ctx) -> bool | None:
        import pandas as pd  # noqa: PLC0415
        sd = (ctx.config or {}).get("_strategy_dir", "")
        path = Path(self.path_template.format(strategy_dir=sd))
        if not path.exists():
            if self.on_missing == "raise":
                raise FileNotFoundError(str(path))
            log.info("LoadParquetTask: %s missing — skipping", path)
            return False
        try:
            df = pd.read_parquet(path)
            from .ctx_ops import _set_path
            _set_path(ctx, self.target_field, df)
        except Exception as exc:
            log.warning("LoadParquetTask: %s read failed — %s", path, exc)
            return False


__all__ = ["WriteJSONArtifactTask", "LoadParquetTask"]
