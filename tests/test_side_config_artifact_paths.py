"""Regression: side-experiment configs must NOT overwrite production artifacts.

Pre-fix incident (2026-05-02): a side config (strategy_config.insider_off.json)
set `panel_ltr.ngboost.artifact_path` to a side path, but missed
`ranking.panel_scoring.ngboost.artifact_path` (the inference-side path).
NGBoostSaveTask logged a warning and used the inference-side default —
overwriting the production NGBoost head with the experimental insider_off
variant. Symptom: production daily_104 retrain runs would have served
the wrong σ head until the next clean retrain.

Invariant pinned by these tests:
  Any strategy_config.*.json with `_audit_label` set (i.e. a side experiment)
  must override ALL six artifact_path keys to a side path matching its
  audit label, OR to None.

If a side config leaves any artifact_path pointing at the production
default (`artifacts/{panel-ltr,ngboost-head,panel-rank-calibration}.json`),
this test fails — pre-merge, not after a multi-hour retrain trashes prod.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

PRODUCTION_DEFAULTS = {
    "artifacts/panel-ltr.json",
    "artifacts/ngboost-head.json",
    "artifacts/panel-rank-calibration.json",
}


def _walk_artifact_paths(obj, path=""):
    """Yield all (dotted_path, value) for keys named 'artifact_path'."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_path = f"{path}.{k}" if path else k
            if k == "artifact_path":
                yield (new_path, v)
            yield from _walk_artifact_paths(v, new_path)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _walk_artifact_paths(v, f"{path}[{i}]")


def _side_config_files() -> list[Path]:
    """All strategy_config.*.json files except production
    (strategy_config.json, strategy_config.golden.json).

    Coverage chosen by file name: any non-production config could be
    invoked via `--strategy-config-name` and clobber production unless
    its artifact paths are overridden. The original v1 test scoped to
    `_audit_label`-tagged configs only — but several historical configs
    (lgbm_*, emb_*, macro_*, wl174, wl178) lack the label and had the
    same leak. Expanded scope catches them.
    """
    cfg_dir = REPO_ROOT / "backtesting" / "renquant_104"
    out = []
    for p in cfg_dir.glob("strategy_config.*.json"):
        if p.name in ("strategy_config.json", "strategy_config.golden.json"):
            continue
        try:
            json.loads(p.read_text())
        except Exception:
            continue
        out.append(p)
    return out


def test_at_least_one_side_config_present():
    """Sanity: ensure we're actually testing something."""
    sides = _side_config_files()
    assert len(sides) > 0, "No side configs with _audit_label found — did the convention change?"


@pytest.mark.parametrize("config_path", _side_config_files(), ids=lambda p: p.stem)
def test_side_config_does_not_use_production_artifact_paths(config_path):
    cfg = json.loads(config_path.read_text())
    # Fall back to filename stem when _audit_label is missing (older
    # historical configs don't carry the label).
    label = cfg.get("_audit_label") or config_path.stem.replace("strategy_config.", "")
    violations = []
    for dotted_path, value in _walk_artifact_paths(cfg):
        if value in PRODUCTION_DEFAULTS:
            violations.append(f"{dotted_path} = {value!r}")
    assert not violations, (
        f"Side config {config_path.name} (label={label!r}) uses production "
        f"artifact paths — running this config would overwrite production:\n"
        + "\n".join(f"  {v}" for v in violations)
        + f"\n\nFix: set every artifact_path in this config to a side path "
        f"containing the label, e.g. 'artifacts/panel-ltr.{label}.json'."
    )
