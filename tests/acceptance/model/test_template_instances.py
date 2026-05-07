"""Concrete instantiations of the standard artifact-acceptance template.

Each class subclasses _StandardChecks and sets ARTIFACT_NAME, SCHEMA,
FILES. The conftest hook auto-expands (path × attr) parametrization.

2026-05-04 update: FILES is now AUTO-DISCOVERED via glob at collection
time, so new ablation/side-config artifacts (e.g. produced by
strategy_config.macro_v2.json) get picked up without editing this file.
The discovery is ONLY for current production + flagged variants —
historical .bak.json and .pre-train.json are explicitly excluded.

Adding a new artifact filename pattern? Add it to the GLOB_PATTERNS
list in the relevant subclass. Per-ticker tests already auto-scan
models/ directory.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACTS_DIR = REPO / "backtesting" / "renquant_104" / "artifacts"
sys.path.insert(0, str(REPO / "tests"))

from acceptance.model._template import _StandardChecks  # noqa: E402
from acceptance.model.schemas import (                     # noqa: E402
    PANEL_LTR_SCHEMA,
    NGBOOST_SCHEMA,
    CALIBRATOR_SCHEMA,
)


def _auto_discover(prefix: str, exclude_substrings: tuple[str, ...] = ()) -> list[str]:
    """Glob artifacts/ for files matching `{prefix}.json` or `{prefix}.*.json`.

    Excludes:
      * .bak.json (historical backups)
      * any filename containing one of `exclude_substrings`
    Returns just the basenames (caller resolves to full paths).
    """
    if not ARTIFACTS_DIR.exists():
        return []
    out: list[str] = []
    for p in sorted(ARTIFACTS_DIR.iterdir()):
        if not p.is_file() or not p.name.endswith(".json"):
            continue
        if not (p.name == f"{prefix}.json" or p.name.startswith(f"{prefix}.")):
            continue
        # Skip .bak/.pre-train backups
        if ".bak.json" in p.name or ".pre-train.json" in p.name:
            continue
        # Skip diagnostic/ablation noise — these are research outputs
        # not meant for ship; would clutter the green pass count.
        if any(sub in p.name for sub in exclude_substrings):
            continue
        out.append(p.name)
    return out


# ── panel-LTR family ─────────────────────────────────────────────────────────

class TestPanelLTRStandard(_StandardChecks):
    """Standard 10-check suite for panel-LTR artifacts.

    Auto-discovers all `panel-ltr*.json` files in artifacts/ except
    .bak / .pre-train / contaminated diagnostic outputs.
    """
    ARTIFACT_NAME = "panel-ltr"
    SCHEMA = PANEL_LTR_SCHEMA
    # Exclude obvious research/diagnostic files (optional — drop these
    # from the exclusion list when you want to validate them too).
    FILES = _auto_discover("panel-ltr", exclude_substrings=(
        "diag", "contaminated", "PRE-MINUTE", "previous",
        "stage3_batch", "ablation",
        "fwd5d_placebo", "triple_barrier_on_placebo",
        "triple_barrier_on_shuffled", "triple_barrier_on_repro",
        "macro_v2_retest", "emb_retest", "wl_sweep", "topdown",
    ))
    # panel_transformer + panel_lgbm + panel_linear use different artifact
    # schemas (no `booster_raw_json` etc.). These have their own scorer
    # classes; skipping XGB-specific schema checks here.
    SKIP_KIND = {"panel_transformer", "panel_lgbm", "panel_linear"}


# ── NGBoost family ───────────────────────────────────────────────────────────

class TestNGBoostStandard(_StandardChecks):
    """Standard 10-check suite for ngboost-head artifacts."""
    ARTIFACT_NAME = "ngboost-head"
    SCHEMA = NGBOOST_SCHEMA
    FILES = _auto_discover("ngboost-head", exclude_substrings=(
        "diag", "contaminated", "PRE-MINUTE",
        "stage3_batch", "ablation",
        "fwd5d_placebo", "triple_barrier_on_placebo",
        "triple_barrier_on_shuffled", "triple_barrier_on_repro",
        "macro_v2_retest", "emb_retest", "wl_sweep", "topdown",
    ))


# ── Calibrator family — pooled + per-regime ─────────────────────────────────

class TestCalibratorStandard(_StandardChecks):
    """Standard 10-check suite for panel-rank-calibration artifacts."""
    ARTIFACT_NAME = "panel-rank-calibration"
    SCHEMA = CALIBRATOR_SCHEMA
    FILES = _auto_discover("panel-rank-calibration", exclude_substrings=(
        "stage3_batch", "wl_sweep", "topdown",
        "macro_v2_retest", "emb_retest", "triple_barrier_on_repro",
    )) + _auto_discover("panel-calibration", exclude_substrings=())
