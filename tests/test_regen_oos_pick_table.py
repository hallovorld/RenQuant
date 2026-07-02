"""Unit tests for scripts/regen_oos_pick_table.py.

Deliberately does NOT re-run the full manifest re-scoring pipeline (43
point-in-time artifacts against the full panel) — that is expensive
(real model inference) and was run once, manually, as this PR's own
verification (see doc/progress/2026-07-02-regen-oos-pick-table.md for the
numbers). These tests cover the parts that are cheap and deterministic to
check in isolation: module import, and the decile-bucketing logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "scripts"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


def test_module_imports_without_error():
    import scripts.regen_oos_pick_table as mod  # noqa: F401


def test_decile_rank_top_decile_is_highest_score():
    from scripts.regen_oos_pick_table import _decile_rank, N_DECILES

    scores = pd.Series(np.arange(100, dtype=float))  # strictly increasing
    deciles = _decile_rank(scores)
    assert deciles.min() == 0
    assert deciles.max() == N_DECILES - 1
    # the single highest score must land in the top bucket, the lowest in
    # the bottom bucket -- decile 9 is "top-decile long candidates".
    assert deciles.iloc[scores.idxmax()] == N_DECILES - 1
    assert deciles.iloc[scores.idxmin()] == 0


def test_decile_rank_is_monotonic_in_score():
    from scripts.regen_oos_pick_table import _decile_rank

    rng = np.random.default_rng(0)
    scores = pd.Series(rng.normal(size=200))
    deciles = _decile_rank(scores)
    order = scores.sort_values().index
    ordered_deciles = deciles.loc[order].to_numpy()
    # non-decreasing as score increases
    assert (np.diff(ordered_deciles) >= 0).all()


def test_decile_rank_roughly_balanced_bucket_sizes():
    from scripts.regen_oos_pick_table import _decile_rank, N_DECILES

    rng = np.random.default_rng(1)
    scores = pd.Series(rng.normal(size=1000))
    deciles = _decile_rank(scores)
    counts = deciles.value_counts()
    assert set(counts.index) == set(range(N_DECILES))
    assert counts.min() >= 90  # ~100/bucket, some qcut slack for ties
    assert counts.max() <= 110


def test_decile_rank_falls_back_gracefully_with_too_few_names():
    from scripts.regen_oos_pick_table import _decile_rank

    scores = pd.Series([1.0, 2.0, 3.0])  # only 3 distinct values, < N_DECILES
    deciles = _decile_rank(scores)
    assert deciles.isna().sum() == 0
    assert deciles.nunique() <= 3
    assert deciles.iloc[2] > deciles.iloc[0]  # still ordered by score


def test_decile_rank_constant_scores_does_not_crash():
    from scripts.regen_oos_pick_table import _decile_rank

    scores = pd.Series([5.0, 5.0, 5.0, 5.0])
    deciles = _decile_rank(scores)
    assert (deciles == 0).all()


def test_reference_artifact_path_uses_last_retrain(tmp_path):
    from scripts.regen_oos_pick_table import _reference_artifact_path
    import json

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    art_path = artifact_dir / "panel-ltr.json"
    art_path.write_text("{}")

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({
        "retrains": [
            {"artifact_uri": str(artifact_dir / "old.json"), "cutoff_date": "2023-01-01"},
            {"artifact_uri": str(art_path), "cutoff_date": "2024-01-01"},
        ]
    }))
    (artifact_dir / "old.json").write_text("{}")

    resolved = _reference_artifact_path(manifest_path)
    assert resolved == art_path.resolve()


def test_reference_artifact_path_raises_on_empty_manifest(tmp_path):
    from scripts.regen_oos_pick_table import _reference_artifact_path
    import json

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"retrains": []}))
    with pytest.raises(ValueError, match="no retrain entries"):
        _reference_artifact_path(manifest_path)
