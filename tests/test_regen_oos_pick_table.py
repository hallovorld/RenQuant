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


# ─── manifest (Codex review on #430: the parquet payload must never be
# committed to git — it matches renquant-orchestrator's PROD_PATH_RULES
# protected-path regex — so this regeneration manifest is the actual durable
# artifact) ─────────────────────────────────────────────────────────────────


def test_relpath_uses_repo_relative_when_under_root(tmp_path):
    from scripts.regen_oos_pick_table import _relpath

    root = tmp_path / "repo"
    nested = root / "backtesting" / "renquant_104" / "artifacts" / "x.json"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}")
    assert _relpath(nested, root) == "backtesting/renquant_104/artifacts/x.json"


def test_relpath_falls_back_to_backtesting_suffix_across_worktrees(tmp_path):
    """The real-world case this exists for: a worktree's input file is a
    symlink to a SIBLING checkout's absolute path (outside `root` entirely,
    e.g. crossing from /private/tmp/some-worktree to /Users/x/git/RenQuant) —
    the manifest must record the repo-internal `backtesting/...` path, never
    leak that machine's home directory into a committed artifact."""
    from scripts.regen_oos_pick_table import _relpath

    root = tmp_path / "worktree"
    root.mkdir()
    other_checkout = tmp_path / "other-checkout"
    real_file = other_checkout / "backtesting" / "renquant_104" / "artifacts" / "y.json"
    real_file.parent.mkdir(parents=True)
    real_file.write_text("{}")
    symlink = root / "y.json"
    symlink.symlink_to(real_file)

    assert _relpath(symlink, root) == "backtesting/renquant_104/artifacts/y.json"


def test_relpath_absolute_fallback_when_no_backtesting_marker(tmp_path):
    from scripts.regen_oos_pick_table import _relpath

    root = tmp_path / "worktree"
    root.mkdir()
    stray = tmp_path / "elsewhere" / "z.json"
    stray.parent.mkdir(parents=True)
    stray.write_text("{}")
    assert _relpath(stray, root) == str(stray)


def test_build_manifest_never_embeds_the_parquet_payload_path_as_object_uri(tmp_path):
    """The manifest's object_uri must say NOTHING is persisted — proves this
    test can't silently regress into claiming an object-storage location
    that doesn't exist."""
    from scripts.regen_oos_pick_table import build_manifest

    manifest_path = tmp_path / "wf_manifest.json"
    manifest_path.write_text("{}")
    artifact_path = tmp_path / "artifact.json"
    artifact_path.write_text("{}")
    meta = {"label": "fwd_60d_excess", "val_cut": "2024-02-01",
            "n_rows": 100, "n_dates": 10, "n_names": 20}

    out = build_manifest(meta, manifest_path=manifest_path, reference_artifact_path=artifact_path)

    assert out["schema"]["columns"] == [
        "date", "name", "score", "decile_rank", "fwd_60d_excess", "regime",
    ]
    assert out["counts"] == {"n_rows": 100, "n_dates": 10, "n_names": 20}
    assert "NOT PERSISTED" in out["object_uri"]
    assert "regen_oos_pick_table.py" in out["object_uri"]
    assert out["recipe"]["generator"] == "scripts/regen_oos_pick_table.py"
    # sha256 hex digest length, not a guessed/placeholder string
    assert len(out["recipe"]["manifest_input_sha256"]) == 64
    assert len(out["recipe"]["reference_artifact_sha256"]) == 64
    # matches a hand-computed sha256 of the fixture's literal content — pins
    # the hash is REAL, not a stub
    import hashlib
    assert out["recipe"]["manifest_input_sha256"] == hashlib.sha256(b"{}").hexdigest()


def test_main_writes_manifest_json_and_gitignored_parquet(tmp_path, monkeypatch):
    """End-to-end check of main()'s two-output contract, without running the
    expensive real scoring pipeline: stub build_oos_pick_table."""
    import scripts.regen_oos_pick_table as mod

    fake_table = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
        "name": ["AAPL", "MSFT"],
        "score": [0.1, -0.1],
        "decile_rank": [9, 0],
        "fwd_60d_excess": [0.02, -0.01],
        "regime": ["BULL_CALM", "BULL_CALM"],
    })
    fake_meta = {
        "manifest": "m.json", "reference_artifact": str(tmp_path / "ref.json"),
        "label": "fwd_60d_excess", "val_cut": "2024-01-01",
        "n_rows": 2, "n_dates": 1, "n_names": 2,
        "panel_meta": {}, "score_meta": {},
    }
    (tmp_path / "ref.json").write_text("{}")
    (tmp_path / "m.json").write_text("{}")
    monkeypatch.setattr(mod, "build_oos_pick_table",
                         lambda **kw: (fake_table, fake_meta))
    monkeypatch.setattr(sys, "argv", [
        "regen_oos_pick_table.py",
        "--manifest", str(tmp_path / "m.json"),
        "--output", str(tmp_path / "out" / "table.parquet"),
    ])

    mod.main()

    out_parquet = tmp_path / "out" / "table.parquet"
    out_manifest = tmp_path / "out" / "table.manifest.json"
    assert out_parquet.exists()
    assert out_manifest.exists()
    import json
    written = json.loads(out_manifest.read_text())
    assert written["counts"] == {"n_rows": 2, "n_dates": 1, "n_names": 2}
    assert "NOT PERSISTED" in written["object_uri"]
