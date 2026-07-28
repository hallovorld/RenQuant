"""Focused tests for the umbrella shadow-scoring artifact resolution
(#537: strategy_dir-first, honoring the runtime _strategy_dir stamp)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting" / "renquant_104"))

from kernel.panel_pipeline.shadow_scoring import resolve_shadow_artifact_path  # noqa: E402


def test_absolute_path_passes_through(tmp_path):
    p = tmp_path / "abs.json"
    assert resolve_shadow_artifact_path(str(p), {}, tmp_path) == p


def test_stamped_strategy_dir_wins_when_artifact_exists(tmp_path):
    sd = tmp_path / "alt_assembly" / "renquant_104"
    (sd / "artifacts" / "shadow").mkdir(parents=True)
    art = sd / "artifacts" / "shadow" / "m.json"
    art.write_text("{}")
    cfg = {"_strategy_dir": str(sd)}
    got = resolve_shadow_artifact_path("artifacts/shadow/m.json", cfg, tmp_path / "repo")
    assert got == art


def test_falls_back_to_repo_root_when_absent_from_strategy_dir(tmp_path):
    sd = tmp_path / "sd"
    sd.mkdir()
    cfg = {"_strategy_dir": str(sd)}
    got = resolve_shadow_artifact_path("artifacts/shadow/m.json", cfg, tmp_path / "repo")
    assert got == tmp_path / "repo" / "artifacts" / "shadow" / "m.json"


def test_missing_stamp_uses_module_strategy_dir_fallback(tmp_path):
    got = resolve_shadow_artifact_path("artifacts/shadow/nonexistent-xyz.json", {}, tmp_path / "repo")
    assert got == tmp_path / "repo" / "artifacts" / "shadow" / "nonexistent-xyz.json"
