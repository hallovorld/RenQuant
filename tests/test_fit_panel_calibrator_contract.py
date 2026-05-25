"""Contract tests for scripts/fit_panel_calibrator.py."""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "fit_panel_calibrator.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fit_panel_calibrator_under_test", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scorer_fingerprint_prefers_artifact_identity(tmp_path: Path) -> None:
    mod = _load_module()
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("artifact bytes")
    scorer = SimpleNamespace(
        metadata={
            "artifact_fingerprint": "sha256:artifact123",
            "config_fingerprint": "sha256:config999",
        }
    )

    assert mod._scorer_fingerprint(scorer_path, scorer) == "sha256:artifact123"


def test_scorer_fingerprint_prefers_model_content_identity(tmp_path: Path) -> None:
    mod = _load_module()
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("artifact bytes")
    scorer = SimpleNamespace(
        metadata={
            "model_content_fingerprint": "sha256:modelcontent123",
            "artifact_fingerprint": "sha256:artifact123",
        }
    )

    assert mod._scorer_fingerprint(scorer_path, scorer) == "sha256:modelcontent123"


def test_scorer_fingerprint_ignores_config_identity(tmp_path: Path) -> None:
    """Config fingerprints are shared recipe IDs, not scorer-file identity."""
    mod = _load_module()
    scorer_path = tmp_path / "panel-ltr.json"
    scorer_path.write_text("artifact bytes")
    scorer = SimpleNamespace(metadata={"config_fingerprint": "sha256:config999"})

    expected = "sha256:" + hashlib.sha256(scorer_path.read_bytes()).hexdigest()

    assert mod._scorer_fingerprint(scorer_path, scorer) == expected
