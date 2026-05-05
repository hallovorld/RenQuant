"""Regression tests for the auto-discovery contract.

Goal: prove that adding a new ticker to the watchlist OR a new
side-config artifact to artifacts/ AUTO-EXTENDS the test matrix —
no manual file edit required.

Concretely:
  1. Per-ticker tests scan models/ at collection time and pick up
     every <TICKER>/<TICKER>-policy-metadata.json automatically.
  2. Panel-level tests glob artifacts/panel-ltr*.json /
     artifacts/ngboost-head*.json / artifacts/panel-rank-calibration*.json
     and exclude .bak/.pre-train backups + diagnostic noise.

If a future refactor accidentally hardcodes a list, these tests fail.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))


class TestAutoDiscoveryContract:
    def test_per_ticker_uses_directory_scan(self):
        """Per-ticker tests must read from MODELS dir at collection time,
        not a hardcoded ticker list."""
        path = REPO / "tests" / "acceptance" / "model" / "test_per_ticker_exhaustive.py"
        src = path.read_text()
        # Prove the function exists + is called at module-eval time
        assert "def _all_tickers" in src, \
            "per-ticker file must define _all_tickers() helper"
        assert "_TICKERS = _all_tickers()" in src, \
            "per-ticker tests must populate _TICKERS at collection time " \
            "(reading models/ directory). Pre-fix this would silently " \
            "miss new watchlist additions."
        # Must NOT contain a hardcoded ticker list
        assert '_TICKERS = ["' not in src, \
            "per-ticker file must NOT hardcode a ticker list — " \
            "use _all_tickers() so adding NVDA/AAPL/etc to watchlist " \
            "auto-extends the test matrix"

    def test_panel_level_uses_glob_discovery(self):
        """Panel-level template instances must use _auto_discover, not
        hardcoded FILES lists."""
        path = REPO / "tests" / "acceptance" / "model" / "test_template_instances.py"
        src = path.read_text()
        assert "_auto_discover(" in src, (
            "test_template_instances.py must use _auto_discover() so "
            "new panel-ltr.<sidecfg>.json / ngboost-head.<sidecfg>.json "
            "files are tested automatically"
        )
        # The hardcoded literal pattern must not be re-introduced
        assert 'FILES = [\n        "panel-ltr.json",' not in src, (
            "FILES must use auto-discovery, not literal list (regression: "
            "hardcoded lists silently miss new ablation outputs)"
        )

    def test_at_least_one_panel_ltr_discovered(self):
        """Sanity: production has panel-ltr.json on disk → the
        discovery returns a non-empty list."""
        from acceptance.model.test_template_instances import TestPanelLTRStandard
        if not (REPO / "backtesting" / "renquant_104" / "artifacts" / "panel-ltr.json").exists():
            import pytest
            pytest.skip("panel-ltr.json not on disk — fresh checkout")
        assert len(TestPanelLTRStandard.FILES) >= 1, (
            "Auto-discovery returned 0 panel-ltr files; either the glob "
            "is broken or all matches were excluded by the exclude_substrings list"
        )

    def test_per_ticker_count_matches_models_dir(self):
        """The number of tickers found = number of models/<TICKER>/ dirs
        with a policy-metadata.json. Pin this so a future refactor can't
        silently drop a discovery dimension."""
        from acceptance.model.test_per_ticker_exhaustive import _TICKERS
        models_dir = REPO / "backtesting" / "renquant_104" / "models"
        if not models_dir.exists():
            import pytest
            pytest.skip("models/ dir absent")
        on_disk = sum(
            1 for d in models_dir.iterdir()
            if d.is_dir() and (d / f"{d.name}-policy-metadata.json").exists()
        )
        assert len(_TICKERS) == on_disk, (
            f"Auto-discovery found {len(_TICKERS)} tickers but disk has "
            f"{on_disk}. Discovery may be filtering — investigate."
        )
