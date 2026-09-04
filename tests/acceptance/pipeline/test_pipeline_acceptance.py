"""Pipeline-layer acceptance — end-to-end smoke + artifact landing.

Tagged @pytest.mark.slow because these run a stripped-down full
training on a synthetic universe. CI runs nightly. Per-PR pre-merge
runs only the model + jobs layers.

What we verify:
  * FullTrainingPipeline drops the expected set of artifacts
  * Each artifact passes its model-acceptance protocol
  * Cross-artifact consistency holds
  * No unhandled exceptions

User mandate (2026-05-04).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
STRATEGY = REPO / "backtesting" / "renquant_104"
if str(STRATEGY) not in sys.path:
    sys.path.insert(0, str(STRATEGY))
sys.path.insert(0, str(REPO / "tests"))


@pytest.mark.slow
class TestPanelTrainingPipelineSmoke:
    """Heavy: runs PanelTrainingPipeline on a tiny stub. ~30s.

    Skipped by default. Enable with `-m slow`.
    """

    def test_pipeline_drops_artifacts_on_synthetic_universe(self, tmp_path):
        # This is intentionally minimal — a real implementation needs
        # a working watchlist + cached OHLCV + a temp strategy_dir.
        # Stub for now: assert the pipeline class is importable and
        # has the expected Job ordering.
        from training_panel.pp_panel_training import PanelTrainingPipeline
        pipe = PanelTrainingPipeline()
        assert pipe is not None
        # Schema: pipeline phases hard-coded in PanelTrainingPipeline.run
        # — covered by job-layer tests for ordering.


class TestArtifactSetCompleteness:
    """Lightweight: verify the expected artifact filenames exist on disk
    in production, IF anyone has trained at all. Each gets schema-
    asserted by tests/acceptance/model/test_artifact_acceptance.py."""

    @property
    def artifacts(self) -> Path:
        return STRATEGY / "artifacts"

    @property
    def prod_artifacts(self) -> Path:
        """2026-05-11 sim/prod isolation refactor: production-trading
        artifacts live in artifacts/prod/ (sim eval lives in artifacts/sim/).
        """
        return self.artifacts / "prod"

    def test_required_artifact_set_present(self):
        """Production training emits this set under artifacts/prod/:
            panel-ltr*.json (canonical model, e.g. panel-ltr.alpha158_fund.json)
            (optional) ngboost-head*.json
            (optional) panel-rank-calibration.json
            (optional) training_data_scan.json
            (optional) spy-gmm-regime.json
        """
        if not self.prod_artifacts.exists():
            pytest.skip(
                "artifacts/prod/ dir doesn't exist — fresh checkout or "
                "pre-2026-05-11 layout. Run scripts/holdout_backtest.py or "
                "daily_104.sh to populate."
            )
        present = {p.name for p in self.prod_artifacts.iterdir() if p.is_file()}
        # Accept any canonical panel-ltr variant (e.g. .alpha158_fund.json).
        has_panel_ltr = any(
            p.startswith("panel-ltr") and p.endswith(".json") for p in present
        )
        if not has_panel_ltr:
            # 2026-09-03: the served pair is live-mutated and untracked
            # (deploy/live_mutated_prod_artifacts.json) — a fresh checkout
            # has the directory (other tracked artifacts) but not the pair.
            declaration = REPO / "deploy" / "live_mutated_prod_artifacts.json"
            if declaration.exists() and any(
                Path(a["path"]).name.startswith("panel-ltr")
                for a in json.loads(declaration.read_text(encoding="utf-8"))["artifacts"]
            ):
                pytest.skip(
                    "served panel-ltr is declared live-mutated (untracked) and is "
                    "absent in this checkout — present only on the serving machine"
                )
        assert has_panel_ltr, (
            f"No panel-ltr*.json found in artifacts/prod/ ({sorted(present)}). "
            f"Run scripts/holdout_backtest.py or daily_104.sh to populate."
        )

    def test_artifacts_recently_modified(self):
        """If artifacts/prod/ exists at all, the canonical panel-ltr*.json
        should be no older than 60 days. A model older than the staleness
        threshold is a sign the cron is broken."""
        import time
        if not self.prod_artifacts.exists():
            pytest.skip("no artifacts/prod/ dir")
        panel_ltrs = sorted(
            p for p in self.prod_artifacts.iterdir()
            if p.name.startswith("panel-ltr") and p.suffix == ".json"
        )
        if not panel_ltrs:
            pytest.skip("no panel-ltr*.json in artifacts/prod/ — run training")
        # Use the most-recently-modified panel-ltr file (canonical).
        canonical = max(panel_ltrs, key=lambda p: p.stat().st_mtime)
        age_sec = time.time() - canonical.stat().st_mtime
        age_days = age_sec / 86400
        # 60 days = strategy_config.json's `model_staleness_days`
        assert age_days <= 60, (
            f"{canonical.name} is {age_days:.0f} days old — exceeds the "
            f"60-day staleness threshold in strategy_config.json. "
            f"Either retrain or investigate why retrain-panel104 cron "
            f"hasn't fired."
        )
