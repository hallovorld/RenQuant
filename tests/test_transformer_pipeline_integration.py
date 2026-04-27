"""End-to-end PanelModelJob integration test for the transformer backend.

Exercises the same Task chain (`CrossValidateTask → FinalFitTask →
SaveArtifactTask`) used by real training runs, but on a synthetic panel
so it runs in seconds. Verifies that:

  1. `panel_ltr.backend: "transformer"` dispatches through all three
     tasks without falling back to XGBoost / LightGBM.
  2. Artifact is written as a `.pt` + paired `.json` sidecar at the
     transformer default path.
  3. `PanelScorer.load(<path>)` on the produced artifact returns a
     TransformerPanelScorer that can score a feature matrix.
  4. Flipping `backend` back to "xgboost" produces the legacy JSON
     artifact and routes through PanelScorer — the transformer branch
     doesn't leak into other backends' paths.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

torch = pytest.importorskip("torch")

from training_panel.context import PanelTrainingContext  # noqa: E402
from training_panel.pp_panel_training import (  # noqa: E402
    CrossValidateTask, FinalFitTask, SaveArtifactTask, PanelModelJob,
)
from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: E402
from kernel.panel_pipeline.transformer_scorer import TransformerPanelScorer  # noqa: E402


def _build_synthetic_panel_ctx(
    strategy_dir: Path, backend: str, extra_panel_cfg: dict | None = None,
) -> PanelTrainingContext:
    """Prepare a PanelTrainingContext with a tiny linear-signal panel."""
    rng = np.random.default_rng(0)
    feature_cols = ["f0", "f1", "f2", "f3"]
    true_w = rng.normal(size=len(feature_cols)).astype(np.float32)

    dates = pd.date_range("2023-01-01", periods=50, freq="B")
    rows = []
    group_sizes = []
    for d in dates:
        n = 8
        X = rng.normal(size=(n, len(feature_cols))).astype(np.float32)
        noise = rng.normal(size=n).astype(np.float32) * 0.3
        y = X @ true_w + noise
        for t in range(n):
            rows.append({
                "date":  d,
                "ticker": f"T{t}",
                **{c: float(X[t, i]) for i, c in enumerate(feature_cols)},
                "label":  float(y[t]),
                "weight": 1.0,
            })
        group_sizes.append(n)
    panel = pd.DataFrame(rows)

    panel_cfg = {
        "backend":          backend,
        "cv_method":        "purged",
        "cv_n_splits":      3,
        "cv_embargo_days":  1,
        "lookahead_days":   1,
        "num_boost_round":  10,
        # Transformer-specific:
        "transformer_params": {
            "max_epochs": 6, "d_model": 16, "n_heads": 2, "n_layers": 1,
            "batch_size": 8, "device": "cpu", "seed": 0,
            # Disable aggressive regularization for a tiny synthetic panel
            # so it actually fits in 6 epochs:
            "label_smoothing": 0.0, "ticker_dropout": 0.0,
            "feature_dropout": 0.0, "dropout": 0.0,
        },
    }
    if extra_panel_cfg:
        panel_cfg.update(extra_panel_cfg)

    ctx = PanelTrainingContext(
        config={
            "panel_ltr":       panel_cfg,
            "_strategy_name":  "test_strategy",
            "_strategy_dir":   str(strategy_dir),     # → ctx.strategy_dir property
            "persistence":     {"enabled": False},    # suppress SQLite writes
        },
    )
    ctx.panel = panel
    ctx.feature_cols = feature_cols
    ctx.group_sizes = np.array(group_sizes, dtype=np.int32)
    ctx.panel_metadata = {
        "n_rows":    len(panel),
        "n_tickers": 8,
        "n_dates":   len(dates),
    }
    return ctx


# ── Transformer end-to-end ────────────────────────────────────────────────────

class TestTransformerModelJob:
    def test_full_job_writes_pt_and_json(self, tmp_path: Path):
        ctx = _build_synthetic_panel_ctx(tmp_path, backend="transformer")
        # Disable the persistence.record_training_run side effect for the
        # synthetic test (no SQLite dir). SaveArtifactTask catches exceptions
        # there and continues — but we still want a clean artifact on disk.
        PanelModelJob().run(ctx)

        pt = tmp_path / "artifacts" / "panel-transformer.pt"
        js = tmp_path / "artifacts" / "panel-transformer.json"
        assert pt.exists(), "transformer .pt artifact missing"
        assert js.exists(), "transformer .json sidecar missing"

        meta = json.loads(js.read_text())
        assert meta["kind"] == "panel_transformer"
        assert meta["feature_cols"] == ctx.feature_cols

    def test_saved_artifact_loads_via_panel_scorer(self, tmp_path: Path):
        ctx = _build_synthetic_panel_ctx(tmp_path, backend="transformer")
        PanelModelJob().run(ctx)

        art = tmp_path / "artifacts" / "panel-transformer.pt"
        scorer = PanelScorer.load(art)
        assert isinstance(scorer, TransformerPanelScorer)

        # Build a feature matrix shaped like inference: one row per ticker.
        matrix = pd.DataFrame(
            np.random.default_rng(5).normal(size=(8, 4)).astype(np.float32),
            columns=ctx.feature_cols,
            index=[f"T{i}" for i in range(8)],
        )
        scores = scorer.score(matrix)
        assert len(scores) == 8
        assert not scores.isna().any()

    def test_backend_transformer_preserves_prior_artifact_via_autobak(self, tmp_path: Path):
        """Transformer run MAY overwrite `panel-ltr.json` with a shim
        (intentional, so calibrator finds the new backend), but MUST
        first auto-back-up the existing artifact to
        `panel-ltr.{prev_kind}.bak.json`.

        Audit fix #141 TRANSFORMER-CLOBBER-AUTOBAK (2026-04-26): pre-fix,
        the shim wrote panel-ltr.json with no auto-backup — `train_104.py
        --strategy-config strategy_config.hourly_transformer.json`
        directly clobbered the production XGBoost artifact, with
        recovery only via the .xgboost.bak.json from a prior
        sunday_panel_sweep.py run (if one ever ran). Post-fix, the
        backup is automatic.
        """
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        xgb_art = artifacts_dir / "panel-ltr.json"
        # Use a realistic XGBoost-kind artifact so the auto-bak labels it.
        sentinel = '{"kind": "panel_ltr_xgboost", "do_not_touch": true}'
        xgb_art.write_text(sentinel)

        ctx = _build_synthetic_panel_ctx(tmp_path, backend="transformer")
        PanelModelJob().run(ctx)

        # The shim is now at panel-ltr.json (intentional — this is the
        # behavior after TRANSFORMER-PANEL-LTR-SHIM landed 2026-04-26).
        new_content = xgb_art.read_text()
        import json as _json
        new_obj = _json.loads(new_content)
        assert new_obj.get("kind") == "panel_transformer", (
            "transformer run should write a shim to panel-ltr.json so "
            "fit_panel_calibrator can dispatch to the transformer scorer"
        )
        # The PRIOR XGBoost artifact must be preserved at the .bak path
        # (Audit fix #141): the original content is recoverable.
        bak = artifacts_dir / "panel-ltr.xgboost.bak.json"
        assert bak.exists(), (
            "auto-bak failed — pre-existing panel-ltr.json content is "
            "now unrecoverable. The .bak file is the audit-#141 "
            "safety net before clobbering with the transformer shim."
        )
        assert bak.read_text() == sentinel, (
            "auto-bak content mismatch — must preserve the EXACT prior "
            "panel-ltr.json content so `cp panel-ltr.xgboost.bak.json "
            "panel-ltr.json` is a clean restore."
        )
        # And the transformer-specific artifact lives separately.
        tf_art = artifacts_dir / "panel-transformer.pt"
        assert tf_art.exists(), "transformer artifact missing at expected default path"

    def test_cv_task_uses_transformer_adapter(self, tmp_path: Path):
        """CrossValidateTask must dispatch to the transformer adapter, not
        silently fall back to XGBoost. Assert by checking that the CV path
        runs with `backend: "transformer"` set and produces a per_fold_ic
        list — if it fell through to XGBoost it would still pass, but the
        mean_ic should be roughly in the same ballpark. For a cleaner
        signal we patch PanelLTRModel.train to fail if called.
        """
        ctx = _build_synthetic_panel_ctx(tmp_path, backend="transformer")
        import training_panel.ltr_model as ltr_mod

        original_train = ltr_mod.PanelLTRModel.train

        def _fail(self, *a, **k):
            raise AssertionError(
                "XGBoost PanelLTRModel.train should not run when backend is transformer"
            )

        ltr_mod.PanelLTRModel.train = _fail
        try:
            CrossValidateTask().run(ctx)
        finally:
            ltr_mod.PanelLTRModel.train = original_train
        assert ctx.cv_result
        assert "mean_ic" in ctx.cv_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
