"""Regression tests pinning the panel-LTR ``lookahead_days`` propagation chain.

Track C8 / P3.3 (2026-05-10). Before this fix the codebase carried THREE
contradictory horizons:

  * ``strategy_config.json::panel_ltr.lookahead_days``   = 10   (config)
  * ``strategy_config.json::model_params.lookahead``     = 5    (legacy shadow)
  * Production artifact ``panel-ltr.alpha158_fund.json`` = 60   (actual)

The production artifact was trained by
``scripts/train_production_model.py`` which hardcodes 60, bypassing the
panel pipeline that reads ``panel_ltr.lookahead_days``. Meanwhile the
walk-forward driver (``scripts/train_walkforward_panel.py``) routed
``panel_ltr.lookahead_days`` (=10) into ``LabelsTask`` and
``BuildPanelTask`` — producing fwd_10d label models while inference
relied on a fwd_60d model.

The fix:

  1. ``panel_frame.resolve_lookahead_days(cfg)`` is the **single source
     of truth** (§5.13.5). All Tasks call it instead of duplicating
     ``int(cfg.get("lookahead_days", DEFAULT))``.
  2. ``panel_ltr.lookahead_days`` is bumped to 60 in
     ``strategy_config.json`` and both side configs
     (``strategy_config.golden.json`` /
     ``strategy_config.alpha158_fund_paper.json``) to match the
     production artifact.
  3. ``model_params.lookahead`` is deleted from all three configs as
     the shadow source — tournament-side code (``training/features.py``)
     still defaults internally to 5 so per-ticker tournament behavior is
     unchanged.

These tests are AUDIT REGRESSION GUARDS per §5.13.3 — they fail loud if
a future change reintroduces a parallel default, lets the shadow key
return, or otherwise breaks the propagation chain.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_STRATEGY_DIR = _REPO_ROOT / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── 1. The resolver itself ──────────────────────────────────────────────────

class TestResolveLookaheadDays:
    """Exercises the canonical helper from ``training_panel.panel_frame``."""

    def test_reads_from_panel_ltr_subdict(self):
        from training_panel.panel_frame import resolve_lookahead_days
        cfg = {"panel_ltr": {"lookahead_days": 60}}
        assert resolve_lookahead_days(cfg) == 60

    def test_accepts_inner_panel_ltr_dict_directly(self):
        from training_panel.panel_frame import resolve_lookahead_days
        # tasks_build_panel.AssemblePanelFrameTask passes the panel_ltr
        # sub-dict (already extracted by SliceWatchlistFramesTask).
        inner = {"lookahead_days": 42}
        assert resolve_lookahead_days(inner) == 42

    def test_returns_int_not_float(self):
        from training_panel.panel_frame import resolve_lookahead_days
        cfg = {"panel_ltr": {"lookahead_days": 60.0}}
        result = resolve_lookahead_days(cfg)
        assert isinstance(result, int) and result == 60

    def test_warns_and_falls_back_when_missing(self, caplog):
        from training_panel.panel_frame import (
            resolve_lookahead_days, DEFAULT_LOOKAHEAD_DAYS,
        )
        cfg = {"panel_ltr": {}}
        with caplog.at_level("WARNING", logger="panel_frame"):
            result = resolve_lookahead_days(cfg)
        assert result == DEFAULT_LOOKAHEAD_DAYS
        assert any("lookahead_days missing" in r.message
                   for r in caplog.records)

    def test_default_matches_production_artifact_horizon(self):
        from training_panel.panel_frame import DEFAULT_LOOKAHEAD_DAYS
        # Production training (scripts/train_production_model.py) and
        # the live artifact panel-ltr.alpha158_fund.json both use 60.
        assert DEFAULT_LOOKAHEAD_DAYS == 60, (
            "DEFAULT_LOOKAHEAD_DAYS must mirror the production artifact "
            "horizon. If you change this you must also update "
            "scripts/train_production_model.py and retrain."
        )


class TestProductionTrainingLookaheadStamp:
    """Production/WF artifacts and calibrators must stamp the label horizon."""

    def test_train_production_artifact_lookahead_follows_label_name(self):
        from scripts.train_production_model import build_artifact
        import numpy as np
        import pandas as pd

        class _FakeBooster:
            def save_raw(self, raw_format="json"):
                assert raw_format == "json"
                return b"{}"

        train = pd.DataFrame({
            "ticker": ["AAA", "BBB"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
        })
        artifact = build_artifact(
            _FakeBooster(),
            ["f1", "f2"],
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            train,
            pd.Timestamp("2024-01-31"),
            "test",
            label_used="fwd_20d_excess",
        )

        assert artifact["lookahead_days"] == 20
        assert artifact["cutoff_embargo_days"] == 20

    def test_fit_calibrator_alpha158_fund_infers_lookahead_from_label(self):
        from scripts.fit_calibrator_alpha158_fund import _infer_label_lookahead_days

        assert _infer_label_lookahead_days("fwd_5d_excess") == 5
        assert _infer_label_lookahead_days("fwd_20d_excess") == 20
        assert _infer_label_lookahead_days("fwd_60d_excess_raw") == 60

    def test_fit_calibrator_main_uses_inferred_lookahead(self):
        import inspect
        from scripts import fit_calibrator_alpha158_fund as mod

        src = inspect.getsource(mod.main)
        assert "lookahead_days = _infer_label_lookahead_days(label_col)" in src
        assert "lookahead_days=lookahead_days" in src
        assert 'metadata["lookahead_days_used"] = lookahead_days' in src


# ── 2. BuildPanelTask propagation (legacy monolith path) ────────────────────

class TestBuildPanelTaskLookaheadPropagation:
    """BuildPanelTask in pp_panel_training.py must forward the resolved
    horizon into ``build_panel_frame(lookahead_days=...)``."""

    def _make_ctx(self, lookahead_days: int):
        from training_panel.context import PanelTrainingContext
        cfg = {
            "panel_ltr": {
                "lookahead_days": lookahead_days,
                "min_history_days": 252,
                "age_warmup_days": 504,
                "nan_prone_cols": [],
            },
        }
        ctx = PanelTrainingContext(config=cfg, watchlist=[])
        # neutralized_frames / labels / etc. all empty — build_panel_frame
        # is patched out below so we don't need real data.
        return ctx

    def _run_with_capture(self, lookahead_days: int) -> dict:
        from training_panel.pp_panel_training import BuildPanelTask
        ctx = self._make_ctx(lookahead_days)
        captured: dict = {}

        def _fake_build_panel_frame(*args, **kwargs):
            captured.update(kwargs)
            # Return minimal valid shape: empty panel, empty groups, meta
            # with all keys the downstream log uses (n_rows / n_tickers /
            # n_dates / feature_cols).
            import numpy as np
            import pandas as pd
            empty = pd.DataFrame({
                "ticker": [], "date": [], "label": [],
            })
            meta = {
                "n_rows": 0, "n_tickers": 0, "n_dates": 0,
                "feature_cols": [],
            }
            return empty, np.array([], dtype="int32"), meta

        with patch(
            "training_panel.panel_frame.build_panel_frame",
            side_effect=_fake_build_panel_frame,
        ):
            BuildPanelTask().run(ctx)
        return captured

    def test_lookahead_10_propagates(self):
        captured = self._run_with_capture(10)
        assert captured.get("lookahead_days") == 10

    def test_lookahead_60_propagates(self):
        captured = self._run_with_capture(60)
        assert captured.get("lookahead_days") == 60

    def test_log_line_present(self, caplog):
        with caplog.at_level("INFO", logger="training_panel"):
            self._run_with_capture(60)
        joined = "\n".join(r.message for r in caplog.records)
        # Match the canonical log statement (§5.13.10: scale-named).
        assert re.search(
            r"lookahead horizon\s*=\s*60 trading days", joined, re.IGNORECASE
        ), f"expected lookahead log line; got:\n{joined}"


# ── 3. LabelsTask uses the same helper ──────────────────────────────────────

class TestLabelsTaskUsesCanonicalHelper:
    """LabelsTask must also route through resolve_lookahead_days so the
    label-horizon and panel-frame horizon cannot drift apart."""

    def test_labelstask_calls_resolve_lookahead_days(self):
        # Grep the source: the runtime call is inside `run()`. Re-importing
        # gives us the literal source text without executing data-heavy
        # branches.
        import inspect
        from training_panel.pp_panel_training import LabelsTask
        src = inspect.getsource(LabelsTask.run)
        assert "resolve_lookahead_days" in src, (
            "LabelsTask.run must call resolve_lookahead_days() to remain "
            "synchronized with BuildPanelTask. Inline `cfg.get(...)` "
            "reads are forbidden — they reintroduce drift."
        )

    def test_buildpaneltask_calls_resolve_lookahead_days(self):
        import inspect
        from training_panel.pp_panel_training import BuildPanelTask
        src = inspect.getsource(BuildPanelTask.run)
        assert "resolve_lookahead_days" in src

    def test_assemblepanelframe_task_calls_resolve_lookahead_days(self):
        # The split Job version under tasks_build_panel.py must also use
        # the helper — historically it had its own `cfg.get(...)` line.
        import inspect
        from training_panel.tasks_build_panel import AssemblePanelFrameTask
        src = inspect.getsource(AssemblePanelFrameTask.run)
        assert "resolve_lookahead_days" in src


# ── 4. Single source of truth — config + grep audit (§5.13.5) ───────────────

class TestSingleSourceOfTruth:
    """Audit-style guards: no shadow key, no parallel default."""

    def test_strategy_config_has_no_model_params_lookahead(self):
        cfg_path = _STRATEGY_DIR / "strategy_config.json"
        cfg = json.loads(cfg_path.read_text())
        mp = cfg.get("model_params", {})
        assert "lookahead" not in mp, (
            "model_params.lookahead must be deleted — it shadowed "
            "panel_ltr.lookahead_days and caused horizon ambiguity."
        )

    def test_side_configs_have_no_model_params_lookahead(self):
        for name in (
            "strategy_config.golden.json",
            "strategy_config.alpha158_fund_paper.json",
        ):
            cfg = json.loads((_STRATEGY_DIR / name).read_text())
            mp = cfg.get("model_params", {})
            assert "lookahead" not in mp, (
                f"{name}: model_params.lookahead must be deleted "
                f"(§5.13.13 side configs must remain coherent with the "
                f"main config)."
            )

    def test_main_and_side_configs_agree_on_panel_ltr_lookahead_days(self):
        names = (
            "strategy_config.json",
            "strategy_config.golden.json",
            "strategy_config.alpha158_fund_paper.json",
        )
        values = {}
        for n in names:
            cfg = json.loads((_STRATEGY_DIR / n).read_text())
            values[n] = cfg["panel_ltr"]["lookahead_days"]
        assert len(set(values.values())) == 1, (
            f"panel_ltr.lookahead_days disagrees across configs: {values}"
        )
        assert next(iter(values.values())) == 60, (
            "Production artifact (panel-ltr.alpha158_fund.json) is "
            "fwd_60d. All configs must say 60."
        )


# ── 5. AUDIT REGRESSION GUARD (§5.13.3) — the invariant ─────────────────────

class TestLookaheadPropagationRegression:
    """The invariant: whatever ``panel_ltr.lookahead_days`` says is what
    ``build_panel_frame`` receives. No magic default, no shadow key."""

    @pytest.mark.parametrize("horizon", [5, 10, 20, 60, 120])
    def test_horizon_flows_through_build_panel_task(self, horizon):
        from training_panel.context import PanelTrainingContext
        from training_panel.pp_panel_training import BuildPanelTask
        cfg = {"panel_ltr": {
            "lookahead_days": horizon,
            "min_history_days": 252,
            "age_warmup_days": 504,
            "nan_prone_cols": [],
        }}
        ctx = PanelTrainingContext(config=cfg, watchlist=[])
        seen = {}

        def _spy(*args, **kwargs):
            seen.update(kwargs)
            import numpy as np
            import pandas as pd
            meta = {"n_rows": 0, "n_tickers": 0, "n_dates": 0,
                    "feature_cols": []}
            return pd.DataFrame({"ticker": [], "date": [], "label": []}), \
                np.array([], dtype="int32"), meta

        with patch(
            "training_panel.panel_frame.build_panel_frame", side_effect=_spy,
        ):
            BuildPanelTask().run(ctx)
        assert seen.get("lookahead_days") == horizon, (
            f"BuildPanelTask did not propagate lookahead_days={horizon} "
            f"into build_panel_frame; observed kwargs={seen}"
        )

    def test_no_more_inline_magic_defaults_in_build_or_labels_tasks(self):
        """Inline ``cfg.get('lookahead_days', N)`` reads with a magic
        numeric default are forbidden inside BuildPanelTask.run and
        LabelsTask.run — they bypass the canonical helper and were the
        2026-05-09 bug pattern."""
        import inspect
        from training_panel.pp_panel_training import BuildPanelTask, LabelsTask
        for cls in (BuildPanelTask, LabelsTask):
            src = inspect.getsource(cls.run)
            assert not re.search(
                r"cfg\.get\(\s*['\"]lookahead_days['\"]\s*,\s*\d+\s*\)",
                src,
            ), (
                f"{cls.__name__}.run still contains inline "
                f"cfg.get('lookahead_days', N) — must route through "
                f"resolve_lookahead_days() instead."
            )

    def test_model_params_lookahead_not_read_anywhere_in_panel_pipeline(self):
        """Grep the panel pipeline source: `mp.get('lookahead', ...)` must
        only appear in the per-ticker feature path (TickerPanelFeatureJob),
        which is the tournament side. It must NOT appear in the panel-LTR
        label / build path."""
        pp_src = (_STRATEGY_DIR / "training_panel" / "pp_panel_training.py").read_text()
        # Count usages — there's exactly one legitimate read for the
        # per-ticker tournament features (inside TickerPanelFeatureJob).
        mp_reads = re.findall(
            r"mp\.get\(\s*['\"]lookahead['\"]\s*,",
            pp_src,
        )
        assert len(mp_reads) <= 1, (
            f"Unexpected mp.get('lookahead', ...) reads in pp_panel_training.py: "
            f"{len(mp_reads)} (expected ≤ 1 for TickerPanelFeatureJob). "
            f"Panel-LTR label/build code must not consult model_params."
        )
