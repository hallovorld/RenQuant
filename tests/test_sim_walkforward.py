"""SimAdapter walk-forward integration tests (Track P2, 2026-05-10).

Defends the leakage class documented in CLAUDE.md §5.13:
    "prod model trained 2026-05-09 used in sim covering 2024-01 → 2026-03"

Coverage:
    1. Legacy static-model behaviour preserved when walkforward.enabled=false.
    2. Legacy path's `assert_no_leakage` fires when model.trained_date
       >= backtest_end (the bug we're fixing).
    3. Walk-forward path: SimAdapter loads via manifest.
    4. `make_context(today)` returns a model with cutoff < today.
    5. ValueError raised when manifest entry's cutoff > today (sim's
       first bar predates every retrain in the manifest).
    6. Per §5.13.1: e2e walk through 3 bars with walk-forward enabled —
       different model used at each retrain boundary (synthetic small
       panel + 2 retrains in manifest).
    7. Per §5.13.2: production sim path imports
       `kernel.walk_forward.leakage_guard` (proves not orphan).
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

_STRATEGY_DIR = (
    Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
)
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ─── Fixtures ──────────────────────────────────────────────────────────────


def _tiny_ohlcv(days: int = 200, seed: int = 0) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-02", periods=days)
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, days)))
    return pd.DataFrame({
        "open": close, "high": close * 1.005, "low": close * 0.995,
        "close": close, "volume": np.ones(days) * 1e6,
    }, index=idx)


def _write_synthetic_panel_artifact(
    path: Path,
    *,
    trained_date: str,
    tag: str = "synthetic",
    **extra_payload,
) -> None:
    """Write a minimal panel-LTR XGBoost artifact PanelScorer.load can read.

    Trains a tiny one-tree booster on 50 random rows × 3 features so the
    artifact is real (loadable) but cheap to build.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3)).astype("float32")
    y = rng.normal(size=50).astype("float32")
    dtrain = xgb.DMatrix(X, label=y)
    booster = xgb.train(
        params={"objective": "reg:squarederror", "max_depth": 2,
                "tree_method": "hist", "verbosity": 0},
        dtrain=dtrain, num_boost_round=2,
    )
    raw = bytes(booster.save_raw(raw_format="json")).decode("utf-8")
    payload = {
        "version": 2,
        "kind": "panel_ltr_xgboost",
        "trained_date": trained_date,
        "feature_cols": ["f0", "f1", "f2"],
        "params": {},
        "best_iter": 1,
        "booster_raw_json": raw,
        "tag": tag,
    }
    payload.update(extra_payload)
    path.write_text(json.dumps(payload, default=str))


@pytest.fixture
def manifest_with_two_retrains(tmp_path: Path) -> Path:
    art_a = tmp_path / "wf_modelA.json"
    art_b = tmp_path / "wf_modelB.json"
    _write_synthetic_panel_artifact(art_a, trained_date="2024-03-15", tag="A")
    _write_synthetic_panel_artifact(art_b, trained_date="2024-06-15", tag="B")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "retrains": [
            {"cutoff_date": "2024-03-01", "trained_date": "2024-03-15",
             "artifact_uri": str(art_a)},
            {"cutoff_date": "2024-06-01", "trained_date": "2024-06-15",
             "artifact_uri": str(art_b)},
        ]
    }))
    return manifest


# ─── Tests ─────────────────────────────────────────────────────────────────


class TestStaticModelBehaviorPreserved:
    """Walk-forward feature flag default OFF — legacy paths unchanged.

    AUDIT REGRESSION GUARD (§5.13.3): SimAdapter MUST behave identically
    when walkforward.enabled is absent or false. Removing/breaking the
    default-off path would silently change every existing sim run.
    """

    def test_default_no_walkforward_loader(self):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {"watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {}}
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._walkforward_loader is None  # noqa: SLF001

    def test_alpha158_static_scorer_skips_legacy_panel_frame_prep(self, tmp_path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(art, trained_date="2024-03-15")
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [],
            "sector_etf_map": {},
            "tax": {},
            "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }

        adapter = SimAdapter(
            config=cfg,
            strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv,
            spy_df=ohlcv["SPY"],
            sector_etf_map={},
            initial_cash=100_000,
        )

        assert adapter._panel_frames_required() is False  # noqa: SLF001
        assert adapter._panel_feature_frames is None  # noqa: SLF001

    def test_walkforward_disabled_explicit(self):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {"enabled": False},
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=_STRATEGY_DIR,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._walkforward_loader is None  # noqa: SLF001


class TestLegacyPathLeakageGuard:
    """Per §5.13.3: invariant — legacy load can't bind a model whose
    trained_date >= backtest_end. The 2026-05-10 audit-class bug must
    never recur."""

    def test_legacy_leakage_raises(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        # Stand up a strategy_dir with an artifact dated AFTER the sim's
        # backtest_end. The default config's panel scoring is off so we
        # must enable it pointing at our synthetic artifact.
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(art, trained_date="2026-05-09")
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        with pytest.raises(ValueError, match="leakage"):
            SimAdapter(
                config=cfg, strategy_dir=tmp_path,
                ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
                initial_cash=100_000,
                backtest_end="2026-03-31",  # before trained_date 2026-05-09
            )

    def test_legacy_no_leakage_passes(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(art, trained_date="2024-01-01")
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        # backtest_end after trained_date — should NOT raise.
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
            backtest_end="2024-09-30",
        )
        assert adapter._panel_scorer is not None  # noqa: SLF001

    def test_static_cutoff_artifact_uses_first_sim_bar(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(
            art,
            trained_date="2026-05-22",
            effective_train_cutoff_date="2024-04-08",
            lookahead_days=60,
        )
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
            backtest_start="2024-07-02",
            backtest_end="2026-03-31",
        )
        assert adapter._panel_scorer is not None  # noqa: SLF001

    def test_static_cutoff_artifact_blocks_pre_cutoff_window(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(
            art,
            trained_date="2026-05-22",
            effective_train_cutoff_date="2024-04-08",
            lookahead_days=60,
        )
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        with pytest.raises(ValueError, match="leakage"):
            SimAdapter(
                config=cfg, strategy_dir=tmp_path,
                ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
                initial_cash=100_000,
                backtest_start="2024-06-28",
                backtest_end="2026-03-31",
            )

    def test_static_artifact_missing_trained_date_hard_fails(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(art, trained_date=None)
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        with pytest.raises(ValueError, match="missing trained_date"):
            SimAdapter(
                config=cfg, strategy_dir=tmp_path,
                ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
                initial_cash=100_000,
                backtest_start="2024-01-02",
                backtest_end="2024-03-31",
            )

    def test_static_artifact_blocks_validation_selection_leakage(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        art = tmp_path / "panel-ltr.json"
        _write_synthetic_panel_artifact(
            art,
            trained_date="2026-05-22",
            effective_train_cutoff_date="2024-04-08",
            lookahead_days=60,
            split_date_ranges={
                "train": {"start": "2020-01-01", "end": "2024-04-08"},
                "val": {"start": "2024-07-02", "end": "2024-09-30"},
            },
        )
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "ranking": {
                "panel_scoring": {
                    "enabled": True,
                    "artifact_path": str(art),
                },
            },
        }
        with pytest.raises(ValueError, match="leakage"):
            SimAdapter(
                config=cfg, strategy_dir=tmp_path,
                ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
                initial_cash=100_000,
                backtest_start="2024-10-01",
                backtest_end="2024-12-31",
            )


class TestWalkforwardLoading:
    def test_walkforward_loader_instantiated(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(manifest_with_two_retrains),
                "fail_on_no_model": True,
            },
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._walkforward_loader is not None  # noqa: SLF001
        # Legacy static scorer must be None when walkforward owns it.
        assert adapter._panel_scorer is None  # noqa: SLF001

    def test_missing_manifest_fail_on_no_model_true(self, tmp_path: Path):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(tmp_path / "missing.json"),
                "fail_on_no_model": True,
            },
        }
        with pytest.raises(FileNotFoundError, match="walkforward manifest"):
            SimAdapter(
                config=cfg, strategy_dir=tmp_path,
                ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
                initial_cash=100_000,
            )

    def test_missing_manifest_fail_on_no_model_false_falls_back(
        self, tmp_path: Path,
    ):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(tmp_path / "missing.json"),
                "fail_on_no_model": False,
            },
        }
        # No raise — falls back to legacy (which has no artifact, so
        # _panel_scorer remains None too).
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        assert adapter._walkforward_loader is None  # noqa: SLF001


class TestMakeContextPerBarLookup:
    def test_make_context_picks_correct_retrain(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(manifest_with_two_retrains),
            },
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        # Bar between cutoff A (2024-03-01) and cutoff B (2024-06-01) →
        # model A. Bar after cutoff B → model B.
        bar_after_a = pd.Timestamp("2024-04-15")
        scorer_a = adapter._get_panel_scorer_for_bar(bar_after_a)  # noqa: SLF001
        assert scorer_a is not None
        assert scorer_a.metadata.get("tag") == "A"

        bar_after_b = pd.Timestamp("2024-09-15")
        scorer_b = adapter._get_panel_scorer_for_bar(bar_after_b)  # noqa: SLF001
        assert scorer_b is not None
        assert scorer_b.metadata.get("tag") == "B"

    def test_make_context_raises_when_no_eligible_retrain(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv()}
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(manifest_with_two_retrains),
            },
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        # First retrain's cutoff is 2024-03-01 → any bar at/before that
        # has zero eligible retrains.
        with pytest.raises(ValueError, match="no retrain"):
            adapter._get_panel_scorer_for_bar(pd.Timestamp("2024-02-01"))  # noqa: SLF001


class TestEndToEndPerBarSwitch:
    """§5.13.1 — synthetic panel through actual SimAdapter, walking 3
    bars with walk-forward on. Verifies different model is used at each
    retrain boundary (not a hand-rolled fixture in isolation)."""

    def test_three_bars_two_retrain_boundaries(
        self, tmp_path: Path, manifest_with_two_retrains: Path,
    ):
        from adapters.sim import SimAdapter
        ohlcv = {"SPY": _tiny_ohlcv(days=400)}  # spans 2024 fully
        cfg = {
            "watchlist": [], "sector_etf_map": {}, "tax": {}, "regime": {},
            "walkforward": {
                "enabled": True,
                "manifest_path": str(manifest_with_two_retrains),
            },
        }
        adapter = SimAdapter(
            config=cfg, strategy_dir=tmp_path,
            ohlcv=ohlcv, spy_df=ohlcv["SPY"], sector_etf_map={},
            initial_cash=100_000,
        )
        # Bar 1: 2024-04-15 — after cutoff A only. Bar 2: 2024-07-15 —
        # after cutoff B. Bar 3: 2024-10-15 — still after cutoff B.
        # Verify scorer identity changes between bars 1 and 2, but
        # bars 2 and 3 share the same scorer.
        # Pick actual SPY index dates so make_context's SPY-return
        # buffer update doesn't crash.
        idx = ohlcv["SPY"].index
        bar1 = next(d for d in idx if d >= pd.Timestamp("2024-04-15"))
        bar2 = next(d for d in idx if d >= pd.Timestamp("2024-07-15"))
        bar3 = next(d for d in idx if d >= pd.Timestamp("2024-10-15"))

        ctx1 = adapter.make_context(bar1)
        ctx2 = adapter.make_context(bar2)
        ctx3 = adapter.make_context(bar3)

        s1 = getattr(ctx1, "_panel_scorer", None)
        s2 = getattr(ctx2, "_panel_scorer", None)
        s3 = getattr(ctx3, "_panel_scorer", None)

        assert s1 is not None and s2 is not None and s3 is not None
        assert s1.metadata.get("tag") == "A"
        assert s2.metadata.get("tag") == "B"
        assert s3.metadata.get("tag") == "B"
        # Bar2 and bar3 must share the cached scorer instance.
        assert s2 is s3
        # Bar1 must differ from bar2.
        assert s1 is not s2


class TestNotOrphan:
    """§5.13.2: a module is dead code until grep proves prod imports it.

    Verify SimAdapter (the production sim entry point) actually imports
    and uses `kernel.walk_forward.leakage_guard`.
    """

    def test_sim_imports_leakage_guard(self):
        # Import the production module and check the guard symbol is
        # reachable from the same code path that's loaded in production.
        import adapters.sim as sim_mod
        # The function isn't a module-level symbol on sim_mod (it's
        # imported lazily inside _assert_legacy_no_leakage) — so we
        # test the production path by reading the source for the call.
        src = Path(sim_mod.__file__).read_text()
        assert "from kernel.walk_forward.leakage_guard import assert_no_leakage" in src, (
            "adapters.sim must import assert_no_leakage from "
            "kernel.walk_forward.leakage_guard. If this assertion fails, "
            "the leakage guard module is orphaned (§5.13.2)."
        )
        # Belt-and-braces: the guard module itself must import cleanly.
        from kernel.walk_forward.leakage_guard import assert_no_leakage  # noqa: F401
