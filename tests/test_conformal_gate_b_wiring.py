"""Tests for M3 conformal Gate B wiring + fitted artifact contract.

Roadmap §M3: regime-conditional Gate B threshold replaces the static
``ranking.panel_scoring.quality_floor.edge_sharpe_floor.threshold``
when ``artifacts/gate_b_thresholds.json`` is present and fresh.

Behavior under test:
  * Artifact missing → fall back to config static τ.
  * Artifact stale (> conformal_max_age_days) → fall back + warn.
  * Regime missing from artifact (insufficient samples in fit) → fall
    back to static τ for that regime only.
  * Valid artifact + known regime → conformal τ used.
  * τ out of [0,1] in artifact → fall back (defensive).

The fitted artifact (produced by scripts/fit_conformal_gate_b.py) on
2026-05-01 contains BULL_CALM and CHOPPY thresholds; BULL_VOLATILE and
BEAR fell out of the fit due to <100 historical samples in those
regimes — the wiring must handle the partial-fit case gracefully.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.pipeline.context import InferenceContext  # noqa: E402
from kernel.panel_pipeline.task_quality_floor import (  # noqa: E402
    QualityFloorTask,
)


def _ctx_with_strategy_dir(strategy_dir: Path,
                            regime: str = "BULL_CALM") -> InferenceContext:
    ctx = InferenceContext(
        config={
            "_strategy_dir": str(strategy_dir),
            "ranking": {"panel_scoring": {"quality_floor": {
                "edge_sharpe_floor": {
                    "enabled": True,
                    "use_conformal": True,
                    "threshold": 0.20,         # static fallback
                    "conformal_max_age_days": 7,
                },
            }}},
        },
        today=_dt.date(2026, 5, 1),
    )
    ctx.regime = regime
    return ctx


# ── Artifact missing → fall back to static τ ──────────────────────────────────

class TestArtifactMissing:
    def test_returns_none_when_artifact_absent(self, tmp_path: Path):
        # tmp_path has no artifacts/gate_b_thresholds.json
        (tmp_path / "artifacts").mkdir()
        ctx = _ctx_with_strategy_dir(tmp_path)
        tau = QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM")
        assert tau is None, "missing artifact must fall back to static τ"


# ── Artifact stale → fall back ───────────────────────────────────────────────

class TestStaleness:
    def test_artifact_older_than_max_age_falls_back(self, tmp_path: Path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        # Fit timestamp 30 days ago
        old = (_dt.datetime.utcnow() - _dt.timedelta(days=30)).isoformat()
        (artifacts / "gate_b_thresholds.json").write_text(json.dumps({
            "fitted_at": old,
            "thresholds": {"BULL_CALM": 0.10, "CHOPPY": 0.15},
        }))
        ctx = _ctx_with_strategy_dir(tmp_path)
        tau = QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM")
        assert tau is None, "stale artifact must fall back to static τ"

    def test_max_age_zero_disables_staleness_check(self, tmp_path: Path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        old = (_dt.datetime.utcnow() - _dt.timedelta(days=365)).isoformat()
        (artifacts / "gate_b_thresholds.json").write_text(json.dumps({
            "fitted_at": old,
            "thresholds": {"BULL_CALM": 0.10},
        }))
        ctx = _ctx_with_strategy_dir(tmp_path)
        ctx.config["ranking"]["panel_scoring"]["quality_floor"][
            "edge_sharpe_floor"]["conformal_max_age_days"] = 0
        tau = QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM")
        assert tau == 0.10


# ── Partial fit (BULL_VOLATILE / BEAR missing) → per-regime fallback ─────────

class TestPartialFit:
    """The 2026-05-01 fit produced thresholds only for BULL_CALM + CHOPPY
    (BULL_VOLATILE n=408, BEAR n=101 — both below min_samples=100 floor
    after audit constraints). Those regimes must fall back per-regime to
    the static τ, NOT take down the whole gate.
    """

    def _make_partial_artifact(self, dir_: Path) -> None:
        artifacts = dir_ / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "gate_b_thresholds.json").write_text(json.dumps({
            "fitted_at": _dt.datetime.utcnow().isoformat(),
            "thresholds": {
                "BULL_CALM": 0.090,
                "CHOPPY":    0.020,
                # BULL_VOLATILE + BEAR omitted — under-sampled
            },
        }))

    def test_fitted_regime_returns_tau(self, tmp_path: Path):
        self._make_partial_artifact(tmp_path)
        ctx = _ctx_with_strategy_dir(tmp_path, regime="BULL_CALM")
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") == 0.090

    def test_fitted_regime_choppy_returns_tau(self, tmp_path: Path):
        self._make_partial_artifact(tmp_path)
        ctx = _ctx_with_strategy_dir(tmp_path, regime="CHOPPY")
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "CHOPPY") == 0.020

    def test_unfit_regime_returns_none(self, tmp_path: Path):
        """BULL_VOLATILE / BEAR not in the artifact → caller falls back to
        the config static τ. Without this, the whole panel gates off in
        those regimes."""
        self._make_partial_artifact(tmp_path)
        ctx = _ctx_with_strategy_dir(tmp_path, regime="BULL_VOLATILE")
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_VOLATILE") is None
        ctx2 = _ctx_with_strategy_dir(tmp_path, regime="BEAR")
        assert QualityFloorTask._gate_b_conformal_tau(ctx2, "BEAR") is None


# ── Defensive: malformed τ values ────────────────────────────────────────────

class TestDefensive:
    def _write_artifact(self, dir_: Path, thresholds: dict) -> None:
        artifacts = dir_ / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / "gate_b_thresholds.json").write_text(json.dumps({
            "fitted_at": _dt.datetime.utcnow().isoformat(),
            "thresholds": thresholds,
        }))

    def test_negative_tau_rejected(self, tmp_path: Path):
        self._write_artifact(tmp_path, {"BULL_CALM": -0.10})
        ctx = _ctx_with_strategy_dir(tmp_path)
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None

    def test_above_one_tau_rejected(self, tmp_path: Path):
        self._write_artifact(tmp_path, {"BULL_CALM": 1.5})
        ctx = _ctx_with_strategy_dir(tmp_path)
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None

    def test_string_tau_rejected(self, tmp_path: Path):
        self._write_artifact(tmp_path, {"BULL_CALM": "garbage"})
        ctx = _ctx_with_strategy_dir(tmp_path)
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None

    def test_none_regime_returns_none(self, tmp_path: Path):
        self._write_artifact(tmp_path, {"BULL_CALM": 0.10})
        ctx = _ctx_with_strategy_dir(tmp_path)
        assert QualityFloorTask._gate_b_conformal_tau(ctx, None) is None

    def test_thresholds_not_dict_rejected(self, tmp_path: Path):
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (artifacts / "gate_b_thresholds.json").write_text(json.dumps({
            "fitted_at": _dt.datetime.utcnow().isoformat(),
            "thresholds": "not a dict",
        }))
        ctx = _ctx_with_strategy_dir(tmp_path)
        assert QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM") is None


# ── Live artifact smoke test: the freshly fitted prod artifact loads ─────────

class TestLiveArtifact:
    """Exercise the actual on-disk artifact produced by the 2026-05-01
    fit. If this test fails, either the artifact got corrupted or the
    schema drifted and the wiring would silently fall back at inference.
    """

    def test_production_artifact_loads_and_returns_known_regime(self):
        strategy_dir = REPO_ROOT / "backtesting" / "renquant_104"
        artifact = strategy_dir / "artifacts" / "gate_b_thresholds.json"
        if not artifact.exists():
            pytest.skip("gate_b_thresholds.json not present (run fit_conformal_gate_b.py)")
        ctx = _ctx_with_strategy_dir(strategy_dir, regime="BULL_CALM")
        # Disable staleness for this smoke test — the artifact may be
        # weeks old in CI; we only care that the loader/parser path
        # works end-to-end and returns a reasonable τ.
        ctx.config["ranking"]["panel_scoring"]["quality_floor"][
            "edge_sharpe_floor"]["conformal_max_age_days"] = 0
        tau = QualityFloorTask._gate_b_conformal_tau(ctx, "BULL_CALM")
        assert tau is not None, (
            "BULL_CALM is the highest-volume regime; the live fit MUST "
            "produce a τ for it. Missing → fit pipeline broken."
        )
        assert 0.0 < tau < 1.0


# ── Backfill script enhancement: benchmarks now covered ──────────────────────

class TestBackfillBenchmark:
    """Pre-2026-05-01, scripts/backfill_forward_returns.py only covered
    candidate tickers. SPY (the benchmark) was missing from
    ticker_forward_returns, so the conformal Gate B fit's LEFT JOIN
    nulled every row. Now the script accepts --benchmarks (default SPY)
    and emits (run_date, benchmark) pairs for missing fwd values.
    """

    def test_script_accepts_benchmarks_flag(self):
        import subprocess
        out = subprocess.run(
            ["python", str(REPO_ROOT / "scripts" / "backfill_forward_returns.py"),
             "--help"],
            capture_output=True, text=True, timeout=30,
        )
        assert "--benchmarks" in out.stdout, (
            "scripts/backfill_forward_returns.py must expose --benchmarks "
            "so operators can ensure SPY is covered for M3 fits"
        )

    def test_benchmark_pairs_helper_returns_missing_only(self, tmp_path: Path):
        """The helper must NOT emit pairs that already have all 4
        forward returns — otherwise we'd re-fetch tens of thousands of
        already-backfilled (date, benchmark) rows on every run."""
        import sqlite3
        # Build a tiny test DB: one already-complete row, one incomplete
        db = tmp_path / "test.db"
        conn = sqlite3.connect(db)
        conn.executescript("""
            CREATE TABLE pipeline_runs (run_id TEXT, run_date TEXT);
            CREATE TABLE ticker_forward_returns (
                as_of_date TEXT, ticker TEXT,
                fwd_1d REAL, fwd_5d REAL, fwd_10d REAL, fwd_20d REAL,
                PRIMARY KEY (as_of_date, ticker)
            );
            INSERT INTO pipeline_runs VALUES ('r1', '2025-01-02');
            INSERT INTO pipeline_runs VALUES ('r2', '2025-01-03');
            INSERT INTO ticker_forward_returns VALUES
                ('2025-01-02','SPY', 0.01, 0.02, 0.03, 0.04);
        """)
        conn.commit()

        # Import the helper from the script
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "_bf", REPO_ROOT / "scripts" / "backfill_forward_returns.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)   # type: ignore[union-attr]

        pairs = mod._benchmark_pairs(conn, ["SPY"], None)
        # Only 2025-01-03 should appear (2025-01-02 already has all fwd_*)
        assert pairs == [("2025-01-03", "SPY")], (
            f"benchmark backfill must skip already-complete rows; got {pairs}"
        )
