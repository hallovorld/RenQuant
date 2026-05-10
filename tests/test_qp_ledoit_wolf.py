"""C3 — Ledoit-Wolf shrinkage default ON (λ=0.2).

Pre-fix: `qp_ledoit_wolf_lambda=0.0` → ShrinkSigmaLedoitWolfTask short-
circuited → 169-stock universe Σ used the raw sample correlation, often
near-singular and dominated by noise eigenvalues. Diagonal degeneration
when Σ_full was sanitized away.

Post-fix:
  1. DEFAULT_LAMBDA = 0.2 (Ledoit-Wolf 2004 mid-of-recommended range).
  2. Eigenvalue floor = 1e-8 to guarantee strict PSD post-shrinkage.
  3. Production strategy_config.json sets qp_ledoit_wolf_lambda = 0.2.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backtesting" / "renquant_104"))

from kernel.portfolio_qp.tasks import ShrinkSigmaLedoitWolfTask  # noqa: E402


# ── Mathematical correctness ──────────────────────────────────────────────────

class TestLedoitWolfMath:

    def _ctx(self, S, lam):
        from types import SimpleNamespace
        ctx = SimpleNamespace(
            _qp_Sigma_full=S,
            config={"rotation": {"joint_actions": {
                "qp_ledoit_wolf_lambda": lam,
            }}},
        )
        return ctx

    def test_zero_lambda_no_change(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        ctx = self._ctx(S.copy(), 0.0)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        # λ=0 → off (no eigenvalue floor either)
        np.testing.assert_array_equal(ctx._qp_Sigma_full, S)

    def test_one_lambda_yields_identity(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        avg = float(np.trace(S)) / 2.0
        ctx = self._ctx(S.copy(), 1.0)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        # Should be ≈ avg·I (with eigenvalue floor at 1e-8 untouched here)
        np.testing.assert_allclose(
            ctx._qp_Sigma_full, avg * np.eye(2), atol=1e-10,
        )

    def test_half_lambda_blends(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        avg = float(np.trace(S)) / 2.0
        F = avg * np.eye(2)
        expected = 0.5 * S + 0.5 * F
        ctx = self._ctx(S.copy(), 0.5)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        np.testing.assert_allclose(ctx._qp_Sigma_full, expected, atol=1e-10)

    def test_eigenvalues_floored_above_1e_minus_8(self):
        """Σ with one near-zero eigenvalue must be lifted ≥ 1e-8."""
        # Construct a near-singular Σ: rank-1 plus tiny noise.
        v = np.array([[1.0, 1.0]])
        S = v.T @ v + 1e-20 * np.eye(2)        # near-singular (eig = ~2, ~1e-20)
        ctx = self._ctx(S.copy(), 0.5)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        eigvals = np.linalg.eigvalsh(ctx._qp_Sigma_full)
        assert eigvals.min() >= 1e-8 - 1e-12

    def test_small_n_more_stable_with_lw(self):
        """Small-N: LW shrinkage condition number better than raw."""
        # 3-stock highly-correlated near-singular Σ
        rho = 0.99
        s = 0.15
        S_raw = np.array([
            [s*s,         rho*s*s,     rho*s*s],
            [rho*s*s,     s*s,         rho*s*s],
            [rho*s*s,     rho*s*s,     s*s    ],
        ])
        cond_raw = np.linalg.cond(S_raw)
        ctx = self._ctx(S_raw.copy(), 0.3)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        cond_shrunk = np.linalg.cond(ctx._qp_Sigma_full)
        assert cond_shrunk < cond_raw, \
            f"shrinkage should improve conditioning: {cond_shrunk} vs {cond_raw}"

    def test_negative_lambda_treated_as_zero(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        ctx = self._ctx(S.copy(), -0.5)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        # Negative λ ignored → off
        np.testing.assert_array_equal(ctx._qp_Sigma_full, S)

    def test_nan_lambda_treated_as_zero(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        ctx = self._ctx(S.copy(), float("nan"))
        ShrinkSigmaLedoitWolfTask().run(ctx)
        np.testing.assert_array_equal(ctx._qp_Sigma_full, S)

    def test_above_one_clamped(self):
        S = np.array([[0.04, 0.01], [0.01, 0.09]])
        avg = float(np.trace(S)) / 2.0
        ctx = self._ctx(S.copy(), 5.0)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        np.testing.assert_allclose(
            ctx._qp_Sigma_full, avg * np.eye(2), atol=1e-10,
        )

    def test_none_sigma_full_skipped(self):
        ctx = self._ctx(None, 0.5)
        ShrinkSigmaLedoitWolfTask().run(ctx)
        assert ctx._qp_Sigma_full is None


# ── §5.13.3 audit-regression-guard: default ON ────────────────────────────────

class TestLedoitWolfDefaultEnabled:
    """Pin invariant: shrinkage default lambda is positive, not 0.

    Pre-fix bug-class: someone reverts qp_ledoit_wolf_lambda back to 0.0
    in a refactor → 169-stock Σ degenerates → solver instability.
    Post-fix invariant: ShrinkSigmaLedoitWolfTask.DEFAULT_LAMBDA > 0.
    """

    def test_class_default_lambda_positive(self):
        assert ShrinkSigmaLedoitWolfTask.DEFAULT_LAMBDA > 0.0
        # Industry-standard range from Ledoit-Wolf 2004
        assert 0.10 <= ShrinkSigmaLedoitWolfTask.DEFAULT_LAMBDA <= 0.30

    def test_eigen_floor_positive(self):
        """Eigenvalue floor is the §5.13.12 invariant. Must be > 0."""
        assert ShrinkSigmaLedoitWolfTask.EIGEN_FLOOR > 0.0
        assert ShrinkSigmaLedoitWolfTask.EIGEN_FLOOR <= 1e-6

    def test_runs_without_explicit_lambda_in_config(self):
        """When user config omits qp_ledoit_wolf_lambda, default ≥ 0.1 fires."""
        from types import SimpleNamespace
        S = np.array([[0.04, 0.02], [0.02, 0.09]])
        ctx = SimpleNamespace(
            _qp_Sigma_full=S.copy(),
            # Note: no qp_ledoit_wolf_lambda key → default applies
            config={"rotation": {"joint_actions": {}}},
        )
        ShrinkSigmaLedoitWolfTask().run(ctx)
        # Should have changed Σ (default λ=0.2 blend, not no-op).
        assert not np.allclose(ctx._qp_Sigma_full, S)

    def test_strategy_config_json_uses_positive_lambda(self):
        """Production strategy_config.json must opt into shrinkage."""
        cfg_path = (REPO_ROOT / "backtesting" / "renquant_104"
                    / "strategy_config.json")
        cfg = json.loads(cfg_path.read_text())
        joint = cfg.get("rotation", {}).get("joint_actions", {})
        lam = joint.get("qp_ledoit_wolf_lambda")
        # Either explicitly set to positive, or absent (Task default fires).
        if lam is not None:
            assert float(lam) > 0.0, \
                "Production qp_ledoit_wolf_lambda must be > 0 (LW2004)"

    def test_side_configs_inherit_default(self):
        """Side configs (golden / alpha158_fund_paper) — same invariant.

        Per §5.13.13, side configs have aliased artifact paths but the
        portfolio-construction defaults should match production behaviour
        (or be absent → Task default).
        """
        for name in ("strategy_config.golden.json",
                      "strategy_config.alpha158_fund_paper.json"):
            cfg_path = (REPO_ROOT / "backtesting" / "renquant_104" / name)
            if not cfg_path.exists():
                continue
            cfg = json.loads(cfg_path.read_text())
            joint = cfg.get("rotation", {}).get("joint_actions", {})
            lam = joint.get("qp_ledoit_wolf_lambda")
            if lam is not None:
                assert float(lam) > 0.0, f"{name} has λ ≤ 0"
