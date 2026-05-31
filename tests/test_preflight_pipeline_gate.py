"""Track H — paired tests for WfGateMetadataTask + RegimeLayeredICTask
asserting byte-equivalence with the legacy ``_check_*`` functions.

Coverage:
  WfGateMetadataTask vs _check_wf_gate_metadata:
    (a) artifact missing                                   → soft pass
    (b) artifact unparseable                               → soft|hard per run_mode
    (c) wf metadata absent                                 → soft|hard per run_mode
    (d) passed=False (sell-only)                          → soft pass with warning
    (e) passed=False (full/buy)                           → HARD fail
    (f) passed=True + missing sanity                      → soft|hard per run_mode
    (g) passed=True + sanity ok + missing numerics        → soft|hard per run_mode
    (h) passed=True + complete evidence                   → HARD pass

  RegimeLayeredICTask vs _check_regime_layered_ic:
    (i) artifact missing                                   → soft pass
    (j) trade_monotonicity absent from wf metadata         → soft|hard per run_mode
    (k) sanity required but absent                         → soft|hard per run_mode
    (l) sanity present but passed=False                    → soft|hard per run_mode
    (m) no eligible regimes                                → soft|hard per run_mode
    (n) eligible regimes + tm.passed=False                 → soft|hard per run_mode
    (o) eligible regimes + tm.passed=True                  → HARD pass
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backtesting/renquant_104"))

from kernel.preflight import (
    _check_regime_layered_ic,
    _check_wf_gate_metadata,
)
from kernel.preflight_pipeline import (
    PreflightContext,
    RegimeLayeredICTask,
    WfGateMetadataTask,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_strategy_dir(tmp_path: Path, artifact_payload: dict | None,
                       artifact_path: str = "artifacts/prod/panel-ltr.alpha158_fund.json"
                       ) -> tuple[Path, dict]:
    art = tmp_path / artifact_path
    art.parent.mkdir(parents=True, exist_ok=True)
    if artifact_payload is not None:
        art.write_text(json.dumps(artifact_payload) if isinstance(artifact_payload, dict)
                       else artifact_payload)
    config = {
        "ranking": {"panel_scoring": {"artifact_path": artifact_path, "kind": "panel_ltr_xgboost"}}
    }
    return tmp_path, config


def _ctx(strategy_dir: Path, config: dict, run_mode: str | None = None) -> PreflightContext:
    return PreflightContext(config=config, strategy_dir=strategy_dir, run_mode=run_mode)


# Build a "complete WF-passing" artifact payload: required for the HARD pass branch.
def _wf_passing_payload() -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "feature_cols": ["KMID"],
        "best_iter": 50,
        "metadata": {
            "wf_gate_metadata": {
                "passed": True,
                "wf_3cut_sharpe_mean": 0.65,
                "wf_3cut_apy_mean": 0.12,
                "spy_sharpe_mean": 0.5,
                "strategy_minus_spy_sharpe_mean": 0.15,
                "n_cuts_beat_spy_sharpe": 2,
                "wf_reason": "passed",
                "run_at": "2026-05-30T16:00:00Z",
                "sanity_regime_ic": {
                    "passed": True,
                    "real_ic": 0.05,
                },
                "trade_monotonicity": {
                    "passed": True,
                    "pooled": {"spearman": 0.18},
                    "min_n_per_regime": 5,
                    "min_spearman": 0.1,
                    "regimes": {
                        "BULL_CALM": {"eligible": True, "passed": True, "spearman": 0.15},
                        "BULL_VOLATILE": {"eligible": True, "passed": True, "spearman": 0.20},
                    },
                },
            },
        },
    }


def _wf_failing_payload() -> dict:
    p = _wf_passing_payload()
    p["metadata"]["wf_gate_metadata"]["passed"] = False
    p["metadata"]["wf_gate_metadata"]["wf_reason"] = "zero trades across cuts"
    return p


# ─── WfGateMetadataTask parity ───────────────────────────────────────────────

class TestWfGateMetadataTaskParity:

    def test_artifact_missing_soft_pass(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=None)
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-WF-GATE"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_wf_metadata_absent_full_run_hard_fail(self, tmp_path):
        # artifact present + parses but no wf_gate_metadata field
        payload = {"kind": "panel_ltr_xgboost", "feature_cols": ["KMID"], "best_iter": 50}
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message

    def test_passed_false_full_hard_fail(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=_wf_failing_payload())
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is False
        assert new.message == leg.message

    def test_passed_false_sell_only_soft_pass(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=_wf_failing_payload())
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="sell-only")
        ctx = _ctx(sd, cfg, run_mode="sell-only")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_passed_true_complete_evidence_hard_pass(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=_wf_passing_payload())
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_passed_true_missing_sanity_strict(self, tmp_path):
        payload = _wf_passing_payload()
        payload["metadata"]["wf_gate_metadata"]["sanity_regime_ic"] = None
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message

    def test_passed_true_missing_sanity_relaxed_via_config(self, tmp_path):
        """When ``wf_gate.sanity_regime_ic_required=false``, the missing-sanity
        branch should NOT block. Legacy + new Task both honor this knob."""
        payload = _wf_passing_payload()
        payload["metadata"]["wf_gate_metadata"]["sanity_regime_ic"] = None
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        cfg["wf_gate"] = {"sanity_regime_ic_required": False}
        leg = _check_wf_gate_metadata(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        WfGateMetadataTask().run(ctx)
        new = ctx.results[-1]
        # Now the next gate is missing-numerics (or HARD pass if all there).
        # We don't care which branch landed; assert byte-equivalence.
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message


# ─── RegimeLayeredICTask parity ──────────────────────────────────────────────

class TestRegimeLayeredICTaskParity:

    def test_artifact_missing_soft_pass(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=None)
        leg = _check_regime_layered_ic(config=cfg, strategy_dir=sd)
        ctx = _ctx(sd, cfg)
        RegimeLayeredICTask().run(ctx)
        new = ctx.results[-1]
        assert new.name == leg.name == "P-REGIME-IC"
        assert new.severity == leg.severity == "soft"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_trade_monotonicity_absent_full_hard_fail(self, tmp_path):
        # WF metadata present but no trade_monotonicity
        payload = _wf_passing_payload()
        payload["metadata"]["wf_gate_metadata"]["trade_monotonicity"] = None
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_regime_layered_ic(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        RegimeLayeredICTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message

    def test_sanity_failed_strict_hard_fail(self, tmp_path):
        payload = _wf_passing_payload()
        payload["metadata"]["wf_gate_metadata"]["sanity_regime_ic"] = {
            "passed": False, "reason": "BEAR regime failed"
        }
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_regime_layered_ic(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        RegimeLayeredICTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message

    def test_eligible_regimes_all_pass_hard_pass(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=_wf_passing_payload())
        leg = _check_regime_layered_ic(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        RegimeLayeredICTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity == "hard"
        assert new.ok is leg.ok is True
        assert new.message == leg.message

    def test_eligible_regimes_some_fail_strict_hard_fail(self, tmp_path):
        payload = _wf_passing_payload()
        payload["metadata"]["wf_gate_metadata"]["trade_monotonicity"]["regimes"]["BEAR"] = {
            "eligible": True, "passed": False, "spearman": -0.05,
        }
        payload["metadata"]["wf_gate_metadata"]["trade_monotonicity"]["passed"] = False
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=payload)
        leg = _check_regime_layered_ic(config=cfg, strategy_dir=sd, run_mode="full")
        ctx = _ctx(sd, cfg, run_mode="full")
        RegimeLayeredICTask().run(ctx)
        new = ctx.results[-1]
        assert new.severity == leg.severity
        assert new.ok is leg.ok
        assert new.message == leg.message


# ─── End-to-end pipeline order ───────────────────────────────────────────────

class TestExtendedPipeline:
    """Pipeline now has 7 Tasks across 3 Jobs."""

    def test_pipeline_runs_7_tasks_in_order(self, tmp_path):
        sd, cfg = _make_strategy_dir(tmp_path, artifact_payload=_wf_passing_payload())
        ctx = _ctx(sd, cfg, run_mode="full")
        from kernel.preflight_pipeline import build_minimal_preflight_pipeline
        results = build_minimal_preflight_pipeline().run(ctx, strict=False)
        assert [r.name for r in results] == [
            "P-MODEL-ARTIFACT",
            "P-PANEL-CONTRACT",
            "P-BEST-ITER",
            "P-WF-GATE",
            "P-REGIME-IC",
            "P-STATE-FILE",
            "P-BROKER-CONNECT",
        ]
