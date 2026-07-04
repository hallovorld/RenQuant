"""Tournament acceptance gate — campaign A2 (F-17, RQ PR #444 / orch PR #297).

The Sunday weekly tournament (`weekly_tournament_retrain.sh` →
`train_104.py --skip-panel --force`) wrote per-ticker buy-admission models
straight to production `models/<TICKER>/` with acceptance auto-disabled.
This suite pins the fix:

1. PROTECTION CONTRACT (P0, operator-pinned): a HEALTHY candidate produces
   byte-identical `models/<TICKER>/` contents whether the gate is enabled
   (staging-then-swap path) or disabled (the pre-fix direct-write path).
2. Degenerate candidates (constant predictions / stale-regressed data /
   metric collapse) are REJECTED: the incumbent files stay byte- and
   mtime-untouched, and the rejection is counted with a per-ticker reason.
3. Partial-batch isolation: one bad ticker never blocks the other tickers.
4. Fail-closed: a crash inside the gate or the staged write rejects that
   ticker instead of shipping an unverified model.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STRATEGY_DIR = REPO_ROOT / "backtesting" / "renquant_104"
if str(STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(STRATEGY_DIR))

from kernel import tournament_acceptance as ta  # noqa: E402
from kernel.pipeline import pp_training  # noqa: E402
from kernel.pipeline.pp_training import (  # noqa: E402
    TickerTrainingContext,
    TrainingContext,
    _run_ticker_chain,
)

# All dates are derived RELATIVE to the wall clock so the suite stays
# deterministic no matter when it runs (the gate's staleness leg compares
# against today).
_TODAY = pd.Timestamp.today().normalize()


def _days_ago(n: int) -> str:
    return str((_TODAY - pd.Timedelta(days=n)).date())


# ── Fixtures ──────────────────────────────────────────────────────────────────

class FakeModel:
    """Deterministic stand-in that mimics the RF/QL/Manual save() contract:
    it writes a weights artifact plus a fresh policy-metadata.json whose
    `artifacts` values embed the ABSOLUTE write path (the byte-invariance
    hazard the staging swap must rewrite)."""

    def __init__(self, payload: str = "weights-v1"):
        self.payload = payload

    def save(self, directory: Path, model_name: str) -> dict:
        directory = Path(directory)
        artifact_path = directory / f"{model_name}-fake-weights.json"
        artifact_path.write_text(json.dumps({"weights": self.payload}, indent=2))
        metadata = {
            "model_name": model_name,
            "policy_type": "fake",
            "artifacts": {"weights": str(artifact_path)},
        }
        meta_path = directory / f"{model_name}-policy-metadata.json"
        meta_path.write_text(json.dumps(metadata, indent=2))
        return metadata


def _feature_frame(rows: int = 40, days_ago: int = 7) -> pd.DataFrame:
    idx = pd.bdate_range(end=_TODAY - pd.Timedelta(days=days_ago), periods=rows)
    rng = np.random.RandomState(7)
    return pd.DataFrame({"feat": rng.randn(rows), "label": rng.randn(rows) * 0.01},
                        index=idx)


def _healthy_result(ticker: str = "AAA", sharpe: float = 1.25) -> dict:
    scores = pd.Series(np.linspace(-0.4, 0.6, 30))
    return {
        "sharpe": sharpe,
        "selection_metric": "sharpe",
        "selection_score": sharpe,
        "best_approach": "FakeApproach",   # retrain_live_models skips unknown
        "model": FakeModel(),
        "oos_signals": None,
        "oos_raw_scores": scores,
        "score_calibration": None,
        "train_rows": 200,
        "oos_rows": 40,
        "passes_floor": True,
    }


def _config(gate_enabled: bool) -> dict:
    return {
        "watchlist": ["AAA"],
        "acceptance": {"tournament": {"enabled": gate_enabled}},
        "model_params": {
            "feature_columns": ["feat"],
            "lookahead": 5,
            "threshold": 0.03,
            "bags": 5,
            "leaf_size": 5,
            "buy_threshold": 0.5,
            "sell_threshold": -0.5,
        },
        "_strategy_name": "renquant_104",
    }


def _tc(strategy_dir: Path, ticker: str = "AAA", *, gate_enabled: bool,
        result: dict | None = None,
        feature_frame: pd.DataFrame | None = None) -> TickerTrainingContext:
    tc = TickerTrainingContext(
        ticker=ticker,
        ohlcv={},
        config=_config(gate_enabled),
        strategy_dir=strategy_dir,
    )
    tc.feature_frame = feature_frame if feature_frame is not None else _feature_frame()
    tc.result = result if result is not None else _healthy_result(ticker)
    return tc


class _NoopJob:
    """Stands in for TickerFeatureJob / TickerTournamentJob so the REAL
    export / calibration / gate / staging code runs against pre-seeded
    deterministic contexts."""

    def run(self, tc):  # noqa: D102
        return None


@pytest.fixture()
def noop_feature_tournament(monkeypatch):
    monkeypatch.setattr(pp_training, "TickerFeatureJob", _NoopJob)
    monkeypatch.setattr(pp_training, "TickerTournamentJob", _NoopJob)


def _snapshot_dir(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*")) if p.is_file()
    }


def _write_incumbent(strategy_dir: Path, ticker: str, *, sharpe: float = 1.0,
                     live_train_end: str | None = None) -> dict[str, tuple[bytes, int]]:
    """Create a plausible incumbent bundle; return {name: (bytes, mtime_ns)}."""
    if live_train_end is None:
        live_train_end = _days_ago(14)
    sym_dir = strategy_dir / "models" / ticker
    sym_dir.mkdir(parents=True, exist_ok=True)
    (sym_dir / f"{ticker}-fake-weights.json").write_text(
        json.dumps({"weights": "incumbent"}, indent=2))
    meta = {
        "model_name": ticker,
        "policy_type": "fake",
        "artifacts": {"weights": str(sym_dir / f"{ticker}-fake-weights.json")},
        "trained_date": _days_ago(7),
        "best_approach": "FakeApproach",
        "sharpe": sharpe,
        "live_train_end": live_train_end,
    }
    (sym_dir / f"{ticker}-policy-metadata.json").write_text(json.dumps(meta, indent=2))
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sym_dir.iterdir() if p.is_file()
    }


def _assert_incumbent_untouched(strategy_dir: Path, ticker: str,
                                before: dict[str, tuple[bytes, int]]) -> None:
    sym_dir = strategy_dir / "models" / ticker
    now = {p.name: (p.read_bytes(), p.stat().st_mtime_ns)
           for p in sym_dir.iterdir() if p.is_file()}
    assert now == before, "incumbent model files must be byte- and mtime-untouched"


# ── 1. Protection contract: healthy path A/B byte-identical ─────────────────

class TestHealthyPathInvariance:
    def test_ab_byte_identical_no_incumbent(self, tmp_path, noop_feature_tournament):
        """Gate-enabled (staging-then-swap) output == gate-disabled (pre-fix
        direct write) output, byte for byte, modulo the tmp-root prefix."""
        outputs: dict[str, dict[str, bytes]] = {}
        roots: dict[str, Path] = {}
        for label, enabled in (("legacy", False), ("gated", True)):
            strategy_dir = tmp_path / label
            (strategy_dir / "models").mkdir(parents=True)
            tc = _tc(strategy_dir, gate_enabled=enabled)
            _run_ticker_chain(tc)
            assert tc.exported is True
            assert tc.rejected is False
            # staging must be fully cleaned up
            staging = strategy_dir / "models" / ".staging"
            assert not staging.exists() or not any(staging.iterdir())
            outputs[label] = _snapshot_dir(strategy_dir / "models")
            roots[label] = strategy_dir

        assert set(outputs["legacy"]) == set(outputs["gated"])
        for rel in outputs["legacy"]:
            legacy = outputs["legacy"][rel].replace(
                str(roots["legacy"]).encode(), b"<ROOT>")
            gated = outputs["gated"][rel].replace(
                str(roots["gated"]).encode(), b"<ROOT>")
            assert legacy == gated, f"byte mismatch in {rel}"

    def test_ab_byte_identical_with_incumbent(self, tmp_path, noop_feature_tournament):
        """Same A/B with an incumbent present — the healthy candidate must
        replace it identically on both paths."""
        outputs = {}
        roots = {}
        for label, enabled in (("legacy", False), ("gated", True)):
            strategy_dir = tmp_path / label
            _write_incumbent(strategy_dir, "AAA", sharpe=0.9)
            tc = _tc(strategy_dir, gate_enabled=enabled)
            _run_ticker_chain(tc)
            assert tc.exported is True
            assert tc.rejected is False
            outputs[label] = _snapshot_dir(strategy_dir / "models")
            roots[label] = strategy_dir

        assert set(outputs["legacy"]) == set(outputs["gated"])
        for rel in outputs["legacy"]:
            legacy = outputs["legacy"][rel].replace(
                str(roots["legacy"]).encode(), b"<ROOT>")
            gated = outputs["gated"][rel].replace(
                str(roots["gated"]).encode(), b"<ROOT>")
            assert legacy == gated, f"byte mismatch in {rel}"

    def test_promoted_metadata_points_at_live_dir_not_staging(
            self, tmp_path, noop_feature_tournament):
        strategy_dir = tmp_path / "s"
        (strategy_dir / "models").mkdir(parents=True)
        tc = _tc(strategy_dir, gate_enabled=True)
        _run_ticker_chain(tc)
        meta = json.loads(
            (strategy_dir / "models" / "AAA" / "AAA-policy-metadata.json").read_text())
        weights_path = meta["artifacts"]["weights"]
        assert ".staging" not in weights_path
        assert weights_path == str(
            strategy_dir / "models" / "AAA" / "AAA-fake-weights.json")
        assert Path(weights_path).exists()

    def test_ab_byte_identical_real_manual_model(self, tmp_path,
                                                 noop_feature_tournament):
        """Same A/B through the REAL ManualModel save contract — its
        metadata embeds str(directory / rules-file), the absolute-path
        byte-invariance hazard, and retrain_live_models() rebuilds and
        re-saves it on the 4y window (the in-place double-write the staging
        swap must reproduce exactly)."""
        from training.models import create_model

        outputs = {}
        roots = {}
        for label, enabled in (("legacy", False), ("gated", True)):
            strategy_dir = tmp_path / label
            (strategy_dir / "models").mkdir(parents=True)
            result = _healthy_result()
            result["best_approach"] = "Manual"
            result["model"] = create_model(
                "manual", buy_threshold=2, sell_threshold=-2)
            tc = _tc(strategy_dir, gate_enabled=enabled, result=result)
            _run_ticker_chain(tc)
            assert tc.exported is True
            assert tc.rejected is False
            outputs[label] = _snapshot_dir(strategy_dir / "models")
            roots[label] = strategy_dir

        assert set(outputs["legacy"]) == set(outputs["gated"])
        assert any("manual-rules" in rel for rel in outputs["legacy"])
        for rel in outputs["legacy"]:
            legacy = outputs["legacy"][rel].replace(
                str(roots["legacy"]).encode(), b"<ROOT>")
            gated = outputs["gated"][rel].replace(
                str(roots["gated"]).encode(), b"<ROOT>")
            assert legacy == gated, f"byte mismatch in {rel}"
        # the retrain leg must actually have run (live_train_* fields present)
        meta = json.loads(
            (roots["gated"] / "models" / "AAA"
             / "AAA-policy-metadata.json").read_text())
        assert meta["best_approach"] == "Manual"
        assert "live_train_end" in meta
        assert meta["artifacts"]["rules"] == str(
            roots["gated"] / "models" / "AAA" / "AAA-manual-rules.json")

    def test_honest_degradation_still_ships(self, tmp_path, noop_feature_tournament):
        """A worse-but-not-collapsed fresh Sharpe is admission information —
        it must still be written (LoadUniverseJob floors act at load time)."""
        strategy_dir = tmp_path / "s"
        _write_incumbent(strategy_dir, "AAA", sharpe=2.5)
        tc = _tc(strategy_dir, gate_enabled=True,
                 result=_healthy_result(sharpe=0.3))
        _run_ticker_chain(tc)
        assert tc.rejected is False
        assert tc.exported is True
        meta = json.loads(
            (strategy_dir / "models" / "AAA" / "AAA-policy-metadata.json").read_text())
        assert meta["sharpe"] == 0.3


# ── 2. Degenerate candidates are rejected, incumbent kept ───────────────────

class TestDegenerateRejection:
    def test_constant_scores_rejected(self, tmp_path, noop_feature_tournament):
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA")
        bad = _healthy_result()
        bad["oos_raw_scores"] = pd.Series(np.full(30, 0.123))
        tc = _tc(strategy_dir, gate_enabled=True, result=bad)
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert tc.exported is False
        assert "T3_nondegenerate" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)

    def test_all_nan_scores_rejected(self, tmp_path, noop_feature_tournament):
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA")
        bad = _healthy_result()
        bad["oos_raw_scores"] = pd.Series(np.full(30, np.nan))
        tc = _tc(strategy_dir, gate_enabled=True, result=bad)
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert "T3_nondegenerate" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)

    def test_regressed_data_cutoff_rejected(self, tmp_path, noop_feature_tournament):
        """Candidate trained on data ending BEFORE the incumbent's
        live_train_end (stale/regressed feed) must be rejected — while still
        inside the absolute staleness budget, so the regression leg is what
        fires."""
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA",
                                  live_train_end=_days_ago(8))
        tc = _tc(strategy_dir, gate_enabled=True,
                 feature_frame=_feature_frame(days_ago=23))
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert "T4_data_cutoff" in (tc.reject_reason or "")
        assert "REGRESSES" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)

    def test_metric_collapse_rejected(self, tmp_path, noop_feature_tournament):
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA", sharpe=2.5)
        tc = _tc(strategy_dir, gate_enabled=True,
                 result=_healthy_result(sharpe=-3.0))
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert "T5_metric_collapse" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)

    def test_rejection_verdict_archived(self, tmp_path, noop_feature_tournament):
        strategy_dir = tmp_path / "s"
        _write_incumbent(strategy_dir, "AAA")
        bad = _healthy_result()
        bad["oos_raw_scores"] = pd.Series(np.full(30, 0.5))
        tc = _tc(strategy_dir, gate_enabled=True, result=bad)
        _run_ticker_chain(tc)
        log_dir = strategy_dir / "artifacts" / "_tournament_acceptance_log"
        archived = list(log_dir.glob("*_REJECTED_AAA.txt"))
        assert len(archived) == 1
        assert "T3_nondegenerate" in archived[0].read_text()

    def test_no_partial_writes_on_rejection(self, tmp_path, noop_feature_tournament):
        """A rejected NEW ticker (no incumbent) must leave models/ empty —
        never a partially-created dir."""
        strategy_dir = tmp_path / "s"
        (strategy_dir / "models").mkdir(parents=True)
        bad = _healthy_result()
        bad["oos_raw_scores"] = pd.Series(np.full(30, 0.5))
        tc = _tc(strategy_dir, gate_enabled=True, result=bad)
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert not (strategy_dir / "models" / "AAA").exists()
        staging = strategy_dir / "models" / ".staging"
        assert not staging.exists() or not any(staging.iterdir())


# ── 3. Fail-closed on machinery crashes ──────────────────────────────────────

class TestFailClosed:
    def test_gate_crash_rejects_instead_of_shipping(self, tmp_path,
                                                    noop_feature_tournament,
                                                    monkeypatch):
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA")

        def _boom(*a, **k):
            raise RuntimeError("gate exploded")

        monkeypatch.setattr(ta, "evaluate_tournament_candidate", _boom)
        tc = _tc(strategy_dir, gate_enabled=True)
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert tc.exported is False
        assert "gate exploded" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)

    def test_staged_write_crash_keeps_incumbent(self, tmp_path,
                                                noop_feature_tournament,
                                                monkeypatch):
        """A crash during promote leaves production byte-untouched and the
        staging dir cleaned up."""
        strategy_dir = tmp_path / "s"
        before = _write_incumbent(strategy_dir, "AAA")

        def _boom(*a, **k):
            raise OSError("disk went away")

        monkeypatch.setattr(ta, "promote_staged_ticker_dir", _boom)
        tc = _tc(strategy_dir, gate_enabled=True)
        _run_ticker_chain(tc)
        assert tc.rejected is True
        assert tc.exported is False
        assert "disk went away" in (tc.reject_reason or "")
        _assert_incumbent_untouched(strategy_dir, "AAA", before)
        staging = strategy_dir / "models" / ".staging"
        assert not staging.exists() or not any(staging.iterdir())


# ── 4. Partial-batch isolation ────────────────────────────────────────────────

class TestPartialBatchIsolation:
    def test_one_bad_ticker_does_not_block_the_rest(self, tmp_path, monkeypatch):
        strategy_dir = tmp_path / "s"
        (strategy_dir / "models").mkdir(parents=True)
        incumbent_before = _write_incumbent(strategy_dir, "BAD")

        frames = {t: _feature_frame() for t in ("GOOD1", "BAD", "GOOD2")}
        results = {
            "GOOD1": _healthy_result("GOOD1"),
            "BAD": _healthy_result("BAD"),
            "GOOD2": _healthy_result("GOOD2"),
        }
        results["BAD"]["oos_raw_scores"] = pd.Series(np.full(30, 0.9))  # constant

        class _SeedFeature:
            def run(self, tc):
                tc.feature_frame = frames[tc.ticker]

        class _SeedTournament:
            def run(self, tc):
                tc.result = results[tc.ticker]

        monkeypatch.setattr(pp_training, "TickerFeatureJob", _SeedFeature)
        monkeypatch.setattr(pp_training, "TickerTournamentJob", _SeedTournament)

        config = _config(gate_enabled=True)
        config["watchlist"] = ["GOOD1", "BAD", "GOOD2"]
        config["_strategy_dir"] = str(strategy_dir)
        ctx = TrainingContext(config=config)
        pp_training.FeatureJob().run(ctx)

        assert sorted(ctx.exported) == ["GOOD1", "GOOD2"]
        assert set(ctx.rejected) == {"BAD"}
        assert "T3_nondegenerate" in ctx.rejected["BAD"]
        for good in ("GOOD1", "GOOD2"):
            assert (strategy_dir / "models" / good
                    / f"{good}-policy-metadata.json").exists()
        _assert_incumbent_untouched(strategy_dir, "BAD", incumbent_before)


# ── 5. Gate unit semantics ────────────────────────────────────────────────────

class TestGateSemantics:
    CFG = dict(ta._DEFAULTS)

    def test_config_default_enabled(self):
        assert ta.tournament_acceptance_config({}) is not None
        assert ta.tournament_acceptance_config({"acceptance": {}}) is not None

    def test_config_follows_acceptance_master_switch(self):
        assert ta.tournament_acceptance_config(
            {"acceptance": {"enabled": False}}) is None
        # explicit tournament.enabled wins over the master switch
        assert ta.tournament_acceptance_config(
            {"acceptance": {"enabled": False,
                            "tournament": {"enabled": True}}}) is not None

    def test_config_explicit_disable(self):
        assert ta.tournament_acceptance_config(
            {"acceptance": {"tournament": {"enabled": False}}}) is None

    def test_config_knob_override(self):
        cfg = ta.tournament_acceptance_config(
            {"acceptance": {"tournament": {"max_data_staleness_days": 90}}})
        assert cfg["max_data_staleness_days"] == 90

    def test_no_incumbent_comparison_gates_skip_pass(self):
        verdict = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(), _feature_frame(), None, self.CFG)
        assert verdict.all_hard_passed

    def test_future_data_end_rejected(self):
        verdict = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(), _feature_frame(days_ago=-30),
            None, self.CFG)
        assert not verdict.all_hard_passed
        assert any(r.name == "T4_data_cutoff" and not r.passed
                   for r in verdict.results)

    def test_stale_data_end_rejected(self):
        verdict = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(), _feature_frame(days_ago=90),
            None, self.CFG)
        assert not verdict.all_hard_passed
        assert any(r.name == "T4_data_cutoff" and not r.passed
                   for r in verdict.results)

    def test_missing_scores_fail_closed(self):
        bad = _healthy_result()
        bad["oos_raw_scores"] = None
        verdict = ta.evaluate_tournament_candidate(
            "AAA", bad, _feature_frame(), None, self.CFG)
        assert not verdict.all_hard_passed

    def test_too_few_rows_fail(self):
        bad = _healthy_result()
        bad["train_rows"] = 10
        verdict = ta.evaluate_tournament_candidate(
            "AAA", bad, _feature_frame(), None, self.CFG)
        assert any(r.name == "T2_sample_size" and not r.passed
                   for r in verdict.results)

    def test_collapse_requires_both_floor_and_drop(self):
        incumbent = {"sharpe": 2.5, "live_train_end": _days_ago(14)}
        # below floor but small drop vs incumbent (incumbent also bad) → pass
        ok = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(sharpe=-1.5),
            _feature_frame(), {"sharpe": -1.2}, self.CFG)
        assert all(r.passed for r in ok.results if r.name == "T5_metric_collapse")
        # above floor, big drop → pass (honest degradation)
        ok2 = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(sharpe=0.2), _feature_frame(),
            incumbent, self.CFG)
        assert all(r.passed for r in ok2.results if r.name == "T5_metric_collapse")
        # below floor AND big drop → reject
        bad = ta.evaluate_tournament_candidate(
            "AAA", _healthy_result(sharpe=-3.0), _feature_frame(),
            incumbent, self.CFG)
        assert any(r.name == "T5_metric_collapse" and not r.passed
                   for r in bad.results)


# ── 6. promote_staged_ticker_dir unit ─────────────────────────────────────────

class TestPromoteStagedDir:
    def test_absolute_artifact_paths_rewritten(self, tmp_path):
        staged = tmp_path / "staging" / "AAA"
        live = tmp_path / "models" / "AAA"
        staged.mkdir(parents=True)
        (staged / "AAA-fake-weights.json").write_text("{}")
        meta = {
            "artifacts": {
                "weights": str(staged / "AAA-fake-weights.json"),
                "relative": "AAA-other.json",   # basename style (XGB) untouched
            },
        }
        (staged / "AAA-policy-metadata.json").write_text(json.dumps(meta, indent=2))
        promoted = ta.promote_staged_ticker_dir(staged, live, "AAA")
        assert sorted(promoted) == ["AAA-fake-weights.json",
                                    "AAA-policy-metadata.json"]
        out = json.loads((live / "AAA-policy-metadata.json").read_text())
        assert out["artifacts"]["weights"] == str(live / "AAA-fake-weights.json")
        assert out["artifacts"]["relative"] == "AAA-other.json"
        # staged dir fully drained of promoted files
        assert not any(staged.iterdir())

    def test_untracked_live_files_left_alone(self, tmp_path):
        """Files in the live dir that the staged bundle does not contain
        (a previous winner's other-model artifacts) survive the swap — same
        as the pre-fix in-place write."""
        staged = tmp_path / "staging" / "AAA"
        live = tmp_path / "models" / "AAA"
        staged.mkdir(parents=True)
        live.mkdir(parents=True)
        (live / "AAA-qtable.json").write_text("legacy-qtable")
        (staged / "AAA-policy-metadata.json").write_text(json.dumps({"a": 1}))
        ta.promote_staged_ticker_dir(staged, live, "AAA")
        assert (live / "AAA-qtable.json").read_text() == "legacy-qtable"
        assert (live / "AAA-policy-metadata.json").exists()


# ── 7. train_104.py wiring (string contracts, same pattern as
#       test_train_104_acceptance_wiring.py) ─────────────────────────────────

TRAIN_SRC = (REPO_ROOT / "scripts" / "train_104.py").read_text()
PP_SRC = (STRATEGY_DIR / "kernel" / "pipeline" / "pp_training.py").read_text()


class TestTrain104Wiring:
    def test_skip_acceptance_disables_tournament_gate(self):
        assert '.setdefault("tournament", {})["enabled"] = False' in TRAIN_SRC

    def test_skip_panel_message_notes_tournament_gate_active(self):
        assert "remains ACTIVE" in TRAIN_SRC

    def test_rejections_logged_and_notified(self):
        assert "TOURNAMENT ACCEPTANCE REJECT" in TRAIN_SRC
        assert "_notify_tournament_rejections" in TRAIN_SRC
        assert "TOURNAMENT ACCEPTANCE WARN" in TRAIN_SRC

    def test_notify_honors_no_notify_env(self):
        idx = TRAIN_SRC.find("def _notify_tournament_rejections")
        assert idx >= 0
        block = TRAIN_SRC[idx:idx + 1200]
        assert "RENQUANT_NO_NOTIFY" in block

    def test_gate_evaluated_before_export_in_chain(self):
        """The verdict must come BEFORE any Export/Calibration write in the
        gated branch of _run_ticker_chain."""
        gated = PP_SRC.find("def _run_gated_export")
        assert gated >= 0
        block = PP_SRC[gated:]
        verdict_idx = block.find("evaluate_tournament_candidate")
        export_idx = block.find("TickerExportJob().run(tc)")
        assert 0 <= verdict_idx < export_idx

    def test_default_on_in_kernel(self):
        """kernel gate must be default-ON (fail-closed) so Sunday's
        `--skip-panel --force` run is protected without config edits."""
        ta_src = (STRATEGY_DIR / "kernel" / "tournament_acceptance.py").read_text()
        assert 'acc.get("enabled", True)' in ta_src
        assert 'tourn.get("enabled", enabled_default)' in ta_src
