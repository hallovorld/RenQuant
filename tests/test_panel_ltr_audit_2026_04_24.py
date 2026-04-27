"""Panel-LTR deep-audit regression tests (2026-04-24).

User mandate: "panel ltr 是我的心腹大患". Walked the panel-LTR pipeline
end-to-end on three surfaces (notebook / LEAN / live). Found 37 issues;
this file pins the fixes shipped in this commit so they don't regress.

See `doc/panel_ltr_audit_2026-04-24.md` for the full register.
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))


# ── P-21: ApplyScoresTask returns None on missing prereqs ──────────────────

class TestApplyScoresChainContinues:
    """Pre-fix: `return False` short-circuited the rest of PanelScoringJob,
    leaving Kelly target / calibration stale. Now returns None so downstream
    tasks (Veto / NGBoost / GlobalCal / Kelly) run with their own None guards."""

    def test_returns_none_when_matrix_empty(self):
        from kernel.panel_pipeline.job_panel_scoring import ApplyScoresTask
        ctx = SimpleNamespace(_panel_scorer=None, _panel_matrix=None,
                               candidates=[], holdings={})
        out = ApplyScoresTask().run(ctx)
        assert out is None

    def test_source_uses_return_none(self):
        """Sentinel: ApplyScoresTask's empty-matrix branch must use
        `return None` (continue chain), not `return False` (short-circuit)."""
        src = (_STRATEGY_DIR / "kernel" / "panel_pipeline"
               / "job_panel_scoring.py").read_text()
        idx = src.find("class ApplyScoresTask")
        end = src.find("class ", idx + 5)
        body = src[idx:end]
        early_branch = body[:body.find("scores: pd.Series")]
        # Strip comments + docstrings before scanning for active code patterns.
        active = "\n".join(
            line for line in early_branch.splitlines()
            if line.strip() and not line.strip().startswith("#")
            and not line.strip().startswith('"""')
            and not line.strip().startswith("'''")
        )
        # Drop any line that's inside a docstring (between triple quotes).
        # A simple heuristic: docstring spans are wrapped in `"""`-only lines.
        in_doc = False
        kept: list[str] = []
        for line in active.splitlines():
            if line.strip().startswith('"""') and line.strip() != '"""':
                continue   # one-line docstring
            if '"""' in line and not in_doc:
                in_doc = True
                continue
            elif '"""' in line and in_doc:
                in_doc = False
                continue
            if in_doc:
                continue
            kept.append(line)
        active_code = "\n".join(kept)
        assert "return False" not in active_code, \
            "ApplyScoresTask still short-circuits via return False on empty matrix"
        assert "return None" in active_code, \
            "ApplyScoresTask's empty-matrix branch must continue chain (return None)"


# ── P-22: NaN panel_score is vetoed (NaN < float = False bug) ──────────────

class TestVetoNaNPanelScore:
    def test_nan_panel_score_is_dropped(self):
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        from kernel.selection import CandidateResult
        ctx = SimpleNamespace(
            config={"ranking": {"panel_scoring": {"buy_floor": 0.5}}},
            candidates=[
                CandidateResult(ticker="A", raw_score=0, rank_score=0.1,
                                rs_score=0, detail="",
                                panel_score=float("nan")),
                CandidateResult(ticker="B", raw_score=0, rank_score=0.2,
                                rs_score=0, detail="",
                                panel_score=0.7),
            ],
            counters={},
        )
        VetoWeakBuysTask().run(ctx)
        # Pre-fix: NaN < 0.5 is False so A survived. Now A is dropped.
        tickers = {c.ticker for c in ctx.candidates}
        assert "A" not in tickers
        assert "B" in tickers
        assert ctx.counters["panel_vetoed"] == 1

    def test_none_panel_score_is_kept(self):
        """Distinct from NaN: ps=None means the matrix didn't include
        this ticker (e.g., no factor frame). RS still ranks it."""
        from kernel.panel_pipeline.job_panel_scoring import VetoWeakBuysTask
        from kernel.selection import CandidateResult
        ctx = SimpleNamespace(
            config={"ranking": {"panel_scoring": {"buy_floor": 0.5}}},
            candidates=[
                CandidateResult(ticker="A", raw_score=0, rank_score=0.1,
                                rs_score=0, detail="", panel_score=None),
            ],
            counters={},
        )
        VetoWeakBuysTask().run(ctx)
        assert {c.ticker for c in ctx.candidates} == {"A"}


# ── P-16: FactorZScoreTask doesn't early-return on partial dict ────────────

class TestFactorZScoreCompletenessGate:
    def test_source_compares_against_raw_factor_count(self):
        """Pre-fix: `if ctx.factor_frames: return` short-circuited on any
        non-empty dict, allowing stale partial state to skip the recompute.
        Now: only skip when factor_frames covers all raw_factor_frames."""
        src = (_STRATEGY_DIR / "training_panel" / "pp_panel_training.py").read_text()
        idx = src.find("class FactorZScoreTask")
        body = src[idx:idx + 1500]
        assert "len(ctx.factor_frames)" in body and "len(ctx.raw_factor_frames" in body


# ── P-9: prepare_inference_panel_frames isolates per-ticker errors ────────

class TestPanelFrameInferenceErrorIsolation:
    def test_one_ticker_failure_does_not_kill_others(self, tmp_path):
        """Pre-fix: any single ticker raising in `_chain` propagated through
        `f.result()` and killed the whole bar's panel inference. Now per-ticker
        try/except logs + skips, mirroring training-side run_panel_ticker_parallel."""
        src = (_STRATEGY_DIR / "training_panel" / "pipeline.py").read_text()
        # Sentinel: post-fix wraps f.result() in try/except
        idx = src.find("with ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix=\"panel-inf\"")
        body = src[idx:idx + 1500]
        assert "try:" in body
        assert "f.result()" in body
        assert "except Exception" in body


# ── P-37: cache_dir resolves via _strategy_dir, not cwd ────────────────────
#
# Audit fix #142 (2026-04-26 round-7): the original P-37 tests assumed
# the resolver returns strategy_dir.parent.parent / p UNCONDITIONALLY,
# and a bare relative path on missing _strategy_dir. Both assumptions
# conflict with the CACHE-DIR-SNAPSHOT fix (2026-04-26 earlier) which:
#   - falls back to cwd when the strategy_dir-derived path doesn't
#     exist (snapshot mode safety — sims wrap strategy_dir in tmpdir)
#   - returns absolute cwd-resolved path on missing _strategy_dir so
#     callers can stat() / read_text() without prepending cwd themselves
#
# Tests rewritten to match the CURRENT (snapshot-aware) contract.

class TestCacheDirResolution:
    def test_resolve_cache_dir_with_strategy_dir(self, tmp_path):
        """When the strategy_dir-derived cache exists, use it.

        Pre-CACHE-DIR-SNAPSHOT: returned strategy_dir.parent.parent / cfg
        unconditionally. Now: prefers derived if it exists; falls back
        to cwd otherwise.
        """
        from training_panel.pp_panel_training import _resolve_cache_dir
        # Build a real strategy_dir layout in tmp so derived path exists.
        repo_root = tmp_path
        (repo_root / "backtesting" / "renquant_104").mkdir(parents=True)
        (repo_root / "data" / "fundamentals").mkdir(parents=True)
        ctx_config = {"_strategy_dir": str(repo_root / "backtesting" / "renquant_104")}
        result = _resolve_cache_dir("data/fundamentals", ctx_config)
        assert result == repo_root / "data" / "fundamentals"

    def test_resolve_cache_dir_snapshot_falls_back_to_cwd(self, tmp_path):
        """Snapshot edge case (CACHE-DIR-SNAPSHOT): when strategy_dir
        is in a tmpdir without the data/ hierarchy, the resolver falls
        back to cwd so the real cache is found.

        This is the load-bearing test for the snapshot-fix —
        previously sim A/B silently saw 0 fundamentals.
        """
        from training_panel.pp_panel_training import _resolve_cache_dir
        from pathlib import Path
        snap_dir = tmp_path / "snapshot" / "renquant_104"
        snap_dir.mkdir(parents=True)
        ctx_config = {"_strategy_dir": str(snap_dir)}
        # data/ohlcv exists in repo cwd; the snapshot tmpdir doesn't
        # have a data/ subdir at all.
        result = _resolve_cache_dir("data/ohlcv", ctx_config)
        cwd_candidate = Path.cwd() / "data" / "ohlcv"
        if cwd_candidate.exists():
            assert result == cwd_candidate, (
                "snapshot fallback should return cwd-resolved cache, "
                "not the missing strategy_dir-derived path"
            )
        else:
            # No cache to fall back to — resolver returns derived
            # (caller's not-found logic handles it).
            assert "snapshot" in str(result)

    def test_resolve_cache_dir_absolute_passthrough(self):
        from training_panel.pp_panel_training import _resolve_cache_dir
        ctx_config = {"_strategy_dir": "/anywhere"}
        result = _resolve_cache_dir("/abs/cache", ctx_config)
        assert str(result) == "/abs/cache"

    def test_resolve_cache_dir_no_strategy_dir_fallback(self):
        """Without _strategy_dir, resolver returns cwd-resolved
        ABSOLUTE path so callers can stat() against it.

        Pre-CACHE-DIR-SNAPSHOT: returned bare relative string.
        Now: returns Path.cwd() / cfg with a warning. Live runner
        always sets _strategy_dir; this branch only fires in
        ad-hoc test/debug contexts.
        """
        from training_panel.pp_panel_training import _resolve_cache_dir
        from pathlib import Path
        result = _resolve_cache_dir("data/fundamentals", {})
        assert result == Path.cwd() / "data" / "fundamentals"

    def test_all_load_tasks_use_resolver(self):
        """Sentinel: every Load*Task with a cache_dir must call _resolve_cache_dir."""
        src = (_STRATEGY_DIR / "training_panel" / "pp_panel_training.py").read_text()
        # The 5 task classes that load from disk caches all need the helper
        for marker in ("FundamentalsStore(data_dir=cache_dir)",
                       "EarningsSurpriseStore(data_dir=cache_dir)",
                       "InsiderTradesStore(data_dir=cache_dir)",
                       "HourlyBarStore(data_dir=cache_dir)",
                       "MinuteBarStore(data_dir=cache_dir)"):
            idx = src.find(marker)
            assert idx > 0, f"missing {marker}"
            # Find the preceding cache_dir = ... line
            preceding = src[max(0, idx - 200):idx]
            assert "_resolve_cache_dir(" in preceding, \
                f"task using {marker} doesn't resolve via _resolve_cache_dir"


# ── Notebook cell 15 routes through prepare_inference_panel_frames ─────────

class TestNotebookCellUsesPrepareFunction:
    def test_cell_15_calls_prepare_inference_panel_frames(self):
        """Pre-fix: cell 15 manually chained 4 of 11 needed tasks. Now
        delegates to prepare_inference_panel_frames (same function LEAN
        and live runner use). Verifies the notebook cell source."""
        import json as _json
        nb_path = _STRATEGY_DIR / "renquant_104.ipynb"
        nb = _json.loads(nb_path.read_text())
        cell15 = nb["cells"][15]
        if cell15.get("cell_type") != "code":
            pytest.skip("cell 15 is not a code cell")
        src = "".join(cell15["source"])
        assert "prepare_inference_panel_frames" in src, \
            "Cell 15 must call prepare_inference_panel_frames (audit P-Notebook-1)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
