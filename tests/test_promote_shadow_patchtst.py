"""Unit tests for scripts/promote_shadow_patchtst.py.

Covers the PURE fail-closed decision logic (freshness / cutoff-advance / §3.4
gate helpers / monitor tier) plus a dry-run integration of run_promote() with the
heavy load+smoke-inference gate mocked (no torch needed in CI). The atomic pin
swap is not exercised here (it needs a real .pt); the promote refuses to swap in
dry-run, which is what we assert.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_mod():
    import sys
    name = "_promote_shadow_patchtst"
    spec = importlib.util.spec_from_file_location(
        name, REPO_ROOT / "scripts" / "promote_shadow_patchtst.py",
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so @dataclass (which reads sys.modules[cls.__module__])
    # resolves the module during class creation.
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load_mod()


# --- parse_date -------------------------------------------------------------

class TestParseDate:
    def test_iso_date(self):
        assert M.parse_date("2026-02-10") == dt.date(2026, 2, 10)

    def test_iso_datetime_truncates(self):
        assert M.parse_date("2026-02-10T09:06:00Z") == dt.date(2026, 2, 10)

    def test_none_and_empty(self):
        assert M.parse_date(None) is None
        assert M.parse_date("") is None
        assert M.parse_date("not-a-date") is None


# --- dotted get/set ---------------------------------------------------------

class TestDotted:
    def test_get(self):
        d = {"ranking": {"panel_scoring": {"artifact_path": "x"}}}
        assert M.get_dotted(d, "ranking.panel_scoring.artifact_path") == "x"
        assert M.get_dotted(d, "ranking.missing.key") is None

    def test_set_creates(self):
        d = {}
        M.set_dotted(d, "ranking.panel_scoring.artifact_path", "y")
        assert d == {"ranking": {"panel_scoring": {"artifact_path": "y"}}}


# --- source SLA verdict -----------------------------------------------------

class TestSourceSLA:
    def test_on_sla(self):
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), dt.date(2026, 6, 20))
        assert v.on_sla and v.age_days == 10

    def test_off_sla_stale_panel(self):
        # The real 2026-02-10 panel vs now 2026-06-30 -> ~140d, far off 28d.
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), dt.date(2026, 2, 10))
        assert not v.on_sla and v.age_days == 140

    def test_missing_cutoff_fails_closed(self):
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), None)
        assert not v.on_sla

    def test_missing_cutoff_tolerated(self):
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), None, missing_ok=True)
        assert v.on_sla

    # --- Codex #419 review 3: a future-dated cutoff must FAIL CLOSED, never pass -
    # --- trivially because a negative age is "<= sla_days" (look-ahead leak). ----

    def test_future_dated_cutoff_fails_closed(self):
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), dt.date(2026, 7, 5))
        assert not v.on_sla
        assert M.FUTURE_DATED in v.detail

    def test_one_day_future_cutoff_fails_closed(self):
        # Just past the (zero-day) clock-skew tolerance -> must fail.
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), dt.date(2026, 7, 1))
        assert not v.on_sla and M.FUTURE_DATED in v.detail

    def test_cutoff_exactly_now_is_boundary_and_passes(self):
        # A cutoff dated exactly "now" (age == 0) is the boundary: not future.
        src = {"name": "panel", "axis": "fast", "sla_days": 28}
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), dt.date(2026, 6, 30))
        assert v.on_sla and v.age_days == 0


# --- cutoff advance ---------------------------------------------------------

class TestCutoffsAdvance:
    def test_both_advance(self):
        served = {"effective_train_cutoff_date": "2024-11-13",
                  "effective_selection_cutoff_date": "2026-02-10"}
        cand = {"effective_train_cutoff_date": "2025-12-12",
                "effective_selection_cutoff_date": "2026-05-01"}
        assert M.cutoffs_advance(served, cand).advanced

    def test_selection_no_advance_is_non_fresh(self):
        # The real gap: train advances but the panel still ends 2026-02-10, so the
        # selection cutoff does NOT advance -> not fresh (RFC §3.1).
        served = {"effective_train_cutoff_date": "2024-11-13",
                  "effective_selection_cutoff_date": "2026-02-10"}
        cand = {"effective_train_cutoff_date": "2025-12-12",
                "effective_selection_cutoff_date": "2026-02-10"}
        v = M.cutoffs_advance(served, cand)
        assert not v.advanced and "selection cutoff did not advance" in v.detail

    def test_missing_candidate_axis_fails_closed(self):
        served = {"effective_train_cutoff_date": "2024-11-13",
                  "effective_selection_cutoff_date": "2026-02-10"}
        cand = {"effective_train_cutoff_date": "2025-12-12"}
        assert not M.cutoffs_advance(served, cand).advanced


# --- freshness tier ---------------------------------------------------------

class TestFreshnessTier:
    def test_healthy(self):
        assert M.freshness_tier(10, all_sources_on_sla=True,
                                validated_advancing_promote=True) == "healthy"

    def test_not_healthy_if_promote_not_validated(self):
        # A run "completing on schedule" is not enough for healthy (RFC §3.2).
        assert M.freshness_tier(10, all_sources_on_sla=True,
                                validated_advancing_promote=False) == "breach"

    def test_warn_band(self):
        assert M.freshness_tier(30, all_sources_on_sla=True,
                                validated_advancing_promote=True) == "warn"

    def test_breach_over_35d(self):
        assert M.freshness_tier(140, all_sources_on_sla=False,
                                validated_advancing_promote=False) == "breach"


# --- §3.4 gate helpers ------------------------------------------------------

class TestGateHelpers:
    def test_non_degenerate_ok(self):
        ok, _ = M.check_non_degenerate([0.1, -0.2, 0.3, 0.05])
        assert ok

    def test_non_degenerate_rejects_constant(self):
        ok, why = M.check_non_degenerate([0.5, 0.5, 0.5])
        assert not ok and "constant" in why

    def test_non_degenerate_rejects_nan(self):
        ok, why = M.check_non_degenerate([float("nan"), 0.1])
        assert not ok and "NaN" in why

    def test_non_degenerate_rejects_empty(self):
        ok, _ = M.check_non_degenerate([])
        assert not ok

    def test_resource_within_budget(self):
        ok, _ = M.check_resource(5.0, 200.0, max_seconds=120, max_rss_mb=4096)
        assert ok

    def test_resource_latency_breach(self):
        ok, why = M.check_resource(500.0, 200.0, max_seconds=120, max_rss_mb=4096)
        assert not ok and "latency" in why

    def test_resource_rss_breach(self):
        ok, why = M.check_resource(5.0, 9000.0, max_seconds=120, max_rss_mb=4096)
        assert not ok and "RSS" in why

    def test_sanity_floor_pass_fail_missing(self):
        assert M.check_sanity_floor(0.03, 0.0)[0]
        assert not M.check_sanity_floor(-0.05, 0.0)[0]
        assert not M.check_sanity_floor(None, 0.0)[0]


class TestCandidateQualityMetric:
    def test_top_level(self):
        assert M.candidate_quality_metric({"wf_ic": 0.031}) == pytest.approx(0.031)

    def test_nested_selection(self):
        assert M.candidate_quality_metric({"selection": {"ic": 0.02}}) == pytest.approx(0.02)

    def test_absent(self):
        assert M.candidate_quality_metric({"cutoff_date": "2026-03-09"}) is None


# --- run_promote dry-run integration ---------------------------------------

def _base_args(**over):
    d = dict(
        served_config="", pin_key=M.DEFAULT_PIN_KEY, wf_manifest="", candidate=None,
        served_root=M.DEFAULT_SERVED_ROOT, stamp_script=M.DEFAULT_STAMP_SCRIPT,
        sources_json=None, fast_ceiling_days=28, sanity_floor=0.0,
        resource_max_seconds=120.0, resource_max_rss_mb=4096.0,
        allow_non_fresh=False, reason=None, skip_inference_gate=False,
        apply=False, check=True, json=False, now=dt.date(2026, 6, 30),
    )
    d.update(over)
    return SimpleNamespace(**d)


def _write_pt_stub(tmp_path: Path, name: str, axes: dict) -> Path:
    pt = tmp_path / name
    pt.write_bytes(b"stub")
    meta = {**axes, "training_contract": {"label_col": "fwd_60d_excess",
                                          **{k: v for k, v in axes.items()}}}
    (tmp_path / (name + ".metadata.json")).write_text(json.dumps(meta))
    return pt


def test_run_promote_refuses_non_hf_patchtst_config(tmp_path, monkeypatch):
    cfg = tmp_path / "strategy_config.shadow.json"
    cfg.write_text(json.dumps({"ranking": {"panel_scoring": {
        "kind": "xgb", "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json"}}}))
    monkeypatch.setattr(M, "REPO", tmp_path)
    rep = M.run_promote(_base_args(served_config=str(cfg)))
    assert rep.rc == M.RC_USAGE and "not 'hf_patchtst'" in rep.verdict


def test_run_promote_refuses_when_not_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "REPO", tmp_path)
    # served pin (stale) + candidate whose selection cutoff does NOT advance
    served_pt = _write_pt_stub(tmp_path, "served.pt", {
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60})
    cand_pt = _write_pt_stub(tmp_path, "cand.pt", {
        "effective_train_cutoff_date": "2025-12-12",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60})
    cfg = tmp_path / "strategy_config.shadow.json"
    cfg.write_text(json.dumps({"ranking": {"panel_scoring": {
        "kind": "hf_patchtst", "artifact_path": "served.pt"}}}))
    # sources all off-SLA (no files) -> also not fresh
    args = _base_args(served_config=str(cfg), candidate=str(cand_pt),
                      sources_json=json.dumps([{"name": "panel", "path": "missing.parquet",
                                               "axis": "fast", "sla_days": 28}]))
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH
    assert not rep.fresh


def test_run_promote_dryrun_ok_when_fresh_and_gates_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "REPO", tmp_path)
    served_pt = _write_pt_stub(tmp_path, "served.pt", {
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60})
    cand_pt = _write_pt_stub(tmp_path, "cand.pt", {
        "effective_train_cutoff_date": "2026-05-01",
        "effective_selection_cutoff_date": "2026-06-15", "lookahead_days": 60,
        "wf_ic": 0.03})
    cfg = tmp_path / "strategy_config.shadow.json"
    cfg.write_text(json.dumps({"ranking": {"panel_scoring": {
        "kind": "hf_patchtst", "artifact_path": "served.pt", "lookahead_days": 60}}}))
    # a fresh source file, mtime pinned to the test's `now` (2026-06-30) — NOT the
    # real wall clock, so this test is not future-dated (and thus fail-closed) once
    # real time moves past 2026-06-30 (Codex #419 review 3 fail-closed fix).
    src = tmp_path / "panel.parquet"
    src.write_bytes(b"x")
    _touch(src, dt.datetime(2026, 6, 30, 12, 0, 0))
    sources = json.dumps([{"name": "panel", "path": "panel.parquet",
                           "axis": "fast", "sla_days": 100000}])
    # mock the heavy smoke inference + fingerprint parity (no torch / no stamp tool)
    monkeypatch.setattr(M, "load_and_smoke_infer", lambda *a, **k: {
        "ok": True, "reason": "mock", "scores": [0.1, -0.2, 0.3],
        "elapsed_s": 1.0, "peak_rss_mb": 100.0})
    monkeypatch.setattr(M, "_parity_gate", lambda *a, **k: (True, "mock parity OK"))
    args = _base_args(served_config=str(cfg), candidate=str(cand_pt), sources_json=sources)
    rep = M.run_promote(args)
    assert rep.rc == M.RC_OK, rep.verdict
    assert rep.fresh and "DRY-RUN OK" in rep.verdict
    assert rep.promoted_pin is None  # dry-run never swaps


def test_run_promote_gate_failure_when_smoke_degenerate(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "REPO", tmp_path)
    _write_pt_stub(tmp_path, "served.pt", {
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60})
    cand_pt = _write_pt_stub(tmp_path, "cand.pt", {
        "effective_train_cutoff_date": "2026-05-01",
        "effective_selection_cutoff_date": "2026-06-15", "lookahead_days": 60,
        "wf_ic": 0.03})
    cfg = tmp_path / "strategy_config.shadow.json"
    cfg.write_text(json.dumps({"ranking": {"panel_scoring": {
        "kind": "hf_patchtst", "artifact_path": "served.pt", "lookahead_days": 60}}}))
    src = tmp_path / "panel.parquet"
    src.write_bytes(b"x")
    _touch(src, dt.datetime(2026, 6, 30, 12, 0, 0))  # pinned mtime, see comment above
    sources = json.dumps([{"name": "panel", "path": "panel.parquet",
                           "axis": "fast", "sla_days": 100000}])
    # degenerate (constant) probe scores -> non_degenerate gate fails
    monkeypatch.setattr(M, "load_and_smoke_infer", lambda *a, **k: {
        "ok": True, "reason": "mock", "scores": [0.5, 0.5, 0.5],
        "elapsed_s": 1.0, "peak_rss_mb": 100.0})
    monkeypatch.setattr(M, "_parity_gate", lambda *a, **k: (True, "mock parity OK"))
    args = _base_args(served_config=str(cfg), candidate=str(cand_pt), sources_json=sources)
    rep = M.run_promote(args)
    assert rep.rc == M.RC_GATE_FAILED
    assert any(g.name == "non_degenerate" and not g.ok for g in rep.gates)


# ===========================================================================
# Codex review #419 fixes — freshness must FAIL CLOSED (never fall to mtime),
# and fundamentals must test QUARTERLY availability, not just max daily date.
# Every case below must keep the OLD pin (rc == RC_NOT_FRESH, never promoted).
# ===========================================================================

def _write_parquet(path: Path, rows: dict) -> Path:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _touch(path: Path, when: dt.datetime) -> None:
    ts = when.timestamp()
    os.utime(path, (ts, ts))


# --- resolve_data_cutoff: declared parquet/date-col sources fail closed ------

class TestResolveDataCutoffFailsClosed:
    def test_corrupt_parquet_is_unresolved_not_mtime(self, tmp_path):
        # (a) unreadable/corrupt parquet with a FRESH mtime must NOT pass on mtime.
        p = tmp_path / "corrupt.parquet"
        p.write_bytes(b"not a parquet file")
        _touch(p, dt.datetime(2026, 6, 30, 12, 0, 0))  # mtime "today"
        src = {"name": "panel", "path": str(p), "axis": "fast",
               "sla_days": 28, "date_col": "date"}
        assert M.resolve_data_cutoff(tmp_path, src) is None  # fail-closed

    def test_missing_date_column_is_unresolved(self, tmp_path):
        # (b) valid parquet WITHOUT the declared date column -> unresolved.
        pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "nocol.parquet", {"ticker": ["A", "B"]})
        _touch(p, dt.datetime(2026, 6, 30, 12, 0, 0))
        src = {"name": "panel", "path": str(p), "axis": "fast",
               "sla_days": 28, "date_col": "date"}
        assert M.resolve_data_cutoff(tmp_path, src) is None

    def test_empty_date_column_is_unresolved(self, tmp_path):
        # (b') present but all-null date column -> no parseable cutoff -> unresolved.
        pd = pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "emptycol.parquet",
                           {"date": pd.to_datetime([None, None])})
        src = {"name": "panel", "path": str(p), "axis": "fast",
               "sla_days": 28, "date_col": "date"}
        assert M.resolve_data_cutoff(tmp_path, src) is None

    def test_freshly_touched_old_data_uses_data_cutoff_not_mtime(self, tmp_path):
        # (c) readable parquet with OLD data but a brand-new mtime: the resolved
        # cutoff is the OLD data date, never the fresh mtime.
        pd = pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "old.parquet",
                           {"date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
        _touch(p, dt.datetime(2026, 6, 30, 12, 0, 0))  # mtime "today", data old
        src = {"name": "panel", "path": str(p), "axis": "fast",
               "sla_days": 28, "date_col": "date"}
        cutoff = M.resolve_data_cutoff(tmp_path, src)
        assert cutoff == dt.date(2026, 1, 2)  # data cutoff, NOT today's mtime
        v = M.source_sla_verdict(src, dt.date(2026, 6, 30), cutoff)
        assert not v.on_sla  # ~179d old -> off 28d SLA

    def test_readable_recent_parquet_resolves(self, tmp_path):
        pd = pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "recent.parquet",
                           {"date": pd.to_datetime(["2026-06-20", "2026-06-25"])})
        src = {"name": "panel", "path": str(p), "axis": "fast",
               "sla_days": 28, "date_col": "date"}
        assert M.resolve_data_cutoff(tmp_path, src) == dt.date(2026, 6, 25)


# --- fundamentals TWO-AXIS, per-ENTITY coverage -----------------------------
# Codex #419 review 2: a single global max fiscal date lets ONE fresh issuer certify a
# frozen panel. Freshness is per-entity (own fiscal-period end, no calendar snapping)
# gated on a preregistered COVERAGE DISTRIBUTION.

def _fund_src(**over):
    s = {"name": "fundamentals", "axis": "slow"}
    s.update(over)
    return s


class TestEntityQuartersBehind:
    _NOW = dt.date(2026, 6, 30)

    def test_calendar_current_is_zero_behind(self):
        # Calendar FY: latest period Mar-31 (Q1) is current on late June.
        assert M.entity_quarters_behind(dt.date(2026, 3, 31), self._NOW, 45) == 0

    def test_calendar_one_quarter_stale(self):
        # Stuck at Q4-2025 -> 1 quarter behind.
        assert M.entity_quarters_behind(dt.date(2025, 12, 31), self._NOW, 45) == 1

    def test_calendar_two_quarters_stale(self):
        assert M.entity_quarters_behind(dt.date(2025, 9, 30), self._NOW, 45) == 2

    def test_non_calendar_fiscal_year_current(self):
        # Jan-FYE issuer: quarters end Apr/Jul/Oct/Jan. Latest = Apr-30 (61d) is current
        # WITHOUT any calendar-quarter snapping — the reviewer's non-calendar case.
        assert M.entity_quarters_behind(dt.date(2026, 4, 30), self._NOW, 45) == 0

    def test_non_calendar_fiscal_year_stale(self):
        # Same issuer stuck at Jan-31 (missed the Apr-30 quarter) -> 1 behind.
        assert M.entity_quarters_behind(dt.date(2026, 1, 31), self._NOW, 45) == 1

    def test_missing_is_none(self):
        assert M.entity_quarters_behind(None, self._NOW, 45) is None

    # --- Codex #419 review 3: future-dated fiscal periods must FAIL CLOSED, ------
    # --- never be silently treated as "0 quarters behind" (maximally fresh). ----

    def test_future_dated_fails_closed_not_zero(self):
        # An impossible fiscal-period end (later than "today") must return the
        # FUTURE_DATED sentinel, NOT 0 (which would read as "current").
        v = M.entity_quarters_behind(dt.date(2026, 9, 30), self._NOW, 45)
        assert v == M.FUTURE_DATED
        assert v != 0

    def test_one_day_future_fails_closed(self):
        # Just past the (zero-day) clock-skew tolerance -> must fail.
        v = M.entity_quarters_behind(self._NOW + dt.timedelta(days=1), self._NOW, 45)
        assert v == M.FUTURE_DATED

    def test_exactly_today_is_boundary_and_passes(self):
        # A fiscal-period end dated exactly "today" (staleness == 0) is the boundary:
        # not future, and current (0 quarters behind).
        assert M.entity_quarters_behind(self._NOW, self._NOW, 45) == 0


class TestFundamentalsCoverage:
    _NOW = dt.date(2026, 6, 30)

    def test_one_fresh_many_stale_is_almost_all_stale(self):
        # The reviewer's regression at the distribution level: 1 current + 291 stale.
        fbe = {"FRESH": dt.date(2026, 3, 31)}
        fbe.update({f"S{i}": dt.date(2025, 12, 31) for i in range(291)})
        cov = M.fundamentals_coverage(fbe, self._NOW, filing_lag_days=45,
                                      max_quarters_behind=1)
        assert cov.n_entities == 292 and cov.n_current == 1
        assert cov.n_stale == 291 and cov.stale_fraction > 0.99
        assert cov.worst_quarters_behind == 1

    def test_all_current(self):
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        cov = M.fundamentals_coverage(fbe, self._NOW, filing_lag_days=45,
                                      max_quarters_behind=1)
        assert cov.stale_fraction == 0.0 and cov.missing_fraction == 0.0

    def test_missing_counts_as_stale(self):
        fbe = {"A": dt.date(2026, 3, 31), "B": None, "C": None, "D": None}
        cov = M.fundamentals_coverage(fbe, self._NOW, filing_lag_days=45,
                                      max_quarters_behind=1)
        assert cov.n_missing == 3 and cov.n_stale == 3
        assert cov.missing_fraction == pytest.approx(0.75)

    def test_future_dated_counts_as_stale_not_current(self):
        # Codex #419 review 3: a future-dated entity must count against staleness,
        # never be silently folded into n_current (which "0 quarters behind" would do).
        fbe = {"A": dt.date(2026, 3, 31), "FUTURE": dt.date(2026, 9, 30)}
        cov = M.fundamentals_coverage(fbe, self._NOW, filing_lag_days=45,
                                      max_quarters_behind=1)
        assert cov.n_future_dated == 1
        assert cov.n_current == 1  # only the genuinely current entity
        assert cov.n_stale == 1 and cov.n_missing == 0


class TestFundamentalsVerdict:
    _NOW = dt.date(2026, 6, 30)
    _KW = dict(max_feed_stale_days=20, filing_lag_days=45, max_quarters_behind=1,
               max_stale_fraction=0.05, max_missing_fraction=0.02,
               max_worst_quarters_behind=1, min_entities=1)

    def test_no_provenance_fails_closed(self):
        # Current daily feed but NO per-entity fiscal/available-at provenance ->
        # UNVERIFIABLE -> fail closed (a global max cannot establish freshness).
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity=None, provenance_present=False, **self._KW)
        assert not v.on_sla and "UNVERIFIABLE" in v.detail

    def test_one_fresh_291_stale_fails_closed(self):
        # THE reviewer's regression: 1 Q1-current + 291 Q4-stale MUST fail, even
        # though a global max(fiscal) would read Q1 (fresh).
        fbe = {"FRESH": dt.date(2026, 3, 31)}
        fbe.update({f"S{i}": dt.date(2025, 12, 31) for i in range(291)})
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity=fbe, provenance_present=True,
            **{**self._KW, "min_entities": 50})
        assert not v.on_sla and "STALE-COVERAGE" in v.detail
        assert v.coverage["n_current"] == 1 and v.coverage["stale_fraction"] > 0.99

    def test_all_current_passes(self):
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity=fbe, provenance_present=True, **self._KW)
        assert v.on_sla

    def test_stale_daily_feed_fails_even_if_coverage_ok(self):
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 3, 31),
            fiscal_by_entity=fbe, provenance_present=True, **self._KW)
        assert not v.on_sla and "STALE" in v.detail

    def test_too_few_entities_fails_closed(self):
        # A tiny cross-section (1 fresh issuer) must not certify the panel.
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity={"ONLY": dt.date(2026, 3, 31)}, provenance_present=True,
            **{**self._KW, "min_entities": 50})
        assert not v.on_sla and "too few entities" in v.detail

    def test_worst_case_outlier_fails_even_if_fraction_ok(self):
        # 199 current + 1 deeply-stale (3 quarters behind): stale_fraction tiny but the
        # worst-case cap catches the outlier.
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(199)}
        fbe["OUTLIER"] = dt.date(2025, 6, 30)  # ~3 quarters behind
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity=fbe, provenance_present=True,
            **{**self._KW, "min_entities": 50, "max_stale_fraction": 0.10})
        assert not v.on_sla and "worst=" in v.detail

    # --- Codex #419 review 3: future-dated observations must FAIL CLOSED --------
    # --- (BOTH the global daily-feed axis and the per-entity fiscal axis). ------

    def test_global_future_dated_feed_fails_closed(self):
        # The daily-feed as-of date itself postdates "now" -> impossible; must fail
        # closed with a distinct reason, never be clamped to age=0 (maximally fresh).
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 7, 5),
            fiscal_by_entity=fbe, provenance_present=True, **self._KW)
        assert not v.on_sla
        assert M.FUTURE_DATED in v.detail

    def test_global_feed_one_day_future_fails_closed(self):
        # Just past the (zero-day) clock-skew tolerance -> must fail.
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=self._NOW + dt.timedelta(days=1),
            fiscal_by_entity=fbe, provenance_present=True, **self._KW)
        assert not v.on_sla and M.FUTURE_DATED in v.detail

    def test_global_feed_exactly_now_is_boundary_and_passes(self):
        # A daily feed dated exactly "now" is the boundary: not future, and current.
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(100)}
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=self._NOW,
            fiscal_by_entity=fbe, provenance_present=True, **self._KW)
        assert v.on_sla

    def test_per_entity_future_dated_fiscal_period_fails_closed(self):
        # A daily-fresh feed with a per-entity fiscal-period end LATER than "now" is
        # impossible (a real look-ahead-leak signal) -> must fail closed even though
        # the naive "0 quarters behind" reading would otherwise pass as current.
        fbe = {f"T{i}": dt.date(2026, 3, 31) for i in range(99)}
        fbe["FUTURE"] = dt.date(2026, 9, 30)
        v = M.fundamentals_sla_verdict(
            _fund_src(), self._NOW, feed_max_date=dt.date(2026, 6, 29),
            fiscal_by_entity=fbe, provenance_present=True,
            **{**self._KW, "min_entities": 50})
        assert not v.on_sla
        assert M.FUTURE_DATED in v.detail
        assert v.coverage["n_future_dated"] == 1


class TestResolveFundamentalsVerdict:
    _NOW = dt.date(2026, 6, 30)

    def test_only_daily_date_no_fiscal_column_fails_closed(self, tmp_path):
        # The real prod schema today (ticker + daily as-of ``date`` + features, NO
        # fiscal-period column) -> per-entity coverage UNVERIFIABLE -> fail closed.
        pd = pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "fund.parquet",
                           {"ticker": ["A", "B"],
                            "date": pd.to_datetime(["2026-06-28", "2026-06-29"]),
                            "book_to_price": [0.5, 0.5]})
        src = {"name": "fundamentals", "path": str(p), "axis": "slow",
               "kind": "fundamentals", "date_col": "date",
               "fiscal_period_cols": ["period_end", "fiscal_period_end"]}
        v = M.resolve_fundamentals_verdict(tmp_path, src, self._NOW)
        assert not v.on_sla and "UNVERIFIABLE" in v.detail

    def test_no_entity_column_fails_closed(self, tmp_path):
        # A fiscal column but NO entity id -> per-entity coverage UNVERIFIABLE.
        pd = pytest.importorskip("pandas")
        p = _write_parquet(tmp_path / "fund.parquet",
                           {"date": pd.to_datetime(["2026-06-28", "2026-06-29"]),
                            "period_end": pd.to_datetime(["2026-03-31", "2026-03-31"])})
        src = {"name": "fundamentals", "path": str(p), "axis": "slow",
               "kind": "fundamentals", "date_col": "date",
               "entity_cols": ["ticker"], "fiscal_period_cols": ["period_end"]}
        v = M.resolve_fundamentals_verdict(tmp_path, src, self._NOW)
        assert not v.on_sla and "UNVERIFIABLE" in v.detail

    def test_per_entity_one_fresh_many_stale_fails_closed(self, tmp_path):
        # (KEY end-to-end regression) 1 ticker Q1-current + 291 tickers Q4-stale.
        pd = pytest.importorskip("pandas")
        tickers = ["FRESH"] + [f"S{i}" for i in range(291)]
        periods = [dt.date(2026, 3, 31)] + [dt.date(2025, 12, 31)] * 291
        p = _write_parquet(tmp_path / "fund.parquet",
                           {"ticker": tickers, "date": [dt.date(2026, 6, 29)] * 292,
                            "period_end": pd.to_datetime(periods)})
        src = {"name": "fundamentals", "path": str(p), "axis": "slow",
               "kind": "fundamentals", "date_col": "date",
               "entity_cols": ["ticker"], "fiscal_period_cols": ["period_end"],
               "min_entities": 50}
        v = M.resolve_fundamentals_verdict(tmp_path, src, self._NOW)
        assert not v.on_sla and "STALE-COVERAGE" in v.detail
        assert v.coverage["n_entities"] == 292 and v.coverage["n_current"] == 1

    def test_per_entity_all_current_passes(self, tmp_path):
        pd = pytest.importorskip("pandas")
        tickers = [f"T{i}" for i in range(60)]
        p = _write_parquet(tmp_path / "fund.parquet",
                           {"ticker": tickers, "date": [dt.date(2026, 6, 29)] * 60,
                            "period_end": pd.to_datetime([dt.date(2026, 3, 31)] * 60)})
        src = {"name": "fundamentals", "path": str(p), "axis": "slow",
               "kind": "fundamentals", "date_col": "date",
               "entity_cols": ["ticker"], "fiscal_period_cols": ["period_end"],
               "min_entities": 50}
        v = M.resolve_fundamentals_verdict(tmp_path, src, self._NOW)
        assert v.on_sla


# --- run_promote integration: pin unchanged in every fail-closed case -------

def _promote_setup(tmp_path, sources, *, cand_fresh=True):
    """Serve a stale pin + a candidate whose cutoffs advance; the ONLY reason a
    given case is not-fresh is the source SLA under test. Returns run args."""
    _write_pt_stub(tmp_path, "served.pt", {
        "effective_train_cutoff_date": "2024-11-13",
        "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60})
    cand_axes = {"effective_train_cutoff_date": "2026-05-01",
                 "effective_selection_cutoff_date": "2026-06-15",
                 "lookahead_days": 60, "wf_ic": 0.03} if cand_fresh else {
                 "effective_train_cutoff_date": "2024-11-13",
                 "effective_selection_cutoff_date": "2026-02-10", "lookahead_days": 60}
    cand_pt = _write_pt_stub(tmp_path, "cand.pt", cand_axes)
    cfg = tmp_path / "strategy_config.shadow.json"
    cfg.write_text(json.dumps({"ranking": {"panel_scoring": {
        "kind": "hf_patchtst", "artifact_path": "served.pt", "lookahead_days": 60}}}))
    return _base_args(served_config=str(cfg), candidate=str(cand_pt),
                      sources_json=json.dumps(sources)), cfg


def test_promote_keeps_pin_on_corrupt_parquet(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "REPO", tmp_path)
    (tmp_path / "panel.parquet").write_bytes(b"corrupt")
    _touch(tmp_path / "panel.parquet", dt.datetime(2026, 6, 30, 12, 0, 0))
    args, cfg = _promote_setup(tmp_path, [{"name": "panel", "path": "panel.parquet",
                                           "axis": "fast", "sla_days": 28, "date_col": "date"}])
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before  # pin unchanged


def test_promote_keeps_pin_on_missing_date_column(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(M, "REPO", tmp_path)
    _write_parquet(tmp_path / "panel.parquet", {"ticker": ["A", "B"]})
    args, cfg = _promote_setup(tmp_path, [{"name": "panel", "path": "panel.parquet",
                                           "axis": "fast", "sla_days": 28, "date_col": "date"}])
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before


def test_promote_keeps_pin_on_freshly_touched_old_data(tmp_path, monkeypatch):
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(M, "REPO", tmp_path)
    _write_parquet(tmp_path / "panel.parquet",
                   {"date": pd.to_datetime(["2026-01-01", "2026-01-02"])})
    _touch(tmp_path / "panel.parquet", dt.datetime(2026, 6, 30, 12, 0, 0))  # fresh mtime
    args, cfg = _promote_setup(tmp_path, [{"name": "panel", "path": "panel.parquet",
                                           "axis": "fast", "sla_days": 28, "date_col": "date"}])
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before


def test_promote_keeps_pin_on_one_fresh_many_stale_fundamentals(tmp_path, monkeypatch):
    # The reviewer's explicit end-to-end regression: 1 ticker Q1-current + 291 tickers
    # Q4-stale -> the pin MUST stay unchanged (a single fresh issuer must not promote a
    # challenger trained on a frozen fundamentals cross-section).
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(M, "REPO", tmp_path)
    # A FRESH fast source (so freshness fails ONLY on the fundamentals coverage axis).
    _write_parquet(tmp_path / "panel.parquet",
                   {"date": pd.to_datetime(["2026-06-28", "2026-06-29"])})
    tickers = ["FRESH"] + [f"S{i}" for i in range(291)]
    periods = [dt.date(2026, 3, 31)] + [dt.date(2025, 12, 31)] * 291
    _write_parquet(tmp_path / "fund.parquet",
                   {"ticker": tickers, "date": [dt.date(2026, 6, 29)] * 292,
                    "period_end": pd.to_datetime(periods)})
    sources = [
        {"name": "panel", "path": "panel.parquet", "axis": "fast",
         "sla_days": 28, "date_col": "date"},
        {"name": "fundamentals", "path": "fund.parquet", "axis": "slow",
         "kind": "fundamentals", "date_col": "date", "entity_cols": ["ticker"],
         "fiscal_period_cols": ["period_end"], "min_entities": 50},
    ]
    args, cfg = _promote_setup(tmp_path, sources)
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before  # pin unchanged
    fund_v = next(v for v in rep.source_verdicts if v.name == "fundamentals")
    assert not fund_v.on_sla and "STALE-COVERAGE" in fund_v.detail
    assert fund_v.coverage["n_current"] == 1 and fund_v.coverage["n_entities"] == 292
    panel_v = next(v for v in rep.source_verdicts if v.name == "panel")
    assert panel_v.on_sla  # the fast source IS fresh; only fundamentals blocks


# --- Codex #419 review 3: future-dated observations must FAIL CLOSED end-to-end ---

def test_promote_keeps_pin_on_future_dated_panel(tmp_path, monkeypatch):
    # A fast-axis cutoff LATER than the decision timestamp is impossible and must
    # fail closed -- never pass trivially because a negative age is "<= sla_days".
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(M, "REPO", tmp_path)
    # now == 2026-06-30 (see _base_args); 2026-07-05 is future-dated.
    _write_parquet(tmp_path / "panel.parquet",
                   {"date": pd.to_datetime(["2026-06-29", "2026-07-05"])})
    args, cfg = _promote_setup(tmp_path, [{"name": "panel", "path": "panel.parquet",
                                           "axis": "fast", "sla_days": 28, "date_col": "date"}])
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before  # pin unchanged
    panel_v = next(v for v in rep.source_verdicts if v.name == "panel")
    assert not panel_v.on_sla and M.FUTURE_DATED in panel_v.detail


def test_promote_keeps_pin_on_future_dated_fundamentals_entity(tmp_path, monkeypatch):
    # A per-entity fiscal-period end LATER than the decision timestamp is impossible
    # and must fail closed, not be read as "0 quarters behind" (maximally fresh) --
    # even though the daily feed itself and the other 99 entities are current.
    pd = pytest.importorskip("pandas")
    monkeypatch.setattr(M, "REPO", tmp_path)
    _write_parquet(tmp_path / "panel.parquet",
                   {"date": pd.to_datetime(["2026-06-28", "2026-06-29"])})
    tickers = [f"T{i}" for i in range(99)] + ["FUTURE"]
    periods = [dt.date(2026, 3, 31)] * 99 + [dt.date(2026, 9, 30)]
    _write_parquet(tmp_path / "fund.parquet",
                   {"ticker": tickers, "date": [dt.date(2026, 6, 29)] * 100,
                    "period_end": pd.to_datetime(periods)})
    sources = [
        {"name": "panel", "path": "panel.parquet", "axis": "fast",
         "sla_days": 28, "date_col": "date"},
        {"name": "fundamentals", "path": "fund.parquet", "axis": "slow",
         "kind": "fundamentals", "date_col": "date", "entity_cols": ["ticker"],
         "fiscal_period_cols": ["period_end"], "min_entities": 50},
    ]
    args, cfg = _promote_setup(tmp_path, sources)
    before = cfg.read_text()
    rep = M.run_promote(args)
    assert rep.rc == M.RC_NOT_FRESH and not rep.fresh and rep.promoted_pin is None
    assert cfg.read_text() == before  # pin unchanged
    fund_v = next(v for v in rep.source_verdicts if v.name == "fundamentals")
    assert not fund_v.on_sla and M.FUTURE_DATED in fund_v.detail
    assert fund_v.coverage["n_future_dated"] == 1
