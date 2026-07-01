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
    # a fresh source file (mtime ~ now)
    src = tmp_path / "panel.parquet"
    src.write_bytes(b"x")
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
