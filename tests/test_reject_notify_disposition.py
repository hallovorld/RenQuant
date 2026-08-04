"""Unit contract for scripts/reject_notify_disposition.py (operator directive
2026-08-04: the weekly reject notification must stop reporting the healthy
fresh-prod steady state with a failure tone/exit).

The helper's one job: prove the fresh-refusal shape from the verdict JSON, or
land in ALARM. Every malformed twin below is a case that MUST NOT read as calm
— the enumerated-allowlist lesson inverted into tests: the default is ALARM.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HELPER = REPO_ROOT / "scripts" / "reject_notify_disposition.py"

spec = importlib.util.spec_from_file_location("reject_notify_disposition", HELPER)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def _fresh_verdict(**overrides):
    """The REAL shape freshness_fallback --stamp wrote on 2026-08-04 (live
    example: logs/weekly_wf_promote/20260804T200020Z.fallback_verdict.json)."""
    v = {
        "as_of": "2026-08-04",
        "decision": "REFUSE",
        "policy": "freshness_fallback_rfc210",
        "refused_on": "prod_stale",
        "checks": [
            {"check": "gate_rejected", "ok": True, "stamped_verdict": False},
            {"check": "prod_stale", "ok": False, "prod_trained": "2026-08-02",
             "staleness_days": 2,
             "why": "served model is 2d old (<= 28d SLA)"},
        ],
    }
    v.update(overrides)
    return v


def _dispose(tmp_path, payload) -> str:
    p = tmp_path / "verdict.json"
    p.write_text(json.dumps(payload) if not isinstance(payload, str) else payload,
                 encoding="utf-8")
    return mod.dispose(str(p))


def test_the_live_fresh_shape_is_calm(tmp_path):
    assert _dispose(tmp_path, _fresh_verdict()) == "CALM_FRESH|2|2026-08-02"


def test_missing_file_alarms():
    assert mod.dispose("/nonexistent/verdict.json").startswith("ALARM|")


def test_malformed_json_alarms(tmp_path):
    assert _dispose(tmp_path, "{not json").startswith("ALARM|")


def test_the_test_harness_stub_shape_alarms(tmp_path):
    # The RFC210 harness stub writes {"verdict": "REFUSE"} — a DIFFERENT
    # schema. An unproven verdict must never read as calm.
    assert _dispose(tmp_path, {"verdict": "REFUSE", "reason": "x"}).startswith("ALARM|")


def test_promote_decision_alarms(tmp_path):
    assert _dispose(tmp_path, _fresh_verdict(decision="FALLBACK_PROMOTE")).startswith("ALARM|")


def test_refusal_on_another_check_alarms(tmp_path):
    out = _dispose(tmp_path, _fresh_verdict(refused_on="candidate_stale"))
    assert out.startswith("ALARM|") and "candidate_stale" in out


def test_prod_stale_ok_true_alarms(tmp_path):
    v = _fresh_verdict()
    v["checks"][1]["ok"] = True
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_prod_stale_ok_absent_or_none_alarms(tmp_path):
    v = _fresh_verdict()
    del v["checks"][1]["ok"]
    assert _dispose(tmp_path, v).startswith("ALARM|")
    v["checks"][1]["ok"] = None
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_prod_stale_ok_falsy_nonbool_alarms(tmp_path):
    # 0 / "" are falsy but are NOT the explicit sentinel False.
    v = _fresh_verdict()
    v["checks"][1]["ok"] = 0
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_staleness_days_string_alarms(tmp_path):
    v = _fresh_verdict()
    v["checks"][1]["staleness_days"] = "2"
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_staleness_days_bool_alarms(tmp_path):
    v = _fresh_verdict()
    v["checks"][1]["staleness_days"] = True
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_staleness_beyond_sla_alarms_even_with_fresh_shape(tmp_path):
    # Internally inconsistent verdict: refused_on says fresh, number says 44d.
    v = _fresh_verdict()
    v["checks"][1]["staleness_days"] = 44
    out = _dispose(tmp_path, v)
    assert out.startswith("ALARM|") and "44" in out


def test_empty_prod_trained_alarms(tmp_path):
    v = _fresh_verdict()
    v["checks"][1]["prod_trained"] = "  "
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_checks_missing_or_wrong_type_alarms(tmp_path):
    assert _dispose(tmp_path, _fresh_verdict(checks=None)).startswith("ALARM|")
    assert _dispose(tmp_path, _fresh_verdict(checks={})).startswith("ALARM|")


def test_prod_stale_check_absent_alarms(tmp_path):
    v = _fresh_verdict()
    v["checks"] = [c for c in v["checks"] if c["check"] != "prod_stale"]
    assert _dispose(tmp_path, v).startswith("ALARM|")


def test_top_level_list_alarms(tmp_path):
    assert _dispose(tmp_path, [_fresh_verdict()]).startswith("ALARM|")


def test_cli_always_exits_zero_and_prints_one_line(tmp_path):
    p = tmp_path / "v.json"
    p.write_text(json.dumps(_fresh_verdict()), encoding="utf-8")
    for args in ([str(p)], ["/nonexistent"], []):
        r = subprocess.run([sys.executable, str(HELPER), *args],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, (args, r.stderr)
        lines = r.stdout.strip().splitlines()
        assert len(lines) == 1, r.stdout
        assert lines[0].startswith(("CALM_FRESH|", "ALARM|"))
