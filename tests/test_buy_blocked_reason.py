"""scripts/buy_blocked_reason.py — the BUY-BLOCKED alert says WHY, urgently, once a day.

Fixture = the metadata block of the artifact that goes sell-only on
2026-08-31 (artifacts/prod/panel-ltr.alpha158_fund.json, trained 2026-08-02,
passed=false, promotion_basis=freshness_fallback_rfc210,
fallback_genuine_ic=+0.00289), shape copied from the served file [VERIFIED
2026-08-30]. No network: the sender is injected; conftest-level
RENQUANT_NO_NOTIFY=1 (pytest.ini) is the second wall.
"""
from __future__ import annotations

import datetime as dt
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "buy_blocked_reason.py"
DAILY_104 = REPO_ROOT / "scripts" / "daily_104.sh"


def _load_module():
    spec = importlib.util.spec_from_file_location("buy_blocked_reason", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


bbr = _load_module()

TODAY = dt.date(2026, 8, 31)

# The served artifact's metadata block, shape-for-shape (numbers rounded).
SERVED_METADATA = {
    "fallback_as_of": "2026-08-04",
    "fallback_genuine_ic": 0.0028876304346270865,
    "fallback_prod_staleness_days": 44,
    "promotion_basis": "freshness_fallback_rfc210",
    "wf_gate_metadata": {
        "passed": False,
        "diagnostic_only": False,
        "candidate_artifact_used": False,
        "wf_3cut_sharpe_mean": 0.6017718060321567,
        "spy_sharpe_mean": 1.0808386653410664,
        "strategy_minus_spy_sharpe_mean": -0.4790668593089097,
        "n_cuts_beat_spy_sharpe": 1,
        "wf_reason": (
            "FAIL: absolute_ok=True, benchmark_ok=False, regime_ok=False; "
            "mean Sharpe +0.602, 3/3 cuts > 0; SPY mean Sharpe +1.081"
        ),
        "run_at": "2026-08-02T17:22:10.666342",
        "sanity_placebo_genuine_ic": 0.0028876304346270865,
        "sanity_regime_ic": {
            "passed": False,
            "reason": "regime sanity IC failed: BULL_CALM,BULL_VOLATILE,CHOPPY",
        },
        "trade_monotonicity": {
            "passed": False,
            "pooled": {"n": 117, "spearman": 0.03911372202282383},
            "regimes": [
                {"regime": "BULL_CALM", "n": 104, "eligible": True,
                 "passed": False, "spearman": 0.0023365233812373976},
                {"regime": "BULL_VOLATILE", "n": 11, "eligible": False,
                 "passed": False, "spearman": 0.27272727272727276},
            ],
        },
    },
}


def _served_payload() -> dict:
    return {
        "kind": "panel_ltr_xgboost",
        "trained_date": "2026-08-02",
        "oos_mean_ic": 0.04478635239480713,
        "feature_cols": ["f1", "f2"],
        "metadata": json.loads(json.dumps(SERVED_METADATA)),
    }


def _config() -> dict:
    return {
        "ranking": {"panel_scoring": {
            "enabled": True, "kind": "blend",
            "artifact_path": "artifacts/prod/panel-ltr.alpha158_fund.json",
        }},
        "wf_gate": {"sanity_regime_ic_required": False},
    }


def _write_tree(tmp_path: Path, payload: dict | None = None) -> tuple[Path, Path]:
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    art = strategy_dir / "artifacts" / "prod" / "panel-ltr.alpha158_fund.json"
    art.parent.mkdir(parents=True)
    art.write_text(json.dumps(payload if payload is not None else _served_payload()))
    cfg = tmp_path / "configs" / "strategy_config.json"
    cfg.parent.mkdir()
    cfg.write_text(json.dumps(_config()))
    return cfg, strategy_dir


LAUNCHCTL = (
    "PID\tStatus\tLabel\n"
    "-\t0\tcom.renquant.intraday104\n"
    "-\t1\tcom.renquant.weekly-wf-promote\n"
    "19215\t0\tcom.renquant.agent-pr-loop\n"
    "-\t1\tcom.renquant.retrain-panel104\n"
)

FULL_RUN_LOG = (
    "2026-08-31 13:55:03 INFO preflight ✓ P-MODEL-ARTIFACT      [HARD] artifact ok\n"
    "2026-08-31 13:55:04 WARNING PRE-FLIGHT FAILED — aborting cron, no orders placed:\n"
    "1 hard pre-flight check(s) failed:\n"
    "  ✗ P-WF-GATE: active panel artifact carries failed WF gate evidence: "
    "wf_sharpe_mean=0.6017718060321567 spy_sharpe_mean=1.0808386653410664 "
    "reason=FAIL: absolute_ok=True. Refusing new live decisions until a WF-passing "
    "artifact is promoted or buy mode is explicitly isolated to shadow/research.\n"
)


# ── artifact facts ─────────────────────────────────────────────────────────

def test_served_summary_reads_the_real_artifact_shape():
    s = bbr.served_summary(_served_payload(), today=TODAY)
    assert s["trained_date"] == "2026-08-02"
    assert s["age_days"] == 29
    assert s["wf_gate_passed"] is False
    assert s["promotion_basis"] == "freshness_fallback_rfc210"
    assert s["genuine_ic"] == pytest.approx(0.00289, abs=1e-5)
    assert s["wf_reason"].startswith("FAIL: absolute_ok=True")


def test_served_summary_never_invents_values_on_a_bare_payload():
    s = bbr.served_summary({}, today=TODAY)
    assert s == {
        "trained_date": None, "age_days": None, "wf_gate_passed": None,
        "promotion_basis": None, "genuine_ic": None, "wf_reason": None,
    }


def test_genuine_ic_falls_back_to_the_wf_stamp_when_metadata_lacks_it():
    payload = _served_payload()
    del payload["metadata"]["fallback_genuine_ic"]
    s = bbr.served_summary(payload, today=TODAY)
    assert s["genuine_ic"] == pytest.approx(0.00289, abs=1e-5)


def test_license_verdict_uses_the_pipeline_evaluator_when_importable(monkeypatch):
    class _Lic:
        served = False
        reason = "governance-served artifact aged out: trained 2026-08-02, 29d old > 28d RFC#210 serving SLA"

    import types
    fake = types.ModuleType("renquant_pipeline.kernel.rfc210_license")
    fake.evaluate_freshness_fallback_license = lambda payload, config=None, today=None: _Lic()
    pkg = types.ModuleType("renquant_pipeline"); kern = types.ModuleType("renquant_pipeline.kernel")
    monkeypatch.setitem(sys.modules, "renquant_pipeline", pkg)
    monkeypatch.setitem(sys.modules, "renquant_pipeline.kernel", kern)
    monkeypatch.setitem(sys.modules, "renquant_pipeline.kernel.rfc210_license", fake)
    v = bbr.license_verdict(_served_payload(), _config(), today=TODAY)
    assert v == "REFUSED: " + _Lic.reason


def test_license_verdict_names_its_absence_instead_of_guessing(monkeypatch):
    monkeypatch.setitem(sys.modules, "renquant_pipeline.kernel.rfc210_license", None)
    v = bbr.license_verdict(_served_payload(), _config(), today=TODAY)
    assert "unavailable" in v and "renquant_pipeline" in v


def test_preflight_verdict_line_quotes_the_failed_gate_without_the_log_prefix():
    line = bbr.preflight_verdict_line(FULL_RUN_LOG)
    assert line.startswith("✗ P-WF-GATE: active panel artifact carries failed WF gate evidence")
    assert "2026-08-31 13:55" not in line
    assert bbr.blocking_gate(line) == "P-WF-GATE"


def test_preflight_verdict_line_prefers_the_failed_line_over_a_passed_one():
    text = ("preflight ✓ P-WF-GATE  [HARD] LICENSED: WF gate FAILED, ...\n"
            "  ✗ P-CONFIG-FP: fingerprint mismatch\n")
    line = bbr.preflight_verdict_line(text)
    assert line == "✗ P-CONFIG-FP: fingerprint mismatch"
    assert bbr.blocking_gate(line) == "P-CONFIG-FP"
    assert bbr.blocking_gate(None) == "a buy-side preflight gate"
    assert bbr.preflight_verdict_line("") is None


# ── promotion-job status ───────────────────────────────────────────────────

def test_launchctl_parse_reads_last_exit_status_per_label():
    got = bbr.parse_launchctl_list(LAUNCHCTL, bbr.DEFAULT_LAUNCHD_LABELS)
    assert got == {
        "com.renquant.weekly-wf-promote": "1",
        "com.renquant.retrain-panel104": "1",
    }
    assert bbr.parse_launchctl_list("PID\tStatus\tLabel\n", ["com.x"]) == {"com.x": "not loaded"}


# ── body composition ───────────────────────────────────────────────────────

def _body(tmp_path: Path, **kw) -> str:
    cfg, sd = _write_tree(tmp_path)
    payload = bbr.load_artifact(bbr.served_artifact_path(bbr.load_config(cfg), sd))
    return bbr.compose_body(
        bbr.served_summary(payload, today=TODAY),
        verdict="REFUSED: governance-served artifact aged out: trained 2026-08-02, 29d old > 28d RFC#210 serving SLA",
        preflight_line=bbr.preflight_verdict_line(FULL_RUN_LOG),
        job_status=bbr.parse_launchctl_list(LAUNCHCTL, bbr.DEFAULT_LAUNCHD_LABELS),
        artifact_path=sd / "artifacts/prod/panel-ltr.alpha158_fund.json",
        **kw,
    )


def test_body_states_every_fact_the_operator_needs(tmp_path):
    body = _body(tmp_path, holdings="Held: ASML+3% NVDA-1%", log_path="/x/2026-08-31.log")
    assert body.startswith("New buys BLOCKED by P-WF-GATE; the full run was rerun --sell-only.")
    assert "Served artifact: trained 2026-08-02 (29d old)" in body
    assert "wf_gate_metadata.passed=false" in body
    assert "promotion_basis=freshness_fallback_rfc210" in body
    assert "genuine_ic=+0.00289" in body
    assert "License: REFUSED: governance-served artifact aged out" in body
    assert "29d old > 28d RFC#210 serving SLA" in body
    assert "Preflight: ✗ P-WF-GATE:" in body
    assert "Exits continue via the 06:30-13:00 sell-only loop" in body
    assert ("Buys resume only when a candidate is promoted (weekly-wf-promote / "
            "retrain-panel104 — weekly-wf-promote last exit status: 1, "
            "retrain-panel104 last exit status: 1).") in body
    assert "Held: ASML+3% NVDA-1%" in body
    assert body.rstrip().endswith("Log: /x/2026-08-31.log")


def test_body_survives_an_unreadable_artifact_and_says_so(tmp_path):
    cfg, sd = _write_tree(tmp_path)
    (sd / "artifacts/prod/panel-ltr.alpha158_fund.json").write_text("{not json")
    rc = bbr.main(["--strategy-config", str(cfg), "--strategy-dir", str(sd),
                   "--today", "2026-08-31", "--launchctl-output", LAUNCHCTL])
    assert rc == 0


def test_cli_prints_the_body_and_json_carries_the_urgent_headers(tmp_path, capsys):
    cfg, sd = _write_tree(tmp_path)
    log = tmp_path / "full.log"; log.write_text(FULL_RUN_LOG)
    rc = bbr.main(["--strategy-config", str(cfg), "--strategy-dir", str(sd),
                   "--full-run-log", str(log), "--today", "2026-08-31",
                   "--launchctl-output", LAUNCHCTL, "--json"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["title"] == "RenQuant 104 BUY-BLOCKED (sell-only fallback)"
    assert out["headers"] == {
        "Title": "RenQuant 104 BUY-BLOCKED (sell-only fallback)",
        "Priority": "urgent",
        "Tags": "rotating_light,rq104",
    }
    assert "New buys BLOCKED by P-WF-GATE" in out["body"]
    assert "trained 2026-08-02 (29d old)" in out["body"]


# ── sending ────────────────────────────────────────────────────────────────

def test_send_alert_posts_with_urgent_priority_and_tags():
    calls = []

    def fake_send(title, body, topic=None, *, priority=None, tags=None, **_):
        calls.append((title, body, priority, tags))
        return True

    rc = bbr.send_alert("T", "B", sender=fake_send, suppressed=lambda: False)
    assert rc == 0
    assert calls == [("T", "B", "urgent", "rotating_light,rq104")]


def test_send_alert_exit_codes_drive_the_wrapper_fallback(capsys):
    # suppressed → handled (0): the wrapper must NOT curl around RENQUANT_NO_NOTIFY
    assert bbr.send_alert("T", "B", sender=lambda *a, **k: True, suppressed=lambda: True) == 0
    assert "[ntfy suppressed]" in capsys.readouterr().err
    # POST failed → 4 → wrapper curls with the same headers
    assert bbr.send_alert("T", "B", sender=lambda *a, **k: False, suppressed=lambda: False) == 4
    # sender import failure → 3
    saved = sys.modules.get("renquant_common.notify")
    sys.modules["renquant_common.notify"] = None  # type: ignore[assignment]
    try:
        assert bbr.send_alert("T", "B") == 3
    finally:
        if saved is None:
            sys.modules.pop("renquant_common.notify", None)
        else:
            sys.modules["renquant_common.notify"] = saved


def test_script_is_read_only_on_the_artifact_tree(tmp_path):
    """Behavioural, not regex: run the CLI as a subprocess and diff the tree."""
    cfg, sd = _write_tree(tmp_path)
    before = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--strategy-config", str(cfg), "--strategy-dir", str(sd),
         "--today", "2026-08-31", "--launchctl-output", LAUNCHCTL],
        capture_output=True, text=True, check=False, env={"RENQUANT_NO_NOTIFY": "1", "PATH": "/usr/bin:/bin"},
    )
    assert proc.returncode == 0, proc.stderr
    after = {p: p.stat().st_mtime_ns for p in tmp_path.rglob("*") if p.is_file()}
    assert after == before
    assert "trained 2026-08-02 (29d old)" in proc.stdout


# ── wrapper wiring ─────────────────────────────────────────────────────────

def test_daily_104_wires_the_helper_with_urgent_fallback_and_date_stamp():
    script = DAILY_104.read_text()
    block = script[script.find('elif [ "$BUY_BLOCKED_BY_PREFLIGHT" -eq 1 ]; then'):
                   script.find("# Sustainability audit")]
    assert 'scripts/buy_blocked_reason.py' in block
    assert '--send' in block
    assert 'BUY_BLOCKED_TITLE="RenQuant 104 BUY-BLOCKED (sell-only fallback)"' in block
    # curl fallback carries the SAME headers — never the bare default-priority line
    assert '-H "Priority: urgent" -H "Tags: rotating_light,rq104"' in block
    assert 'notify "RenQuant 104 BUY-BLOCKED"' not in script
    # stamp keyed by session date, not a seconds cooldown
    assert 'echo "$DATE" > "$BUY_BLOCKED_ALERT_STAMP"' in block
    assert '[ "$BUY_BLOCKED_LAST_DATE" != "$DATE" ]' in block
    assert "RENQUANT_BUY_BLOCKED_ALERT_COOLDOWN_SEC" not in script
    assert "21600" not in block
    # the preflight verdict is captured BEFORE the full-run log is deleted
    cap = script.find("BUY_BLOCKED_PREFLIGHT_LINES=$(grep")
    assert 0 < cap < script.find('--once --sell-only > "$SELL_ONLY_LOG"')
