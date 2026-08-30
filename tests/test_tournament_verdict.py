"""The weekly tournament verdict names its rejections (2026-08-30).

Incident: ``RenQuant 104 TOURNAMENT-RETRAIN ✓`` ("CERTIFIED") fired 1 s after
``TOURNAMENT ACCEPTANCE WARN: rejected 2 per-ticker candidate(s)`` (APP, SPY).
Certification measures coverage / freshness / exit code — not acceptance — so
both messages were true, and the operator read the ✓ as "all good".

Contract pinned here:
- the count comes from a run-bound RECEIPT written by train_104.py, not a log
  grep, and never defaults to 0;
- 0 rejections → ✓; N > 0 → ⚠ and "CERTIFIED WITH N REJECTIONS (A, B)";
- missing / stale receipt → ⚠ and UNKNOWN, with the reason;
- the shell exports the receipt env before launch and composes the final
  ntfy from the CLI in the CERTIFIED branch (source-level, like the other
  operator-script contracts in this repo).
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
MODULE = REPO_ROOT / "scripts" / "tournament_verdict.py"
SHELL = REPO_ROOT / "scripts" / "weekly_tournament_retrain.sh"
TRAIN = REPO_ROOT / "scripts" / "train_104.py"

#: This file asserts on the composed ntfy title/body the operator reads.
pytestmark = pytest.mark.notification_contract


@pytest.fixture(scope="module")
def tv():
    spec = importlib.util.spec_from_file_location("tournament_verdict_for_test", MODULE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _compose(tv, receipt: Path, run_id="RUN-1"):
    return tv.compose_tournament_verdict(
        receipt_path=receipt, expected_run_id=run_id,
        marker="/models/.last_tournament_retrain.json", log="/logs/x.log",
    )


class TestComposition:
    def test_zero_rejections_is_a_check_mark_and_says_zero(self, tv, tmp_path):
        receipt = tv.write_rejection_receipt(
            tmp_path / "r.json", run_id="RUN-1", trigger="weekly_tournament_cadence", rejected={},
        )
        title, body = _compose(tv, receipt)
        assert title == "RenQuant 104 TOURNAMENT-RETRAIN ✓"
        assert "CERTIFIED, 0 rejections" in body
        assert "WITH" not in body and "UNKNOWN" not in body
        assert "/models/.last_tournament_retrain.json" in body and "Log: /logs/x.log" in body

    def test_two_rejections_is_a_warning_naming_both(self, tv, tmp_path):
        """The 2026-08-30 shape: APP and SPY rejected, run still certified."""
        receipt = tv.write_rejection_receipt(
            tmp_path / "r.json", run_id="RUN-1", trigger="weekly_tournament_cadence",
            rejected={"SPY": "val_ic_below_floor", "APP": "insufficient_history"},
        )
        title, body = _compose(tv, receipt)
        assert title == "RenQuant 104 TOURNAMENT-RETRAIN ⚠"
        assert "CERTIFIED WITH 2 REJECTIONS (APP, SPY)" in body
        assert "previous models kept" in body
        assert "\n" not in title and "\n" not in body

    def test_one_rejection_is_singular(self, tv, tmp_path):
        receipt = tv.write_rejection_receipt(
            tmp_path / "r.json", run_id="RUN-1", trigger="t", rejected={"APP": "x"},
        )
        title, body = _compose(tv, receipt)
        assert title.endswith("⚠")
        assert "CERTIFIED WITH 1 REJECTION (APP)" in body

    def test_many_rejections_list_ten_and_count_the_rest(self, tv, tmp_path):
        rejected = {f"T{i:02d}": "r" for i in range(13)}
        receipt = tv.write_rejection_receipt(tmp_path / "r.json", run_id="RUN-1", trigger="t", rejected=rejected)
        _, body = _compose(tv, receipt)
        assert "CERTIFIED WITH 13 REJECTIONS (" in body
        assert "T09, +3 more)" in body
        assert "T10" not in body

    def test_missing_receipt_is_unknown_not_zero(self, tv, tmp_path):
        title, body = _compose(tv, tmp_path / "absent.json")
        assert title == "RenQuant 104 TOURNAMENT-RETRAIN ⚠"
        assert "UNKNOWN" in body and "rejection receipt MISSING" in body
        assert "0 rejections" not in body

    def test_receipt_from_another_run_is_stale_not_trusted(self, tv, tmp_path):
        receipt = tv.write_rejection_receipt(tmp_path / "r.json", run_id="RUN-OLD", trigger="t", rejected={})
        title, body = _compose(tv, receipt, run_id="RUN-1")
        assert title.endswith("⚠")
        assert "UNKNOWN" in body and "STALE" in body and "RUN-OLD" in body and "RUN-1" in body

    def test_malformed_receipt_is_reported(self, tv, tmp_path):
        bad = tmp_path / "r.json"
        bad.write_text('{"rejected": "not-a-dict"}')
        title, body = _compose(tv, bad)
        assert title.endswith("⚠") and "MALFORMED" in body
        bad.write_text("{not json")
        title, body = _compose(tv, bad)
        assert title.endswith("⚠") and "UNREADABLE" in body

    def test_no_expected_run_id_accepts_any_receipt(self, tv, tmp_path):
        """Ad-hoc invocation without --run-id: the receipt is still read."""
        receipt = tv.write_rejection_receipt(tmp_path / "r.json", run_id="whatever", trigger="t", rejected={})
        title, _ = _compose(tv, receipt, run_id=None)
        assert title.endswith("✓")


class TestReceipt:
    def test_receipt_shape_and_atomic_write(self, tv, tmp_path):
        path = tv.write_rejection_receipt(
            tmp_path / "deep" / "r.json", run_id="RUN-1", trigger="weekly_tournament_cadence",
            rejected={"SPY": "b", "APP": "a"}, train_run_id="abcd1234",
        )
        raw = json.loads(path.read_text())
        assert raw["schema"] == 1
        assert raw["run_id"] == "RUN-1" and raw["train_run_id"] == "abcd1234"
        assert raw["n_rejected"] == 2 and list(raw["rejected"]) == ["APP", "SPY"]
        assert raw["written_at"].endswith("+00:00")
        assert not list(path.parent.glob(".*.tmp-*")), "tmp file must be renamed away"


class TestCLI:
    def test_prints_title_then_body(self, tv, tmp_path):
        receipt = tv.write_rejection_receipt(
            tmp_path / "r.json", run_id="RUN-1", trigger="t", rejected={"APP": "a", "SPY": "b"},
        )
        out = subprocess.run(
            [sys.executable, str(MODULE), "--receipt", str(receipt), "--run-id", "RUN-1",
             "--marker", "M", "--log", "L"],
            capture_output=True, text=True, check=True,
        ).stdout.splitlines()
        assert out[0] == "RenQuant 104 TOURNAMENT-RETRAIN ⚠"
        assert "CERTIFIED WITH 2 REJECTIONS (APP, SPY)" in out[1]
        assert len(out) == 2


class TestWiring:
    """Source-level: the producer and the consumer are actually connected."""

    def test_shell_exports_receipt_env_before_launch_and_binds_run_id(self):
        src = SHELL.read_text()
        i_export = src.find('export RENQUANT_TOURNAMENT_REJECTIONS_OUT="$REJECTIONS_FILE"')
        i_runid = src.find('export RENQUANT_TOURNAMENT_RUN_ID="$RUN_ID"')
        i_launch = src.find("scripts/train_104.py --skip-panel --force --trigger")
        assert 0 < i_export < i_launch and 0 < i_runid < i_launch

    def test_shell_certified_branch_uses_the_composer_not_a_literal(self):
        src = SHELL.read_text()
        i_pass = src.find("completion CERTIFIED, marker stamped")
        i_cli = src.find("scripts/tournament_verdict.py", i_pass)
        i_notify = src.find('notify "$VERDICT_TITLE" "$VERDICT_BODY"', i_pass)
        assert 0 < i_pass < i_cli < i_notify
        assert '--run-id "$RUN_ID"' in src[i_cli:i_notify]
        assert 'notify "RenQuant 104 TOURNAMENT-RETRAIN ✓"' not in src, (
            "the ✓ literal must not be hardcoded in the shell any more"
        )

    def test_train_104_writes_the_receipt_after_rejections_are_known(self):
        src = TRAIN.read_text()
        i_rej = src.find('baseline_rejected = dict(getattr(ctx, "baseline_rejected", {}) or {})')
        i_write = src.find("_write_tournament_rejection_receipt(\n        baseline_rejected", i_rej)
        assert 0 < i_rej < i_write
        assert 'RENQUANT_TOURNAMENT_REJECTIONS_OUT' in src
        assert 'tournament_ran=not args.skip_baseline' in src

    def test_train_104_helper_writes_a_receipt_the_composer_reads(self, tv, tmp_path, monkeypatch):
        """Exercise the real helper from train_104.py (loaded WITHOUT running
        main) end-to-end into the composer."""
        src = TRAIN.read_text()
        start = src.find("def _write_tournament_rejection_receipt(")
        end = src.find("\ndef main() -> None:")
        assert 0 < start < end
        ns: dict = {}
        exec("import os as _os\nimport logging\nfrom pathlib import Path\n"
             "log = logging.getLogger('t')\n__file__ = %r\n" % str(TRAIN) + src[start:end], ns)
        helper = ns["_write_tournament_rejection_receipt"]
        out = tmp_path / "r.json"
        monkeypatch.setenv("RENQUANT_TOURNAMENT_REJECTIONS_OUT", str(out))
        monkeypatch.setenv("RENQUANT_TOURNAMENT_RUN_ID", "RUN-1")
        # --skip-baseline: the tournament did not run → no receipt (not a "0").
        assert helper({}, trigger="t", train_run_id="x", tournament_ran=False) is None
        assert not out.exists()
        assert helper({"APP": "a", "SPY": "b"}, trigger="t", train_run_id="x", tournament_ran=True) == out
        title, body = _compose(tv, out)
        assert title.endswith("⚠") and "CERTIFIED WITH 2 REJECTIONS (APP, SPY)" in body
        # Unset env (ad-hoc run): no-op.
        monkeypatch.delenv("RENQUANT_TOURNAMENT_REJECTIONS_OUT")
        assert helper({}, trigger="t", train_run_id="x", tournament_ran=True) is None
