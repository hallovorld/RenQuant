"""Snapshot freshness backstop wired into restamp_prod_fingerprint.py (M9/A6
round 4, Codex PR #432 review): a re-stamp that mutates the active production
artifact's metadata in place must NOT be able to report overall success while
leaving doc/arch/strategy-104-snapshot.md stale — the exact operational gap
weekly_wf_promote.sh/manual_promote.sh/promote_pin.py also close.

This test proves the WIRING (main() -> check_snapshot_freshness -> exit code),
not check_snapshot_freshness's own diff/regenerate logic, which
tests/test_promote_pin.py already covers end-to-end (real, non-mocked).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import restamp_prod_fingerprint as rpf  # noqa: E402


def _watchlist_config(sector_map: dict, sector_etf_map: dict) -> dict:
    return {
        "watchlist": ["AAA"],
        "benchmark": "SPY",
        "sector_map": sector_map,
        "sector_etf_map": sector_etf_map,
        "panel_ltr": {
            "lookahead_days": 10,
            "xgb_params": {"objective": "rank:pairwise"},
            "asset_embeddings": {"enabled": False},
            "training_resolution": "daily",
            "hourly": {"enabled": False},
            "minute": {"enabled": False},
        },
        "ranking": {"panel_scoring": {"artifact_path": "artifacts/prod/primary.json"}},
    }


@pytest.fixture()
def sector_only_diff_repo(tmp_path, monkeypatch):
    """A fixture repo where the artifact's stored fingerprint_fields differ
    from live config ONLY in sector_map/sector_etf_map — the exact
    precondition restamp_prod_fingerprint.py requires to proceed (a
    non-sector diff would REFUSE with rc=2, unrelated to this backstop)."""
    strategy_dir = tmp_path / "backtesting" / "renquant_104"
    (strategy_dir / "artifacts" / "prod").mkdir(parents=True)

    cfg = _watchlist_config(sector_map={"AAA": "Tech"}, sector_etf_map={"Tech": "XLK"})
    (strategy_dir / "strategy_config.json").write_text(json.dumps(cfg))

    live_fields = rpf._model_relevant_fields(cfg)
    stale_fields = dict(live_fields)
    stale_fields["sector_map"] = {}  # the only diff — legacy pre-sector-map artifact
    stale_fields["sector_etf_map"] = {}
    artifact = {
        "config_fingerprint_fields": stale_fields,
        "config_fingerprint": "sha256:stale0000000000",
    }
    art_path = strategy_dir / "artifacts" / "prod" / "primary.json"
    art_path.write_text(json.dumps(artifact))

    monkeypatch.setattr(rpf, "REPO", tmp_path)
    monkeypatch.setattr(rpf, "STRATEGY_DIR", strategy_dir)
    monkeypatch.setattr(sys, "argv", ["restamp_prod_fingerprint.py"])
    return tmp_path, art_path


def test_restamp_succeeds_but_backstop_fails_the_run_when_snapshot_stale(
    sector_only_diff_repo, monkeypatch,
):
    tmp_path, art_path = sector_only_diff_repo
    calls = []

    def fake_check(python, repo=None):
        calls.append((python, repo))
        return False, "ACTION REQUIRED: fake staleness for the test"

    # check_snapshot_freshness is imported from promote_pin INSIDE main() at
    # call time — patch the real promote_pin module so that local import
    # picks up the fake.
    import promote_pin
    monkeypatch.setattr(promote_pin, "check_snapshot_freshness", fake_check)

    rc = rpf.main()

    assert rc == 1, "a stale snapshot must fail the run even though the re-stamp itself succeeded"
    assert len(calls) == 1
    assert calls[0][1] == tmp_path, "backstop must check against THIS run's repo, not the real one"

    # The re-stamp itself was NOT reverted for a stale-snapshot finding alone.
    written = json.loads(art_path.read_text())
    assert written["config_fingerprint_fields"]["sector_map"] == {"AAA": "Tech"}, (
        "the actual re-stamp must still be applied — only the overall exit "
        "code reflects the stale snapshot, per the no-auto-revert contract"
    )


def test_restamp_reports_success_when_snapshot_is_fresh(sector_only_diff_repo, monkeypatch):
    _tmp_path, art_path = sector_only_diff_repo
    calls = []

    def fake_check(python, repo=None):
        calls.append(1)
        return True, "strategy-104 snapshot is fresh"

    import promote_pin
    monkeypatch.setattr(promote_pin, "check_snapshot_freshness", fake_check)

    rc = rpf.main()

    assert rc == 0
    assert len(calls) == 1
    written = json.loads(art_path.read_text())
    assert written["config_fingerprint_fields"]["sector_map"] == {"AAA": "Tech"}


def test_restamp_dry_run_never_reaches_the_snapshot_backstop(sector_only_diff_repo, monkeypatch):
    """--dry-run returns before any write happens; the snapshot backstop
    (which only matters once production state has actually changed) must
    not fire in that path."""
    _tmp_path, _art_path = sector_only_diff_repo
    monkeypatch.setattr(sys, "argv", ["restamp_prod_fingerprint.py", "--dry-run"])
    calls = []

    import promote_pin
    monkeypatch.setattr(
        promote_pin, "check_snapshot_freshness",
        lambda *a, **k: calls.append(1) or (False, "should not run"),
    )

    rc = rpf.main()
    assert rc == 0
    assert not calls
