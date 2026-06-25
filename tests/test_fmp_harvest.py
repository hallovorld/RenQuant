"""Tests for the FMP harvester — classification, atomic write+manifest, skip/rerun,
fail-closed exit, and bounded retry. Network is never touched: a fake `get` is injected.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "fmp_harvest", Path(__file__).resolve().parent.parent / "scripts" / "fmp_harvest.py")
fh = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fh)


# ---- classify (pure) -------------------------------------------------------
def test_classify_list_with_rows():
    assert fh.classify([{"a": 1}, {"b": 2}]) == ("with_data", [{"a": 1}, {"b": 2}])


def test_classify_empty_list_is_no_data():
    assert fh.classify([]) == ("no_data", [])


def test_classify_single_dict_is_data():
    assert fh.classify({"symbol": "AAPL", "v": 1}) == ("with_data", [{"symbol": "AAPL", "v": 1}])


def test_classify_http_and_fetch_errors():
    assert fh.classify({"_http": 402}) == ("http_error", [])
    assert fh.classify({"_err": "TimeoutError"}) == ("fetch_error", [])


def test_classify_garbage_is_no_data():
    assert fh.classify(None) == ("no_data", [])
    assert fh.classify("oops") == ("no_data", [])


# ---- harvest_endpoint: write + manifest ------------------------------------
def _fake_get(mapping, default):
    """Return a get(path,key,retries,backoff) that looks payloads up by ticker substring."""
    def get(path, key, retries, backoff):
        for sym, payload in mapping.items():
            if f"={sym}" in path or path == sym:   # matches symbol=X and name=X
                return payload
        return default
    return get


def test_harvest_endpoint_writes_parquet_and_ok_manifest(tmp_path):
    uni = ["AAA", "BBB"]
    get = _fake_get({"AAA": [{"x": 1}], "BBB": [{"x": 2}]}, [])
    m = fh.harvest_endpoint("demo", "demo?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "ok"
    assert m["with_data"] == 2 and m["http_error"] == 0
    assert m["rows"] == 2 and m["tickers"] == 2
    assert (tmp_path / "demo_291.parquet").exists()
    man = json.loads((tmp_path / "demo_291.manifest.json").read_text())
    assert man["status"] == "ok" and man["sha256"] and len(man["sha256"]) == 64
    # rows are stamped with ticker/source
    import pandas as pd
    df = pd.read_parquet(tmp_path / "demo_291.parquet")
    assert set(df["ticker"]) == {"AAA", "BBB"} and (df["source"] == "fmp_demo").all()


def test_harvest_endpoint_list_targets_iterates_names(tmp_path):
    # per_ticker as a tuple → iterate those exact {sym} values (macro indicators),
    # NOT the ticker universe.
    get = _fake_get({"GDP": [{"value": 1}], "CPI": [{"value": 2}]}, [])
    m = fh.harvest_endpoint("econ", "econ?name={sym}", ("GDP", "CPI"), ["AAA", "BBB"],
                            tmp_path, 0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "ok"
    assert m["requested"] == 2 and m["with_data"] == 2
    import pandas as pd
    df = pd.read_parquet(tmp_path / "econ_291.parquet")
    assert set(df["ticker"]) == {"GDP", "CPI"}      # stamped by indicator name, not AAA/BBB


def test_harvest_endpoint_no_data_is_ok_no_parquet(tmp_path):
    uni = ["AAA", "BBB"]
    get = _fake_get({}, [])  # every symbol returns empty list
    m = fh.harvest_endpoint("none", "none?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "ok"          # no_data does NOT fail
    assert m["no_data"] == 2 and m["rows"] == 0
    assert m["output"] is None and m["sha256"] is None
    assert not (tmp_path / "none_291.parquet").exists()


def test_harvest_endpoint_http_error_sets_errors_status(tmp_path):
    uni = ["AAA", "BBB"]
    get = _fake_get({"AAA": [{"x": 1}]}, {"_http": 402})  # BBB → 402
    m = fh.harvest_endpoint("locked", "locked?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "errors"
    assert m["http_error"] == 1
    # error sample carries the HTTP code, not just the ticker (audit record)
    assert m["error_samples"] == [{"ticker": "BBB", "http": 402}]
    man = json.loads((tmp_path / "locked_291.manifest.json").read_text())
    assert man["status"] == "errors"
    assert man["error_samples"][0]["http"] == 402


def test_harvest_endpoint_fetch_error_sample_carries_err_type(tmp_path):
    uni = ["AAA", "BBB"]
    get = _fake_get({"AAA": [{"x": 1}]}, {"_err": "TimeoutError"})  # BBB → transport error
    m = fh.harvest_endpoint("flaky", "flaky?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "errors"
    assert m["fetch_error"] == 1
    assert m["error_samples"] == [{"ticker": "BBB", "err": "TimeoutError"}]


# ---- skip / rerun on manifest ----------------------------------------------
def test_manifest_ok_gate(tmp_path):
    p = tmp_path / "m.json"
    assert fh._manifest_ok(p) is False                 # missing
    p.write_text(json.dumps({"status": "errors"}))
    assert fh._manifest_ok(p) is False                 # errored → re-pull
    p.write_text(json.dumps({"status": "ok"}))
    assert fh._manifest_ok(p) is True                  # ok → skip (pure status gate)


def _write_endpoint(tmp_path, key, tmpl, targets, get):
    """Run one endpoint offline and return (manifest_path, parquet_path)."""
    m = fh.harvest_endpoint(key, tmpl, list(targets), [], tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    return (tmp_path / f"{key}_291.manifest.json",
            tmp_path / f"{key}_291.parquet", m)


def test_skip_requires_matching_sha256(tmp_path):
    # A data endpoint: ok-manifest + parquet present + sha256 match → skip.
    get = _fake_get({"AAA": [{"x": 1}], "BBB": [{"x": 2}]}, [])
    mp, pp, _ = _write_endpoint(tmp_path, "demo", "demo?symbol={sym}", ("AAA", "BBB"), get)
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}", targets=["AAA", "BBB"],
                           parquet_path=pp) is True
    # Tamper the parquet → sha256 no longer matches → must re-pull.
    pp.write_bytes(pp.read_bytes() + b"corrupt")
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}", targets=["AAA", "BBB"],
                           parquet_path=pp) is False
    # Missing parquet → must re-pull.
    pp.unlink()
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}", targets=["AAA", "BBB"],
                           parquet_path=pp) is False


def test_skip_invalidated_by_changed_template_or_universe(tmp_path):
    get = _fake_get({"AAA": [{"x": 1}], "BBB": [{"x": 2}]}, [])
    mp, pp, _ = _write_endpoint(tmp_path, "demo", "demo?symbol={sym}", ("AAA", "BBB"), get)
    # Same template + universe → skip.
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp) is True
    # Endpoint/request-config changed (e.g. period/limit) → NOT skipped.
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}&limit=20",
                           targets=["AAA", "BBB"], parquet_path=pp) is False
    # Universe changed (CCC added) → NOT skipped.
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}",
                           targets=["AAA", "BBB", "CCC"], parquet_path=pp) is False
    # Universe changed (BBB renamed) → NOT skipped.
    assert fh._manifest_ok(mp, tmpl="demo?symbol={sym}",
                           targets=["AAA", "ZZZ"], parquet_path=pp) is False


def test_zero_data_is_valid_completion_and_skips_without_parquet(tmp_path):
    # A legitimately empty endpoint writes ok + output null + no parquet, and a
    # later run must SKIP it (manifest is the completion record) — not re-pull forever.
    get = _fake_get({}, [])  # every symbol returns empty list
    mp, pp, m = _write_endpoint(tmp_path, "none", "none?symbol={sym}", ("AAA", "BBB"), get)
    assert m["status"] == "ok" and m["output"] is None and m["rows"] == 0
    assert not pp.exists()
    assert fh._manifest_ok(mp, tmpl="none?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp) is True   # skip, no parquet
    # But a changed universe still invalidates the zero-data record.
    assert fh._manifest_ok(mp, tmpl="none?symbol={sym}",
                           targets=["AAA", "CCC"], parquet_path=pp) is False


def test_zero_data_rerun_retires_stale_parquet(tmp_path):
    # First run has data → parquet written.
    get1 = _fake_get({"AAA": [{"x": 1}]}, [])
    _, pp, m1 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"), get1)
    assert pp.exists() and m1["output"] == "ev_291.parquet"
    # Second run returns zero rows for everyone → old parquet must be RETIRED, not left
    # behind to let a future run skip on a stale parquet + output:null manifest.
    get2 = _fake_get({}, [])
    mp, pp2, m2 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"), get2)
    assert m2["status"] == "ok" and m2["output"] is None and m2["rows"] == 0
    assert not pp2.exists()                              # stale parquet gone
    assert (tmp_path / "ev_291.parquet.retired").exists()  # retired, not deleted
    # And the resulting zero-data manifest is a valid skip (no parquet required).
    assert fh._manifest_ok(mp, tmpl="ev?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp2) is True


def test_harvest_loop_skips_zero_data_endpoint_without_repull(tmp_path, monkeypatch):
    # End-to-end through harvest(): a zero-data endpoint completes ok once, then a
    # second harvest() must SKIP it (no second pull), proving no "re-pull forever".
    monkeypatch.setattr(fh, "_universe", lambda repo: ["AAA", "BBB"])
    pulls = {"n": 0}

    def counting_get(path, key, retries, backoff):
        pulls["n"] += 1
        return []   # always empty → no_data

    monkeypatch.setattr(fh, "_get", counting_get)
    rc1 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc1 == 0
    after_first = pulls["n"]
    assert after_first >= 1                              # it pulled once
    rc2 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc2 == 0
    assert pulls["n"] == after_first                     # second run did NOT re-pull


# ---- bounded retry ---------------------------------------------------------
def test_get_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'[{"ok": 1}]'

    def fake_urlopen(url, timeout=30):
        calls["n"] += 1
        if calls["n"] < 3:
            import urllib.error
            raise urllib.error.HTTPError(url, 503, "busy", {}, None)
        return _Resp()

    monkeypatch.setattr(fh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fh.time, "sleep", lambda *_: None)
    out = fh._get("x?symbol=AAA", "k", retries=3, backoff=0.0)
    assert out == [{"ok": 1}] and calls["n"] == 3


def test_get_non_retryable_returns_immediately(monkeypatch):
    calls = {"n": 0}

    def fake_urlopen(url, timeout=30):
        calls["n"] += 1
        import urllib.error
        raise urllib.error.HTTPError(url, 404, "nope", {}, None)

    monkeypatch.setattr(fh.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(fh.time, "sleep", lambda *_: None)
    out = fh._get("x?symbol=AAA", "k", retries=3, backoff=0.0)
    assert out == {"_http": 404} and calls["n"] == 1   # 404 not retried
