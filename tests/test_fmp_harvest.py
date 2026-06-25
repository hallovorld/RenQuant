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
    assert m["error_samples"] == [{"ticker": "BBB"}]
    man = json.loads((tmp_path / "locked_291.manifest.json").read_text())
    assert man["status"] == "errors"


# ---- skip / rerun on manifest ----------------------------------------------
def test_manifest_ok_gate(tmp_path):
    p = tmp_path / "m.json"
    assert fh._manifest_ok(p) is False                 # missing
    p.write_text(json.dumps({"status": "errors"}))
    assert fh._manifest_ok(p) is False                 # errored → re-pull
    p.write_text(json.dumps({"status": "ok"}))
    assert fh._manifest_ok(p) is True                  # ok → skip


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
