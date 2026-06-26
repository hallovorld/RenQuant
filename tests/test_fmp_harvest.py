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


# ---- classify: app-level error bodies (HTTP 200 + error payload) ------------
def test_classify_error_message_dict_is_app_error():
    # FMP's canonical entitlement/schema error shape, returned with HTTP 200.
    payload = {"Error Message": "This endpoint is not available under your plan."}
    assert fh.classify(payload) == ("app_error", [payload])


def test_classify_error_and_message_dicts_are_app_error():
    # `{"error": ...}` (explicit error key) and a BARE `{"message": ...}`.
    assert fh.classify({"error": "Invalid API KEY."}) == ("app_error", [{"error": "Invalid API KEY."}])
    assert fh.classify({"message": "Limit Reach."}) == ("app_error", [{"message": "Limit Reach."}])


def test_classify_data_row_with_message_field_is_data():
    # A legitimate data row that merely CONTAINS a 'message' field (alongside real
    # columns) must NOT be misclassified as an error.
    row = {"symbol": "AAPL", "date": "2026-01-01", "value": 1.0, "message": "ok"}
    assert fh.classify(row) == ("with_data", [row])


def test_classify_data_dict_with_error_substring_key_is_data():
    # A key that merely contains 'error' as a substring (e.g. 'errorRate') is NOT an
    # error-signal key — a real metric row stays data.
    row = {"symbol": "AAPL", "errorRate": 0.02}
    assert fh.classify(row) == ("with_data", [row])


def test_classify_list_of_only_error_dicts_is_app_error():
    # A list whose every element is an app-error (e.g. [{"Error Message": ...}]).
    payload = [{"Error Message": "Restricted Endpoint"}]
    assert fh.classify(payload) == ("app_error", payload)


def test_classify_list_with_real_rows_passes_despite_error_dict():
    # A list with a real data row passes as data even if an error-ish dict is mixed in
    # (any real row → the response carried data).
    payload = [{"symbol": "AAPL", "v": 1}, {"Error Message": "partial"}]
    status, rows = fh.classify(payload)
    assert status == "with_data" and rows == payload


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


def test_harvest_endpoint_partial_no_data_is_ok(tmp_path):
    # Per-symbol no_data MIXED with real data does NOT fail — only ALL-target zero
    # data is suspicious. AAA has data, BBB is empty → status ok, BBB just absent.
    uni = ["AAA", "BBB"]
    get = _fake_get({"AAA": [{"x": 1}]}, [])  # BBB → empty list (no_data)
    m = fh.harvest_endpoint("mixed", "mixed?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "ok"
    assert m["with_data"] == 1 and m["no_data"] == 1 and m["rows"] == 1
    assert m["output"] == "mixed_291.parquet"
    assert (tmp_path / "mixed_291.parquet").exists()


def test_harvest_endpoint_all_zero_data_default_fails_closed(tmp_path):
    # DEFAULT (allow_zero_data=False): every target empty → NOT a success.
    # An entitlement change / vendor-schema failure / wrong param / outage must
    # fail closed, not be permanently accepted as ok during the paid window.
    uni = ["AAA", "BBB"]
    get = _fake_get({}, [])  # every symbol returns empty list
    m = fh.harvest_endpoint("none", "none?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "zero_data_unexpected"   # fails closed
    assert m["allow_zero_data"] is False
    assert m["no_data"] == 2 and m["rows"] == 0 and m["with_data"] == 0
    assert m["output"] is None and m["sha256"] is None
    assert not (tmp_path / "none_291.parquet").exists()
    # NOT a skippable state: the loop must re-pull it next run.
    assert fh._manifest_ok(tmp_path / "none_291.manifest.json",
                           tmpl="none?symbol={sym}", targets=["AAA", "BBB"]) is False


def test_harvest_endpoint_all_zero_data_allowed_is_ok(tmp_path):
    # EXPLICITLY allowed empty endpoint → valid zero-data completion (status ok,
    # full invariant satisfied), and a later run SKIPS it (no re-pull forever).
    uni = ["AAA", "BBB"]
    get = _fake_get({}, [])  # every symbol returns empty list
    m = fh.harvest_endpoint("ok_empty", "ok_empty?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get,
                            allow_zero_data=True)
    assert m["status"] == "ok"
    assert m["allow_zero_data"] is True
    assert m["no_data"] == 2 and m["requested"] == 2 and m["rows"] == 0
    assert m["with_data"] == 0 and m["tickers"] == 0
    assert m["output"] is None and m["sha256"] is None
    assert m["manifest_version"] == fh.MANIFEST_VERSION
    assert not (tmp_path / "ok_empty_291.parquet").exists()
    assert fh._manifest_ok(tmp_path / "ok_empty_291.manifest.json",
                           tmpl="ok_empty?symbol={sym}", targets=["AAA", "BBB"]) is True


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


def test_harvest_endpoint_app_error_body_fails_closed(tmp_path):
    # An HTTP-200 error body ({"Error Message": ...}) on every target must FAIL CLOSED:
    # status "errors" (NOT ok), no parquet written, and the sample carries the message
    # so the audit trail says *why*. Accepting it as a data row would write an ok-skip
    # manifest and burn the paid window silently.
    uni = ["AAA", "BBB"]
    get = _fake_get(
        {"AAA": {"Error Message": "Restricted under your plan."}},
        {"Error Message": "Restricted under your plan."})  # both targets → error body
    m = fh.harvest_endpoint("locked200", "locked200?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "errors"
    assert m["app_error"] == 2 and m["with_data"] == 0 and m["rows"] == 0
    assert m["output"] is None and m["sha256"] is None
    assert not (tmp_path / "locked200_291.parquet").exists()
    assert m["error_samples"][0] == {
        "ticker": "AAA", "key": "Error Message", "message": "Restricted under your plan."}
    # Not a skippable state — must re-pull next run.
    assert fh._manifest_ok(tmp_path / "locked200_291.manifest.json",
                           tmpl="locked200?symbol={sym}", targets=["AAA", "BBB"]) is False


def test_harvest_endpoint_list_of_error_dicts_fails_closed(tmp_path):
    # A LIST containing only error dicts (e.g. [{"error": ...}]) is a wholesale error
    # body → app_error → fail closed, just like a bare error dict.
    uni = ["AAA", "BBB"]
    get = _fake_get({}, [{"error": "Invalid API KEY."}])  # every target → list-of-error
    m = fh.harvest_endpoint("listerr", "listerr?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "errors"
    assert m["app_error"] == 2 and m["rows"] == 0
    assert not (tmp_path / "listerr_291.parquet").exists()
    assert m["error_samples"][0] == {"ticker": "AAA", "key": "error", "message": "Invalid API KEY."}


def test_harvest_endpoint_real_data_dict_still_passes(tmp_path):
    # A legitimate single-dict endpoint row (price-target-consensus shape) that has no
    # error key must still be written as data — the app-error guard must not over-trip.
    uni = ["AAA", "BBB"]
    get = _fake_get({"AAA": {"symbol": "AAA", "targetConsensus": 210.0},
                     "BBB": {"symbol": "BBB", "targetConsensus": 95.0}}, [])
    m = fh.harvest_endpoint("consensus", "consensus?symbol={sym}", True, uni, tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get)
    assert m["status"] == "ok"
    assert m["with_data"] == 2 and m["app_error"] == 0 and m["rows"] == 2
    assert (tmp_path / "consensus_291.parquet").exists()


def test_harvest_loop_app_error_endpoint_fails_closed_and_repulls(tmp_path, monkeypatch):
    # End-to-end through harvest(): every target returns an HTTP-200 error body on a
    # non-allowed endpoint → the run FAILS CLOSED (rc 1) and the next run RE-PULLS
    # (never canonicalizes the error body as a permanent ok completion).
    monkeypatch.setattr(fh, "_universe", lambda repo: ["AAA", "BBB"])
    pulls = {"n": 0}

    def err_get(path, key, retries, backoff):
        pulls["n"] += 1
        return {"Error Message": "Restricted Endpoint"}   # HTTP-200 error body

    monkeypatch.setattr(fh, "_get", err_get)
    rc1 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc1 == 1                                      # fails closed
    after_first = pulls["n"]
    assert after_first >= 1
    rc2 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc2 == 1                                      # still fails closed
    assert pulls["n"] > after_first                      # second run DID re-pull


# ---- skip / rerun on manifest ----------------------------------------------
def test_manifest_ok_gate(tmp_path):
    p = tmp_path / "m.json"
    assert fh._manifest_ok(p) is False                 # missing
    p.write_text(json.dumps({"status": "errors"}))
    assert fh._manifest_ok(p) is False                 # errored → re-pull
    p.write_text(json.dumps({"status": "ok"}))
    assert fh._manifest_ok(p) is True                  # ok → skip (pure status gate)


def _write_endpoint(tmp_path, key, tmpl, targets, get, allow_zero_data=False):
    """Run one endpoint offline and return (manifest_path, parquet_path, manifest)."""
    m = fh.harvest_endpoint(key, tmpl, list(targets), [], tmp_path,
                            0.0, "k", "2026-06-25", 0, 0.0, get=get,
                            allow_zero_data=allow_zero_data)
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
    # An EXPLICITLY-ALLOWED empty endpoint writes ok + output null + no parquet, and a
    # later run must SKIP it (manifest is the completion record) — not re-pull forever.
    get = _fake_get({}, [])  # every symbol returns empty list
    mp, pp, m = _write_endpoint(tmp_path, "none", "none?symbol={sym}", ("AAA", "BBB"),
                                get, allow_zero_data=True)
    assert m["status"] == "ok" and m["output"] is None and m["rows"] == 0
    assert not pp.exists()
    assert fh._manifest_ok(mp, tmpl="none?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp) is True   # skip, no parquet
    # But a changed universe still invalidates the zero-data record.
    assert fh._manifest_ok(mp, tmpl="none?symbol={sym}",
                           targets=["AAA", "CCC"], parquet_path=pp) is False


def test_zero_data_manifest_invariant_rejects_inconsistent_records(tmp_path):
    # A valid allowed-empty manifest must satisfy the FULL invariant. Mutating any
    # field that breaks internal consistency makes it a NON-skippable (re-pull) state,
    # so a corrupt/forged/stale "successful empty" record is never silently trusted.
    get = _fake_get({}, [])
    mp, pp, m = _write_endpoint(tmp_path, "inv", "inv?symbol={sym}", ("AAA", "BBB"),
                                get, allow_zero_data=True)
    base = json.loads(mp.read_text())
    assert fh._manifest_ok(mp, tmpl="inv?symbol={sym}", targets=["AAA", "BBB"]) is True

    def _check_rejected(**override):
        man = dict(base, **override)
        mp.write_text(json.dumps(man))
        assert fh._manifest_ok(mp, tmpl="inv?symbol={sym}", targets=["AAA", "BBB"]) is False

    _check_rejected(allow_zero_data=False)        # not actually allowed
    _check_rejected(manifest_version=999)         # wrong/old state-machine version
    _check_rejected(with_data=1)                  # claims data but output null
    _check_rejected(rows=5)                        # nonzero rows with null output
    _check_rejected(tickers=3)                     # nonzero tickers
    _check_rejected(http_error=1)                  # an error slipped in
    _check_rejected(fetch_error=1)
    _check_rejected(no_data=1)                     # no_data != requested (unaccounted)
    _check_rejected(output="inv_291.parquet")     # output set but no sha → inconsistent
    _check_rejected(sha256="deadbeef")             # sha set with null output
    _check_rejected(requested=0, no_data=0)        # nothing requested → not a completion


def test_unexpected_zero_data_preserves_last_verified_parquet(tmp_path):
    # First run has data → verified parquet written.
    get1 = _fake_get({"AAA": [{"x": 1}]}, [])
    _, pp, m1 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"), get1)
    assert pp.exists() and m1["output"] == "ev_291.parquet"
    sha_first = m1["sha256"]
    # Second run returns zero rows for everyone on a NON-allowed endpoint → suspicious.
    # The last verified parquet must be PRESERVED (NOT retired), and the run fails closed
    # so the next invocation re-pulls instead of canonicalizing a suspicious empty state.
    get2 = _fake_get({}, [])
    mp, pp2, m2 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"), get2)
    assert m2["status"] == "zero_data_unexpected"
    assert pp2.exists()                                       # verified parquet preserved
    assert not (tmp_path / "ev_291.parquet.retired").exists() # NOT retired
    assert fh._sha256(pp2) == sha_first                       # untouched
    # The unexpected-empty manifest is NOT a skippable state → re-pull next run.
    assert fh._manifest_ok(mp, tmpl="ev?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp2) is False


def test_allowed_zero_data_rerun_retires_stale_parquet(tmp_path):
    # First run has data → parquet written.
    get1 = _fake_get({"AAA": [{"x": 1}]}, [])
    _, pp, m1 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"),
                                get1, allow_zero_data=True)
    assert pp.exists() and m1["output"] == "ev_291.parquet"
    # Second run returns zero rows for everyone on an ALLOWED-empty endpoint → the
    # replacement state is accepted, so the old parquet is RETIRED (not deleted), and
    # the zero-data record becomes a valid skip.
    get2 = _fake_get({}, [])
    mp, pp2, m2 = _write_endpoint(tmp_path, "ev", "ev?symbol={sym}", ("AAA", "BBB"),
                                  get2, allow_zero_data=True)
    assert m2["status"] == "ok" and m2["output"] is None and m2["rows"] == 0
    assert not pp2.exists()                              # stale parquet gone
    assert (tmp_path / "ev_291.parquet.retired").exists()  # retired, not deleted
    assert fh._manifest_ok(mp, tmpl="ev?symbol={sym}",
                           targets=["AAA", "BBB"], parquet_path=pp2) is True


def test_harvest_loop_systemic_empty_fails_closed_and_repulls(tmp_path, monkeypatch):
    # End-to-end through harvest(): with EVERY target returning empty on a non-allowed
    # endpoint (systemic empty), the run must FAIL CLOSED (non-zero rc) and the next
    # run must RE-PULL — never accept the suspicious empty as a permanent completion.
    monkeypatch.setattr(fh, "_universe", lambda repo: ["AAA", "BBB"])
    pulls = {"n": 0}

    def counting_get(path, key, retries, backoff):
        pulls["n"] += 1
        return []   # always empty → no_data across the whole universe

    monkeypatch.setattr(fh, "_get", counting_get)
    # "treasury" matches the treasury_rates endpoint (allow_zero_data=False).
    rc1 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc1 == 1                                      # fails closed
    after_first = pulls["n"]
    assert after_first >= 1                              # it pulled once
    rc2 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc2 == 1                                      # still fails closed
    assert pulls["n"] > after_first                      # second run DID re-pull (not skipped)


def test_harvest_loop_skips_verified_data_endpoint_without_repull(tmp_path, monkeypatch):
    # End-to-end: a data endpoint completes ok once, then a second harvest() SKIPS it
    # (no second pull) on the verified parquet sha256 — proving valid completions don't
    # re-pull while suspicious empties (above) do.
    monkeypatch.setattr(fh, "_universe", lambda repo: ["AAA", "BBB"])
    pulls = {"n": 0}

    def counting_get(path, key, retries, backoff):
        pulls["n"] += 1
        return [{"v": 1}]   # always returns data

    monkeypatch.setattr(fh, "_get", counting_get)
    rc1 = fh.harvest(tmp_path, 0.0, "treasury", "k", Path("/repo"), 0, 0.0, False)
    assert rc1 == 0
    after_first = pulls["n"]
    assert after_first >= 1
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
