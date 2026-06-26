#!/usr/bin/env python3
"""One-time FMP harvest — store everything for the universe while the paid month
is active (see doc/research/2026-06-25-fmp-harvest-plan.md).

Auditable + fail-closed + resumable:
- Every endpoint writes a parquet AND a sidecar manifest (`<key>_291.manifest.json`)
  recording requested/with_data/no_data/http_error/fetch_error/app_error counts, error
  samples (HTTP code / error type / app-error key+message), endpoint URL template, universe
  hash, started/finished,
  row+ticker counts, and the output sha256.
- Resumable on the MANIFEST, content/config aware: an endpoint is skipped only when its
  manifest status == "ok" AND its `manifest_version` matches AND its recorded `path_template`
  matches the current endpoint AND its recorded `universe_hash` matches the current target
  list AND either (a) the parquet exists and its sha256 matches the recorded sha256 (data
  completion) or (b) the manifest is a valid ALLOWED ZERO-DATA record (the full zero-data
  invariant holds: with_data/errors/tickers/rows all 0, output/sha256 null) — which skips
  WITHOUT needing a parquet. A partial/errored run, a tampered/stale/missing parquet, or a
  changed endpoint/universe all re-pull on the next invocation.
- FAIL-CLOSED on all-target zero data by DEFAULT: an endpoint where every target returns an
  empty payload (with_data == 0, no errors) is `zero_data_unexpected` — it counts toward the
  non-zero exit and is NOT accepted as a completion, because an entitlement change, a
  vendor/schema failure returning empty lists, a wrong endpoint param, or a systemic outage
  must not silently burn the only paid collection window. Only endpoints explicitly marked
  `allow_zero_data=True` may record a valid empty completion (none currently are).
- Preserve the last verified artifact: a suspicious refresh (zero_data_unexpected, or any
  http/fetch/app error) NEVER replaces or retires the existing good parquet/manifest — the
  prior verified state stands until a CLEAN full replacement passes its acceptance rule. A
  PARTIAL-error pull (some targets data, some errored) does NOT overwrite the canonical
  parquet: its partial rows are quarantined to `<key>_291.parquet.staging` and the manifest
  records `output:null, status:"errors"` (so a later run re-pulls, never skips on the partial).
  Only an ALLOWED zero-data completion atomically RETIRES an older parquet (-> `.parquet.retired`).
- Writes are atomic (tmp file -> os.replace) so an interruption never leaves a half file
  that a later run would trust.
- Fail-closed: any http_error/fetch_error/app_error OR unexpected all-target zero data makes
  the run exit non-zero unless --allow-errors. `app_error` is an FMP HTTP-200 error body
  (e.g. {"Error Message": ...} entitlement/plan/schema message) — caught BEFORE the
  dict-as-data path so it never writes an ok-skip manifest. Per-symbol `no_data` (e.g. an ETF
  with no fundamentals, a name with no dividends) mixed with real data is EXPECTED and does
  not fail the run.
- Bounded retry/backoff on 429/5xx/timeouts.

Per-ticker endpoints are pulled for the alpha158 training universe (291); treasury is
universe-agnostic. Throttle ~0.2s (≈300/min Starter cap). Key from FMP_API_KEY (.env,
never committed).

  fmp_harvest.py --out data/fmp_harvest --rate 0.2 [--only analyst] [--allow-errors]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://financialmodelingprep.com/stable"
RETRY_CODES = {429, 500, 502, 503, 504}

# Schema / state-machine version stamped into every manifest. A zero-data
# completion is only honored as a valid skip when this matches (so a manifest
# written by an older state machine is re-pulled, never silently trusted).
MANIFEST_VERSION = 2

# Macro indicators verified available on Starter 2026-06-25 (each a single call;
# rows already carry a `name` field, so we stamp ticker=<indicator>).
ECON_NAMES = ("GDP", "realGDP", "CPI", "inflationRate", "unemploymentRate",
              "federalFunds", "retailSales", "consumerSentiment")

# (endpoint_key, path_template, per_ticker?, allow_zero_data) — path uses {sym}
# for per-ticker. `allow_zero_data`: when True an all-target-empty pull is a VALID
# completion (skip on rerun); when False (the DEFAULT) an all-target-empty pull is
# treated as `zero_data_unexpected` and FAILS CLOSED — an entitlement change, a
# vendor/schema failure returning empty lists, a wrong endpoint param, or a systemic
# outage must NOT be silently accepted during the one-month paid window.
# NOTE: institutional-ownership is intentionally omitted — verified 2026-06-25 to be
# plan-locked above Starter (402 "Restricted Endpoint"); keeping it would always fail-close.
# NOTE: every shipped endpoint is allow_zero_data=False — none is known to legitimately
# return all-empty across the whole 291 universe (each has broad real coverage). If a
# future endpoint is genuinely all-empty by design, mark it True with a justification.
ENDPOINTS = [
    # A. analyst (high value — feeds the retrain)
    ("grades_historical", "grades-historical?symbol={sym}", True, False),
    ("grades_consensus", "grades-consensus?symbol={sym}", True, False),
    ("analyst_estimates", "analyst-estimates?symbol={sym}&period=annual", True, False),
    ("price_target_consensus", "price-target-consensus?symbol={sym}", True, False),
    ("price_target_summary", "price-target-summary?symbol={sym}", True, False),
    # B. fundamentals
    ("income_statement", "income-statement?symbol={sym}&period=annual&limit=20", True, False),
    ("balance_sheet", "balance-sheet-statement?symbol={sym}&period=annual&limit=20", True, False),
    ("cash_flow", "cash-flow-statement?symbol={sym}&period=annual&limit=20", True, False),
    ("ratios", "ratios?symbol={sym}&period=annual&limit=20", True, False),
    ("key_metrics", "key-metrics?symbol={sym}&period=annual&limit=20", True, False),
    ("financial_growth", "financial-growth?symbol={sym}&period=annual&limit=20", True, False),
    ("enterprise_values", "enterprise-values?symbol={sym}&limit=20", True, False),
    ("market_cap", "historical-market-capitalization?symbol={sym}", True, False),
    # C. earnings & events
    ("earnings", "earnings?symbol={sym}", True, False),
    ("dividends", "dividends?symbol={sym}", True, False),
    ("splits", "splits?symbol={sym}", True, False),
    # D. ownership & flow
    ("insider_trading", "insider-trading/search?symbol={sym}", True, False),
    ("shares_float", "shares-float?symbol={sym}", True, False),
    # F. macro (universe-agnostic). A tuple/list `targets` iterates those values as
    # {sym} instead of the ticker universe — economic_indicators is one call per name.
    ("treasury_rates", "treasury-rates", False, False),
    ("economic_indicators", "economic-indicators?name={sym}", ECON_NAMES, False),
]


# App-level error keys FMP uses to deliver entitlement/plan/schema messages with an
# HTTP 200 body (e.g. {"Error Message": ...}, {"error": ...}, {"message": ...}).
# Matched case-insensitively, whitespace-folded ("error message" == "errormessage").
_APP_ERROR_KEYS = frozenset({"error", "errormessage", "error message", "message"})


def _norm_key(k) -> str:
    """Lowercase + strip + drop interior whitespace so 'Error Message' == 'errormessage'."""
    return "".join(str(k).split()).lower()


def _is_app_error(rec) -> bool:
    """True when a parsed dict is an FMP application-level error (HTTP 200 + error body).

    Heuristic (deliberately conservative — must NOT swallow a real data row that merely
    carries a 'message' column):
      * `{"Error Message": ...}` / `{"error": ...}` — FMP's canonical error shapes:
        the presence of an explicit error/error-message key marks the record an error,
        even alongside other keys; AND
      * a SINGLE-key dict whose only key is an error-signal key (incl. a bare
        `{"message": ...}` — a lone message with no data fields is an error payload,
        but a data row that also has a 'message' field is NOT, because it has other
        keys too).
    A legitimate endpoint row (symbol/date/value/… possibly *plus* a message field) has
    a non-error key set and is left as data.
    """
    if not isinstance(rec, dict) or not rec:
        return False
    norm = {_norm_key(k) for k in rec}
    # Canonical FMP error shapes: explicit error / error-message key anywhere.
    if "error" in norm or "errormessage" in norm or "error message" in norm:
        return True
    # Otherwise only a dict whose SOLE content is an error-signal key (e.g. a bare
    # {"message": ...}) is an error; a multi-field data row is not.
    return len(norm) == 1 and norm <= _APP_ERROR_KEYS


def _app_error_sample(rec) -> dict:
    """Audit sample for an app-error record: the offending error key + its message."""
    for k, v in rec.items():
        if _norm_key(k) in _APP_ERROR_KEYS:
            return {"key": str(k), "message": str(v)}
    # Fallback (shouldn't happen): record the first key/value.
    k, v = next(iter(rec.items()))
    return {"key": str(k), "message": str(v)}


def classify(payload):
    """Pure classifier of a parsed FMP response → (status, rows).

    status ∈ {with_data, no_data, http_error, fetch_error, app_error}. `_get` returns
    {"_http": code} / {"_err": name} sentinels for transport failures; a real
    list/dict payload is data. An empty list is legitimate no-coverage (no_data).

    App-level errors (entitlement/plan/schema messages FMP returns with HTTP 200,
    e.g. {"Error Message": ...}) are detected BEFORE the generic dict-as-data path and
    classified `app_error` so they FAIL CLOSED — accepting one as a single data row would
    write an `ok` manifest and skip the endpoint forever, defeating the paid-window audit.
    For a list, ANY app-error element makes the whole response `app_error` (fail closed):
    a paid harvest must not canonicalize an FMP top-level error object into the parquet as a
    data row just because another element happens to be real. The `rows` returned for an
    app_error are the offending error dicts ONLY, so the caller attaches the message/key to
    the audit trail and writes no data for that target.
    """
    if isinstance(payload, dict):
        if "_http" in payload:
            return "http_error", []
        if "_err" in payload:
            return "fetch_error", []
        # App-level error BEFORE treating a dict as one data row (fail closed).
        if _is_app_error(payload):
            return "app_error", [payload]
        return "with_data", [payload]
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
        if not rows:
            return "no_data", []
        # FAIL CLOSED on a mixed list: if ANY element is an app-error, the response is
        # untrustworthy — a top-level FMP error object must not be written as data just
        # because a sibling row is real. Return only the offending error dicts as the
        # error sample; no data is written for this target.
        errs = [r for r in rows if _is_app_error(r)]
        if errs:
            return "app_error", errs
        return "with_data", rows
    return "no_data", []


def _get(path: str, key: str, retries: int = 3, backoff: float = 0.5):
    """GET with bounded exponential backoff on 429/5xx/timeouts."""
    sep = "&" if "?" in path else "?"
    url = f"{BASE}/{path}{sep}apikey={key}"
    last = {"_err": "Unattempted"}
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.loads(r.read().decode("utf-8", "ignore"))
        except urllib.error.HTTPError as e:
            last = {"_http": e.code}
            if e.code in RETRY_CODES and attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            return last
        except Exception as e:  # noqa: BLE001
            last = {"_err": type(e).__name__}
            if attempt < retries:
                time.sleep(backoff * (2 ** attempt))
                continue
            return last
    return last


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_parquet(df, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.parquet")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dst)


def _atomic_write_json(obj, dst: Path) -> None:
    tmp = dst.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, dst)


def _targets_for(per_ticker, uni) -> list[str]:
    """Resolve the {sym} iteration list for an endpoint.

    per_ticker: True → the ticker universe; a tuple/list → those exact {sym} values
    (e.g. macro indicator names); falsy → a single universe-agnostic call.
    """
    if per_ticker is True:
        return list(uni)
    if isinstance(per_ticker, (list, tuple)):
        return list(per_ticker)
    return ["_"]


def _universe_hash(targets) -> str:
    """Identity of the exact target list (order-independent), so a changed
    universe (added/removed/renamed names) invalidates a stale manifest."""
    h = hashlib.sha256()
    h.update("\n".join(sorted(str(t) for t in targets)).encode("utf-8"))
    return h.hexdigest()


def _is_valid_zero_data_completion(man, *, tmpl=None, targets=None) -> bool:
    """Full ZERO-DATA invariant: only an explicitly-allowed, internally-consistent,
    config-matched empty completion may be honored as a skippable state.

    EVERY clause must hold (any inconsistency → re-pull, never silently trusted):
      * manifest_version matches the current state machine; AND
      * allow_zero_data is True (endpoint explicitly permitted to be all-empty); AND
      * status == "ok"; AND
      * with_data == 0 AND errors == 0 (http_error + fetch_error + app_error) AND
        tickers == 0 AND rows == 0 (truly empty, no partial/errored residue); AND
      * requested == no_data (every target classified as no_data, none unaccounted); AND
      * output is None AND sha256 is None (no parquet, no checksum); AND
      * path_template / universe_hash match the current endpoint + target list.
    """
    if man.get("manifest_version") != MANIFEST_VERSION:
        return False
    if not man.get("allow_zero_data"):
        return False
    if man.get("status") != "ok":
        return False
    if man.get("output") is not None or man.get("sha256") is not None:
        return False
    counts = (int(man.get("with_data") or 0), int(man.get("http_error") or 0),
              int(man.get("fetch_error") or 0), int(man.get("app_error") or 0),
              int(man.get("tickers") or 0), int(man.get("rows") or 0))
    if any(counts):
        return False
    requested = int(man.get("requested") or 0)
    if requested == 0 or int(man.get("no_data") or 0) != requested:
        return False
    if tmpl is not None and man.get("path_template") != tmpl:
        return False
    if targets is not None and man.get("universe_hash") != _universe_hash(targets):
        return False
    return True


def _manifest_ok(manifest_path: Path, *, tmpl=None, targets=None,
                 parquet_path: Path | None = None) -> bool:
    """Decide whether a recorded manifest lets us SKIP a re-pull.

    Content/config aware — a manifest is "ok to skip" only when ALL hold:
      * status == "ok"; AND
      * its recorded `manifest_version` matches the current state machine; AND
      * its recorded `path_template` matches the current endpoint template (request
        config / endpoint changed → re-pull); AND
      * its recorded `universe_hash` matches the current target list (universe
        added/removed/renamed → re-pull); AND
      * the data-vs-zero-data shape is internally consistent and verified:
          - DATA completion  (output set): the parquet must exist AND its sha256
            must equal the recorded sha256 (missing/tampered/stale parquet → re-pull);
          - ZERO-DATA completion (output is null): honored ONLY when the FULL allowed
            zero-data invariant holds (see `_is_valid_zero_data_completion`) — an
            explicitly-allowed, internally-consistent, config-matched empty record.
            An unexpected all-empty pull is `zero_data_unexpected` (not "ok") so it
            never reaches here.

    When called with only a path (no tmpl/targets) it falls back to a pure status
    check (used by the unit gate test); the harvest loop always passes config.
    """
    if not manifest_path.exists():
        return False
    try:
        man = json.loads(manifest_path.read_text())
    except (ValueError, OSError):
        return False
    if man.get("status") != "ok":
        return False
    if tmpl is None and targets is None:
        return True  # pure status gate (no config to validate against)
    if man.get("manifest_version") != MANIFEST_VERSION:
        return False
    if tmpl is not None and man.get("path_template") != tmpl:
        return False
    if targets is not None and man.get("universe_hash") != _universe_hash(targets):
        return False
    output = man.get("output")
    if output is None:
        # Zero-data completion: honor ONLY a fully valid allowed-empty record.
        return _is_valid_zero_data_completion(man, tmpl=tmpl, targets=targets)
    # Data completion: the parquet must exist and match the recorded sha256.
    if parquet_path is None:
        parquet_path = manifest_path.parent / output
    if not parquet_path.exists():
        return False
    return _sha256(parquet_path) == man.get("sha256")


def harvest_endpoint(key_name, tmpl, per_ticker, uni, out, rate, key, fetched,
                     retries, backoff, get=None, allow_zero_data=False):
    """Pull one endpoint, write parquet + manifest atomically, return the manifest.

    `get` is injectable so tests can drive the classify/write/manifest path offline.
    Resolved at call time (not bound as a default) so a monkeypatched `_get` is honored.

    `allow_zero_data`: when False (the DEFAULT) an all-target-empty pull is treated as
    `zero_data_unexpected` — it FAILS CLOSED (counts toward the non-zero exit) and the
    existing verified parquet is PRESERVED (a suspicious refresh must not retire the last
    good artifact). When True an all-empty pull is a valid zero-data completion (status
    "ok", output null) and any older parquet is atomically retired.

    PARTIAL-ERROR PRESERVATION: when ANY target errors (`bad > 0`) the canonical parquet
    is NEVER replaced, even if other targets returned real rows. The partial rows are
    quarantined to `<key>_291.parquet.staging` and the manifest records `output: null`
    (canonical untouched), `status: "errors"`, and a `staging`/`staged_rows` pointer for
    audit. The "last verified" canonical parquet+manifest only advance when a full
    replacement passes the endpoint acceptance rule (all targets data/no_data, no errors).
    """
    import pandas as pd  # noqa: PLC0415

    if get is None:
        get = _get
    targets = _targets_for(per_ticker, uni)
    rows, counts, errs = [], {"with_data": 0, "no_data": 0, "http_error": 0,
                              "fetch_error": 0, "app_error": 0}, []
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    for sym in targets:
        payload = get(tmpl.format(sym=sym), key, retries, backoff)
        status, recs = classify(payload)
        counts[status] += 1
        if status == "with_data":
            rows.extend({**r, "ticker": sym} for r in recs)
        elif status in ("http_error", "fetch_error", "app_error") and len(errs) < 10:
            # Record *why* a name failed, not just the ticker, so the audit trail is
            # actionable. `_get` returns {"_http": code} / {"_err": name} sentinels for
            # transport failures; an app_error carries the offending error key/message
            # from the HTTP-200 body — thread the relevant detail through.
            sample = {"ticker": sym}
            if status == "app_error":
                sample.update(_app_error_sample(recs[0]) if recs else {})
            elif isinstance(payload, dict):
                if "_http" in payload:
                    sample["http"] = payload["_http"]
                elif "_err" in payload:
                    sample["err"] = payload["_err"]
            errs.append(sample)
        time.sleep(rate)

    df = pd.DataFrame(rows)
    dst = out / f"{key_name}_291.parquet"
    staging = dst.with_suffix(".parquet.staging")
    # app_error folds into the error bucket so an HTTP-200 error body FAILS CLOSED
    # exactly like an http/fetch error (never written as an ok-skip manifest).
    bad = counts["http_error"] + counts["fetch_error"] + counts["app_error"]
    # All-target zero data with no errors: legitimate only if this endpoint is
    # explicitly allowed to be all-empty; otherwise it is SUSPICIOUS (entitlement
    # change / vendor-schema failure / wrong param / outage) → fail closed.
    zero_data = (not len(df)) and bad == 0
    zero_data_unexpected = zero_data and not allow_zero_data

    if len(df):
        df["fetched_at"] = fetched
        df["source"] = f"fmp_{key_name}"
    staged = None
    if bad:
        # PARTIAL ERROR: do NOT replace the canonical parquet (preserve the last
        # verified artifact). Quarantine any partial rows to a staging path for audit;
        # the canonical dst/manifest only advance on a clean full replacement.
        if len(df):
            _atomic_write_parquet(df, staging)
            staged = staging.name
        elif staging.exists():
            staging.unlink()   # no partial rows this run → clear a prior staging file
    elif len(df):
        # Clean replacement: every target was data/no_data, no errors.
        _atomic_write_parquet(df, dst)
        if staging.exists():
            staging.unlink()   # supersede any earlier quarantined partial
    elif zero_data and allow_zero_data:
        # ALLOWED zero-data completion (a clean run, no errors): a prior quarantined
        # partial is now stale → clear it. If an OLDER parquet is on disk, atomically
        # RETIRE it so a later run can't skip on a stale parquet + an output==null
        # manifest. (Only an accepted-legitimate empty result may retire.)
        if staging.exists():
            staging.unlink()
        if dst.exists():
            os.replace(dst, dst.with_suffix(".parquet.retired"))
    # else (zero_data_unexpected): PRESERVE the last verified parquet —
    # a suspicious/failed refresh must not retire the prior good artifact.

    # On a clean run the canonical parquet reflects this pull; on an errored run the
    # canonical parquet is whatever (verified) state was already on disk — manifest
    # output/rows/sha reference ONLY the canonical artifact, never the staged partial.
    canonical_written = bool(len(df)) and not bad
    if bad:
        status = "errors"
    elif zero_data_unexpected:
        status = "zero_data_unexpected"
    else:
        status = "ok"
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "endpoint": key_name,
        "path_template": tmpl,
        "url_base": BASE,
        "allow_zero_data": bool(allow_zero_data),
        "requested": len(targets),
        "universe_hash": _universe_hash(targets),
        **counts,
        "error_samples": errs,
        "rows": int(len(df)) if canonical_written else 0,
        "tickers": int(df["ticker"].nunique()) if canonical_written else 0,
        "output": dst.name if canonical_written else None,
        "sha256": _sha256(dst) if canonical_written else None,
        # Audit pointer for a quarantined partial-error pull (NOT a skippable artifact).
        "staging": staged,
        "staged_rows": int(len(df)) if (bad and staged) else 0,
        "started_at": started,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": status,
    }
    _atomic_write_json(manifest, out / f"{key_name}_291.manifest.json")
    return manifest


def _universe(repo: Path) -> list[str]:
    import pandas as pd  # noqa: PLC0415

    d = pd.read_parquet(repo / "data" / "alpha158_291_fund_regime_dataset.parquet",
                        columns=["ticker"])
    return sorted(str(t).upper() for t in d["ticker"].unique())


def harvest(out: Path, rate: float, only, key, repo, retries, backoff,
            allow_errors) -> int:
    import pandas as pd  # noqa: PLC0415

    out.mkdir(parents=True, exist_ok=True)
    uni = _universe(repo)
    fetched = pd.Timestamp("today").normalize()
    eps = [e for e in ENDPOINTS if (only is None or only in e[0])]
    print(f"universe={len(uni)} endpoints={len(eps)} → {out}", flush=True)
    failed = []
    for key_name, tmpl, per_ticker, allow_zero in eps:
        manifest_path = out / f"{key_name}_291.manifest.json"
        targets = _targets_for(per_ticker, uni)
        # Content/config-aware skip: ok manifest + matching version + matching
        # template + matching universe + (verified parquet sha256 OR a valid
        # allowed zero-data record).
        if _manifest_ok(manifest_path, tmpl=tmpl, targets=targets,
                        parquet_path=out / f"{key_name}_291.parquet"):
            print(f"  skip {key_name} (manifest ok, verified)", flush=True)
            continue
        m = harvest_endpoint(key_name, tmpl, per_ticker, uni, out, rate, key,
                             fetched, retries, backoff, allow_zero_data=allow_zero)
        print(f"  {key_name}: tickers={m['tickers']} rows={m['rows']} "
              f"no_data={m['no_data']} http_err={m['http_error']} "
              f"fetch_err={m['fetch_error']} → {m['status']}", flush=True)
        if m["status"] != "ok":
            failed.append(key_name)
    if failed and not allow_errors:
        print(f"HARVEST FAILED — endpoints with errors/unexpected-zero-data: {failed} "
              f"(rerun re-pulls them; verified artifacts preserved; "
              f"--allow-errors to tolerate)", flush=True)
        return 1
    print(f"HARVEST DONE (errored={failed or 'none'})", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="data/fmp_harvest")
    ap.add_argument("--rate", type=float, default=0.2, help="sleep between calls (≈300/min at 0.2)")
    ap.add_argument("--only", default=None, help="substring filter on endpoint key (e.g. 'analyst')")
    ap.add_argument("--repo", default="/Users/renhao/git/github/RenQuant")
    ap.add_argument("--retries", type=int, default=3, help="retry attempts on 429/5xx/timeout")
    ap.add_argument("--backoff", type=float, default=0.5, help="base backoff seconds (exponential)")
    ap.add_argument("--allow-errors", action="store_true",
                    help="exit 0 even if some endpoints errored (default: fail closed)")
    args = ap.parse_args(argv)
    key = os.environ.get("FMP_API_KEY")
    if not key:
        print("FMP_API_KEY not set")
        return 1
    return harvest(Path(args.out), args.rate, args.only, key, Path(args.repo),
                   args.retries, args.backoff, args.allow_errors)


if __name__ == "__main__":
    raise SystemExit(main())
