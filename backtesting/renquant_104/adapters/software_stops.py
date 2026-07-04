"""S-FRAC stage 3 (core) — the software-stop registry + evaluator.

Design: renquant-orchestrator doc/design/2026-07-02-s-frac-fractional-v2.md
§3.2 (where the software stop lives) + §3.3/§3.4 (failure-mode analysis).
Sprint D2. This module supplies the protection layer that stage 0
(adapters/commit_contract.py, PR #439) fail-closes on: a fractional
quantity cannot ride a broker-resident GTC stop at this broker (Alpaca
fractional orders are TIF=DAY only — design §4), so its stop lives HERE,
evaluated by the 12-minute intraday sell-only loop
(scripts/intraday_sell_104.sh → SellOnlyPipeline →
kernel/pipeline/task_software_stops.SoftwareStopExitTask).

Contract with stage 0 (never reimplemented, only satisfied):

* ``is_armed()`` — the probe ``commit_contract.software_stops_armed``
  calls. Returns True ONLY when the layer is enabled AND the persisted
  registry loaded cleanly. A corrupt registry is NOT armed ⇒ the stage-0
  capability gate blocks every new fractional BUY (fail-closed, never
  silently unprotected).
* ``register(symbol, qty, stop_price, ...)`` — called by the Z9 stop
  router (adapters/z9_stops.place_or_replace_stop) when
  ``route_stop_protection`` selects ``"software"``.
* ``deregister(symbol, ...)`` / ``gc(currently_held)`` — called on full
  liquidation (Z9 cancel path) and at STATE-GC for externally-disposed
  positions.

Registry state file (default ``data/rq105/software_stops.json``,
broker-tagged like every other live state file — the 2026-04-27
paper-contaminates-alpaca incident):

.. code-block:: json

    {
      "version": 1,
      "contract": "software-stops-v1",
      "max_staleness_minutes": 30.0,
      "last_evaluated_at": "2026-07-03T14:32:11-04:00",
      "stops": {
        "BLK": {
          "symbol": "BLK",
          "qty": 0.341052,
          "stop_price": 760.0,
          "armed_at": "2026-07-03",
          "source": "z9",
          "history": [
            {"ts": "...", "action": "register", "stop_price": 760.0,
             "qty": 0.341052, "reason": ""}
          ]
        }
      }
    }

Invariants:

* **Never-loosen (ratchet-only).** ``register`` may only RAISE
  ``stop_price`` (tightening). A lower price is refused and recorded in
  the entry's history (``ratchet_refused``). Loosening requires the
  explicit ``rewrite_stop(symbol, price, reason=...)`` path with a
  non-empty logged reason. NOTE the deliberate difference from the Z9
  broker-stop convention (z9_stops takes ``min`` — it never MOVES a
  catastrophe line once set): the software stop is re-derived from a
  reference price on every placement/top-up, so ratcheting UP keeps the
  tightest protection ever computed while a top-up at a lower reference
  can never widen it. Task spec (sprint D2): "stop may only RATCHET UP".
* **Corrupt registry fail-closes writes, loudly.** Load failure ⇒
  ``is_armed() == False`` (new fractional entries blocked by the stage-0
  gate), every mutation raises ``SoftwareStopRegistryCorrupt``, the
  corrupt file bytes are never overwritten (evidence preserved), and
  ``evaluate`` logs ERROR every pass — a corrupt registry is an
  operator page (scripts/check_software_stops_liveness.py exits 2),
  never a silent unprotect.
* **Gap-down-through-stop** (design §3.3): a breach fires a market exit
  for the FULL registered qty at the next loop pass regardless of how
  far through the stop the print is; the gap size is measured, logged,
  and carried on the intent — slippage is accepted, not hidden.
* **Loop-dead watchdog** (design §3.4): every ``evaluate`` stamps
  ``last_evaluated_at`` (the heartbeat). ``max_staleness_minutes`` rides
  in the registry file so the liveness checker
  (scripts/check_software_stops_liveness.py) is self-configuring: armed
  entries + a heartbeat older than the budget during a market session ⇒
  alarm.

Default OFF: ``execution.software_stops.enabled`` absent/false ⇒
``from_config`` returns None ⇒ RunnerAdapter._software_stops stays None
⇒ every stage-0 consumer behaves exactly as before this module existed.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Iterable

log = logging.getLogger("live.runner")  # same logger as z9_stops / runner

REGISTRY_CONTRACT = "software-stops-v1"
REGISTRY_VERSION = 1

# Default registry state-file path, relative to the umbrella repo root
# (the live entry points `cd $REPO_DIR` before invoking the runner —
# same convention as persistence.db_path "data/runs.db").
DEFAULT_REGISTRY_PATH = "data/rq105/software_stops.json"

# Heartbeat budget: the sell-only loop runs on a 12-minute launchd
# cadence (com.renquant.intraday104.plist); 30 minutes ≈ two missed
# passes + scheduling slack. Overridable per-file / per-config.
DEFAULT_MAX_STALENESS_MINUTES = 30.0

VALID_SOURCES = frozenset({"z9", "manual", "fractional-auto"})


class SoftwareStopRegistryCorrupt(RuntimeError):
    """Raised on any attempted WRITE to a registry that failed to load.

    The corrupt file is evidence — it is never overwritten. is_armed()
    is already False, so the stage-0 capability gate blocks new
    fractional entries; existing entries are an operator page (watchdog
    exit 2), not a silent drop.
    """


def registry_path_for(base_path: "Path | str", broker_name: str | None) -> Path:
    """Broker-isolated registry path (the 2026-04-27 incident lesson).

    ``data/rq105/software_stops.json`` + ``alpaca`` →
    ``data/rq105/software_stops.alpaca.json``. Idempotent; unknown/None
    broker returns the base path unchanged (sim/tests).
    """
    p = Path(base_path)
    if not broker_name:
        return p
    from kernel.state_paths import _safe_broker  # noqa: PLC0415

    safe = _safe_broker(broker_name)
    if safe == "unknown" or p.stem.endswith(f".{safe}"):
        return p
    return p.with_stem(f"{p.stem}.{safe}")


def _now_iso(now: "datetime.datetime | None" = None) -> str:
    dt = now or datetime.datetime.now().astimezone()
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.isoformat()


def _finite_positive(value: Any) -> "float | None":
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f <= 0:
        return None
    return f


def _validate_snapshot(raw: Any) -> dict:
    """Schema-validate a loaded registry dict. Raises ValueError on ANY
    violation — one bad entry corrupts the whole file (fail-closed: a
    partially-readable registry must not pretend the readable subset is
    the whole protection surface)."""
    if not isinstance(raw, dict):
        raise ValueError(f"registry root must be an object, got {type(raw).__name__}")
    version = raw.get("version")
    if version != REGISTRY_VERSION:
        raise ValueError(f"unsupported registry version {version!r}")
    stops = raw.get("stops")
    if not isinstance(stops, dict):
        raise ValueError("registry 'stops' must be an object")
    for sym, entry in stops.items():
        if not isinstance(entry, dict):
            raise ValueError(f"stop entry {sym!r} must be an object")
        if entry.get("symbol") != sym:
            raise ValueError(f"stop entry {sym!r} symbol mismatch: {entry.get('symbol')!r}")
        if _finite_positive(entry.get("qty")) is None:
            raise ValueError(f"stop entry {sym!r} qty invalid: {entry.get('qty')!r}")
        if _finite_positive(entry.get("stop_price")) is None:
            raise ValueError(
                f"stop entry {sym!r} stop_price invalid: {entry.get('stop_price')!r}"
            )
        if entry.get("source") not in VALID_SOURCES:
            raise ValueError(f"stop entry {sym!r} source invalid: {entry.get('source')!r}")
    ms = raw.get("max_staleness_minutes", DEFAULT_MAX_STALENESS_MINUTES)
    if _finite_positive(ms) is None:
        raise ValueError(f"max_staleness_minutes invalid: {ms!r}")
    return raw


def compute_staleness(
    snapshot: "dict | None",
    *,
    now: "datetime.datetime | None" = None,
    corrupt: bool = False,
) -> dict:
    """Pure watchdog arithmetic — shared by the registry and the liveness
    check CLI (scripts/check_software_stops_liveness.py) so the two can
    never disagree.

    ``stale`` is True only when there ARE armed entries whose heartbeat
    is missing or older than the staleness budget: an empty registry has
    nothing unprotected, so a quiet loop is not an alarm. A corrupt
    registry reports its entries unknowable and is the caller's exit-2.
    """
    now_dt = now or datetime.datetime.now().astimezone()
    if now_dt.tzinfo is None:
        now_dt = now_dt.astimezone()
    if corrupt:
        return {
            "exists": True, "corrupt": True, "n_stops": None,
            "last_evaluated_at": None, "age_minutes": None,
            "max_staleness_minutes": None, "stale": True,
        }
    if snapshot is None:
        return {
            "exists": False, "corrupt": False, "n_stops": 0,
            "last_evaluated_at": None, "age_minutes": None,
            "max_staleness_minutes": None, "stale": False,
        }
    stops = snapshot.get("stops") or {}
    n = len(stops)
    budget = float(
        snapshot.get("max_staleness_minutes", DEFAULT_MAX_STALENESS_MINUTES)
    )
    hb = snapshot.get("last_evaluated_at")
    age_minutes: "float | None" = None
    if hb:
        try:
            hb_dt = datetime.datetime.fromisoformat(hb)
            if hb_dt.tzinfo is None:
                hb_dt = hb_dt.astimezone()
            age_minutes = (now_dt - hb_dt).total_seconds() / 60.0
        except ValueError:
            age_minutes = None
    stale = n > 0 and (age_minutes is None or age_minutes > budget)
    return {
        "exists": True, "corrupt": False, "n_stops": n,
        "last_evaluated_at": hb, "age_minutes": age_minutes,
        "max_staleness_minutes": budget, "stale": stale,
    }


class SoftwareStopRegistry:
    """Persisted, ratchet-only stop registry for quantities the broker
    cannot protect. See module docstring for the full contract."""

    def __init__(
        self,
        registry_path: "Path | str",
        *,
        max_staleness_minutes: "float | None" = None,
    ) -> None:
        self._path = Path(registry_path)
        self._corrupt = False
        self._corrupt_error: "str | None" = None
        self._stops: dict[str, dict] = {}
        self._last_evaluated_at: "str | None" = None
        self._max_staleness_minutes = (
            _finite_positive(max_staleness_minutes)
            or DEFAULT_MAX_STALENESS_MINUTES
        )
        self._load()

    # ── construction ────────────────────────────────────────────────────

    @classmethod
    def from_config(
        cls,
        config: "dict | None",
        *,
        broker_name: str | None = None,
        repo_root: "Path | str | None" = None,
    ) -> "SoftwareStopRegistry | None":
        """Flag-gated constructor: ``execution.software_stops.enabled``
        absent/false ⇒ None (the layer does not exist; stage-0 semantics
        are untouched — byte-inert)."""
        ss_cfg = ((config or {}).get("execution") or {}).get("software_stops") or {}
        if not ss_cfg.get("enabled", False):
            return None
        base = ss_cfg.get("registry_path", DEFAULT_REGISTRY_PATH)
        base_p = Path(base)
        if not base_p.is_absolute():
            base_p = Path(repo_root) / base_p if repo_root else base_p
        path = registry_path_for(base_p, broker_name)
        return cls(
            path,
            max_staleness_minutes=ss_cfg.get("max_staleness_minutes"),
        )

    def _load(self) -> None:
        if not self._path.exists():
            return  # empty registry — armed, created on first write
        try:
            raw = _validate_snapshot(json.loads(self._path.read_text()))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._corrupt = True
            self._corrupt_error = f"{type(exc).__name__}: {exc}"
            log.error(
                "SOFTWARE-STOP registry CORRUPT at %s (%s) — layer NOT armed: "
                "new fractional entries are blocked by the stage-0 capability "
                "gate; existing registered stops CANNOT be evaluated. This is "
                "an operator page (check_software_stops_liveness.py exits 2). "
                "The corrupt file is preserved untouched for forensics.",
                self._path, self._corrupt_error,
            )
            return
        self._stops = dict(raw.get("stops") or {})
        self._last_evaluated_at = raw.get("last_evaluated_at")
        self._max_staleness_minutes = float(
            raw.get("max_staleness_minutes", self._max_staleness_minutes)
        )

    # ── stage-0 probe ───────────────────────────────────────────────────

    def is_armed(self) -> bool:
        """The exact probe commit_contract.software_stops_armed calls.
        Must return literal True to count as armed (stage-0 contract)."""
        return not self._corrupt

    # ── persistence ─────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        return {
            "version": REGISTRY_VERSION,
            "contract": REGISTRY_CONTRACT,
            "max_staleness_minutes": self._max_staleness_minutes,
            "last_evaluated_at": self._last_evaluated_at,
            "stops": self._stops,
        }

    def _persist(self) -> None:
        if self._corrupt:
            raise SoftwareStopRegistryCorrupt(
                f"refusing to write over corrupt registry {self._path}: "
                f"{self._corrupt_error}"
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # LS-ATOM pattern (state_store.save_live_state_atomic): tmp +
        # rename so a SIGKILL mid-write can never leave a truncated
        # registry — a truncated registry would fail-close ALL new
        # fractional entries until an operator intervenes.
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.snapshot(), indent=2, sort_keys=True))
        os.replace(tmp, self._path)

    # ── mutation (ratchet-only) ─────────────────────────────────────────

    def _require_writable(self) -> None:
        if self._corrupt:
            raise SoftwareStopRegistryCorrupt(
                f"registry {self._path} is corrupt ({self._corrupt_error}); "
                "writes refused — fix or quarantine the file first"
            )

    def register(
        self,
        symbol: str,
        qty: Any,
        stop_price: Any,
        *,
        source: str = "z9",
        today_str: "str | None" = None,
        reason: str = "",
    ) -> dict:
        """Arm (or ratchet) the stop for *symbol*.

        New entry: recorded verbatim. Existing entry: ``stop_price`` may
        only move UP (never-loosen); a lower proposal keeps the existing
        stop and records ``ratchet_refused``. ``qty`` always refreshes to
        the CURRENT held quantity (a top-up grows the protected qty; the
        price ratchet is what never loosens).
        """
        self._require_writable()
        q = _finite_positive(qty)
        p = _finite_positive(stop_price)
        if q is None or p is None:
            raise ValueError(
                f"register({symbol!r}) needs finite positive qty/stop_price, "
                f"got qty={qty!r} stop_price={stop_price!r}"
            )
        if source not in VALID_SOURCES:
            raise ValueError(
                f"register({symbol!r}) source must be one of "
                f"{sorted(VALID_SOURCES)}, got {source!r}"
            )
        ts = _now_iso()
        existing = self._stops.get(symbol)
        if existing is None:
            entry = {
                "symbol": symbol,
                "qty": q,
                "stop_price": p,
                "armed_at": today_str or ts[:10],
                "source": source,
                "history": [{
                    "ts": ts, "action": "register",
                    "stop_price": p, "qty": q, "reason": reason,
                }],
            }
            self._stops[symbol] = entry
            self._persist()
            log.info(
                "SOFTWARE-STOP armed: %s qty=%s stop=$%.4f source=%s",
                symbol, q, p, source,
            )
            return dict(entry)
        old_stop = float(existing["stop_price"])
        if p > old_stop:
            action, new_stop = "ratchet_up", p
        elif p < old_stop:
            # Never-loosen: refuse, keep the existing (tighter) stop.
            action, new_stop = "ratchet_refused", old_stop
            log.warning(
                "SOFTWARE-STOP never-loosen: %s proposed stop $%.4f < "
                "existing $%.4f — REFUSED (loosening requires the explicit "
                "rewrite_stop path with a logged reason)",
                symbol, p, old_stop,
            )
        else:
            action, new_stop = "refresh", old_stop
        existing["stop_price"] = new_stop
        existing["qty"] = q
        existing["history"].append({
            "ts": ts, "action": action,
            "stop_price": new_stop, "qty": q,
            "proposed_stop_price": p, "reason": reason,
        })
        self._persist()
        if action == "ratchet_up":
            log.info(
                "SOFTWARE-STOP ratchet: %s stop $%.4f → $%.4f qty=%s",
                symbol, old_stop, new_stop, q,
            )
        return dict(existing)

    def rewrite_stop(self, symbol: str, stop_price: Any, *, reason: str) -> dict:
        """The ONLY loosening path. Requires an explicit non-empty reason,
        which is logged and recorded in the entry's audit history."""
        self._require_writable()
        if not reason or not str(reason).strip():
            raise ValueError(
                f"rewrite_stop({symbol!r}) requires an explicit non-empty "
                "reason — loosening a protective stop is never implicit"
            )
        p = _finite_positive(stop_price)
        if p is None:
            raise ValueError(
                f"rewrite_stop({symbol!r}) needs a finite positive "
                f"stop_price, got {stop_price!r}"
            )
        entry = self._stops.get(symbol)
        if entry is None:
            raise KeyError(f"rewrite_stop({symbol!r}): no registered stop")
        old_stop = float(entry["stop_price"])
        entry["stop_price"] = p
        entry["history"].append({
            "ts": _now_iso(), "action": "explicit_rewrite",
            "stop_price": p, "qty": float(entry["qty"]),
            "previous_stop_price": old_stop, "reason": str(reason),
        })
        self._persist()
        log.warning(
            "SOFTWARE-STOP explicit rewrite: %s stop $%.4f → $%.4f "
            "(reason: %s)", symbol, old_stop, p, reason,
        )
        return dict(entry)

    def deregister(self, symbol: str, *, reason: str = "") -> bool:
        """Remove the stop for *symbol* (full liquidation / GC). No-op
        (False) when absent. Refused on a corrupt registry."""
        if symbol not in self._stops:
            return False
        self._require_writable()
        self._stops.pop(symbol, None)
        self._persist()
        log.info("SOFTWARE-STOP disarmed: %s (%s)", symbol, reason or "no reason")
        return True

    def gc(self, currently_held: Iterable[str]) -> list[str]:
        """Drop entries whose position is gone (external disposition).
        Mirrors the Z9 stop_orders STATE-GC. Returns dropped symbols."""
        held = set(currently_held)
        stale = [s for s in self._stops if s not in held]
        for s in stale:
            self.deregister(s, reason="software-stop GC: position gone")
        return stale

    # ── evaluation (the sell-only loop consumer) ────────────────────────

    def evaluate(
        self,
        live_quotes: "dict | None",
        *,
        now: "datetime.datetime | None" = None,
    ) -> list[dict]:
        """One sell-only-loop pass: breach checks + heartbeat stamp.

        Returns a list of market-exit intents, one per breached stop:
        ``{symbol, qty, stop_price, trigger_price, gap_pct, source,
        armed_at, reason}``. ``qty`` is the FULL registered quantity
        (fractional-capable — the stage-0 float commit contract carries
        it end-to-end). Entries stay registered until the exit is
        broker-confirmed (commit deregisters on full liquidation), so a
        failed SELL re-fires next pass instead of silently unprotecting.

        Gap-down-through-stop (design §3.3): the intent fires however far
        through the stop the print is; ``gap_pct`` measures the gap and
        is logged — slippage accepted, never hidden.
        """
        if self._corrupt:
            log.error(
                "SOFTWARE-STOP evaluate on CORRUPT registry %s (%s) — "
                "registered stops CANNOT be checked this pass. Operator "
                "action required (watchdog exits 2).",
                self._path, self._corrupt_error,
            )
            return []
        quotes = live_quotes or {}
        intents: list[dict] = []
        for symbol, entry in self._stops.items():
            stop = float(entry["stop_price"])
            quote = _finite_positive(quotes.get(symbol))
            if quote is None:
                log.warning(
                    "SOFTWARE-STOP: no finite quote for %s this pass — "
                    "stop @ $%.4f NOT evaluated (stays armed)",
                    symbol, stop,
                )
                continue
            if quote > stop:
                continue
            gap_pct = max(0.0, (stop - quote) / stop)
            reason = (
                f"software_stop breach: price {quote:.4f} <= stop {stop:.4f} "
                f"(gap {gap_pct:.2%}, qty={float(entry['qty'])}, "
                f"source={entry.get('source')}, armed_at={entry.get('armed_at')})"
            )
            log.warning(
                "SOFTWARE-STOP BREACH %s: price $%.4f <= stop $%.4f "
                "gap=%.2f%% qty=%s — emitting market exit for the FULL "
                "registered qty; gap slippage accepted by design (§3.3)",
                symbol, quote, stop, gap_pct * 100.0, float(entry["qty"]),
            )
            intents.append({
                "symbol": symbol,
                "qty": float(entry["qty"]),
                "stop_price": stop,
                "trigger_price": quote,
                "gap_pct": gap_pct,
                "source": entry.get("source"),
                "armed_at": entry.get("armed_at"),
                "reason": reason,
            })
        # Heartbeat AFTER the pass: the watchdog's liveness signal is
        # "the loop finished evaluating the registry", not "it started".
        self._last_evaluated_at = _now_iso(now)
        self._persist()
        return intents

    # ── watchdog surface ────────────────────────────────────────────────

    def staleness_state(
        self, *, now: "datetime.datetime | None" = None,
    ) -> dict:
        if self._corrupt:
            return compute_staleness(None, now=now, corrupt=True)
        return compute_staleness(self.snapshot(), now=now)

    # ── introspection (tests / forensics) ───────────────────────────────

    @property
    def path(self) -> Path:
        return self._path

    @property
    def corrupt(self) -> bool:
        return self._corrupt

    def get(self, symbol: str) -> "dict | None":
        entry = self._stops.get(symbol)
        return dict(entry) if entry is not None else None

    def symbols(self) -> list[str]:
        return sorted(self._stops)

    def __len__(self) -> int:
        return len(self._stops)
