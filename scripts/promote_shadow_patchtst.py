#!/usr/bin/env python3
"""promote_shadow_patchtst.py — validated served-pin promote for the SHADOW PatchTST scorer.

Design: ``doc/design/2026-06-30-shadow-scorer-freshness.md`` (RFC r2; orchestrator
PR #212). This closes the deeper of the two shadow freezes the RFC diagnoses (§1.3):
a successful ``weekly_retrain_patchtst.sh`` (rc=0) writes only the walk-forward
corpus (``walkforward_patchtst/``); it does **not** advance the SERVED shadow pin
(``strategy_config.shadow.json`` ``ranking.panel_scoring.artifact_path`` is a fixed
path). Without a promote step the model ages in place while the retrain "succeeds"
— the repo's recurring *"merged is not deployed / deployed-but-dark"* failure.

SHADOW-SCOPED. PatchTST is the shadow (champion–challenger) scorer, not the live
decision, so this moves **no capital**. But it shares the daily inference + reporting
paths (RFC §2), so a broken/degenerate artifact can still fail the daily run or
corrupt the challenger evidence. Therefore the pin swap FAILS CLOSED unless ALL of:

  §3.1 freshness  (a) every recipe-required source is on its source-specific SLA, AND
                  (b) the candidate's effective train/selection cutoffs ACTUALLY
                      ADVANCE past the served pin's. A no-advance retrain (e.g. a
                      recipe/code fix on an unrefreshed panel) is LABELED non-fresh
                      (``--allow-non-fresh --reason ...``): it may be served for the
                      stated reason but does NOT reset the freshness clock.
  §3.4 validation (1) artifact LOAD + SMOKE INFERENCE, (2) schema/recipe/config-
                      fingerprint PARITY (stamped from the CURRENT pinned config —
                      reconciles with the ``panel_scorer_config_mismatch`` re-stamp,
                      §3.3), (3) NON-DEGENERATE outputs, (4) RESOURCE bounds,
                      (5) a minimum shadow-quality SANITY FLOOR.

Only then is the served pin swapped ATOMICALLY (write-new-then-swap); the shadow
decision never reads a half-written artifact, and the superseded artifact + config
backup are retained for rollback.

DRY-RUN by default. Nothing is written without ``--apply``. ``--check`` is a verbose
dry-run that runs every gate it can and prints the verdict.

Exit codes:  0 promoted, or clean dry-run/check
             10 refused: NOT FRESH (expected on a stale panel — informational, not a bug)
             20 refused: a §3.4 VALIDATION gate failed (a real problem — alert-worthy)
             2  usage / precondition error

Owner split (RFC §5): the umbrella owns the script + launchd schedule; the served
``artifact_path`` pin lives in strategy-104 config; the freshness *monitor* (Phase 1,
observe-only) is renquant-pipeline / renquant-orchestrator work and is NOT this script.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(os.environ["RENQUANT_REPO_ROOT"]).resolve() \
    if os.environ.get("RENQUANT_REPO_ROOT") else Path(__file__).resolve().parent.parent

# --- #210 §2/§3 SLA defaults -------------------------------------------------
FAST_CEILING_DAYS = 28          # #210 fast-axis (price/retrain-data) ceiling

# --- fundamentals TWO-AXIS SLA (P-FUND-FRESHNESS contract) --------------------
# The quarterly fundamentals source is NOT a generic max-date slow feed: a daily
# forward-filled feed can show yesterday's ``date`` while still MISSING the latest
# expected 10-Q. It is judged on TWO independent axes (see ``fundamentals_sla_verdict``),
# mirroring renquant-pipeline
#   src/renquant_pipeline/kernel/preflight_pipeline/tasks/fundamentals_freshness.py
# (the merged P-FUND-FRESHNESS gate): daily-feed liveness AND per-ENTITY fiscal-quarter
# coverage. It FAILS CLOSED until a real ENTITY id + fiscal-period/available-at field
# exist — the as-of ``date`` alone (or a single global max fiscal date) cannot establish
# that the WHOLE cross-section is fresh.
#
# Axis-2 (Codex #419 review 2): a single ``max(fiscal_period)`` over the whole parquet lets
# ONE current issuer certify the entire panel while ~291 others stay quarters stale/missing.
# Instead freshness is evaluated PER ENTITY from each entity's OWN latest fiscal-period end
# (no calendar-quarter snapping -> valid for non-calendar fiscal years) and gated on a
# PREREGISTERED COVERAGE DISTRIBUTION (missing fraction, stale fraction, worst-quarters-
# behind, quantiles), not one global maximum.
FUND_MAX_FEED_STALE_DAYS = 20   # daily forward-filled feed liveness ceiling (aligns pipeline)
FUND_FILING_LAG_DAYS = 45       # days after fiscal-period end a 10-Q is expected filed+ingested
FUND_QUARTER_DAYS = 92          # nominal quarter length (per-entity staleness; no calendar snap)
FUND_MAX_QUARTERS_BEHIND = 1    # per entity: current == 0 quarters behind (q_behind < this)
# Preregistered coverage policy (PROVISIONAL — conservative/fail-closed pending the
# IC/rank/turnover sensitivity study in doc/design/2026-06-30-shadow-scorer-freshness.md).
FUND_MAX_STALE_FRACTION = 0.05      # <=5% of the universe may be >= max_quarters_behind stale
FUND_MAX_MISSING_FRACTION = 0.02    # <=2% of the universe may lack any fiscal-period provenance
FUND_MAX_WORST_QUARTERS_BEHIND = 1  # no single entity may be more than this many quarters behind
FUND_MIN_ENTITIES = 50              # refuse to certify quarterly coverage on a tiny cross-section
# Entity/issuer id columns (the first present is the grouping key). NONE present ->
# per-entity coverage is UNVERIFIABLE -> fail closed.
DEFAULT_ENTITY_COLS = ["ticker", "symbol", "entity_id", "cik", "figi", "permaticker", "sid"]
# Real fiscal-period / available-at columns that carry the true per-entity fiscal quarter
# (NOT the forward-fillable as-of ``date``). The first present column is used per entity;
# NONE present -> quarterly availability is UNVERIFIABLE -> fail closed. Ordered date-like
# first (ambiguous string quarter labels like "2026Q1" are deliberately excluded).
DEFAULT_FISCAL_PERIOD_COLS = ["fiscal_period_end", "period_end", "report_date",
                              "filed_date", "acceptance_datetime", "available_at"]

# Recipe-required sources the served shadow model's data cutoff is capped by (RFC §3.1).
# ``date_col`` present -> read the max of that column as the data cutoff (a declared
# data-cutoff source: fail CLOSED on any read failure, NEVER fall back to file mtime);
# ``kind=fundamentals`` -> the two-axis P-FUND-FRESHNESS check (above).
DEFAULT_SOURCES: list[dict] = [
    {"name": "transformer_panel", "path": "data/transformer_v4_wl200_clean.parquet",
     "axis": "fast", "sla_days": FAST_CEILING_DAYS, "date_col": "date"},
    {"name": "rawlabel", "path": "data/alpha158_291_fundamental_dataset_rawlabel.parquet",
     "axis": "fast", "sla_days": FAST_CEILING_DAYS, "date_col": "date"},
    {"name": "fundamentals", "path": "data/sec_fundamentals_daily.parquet",
     "axis": "slow", "kind": "fundamentals", "date_col": "date",
     "max_feed_stale_days": FUND_MAX_FEED_STALE_DAYS,
     "filing_lag_days": FUND_FILING_LAG_DAYS,
     "quarter_days": FUND_QUARTER_DAYS,
     "max_quarters_behind": FUND_MAX_QUARTERS_BEHIND,
     "max_stale_fraction": FUND_MAX_STALE_FRACTION,
     "max_missing_fraction": FUND_MAX_MISSING_FRACTION,
     "max_worst_quarters_behind": FUND_MAX_WORST_QUARTERS_BEHIND,
     "min_entities": FUND_MIN_ENTITIES,
     "entity_cols": DEFAULT_ENTITY_COLS,
     "fiscal_period_cols": DEFAULT_FISCAL_PERIOD_COLS},
]

DEFAULT_SERVED_CONFIG = "backtesting/renquant_104/strategy_config.shadow.json"
DEFAULT_PIN_KEY = "ranking.panel_scoring.artifact_path"
DEFAULT_WF_MANIFEST = "backtesting/renquant_104/artifacts/walkforward_patchtst_manifest.json"
DEFAULT_SERVED_ROOT = "artifacts/patchtst_shadow"
DEFAULT_STAMP_SCRIPT = "scripts/stamp_patchtst_fingerprint.py"

RC_OK = 0
RC_NOT_FRESH = 10
RC_GATE_FAILED = 20
RC_USAGE = 2

# --- future-dated / clock-skew guard (Codex #419 review 3) -------------------
# EVERY freshness/staleness computation below compares a data cutoff (a source's
# max date, the fundamentals daily-feed as-of date, or a per-entity fiscal-period
# end) to ``now`` (the decision timestamp, ``dt.date.today()``). A cutoff LATER
# than ``now`` is IMPOSSIBLE — the source/availability/fiscal date cannot postdate
# the decision that reads it — and is itself a real look-ahead-leak signal (a
# clock bug, a corrupted/rewritten date column, or an actual future-dated write).
# It must FAIL CLOSED with a distinct "future-dated" reason. It must NEVER be
# clamped to age=0 / quarters-behind=0: that silently turns IMPOSSIBLE data into
# MAXIMALLY FRESH data — exactly backwards, and exactly the bug this guards.
#
# Tolerance: 0 days. Every comparison here is done at ``dt.date`` (day) granularity
# (``dt.date.today()`` vs a parsed ``date``/fiscal-period column) — there is no
# sub-day timestamp involved, so there is no clock-skew jitter to tolerate. A
# cutoff dated exactly today (age/staleness == 0) is the boundary and passes as
# current; a cutoff dated ANY day after today is never legitimate at this
# granularity, so a 0-day tolerance is the correct (not merely convenient) value.
FUTURE_DATE_TOLERANCE_DAYS = 0
# Sentinel returned by ``entity_quarters_behind`` for an impossible (future-dated)
# per-entity fiscal-period end. Distinct from ``None`` (MISSING: no parseable date
# at all) — both fail closed, but future-dated is flagged with its own reason.
FUTURE_DATED = "future_dated"


# ============================================================================
# Pure helpers (unit-tested in tests/test_promote_shadow_patchtst.py)
# ============================================================================

def parse_date(value) -> dt.date | None:
    """Parse an ISO date/datetime string (or date) to a ``date``; None if unparseable."""
    if value is None:
        return None
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    s = str(value).strip()
    if not s:
        return None
    s = s.split("T")[0]
    try:
        return dt.date.fromisoformat(s)
    except ValueError:
        return None


def get_dotted(d: dict, dotted: str):
    """Read a nested value by a dotted path; None if any segment is missing."""
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def set_dotted(d: dict, dotted: str, value) -> None:
    """Set a nested value by a dotted path, creating intermediate dicts."""
    cur = d
    parts = dotted.split(".")
    for part in parts[:-1]:
        cur = cur.setdefault(part, {})
    cur[parts[-1]] = value


@dataclass
class SourceVerdict:
    name: str
    axis: str
    sla_days: int
    data_cutoff: dt.date | None
    age_days: int | None
    on_sla: bool
    detail: str
    coverage: dict | None = None  # per-entity fundamentals coverage distribution (if any)


def source_sla_verdict(source: dict, now: dt.date, cutoff: dt.date | None,
                       *, missing_ok: bool = False) -> SourceVerdict:
    """Judge one recipe source against its source-specific SLA (#210 §2/§3).

    ``cutoff`` is the source's data cutoff (max date column, or file mtime),
    resolved by the caller (I/O is kept out of this pure function so it is
    testable). A missing cutoff is OFF-SLA (fail-closed) unless ``missing_ok``.
    """
    sla_days = int(source["sla_days"])
    if cutoff is None:
        return SourceVerdict(source["name"], source["axis"], sla_days, None, None,
                             on_sla=bool(missing_ok),
                             detail="cutoff unresolved" + (" (tolerated)" if missing_ok else ""))
    age = (now - cutoff).days
    if age < -FUTURE_DATE_TOLERANCE_DAYS:
        # Impossible: the source's data cutoff postdates the decision timestamp.
        # FAIL CLOSED with a distinct reason — never let a negative age pass the
        # `age <= sla_days` check trivially (that would score future-dated data
        # as maximally fresh, i.e. a look-ahead leak in the freshness gate).
        return SourceVerdict(source["name"], source["axis"], sla_days, cutoff, age, False,
                             detail=f"cutoff={cutoff.isoformat()} is FUTURE-DATED "
                                    f"({-age}d after now={now.isoformat()}) — impossible / "
                                    f"look-ahead value, fail-closed ({FUTURE_DATED})")
    on_sla = age <= sla_days
    return SourceVerdict(source["name"], source["axis"], sla_days, cutoff, age, on_sla,
                         detail=f"cutoff={cutoff.isoformat()} age={age}d sla={sla_days}d "
                                f"{'OK' if on_sla else 'OFF-SLA'}")


@dataclass
class AdvanceVerdict:
    train_served: dt.date | None
    train_candidate: dt.date | None
    selection_served: dt.date | None
    selection_candidate: dt.date | None
    advanced: bool
    detail: str


def cutoffs_advance(served: dict, candidate: dict) -> AdvanceVerdict:
    """True iff BOTH effective train- and selection-cutoffs strictly advance.

    A missing candidate axis is treated as NON-advancing (fail-closed): we cannot
    prove freshness we cannot read. ``served``/``candidate`` are dicts with keys
    ``effective_train_cutoff_date`` / ``effective_selection_cutoff_date``.
    """
    ts = parse_date(served.get("effective_train_cutoff_date"))
    tc = parse_date(candidate.get("effective_train_cutoff_date"))
    ss = parse_date(served.get("effective_selection_cutoff_date"))
    sc = parse_date(candidate.get("effective_selection_cutoff_date"))

    reasons: list[str] = []
    train_adv = tc is not None and ts is not None and tc > ts
    sel_adv = sc is not None and ss is not None and sc > ss
    if tc is None:
        reasons.append("candidate train cutoff missing")
    elif ts is None:
        reasons.append("served train cutoff missing")
    elif not train_adv:
        reasons.append(f"train cutoff did not advance ({ts.isoformat()} -> {tc.isoformat()})")
    if sc is None:
        reasons.append("candidate selection cutoff missing")
    elif ss is None:
        reasons.append("served selection cutoff missing")
    elif not sel_adv:
        reasons.append(f"selection cutoff did not advance ({ss.isoformat()} -> {sc.isoformat()})")

    advanced = train_adv and sel_adv
    detail = "both cutoffs advanced" if advanced else "; ".join(reasons)
    return AdvanceVerdict(ts, tc, ss, sc, advanced, detail)


def freshness_tier(fast_age_days: int | None, *, all_sources_on_sla: bool,
                   validated_advancing_promote: bool,
                   fast_ceiling: int = FAST_CEILING_DAYS,
                   breach_days: int = 35) -> str:
    """RFC §3.2 monitor tier keyed on the served artifact's BINDING DATA CUTOFF.

    Reused here for the promote's reporting + run-bundle stamp. A run merely
    "completing on schedule" is NOT sufficient for healthy: healthy also requires
    the pin was set by a validated, advancing promote.
    """
    if fast_age_days is None:
        return "breach"
    if fast_age_days > breach_days or not validated_advancing_promote:
        return "breach"
    if fast_age_days <= fast_ceiling and all_sources_on_sla and validated_advancing_promote:
        return "healthy"
    if fast_age_days <= fast_ceiling + 5 and all_sources_on_sla:
        return "warn"
    return "escalate"


def check_non_degenerate(scores) -> tuple[bool, str]:
    """§3.4(3): probe scores must be finite, non-constant, in a sane range."""
    import math
    vals = [float(v) for v in scores]
    if not vals:
        return False, "no probe scores produced"
    if any(math.isnan(v) or math.isinf(v) for v in vals):
        return False, "probe scores contain NaN/Inf"
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return False, f"probe scores are constant ({lo:.6g})"
    if max(abs(lo), abs(hi)) > 1e3:
        return False, f"probe scores outside sane range [{lo:.4g}, {hi:.4g}]"
    return True, f"n={len(vals)} range=[{lo:.4g}, {hi:.4g}] spread={hi - lo:.4g}"


def check_resource(elapsed_s: float, peak_rss_mb: float | None,
                   max_seconds: float, max_rss_mb: float) -> tuple[bool, str]:
    """§3.4(4): load+inference must stay within a latency / memory budget."""
    if elapsed_s > max_seconds:
        return False, f"latency {elapsed_s:.1f}s > budget {max_seconds:.0f}s"
    if peak_rss_mb is not None and peak_rss_mb > max_rss_mb:
        return False, f"peak RSS {peak_rss_mb:.0f}MB > budget {max_rss_mb:.0f}MB"
    rss = f"{peak_rss_mb:.0f}MB" if peak_rss_mb is not None else "n/a"
    return True, f"elapsed={elapsed_s:.1f}s peak_rss={rss}"


def check_sanity_floor(metric: float | None, floor: float) -> tuple[bool, str]:
    """§3.4(5): the fresh challenger clears a low, pre-declared WF/holdout floor.

    Not a trading gate — just a floor to reject a broken / collapsed model. A
    missing metric is fail-closed (cannot prove the floor is cleared).
    """
    if metric is None:
        return False, "no WF/holdout quality metric available for candidate"
    if metric < floor:
        return False, f"quality {metric:.4f} < floor {floor:.4f}"
    return True, f"quality {metric:.4f} >= floor {floor:.4f}"


# ============================================================================
# I/O helpers
# ============================================================================

def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON through a temp file in the same dir, then os.replace (atomic)."""
    tmp = path.with_suffix(path.suffix + ".promote-tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))  # validate it parses
    os.replace(tmp, path)


def _source_path(repo: Path, source: dict) -> Path:
    return (repo / source["path"]) if not Path(source["path"]).is_absolute() \
        else Path(source["path"])


def _read_parquet_max_date(path: Path, date_col: str) -> dt.date | None:
    """Max of ``date_col`` in a parquet, or None on ANY failure (fail-closed).

    Returns None for: unreadable/corrupt parquet, incompatible engine, a missing
    ``date_col``, an empty frame, or a column with no parseable dates. The caller
    treats None as UNRESOLVED and fails closed — file mtime is never substituted.
    """
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(path, columns=[date_col])
    except Exception:
        return None  # corrupt / incompatible engine / missing column at read
    if date_col not in getattr(df, "columns", []) or df.empty:
        return None
    try:
        s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    except Exception:
        return None
    if len(s) == 0:
        return None
    return s.max().date()


def _parquet_columns(path: Path) -> set[str] | None:
    """Column names of a parquet WITHOUT loading data, or None if unreadable."""
    try:
        import pyarrow.parquet as pq  # noqa: PLC0415
        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        try:
            import pandas as pd  # noqa: PLC0415
            return set(pd.read_parquet(path).columns)
        except Exception:
            return None


def _read_parquet_max_by_group(path: Path, group_col: str,
                               date_col: str) -> dict[str, dt.date | None] | None:
    """Per-entity latest ``date_col`` value: ``{entity -> max date | None}``.

    ``None`` (whole result) on ANY read failure or a missing/empty required column
    (fail-closed). A present entity whose ``date_col`` has no parseable value maps to
    ``None`` (a MISSING entity — cannot prove its fiscal freshness)."""
    try:
        import pandas as pd  # noqa: PLC0415
        df = pd.read_parquet(path, columns=[group_col, date_col])
    except Exception:
        return None
    cols = getattr(df, "columns", [])
    if group_col not in cols or date_col not in cols or df.empty:
        return None
    try:
        parsed = pd.to_datetime(df[date_col], errors="coerce")
    except Exception:
        return None
    out: dict[str, dt.date | None] = {}
    for ent, idx in df.groupby(group_col).groups.items():
        vals = parsed.loc[idx].dropna()
        out[str(ent)] = (vals.max().date() if len(vals) else None)
    return out or None


def resolve_data_cutoff(repo: Path, source: dict) -> dt.date | None:
    """Resolve a source's data cutoff.

    For a DECLARED parquet + ``date_col`` source the cutoff is ``max(date_col)``;
    ANY read/parse/empty/max-date failure returns None (UNRESOLVED -> fail closed
    at the SLA gate). File mtime is NEVER substituted for a declared data cutoff:
    a corrupt / incompatible / rewritten-old-data copy touched recently must not
    pass the SLA on filesystem liveness alone. mtime is used ONLY for a source
    that declares no ``date_col`` (there it is the intended liveness proxy, not a
    data cutoff)."""
    p = _source_path(repo, source)
    if not p.exists():
        return None
    date_col = source.get("date_col")
    if date_col and p.suffix == ".parquet":
        # Declared data-cutoff source: fail CLOSED on failure; do NOT fall to mtime.
        return _read_parquet_max_date(p, date_col)
    if date_col:
        # A date column was declared but the path is not parquet we can read as a
        # data cutoff -> unresolved (fail closed) rather than pretending mtime is it.
        return None
    # No declared date column: mtime is the intended liveness/provenance proxy.
    return dt.date.fromtimestamp(p.stat().st_mtime)


# --- fundamentals TWO-AXIS (P-FUND-FRESHNESS), per-ENTITY coverage -----------
# Aligns with renquant-pipeline preflight_pipeline/tasks/fundamentals_freshness.py
# (the merged P-FUND-FRESHNESS contract): daily-feed liveness AND per-entity fiscal
# availability. Kept self-contained — this script runs standalone under launchd and
# must not import the pipeline package at promote time.
#
# Codex #419 review 2: a single ``max(fiscal_period)`` over the whole parquet lets ONE
# current issuer certify the entire cross-section. Instead each entity is judged from
# its OWN latest fiscal-period end (staleness window, NO calendar-quarter snapping ->
# valid for non-calendar fiscal years), and the panel is gated on a PREREGISTERED
# COVERAGE DISTRIBUTION rather than one global maximum.


def entity_quarters_behind(fiscal_period_end: dt.date | None, today: dt.date,
                           filing_lag_days: int,
                           quarter_days: int = FUND_QUARTER_DAYS) -> int | None | str:
    """Quarters ONE entity's latest fiscal period lags, from its OWN fiscal-period end.

    Uses a rolling staleness window (``today - fiscal_period_end``) with a
    ``filing_lag_days`` grace period, NOT calendar-quarter snapping — so a non-calendar
    fiscal year (e.g. Jan/Apr/Jul/Oct ends) is judged on its own filing cadence.

    Returns 0 for a current entity, >=1 when behind, ``None`` when the entity has no
    parseable fiscal-period end (MISSING — its freshness cannot be proven), and the
    ``FUTURE_DATED`` sentinel when ``fiscal_period_end`` is LATER than ``today`` beyond
    ``FUTURE_DATE_TOLERANCE_DAYS``. Codex #419 review 3: a future-dated fiscal period is
    an IMPOSSIBLE / look-ahead value — it must FAIL CLOSED, never be silently treated as
    "0 quarters behind" (maximally fresh)."""
    if fiscal_period_end is None:
        return None
    staleness = (today - fiscal_period_end).days
    if staleness < -FUTURE_DATE_TOLERANCE_DAYS:
        return FUTURE_DATED  # impossible / look-ahead fiscal-period end -> fail closed
    # A full new quarter's filing is overdue once staleness exceeds one quarter + lag.
    return max(0, (staleness - filing_lag_days) // max(1, quarter_days))


def _quantiles(values: list[int], ps=(0.5, 0.9, 0.99)) -> dict:
    """Nearest-rank quantiles of integer quarters-behind (empty -> {})."""
    if not values:
        return {}
    s = sorted(values)
    out: dict[str, int] = {}
    import math  # noqa: PLC0415
    for p in ps:
        idx = min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))
        out[f"p{int(round(p * 100))}"] = int(s[idx])
    return out


@dataclass
class FundamentalsCoverage:
    n_entities: int
    n_missing: int            # entities present but with no parseable fiscal-period end
    n_future_dated: int       # entities whose fiscal-period end is LATER than now (impossible)
    n_stale: int              # entities >= max_quarters_behind behind, MISSING, or FUTURE-DATED
    n_current: int            # entities with q_behind < max_quarters_behind
    stale_fraction: float     # (n_missing + n_future_dated + n_behind) / n_entities
    missing_fraction: float   # n_missing / n_entities
    worst_quarters_behind: int | None   # max q_behind over PRESENT entities (None if all missing)
    quantiles: dict           # nearest-rank quantiles of present entities' q_behind

    def as_dict(self) -> dict:
        return {"n_entities": self.n_entities, "n_missing": self.n_missing,
                "n_future_dated": self.n_future_dated,
                "n_stale": self.n_stale, "n_current": self.n_current,
                "stale_fraction": round(self.stale_fraction, 6),
                "missing_fraction": round(self.missing_fraction, 6),
                "worst_quarters_behind": self.worst_quarters_behind,
                "quantiles": self.quantiles}


def fundamentals_coverage(fiscal_by_entity: dict, today: dt.date, *,
                          filing_lag_days: int, max_quarters_behind: int,
                          quarter_days: int = FUND_QUARTER_DAYS) -> FundamentalsCoverage:
    """PER-ENTITY coverage distribution of the fundamentals cross-section.

    ``fiscal_by_entity``: ``{entity -> latest fiscal-period end | None}``. An entity is
    STALE if MISSING (None), FUTURE-DATED (an impossible fiscal-period end later than
    ``today`` — Codex #419 review 3, never treated as current), or
    ``q_behind >= max_quarters_behind``. Aggregates the missing/future-dated counts,
    stale fraction, worst-case and quantiles — the distribution the coverage gate
    enforces (NOT a single global maximum)."""
    n = len(fiscal_by_entity)
    present_behind: list[int] = []
    n_missing = 0
    n_future = 0
    for _ent, fpe in fiscal_by_entity.items():
        qb = entity_quarters_behind(fpe, today, filing_lag_days, quarter_days)
        if qb is None:
            n_missing += 1
        elif qb == FUTURE_DATED:
            n_future += 1
        else:
            present_behind.append(int(qb))
    n_behind = sum(1 for q in present_behind if q >= max_quarters_behind)
    n_current = sum(1 for q in present_behind if q < max_quarters_behind)
    n_stale = n_missing + n_future + n_behind
    worst = max(present_behind) if present_behind else None
    return FundamentalsCoverage(
        n_entities=n, n_missing=n_missing, n_future_dated=n_future,
        n_stale=n_stale, n_current=n_current,
        stale_fraction=(n_stale / n if n else 1.0),
        missing_fraction=(n_missing / n if n else 1.0),
        worst_quarters_behind=worst, quantiles=_quantiles(present_behind))


def fundamentals_sla_verdict(source: dict, now: dt.date, *,
                             feed_max_date: dt.date | None,
                             fiscal_by_entity: dict | None,
                             provenance_present: bool,
                             max_feed_stale_days: int,
                             filing_lag_days: int,
                             max_quarters_behind: int,
                             max_stale_fraction: float,
                             max_missing_fraction: float,
                             max_worst_quarters_behind: int,
                             min_entities: int,
                             quarter_days: int = FUND_QUARTER_DAYS,
                             resolve_detail: str = "") -> SourceVerdict:
    """PURE two-axis judgement for the fundamentals source (P-FUND-FRESHNESS).

    ON-SLA only when BOTH axes hold:
      1. DAILY-FEED liveness: ``feed_age = now - feed_max_date`` within
         ``max_feed_stale_days`` (the daily forward-filled refresh is current).
      2. PER-ENTITY QUARTERLY coverage: real ENTITY id + fiscal-period/available-at
         provenance exist (else UNVERIFIABLE -> fail closed), the cross-section has at
         least ``min_entities`` names, and its coverage DISTRIBUTION clears the
         preregistered policy — ``stale_fraction <= max_stale_fraction``,
         ``missing_fraction <= max_missing_fraction`` and
         ``worst_quarters_behind <= max_worst_quarters_behind``. A single global max
         date is NOT sufficient: one fresh issuer must not certify a frozen panel.
    Any unresolved input fails closed (keep the old pin)."""
    name = source["name"]
    axis = source.get("axis", "slow")
    if feed_max_date is None:
        return SourceVerdict(name, axis, max_feed_stale_days, None, None, False,
                             detail=f"daily-feed cutoff unresolved ({resolve_detail}) "
                                    f"— fail-closed (mtime is not a data cutoff)")
    feed_age = (now - feed_max_date).days
    if feed_age < -FUTURE_DATE_TOLERANCE_DAYS:
        # Impossible: the daily-feed as-of date postdates the decision timestamp.
        # FAIL CLOSED with a distinct reason — NEVER clamp this to age=0, which would
        # silently turn a future-dated (impossible) feed into "maximally fresh".
        return SourceVerdict(name, axis, max_feed_stale_days, feed_max_date, feed_age, False,
                             detail=f"daily feed as-of {feed_max_date.isoformat()} is "
                                    f"FUTURE-DATED ({-feed_age}d after now={now.isoformat()}) "
                                    f"— impossible / look-ahead value, fail-closed "
                                    f"({FUTURE_DATED})")
    feed_ok = feed_age <= max_feed_stale_days
    feed_fact = (f"daily feed as-of {feed_max_date.isoformat()} age={feed_age}d "
                 f"(max={max_feed_stale_days}d {'OK' if feed_ok else 'STALE'})")
    if not provenance_present or fiscal_by_entity is None:
        return SourceVerdict(name, axis, max_feed_stale_days, feed_max_date, feed_age, False,
                             detail=f"{feed_fact}; QUARTERLY UNVERIFIABLE — no per-entity "
                                    f"fiscal-period/available-at provenance ({resolve_detail}); "
                                    f"fail-closed until it exists (a single global max date, or "
                                    f"the as-of date alone, cannot establish cross-section "
                                    f"freshness)")
    cov = fundamentals_coverage(fiscal_by_entity, now, filing_lag_days=filing_lag_days,
                                max_quarters_behind=max_quarters_behind,
                                quarter_days=quarter_days)
    reasons: list[str] = []
    if cov.n_future_dated > 0:
        # An impossible (future-dated) fiscal-period end is a distinct, unconditional
        # fail-closed — never let it hide inside a coverage-fraction threshold that
        # might otherwise tolerate a small count of "stale" entities (Codex #419
        # review 3: future-dated data is a look-ahead-leak signal, not ordinary staleness).
        reasons.append(f"{FUTURE_DATED}={cov.n_future_dated} entit"
                       f"{'y' if cov.n_future_dated == 1 else 'ies'} with an impossible "
                       f"fiscal-period end later than now — fail-closed")
    if cov.n_entities < min_entities:
        reasons.append(f"too few entities ({cov.n_entities}<{min_entities}) to certify coverage")
    if cov.missing_fraction > max_missing_fraction:
        reasons.append(f"missing_frac={cov.missing_fraction:.3f}>{max_missing_fraction}")
    if cov.stale_fraction > max_stale_fraction:
        reasons.append(f"stale_frac={cov.stale_fraction:.3f}>{max_stale_fraction}")
    if cov.worst_quarters_behind is not None \
            and cov.worst_quarters_behind > max_worst_quarters_behind:
        reasons.append(f"worst={cov.worst_quarters_behind}q>{max_worst_quarters_behind}")
    quarter_ok = not reasons
    on_sla = feed_ok and quarter_ok
    coverage_fact = (f"coverage n={cov.n_entities} current={cov.n_current} "
                     f"missing={cov.n_missing} {FUTURE_DATED}={cov.n_future_dated} "
                     f"stale={cov.n_stale} "
                     f"stale_frac={cov.stale_fraction:.3f}(max={max_stale_fraction}) "
                     f"worst={cov.worst_quarters_behind}q q={cov.quantiles} "
                     f"{'OK' if quarter_ok else 'STALE-COVERAGE: ' + '; '.join(reasons)}")
    return SourceVerdict(name, axis, max_feed_stale_days, feed_max_date, feed_age, on_sla,
                         detail=f"{feed_fact}; {coverage_fact} "
                                f"{'OK' if on_sla else 'OFF-SLA'}",
                         coverage=cov.as_dict())


def resolve_fundamentals_verdict(repo: Path, source: dict, now: dt.date) -> SourceVerdict:
    """I/O wrapper: read the fundamentals parquet's daily as-of date AND a PER-ENTITY
    (entity id x fiscal-period/available-at) cross-section, then apply
    ``fundamentals_sla_verdict``. All read failures fail closed (keep the old pin)."""
    kw = dict(
        max_feed_stale_days=int(source.get("max_feed_stale_days", FUND_MAX_FEED_STALE_DAYS)),
        filing_lag_days=int(source.get("filing_lag_days", FUND_FILING_LAG_DAYS)),
        max_quarters_behind=int(source.get("max_quarters_behind", FUND_MAX_QUARTERS_BEHIND)),
        max_stale_fraction=float(source.get("max_stale_fraction", FUND_MAX_STALE_FRACTION)),
        max_missing_fraction=float(source.get("max_missing_fraction", FUND_MAX_MISSING_FRACTION)),
        max_worst_quarters_behind=int(source.get("max_worst_quarters_behind",
                                                 FUND_MAX_WORST_QUARTERS_BEHIND)),
        min_entities=int(source.get("min_entities", FUND_MIN_ENTITIES)),
        quarter_days=int(source.get("quarter_days", FUND_QUARTER_DAYS)))
    entity_cols = source.get("entity_cols") or DEFAULT_ENTITY_COLS
    fiscal_cols = source.get("fiscal_period_cols") or DEFAULT_FISCAL_PERIOD_COLS
    date_col = source.get("date_col", "date")

    def _verdict(feed_max, fiscal_by_entity, provenance_present, detail):
        return fundamentals_sla_verdict(
            source, now, feed_max_date=feed_max, fiscal_by_entity=fiscal_by_entity,
            provenance_present=provenance_present, resolve_detail=detail, **kw)

    p = _source_path(repo, source)
    if not p.exists():
        return _verdict(None, None, False, "file missing")
    cols = _parquet_columns(p)
    if cols is None:
        return _verdict(None, None, False, "parquet schema unreadable")
    feed_max = _read_parquet_max_date(p, date_col)
    entity_col = next((c for c in entity_cols if c in cols), None)
    fiscal_col = next((c for c in fiscal_cols if c in cols), None)
    if entity_col is None or fiscal_col is None:
        return _verdict(feed_max, None, False,
                        f"no per-entity provenance (entity_col={entity_col}, "
                        f"fiscal_col={fiscal_col}, columns={sorted(cols)[:8]})")
    fiscal_by_entity = _read_parquet_max_by_group(p, entity_col, fiscal_col)
    if fiscal_by_entity is None:
        return _verdict(feed_max, None, False,
                        f"entity/fiscal columns unreadable "
                        f"(entity_col={entity_col}, fiscal_col={fiscal_col})")
    return _verdict(feed_max, fiscal_by_entity, True,
                    f"entity_col={entity_col} fiscal_col={fiscal_col} "
                    f"n_entities={len(fiscal_by_entity)}")


def resolve_pin_path(pin: str, config_path: Path, repo: Path) -> Path:
    """Resolve a (possibly relative) served pin the way the runtime does
    (job_panel_scoring._resolve_artifact_path -> locate_artifact): relative to the
    strategy config's directory FIRST, with a repo-root fallback if that exists."""
    p = Path(pin)
    if p.is_absolute():
        return p
    by_config = (config_path.parent / p).resolve()
    by_repo = (repo / p).resolve()
    # Prefer whichever the .pt (or its sidecar) actually exists at; default config-dir.
    for cand in (by_config, by_repo):
        if cand.exists() or Path(str(cand) + ".metadata.json").exists():
            return cand
    return by_config


def read_artifact_axes(pt_path: Path) -> dict:
    """Read {trained_date, effective_train_cutoff_date, effective_selection_cutoff_date,
    config_fingerprint, lookahead_days, label_col} from a .pt's .metadata.json sidecar."""
    meta_path = Path(str(pt_path) + ".metadata.json")
    out = {"_meta_path": str(meta_path)}
    if not meta_path.exists():
        return out
    meta = load_json(meta_path)
    tc = meta.get("training_contract") or {}
    for k in ("trained_date", "effective_train_cutoff_date",
              "effective_selection_cutoff_date", "config_fingerprint",
              "lookahead_days"):
        out[k] = meta.get(k) if meta.get(k) is not None else tc.get(k)
    out["label_col"] = tc.get("label_col") or meta.get("label_col")
    # Surface any WF/holdout quality metric the sidecar carries so the sanity
    # floor (§3.4.5) can read it when a candidate is given explicitly (no manifest).
    for k in ("wf_ic", "holdout_ic", "selection_ic", "ic", "val_ic", "sanity_metric"):
        if meta.get(k) is not None:
            out[k] = meta[k]
        elif tc.get(k) is not None:
            out[k] = tc[k]
    for k in ("selection", "eval"):
        if isinstance(meta.get(k), dict):
            out[k] = meta[k]
    return out


def discover_candidate(repo: Path, wf_manifest: Path) -> dict | None:
    """Latest retrain in the WF manifest (max cutoff_date). Returns the manifest entry
    merged with the artifact's sidecar axes, or None if the manifest is absent/empty."""
    if not wf_manifest.exists():
        return None
    man = load_json(wf_manifest)
    retrains = man.get("retrains") or []
    if not retrains:
        return None
    entry = max(retrains, key=lambda r: (r.get("cutoff_date") or ""))
    uri = entry.get("artifact_uri")
    if not uri:
        return None
    pt = Path(uri)
    if not pt.is_absolute():
        pt = repo / pt
    axes = read_artifact_axes(pt)
    merged = dict(entry)
    merged["artifact_pt"] = str(pt)
    # Manifest axes take precedence when present (authoritative retrain record).
    for k in ("effective_train_cutoff_date", "effective_selection_cutoff_date",
              "trained_date", "lookahead_days"):
        if entry.get(k) is not None:
            merged[k] = entry[k]
        elif axes.get(k) is not None:
            merged[k] = axes[k]
    merged.setdefault("config_fingerprint", axes.get("config_fingerprint"))
    merged.setdefault("label_col", axes.get("label_col"))
    merged["_sidecar_axes"] = axes
    return merged


def candidate_quality_metric(entry: dict) -> float | None:
    """Best-effort WF/holdout quality metric for the sanity floor (§3.4.5).

    Looks at common keys the WF manifest / eval sidecar may carry. None if absent
    (fail-closed at the gate)."""
    for k in ("wf_ic", "holdout_ic", "selection_ic", "ic", "val_ic", "sanity_metric"):
        v = entry.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    sel = entry.get("selection") or entry.get("eval") or {}
    if isinstance(sel, dict):
        for k in ("ic", "wf_ic", "holdout_ic", "score"):
            v = sel.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
    return None


# ============================================================================
# §3.4(1) LOAD + SMOKE INFERENCE  (heavy runtime; lazy-imported, fail-closed)
# ============================================================================

def load_and_smoke_infer(pt_path: Path, served_config_path: Path) -> dict:
    """Load the shadow scorer via the model registry and score a small synthetic
    probe panel end-to-end. Returns a dict:
        {ok, reason, scores, elapsed_s, peak_rss_mb, feature_cols, seq_len}
    ``ok=False`` on any import/load/inference failure (fail-closed)."""
    result: dict = {"ok": False, "reason": "", "scores": [],
                    "elapsed_s": 0.0, "peak_rss_mb": None}
    kernel = REPO / "backtesting" / "renquant_104"
    if str(kernel) not in sys.path:
        sys.path.insert(0, str(kernel))
    t0 = time.perf_counter()
    try:
        import numpy as np  # noqa: PLC0415
        import pandas as pd  # noqa: PLC0415
        from kernel.panel_pipeline.model_registry import registry  # noqa: PLC0415
    except Exception as exc:  # torch/pandas/registry unavailable
        result["reason"] = f"runtime unavailable for smoke inference: {exc}"
        return result

    try:
        rss0 = _peak_rss_mb()
        cfg = load_json(served_config_path)
        handler = registry.get(cfg.get("ranking", {}).get("panel_scoring", {})
                               .get("kind", "hf_patchtst"))
        scorer = handler.scorer_loader(pt_path, cfg)
        feature_cols = list(getattr(scorer, "feature_cols", []) or [])
        seq_len = int(getattr(scorer, "seq_len", 24) or 24)
        if not feature_cols:
            result["reason"] = "loaded scorer exposes no feature_cols"
            return result
        # Build a tiny synthetic probe panel: 5 tickers x seq_len dates.
        tickers = [f"PROBE{i}" for i in range(5)]
        base = pd.Timestamp("2020-01-01")
        rng = np.random.default_rng(0)
        rows = []
        for t in tickers:
            for d in range(seq_len):
                row = {"ticker": t, "date": base + pd.Timedelta(days=d)}
                feats = rng.standard_normal(len(feature_cols)).astype("float32")
                row.update(dict(zip(feature_cols, feats)))
                rows.append(row)
        probe = pd.DataFrame(rows)
        if getattr(scorer, "requires_history", False):
            series = scorer.score_with_history(probe, tickers)
        else:  # non-sequence scorer: score the latest snapshot
            latest = probe[probe["date"] == probe["date"].max()]
            series = scorer.score(latest) if hasattr(scorer, "score") else None
        scores = [float(v) for v in (series.tolist() if series is not None else [])]
        result["scores"] = scores
        result["feature_cols"] = len(feature_cols)
        result["seq_len"] = seq_len
        result["elapsed_s"] = time.perf_counter() - t0
        result["peak_rss_mb"] = (_peak_rss_mb() - rss0) if rss0 is not None else None
        if not scores:
            result["reason"] = "smoke inference produced 0 scores"
            return result
        result["ok"] = True
        result["reason"] = f"scored {len(scores)} probe tickers"
        return result
    except Exception as exc:
        result["elapsed_s"] = time.perf_counter() - t0
        result["reason"] = f"load/smoke-inference raised: {exc}"
        return result


def _peak_rss_mb() -> float | None:
    try:
        import resource  # noqa: PLC0415
        ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        return ru / (1024 * 1024) if platform.system() == "Darwin" else ru / 1024
    except Exception:
        return None


# ============================================================================
# Fingerprint parity (§3.4.2) via the existing stamp tool
# ============================================================================

def live_config_fingerprint(served_config_path: Path) -> str | None:
    """Compute the live config fingerprint the same way the LoadScorerTask gate does."""
    kernel = REPO / "backtesting" / "renquant_104"
    if str(kernel) not in sys.path:
        sys.path.insert(0, str(kernel))
    try:
        from kernel.config_consistency import fingerprint_config  # noqa: PLC0415
        return fingerprint_config(load_json(served_config_path))
    except Exception:
        return None


def stamp_fingerprint(stamp_script: Path, meta_path: Path, served_config_path: Path,
                      *, write: bool) -> tuple[int, str]:
    """Invoke scripts/stamp_patchtst_fingerprint.py (fail-closed compat check + stamp)."""
    cmd = [sys.executable, str(stamp_script),
           "--artifact-meta", str(meta_path),
           "--strategy-config", str(served_config_path)]
    if write:
        cmd.append("--write")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout + proc.stderr)


# ============================================================================
# Promote orchestration
# ============================================================================

@dataclass
class GateResult:
    name: str
    ok: bool
    detail: str


@dataclass
class PromoteReport:
    verdict: str = "unknown"
    rc: int = RC_USAGE
    fresh: bool = False
    labeled_non_fresh: bool = False
    source_verdicts: list = field(default_factory=list)
    advance: AdvanceVerdict | None = None
    gates: list = field(default_factory=list)
    served_pin: str = ""
    candidate_pt: str = ""
    promoted_pin: str | None = None
    superseded_backup: str | None = None
    tier: str = "unknown"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "rc": self.rc, "fresh": self.fresh,
            "labeled_non_fresh": self.labeled_non_fresh, "tier": self.tier,
            "served_pin": self.served_pin, "candidate_pt": self.candidate_pt,
            "promoted_pin": self.promoted_pin, "superseded_backup": self.superseded_backup,
            "source_verdicts": [v.__dict__ | {"data_cutoff":
                                (v.data_cutoff.isoformat() if v.data_cutoff else None)}
                                for v in self.source_verdicts],
            "advance": (None if self.advance is None else {
                "advanced": self.advance.advanced, "detail": self.advance.detail}),
            "gates": [g.__dict__ for g in self.gates],
        }


def run_promote(args) -> PromoteReport:
    repo = REPO
    now = args.now
    rep = PromoteReport()
    served_config = (repo / args.served_config) if not Path(args.served_config).is_absolute() \
        else Path(args.served_config)
    stamp_script = (repo / args.stamp_script) if not Path(args.stamp_script).is_absolute() \
        else Path(args.stamp_script)

    if not served_config.exists():
        rep.verdict = f"precondition: served config {served_config} not found"
        rep.rc = RC_USAGE
        return rep

    cfg = load_json(served_config)
    panel = get_dotted(cfg, "ranking.panel_scoring") or {}
    kind = panel.get("kind")
    if kind != "hf_patchtst":
        rep.verdict = (f"precondition: {served_config.name} panel_scoring.kind={kind!r} "
                       f"is not 'hf_patchtst' — refusing to edit a non-PatchTST pin")
        rep.rc = RC_USAGE
        return rep

    served_pin = get_dotted(cfg, args.pin_key)
    if not served_pin:
        rep.verdict = f"precondition: pin key {args.pin_key} absent in {served_config.name}"
        rep.rc = RC_USAGE
        return rep
    rep.served_pin = served_pin
    served_pt = resolve_pin_path(served_pin, served_config, repo)
    served_axes = read_artifact_axes(served_pt)

    # --- candidate discovery ---
    if args.candidate:
        cand_pt = Path(args.candidate)
        if not cand_pt.is_absolute():
            cand_pt = repo / cand_pt
        cand = read_artifact_axes(cand_pt)
        cand["artifact_pt"] = str(cand_pt)
    else:
        wf_manifest = (repo / args.wf_manifest) if not Path(args.wf_manifest).is_absolute() \
            else Path(args.wf_manifest)
        cand = discover_candidate(repo, wf_manifest)
        if cand is None:
            rep.verdict = (f"precondition: no candidate — WF manifest {wf_manifest} "
                           f"absent/empty and no --candidate given")
            rep.rc = RC_USAGE
            return rep
        cand_pt = Path(cand["artifact_pt"])
    rep.candidate_pt = str(cand_pt)
    if not cand_pt.exists():
        rep.verdict = f"precondition: candidate artifact {cand_pt} not found"
        rep.rc = RC_USAGE
        return rep

    # --- §3.1 freshness: source SLA + cutoff advance ---
    sources = json.loads(args.sources_json) if args.sources_json else DEFAULT_SOURCES
    for src in sources:
        if src.get("kind") == "fundamentals":
            # Two-axis P-FUND-FRESHNESS: daily-feed liveness AND quarterly filing
            # availability — NOT a generic max-date slow feed (fail-closed until a
            # real fiscal-period/available-at field exists).
            rep.source_verdicts.append(resolve_fundamentals_verdict(repo, src, now))
        else:
            cutoff = resolve_data_cutoff(repo, src)
            rep.source_verdicts.append(source_sla_verdict(src, now, cutoff))
    all_on_sla = all(v.on_sla for v in rep.source_verdicts)
    rep.advance = cutoffs_advance(served_axes, cand)

    fast_ages = [v.age_days for v in rep.source_verdicts
                 if v.axis == "fast" and v.age_days is not None]
    fast_age = max(fast_ages) if fast_ages else None

    rep.fresh = all_on_sla and rep.advance.advanced
    if rep.fresh:
        rep.labeled_non_fresh = False
    elif args.allow_non_fresh:
        rep.labeled_non_fresh = True  # served for --reason; does NOT reset the freshness clock
    else:
        rep.tier = freshness_tier(fast_age, all_sources_on_sla=all_on_sla,
                                  validated_advancing_promote=False)
        rep.verdict = ("REFUSED (not fresh): "
                       + ("; ".join(f"{v.name} {v.detail}"
                                    for v in rep.source_verdicts if not v.on_sla)
                          or rep.advance.detail)
                       + " — pass --allow-non-fresh --reason for a deliberate recipe-fix promote")
        rep.rc = RC_NOT_FRESH
        return rep

    # --- §3.4 validation gate (always, even for labeled-non-fresh) ---
    gates: list[GateResult] = []

    if args.skip_inference_gate:
        gates.append(GateResult("load_smoke_inference", True,
                                "SKIPPED via --skip-inference-gate (weakened promote)"))
        smoke = {"scores": [], "elapsed_s": 0.0, "peak_rss_mb": None}
    else:
        smoke = load_and_smoke_infer(cand_pt, served_config)
        gates.append(GateResult("load_smoke_inference", bool(smoke["ok"]), smoke["reason"]))
        # non-degenerate + resource depend on smoke succeeding
        nd_ok, nd_detail = check_non_degenerate(smoke["scores"]) if smoke["ok"] \
            else (False, "smoke inference did not produce scores")
        gates.append(GateResult("non_degenerate", nd_ok, nd_detail))
        res_ok, res_detail = check_resource(smoke["elapsed_s"], smoke["peak_rss_mb"],
                                            args.resource_max_seconds, args.resource_max_rss_mb) \
            if smoke["ok"] else (False, "no resource sample (smoke failed)")
        gates.append(GateResult("resource_bounds", res_ok, res_detail))

    # parity: lookahead/label + config fingerprint (via re-stamp against served config)
    parity_ok, parity_detail = _parity_gate(cand, cand_pt, served_config, stamp_script,
                                             apply=args.apply)
    gates.append(GateResult("schema_recipe_fingerprint_parity", parity_ok, parity_detail))

    metric = candidate_quality_metric(cand)
    floor_ok, floor_detail = check_sanity_floor(metric, args.sanity_floor)
    gates.append(GateResult("sanity_floor", floor_ok, floor_detail))

    rep.gates = gates
    rep.tier = freshness_tier(fast_age, all_sources_on_sla=all_on_sla,
                              validated_advancing_promote=(rep.fresh and all(g.ok for g in gates)))

    if not all(g.ok for g in gates):
        failed = [g.name for g in gates if not g.ok]
        rep.verdict = f"REFUSED (validation gate failed): {', '.join(failed)} — kept old pin"
        rep.rc = RC_GATE_FAILED
        return rep

    # --- all gates pass ---
    label = " [LABELED NON-FRESH]" if rep.labeled_non_fresh else ""
    if not args.apply:
        rep.verdict = f"DRY-RUN OK — would promote{label}: {served_pin} -> {cand_pt}"
        rep.rc = RC_OK
        return rep

    # --- §3.1 atomic write-new-then-swap promote ---
    try:
        promoted_pin, backup = _execute_swap(repo, cfg, served_config, args.pin_key,
                                             cand_pt, served_pt, args.served_root,
                                             stamp_script, rep, args)
    except Exception as exc:
        rep.verdict = f"REFUSED (swap failed, old pin retained): {exc}"
        rep.rc = RC_GATE_FAILED
        return rep
    rep.promoted_pin = promoted_pin
    rep.superseded_backup = backup
    rep.verdict = f"PROMOTED{label}: {served_pin} -> {promoted_pin}"
    rep.rc = RC_OK

    # strategy-104 snapshot freshness backstop (M9/A6 round 4, same as
    # weekly_wf_promote.sh/manual_promote.sh/restamp_prod_fingerprint.py):
    # this swap just changed strategy_config.shadow.json's served pin, which
    # doc/arch/strategy-104-snapshot.md's collect_snapshot() also reads.
    # Reuses promote_pin.py's scratch-rendered, diff-preview, never-auto-
    # commits check; does NOT revert the already-executed swap for a
    # stale-snapshot finding alone (this scorer is shadow-scoped and moves
    # no capital, but a stale snapshot doc is still a real drift to surface).
    sys.path.insert(0, str(repo / "scripts"))
    from promote_pin import check_snapshot_freshness  # noqa: E402

    fresh, msg = check_snapshot_freshness(sys.executable, repo=repo)
    rep.verdict += f" | snapshot: {msg}"
    if not fresh:
        rep.rc = RC_GATE_FAILED
    return rep


def _parity_gate(cand: dict, cand_pt: Path, served_config: Path, stamp_script: Path,
                 *, apply: bool) -> tuple[bool, str]:
    """§3.4(2): lookahead/label parity + config-fingerprint stamped from the CURRENT config."""
    meta_path = Path(str(cand_pt) + ".metadata.json")
    if not meta_path.exists():
        return False, f"candidate metadata sidecar missing: {meta_path.name}"
    cfg = load_json(served_config)
    live_lookahead = get_dotted(cfg, "ranking.panel_scoring.lookahead_days") \
        or cfg.get("panel_ltr", {}).get("lookahead_days") or cfg.get("lookahead_days")
    cand_lookahead = cand.get("lookahead_days")
    if live_lookahead is not None and cand_lookahead is not None \
            and int(live_lookahead) != int(cand_lookahead):
        return False, f"lookahead mismatch: live={live_lookahead} candidate={cand_lookahead}"
    # Fingerprint parity: dry-run the stamp tool (fail-closed compat check). On --apply
    # the real stamp+write happens in the swap; here we confirm it WOULD accept.
    rc, out = stamp_fingerprint(stamp_script, meta_path, served_config, write=False)
    if rc != 0:
        tail = out.strip().splitlines()[-3:]
        return False, "stamp compat check failed: " + " | ".join(tail)
    live_fp = live_config_fingerprint(served_config)
    detail = f"lookahead={cand_lookahead} label={cand.get('label_col')} stamp-compat=OK"
    if live_fp:
        detail += f" live_fp={live_fp}"
    return True, detail


def _execute_swap(repo, cfg, served_config, pin_key, cand_pt, served_pt, served_root,
                  stamp_script, rep, args) -> tuple[str, str]:
    """Write-new (copy candidate into a fresh served snapshot) -> re-stamp -> atomic
    pin swap -> retain superseded config backup + snapshot. Returns (new_pin, backup_path)."""
    root = (repo / served_root) if not Path(served_root).is_absolute() else Path(served_root)
    trained = parse_date(rep.advance.train_candidate) or dt.date.today()
    sel = rep.advance.selection_candidate
    stamp = f"promoted_{dt.datetime.utcnow():%Y%m%dT%H%M%SZ}"
    snap_name = f"pt_shadow_selcut_{sel.isoformat() if sel else 'na'}_{stamp}"
    snap_dir = root / snap_name / "seed_44"
    snap_dir.mkdir(parents=True, exist_ok=True)
    new_pt = snap_dir / cand_pt.name
    # write-new: copy .pt + sidecar + calibrator siblings
    shutil.copy2(cand_pt, new_pt)
    for sib_suffix in (".metadata.json",):
        src = Path(str(cand_pt) + sib_suffix)
        if src.exists():
            shutil.copy2(src, Path(str(new_pt) + sib_suffix))
    cal = cand_pt.parent / "hf_patchtst-calibration.json"
    if cal.exists():
        shutil.copy2(cal, snap_dir / cal.name)

    # re-stamp the NEW copy against the current pinned config (§3.3)
    new_meta = Path(str(new_pt) + ".metadata.json")
    rc, out = stamp_fingerprint(stamp_script, new_meta, served_config, write=True)
    if rc != 0:
        raise RuntimeError("re-stamp of promoted artifact failed: "
                           + " | ".join(out.strip().splitlines()[-3:]))
    # defensive parity re-check
    live_fp = live_config_fingerprint(served_config)
    if live_fp is not None:
        stamped = load_json(new_meta).get("config_fingerprint")
        if stamped != live_fp:
            raise RuntimeError(f"post-stamp fingerprint {stamped} != live {live_fp}")

    # backup old config, then atomic pin swap
    backup = served_config.with_name(
        served_config.name + f".promote-bak.{stamp}")
    backup.write_text(served_config.read_text(encoding="utf-8"), encoding="utf-8")
    # Write the pin relative to the CONFIG's directory — the runtime resolves it
    # against _strategy_dir (job_panel_scoring._resolve_artifact_path), matching the
    # existing "../../artifacts/..." convention.
    new_pin = os.path.relpath(new_pt, served_config.parent)
    new_cfg = json.loads(json.dumps(cfg))  # deep copy
    set_dotted(new_cfg, pin_key, new_pin)
    atomic_write_json(served_config, new_cfg)

    # run-bundle provenance (§5)
    _write_promote_log(repo, rep, new_pin, str(backup), args)
    return new_pin, str(backup)


def _write_promote_log(repo, rep, new_pin, backup, args) -> None:
    log_dir = repo / "logs" / "promote_shadow_patchtst"
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = rep.to_dict()
    entry["promoted_pin"] = new_pin
    entry["superseded_backup"] = backup
    entry["promoted_at"] = dt.datetime.utcnow().isoformat() + "Z"
    entry["reason"] = args.reason
    path = log_dir / f"{dt.datetime.utcnow():%Y-%m-%dT%H%M%SZ}.json"
    path.write_text(json.dumps(entry, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--served-config", default=DEFAULT_SERVED_CONFIG,
                    help="JSON config carrying the served PatchTST pin (must have "
                         "panel_scoring.kind=hf_patchtst). Default: %(default)s")
    ap.add_argument("--pin-key", default=DEFAULT_PIN_KEY,
                    help="Dotted path to the served artifact_path. Default: %(default)s")
    ap.add_argument("--wf-manifest", default=DEFAULT_WF_MANIFEST,
                    help="WF manifest to auto-discover the candidate. Default: %(default)s")
    ap.add_argument("--candidate", default=None,
                    help="Explicit candidate .pt path (overrides WF auto-discovery).")
    ap.add_argument("--served-root", default=DEFAULT_SERVED_ROOT,
                    help="Root dir for promoted served snapshots. Default: %(default)s")
    ap.add_argument("--stamp-script", default=DEFAULT_STAMP_SCRIPT)
    ap.add_argument("--sources-json", default=None,
                    help="Override recipe-required sources (JSON list of "
                         "{name,path,axis,sla_days,date_col}).")
    ap.add_argument("--fast-ceiling-days", type=int, default=FAST_CEILING_DAYS)
    ap.add_argument("--sanity-floor", type=float, default=0.0,
                    help="Minimum WF/holdout quality floor (§3.4.5). Default: %(default)s")
    ap.add_argument("--resource-max-seconds", type=float, default=120.0)
    ap.add_argument("--resource-max-rss-mb", type=float, default=4096.0)
    ap.add_argument("--allow-non-fresh", action="store_true",
                    help="Promote a non-advancing candidate for a deliberate recipe/code "
                         "fix; the pin is LABELED non-fresh and does NOT reset the "
                         "freshness clock (§3.1). Requires --reason.")
    ap.add_argument("--reason", default=None,
                    help="Required with --allow-non-fresh: the recipe/code-fix reason.")
    ap.add_argument("--skip-inference-gate", action="store_true",
                    help="Skip the load+smoke-inference gate (weakened promote; logged). "
                         "For environments without the torch runtime.")
    ap.add_argument("--apply", action="store_true",
                    help="Actually swap the served pin (default: dry-run).")
    ap.add_argument("--check", action="store_true",
                    help="Verbose dry-run: run every gate and print the full verdict.")
    ap.add_argument("--json", action="store_true", help="Emit the report as JSON.")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    args.now = dt.date.today()
    if args.allow_non_fresh and not args.reason:
        print("ERROR: --allow-non-fresh requires --reason", file=sys.stderr)
        return RC_USAGE
    rep = run_promote(args)

    if args.json:
        print(json.dumps(rep.to_dict(), indent=2))
    else:
        print(f"═══ promote_shadow_patchtst — {rep.verdict} ═══")
        print(f"  served pin : {rep.served_pin}")
        print(f"  candidate  : {rep.candidate_pt}")
        print(f"  fresh={rep.fresh} labeled_non_fresh={rep.labeled_non_fresh} tier={rep.tier}")
        for v in rep.source_verdicts:
            print(f"  source[{v.axis}] {v.name}: {v.detail}")
        if rep.advance is not None:
            print(f"  cutoff advance: {rep.advance.detail}")
        for g in rep.gates:
            print(f"  gate {g.name}: {'PASS' if g.ok else 'FAIL'} — {g.detail}")
        if rep.promoted_pin:
            print(f"  promoted -> {rep.promoted_pin}")
            print(f"  superseded config backup: {rep.superseded_backup}")
    return rep.rc


if __name__ == "__main__":
    raise SystemExit(main())
