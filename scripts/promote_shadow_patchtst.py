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
SLOW_SLA_DAYS = 55              # quarterly fundamentals filing-calendar SLA (~45d + buffer)

# Recipe-required sources the served shadow model's data cutoff is capped by (RFC §3.1).
# ``date_col`` present -> read the max of that column as the data cutoff; else file mtime.
DEFAULT_SOURCES: list[dict] = [
    {"name": "transformer_panel", "path": "data/transformer_v4_wl200_clean.parquet",
     "axis": "fast", "sla_days": FAST_CEILING_DAYS, "date_col": "date"},
    {"name": "rawlabel", "path": "data/alpha158_291_fundamental_dataset_rawlabel.parquet",
     "axis": "fast", "sla_days": FAST_CEILING_DAYS, "date_col": "date"},
    {"name": "fundamentals", "path": "data/sec_fundamentals_daily.parquet",
     "axis": "slow", "sla_days": SLOW_SLA_DAYS, "date_col": "date"},
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


def resolve_data_cutoff(repo: Path, source: dict) -> dt.date | None:
    """Resolve a source's data cutoff: max(date_col) for parquet, else file mtime."""
    p = (repo / source["path"]) if not Path(source["path"]).is_absolute() else Path(source["path"])
    if not p.exists():
        return None
    date_col = source.get("date_col")
    if date_col and p.suffix == ".parquet":
        try:
            import pandas as pd  # noqa: PLC0415
            df = pd.read_parquet(p, columns=[date_col])
            return pd.to_datetime(df[date_col]).max().date()
        except Exception:
            pass  # fall through to mtime
    return dt.date.fromtimestamp(p.stat().st_mtime)


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
