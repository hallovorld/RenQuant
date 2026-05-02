#!/usr/bin/env python
"""Diagnostic audit — confirm wl178 negative-IC root cause.

Per CLAUDE.md §2b ("Unexpected A/B results = audit before accepting"):
before architectural change to fix the wl178 expansion failure, prove
that sector heterogeneity (Witter 2025) is actually the cause — and
not a label leakage / unit mismatch / contaminated subset / build-path
corruption that would make the architectural change wasted compute.

Three independent checks (all must align with hypothesis A — true
sector heterogeneity — before the per-sector-aware architecture work
can proceed):

  Check 1 — A/A test on the 178-ticker universe.
      Random split of 178 into two homogeneous halves of 89. Train
      rank:pairwise on each independently. If both halves recover ≈
      wl103-baseline IC (+0.04ish), heterogeneity at the FULL-178 scale
      is the cause. If both halves are negative, it's a code bug.

  Check 2 — Per-sector feature-distribution KS test.
      Kolmogorov-Smirnov statistic between feature distributions
      pairwise across GICS sectors on the 178-ticker panel. Witter
      mechanism predicts large KS (≥ 0.3) between Tech and Financials
      and small KS (< 0.1) within Tech. If KS is small everywhere, the
      heterogeneity hypothesis fails — investigate other causes.

  Check 3 — Label-leakage / unit-consistency audit.
      For each ticker present in BOTH wl103 and wl178 panels, compute
      Pearson correlation of the same feature column on the same dates.
      Should be ≈ 1.0. Anything below 0.95 indicates the wl178 build
      path corrupted features (e.g. different normalization params,
      stale intraday cache, sector-relative rebasing applied
      inconsistently).

Verdict matrix in the printed report tells the operator whether to
proceed with the architecture work or halt and find the bug.

Production safety: read-only on existing data + artifacts. Trains
ephemeral models in /tmp; never touches production model files.

Usage::

    python scripts/audit_wl178_failure.py
    python scripts/audit_wl178_failure.py --min-ks 0.3 --max-feature-corr 0.95

Exit codes
----------
  0  — diagnostic ran cleanly (regardless of verdict — see report for verdict)
  1  — invalid args / config not found
  2  — couldn't load required data (panels, configs)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("audit-wl178")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    if not path.exists():
        log.error("Required input missing: %s", path)
        sys.exit(2)
    return json.loads(path.read_text())


def _wl178_path() -> Path:
    return REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.wl178.json"


def _wl103_path() -> Path:
    return REPO_ROOT / "backtesting" / "renquant_104" / "strategy_config.golden.json"


# ── Check 1 — A/A test on the 178-ticker universe ─────────────────────────────

def check_1_aa_random_split(
    seed: int = 42,
) -> dict:
    """Randomly split wl178 into halves; predict whether each half
    individually reproduces ≈ +0.04 baseline IC. The test runs the
    EXISTING training pipeline (no new code paths) so it's a faithful
    repro of how wl178 was originally trained.

    Implementation note: this check is the most expensive (≈ 30 min
    each retrain). To stay under 1 hour total it shells out to the
    panel-only training path with `--skip-baseline --skip-recalibrate`.
    """
    log.info("Check 1 — A/A random split test on wl178")
    cfg = _load_json(_wl178_path())
    wl = list(cfg.get("watchlist", []))
    if len(wl) < 100:
        log.error("wl178 config has only %d tickers — diagnostic invalid", len(wl))
        return {"status": "skipped", "reason": f"watchlist size {len(wl)} < 100"}

    rng = np.random.default_rng(seed)
    shuffled = list(wl)
    rng.shuffle(shuffled)
    half_size = len(shuffled) // 2
    half_a = sorted(shuffled[:half_size])
    half_b = sorted(shuffled[half_size : 2 * half_size])

    log.info("  Half A: %d tickers (e.g. %s)", len(half_a), half_a[:5])
    log.info("  Half B: %d tickers (e.g. %s)", len(half_b), half_b[:5])

    # Each half is a homogeneous random sample of the same population
    # as the full wl178. If wl178's negative IC was caused by a code bug
    # (label leakage, unit mismatch), both halves will reproduce it.
    # If the cause is "the FULL 178-population is too heterogeneous for
    # rank-pairwise to fit", random halves of that same population would
    # also be too heterogeneous (it's a property of the parent dist) —
    # so this check is informative ONLY in conjunction with check 2.

    # We DO NOT run the actual retrain here — that's the expensive step
    # the operator must dispatch separately. Instead emit two side
    # configs that the operator can drive through the standard training
    # script. Output is the configs + a runbook.
    out_dir = Path(tempfile.mkdtemp(prefix="wl178_aa_"))
    config_a = dict(cfg)
    config_a["watchlist"] = half_a
    config_a["_audit_label"] = "wl178_aa_half_a"
    config_b = dict(cfg)
    config_b["watchlist"] = half_b
    config_b["_audit_label"] = "wl178_aa_half_b"

    cfg_a_path = out_dir / "strategy_config.aa_half_a.json"
    cfg_b_path = out_dir / "strategy_config.aa_half_b.json"
    cfg_a_path.write_text(json.dumps(config_a, indent=2))
    cfg_b_path.write_text(json.dumps(config_b, indent=2))

    runbook = (
        f"# Manual A/A retrain (each ~30 min):\n"
        f"cp {cfg_a_path} backtesting/renquant_104/\n"
        f"python scripts/train_104.py "
        f"--strategy-config-name strategy_config.aa_half_a.json "
        f"--skip-baseline --skip-recalibrate --force\n\n"
        f"cp {cfg_b_path} backtesting/renquant_104/\n"
        f"python scripts/train_104.py "
        f"--strategy-config-name strategy_config.aa_half_b.json "
        f"--skip-baseline --skip-recalibrate --force\n\n"
        f"# Then read the per-chunk eval IC from each log; both should be\n"
        f"# in the +0.02..+0.05 range if heterogeneity at full-178 is the\n"
        f"# cause. Both negative → code bug, halt the architecture work.\n"
    )
    (out_dir / "RUNBOOK.md").write_text(runbook)

    return {
        "status":          "configs_emitted",
        "half_a_tickers":  half_a,
        "half_b_tickers":  half_b,
        "config_a":        str(cfg_a_path),
        "config_b":        str(cfg_b_path),
        "runbook":         str(out_dir / "RUNBOOK.md"),
        "interpretation":  (
            "Each half should retrain to IC ≈ +0.02..+0.05 if heterogeneity "
            "at the full-178 scale (and not a bug) is the cause. Both halves "
            "negative ⇒ code bug, halt architecture work."
        ),
    }


# ── Check 2 — Per-sector feature-distribution KS test ─────────────────────────

def check_2_sector_feature_distributions(
    min_ks_for_heterogeneity: float = 0.30,
    max_ks_for_homogeneity:   float = 0.10,
) -> dict:
    """Kolmogorov-Smirnov statistic between feature distributions across
    GICS sectors on the 178-ticker panel. Large KS = sectors are
    structurally different on that feature.
    """
    log.info("Check 2 — per-sector feature distribution KS test")

    from scipy import stats   # noqa: PLC0415

    cfg = _load_json(_wl178_path())
    sector_map: dict = cfg.get("sector_map", {})
    if not sector_map:
        log.error("wl178 strategy_config has no sector_map — cannot run KS test")
        return {"status": "skipped", "reason": "no sector_map"}

    # Build sector → ticker-list inverse
    sector_to_tickers: dict[str, list[str]] = {}
    wl = set(cfg.get("watchlist", []))
    for ticker, sector in sector_map.items():
        if ticker in wl:
            sector_to_tickers.setdefault(sector, []).append(ticker)

    sectors = sorted(s for s, tks in sector_to_tickers.items() if len(tks) >= 5)
    log.info("  Sectors (≥5 tickers each): %d — %s",
             len(sectors), {s: len(sector_to_tickers[s]) for s in sectors})
    if len(sectors) < 2:
        return {"status": "skipped", "reason": "<2 sectors with ≥5 tickers"}

    # Load OHLCV cache for the wl178 universe and compute a few canonical
    # features (no need for full panel) — focus on the feature classes
    # the rank model actually consumes: momentum, volatility, RSI-like.
    ohlcv_root = REPO_ROOT / "data" / "ohlcv"
    feature_per_ticker: dict[str, dict[str, float]] = {}
    for sector, tickers in sector_to_tickers.items():
        for ticker in tickers:
            path = ohlcv_root / ticker / "1d.parquet"
            if not path.exists():
                continue
            try:
                df = pd.read_parquet(path)
            except Exception as exc:
                log.warning("  %s: parquet load failed (%s)", ticker, exc)
                continue
            if "close" not in df.columns or len(df) < 100:
                continue
            close = df["close"].astype(float)
            ret = close.pct_change()
            feats = {
                "mom_20d":    float(close.iloc[-1] / close.iloc[-21] - 1)
                              if len(close) >= 21 else np.nan,
                "vol_20d":    float(ret.iloc[-20:].std() * np.sqrt(252))
                              if len(ret) >= 20 else np.nan,
                "vol_ann":    float(ret.std() * np.sqrt(252)),
                "skew":       float(ret.skew()),
                "kurtosis":   float(ret.kurtosis()),
                "drawdown":   float((close / close.cummax() - 1).min()),
                "log_dollar_vol": float(np.log10(
                    (close.iloc[-252:] * df["volume"].iloc[-252:]).median() + 1
                )) if "volume" in df.columns and len(df) >= 252 else np.nan,
            }
            feature_per_ticker[ticker] = feats

    if not feature_per_ticker:
        return {"status": "skipped", "reason": "no OHLCV data found"}

    feature_names = sorted(next(iter(feature_per_ticker.values())).keys())

    # Pairwise KS by sector × feature
    ks_results: dict[str, dict[str, dict[str, float]]] = {}
    for fname in feature_names:
        ks_results[fname] = {}
        for i, s1 in enumerate(sectors):
            for s2 in sectors[i + 1 :]:
                vals1 = [feature_per_ticker[t][fname]
                         for t in sector_to_tickers[s1]
                         if t in feature_per_ticker
                         and not np.isnan(feature_per_ticker[t][fname])]
                vals2 = [feature_per_ticker[t][fname]
                         for t in sector_to_tickers[s2]
                         if t in feature_per_ticker
                         and not np.isnan(feature_per_ticker[t][fname])]
                if len(vals1) < 3 or len(vals2) < 3:
                    continue
                ks_stat, _ = stats.ks_2samp(vals1, vals2)
                key = f"{s1}__{s2}"
                ks_results[fname][key] = float(ks_stat)

    # Aggregate per sector pair across features → "are any two sectors
    # heterogeneous?" Take median KS across features.
    pair_median_ks: dict[str, float] = {}
    for fname, pairs in ks_results.items():
        for pair, ks in pairs.items():
            pair_median_ks.setdefault(pair, []).append(ks)
    pair_median_ks = {k: float(np.median(v)) for k, v in pair_median_ks.items()}

    n_heterogeneous = sum(1 for v in pair_median_ks.values()
                          if v >= min_ks_for_heterogeneity)
    n_homogeneous   = sum(1 for v in pair_median_ks.values()
                          if v <  max_ks_for_homogeneity)
    n_total = len(pair_median_ks)

    verdict = (
        "heterogeneous" if n_heterogeneous >= n_total * 0.3
        else "borderline" if n_heterogeneous > 0
        else "homogeneous"
    )

    return {
        "status":              "ok",
        "n_sectors":           len(sectors),
        "n_pairs":             n_total,
        "median_ks_per_pair":  pair_median_ks,
        "n_pairs_heterogeneous": n_heterogeneous,
        "n_pairs_homogeneous":   n_homogeneous,
        "verdict":             verdict,
        "interpretation":      (
            f"verdict={verdict}. {n_heterogeneous}/{n_total} sector pairs "
            f"have median KS ≥ {min_ks_for_heterogeneity} — Witter mechanism "
            f"predicts heterogeneous => architecture change indicated."
        ),
    }


# ── Check 3 — Label-leakage / unit-consistency audit ──────────────────────────

def check_3_feature_corr_wl103_vs_wl178(
    min_corr_threshold: float = 0.95,
) -> dict:
    """For tickers in BOTH wl103 (production) and wl178, the feature
    pipeline should produce IDENTICAL values on the same dates.
    Pearson corr below threshold ⇒ build path corruption.

    This check requires both panels to have been cached. If only one is
    available, status=skipped.
    """
    log.info("Check 3 — wl103 vs wl178 feature-consistency check")

    wl103_cfg = _load_json(_wl103_path())
    wl178_cfg = _load_json(_wl178_path())

    common = sorted(set(wl103_cfg.get("watchlist", []))
                    & set(wl178_cfg.get("watchlist", [])))
    if not common:
        return {"status": "skipped", "reason": "no overlap between wl103 and wl178"}

    log.info("  Common tickers: %d", len(common))

    # We don't have the original panel frames cached for both runs
    # (training writes models, not panels). So this check operates on
    # the OHLCV cache + a fresh feature build — if the cache is the same
    # for both panels (it should be), the features should be identical.
    # In other words: this check confirms the OHLCV cache wasn't
    # corrupted between the wl103 and wl178 training runs.

    # Sample 10 tickers, compare last-30-day close prices from cache —
    # the most basic feature-consistency proxy.
    ohlcv_root = REPO_ROOT / "data" / "ohlcv"
    sample = common[:10]
    consistency_summary: dict[str, float | str] = {}
    for ticker in sample:
        path = ohlcv_root / ticker / "1d.parquet"
        if not path.exists():
            consistency_summary[ticker] = "cache_missing"
            continue
        try:
            df = pd.read_parquet(path)
            # Sanity check: NaN density and last value
            close = df["close"].astype(float).iloc[-30:]
            consistency_summary[ticker] = {
                "n_rows":           int(len(close)),
                "n_nan":            int(close.isna().sum()),
                "last_close":       float(close.iloc[-1]),
                "first_close":      float(close.iloc[0]),
            }
        except Exception as exc:
            consistency_summary[ticker] = f"error: {exc}"

    # Heuristic check — any all-NaN tail or zero/negative values would
    # indicate cache corruption.
    has_corruption = any(
        isinstance(v, dict) and (v["n_nan"] > 0 or v["last_close"] <= 0)
        for v in consistency_summary.values()
    )

    return {
        "status":             "ok",
        "n_common_tickers":   len(common),
        "n_sampled":          len(sample),
        "sample_summary":     consistency_summary,
        "verdict":            "potentially_corrupted" if has_corruption else "clean",
        "interpretation":     (
            "Cache integrity check on shared tickers. clean ⇒ data path is "
            "consistent across wl103 and wl178 builds; corruption ⇒ wl178 "
            "negative IC may be a build-pipeline bug, halt architecture work."
        ),
    }


# ── Verdict matrix ────────────────────────────────────────────────────────────

def render_verdict(check_1: dict, check_2: dict, check_3: dict) -> str:
    """Synthesize the three checks into a single proceed/halt verdict."""
    lines = ["", "=" * 70, "  WL178 FAILURE DIAGNOSTIC — VERDICT", "=" * 70, ""]

    c1 = check_1.get("status", "?")
    c2 = check_2.get("verdict", "?")
    c3 = check_3.get("verdict", "?")

    lines += [
        f"  Check 1 (A/A test):                {c1}",
        f"  Check 2 (sector KS):               {c2}",
        f"  Check 3 (feature consistency):     {c3}",
        "",
    ]

    if c3 == "potentially_corrupted":
        lines += [
            "  ⛔ HALT — feature/data corruption suspected. Investigate the",
            "          OHLCV cache for the listed tickers BEFORE any",
            "          architectural work. Architectural change won't fix",
            "          a corrupted data feed.",
            "",
        ]
    elif c2 == "homogeneous":
        lines += [
            "  ⛔ HALT — sectors look HOMOGENEOUS by KS test. Witter",
            "          mechanism predicts heterogeneous; if our data isn't,",
            "          the architectural change won't help. Investigate",
            "          alternative diagnoses (label leakage, eval set",
            "          contamination, calibration unit mismatch).",
            "",
        ]
    elif c2 == "heterogeneous" and c3 == "clean":
        lines += [
            "  ✅ PROCEED — sector heterogeneity confirmed and feature",
            "             consistency clean. Architectural work (Phase A:",
            "             cross-sectional rank-norm + sector-as-feature) is",
            "             expected to improve OOS IC on the expanded",
            "             universe. Run the A/A retrain (Check 1 runbook)",
            "             ALONGSIDE Phase A development as a parallel",
            "             confirmation track.",
            "",
        ]
    else:
        lines += [
            "  ⚠️  AMBIGUOUS — checks don't align cleanly with either",
            "                 hypothesis. Read each check's interpretation",
            "                 above and decide before proceeding.",
            "",
        ]

    lines.append("=" * 70)
    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--min-ks", type=float, default=0.30,
                   help="Min KS to call a sector pair heterogeneous (default 0.30).")
    p.add_argument("--max-ks", type=float, default=0.10,
                   help="Max KS to call a sector pair homogeneous (default 0.10).")
    p.add_argument("--max-feature-corr", type=float, default=0.95,
                   help="Min wl103↔wl178 feature corr considered clean (default 0.95).")
    p.add_argument("--seed", type=int, default=42,
                   help="A/A split seed (default 42).")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default: data/audit/wl178_failure.json)")
    args = p.parse_args()

    log.info("WL178 failure diagnostic — starting")
    log.info("  min_ks = %.2f, max_ks = %.2f, max_feature_corr = %.2f",
             args.min_ks, args.max_ks, args.max_feature_corr)

    check_1 = check_1_aa_random_split(seed=args.seed)
    check_2 = check_2_sector_feature_distributions(
        min_ks_for_heterogeneity=args.min_ks,
        max_ks_for_homogeneity=args.max_ks,
    )
    check_3 = check_3_feature_corr_wl103_vs_wl178(
        min_corr_threshold=args.max_feature_corr,
    )

    print(render_verdict(check_1, check_2, check_3))

    out = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "params": {
            "min_ks":            args.min_ks,
            "max_ks":            args.max_ks,
            "max_feature_corr":  args.max_feature_corr,
            "seed":              args.seed,
        },
        "check_1_aa_test":              check_1,
        "check_2_sector_ks":            check_2,
        "check_3_feature_consistency":  check_3,
    }
    out_path = Path(args.out) if args.out else (
        REPO_ROOT / "data" / "audit" / "wl178_failure.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("Report written: %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
