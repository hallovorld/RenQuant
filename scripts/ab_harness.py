"""Reusable A/B sim harness — isolated artifacts + multi-variant.

Diagnosed 2026-04-24: direct `/tmp/ab.py` scripts reading live
artifacts directly produce non-reproducible results because the
user's notebook retrains in parallel, overwriting panel/NGBoost/GMM
files. Every call to this harness:

1. **Snapshots artifacts** into a tmp dir at the start (via
   `snapshot_artifacts_ctx`). All variants see the SAME frozen
   artifact set regardless of concurrent retraining.
2. **Builds panel frames ONCE** — shared across all variants for
   the A/B (not a new snapshot for each variant).
3. **Runs each variant** with a `config_mutator` function — a closure
   that takes a deep-copied base config and mutates it.
4. **Cleans up** the snapshot on exit (context manager guarantees).

Usage example (run directly OR import)::

    from scripts.ab_harness import run_ab

    variants = [
        ("A_GOLDEN",          lambda c: None),
        ("B_flag_enabled",    lambda c: c["ranking"]["kelly_sizing"].__setitem__(
            "trim_enabled", True)),
    ]
    results = run_ab(
        strategy="renquant_104",
        variants=variants,
    )
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("ab-harness")


def _load_cached_ohlcv(symbols: set[str], cache_root: Path) -> dict:
    out = {}
    for s in symbols:
        p = cache_root / s / "1d.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if not df.empty:
            out[s] = df
    return out


def run_ab(
    strategy: str = "renquant_104",
    variants: "list[tuple[str, callable]] | None" = None,
    initial_cash: float = 100_000.0,
    cache_root: "Path | None" = None,
) -> list[dict]:
    """Run an A/B sweep with artifact isolation.

    Args:
        strategy: subdir under `backtesting/` (e.g. "renquant_104").
        variants: list of `(label, mutator_fn)`. mutator takes a dict and
                  mutates in place. `lambda c: None` = baseline (no-op).
        initial_cash: starting portfolio value.
        cache_root: where to find OHLCV parquets (default `data/ohlcv`).

    Returns: list of result dicts with `label, apy, final, win, buys, sells,
    rotations, streak, elapsed, snapshot_sha`.
    """
    if variants is None or not variants:
        raise ValueError("Must specify at least one variant")

    strategy_dir = REPO_ROOT / "backtesting" / strategy
    cache_root   = cache_root or (REPO_ROOT / "data" / "ohlcv")

    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    from kernel.artifact_snapshot import snapshot_artifacts_ctx  # noqa: PLC0415
    from kernel.config import load_strategy_config               # noqa: PLC0415
    from training_panel.pipeline import prepare_inference_panel_frames  # noqa: PLC0415
    from sim.runner import run_backtest                          # noqa: PLC0415

    results: list[dict] = []

    with snapshot_artifacts_ctx(strategy_dir) as snap:
        log.info("Snapshot: %s", snap)
        sha = (snap / ".snapshot_sha").read_text().strip() if (snap / ".snapshot_sha").exists() else "unknown"
        log.info("Commit at snapshot time: %s", sha[:12])

        # Load config FROM SNAPSHOT — not from live dir
        cfg_base = load_strategy_config(snap / "strategy_config.json")
        cfg_base["_strategy_dir"] = str(snap)
        cfg_base.setdefault("initial_cash", initial_cash)

        symbols = (set(cfg_base["watchlist"])
                    | set(cfg_base.get("sector_etf_map", {}).values())
                    | {"SPY"})

        log.info("Loading OHLCV for %d symbols...", len(symbols))
        ohlcv = _load_cached_ohlcv(symbols, cache_root)
        log.info("  loaded %d/%d", len(ohlcv), len(symbols))

        log.info("Building panel frames (shared across variants)...")
        ticker_sectors = {t: cfg_base["sector_map"][t] for t in cfg_base["watchlist"]
                          if t in cfg_base.get("sector_map", {})}
        ff, fac = prepare_inference_panel_frames(
            watchlist=cfg_base["watchlist"], ohlcv=ohlcv,
            ticker_sectors=ticker_sectors,
            config={**cfg_base, "_strategy_dir": str(snap)},
        )

        for label, mutator in variants:
            cfg = copy.deepcopy(cfg_base)
            if mutator is not None:
                mutator(cfg)
            t0 = time.time()
            # Audit fix VALIDATE-SNAPSHOT-OVERRIDE (2026-04-26):
            # snapshot=False because we're already INSIDE a
            # snapshot_artifacts_ctx (line 99) — re-snapshotting would
            # discard the variant mutator's config flips. Pre-fix,
            # variants would have all run cfg_base instead of mutator(cfg).
            r = run_backtest(
                config=cfg, strategy_dir=snap, ohlcv=ohlcv,
                spy_df=ohlcv["SPY"], sector_etf_map=cfg.get("sector_etf_map", {}),
                panel_feature_frames=ff, panel_factor_frames=fac,
                snapshot=False,
            )
            results.append({
                "label":     label,
                "apy":       r.apy,
                "final":     r.final_value,
                "win":       r.win_rate,
                "buys":      len(r.buys),
                "sells":     len(r.sells),
                "rotations": len(r.rotations),
                "streak":    r.longest_no_trade_streak,
                "elapsed":   time.time() - t0,
                "snapshot_sha": sha[:12],
            })
            log.info("  %s  →  APY %+.2f%%  %d buys  %d rot",
                     label, r.apy * 100, len(r.buys), len(r.rotations))

    # Pretty table on exit
    if results:
        baseline_apy = results[0]["apy"]
        print(f"\n{'Variant':<30} {'APY':>9} {'ΔAPY':>9} {'Final':>12} {'Win':>5} "
              f"{'Buys':>5} {'Sells':>6} {'Rot':>5} {'Streak':>7} {'Time':>7}")
        print("─" * 108)
        for r in results:
            d = r["apy"] - baseline_apy
            print(f"{r['label']:<30} {r['apy']*100:>+8.2f}% {d*100:>+8.2f}% "
                  f"${r['final']:>10,.0f} {r['win']*100:>4.0f}% {r['buys']:>5} "
                  f"{r['sells']:>6} {r['rotations']:>5} {r['streak']:>6}d "
                  f"{r['elapsed']:>6.1f}s")
        print(f"\n(artifacts snapshot @ commit {results[0]['snapshot_sha']})")

    return results


def _cli() -> None:
    """Self-test — runs a minimal 1-variant A/B to verify the harness."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="renquant_104")
    args = p.parse_args()
    variants = [("A_BASELINE_SMOKE", lambda c: None)]
    run_ab(strategy=args.strategy, variants=variants)


if __name__ == "__main__":
    _cli()
