#!/usr/bin/env python
"""Train asset embeddings for the renquant_104 watchlist.

T2-2 Phase B from doc/components/asset-embeddings-design.md. CLI driver
for AssetEmbeddingTrainer. Reads watchlist + OHLCV cache, trains
contrastive embeddings, persists to artifacts/asset-embeddings.json.

Run weekly via cron (proposed: Sunday 12:30 PT, after screen_watchlist).

Usage::

    python scripts/train_asset_embeddings.py
    python scripts/train_asset_embeddings.py --strategy renquant_104
    python scripts/train_asset_embeddings.py --as-of 2026-04-26
    python scripts/train_asset_embeddings.py --embedding-dim 32 --epochs 50

Exit codes:
    0  — success, embeddings persisted, smoke check passed
    2  — too few tickers had enough history; no embeddings written
    3  — collapse smoke check failed (mean cosine > 0.95) — artifact NOT saved
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("train-asset-embeddings")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategy",      default="renquant_104")
    p.add_argument("--as-of",         default=None,
                   help="ISO date; default = today UTC")
    p.add_argument("--embedding-dim", type=int, default=16)
    p.add_argument("--lookback-days", type=int, default=504)
    p.add_argument("--epochs",        type=int, default=30)
    p.add_argument("--out",           default=None,
                   help="Override artifact output path "
                        "(default: artifacts/asset-embeddings.json under strategy dir)")
    p.add_argument("--no-smoke-fail", action="store_true",
                   help="Persist artifact even if smoke test fails (debugging)")
    p.add_argument("--strategy-config-name", default="strategy_config.json",
                   help="Filename of the strategy config under the strategy dir.")
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    config_path = strategy_dir / args.strategy_config_name
    if not config_path.exists():
        log.error("strategy config missing: %s", config_path)
        return 2
    config = json.loads(config_path.read_text())
    config["_strategy_dir"] = str(strategy_dir)

    # Load OHLCV via the same data layer the panel pipeline uses
    import pandas as pd  # noqa: PLC0415
    from kernel.data import LocalStore, fetch_ohlcv  # noqa: PLC0415
    from training_panel.asset_embeddings import AssetEmbeddingTrainer  # noqa: PLC0415

    watchlist = list(config.get("watchlist", []))
    benchmark = config.get("benchmark", "SPY")
    if benchmark not in watchlist:
        watchlist = watchlist + [benchmark]

    log.info("Fetching OHLCV for %d tickers (watchlist + benchmark)", len(watchlist))
    # BUG FIX (T2-2): original call used unsupported kwargs `config=` and
    # `allow_fetch=False` which don't exist in fetch_ohlcv's signature.
    # Use LocalStore.load() directly for cache-only access (no network fetch).
    # 2026-04-27: fall back to repo-root data/ohlcv when the strategy-local
    # cache is missing tickers (the strategy cache only covers the original
    # 56-ticker watchlist, but the canonical cache at REPO_ROOT/data/ohlcv
    # has the full 114).
    primary_store = LocalStore(data_dir=strategy_dir / "data" / "ohlcv")
    fallback_store = LocalStore(data_dir=REPO_ROOT / "data" / "ohlcv")
    ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in watchlist:
        try:
            df = primary_store.load(ticker)
            if (df is None or df.empty) and fallback_store.data_dir != primary_store.data_dir:
                df = fallback_store.load(ticker)
            if df is not None and not df.empty:
                ohlcv[ticker] = df
        except Exception as exc:
            log.warning("fetch_ohlcv(%s) failed: %s — skipping", ticker, exc)

    log.info("Loaded OHLCV for %d/%d tickers", len(ohlcv), len(watchlist))

    if len(ohlcv) < 10:
        log.error("Too few tickers (%d) — refusing to train embeddings", len(ohlcv))
        return 2

    # BUG FIX (2026-04-27): pd.Timestamp.utcnow() is tz-aware; the OHLCV
    # parquet index is tz-naive — comparison raises TypeError. Strip tz.
    as_of_date = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.now()
    if getattr(as_of_date, "tz", None) is not None:
        as_of_date = as_of_date.tz_localize(None)

    log.info("Training %dD embeddings (lookback %dd, epochs %d, as_of %s)",
             args.embedding_dim, args.lookback_days, args.epochs,
             as_of_date.date().isoformat())

    trainer = AssetEmbeddingTrainer(
        embedding_dim = args.embedding_dim,
        lookback_days = args.lookback_days,
        n_epochs      = args.epochs,
    )

    embeddings = trainer.fit(ohlcv, as_of_date)
    if not embeddings:
        log.error("Trainer returned empty — no ticker had enough history "
                  "(min %d bars). Aborting.", args.lookback_days + 30)
        return 2

    # Smoke-test collapse
    healthy = trainer.smoke_test_collapse()
    if not healthy and not args.no_smoke_fail:
        log.error("Embeddings COLLAPSED (mean off-diagonal cosine > 0.95). "
                  "Artifact NOT saved. Re-run with different hyperparams or "
                  "--no-smoke-fail to override.")
        return 3

    # Persist
    out_path = Path(args.out) if args.out else (
        strategy_dir / "artifacts" / "asset-embeddings.json"
    )
    trainer.save(out_path)
    log.info("Persisted %d-dim embeddings for %d tickers → %s",
             args.embedding_dim, len(embeddings), out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
