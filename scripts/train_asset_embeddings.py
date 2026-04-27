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
    args = p.parse_args()

    strategy_dir = REPO_ROOT / "backtesting" / args.strategy
    if str(strategy_dir) not in sys.path:
        sys.path.insert(0, str(strategy_dir))

    config_path = strategy_dir / "strategy_config.json"
    if not config_path.exists():
        log.error("strategy config missing: %s", config_path)
        return 2
    config = json.loads(config_path.read_text())
    config["_strategy_dir"] = str(strategy_dir)

    # Load OHLCV via the same data layer the panel pipeline uses
    import pandas as pd  # noqa: PLC0415
    from kernel.data import fetch_ohlcv  # noqa: PLC0415
    from training_panel.asset_embeddings import AssetEmbeddingTrainer  # noqa: PLC0415

    watchlist = list(config.get("watchlist", []))
    benchmark = config.get("benchmark", "SPY")
    if benchmark not in watchlist:
        watchlist = watchlist + [benchmark]

    log.info("Fetching OHLCV for %d tickers (watchlist + benchmark)", len(watchlist))
    ohlcv: dict[str, pd.DataFrame] = {}
    for ticker in watchlist:
        try:
            df = fetch_ohlcv(ticker, config=config, allow_fetch=False)
            if df is not None and not df.empty:
                ohlcv[ticker] = df
        except Exception as exc:
            log.warning("fetch_ohlcv(%s) failed: %s — skipping", ticker, exc)

    log.info("Loaded OHLCV for %d/%d tickers", len(ohlcv), len(watchlist))

    if len(ohlcv) < 10:
        log.error("Too few tickers (%d) — refusing to train embeddings", len(ohlcv))
        return 2

    as_of_date = pd.Timestamp(args.as_of) if args.as_of else pd.Timestamp.utcnow()

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
