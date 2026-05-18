#!/usr/bin/env python3
"""Score news headlines with FinBERT, aggregate per-ticker-per-date.

Roadmap C5 step 2 (2026-05-18, user mandate $0/mo).

Input:  data/news_alpaca/{ticker}.parquet  (from fetch_news_alpaca.py)
Output: data/news_sentiment_alpaca/{ticker}.parquet

Per-day features:
  • mean_sentiment       — average per-article signed score in [-1, +1]
  • sentiment_dispersion — std-dev (Garcia 2013 "disagreement" proxy)
  • n_articles           — count of articles that day
  • sentiment_pos_share  — fraction with score > +0.2
  • sentiment_neg_share  — fraction with score < -0.2

Per-article signed score = (P_pos - P_neg). Standard FinBERT
convention (Araci 2019 *arXiv 1908.10063*).

Model: ProsusAI/finbert via HuggingFace. 440MB, 110M params,
3-class output: positive / neutral / negative.

Concurrency: batched inference on MPS (Apple GPU) when available,
otherwise CPU multi-thread. Default batch=64.

Sanity gate (CLAUDE.md §5.2): reject scored output if degenerate
distribution (all zero / all saturated / >95% zero). Logs warning
and does NOT write the parquet — operator must investigate.

References:
  - Araci 2019 "FinBERT: Financial Sentiment Analysis with Pre-trained
    Language Models" arXiv 1908.10063
  - Garcia 2013 *J. Finance* "Sentiment During Recessions"
  - Tetlock 2007 *J. Finance* "Giving Content to Investor Sentiment"
  - Ke-Kelly-Xiu 2019 NBER w26261 "Predicting Returns with Text Data"

CLI:
  # Score all backfilled news, output to data/news_sentiment_alpaca/
  python scripts/score_news_finbert.py

  # Limit to specific symbols
  python scripts/score_news_finbert.py --symbols AAPL META TSLA

  # Override batch size (default 64; MPS-friendly, ~6GB peak VRAM)
  python scripts/score_news_finbert.py --batch-size 32

  # Dry-run: no writes (sanity check only)
  python scripts/score_news_finbert.py --dry-run
"""
from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
IN_DIR = REPO / "data" / "news_alpaca"
OUT_DIR = REPO / "data" / "news_sentiment_alpaca"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("score_news_finbert")

DEFAULT_BATCH_SIZE = 64
POS_THRESHOLD = 0.2
NEG_THRESHOLD = -0.2

# Sanity-gate thresholds
SAT_FRACTION_LIMIT = 0.95  # > 95% at one value = degenerate
ZERO_FRACTION_LIMIT = 0.95  # > 95% exactly 0 = tokenizer fail


# ── Pure helpers (model-free, fully tested) ──────────────────────────────

def probs_to_signed(p_pos: float, p_neu: float, p_neg: float) -> float:
    """3-class probs → signed scalar in [-1, +1].

    Standard FinBERT convention. P_neu is excluded; we keep only the
    directional contrast P_pos - P_neg. This means a "60% positive
    40% neutral 0% negative" headline scores +0.6, NOT +1.0.
    """
    return float(p_pos - p_neg)


def is_skippable_text(text) -> bool:
    """Skip empty / whitespace / None headlines (scores neutral)."""
    if text is None:
        return True
    s = str(text).strip()
    return len(s) == 0


def _chunk(items: list, size: int) -> Iterator[list]:
    """Split into batches of `size`."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Roll per-article scores up to per-ticker-per-date features.

    Input columns: symbol, date, sentiment.
    Output columns: symbol, date, mean_sentiment, sentiment_dispersion,
                    n_articles, sentiment_pos_share, sentiment_neg_share.
    """
    if df.empty:
        return pd.DataFrame(columns=[
            "symbol", "date", "mean_sentiment", "sentiment_dispersion",
            "n_articles", "sentiment_pos_share", "sentiment_neg_share",
        ])

    g = df.groupby(["symbol", "date"])
    agg = g["sentiment"].agg(
        mean_sentiment="mean",
        sentiment_dispersion="std",
        n_articles="count",
    ).reset_index()
    # std() returns NaN for single-row groups → impute zero
    agg["sentiment_dispersion"] = agg["sentiment_dispersion"].fillna(0.0)

    # Pos/neg shares (compute on the raw per-article frame, then join)
    pos = df[df["sentiment"] > POS_THRESHOLD].groupby(["symbol", "date"]).size()
    neg = df[df["sentiment"] < NEG_THRESHOLD].groupby(["symbol", "date"]).size()
    pos.name = "_n_pos"
    neg.name = "_n_neg"
    agg = agg.merge(pos.reset_index(), on=["symbol", "date"], how="left")
    agg = agg.merge(neg.reset_index(), on=["symbol", "date"], how="left")
    agg["_n_pos"] = agg["_n_pos"].fillna(0)
    agg["_n_neg"] = agg["_n_neg"].fillna(0)
    agg["sentiment_pos_share"] = agg["_n_pos"] / agg["n_articles"]
    agg["sentiment_neg_share"] = agg["_n_neg"] / agg["n_articles"]
    return agg.drop(columns=["_n_pos", "_n_neg"])


def validate_sanity(scores: pd.Series) -> tuple[bool, str]:
    """Return (ok, reason) — refuse to save degenerate distributions."""
    if scores.empty:
        return False, "empty score series"
    n = len(scores)
    n_zero = (scores == 0.0).sum()
    n_sat_pos = (scores == 1.0).sum()
    n_sat_neg = (scores == -1.0).sum()
    if n_zero / n > ZERO_FRACTION_LIMIT:
        return False, f"degenerate: {n_zero}/{n}={n_zero/n:.1%} scores exactly zero"
    if (n_sat_pos / n) > SAT_FRACTION_LIMIT:
        return False, f"degenerate: {n_sat_pos}/{n}={n_sat_pos/n:.1%} saturated at +1"
    if (n_sat_neg / n) > SAT_FRACTION_LIMIT:
        return False, f"degenerate: {n_sat_neg}/{n}={n_sat_neg/n:.1%} saturated at -1"
    # Constant
    if scores.std() < 1e-6:
        return False, f"constant distribution (std={scores.std():.2e})"
    return True, "ok"


# ── FinBERT model wrapper ────────────────────────────────────────────────

class FinBertScorer:
    """Lazy-loading FinBERT scorer with batched inference on MPS/CUDA/CPU.

    Auto-detects best device on init. Caches model + tokenizer.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, device: str | None = None, max_length: int = 128):
        import torch
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self.device = device
        self.max_length = max_length

        log.info("loading %s on device=%s ...", self.MODEL_NAME, device)
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)
        self.model = self.model.to(device).eval()
        # Pin the label order so we don't depend on tokenizer.config order
        # FinBERT id2label: 0 -> positive, 1 -> negative, 2 -> neutral
        id2lab = {int(k): v for k, v in self.model.config.id2label.items()}
        self._idx_pos = next(i for i, v in id2lab.items() if v.lower().startswith("pos"))
        self._idx_neg = next(i for i, v in id2lab.items() if v.lower().startswith("neg"))
        self._idx_neu = next(i for i, v in id2lab.items() if v.lower().startswith("neu"))
        log.info("FinBERT ready (pos=%d neg=%d neu=%d device=%s)",
                 self._idx_pos, self._idx_neg, self._idx_neu, device)

    def score_text(self, text: str) -> float:
        if is_skippable_text(text):
            return 0.0
        return self.score_batch([text])[0]

    def score_batch(self, texts: list[str]) -> list[float]:
        """Score a batch of texts. Skippable texts return 0.0."""
        import torch
        if not texts:
            return []

        # Substitute skippable with placeholder; we'll override post-inference
        cleaned = [t if not is_skippable_text(t) else "neutral" for t in texts]
        skip_mask = [is_skippable_text(t) for t in texts]

        enc = self.tokenizer(cleaned, padding=True, truncation=True,
                              max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
            probs = torch.softmax(out.logits, dim=-1).cpu().numpy()

        results = []
        for i in range(len(texts)):
            if skip_mask[i]:
                results.append(0.0)
                continue
            results.append(probs_to_signed(
                p_pos=float(probs[i][self._idx_pos]),
                p_neu=float(probs[i][self._idx_neu]),
                p_neg=float(probs[i][self._idx_neg]),
            ))
        return results


# ── Driver ──────────────────────────────────────────────────────────────

def _score_one_parquet(scorer: FinBertScorer, in_path: Path, out_path: Path,
                       batch_size: int, dry_run: bool) -> tuple[int, int]:
    """Score one ticker's news parquet → write aggregated daily parquet.

    Returns (n_articles_scored, n_daily_rows). Skips if sanity gate fails.
    """
    df_in = pd.read_parquet(in_path)
    if df_in.empty:
        return 0, 0

    # Use headline (Alpaca summary often empty)
    texts = df_in["headline"].fillna("").tolist()
    n = len(texts)

    scores: list[float] = []
    for batch in _chunk(texts, batch_size):
        scores.extend(scorer.score_batch(batch))

    df_in["sentiment"] = scores
    df_in["date"] = pd.to_datetime(df_in["created_at"]).dt.tz_convert("UTC").dt.date
    df_in["date"] = pd.to_datetime(df_in["date"])

    # Sanity gate on raw per-article scores
    ok, reason = validate_sanity(pd.Series(scores))
    if not ok:
        log.warning("  %s: SANITY FAIL — %s (skip write)",
                    in_path.stem, reason)
        return n, 0

    agg = aggregate_daily(df_in[["symbol", "date", "sentiment"]])

    if not dry_run:
        agg.to_parquet(out_path, index=False)
    return n, len(agg)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="*", default=None,
                   help="restrict to specific tickers; default = all")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--dry-run", action="store_true",
                   help="score + sanity-check but do NOT write parquet")
    p.add_argument("--device", default=None,
                   help="force device: mps / cuda / cpu (default = auto)")
    args = p.parse_args()

    in_files = sorted(IN_DIR.glob("*.parquet"))
    if args.symbols:
        wanted = {s.upper() for s in args.symbols}
        in_files = [f for f in in_files if f.stem in wanted]

    log.info("scoring %d ticker files; batch=%d  dry_run=%s",
             len(in_files), args.batch_size, args.dry_run)

    scorer = FinBertScorer(device=args.device)

    n_total = 0
    n_files_written = 0
    n_sanity_fail = 0
    for i, f in enumerate(in_files):
        out_p = OUT_DIR / f"{f.stem}.parquet"
        try:
            n_art, n_days = _score_one_parquet(
                scorer, f, out_p, args.batch_size, args.dry_run)
            n_total += n_art
            if n_days > 0:
                n_files_written += 1
            elif n_art > 0:
                n_sanity_fail += 1
        except Exception as exc:
            log.warning("  %s: scorer failed — %s", f.stem, exc)
            continue
        if (i + 1) % 10 == 0:
            log.info("  %d/%d files  cumulative articles scored: %d",
                     i + 1, len(in_files), n_total)

    log.info("DONE. %d files scored (%d sanity-fail) / %d articles total → %s/",
             n_files_written, n_sanity_fail, n_total,
             OUT_DIR.relative_to(REPO))
    return 0


if __name__ == "__main__":
    sys.exit(main())
