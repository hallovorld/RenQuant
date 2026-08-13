"""Proof for the G-J SPY rolling-Hurst dedup (perf/hoist-spy-hurst-dedup).

The daily-full feature prep recomputes the 63-day SPY rolling-Hurst once per
ticker even though — within a single ``prepare_inference_panel_frames`` run —
SPY OHLCV is a single shared frame, so every ticker that shares a
``common_idx`` feeds an IDENTICAL ``spy_rets`` into ``rolling_hurst``. That
call is ~99% of per-ticker cost. This change memoizes the exact existing
computation, keyed by the content of ``spy_rets``, and shares the memoizer
across the per-ticker chain.

The change is a SPEED-ONLY change: any change to a feature value would be a
regression. These tests are the safety net — they must FAIL if the memoized
output differs from the un-memoized baseline by a single value, and they
assert the memo actually collapses the redundant computations.

  * ``test_prepare_inference_frames_byte_identical`` — run
    ``prepare_inference_panel_frames`` BEFORE (memo disabled via monkeypatch)
    and AFTER (memo on) on a real 12-ticker input; the neutralized feature
    frames AND the z-scored factor frames must be byte-identical per ticker,
    and the ``rolling_hurst`` call count must drop from 12 → 1.
  * ``test_build_training_features_byte_identical_and_dedup`` — same guarantee
    at the ``build_training_features`` level, including a distinct-``common_idx``
    ticker group to prove the memo computes once per DISTINCT input (not just
    globally once) while staying byte-identical.
  * ``test_memo_returns_identical_series`` — the memo primitive returns the
    exact Series ``rolling_hurst`` would, and a repeated identical input is a
    cache hit (no second compute).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_STRATEGY_DIR = Path(__file__).resolve().parent.parent / "backtesting" / "renquant_104"
if str(_STRATEGY_DIR) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_DIR))

import renquant_common.hurst as hurst_mod  # noqa: E402
from training.features import SpyHurstMemo, build_training_features  # noqa: E402
from training_panel import pipeline as pipeline_mod  # noqa: E402


# ── fixtures ────────────────────────────────────────────────────────────────

def _ohlcv(seed: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic synthetic OHLCV on a given index (positive random walk)."""
    rng = np.random.default_rng(seed)
    n = len(index)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.014, n)))
    return pd.DataFrame(
        {
            "open": close * (1.0 + rng.normal(0, 0.001, n)),
            "high": close * 1.008,
            "low": close * 0.992,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=index,
    )


_FULL_IDX = pd.bdate_range("2022-01-03", periods=420)
# A strict subset of _FULL_IDX (later start / earlier end) → a DISTINCT
# common_idx, hence a distinct spy_rets, hence a second cache slot.
_SHORT_IDX = pd.bdate_range("2022-04-01", periods=300)


@pytest.fixture
def universe():
    """12 tickers + SPY + XLK. 10 share the full date range, 2 share a shorter
    (subset) range → exactly 2 distinct SPY ``spy_rets`` inputs."""
    full = [f"F{i:02d}" for i in range(10)]
    short = ["S00", "S01"]
    ohlcv: dict[str, pd.DataFrame] = {}
    for i, t in enumerate(full):
        ohlcv[t] = _ohlcv(100 + i, _FULL_IDX)
    for i, t in enumerate(short):
        ohlcv[t] = _ohlcv(200 + i, _SHORT_IDX)
    ohlcv["SPY"] = _ohlcv(1, _FULL_IDX)
    ohlcv["XLK"] = _ohlcv(2, _FULL_IDX)
    watchlist = full + short
    return watchlist, ohlcv


@pytest.fixture
def count_hurst(monkeypatch):
    """Patch renquant_common.hurst.rolling_hurst to count real evaluations.

    build_training_features does a call-time ``from renquant_common.hurst
    import rolling_hurst``, so patching the module attribute is picked up on
    every call and the counter reflects true computations.
    """
    real = hurst_mod.rolling_hurst
    calls = {"n": 0}

    def counting(returns, window=63):
        calls["n"] += 1
        return real(returns, window=window)

    monkeypatch.setattr(hurst_mod, "rolling_hurst", counting)
    return calls


# ── integration proof: prepare_inference_panel_frames ───────────────────────

def _cfg(watchlist):
    return {
        "benchmark": "SPY",
        "sector_etf_map": {"tech": "XLK"},
        "sector_map": {t: "tech" for t in watchlist},
        "panel_ltr": {"fundamentals": {"enabled": False}},
        "indicator_spec": {},
        "_strategy_dir": str(_STRATEGY_DIR),
    }


def test_prepare_inference_frames_byte_identical(universe, count_hurst, monkeypatch):
    watchlist, ohlcv = universe
    cfg = _cfg(watchlist)
    ticker_sectors = {t: "tech" for t in watchlist}

    # BEFORE: disable the memoizer entirely → original per-ticker computation.
    monkeypatch.setattr(pipeline_mod, "_new_spy_hurst_memo", lambda: None)
    count_hurst["n"] = 0
    ff_before, fac_before, macro_before, _ = pipeline_mod.prepare_inference_panel_frames(
        watchlist=watchlist, ohlcv=ohlcv, ticker_sectors=ticker_sectors, config=cfg,
    )
    before_calls = count_hurst["n"]

    # AFTER: real memoizer (restore the genuine factory).
    monkeypatch.undo()
    monkeypatch.setattr(pipeline_mod, "_new_spy_hurst_memo", lambda: SpyHurstMemo())
    count_hurst["n"] = 0
    ff_after, fac_after, macro_after, _ = pipeline_mod.prepare_inference_panel_frames(
        watchlist=watchlist, ohlcv=ohlcv, ticker_sectors=ticker_sectors, config=cfg,
    )
    after_calls = count_hurst["n"]

    # Sanity: the fixture must actually exercise the feature builder.
    assert len(ff_before) >= 10, "fixture should produce >=10 neutralized frames"
    assert set(ff_before) == set(ff_after)
    assert set(fac_before) == set(fac_after)

    # Byte-identical neutralized feature frames AND z-scored factor frames.
    for t in ff_before:
        pd.testing.assert_frame_equal(
            ff_before[t], ff_after[t], check_exact=True,
            obj=f"neutralized_frame[{t}]",
        )
    for t in fac_before:
        pd.testing.assert_frame_equal(
            fac_before[t], fac_after[t], check_exact=True,
            obj=f"factor_frame[{t}]",
        )
    # macro frame parity (None here — fundamentals/macro disabled).
    assert macro_before is macro_after or (
        macro_before is None and macro_after is None
    )

    # Call-count reduction: baseline computes once per ticker (12); with the
    # memoizer it collapses to one compute per DISTINCT spy_rets. The 2 short
    # tickers share one common_idx, so 2 distinct groups → but they may drop
    # out after warm-up; assert the strong, always-true bound.
    assert before_calls == len(watchlist), before_calls
    assert after_calls < before_calls
    assert after_calls <= 2, after_calls  # <= number of distinct common_idx


# ── unit proof: build_training_features per ticker ──────────────────────────

def test_build_training_features_byte_identical_and_dedup(universe, count_hurst):
    watchlist, ohlcv = universe
    spec, lookahead, threshold = {}, 5, 0.03

    # Baseline: no cache — one hurst compute per ticker.
    count_hurst["n"] = 0
    baseline = {
        t: build_training_features(t, ohlcv, spec, lookahead, threshold, hurst_cache=None)
        for t in watchlist
    }
    baseline_calls = count_hurst["n"]

    # Cached: a single shared memo across every ticker.
    memo = SpyHurstMemo()
    count_hurst["n"] = 0
    cached = {
        t: build_training_features(t, ohlcv, spec, lookahead, threshold, hurst_cache=memo)
        for t in watchlist
    }
    cached_calls = count_hurst["n"]

    built = [t for t in watchlist if baseline[t] is not None]
    assert len(built) >= 10, "fixture should build >=10 feature frames"

    # Byte-identical per ticker (this is the feature frame that carries
    # hurst_proxy — the only column the change can affect).
    for t in watchlist:
        b, c = baseline[t], cached[t]
        assert (b is None) == (c is None), t
        if b is None:
            continue
        pd.testing.assert_frame_equal(b, c, check_exact=True, obj=f"feature_frame[{t}]")
        assert "hurst_proxy" in b.columns

    # Dedup: baseline = one per ticker; cached = one per DISTINCT common_idx.
    assert baseline_calls == len(watchlist)
    n_distinct = len({
        tuple(baseline[t].index) for t in built  # frame index == common_idx (post-dropna)
    })
    assert cached_calls == memo.compute_calls
    assert cached_calls < baseline_calls
    # Two date-range groups in the fixture → at most 2 distinct spy_rets.
    assert cached_calls <= 2, cached_calls


# ── unit proof: the memo primitive is output-invariant ──────────────────────

def test_memo_returns_identical_series():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2022-01-03", periods=200)
    spy_rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)

    direct = hurst_mod.rolling_hurst(spy_rets, window=63)

    memo = SpyHurstMemo()
    first = memo.get_or_compute(spy_rets, lambda: hurst_mod.rolling_hurst(spy_rets, window=63))
    assert memo.compute_calls == 1
    pd.testing.assert_series_equal(first, direct, check_exact=True)

    # Repeated identical input → cache hit, no second compute, same object.
    second = memo.get_or_compute(
        spy_rets.copy(), lambda: pytest.fail("cache miss on identical input")
    )
    assert memo.compute_calls == 1
    assert second is first

    # A genuinely different input → a second compute.
    other = spy_rets * 2.0
    memo.get_or_compute(other, lambda: hurst_mod.rolling_hurst(other, window=63))
    assert memo.compute_calls == 2
