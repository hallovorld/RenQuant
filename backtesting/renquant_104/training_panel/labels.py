"""Beta-neutral, cross-sectionally Gaussianized labels for Stage-1 LTR.

Pipeline::

    fwd_return_i,t
        ─► residualize vs SPY and sector ETF forward returns (purged β)
        ─► per-date rank → [0,1] → inverse-normal CDF ⇒ N(0,1)

The output label for row (ticker=i, date=t) is the Gaussianized residual
forward return — a cross-sectionally comparable, beta-neutral target that
encodes "outperforms peers" rather than "goes up with the market".

Public API::

    compute_residual_returns(...)    — purged rolling β neutralization
    gaussianize_cross_section(...)   — per-date rank → normal quantile
    build_labels(...)                — full pipeline
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _rolling_beta_purged(
    y: pd.Series, x: pd.Series, window: int, purge: int,
    *, clip_low: float = -3.0, clip_high: float = 5.0,
) -> pd.Series:
    """Rolling OLS slope of y on x, using only data strictly prior to t.

    At time t we fit on the window ending at t-purge (inclusive), so the
    current bar's forward return cannot leak into β. Returns β_t aligned
    to y's index. Rows without enough prior history are NaN.

    Audit fix D-1 (2026-04-25): β is clipped to [clip_low, clip_high]
    (default [-3, +5], the typical equity β range). Pre-fix, an
    illiquid/halted ticker × SPY could produce var≈0 → β ∈ [-50, +50]
    which then dominated `residual = fwd - β · spy_fwd` and turned
    labels into noise.
    """
    y = y.astype(float)
    x = x.astype(float)
    # pair values are the training observations; shift by `purge` bars so
    # the window for index t excludes bars t-purge+1 .. t.
    y_s = y.shift(purge)
    x_s = x.shift(purge)
    # rolling covariance / variance
    cov = y_s.rolling(window, min_periods=window).cov(x_s)
    var = x_s.rolling(window, min_periods=window).var()
    beta = cov / var.replace(0, np.nan)
    return beta.clip(lower=clip_low, upper=clip_high)


def compute_residual_returns(
    fwd_returns: dict[str, pd.Series],
    spy_returns: pd.Series,
    sector_returns_by_ticker: dict[str, pd.Series],
    beta_window: int = 60,
    lookahead_days: int = 5,
) -> dict[str, pd.Series]:
    """Per-ticker residualize fwd_return vs SPY and sector forward returns.

    residual_i,t = fwd_i,t − β_spy_i,t · spy_fwd_t − β_sec_i,t · sec_fwd_t

    β is fit by rolling OLS on strictly-prior data (purged by
    `lookahead_days` bars to avoid label overlap with the current bar).

    Missing β (insufficient history) leaves that row as NaN. Missing sector
    series falls back to β_spy only.
    """
    out: dict[str, pd.Series] = {}
    for t, fwd in fwd_returns.items():
        fwd = fwd.astype(float)
        spy_fwd = spy_returns.reindex(fwd.index).astype(float)

        sec_fwd = sector_returns_by_ticker.get(t)
        if sec_fwd is not None:
            sec_fwd = sec_fwd.reindex(fwd.index).astype(float)

        beta_spy = _rolling_beta_purged(
            fwd, spy_fwd, window=beta_window, purge=lookahead_days,
        )

        residual = fwd - beta_spy * spy_fwd
        if sec_fwd is not None:
            # Audit fix LBL-1 (Round 2 deep audit, 2026-04-25):
            # Pre-fix, beta_sec was fit against raw `fwd` and raw `sec_fwd`.
            # But sector ETFs are ~85-95% SPY-correlated, so the resulting
            # β_sec captured *both* the sector AND the SPY-driven component
            # of sec_fwd. Subtracting `β_spy·spy_fwd` then `β_sec·sec_fwd`
            # then over-removes the market component by a factor of
            # ~β·ρ²(spy,sec) — labels were biased systematically lower
            # for tickers in high-β sectors during up moves.
            #
            # Fix: orthogonalize sec_fwd against spy_fwd first
            # (Frisch-Waugh-Lovell), so β_sec captures only the
            # sector-specific component net of SPY. Final residual is
            # then equivalent to a joint OLS residual on [spy, sec].
            beta_sec_on_spy = _rolling_beta_purged(
                sec_fwd, spy_fwd, window=beta_window, purge=lookahead_days,
            )
            sec_fwd_orth = sec_fwd - beta_sec_on_spy * spy_fwd
            beta_sec = _rolling_beta_purged(
                fwd, sec_fwd_orth, window=beta_window, purge=lookahead_days,
            )
            residual = residual - beta_sec * sec_fwd_orth

        out[t] = residual
    return out


def gaussianize_cross_section(
    residuals: dict[str, pd.Series],
) -> dict[str, pd.Series]:
    """Per date: rank residuals across tickers, map uniform → N(0,1).

    rank / (N+1) ∈ (0,1)  →  norm.ppf(...)  ⇒  approximately standard normal.

    Ties get average rank. A single-ticker day yields NaN (cross-sectional
    rank is undefined with N=1; emitting 0.0 silently injected a "neutral"
    label into training, biasing weights for any date the universe shrinks
    to one ticker — common at 200+-watchlist start dates where younger names
    enter progressively. External audit fix 2026-04-29 (#9).
    """
    # Assemble long frame indexed by (date, ticker)
    frames = []
    for t, s in residuals.items():
        df = pd.DataFrame({"date": s.index, "ticker": t, "residual": s.values})
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)

    # Per-date rank → uniform → normal. Ties get average rank (constant → 0.5).
    def _per_date(r: pd.Series) -> pd.Series:
        mask = r.notna()
        n = int(mask.sum())
        out_v = pd.Series(np.nan, index=r.index)
        if n == 0:
            return out_v
        if n == 1:
            # Cross-sectional rank with N=1 is undefined — return NaN so the
            # row is dropped from training, not silently labeled 0.0.
            return out_v
        ranks = r[mask].rank(method="average")
        u = ranks / (n + 1)
        out_v[mask] = norm.ppf(u.values)
        return out_v

    long["gauss"] = long.groupby("date")["residual"].transform(_per_date)

    out: dict[str, pd.Series] = {}
    for t, sub in long.groupby("ticker", sort=False):
        s = pd.Series(sub["gauss"].values, index=pd.Index(sub["date"].values))
        s = s.sort_index()
        out[t] = s
    return out


def build_labels(
    fwd_returns: dict[str, pd.Series],
    spy_returns: pd.Series,
    sector_returns_by_ticker: dict[str, pd.Series],
    *,
    beta_window: int = 60,
    lookahead_days: int = 5,
) -> dict[str, pd.Series]:
    """Full pipeline: residualize vs SPY+sector → Gaussianize per date."""
    residuals = compute_residual_returns(
        fwd_returns, spy_returns, sector_returns_by_ticker,
        beta_window=beta_window, lookahead_days=lookahead_days,
    )
    return gaussianize_cross_section(residuals)
