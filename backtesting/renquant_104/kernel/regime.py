"""3-layer regime detection — Hurst, CUSUM, GMM.

Self-contained: only numpy, json, math.  No common/ imports.
"""
from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES


# ── Data container ────────────────────────────────────────────────────────────

@dataclass
class RegimeState:
    regime: str = BULL_CALM
    confidence: float = 0.5
    in_transition: bool = False
    countdown: int = 0
    # mutable; passed in/out so callers persist CUSUM state across bars
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    # CUSUM-cooldown-v2 Design C (user-locked 2026-04-24): wall-clock
    # start of the regime-switch cooldown window. Stamped at the same
    # moment `countdown` is set. Persisted alongside `countdown` in
    # live_state.json + live_state_snapshots so intraday runners don't
    # tick a bar-based cooldown 10x per day.
    cooldown_start: "datetime.datetime | None" = None

    # Intermediate layer outputs — written by individual tasks, read by later tasks
    hurst: float = 0.5                       # Layer 1 output
    hurst_regime: str = "AMBIGUOUS"          # MOMENTUM | REVERSION | AMBIGUOUS
    gmm_probs: dict = field(default_factory=dict)  # Layer 3: P(regime) for each label
    hard_bear: bool = False                  # BEAR hard-override flag
    cusum_triggered: bool = False            # Layer 2 raw flag (diagnostic; cooldown
                                              # armed by RegimeFinalizeTask on regime switch)


def cusum_cooldown_progress(
    now: "datetime.datetime | datetime.date | None",
    cooldown_start: "datetime.datetime | None",
    cooldown_days: float,
) -> float:
    """Fraction of cooldown elapsed (0.0 = just switched, 1.0 = done).

    Used by Design C confidence-scaled sizing: `max_position_pct *= progress`.

    Returns 1.0 (no penalty) when:
      * `cooldown_start` is None (no cooldown active), OR
      * cooldown_days <= 0 (disabled)
    """
    if cooldown_start is None or cooldown_days <= 0:
        return 1.0
    if now is None:
        # Audit #13: failing open (returning 1.0 = no penalty) silently
        # discards the cooldown when callers forget to pass `now`. Failing
        # closed (0.0 = full penalty) makes the bug noisy — sizing collapses
        # immediately and the operator notices.
        return 0.0
    # Accept date (sim bars) by midnight-aligning to datetime
    if isinstance(now, datetime.date) and not isinstance(now, datetime.datetime):
        now = datetime.datetime(now.year, now.month, now.day)
    if isinstance(cooldown_start, datetime.date) and not isinstance(cooldown_start, datetime.datetime):
        cooldown_start = datetime.datetime(cooldown_start.year,
                                           cooldown_start.month,
                                           cooldown_start.day)
    elapsed_days = (now - cooldown_start).total_seconds() / 86400.0
    return max(0.0, min(1.0, elapsed_days / float(cooldown_days)))


# ── Layer 1: Hurst Exponent ───────────────────────────────────────────────────

def compute_hurst(returns: np.ndarray, window: int | None = None) -> float:
    """Rescaled-range (R/S) Hurst exponent. Returns H ∈ [0, 1].

    2026-04-24 fixes:
      - chunk loop was off-by-one (`range(0, n - lag, lag)` skipped the
        trailing arr[n-lag:n]) — now uses `range(0, n - lag + 1, lag)`.
      - `lags_used` was regenerated from `range(2, 2+len(rs_vals))`,
        misaligning when a particular lag produced no chunks. Now we
        pair (lag, rs) explicitly.
    """
    arr = returns if window is None else returns[-window:]
    n = len(arr)
    if n < 10:
        return 0.5
    max_lag = min(n // 2, 40)
    lags_used: list[int] = []
    rs_vals:   list[float] = []
    for lag in range(2, max_lag):
        chunks = [arr[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_chunk: list[float] = []
        for chunk in chunks:
            if len(chunk) < 2:
                continue
            mean = chunk.mean()
            devs = np.cumsum(chunk - mean)
            R    = devs.max() - devs.min()
            S    = chunk.std(ddof=1)
            if S > 0:
                rs_chunk.append(R / S)
        if rs_chunk:
            lags_used.append(lag)
            rs_vals.append(float(np.mean(rs_chunk)))
    if len(rs_vals) < 2:
        return 0.5
    try:
        poly = np.polyfit(np.log(lags_used), np.log(rs_vals), 1)
        return float(np.clip(poly[0], 0.0, 1.0))
    except Exception:
        return 0.5


def rolling_hurst(returns: pd.Series, window: int = 63) -> pd.Series:
    """Rolling Hurst exponent on a return series."""
    result = pd.Series(index=returns.index, dtype=float)
    arr = returns.values
    for i in range(window, len(arr) + 1):
        result.iloc[i - 1] = compute_hurst(arr[i - window:i])
    return result


# ── Layer 2: CUSUM Changepoint ────────────────────────────────────────────────

def compute_cusum(
    returns: np.ndarray,
    lookback: int,
    threshold: float,
    drift: float,
) -> bool:
    """Return True if the latest *lookback* window deviates from the prior window."""
    if len(returns) < lookback * 2:
        return False
    reference = returns[-(lookback * 2):-lookback]
    window    = returns[-lookback:]
    mu        = reference.mean()
    sigma     = reference.std(ddof=1)
    if sigma <= 0:
        return False
    s_pos = s_neg = 0.0
    for r in window:
        z     = (r - mu) / sigma
        s_pos = max(0.0, s_pos + z - drift)
        s_neg = max(0.0, s_neg - z - drift)
        if s_pos > threshold or s_neg > threshold:
            return True
    return False


def rolling_cusum(
    returns: pd.Series,
    window: int = 20,
    threshold: float = 3.0,
    drift: float = 0.5,
) -> pd.Series:
    """Rolling CUSUM: returns True bars where a changepoint is detected."""
    result = pd.Series(False, index=returns.index)
    arr = returns.values
    for i in range(window * 2, len(arr) + 1):
        result.iloc[i - 1] = compute_cusum(
            arr[i - window * 2:i],
            lookback=window,
            threshold=threshold,
            drift=drift,
        )
    return result


# ── Layer 3: GMM ─────────────────────────────────────────────────────────────

def compute_spy_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Compute ADX(*period*) from an OHLCV DataFrame.

    ATR delegated to kernel.indicators.compute_atr (single source of truth).
    """
    from kernel.indicators import compute_atr  # noqa: PLC0415
    if df is None or df.empty or len(df) < period + 1:
        return 25.0
    rows  = df.tail(max(period * 2, period + 1)).copy()
    high  = rows["high"].astype(float)
    low   = rows["low"].astype(float)
    close = rows["close"].astype(float)
    up    = high.diff()
    down  = -low.diff()
    plus_dm  = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    atr      = compute_atr(high, low, close, period=period)
    plus_di  = (100 * pd.Series(plus_dm, index=rows.index)
                .ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
                / atr.replace(0, float("nan")))
    minus_di = (100 * pd.Series(minus_dm, index=rows.index)
                .ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
                / atr.replace(0, float("nan")))
    dx  = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, float("nan"))
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().dropna()
    return float(adx.iloc[-1]) if not adx.empty and not math.isnan(adx.iloc[-1]) else 25.0


def gmm_predict(
    gmm_artifact: dict,
    spy_returns: np.ndarray,
    spy_df: pd.DataFrame | None,
    vol_window: int = 20,
) -> dict[str, float]:
    """Return P(regime) dict using a pre-trained GMM artifact.

    Audit fix GMM-1 (Round 2 deep audit, 2026-04-25): pre-fix, NaN/inf
    in spy_returns propagated through np.sum / np.std into x → through
    matrix algebra → into log_probs as NaN. Then np.exp(NaN) = NaN,
    probs.sum() = NaN, normalisation gave {label: NaN} for every regime.
    Downstream `max(gmm_probs, key=gmm_probs.get)` then returned an
    arbitrary key (NaN comparisons unstable) → non-deterministic regime
    detection on bad SPY data.

    Post-fix: explicit isnan/isinf check; uniform prior on bad data.
    """
    if gmm_artifact is None or len(spy_returns) < vol_window + 10:
        return {r: 1.0 / len(REGIMES) for r in REGIMES}

    recent = spy_returns[-max(vol_window, 11):]
    if np.isnan(recent).any() or np.isinf(recent).any():
        # Bad data → uniform prior (no opinion); RegimeFinalizeTask still
        # gets a usable dict and BEAR override may still fire on its own
        # signal.
        import logging  # noqa: PLC0415
        logging.getLogger("kernel.regime").warning(
            "gmm_predict: NaN/inf in spy_returns — returning uniform prior",
        )
        return {r: 1.0 / len(REGIMES) for r in REGIMES}
    r10d   = float(np.sum(recent[-10:]))
    vol20  = float(np.std(recent[-vol_window:], ddof=1) * math.sqrt(252))
    spy_adx = compute_spy_adx(spy_df) if spy_df is not None else 25.0

    if len(recent) >= 12:
        arr        = np.array(recent[-20:]) if len(recent) >= 20 else np.array(recent)
        r_autocorr = float(np.corrcoef(arr[:-1], arr[1:])[0, 1]) if len(arr) > 2 else 0.0
    else:
        r_autocorr = 0.0

    x = np.array([r10d, vol20, spy_adx, r_autocorr])
    scaler_mean  = np.array(gmm_artifact.get("scaler_mean",  [0.0] * 4))
    scaler_scale = np.array(gmm_artifact.get("scaler_scale", [1.0] * 4))
    scaler_scale = np.where(scaler_scale > 0, scaler_scale, 1.0)
    x = (x - scaler_mean) / scaler_scale

    means   = gmm_artifact["means"]
    covs    = gmm_artifact["covariances"]
    weights = gmm_artifact["weights"]
    labels  = gmm_artifact["cluster_labels"]

    log_probs: list[float] = []
    for k in range(len(means)):
        mu   = np.array(means[k])
        sig  = np.array(covs[k])
        diff = x - mu
        try:
            _sign, logdet = np.linalg.slogdet(sig)
            inv_s = np.linalg.inv(sig)
            mahal = float(diff @ inv_s @ diff)
            lp    = -0.5 * (mahal + logdet) + math.log(max(weights[k], 1e-10))
        except Exception:
            lp = math.log(max(weights[k], 1e-10))
        log_probs.append(lp)

    arr_lp = np.array(log_probs)
    arr_lp -= arr_lp.max()
    probs   = np.exp(arr_lp)
    probs  /= probs.sum()
    return {label: float(p) for label, p in zip(labels, probs)}


def load_gmm_artifact(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Confidence formula ────────────────────────────────────────────────────────

def compute_regime_confidence(
    regime: str,
    hurst: float,
    gmm_probs: dict[str, float],
    in_transition: bool,
    config: dict,
    hurst_regime: str | None = None,
    hard_bear: bool = False,
) -> float:
    """Return confidence ∈ [0, 1] for position-sizing.

    Audit fix RC-MISMATCH (Round 4 deep audit, 2026-04-25, user spec
    "你那个sizing math靠谱吗？confidence也太低了"): the confidence formula
    must MATCH the source of the regime decision in `RegimeFinalizeTask`,
    not blindly query GMM.

    Decision tree (mirrors `RegimeFinalizeTask`):
      hard_bear OR gmm_probs[BEAR] > 0.5     → BEAR decided definitively → confidence = 1.0
      hurst_regime == MOMENTUM               → BULL_CALM via Hurst → use Hurst distance from
                                               trending threshold (deeper into MOMENTUM = higher conf)
      hurst_regime == REVERSION              → CHOPPY via Hurst → use Hurst distance from
                                               reversion threshold (existing logic)
      else (dominant_gmm route)              → use GMM posterior for the regime

    Pre-fix scenario this catches: Hurst MOMENTUM forces BULL_CALM, GMM probability for
    BULL_CALM is 0.0041 (because GMM dominant cluster is something else), confidence
    returned 0.0041 → max_position × confidence ≈ $6 → no buys ever fired.

    During transition: flat 0.5 (uncertainty window after CUSUM changepoint).
    """
    if in_transition:
        return 0.5

    # BEAR override — if we forced BEAR via hard_bear or GMM-BEAR-dominant,
    # the decision is definitive.
    if hard_bear or gmm_probs.get(BEAR, 0.0) > 0.5:
        if regime == BEAR:
            return 1.0

    # Hurst-forced regimes: confidence is Hurst-distance-based (matches source).
    if hurst_regime == "MOMENTUM" and regime == "BULL_CALM":
        hurst_trend = float(config.get("regime", {}).get("hurst_trending_threshold", 0.65))
        # Linear ramp from `hurst_trend` (conf=0) to 1.0 (conf=1).
        # Floor at 0.5: trending Hurst is itself a meaningful signal even at threshold.
        conf = 0.5 + 0.5 * (hurst - hurst_trend) / max(1.0 - hurst_trend, 1e-6)
        return float(min(1.0, max(0.5, conf)))

    if regime == CHOPPY:
        hurst_floor = float(config.get("regime", {}).get("choppy_hurst_floor", 0.20))
        hurst_rev   = float(config.get("regime", {}).get("hurst_reversion_threshold", 0.52))
        conf = (hurst_rev - hurst) / max(hurst_rev - hurst_floor, 1e-6)
        return float(min(1.0, max(0.0, conf)))

    # Audit fix DBT-4 (2026-04-25 followups): defensive — floor at 0.0
    # in case GMM somehow returned a negative posterior for the regime
    # (shouldn't happen but no defense). Downstream confidence_to_size_multiplier
    # already clamps to [floor, 1.0]; this just keeps the contract.
    return float(max(0.0, gmm_probs.get(regime, 0.5)))


def confidence_to_size_multiplier(confidence: float | None, floor: float = 0.5) -> float:
    """Map raw confidence ∈ [0, 1] → size multiplier ∈ [floor, 1.0].

    Audit fix CONF-MULT (Round 4 deep audit, 2026-04-25, user spec
    "直接*confidence不就是降档么这特么的在搞笑吧"): the previous
    callsites multiplied `max_position_pct *= ctx.confidence` directly,
    which has no upside (range [0, 1]) and degenerates to zero on low
    confidence. Floor at 0.5 means even worst-case confidence still
    deploys 50% of the max position — risk-aware, not capital-starved.

    Use this everywhere instead of `value * ctx.confidence`.

    Audit fix CONF-MULT-NONE (2026-04-25 follow-up): pre-fix, calling
    with confidence=None raised TypeError because `math.isfinite(None)`
    crashes. Now: None coerces to floor (same as NaN/inf handling).
    """
    import math
    if confidence is None:
        return float(floor)
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return float(floor)
    if not math.isfinite(c):
        return float(floor)
    return float(max(floor, min(1.0, c)))


# ── Top-level orchestrator ────────────────────────────────────────────────────

def detect_regime(
    spy_returns: np.ndarray,
    spy_df: pd.DataFrame | None,
    gmm_artifact: dict | None,
    state: RegimeState,
    config: dict,
) -> RegimeState:
    """Run all three layers; mutate and return *state*.

    Callers own the RegimeState instance and pass it back each bar so
    CUSUM countdown persists across calls.

    spy_returns: 1-D array of SPY daily returns (most recent last).
    spy_df:      OHLCV DataFrame used for ADX computation inside GMM features.
    gmm_artifact: loaded JSON dict, or None to fall back to Hurst-only.
    state:       previous RegimeState; mutated in place and returned.
    config:      full strategy_config dict.
    """
    regime_cfg    = config.get("regime", {})
    hurst_window  = int(regime_cfg.get("hurst_window", 63))
    hurst_trend   = float(regime_cfg.get("hurst_trending_threshold",   0.65))
    hurst_rev     = float(regime_cfg.get("hurst_reversion_threshold",  0.52))
    cusum_lookback = int(regime_cfg.get("cusum_lookback", 20))
    cusum_thresh  = float(regime_cfg.get("cusum_threshold", 5.5))
    cusum_drift   = float(regime_cfg.get("cusum_drift", 0.5))
    trans_bars    = int(regime_cfg.get("transition_uncertainty_bars", 3))
    vol_window    = int(regime_cfg.get("vol_realized_window", 20))
    bear_vol_thr  = float(regime_cfg.get("bear_vol_threshold",    0.35))
    bear_ret_thr  = float(regime_cfg.get("bear_return_threshold", -0.08))

    if len(spy_returns) < 30:
        return state   # not enough data yet

    prev_regime = state.regime   # snapshot BEFORE mutating — used below for
                                  # switch detection (Plan B: only trigger
                                  # the cooldown when regime *actually*
                                  # changes, not on every CUSUM fire)

    # Layer 1 — Hurst
    hurst = compute_hurst(spy_returns, window=hurst_window)
    if hurst > hurst_trend:
        hurst_regime = "MOMENTUM"
    elif hurst < hurst_rev:
        hurst_regime = "REVERSION"
    else:
        hurst_regime = "AMBIGUOUS"

    # Layer 2 — CUSUM (diagnostic only; actual cooldown trigger below)
    triggered = compute_cusum(spy_returns, cusum_lookback, cusum_thresh, cusum_drift)

    # Layer 3 — GMM
    gmm_probs = gmm_predict(gmm_artifact, spy_returns, spy_df, vol_window=vol_window)
    dominant_gmm = max(gmm_probs, key=gmm_probs.get)

    # BEAR hard override — fire if realized vol or cumulative return cross thresholds
    # regardless of GMM output (GMM alone reacts too slowly to macro shocks)
    if len(spy_returns) >= vol_window:
        window = spy_returns[-vol_window:]
        spy_20d_vol = float(np.std(window, ddof=1) * math.sqrt(252))
        # Audit #11 — strict cumulative product, not arithmetic sum.
        spy_20d_ret = float(np.prod(1.0 + window) - 1.0)
    else:
        spy_20d_vol = 0.0
        spy_20d_ret = 0.0
    hard_bear = spy_20d_vol > bear_vol_thr or spy_20d_ret < bear_ret_thr

    # Resolve regime
    if hard_bear or gmm_probs.get(BEAR, 0) > 0.5:
        new_regime = BEAR
    elif hurst_regime == "MOMENTUM":
        new_regime = BULL_CALM
    elif hurst_regime == "REVERSION":
        new_regime = CHOPPY
    else:
        new_regime = dominant_gmm if dominant_gmm != BEAR else BULL_VOLATILE

    # Plan B (2026-04-23): CUSUM cooldown only fires on a REGIME SWITCH.
    # Previously `if triggered and state.countdown == 0: countdown = trans_bars`
    # would re-arm every time CUSUM flagged a change in the 20-bar SPY
    # window — but CUSUM detects *any* shift (e.g. bull recovery, vol up/
    # down) even when the resolved regime stayed BULL_CALM. During the
    # 2026-04-12 → 04-23 bull recovery that made CUSUM fire 10 bars in a
    # row → `in_transition=True` locked → all live buys blocked for 3 days.
    # New rule: cooldown only applies when `prev_regime != new_regime`.
    if new_regime != prev_regime and state.countdown == 0:
        state.countdown = trans_bars

    # Confidence
    in_transition = state.countdown > 0
    confidence = compute_regime_confidence(new_regime, hurst, gmm_probs, in_transition, config)

    # Decrement countdown after use (so last bar of window still shows in_transition=True)
    if state.countdown > 0:
        state.countdown -= 1

    state.regime     = new_regime
    state.confidence = confidence
    state.in_transition = in_transition
    return state
