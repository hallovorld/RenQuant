"""Macro factor frame — cross-asset signals broadcast to every panel row.

User spec 2026-04-26: add VIX/HYG/UUP/DBC/etc. as panel features WITHOUT
trying to trade them. Per-date z-scored levels + changes broadcast across
all watchlist tickers on the same date.

See `doc/components/macro-factor-frame-design.md` for the full design.

Cache layout mirrors `FundamentalsStore`:

  data/macro/{SYMBOL}.parquet

Each file = standard OHLCV (open/high/low/close/volume) indexed by date.
Fetched via yfinance (one symbol per file). Read by `LoadMacroFactorsTask`
during training; computed z-score features broadcast to all rows of the
panel via `panel_frame.build_panel_frame(macro_frame=...)`.

Safety (per design §11):
- Per-symbol try/except: one missing symbol doesn't kill the others
- F4 short-window guard: rolling-window warmup must cover ≥95% of training
  window or that macro is dropped (default knob)
- F5 zero-variance clamp: z-score divisions by zero → 0.0 (not inf/NaN)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

log = logging.getLogger("kernel.macro")


# Default symbol set — see design §2 for rationale per symbol.
# Note: VXX is the tradable VIX-proxy ETF. ^VIX is the index itself
# (some yfinance versions return raw, some return normalized — VXX is
# more reliable). Operator can override via config.
#
# Tier 1 expansion (2026-04-27, per `doc/research/macro-data-expansion-
# plan-2026-04-27.md`): added 18 ETF symbols covering defense, oil,
# semis, banks, biotech, gold miners, SaaS, China, EM, Europe, Japan,
# FX, vol regimes, long Treasury, IG credit, broader commodities, TIPS.
# These get ingested as per-ticker β features (NOT raw broadcast — the
# 8-variant tournament showed pure broadcast adds 0 within-date variance
# on the cross-sectional rank loss). Total 11 + 18 = 29 macro symbols.
DEFAULT_MACRO_SYMBOLS: list[str] = [
    # ── Original 11 (vol/rates/credit/factor/sector core) ─────────
    "VXX",    # volatility regime
    "HYG",    # credit spread / risk-on-off
    "UUP",    # dollar strength (DXY proxy)
    "DBC",    # broad commodities (inflation/growth)
    "GLD",    # gold (safe-haven)
    "TLT",    # long-bond rates
    "XLV",    # healthcare (defensive)
    "XLU",    # utilities (low-beta defensive)
    "KRE",    # regional banks (credit health)
    "MTUM",   # momentum factor crowdedness
    "USMV",   # low-vol factor
    # ── Tier 1 expansion: sector / industry ETFs (paper refs in
    #     doc/research/macro-data-expansion-plan-2026-04-27.md) ──
    "ITA",    # US Aerospace & Defense (geopolitical risk)
    "USO",    # crude oil ETF (Driesprong et al. 2008)
    "XLE",    # energy SPDR (oil-price exposure)
    "SMH",    # semiconductors (semi-cycle exposure)
    "KBE",    # bank ETF (yield-curve sensitivity)
    "XBI",    # biotech ETF (FDA / pharma cycle)
    "GDX",    # gold miners (inflation hedge β)
    "WCLD",   # cloud/SaaS (tech subsegment)
    # ── Tier 1: international / FX ───────────────────────────────
    "FXI",    # China large-cap (CNY exposure)
    "EEM",    # emerging markets (EM growth)
    "VGK",    # Europe (European cycle)
    "EWJ",    # Japan (JPY/Nikkei)
    "FXE",    # EUR/USD (FX exposure)
    "FXY",    # JPY/USD
    # ── Tier 1: rates / credit / vol / commodities / TIPS ─────────
    "VIXY",   # VIX-tracking ETF (vol-regime)
    "EDV",    # Vanguard Extended Duration Treasury (>20y)
    "LQD",    # IG corporate bonds (credit spread vs HYG)
    "DBA",    # agriculture (Boons 2016)
    "TIP",    # TIPS (real-rate exposure, breakeven inflation)
]

DEFAULT_TRANSFORMS: list[str] = ["level_z", "chg_5d_z", "chg_20d_z"]
DEFAULT_ROLLING_WINDOW: int = 252


# ── Storage ───────────────────────────────────────────────────────────────────

@dataclass
class MacroFactorStore:
    """Parquet-backed cache at `data/macro/{SYMBOL}.parquet`.

    Mirrors `FundamentalsStore` semantics:
      - `load(symbol)` → DataFrame with DatetimeIndex (or None on miss/corrupt)
      - `save(df, symbol)` → atomic write, dedupes on index
    """
    data_dir: Path = Path("data/macro")

    def __post_init__(self) -> None:
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)

    def _path(self, symbol: str) -> Path:
        return self.data_dir / f"{symbol.upper()}.parquet"

    def load(self, symbol: str) -> pd.DataFrame | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        try:
            df = pd.read_parquet(p)
        except Exception as exc:
            log.warning(
                "MacroFactorStore.load(%s): corrupt parquet — %s; "
                "treating as cache-miss", symbol, exc,
            )
            return None
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def save(self, df: pd.DataFrame, symbol: str) -> Path:
        """Atomic write (.tmp + rename), dedupes on index."""
        p = self._path(symbol)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        existing = self.load(symbol)
        if existing is not None:
            df = pd.concat([existing, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
        tmp = p.with_suffix(p.suffix + ".tmp")
        df.to_parquet(tmp)
        tmp.replace(p)
        return p


# ── Z-score transforms ────────────────────────────────────────────────────────

def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    """Z-score of series against its rolling mean+std over `window` days.

    Safety F5: zero-variance windows or division-by-zero produce 0.0
    (not inf). NaN values from warmup are PRESERVED so callers can
    distinguish "not enough data yet" from "valid but zero". This is
    load-bearing for the F4 coverage check in build_macro_frame.
    """
    mean = series.rolling(window, min_periods=max(2, window // 4)).mean()
    std = series.rolling(window, min_periods=max(2, window // 4)).std()
    z = (series - mean) / std.replace(0.0, np.nan)
    # Clamp +/- inf to 0.0; leave NaN as NaN.
    z = z.where(~np.isinf(z), 0.0)
    return z


def compute_macro_features(
    ohlcv: pd.DataFrame,
    *,
    symbol: str,
    transforms: Iterable[str] = DEFAULT_TRANSFORMS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
) -> dict[str, pd.Series]:
    """Compute z-scored features for one macro symbol.

    Returns dict[col_name → Series]. Each col_name is `{sym_lower}_{transform}`.
    Skips unknown transforms (logs WARN).

    Safety:
    - Returns empty dict if `close` column missing or empty.
    - Each series fully NaN-clamped via _rolling_z.
    """
    if ohlcv is None or ohlcv.empty or "close" not in ohlcv.columns:
        return {}
    # AUDIT 2026-05-10 C9 — sort_index() alone preserves duplicate index
    # rows (yfinance occasionally returns dups on splits/dividends). Mirror
    # the dedup pattern at line 135 to ensure macro_frame index is unique
    # per §5.13.11. Otherwise downstream reindex (see macro_per_ticker.py:122)
    # raises "cannot reindex on an axis with duplicate labels".
    close = ohlcv["close"].astype(float)
    close = close[~close.index.duplicated(keep="last")].sort_index()
    sym_lower = symbol.lower()
    out: dict[str, pd.Series] = {}
    for t in transforms:
        col_name = f"{sym_lower}_{t}"
        if t == "level_z":
            out[col_name] = _rolling_z(close, rolling_window)
        elif t == "chg_5d_z":
            out[col_name] = _rolling_z(close.pct_change(5), rolling_window)
        elif t == "chg_20d_z":
            out[col_name] = _rolling_z(close.pct_change(20), rolling_window)
        else:
            log.warning("compute_macro_features: unknown transform %s — skipping", t)
    return out


def build_macro_frame(
    store: MacroFactorStore,
    *,
    symbols: Iterable[str] = DEFAULT_MACRO_SYMBOLS,
    transforms: Iterable[str] = DEFAULT_TRANSFORMS,
    rolling_window: int = DEFAULT_ROLLING_WINDOW,
    min_window_overlap_pct: float = 0.95,
    training_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Assemble the full date-indexed macro frame from cached symbols.

    Safety:
    - F1: per-symbol try/except — one bad symbol doesn't kill others
    - F2: missing symbols logged + skipped
    - F4: drops macros whose rolling-window coverage < min_window_overlap_pct
      (default 95%) of the training window
    - F5: z-score transforms already clamp inf/NaN

    Returns (frame, metadata):
      frame    — DataFrame indexed by date, columns = macro features
      metadata — dict {symbols_used, symbols_skipped, n_features,
                       rolling_window, transforms}
    """
    cols: dict[str, pd.Series] = {}
    skipped: list[tuple[str, str]] = []
    used: list[str] = []

    for sym in symbols:
        try:
            ohlcv = store.load(sym)
        except Exception as exc:
            skipped.append((sym, f"load_failed: {type(exc).__name__}"))
            log.warning("build_macro_frame: load %s failed — %s", sym, exc)
            continue
        if ohlcv is None or ohlcv.empty:
            skipped.append((sym, "no_cache"))
            continue
        try:
            sym_features = compute_macro_features(
                ohlcv, symbol=sym,
                transforms=transforms, rolling_window=rolling_window,
            )
        except Exception as exc:
            skipped.append((sym, f"compute_failed: {type(exc).__name__}"))
            log.warning("build_macro_frame: compute %s failed — %s", sym, exc)
            continue

        # F4: drop the symbol if its rolling-window coverage on the
        # training window is too thin.
        if training_end is not None and sym_features:
            first_col = next(iter(sym_features.values()))
            window_start = training_end - pd.Timedelta(days=rolling_window * 2)
            in_window = first_col.loc[window_start:training_end]
            if len(in_window) > 0:
                non_nan_pct = float(in_window.notna().mean())
                if non_nan_pct < min_window_overlap_pct:
                    skipped.append(
                        (sym, f"insufficient_coverage_{non_nan_pct:.2f}<{min_window_overlap_pct}"),
                    )
                    log.warning(
                        "build_macro_frame: %s coverage %.0f%% < min %.0f%% — skipping",
                        sym, non_nan_pct * 100, min_window_overlap_pct * 100,
                    )
                    continue

        cols.update(sym_features)
        used.append(sym)

    if not cols:
        log.warning("build_macro_frame: no macro features built (all symbols skipped)")
        return pd.DataFrame(), {
            "symbols_used":    [],
            "symbols_skipped": skipped,
            "n_features":      0,
            "rolling_window":  rolling_window,
            "transforms":      list(transforms),
        }

    frame = pd.DataFrame(cols).sort_index()
    metadata = {
        "symbols_used":    used,
        "symbols_skipped": skipped,
        "n_features":      len(cols),
        "rolling_window":  rolling_window,
        "transforms":      list(transforms),
        "n_dates":         int(len(frame)),
    }
    log.info(
        "build_macro_frame: %d features from %d symbols (%d skipped); "
        "%d dates",
        len(cols), len(used), len(skipped), len(frame),
    )
    return frame, metadata


__all__ = [
    "MacroFactorStore",
    "compute_macro_features",
    "build_macro_frame",
    "DEFAULT_MACRO_SYMBOLS",
    "DEFAULT_TRANSFORMS",
    "DEFAULT_ROLLING_WINDOW",
]
