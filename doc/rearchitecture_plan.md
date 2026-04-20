# renquant_103 Re-Architecture Plan

> **Status: IMPLEMENTED** — All phases complete as of 2026-04-19. See `doc/architecture.md` for the current state. This document is preserved for historical context.

## Problem

Three files implement the same decision logic independently:
- `backtesting/renquant_103/main.py` — 1,388 lines, LEAN QCAlgorithm
- `Notebooks/renquant_103.ipynb` — 423-line simulation cell + training cells
- `live/runner.py` — 1,597 lines, live trading runner

Every strategy change requires triple-updating. Drift is inevitable — the Hurst-confidence
change required three separate commits across three sessions. `common/` is not accessible
from LEAN's Docker, so all shared logic was copied inline.

---

## Guiding Principles

1. **The strategy directory owns the strategy.** `backtesting/renquant_103/` contains
   everything: kernel, notebook, training code, simulation engine, LEAN entry point.
2. **Kernel is the single source of truth for decisions.** All entry/exit/sizing/regime logic
   lives in `kernel/`. LEAN's `main.py`, the notebook, and the live runner are thin callers.
3. **Kernel is Docker-safe.** `kernel/` has zero `common/` imports. Only `numpy`, `pandas`,
   `json`, `math`, `datetime`. This is enforced by a CI test.
4. **No `common/` for renquant_103.** Training, data fetching, and charting move into
   strategy-local modules. `common/` remains untouched for other strategies.
5. **Parity tests become unit tests.** When notebook and LEAN call the same function, parity
   is structural — not something to test. Replace parity mocks with kernel unit tests.

---

## Target Directory Structure

```
backtesting/renquant_103/
│
├── strategy_config.json          ← unchanged
├── renquant_103.ipynb            ← MOVED from Notebooks/; thin glue (~100 lines)
├── main.py                       ← LEAN entry point; stays at root (LEAN CLI requires it)
│                                    reduced to ~250 lines of Initialize + OnData glue
│
├── kernel/                       ← LEAN-safe; no common/ imports
│   ├── __init__.py
│   ├── config.py                 ← load_config(path) → dict
│   ├── regime.py                 ← Hurst, CUSUM, GMM predict, confidence formula
│   ├── indicators.py             ← inference-time indicators (RSI, MACD, CCI, etc.)
│   ├── models.py                 ← all 4 model types + calibration (load & score)
│   ├── exits.py                  ← all 5 exit types as pure functions
│   ├── selection.py              ← RS score, candidate scoring, tiered select, guards
│   └── sizing.py                 ← position sizing, oversize fallback
│
├── data/                         ← notebook + runner only; NOT imported by main.py
│   ├── __init__.py
│   ├── fetch.py                  ← yfinance fetch, parquet cache
│   └── artifacts.py              ← load/save GMM artifact, corr matrix, earnings cal
│
├── training/                     ← notebook only
│   ├── __init__.py
│   ├── features.py               ← build training feature frames (indicators + labels)
│   ├── tournament.py             ← 4-model tournament, OOS Sharpe eval, model export
│   ├── calibration.py            ← score calibration fit, blend weight fit + export
│   └── regime_fit.py             ← GMM fit, correlation matrix, earnings calendar fetch
│
├── simulation/                   ← notebook simulation engine
│   ├── __init__.py
│   ├── engine.py                 ← daily loop; calls kernel functions; pure Python
│   └── reporter.py               ← charts, stats table, trade log
│
├── models/                       ← model artifacts (unchanged)
│   └── {TICKER}/
│       ├── {ticker}-policy-metadata.json
│       └── ...
│
└── artifacts/                    ← runtime JSON artifacts (renamed from root level)
    ├── spy-gmm-regime.json
    ├── watchlist-correlation.json
    └── earnings-calendar.json
```

```
live/
├── runner.py                     ← entry point; reduced to ~150 lines
├── broker/
│   ├── __init__.py
│   ├── base.py                   ← Broker ABC (moved from inline classes)
│   ├── alpaca.py                 ← AlpacaBroker
│   ├── paper.py                  ← PaperBroker
│   └── ibkr.py                   ← IBKRBroker stub
└── execution/
    ├── __init__.py
    ├── signals.py                ← compute_signals_for_today() using kernel
    └── state.py                  ← portfolio state, position tracking, wash-sale log
```

```
tests/
├── conftest.py                   ← shared fixtures: synthetic price DataFrames, configs,
│                                    minimal model artifacts
├── kernel/
│   ├── test_regime.py            ← Hurst, CUSUM, GMM predict, confidence formula
│   ├── test_exits.py             ← all 5 exit types, edge cases
│   ├── test_selection.py         ← RS score, tiered thresholds, wash-sale, sector,
│   │                                correlation guards
│   ├── test_sizing.py            ← position sizing, oversize fallback
│   ├── test_indicators.py        ← inference-time indicator correctness
│   └── test_models.py            ← model inference (all 4 types), calibration
├── simulation/
│   └── test_engine.py            ← simulation engine: multi-day replay
├── live/
│   └── test_runner.py            ← live runner: broker calls, state tracking, signals
└── integration/
    └── test_parity.py            ← ~5 tests: same kernel inputs → sim engine and
                                     main.py OnData produce identical decisions
```

---

## Module Breakdown

### `kernel/config.py`
Replaces `common.config.load_strategy_config` and the current `config.py`.

```python
def load_config(path: str | Path) -> dict: ...
# Loads strategy_config.json. Both main.py and notebook call this.
```

---

### `kernel/regime.py`
All three regime detection layers + confidence formula. Currently split between
LEAN's `_compute_hurst`, `_compute_cusum`, `_gmm_predict` (main.py lines 545–659)
and notebook cells 5–7.

```python
@dataclass
class RegimeState:
    regime: str           # "BULL_CALM" | "BULL_VOLATILE" | "CHOPPY" | "BEAR"
    confidence: float     # position sizing multiplier
    in_transition: bool
    countdown: int
    cusum_pos: float      # passed in/out so callers persist across bars
    cusum_neg: float

def compute_hurst(returns: np.ndarray, window: int) -> float: ...
def update_cusum(ret: float, pos: float, neg: float,
                 threshold: float, drift: float) -> tuple[float, float, bool]: ...
def gmm_predict(spy_features: dict, artifact: dict) -> tuple[str, dict[str, float]]: ...
def compute_regime_confidence(regime: str, hurst: float,
                               gmm_probs: dict[str, float],
                               in_transition: bool, config: dict) -> float: ...
def detect_regime(spy_returns: np.ndarray, spy_df: pd.DataFrame,
                  gmm_artifact: dict, cusum_state: dict, config: dict) -> RegimeState: ...
```

---

### `kernel/indicators.py`
Inference-time indicator computation only (no training labels). Currently:
- LEAN: `_compute_indicators()` (lines 941–998), `_get_spy_adx()` (661–682)
- Notebook: cell 11 (inline in feature frame construction)

```python
def compute_rsi(close: pd.Series, period: int) -> pd.Series: ...
def compute_macd(close: pd.Series, fast, slow, signal) -> pd.DataFrame: ...
def compute_cci(high, low, close, period) -> pd.Series: ...
def compute_bbp(close, period) -> pd.Series: ...
def compute_adx(high, low, close, period) -> pd.Series: ...
def compute_williams_r(high, low, close, period) -> pd.Series: ...
def compute_obv_slope(close, volume, signal_period) -> pd.Series: ...
def compute_hurst_proxy(close: pd.Series, window: int = 20) -> pd.Series: ...
def compute_spy_features(spy_df: pd.DataFrame, config: dict) -> pd.DataFrame: ...
    # returns spy_realized_vol, spy_adx, spy_trend columns

def compute_all(df: pd.DataFrame, spec: dict) -> pd.DataFrame: ...
    # applies all indicators from indicator_spec; used at inference time
```

---

### `kernel/models.py`
All four model type implementations + calibration. Currently:
- LEAN: lines 1049–1135 (calibration) + `_traverse_tree`, `_bag_predict`,
  `_encode_q_state`, `_score_manual_rules`, `_xgb_predict`
- Notebook: via `common.models.*` at training time; same inference needed at sim time

```python
def load_artifact(path: Path) -> dict: ...

def predict_classification(artifact: dict, feature_row: pd.Series) -> float: ...
def predict_qlearning(artifact: dict, feature_row: pd.Series) -> float: ...
def predict_manual(artifact: dict, feature_row: pd.Series) -> float: ...
def predict_xgboost(artifact: dict, feature_row: pd.Series) -> float: ...

def calibrate_score(raw_score: float, calibration_meta: dict) -> float: ...
    # selects isotonic / Platt / constant based on n_samples in metadata

def load_and_score(model_dir: Path, indicator_row: pd.Series,
                   feature_columns: list[str]) -> tuple[float, float, str]:
    # returns (raw_score, calibrated_rank_score, model_action)
    # model_action: "buy" | "hold" | "sell"
```

---

### `kernel/exits.py`
All five exit types as pure functions. Currently:
- LEAN: OnData sell loop, lines 190–285
- Notebook: simulation cell lines ~80–160

```python
@dataclass
class HoldingState:
    ticker: str
    entry_price: float
    entry_date: date
    shares: int
    high_price: float    # trailing stop high-water mark
    sell_streak: int
    prev_close: float    # for single-day loss gate

@dataclass
class ExitSignal:
    ticker: str
    reason: str   # "trailing_stop" | "stop_loss" | "single_day_loss" |
                  # "max_hold" | "model_sell"

def check_trailing_stop(h: HoldingState, price: float, rp: dict) -> bool: ...
def check_cumulative_stop(h: HoldingState, price: float, rp: dict) -> bool: ...
def check_single_day_loss(h: HoldingState, price: float, rp: dict) -> bool: ...
def check_max_hold(h: HoldingState, today: date, rp: dict) -> bool: ...
def check_sell_streak(h: HoldingState, model_action: str,
                      n_required: int) -> tuple[bool, int]: ...
    # returns (should_exit, updated_streak)

def compute_exits(holdings: list[HoldingState],
                  prices: dict[str, float],
                  model_actions: dict[str, str],
                  today: date,
                  regime_state: RegimeState,
                  config: dict) -> list[ExitSignal]: ...
    # applies all exits in priority order; returns signals for caller to act on
```

---

### `kernel/selection.py`
Candidate scoring and selection pipeline. Currently the most-duplicated section:
- LEAN: lines 340–480 (buy phase)
- Notebook: simulation cell lines ~200–380

```python
@dataclass
class CandidateResult:
    ticker: str
    rank_score: float
    rs_score: float
    blend_score: float
    model_action: str

def compute_rs_score(ticker_close: pd.Series, etf_close: pd.Series,
                     lookback: int = 20) -> float: ...

def blend(rank_score: float, rs_score: float,
          weights: list[float]) -> float: ...

def is_earnings_blocked(ticker: str, today: date,
                        earnings_cal: dict, buffer_days: int) -> bool: ...

def is_wash_sale_blocked(ticker: str, last_sell: date | None,
                         wash_sale_days: int) -> bool: ...

def passes_sector_guard(ticker: str, sector_map: dict,
                        held: set[str], selected: set[str],
                        max_per_sector: int) -> bool: ...

def passes_correlation_guard(ticker: str, corr_matrix: pd.DataFrame,
                              held: set[str], selected: set[str],
                              threshold: float) -> bool: ...

def score_candidates(
    universe: list[str],
    price_data: dict[str, pd.DataFrame],     # ticker → OHLCV
    model_dir: Path,
    etf_prices: dict[str, pd.DataFrame],
    earnings_cal: dict,
    regime_state: RegimeState,
    config: dict,
) -> list[CandidateResult]: ...             # sorted by blend_score desc

def run_selection(
    candidates: list[CandidateResult],
    held_tickers: set[str],
    wash_sale_log: dict[str, date],
    corr_matrix: pd.DataFrame,
    regime_state: RegimeState,
    config: dict,
    n_open_slots: int,
) -> list[CandidateResult]: ...             # selected, in entry order
```

---

### `kernel/sizing.py`
Position sizing and oversize fallback. Currently:
- LEAN: `_size_position()` (lines 780–818)
- Notebook: simulation cell lines ~340–365

```python
def compute_position_size(
    price: float,
    portfolio_value: float,
    available_cash: float,
    reserve: float,
    regime_state: RegimeState,
    config: dict,
) -> int: ...
# Returns shares. 0 = skip.
# Applies regime_confidence scaling, then oversize fallback to 25% cap.
```

---

### `data/fetch.py`
Replaces `common.fetch_ohlcv`. Not imported by LEAN.

```python
def fetch_ohlcv(ticker: str, start: str, end: str,
                cache_dir: Path) -> pd.DataFrame: ...
def fetch_multiple(tickers: list[str], start: str, end: str,
                   cache_dir: Path) -> dict[str, pd.DataFrame]: ...
```

---

### `data/artifacts.py`
Load and save the three JSON artifacts.

```python
def load_gmm_artifact(path: Path) -> dict: ...
def load_correlation_matrix(path: Path) -> pd.DataFrame: ...
def load_earnings_calendar(path: Path) -> dict: ...
def save_gmm_artifact(gmm, path: Path): ...
def save_correlation_matrix(corr: pd.DataFrame, path: Path): ...
def save_earnings_calendar(cal: dict, path: Path): ...
```

---

### `training/features.py`
Build training feature frames (indicators + relative-close + forward-return labels).
Currently notebook cell 11 (~81 lines).

```python
def build_feature_frame(ticker: str, ticker_df: pd.DataFrame,
                         spy_df: pd.DataFrame, config: dict) -> pd.DataFrame: ...
    # adds indicators, relative close (stock/SPY×100), forward return labels,
    # train/test split column
```

---

### `training/tournament.py`
4-model tournament, OOS Sharpe evaluation, model export. Currently notebook cell 13
(168 lines) + cell 15 (115 lines).

```python
def train_one(model_type: str, train_df: pd.DataFrame,
              test_df: pd.DataFrame, config: dict) -> tuple[object, float, float]:
    # returns (model, oos_sharpe, is_sharpe)

def run_tournament(ticker: str, feature_df: pd.DataFrame,
                   config: dict) -> dict:
    # runs all 4 types; returns result dict with winner, metadata

def export_all(results: dict, strategy_dir: Path, config: dict): ...
    # saves best model + policy-metadata.json per ticker
```

---

### `training/calibration.py`
Score calibration and blend weight fitting. Currently in `scripts/recalibrate_scores.py`
and `common/models/scoring.py`.

```python
def fit_calibration(raw_scores: np.ndarray, labels: np.ndarray,
                    n_samples: int) -> dict: ...
    # returns calibration_meta dict; method selected by sample size

def fit_blend_weights(rank_scores: np.ndarray, rs_scores: np.ndarray,
                      outcomes: np.ndarray) -> list[float]: ...
    # logistic regression → normalised [w_rank, w_rs]

def recalibrate_all(strategy_dir: Path, config: dict): ...
    # updates score_calibration in each metadata file + blend_weights in config
```

---

### `training/regime_fit.py`
GMM fitting, correlation matrix computation, earnings calendar fetch. Currently:
- GMM: notebook cell 6
- Correlation: notebook cell 17
- Earnings: notebook cell 31 + `scripts/fetch_earnings_calendar.py`

```python
def fit_gmm(spy_df: pd.DataFrame, config: dict) -> object: ...
    # trains RegimeGMM; caller saves via artifacts.save_gmm_artifact()

def compute_correlation_matrix(price_data: dict[str, pd.DataFrame],
                                 config: dict) -> pd.DataFrame: ...

def fetch_earnings_calendar(tickers: list[str],
                             config: dict) -> dict: ...
```

---

### `simulation/engine.py`
Pure Python daily-loop simulation. Currently the 423-line notebook simulation cell.
Calls kernel functions exclusively — no `common/`, no notebook-only code.

```python
@dataclass
class SimResult:
    portfolio_values: pd.Series
    trade_log: list[dict]
    regime_log: pd.Series

def run_simulation(
    price_data: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    model_dir: Path,
    gmm_artifact: dict,
    corr_matrix: pd.DataFrame,
    earnings_cal: dict,
    config: dict,
    backtest_start: str,
    backtest_end: str,
) -> SimResult: ...
```

---

### `simulation/reporter.py`
Charts and stats. Currently notebook cells 22–29.

```python
def plot_results(result: SimResult, spy_df: pd.DataFrame, config: dict): ...
def print_stats(result: SimResult, spy_df: pd.DataFrame): ...
def plot_trade_detail(result: SimResult, price_data: dict): ...
def print_trade_log(result: SimResult): ...
```

---

### `main.py` (LEAN — stays at root, reduced to ~250 lines)

```python
from kernel.config    import load_config
from kernel.regime    import detect_regime, RegimeState
from kernel.exits     import compute_exits, HoldingState
from kernel.selection import score_candidates, run_selection
from kernel.sizing    import compute_position_size
from kernel.models    import load_artifact
from kernel.indicators import compute_spy_features

class RenQuant103(QCAlgorithm):
    def Initialize(self):
        # LEAN-only: symbols, consolidators, history warmup
        # Load config, artifacts, model artifacts into memory

    def OnData(self, data):
        # 1. Collect prices from LEAN slice (LEAN-only)
        # 2. detect_regime()     ← kernel
        # 3. compute_exits()     ← kernel → self.Liquidate()
        # 4. score_candidates()  ← kernel
        # 5. run_selection()     ← kernel → self.MarketOrder()
        # 6. compute_position_size() ← kernel
        # 7. Plot(), Debug()     ← LEAN-only
```

---

### `renquant_103.ipynb` (at strategy root, ~100 lines of glue)

```python
# Cell: Setup
from kernel.config    import load_config
from data.fetch       import fetch_multiple
from data.artifacts   import load_gmm_artifact, load_correlation_matrix

# Cell: Fit regime (run once or weekly)
from training.regime_fit import fit_gmm, compute_correlation_matrix

# Cell: Build features + train
from training.features   import build_feature_frame
from training.tournament import run_tournament, export_all

# Cell: Calibrate scores
from training.calibration import recalibrate_all

# Cell: Run simulation (calls same kernel as LEAN)
from simulation.engine   import run_simulation

# Cell: Charts
from simulation.reporter import plot_results, print_stats
```

---

### `live/runner.py` (reduced to ~150 lines)

```python
import sys
sys.path.insert(0, str(Path(__file__).parents[1] / "backtesting" / "renquant_103"))

from kernel.regime    import detect_regime
from kernel.exits     import compute_exits
from kernel.selection import score_candidates, run_selection
from kernel.sizing    import compute_position_size
from execution.signals import compute_signals_for_today
from execution.state   import PortfolioState
from broker.alpaca     import AlpacaBroker
```

---

### `live/broker/` (split from inline classes in runner.py)

| File | Lines (est.) | Contents |
|------|-------------|----------|
| `base.py` | 40 | `Broker` ABC: `get_positions`, `place_order`, `cancel_order`, `get_open_orders` |
| `alpaca.py` | 150 | `AlpacaBroker` — real + paper via env flag |
| `paper.py` | 80 | `PaperBroker` — in-memory simulation |
| `ibkr.py` | 30 | `IBKRBroker` stub |

### `live/execution/` (split from runner.py)

| File | Lines (est.) | Contents |
|------|-------------|----------|
| `signals.py` | 120 | `compute_signals_for_today()` — fetches data, calls kernel, returns buy/sell list |
| `state.py` | 100 | `PortfolioState` — holdings, wash-sale log, entry prices, position HWM |

---

## Test Refactor Plan

### Why the count drops

Today's 464 tests include ~200 tests that exist to verify notebook and LEAN agree with
each other (parity tests, cross-checks). After refactor, notebook and LEAN call the same
kernel functions — parity is structural, not testable. Those 200 tests collapse into
~5 integration smoke tests.

The remaining tests become cleaner kernel unit tests with no mocking.

### Estimated target test count: ~260

| Suite | Today | After |
|-------|-------|-------|
| `tests/kernel/test_regime.py` | scattered | ~50 |
| `tests/kernel/test_exits.py` | scattered | ~60 |
| `tests/kernel/test_selection.py` | scattered | ~60 |
| `tests/kernel/test_sizing.py` | scattered | ~20 |
| `tests/kernel/test_indicators.py` | 0 | ~20 |
| `tests/kernel/test_models.py` | scattered | ~30 |
| `tests/simulation/test_engine.py` | 5 (ledger parity) | ~15 |
| `tests/live/test_runner.py` | ~70 (runner_ranking) | ~50 |
| `tests/integration/test_parity.py` | ~222 (alignment) | ~5 |
| **Total** | **464** | **~310** |

310 tests, no mocking, all fast, no parity duplicates.

### Test file migration map

| Old file | New home | What happens |
|----------|----------|--------------|
| `test_policy_alignment.py` (222) | `tests/kernel/test_*.py` | Split by kernel module; parity cross-checks deleted |
| `test_lean_policies.py` (172) | `tests/kernel/test_*.py` + `tests/integration/` | Most become kernel unit tests |
| `test_simulation_policies.py` | `tests/kernel/test_*.py` | Become kernel unit tests |
| `test_strategy_ledger_parity.py` | `tests/simulation/test_engine.py` | Engine-level replay test |
| `test_runner_ranking.py` | `tests/live/test_runner.py` | Runner-specific tests stay |

### `tests/conftest.py` (new)

Shared fixtures extracted from all test files:

```python
@pytest.fixture
def minimal_config():
    return {
        "wash_sale_days": 30, "min_hold_days": 20,
        "consecutive_sell_signals": 3,
        "regime": {"hurst_window": 63, "choppy_hurst_floor": 0.20, ...},
        "regime_params": {"BULL_CALM": {...}, "CHOPPY": {...}, ...},
        ...
    }

@pytest.fixture
def flat_price_series():  # deterministic synthetic prices
    ...

@pytest.fixture
def minimal_model_artifact():  # smallest valid Classification artifact
    ...
```

### `tests/kernel/test_exits.py` structure

Each exit type gets its own test class. Example:

```python
class TestTrailingStop:
    def test_triggers_after_gain_threshold(self): ...
    def test_does_not_trigger_before_threshold(self): ...
    def test_trails_from_hwm_not_entry(self): ...
    def test_disabled_when_trigger_zero(self): ...

class TestCumulativeStop:
    def test_triggers_at_threshold(self): ...
    def test_bull_calm_wider_stop(self): ...
    def test_others_use_5pct(self): ...

class TestMaxHold:
    def test_triggers_at_limit(self): ...
    def test_not_triggered_within_window(self): ...
    def test_choppy_23d_vs_bull_500d(self): ...
```

No patching. No mocking. Pure inputs → assert output.

### `tests/integration/test_parity.py`

Only ~5 tests. The question is: given the exact same inputs, does `simulation/engine.py`
and `main.py` OnData produce the same decision? Since both call kernel functions, these
should trivially pass and serve as regression guards.

```python
def test_single_day_buy_decision_matches():
    # Feed identical state to engine.py step and to a thin OnData simulator
    # Assert same ticker selected, same shares computed

def test_single_day_sell_decision_matches():
    # Same for exits
```

---

## What Gets Eliminated

| Item | Why |
|------|-----|
| `common/` imports in renquant_103 | Replaced by kernel/ and strategy-local modules |
| `Notebooks/renquant_103.ipynb` | Moved to strategy dir |
| `scripts/fetch_earnings_calendar.py` | Logic moves to `training/regime_fit.py` |
| `scripts/recalibrate_scores.py` | Logic moves to `training/calibration.py` |
| LEAN's inline Hurst/CUSUM/GMM/ADX | Move to kernel/regime.py, kernel/indicators.py |
| LEAN's inline model inference (all 4 types) | Move to kernel/models.py |
| Runner's `_hurst_choppy_confidence()` | Replaced by kernel/regime.py |
| Parity test cross-checks (~150 tests) | Structural guarantee replaces test overhead |

---

## Migration Phases

Phases are ordered so each one is independently testable and committable.

### Phase 1 — Scaffold and move notebook (1–2 hours)

1. Create directories: `kernel/`, `data/`, `training/`, `simulation/` with `__init__.py`
2. Create `artifacts/` directory; move `spy-gmm-regime.json`, `watchlist-correlation.json`,
   `earnings-calendar.json` into it; update all path references
3. Move `Notebooks/renquant_103.ipynb` → `backtesting/renquant_103/renquant_103.ipynb`
4. Move `backtesting/renquant_103/config.py` → `kernel/config.py`; update `main.py` import
5. Run `pytest tests/ -q` — all 464 should still pass

**Commit: `chore: scaffold strategy dir structure and move notebook`**

---

### Phase 2 — Extract kernel from main.py (2–3 hours)

This is the core work. No behavior change — only moving code.

1. `kernel/regime.py` — move `_compute_hurst`, `_compute_cusum`, `_gmm_predict`,
   `_detect_regime` logic; add `detect_regime()` wrapper; add `RegimeState` dataclass
2. `kernel/indicators.py` — move `_compute_indicators`, `_get_spy_adx`
3. `kernel/models.py` — move `_calibrate_model_score`, `_traverse_tree`, `_bag_predict`,
   `_encode_q_state`, `_score_manual_rules`, `_xgb_predict`, `load_and_score`
4. `kernel/exits.py` — extract exit checks from OnData sell loop; add dataclasses
5. `kernel/selection.py` — extract buy phase: `_compute_rs_score`, `_is_earnings_blocked`,
   scoring loop, tiered selection, guards
6. `kernel/sizing.py` — extract `_size_position`
7. Rewrite `main.py` to import from kernel; target ~250 lines
8. Run `lean backtest .` — results must be identical to pre-refactor

**Commit: `refactor: extract strategy kernel from LEAN main.py`**

---

### Phase 3 — Wire notebook to kernel (1–2 hours)

1. `data/fetch.py` — extract data fetching (currently via `common.fetch_ohlcv`)
2. `data/artifacts.py` — extract artifact load/save
3. `training/features.py` — extract notebook cell 11 (feature frame construction)
4. `training/tournament.py` — extract cells 13 + 15 (training + export)
5. `training/calibration.py` — pull from `scripts/recalibrate_scores.py`
6. `training/regime_fit.py` — extract cells 6, 17, 31 (GMM, correlation, earnings)
7. `simulation/engine.py` — replace cell 21 (423 lines) with kernel calls
8. `simulation/reporter.py` — extract cells 22–29 (charts, stats, trade log)
9. Rewrite notebook to ~100 lines of glue; run end-to-end; charts must match

**Commit: `refactor: wire notebook to kernel; extract training and simulation modules`**

---

### Phase 4 — Wire live runner to kernel (1 hour)

1. Create `live/broker/` and `live/execution/` structure
2. Move broker classes into `live/broker/`
3. Create `live/execution/signals.py` using kernel
4. Create `live/execution/state.py`
5. Reduce `live/runner.py` to ~150 lines
6. Run live paper mode: `python -m live.runner --strategy renquant_103 --broker paper --once`

**Commit: `refactor: restructure live runner; wire to kernel`**

---

### Phase 5 — Refactor tests (2–3 hours)

1. Create `tests/conftest.py` with shared fixtures
2. Create `tests/kernel/test_regime.py` — port regime tests from `test_policy_alignment.py`
   and `test_lean_policies.py`; write as pure function tests
3. Similarly create `test_exits.py`, `test_selection.py`, `test_sizing.py`,
   `test_indicators.py`, `test_models.py`
4. Create `tests/simulation/test_engine.py` — port `test_strategy_ledger_parity.py`
5. Create `tests/live/test_runner.py` — port `test_runner_ranking.py`
6. Create `tests/integration/test_parity.py` — 5 smoke tests
7. Delete old test files once new coverage matches
8. Verify: `pytest tests/ -q` passes with ~310 tests, no mocking

**Commit: `test: refactor tests to kernel unit tests; remove parity duplicates`**

---

### Phase 6 — CI enforcement (30 minutes)

Add one test that the kernel has no `common/` imports:

```python
# tests/test_kernel_isolation.py
def test_kernel_has_no_common_imports():
    kernel_dir = Path("backtesting/renquant_103/kernel")
    for f in kernel_dir.glob("*.py"):
        src = f.read_text()
        assert "from common" not in src, f"{f.name} imports from common/"
        assert "import common" not in src, f"{f.name} imports common"
```

**Commit: `ci: enforce kernel isolation from common/`**

---

## Risks

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| LEAN backtest diverges after Phase 2 | Medium | Run backtest before and after; compare trade log line by line |
| LEAN can't import kernel subdirectory | Low | LEAN already imports `from config import ...` from same dir; subdirs work the same way. Verify with `lean backtest .` in Phase 2 step 8 before proceeding |
| Notebook simulation diverges from engine.py | Medium | Port cell 21 to engine.py first, validate on same date range before deleting cell |
| `live/` sys.path import is fragile | Low | Alternative: make `backtesting/renquant_103` a proper package with `pip install -e .` in setup; but sys.path is already used in the codebase and is fine |
| Phase 5 test count drops too far | Low | Track coverage with `pytest --cov`; any uncovered kernel branch gets a test |

---

## What Does NOT Change

- `lean backtest .` CLI interface — `main.py` stays at strategy root
- `python -m live.runner ...` CLI interface
- `strategy_config.json` schema
- Model artifact format (`policy-metadata.json`, model JSON files)
- Artifact JSON schemas (GMM, correlation, earnings)
- Scheduled scripts (`daily_103.sh`, `live_only_103.sh`) — they call runner.py, unchanged
- `common/` — untouched; renquant_101 and renquant_102 still depend on it

---

## Summary

| Metric | Before | After |
|--------|--------|-------|
| Lines in main.py | 1,388 | ~250 |
| Lines in runner.py | 1,597 | ~150 |
| Notebook simulation cell | 423 | ~30 (calls engine.py) |
| Files with copied decision logic | 3 | 0 (kernel is the one copy) |
| Test count | 464 | ~310 |
| Tests requiring mocking | ~200 | ~10 |
| Time to propagate a logic change | 3 files, 2 sessions | 1 function in kernel |
