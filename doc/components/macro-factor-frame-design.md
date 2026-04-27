# Macro Factor Frame — Design + Integration Plan

**Status:** 🔴 Design only.
**Driver:** User spec 2026-04-26 — *"按照proper的#3推进，计划好！把watchlist里的一部分item也可以换到 _macro_factor_frame里面，你给我一个完整的设计，如何把新的设计放到现有的架构中，如何确认python和rust的所有模型接受这个新的参数"*
**Goal:** add macro / cross-asset signals (VIX, HYG, UUP, DBC, …) as broadcast features to every panel row WITHOUT trying to trade them. Also: migrate non-tradable items currently in the watchlist (defensive ETFs, sector ETFs as RS comparators) into this frame.

---

## 1. What's a "macro factor frame"?

A panel-LTR feature today is **per-ticker, per-date** (e.g., `PLTR.rsi[2026-04-26]`).
A **macro factor** is **per-date only** — every ticker on the same date gets the same value (e.g., `vix_level[2026-04-26] = 14.32`).

The frame is **broadcast** at panel-build time: each row inherits the macro value for its date.

**Why "frame" (not "factor")?** We already have `raw_factor_frames` (per-ticker fundamentals/factor data). Macro is a **single shared frame** indexed by date alone.

---

## 2. What goes IN the macro frame

**Hard recommendation, top to bottom by info-density:**

| Symbol | Type | Signal | Why include |
|---|---|---|---|
| `^VIX` (or VXX as proxy) | Index | volatility regime | Highest info-density; fear gauge; leads BEAR detection |
| `HYG` | ETF | credit spread / risk-on-off | Leads equity 2-5d in stress |
| `UUP` | ETF | dollar strength (DXY proxy) | Macro driver; affects multinationals + commodities |
| `DBC` | ETF | broad commodities | Inflation + growth cycle, orthogonal to gold |
| `TLT` | ETF | long-bond rates | Already in `defensive_tickers` — MOVE here |
| `GLD` | ETF | gold / USD-debasement hedge | Already in `defensive_tickers` — MOVE here |
| `XLU` | ETF | utilities (low-beta defensive) | Already in `defensive_tickers` — MOVE here |
| `XLV` | ETF | healthcare (defensive) | Already in `defensive_tickers` — MOVE here |
| `KRE` | ETF | regional banks / credit health | 2023 SVB-style early warning |
| `MTUM` | ETF | momentum factor crowdedness | Confirms / contradicts our momentum thesis |
| `USMV` | ETF | low-vol factor | Defensive rotation indicator |

**What stays in `watchlist` as tradable:**
- All 99 individual equities (AAPL, AMZN, …)
- Nothing else.

**What stays in `sector_etf_map` as RS comparators:** sector ETFs that map a watchlist ticker to its sector benchmark (XLK for tech, XLI for industrials, etc.). The map's purpose is per-ticker RS scoring, **distinct** from macro broadcast.

**What MIGRATES from `watchlist` → macro frame:**
- `defensive_tickers = [GLD, TLT, XLV, XLU]` → all four become macro features.
- `BEAR` regime currently slots a defensive — needs a successor. **Recommendation:** in BEAR, hold cash + add a hedge ETF if equity exposure > 0. The hedge ETF would be a SINGLE explicit holding (e.g., `bear_hedge_ticker = SH` — inverse SPY) rather than picking among defensives.

---

## 3. Per-feature transformations

For each macro symbol, expose **3 features** to the panel:

| Feature | Formula | Why |
|---|---|---|
| `{sym}_level_z` | z-score of `close` over 252-day rolling window | Where are we in the regime band? |
| `{sym}_chg_5d_z` | z-score of `close.pct_change(5)` over 252-day rolling window | Recent acceleration |
| `{sym}_chg_20d_z` | z-score of `close.pct_change(20)` over 252-day rolling window | Cycle position |

So **11 macros × 3 = 33 new feature columns**. Combined with the current 24 panel features = **57 columns**. Within transformer's parameter budget (606k params today; +33 features × 128 d_model = ~4k extra params; trivial).

**Why z-score not raw**: trees are scale-invariant; transformer is NOT (saw the panel-ltr.json XGBoost trained on raw `close` blow up the attention softmax). Always z-score.

**Why rolling not full-history**: prevents look-ahead bias. The 252-day window is long enough to capture full vol regime but short enough to be reactive.

**Monotone constraints (XGBoost / LightGBM)** — when economically signed:

| Feature | Sign | Reasoning |
|---|---|---|
| `vix_level_z` | -1 | High VIX → expected return goes DOWN |
| `vix_chg_5d_z` | -1 | Vol spike up → risk-off |
| `hyg_level_z` | +1 | Tight credit → risk-on |
| `hyg_chg_5d_z` | +1 | Credit improving → risk-on |
| `uup_chg_5d_z` | -1 | $ strength up → US equity headwind |
| `dbc_level_z` | +0 (no constraint) | Inflation context — sign depends on regime |
| `gld_level_z` | -1 | Gold up → flight to safety → equity headwind |
| `tlt_chg_5d_z` | -1 | Bonds rallying → risk-off |
| `xlu_chg_5d_z` | -1 | Utilities leading → defensive rotation |
| Others | 0 | Sign ambiguous; let model learn |

---

## 4. Integration into existing architecture

### 4.1 New components to create

```
backtesting/renquant_104/
├── kernel/
│   └── macro.py                    # MacroFactorStore (parquet cache)
├── training_panel/
│   ├── context.py                  # +macro_factor_frame: pd.DataFrame | None
│   ├── pp_panel_training.py        # +LoadMacroFactorsTask, +MacroBroadcastTask
│   └── panel_frame.py              # build_panel_frame: +macro_frame param
└── strategy_config.json            # +panel_ltr.macro: {enabled, symbols, ...}
```

### 4.2 Schema changes

**`PanelTrainingContext` (training_panel/context.py)** — add field:

```python
@dataclass
class PanelTrainingContext:
    # … existing fields …
    macro_factor_frame: pd.DataFrame | None = None  # date-indexed, columns = z-scored macros
    macro_metadata: dict = field(default_factory=dict)  # {"symbols": [...], "n_features": int}
```

**`MacroFactorStore` (kernel/macro.py)** — new module mirroring `FundamentalsStore`:

```python
class MacroFactorStore:
    """Parquet cache for macro symbol OHLCV. One file per symbol.
    Schema: data/macro/{SYMBOL}.parquet with cols [open, high, low, close, volume]
    indexed by date.
    """
    def __init__(self, data_dir: str | Path = "data/macro"): ...
    def load(self, symbol: str) -> pd.DataFrame | None: ...
    def save(self, symbol: str, df: pd.DataFrame) -> None: ...
```

**Config block (strategy_config.json + golden)**:

```json
{
  "panel_ltr": {
    "macro": {
      "enabled": true,
      "cache_dir": "data/macro",
      "symbols": ["VXX", "HYG", "UUP", "DBC", "GLD", "TLT", "XLV", "XLU", "KRE", "MTUM", "USMV"],
      "transforms": ["level_z", "chg_5d_z", "chg_20d_z"],
      "rolling_window": 252,
      "monotone_constraints": {
        "vix_level_z": -1,
        "vix_chg_5d_z": -1,
        "hyg_level_z": 1,
        "hyg_chg_5d_z": 1,
        "uup_chg_5d_z": -1,
        "gld_level_z": -1,
        "tlt_chg_5d_z": -1,
        "xlu_chg_5d_z": -1
      }
    }
  }
}
```

### 4.3 New tasks (pp_panel_training.py)

**`LoadMacroFactorsTask`** (in PanelDataJob, after LoadFundamentalsTask):

```python
class LoadMacroFactorsTask(PanelTask):
    """Load OHLCV for each macro symbol; compute level/chg z-scores;
    assemble single date-indexed DataFrame with all macro features."""

    def run(self, ctx: PanelTrainingContext) -> bool | None:
        cfg = ctx.config.get("panel_ltr", {}).get("macro", {})
        if not cfg.get("enabled", False):
            return True

        store = MacroFactorStore(_resolve_cache_dir(cfg["cache_dir"], ctx.config))
        rolling_window = int(cfg.get("rolling_window", 252))
        symbols = list(cfg.get("symbols", []))
        transforms = list(cfg.get("transforms", ["level_z", "chg_5d_z", "chg_20d_z"]))

        cols: dict[str, pd.Series] = {}
        for sym in symbols:
            ohlcv = store.load(sym)
            if ohlcv is None or ohlcv.empty:
                log.warning("LoadMacroFactorsTask: %s missing — skip", sym)
                continue
            close = ohlcv["close"]
            sym_low = sym.lower()
            for t in transforms:
                col_name = f"{sym_low}_{t}"
                if t == "level_z":
                    s = (close - close.rolling(rolling_window).mean()) \
                        / close.rolling(rolling_window).std()
                elif t == "chg_5d_z":
                    chg = close.pct_change(5)
                    s = (chg - chg.rolling(rolling_window).mean()) \
                        / chg.rolling(rolling_window).std()
                elif t == "chg_20d_z":
                    chg = close.pct_change(20)
                    s = (chg - chg.rolling(rolling_window).mean()) \
                        / chg.rolling(rolling_window).std()
                else:
                    log.warning("Unknown macro transform %s — skip", t)
                    continue
                cols[col_name] = s

        if not cols:
            log.warning("LoadMacroFactorsTask: no macro features built")
            return True
        ctx.macro_factor_frame = pd.DataFrame(cols).sort_index()
        ctx.macro_metadata = {
            "symbols":    symbols,
            "transforms": transforms,
            "n_features": len(cols),
            "rolling_window": rolling_window,
        }
        log.info("LoadMacroFactorsTask: %d features, %d dates",
                 len(cols), len(ctx.macro_factor_frame))
        return True
```

**Modification to `BuildPanelTask`** (line ~1187 of pp_panel_training.py) — pass macro to `build_panel_frame`:

```python
panel, group_sizes, meta = build_panel_frame(
    ff_wl, lab_wl, sec_wl,
    factor_frames=fac_wl,
    macro_frame=ctx.macro_factor_frame,   # NEW param
    listing_dates=ctx.listing_dates,
    # … rest unchanged …
)
```

**Modification to `BuildHourlyResolutionPanelTask`** — same `macro_frame=` injection (the hourly path). Macro values broadcast per (date, ticker) hour by hour — same value for all hours within a date.

### 4.4 Modification to `panel_frame.py::build_panel_frame`

The function builds a long-format panel by iterating tickers. Add a final step:

```python
def build_panel_frame(
    feature_frames: dict[str, pd.DataFrame],
    label_series:   dict[str, pd.Series],
    sector_map:     dict[str, str],
    *,
    factor_frames:  dict[str, pd.DataFrame] | None = None,
    macro_frame:    pd.DataFrame | None = None,        # NEW
    # … rest unchanged …
):
    # … existing loop builds rows ticker-by-ticker …
    panel = pd.concat(rows, ignore_index=True)

    # NEW: broadcast macro frame onto panel by date
    if macro_frame is not None and not macro_frame.empty:
        # macro_frame is indexed by date; reindex to panel dates and merge
        panel = panel.merge(
            macro_frame, left_on="date", right_index=True, how="left",
        )
        # Forward-fill within ticker (in case some macros have weekend gaps)
        macro_cols = [c for c in macro_frame.columns]
        panel[macro_cols] = panel.groupby("ticker", group_keys=False)[macro_cols].ffill()
        # Trailing NaN (warmup for rolling-z) goes to 0 — same convention as
        # existing factor frame handling.
        panel[macro_cols] = panel[macro_cols].fillna(0.0)
    # … rest unchanged …
```

### 4.5 Fetch script

**`scripts/fetch_macro_factors.py`** — driver to populate `data/macro/`:

```python
"""Fetch OHLCV for each macro symbol via yfinance + save to MacroFactorStore."""
import yfinance as yf
from kernel.macro import MacroFactorStore

SYMBOLS = ["VXX", "HYG", "UUP", "DBC", "GLD", "TLT", "XLV", "XLU", "KRE", "MTUM", "USMV"]
store = MacroFactorStore("data/macro")
for sym in SYMBOLS:
    df = yf.Ticker(sym).history(period="10y", interval="1d")
    df.columns = df.columns.str.lower()
    store.save(sym, df[["open", "high", "low", "close", "volume"]])
    print(f"✓ {sym}: {len(df)} rows")
```

Hook into `daily_104.sh` as Step 1c (after baseline tournament, before LEAN export).

### 4.6 Inference / live runner

**`adapters/runner.py::make_context`** — load macro frame:

```python
# Existing: panel-LTR feature prep when ranking.panel_scoring.enabled
if panel_enabled:
    # … existing feature_frame, factor_frame load …

    # NEW: macro_factor_frame
    macro_cfg = ctx.config.get("panel_ltr", {}).get("macro", {})
    if macro_cfg.get("enabled"):
        from kernel.macro import build_macro_frame_for_inference
        ctx.macro_factor_frame = build_macro_frame_for_inference(
            macro_cfg, ctx.config, today=ctx.today,
        )
```

`build_macro_frame_for_inference` is the inference-time twin of `LoadMacroFactorsTask` — same z-score logic but uses live `data/macro/` parquet.

---

## 5. Backend integration

### 5.1 XGBoost / LightGBM (no schema change)

**Feature columns are dataframe columns.** The `feature_cols` list (set by `BuildPanelTask`) is computed from `panel.columns - exclude`. After `build_panel_frame` merges macro columns in, they're automatically picked up by `feature_cols` and used by both XGBoost and LightGBM.

**Required change to LightGBM/XGBoost:**
- Update `monotone_constraints` config to include macro signs (already in §4.2 config).
- **Test parity:** add `tests/test_panel_macro.py::test_macro_columns_in_feature_cols` to assert `vix_level_z` etc. appear in `ctx.feature_cols` after panel build.

### 5.2 NGBoost head

NGBoost head fits on `feature_cols` directly. Same story as XGBoost — macro cols come along for the ride.

**Required change:** none. `tests/test_ngboost_head.py` automatically tests against whatever `feature_cols` is.

### 5.3 PyTorch transformer (`training_panel/transformer_model.py`)

The transformer takes `(n_groups, max_tickers, n_features)` tensor. `n_features` is `len(feature_cols)`. Adding macros increases `n_features` from 24 → 57.

**Required change:** the `_PanelTransformer.__init__` already accepts `n_features` as a constructor arg (line 162-220). The input projection layer auto-adapts: `nn.Linear(n_features, d_model)`.

**Test:** `tests/test_panel_transformer.py::test_handles_57_features` — pin via shape assertion.

**Performance:** 33 extra features × 128 d_model × 3 layers ≈ 25k extra params (out of 606k). Negligible.

### 5.4 Rust transformer scorer (`rust/transformer_scorer/`)

This is the inference-time replica. The Rust code reads `feature_cols` from the artifact's JSON sidecar (`config.rs::PanelConfig.feature_cols: Vec<String>`).

**Required change:**
- `rust/transformer_scorer/src/dataset.rs`: ensure feature parsing handles arbitrary column names (already does — it iterates `feature_cols`).
- `rust/transformer_scorer/src/config.rs::PanelConfig`: no change needed — it reads `feature_cols` from JSON.
- **Critical**: ensure the artifact JSON's `feature_cols` order matches the Python training-time order. Today this is guaranteed by `ctx.feature_cols` being a list (ordered), so the Rust scorer reads the exact same order.

**Tests:**
- `rust/transformer_scorer/tests/poc_parity.rs` — already does Python-vs-Rust parity on a fixed panel. Add a regression test with a panel containing macro columns to assert per-row score parity within 1e-5.
- New test: `rust/transformer_scorer/tests/macro_parity.rs` — load a macro-trained artifact, score a synthetic panel, compare to Python's score.

### 5.5 Calibrator + global_calibration

Calibrator fits `panel_score → P(outperform)` on raw scores. Macros change the raw scores but the calibration mechanism is **agnostic** to feature provenance. No change needed.

**Test:** `tests/test_global_calibrator.py` already validates round-trip; runs against any feature set.

---

## 6. Validation: how to confirm every model accepts the new parameter

**Phased verification** (each phase is a commit):

### Phase A — Unit tests for each backend (~2 hours)

```python
# tests/test_panel_macro.py
class TestMacroFrameLoad:
    def test_load_macro_factors_task_emits_frame()
    def test_macro_frame_indexed_by_date()
    def test_macro_z_score_columns_match_config()
    def test_disabled_flag_is_no_op()

class TestBuildPanelWithMacro:
    def test_macro_columns_in_feature_cols()
    def test_macro_columns_broadcast_to_all_tickers_per_date()
    def test_panel_size_unchanged_with_macro_added()  # macros don't add ROWS
    def test_macro_rolling_window_warmup_filled_zero()

class TestBackendsConsumeMacro:
    def test_xgboost_train_with_macro_feature_set()
    def test_lightgbm_train_with_macro_feature_set()
    def test_ngboost_train_with_macro_feature_set()
    def test_transformer_train_with_macro_feature_set()  # CPU device, 2 epochs

class TestInferenceParity:
    def test_runner_loads_macro_frame_when_enabled()
    def test_inference_panel_has_macro_columns()
    def test_panel_scorer_consumes_macro_at_inference()
```

### Phase B — Rust parity test (~1 hour)

```rust
// rust/transformer_scorer/tests/macro_parity.rs
#[test]
fn rust_python_score_parity_with_macro() {
    // 1. Train a small Python transformer on synthetic panel WITH macro cols
    // 2. Save the .pt + .json sidecar to a tmp path
    // 3. Load via Rust PanelScorer::load(path)
    // 4. Score a synthetic eval panel (same shape) in both Python + Rust
    // 5. Assert per-row score within 1e-5
}
```

### Phase C — Sim A/B (~50 min)

```bash
# Variant A: golden v4.1 (no macro)
python scripts/validate_buy_logic.py --baseline

# Variant B: macro_enabled
python scripts/validate_buy_logic.py --macro-enabled
```

Promotion criteria per CLAUDE.md §2a:
- Sim A/B variant B beats A by ≥ +2 APY pts → promote macro-enabled to golden
- < +2 pts but mechanism-clean (just added features) → ship at any positive margin
- Negative margin → audit per CLAUDE.md §2b before shipping

### Phase D — End-to-end smoke test (~30 min)

`tests/test_macro_e2e.py` (opt-in via `RENQUANT_FULL_SIM=1`):
1. Fetch live data for 11 macros
2. Run full `train_104.py` with macro enabled
3. Run live runner once with paper broker
4. Assert: panel-ltr.json metadata.feature_cols includes macro columns
5. Assert: trade-decision logs reference macro feature values

---

## 7. Migration plan (existing config → new config)

### Step 1 — Cache backfill (1-time)

```bash
python scripts/fetch_macro_factors.py
# Writes data/macro/{VXX,HYG,UUP,DBC,GLD,TLT,XLV,XLU,KRE,MTUM,USMV}.parquet
```

### Step 2 — Config update (paired)

In `strategy_config.json` AND `strategy_config.golden.json`:
```json
{
  "panel_ltr": {
    "macro": { "enabled": false, ... }    // ship as DEFAULT OFF
  },
  "defensive_tickers": [],                  // empty — they're macro features now
  "bear_defensive_slots": 0,
  "bear_defensive_pct": 0,
  "bear_hedge_ticker": "SH"                 // single inverse-SPY hedge
}
```

### Step 3 — Code ship (with `enabled=false` to start)

- All Phase A/B/C tests passing
- Default OFF — production unaffected
- Roll out: flip `enabled=true` after sim A/B confirms uplift

### Step 4 — Sim A/B (Phase C above)

If positive: flip enabled=true in golden + production-config in same commit.
If negative: keep code merged (defenses against future regression), keep flag off, write postmortem.

### Step 5 — Cleanup (after 4 weeks of live data)

- Retire `defensive_tickers` config slot entirely (currently empty after Step 2)
- Remove BEAR-defensive code paths (`task_candidates.py` BEAR branch)
- Replace with single `bear_hedge_ticker` logic

---

## 8. Risks + mitigations

| Risk | Mitigation |
|---|---|
| Macro data has weekend/holiday gaps mismatching equity calendar | `panel.merge` with ffill within ticker handles weekends. NYSE-holiday alignment via `pandas_market_calendars` (already a dep) |
| 252-day rolling z requires 1 year of warmup → can't backfill far | Document training_window_years requirement = old window + 252d warmup |
| 11 macro symbols × 3 transforms = 33 features doubles feature count → overfitting risk | XGBoost L1/L2 regularization + monotone constraints. CPCV will catch overfitting in oos_mean_ic. If IC drops, drop 5 weakest macros via `panel_ltr.drop_cols`. |
| Rust scorer trained on old panel won't have macro feature_cols → must retrain | Versioned artifact: `panel-transformer.pt.macro.bak.json`. Load checks feature_cols match config; mismatch → error not silent miscalibration. |
| BEAR regime currently uses defensive_tickers — losing that needs replacement strategy | Step 5 above replaces with explicit `bear_hedge_ticker` (SH/PSQ/SDS) — always-on inverse hedge during BEAR rather than rotating defensives. Cleaner, simpler |
| yfinance VXX is an ETF (not raw VIX index); rolls cause drift | Document. Switch to `^VIX` direct index later if precision matters. VXX is OK for regime signal even with roll drift |

---

## 9. Implementation effort summary

| Phase | Component | Effort | Reversibility |
|---|---|---|---|
| 1 | `kernel/macro.py` + `MacroFactorStore` | 1 hr | trivial — new file |
| 2 | `LoadMacroFactorsTask` | 2 hr | trivial — new task |
| 3 | `panel_frame.py::build_panel_frame` macro broadcast | 1 hr | needs flag default-off |
| 4 | `BuildPanelTask` + `BuildHourlyResolutionPanelTask` wiring | 30 min | trivial |
| 5 | Live runner inference path | 2 hr | flag default-off |
| 6 | Test phase A (Python backends) | 2 hr | additive |
| 7 | Test phase B (Rust parity) | 1 hr | additive |
| 8 | `scripts/fetch_macro_factors.py` + cache backfill | 1 hr | new file |
| 9 | Test phase C (sim A/B 50min × 2) | 2 hr | data only |
| 10 | Documentation + roadmap update | 1 hr | additive |
| **Total** | | **~14 hr** | one full coding day |

---

## 10. References

- **Cont (2001)** — *Empirical properties of asset returns* (Quant. Finance) — establishes that volatility (VIX) is the dominant cross-asset signal
- **Whaley (2000)** — *Investor Fear Gauge* (J. Portfolio Management) — VIX as risk signal
- **Cochrane (2011)** — *Discount Rates* (J. Finance) — credit spreads (HYG) explain ~30% of equity premium variation
- **Ang & Bekaert (2007)** — *Stock return predictability: Is it there?* — short-rate + credit spread are robust predictors
- **Asness, Frazzini, Pedersen (2019)** — *Quality minus Junk* — defensive factor (low-vol) as risk premium
- **López de Prado (2018)** — *Advances in Financial ML* Ch. 8 (Feature Importance) — methodology to validate the proposed monotone constraints empirically

## 11. Cross-references

- `doc/components/panel-ltr.md` — primer for the consuming model
- `doc/components/training-pipeline.md` — where this fits in the data flow
- `doc/arch/strategy-104.md` — overall strategy where macro features compose with ticker features
- `doc/roadmap.md` — schedule + ROI for this work
- This doc itself: `doc/components/macro-factor-frame-design.md`
