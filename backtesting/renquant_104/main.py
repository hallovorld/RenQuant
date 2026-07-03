"""Adaptive Regime Multi-Stock Strategy — LEAN entry point.

Thin wrapper: Initialize sets up state and wires the adapter + pipeline.
OnData calls adapter.make_context → pipeline.run → adapter.commit.

LEAN-safe: no common/ imports.  Docker can access kernel/ and adapters/ locally.

LEAN data normalization: Adjusted (split + dividend).
Sim data source: yfinance (default Adjusted).
Both produce continuous price series with dividends reinvested.
Cash dividend events are NOT modeled separately; they appear as price
continuity instead. (Trade-off: ignores tax timing but maintains
path-equivalence between sim and LEAN.)
"""
from AlgorithmImports import *  # noqa: F401,F403
import json
import math
from pathlib import Path

from kernel.config       import load_config, split_date_parts, BULL_CALM, BULL_VOLATILE, CHOPPY, BEAR, REGIMES, artifact_path
from kernel.regime       import RegimeState, load_gmm_artifact
from kernel.pipeline     import InferencePipeline, SellOnlyPipeline
from kernel.pipeline.task_benchmark_sleeve import (
    benchmark_sleeve_ticker,
    is_benchmark_sleeve_enabled,
)
from kernel.walk_forward import (
    assert_correlation_no_leakage,
    assert_gmm_no_leakage,
    assert_lean_panel_no_leakage,
    parse_correlation_artifact,
)
from kernel.execution.slippage import SlippageConfig, slip_fill_price
from kernel.preflight    import run_preflight
from adapters.lean       import LeanAdapter
from adapters.sleeve_prices import parking_sleeve_price_tickers

CONFIG = load_config()


class LeanHalfSpreadSlippageModel:
    """LEAN slippage adapter backed by kernel.execution.slippage.

    LEAN expects a positive price-distance; its fill model applies the buy/sell
    sign. Sim computes the signed fill price directly through the same helper.
    """

    def __init__(self, cfg: SlippageConfig):
        self._cfg = cfg

    def GetSlippageApproximation(self, asset, order):  # noqa: N802 - LEAN API
        try:
            price = float(getattr(asset, "Price", 0.0) or 0.0)
            qty = abs(float(getattr(order, "Quantity", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(price) or price <= 0.0 or qty <= 0.0:
            return 0.0
        side = "buy" if float(getattr(order, "Quantity", 0.0) or 0.0) > 0 else "sell"
        fill = slip_fill_price(
            market_price=price,
            side=side,
            shares=qty,
            adv_shares=None,
            cfg=self._cfg,
        )
        if not math.isfinite(fill) or fill <= 0.0:
            return 0.0
        return abs(fill - price)


class AdaptiveRegimeMultiStockStrategy(QCAlgorithm):

    # ── Initialization ─────────────────────────────────────────────────────────

    def Initialize(self):
        sy, sm, sd = split_date_parts(CONFIG["backtest_start"])
        ey, em, ed = split_date_parts(CONFIG["backtest_end"])
        self.SetStartDate(sy, sm, sd)
        self.SetEndDate(ey, em, ed)
        self.SetCash(CONFIG["initial_cash"])

        # ── Data normalization (Track C7, 2026-05-10) ────────────────────────
        # Adjusted matches yfinance default (split-adjusted + dividend-adjusted),
        # keeping LEAN/sim parity. See execution audit 2026-05-10. Setting on
        # UniverseSettings applies the default to ALL subsequent AddEquity
        # calls (watchlist, sector ETFs, benchmark) — single source of truth.
        # Without this, LEAN's default depends on the data source and may
        # diverge from sim on dividend-paying stocks (e.g. AAPL ~0.6%/y).
        try:
            self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted
        except (AttributeError, NameError):
            pass

        # ── Execution model parity with SimAdapter (Track Batch A, 2026-05-10).
        # LEAN's brokerage model handles commission + slippage + T+2 cash
        # settlement natively when SetBrokerageModel is set — this is the
        # single place we tell LEAN "treat orders like an Alpaca account."
        # Without this, LEAN fills at exact bar-close and settles T+0,
        # inflating reported APY by ~1.5-2.5%/yr.
        #
        # AlpacaBrokerageModel uses:
        #   - Commission: $0 (Alpaca zero-commission for stocks)
        #   - Slippage:   ConstantSlippageModel (overridden below to mirror
        #                 sim's bps-based model exactly)
        #   - Settlement: AccountType.Margin (T+2 for US equity by default)
        #
        # SimAdapter's industry-grade analog lives in
        # `kernel.execution.{fees,slippage,t2_settlement}`. Defaults match.
        # Parity-check: see tests/test_sim_execution_integration.py.
        exec_cfg = CONFIG.get("execution", {}) or {}
        if bool(exec_cfg.get("enabled", True)) and not bool(exec_cfg.get("legacy_no_fees", False)):
            # AlpacaBrokerageModel: matches our zero-commission default;
            # falls back to InteractiveBrokers when Alpaca constant is
            # absent on older LEAN builds.
            try:
                self.SetBrokerageModel(BrokerageName.Alpaca, AccountType.Margin)
            except (AttributeError, NameError, RuntimeError):
                # Older LEAN: Alpaca constant may not exist → use IB
                # which has a comparable commission schedule for retail
                # equities ($0.005/share, capped; ≈ 2 bps on $100 stocks).
                try:
                    self.SetBrokerageModel(BrokerageName.InteractiveBrokers, AccountType.Margin)
                except Exception:
                    pass
            slip_cfg = SlippageConfig(
                half_spread_bps=float(exec_cfg.get("half_spread_bps", 2.0)),
                impact_bps_per_pct_adv=float(exec_cfg.get("impact_bps_per_adv", 0.0)),
            )
            self.SetSlippageModel(LeanHalfSpreadSlippageModel(slip_cfg))

        self._config       = CONFIG
        self._strategy_dir = Path(__file__).resolve().parent
        self._config["_strategy_dir"] = str(self._strategy_dir)
        self._preflight_ok = False
        run_preflight(
            CONFIG,
            broker=None,
            strategy_dir=self._strategy_dir,
            strict=True,
            run_mode="full",
        )
        self._preflight_ok = True
        self._watchlist    = CONFIG["watchlist"]
        self._benchmark    = CONFIG.get("benchmark", "SPY")

        # ── Symbols ──────────────────────────────────────────────────────────
        self.symbols: dict[str, Symbol] = {}
        for ticker in self._watchlist:
            self.symbols[ticker] = self.AddEquity(ticker, Resolution.Daily).Symbol

        self._sector_etf_map     = CONFIG.get("sector_etf_map", {})
        self._sector_etf_symbols: dict[str, Symbol] = {}
        for etf in set(self._sector_etf_map.values()):
            if etf not in self.symbols:
                self._sector_etf_symbols[etf] = self.AddEquity(etf, Resolution.Daily).Symbol

        self._spy_sym = self.AddEquity(self._benchmark, Resolution.Daily).Symbol
        sleeve_ticker = benchmark_sleeve_ticker(CONFIG)
        if (
            is_benchmark_sleeve_enabled(CONFIG)
            and sleeve_ticker
            and sleeve_ticker != self._benchmark
            and sleeve_ticker not in self.symbols
            and sleeve_ticker not in self._sector_etf_symbols
        ):
            self._sector_etf_symbols[sleeve_ticker] = self.AddEquity(
                sleeve_ticker,
                Resolution.Daily,
            ).Symbol

        # Parking-sleeve legs (st104 #39 follow-up): subscribe
        # sleeve.spy_symbol / sleeve.sgov_symbol only when sleeve.enabled —
        # same conditional-subscription pattern as the benchmark sleeve
        # above. Returns [] when the flag is off/absent (the shipped
        # default), so this block is byte-inert in production.
        for parking_ticker in parking_sleeve_price_tickers(CONFIG):
            if (
                parking_ticker != self._benchmark
                and parking_ticker not in self.symbols
                and parking_ticker not in self._sector_etf_symbols
            ):
                self._sector_etf_symbols[parking_ticker] = self.AddEquity(
                    parking_ticker,
                    Resolution.Daily,
                ).Symbol

        # ── Per-run state ────────────────────────────────────────────────────
        from kernel.exits import HoldingState  # local import keeps global scope clean
        self._holdings: dict[str, HoldingState] = {}
        self._last_sell_dates: dict = {}
        self._last_sell_pls: dict = {}
        self._regime_state  = RegimeState()
        self._spy_returns: list[float] = []
        self._prev_closes: dict[str, float] = {}
        self._regime_counts = {r: 0 for r in REGIMES}
        self._hwm           = float(CONFIG["initial_cash"])
        self._skip_buys     = False

        # ── Tax config ───────────────────────────────────────────────────────
        # Audit #87 #92 — every fallback uses the HIGHER bracket (50%/32%)
        # so a missing tax config block conservatively over-estimates rather
        # than under-estimates tax drag. Aligned across LEAN, kernel.rotation,
        # task_rotation, and SimAdapter (2026-04-24).
        tax_cfg               = CONFIG.get("tax", {})
        self._tax_short       = float(tax_cfg.get("short_term_rate", 0.50))
        self._tax_long        = float(tax_cfg.get("long_term_rate", 0.32))
        self._tax_thresh_days = int(tax_cfg.get("long_term_threshold_days", 365))

        # ── Telemetry counters ────────────────────────────────────────────────
        self._total_tax        = 0.0
        self._st_trades        = 0
        self._lt_trades        = 0
        self._executed_buys    = 0
        self._executed_sells   = 0
        self._stop_exits       = 0
        self._trail_exits      = 0
        self._sdl_exits        = 0
        self._rotation_exits   = 0
        self._blocked_wash     = 0
        self._blocked_min_hold = 0
        self._blocked_streak   = 0
        self._sector_blocks    = 0
        self._corr_blocks      = 0
        self._earnings_blocks  = 0
        self._transition_blocks = 0
        self._velocity_blocks  = 0

        # ── Artifacts ────────────────────────────────────────────────────────
        self._models: dict[str, dict] = {}
        self._load_all_models()

        # AUDIT 2026-05-10 §5.13.5 — same leakage guard as SimAdapter.
        # Routes through kernel.walk_forward.leakage_guard.assert_no_leakage.
        # Regression invariant pinned in tests/test_lean_guard.py.
        assert_lean_panel_no_leakage(
            config=self._config,
            strategy_dir=self._strategy_dir,
            is_live_mode=getattr(self, "LiveMode", False),
        )

        regime_cfg  = CONFIG.get("regime", {})
        # 2026-05-11 sim/prod isolation: defaults relocated to prod/.
        # Sim configs override these keys to sim/<file>.
        self._gmm   = load_gmm_artifact(artifact_path(
            regime_cfg.get("gmm_artifact", "prod/spy-gmm-regime.json")))
        assert_gmm_no_leakage(
            self._gmm,
            CONFIG.get("backtest_start"),
            is_live_mode=getattr(self, "LiveMode", False),
            context="LEAN main.py gmm",
        )
        # AUDIT 2026-05-10 §5.13.5 — correlation as-of-date leakage guard.
        # Loads raw artifact, parses v1/v2 schema, asserts as_of_date <=
        # backtest_start in backtest mode (LiveMode skips). Routes through
        # kernel.walk_forward.correlation_guard.
        _corr_raw = self._load_json_artifact(
            regime_cfg.get("correlation_artifact", "prod/watchlist-correlation.json"), "Correlation")
        self._corr, _corr_as_of = parse_correlation_artifact(_corr_raw)
        assert_correlation_no_leakage(
            _corr_as_of,
            CONFIG.get("backtest_start"),
            is_live_mode=getattr(self, "LiveMode", False),
            allow_legacy_without_as_of=bool(
                regime_cfg.get("allow_legacy_correlation_without_as_of", False)
            ),
            context="LEAN main.py corr",
        )
        self._earnings = self._load_json_artifact(
            regime_cfg.get("earnings_artifact", "prod/earnings-calendar.json"), "Earnings")

        # ── Pipeline + adapter ───────────────────────────────────────────────
        self._pipeline = InferencePipeline()
        self._adapter  = LeanAdapter(self)

        self._setup_charts()
        self.SetWarmUp(90)

    # ── Main event loop ────────────────────────────────────────────────────────

    def OnData(self, data: Slice):
        if self.IsWarmingUp:
            return

        ctx = self._adapter.make_context(data)
        self._pipeline.run(ctx)
        self._adapter.commit(ctx)

        self._plot_regime(ctx.confidence, ctx.regime)
        self.Plot("Portfolio", "Positions", len(self._holdings))
        self.Debug(
            f"{self.Time.date()} REGIME={ctx.regime} conf={ctx.confidence:.2f} "
            f"held={list(self._holdings.keys())}"
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _load_all_models(self) -> None:
        from kernel.pipeline.job_universe import UniverseContext, LoadUniverseJob
        uctx = UniverseContext(
            config=CONFIG,
            strategy_dir=self._strategy_dir,
            held_tickers=self._current_held_tickers(),
        )
        LoadUniverseJob().run(uctx)
        self._models = uctx.loaded_models
        self._universe_rejections = dict(uctx.rejections)
        for ticker, reason in uctx.rejections:
            self.Log(f"WARNING: {ticker} {reason}, skipping")
        self.Log(f"Loaded {len(self._models)}/{len(self._watchlist)} models: {sorted(self._models)}")

    def _current_held_tickers(self) -> set[str]:
        """Return broker/LEAN portfolio-held tickers for universe exemptions.

        Universe floors and auto-drop gates protect new buys. They must not
        remove a currently-held ticker's model, because that kills model-sell
        and rotation logic for an open position.
        """
        held: set[str] = set(getattr(self, "_holdings", {}) or {})
        for ticker, sym in getattr(self, "symbols", {}).items():
            try:
                qty = float(self.Portfolio[sym].Quantity)
            except Exception:
                continue
            if math.isfinite(qty) and abs(qty) > 1e-9:
                held.add(ticker)
        return held

    def _load_json_artifact(self, filename: str, label: str) -> dict | None:
        p = artifact_path(filename)
        if not p.exists():
            self.Log(f"WARNING: {label} artifact {filename} not found")
            return None
        with open(p) as f:
            return json.load(f)

    # ── Charts & end-of-algo ───────────────────────────────────────────────────

    def _setup_charts(self) -> None:
        rc = Chart("Regime")
        rc.AddSeries(Series("State",      SeriesType.Line, ""))
        rc.AddSeries(Series("Confidence", SeriesType.Line, "%"))
        self.AddChart(rc)
        pc = Chart("Portfolio")
        pc.AddSeries(Series("Positions",  SeriesType.Line, "count"))
        self.AddChart(pc)

    def _plot_regime(self, conf: float, regime: str) -> None:
        numeric = {BULL_CALM: 3, BULL_VOLATILE: 2, CHOPPY: 1, BEAR: 0}
        self.Plot("Regime", "State",      numeric.get(regime, -1))
        self.Plot("Regime", "Confidence", conf * 100)

    def OnEndOfAlgorithm(self):
        total_days = sum(self._regime_counts.values()) or 1
        pcts = {r: f"{v/total_days:.1%}" for r, v in self._regime_counts.items()}
        stats = {
            "Watchlist":              str(len(self._watchlist)),
            "Active Models":          str(len(self._models)),
            "Executed Buys":          str(self._executed_buys),
            "Executed Sells":         str(self._executed_sells),
            "Stop Loss Exits":        str(self._stop_exits),
            "Single Day Loss Exits":  str(self._sdl_exits),
            "Trailing Stop Exits":    str(self._trail_exits),
            "Rotation Exits":         str(self._rotation_exits),
            "Blocked Wash Sales":     str(self._blocked_wash),
            "Blocked Min Hold":       str(self._blocked_min_hold),
            "Blocked Sell Streak":    str(self._blocked_streak),
            "Sector Blocks":          str(self._sector_blocks),
            "Corr Guard Blocks":      str(self._corr_blocks),
            "Earnings Blocks":        str(self._earnings_blocks),
            "Transition Blocks":      str(self._transition_blocks),
            "Velocity Blocks":        str(self._velocity_blocks),
            "Total Tax":              f"${self._total_tax:,.2f}",
            "Short-Term Trades":      str(self._st_trades),
            "Long-Term Trades":       str(self._lt_trades),
            "Regime BULL_CALM":       pcts.get(BULL_CALM,     "0%"),
            "Regime BULL_VOL":        pcts.get(BULL_VOLATILE, "0%"),
            "Regime CHOPPY":          pcts.get(CHOPPY,        "0%"),
            "Regime BEAR":            pcts.get(BEAR,          "0%"),
        }
        for k, v in stats.items():
            self.SetRuntimeStatistic(k, v)
        self.Log(
            f"End | models={len(self._models)} buys={self._executed_buys} "
            f"sells={self._executed_sells} tax=${self._total_tax:,.0f} regimes={pcts}"
        )
