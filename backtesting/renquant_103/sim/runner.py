"""run_backtest — walk OOS bars, call kernel primitives, simulate the portfolio.

This is the notebook's simulation entry point. It mirrors the live
InferencePipeline's decision order (regime → mark-to-market → drawdown →
sells → market gates → BEAR branch → candidate scan → ranking → rotation →
selection) but drives a sim portfolio instead of a broker. Precomputed OOS
signals come in via the `results` dict (built by `training/`), so the loop
is O(N_bars) without re-running model inference per bar.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from kernel.exits        import HoldingState, compute_exits
from kernel.market_gates import check_spy_ema_trend, check_spy_velocity_crash
from kernel.models       import expected_return_from_calibration
from kernel.portfolio    import compute_trade_tax, update_drawdown_circuit_breaker
from kernel.regime       import RegimeState, detect_regime, load_gmm_artifact
from kernel.rotation     import find_rotation_pairs
from kernel.selection    import (
    CandidateResult,
    SelectionContext,
    compute_relative_strength,
    is_wash_sale_blocked,
    passes_correlation_guard,
    passes_sector_guard,
    run_selection_loop,
    score_candidates,
)
from kernel.sizing       import compute_position_size

log = logging.getLogger("sim.runner")


@dataclass
class SimResult:
    equity_df:    pd.DataFrame                  # date → portfolio, regime
    trade_log:    list[dict]                    # buy/sell records
    rotation_log: list[dict]                    # ROTATION_TREE/REJECT/EXEC events
    final_value:  float
    total_return: float
    apy:          float
    win_rate:     float
    avg_hold:     float
    avg_pnl:      float
    total_tax:    float
    exit_reasons: dict[str, int]
    rotations:    list[dict]                    # paired sell/buy summary

    @property
    def buys(self) -> list[dict]:
        return [t for t in self.trade_log if t["action"] == "buy"]

    @property
    def sells(self) -> list[dict]:
        return [t for t in self.trade_log if t["action"] == "sell"]

    def print_summary(self) -> None:
        print(f"Simulation complete: {len(self.equity_df)} days")
        print(f"Final value: ${self.final_value:,.0f}  |  "
              f"Return: {self.total_return:.1%}  |  APY: {self.apy:.1%}")
        print(f"Trades: {len(self.buys)} buys, {len(self.sells)} sells  |  "
              f"Win rate: {self.win_rate:.0%}")
        if self.sells:
            print(f"Avg hold: {self.avg_hold:.0f}d  |  "
                  f"Avg P&L/trade: {self.avg_pnl:.1%}  |  "
                  f"Total tax: ${self.total_tax:,.0f}")
            print(f"Exit reasons: {self.exit_reasons}")
        if self.rotations:
            print(f"\n── Rotations ({len(self.rotations)}) ──")
            for r in self.rotations:
                print(f"  [{r['date']}] {r['sell']:<5} → {r['buy']:<5}  "
                      f"sold_pnl={r['pnl_pct']:+.1%}  hold={r['hold_days']:>3}d  "
                      f"tax=${r['tax']:>7,.0f}")
        else:
            print("\nNo rotations triggered this run.")


def _load_artifacts(strategy_dir: Path, config: dict, fallback_corr: dict | None):
    artifacts_dir = strategy_dir / "artifacts"
    if not artifacts_dir.exists():
        artifacts_dir = strategy_dir
    regime_cfg = config.get("regime", {})

    earnings_path = artifacts_dir / regime_cfg.get("earnings_artifact", "earnings-calendar.json")
    earnings_cal  = json.loads(earnings_path.read_text()) if earnings_path.exists() else {}

    gmm = load_gmm_artifact(artifacts_dir / regime_cfg.get("gmm_artifact", "spy-gmm-regime.json"))

    corr_path = artifacts_dir / regime_cfg.get("correlation_artifact", "watchlist-correlation.json")
    if corr_path.exists():
        corr_dict = json.loads(corr_path.read_text())
    elif fallback_corr is not None:
        corr_dict = fallback_corr
    else:
        corr_dict = {}

    return gmm, earnings_cal, corr_dict


def run_backtest(
    *,
    config:           dict,
    strategy_dir:     Path,
    results:          dict,                     # {ticker: {oos_signals, oos_raw_scores, score_calibration, passes_floor}}
    ohlcv:            dict[str, pd.DataFrame],
    spy_df:           pd.DataFrame,
    sector_etf_map:   dict[str, str],
    fallback_corr:    dict | None = None,       # used if correlation artifact is missing
) -> SimResult:
    """Run the OOS portfolio simulation. Mirrors LEAN's decision order."""

    # ── Config shortcuts ────────────────────────────────────────────────────
    BACKTEST_START    = config["backtest_start"]
    BACKTEST_END      = config["backtest_end"]
    INITIAL_CASH      = config["initial_cash"]
    MAX_POSITIONS     = config["max_concurrent_positions"]
    CORR_THRESHOLD    = config["regime"]["correlation_guard_threshold"]
    WASH_SALE_DAYS    = config.get("wash_sale_days", 30)
    MIN_HOLD_DAYS     = config.get("min_hold_days", 30)
    CONSECUTIVE_SELLS = config.get("consecutive_sell_signals", 3)
    TIERED_THRESHOLDS = config.get("tiered_thresholds", [{"min_model_score": 0.10}])
    MAX_PER_SECTOR    = config.get("max_positions_per_sector", 3)
    EARNINGS_BUFFER   = config["regime"].get("earnings_buffer_days", 3)
    DEFENSIVE_TICKERS = set(config.get("defensive_tickers", []))
    BEAR_DEFENSIVE_PCT, BEAR_DEFENSIVE_SLOTS = 0.15, 1
    LT_HOLD_GATE_DAYS = config.get("lt_hold_gate_days", 330)
    LT_HOLD_MIN_GAIN  = config.get("lt_hold_min_gain", 0.10)
    ROTATION_CFG      = config.get("rotation", {})
    ROTATION_HORIZON  = int(ROTATION_CFG.get("target_horizon_days", 20))
    ROTATION_THRESH   = float(ROTATION_CFG.get("min_expected_advantage_pct", 0.03))
    ROTATION_COST     = float(ROTATION_CFG.get("transaction_cost_pct", 0.0))

    ST_RATE   = config["tax"]["short_term_rate"]
    LT_RATE   = config["tax"]["long_term_rate"]
    LT_THRESH = config["tax"]["long_term_threshold_days"]

    RP = config["regime_params"]
    bc = RP["BULL_CALM"]
    SPY_VEL_HALT_PCT      = bc.get("spy_velocity_halt_pct", 0.03)
    SPY_VEL_LOOKBACK_DAYS = int(bc.get("spy_velocity_lookback_days", 3))

    BW = config.get("ranking", {}).get("blend_weights", [0.5, 0.5])
    _bt = float(BW[0]) + float(BW[1]) or 1.0
    RANK_W, RS_W = float(BW[0]) / _bt, float(BW[1]) / _bt

    # ── Artifacts ───────────────────────────────────────────────────────────
    gmm, earnings_cal, corr_dict = _load_artifacts(strategy_dir, config, fallback_corr)
    print(f"GMM artifact loaded: {gmm is not None}")
    print(f"Earnings calendar entries: {sum(len(v) for v in earnings_cal.values())}")
    print(f"Correlation matrix entries: {len(corr_dict)}")
    print(f"Rotation enabled: {ROTATION_CFG.get('enabled', False)}  "
          f"horizon={ROTATION_HORIZON}d  threshold={ROTATION_THRESH:+.3f}  "
          f"cost={ROTATION_COST:.4f}")

    # ── Per-ticker score lookups (precomputed in training) ─────────────────
    def _is_earnings_blocked(ticker, today_ts):
        for d in earnings_cal.get(ticker, []):
            try:
                if abs((pd.Timestamp(d).date() - today_ts.date()).days) <= EARNINGS_BUFFER:
                    return True
            except Exception:
                pass
        return False

    def _rank_score(ticker, today_ts):
        raw = results[ticker].get("oos_raw_scores")
        if raw is None or today_ts not in raw.index:
            return None
        cal = results[ticker].get("score_calibration")
        return float(cal.calibrate(float(raw.loc[today_ts]))) if cal else float(raw.loc[today_ts])

    def _expected_return(ticker, today_ts):
        raw = results[ticker].get("oos_raw_scores")
        if raw is None or today_ts not in raw.index:
            return None
        cal = results[ticker].get("score_calibration")
        if cal is None:
            return None
        return float(expected_return_from_calibration(
            float(raw.loc[today_ts]), cal.to_dict(), horizon_days=ROTATION_HORIZON,
        ))

    def _action(ticker, today_ts):
        sigs = results[ticker].get("oos_signals")
        if sigs is None or today_ts not in sigs.index:
            return "hold"
        v = sigs.loc[today_ts]
        return "buy" if v == 1 else ("sell" if v == -1 else "hold")

    # ── State ───────────────────────────────────────────────────────────────
    cash:              float = float(INITIAL_CASH)
    hwm:               float = float(INITIAL_CASH)
    skip_buys:         bool  = False
    holdings:          dict[str, HoldingState]   = {}
    pos_shares:        dict[str, float]          = {}
    last_sell_date:    dict[str, pd.Timestamp]   = {}
    equity_curve:      list[dict]                = []
    trade_log:         list[dict]                = []
    rotation_log:      list[dict]                = []
    exportable                                   = {t for t, r in results.items() if r.get("passes_floor")}
    regime_state                                 = RegimeState()

    bt_dates      = spy_df.loc[BACKTEST_START:BACKTEST_END].index
    spy_daily_ret = spy_df["close"].pct_change().fillna(0)

    # ── Main loop ───────────────────────────────────────────────────────────
    for today in bt_dates:
        # Streaming regime detection (Hurst + CUSUM + GMM)
        spy_rets_arr  = spy_daily_ret.loc[:today].values.astype(float)
        spy_df_window = spy_df.loc[:today]
        regime_state  = detect_regime(spy_rets_arr, spy_df_window, gmm,
                                      regime_state, config)
        regime        = regime_state.regime
        regime_conf   = regime_state.confidence
        in_transition = regime_state.in_transition

        rp = RP.get(regime, RP["BULL_CALM"])

        # Mark-to-market
        port_val = cash + sum(
            pos_shares[t] * float(ohlcv[t].loc[today, "close"])
            for t in holdings if t in ohlcv and today in ohlcv[t].index
        )

        # Drawdown circuit breaker
        hwm, skip_buys = update_drawdown_circuit_breaker(
            port_val, hwm, float(rp["drawdown_halt_pct"]))
        equity_curve.append({"date": today, "portfolio": port_val, "regime": regime})

        # ── SELLS ────────────────────────────────────────────────────────────
        exit_params = {
            "trailing_stop_trigger_pct": rp.get("trailing_stop_trigger_pct", 0),
            "trailing_stop_trail_pct":   rp.get("trailing_stop_trail_pct", 0),
            "stop_loss_pct":             rp["stop_loss_pct"],
            "max_single_day_loss_pct":   rp.get("max_single_day_loss_pct", 0),
            "max_hold_days":             rp["max_hold_days"],
            "consecutive_sell_signals":  CONSECUTIVE_SELLS,
            "min_hold_days":             MIN_HOLD_DAYS,
            "lt_hold_gate_days":         LT_HOLD_GATE_DAYS,
            "lt_hold_min_gain":          LT_HOLD_MIN_GAIN,
        }
        to_sell = []
        for t, st in list(holdings.items()):
            if today not in ohlcv.get(t, pd.DataFrame()).index:
                continue
            price = float(ohlcv[t].loc[today, "close"])
            sig, updated = compute_exits(price, today.date(), _action(t, today), st, exit_params)
            updated.prev_close      = price
            updated.rank_score      = _rank_score(t, today)
            updated.expected_return = _expected_return(t, today)
            holdings[t] = updated
            if sig.should_exit:
                to_sell.append((t, price, sig.exit_type))

        for t, price, reason in to_sell:
            st        = holdings.pop(t)
            shares    = pos_shares.pop(t)
            hold_days = (today.date() - st.entry_date).days
            gross_pnl = shares * (price - st.entry_price)
            tax       = compute_trade_tax(gross_pnl, hold_days, ST_RATE, LT_RATE, LT_THRESH)
            cash     += shares * price - tax
            last_sell_date[t] = today
            trade_log.append({"action": "sell", "ticker": t, "date": today,
                              "pnl_pct": (price - st.entry_price) / st.entry_price,
                              "hold_days": hold_days, "tax": tax, "exit_reason": reason})

        # Transition uncertainty window — no new buys
        if in_transition:
            continue

        # ── BEAR branch: defensives only ────────────────────────────────────
        if regime == "BEAR":
            defensive_held = sum(1 for t in holdings if t in DEFENSIVE_TICKERS)
            if not skip_buys and defensive_held < BEAR_DEFENSIVE_SLOTS:
                def_sorted = sorted(
                    (t for t in exportable & DEFENSIVE_TICKERS
                     if t not in holdings
                     and today in ohlcv.get(t, pd.DataFrame()).index
                     and _action(t, today) == "buy"
                     and (not last_sell_date.get(t) or
                          (today - last_sell_date[t]).days >= WASH_SALE_DAYS)
                     and _rank_score(t, today) is not None),
                    key=lambda x: _rank_score(x, today), reverse=True,
                )
                for t in def_sorted:
                    price = float(ohlcv[t].loc[today, "close"])
                    _, shares = compute_position_size(
                        port_val, cash, 0, 0, price, override_pct=BEAR_DEFENSIVE_PCT)
                    if shares < 1:
                        continue
                    invest = shares * price
                    cash  -= invest
                    holdings[t]   = HoldingState(entry_price=price, entry_date=today.date(),
                                                 high_watermark=price, prev_close=price)
                    pos_shares[t] = shares
                    trade_log.append({"action": "buy", "ticker": t, "date": today,
                                      "price": price, "shares": shares, "invest": invest,
                                      "regime": "BEAR_defensive"})
                    break
            continue

        # ── Market gates ────────────────────────────────────────────────────
        spy_rets_window = list(spy_daily_ret.loc[:today].values)
        spy_close_hist  = spy_df["close"].loc[:today]
        if (skip_buys
                or len(holdings) >= MAX_POSITIONS
                or check_spy_velocity_crash(spy_rets_window, SPY_VEL_LOOKBACK_DAYS, SPY_VEL_HALT_PCT)
                or check_spy_ema_trend(spy_close_hist, ema_span=50)):
            continue

        # ── Candidate scan ──────────────────────────────────────────────────
        open_slots   = MAX_POSITIONS - len(holdings)
        max_pos_pct  = float(rp.get("max_position_pct", 0.15)) * regime_conf
        cash_res_pct = float(rp.get("cash_reserve_pct", 0.0)) * regime_conf
        min_score    = float(rp.get("min_model_score", 0.10))

        candidates = []
        for t in exportable:
            if (t in holdings
                    or today not in ohlcv.get(t, pd.DataFrame()).index
                    or _action(t, today) != "buy"):
                continue
            last_s = last_sell_date.get(t)
            if last_s is not None and (today - last_s).days < WASH_SALE_DAYS:
                continue
            if _is_earnings_blocked(t, today):
                continue
            rs = _rank_score(t, today)
            if rs is None or rs < min_score:
                continue
            sector = config["sector_map"].get(t, "other")
            etf    = sector_etf_map.get(sector)
            rs_sc  = 0.0
            if etf and etf in ohlcv and today in ohlcv[etf].index:
                try:
                    rs_sc = compute_relative_strength(
                        float(ohlcv[t]["close"].pct_change(20).loc[today]),
                        float(ohlcv[etf]["close"].pct_change(20).loc[today]),
                    )
                except Exception:
                    pass
            er = _expected_return(t, today) or 0.0
            candidates.append(CandidateResult(
                ticker=t, raw_score=rs, rank_score=rs, rs_score=rs_sc,
                expected_return=er,
            ))

        if not candidates:
            continue

        # Rank
        ranked = score_candidates(candidates, RANK_W, RS_W)

        # ── ROTATION: expected-return swap rule ─────────────────────────────
        if ROTATION_CFG.get("enabled", False) and holdings:
            held_scores: dict = {}
            held_er:     dict = {}
            held_meta:   dict = {}
            for t, st in holdings.items():
                s  = getattr(st, "rank_score", None)
                er = getattr(st, "expected_return", None)
                if (s is None or er is None
                        or t not in ohlcv or today not in ohlcv[t].index):
                    continue
                held_scores[t] = float(s)
                held_er[t]     = float(er)
                held_meta[t]   = {
                    "entry_date":    st.entry_date,
                    "entry_price":   st.entry_price,
                    "current_price": float(ohlcv[t].loc[today, "close"]),
                }

            eligible_cands = [c for c in ranked if c.ticker not in holdings]
            pairs = find_rotation_pairs(
                held_scores  = held_scores,
                held_er      = held_er,
                held_meta    = held_meta,
                candidates   = eligible_cands,
                today        = today.date(),
                rotation_cfg = ROTATION_CFG,
                tax_cfg      = config.get("tax", {}),
            )

            chosen_pairs = {p.buy_ticker: p for p in pairs}
            for c in eligible_cands[:5]:
                cand_er = float(getattr(c, "expected_return", 0.0) or 0.0)
                chosen  = chosen_pairs.get(c.ticker)
                rotation_log.append({
                    "kind": "TREE", "date": today.date(), "cand": c.ticker,
                    "cand_er": cand_er, "cand_rank": float(c.rank_score),
                    "chosen": chosen.sell_ticker if chosen else None,
                })

            validated = []
            last_sell_dates_d = {t: d.date() for t, d in last_sell_date.items()}
            for pair in pairs:
                if is_wash_sale_blocked(pair.buy_ticker, today.date(),
                                        last_sell_dates_d, WASH_SALE_DAYS):
                    rotation_log.append({"kind": "REJECT", "date": today.date(),
                                         "swap": (pair.sell_ticker, pair.buy_ticker),
                                         "reason": "wash_sale"})
                    continue
                virtual_held = (
                    set(holdings.keys())
                    - {p.sell_ticker for p in validated} - {pair.sell_ticker}
                    | {p.buy_ticker for p in validated}
                )
                if not passes_sector_guard(pair.buy_ticker, list(virtual_held),
                                           config["sector_map"], MAX_PER_SECTOR,
                                           DEFENSIVE_TICKERS):
                    rotation_log.append({"kind": "REJECT", "date": today.date(),
                                         "swap": (pair.sell_ticker, pair.buy_ticker),
                                         "reason": "sector_cap"})
                    continue
                if not passes_correlation_guard(pair.buy_ticker, list(virtual_held),
                                                corr_dict, CORR_THRESHOLD):
                    rotation_log.append({"kind": "REJECT", "date": today.date(),
                                         "swap": (pair.sell_ticker, pair.buy_ticker),
                                         "reason": "correlation_guard"})
                    continue
                validated.append(pair)

            rotated_buys: set = set()
            for pair in validated:
                if pair.sell_ticker not in holdings:
                    continue
                sell_price = float(ohlcv[pair.sell_ticker].loc[today, "close"])
                st_p       = holdings.pop(pair.sell_ticker)
                shares     = pos_shares.pop(pair.sell_ticker)
                hold_days  = (today.date() - st_p.entry_date).days
                gross_pnl  = shares * (sell_price - st_p.entry_price)
                tax        = compute_trade_tax(gross_pnl, hold_days, ST_RATE, LT_RATE, LT_THRESH)
                cash      += shares * sell_price - tax
                last_sell_date[pair.sell_ticker] = today
                trade_log.append({"action": "sell", "ticker": pair.sell_ticker, "date": today,
                                  "pnl_pct": (sell_price - st_p.entry_price) / st_p.entry_price,
                                  "hold_days": hold_days, "tax": tax, "exit_reason": "rotation"})

                if pair.buy_ticker not in ohlcv or today not in ohlcv[pair.buy_ticker].index:
                    continue
                buy_price   = float(ohlcv[pair.buy_ticker].loc[today, "close"])
                _, b_shares = compute_position_size(port_val, cash,
                                                    max_pos_pct, cash_res_pct, buy_price)
                if b_shares < 1:
                    continue
                invest = b_shares * buy_price
                cash  -= invest
                holdings[pair.buy_ticker]   = HoldingState(entry_price=buy_price,
                                                           entry_date=today.date(),
                                                           high_watermark=buy_price,
                                                           prev_close=buy_price)
                pos_shares[pair.buy_ticker] = b_shares
                trade_log.append({"action": "buy", "ticker": pair.buy_ticker, "date": today,
                                  "price": buy_price, "shares": b_shares, "invest": invest,
                                  "regime": "rotation"})
                rotated_buys.add(pair.buy_ticker)
                rotation_log.append({
                    "kind": "EXEC", "date": today.date(),
                    "swap": (pair.sell_ticker, pair.buy_ticker), "shares": b_shares,
                    "net_adv": pair.net_advantage, "raw_adv": pair.raw_advantage,
                    "tax": pair.tax_drag, "cost": pair.transaction_cost,
                    "threshold": pair.threshold,
                })

            if rotated_buys:
                ranked = [c for c in ranked if c.ticker not in rotated_buys]
            open_slots = MAX_POSITIONS - len(holdings)

        # ── Selection loop ──────────────────────────────────────────────────
        if open_slots <= 0 or not ranked:
            continue
        sel_ctx = SelectionContext(
            today=today.date(),
            held_tickers=list(holdings.keys()),
            last_sell_dates={t: d.date() for t, d in last_sell_date.items()},
            earnings_calendar=earnings_cal,
            corr_matrix=corr_dict,
            sector_map=config["sector_map"],
            defensive_set=DEFENSIVE_TICKERS,
            wash_sale_days=WASH_SALE_DAYS,
            earnings_buffer=EARNINGS_BUFFER,
            corr_threshold=CORR_THRESHOLD,
            max_per_sector=MAX_PER_SECTOR,
            tiered_thresholds=TIERED_THRESHOLDS,
            open_slots=open_slots,
        )
        selected, _ = run_selection_loop(ranked, sel_ctx)

        for t in selected:
            price = float(ohlcv[t].loc[today, "close"])
            _, shares = compute_position_size(port_val, cash, max_pos_pct, cash_res_pct, price)
            if shares < 1:
                continue
            invest = shares * price
            cash  -= invest
            holdings[t]   = HoldingState(entry_price=price, entry_date=today.date(),
                                         high_watermark=price, prev_close=price)
            pos_shares[t] = shares
            trade_log.append({"action": "buy", "ticker": t, "date": today,
                              "price": price, "shares": shares, "invest": invest})

    # ── Summary ─────────────────────────────────────────────────────────────
    equity_df = pd.DataFrame(equity_curve).set_index("date")
    final_val = equity_df["portfolio"].iloc[-1]
    total_ret = final_val / INITIAL_CASH - 1
    n_years   = len(equity_df) / 252
    apy       = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    sells = [t for t in trade_log if t["action"] == "sell"]
    buys  = [t for t in trade_log if t["action"] == "buy"]
    wins  = [t for t in sells if t["pnl_pct"] > 0]
    win_rate  = len(wins) / max(1, len(sells))
    avg_hold  = sum(t["hold_days"] for t in sells) / len(sells) if sells else 0.0
    avg_pnl   = sum(t["pnl_pct"]   for t in sells) / len(sells) if sells else 0.0
    total_tax = sum(t["tax"]       for t in sells)
    exit_reasons = dict(Counter(t.get("exit_reason", "?") for t in sells))

    rotation_sells = [t for t in sells if t.get("exit_reason") == "rotation"]
    rotations: list[dict] = []
    for s in rotation_sells:
        d = s["date"].date() if hasattr(s["date"], "date") else s["date"]
        same_day_buys = [b for b in buys
                         if (b["date"].date() if hasattr(b["date"], "date") else b["date"]) == d
                         and b.get("regime") == "rotation"]
        rotations.append({
            "date": d, "sell": s["ticker"],
            "buy": same_day_buys[0]["ticker"] if same_day_buys else "?",
            "pnl_pct": s["pnl_pct"], "hold_days": s["hold_days"], "tax": s["tax"],
        })

    return SimResult(
        equity_df=equity_df, trade_log=trade_log, rotation_log=rotation_log,
        final_value=final_val, total_return=total_ret, apy=apy,
        win_rate=win_rate, avg_hold=avg_hold, avg_pnl=avg_pnl, total_tax=total_tax,
        exit_reasons=exit_reasons, rotations=rotations,
    )
