"""Replay-style parity checks for renquant_103 selection semantics.

This file adds a short synthetic tape that verifies notebook-like and LEAN-like
selection ledgers stay aligned on the currently-canonical semantics:

- raw score drives buy/sell action eligibility upstream
- calibrated rank_score drives filtering, ranking, and tier thresholds
- blend weights come from config
- max_position_pct and cash_reserve_pct scale with regime confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta


def _norm(value: float, low: float, high: float) -> float:
    return (value - low) / (high - low) if high > low else 0.5


@dataclass
class Candidate:
    ticker: str
    raw_score: float
    rank_score: float
    rs_score: float
    sector: str
    action: str = "buy"


@dataclass
class DayScenario:
    day: date
    regime_confidence: float
    regime_params: dict
    candidates: list[Candidate]


def _notebook_replay(scenarios: list[DayScenario], config: dict, correlations: dict, last_sell_dates: dict) -> list[dict]:
    held: list[str] = []
    cash = 10_000.0
    portfolio_value = 10_000.0
    ledger: list[dict] = []

    for scenario in scenarios:
        rp = scenario.regime_params
        open_slots = config["max_positions"] - len(held)
        scan_pass = []
        for candidate in scenario.candidates:
            if candidate.ticker in held or candidate.action != "buy":
                continue
            last_sell = last_sell_dates.get(candidate.ticker)
            if last_sell and (scenario.day - last_sell).days < config["wash_sale_days"]:
                continue
            if candidate.rank_score < rp.get("min_model_score", 0.0):
                continue
            scan_pass.append(candidate)

        if len(scan_pass) > 1:
            rank_scores = [candidate.rank_score for candidate in scan_pass]
            rs_scores = [candidate.rs_score for candidate in scan_pass]
            scan_pass = sorted(
                scan_pass,
                key=lambda candidate: config["weights"][0] * _norm(candidate.rank_score, min(rank_scores), max(rank_scores))
                + config["weights"][1] * _norm(candidate.rs_score, min(rs_scores), max(rs_scores)),
                reverse=True,
            )

        selected = []
        investments = {}
        for candidate in scan_pass:
            if len(selected) >= open_slots:
                break
            tier_idx = min(len(selected), len(config["tiers"]) - 1)
            if candidate.rank_score < config["tiers"][tier_idx]["min_model_score"]:
                continue
            last_sell = last_sell_dates.get(candidate.ticker)
            if last_sell and (scenario.day - last_sell).days < config["wash_sale_days"]:
                continue
            if candidate.sector not in config["defensive_sectors"]:
                same_sector = [ticker for ticker in held + selected if config["sector_map"][ticker] == candidate.sector]
                if len(same_sector) >= config["max_positions_per_sector"]:
                    continue
            blocked = False
            for held_ticker in held + selected:
                corr = correlations.get(candidate.ticker, {}).get(held_ticker, correlations.get(held_ticker, {}).get(candidate.ticker, 0.0))
                if abs(corr) >= config["correlation_threshold"]:
                    blocked = True
                    break
            if blocked:
                continue
            selected.append(candidate.ticker)
            cash_reserve = portfolio_value * rp.get("cash_reserve_pct", 0.0) * scenario.regime_confidence
            max_position_pct = rp.get("max_position_pct", 0.0) * scenario.regime_confidence
            investments[candidate.ticker] = min(cash - cash_reserve, portfolio_value * max_position_pct)
            cash -= investments[candidate.ticker]

        held.extend(selected)
        ledger.append(
            {
                "day": scenario.day,
                "scan_pass": [candidate.ticker for candidate in scan_pass],
                "ranked": [candidate.ticker for candidate in scan_pass],
                "selected": selected,
                "investments": investments,
            }
        )

    return ledger


def _lean_replay(scenarios: list[DayScenario], config: dict, correlations: dict, last_sell_dates: dict) -> list[dict]:
    held_tickers: list[str] = []
    available_cash = 10_000.0
    portfolio_value = 10_000.0
    ledger: list[dict] = []

    for scenario in scenarios:
        rp = scenario.regime_params
        open_slots = config["max_positions"] - len(held_tickers)
        scored = []
        for candidate in scenario.candidates:
            if candidate.ticker in held_tickers or candidate.action != "buy":
                continue
            last_sell = last_sell_dates.get(candidate.ticker)
            if last_sell and (scenario.day - last_sell).days < config["wash_sale_days"]:
                continue
            if candidate.rank_score < rp.get("min_model_score", 0.0):
                continue
            scored.append(candidate)

        if len(scored) > 1:
            rank_scores = [candidate.rank_score for candidate in scored]
            rs_scores = [candidate.rs_score for candidate in scored]
            scored = sorted(
                scored,
                key=lambda candidate: config["weights"][0] * _norm(candidate.rank_score, min(rank_scores), max(rank_scores))
                + config["weights"][1] * _norm(candidate.rs_score, min(rs_scores), max(rs_scores)),
                reverse=True,
            )

        selected = []
        investments = {}
        for candidate in scored:
            if len(selected) >= open_slots:
                break
            tier_idx = min(len(selected), len(config["tiers"]) - 1)
            if candidate.rank_score < config["tiers"][tier_idx]["min_model_score"]:
                continue
            last_sell = last_sell_dates.get(candidate.ticker)
            if last_sell and (scenario.day - last_sell).days < config["wash_sale_days"]:
                continue
            if candidate.sector not in config["defensive_sectors"]:
                same_sector = [ticker for ticker in held_tickers + selected if config["sector_map"][ticker] == candidate.sector]
                if len(same_sector) >= config["max_positions_per_sector"]:
                    continue
            correlated = False
            for held_ticker in held_tickers + selected:
                corr = correlations.get(candidate.ticker, {}).get(held_ticker, correlations.get(held_ticker, {}).get(candidate.ticker, 0.0))
                if abs(corr) >= config["correlation_threshold"]:
                    correlated = True
                    break
            if correlated:
                continue
            selected.append(candidate.ticker)
            cash_reserve = portfolio_value * rp.get("cash_reserve_pct", 0.0) * scenario.regime_confidence
            scaled_max_pct = rp.get("max_position_pct", 0.0) * scenario.regime_confidence
            investable = max(available_cash - cash_reserve, 0.0)
            target_pct = min(scaled_max_pct, investable / max(portfolio_value, 1.0))
            investments[candidate.ticker] = target_pct * portfolio_value
            available_cash -= investments[candidate.ticker]

        held_tickers.extend(selected)
        ledger.append(
            {
                "day": scenario.day,
                "scan_pass": [candidate.ticker for candidate in scored],
                "ranked": [candidate.ticker for candidate in scored],
                "selected": selected,
                "investments": investments,
            }
        )

    return ledger


def test_selection_ledger_replay_matches_between_notebook_and_lean():
    config = {
        "wash_sale_days": 30,
        "max_positions": 3,
        "max_positions_per_sector": 1,
        "correlation_threshold": 0.70,
        "weights": (0.8, 0.2),
        "tiers": [
            {"min_model_score": 0.10},
            {"min_model_score": 0.80},
            {"min_model_score": 0.50},
        ],
        "defensive_sectors": {"commodity", "bond", "healthcare", "utility"},
        "sector_map": {
            "AAPL": "tech",
            "JPM": "finance",
            "XOM": "energy",
            "MSFT": "tech",
            "NVDA": "tech",
            "CVX": "energy",
        },
    }
    last_sell_dates = {
        "MSFT": date(2026, 4, 11),
    }
    scenarios = [
        DayScenario(
            day=date(2026, 4, 16),
            regime_confidence=0.5,
            regime_params={"min_model_score": 0.15, "max_position_pct": 0.20, "cash_reserve_pct": 0.20},
            candidates=[
                Candidate("AAPL", 1.4, 0.62, 0.10, "tech"),
                Candidate("JPM", 0.9, 0.55, 0.40, "finance"),
                Candidate("XOM", 0.5, 0.48, 0.90, "energy"),
                Candidate("MSFT", 1.1, 0.95, 0.30, "tech"),
            ],
        ),
        DayScenario(
            day=date(2026, 5, 20),
            regime_confidence=1.0,
            regime_params={"min_model_score": 0.10, "max_position_pct": 0.15, "cash_reserve_pct": 0.00},
            candidates=[
                Candidate("JPM", 0.8, 0.67, 0.20, "finance"),
                Candidate("XOM", 0.7, 0.61, 0.30, "energy"),
                Candidate("NVDA", 1.0, 0.70, 0.80, "tech"),
            ],
        ),
        DayScenario(
            day=date(2026, 5, 21),
            regime_confidence=0.8,
            regime_params={"min_model_score": 0.10, "max_position_pct": 0.15, "cash_reserve_pct": 0.00},
            candidates=[
                Candidate("XOM", 0.7, 0.52, 0.50, "energy"),
                Candidate("CVX", 0.9, 0.60, 0.40, "energy"),
                Candidate("MSFT", 1.0, 0.58, 0.10, "tech"),
            ],
        ),
    ]
    correlations = {
        "CVX": {"JPM": 0.75, "AAPL": 0.10},
        "XOM": {"JPM": 0.20, "AAPL": 0.10},
        "MSFT": {"AAPL": 0.20, "JPM": 0.10},
        "NVDA": {"AAPL": 0.20},
        "JPM": {"AAPL": 0.10},
    }

    notebook_ledger = _notebook_replay(scenarios, config, correlations, last_sell_dates)
    lean_ledger = _lean_replay(scenarios, config, correlations, last_sell_dates)

    assert notebook_ledger == lean_ledger
    assert notebook_ledger == [
        {
            "day": date(2026, 4, 16),
            "scan_pass": ["AAPL", "JPM", "XOM"],
            "ranked": ["AAPL", "JPM", "XOM"],
            "selected": ["AAPL"],
            "investments": {"AAPL": 1000.0},
        },
        {
            "day": date(2026, 5, 20),
            "scan_pass": ["NVDA", "JPM", "XOM"],
            "ranked": ["NVDA", "JPM", "XOM"],
            "selected": ["JPM"],
            "investments": {"JPM": 1500.0},
        },
        {
            "day": date(2026, 5, 21),
            "scan_pass": ["CVX", "MSFT", "XOM"],
            "ranked": ["CVX", "MSFT", "XOM"],
            "selected": ["XOM"],
            "investments": {"XOM": 1200.0},
        },
    ]