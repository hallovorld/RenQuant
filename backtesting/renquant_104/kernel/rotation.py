"""Rotation primitives — when to swap a held position for a better candidate.

Self-contained: stdlib only.  Shared by RotationJob (LEAN + live runner) and
the notebook simulation cell.

Decision rule (expected-return units, all in fraction of position value):

    raw_advantage = E[R_buy_horizon] - E[R_sell_horizon]
    cost          = transaction_cost_pct
    tax_drag      = unrealized_pnl_pct * tax_rate(hold_days)
    net_advantage = raw_advantage - cost - tax_drag
    swap when     net_advantage >= min_expected_advantage_pct

Where E[R_horizon] is `ScoreCalibration.expected_return(raw_score, horizon)` —
i.e. expected stock-minus-SPY return over `target_horizon_days` trading days.

LT protection: if the held position sits on a gain and is within
`lt_protection_days` of the long-term threshold, we pin its required margin to
+inf — a forced swap would burn the upcoming LT tax discount.

The kernel returns rich `RotationPair` records carrying every component above
so that `task_rotation.py` can emit a structured decision-tree log.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass


@dataclass
class RotationPair:
    """One swap intent emitted by find_rotation_pairs.  All values fraction units."""
    sell_ticker:     str
    buy_ticker:      str
    sell_score:      float    # rank_score (probability) — kept for log readability
    buy_score:       float    # rank_score (probability)
    sell_er:         float    # E[R - SPY] over horizon for held position
    buy_er:          float    # E[R - SPY] over horizon for candidate
    horizon_days:    int
    raw_advantage:   float    # buy_er - sell_er
    tax_drag:        float    # tax cost on realised gain (held side)
    transaction_cost: float
    net_advantage:   float    # raw_advantage - tax_drag - transaction_cost
    threshold:       float    # min_expected_advantage_pct
    margin_realized: float    # net_advantage - threshold (>=0 when emitted)


# ── Tax helpers ────────────────────────────────────────────────────────────────

def tax_drag(
    unrealized_pnl_pct: float,
    hold_days: int,
    short_term_rate: float,
    long_term_rate: float,
    long_term_threshold_days: int = 365,
) -> float:
    """Cost of realizing a gain, as fraction of position value.

    A 20% gain held short-term at 50% rate → 0.20 * 0.50 = 0.10 of position
    paid in tax.  Losses give zero drag (loss harvesting helps the swap).
    """
    if unrealized_pnl_pct <= 0:
        return 0.0
    rate = long_term_rate if hold_days >= long_term_threshold_days else short_term_rate
    return unrealized_pnl_pct * rate


def is_lt_protected(
    unrealized_pnl_pct: float,
    hold_days: int,
    lt_threshold_days: int,
    lt_protection_days: int,
) -> bool:
    """True iff the position would lose an upcoming LT tax discount on swap."""
    return (
        unrealized_pnl_pct > 0
        and 0 < (lt_threshold_days - hold_days) <= lt_protection_days
    )


# ── Pair selection ─────────────────────────────────────────────────────────────

def find_thesis_primary_pairs(
    held_entry_scores: dict[str, "float | None"],   # ticker → entry-time rank_score (BASELINE)
    held_today_scores: dict[str, "float | None"],   # ticker → today rank_score
    held_meta:         dict[str, dict],             # ticker → {entry_date, entry_price, current_price}
    candidates:        list,                         # CandidateResult-like
    today:             datetime.date,
    rotation_cfg:      dict,
    tax_cfg:           dict,
) -> list[RotationPair]:
    """Route B — thesis-degradation as the PRIMARY rotation gate.

    Use this when `rotation.mode == "thesis_primary"`. Bypasses the
    ER-based pair discovery (which requires `min_expected_advantage_pct`
    to clear — impossible when realistic ER deltas are smaller than the
    threshold). Instead emits a pair when:

      * held's thesis has DEGRADED (entry - today >= degradation_pct)
      * candidate beats held's entry baseline (cand.rank - entry >= uplift_pct)

    Same guardrails as ER mode: min_hold_days, lt_protection_days,
    max_rotations_per_bar, wash-sale + sector + correlation handled
    downstream in ValidatePairsTask.

    Still produces RotationPair records with ER/tax_drag/net_advantage
    fields populated (for log compatibility) even though they don't
    drive the decision.
    """
    if not rotation_cfg.get("enabled", False):
        return []

    thesis_cfg      = rotation_cfg.get("thesis", {})
    degradation_pct = float(thesis_cfg.get("degradation_pct", 0.30))
    uplift_pct      = float(thesis_cfg.get("uplift_pct", 0.10))
    horizon         = int(rotation_cfg.get("target_horizon_days", 20))
    txn_cost        = float(rotation_cfg.get("transaction_cost_pct", 0.0))
    min_hold        = int(rotation_cfg.get("min_rotation_hold_days", 30))
    lt_protect      = int(rotation_cfg.get("lt_protection_days", 30))
    max_per_bar     = int(rotation_cfg.get("max_rotations_per_bar", 2))

    st_rate         = float(tax_cfg.get("short_term_rate", 0.37))
    lt_rate         = float(tax_cfg.get("long_term_rate", 0.20))
    lt_threshold    = int(tax_cfg.get("long_term_threshold_days", 365))

    eligible: dict[str, dict] = {}
    for ticker, entry_score in held_entry_scores.items():
        if entry_score is None or entry_score <= 0:
            continue
        today_score = held_today_scores.get(ticker)
        if today_score is None:
            continue
        meta = held_meta.get(ticker)
        if meta is None:
            continue
        entry_date  = meta.get("entry_date")
        entry_price = float(meta.get("entry_price", 0.0))
        cur_price   = float(meta.get("current_price", 0.0))
        if entry_date is None or entry_price <= 0:
            continue
        hold_days = (today - entry_date).days
        if hold_days < min_hold:
            continue
        unreal_pct = (cur_price - entry_price) / entry_price
        if is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
            continue
        degradation = (entry_score - today_score) / entry_score
        if degradation < degradation_pct:
            continue   # held thesis still intact
        eligible[ticker] = {
            "entry_score": float(entry_score),
            "today_score": float(today_score),
            "degradation": degradation,
            "unreal_pct":  unreal_pct,
            "tax_drag":    tax_drag(unreal_pct, hold_days,
                                    st_rate, lt_rate, lt_threshold),
        }

    if not eligible or not candidates:
        return []

    used_holds: set[str] = set()
    pairs: list[RotationPair] = []

    for c in candidates:
        if len(pairs) >= max_per_bar:
            break
        cand_ticker = c.ticker
        if cand_ticker in held_entry_scores:
            continue
        cand_score = float(c.rank_score)

        # Find the most-degraded held whose entry baseline cand also beats
        best_match: str | None = None
        best_deg: float = -math.inf
        for held_ticker, info in eligible.items():
            if held_ticker in used_holds:
                continue
            uplift = cand_score - info["entry_score"]
            if uplift < uplift_pct:
                continue
            if info["degradation"] > best_deg:
                best_match = held_ticker
                best_deg   = info["degradation"]

        if best_match is None:
            continue

        info = eligible[best_match]
        pairs.append(RotationPair(
            sell_ticker      = best_match,
            buy_ticker       = cand_ticker,
            sell_score       = info["today_score"],
            buy_score        = cand_score,
            sell_er          = 0.0,   # N/A in thesis mode
            buy_er           = 0.0,
            horizon_days     = horizon,
            raw_advantage    = cand_score - info["entry_score"],
            tax_drag         = info["tax_drag"],
            transaction_cost = txn_cost,
            net_advantage    = cand_score - info["entry_score"] - info["tax_drag"] - txn_cost,
            threshold        = uplift_pct,
            margin_realized  = (cand_score - info["entry_score"]) - uplift_pct,
        ))
        used_holds.add(best_match)

    pairs.sort(key=lambda p: p.margin_realized, reverse=True)
    return pairs


def find_rotation_pairs(
    held_scores:    dict[str, float],          # ticker → rank_score (prob)
    held_er:        dict[str, float],          # ticker → E[R - SPY] over horizon
    held_meta:      dict[str, dict],           # ticker → {entry_date, entry_price, current_price}
    candidates:     list,                      # CandidateResult-like (.ticker, .rank_score, .expected_return)
    today:          datetime.date,
    rotation_cfg:   dict,
    tax_cfg:        dict,
) -> list[RotationPair]:
    """Greedy pairing using expected-return decision rule.

    Walks ranked candidates; for each, picks the held with the lowest
    expected-return whose net_advantage clears `min_expected_advantage_pct`.
    Each ticker (held or candidate) appears in at most one pair.
    """
    if not rotation_cfg.get("enabled", False):
        return []

    threshold       = float(rotation_cfg.get("min_expected_advantage_pct", 0.03))
    horizon         = int(rotation_cfg.get("target_horizon_days", 20))
    txn_cost        = float(rotation_cfg.get("transaction_cost_pct", 0.0))
    min_hold        = int(rotation_cfg.get("min_rotation_hold_days", 30))
    lt_protect      = int(rotation_cfg.get("lt_protection_days", 30))
    max_per_bar     = int(rotation_cfg.get("max_rotations_per_bar", 2))
    # Rotation V1 (2026-04-24): two additional depth / persistence gates.
    # User hypothesis: current rotations lose money because the net-adv
    # threshold alone can clear on marginal signal-vs-noise edges. Gate
    # on BOTH raw_advantage depth AND signal persistence to require a
    # deeper and more stable divergence before firing.
    #
    #   min_raw_advantage_pct (default 0.0 = off) — raw_adv (pre-tax,
    #     pre-cost) must clear this. Default matches original behaviour.
    #   persistence_bars      (default 0 = off) — the same (sell,buy)
    #     pair must have been proposed on the prior N bars. State is
    #     held by the caller (InferenceContext.prior_rotation_proposals
    #     set) and passed in via rotation_cfg["_prior_proposals"] as a
    #     list of sets of (sell,buy) tuples (most recent last).
    min_raw_adv     = float(rotation_cfg.get("min_raw_advantage_pct", 0.0))
    persistence     = int(rotation_cfg.get("persistence_bars", 0))
    prior_proposals = rotation_cfg.get("_prior_proposals") or []

    st_rate         = float(tax_cfg.get("short_term_rate", 0.37))
    lt_rate         = float(tax_cfg.get("long_term_rate", 0.20))
    lt_threshold    = int(tax_cfg.get("long_term_threshold_days", 365))

    # Eligible held positions (past min hold, both score + ER available, not LT-pinned)
    eligible: dict[str, dict] = {}
    for ticker, score in held_scores.items():
        if score is None:
            continue
        er = held_er.get(ticker)
        if er is None or not math.isfinite(er):
            continue
        meta = held_meta.get(ticker)
        if meta is None:
            continue
        entry_date  = meta.get("entry_date")
        entry_price = float(meta.get("entry_price", 0.0))
        cur_price   = float(meta.get("current_price", 0.0))
        if entry_date is None or entry_price <= 0:
            continue
        hold_days = (today - entry_date).days
        if hold_days < min_hold:
            continue
        unreal_pct = (cur_price - entry_price) / entry_price
        if is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
            continue
        eligible[ticker] = {
            "score":      float(score),
            "er":         float(er),
            "unreal_pct": unreal_pct,
            "tax_drag":   tax_drag(unreal_pct, hold_days,
                                   st_rate, lt_rate, lt_threshold),
        }

    if not eligible or not candidates:
        return []

    used_holds: set[str] = set()
    pairs: list[RotationPair] = []

    for c in candidates:
        if len(pairs) >= max_per_bar:
            break
        cand_ticker = c.ticker
        if cand_ticker in held_scores:
            continue
        cand_score = float(c.rank_score)
        cand_er    = float(getattr(c, "expected_return", 0.0) or 0.0)
        if not math.isfinite(cand_er):
            continue

        # Pick weakest-ER eligible held that this candidate beats by ≥ threshold
        # after costs.  "Weakest" = lowest E[R_horizon] — since the candidate's
        # ER is fixed in this loop iteration, picking the held with the smallest
        # ER maximizes raw_advantage and (for ties on raw_advantage) leaves
        # higher-ER holds available for later, stronger candidates.
        best_match: str | None = None
        best_er: float = math.inf
        for held_ticker, info in eligible.items():
            if held_ticker in used_holds:
                continue
            raw_adv = cand_er - info["er"]
            # V1 gate 1: raw_advantage depth
            if min_raw_adv > 0.0 and raw_adv < min_raw_adv:
                continue
            net_adv = raw_adv - info["tax_drag"] - txn_cost
            if net_adv < threshold:
                continue
            if info["er"] < best_er:
                best_match = held_ticker
                best_er    = info["er"]

        if best_match is None:
            continue

        # V1 gate 2: persistence — the same pair must have appeared on
        # the prior `persistence` bars. When fewer than N bars of history
        # have accumulated, we require all history to contain the pair
        # (fail-closed on cold start so the gate can't be bypassed by
        # restarting the sim).
        if persistence > 0:
            required = min(persistence, len(prior_proposals))
            if required < persistence:
                # Not enough history accumulated yet — skip
                continue
            relevant = prior_proposals[-required:]
            pair_key = (best_match, cand_ticker)
            if not all(pair_key in bar for bar in relevant):
                continue

        info    = eligible[best_match]
        raw_adv = cand_er - info["er"]
        net_adv = raw_adv - info["tax_drag"] - txn_cost
        pairs.append(RotationPair(
            sell_ticker      = best_match,
            buy_ticker       = cand_ticker,
            sell_score       = info["score"],
            buy_score        = cand_score,
            sell_er          = info["er"],
            buy_er           = cand_er,
            horizon_days     = horizon,
            raw_advantage    = raw_adv,
            tax_drag         = info["tax_drag"],
            transaction_cost = txn_cost,
            net_advantage    = net_adv,
            threshold        = threshold,
            margin_realized  = net_adv - threshold,
        ))
        used_holds.add(best_match)

    pairs.sort(key=lambda p: p.margin_realized, reverse=True)
    return pairs
