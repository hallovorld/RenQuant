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

    Audit fix R-1 (Round 8, 2026-04-25): pre-fix, NaN unrealized_pnl_pct
    slipped past `<= 0` and propagated into the multiplication →
    `tax_drag = NaN` → `effective_swap_margin = base + NaN = NaN` →
    every rotation pair comparison returned False (NaN < margin is False).
    Net effect: rotation gate silently rejected all swaps.
    """
    import math
    if unrealized_pnl_pct is None or not math.isfinite(float(unrealized_pnl_pct)):
        return 0.0
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
    panel_buy_floor:   "float | None" = None,        # candidate rank_score must be >= this
    panel_sell_floor:  "float | None" = None,        # held today rank_score must be <= this
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

    st_rate         = float(tax_cfg.get("short_term_rate", 0.50))
    lt_rate         = float(tax_cfg.get("long_term_rate", 0.32))
    lt_threshold    = int(tax_cfg.get("long_term_threshold_days", 365))

    eligible: dict[str, dict] = {}
    for ticker, entry_score in held_entry_scores.items():
        if entry_score is None or entry_score <= 0:
            continue
        today_score = held_today_scores.get(ticker)
        if today_score is None:
            continue
        # Phase 1 (2026-04-25): panel_sell_floor — held position only
        # eligible to rotate OUT when today's calibrated rank_score is
        # weak enough (<= floor). Spec: 被替换的 portfolio 里的 stock
        # 的 score 要低于一个值。
        if panel_sell_floor is not None and float(today_score) > float(panel_sell_floor):
            continue
        meta = held_meta.get(ticker)
        if meta is None:
            continue
        entry_date  = meta.get("entry_date")
        entry_price = float(meta.get("entry_price", 0.0))
        cur_price   = float(meta.get("current_price", 0.0))
        # Audit fix ROT-KERNEL-PRICE-NaN (Round 2 deep audit, 2026-04-25):
        # NaN current_price slipped past `entry_price <= 0` check, then
        # unreal_pct = NaN propagated through is_lt_protected() (NaN > 0
        # False → not LT-protected → swap allowed) and into tax_drag()
        # which returns 0 on NaN (R-1 fix), so the rotation got compared
        # at zero-tax-cost when it should have been skipped entirely.
        # Now: explicit isfinite check on cur_price.
        if (entry_date is None or entry_price <= 0
                or not math.isfinite(cur_price) or cur_price <= 0):
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
        # Phase 1 (2026-04-25): panel_buy_floor — candidate must clear
        # this calibrated rank_score before it can replace anyone.
        # Spec: 进到 portfolio 的 stock 的 score 要高于一个值。
        if panel_buy_floor is not None and cand_score < float(panel_buy_floor):
            continue

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


def find_thesis_symmetric_pairs(
    held_entry_scores: dict[str, "float | None"],       # A's rank at A's entry
    held_today_scores: dict[str, "float | None"],       # A's rank today
    held_meta:         dict[str, dict],                 # {entry_date, entry_price, current_price}
    candidates:        list,                            # CandidateResult-like (today's)
    entry_day_lookup:  "dict[tuple[str, datetime.date], float | None]",
                                                        # (B_ticker, A_entry_date) → B's rank on that date
    today:             datetime.date,
    rotation_cfg:      dict,
    tax_cfg:           dict,
    own_momentum:      "dict[str, float] | None" = None,
                                                        # {ticker: 63d return}
    panel_buy_floor:   "float | None" = None,            # candidate rank_score >= this
    panel_sell_floor:  "float | None" = None,            # held today rank_score <= this
) -> list[RotationPair]:
    """Rotation V4 — full 4-point symmetric thesis mode (2026-04-24).

    User spec: "综合考虑买进 A 当天 AB 的 decision factor（from DB）和
    今天 AB 的 scores，来决定是否 rotate". Four-way comparison:

        a_velocity = A_today − A_entry      # held decay (more negative = better to swap)
        b_velocity = B_today − B_entry      # cand momentum (more positive = better)
        cross_flip = (B_today − A_today) − (B_entry − A_entry)
                   = today gap − entry gap  # has B overtaken A since A's entry?

    Thresholds:
      * a_velocity ≤ −max_a_velocity          (A must have lost >= X)
      * b_velocity ≥ +min_b_velocity          (B must have gained >= Y)
      * cross_flip ≥ min_cross_flip           (gap must have widened >= Z)

    All three must hold for a pair to fire. `entry_day_lookup` is a dict
    keyed by (cand_ticker, A_entry_date) → rank_score at that historical
    point. When missing, the pair is skipped (no B_entry info → can't
    decide); downstream can fall back to other rotation modes.
    """
    if not rotation_cfg.get("enabled", False):
        return []

    thesis_cfg         = rotation_cfg.get("thesis_symmetric", {})
    max_a_velocity     = float(thesis_cfg.get("max_a_velocity", 0.10))
    min_b_velocity     = float(thesis_cfg.get("min_b_velocity", 0.05))
    min_cross_flip     = float(thesis_cfg.get("min_cross_flip", 0.15))
    # Proposal 1 from rotation_research_2026-04-24.md — own time-series
    # momentum gate (Moskowitz-Ooi-Pedersen 2012). When enabled and
    # own_momentum dict provided, require A's own 63d return ≤ a_mom_max
    # (has broken down) AND B's own 63d return ≥ b_mom_min (still trending).
    # Jegadeesh 1993: winners keep winning; don't rotate out of an asset
    # whose own time-series momentum is intact.
    own_mom_enabled    = bool(thesis_cfg.get("own_momentum_enabled", False))
    a_mom_max          = float(thesis_cfg.get("a_own_mom_max", 0.0))   # A must be ≤ this
    b_mom_min          = float(thesis_cfg.get("b_own_mom_min", 0.0))   # B must be ≥ this
    min_hold           = int(rotation_cfg.get("min_rotation_hold_days", 30))
    lt_protect         = int(rotation_cfg.get("lt_protection_days", 30))
    max_per_bar        = int(rotation_cfg.get("max_rotations_per_bar", 2))
    txn_cost           = float(rotation_cfg.get("transaction_cost_pct", 0.0))
    horizon            = int(rotation_cfg.get("target_horizon_days", 20))

    st_rate            = float(tax_cfg.get("short_term_rate", 0.50))
    lt_rate            = float(tax_cfg.get("long_term_rate", 0.32))
    lt_threshold       = int(tax_cfg.get("long_term_threshold_days", 365))

    # Eligible held positions — past min hold, have entry score, not LT-pinned
    eligible: dict[str, dict] = {}
    for ticker, a_entry in held_entry_scores.items():
        if a_entry is None or a_entry <= 0:
            continue
        a_today = held_today_scores.get(ticker)
        if a_today is None:
            continue
        # Phase 1 (2026-04-25): panel_sell_floor — held weak enough to swap out.
        if panel_sell_floor is not None and float(a_today) > float(panel_sell_floor):
            continue
        meta = held_meta.get(ticker)
        if meta is None:
            continue
        entry_date  = meta.get("entry_date")
        entry_price = float(meta.get("entry_price", 0.0))
        cur_price   = float(meta.get("current_price", 0.0))
        # Audit fix ROT-KERNEL-PRICE-NaN (mirror of fix in
        # find_thesis_primary_pairs): NaN cur_price → NaN unreal_pct →
        # NaN tax_drag (returns 0) → swap looks free, gets emitted.
        if (entry_date is None or entry_price <= 0
                or not math.isfinite(cur_price) or cur_price <= 0):
            continue
        hold_days = (today - entry_date).days
        if hold_days < min_hold:
            continue
        unreal_pct = (cur_price - entry_price) / entry_price
        if is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
            continue
        a_velocity = float(a_today) - float(a_entry)
        if a_velocity > -max_a_velocity:
            continue   # A hasn't decayed enough
        # Proposal 1 own-momentum gate — A's OWN time-series momentum
        # must have also broken (negative or below threshold). Without
        # this, we'd rotate out of winners that Jegadeesh says keep winning.
        if own_mom_enabled and own_momentum is not None:
            a_mom = own_momentum.get(ticker)
            if a_mom is not None and a_mom > a_mom_max:
                continue
        eligible[ticker] = {
            "a_entry":   float(a_entry),
            "a_today":   float(a_today),
            "a_vel":     a_velocity,
            "entry_date": entry_date,
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
        if cand_ticker in held_entry_scores:
            continue
        b_today = float(getattr(c, "rank_score", 0.0) or 0.0)
        # Phase 1 (2026-04-25): panel_buy_floor — candidate strong enough to enter.
        if panel_buy_floor is not None and b_today < float(panel_buy_floor):
            continue

        best_match: "str | None" = None
        best_flip: float = -math.inf

        for held_ticker, info in eligible.items():
            if held_ticker in used_holds:
                continue
            # Look up B's rank_score on A's entry date
            b_entry = entry_day_lookup.get((cand_ticker, info["entry_date"]))
            if b_entry is None:
                continue
            b_entry = float(b_entry)
            b_velocity = b_today - b_entry
            if b_velocity < min_b_velocity:
                continue
            # Proposal 1 — B's own momentum must be intact (≥ threshold).
            # Don't rotate INTO a falling knife just because rank lifted.
            if own_mom_enabled and own_momentum is not None:
                b_mom = own_momentum.get(cand_ticker)
                if b_mom is not None and b_mom < b_mom_min:
                    continue
            cross_flip = (b_today - info["a_today"]) - (b_entry - info["a_entry"])
            if cross_flip < min_cross_flip:
                continue
            # Pick the held with the BIGGEST positive cross_flip — the
            # pair where B has overtaken A the most.
            if cross_flip > best_flip:
                best_match = held_ticker
                best_flip  = cross_flip

        if best_match is None:
            continue

        info   = eligible[best_match]
        b_entry = float(entry_day_lookup[(cand_ticker, info["entry_date"])])
        b_velocity = b_today - b_entry
        cross_flip = (b_today - info["a_today"]) - (b_entry - info["a_entry"])

        pairs.append(RotationPair(
            sell_ticker      = best_match,
            buy_ticker       = cand_ticker,
            sell_score       = info["a_today"],
            buy_score        = b_today,
            sell_er          = info["a_vel"],       # re-purposed: a_velocity
            buy_er           = b_velocity,          # re-purposed: b_velocity
            horizon_days     = horizon,
            raw_advantage    = cross_flip,          # re-purposed: cross_flip
            tax_drag         = info["tax_drag"],
            transaction_cost = txn_cost,
            net_advantage    = cross_flip - info["tax_drag"] - txn_cost,
            threshold        = min_cross_flip,
            margin_realized  = cross_flip - min_cross_flip,
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
    panel_buy_floor:  "float | None" = None,    # candidate rank_score >= this
    panel_sell_floor: "float | None" = None,    # held rank_score <= this
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
    # V3 (2026-04-24): drawdown-of-held gate — protect hot runners.
    # When set (e.g. 0.05), only holdings whose unrealized_pct <= this
    # ceiling are eligible to rotate OUT. So positions up ≤5% (including
    # losers) can swap; positions up >5% are protected. Default None =
    # no protection (all eligible).
    held_max_unreal_raw = rotation_cfg.get("held_max_unrealized_pct")
    held_max_unreal     = (float(held_max_unreal_raw)
                            if held_max_unreal_raw is not None else None)

    st_rate         = float(tax_cfg.get("short_term_rate", 0.50))
    lt_rate         = float(tax_cfg.get("long_term_rate", 0.32))
    lt_threshold    = int(tax_cfg.get("long_term_threshold_days", 365))

    # Eligible held positions (past min hold, both score + ER available, not LT-pinned)
    eligible: dict[str, dict] = {}
    for ticker, score in held_scores.items():
        if score is None:
            continue
        # Phase 1 (2026-04-25): panel_sell_floor — held rank_score weak
        # enough to swap out. Spec: 被替换的 portfolio 里的 stock 的
        # score 要低于一个值。
        if panel_sell_floor is not None and float(score) > float(panel_sell_floor):
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
        # Audit fix ROT-KERNEL-PRICE-NaN (mirror of fix in
        # find_thesis_primary_pairs / find_thesis_symmetric_pairs).
        if (entry_date is None or entry_price <= 0
                or not math.isfinite(cur_price) or cur_price <= 0):
            continue
        hold_days = (today - entry_date).days
        if hold_days < min_hold:
            continue
        unreal_pct = (cur_price - entry_price) / entry_price
        if is_lt_protected(unreal_pct, hold_days, lt_threshold, lt_protect):
            continue
        # V3 gate: don't rotate out of hot runners
        if held_max_unreal is not None and unreal_pct > held_max_unreal:
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
        # 2026-05-04 audit Issue 33 fix: NaN rank_score slipped past the
        # `cand_score < float(panel_buy_floor)` check (NaN < X is False)
        # → candidate proceeded as if it crossed the buy floor. Same
        # NaN-slip class as Issues 6/7/18/19/22. Reject NaN here.
        cand_score = float(c.rank_score) if c.rank_score is not None else float("nan")
        if not math.isfinite(cand_score):
            continue
        # Phase 1 (2026-04-25): panel_buy_floor — candidate strong enough.
        if panel_buy_floor is not None and cand_score < float(panel_buy_floor):
            continue
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
