"""MetaLabelVetoTask — veto false-positive path-rule exits.

Stage 2 of the meta-labeling pipeline (López de Prado AFML 2018 ch.20).
Sits in pp_inference.py AFTER the parallel Phase-2a TickerSellJob has
populated ctx.exits and BEFORE the buy phase, in the same architectural
slot as DrawdownFlattenTask.

For each path-rule exit (stop_loss / trailing_stop / single_day_loss /
max_hold) in ctx.exits, query the meta-label predictor with the SAME
feature set used at training time (FEATURE_COLUMNS minus identifiers
and outcomes). If P(exit_is_profitable) < threshold, **remove the
exit** — the path rule is deemed a false positive and the position is
held.

Model-driven exits (model_sell, qp_*, panel_conviction) are NEVER
vetoed: their generation is itself a model decision; layering a second
model on top is double-counting.

Fail-safe contract (CLAUDE.md §5.13.10):
  * predictor missing / None → no-op (keep all exits)
  * predictor returns NaN / raises → keep the specific exit
  * config block missing → no-op
"""
from __future__ import annotations

import math
import datetime
from typing import Any

from kernel.pipeline.context import InferenceContext
from kernel.pipeline.pipeline import Task

# Canonical exit-type taxonomy (CLAUDE.md §5.13.5 — single source).
# Refactored 2026-05-11 — kernel/exit_types.META_LABEL_VETO_ELIGIBLE
# owns the lookup. Only PATH_RULE_CORE names (no synonyms) because the
# meta-label classifier was trained on those canonical exit_types.
from renquant_pipeline.kernel.exit_types import META_LABEL_VETO_ELIGIBLE as _PATH_RULE_EXITS  # noqa: E402

# Regime → integer encoding — must match
# kernel/meta_label/task_snapshot.py::_REGIME_CODE.
_REGIME_CODE: dict[str, int] = {
    "BULL_CALM": 0, "BULL_VOLATILE": 1, "CHOPPY": 2, "BEAR": 3,
}


def _safe_div(a: Any, b: Any) -> float:
    try:
        af = float(a); bf = float(b)
    except (TypeError, ValueError):
        return float("nan")
    if not (math.isfinite(af) and math.isfinite(bf)) or bf == 0:
        return float("nan")
    return af / bf


def _build_features_for_ticker(
    ticker:        str,
    sig,
    holding,
    ctx:           InferenceContext,
    n_concurrent_triggers: int,
) -> dict:
    """Build the per-ticker feature dict matching
    ``kernel.meta_label.snapshot.FEATURE_COLUMNS``.

    Identifier columns (date, ticker) are included so the predictor
    contract is symmetric with the trained model's input. The forward-
    return columns (fwd_5d_ret / fwd_20d_ret) are NOT included at
    inference time (they are the LABEL source, only known
    post-hoc) — predictor should drop them per training-time
    feature_cols.
    """
    today = ctx.today
    date_iso = today.isoformat() if hasattr(today, "isoformat") else str(today)
    price = ctx.prices.get(ticker) if hasattr(ctx, "prices") else None

    cum_pnl = _safe_div(
        price - holding.entry_price if price is not None else float("nan"),
        holding.entry_price,
    )
    peak_gain = _safe_div(holding.high_watermark - holding.entry_price,
                          holding.entry_price)
    dd_from_peak = _safe_div(
        holding.high_watermark - price if price is not None else float("nan"),
        holding.high_watermark,
    )
    days_held = 0
    if holding.entry_date is not None and isinstance(today, datetime.date):
        days_held = max(0, (today - holding.entry_date).days)

    exit_type = getattr(sig, "exit_type", "")
    feats = {
        "date": date_iso,
        "ticker": ticker,
        "cum_pnl_pct":             cum_pnl,
        "peak_gain_pct":           peak_gain,
        "drawdown_from_peak_pct":  dd_from_peak,
        "days_held":               days_held,
        "consec_underwater_days":  0,
        "prev_day_return":         float("nan"),
        "gap_open_pct":            float("nan"),
        "realized_vol_20d": (holding.realized_sigma_daily * math.sqrt(252.0)
                              if holding.realized_sigma_daily is not None
                              else float("nan")),

        # Market features — same code as SnapshotHoldingsTask
        "spy_5d_ret":              _cum_ret(ctx.spy_returns, 5),
        "spy_20d_ret":             _cum_ret(ctx.spy_returns, 20),
        "spy_60d_ret":             _cum_ret(ctx.spy_returns, 60),
        "spy_realized_vol_20d":    _rv(ctx.spy_returns, 20),
        "regime_code":             _REGIME_CODE.get(ctx.regime, -1),
        "regime_just_switched":    0,
        "regime_confidence":       float(getattr(ctx, "confidence", float("nan"))),

        "panel_score_current":             (holding.panel_score
                                              if holding.panel_score is not None
                                              else float("nan")),
        "panel_score_at_entry":            (holding.entry_panel_score
                                              if holding.entry_panel_score is not None
                                              else float("nan")),
        "panel_score_delta":               ((holding.panel_score - holding.entry_panel_score)
                                              if holding.panel_score is not None
                                              and holding.entry_panel_score is not None
                                              else float("nan")),
        "panel_score_rank_among_holdings": -1,
        "mu_current":              (holding.mu    if holding.mu    is not None else float("nan")),
        "sigma_current":           (holding.sigma if holding.sigma is not None else float("nan")),

        "position_weight":               float("nan"),
        "sector_concentration":          float("nan"),
        "portfolio_drawdown_now":        _safe_div(
            getattr(ctx, "hwm", 0.0) - getattr(ctx, "portfolio_value", 0.0),
            getattr(ctx, "hwm", 0.0)
        ),
        "n_concurrent_exits_this_bar":   int(n_concurrent_triggers),

        "trigger_stop_loss":         1 if exit_type == "stop_loss"       else 0,
        "trigger_trailing_stop":     1 if exit_type == "trailing_stop"   else 0,
        "trigger_single_day_loss":   1 if exit_type == "single_day_loss" else 0,
        "trigger_max_hold":          1 if exit_type == "max_hold"        else 0,
        "any_trigger":               1,

        "fwd_5d_ret":  float("nan"),
        "fwd_20d_ret": float("nan"),
    }
    return feats


def _cum_ret(rets, window: int) -> float:
    if not rets or len(rets) < window:
        return float("nan")
    prod = 1.0
    for r in rets[-window:]:
        try:
            f = float(r)
        except (TypeError, ValueError):
            return float("nan")
        if not math.isfinite(f):
            return float("nan")
        prod *= (1.0 + f)
    return prod - 1.0


def _rv(rets, window: int) -> float:
    if not rets or len(rets) < window:
        return float("nan")
    seg = [float(r) for r in rets[-window:]
           if r is not None and math.isfinite(float(r))]
    if len(seg) < 2:
        return float("nan")
    mean = sum(seg) / len(seg)
    var  = sum((r - mean) ** 2 for r in seg) / (len(seg) - 1)
    if var <= 0 or not math.isfinite(var):
        return float("nan")
    return math.sqrt(var) * math.sqrt(252.0)


class MetaLabelVetoTask(Task):
    """Drop ctx.exits entries where the meta-label predictor says the
    path-rule exit is a false positive (low P(profitable_exit))."""

    def run(self, ctx: InferenceContext) -> bool | None:
        cfg = (ctx.config.get("ranking") or {}).get("meta_label") or {}
        if not cfg.get("enabled", False):
            return None
        predictor = getattr(ctx, "_meta_label_predictor", None)
        if predictor is None:
            return None
        try:
            threshold = float(cfg.get("threshold", 0.5))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(threshold):
            return None

        all_exits = list(ctx.exits or [])
        n_triggers = sum(
            1 for (_t, s) in all_exits
            if s is not None and getattr(s, "exit_type", "") in _PATH_RULE_EXITS
        )
        kept: list = []
        n_veto = 0
        for (ticker, sig) in all_exits:
            if sig is None or not getattr(sig, "should_exit", False):
                kept.append((ticker, sig))
                continue
            exit_type = getattr(sig, "exit_type", "")
            # Only path-rule exits are eligible for meta-veto.
            if exit_type not in _PATH_RULE_EXITS:
                kept.append((ticker, sig))
                continue
            holding = ctx.holdings.get(ticker)
            if holding is None:
                kept.append((ticker, sig))
                continue
            feats = _build_features_for_ticker(
                ticker, sig, holding, ctx, n_triggers,
            )
            try:
                p = float(predictor(feats))
            except Exception:  # noqa: BLE001
                # Fail-safe: keep the exit if model crashed
                kept.append((ticker, sig))
                continue
            if not math.isfinite(p):
                # Fail-safe: NaN prediction → keep exit (don't accidentally
                # veto on garbage)
                kept.append((ticker, sig))
                continue
            if p < threshold:
                # VETO — remove the exit
                n_veto += 1
                continue
            kept.append((ticker, sig))

        ctx.exits = kept
        if n_veto > 0:
            ctx.counters["meta_veto"] = ctx.counters.get("meta_veto", 0) + n_veto
        return None
