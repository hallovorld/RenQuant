# §8 Step 5 — Live shadow telemetry envelope schema

**Date**: 2026-06-03
**Status**: Pre-implementation schema spec for the live-shadow
telemetry JSONL the §8 Step 5 daily run will append to under
`doc/research/evidence/`.
**Author**: Claude
**Sibling spec**: [2026-06-03-qp-ab-replay-evidence-schema.md](2026-06-03-qp-ab-replay-evidence-schema.md) (PR #134) — Step 4g offline A/B verdict envelope.

## Purpose

Step 5 fires **only if** Step 4g's verdict selects a non-incumbent
allocator (`next_action: "live_shadow"`). It runs the selected
candidate ALONGSIDE current QP on the live decision path, logging
per-bar telemetry for operational parity verification + fallback-rate
drift detection.

**Critical**: Step 5 is NOT a Sharpe gate. Codex MED-6 on PR #125
rejected the original "30-day shadow → promote" plan. Live shadow
exists to catch implementation-parity issues the offline replay can't
see (broker rounding, real fill prices, live broker fees, intraday
fill latency). The promote/reject decision was already made at Step
4g; Step 5 just confirms the offline result generalises to live
execution.

## Operating mode

The cron path is `live.runner --strategy renquant_104 --broker
readonly-alpaca --once --shadow-allocator <name>` — the existing
shadow flag pattern (PAPER mandate per CLAUDE.md §4) extended to also
run the candidate allocator on the same ctx/snapshot the incumbent
sees.

```
LeanAdapter / RunnerAdapter
  → InferencePipeline runs (incumbent QP)
  → ctx._qp_constraint_snapshot is built (PR #129)
  → SolveMarkowitzQPTask emits incumbent orders (live)
  → BuildLiveShadowTelemetryTask emits candidate AllocatorResult
     (does NOT submit orders; only logs)
  → telemetry appended to doc/research/evidence/qp-live-shadow-{date}.jsonl
```

## Schema — one JSONL row per bar

```json
{
  "as_of_date": "<ISO date>",
  "as_of_time": "<ISO datetime, bar-end>",
  "broker": "alpaca-paper",
  "incumbent": "current_qp",
  "candidate": "hybrid_option_f",
  "constraint_snapshot_contract_version": "v1-2026-06-03",
  "ctx_fingerprint": "<sha256 of snap + mu + sigma>",

  "incumbent": {
    "status": "optimal",
    "target_w": {"AAPL": 0.10, "MSFT": 0.08, ...},
    "delta_w": {"AAPL": 0.02, ...},
    "n_buys": 3, "n_sells": 1,
    "turnover_l1": 0.15,
    "violations_per_family": { "w_upper_hard": 0, "wash_sale": 0, ... },
    "live_orders_emitted": [
      {"ticker": "AAPL", "side": "buy", "qty": 12, "px_est": 188.50},
      ...
    ]
  },

  "candidate": {
    "status": "optimal",
    "target_w": {...},
    "delta_w": {...},
    "n_buys": 4, "n_sells": 0,
    "turnover_l1": 0.18,
    "violations_per_family": { ... },
    "would_have_orders": [
      {"ticker": "TSLA", "side": "buy", "qty": 8, "px_est": 245.00},
      ...
    ]
  },

  "divergence": {
    "abs_target_w_l1": 0.12,         // ‖target_w_inc - target_w_cand‖₁
    "ticker_overlap_pct": 0.80,      // fraction of held names common
    "delta_w_sign_agreement_pct": 0.75,
    "would_be_friendly_fire": [],    // tickers candidate would SELL that incumbent BOUGHT today
    "would_be_missed_alpha": []      // tickers candidate would BUY that incumbent didn't
  },

  "broker_fidelity": {
    "incumbent_orders_filled": 4,
    "incumbent_orders_partial": 0,
    "incumbent_orders_rejected": 0,
    "incumbent_total_slippage_bps": 1.8,
    "incumbent_fee_total_usd": 0.0,    // alpaca: zero unless leverage / shorts
    "share_rounding_loss_bps": 0.5,     // capped + rounded shares vs continuous target
    "candidate_predicted_slippage_bps": <float>,  // candidate's would-have orders × Almgren-Chriss
    "candidate_predicted_fee_total_usd": <float>
  },

  "regime": "BULL_CALM",
  "regime_confidence": 0.72,
  "panel_artifact": "<panel-ltr.alpha158_fund.json sha>",

  "anomalies": [
    // any infeasibility / fallback that fired on either path:
    // "candidate_status=infeasible:sector_cap",
    // "incumbent_qp_used_cap_compliance_fallback",
    // etc.
  ]
}
```

## File-of-record convention

```
doc/research/evidence/
  qp-live-shadow-YYYY-MM-DD.jsonl   ← one row per bar; new file per day
  qp-live-shadow-summary.json       ← rolling N-day aggregate (regenerated nightly)
```

The aggregate `qp-live-shadow-summary.json` distills the JSONL stream
into the metrics the user reads:

```json
{
  "as_of_date": "<ISO date>",
  "window_days": 30,
  "incumbent": "current_qp",
  "candidate": "hybrid_option_f",
  "n_bars": <int>,
  "candidate_status_distribution": {
    "optimal": <pct>,
    "qp_fallback_fired": <pct>,
    "infeasible:hybrid_qp_fallback": <pct>,
    "no_candidates": <pct>
  },
  "mean_divergence_target_w_l1": <float>,
  "mean_ticker_overlap_pct": <float>,
  "candidate_predicted_vs_realised_slippage_bps": <float>,
  "implementation_parity_score": <float in [0,1]>,
  "anomaly_count_by_type": { ... },
  "ready_for_promotion": <bool>,
  "promotion_reason": "<text>",
  "shadow_days_logged": <int>,
  "shadow_days_needed": <int>
}
```

`ready_for_promotion` becomes `true` when:
- `shadow_days_logged ≥ shadow_days_needed` (default 30 trading days);
- `candidate_status_distribution["infeasible:*"] < 5%`;
- `mean_ticker_overlap_pct ≥ 0.70` (the candidate isn't picking a
  wildly different universe — that would suggest a contract bug);
- `anomaly_count_by_type` has zero `qp_constraint_snapshot_invalid`
  events (any architecture-level failure must be investigated before
  promotion).

## What's NOT in the shadow

- The candidate's orders are **never submitted**. Only the incumbent's
  orders reach Alpaca. This is the safety invariant per CLAUDE.md
  §4.1 (PAPER mandate); broker-side state stays driven by the
  incumbent until promotion completes.
- Per-fill broker confirmations are stamped on the INCUMBENT side
  only — the candidate's "would_have_orders" uses Almgren-Chriss
  predicted slippage from the prod σ̂ estimate.
- Tax / wash-sale lots track the incumbent only. Candidate-side
  wash-sale projection uses the incumbent's lot ledger as if the
  candidate had been driving since session start — flagged in the
  anomalies array when this materially diverges.

## Path from Step 4g to Step 5

```
                Step 4g verdict
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   reject_all     iterate         live_shadow
   (do nothing)   (refine 4d/4e   (THIS PR's scope:
                  and re-run)      30 trading days
                                   of shadow + summary)
                                          │
                                          ▼
                                ready_for_promotion?
                                          │
                            ┌─────────────┼─────────────┐
                            ▼                            ▼
                          true                        false
                            │                            │
                            ▼                            ▼
                       human-approval                 abandon
                       + flip config                  (verdict
                       + audit trail                  promote-to-
                                                      reject)
```

## Reviewer pre-emption

| Likely concern | Pre-emptive resolution |
|---|---|
| "Why 30 trading days?" | Operational-telemetry gate, not Sharpe. 30 trading days is enough to catch fall-back-rate drift and broker-side anomalies; codex MED-6 on PR #125 explicitly rejected the original "30-day Sharpe-gate" framing — this is a separate concern. |
| "What if the candidate would have submitted orders that conflict with the incumbent's?" | The candidate never submits. Conflicts are logged in `divergence.would_be_friendly_fire` for human review during the promotion-approval step. |
| "Why is the candidate's wash-sale projection based on the incumbent's ledger?" | The candidate has no separate broker book to track; projecting against the incumbent's ledger lets the operator see how often the candidate would have hit a wash-sale gate without standing up a parallel lot tracker. |
| "When is this run?" | The §8 plan calls for daily cadence — same cron as the prod `daily_104.sh`. The shadow Task is additive: insert at the end of `InferencePipeline` before order emission, no effect on live orders. |
