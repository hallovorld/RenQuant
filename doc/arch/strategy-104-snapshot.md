# renquant_104 — generated production snapshot

GENERATED FILE — do not hand-edit. Regenerate with:
`python3 scripts/render_strategy_104_snapshot.py` (or `make snapshot`);
verify with `make snapshot-check`.

Rendered from the PINNED strategy-104 config (the `.subrepo_runtime/repos/renquant-strategy-104/configs/` checkout that the
daily run actually consumes — NOT the umbrella working copy, which is a
known rot vector) plus each referenced artifact's own stamped metadata —
see amendment A6, doc/design/2026-07-01-104-105-design-review-amendments.md
and the unified 107 master plan M9. This states ONLY what the pinned
sources say AS OF the last regeneration — a current fact, never a
historical/promotion claim ("active since <date>", "promoted on <date>");
that narrative, with its own dating and provenance, belongs in
doc/arch/strategy-104.md instead. Fields the sources do not stamp are
rendered as explicit unknowns, never invented.

Source fingerprint: 0558d81d119bd98f6c60431b8e0f1885d8e5e5d571eec6cec2598ab6ba804a94 (sha256 over the sorted per-file source hashes below — deterministic; changes iff pinned/artifact CONTENT changes, never on a bare regeneration. EXCLUDES the pooled calibrators: they are re-fit per promote (mutable live state) and are recorded below as runtime observations, NOT folded into this candidate-interface fingerprint)

## Provenance

| | |
|---|---|
| Pinned config root | `.subrepo_runtime/repos/renquant-strategy-104/configs` |
| strategy-104 runtime checkout commit | unknown (field absent) |
| subrepos.lock.json strategy-104 pin | d3c8026a8ac222c4b92b5070a906445ce0153ff6 |

### Source warnings

- **pinned active config unreadable or missing: .subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json**
- **pinned shadow config unreadable or missing: .subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json**

## Active scorer

| | |
|---|---|
| Scorer kind | unknown (field absent) |
| Artifact | unknown (field absent) |
| Artifact metadata file fingerprint | unknown (file missing) |
| trained_date | unknown (field absent) |
| Binding data cutoff | unknown (field absent) |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| label_col | unknown (field absent) |
| Feature count | unknown (field absent) |
| train_run_id | unknown (field absent) |
| oos_mean_ic (stamped) | unknown (field absent) |
| promotion_status | unknown (field absent) |
| config_fingerprint | unknown (field absent) |
| WF gate (stamped) | unknown (field absent) |

## Active calibrator

> Runtime observation, not a locked identity: the pooled calibrator is re-fit per promote, so the digest below reflects the live artifact as of this regeneration and is EXCLUDED from the candidate Source fingerprint above.

| | |
|---|---|
| Artifact | unknown (field absent) |
| Artifact file fingerprint | unknown (file missing) |
| kind | unknown (field absent) |
| trained_date | unknown (field absent) |
| Method | unknown (field absent) |
| pool_ic (stamped) | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| Bound scorer content fingerprint | unknown (field absent) |
| Fit data window | unknown (field absent) |

## In-run shadow scorers (readonly, same run)

(none configured)

## Shadow e2e config (strategy_config.shadow.json)

| | |
|---|---|
| Scorer kind | unknown (field absent) |
| Artifact | unknown (field absent) |
| Artifact metadata file fingerprint | unknown (file missing) |
| trained_date | unknown (field absent) |
| Binding data cutoff | unknown (field absent) |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| label_col | unknown (field absent) |
| Feature count | unknown (field absent) |
| train_run_id | unknown (field absent) |
| oos_mean_ic (stamped) | unknown (field absent) |
| promotion_status | unknown (field absent) |
| config_fingerprint | unknown (field absent) |
| WF gate (stamped) | unknown (field absent) |

### Shadow e2e calibrator

> Runtime observation (re-fit per promote), excluded from the candidate Source fingerprint — same as the active calibrator.

| | |
|---|---|
| Artifact | unknown (field absent) |
| Artifact file fingerprint | unknown (file missing) |
| kind | unknown (field absent) |
| trained_date | unknown (field absent) |
| Method | unknown (field absent) |
| pool_ic (stamped) | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| Bound scorer content fingerprint | unknown (field absent) |
| Fit data window | unknown (field absent) |

## Key policy knobs (active pinned config)

| | |
|---|---|
| Watchlist size | 0 tickers |
| Conviction gate μ floor | enabled=unknown (field absent); mu_floor=unknown (field absent); demean_cross_sectional=unknown (field absent) |
| signal_gate_prefer_calibrated_mu | unknown (field absent) |
| Buy floor | mode=unknown (field absent); min=unknown (field absent) |
| panel_buy_top_n | unknown (field absent) |
| Rotation | min_expected_advantage_pct=unknown (field absent); target_horizon_days=unknown (field absent) |
| Kelly sizing | enabled=unknown (field absent); fractional=unknown (field absent); max_concentration=unknown (field absent); min_edge=unknown (field absent); use_calibrator_mu=unknown (field absent) |
| Position caps | max_concurrent_positions=unknown (field absent); max_position_pct=unknown (field absent); max_positions_per_sector=unknown (field absent) |
| model_staleness_days | unknown (field absent) |
| QP | risk_aversion=unknown (field absent); turnover_max=unknown (field absent); no_trade_band_cap=unknown (field absent); mu_horizon_days=unknown (field absent); admission_min_rank_score=unknown (field absent) |
| WF gate relaxations (lock-declared) | benchmark_required=unknown (field absent); regime_required=unknown (field absent); sanity_regime_ic_required=unknown (field absent) |

## Subrepo pins (subrepos.lock.json)

| Subrepo | Branch | Commit | Status |
|---|---|---|---|
| renquant-artifacts | main | `c09d66f8dd09` | bootstrapped |
| renquant-backtesting | main | `e5f9bae3b1e2` | bootstrapped |
| renquant-base-data | main | `f8514066b53f` | active |
| renquant-common | main | `ef7726dd6c90` | bootstrapped |
| renquant-execution | main | `91c7bf8873fd` | bootstrapped |
| renquant-model | main | `bd0fa488d216` | active |
| renquant-orchestrator | main | `64238032744f` | active |
| renquant-pipeline | main | `b1905d5b3a55` | bootstrapped |
| renquant-strategy-104 | main | `d3c8026a8ac2` | bootstrapped |

## Source fingerprints

- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json` — unknown (file missing)
- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json` — unknown (file missing)
- `subrepos.lock.json` — sha256:66ccd65cbbf10c9b

<!-- snapshot-machine-block
{
 "active_artifact": null,
 "active_kind": null,
 "in_run_shadow_kinds": [],
 "lock_strategy_104_pin": "d3c8026a8ac222c4b92b5070a906445ce0153ff6",
 "schema_version": 2,
 "shadow_e2e_kind": null,
 "sources_sha256": {
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json": null,
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json": null,
  "subrepos.lock.json": "sha256:66ccd65cbbf10c9b"
 },
 "strategy_104_pin": null
}
-->
