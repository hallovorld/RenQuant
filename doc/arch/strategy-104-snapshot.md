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

Source fingerprint: fa2f0201dd37874624038c3cb72a9445473fb4ea6db342b606368cb476f99a1c (sha256 over the sorted per-file source hashes below — deterministic; changes iff pinned/artifact CONTENT changes, never on a bare regeneration)

## Provenance

| | |
|---|---|
| Pinned config root | `.subrepo_runtime/repos/renquant-strategy-104/configs` |
| strategy-104 runtime checkout commit | 1e840cd7fc26c0ef277ffb0ce7d1375abe398b7c |
| subrepos.lock.json strategy-104 pin | 1e840cd7fc26c0ef277ffb0ce7d1375abe398b7c |

### Source warnings

- **UMBRELLA WORKING-COPY DRIFT: backtesting/renquant_104/strategy_config.json declares kind='hf_patchtst' but the pinned config declares kind='xgb' — the pinned config is what the daily run consumes; the working copy is stale**

## Active scorer

| | |
|---|---|
| Scorer kind | `xgb` |
| Artifact | `artifacts/prod/panel-ltr.alpha158_fund.json` |
| Artifact metadata file fingerprint | sha256:04d7a381cd6df847 |
| trained_date | 2026-06-21 |
| Binding data cutoff | unknown (field absent) |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | 60 |
| label_col | fwd_60d_excess |
| Feature count | 172 |
| train_run_id | 8daa5b88 |
| oos_mean_ic (stamped) | +0.0533 |
| promotion_status | unknown (field absent) |
| config_fingerprint | sha256:f8fb2259b2bf1537 |
| WF gate (stamped) | passed=true; run_at=2026-06-22T05:37:02.965705; sanity_eval_end=2026-02-11 |

## Active calibrator

| | |
|---|---|
| Artifact | `artifacts/prod/panel-rank-calibration.json` |
| Artifact file fingerprint | sha256:d2b4d6ab197f0503 |
| kind | global_panel_calibration |
| trained_date | 2026-05-21 |
| Method | platt |
| pool_ic (stamped) | +0.1149 |
| lookahead_days | 60 |
| Bound scorer content fingerprint | sha256:6fc9985e6e53e3883a13d572b1e11b7dea484e14df23819dab5474f2cd592372 |
| Fit data window | unknown (field absent) |

## In-run shadow scorers (readonly, same run)

| | |
|---|---|
| Name | `hf_patchtst_pt07_strict_seed44_previous_primary` |
| Scorer kind | `hf_patchtst` |
| Artifact | `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt` |
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
| Metadata | unknown (metadata file missing or unreadable) |

## Shadow e2e config (strategy_config.shadow.json)

| | |
|---|---|
| Scorer kind | `hf_patchtst` |
| Artifact | `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt` |
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
| Metadata | unknown (metadata file missing or unreadable) |

### Shadow e2e calibrator

| | |
|---|---|
| Artifact | `artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json` |
| Artifact file fingerprint | sha256:bc3b8a8f803e4685 |
| kind | global_panel_calibration |
| trained_date | 2026-06-01 |
| Method | platt |
| pool_ic (stamped) | +0.1309 |
| lookahead_days | 60 |
| Bound scorer content fingerprint | sha256:07046963994dbb8da29bfc66f99d21399e39d6d2dbd842c180299bce67c07571 |
| Fit data window | 2023-01-03 → 2024-04-09 |

## Key policy knobs (active pinned config)

| | |
|---|---|
| Watchlist size | 145 tickers |
| Conviction gate μ floor | enabled=true; mu_floor=+0.0300; demean_cross_sectional=false |
| signal_gate_prefer_calibrated_mu | unknown (field absent) |
| Buy floor | mode=adaptive_mean_std; min=+0.2000 |
| panel_buy_top_n | 3 |
| Rotation | min_expected_advantage_pct=+0.0600; target_horizon_days=60 |
| Kelly sizing | enabled=true; fractional=+0.3000; max_concentration=+0.1200; min_edge=0; use_calibrator_mu=true |
| Position caps | max_concurrent_positions=8; max_position_pct=+0.1500; max_positions_per_sector=6 |
| model_staleness_days | 60 |
| QP | risk_aversion=3; turnover_max=+0.2000; no_trade_band_cap=+0.0500; mu_horizon_days=60; admission_min_rank_score=+0.5500 |
| WF gate relaxations (lock-declared) | benchmark_required=false; regime_required=false; sanity_regime_ic_required=false |

### Per-regime caps

| Regime | max_position_pct | qp_turnover_max | cash_reserve_pct | stop_loss_pct |
|---|---|---|---|---|
| BEAR | 0 | unknown (field absent) | 1 | +0.0500 |
| BULL_CALM | +0.1200 | +0.1500 | 0 | +0.1500 |
| BULL_VOLATILE | +0.2000 | unknown (field absent) | +0.2000 | +0.0500 |
| CHOPPY | +0.1500 | unknown (field absent) | +0.3000 | +0.0800 |

## Subrepo pins (subrepos.lock.json)

| Subrepo | Branch | Commit | Status |
|---|---|---|---|
| renquant-artifacts | main | `c09d66f8dd09` | bootstrapped |
| renquant-backtesting | main | `8f6700ab3558` | bootstrapped |
| renquant-base-data | main | `021ca6474102` | active |
| renquant-common | main | `6f09cb99ae47` | bootstrapped |
| renquant-execution | main | `c41639840b2c` | bootstrapped |
| renquant-model | main | `5ef1c2d94f64` | active |
| renquant-orchestrator | main | `ade07dd797b0` | active |
| renquant-pipeline | main | `d32f7017ff05` | bootstrapped |
| renquant-strategy-104 | main | `1e840cd7fc26` | bootstrapped |

## Source fingerprints

- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json` — sha256:cdd26608f7dbc610
- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json` — sha256:dd13dee4f8906576
- `backtesting/renquant_104/artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json` — unknown (file missing)
- `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` — sha256:04d7a381cd6df847
- `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json` — sha256:d2b4d6ab197f0503
- `backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json` — sha256:bc3b8a8f803e4685
- `subrepos.lock.json` — sha256:f46f4b919075de4d

<!-- snapshot-machine-block
{
 "active_artifact": "artifacts/prod/panel-ltr.alpha158_fund.json",
 "active_kind": "xgb",
 "in_run_shadow_kinds": [
  "hf_patchtst"
 ],
 "lock_strategy_104_pin": "1e840cd7fc26c0ef277ffb0ce7d1375abe398b7c",
 "schema_version": 2,
 "shadow_e2e_kind": "hf_patchtst",
 "sources_sha256": {
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json": "sha256:cdd26608f7dbc610",
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json": "sha256:dd13dee4f8906576",
  "backtesting/renquant_104/artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json": null,
  "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json": "sha256:04d7a381cd6df847",
  "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json": "sha256:d2b4d6ab197f0503",
  "backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json": "sha256:bc3b8a8f803e4685",
  "subrepos.lock.json": "sha256:f46f4b919075de4d"
 },
 "strategy_104_pin": "1e840cd7fc26c0ef277ffb0ce7d1375abe398b7c"
}
-->
