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

Source fingerprint: 38f6a5eb973e8946446fe8db128fcbd92a5108f7d72d2786876cd063a5889b49 (sha256 over the sorted per-file source hashes below — deterministic; changes iff pinned/artifact CONTENT changes, never on a bare regeneration)

## Provenance

| | |
|---|---|
| Pinned config root | `.subrepo_runtime/repos/renquant-strategy-104/configs` |
| strategy-104 runtime checkout commit | 74a643e9a4495262df34b9ca47afc1e2c5e1b0da |
| subrepos.lock.json strategy-104 pin | 74a643e9a4495262df34b9ca47afc1e2c5e1b0da |

## Active scorer

| | |
|---|---|
| Scorer kind | `xgb` |
| Artifact | `artifacts/prod/panel-ltr.alpha158_fund.json` |
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

## Active calibrator

| | |
|---|---|
| Artifact | `artifacts/prod/panel-rank-calibration.json` |
| Artifact file fingerprint | unknown (file missing) |
| kind | unknown (field absent) |
| trained_date | unknown (field absent) |
| Method | unknown (field absent) |
| pool_ic (stamped) | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| Bound scorer content fingerprint | unknown (field absent) |
| Fit data window | unknown (field absent) |
| Metadata | unknown (file missing or unreadable) |

## In-run shadow scorers (readonly, same run)

| | |
|---|---|
| Name | `hf_patchtst_pt07_strict_seed44_previous_primary` |
| Scorer kind | `hf_patchtst` |
| Artifact | `../../artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt` |
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
| Artifact | `../../artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt` |
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
| Artifact file fingerprint | unknown (file missing) |
| kind | unknown (field absent) |
| trained_date | unknown (field absent) |
| Method | unknown (field absent) |
| pool_ic (stamped) | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| Bound scorer content fingerprint | unknown (field absent) |
| Fit data window | unknown (field absent) |
| Metadata | unknown (file missing or unreadable) |

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
| renquant-base-data | main | `0678958ec2f5` | active |
| renquant-common | main | `df620a6ecc35` | bootstrapped |
| renquant-execution | main | `43a8bdd36539` | bootstrapped |
| renquant-model | main | `775804dbb0bc` | active |
| renquant-orchestrator | main | `6a6a1bd371f6` | active |
| renquant-pipeline | main | `b6139e6a3ad7` | bootstrapped |
| renquant-strategy-104 | main | `74a643e9a449` | bootstrapped |

## Source fingerprints

- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json` — sha256:4af12d2ac3efab86
- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json` — sha256:d1b94ac4aa89a8bf
- `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json` — unknown (file missing)
- `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` — unknown (file missing)
- `backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json` — unknown (file missing)
- `backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json` — unknown (file missing)
- `subrepos.lock.json` — sha256:6b30925d6abaaf67

<!-- snapshot-machine-block
{
 "active_artifact": "artifacts/prod/panel-ltr.alpha158_fund.json",
 "active_kind": "xgb",
 "in_run_shadow_kinds": [
  "hf_patchtst"
 ],
 "lock_strategy_104_pin": "74a643e9a4495262df34b9ca47afc1e2c5e1b0da",
 "schema_version": 2,
 "shadow_e2e_kind": "hf_patchtst",
 "sources_sha256": {
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json": "sha256:4af12d2ac3efab86",
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json": "sha256:d1b94ac4aa89a8bf",
  "artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json": null,
  "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json": null,
  "backtesting/renquant_104/artifacts/prod/panel-rank-calibration.json": null,
  "backtesting/renquant_104/artifacts/shadow/panel-rank-calibration.hf_patchtst_seed44_trainfit_20230103_20240409.json": null,
  "subrepos.lock.json": "sha256:6b30925d6abaaf67"
 },
 "strategy_104_pin": "74a643e9a4495262df34b9ca47afc1e2c5e1b0da"
}
-->
