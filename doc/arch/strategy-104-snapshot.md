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

Source fingerprint: fadc8f8130756da6e87026abc5e28c8c0ebbe8f9295781f4ff82e09286a7253b (sha256 over the sorted per-file source hashes below — deterministic; changes iff pinned/artifact CONTENT changes, never on a bare regeneration. EXCLUDES the pooled calibrators: they are re-fit per promote (mutable live state) and are recorded below as runtime observations, NOT folded into this candidate-interface fingerprint)

## Provenance

| | |
|---|---|
| Pinned config root | `.subrepo_runtime/repos/renquant-strategy-104/configs` |
| strategy-104 runtime checkout commit | 799821245b0e17ac84f5b969c368034a4d1ccf32 |
| subrepos.lock.json strategy-104 pin | 799821245b0e17ac84f5b969c368034a4d1ccf32 |

### Source warnings

- **SCORER PROVENANCE UNRESOLVED: shadow scorer momentum_fast_v1_shadow artifact 'artifacts/momentum_fast/momentum_artifact_ledger.jsonl' did not resolve to a metadata-bearing file under the canonical resolver — no digest; the candidate pin does not provably restore a traceable scorer**
- **UMBRELLA WORKING-COPY DRIFT: backtesting/renquant_104/strategy_config.json declares kind='hf_patchtst' but the pinned config declares kind='blend' — the pinned config is what the daily run consumes; the working copy is stale**

## Active scorer

| | |
|---|---|
| Scorer kind | `blend` |
| Artifact | `artifacts/prod/panel-ltr.alpha158_fund.json` |
| Artifact metadata file fingerprint | sha256:f1b1c1322e3b66f7 |
| trained_date | 2026-08-31 |
| Binding data cutoff | effective_train_cutoff_date=2026-06-03 |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | 60 |
| label_col | fwd_60d_excess |
| Feature count | 172 |
| train_run_id | 8b9b8093 |
| oos_mean_ic (stamped) | +0.0534 |
| promotion_status | unknown (field absent) |
| config_fingerprint | sha256:f8fb2259b2bf1537 |
| WF gate (stamped) | passed=false; run_at=2026-08-31T14:26:19.641333; sanity_eval_end=2026-06-03 |

## Active calibrator

> Runtime observation, not a locked identity: the pooled calibrator is re-fit per promote, so the digest below reflects the live artifact as of this regeneration and is EXCLUDED from the candidate Source fingerprint above.

| | |
|---|---|
| Artifact | `artifacts/prod/panel-rank-calibration.json` |
| Artifact file fingerprint | sha256:43e859b4ef5f7641 |
| kind | global_panel_calibration |
| trained_date | 2026-08-31 |
| Method | platt |
| pool_ic (stamped) | +0.1250 |
| lookahead_days | 60 |
| Bound scorer content fingerprint | sha256:2d5a0288545c8a48c1aa34c83b951ad0af5591defb938783a0862a7eab25ca54 |
| Fit data window | unknown (field absent) |

## In-run shadow scorers (readonly, same run)

| | |
|---|---|
| Name | `topdecile_clf_blend_leg` |
| Scorer kind | `xgb` |
| Artifact | `artifacts/shadow/panel-clf.top-decile.fwd60.json` |
| Artifact metadata file fingerprint | sha256:1e644354e0981f47 |
| trained_date | 2026-07-28 |
| Binding data cutoff | effective_train_cutoff_date=2026-04-28 |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | 60 |
| label_col | fwd_60d_excess |
| Feature count | 172 |
| train_run_id | unknown (field absent) |
| oos_mean_ic (stamped) | unknown (field absent) |
| promotion_status | unknown (field absent) |
| config_fingerprint | sha256:1d8f167fed18cd8cb1e0760251fdd5398724e630462d92b41561d2e19973e41b |
| WF gate (stamped) | unknown (field absent) |

| | |
|---|---|
| Name | `momentum_residual_v0_shadow` |
| Scorer kind | `momentum_residual` |
| Artifact | `artifacts/momentum/momentum_artifact_ledger.jsonl` |
| Artifact metadata file fingerprint | sha256:9aa2d8c9571bad95 |
| trained_date | unknown (field absent) |
| Binding data cutoff | effective_train_cutoff_date=2026-07-02 |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | unknown (field absent) |
| label_col | unknown (field absent) |
| Feature count | unknown (field absent) |
| train_run_id | unknown (field absent) |
| oos_mean_ic (stamped) | unknown (field absent) |
| promotion_status | unknown (field absent) |
| config_fingerprint | unknown (field absent) |
| WF gate (stamped) | unknown (field absent) |

| | |
|---|---|
| Name | `momentum_fast_v1_shadow` |
| Scorer kind | `momentum_residual` |
| Artifact | `artifacts/momentum_fast/momentum_artifact_ledger.jsonl` |
| Artifact metadata file fingerprint | sha256:618fa993fa65f840 |
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
| Artifact metadata file fingerprint | sha256:447b7efa5fa3f64f |
| trained_date | 2026-05-22 |
| Binding data cutoff | effective_selection_cutoff_date=2026-02-10 |
| label_observation_cutoff | unknown (field absent) |
| lookahead_days | 60 |
| label_col | unknown (field absent) |
| Feature count | 172 |
| train_run_id | unknown (field absent) |
| oos_mean_ic (stamped) | unknown (field absent) |
| promotion_status | unknown (field absent) |
| config_fingerprint | sha256:f8fb2259b2bf1537 |
| WF gate (stamped) | unknown (field absent) |

### Shadow e2e calibrator

> Runtime observation (re-fit per promote), excluded from the candidate Source fingerprint — same as the active calibrator.

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
| Conviction gate μ floor | enabled=false; mu_floor=+0.0300; demean_cross_sectional=false |
| signal_gate_prefer_calibrated_mu | unknown (field absent) |
| Buy floor | mode=unknown (field absent); min=+0.2000 |
| panel_buy_top_n | 3 |
| Rotation | min_expected_advantage_pct=+0.0600; target_horizon_days=60 |
| Kelly sizing | enabled=false; fractional=+0.3000; max_concentration=+0.3000; min_edge=0; use_calibrator_mu=true |
| Position caps | max_concurrent_positions=10; max_position_pct=+0.1500; max_positions_per_sector=6 |
| model_staleness_days | 60 |
| QP | risk_aversion=3; turnover_max=+0.2000; no_trade_band_cap=+0.0500; mu_horizon_days=60; admission_min_rank_score=unknown (field absent) |
| WF gate relaxations (lock-declared) | benchmark_required=false; regime_required=false; sanity_regime_ic_required=false |

### Per-regime caps

| Regime | max_position_pct | qp_turnover_max | cash_reserve_pct | stop_loss_pct |
|---|---|---|---|---|
| BEAR | 0 | unknown (field absent) | 1 | +0.0500 |
| BULL_CALM | +0.3000 | +0.1500 | 0 | +0.1500 |
| BULL_VOLATILE | +0.2000 | unknown (field absent) | +0.2000 | +0.0500 |
| CHOPPY | +0.1500 | unknown (field absent) | +0.3000 | +0.0800 |

## Subrepo pins (subrepos.lock.json)

| Subrepo | Branch | Commit | Status |
|---|---|---|---|
| renquant-artifacts | main | `c09d66f8dd09` | bootstrapped |
| renquant-backtesting | main | `31b6d6e41ed7` | bootstrapped |
| renquant-base-data | main | `f8514066b53f` | active |
| renquant-common | main | `ef7726dd6c90` | bootstrapped |
| renquant-execution | main | `91c7bf8873fd` | bootstrapped |
| renquant-model | main | `36085810fffe` | active |
| renquant-orchestrator | main | `0474af18edf9` | active |
| renquant-pipeline | main | `faf1416a342b` | bootstrapped |
| renquant-strategy-104 | main | `799821245b0e` | bootstrapped |

## Source fingerprints

- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json` — sha256:c1a8a9ddc15d746e
- `.subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json` — sha256:ae2cd4431755c3b9
- `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json` — sha256:447b7efa5fa3f64f
- `backtesting/renquant_104/artifacts/momentum/momentum_artifact_ledger.jsonl` — sha256:9aa2d8c9571bad95
- `backtesting/renquant_104/artifacts/momentum_fast/momentum_artifact_ledger.jsonl` — sha256:618fa993fa65f840
- `backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json` — sha256:f1b1c1322e3b66f7
- `backtesting/renquant_104/artifacts/shadow/panel-clf.top-decile.fwd60.json` — sha256:1e644354e0981f47
- `subrepos.lock.json` — sha256:305eb3162df0addc

<!-- snapshot-machine-block
{
 "active_artifact": "artifacts/prod/panel-ltr.alpha158_fund.json",
 "active_kind": "blend",
 "in_run_shadow_kinds": [
  "xgb",
  "momentum_residual",
  "momentum_residual"
 ],
 "lock_strategy_104_pin": "799821245b0e17ac84f5b969c368034a4d1ccf32",
 "schema_version": 2,
 "shadow_e2e_kind": "hf_patchtst",
 "sources_sha256": {
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.json": "sha256:c1a8a9ddc15d746e",
  ".subrepo_runtime/repos/renquant-strategy-104/configs/strategy_config.shadow.json": "sha256:ae2cd4431755c3b9",
  "artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt.metadata.json": "sha256:447b7efa5fa3f64f",
  "backtesting/renquant_104/artifacts/momentum/momentum_artifact_ledger.jsonl": "sha256:9aa2d8c9571bad95",
  "backtesting/renquant_104/artifacts/momentum_fast/momentum_artifact_ledger.jsonl": "sha256:618fa993fa65f840",
  "backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.json": "sha256:f1b1c1322e3b66f7",
  "backtesting/renquant_104/artifacts/shadow/panel-clf.top-decile.fwd60.json": "sha256:1e644354e0981f47",
  "subrepos.lock.json": "sha256:305eb3162df0addc"
 },
 "strategy_104_pin": "799821245b0e17ac84f5b969c368034a4d1ccf32"
}
-->
