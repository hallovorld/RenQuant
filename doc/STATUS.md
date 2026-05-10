# RenQuant — Status

**Last updated:** 2026-05-09 EOD (post BUG #6/#7/#5 + cost-aware wash-sale + NGB on/off A/B revert + WF 3-cut)

---

## Production state

| | |
|---|---|
| Strategy        | `renquant_104` panel-LTR cross-sectional ranking |
| Active model    | **XGB rank:pairwise on 169 features** (alpha158 + 5 fund + 3 PEAD + 3 SUE), `panel-ltr.alpha158_fund.json` fingerprint `4f1e25989d475225` |
| NGBoost head    | DISABLED (27-mo A/B: -3.78 APY pp / -0.14 Sharpe; persistence ratio 63% per E55). Artifact retained for future re-enable |
| Watchlist       | 103 live / 292 train panel / wl162 quality-first selected (panel build pending) |
| Portfolio QP    | cvxpy + CLARABEL (cvxportfolio idiom). σ-band cap 5% per BUG #7 fix |
| Wash-sale       | Cost-aware per IRC §1091 (gain → no cost; loss → NPV deferred-tax cost) |
| Calibrator      | Pooled `panel-rank-calibration.json`, n_unique_prob_y=79, pool_ic=+0.094 |
| Broker          | Alpaca live, equity ~$10,580 (deposits 2× $5k 2026-04-06 + 04-16) |
| Cron            | daily104 / open104 / preclose104 / intraday104 / Sunday retrain ENABLED |
| Walk-forward gate | promote() requires `wf_gate_metadata.passed=True` (commit 5b8c891) |
| Test count      | ~14k passing (60 acceptance + 14 WF gate + 27 wash-sale + 14 e2e smoke + invariants) |

---

## Performance baseline — TBD (audit 2026-05-09 invalidated prior claims)

> **All previously published APY/Sharpe numbers are single-measurement and not reproducible.** Per `doc/AUDIT_2026-05-09.md`, the morning's "27-mo APY +6.77% / Sharpe +0.40 honest baseline" failed reproduction the same evening (re-run with same config + same artifact produced +1.97% / +0.20). Root cause not yet isolated; could be sim non-determinism, config drift, or one of the bugs the audit found.

**The work to establish a trustworthy baseline:**
1. Fix all RED bugs from audit doc → done for: BUG #1/#2/#6/#7, fund parity, dashboard DB path, panel-ltr.json sync, sim/QP/selection cost-aware wash-sale (5 commits today).
2. Fix remaining YELLOW bugs (BUG #5 parquet regen, WF gate cron schedule).
3. Establish A/A multi-seed protocol per CLAUDE.md §5.2 to characterize σ_APY / σ_Sharpe.
4. Only THEN report any number, with mean ± std from ≥5 seeds.

Per user mandate 2026-05-09: "no number trustworthy until bugs fixed". Walk-forward, single-seed, and cross-cut numbers all currently unreliable.

---

## Today's commits (2026-05-09, 14 total)

```
42e3adb  fix(features): BUG #5 asset_growth periods=4d → 252d (Cooper-Gulen-Schill 2008)
4c16ce0  exp(track2): wl200 selection (→wl162) + insider feature test (REJECT)
35a08c2  fix(tests): WF gate compat + options-IV scaffold (P0 #2 start)
5b8c891  feat(promote): walk-forward gate enforcement (roadmap P0 #1)
00506bf  docs(roadmap): rewrite by ROI w/ paper+open-source cite per item
d681fe8  docs: E55 NGBoost-proper SIGNIFICANT 5-seed lift but 63% persistence
fb74bb4  revert(prod): disable NGB after 27-month A/B losses by 3.78 APY pts
36d79ef  docs+infra: E54 production deploy chain + sim adapter polymorphic loader
ebbc158  fix(qp): BUG #7 cap σ-derived no-trade band at 5% — unblock high-σ exits
3549c51  feat(filter): cost-aware wash-sale per IRC §1091
ac468e7  fix(prod): BUG #6 prod μ̂ collapse + 4 universal model contracts
fa7d005  exp(track2): Phase C neural QHead E52 → 1.7σ + 42% persistence audit
022ade8  exp(track2): NGB raw-label promote + Phase A/B experiments E51 → null
507cef6  fix(pipeline): BUG #1 runtime fund parity + BUG #2 SEC date guard
```

---

## Track 2 NULL experiments (today, all consistent — panel signal-bound)

| Experiment | Result | Notes |
|---|---|---|
| Phase A NGB QHead variants (X-std, y-demean, +K) | NULL | within 5-seed σ noise |
| Phase C neural QHead MLP | 1.7σ borderline | 42% persistence (E52) |
| Phase D2 NGBoost-proper | sig +60bp BUT 63% persistence | pure-alpha worse than baseline |
| Per-sector excess label | REJECT | persistence 89%, pure-alpha drops to +0.005 |
| Vol-adjusted label | REJECT | -5bp on raw_y eval |
| LightGBM + sector categorical | REJECT | LGB structurally weaker than XGB |
| Multi-horizon ensemble (E42 retest) | REJECT | shorter horizons dilute H=60 signal |
| Triple-barrier label (E25 retest) | REJECT | val_ic negative on TB label |
| Explicit momentum-rank features | NULL | persistence 63→69% (autocorrelated) |
| Insider EDGAR features (E22 retest) | REJECT | -10 to -21bp; 8% coverage, no opportunistic split |

**Pattern:** all variants of "tweak architecture/labels/features on the existing 169 panel" return NULL. Real lift requires NEW DATA SOURCES (P0 #2 options-IV, P0 #3 news, P0 #6 insider full backfill).

---

## Open

- **P0 #1 walk-forward gate**: ✅ shipped (commit 5b8c891)
- **P0 #4 wl200 expansion**: wl162 selected; panel build pending (~1h compute)
- **BUG #5 asset_growth fix**: code fixed; `sec_fundamentals_daily.parquet` regen pending (~1h)
- **P0 #2 options-IV**: awaits Polygon.io paid ($30/mo) OR 3-mo daily polling
- **P0 #3 news sentiment FinBERT**: awaits Alpaca News backfill or paid news source

---

## Reference

- Roadmap (ROI-ranked): [`doc/roadmap.md`](roadmap.md)
- Failed experiments (every NO-GO with reason): [`doc/research/failed-experiments-log.md`](research/failed-experiments-log.md)
- Strategy spec: [`doc/arch/strategy-104.md`](arch/strategy-104.md)
- Architecture overview: [`doc/arch/overview.md`](arch/overview.md)
- IC eval methodology: [`doc/research/ic-evaluation-methodology.md`](research/ic-evaluation-methodology.md)
- Operations: [`doc/ops/usage.md`](ops/usage.md), [`doc/ops/golden-config.md`](ops/golden-config.md)
- CLAUDE.md @ repo root for development rules + working principles
