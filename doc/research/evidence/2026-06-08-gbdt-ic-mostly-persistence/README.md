# 2026-06-08 — The GBDT model's IC is ~61% cross-sectional persistence (placebo decomposition)

**Status**: §6.3/§6.4 no-run-path diagnostic (gate output + existing panel, zero
retrain). The 2026-06-08 weekly GBDT retrain reached a clean WF-gate verdict for
the first time in weeks (pipeline P0 fixed) — and the gate **correctly REJECTED**
on the §7.2 time-shift placebo. This doc explains *why*: the `fwd_60d_excess`
label is cross-sectionally autocorrelated (+0.049 at lag-120), so ~61% of the
model's apparent IC is momentum-persistence, not new forward alpha.
**Owner**: Claude.

---

## 1 · What triggered this

After the P0 retrain-pipeline repair (config parity #253, coverage-gap #249,
manifest-URI resolver #43), the weekly GBDT (`alpha158_fund`) WF gate ran
end-to-end to a real verdict:

```
WF config parity: PASS
§5.2 sanity battery (shuffle + time-shift placebo)
Sanity result: FAIL
  shuf_ic    = -0.0005   (need |·| < 0.005)            → PASS
  placebo_ic = +0.0359   at gate_shift=120d (2×horizon) → FAIL
  threshold  = +0.0295   (= 0.5 × aligned_real_ic 0.0590)
VERDICT: FAIL — production unchanged
```

The shuffle placebo passes (the signal is *real*, not data-mined noise), but the
**time-shift placebo is too high**. This doc attributes that.

## 2 · The mechanism — the label is cross-sectionally autocorrelated

`fwd_60d_excess` is a 60-trading-day forward excess return. Measured directly on
`alpha158_291_fundamental_dataset_rawlabel.parquet` (2,422 dates), the per-date
**cross-sectional** autocorrelation of the label against its own lagged value:

| lag | corr( label_t , label_{t−lag} ) |
|---|--:|
| 60 (1× horizon) | **+0.036** |
| **120 (2× horizon, = gate shift)** | **+0.049** |

Forward 60-day returns *persist* in the cross-section: a name that outperformed
over [t−120, t−60] tends to outperform over [t, t+60]. So when the gate shifts
the labels by 120 days, the shifted labels still correlate (+0.049) with the real
ones — and a model tuned to predict the real forward returns therefore scores on
the shifted labels too. That is the `placebo_ic = +0.036`.

## 3 · Decomposing the model's IC

| Component | IC |
|---|--:|
| Real (aligned) | **+0.059** |
| Time-shift placebo (persistence floor) | **+0.036** |
| **Genuine forward alpha** (real − placebo) | **≈ +0.023** |
| Shuffle placebo (cross-sectional noise) | −0.0005 |

**≈61% of the model's apparent IC is cross-sectional persistence** (≈ "winners
keep winning" = momentum); only **≈39% is new forward prediction**. The gate's
`placebo < 0.5 × real` rule therefore fails it — correctly, on its own terms.

## 4 · Interpretation — same truth as the BULL_CALM campaign, now pooled

The whole signal-recovery campaign (Kelly σ, cash overlay, QP, Track B, Track C)
kept bottoming out on "the panel signal is weak." This quantifies it at the
*pooled* level: the GBDT isn't adding much beyond momentum persistence. That is
*why* it cannot clear its own gate. The fix is more genuine forward alpha
(per-regime specialists / non-persistence features), not more
persistence-correlated alpha158/fund features.

## 5 · Two takeaways

1. **Model side (real):** to promote, the GBDT needs alpha *beyond* persistence.
   Adding more alpha158/fundamental features won't help if they are all
   persistence-correlated. Per-regime specialists (Track C, BULL_CALM specialist
   IC +0.0241 placebo-clean) are the live lever.

2. **Methodology side (RFC — §7.2 gate):** for a 60-day **overlapping-window**
   label with +0.049 lag-120 autocorrelation, the `placebo < 0.5 × real`
   threshold may be **structurally near-unreachable for any model** — the
   persistence floor is baked into the *target*, not the model. See the
   companion RFC for the proposed fix (label-autocorrelation-adjusted threshold,
   or a non-overlapping / shorter-horizon sanity label).

## 6 · Caveats

- The label autocorrelation is measured on the rawlabel panel (the gate's own
  sanity panel); the gate's `placebo_ic` is on the model's scored val rows. The
  +0.049 autocorr is the *target* property; the +0.036 placebo is the model
  *inheriting* it. They are consistent (placebo ≈ real × persistence-share) but
  not identical quantities.
- Single retrain, single seed. The decomposition is directional; the conclusion
  (placebo-driven-by-label-persistence) is robust because it follows from a
  target property (autocorrelation), not a model artifact.

## 7 · Reproduction

```bash
# label lag-120 cross-sectional autocorr
python - <<'PY'
import pandas as pd, numpy as np
df=pd.read_parquet('data/alpha158_291_fundamental_dataset_rawlabel.parquet',
                   columns=['ticker','date','fwd_60d_excess_raw']).dropna()
piv=df.assign(date=pd.to_datetime(df.date)).pivot_table(
    index='date',columns='ticker',values='fwd_60d_excess_raw').sort_index()
for lag in (60,120):
    c=[np.corrcoef(piv.iloc[i][m],piv.iloc[i-lag][m])[0,1]
       for i in range(lag,len(piv))
       for m in [piv.iloc[i].notna()&piv.iloc[i-lag].notna()] if m.sum()>=20]
    print(lag, round(float(np.mean(c)),3))
PY
# gate numbers: logs/weekly_wf_promote/2026-06-08.log (Sanity result line)
```
