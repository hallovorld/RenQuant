# Rigorous verdicts: 5 treatments vs baseline `sim_2026-05-16_re_kelly_t1_035`

- bootstrap iterations: 5000 (stationary block, block ≈ N^(1/3))
- DSR n_trials: 5 (= number of knobs tested in this batch)
- CSCV splits: 1000 (or all C(N, N/2) if fewer)
- t-stat: Newey-West HAC (maxlags = N^(1/4))

| treatment | N windows | mean Δ APY | 95% CI (block bootstrap) | NW t | NW p | deflated t | DSR | VERDICT |
|---|---|---|---|---|---|---|---|---|
| sim_2026-05-16_re_stop007 | 16 | -0.0023 | [-0.0496, +0.0448] | -0.10 | 0.919 | -0.29 | 0.39 | **NULL** |
| sim_2026-05-16_re_sdl_n2 | 16 | -0.0124 | [-0.0822, +0.0588] | -0.34 | 0.736 | -0.55 | 0.29 | **NULL** |
| sim_2026-05-16_re_trail015 | 16 | +0.0007 | [-0.0068, +0.0087] | +0.16 | 0.872 | -0.06 | 0.48 | **NULL** |
| sim_2026-05-16_re_cvar025 | 16 | -0.0057 | [-0.0459, +0.0312] | -0.28 | 0.782 | -0.51 | 0.31 | **NULL** |
| sim_2026-05-16_re_cvar050 | 16 | -0.0015 | [-0.0378, +0.0331] | -0.08 | 0.932 | -0.28 | 0.39 | **NULL** |

## Batch-level overfit probability

- **PBO (CSCV) = 0.930**  (93.0%)
- Bailey-López de Prado threshold: PBO > 0.5 ⇒ batch is overfit; no strategy in this batch should be promoted as if it were a single-test discovery.

## How to read these verdicts

- **REAL_EFFECT** — 95% block-bootstrap CI on the per-window paired Δ APY excludes 0, AND deflated t-stat (selection-bias corrected for n_trials peers) is positive. This is the only verdict that warrants further confirmation work (multi-seed, larger panel).
- **SUSPECT_MULTI_COMP** — CI excludes 0 but the deflated t-stat is ≤0. Effect is real in a single-test sense but disappears under multiple-comparison correction. Re-run as a single-hypothesis test (not as part of a sweep) before believing.
- **NULL** — bootstrap CI contains 0. Cannot reject the null that the knob has no effect; pooled mean is within noise. Don't deploy. Don't run a follow-up; the next compute is better spent on a different hypothesis.

```json
{
  "baseline": "data/logs/sim_2026-05-16_re_kelly_t1_035",
  "n_trials": 5,
  "per_strategy": [
    {
      "treatment": "sim_2026-05-16_re_stop007",
      "N": 16,
      "mean_delta_apy": -0.002306669070467296,
      "ci_lo": -0.049609667975975734,
      "ci_hi": 0.044763937114616834,
      "nw_t": -0.10208947425938923,
      "nw_p": 0.9186856556860556,
      "deflated_t": -0.2895340094267735,
      "dsr": 0.3860863790232504,
      "verdict": "NULL"
    },
    {
      "treatment": "sim_2026-05-16_re_sdl_n2",
      "N": 16,
      "mean_delta_apy": -0.01240299498591075,
      "ci_lo": -0.08218802366557904,
      "ci_hi": 0.05882434023303539,
      "nw_t": -0.33767973400524165,
      "nw_p": 0.7356045485122376,
      "deflated_t": -0.5505677180562791,
      "dsr": 0.2909650217731833,
      "verdict": "NULL"
    },
    {
      "treatment": "sim_2026-05-16_re_trail015",
      "N": 16,
      "mean_delta_apy": 0.0006967953087185091,
      "ci_lo": -0.006774178195127357,
      "ci_hi": 0.008664648339734085,
      "nw_t": 0.1616350832193382,
      "nw_p": 0.8715932281599394,
      "deflated_t": -0.062350240705544825,
      "dsr": 0.4751419599878469,
      "verdict": "NULL"
    },
    {
      "treatment": "sim_2026-05-16_re_cvar025",
      "N": 16,
      "mean_delta_apy": -0.005655133275279958,
      "ci_lo": -0.04586017190016643,
      "ci_hi": 0.031203852978844482,
      "nw_t": -0.27710698752166485,
      "nw_p": 0.7816979565814228,
      "deflated_t": -0.5092802172114528,
      "dsr": 0.30527791122443104,
      "verdict": "NULL"
    },
    {
      "treatment": "sim_2026-05-16_re_cvar050",
      "N": 16,
      "mean_delta_apy": -0.0015001533047689572,
      "ci_lo": -0.037783790135005955,
      "ci_hi": 0.0331020672429131,
      "nw_t": -0.08490187391500677,
      "nw_p": 0.9323394019999307,
      "deflated_t": -0.28468245755042904,
      "dsr": 0.3879437158094839,
      "verdict": "NULL"
    }
  ],
  "pbo": 0.93
}
```