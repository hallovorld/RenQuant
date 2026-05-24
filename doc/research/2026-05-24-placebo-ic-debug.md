# 2026-05-24 Placebo IC Debug

## Trigger

The WF sanity gate reported:

- `real_ic = +0.0385`
- `placebo_60_ic = +0.0460`

This looked like the 60-day placebo beat the real label, which would make the
model fundamentally untrustworthy.

## Confirmed Bug

The old gate compared different samples:

- `real_ic` used every manifest-covered validation row: 508 OOS dates.
- `placebo_60_ic` can only use rows that still have a `t+60` label: 448 OOS
  dates.

That made the headline `placebo > real` an apples-to-oranges statement.
The fix is to compute an `aligned_real_ic` on the exact same `(ticker, date)`
rows used by each shifted placebo.

Patched files:

- `scripts/run_wf_gate.py`
- `scripts/analyze_manifest_sanity_placebo.py`
- `backtesting/renquant_104/kernel/panel_pipeline/job_panel_scoring.py`

New metadata:

- Top-level: `sanity_placebo_aligned_real_ic`
- Per-shift: `aligned_real_ic`, `full_real_ic`,
  `abs_ratio_to_aligned_real`, `abs_ratio_to_full_real`
- Per-regime: `placebo_60_aligned_real_ic`

Regression coverage:

- `tests/test_manifest_sanity_placebo_analysis.py`
- `tests/test_wf_gate_regime_sanity_metadata.py`
- `tests/test_wf_gate_cli_contract.py`
- `tests/test_regime_model_admission.py`

Verification command:

```bash
.venv/bin/python -m pytest \
  tests/test_manifest_sanity_placebo_analysis.py \
  tests/test_wf_gate_regime_sanity_metadata.py \
  tests/test_wf_gate_cli_contract.py \
  tests/test_regime_model_admission.py \
  tests/test_promote_wf_gate.py -q
```

Result: `62 passed`.

## Corrected Numbers

Artifact:
`backtesting/renquant_104/artifacts/prod/panel-ltr.alpha158_fund.codex_featspace_20260523-211211.staging.json`

Manifest:
`backtesting/renquant_104/artifacts/sim/walkforward_manifest_172_featspace_20260523.scopefixed.covered.json`

Updated diagnostic output:
`backtesting/renquant_104/artifacts/diagnostics/sanity_placebo_codex_20260524_aligned_fix/panel-ltr_alpha158_fund_codex_featspace_20260523-211211_staging.json`

Correct same-sample comparison:

| Metric | Value |
|---|---:|
| Full real IC, 508 dates | `+0.0385` |
| 60d aligned real IC, 448 dates | `+0.0548` |
| 60d placebo IC, same 448 dates | `+0.0460` |
| Placebo / aligned real | `0.8405` |
| Label autocorr at 60d | `-0.0008` |

Conclusion: the original `placebo > real` headline was a gate reporting bug,
but the model still fails sanity because placebo is 84% of aligned real IC.
Promotion remains correctly blocked.

## Regime Split

| Regime | Full Real | Aligned Real | Placebo 60d | Ratio |
|---|---:|---:|---:|---:|
| BEAR | `+0.2565` | `+0.2563` | `+0.1828` | `0.71` |
| BULL_CALM | `+0.0152` | `+0.0323` | `+0.0312` | `0.97` |
| BULL_VOLATILE | `-0.0296` | `-0.0169` | `+0.0426` | `2.52` |
| CHOPPY | `+0.0315` | `+0.0210` | `-0.0055` | `0.26` |

The production issue is concentrated in BULL_CALM, which is also the regime
where most buys have been generated. BULL_CALM aligned real IC and placebo IC
are nearly identical, so current BULL_CALM buy evidence should be treated as
placebo-dominated.

## Feature/Style Diagnosis

The score is strongly exposed to volatility/range features (`STD60`, `STD30`,
`STD20`, `STD10`, `STD5`, `KLEN`, `MAX*`, `MIN*`, `WVMA*`, `CORD*`, `CORR60`).

Diagnostic residualization on the 448 aligned rows:

| Score Treatment | Real IC | Placebo IC |
|---|---:|---:|
| Raw score | `+0.0548` | `+0.0460` |
| Sector-neutral score | `+0.0620` | `+0.0398` |
| Vol/range-residual score | `-0.0010` | `+0.0107` |
| Sector + vol/range residual score | `+0.0087` | `+0.0056` |

Interpretation: most of the model's apparent IC is carried by slow-moving
volatility/range style exposure, not a clean stock-selection alpha. Sector
neutralization helps placebo separation, but stripping vol/range exposure also
strips nearly all real IC.

## Production Implication

Do not promote or trust this XGB artifact for unrestricted BULL_CALM buys.
The fixed gate will no longer misreport different samples, but it still fails
closed because placebo remains too large relative to same-sample real IC.

`RegimeModelAdmissionTask` now uses `placebo_60_aligned_real_ic` when present,
so runtime regime admission cannot pass BULL_CALM merely because full-sample
real IC looks weaker/stronger than the placebo-evaluable sample.

## Next Repair Path

1. Retrain a paired experiment with explicit volatility/range neutralization
   in either the label or the score, then run the same aligned placebo gate.
2. Test a feature-pruned XGB excluding the dominant vol/range block. This is
   expected to reduce raw IC; acceptance depends on placebo ratio and portfolio
   P&L, not raw IC alone.
3. Keep BULL_CALM buy admission fail-closed until aligned real IC beats placebo
   by the configured margin and trade-domain monotonicity remains positive.
4. Compare with PatchTST under the same aligned placebo diagnostic before any
   shadow promotion decision.

