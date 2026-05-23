# PatchTST / PatchTXT Status — 2026-05-23

## Scope

This note reconciles the long-running PatchTST/PatchTXT experiments that were
started before the 2026-05-23 decision-tree work. It references artifacts but
does not trust old claims without checking the files.

## What Finished

The 5-cut × 5-seed HF PatchTST experiment families have finished:

| artifact | rows | mean best-val IC | std | min | max |
|---|---:|---:|---:|---:|---:|
| `artifacts/hf_trainer_5cut_5seed_pt07_clean/raw_results.json` | 25 | +0.0467 | 0.0816 | -0.0607 | +0.1878 |
| `artifacts/hf_film_5cut_5seed_pt07_clean/raw_results.json` | 25 | +0.0477 | 0.0767 | -0.0502 | +0.1718 |
| `artifacts/hf_cross_stock_5cut_5seed_pt07/raw_results.json` | 25 | +0.0507 | 0.0878 | -0.0594 | +0.2035 |

Regime/cut split is the important part:

| family | cut1_covid | cut2_fed | cut3_inflpk | cut4_svb | cut5_unwind |
|---|---:|---:|---:|---:|---:|
| trainer clean | +0.1220 | -0.0430 | +0.0102 | +0.1594 | -0.0152 |
| FiLM clean | +0.1142 | -0.0342 | +0.0211 | +0.1561 | -0.0186 |
| cross-stock | +0.1128 | -0.0423 | +0.0120 | +0.1865 | -0.0154 |

Conclusion: PatchTST has real positive signal in some stress cuts, but it is
not stable across all regimes/cuts. The pooled positive mean is not enough for
promotion under CLAUDE.md's regime-conditional rule.

## Shadow Config

Current production shadow model in `strategy_config.json` /
`strategy_config.golden.json`:

- name: `hf_patchtst_pt07_strict_seed44`
- artifact:
  `artifacts/patchtst_shadow/pt07_strict_trainfit_embargo60_20260522/seed_44/hf_patchtst_all_seed44_model.pt`
- contract: tail-val 10%, 60 business-day embargo, train-only label winsor,
  `seq_len=24`, `lr=1e-4`, `weight_decay=0.3`, warmup 0.1, distributional head.
- strict min-regime IC: `+0.0307`
  - BULL_VOLATILE `+0.0524`
  - BEAR `+0.1916`
  - CHOPPY `+0.0307`

This is not the same as the older canonical 5-seed MPS artifact set:

| seed | old canonical best-val IC |
|---:|---:|
| 42 | +0.0569 |
| 43 | +0.0615 |
| 44 | +0.0657 |
| 45 | +0.0551 |
| 46 | +0.0834 |

The old canonical numbers are higher, but the strict seed44 artifact is safer:
it carries a stamped training contract and an embargoed validation design. The
right next step is not to revert to the higher old IC. It is to run the strict
contract for all 5 seeds and evaluate regime-stratified IC.

## Raw APY / Sharpe Caveat

`artifacts/diagnostics/raw_signal_baseline_patchtst_seed44_20260522.json` is a
static raw-signal diagnostic over only 5 rebalance dates. It reports pooled
actual after-tax APY `+0.35%` and Sharpe `+0.08`, while shuffle has after-tax
APY `+11.9%` and Sharpe `+1.17`.

That result is not promotable and not a production APY estimate. It is a
warning that static PatchTST APY/Sharpe is currently underpowered and
shuffle-sensitive. PatchTST should stay shadow until strict 5-seed,
regime-stratified IC plus true walk-forward trade simulation pass.

## Current Verdict

- PatchTST is more promising than the clean XGB trade slice on raw IC in some
  cuts, especially stress regimes.
- PatchTST is not yet a trustworthy primary model because cut2_fed and
  cut5_unwind are negative across all tested HF families.
- Current shadow is using the safer strict-contract seed44, not the older
  higher-IC canonical seed44.
- No live promotion. Next scientific step: strict-contract 5-seed rerun, then
  regime-conditional shadow comparison and trade-level monotonicity gate.
