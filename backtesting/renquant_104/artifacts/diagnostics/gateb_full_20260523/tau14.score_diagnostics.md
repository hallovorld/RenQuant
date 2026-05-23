# Trade-Level Score Diagnostics

- closed_only: `True`
- outcome_col: `pnl_pct`
- n_trades: `58`
- win_rate: `+41.38%`
- outcome_mean: `+1.68%`
- outcome_median: `-1.98%`

| score | n | Spearman vs outcome | top-bottom outcome spread | winner mean | loser mean | winner-loser score spread | expected direction |
|---|---:|---:|---:|---:|---:|---:|---|
| entry_rank_score | 58 | -0.1705 | -4.56% | +0.6121 | +0.6275 | -0.0154 | higher better |
| entry_mu | 58 | -0.1705 | -4.56% | +0.0294 | +0.0350 | -0.0055 | higher better |
| entry_sigma | 58 | -0.1110 | -0.29% | +0.2084 | +0.2269 | -0.0185 | lower risk better |
| entry_mu_over_sigma | 58 | +0.0037 | +1.84% | +0.1447 | +0.1554 | -0.0107 | higher better |
| entry_panel_score | 58 | -0.1705 | -4.56% | +0.1081 | +0.1629 | -0.0547 | higher better |
| entry_kelly_target_pct | 41 | -0.1703 | -1.03% | +0.0910 | +0.0945 | -0.0035 | higher better |

Interpretation:

- For alpha scores, positive Spearman and positive top-bottom spread are required.
- For sigma, negative Spearman is desirable because lower risk should realize better P&L.
- If winners and losers have nearly identical rank/μ, the execution slice is not using a discriminative alpha score.
