# Trade-Level Score Diagnostics

- closed_only: `True`
- outcome_col: `pnl_pct`
- n_trades: `56`
- win_rate: `+32.14%`
- outcome_mean: `-0.67%`
- outcome_median: `-3.14%`

| score | n | Spearman vs outcome | top-bottom outcome spread | winner mean | loser mean | winner-loser score spread | expected direction |
|---|---:|---:|---:|---:|---:|---:|---|
| entry_rank_score | 56 | +0.0152 | -0.30% | +0.6130 | +0.6131 | -0.0000 | higher better |
| entry_mu | 56 | +0.0152 | -0.30% | +0.0297 | +0.0297 | -0.0000 | higher better |
| entry_sigma | 56 | -0.4043 | -13.43% | +0.1950 | +0.2542 | -0.0592 | lower risk better |
| entry_mu_over_sigma | 56 | +0.2646 | +9.56% | +0.1597 | +0.1241 | +0.0356 | higher better |
| entry_panel_score | 56 | +0.0152 | -0.30% | +0.1108 | +0.1111 | -0.0003 | higher better |
| entry_kelly_target_pct | 36 | -0.0651 | +2.01% | +0.0841 | +0.0858 | -0.0017 | higher better |

Interpretation:

- For alpha scores, positive Spearman and positive top-bottom spread are required.
- For sigma, negative Spearman is desirable because lower risk should realize better P&L.
- If winners and losers have nearly identical rank/μ, the execution slice is not using a discriminative alpha score.

