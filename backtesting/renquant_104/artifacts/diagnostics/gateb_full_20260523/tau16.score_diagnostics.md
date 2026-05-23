# Trade-Level Score Diagnostics

- closed_only: `True`
- outcome_col: `pnl_pct`
- n_trades: `50`
- win_rate: `+50.00%`
- outcome_mean: `+1.20%`
- outcome_median: `-0.24%`

| score | n | Spearman vs outcome | top-bottom outcome spread | winner mean | loser mean | winner-loser score spread | expected direction |
|---|---:|---:|---:|---:|---:|---:|---|
| entry_rank_score | 50 | -0.4682 | -10.12% | +0.6173 | +0.6293 | -0.0120 | higher better |
| entry_mu | 50 | -0.4682 | -10.12% | +0.0313 | +0.0356 | -0.0043 | higher better |
| entry_sigma | 50 | -0.2458 | -3.46% | +0.2009 | +0.2191 | -0.0182 | lower risk better |
| entry_mu_over_sigma | 50 | -0.1480 | -2.38% | +0.1589 | +0.1642 | -0.0052 | higher better |
| entry_panel_score | 50 | -0.4682 | -10.12% | +0.1258 | +0.1693 | -0.0435 | higher better |
| entry_kelly_target_pct | 35 | -0.2821 | -9.46% | +0.0831 | +0.1038 | -0.0207 | higher better |

Interpretation:

- For alpha scores, positive Spearman and positive top-bottom spread are required.
- For sigma, negative Spearman is desirable because lower risk should realize better P&L.
- If winners and losers have nearly identical rank/μ, the execution slice is not using a discriminative alpha score.
