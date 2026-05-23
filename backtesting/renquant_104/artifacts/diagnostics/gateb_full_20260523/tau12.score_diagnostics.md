# Trade-Level Score Diagnostics

- closed_only: `True`
- outcome_col: `pnl_pct`
- n_trades: `57`
- win_rate: `+43.86%`
- outcome_mean: `+1.88%`
- outcome_median: `-0.68%`

| score | n | Spearman vs outcome | top-bottom outcome spread | winner mean | loser mean | winner-loser score spread | expected direction |
|---|---:|---:|---:|---:|---:|---:|---|
| entry_rank_score | 57 | -0.1829 | -8.39% | +0.6148 | +0.6181 | -0.0033 | higher better |
| entry_mu | 57 | -0.1829 | -8.39% | +0.0304 | +0.0316 | -0.0012 | higher better |
| entry_sigma | 57 | -0.1369 | -3.96% | +0.2265 | +0.2372 | -0.0107 | lower risk better |
| entry_mu_over_sigma | 57 | -0.0963 | -2.89% | +0.1424 | +0.1410 | +0.0015 | higher better |
| entry_panel_score | 57 | -0.1829 | -8.39% | +0.1174 | +0.1292 | -0.0118 | higher better |
| entry_kelly_target_pct | 36 | -0.1171 | +2.00% | +0.0820 | +0.0835 | -0.0015 | higher better |

Interpretation:

- For alpha scores, positive Spearman and positive top-bottom spread are required.
- For sigma, negative Spearman is desirable because lower risk should realize better P&L.
- If winners and losers have nearly identical rank/μ, the execution slice is not using a discriminative alpha score.
