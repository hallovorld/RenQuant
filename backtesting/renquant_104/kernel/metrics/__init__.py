"""kernel.metrics — performance-significance utilities (DSR, PBO, perf triple).

Per CLAUDE.md §5.13.4: every Sharpe / APY / IC claim ships with
(mean ± std, DSR, PBO).  Single-number claims are forbidden.

References:
  - Bailey & López de Prado (2014), "The Deflated Sharpe Ratio: Correcting for
    Selection Bias, Backtest Overfitting and Non-Normality", SSRN 2460551.
  - Bailey, Borwein, López de Prado, Zhu (2017), "The Probability of Backtest
    Overfitting", Journal of Computational Finance.
"""

from .deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
    sharpe_std_error,
)
from .pbo import probability_of_backtest_overfitting
from .perf_summary import compute_perf_triple

__all__ = [
    "deflated_sharpe_ratio",
    "expected_max_sharpe",
    "sharpe_std_error",
    "probability_of_backtest_overfitting",
    "compute_perf_triple",
]
