"""Regression guards for scripts/analyze_decision_factors.py."""
from __future__ import annotations

import pandas as pd

from scripts.analyze_decision_factors import _print_block_outcomes


class TestBlockReasonAttribution:
    def test_selected_rows_override_stale_blocked_by(self, capsys):
        """AUDIT REGRESSION GUARD: selected rows are not blocked rows."""
        df = pd.DataFrame(
            [
                {"selected": 1, "blocked_by": "kelly_zero:mu_none", "fwd": 0.02},
                {"selected": 0, "blocked_by": "tier", "fwd": -0.01},
            ]
        )

        _print_block_outcomes(df)

        out = capsys.readouterr().out
        assert "(selected)" in out
        assert "tier" in out
        assert "kelly_zero:mu_none" not in out
