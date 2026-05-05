"""Regression test for Bug #22 (RS-SCORE-KEYERROR, 2026-04-26 round-7).

The 17:18 live e2e crashed at adapters/runner.py:712 with
    KeyError: 'rs_score'
AFTER orders had been submitted to Alpaca. Trades placed (live money)
but commit() crashed before the trade log + state save → live state
inconsistent.

Root cause: order dicts from JointPortfolioQPTask (task_joint_qp.py
line 240) omit rs_score / regime fields. The runner's BUY-emit path
(line 711-713) reads them as bare keys → KeyError.

CLAUDE.md says rs_score is retired from ranking math but still
populated on CandidateResult for logs. The QP path skips rs_score
entirely. The runner needs defensive .get() with sensible defaults.

Test strategy: same string-level contract pattern as test_runner_state_fixes
+ test_bug20_pending_broker_tickers_scope.py. Asserts the fix line
is in source so a future refactor that re-introduces bare-key access
trips the test.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "backtesting/renquant_104/adapters/runner.py"
RUNNER_SOURCE = RUNNER_PATH.read_text()


class TestBug22RsScoreKeyError:
    def test_audit_tag_present(self):
        assert "Bug #22 fix" in RUNNER_SOURCE

    def test_rank_score_uses_get_with_default(self):
        """No more bare order['rank_score'] in BUY-emit; must use .get()
        because the QP order producer omits it."""
        assert 'order.get("rank_score"' in RUNNER_SOURCE, (
            "BUY-emit must read rank_score with .get() to tolerate orders "
            "from QP path that omit the key."
        )

    def test_rs_score_uses_get_with_default(self):
        assert 'order.get("rs_score"' in RUNNER_SOURCE

    def test_regime_uses_get_with_fallback_to_ctx(self):
        """regime fallback to ctx.regime so the trade log still records
        the real regime even when the QP order omits the field."""
        assert 'order.get("regime"' in RUNNER_SOURCE
        assert 'ctx.regime' in RUNNER_SOURCE   # used as fallback

    def test_no_bare_order_keyaccess_in_buy_log(self):
        """The BUY-emit block must NOT have any `order["rank_score"]` or
        `order["rs_score"]` bare-key reads. (Bracket form would crash
        on QP-produced orders.)"""
        # Find the _log_trade BUY block (search anchors on the dict literal).
        anchor = '"action":     "BUY"'
        assert anchor in RUNNER_SOURCE, "BUY-emit anchor missing"
        idx = RUNNER_SOURCE.find(anchor)
        block = RUNNER_SOURCE[idx:idx + 600]
        # No bare bracket-key access on rank_score / rs_score / regime
        # inside this block. (The .get(...) form is fine — checks above.)
        for forbidden_key in ("rank_score", "rs_score", "regime"):
            bare_form = f'order["{forbidden_key}"]'
            assert bare_form not in block, (
                f"Bare key access {bare_form} re-introduced — would "
                f"KeyError on QP-emitted orders."
            )


class TestQPOrderShape:
    """Confirm the QP order producer omits the keys (so our defensive
    .get() is actually load-bearing). If the QP producer adds these
    fields later, this test points at the right spot."""

    # 2026-05-04: legacy task_joint_qp.py is now a back-compat shim;
    # the order-emit code lives in tasks.py (EmitOrdersFromQPSolutionTask
    # via the _emit_qp_buy helper).
    QP_PATH = REPO_ROOT / "backtesting/renquant_104/kernel/portfolio_qp/tasks.py"
    QP_SOURCE = QP_PATH.read_text()

    def test_qp_order_dict_does_not_set_rs_score(self):
        """QP producer's BUY ctx.orders.append({...}) doesn't include
        rs_score. If it ever does, the runner's defensive .get() default
        of 0.0 still works — but the test reminds us to remove this
        comment block too."""
        anchor = "ctx.orders.append({"
        assert anchor in self.QP_SOURCE
        idx = self.QP_SOURCE.find(anchor)
        block = self.QP_SOURCE[idx:idx + 600]
        assert '"rs_score"' not in block, (
            "QP order producer now sets rs_score — update runner.py "
            "comment block to reflect that the .get() default is no "
            "longer load-bearing for this producer."
        )
