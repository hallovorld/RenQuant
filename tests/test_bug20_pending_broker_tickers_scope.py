"""Regression test for Bug #20 (PENDING-BROKER-SCOPE, 2026-04-26 round-7).

E2E log on 2026-04-26 16:31 surfaced:
    [WARNING] adapters.runner: ticker_daily_state write failed:
              name 'pending_broker_tickers' is not defined

Root cause: `pending_broker_tickers` is a LOCAL variable inside
`make_context()` (defined ~line 170). The TDS-write code in `commit()`
(lines ~963 / ~989) referenced it as a bare name → NameError → caught
by the outer try/except → silent telemetry loss on EVERY live bar.

Fix: read `pending_broker_tickers` from `ctx.pending_broker_tickers`
(set by make_context at line 478) instead of bare scope. Defensive
default to `set()` so a sell-only path that didn't run BROKER-PRECHECK
still writes the row (with pending_at_broker=0).

Test strategy: string-level assertions on adapters/runner.py source.
Same pattern as test_runner_state_fixes.py — catches refactor regressions
without needing a full RunnerAdapter mock surface.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER_PATH = REPO_ROOT / "backtesting/renquant_104/adapters/runner.py"
RUNNER_SOURCE = RUNNER_PATH.read_text()


class TestPendingBrokerScopeFix:
    def test_audit_tag_present(self):
        """The fix line documents the bug for future readers."""
        assert "Bug #20 fix" in RUNNER_SOURCE, (
            "Audit tag for round-7 PENDING-BROKER-SCOPE must be present "
            "in source so future readers can find the rationale."
        )

    def test_tds_write_reads_from_ctx_not_bare_scope(self):
        """The TDS-write block must read pending_broker_tickers from ctx,
        not from make_context()'s local scope.
        """
        # Locate the TDS write block by its anchor comment.
        anchor = "ticker_daily_state — every watchlist ticker"
        assert anchor in RUNNER_SOURCE, (
            f"TDS-write block anchor missing — file may have been "
            f"refactored. Check that the anchor '{anchor}' still exists."
        )
        idx_anchor = RUNNER_SOURCE.find(anchor)
        # Block runs ~80 lines. Read enough downstream context to cover
        # the for-loop body where the variable is referenced.
        block = RUNNER_SOURCE[idx_anchor:idx_anchor + 4000]

        # The fix must rebind pending_broker_tickers from ctx INSIDE the
        # commit() try/except, BEFORE the for-loop uses it.
        assert 'getattr(ctx, "pending_broker_tickers"' in block, (
            "commit() must read pending_broker_tickers from ctx (set by "
            "make_context) — bare-name reference will NameError."
        )

    def test_defensive_default_to_empty_set(self):
        """When ctx doesn't have the attribute (sell-only path skipped
        BROKER-PRECHECK), the rebind must default to set() — NOT None,
        which would crash on `tk in pending_broker_tickers`.
        """
        # Look for the rebind statement specifically in the commit() block.
        anchor = "Bug #20 fix"
        assert anchor in RUNNER_SOURCE
        idx = RUNNER_SOURCE.find(anchor)
        rebind_block = RUNNER_SOURCE[idx:idx + 1200]
        # Both the getattr fallback to None AND the `or set()` must be
        # present so a missing attribute resolves to a usable empty set.
        assert "or set()" in rebind_block, (
            "rebind must default to set() when attribute is None or missing"
        )

    def test_no_bare_pending_broker_tickers_in_commit_loop(self):
        """The for-loop body must reference the LOCAL rebind, not the
        bare name from make_context()'s scope. Audit guard against the
        original bug pattern.
        """
        # Find commit() function start.
        commit_idx = RUNNER_SOURCE.find("    def commit(")
        assert commit_idx > 0
        commit_body = RUNNER_SOURCE[commit_idx:]
        # Within commit(), there must be a local rebind BEFORE any usage.
        rebind_idx = commit_body.find("pending_broker_tickers: set = set(")
        first_use_idx = commit_body.find("tk in pending_broker_tickers")
        assert rebind_idx > 0, "commit() must rebind pending_broker_tickers locally"
        assert first_use_idx > rebind_idx, (
            f"rebind (idx={rebind_idx}) must come BEFORE first use "
            f"(idx={first_use_idx}) — otherwise the NameError returns."
        )
