#!/usr/bin/env python3
"""Backward-compatible alias for the orchestrator daily contract smoke."""
from __future__ import annotations

from subrepo_daily_contract import main

if __name__ == "__main__":
    raise SystemExit(main())
