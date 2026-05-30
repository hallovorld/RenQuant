"""Build a minimal PreflightPipeline (state + broker) for Track H proof-of-life.

This factory is the migration entry point. Today it returns a pipeline with
just StateFileTask + BrokerConnectTask wired into one Job. As remaining 14
``_check_*`` functions migrate (one per PR), each lands as a new Task in an
appropriate Job, and ``build_minimal_preflight_pipeline`` is renamed
``build_preflight_pipeline`` and replaces ``kernel.preflight.run_preflight``.
"""
from __future__ import annotations

from .base import PreflightJob, PreflightPipeline
from .tasks.broker import BrokerConnectTask
from .tasks.state import StateFileTask


class _StateAndBrokerJob(PreflightJob):
    """Light-weight Job for the 2 migrated checks. Will be split as more
    checks migrate (artifact / fingerprint / watchlist / etc.)."""

    tasks = [StateFileTask(), BrokerConnectTask()]


def build_minimal_preflight_pipeline() -> PreflightPipeline:
    """Return a PreflightPipeline holding only the migrated checks.

    Intentionally minimal — exists so ``kernel.preflight.run_preflight`` can
    route a subset of checks through the new T/J/P architecture as a feature
    flag (off by default during migration).
    """
    return PreflightPipeline(jobs=[_StateAndBrokerJob()])
