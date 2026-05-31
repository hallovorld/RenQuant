"""Build a minimal PreflightPipeline (state + broker) for Track H proof-of-life.

This factory is the migration entry point. Today it returns a pipeline with
just StateFileTask + BrokerConnectTask wired into one Job. As remaining 14
``_check_*`` functions migrate (one per PR), each lands as a new Task in an
appropriate Job, and ``build_minimal_preflight_pipeline`` is renamed
``build_preflight_pipeline`` and replaces ``kernel.preflight.run_preflight``.
"""
from __future__ import annotations

from .base import PreflightJob, PreflightPipeline
from .tasks.artifact import BestIterTask, ModelArtifactTask, PanelContractTask
from .tasks.broker import BrokerConnectTask
from .tasks.state import StateFileTask


class _ArtifactJob(PreflightJob):
    """Artifact group — checks the active scorer artifact exists, parses,
    carries the contract metadata, and was trained to a healthy best_iter."""

    tasks = [
        ModelArtifactTask(),
        PanelContractTask(),
        BestIterTask(),
    ]


class _StateAndBrokerJob(PreflightJob):
    """State + broker connectivity — final checks before live decisions."""

    tasks = [StateFileTask(), BrokerConnectTask()]


def build_minimal_preflight_pipeline() -> PreflightPipeline:
    """Return a PreflightPipeline holding the migrated checks (5/16 so far).

    Job order mirrors ``kernel.preflight.run_preflight``'s ALL_CHECKS list:
    artifact validation runs first, state + broker last. As more checks lift
    over, they'll be inserted into appropriate Jobs between these two.
    """
    return PreflightPipeline(jobs=[_ArtifactJob(), _StateAndBrokerJob()])
