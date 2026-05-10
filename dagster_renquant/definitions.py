"""Top-level Dagster Definitions for the RenQuant cron-tier graph.

Loaded by ``dagster dev -m dagster_renquant.definitions``.
"""

from __future__ import annotations

from dagster import Definitions

from dagster_renquant.assets import ALL_ASSETS

defs = Definitions(assets=ALL_ASSETS)
