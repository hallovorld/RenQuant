"""RenQuant Dagster asset graph.

Side-by-side with launchd (NOT a replacement). The graph mechanically encodes
the cron-tier dependency chain so that ``promote_decision`` literally cannot
be materialized unless a fresh ``wf_gate_pass`` exists upstream — kills the
``RQ_ALLOW_NO_WF=1`` bypass class by construction.

See ``dagster_renquant/README.md`` for the migration plan.
"""

from dagster_renquant.definitions import defs

__all__ = ["defs"]
