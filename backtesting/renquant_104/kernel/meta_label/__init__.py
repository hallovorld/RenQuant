"""Meta-Labeling for Smart Exit Policies (López de Prado AFML ch.20).

See doc/research/meta-labeling-exit-policy.md for the full design.
"""
from .snapshot import SnapshotLogger, FEATURE_COLUMNS  # noqa: F401

__all__ = ["SnapshotLogger", "FEATURE_COLUMNS"]
