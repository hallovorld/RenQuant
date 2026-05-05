"""Reusable Task atoms — see README.md for design rules.

Import from this top-level module so call sites stay clean:

    from kernel.pipeline.atoms import (
        IsFiniteGuardTask, BuildVectorFromMappingTask,
        WriteJSONArtifactTask, LogSummaryTask, ...
    )
"""
from .ctx_ops import (
    AssertFieldExistsTask,
    ClearFieldTask,
    CopyFieldTask,
)
from .gates import (
    SkipIfConfigDisabledTask,
    SkipIfFieldEqualsTask,
    SkipIfFieldFalsyTask,
)
from .logging_atoms import IncrementCounterTask, LogSummaryTask
from .numerical import (
    ClampFieldTask,
    IsFiniteGuardTask,
    NonEmptyGuardTask,
    RangeGuardTask,
)
from .persistence import LoadParquetTask, WriteJSONArtifactTask
from .vectors import (
    BuildMaskFromConditionTask,
    BuildVectorFromMappingTask,
    StableTickerOrderTask,
)

__all__ = [
    # ctx_ops
    "AssertFieldExistsTask", "ClearFieldTask", "CopyFieldTask",
    # gates
    "SkipIfConfigDisabledTask", "SkipIfFieldEqualsTask", "SkipIfFieldFalsyTask",
    # logging
    "IncrementCounterTask", "LogSummaryTask",
    # numerical
    "ClampFieldTask", "IsFiniteGuardTask", "NonEmptyGuardTask", "RangeGuardTask",
    # persistence
    "LoadParquetTask", "WriteJSONArtifactTask",
    # vectors
    "BuildMaskFromConditionTask", "BuildVectorFromMappingTask",
    "StableTickerOrderTask",
]
