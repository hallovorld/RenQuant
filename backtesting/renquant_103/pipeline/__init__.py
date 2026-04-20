"""Pipeline orchestration for renquant_103.

Three jobs: DataJob → SignalJob → ExecutionJob.

Usage (live runner)::

    from pipeline import Pipeline, PipelineContext
    from pipeline.jobs.data import DataJob
    from pipeline.jobs.signals import SignalJob
    from pipeline.jobs.execution import ExecutionJob

    ctx = PipelineContext(config=config, strategy_dir=strategy_dir,
                          sell_only=sell_only, broker=broker, models=models)
    Pipeline([DataJob(), SignalJob(), ExecutionJob()]).run(ctx)
"""
from .context import PipelineContext
from .pipeline import Job, Pipeline
from .task import TaskResult, run_tasks

__all__ = [
    "PipelineContext",
    "Job",
    "Pipeline",
    "TaskResult",
    "run_tasks",
]
