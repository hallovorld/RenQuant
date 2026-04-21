"""kernel.pipeline — parallel InferencePipeline for LEAN and live runner.

Architecture: 3-phase pipeline where per-ticker work (sell eval, candidate
scoring) runs in parallel via ThreadPoolExecutor.

Usage::

    from kernel.pipeline import InferencePipeline, InferenceContext

    ctx = InferenceContext(config=cfg, today=today, ...)
    InferencePipeline().run(ctx)
"""
from .context  import InferenceContext, TickerInferenceContext
from .pipeline import Job, TickerJob, InferencePipeline, SellOnlyPipeline, run_parallel

__all__ = [
    "InferenceContext", "TickerInferenceContext",
    "Job", "TickerJob",
    "InferencePipeline", "SellOnlyPipeline",
    "run_parallel",
]
