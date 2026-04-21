"""kernel.pipeline — 7-job InferencePipeline for LEAN and live runner.

Self-contained: no common/ imports.

Usage::

    from kernel.pipeline import InferencePipeline, InferenceContext

    ctx = InferenceContext(config=cfg, today=today, ...)
    InferencePipeline().run(ctx)
"""
from .context  import InferenceContext
from .pipeline import Job, InferencePipeline, SellOnlyPipeline

__all__ = ["InferenceContext", "Job", "InferencePipeline", "SellOnlyPipeline"]
