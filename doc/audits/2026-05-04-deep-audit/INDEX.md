# 2026-05-04 Deep Audit — Pipeline / Job / Task

User mandate: while waiting for compute / permission, audit every
pipeline architecture, every Job logic, every Task implementation —
interfaces, data formats, edge cases. 100×.

## Scope

- **5 Pipelines**: InferencePipeline, SellOnlyPipeline, FullTrainingPipeline, TrainingPipeline, PanelTrainingPipeline
- **25 Jobs**
- **97 Tasks**

## Audit format (per Task)

```
### {ClassName}
File: {path}:{line}
Reads: {ctx fields}
Writes: {ctx fields}
Skip / short-circuit:
Edge cases handled:
Edge cases NOT handled:
Tests:
🔴 Issues found:
🟡 Concerns:
```

## Priority order

1. **InferencePipeline** path (production live)
2. **PanelScoringJob** (already partial — re-audit completeness)
3. **PortfolioQP** (sizing decisions)
4. **PanelTrainingPipeline** (model-creation path)
5. **TrainingPipeline** (legacy per-ticker, deprecated but still wired)

## Per-pipeline audit files

- [`01-inference-pipeline.md`](01-inference-pipeline.md) — production runtime path
- [`02-panel-scoring-job.md`](02-panel-scoring-job.md) — score → calibrate → veto
- [`03-portfolio-qp.md`](03-portfolio-qp.md) — sizing + target weights
- [`04-panel-training-pipeline.md`](04-panel-training-pipeline.md)
- [`05-training-pipeline-legacy.md`](05-training-pipeline-legacy.md)

## Issue tracker

Each finding gets a 🔴 (bug, must fix) or 🟡 (concern, document).
Cross-reference back to MEMORY task list (#14, #15, #16, etc.) where it
makes sense.

## Status

Started: 2026-05-04 00:55 PT
Last updated: (live, scrolling)
