"""Re-export shim — implementation lives in kernel/pipeline/pp_training.py."""
from kernel.pipeline.pp_training import (  # noqa: F401
    TrainingContext,
    TickerTrainingContext,
    TrainingTask,
    TrainingJob,
    TrainingTickerJob,
    run_ticker_parallel,
    DataFetchJob,
    RegimeFitJob,
    HurstCUSUMTask,
    GMMFitTask,
    RegimeCombineTask,
    RegimeSaveTask,
    FeatureJob,
    TournamentJob,
    ExportJob,
    CalibrationJob,
    CorrelationJob,
    TickerFeatureJob,
    TickerTournamentJob,
    TickerExportJob,
    TickerCalibrationJob,
    TrainingPipeline,
)

# Backward-compat aliases for notebook code that used the old ABC names
Job = TrainingJob
TickerJob = TrainingTickerJob
