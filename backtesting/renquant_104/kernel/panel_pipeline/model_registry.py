"""Panel-LTR model registry — single source of truth for kind → handlers.

Extensible: register a new model (e.g. lgbm, catboost, ngboost) with a
single registry.register() call. Train pipeline + inference pipeline
both look up by `kind` from config.

Per CLAUDE.md §1c: thin registry, single responsibility. Each registered
handler is a function that takes config + does its own work.

Example use:

    # Inference (LoadScorerTask):
    from kernel.panel_pipeline.model_registry import registry
    kind = config.get("ranking", {}).get("panel_scoring", {}).get("kind", "xgb")
    handler = registry.get(kind)
    scorer = handler.scorer_loader(artifact_path, config)
    # → scorer.score() or scorer.score_with_history()

    # Train (scripts/train_panel_unified.py):
    handler = registry.get(args.kind)
    handler.train_cmd(args)  # invokes the underlying training script

Adding a new model:

    @registry.register("lgbm")
    class LGBMModelHandler:
        kind = "lgbm"
        requires_history = False
        @classmethod
        def scorer_loader(cls, artifact_path, config):
            from kernel.panel_pipeline.lgbm_scorer import LGBMScorer
            return LGBMScorer.load(artifact_path)
        @classmethod
        def train_cmd(cls, args) -> list[str]:
            return [sys.executable, "scripts/train_panel_lgbm.py",
                    "--dataset", args.dataset, "--output", args.output]
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable, Optional

_REPO = Path(__file__).resolve().parents[4]


class _ModelHandler:
    """Base class for model handlers (kind-specific train + inference)."""
    kind: str = ""                  # name used in config (config.kind = ...)
    requires_history: bool = False  # True for sequence models (PatchTST, etc.)

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        raise NotImplementedError

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        """Return shell command list to invoke training."""
        raise NotImplementedError


class _Registry:
    """Singleton registry mapping kind → handler."""

    def __init__(self):
        self._handlers: dict[str, type[_ModelHandler]] = {}

    def register(self, name: str):
        """Decorator: @registry.register("xgb") class XGBHandler: ..."""
        def _wrap(cls: type[_ModelHandler]):
            cls.kind = name
            self._handlers[name] = cls
            return cls
        return _wrap

    def get(self, name: str) -> type[_ModelHandler]:
        if name not in self._handlers:
            raise ValueError(
                f"model kind {name!r} not registered. "
                f"Available: {sorted(self._handlers.keys())}")
        return self._handlers[name]

    def list(self) -> list[str]:
        return sorted(self._handlers.keys())


registry = _Registry()


# ── Built-in handlers ──────────────────────────────────────────────────────

@registry.register("xgb")
class XGBHandler(_ModelHandler):
    """XGBoost rank:pairwise — the current production model."""
    requires_history = False

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        from kernel.panel_pipeline.panel_scorer import PanelScorer  # noqa: PLC0415
        return PanelScorer.load(artifact_path)

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        return [
            sys.executable, str(_REPO / "scripts/train_panel_alpha158_xgb.py"),
            "--dataset", args.dataset,
            "--output", args.output or str(_REPO / "artifacts/panel-ltr.alpha158_fund.json"),
            "--label", args.label,
            "--seed", str(args.seed),
        ]


@registry.register("patchtst")
class PatchTSTHandler(_ModelHandler):
    """PatchTST (Nie 2023 ICLR) — sequence transformer, needs history at inference."""
    requires_history = True

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        from kernel.panel_pipeline.patchtst_scorer import PatchTSTPanelScorer  # noqa: PLC0415
        panel_cfg = config.get("ranking", {}).get("panel_scoring", {})
        seq_len = int(panel_cfg.get("seq_len", 32))
        feature_cols = panel_cfg.get("feature_cols")
        if not feature_cols:
            raise ValueError("PatchTST requires ranking.panel_scoring.feature_cols "
                              "list in config (model .pt doesn't store them)")
        return PatchTSTPanelScorer.load(artifact_path, feature_cols=list(feature_cols),
                                         seq_len=seq_len)

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        return [
            sys.executable, str(_REPO / "scripts/transformer_v4.py"),
            "--dataset", args.dataset,
            "--arch", "patchtst",
            "--label", args.label,
            "--seq-len", str(getattr(args, "seq_len", 32)),
            "--epochs", str(getattr(args, "epochs", 10)),
            "--num-seeds", str(getattr(args, "num_seeds", 5)),
            "--output-dir", args.output_dir or str(_REPO / "artifacts/patchtst_unified"),
            "--device", getattr(args, "device", "mps"),
        ]


__all__ = ["registry", "_ModelHandler"]
