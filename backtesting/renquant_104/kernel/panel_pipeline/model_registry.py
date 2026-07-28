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


@registry.register("hf_patchtst")
class HFPatchTSTHandler(_ModelHandler):
    """HF transformers PatchTST (replaces custom-impl 'patchtst' kind).

    Trained by scripts/patchtst_hf.py --save-model. Inference via
    HFPatchTSTPanelScorer (mirrors PatchTSTPanelScorer API but uses HF
    backbone). Artifact stores feature_cols + config_dict internally so
    no config-side feature_cols list required.
    """
    requires_history = True

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: PLC0415
        return HFPatchTSTPanelScorer.load(artifact_path)

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        # NOTE: --warmup-epochs not yet wired in scripts/patchtst_hf.py
        # (DOE warmup_epochs knob is decorative until LR scheduler added).
        return [
            sys.executable, str(_REPO / "scripts/patchtst_hf.py"),
            "--dataset", args.dataset,
            "--label", args.label,
            "--cut", getattr(args, "cut", "cut1_covid"),
            "--seq-len", str(getattr(args, "seq_len", 24)),
            "--epochs", str(getattr(args, "epochs", 15)),
            "--lr", str(getattr(args, "lr", 1e-4)),
            "--weight-decay", str(getattr(args, "weight_decay", 1e-2)),
            "--output-dir", args.output_dir or str(_REPO / "artifacts/hf_patchtst_prod"),
            "--device", getattr(args, "device", "mps"),
            "--save-model",
        ]


@registry.register("regime_router")
class RegimeRouterHandler(_ModelHandler):
    """Regime-conditional model router — per Phase 0 finding that XGB and
    HF PatchTST fail in different regimes. Config schema:

      ranking.panel_scoring.kind = "regime_router"
      ranking.panel_scoring.regime_router = {
        "scorers": {
          "xgb": {"kind": "xgb", "artifact_path": "..."},
          "hf_patchtst": {"kind": "hf_patchtst", "artifact_path": "..."},
        },
        "routing": {"BEAR": "hf_patchtst", "CHOPPY": "hf_patchtst",
                     "BULL_CALM": "xgb", "BULL_VOLATILE": "xgb"},
        "default_scorer_key": "xgb",
      }
    """
    requires_history = True  # router may dispatch to history-requiring scorer

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        from kernel.panel_pipeline.regime_router_scorer import (  # noqa: PLC0415
            RegimeRouterScorer, DEFAULT_ROUTING)
        from pathlib import Path as _P  # noqa: PLC0415
        rr_cfg = (config.get("ranking", {}).get("panel_scoring", {})
                          .get("regime_router", {}))
        sub_scorers_cfg = rr_cfg.get("scorers", {})
        if not sub_scorers_cfg:
            raise ValueError("regime_router config missing 'scorers' dict "
                              "(at ranking.panel_scoring.regime_router.scorers)")
        strategy_dir = config.get("_strategy_dir") or _REPO / "backtesting/renquant_104"
        loaded = {}
        for key, sub in sub_scorers_cfg.items():
            sub_kind = sub["kind"]
            sub_handler = registry.get(sub_kind)  # recursive registry dispatch
            p = _P(sub["artifact_path"])
            if not p.is_absolute():
                p = _P(strategy_dir) / p
            # Build a temp config so sub-handler sees its own panel_scoring section
            sub_config = dict(config)
            sub_config.setdefault("ranking", {})
            sub_config["ranking"] = dict(config["ranking"])
            sub_panel_cfg = dict(sub_config["ranking"].get("panel_scoring", {}))
            sub_panel_cfg["kind"] = sub_kind
            sub_panel_cfg["artifact_path"] = str(p)
            # Pass any sub-specific feature_cols / seq_len
            for k in ("feature_cols", "seq_len"):
                if k in sub:
                    sub_panel_cfg[k] = sub[k]
            sub_config["ranking"]["panel_scoring"] = sub_panel_cfg
            loaded[key] = sub_handler.scorer_loader(p, sub_config)
        return RegimeRouterScorer(
            scorers=loaded,
            routing=rr_cfg.get("routing", DEFAULT_ROUTING),
            default_scorer_key=rr_cfg.get("default_scorer_key", "xgb"),
        )

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        raise NotImplementedError(
            "regime_router is an inference-only composition; train each "
            "sub-model independently then wire via config.")


def _import_pipeline_blend_loader():
    """Import the ONE blend implementation from the pinned renquant-pipeline.

    DELEGATION, NOT DUPLICATION (2026-07-28 rehearsal-caught fork
    divergence): pipeline#218's ``blend_scorer`` is the single source of
    truth for the certified z(prod)+z(clf) composite — the umbrella fork
    must not carry a second copy of the scorer/loader (pin verification,
    z-guard, composite-fingerprint recipe) that could drift. The daily
    run's PYTHONPATH (``.subrepo_assembly/current.env``) carries the
    pinned pipeline src, so the direct import works in the live runner;
    fall back to the same RENQUANT_SUBREPO_ROOT / sibling-checkout
    discovery ``job_panel_scoring._bootstrap_gate_registry_import`` uses.
    An unresolvable import RAISES — LoadScorerTask converts it into the
    ``panel_scorer_load_failed`` fail-close (never a silent skip).
    """
    try:
        from renquant_pipeline.kernel.panel_pipeline.blend_scorer import (  # noqa: PLC0415
            load_blend_scorer,
        )
    except ImportError:
        import os  # noqa: PLC0415
        candidates: list[Path] = []
        subrepo_root = os.environ.get("RENQUANT_SUBREPO_ROOT")
        if subrepo_root:
            candidates.append(Path(subrepo_root) / "renquant-pipeline" / "src")
        candidates.append(_REPO.parent / "renquant-pipeline" / "src")
        for src in candidates:
            src_str = str(src)
            if src.is_dir() and src_str not in sys.path:
                sys.path.insert(0, src_str)
        from renquant_pipeline.kernel.panel_pipeline.blend_scorer import (  # noqa: PLC0415
            load_blend_scorer,
        )
    return load_blend_scorer


@registry.register("blend")
class BlendHandler(_ModelHandler):
    """Composite z(prod) + z(clf) blend — the certified construction
    (renquant-model#74/75/76 confirmatory line; pipeline#218 kind).

    Umbrella mirror of the pipeline-side ``BlendHandler``: the loader is
    IMPORTED from the pinned renquant-pipeline (see
    ``_import_pipeline_blend_loader``), so both dispatch sites execute the
    identical fail-closed two-pin load path. Config schema (frozen —
    pipeline ``blend_scorer.py`` module docstring):

      ranking.panel_scoring.kind = "blend"
      ranking.panel_scoring.components = [   # exactly TWO, order-significant
        {"artifact_path": ...,               # 0: production panel scorer
         "expected_content_sha256": ...,     #    abbrev-tolerant content pin
         "expected_config_fingerprint": ...},#    verbatim fp pin (both forms)
        {"artifact_path": ...,               # 1: top-decile classifier
         "expected_content_sha256": ...,
         "expected_config_fingerprint": ...},
      ]

    Both pins are REQUIRED and verified fail-closed at load. The
    ``artifact_path`` argument is ignored — components carry their own
    pinned paths (LoadScorerTask anchors its path-based consistency gate
    on component 0 / the top-level ``artifact_path`` when configured).
    """
    requires_history = False

    @classmethod
    def scorer_loader(cls, artifact_path, config):
        del artifact_path  # components carry their own pinned paths
        load_blend_scorer = _import_pipeline_blend_loader()
        return load_blend_scorer(config)

    @classmethod
    def train_cmd(cls, args) -> list[str]:
        raise NotImplementedError(
            "blend is an inference-only composition; train each component "
            "independently then wire via config.")


__all__ = ["registry", "_ModelHandler"]
