"""Multi-horizon ensemble panel scorer (shadow-first, 2026-06-16).

Design: doc/design/2026-06-16-multi-horizon-ensemble.md (PR #146).

Motivation
----------
A single PatchTST checkpoint is trained against ONE label horizon (e.g.
``fwd_20d_excess``). Different horizons (5d / 20d / 60d) capture different
slices of the return-generating process, and multiple seeds at the same
horizon average away initialization noise. Combining several checkpoints
produces a sturdier cross-sectional ranking than any single model.

Scale invariance
----------------
PatchTST raw scores are NOT comparable across checkpoints — each model
sits on its own (often all-negative) scale, and the means/spreads differ
by horizon and by seed. Averaging the *raw* scores would let the model
with the widest spread dominate. Instead we convert each component's
per-day scores to **cross-sectional percentile ranks in [0, 1]** and
average those. Rank-averaging is scale-invariant (the same principle the
legacy :class:`EnsemblePanelScorer` and the Kelly-Gu-Xiu 2020 CSRankNorm
preprocessing rely on), so a 60d model whose raw scores span [-0.4, -0.1]
composes cleanly with a 5d model spanning [0.0, 2.0].

Interface
---------
Drop-in for the existing history-requiring scorer surface (mirrors
:class:`HFPatchTSTPanelScorer`):

  * ``feature_cols``    — union across components
  * ``seq_len``         — MAX across components (so the panel-history loader
                          fetches enough rows for the longest-context model)
  * ``requires_history``— ``True``
  * ``score_with_history(panel_history, target_tickers) -> pd.Series``
                          indexed by ticker, values = ensemble rank in [0, 1].

This class is NOT returned by ``HFPatchTSTPanelScorer.load(single_path)`` —
like :class:`EnsemblePanelScorer` it is built explicitly from several
component artifacts via :meth:`HorizonEnsembleScorer.load`. Wiring it into
the (shadow) scoring path is a later, separate change; nothing here touches
live state, the runner, or strategy configs.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer

log = logging.getLogger("kernel.panel_pipeline.horizon_ensemble_scorer")


@dataclass
class HorizonComponent:
    """One member of the ensemble: a loaded scorer tagged with its horizon.

    Attributes:
      scorer:  the loaded :class:`HFPatchTSTPanelScorer` (or any object with
               ``score_with_history``, ``feature_cols``, ``seq_len``).
      horizon: human-readable label for the label horizon this checkpoint was
               trained on, e.g. ``"5d"`` / ``"20d"`` / ``"60d"``. Informational
               only — it does not affect scoring, just provenance/logging.
      weight:  non-negative blend weight. Weights are normalized to sum to 1
               across the ensemble; equal weights by default.
      path:    source artifact path (provenance).
    """

    scorer: object
    horizon: str = ""
    weight: float = 1.0
    path: Optional[str] = None


def _cross_sectional_percentile_rank(scores: pd.Series) -> pd.Series:
    """Convert a Series of raw scores to cross-sectional percentile ranks.

    Uses ``rank(pct=True)`` (average rank for ties) so the result lies in
    (0, 1]: the highest score maps near 1.0, the lowest near ~1/N. A single
    element maps to 1.0 (degenerate cross-section); an empty Series passes
    through. NaNs are ranked as NaN by pandas and dropped by the caller's
    reindex/fill, so we leave them be here.
    """
    if scores.empty:
        return scores.astype(float)
    if len(scores) == 1:
        # Degenerate cross-section: a lone name has no relative rank; use the
        # neutral midpoint so it neither tops nor bottoms the blend.
        return pd.Series([0.5], index=scores.index, dtype=float)
    return scores.rank(pct=True)


class HorizonEnsembleScorer:
    """Percentile-rank ensemble over several PatchTST horizon/seed checkpoints.

    For each call to :meth:`score_with_history`:

      1. each component scores the panel → per-component {ticker: raw};
      2. restrict to the requested ``target_tickers`` that the component
         actually scored (a component may skip a ticker lacking enough
         history);
      3. convert that component's raw scores to cross-sectional percentile
         ranks in (0, 1] over the day's scored tickers;
      4. average the ranks across components, weighted by each component's
         (normalized) weight, counting only components that produced a rank
         for a given ticker;
      5. return {ticker: ensemble_rank} as a ``pd.Series``.

    A ticker that no component could score is absent from the result (same
    contract as :class:`HFPatchTSTPanelScorer`).
    """

    def __init__(
        self,
        components: Sequence[HorizonComponent],
        metadata: Optional[dict] = None,
    ) -> None:
        comps = list(components)
        if not comps:
            raise ValueError("HorizonEnsembleScorer: need at least one component")
        self.components = comps

        weights = [float(c.weight) for c in comps]
        if any(w < 0 for w in weights):
            raise ValueError("HorizonEnsembleScorer: component weights must be >= 0")
        wsum = float(sum(weights))
        if wsum <= 0:
            raise ValueError("HorizonEnsembleScorer: weights must sum to > 0")
        self._weights = [w / wsum for w in weights]

        # Union of feature columns across components (each component filters to
        # its own subset at score time, exactly like the other ensembles).
        cols: list[str] = []
        seen: set[str] = set()
        for c in comps:
            for col in getattr(c.scorer, "feature_cols", []):
                if col not in seen:
                    cols.append(col)
                    seen.add(col)
        self.feature_cols = cols

        # MAX seq_len so the history loader pulls enough rows for the longest
        # context window; shorter-context components just use their tail.
        self.seq_len = max(
            (int(getattr(c.scorer, "seq_len", 1)) for c in comps), default=1
        )
        self.requires_history = True

        meta = dict(metadata or {})
        meta["ensemble_kind"] = "multi_horizon_rank"
        meta["n_components"] = len(comps)
        meta["component_horizons"] = [c.horizon for c in comps]
        meta["component_weights"] = list(self._weights)
        meta["component_paths"] = [c.path for c in comps]
        meta["seq_len"] = self.seq_len
        self.metadata = meta

    # ── Scoring ────────────────────────────────────────────────────────────

    def score_with_history(
        self,
        panel_history: pd.DataFrame,
        target_tickers: list[str],
    ) -> pd.Series:
        """Per-day percentile-rank average of the component scorers.

        See the class docstring for the full algorithm. The returned Series is
        indexed by ticker (subset of ``target_tickers`` that at least one
        component scored); values are the weighted-average percentile rank in
        (0, 1].
        """
        if not target_tickers:
            return pd.Series([], dtype=float, name="panel_score")

        targets = list(target_tickers)
        # Accumulate, per ticker, the weighted rank sum and the weight that was
        # actually applied (so tickers some components skipped are renormalized
        # over the components that DID score them).
        rank_sum = pd.Series(0.0, index=targets, dtype=float)
        weight_sum = pd.Series(0.0, index=targets, dtype=float)

        for comp, w in zip(self.components, self._weights):
            scorer = comp.scorer
            feat_cols = list(getattr(scorer, "feature_cols", []))
            # Restrict columns to what this component needs (+ keys), keeping
            # only columns that actually exist so a caller can pass a minimal
            # frame. Harmless if panel_history already matches.
            if feat_cols:
                want = ["ticker", "date", *feat_cols]
                seen: set[str] = set()
                keep = [
                    c for c in want
                    if c in panel_history.columns and not (c in seen or seen.add(c))
                ]
                ph = panel_history[keep]
            else:
                ph = panel_history

            raw = scorer.score_with_history(ph, targets)
            if raw is None or len(raw) == 0:
                log.warning(
                    "HorizonEnsembleScorer: component horizon=%s scored 0 tickers",
                    comp.horizon,
                )
                continue
            # Only rank over the requested targets this component scored.
            raw = raw[[t for t in raw.index if t in rank_sum.index]]
            if raw.empty:
                continue
            ranks = _cross_sectional_percentile_rank(raw)
            rank_sum.loc[ranks.index] += w * ranks.values
            weight_sum.loc[ranks.index] += w

        scored = weight_sum > 0
        if not scored.any():
            return pd.Series([], dtype=float, name="panel_score")
        ensemble = (rank_sum[scored] / weight_sum[scored]).astype(float)
        ensemble.name = "panel_score"
        log.info(
            "HorizonEnsembleScorer.score_with_history: %d components, scored "
            "%d/%d tickers (mean_rank=%.4f)",
            len(self.components), int(scored.sum()), len(targets),
            float(ensemble.mean()),
        )
        return ensemble

    def score(self, feature_matrix: pd.DataFrame) -> pd.Series:  # noqa: D401
        raise NotImplementedError(
            "HorizonEnsembleScorer requires sequence input. Use "
            "score_with_history(panel_history, target_tickers) instead "
            "(requires_history=True)."
        )

    # ── Loader ──────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        manifest_or_paths: "str | Path | Sequence[str | Path]",
        metadata: Optional[dict] = None,
    ) -> "HorizonEnsembleScorer":
        """Build the ensemble from component checkpoints.

        Two input shapes are accepted:

        1. **List of paths** — a sequence of HF PatchTST scorer ``.pt`` paths.
           Each is loaded via :meth:`HFPatchTSTPanelScorer.load`, gets equal
           weight, and its horizon label is read from the checkpoint's
           ``label_col`` metadata when available (informational only)::

               HorizonEnsembleScorer.load([
                   "artifacts/hf_5d_seed0.pt",
                   "artifacts/hf_20d_seed0.pt",
                   "artifacts/hf_60d_seed0.pt",
               ])

        2. **JSON manifest** — a path to a small JSON file describing the
           components explicitly. Either a top-level list, or an object with a
           ``"components"`` list. Each entry is::

               {"path": "...", "horizon": "20d", "weight": 1.0}

           ``horizon`` and ``weight`` are optional (default ``""`` and ``1.0``).
           Relative ``path`` values are resolved against the manifest's
           directory. Example manifest::

               {
                 "components": [
                   {"path": "hf_5d_seed0.pt",  "horizon": "5d",  "weight": 1.0},
                   {"path": "hf_20d_seed0.pt", "horizon": "20d", "weight": 2.0},
                   {"path": "hf_60d_seed0.pt", "horizon": "60d", "weight": 1.0}
                 ]
               }

        Weights are normalized to sum to 1 inside the constructor; the manifest
        may use any non-negative scale.
        """
        entries = cls._resolve_manifest(manifest_or_paths)
        components: list[HorizonComponent] = []
        for entry in entries:
            path = Path(entry["path"])
            scorer = HFPatchTSTPanelScorer.load(path)
            horizon = entry.get("horizon") or cls._horizon_from_scorer(scorer)
            components.append(
                HorizonComponent(
                    scorer=scorer,
                    horizon=str(horizon or ""),
                    weight=float(entry.get("weight", 1.0)),
                    path=str(path),
                )
            )
        log.info(
            "HorizonEnsembleScorer.load: built %d components (horizons=%s)",
            len(components), [c.horizon for c in components],
        )
        return cls(components=components, metadata=metadata)

    # ── Loader helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _resolve_manifest(
        manifest_or_paths: "str | Path | Sequence[str | Path]",
    ) -> list[dict]:
        """Normalize the ``load`` argument to a list of component dicts.

        Returns a list of ``{"path", "horizon"?, "weight"?}`` dicts. A JSON
        manifest path resolves its relative component paths against the
        manifest's directory; a plain list of paths is taken verbatim.
        """
        # A sequence (but not a bare string/Path) → list of paths.
        if isinstance(manifest_or_paths, (list, tuple)):
            return [{"path": str(p)} for p in manifest_or_paths]

        p = Path(manifest_or_paths)
        if p.suffix.lower() == ".json":
            payload = json.loads(p.read_text())
            if isinstance(payload, dict):
                raw_entries = payload.get("components", [])
            elif isinstance(payload, list):
                raw_entries = payload
            else:
                raise ValueError(
                    f"HorizonEnsembleScorer manifest {p} must be a list or an "
                    "object with a 'components' list"
                )
            if not raw_entries:
                raise ValueError(
                    f"HorizonEnsembleScorer manifest {p} has no components"
                )
            base = p.parent
            out: list[dict] = []
            for e in raw_entries:
                if isinstance(e, str):
                    e = {"path": e}
                if "path" not in e:
                    raise ValueError(
                        f"HorizonEnsembleScorer manifest {p}: component entry "
                        f"missing 'path': {e}"
                    )
                comp = dict(e)
                comp_path = Path(comp["path"])
                if not comp_path.is_absolute():
                    comp_path = base / comp_path
                comp["path"] = str(comp_path)
                out.append(comp)
            return out

        # A single non-JSON path → a one-component ensemble.
        return [{"path": str(p)}]

    @staticmethod
    def _horizon_from_scorer(scorer: object) -> str:
        """Best-effort horizon label from a loaded scorer's metadata.

        Reads ``label_col`` (e.g. ``fwd_20d_excess``) and extracts the ``20d``
        token; returns ``""`` when it can't be inferred. Informational only.
        """
        meta = getattr(scorer, "metadata", {}) or {}
        label = meta.get("label_col") or ""
        if not isinstance(label, str):
            return ""
        for tok in label.replace("-", "_").split("_"):
            if tok and tok[0].isdigit() and tok[-1] in "dwmy":
                return tok
        return ""


__all__ = ["HorizonEnsembleScorer", "HorizonComponent"]
