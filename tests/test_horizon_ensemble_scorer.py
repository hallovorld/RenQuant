"""Hermetic tests for HorizonEnsembleScorer (multi-horizon rank ensemble).

Design: doc/design/2026-06-16-multi-horizon-ensemble.md (PR #146).

Two kinds of coverage:

  * REAL load path — build 2-3 TINY HFPatchTSTRanker models (small
    PatchTSTConfig, mirroring test_hf_patchtst_scorer_cross_stock.py), save
    them as scorer .pt files, load via HorizonEnsembleScorer.load([...]) and
    HorizonEnsembleScorer.load(manifest.json), and assert the public surface
    (a score per scored ticker, requires_history, seq_len = max component).

  * RANK-AVERAGE MATH — use lightweight stub components whose raw scores are
    fixed, so the per-day percentile-rank average is exactly predictable. This
    isolates the ensemble's combination logic from the (uncontrollable)
    PatchTST forward pass.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PatchTSTConfig

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "phf_hens", REPO / "scripts/patchtst_hf.py")
phf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phf)

from kernel.panel_pipeline.horizon_ensemble_scorer import (  # noqa: E402
    HorizonComponent,
    HorizonEnsembleScorer,
)

FEATURES = ["f0", "f1", "f2", "f3"]
SEQ_LEN = 8


# ──────────────────────────────────────────────────────────────────────────
# REAL tiny-model fixtures (mirror test_hf_patchtst_scorer_cross_stock.py)
# ──────────────────────────────────────────────────────────────────────────
def _cfg() -> PatchTSTConfig:
    return PatchTSTConfig(
        num_input_channels=len(FEATURES), context_length=SEQ_LEN,
        patch_length=2, patch_stride=2, d_model=16, num_attention_heads=2,
        num_hidden_layers=1, ffn_dim=32)


def _save_tiny_scorer(path: Path, *, seq_len: int = SEQ_LEN,
                      label_col: str = "fwd_20d_excess", seed: int = 0) -> Path:
    """Save a scorer-format .pt for a tiny HFPatchTSTRanker."""
    torch.manual_seed(seed)
    cfg = _cfg()
    cfg.context_length = seq_len
    model = phf.HFPatchTSTRanker(cfg, use_distributional_head=False)
    torch.save({
        "state_dict": model.state_dict(), "config_dict": cfg.to_dict(),
        "feature_cols": list(FEATURES), "seq_len": seq_len,
        "label_col": label_col, "best_val_ic": 0.0,
        "uses_distributional_head": False, "uses_film_regime": False,
        "uses_cross_stock_attn": False,
        # Skip CSRankNorm so the test panel feeds straight through (keeps the
        # real-model assertions about *which* tickers get scored simple).
        "uses_csranknorm_preprocessing": False,
    }, path)
    return path


def _panel(tickers, n_days: int = SEQ_LEN + 2, seed: int = 1) -> pd.DataFrame:
    """Build a hermetic (ticker, date, features) panel with >= seq_len rows."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for tkr in tickers:
        for d in dates:
            row = {"ticker": tkr, "date": d}
            for f in FEATURES:
                row[f] = float(rng.standard_normal())
            rows.append(row)
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────
# Stub component — fixed raw scores, so rank-average is predictable
# ──────────────────────────────────────────────────────────────────────────
class _StubScorer:
    """Minimal scorer with controllable raw outputs and the required surface."""

    def __init__(self, raw: dict, *, seq_len: int = 4,
                 feature_cols=None, label_col: str = ""):
        self._raw = dict(raw)
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols or FEATURES)
        self.requires_history = True
        self.metadata = {"label_col": label_col}

    def score_with_history(self, panel_history, target_tickers) -> pd.Series:
        items = [(t, self._raw[t]) for t in target_tickers if t in self._raw]
        if not items:
            return pd.Series([], dtype=float, name="panel_score")
        idx, vals = zip(*items)
        return pd.Series(vals, index=list(idx), name="panel_score")


def _stub_component(raw, horizon="", weight=1.0, **kw) -> HorizonComponent:
    return HorizonComponent(scorer=_StubScorer(raw, **kw), horizon=horizon,
                            weight=weight)


# ══════════════════════════════════════════════════════════════════════════
# (1) score_with_history returns a score per requested ticker (REAL models)
# ══════════════════════════════════════════════════════════════════════════
def test_real_models_score_every_requested_ticker(tmp_path):
    p1 = _save_tiny_scorer(tmp_path / "h5.pt", label_col="fwd_5d_excess", seed=1)
    p2 = _save_tiny_scorer(tmp_path / "h20.pt", label_col="fwd_20d_excess", seed=2)
    p3 = _save_tiny_scorer(tmp_path / "h60.pt", label_col="fwd_60d_excess", seed=3)
    ens = HorizonEnsembleScorer.load([p1, p2, p3])

    tickers = ["AAA", "BBB", "CCC", "DDD"]
    panel = _panel(tickers)
    out = ens.score_with_history(panel, tickers)

    assert isinstance(out, pd.Series)
    assert set(out.index) == set(tickers)            # one score per ticker
    assert out.between(0.0, 1.0).all()               # ranks in [0, 1]
    assert not out.isna().any()
    # Horizons inferred from each checkpoint's label_col.
    assert ens.metadata["component_horizons"] == ["5d", "20d", "60d"]


# ══════════════════════════════════════════════════════════════════════════
# (2) result is the per-day percentile-rank AVERAGE of the components
# ══════════════════════════════════════════════════════════════════════════
def test_equal_weight_is_percentile_rank_average():
    # Component A ranks AAA > BBB > CCC > DDD (raw scale 0..3).
    # Component B ranks them in the OPPOSITE order on a totally different scale
    # (so raw averaging would be meaningless; rank averaging is well-defined).
    comp_a = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0, "DDD": 0.0}
    comp_b = {"AAA": -100.0, "BBB": -200.0, "CCC": -300.0, "DDD": -400.0}
    # rank(pct=True) over 4 names → {min:0.25, .., max:1.0}.
    #   A ranks: AAA1.00 BBB0.75 CCC0.50 DDD0.25
    #   B ranks: AAA1.00 BBB0.75 CCC0.50 DDD0.25  (same, since B is also
    #            descending AAA..DDD)
    # equal-weight average == that same vector.
    ens = HorizonEnsembleScorer([
        _stub_component(comp_a, horizon="5d"),
        _stub_component(comp_b, horizon="60d"),
    ])
    targets = ["AAA", "BBB", "CCC", "DDD"]
    out = ens.score_with_history(pd.DataFrame(), targets)
    expected = pd.Series({"AAA": 1.00, "BBB": 0.75, "CCC": 0.50, "DDD": 0.25})
    pd.testing.assert_series_equal(
        out.reindex(targets), expected.reindex(targets),
        check_names=False)


def test_opposite_rankings_average_to_neutral():
    # A ranks AAA best→DDD worst; B ranks DDD best→AAA worst. Equal-weight
    # rank-average must be exactly 0.625 for all (mean of {1.0,..,0.25} pairs).
    comp_a = {"AAA": 4.0, "BBB": 3.0, "CCC": 2.0, "DDD": 1.0}
    comp_b = {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0, "DDD": 4.0}
    ens = HorizonEnsembleScorer([
        _stub_component(comp_a), _stub_component(comp_b)])
    targets = ["AAA", "BBB", "CCC", "DDD"]
    out = ens.score_with_history(pd.DataFrame(), targets)
    # A: AAA1.00 BBB0.75 CCC0.50 DDD0.25 ; B: AAA0.25 BBB0.50 CCC0.75 DDD1.00
    # avg = 0.625 everywhere.
    assert np.allclose(out.reindex(targets).values, 0.625)


def test_ranks_are_recomputed_per_day_over_targets_only():
    # A component scoring a SUPERSET of the targets must still rank only over
    # the requested targets (the ensemble restricts before ranking).
    comp = {"AAA": 10.0, "BBB": 5.0, "CCC": 1.0, "ZZZ": 999.0}
    ens = HorizonEnsembleScorer([_stub_component(comp)])
    out = ens.score_with_history(pd.DataFrame(), ["AAA", "BBB", "CCC"])
    # Over {AAA,BBB,CCC} only → AAA top, CCC bottom; ZZZ excluded entirely.
    assert "ZZZ" not in out.index
    expected = pd.Series({"AAA": 1.0, "BBB": 2 / 3, "CCC": 1 / 3})
    pd.testing.assert_series_equal(
        out.reindex(["AAA", "BBB", "CCC"]),
        expected.reindex(["AAA", "BBB", "CCC"]), check_names=False)


# ══════════════════════════════════════════════════════════════════════════
# (3) equal vs custom weights behave correctly
# ══════════════════════════════════════════════════════════════════════════
def test_custom_weights_shift_blend_toward_heavier_component():
    # A: AAA best (rank 1.0), BBB worst (rank 0.5 with 2 names → actually
    # rank(pct) over 2 = {0.5, 1.0}). Use 3 names for clarity.
    comp_a = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}   # ranks 1.0 / 0.667 / 0.333
    comp_b = {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0}   # ranks 0.333 / 0.667 / 1.0
    targets = ["AAA", "BBB", "CCC"]

    # Equal weights → AAA and CCC symmetric around BBB.
    eq = HorizonEnsembleScorer([_stub_component(comp_a), _stub_component(comp_b)])
    out_eq = eq.score_with_history(pd.DataFrame(), targets)
    assert out_eq["AAA"] == pytest.approx(out_eq["CCC"])  # symmetric
    assert out_eq["BBB"] == pytest.approx(2 / 3)

    # Weight A 3x → blend leans toward A's ranking (AAA up, CCC down).
    wt = HorizonEnsembleScorer([
        _stub_component(comp_a, weight=3.0),
        _stub_component(comp_b, weight=1.0),
    ])
    out_wt = wt.score_with_history(pd.DataFrame(), targets)
    # Normalized weights 0.75 / 0.25:
    #   AAA = 0.75*1.0   + 0.25*0.333 = 0.8333
    #   CCC = 0.75*0.333 + 0.25*1.0   = 0.5
    assert out_wt["AAA"] == pytest.approx(0.75 * 1.0 + 0.25 * (1 / 3))
    assert out_wt["CCC"] == pytest.approx(0.75 * (1 / 3) + 0.25 * 1.0)
    assert out_wt["AAA"] > out_eq["AAA"]   # heavier A pulls AAA up
    assert out_wt["CCC"] < out_eq["CCC"]   # and CCC down


def test_weights_are_normalized():
    # Weights need not sum to 1; only their ratio matters.
    comp_a = {"AAA": 3.0, "BBB": 2.0, "CCC": 1.0}
    comp_b = {"AAA": 1.0, "BBB": 2.0, "CCC": 3.0}
    targets = ["AAA", "BBB", "CCC"]
    a = HorizonEnsembleScorer([
        _stub_component(comp_a, weight=3.0), _stub_component(comp_b, weight=1.0)])
    b = HorizonEnsembleScorer([
        _stub_component(comp_a, weight=30.0), _stub_component(comp_b, weight=10.0)])
    pd.testing.assert_series_equal(
        a.score_with_history(pd.DataFrame(), targets).reindex(targets),
        b.score_with_history(pd.DataFrame(), targets).reindex(targets))


def test_bad_weights_rejected():
    comp = {"AAA": 1.0}
    with pytest.raises(ValueError, match="must sum to|>= 0"):
        HorizonEnsembleScorer([_stub_component(comp, weight=0.0)])
    with pytest.raises(ValueError, match=">= 0"):
        HorizonEnsembleScorer([_stub_component(comp, weight=-1.0)])
    with pytest.raises(ValueError, match="at least one"):
        HorizonEnsembleScorer([])


# ══════════════════════════════════════════════════════════════════════════
# (4) seq_len = max(component seq_lens); feature_cols = union
# ══════════════════════════════════════════════════════════════════════════
def test_seq_len_is_max_and_feature_cols_union():
    ens = HorizonEnsembleScorer([
        _stub_component({"AAA": 1.0}, seq_len=5, feature_cols=["f0", "f1"]),
        _stub_component({"AAA": 1.0}, seq_len=12, feature_cols=["f1", "f2"]),
        _stub_component({"AAA": 1.0}, seq_len=8, feature_cols=["f2", "f3"]),
    ])
    assert ens.seq_len == 12
    assert ens.requires_history is True
    assert ens.feature_cols == ["f0", "f1", "f2", "f3"]  # union, first-seen order


def test_real_seq_len_is_max_across_checkpoints(tmp_path):
    p_short = _save_tiny_scorer(tmp_path / "short.pt", seq_len=SEQ_LEN, seed=1)
    # A longer-context checkpoint needs a context_length divisible by stride.
    p_long = _save_tiny_scorer(tmp_path / "long.pt", seq_len=SEQ_LEN + 4, seed=2)
    ens = HorizonEnsembleScorer.load([p_short, p_long])
    assert ens.seq_len == SEQ_LEN + 4


# ══════════════════════════════════════════════════════════════════════════
# Loader: JSON manifest path
# ══════════════════════════════════════════════════════════════════════════
def test_load_from_json_manifest(tmp_path):
    _save_tiny_scorer(tmp_path / "h5.pt", label_col="fwd_5d_excess", seed=1)
    _save_tiny_scorer(tmp_path / "h60.pt", label_col="fwd_60d_excess", seed=2)
    manifest = tmp_path / "ensemble.json"
    manifest.write_text(json.dumps({
        "components": [
            {"path": "h5.pt", "horizon": "5d", "weight": 1.0},
            {"path": "h60.pt", "horizon": "60d", "weight": 2.0},
        ]
    }))
    ens = HorizonEnsembleScorer.load(manifest)
    assert ens.metadata["component_horizons"] == ["5d", "60d"]
    # weights normalized 1:2 → 1/3, 2/3
    assert np.allclose(ens.metadata["component_weights"], [1 / 3, 2 / 3])
    # relative paths resolved against the manifest dir
    assert all(Path(p).exists() for p in ens.metadata["component_paths"])


def test_empty_targets_returns_empty():
    ens = HorizonEnsembleScorer([_stub_component({"AAA": 1.0})])
    out = ens.score_with_history(pd.DataFrame(), [])
    assert out.empty


def test_score_raises_directing_to_history_api():
    ens = HorizonEnsembleScorer([_stub_component({"AAA": 1.0})])
    with pytest.raises(NotImplementedError, match="score_with_history"):
        ens.score(pd.DataFrame())
