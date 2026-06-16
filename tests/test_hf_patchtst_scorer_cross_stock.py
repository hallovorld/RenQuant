"""Regression: HFPatchTSTPanelScorer.load must reconstruct the OPTIONAL
architecture layers (cross-stock attention, FiLM) the checkpoint was trained
with — otherwise a cross-stock model is silently scored through the channel-
independent baseline (load_state_dict(strict=False) drops the unmatched
cross_stock.* tensors), producing wrong scores and an un-promotable model.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import PatchTSTConfig

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("phf_cs", REPO / "scripts/patchtst_hf.py")
phf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(phf)

from kernel.panel_pipeline.hf_patchtst_scorer import HFPatchTSTPanelScorer  # noqa: E402


def _cfg():
    return PatchTSTConfig(
        num_input_channels=4, context_length=8, patch_length=2, patch_stride=2,
        d_model=16, num_attention_heads=2, num_hidden_layers=1, ffn_dim=32)


def _save(tmp_path, *, cross_stock, declared_cross_stock=None):
    """Save a scorer-format .pt. declared_cross_stock lets a test lie about the
    flag (to exercise the fail-loud guard)."""
    cfg = _cfg()
    model = phf.HFPatchTSTRanker(cfg, use_distributional_head=True,
                                 use_cross_stock_attn=cross_stock)
    flag = cross_stock if declared_cross_stock is None else declared_cross_stock
    p = tmp_path / "hf_patchtst_model.pt"
    torch.save({
        "state_dict": model.state_dict(), "config_dict": cfg.to_dict(),
        "feature_cols": ["f0", "f1", "f2", "f3"], "seq_len": 8,
        "label_col": "fwd_60d_excess", "best_val_ic": 0.0,
        "uses_distributional_head": True, "uses_film_regime": False,
        "uses_cross_stock_attn": flag,
    }, p)
    return p


def test_cross_stock_layer_reconstructed(tmp_path):
    s = HFPatchTSTPanelScorer.load(_save(tmp_path, cross_stock=True))
    assert s._model.cross_stock is not None


def test_baseline_loads_without_cross_stock(tmp_path):
    s = HFPatchTSTPanelScorer.load(_save(tmp_path, cross_stock=False))
    assert s._model.cross_stock is None


def test_unexpected_tensor_fails_loud(tmp_path):
    # cross-stock weights in the checkpoint but the flag says baseline -> the
    # cross_stock.* tensors are unexpected; loader must refuse, not silently drop.
    p = _save(tmp_path, cross_stock=True, declared_cross_stock=False)
    with pytest.raises(ValueError, match="did not reconstruct"):
        HFPatchTSTPanelScorer.load(p)
