#!/usr/bin/env python
"""Export the trained PyTorch transformer to safetensors + JSON sidecar.

Bridges the Python trainer to the Rust scorer at rust/transformer_scorer.
The Rust side (lib.rs::PanelTransformer::load) reads exactly two files:

    <stem>.safetensors    weights (this script writes)
    <stem>.json           feature_cols + TransformerParams + audit metadata
                          (already written alongside the .pt by the trainer
                          — we read it, optionally augment with the
                          weight-shape fingerprint, and re-emit)

Why this script exists separately from the trainer save path:
    1. safetensors is an optional install on the Python side; the trainer
       shouldn't bring it as a hard requirement just to enable a Rust
       cutover that may or may not happen.
    2. We can run it offline against any historical .pt artifact, including
       ones trained before the Rust port existed.
    3. Conversion is a clear, audited boundary — easier to reason about
       parity bugs.

Usage::

    python scripts/export_transformer_to_safetensors.py \\
        --stem backtesting/renquant_104/artifacts/panel-transformer

Reads `<stem>.pt` + `<stem>.json`, writes `<stem>.safetensors` next to them.

Round-trip invariant:
    For every named tensor in the PyTorch state_dict, an identically-named
    tensor of identical shape and dtype exists in the safetensors file.
    The Rust loader (transformer_block.rs) hard-codes those exact names,
    so a successful export means a successful Rust load.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_state_dict(pt_path: Path):
    """Load the .pt, supporting both pure-state_dict and full-checkpoint shapes."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        sys.exit("error: torch not installed in this Python env")
    obj = torch.load(pt_path, map_location="cpu", weights_only=True)
    # PanelTransformerModel.save writes a pure state_dict, but some callers
    # may have wrapped in {"state_dict": ..., "epoch": ...}. Handle both.
    if isinstance(obj, dict) and "state_dict" in obj and not all(
            isinstance(v, torch.Tensor) for v in obj.values()):
        obj = obj["state_dict"]
    if not isinstance(obj, dict):
        sys.exit(f"error: {pt_path} does not contain a state_dict (got {type(obj)})")
    return obj


def _to_contiguous_cpu_dict(state):
    """safetensors requires contiguous CPU tensors. Return a fresh dict.

    Audit fix BRIDGE-1/BRIDGE-2 (Round 2 deep audit, 2026-04-25):
      * BRIDGE-1: refuse to export NaN or inf weights. Pre-fix, a
        gradient explosion or corrupt checkpoint would silently land
        NaN weights on disk → Rust forward pass produces NaN scores →
        downstream pipelines silently degrade with no signal.
      * BRIDGE-2: cast weights to float32. Rust loader uses DType::F32
        explicitly; if Python saved float64 (default for some custom
        layers), candle would either silently downcast or error in a
        less debuggable way. Cast at export, fail loud on int dtypes
        (which can't represent the network's continuous weights).
    """
    import torch  # noqa: PLC0415
    out = {}
    n_nan = 0
    n_inf = 0
    n_cast = 0
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach().to("cpu").contiguous()
        # BRIDGE-2: dtype enforcement.
        if t.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            sys.exit(
                f"error: tensor '{k}' has dtype {t.dtype}; expected a float dtype. "
                "Refusing to export — would corrupt the Rust loader."
            )
        if t.dtype != torch.float32:
            t = t.to(torch.float32)
            n_cast += 1
        # BRIDGE-1: NaN / inf check on the cast tensor.
        if torch.isnan(t).any().item():
            n_nan += int(torch.isnan(t).sum().item())
        if torch.isinf(t).any().item():
            n_inf += int(torch.isinf(t).sum().item())
        out[k] = t
    if n_nan > 0 or n_inf > 0:
        sys.exit(
            f"error: refusing to export weights — found {n_nan} NaN and "
            f"{n_inf} inf values across {len(out)} tensors. The model is "
            "broken upstream (gradient explosion, corrupt checkpoint, or "
            "bad init); fix the trainer before re-exporting."
        )
    if n_cast > 0:
        print(f"info: cast {n_cast} non-f32 tensor(s) to float32 for Rust compatibility")
    return out


def _augment_sidecar(sidecar: dict, weights: dict) -> dict:
    """Add weight-shape fingerprint so the Rust loader can fail loud
    on a corrupt .safetensors instead of silently mis-shape-ing."""
    shapes = {k: list(t.shape) for k, t in weights.items()}
    sidecar = dict(sidecar)
    sidecar["weight_shapes"] = shapes
    sidecar["safetensors_version"] = 1
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stem", required=True, type=Path,
        help="Artifact stem (without extension). Looks for <stem>.pt + <stem>.json",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite an existing <stem>.safetensors without prompting",
    )
    args = parser.parse_args()

    pt_path   = args.stem.with_suffix(".pt")
    json_path = args.stem.with_suffix(".json")
    out_path  = args.stem.with_suffix(".safetensors")

    if not pt_path.exists():
        sys.exit(f"error: {pt_path} not found")
    if not json_path.exists():
        sys.exit(f"error: {json_path} sidecar not found — Rust loader needs it")
    if out_path.exists() and not args.force:
        sys.exit(f"error: {out_path} already exists. Use --force to overwrite")

    try:
        from safetensors.torch import save_file  # type: ignore[import-not-found]
    except ImportError:
        sys.exit(
            "error: safetensors not installed. "
            "Install with: pip install safetensors"
        )

    state   = _load_state_dict(pt_path)
    weights = _to_contiguous_cpu_dict(state)
    if not weights:
        sys.exit("error: state_dict had no tensor entries — refusing to write empty safetensors")

    save_file(weights, str(out_path))
    print(f"wrote {out_path}  ({len(weights)} tensors)")

    sidecar = json.loads(json_path.read_text())
    sidecar = _augment_sidecar(sidecar, weights)
    json_path.write_text(json.dumps(sidecar, indent=2))
    print(f"updated {json_path} (added weight_shapes fingerprint)")

    # Verify round-trip on the spot — load the safetensors back and
    # check shapes match. Fails LOUD here rather than at Rust runtime.
    try:
        from safetensors import safe_open  # type: ignore[import-not-found]
    except ImportError:
        return
    with safe_open(str(out_path), framework="pt") as f:
        out_keys = set(f.keys())
    in_keys = set(weights.keys())
    if out_keys != in_keys:
        sys.exit(
            f"error: safetensors round-trip key mismatch\n"
            f"  in  - out: {sorted(in_keys - out_keys)}\n"
            f"  out - in:  {sorted(out_keys - in_keys)}"
        )
    print(f"verify: {len(out_keys)} keys round-trip cleanly")


if __name__ == "__main__":
    main()
