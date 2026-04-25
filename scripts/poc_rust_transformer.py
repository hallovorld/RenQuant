#!/usr/bin/env python
"""Tiny POC: build synthetic transformer + export safetensors + run Rust scorer.

End-to-end demo so we can sanity-check the Rust port without needing
real production data. Run from repo root:

    python scripts/poc_rust_transformer.py

What this does:
  1. Builds a tiny PyTorch transformer matching the scaffolded Rust
     architecture (d_model=8, n_heads=2, n_layers=1).
  2. Initializes weights deterministically (manual seed) so Python
     and Rust outputs are bit-comparable.
  3. Saves <stem>.pt + <stem>.json to /tmp.
  4. Calls scripts/export_transformer_to_safetensors.py to make
     <stem>.safetensors.
  5. Generates a feature CSV with 4 fake tickers.
  6. Computes the Python forward-pass output for each ticker.
  7. Prints both the file paths and the Python scores so a follow-up
     `cargo run` invocation can compare.

The Rust binary command will be:
    cargo run --release --bin score-panel -- \\
        --artifact /tmp/poc_panel \\
        --input    /tmp/poc_features.csv \\
        --output   /tmp/poc_scores.csv
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
STEM = Path("/tmp/poc_panel")
CSV  = Path("/tmp/poc_features.csv")


def main() -> None:
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        sys.exit("error: torch not installed in this Python env")

    # ── 1. Build the matching architecture ───────────────────────────
    # Identical names to the Rust loader's expectations.
    torch.manual_seed(0)
    n_features = 6
    n_tickers  = 4
    d_model    = 8
    n_heads    = 2
    n_layers   = 1
    ff_dim     = 16

    feature_encoder = nn.Linear(n_features, d_model)
    enc_layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=n_heads, dim_feedforward=ff_dim,
        dropout=0.0, batch_first=True, activation="gelu",
    )
    encoder    = nn.TransformerEncoder(enc_layer, num_layers=n_layers,
                                        enable_nested_tensor=False)
    score_head = nn.Linear(d_model, 1)
    model = nn.ModuleDict({
        "feature_encoder": feature_encoder,
        "encoder":         encoder,
        "score_head":      score_head,
    })

    # Deterministic-ish weight init: re-seed and use small uniform.
    torch.manual_seed(7)
    for p in model.parameters():
        p.data.uniform_(-0.1, 0.1)
    model.eval()

    # ── 2. Sample input + Python forward ─────────────────────────────
    feature_cols = [f"f{i}" for i in range(n_features)]
    tickers      = [f"T{i}" for i in range(n_tickers)]
    np.random.seed(42)
    X_np = np.random.randn(n_tickers, n_features).astype(np.float32)

    with torch.no_grad():
        x  = torch.from_numpy(X_np).unsqueeze(0)              # (1, T, F)
        h  = model["feature_encoder"](x)                      # (1, T, d)
        h  = model["encoder"](h)                              # (1, T, d)
        s  = model["score_head"](h).squeeze(-1).squeeze(0)    # (T,)
        py_scores = s.numpy().tolist()

    # ── 3. Save .pt + .json ──────────────────────────────────────────
    state = {}
    state["feature_encoder.weight"] = feature_encoder.weight.data.contiguous()
    state["feature_encoder.bias"]   = feature_encoder.bias.data.contiguous()
    layer = encoder.layers[0]
    state["encoder.layers.0.self_attn.in_proj_weight"]  = layer.self_attn.in_proj_weight.data.contiguous()
    state["encoder.layers.0.self_attn.in_proj_bias"]    = layer.self_attn.in_proj_bias.data.contiguous()
    state["encoder.layers.0.self_attn.out_proj.weight"] = layer.self_attn.out_proj.weight.data.contiguous()
    state["encoder.layers.0.self_attn.out_proj.bias"]   = layer.self_attn.out_proj.bias.data.contiguous()
    state["encoder.layers.0.linear1.weight"]            = layer.linear1.weight.data.contiguous()
    state["encoder.layers.0.linear1.bias"]              = layer.linear1.bias.data.contiguous()
    state["encoder.layers.0.linear2.weight"]            = layer.linear2.weight.data.contiguous()
    state["encoder.layers.0.linear2.bias"]              = layer.linear2.bias.data.contiguous()
    state["encoder.layers.0.norm1.weight"]              = layer.norm1.weight.data.contiguous()
    state["encoder.layers.0.norm1.bias"]                = layer.norm1.bias.data.contiguous()
    state["encoder.layers.0.norm2.weight"]              = layer.norm2.weight.data.contiguous()
    state["encoder.layers.0.norm2.bias"]                = layer.norm2.bias.data.contiguous()
    state["score_head.weight"]                          = score_head.weight.data.contiguous()
    state["score_head.bias"]                            = score_head.bias.data.contiguous()

    pt_path   = STEM.with_suffix(".pt")
    json_path = STEM.with_suffix(".json")
    torch.save(state, pt_path)
    json_path.write_text(json.dumps({
        "feature_cols": feature_cols,
        "params": {
            "d_model":         d_model,
            "n_heads":         n_heads,
            "n_layers":        n_layers,
            "feedforward_dim": ff_dim,
            "dropout":         0.0,
            "feature_dropout": 0.0,
        },
        "trained_on": "POC",
        "n_features": n_features,
        "n_tickers":  n_tickers,
        "n_dates":    1,
        "backend":    "transformer",
    }, indent=2))

    # ── 4. Convert to safetensors ────────────────────────────────────
    res = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "export_transformer_to_safetensors.py"),
         "--stem", str(STEM), "--force"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("STDOUT:", res.stdout)
        print("STDERR:", res.stderr)
        sys.exit("error: safetensors export failed")
    print(res.stdout.strip())

    # ── 5. Write feature CSV ─────────────────────────────────────────
    with CSV.open("w") as f:
        f.write("ticker," + ",".join(feature_cols) + "\n")
        for t, row in zip(tickers, X_np):
            f.write(t + "," + ",".join(f"{v:.6f}" for v in row) + "\n")

    print("\n=== POC artefacts ready ===")
    print(f"  artifact stem:  {STEM}")
    print(f"  feature CSV:    {CSV}")
    print(f"  Python scores:  {[f'{v:+.6f}' for v in py_scores]}")
    print(f"  tickers:        {tickers}")
    print()
    print("Now build + run the Rust scorer:")
    print(f"  cd {REPO/'rust'}")
    print( "  cargo run --release --bin score-panel -- \\")
    print(f"      --artifact {STEM} \\")
    print(f"      --input    {CSV} \\")
    print( "      --output   /tmp/poc_scores.csv")
    print()
    print("Compare /tmp/poc_scores.csv vs the Python scores above —")
    print("absolute diff should be < 1e-4 per ticker.")


if __name__ == "__main__":
    main()
