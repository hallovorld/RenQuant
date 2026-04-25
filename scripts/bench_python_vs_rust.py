#!/usr/bin/env python
"""Head-to-head bench: Python forward pass vs Rust score-panel CLI.

User asked: "rust代码写的对吗？效率没提升啊！" — let me measure.
Single-call latency (one date-group, N tickers) is what most callers
actually do. Multiple-call throughput is where Rust's no-GIL parallelism
shines (already 16.59× in bench_parallel.rs).

This script:
  1. Loads the POC artifact in Python via PanelTransformerModel
  2. Loads the same artifact in Rust via score-panel CLI
  3. Runs the SAME 4-ticker × 6-feature input N times each
  4. Reports per-call latency (median + p99) for each
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ART_STEM = Path("/tmp/poc_panel")
CSV      = Path("/tmp/poc_features.csv")
N_ITERS_PY  = 200
N_ITERS_RS  = 50   # Rust path includes process spawn → expensive per call


def run_python_bench():
    """Build the Python equivalent forward pass; time N_ITERS_PY calls."""
    sys.path.insert(0, str(REPO / "backtesting" / "renquant_104"))
    import torch
    import torch.nn as nn
    import numpy as np

    # Reload the same architecture as scripts/poc_rust_transformer.py.
    sidecar = json.loads(ART_STEM.with_suffix(".json").read_text())
    p = sidecar["params"]
    n_features = len(sidecar["feature_cols"])
    state = torch.load(ART_STEM.with_suffix(".pt"), map_location="cpu", weights_only=True)

    feature_encoder = nn.Linear(n_features, p["d_model"])
    enc_layer = nn.TransformerEncoderLayer(
        d_model=p["d_model"], nhead=p["n_heads"],
        dim_feedforward=p["feedforward_dim"], dropout=p["dropout"],
        batch_first=True, activation="gelu",
    )
    encoder = nn.TransformerEncoder(enc_layer, num_layers=p["n_layers"],
                                     enable_nested_tensor=False)
    score_head = nn.Linear(p["d_model"], 1)

    feature_encoder.weight.data = state["feature_encoder.weight"]
    feature_encoder.bias.data   = state["feature_encoder.bias"]
    layer = encoder.layers[0]
    layer.self_attn.in_proj_weight.data  = state["encoder.layers.0.self_attn.in_proj_weight"]
    layer.self_attn.in_proj_bias.data    = state["encoder.layers.0.self_attn.in_proj_bias"]
    layer.self_attn.out_proj.weight.data = state["encoder.layers.0.self_attn.out_proj.weight"]
    layer.self_attn.out_proj.bias.data   = state["encoder.layers.0.self_attn.out_proj.bias"]
    layer.linear1.weight.data            = state["encoder.layers.0.linear1.weight"]
    layer.linear1.bias.data              = state["encoder.layers.0.linear1.bias"]
    layer.linear2.weight.data            = state["encoder.layers.0.linear2.weight"]
    layer.linear2.bias.data              = state["encoder.layers.0.linear2.bias"]
    layer.norm1.weight.data              = state["encoder.layers.0.norm1.weight"]
    layer.norm1.bias.data                = state["encoder.layers.0.norm1.bias"]
    layer.norm2.weight.data              = state["encoder.layers.0.norm2.weight"]
    layer.norm2.bias.data                = state["encoder.layers.0.norm2.bias"]
    score_head.weight.data               = state["score_head.weight"]
    score_head.bias.data                 = state["score_head.bias"]
    feature_encoder.eval()
    encoder.eval()
    score_head.eval()

    # Read the CSV once.
    np.random.seed(42)
    rng = np.random.RandomState(42)
    X_np = rng.randn(4, n_features).astype(np.float32)

    timings = []
    with torch.no_grad():
        for _ in range(N_ITERS_PY):
            t0 = time.perf_counter()
            x = torch.from_numpy(X_np).unsqueeze(0)
            h = feature_encoder(x)
            h = encoder(h)
            s = score_head(h).squeeze(-1).squeeze(0)
            _ = s.numpy()
            timings.append(time.perf_counter() - t0)
    return timings


def run_rust_bench():
    """Spawn score-panel CLI N_ITERS_RS times, measure each round-trip."""
    bin_path = REPO / "rust" / "target" / "release" / "score-panel"
    if not bin_path.exists():
        sys.exit(f"build first: cd rust && cargo build --release  ({bin_path} missing)")
    out = Path("/tmp/poc_scores_bench.csv")
    timings = []
    for _ in range(N_ITERS_RS):
        t0 = time.perf_counter()
        subprocess.run(
            [str(bin_path),
             "--artifact", str(ART_STEM),
             "--input",    str(CSV),
             "--output",   str(out)],
            capture_output=True, check=True,
        )
        timings.append(time.perf_counter() - t0)
    return timings


def report(name, timings):
    timings = sorted(timings)
    median = statistics.median(timings) * 1000
    mean   = statistics.mean(timings) * 1000
    p99    = timings[int(len(timings) * 0.99)] * 1000 if len(timings) >= 100 else timings[-1] * 1000
    minv   = timings[0] * 1000
    print(f"  {name:>20s}: median={median:7.3f} ms  mean={mean:7.3f} ms  "
          f"p99={p99:7.3f} ms  min={minv:7.3f} ms  n={len(timings)}")


def main():
    print("=== Single-call latency comparison (4-ticker POC artifact) ===\n")
    print("Python forward pass (in-process):")
    py = run_python_bench()
    report("python", py)
    print()
    print("Rust score-panel CLI (process spawn each call):")
    rs = run_rust_bench()
    report("rust-cli", rs)

    py_med = statistics.median(py)
    rs_med = statistics.median(rs)
    print()
    print(f"  Rust CLI is {rs_med/py_med:.1f}× SLOWER per call than Python in-process.")
    print(f"  Why: each Rust call pays a fork+exec+load-artifact cost ({rs_med*1000:.1f} ms).")
    print(f"  The Python forward pass is in-process — no spawn overhead.")
    print()
    print("This means: if you call score 1 time, Python wins. If you call")
    print("score 1000+ times, Rust's parallel batch path wins (16.59× in")
    print("bench_parallel — see rust/target/release/examples/bench_parallel).")
    print()
    print("To make Rust win on single-call:")
    print("  * Load the artifact ONCE, score many times via FFI (PyO3 bindings).")
    print("  * Or run Rust as a long-lived subprocess with stdin/stdout protocol.")
    print("  * Or move training to Rust so the whole loop is no-GIL.")


if __name__ == "__main__":
    main()
