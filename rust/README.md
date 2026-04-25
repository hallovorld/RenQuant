# RenQuant — Rust workspace

Pure-Rust port of the panel transformer scorer used by the renquant_104
strategy. Designed to live alongside the Python implementation: the Python
side stays the source of truth for training; this workspace owns
inference-time scoring + future Rust ports of hot kernels.

## Status

| Component                | State        | Notes                              |
|--------------------------|--------------|------------------------------------|
| transformer_scorer       | scaffolded   | matches Python `_PanelTransformer` |
| safetensors loader       | scaffolded   | needs the Python export script     |
| Metal (MPS-equivalent)   | feature flag | `cargo metal`                       |
| training (forward+back)  | not started  | Python remains the trainer         |
| sim/live integration     | not started  | parity tests come first            |

This is intentionally a parallel track — **no Python pipeline depends on
this Rust workspace yet**. Plan: validate scorer parity vs Python golden,
then opt-in via a `panel_ltr.scorer_backend = "rust"` config flag.

## Environment — professional, conda-style

This workspace uses the same "pin everything, idempotent install" model
as the Python conda env at the repo root:

| Concept                  | Python (conda)          | Rust here                              |
|--------------------------|-------------------------|----------------------------------------|
| Toolchain manager        | `conda` / `miniconda`   | `rustup`                                |
| Toolchain pin            | `environment.yml`       | `rust-toolchain.toml`                   |
| Package manifest         | `requirements.lock.txt` | `Cargo.toml` (workspace + crate)        |
| Reproducible lockfile    | `requirements.lock.txt` | `Cargo.lock` (committed)                |
| Per-project config       | `.condarc`              | `.cargo/config.toml`                    |
| One-shot installer       | `bash SETUP.md`         | `bash rust/setup.sh`                    |

The pinned channel (`1.81.0`) is auto-installed by rustup the first time
cargo is invoked from anywhere inside `rust/`, so **no global side effects**
on the user's system Rust install.

## First-time setup

```bash
cd rust/
bash setup.sh
```

That will:
1. Install rustup if absent (official `rustup-init.sh`, no system writes
   outside `$HOME/.cargo`).
2. Auto-install the pinned toolchain via `rust-toolchain.toml`.
3. Add rustfmt + clippy + rust-analyzer components.
4. `cargo fetch` to prefetch crates into the workspace cache.
5. `cargo check` and `cargo build --release` as a smoke test.

Re-run setup.sh any time — it's idempotent.

## Build / run

```bash
cd rust/
cargo dev          # debug build, all crates
cargo rls          # release build (default; fastest binary)
cargo metal        # release + Apple Metal backend (Mac only)
cargo test-all     # tests, all features
cargo lint         # clippy with -D warnings
cargo fmt-check    # rustfmt CI gate
```

Once built:

```bash
./target/release/score-panel \
    --artifact ../backtesting/renquant_104/artifacts/panel-transformer \
    --input    /tmp/panel_features.csv \
    --output   /tmp/panel_scores.csv
```

## Layout

```
rust/
  Cargo.toml                 workspace + shared dep versions
  rust-toolchain.toml        pinned channel + components
  .cargo/config.toml         build profile, aliases, registry config
  setup.sh                   one-shot installer (idempotent)
  README.md                  you are here
  transformer_scorer/        first crate
    Cargo.toml
    src/
      lib.rs                 PanelScorer + PanelTransformer top-level
      config.rs              sidecar JSON schema
      transformer_block.rs   from-scratch encoder layer (matches PyTorch
                             post-LN signature, weight names mirror torch)
      main.rs                `score-panel` CLI
```

## Artifact format

Two-file pair, written by the Python trainer (export script lives at
`scripts/export_transformer_to_safetensors.py` — TODO for next session):

| File                       | Format       | Source                          |
|----------------------------|--------------|---------------------------------|
| `<stem>.safetensors`       | safetensors  | Python `.pt` state_dict, converted |
| `<stem>.json`              | JSON         | feature_cols + TransformerParams + audit metadata |

The safetensors weight names mirror PyTorch's `nn.TransformerEncoderLayer`
exactly so the conversion is a no-rename round-trip:

```text
feature_encoder.weight        (d_model, F)
feature_encoder.bias          (d_model,)
encoder.layers.{i}.self_attn.in_proj_weight  (3*d, d)
encoder.layers.{i}.self_attn.in_proj_bias    (3*d,)
encoder.layers.{i}.self_attn.out_proj.weight (d, d)
encoder.layers.{i}.self_attn.out_proj.bias   (d,)
encoder.layers.{i}.linear1.weight            (ff, d)
encoder.layers.{i}.linear1.bias              (ff,)
encoder.layers.{i}.linear2.weight            (d, ff)
encoder.layers.{i}.linear2.bias              (d,)
encoder.layers.{i}.norm1.weight              (d,)
encoder.layers.{i}.norm1.bias                (d,)
encoder.layers.{i}.norm2.weight              (d,)
encoder.layers.{i}.norm2.bias                (d,)
score_head.weight                            (1, d)
score_head.bias                              (1,)
```

## What's NOT here yet

Tracked separately so we can ship the Rust scaffold first and parity-
validate later:

- [ ] `scripts/export_transformer_to_safetensors.py` — Python side converter.
- [ ] Golden tests: load Python+Rust on the same artifact, confirm scores
      match within ~1e-4 (FP32 tolerance) on synthetic + production inputs.
- [ ] `panel_ltr.scorer_backend = "rust"` config flag + adapter dispatch.
- [ ] Bench: Rust CLI vs Python `score()` end-to-end latency.
- [ ] Cross-compile target for Linux (`x86_64-unknown-linux-gnu`) once we
      have a real deploy target.

## Why this layout

- **Workspace, not a single crate** — leaves room for sister crates
  (e.g. `regime`, `sizing`) once the scorer parity is validated.
- **Pinned toolchain via rust-toolchain.toml** — every contributor builds
  with the exact same compiler, exactly like an environment.yml channel pin.
- **Cargo.lock committed** — reproducible binary builds. (Library-only
  crates conventionally exclude it; binary crates and applications include
  it. We have a binary, so include.)
- **`.cargo/config.toml` instead of env vars** — the build profile, link
  flags, and macOS deployment target all live in version control.
- **No system-wide Rust install required** — `setup.sh` only writes to
  `$HOME/.cargo`. Uninstalling is `rustup self uninstall`.
