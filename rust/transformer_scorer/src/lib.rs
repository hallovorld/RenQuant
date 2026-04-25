//! Panel transformer scorer — Rust port of `training_panel/transformer_model.py`.
//!
//! Architecture mirrors the Python `_PanelTransformer`:
//!
//! ```text
//!   x: (B, T, F)   pad_mask: (B, T)   →   scores: (B, T)
//!
//!   feature_encoder : Linear(F, d_model)
//!   encoder         : N × TransformerEncoderLayer(d_model, n_heads, ff, dropout, gelu)
//!   score_head      : Linear(d_model, 1)
//! ```
//!
//! The loaded artifact is a pair:
//!   * ``model.safetensors`` — weights (converted from PyTorch `.pt` via
//!     `scripts/export_transformer_to_safetensors.py`)
//!   * ``model.json``        — sidecar with feature_cols + hyperparams
//!
//! Inference shape: per-bar use case feeds a single date-group of N tickers
//! at once (B=1, T=N), mirroring `TransformerPanelScorer.score()`.

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{linear, ops, Activation, Linear, VarBuilder};
use serde::{Deserialize, Serialize};
use std::path::Path;

pub mod config;
pub mod cv;
pub mod dataset;
pub mod loss;
pub mod metrics;
pub mod trainer;
pub mod transformer_block;

pub use config::{ScorerArtifact, TransformerParams};
pub use transformer_block::TransformerEncoderLayer;

/// One forward-pass-ready model.
pub struct PanelTransformer {
    feature_encoder: Linear,
    layers:          Vec<TransformerEncoderLayer>,
    score_head:      Linear,
    pub feature_cols: Vec<String>,
    pub params:       TransformerParams,
    device:           Device,
}

impl PanelTransformer {
    /// Load a `.safetensors + .json` pair.
    pub fn load(stem: impl AsRef<Path>, device: Device) -> Result<Self> {
        let stem = stem.as_ref();
        let json_path = stem.with_extension("json");
        let st_path   = stem.with_extension("safetensors");

        let artifact: ScorerArtifact = serde_json::from_str(
            &std::fs::read_to_string(&json_path).with_context(|| {
                format!("reading sidecar JSON {}", json_path.display())
            })?,
        )
        .with_context(|| format!("parsing sidecar JSON {}", json_path.display()))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&st_path], DType::F32, &device)?
        };

        let p = &artifact.params;
        let n_features = artifact.feature_cols.len();

        // Linear(F → d_model) with bias.
        let feature_encoder = linear(n_features, p.d_model, vb.pp("feature_encoder"))?;

        // N transformer encoder layers.
        let mut layers = Vec::with_capacity(p.n_layers);
        for i in 0..p.n_layers {
            let layer = TransformerEncoderLayer::new(
                p,
                vb.pp(&format!("encoder.layers.{}", i)),
            )?;
            layers.push(layer);
        }

        // Linear(d_model → 1).
        let score_head = linear(p.d_model, 1, vb.pp("score_head"))?;

        Ok(Self {
            feature_encoder,
            layers,
            score_head,
            feature_cols: artifact.feature_cols.clone(),
            params: artifact.params.clone(),
            device,
        })
    }

    /// Forward pass for a single date-group.
    ///
    /// `x` shape: (T, F) — `T` tickers, `F` features (must match
    /// `feature_cols.len()`). Returns scores of length `T`.
    pub fn forward_single_group(&self, x: &Tensor) -> Result<Tensor> {
        // Add batch dimension: (T, F) → (1, T, F)
        let x = x.unsqueeze(0)?;
        // Optional feature dropout is identity at inference (eval mode).
        let mut h = self.feature_encoder.forward(&x)?;          // (1, T, d)
        // No padding at single-group inference, but we still need the
        // mask shape that matches the layer's signature. None means
        // "no padding, attend everything".
        for layer in &self.layers {
            h = layer.forward(&h, None)?;
        }
        let s = self.score_head.forward(&h)?;                   // (1, T, 1)
        let s = s.squeeze(2)?.squeeze(0)?;                      // (T,)
        Ok(s)
    }
}

/// Public scoring API mirrors `TransformerPanelScorer.score(matrix)`.
pub struct PanelScorer {
    model: PanelTransformer,
}

impl PanelScorer {
    pub fn load(stem: impl AsRef<Path>, device: Device) -> Result<Self> {
        Ok(Self {
            model: PanelTransformer::load(stem, device)?,
        })
    }

    pub fn feature_cols(&self) -> &[String] {
        &self.model.feature_cols
    }

    /// Score a (T × F) feature matrix. `feature_names` is the column
    /// order in `matrix` and MUST be a superset of `self.feature_cols()`
    /// — we re-index by name to match the model's feature order.
    pub fn score(
        &self,
        matrix: &ndarray::Array2<f32>,
        feature_names: &[String],
    ) -> Result<Vec<f32>> {
        // Audit fix RUST-R3-22 (Round 2 deep audit, 2026-04-25): mirror
        // the Python `TransformerPanelScorer.score()` empty-input early
        // return. Without this, calling forward on a (0, F) tensor hits
        // the encoder's softmax-over-empty-axis path and panics with
        // "empty tensor for reduce". Empty input is a valid no-op (no
        // candidates this bar); silent zero-row pass-through is the
        // documented Python behaviour, port it here.
        if matrix.is_empty() || matrix.shape()[0] == 0 {
            // Still validate that requested columns exist — fail loud
            // on bad caller config even when there's no data to score.
            for want in &self.model.feature_cols {
                if !feature_names.iter().any(|n| n == want) {
                    return Err(anyhow!("missing feature column '{}'", want));
                }
            }
            return Ok(Vec::new());
        }

        // Build a column-permutation index.
        let mut idx = Vec::with_capacity(self.model.feature_cols.len());
        for want in &self.model.feature_cols {
            let pos = feature_names
                .iter()
                .position(|n| n == want)
                .ok_or_else(|| anyhow!("missing feature column '{}'", want))?;
            idx.push(pos);
        }
        let t = matrix.shape()[0];
        let f = self.model.feature_cols.len();
        let mut buf = Vec::with_capacity(t * f);
        for row in 0..t {
            for &col in &idx {
                buf.push(matrix[[row, col]]);
            }
        }
        let x = Tensor::from_vec(buf, (t, f), &self.model.device)?;
        let s = self.model.forward_single_group(&x)?;
        let v = s.to_vec1::<f32>()?;
        Ok(v)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ScoringResult {
    pub tickers: Vec<String>,
    pub scores:  Vec<f32>,
}
