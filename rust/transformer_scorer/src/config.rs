//! Sidecar JSON schema — mirrors the Python artifact metadata.

use serde::{Deserialize, Serialize};

/// Hyperparameters baked into the artifact (must match the Python
/// `TransformerParams` dataclass at training time).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TransformerParams {
    pub d_model:         usize,
    pub n_heads:         usize,
    pub n_layers:        usize,
    #[serde(default = "default_ff_dim")]
    pub feedforward_dim: usize,
    #[serde(default)]
    pub dropout:         f32,
    #[serde(default)]
    pub feature_dropout: f32,
}

fn default_ff_dim() -> usize { 256 }

/// Top-level sidecar JSON. The Python side (PanelTransformerModel.save)
/// writes this verbatim alongside the .pt → .safetensors file.
///
/// Fields beyond `feature_cols` + `params` are operationally relevant
/// to the trainer but not the inference path; they're retained for
/// audit (so a `.json` round-trip preserves all upstream information).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ScorerArtifact {
    pub feature_cols: Vec<String>,
    pub params:       TransformerParams,
    #[serde(default)]
    pub trained_on:   Option<String>,   // ISO date
    #[serde(default)]
    pub n_features:   Option<usize>,
    #[serde(default)]
    pub n_dates:      Option<usize>,
    #[serde(default)]
    pub n_tickers:    Option<usize>,
    #[serde(default)]
    pub backend:      Option<String>,   // "transformer"
    #[serde(default)]
    pub commit_sha:   Option<String>,
}
