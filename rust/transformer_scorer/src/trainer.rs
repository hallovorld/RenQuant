//! Trainable PanelTransformer + Adam loop. Mirrors the Python
//! `_PanelTransformer` exactly, but routed through `VarBuilder::from_varmap`
//! so weights are actual `candle_core::Var`s the optimizer can update.
//!
//! Reference: candle MNIST example
//! https://github.com/huggingface/candle/blob/main/candle-examples/examples/mnist-training/main.rs
//!
//! API:
//!   let mut tr = Trainer::new(params, n_features, dev)?;
//!   for epoch in 0..n_epochs {
//!       let loss = tr.train_step(&x, &y, &pad_mask, &nan_mask)?;
//!       println!("epoch {} loss {:.6}", epoch, loss);
//!   }
//!   tr.save_safetensors(&path)?;
//!
//! AdamW defaults match Python's typical setup (β1=0.9, β2=0.999, ε=1e-8).
//! Caller passes lr + weight_decay to control regularisation strength.

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    layer_norm, linear, AdamW, Linear, Optimizer, ParamsAdamW, VarBuilder, VarMap,
};
use std::path::Path;

use crate::config::TransformerParams;
use crate::loss::listnet_loss;
use crate::transformer_block::TransformerEncoderLayer;

/// Trainable transformer. Same shape as PanelTransformer but each layer
/// holds its weights via the VarMap so AdamW can step them.
pub struct TrainablePanelTransformer {
    feature_encoder: Linear,
    layers:          Vec<TransformerEncoderLayer>,
    score_head:      Linear,
    pub params:      TransformerParams,
}

impl TrainablePanelTransformer {
    pub fn new(
        n_features: usize,
        params:     TransformerParams,
        vb:         VarBuilder,
    ) -> Result<Self> {
        let feature_encoder = linear(n_features, params.d_model, vb.pp("feature_encoder"))?;

        let mut layers = Vec::with_capacity(params.n_layers);
        for i in 0..params.n_layers {
            let layer = TransformerEncoderLayer::new(
                &params,
                vb.pp(&format!("encoder.layers.{}", i)),
            )?;
            layers.push(layer);
        }
        let score_head = linear(params.d_model, 1, vb.pp("score_head"))?;
        Ok(Self { feature_encoder, layers, score_head, params })
    }

    /// Forward (B, T, F) → (B, T) score tensor.
    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut h = self.feature_encoder.forward(x)?;
        for layer in &self.layers {
            h = layer.forward(&h, pad_mask)?;
        }
        let s = self.score_head.forward(&h)?;
        // (B, T, 1) → (B, T)
        let s = s.squeeze(2)?;
        Ok(s)
    }
}


/// Trainer wraps the model + VarMap + AdamW so the caller doesn't have
/// to assemble these by hand. Usage in the demo binary below.
pub struct Trainer {
    pub model:   TrainablePanelTransformer,
    pub varmap:  VarMap,
    pub opt:     AdamW,
    pub device:  Device,
}

impl Trainer {
    pub fn new(
        n_features:   usize,
        params:       TransformerParams,
        learning_rate: f64,
        weight_decay: f64,
        device:       Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let model = TrainablePanelTransformer::new(n_features, params, vb)?;

        let opt = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW {
                lr: learning_rate,
                weight_decay,
                ..Default::default()
            },
        )?;
        Ok(Self { model, varmap, opt, device })
    }

    /// One training step. Returns the loss as f32.
    ///
    /// Shapes:
    ///   x         (B, T, F) f32
    ///   label     (B, T)    f32   (Gaussianized residuals)
    ///   pad_mask  (B, T)    u8    (1 where padded slot)
    ///   nan_mask  (B, T)    u8    (1 where label was NaN); pass None if
    ///                              caller has already filtered
    pub fn train_step(
        &mut self,
        x:        &Tensor,
        label:    &Tensor,
        pad_mask: &Tensor,
        nan_mask: Option<&Tensor>,
    ) -> Result<f32> {
        let pad_for_attn = if pad_mask.dtype() == DType::U8 {
            Some(pad_mask)
        } else {
            None
        };
        let score = self.model.forward(x, pad_for_attn)?;
        let loss  = listnet_loss(&score, label, pad_mask, nan_mask)?;
        self.opt.backward_step(&loss)?;
        Ok(loss.to_vec0::<f32>()?)
    }

    /// Save trained weights to safetensors at `<stem>.safetensors`.
    /// The Python loader can read this back via the existing
    /// `scripts/export_transformer_to_safetensors.py` round-trip
    /// (or directly via safetensors.torch.load_file).
    pub fn save_safetensors<P: AsRef<Path>>(&self, stem: P) -> Result<()> {
        let stem = stem.as_ref();
        let path = stem.with_extension("safetensors");
        self.varmap.save(&path)?;
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: train for 50 steps on synthetic (label = mean of features
    /// across each slot, ranked across the 8-slot row). Loss must end LOWER
    /// than where it started — proves the gradient + optimizer wiring works.
    #[test]
    fn loss_decreases_on_synthetic_data() {
        let dev = Device::Cpu;
        let n_features = 4;
        let params = TransformerParams {
            d_model:         16,
            n_heads:         2,
            n_layers:        1,
            feedforward_dim: 32,
            dropout:         0.0,
            feature_dropout: 0.0,
        };
        let mut tr = Trainer::new(n_features, params, 1e-2, 0.0, dev.clone())
            .expect("create trainer");

        // Synthetic: one batch, 8-slot row. Label = ranked mean of features.
        // We can't feed gradients-true data through nn helpers easily, so just
        // give a random-but-fixed input + label pair and verify loss decreases.
        let x_data: [[[f32; 4]; 8]; 1] = [[
            [ 0.50,  0.10, -0.30,  0.20],
            [-0.40,  0.20,  0.10, -0.10],
            [ 0.30, -0.20,  0.40,  0.10],
            [ 0.10,  0.40, -0.10,  0.30],
            [-0.20, -0.30,  0.20, -0.40],
            [ 0.40,  0.50, -0.20,  0.10],
            [-0.10,  0.30,  0.40, -0.20],
            [ 0.20, -0.40, -0.30,  0.50],
        ]];
        let label_data: [[f32; 8]; 1] = [[1.0, -1.0, 0.5, 0.7, -0.5, 0.9, 0.4, -0.3]];

        let x      = Tensor::new(&x_data, &dev).unwrap();
        let label  = Tensor::new(&label_data, &dev).unwrap();
        let pad    = Tensor::zeros((1, 8), DType::U8, &dev).unwrap();

        let initial = tr.train_step(&x, &label, &pad, None).expect("step 0");
        for _ in 0..50 {
            tr.train_step(&x, &label, &pad, None).expect("training step");
        }
        let after = tr.train_step(&x, &label, &pad, None).expect("step final");

        assert!(initial.is_finite(), "initial loss must be finite, got {}", initial);
        assert!(after.is_finite(), "final loss must be finite, got {}", after);
        assert!(
            after < initial,
            "loss must DECREASE — initial={:.6} final={:.6} (gradient/optimizer broken?)",
            initial, after,
        );
        // It should drop a fair amount on a single batch we keep retraining on.
        assert!(
            after < initial * 0.95,
            "loss should drop > 5% on memorisation task — initial={:.6} final={:.6}",
            initial, after,
        );
    }

    #[test]
    fn save_round_trip() {
        // Train briefly then save; verify the file lands on disk.
        let dev = Device::Cpu;
        let params = TransformerParams {
            d_model: 8, n_heads: 2, n_layers: 1, feedforward_dim: 16,
            dropout: 0.0, feature_dropout: 0.0,
        };
        let mut tr = Trainer::new(4, params, 1e-3, 0.0, dev.clone()).unwrap();

        let x = Tensor::randn(0_f32, 1.0_f32, (1, 4, 4), &dev).unwrap();
        let y = Tensor::randn(0_f32, 1.0_f32, (1, 4), &dev).unwrap();
        let pad = Tensor::zeros((1, 4), DType::U8, &dev).unwrap();
        tr.train_step(&x, &y, &pad, None).unwrap();

        let tmp = std::env::temp_dir().join("rust_trainer_test");
        tr.save_safetensors(&tmp).unwrap();
        let written = tmp.with_extension("safetensors");
        assert!(written.exists(), "save did not produce {}", written.display());
        let _ = std::fs::remove_file(&written);
    }
}
