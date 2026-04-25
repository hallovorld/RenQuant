//! One transformer encoder block — pre-LN MultiHeadSelfAttention + FFN.
//!
//! PyTorch's `nn.TransformerEncoderLayer(batch_first=True, activation="gelu")`
//! emits the post-LN formulation by default. We mirror that exactly so
//! safetensor weights round-trip without renaming.
//!
//! Weight names follow PyTorch's convention (so the safetensors export
//! step doesn't have to rename):
//!
//!   self_attn.in_proj_weight     (3*d_model, d_model)
//!   self_attn.in_proj_bias       (3*d_model,)
//!   self_attn.out_proj.weight    (d_model, d_model)
//!   self_attn.out_proj.bias      (d_model,)
//!   linear1.weight               (ff_dim, d_model)
//!   linear1.bias                 (ff_dim,)
//!   linear2.weight               (d_model, ff_dim)
//!   linear2.bias                 (d_model,)
//!   norm1.weight, norm1.bias     (d_model,)
//!   norm2.weight, norm2.bias     (d_model,)
//!
//! NOTE: this is a from-scratch attention impl rather than reaching into
//! candle-transformers; we want full control of the mask path so behaviour
//! matches the Python golden output bit-for-bit.

use anyhow::Result;
use candle_core::{Module, Tensor};
use candle_nn::{layer_norm, ops::softmax, LayerNorm, Linear, VarBuilder};

use crate::config::TransformerParams;

pub struct TransformerEncoderLayer {
    // Multi-head self-attention (combined QKV projection, like PyTorch).
    in_proj_w:  Tensor,   // (3*d_model, d_model)
    in_proj_b:  Tensor,   // (3*d_model,)
    out_proj:   Linear,
    // Feed-forward.
    linear1: Linear,
    linear2: Linear,
    // Layer norms.
    norm1: LayerNorm,
    norm2: LayerNorm,
    // Dims.
    d_model: usize,
    n_heads: usize,
    head_d:  usize,
}

impl TransformerEncoderLayer {
    pub fn new(p: &TransformerParams, vb: VarBuilder) -> Result<Self> {
        assert!(p.d_model % p.n_heads == 0, "d_model must be divisible by n_heads");
        let head_d = p.d_model / p.n_heads;

        // Combined QKV — PyTorch packs as (3*d_model, d_model).
        let in_proj_w = vb.pp("self_attn").get((3 * p.d_model, p.d_model), "in_proj_weight")?;
        let in_proj_b = vb.pp("self_attn").get(3 * p.d_model, "in_proj_bias")?;
        // Output proj — Linear with bias.
        let out_proj  = candle_nn::linear(p.d_model, p.d_model, vb.pp("self_attn.out_proj"))?;

        let linear1 = candle_nn::linear(p.d_model, p.feedforward_dim, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(p.feedforward_dim, p.d_model, vb.pp("linear2"))?;

        // PyTorch's TransformerEncoderLayer uses `eps=1e-5` LayerNorm by default.
        let norm1 = layer_norm(p.d_model, 1e-5_f64, vb.pp("norm1"))?;
        let norm2 = layer_norm(p.d_model, 1e-5_f64, vb.pp("norm2"))?;

        Ok(Self {
            in_proj_w,
            in_proj_b,
            out_proj,
            linear1,
            linear2,
            norm1,
            norm2,
            d_model: p.d_model,
            n_heads: p.n_heads,
            head_d,
        })
    }

    /// Forward pass. `x`: (B, T, d_model). `pad_mask`: optional (B, T)
    /// where `true` = padded (won't be attended to).
    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        // ── Self-attention ────────────────────────────────────────────
        let attn_out = self.self_attention(x, pad_mask)?;
        // Residual + LayerNorm (post-LN — PyTorch default).
        let x1 = (x + attn_out)?;
        let x1 = self.norm1.forward(&x1)?;

        // ── Feed-forward ──────────────────────────────────────────────
        let ff_in  = self.linear1.forward(&x1)?;
        let ff_in  = ff_in.gelu()?;
        let ff_out = self.linear2.forward(&ff_in)?;
        let x2 = (x1 + ff_out)?;
        let x2 = self.norm2.forward(&x2)?;
        Ok(x2)
    }

    /// Multi-head self-attention with the combined QKV projection.
    fn self_attention(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, t, d) = x.dims3()?;
        debug_assert_eq!(d, self.d_model);

        // Project to (B, T, 3*d_model).
        let qkv = x.broadcast_matmul(&self.in_proj_w.t()?)?;
        let qkv = qkv.broadcast_add(&self.in_proj_b)?;
        // Split along last dim into Q, K, V each (B, T, d_model).
        let chunks = qkv.chunk(3, 2)?;
        let q = &chunks[0];
        let k = &chunks[1];
        let v = &chunks[2];

        // Reshape to (B, n_heads, T, head_d).
        let to_heads = |t_in: &Tensor| -> Result<Tensor> {
            let r = t_in
                .reshape((b, t, self.n_heads, self.head_d))?
                .transpose(1, 2)?
                .contiguous()?;
            Ok(r)
        };
        let q = to_heads(q)?;
        let k = to_heads(k)?;
        let v = to_heads(v)?;

        // Scores = Q · K^T / sqrt(head_d)   shape (B, H, T, T).
        let scale = 1.0_f64 / (self.head_d as f64).sqrt();
        let kt = k.transpose(2, 3)?.contiguous()?;
        let scores = q.matmul(&kt)?;
        let scores = (scores * scale)?;

        // Apply pad mask: mask shape (B, T), broadcast to (B, 1, 1, T).
        let scores = if let Some(m) = pad_mask {
            let m = m
                .to_dtype(scores.dtype())?
                .reshape((b, 1, 1, t))?;
            // mask=1 → set score = -inf; PyTorch convention.
            let neg_inf = Tensor::full(f32::NEG_INFINITY, scores.shape(), scores.device())?;
            scores.broadcast_add(&(m * &neg_inf)?)?
        } else {
            scores
        };

        let weights = softmax(&scores, candle_core::D::Minus1)?;
        // Attention out: weights · V  → (B, H, T, head_d).
        let out = weights.matmul(&v)?;
        // Reshape back: (B, T, d_model).
        let out = out.transpose(1, 2)?.contiguous()?.reshape((b, t, d))?;
        // Output projection.
        let out = self.out_proj.forward(&out)?;
        Ok(out)
    }
}
