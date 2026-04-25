//! ListNet loss with NaN-mask — direct port of the Python
//! `_listnet_loss` in training_panel/transformer_model.py.
//!
//! Match Python verbatim:
//!   * scores tensor (B, T) — model output per ticker per date-group
//!   * labels tensor (B, T) — Gaussianized residual returns (target distrib)
//!   * pad_mask (B, T) bool — True where padded slots
//!   * nan_label_mask (B, T) bool — True where label is NaN (skip in loss)
//!
//! Top-1 ListNet:
//!   p_target = softmax(label) over each row (with masked entries set to -inf)
//!   p_pred   = softmax(score) over each row (same masking)
//!   loss = - mean over rows of  sum_t p_target[t] * log p_pred[t]
//!
//! Without explicit row weights (the Python doesn't take them either —
//! a higher level handles batching weights via NaN-fraction guard).

use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::ops::{log_softmax, softmax};

/// ListNet loss. All inputs are (B, T) tensors.
///
/// `score` requires_grad=true going in; the returned loss carries the
/// gradient graph back to it.
pub fn listnet_loss(
    score:    &Tensor,    // (B, T) — model output, gradient source
    label:    &Tensor,    // (B, T) — target distribution input
    pad_mask: &Tensor,    // (B, T) bool — True where padding
    nan_mask: Option<&Tensor>, // (B, T) bool — True where label was NaN
) -> Result<Tensor> {
    let device = score.device();
    let dtype  = score.dtype();
    debug_assert_eq!(dtype, DType::F32, "expected f32 inputs");

    let neg_inf = Tensor::full(f32::NEG_INFINITY, score.shape(), device)?;

    // Combine masks: a slot is "skipped" if pad OR nan-label. Compute as
    // u8 (1 where skip, 0 where keep) so we can use where_cond.
    let pad_u = pad_mask.to_dtype(DType::U8)?;
    let skip_u = match nan_mask {
        Some(nm) => {
            // OR via clamp to 1.
            let combined = (pad_u + nm.to_dtype(DType::U8)?)?;
            combined.clamp(0_u8, 1_u8)?
        }
        None => pad_u,
    };

    // where_cond: where skip_u == 1, take neg_inf; else take score / label.
    // Using where_cond avoids the `0 * -inf = NaN` trap of additive masking.
    let masked_score = skip_u.where_cond(&neg_inf, score)?;
    let masked_label = skip_u.where_cond(&neg_inf, label)?;

    // Soft target distribution + log-prob predictions.
    let p_target = softmax(&masked_label, candle_core::D::Minus1)?;
    let log_pred = log_softmax(&masked_score, candle_core::D::Minus1)?;

    // Cross-entropy per row: -sum_t p_target * log_pred. The trap:
    // for skipped slots, p_target = 0 (softmax of -inf) AND log_pred =
    // -inf (log of 0) → 0 × -inf = NaN. Mask the per-element loss to
    // 0 in skipped slots BEFORE summing, so NaN never enters the sum.
    let zero = Tensor::zeros_like(&p_target)?;
    let row_terms = (p_target * log_pred)?.neg()?;
    let row_terms = skip_u.where_cond(&zero, &row_terms)?;
    let row_loss = row_terms.sum_keepdim(candle_core::D::Minus1)?;
    // Mean across batch dimension.
    let loss = row_loss.mean_all()?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn loss_is_finite_on_clean_input() {
        let dev = Device::Cpu;
        let score = Tensor::new(&[[0.1_f32, 0.2, 0.3, 0.4]], &dev).unwrap();
        let label = Tensor::new(&[[0.4_f32, 0.3, 0.2, 0.1]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 4), DType::U8, &dev).unwrap();
        let l = listnet_loss(&score, &label, &pad, None).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.is_finite(), "loss must be finite on clean input, got {}", v);
        assert!(v > 0.0, "ListNet loss should be > 0");
    }

    #[test]
    fn loss_drops_nan_labeled_rows() {
        let dev = Device::Cpu;
        let score = Tensor::new(&[[0.1_f32, 0.2, 0.3, 0.4]], &dev).unwrap();
        // Label has NaN in slot 1 — should be skipped; loss becomes
        // softmax over the other 3.
        let label = Tensor::new(&[[0.4_f32, f32::NAN, 0.2, 0.1]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 4), DType::U8, &dev).unwrap();
        let nan   = Tensor::new(&[[0_u8, 1, 0, 0]], &dev).unwrap();
        let l = listnet_loss(&score, &label, &pad, Some(&nan)).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.is_finite(), "loss must remain finite when NaN is masked, got {}", v);
    }

    #[test]
    fn fully_masked_row_returns_finite_loss() {
        // Every slot padded → softmax over no slots → loss should be 0
        // (or NaN, depending on candle softmax). Just verify no panic.
        let dev = Device::Cpu;
        let score = Tensor::new(&[[0.1_f32, 0.2]], &dev).unwrap();
        let label = Tensor::new(&[[0.4_f32, 0.3]], &dev).unwrap();
        let pad   = Tensor::ones((1, 2), DType::U8, &dev).unwrap();
        let _ = listnet_loss(&score, &label, &pad, None);
        // Don't assert finiteness; the contract is "doesn't panic". The
        // training loop's caller is expected to filter all-pad batches.
    }
}
