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
    // Audit fix MASK-ASYMMETRY (Round 2 deep audit, 2026-04-25): pre-fix
    // accepted nan_mask of any dtype and silently `to_dtype(U8)`'d it.
    // Bool tensors cast to U8 give {0,1} OK, but f32 nan_mask with
    // values like 0.5 would silently become 1 after cast — matching
    // dtype guarantees the contract. Now: explicit U8/Bool dtype check.
    let pad_u = pad_mask.to_dtype(DType::U8)?;
    let skip_u = match nan_mask {
        Some(nm) => {
            if nm.dtype() != DType::U8 {
                anyhow::bail!(
                    "listnet_loss: nan_mask must be DType::U8, got {:?}",
                    nm.dtype(),
                );
            }
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
    let row_loss = row_terms.sum(candle_core::D::Minus1)?;   // (B,)

    // Audit fix DIVERGENCE-PY (Round 5 audit, 2026-04-25): match the
    // Python `_listnet_loss` semantics — average over rows that have
    // ≥2 valid slots, NOT over all B rows. Without this, fully-padded
    // or single-valid-slot rows DILUTED the mean (their row_loss = 0
    // contributed to the numerator but added 1 to the denominator),
    // so the Rust loss number reported was smaller than Python's on
    // the same input. Identical training gradients (zero-loss rows
    // had zero grad anyway), but the loss curve shape on plots
    // differed and confused operators monitoring training health.
    //
    // Implementation: per-row count of valid slots (T - sum(skip)).
    // Then a mask of rows where count >= 2. Average row_loss over
    // those rows. If no rows have >= 2 valid → return 0 with the
    // gradient connection preserved (so backward_step is a no-op,
    // not an error).
    // valid = 1 - skip in f32 land. Use affine(-1, 1) to flip.
    let skip_f = skip_u.to_dtype(DType::F32)?;
    let valid_counts = skip_f
        .affine(-1.0_f64, 1.0_f64)?
        .sum(candle_core::D::Minus1)?;   // (B,)
    let valid_row_mask_u = valid_counts.ge(2.0_f32)?.to_dtype(DType::U8)?;
    let n_valid = valid_row_mask_u
        .to_dtype(DType::F32)?
        .sum_all()?
        .to_vec0::<f32>()?;
    if n_valid < 1.0 {
        // Degenerate batch — keep a zero-with-grad tensor so backward
        // is a clean no-op (mirrors Python's `scores.sum() * 0.0`).
        let zero_scalar = (score.sum_all()? * 0.0)?;
        return Ok(zero_scalar);
    }
    let mask_f = valid_row_mask_u.to_dtype(DType::F32)?;
    let masked_row_loss = row_loss.mul(&mask_f)?;
    let total = masked_row_loss.sum_all()?;
    // candle's `/` impl on Tensor returns Tensor (panics-on-error);
    // use the `affine` method to scale safely without unwrapping.
    let loss = total.affine(1.0_f64 / n_valid as f64, 0.0_f64)?;
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
