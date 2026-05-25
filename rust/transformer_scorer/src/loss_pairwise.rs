//! RankNet pairwise loss — Burges et al 2005, "Learning to Rank using
//! Gradient Descent" (Microsoft Research). Foundational pairwise LTR
//! method, reused by Poh-Lim-Zohren-Roberts 2020 for cross-sectional
//! stock ranking on S&P 500 (arXiv 2012.07149).
//!
//! For each row in the batch, generate all label-distinct pairs (i, j)
//! where label_i > label_j. The loss for the pair is:
//!
//! ```text
//!     L_ij = log(1 + exp(-(s_i - s_j)))
//! ```
//!
//! which is the binary cross-entropy of the "i ranks above j" event.
//! Gradient is `-1/(1+exp(s_i-s_j))` for s_i and `+1/(...)` for s_j —
//! pushes higher-label score up, lower-label score down, magnitude
//! proportional to how WRONG the model currently is on that pair.
//!
//! Compared to ListNet (top-1 softmax CE):
//!   * ListNet: gradient mostly concentrated on top-K items
//!   * RankNet: gradient spread evenly across ALL pairs, so middle-of-
//!     pack ordering also gets corrective signal
//!
//! Trade-off: O(T²) pairs per group vs ListNet's O(T). On our T≈99
//! panel, that's 9801 pairs × 67 val groups = ~650K comparisons per
//! eval. Negligible vs the matmul cost of the forward pass.
//!
//! NaN-safe: pairs where either label is NaN OR either slot is padded
//! are excluded from the sum.

use anyhow::Result;
use candle_core::{DType, Tensor};

/// Pairwise RankNet loss. All inputs (B, T) shape; pad_mask & nan_mask
/// (B, T) u8 with 1 where the slot should be excluded.
pub fn ranknet_loss(
    score:    &Tensor,
    label:    &Tensor,
    pad_mask: &Tensor,
    nan_mask: Option<&Tensor>,
) -> Result<Tensor> {
    let dtype  = score.dtype();
    debug_assert_eq!(dtype, DType::F32, "expected f32 inputs");

    // Combined skip mask (1 where slot is invalid).
    let pad_u  = pad_mask.to_dtype(DType::U8)?;
    let skip_u = match nan_mask {
        Some(nm) => {
            if nm.dtype() != DType::U8 {
                anyhow::bail!("ranknet_loss: nan_mask must be U8");
            }
            (pad_u + nm.to_dtype(DType::U8)?)?.clamp(0_u8, 1_u8)?
        }
        None => pad_u,
    };
    let valid_f = skip_u.to_dtype(DType::F32)?.affine(-1.0_f64, 1.0_f64)?;  // (B,T)
    let (b, t) = score.dims2()?;

    // For each row build (B, T, T) matrices:
    //   s_diff[b, i, j] = score[b, i] - score[b, j]
    //   y_pair[b, i, j] = +1 if label[b,i] > label[b,j], 0 if equal, ignored otherwise
    //   valid_pair[b, i, j] = valid_i AND valid_j AND label[b,i] > label[b,j]
    let s_i = score.unsqueeze(2)?.broadcast_as((b, t, t))?;     // (B, T, T) — score along dim-1
    let s_j = score.unsqueeze(1)?.broadcast_as((b, t, t))?;     // (B, T, T) — score along dim-2
    let s_diff = (s_i - s_j)?;                                   // (B, T, T)

    let l_i = label.unsqueeze(2)?.broadcast_as((b, t, t))?;
    let l_j = label.unsqueeze(1)?.broadcast_as((b, t, t))?;

    // valid_pair = valid_i * valid_j * (label_i > label_j)
    let valid_i = valid_f.unsqueeze(2)?.broadcast_as((b, t, t))?;
    let valid_j = valid_f.unsqueeze(1)?.broadcast_as((b, t, t))?;
    let l_gt    = l_i.gt(&l_j)?.to_dtype(DType::F32)?;
    let pair_w  = (valid_i * valid_j)?.mul(&l_gt)?;              // (B, T, T) f32

    // Stable log(1 + exp(-x)):
    //   softplus(-x) = max(0, -x) + log(1 + exp(-|x|))
    // candle has Tensor::exp(); compute manually for numerical stability.
    let neg = s_diff.neg()?;
    let zero = Tensor::zeros_like(&neg)?;
    let max_part = neg.maximum(&zero)?;
    let abs_diff = s_diff.abs()?;
    let log1pexp = abs_diff.neg()?.exp()?.affine(1.0, 1.0)?.log()?;
    let softplus_neg = (max_part + log1pexp)?;                   // (B, T, T)

    // Sum of pair losses, weighted by pair_w (zero on invalid pairs).
    let weighted = (softplus_neg * &pair_w)?;
    let total = weighted.sum_all()?;
    let n_pairs = pair_w.sum_all()?.to_vec0::<f32>()?;
    if n_pairs < 1.0 {
        // Degenerate batch — return zero with grad through scores.
        let zero_scalar = (score.sum_all()? * 0.0)?;
        return Ok(zero_scalar);
    }
    let loss = total.affine(1.0_f64 / n_pairs as f64, 0.0_f64)?;
    Ok(loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn perfect_ranking_yields_low_loss() {
        let dev = Device::Cpu;
        // Higher score for higher label — ranker is correct everywhere.
        let score = Tensor::new(&[[3.0_f32, 2.0, 1.0, 0.0]], &dev).unwrap();
        let label = Tensor::new(&[[3.0_f32, 2.0, 1.0, 0.0]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 4), DType::U8, &dev).unwrap();
        let l = ranknet_loss(&score, &label, &pad, None).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.is_finite() && v < 0.4,
            "perfect-ranked → small loss; got {}", v);
    }

    #[test]
    fn inverted_ranking_yields_high_loss() {
        let dev = Device::Cpu;
        // Score is inverted relative to label — ranker is maximally wrong.
        let score = Tensor::new(&[[0.0_f32, 1.0, 2.0, 3.0]], &dev).unwrap();
        let label = Tensor::new(&[[3.0_f32, 2.0, 1.0, 0.0]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 4), DType::U8, &dev).unwrap();
        let l = ranknet_loss(&score, &label, &pad, None).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.is_finite() && v > 1.0,
            "inverted-ranked → large loss; got {}", v);
    }

    #[test]
    fn zero_pairs_returns_zero_with_grad() {
        // All slots have the SAME label → no valid pairs → loss=0 (no signal).
        let dev = Device::Cpu;
        let score = Tensor::new(&[[1.0_f32, 2.0, 3.0]], &dev).unwrap();
        let label = Tensor::new(&[[5.0_f32, 5.0, 5.0]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 3), DType::U8, &dev).unwrap();
        let l = ranknet_loss(&score, &label, &pad, None).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.abs() < 1e-6, "no-pairs case → 0 loss; got {}", v);
    }

    #[test]
    fn pad_mask_excludes_pairs() {
        // Mask out slot 1 — only slots 0 and 2 contribute. label[0]>label[2]
        // and score is correct → low loss.
        let dev = Device::Cpu;
        let score = Tensor::new(&[[3.0_f32, 99.0, 1.0]], &dev).unwrap();
        let label = Tensor::new(&[[3.0_f32, 99.0, 1.0]], &dev).unwrap();
        let pad   = Tensor::new(&[[0_u8, 1, 0]], &dev).unwrap();
        let l = ranknet_loss(&score, &label, &pad, None).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        // Without the mask the score/label are still consistent so it'd
        // be small either way — but verify it doesn't blow up + is finite.
        assert!(v.is_finite() && v < 0.5,
            "masked slot → loss still small; got {}", v);
    }

    #[test]
    fn nan_label_excluded() {
        let dev = Device::Cpu;
        let score = Tensor::new(&[[3.0_f32, 99.0, 1.0]], &dev).unwrap();
        let label = Tensor::new(&[[3.0_f32, f32::NAN, 1.0]], &dev).unwrap();
        let pad   = Tensor::zeros((1, 3), DType::U8, &dev).unwrap();
        let nan   = Tensor::new(&[[0_u8, 1, 0]], &dev).unwrap();
        let l = ranknet_loss(&score, &label, &pad, Some(&nan)).unwrap();
        let v = l.to_vec0::<f32>().unwrap();
        assert!(v.is_finite(),
            "NaN label slot must be masked, got {}", v);
    }
}
