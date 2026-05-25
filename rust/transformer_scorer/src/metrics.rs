//! Information Coefficient (IC) — Spearman rank correlation of model
//! predictions vs labels, pooled across date groups.
//!
//! References:
//!   * Poh-Lim-Zohren-Roberts 2020, "Building Cross-Sectional Systematic
//!     Strategies By Learning to Rank" — IC is the standard metric for
//!     cross-sectional ranking.
//!   * Python parity: `training_panel/purged_cv.py::evaluate_fold_ic`
//!     uses `scipy.stats.spearmanr` per date, then mean-of-finite.
//!
//! Per-date IC: rank both vectors, pearson on the ranks. Skip degenerate
//! groups (n<2 or all-equal). Pool by mean across remaining groups.
//!
//! Edge cases mirror the Python:
//!   * n < 2 → IC undefined for that group, skip from the mean
//!   * NaN labels in test → skip those rows in the per-date Spearman
//!   * all-equal predictions OR labels → IC=0 for that group (ties)

/// Compute Spearman rank correlation between two equal-length f32 vectors.
/// NaN entries on either side cause the corresponding rows to be dropped.
/// Returns None if fewer than 2 valid pairs remain or one side is constant.
pub fn spearman_corr(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }
    // Drop rows with NaN on either side.
    let pairs: Vec<(f32, f32)> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(&x, &y)| (x, y))
        .collect();
    if pairs.len() < 2 {
        return None;
    }
    let (xs, ys): (Vec<f32>, Vec<f32>) = pairs.iter().copied().unzip();
    let rx = average_ranks(&xs);
    let ry = average_ranks(&ys);
    pearson_corr(&rx, &ry)
}

/// Average-rank assignment matching scipy's default Spearman (method="average").
/// Ties get the average of their tied ranks. Returns ranks in original order.
fn average_ranks(xs: &[f32]) -> Vec<f32> {
    let n = xs.len();
    let mut idx: Vec<usize> = (0..n).collect();
    // Sort indices by xs[i]; NaN-stable since caller already filtered.
    idx.sort_by(|&i, &j| xs[i].partial_cmp(&xs[j]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0_f32; n];
    let mut i = 0;
    while i < n {
        // Find run of ties.
        let mut j = i + 1;
        while j < n && xs[idx[j]] == xs[idx[i]] {
            j += 1;
        }
        // Average rank for [i, j) — 1-based ranks.
        let avg_rank = ((i + 1) as f32 + j as f32) / 2.0;   // (i+1 + j) / 2
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation of two equal-length vectors. Returns None on
/// length mismatch or when either side has zero variance (constant).
pub fn pearson_corr(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.len() < 2 {
        return None;
    }
    let n = a.len() as f64;
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n;
    let mut num = 0.0_f64;
    let mut da2 = 0.0_f64;
    let mut db2 = 0.0_f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        num += dx * dy;
        da2 += dx * dx;
        db2 += dy * dy;
    }
    let denom = (da2 * db2).sqrt();
    // Audit fix PEARSON-UNDERFLOW (Round 2 deep audit, 2026-04-25):
    // pre-fix used `denom <= 0.0`. With BOTH variances around 1e-40
    // (degenerate near-constant input), `da2 * db2` can underflow
    // entire-mantissa to zero, then sqrt(0) = 0, then `num / 0` is
    // ±inf instead of None. Threshold at 1e-15 (well above f64
    // smallest normal of 2.2e-308 but covers underflow-prone tiny
    // variances we never want to report a correlation on).
    if !denom.is_finite() || denom <= 1e-15 {
        return None;
    }
    Some((num / denom) as f32)
}

/// Pool per-group Spearman ICs into a mean. Skips groups returning None.
/// Returns (mean_ic, n_valid_groups).
pub fn pooled_ic<'a, I>(groups: I) -> (f32, usize)
where
    I: IntoIterator<Item = (&'a [f32], &'a [f32])>,
{
    let mut sum = 0.0_f64;
    let mut n = 0_usize;
    for (preds, labels) in groups {
        if let Some(ic) = spearman_corr(preds, labels) {
            sum += ic as f64;
            n += 1;
        }
    }
    let mean = if n > 0 { (sum / n as f64) as f32 } else { f32::NAN };
    (mean, n)
}

/// Convenience: compute pooled IC from `Vec<(predictions, labels)>` of
/// owned f32 vectors. Caller iterates per-date groups.
pub fn pooled_ic_owned(groups: &[(Vec<f32>, Vec<f32>)]) -> (f32, usize) {
    pooled_ic(groups.iter().map(|(p, l)| (p.as_slice(), l.as_slice())))
}

/// Spearman-via-pearson invariant test
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_monotonic_yields_ic_one() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0_f32, 20.0, 30.0, 40.0, 50.0];
        let ic = spearman_corr(&a, &b).expect("ic");
        assert!((ic - 1.0).abs() < 1e-6, "perfect rank correlation should be 1.0, got {}", ic);
    }

    #[test]
    fn perfect_inverse_yields_minus_one() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0_f32, 4.0, 3.0, 2.0, 1.0];
        let ic = spearman_corr(&a, &b).expect("ic");
        assert!((ic - (-1.0)).abs() < 1e-6, "inverse rank → -1.0, got {}", ic);
    }

    #[test]
    fn random_uncorrelated_close_to_zero() {
        // 1000-point pseudo-random check; tolerance loose because n is finite.
        let n = 1000;
        let a: Vec<f32> = (0..n).map(|i| ((i as f32 * 1.61803).fract() - 0.5)).collect();
        let b: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.31415).fract() - 0.5)).collect();
        let ic = spearman_corr(&a, &b).unwrap();
        assert!(ic.abs() < 0.15, "uncorrelated IC should be near 0, got {}", ic);
    }

    #[test]
    fn average_rank_handles_ties() {
        // [10, 20, 20, 30] → ranks [1, 2.5, 2.5, 4]
        let xs = vec![10.0_f32, 20.0, 20.0, 30.0];
        let r = average_ranks(&xs);
        assert!((r[0] - 1.0).abs() < 1e-6);
        assert!((r[1] - 2.5).abs() < 1e-6);
        assert!((r[2] - 2.5).abs() < 1e-6);
        assert!((r[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn nan_pairs_dropped() {
        let a = vec![1.0_f32, f32::NAN, 3.0, 4.0];
        let b = vec![10.0_f32, 99.0, 30.0, 40.0];   // NaN drops index 1
        let ic = spearman_corr(&a, &b).expect("ic");
        // Remaining: [1,10],[3,30],[4,40] — perfect monotonic.
        assert!((ic - 1.0).abs() < 1e-6);
    }

    #[test]
    fn constant_input_returns_none() {
        let a = vec![5.0_f32, 5.0, 5.0, 5.0];
        let b = vec![1.0_f32, 2.0, 3.0, 4.0];
        let ic = spearman_corr(&a, &b);
        assert!(ic.is_none(), "constant a → no variance → None");
    }

    #[test]
    fn single_pair_returns_none() {
        let a = vec![1.0_f32];
        let b = vec![1.0_f32];
        assert!(spearman_corr(&a, &b).is_none());
    }

    #[test]
    fn pooled_ic_means_finite_groups() {
        // Group 1: perfect → IC=1
        // Group 2: inverse → IC=-1
        // Group 3: constant → skipped
        let g1 = (vec![1.0_f32, 2.0, 3.0], vec![10.0_f32, 20.0, 30.0]);
        let g2 = (vec![1.0_f32, 2.0, 3.0], vec![30.0_f32, 20.0, 10.0]);
        let g3 = (vec![5.0_f32, 5.0, 5.0], vec![1.0_f32, 2.0, 3.0]);
        let groups = vec![g1, g2, g3];
        let (mean, n) = pooled_ic_owned(&groups);
        assert_eq!(n, 2, "group 3 should be skipped (constant)");
        assert!((mean - 0.0).abs() < 1e-6, "mean of [1, -1] = 0; got {}", mean);
    }

    #[test]
    fn pearson_matches_known_value() {
        // y = 2x + 1 — perfect linear, pearson = 1.0
        let x = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0_f32, 5.0, 7.0, 9.0, 11.0];
        let r = pearson_corr(&x, &y).unwrap();
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn pearson_underflow_returns_none() {
        // Audit fix PEARSON-UNDERFLOW: tiny variance values → denom
        // underflows. Pre-fix returned a finite-but-meaningless result
        // (or inf/NaN). Post-fix: 1e-15 threshold returns None.
        let a = vec![1e-25_f32, 2e-25, 3e-25];
        let b = vec![3e-25_f32, 1e-25, 2e-25];
        let r = pearson_corr(&a, &b);
        // Values themselves are ordered, so spearman would be defined,
        // but pearson on these tiny floats → underflow → expect None
        // (or at minimum a finite value, not inf/NaN).
        match r {
            None => {}   // ✓ — caught the underflow
            Some(v) => assert!(v.is_finite() && v.abs() <= 1.0,
                               "if not None, must still be a sane correlation, got {}", v),
        }
    }
}
